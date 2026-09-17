"""LeRobot LIBERO environment instrumented with semantic stage tracking."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
from lerobot.envs.libero import LiberoEnv, _get_suite, _select_task_ids

from .stage_tracker import LiberoStageSignalReader, StageTracker, StageTrackerConfig


class TrackedLiberoEnv(LiberoEnv):
    """Scalar LIBERO environment that persists stage data after every episode."""

    def __init__(
        self,
        *args: Any,
        stage_output_dir: str | Path,
        task_suite_name: str,
        stage_config: StageTrackerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, task_suite_name=task_suite_name, **kwargs)
        self._tracked_suite_name = task_suite_name
        self._stage_output_dir = Path(stage_output_dir)
        self._stage_config = stage_config or StageTrackerConfig()
        self._stage_reader: LiberoStageSignalReader | None = None
        self._stage_tracker: StageTracker | None = None
        self._stage_step = 0
        self._stage_episode_counter = 0
        self._stage_active = False
        self._stage_finalized = False
        self._stage_seed: int | None = None
        self._last_stage_path: Path | None = None

    def reset(self, seed=None, **kwargs):
        if self._stage_active and not self._stage_finalized:
            self._finalize_stage_episode("reset")

        observation, info = super().reset(seed=seed, **kwargs)
        self._stage_reader = LiberoStageSignalReader(self)
        self._stage_tracker = StageTracker(self._stage_config)
        self._stage_step = 0
        self._stage_seed = int(seed) if seed is not None else None
        self._stage_tracker.reset(self._stage_reader.read(step=0))
        self._stage_active = True
        self._stage_finalized = False
        info = dict(info)
        info["stage"] = self._stage_tracker.records[-1].current_stage
        info["max_stage_reached"] = self._stage_tracker.records[-1].max_stage_reached
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if self._stage_tracker is None or self._stage_reader is None:
            raise RuntimeError("TrackedLiberoEnv.step() called before reset()")

        self._stage_step += 1
        record = self._stage_tracker.update(self._stage_reader.read(step=self._stage_step))
        info = dict(info)
        info["stage"] = record.current_stage
        info["max_stage_reached"] = record.max_stage_reached

        hit_time_limit = self._stage_step >= self._max_episode_steps
        if terminated or truncated or hit_time_limit:
            reason = "success" if info.get("is_success", False) else "time_limit" if hit_time_limit else "terminated"
            path = self._finalize_stage_episode(reason)
            info["stage_metrics_path"] = str(path)
        return observation, reward, terminated, truncated, info

    def close(self):
        if self._stage_active and not self._stage_finalized:
            self._finalize_stage_episode("closed")
        super().close()

    def _finalize_stage_episode(self, termination_reason: str) -> Path:
        if self._stage_tracker is None or self._stage_reader is None:
            raise RuntimeError("Cannot finalize stage episode before reset()")
        if self._stage_finalized:
            if self._last_stage_path is None:
                raise RuntimeError("Stage episode is finalized but its output path is missing")
            return self._last_stage_path

        global_episode_index = self.episode_index + self._stage_episode_counter * self._reset_stride
        payload = {
            "schema_version": 1,
            "task_group": self._tracked_suite_name,
            "task_id": self.task_id,
            "task": self.task,
            "task_description": self.task_description,
            "episode_index": global_episode_index,
            "seed": self._stage_seed,
            "object_name": self._stage_reader.object_name,
            "target_name": self._stage_reader.target_name,
            "termination_reason": termination_reason,
            "thresholds": asdict(self._stage_config),
            "summary": self._stage_tracker.summary(),
            "timeline": self._stage_tracker.timeline(),
        }
        path = self._episode_path(global_episode_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)

        self._last_stage_path = path
        self._stage_finalized = True
        self._stage_active = False
        self._stage_episode_counter += 1
        return path

    def _episode_path(self, episode_index: int) -> Path:
        task_dir = self._stage_output_dir / f"{self._tracked_suite_name}_{self.task_id}"
        return task_dir / f"episode_{episode_index:04d}.json"


def create_tracked_libero_envs(
    cfg: Any,
    *,
    n_envs: int,
    use_async_envs: bool,
    stage_output_dir: str | Path,
    stage_config: StageTrackerConfig | None = None,
    **_: Any,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create tracked environments with the shape expected by LeRobot eval."""

    if getattr(cfg, "type", None) != "libero":
        raise ValueError(f"Stage tracking currently supports env.type=libero, got {getattr(cfg, 'type', None)}")
    if n_envs < 1:
        raise ValueError(f"n_envs must be positive, got {n_envs}")

    suite_names = [name.strip() for name in str(cfg.task).split(",") if name.strip()]
    output: dict[str, dict[int, gym.vector.VectorEnv]] = {}
    vec_cls = gym.vector.AsyncVectorEnv if (use_async_envs and n_envs > 1) else gym.vector.SyncVectorEnv

    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        selected_ids = _select_task_ids(len(suite.tasks), cfg.task_ids)
        task_envs: dict[int, gym.vector.VectorEnv] = {}
        for task_id in selected_ids:
            factories = []
            for env_index in range(n_envs):
                factories.append(
                    partial(
                        TrackedLiberoEnv,
                        task_suite=suite,
                        task_id=task_id,
                        task_suite_name=suite_name,
                        stage_output_dir=stage_output_dir,
                        stage_config=stage_config,
                        camera_name=cfg.camera_name,
                        init_states=cfg.init_states,
                        episode_length=cfg.episode_length,
                        episode_index=env_index,
                        n_envs=n_envs,
                        control_mode=cfg.control_mode,
                        camera_name_mapping=cfg.camera_name_mapping,
                        is_libero_plus=cfg.is_libero_plus,
                        **{key: value for key, value in cfg.gym_kwargs.items() if key != "task_ids"},
                    )
                )
            task_envs[task_id] = vec_cls(
                factories,
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            )
        output[suite_name] = task_envs
    return output
