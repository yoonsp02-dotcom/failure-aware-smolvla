"""Semantic stage tracking for LIBERO pick-and-place episodes.

The tracker consumes privileged simulator signals for evaluation and labeling
only. These signals must not be added to the policy observation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any


Vec3 = tuple[float, float, float]


class Stage(IntEnum):
    START = 0
    REACH = 1
    GRASP = 2
    LIFT = 3
    TRANSPORT = 4
    LOWER = 5
    PLACE = 6


@dataclass(frozen=True)
class StageTrackerConfig:
    """Thresholds for the first pick-and-place tracker version."""

    reach_distance_m: float = 0.06 # 로봇 손과 물체가 6cm 이내
    lift_height_m: float = 0.04 # 물체가 초기 높이보다 4cm 이상 상승
    transport_xy_distance_m: float = 0.10 # 물체와 목표 지점의 수평 거리가 10cm 이내
    drop_height_tolerance_m: float = 0.02 # 들었던 물체가 초기 높이에서 2cm 이내까지 추락

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class StageSignals:
    """Privileged state sampled after one environment step."""

    step: int
    end_effector_pos: Vec3
    object_pos: Vec3
    target_pos: Vec3
    grasped: bool
    goal_satisfied: bool

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step must be non-negative, got {self.step}")
        for name in ("end_effector_pos", "object_pos", "target_pos"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(float(x)) for x in value):
                raise ValueError(f"{name} must contain three finite values, got {value}")


@dataclass(frozen=True)
class StageEvent:
    step: int
    event: str
    from_stage: str | None = None
    to_stage: str | None = None


@dataclass(frozen=True)
class StageRecord:
    step: int
    current_stage: str
    max_stage_reached: str
    ee_to_object_distance_m: float
    object_to_target_xy_distance_m: float
    object_height_delta_m: float
    grasped: bool
    goal_satisfied: bool


class StageTracker:
    """Track semantic progress, regressions, and recovery events.

    Stages are derived independently at every step, while
    ``max_stage_reached`` and first-reached steps are monotonic episode-level
    summaries. Call ``reset`` once after the environment reset and ``update``
    after every action.
    """

    def __init__(self, config: StageTrackerConfig | None = None) -> None:
        self.config = config or StageTrackerConfig()
        self._initialized = False
        self._initial_object_z = 0.0
        self._current_stage = Stage.START
        self._max_stage = Stage.START
        self._previous_grasped = False
        self._pending_drop = False
        self._regrasp_seen = False
        self._recovery_success_recorded = False
        self._first_step: dict[Stage, int] = {}
        self._events: list[StageEvent] = []
        self._records: list[StageRecord] = []

    @property
    def events(self) -> tuple[StageEvent, ...]:
        return tuple(self._events)

    @property
    def records(self) -> tuple[StageRecord, ...]:
        return tuple(self._records)

    @property
    def current_stage(self) -> Stage:
        self._require_initialized()
        return self._current_stage

    @property
    def max_stage_reached(self) -> Stage:
        self._require_initialized()
        return self._max_stage

    def reset(self, signals: StageSignals) -> StageRecord:
        self._initialized = True
        self._initial_object_z = float(signals.object_pos[2])
        self._current_stage = Stage.START
        self._max_stage = Stage.START
        self._previous_grasped = False
        self._pending_drop = False
        self._regrasp_seen = False
        self._recovery_success_recorded = False
        self._first_step = {Stage.START: signals.step}
        self._events = []
        self._records = []
        return self._process(signals, is_reset=True)

    def update(self, signals: StageSignals) -> StageRecord:
        self._require_initialized()
        if self._records and signals.step <= self._records[-1].step:
            raise ValueError(
                f"step must increase monotonically: previous={self._records[-1].step}, got={signals.step}"
            )
        return self._process(signals, is_reset=False)

    def summary(self) -> dict[str, Any]:
        self._require_initialized()
        event_counts: dict[str, int] = {}
        for event in self._events:
            event_counts[event.event] = event_counts.get(event.event, 0) + 1

        return {
            "final_step": self._records[-1].step,
            "official_success": self._records[-1].goal_satisfied,
            "current_stage": self._current_stage.name.lower(),
            "max_stage_reached": self._max_stage.name.lower(),
            "first_step_by_stage": {
                stage.name.lower(): self._first_step.get(stage) for stage in Stage
            },
            "event_counts": event_counts,
            "events": [asdict(event) for event in self._events],
        }

    def timeline(self) -> list[dict[str, Any]]:
        self._require_initialized()
        return [asdict(record) for record in self._records]

    def _process(self, signals: StageSignals, *, is_reset: bool) -> StageRecord:
        ee_distance = math.dist(signals.end_effector_pos, signals.object_pos)
        target_xy_distance = math.dist(signals.object_pos[:2], signals.target_pos[:2])
        height_delta = float(signals.object_pos[2]) - self._initial_object_z

        stage = self._derive_stage(
            ee_distance=ee_distance,
            target_xy_distance=target_xy_distance,
            height_delta=height_delta,
            grasped=signals.grasped,
            goal_satisfied=signals.goal_satisfied,
        )

        previous_stage = self._current_stage
        if not is_reset and stage != previous_stage:
            self._events.append(
                StageEvent(
                    step=signals.step,
                    event="stage_changed",
                    from_stage=previous_stage.name.lower(),
                    to_stage=stage.name.lower(),
                )
            )

        if not is_reset and self._previous_grasped and not signals.grasped and not signals.goal_satisfied:
            self._events.append(StageEvent(step=signals.step, event="grasp_lost"))
            self._pending_drop = True

        if not is_reset and signals.grasped and not self._previous_grasped:
            if any(event.event == "grasp_lost" for event in self._events):
                self._events.append(StageEvent(step=signals.step, event="regrasped"))
                self._regrasp_seen = True
            self._pending_drop = False

        if (
            self._pending_drop
            and self._max_stage >= Stage.LIFT
            and not signals.grasped
            and not signals.goal_satisfied
            and height_delta <= self.config.drop_height_tolerance_m
        ):
            self._events.append(StageEvent(step=signals.step, event="object_dropped"))
            self._pending_drop = False

        if signals.goal_satisfied and self._regrasp_seen and not self._recovery_success_recorded:
            self._events.append(StageEvent(step=signals.step, event="recovery_success"))
            self._recovery_success_recorded = True

        self._current_stage = stage
        self._max_stage = max(self._max_stage, stage)
        for reached_stage in Stage:
            if reached_stage <= stage and reached_stage not in self._first_step:
                self._first_step[reached_stage] = signals.step

        record = StageRecord(
            step=signals.step,
            current_stage=stage.name.lower(),
            max_stage_reached=self._max_stage.name.lower(),
            ee_to_object_distance_m=ee_distance,
            object_to_target_xy_distance_m=target_xy_distance,
            object_height_delta_m=height_delta,
            grasped=bool(signals.grasped),
            goal_satisfied=bool(signals.goal_satisfied),
        )
        self._records.append(record)
        self._previous_grasped = bool(signals.grasped)
        return record

    def _derive_stage(
        self,
        *,
        ee_distance: float,
        target_xy_distance: float,
        height_delta: float,
        grasped: bool,
        goal_satisfied: bool,
    ) -> Stage:
        if goal_satisfied:
            return Stage.PLACE
        lifted = height_delta >= self.config.lift_height_m
        if (
            grasped
            and self._max_stage >= Stage.TRANSPORT
            and target_xy_distance <= self.config.transport_xy_distance_m
            and not lifted
        ):
            return Stage.LOWER
        if grasped and lifted and target_xy_distance <= self.config.transport_xy_distance_m:
            return Stage.TRANSPORT
        if grasped and lifted:
            return Stage.LIFT
        if grasped:
            return Stage.GRASP
        if ee_distance <= self.config.reach_distance_m:
            return Stage.REACH
        return Stage.START

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("StageTracker.reset() must be called before reading or updating the tracker")


class LiberoStageSignalReader:
    """Read pick-and-place stage signals from an initialized LIBERO environment.

    The adapter supports raw LIBERO task environments and LeRobot's scalar
    ``LiberoEnv`` wrapper. Vector environments intentionally are not supported
    because their simulator state lives in separate worker processes.
    """

    def __init__(self, env: Any) -> None:
        self.env = self._unwrap(env)
        goal_states = self.env.parsed_problem["goal_state"]
        if len(goal_states) != 1 or len(goal_states[0]) != 3:
            raise ValueError(f"Expected one binary LIBERO goal predicate, got {goal_states}")
        predicate, object_name, target_name = goal_states[0]
        if str(predicate).lower() != "on":
            raise ValueError(f"Only On(object, target) goals are supported in v1, got {goal_states[0]}")
        self.goal_state = goal_states[0]
        self.object_name = object_name
        self.target_name = target_name

    def read(self, step: int) -> StageSignals:
        object_pos = self.env.object_states_dict[self.object_name].get_geom_state()["pos"]
        target_pos = self.env.object_states_dict[self.target_name].get_geom_state()["pos"]
        grasped = self.env._check_grasp(
            gripper=self.env.robots[0].gripper,
            object_geoms=self.env.get_object(self.object_name),
        )
        goal_satisfied = self.env._eval_predicate(self.goal_state)
        return StageSignals(
            step=step,
            end_effector_pos=self._vec3(self.env._eef_xpos),
            object_pos=self._vec3(object_pos),
            target_pos=self._vec3(target_pos),
            grasped=bool(grasped),
            goal_satisfied=bool(goal_satisfied),
        )

    @staticmethod
    def _unwrap(env: Any) -> Any:
        candidate = env
        visited: set[int] = set()
        while id(candidate) not in visited:
            visited.add(id(candidate))
            if all(hasattr(candidate, name) for name in ("parsed_problem", "object_states_dict", "sim")):
                return candidate
            if hasattr(candidate, "_env") and candidate._env is not None:
                candidate = candidate._env
                continue
            if hasattr(candidate, "env"):
                candidate = candidate.env
                continue
            break
        raise TypeError(
            "Could not find an initialized scalar LIBERO environment. "
            "Call reset() first and do not pass a vector environment."
        )

    @staticmethod
    def _vec3(value: Any) -> Vec3:
        return (float(value[0]), float(value[1]), float(value[2]))
