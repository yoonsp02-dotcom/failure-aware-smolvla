#!/usr/bin/env python
"""Smoke-test the stage tracker against one real LIBERO-Spatial environment."""

from __future__ import annotations

import json

import numpy as np
from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action

from failure_aware_smolvla import LiberoStageSignalReader, StageTracker


def main() -> None:
    suite = _get_suite("libero_spatial")
    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_spatial",
        observation_width=256,
        observation_height=256,
        obs_type="pixels_agent_pos",
        init_states=True,
        episode_index=0,
        n_envs=1,
    )
    try:
        env.reset(seed=1000)
        reader = LiberoStageSignalReader(env)
        tracker = StageTracker()
        initial = tracker.reset(reader.read(step=0))

        action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
        env.step(action)
        after_noop = tracker.update(reader.read(step=1))

        print(
            json.dumps(
                {
                    "object_name": reader.object_name,
                    "target_name": reader.target_name,
                    "initial": initial.__dict__,
                    "after_noop": after_noop.__dict__,
                    "summary": tracker.summary(),
                },
                indent=2,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
