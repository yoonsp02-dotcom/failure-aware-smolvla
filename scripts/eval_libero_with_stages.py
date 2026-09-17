#!/usr/bin/env python
"""Run LeRobot evaluation with failure-aware stage instrumentation."""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

from lerobot.scripts import lerobot_eval

from failure_aware_smolvla.stage_metrics import write_stage_outputs
from failure_aware_smolvla.tracked_libero_env import create_tracked_libero_envs


def _required_output_dir(argv: list[str]) -> Path:
    for index, argument in enumerate(argv):
        if argument.startswith("--output_dir="):
            return Path(argument.split("=", 1)[1]).resolve()
        if argument == "--output_dir" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
    raise SystemExit("--output_dir is required so stage metrics have a stable destination")


def main() -> None:
    output_dir = _required_output_dir(sys.argv[1:])
    stage_dir = output_dir / "stages"
    lerobot_eval.make_env = partial(create_tracked_libero_envs, stage_output_dir=stage_dir)
    lerobot_eval.main()
    summary = write_stage_outputs(stage_dir=stage_dir, output_dir=output_dir)
    print(f"Stage metrics: {output_dir / 'stage_summary.json'}")
    print(f"Tracked episodes: {summary['overall']['n_episodes']}")


if __name__ == "__main__":
    main()
