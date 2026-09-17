# Failure-Aware SmolVLA

Experiments on failure-targeted data selection for small vision-language-action policies.

## Local environment

- WSL2 / Ubuntu 24.04
- Conda environment: `failure-aware-vla`
- Python 3.12
- PyTorch 2.11 with CUDA 12.8
- FFmpeg 7.1.1 (TorchCodec-compatible)
- LeRobot 0.6.2 installed in editable mode from `../lerobot`
- LeRobot commit: see `configs/lerobot_commit.txt`

Activate the environment in a new shell:

```bash
conda activate failure-aware-vla
cd /home/sangpil/projects/failure-aware-smolvla
python scripts/check_environment.py
```

The conda environment automatically sets `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl` for headless simulation.

## Recreate the environment

Clone LeRobot next to this repository, then run:

```bash
conda env create -f environment.yml
conda activate failure-aware-vla
python -m pip install uv

uv pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128" "torchvision==0.26.0+cu128"

uv pip install "cmake>=3.29,<4"
uv pip install -e "../lerobot[training,smolvla,libero]"
```

The CMake `<4` constraint is required by the current LIBERO EGL probe build.

## Smoke evaluation

After activating the conda environment, run one pretrained SmolVLA episode on
LIBERO-Spatial task 0:

```bash
./scripts/eval_smolvla_libero_smoke.sh
```

The first successful run is documented in
`reports/2026-09-17-smoke-test.md`.

## Per-task LIBERO evaluation

Run selected LIBERO-Spatial tasks independently so every task gets its own
`eval_info.json`, videos, and persistent log. Task numbers are one-based for
convenience (`1..10`), while LIBERO's internal task IDs are zero-based.

```bash
./scripts/eval_smolvla_libero_tasks.sh 8 9 10
```

Each run is stored under a timestamped directory in `outputs/eval/`. The
top-level `status.tsv` records whether each task is running, done, or failed.

## Semantic stage tracker

`failure_aware_smolvla.stage_tracker` derives pick-and-place stages from
privileged LIBERO simulator state without exposing that state to the policy:

```text
start -> reach -> grasp -> lift -> transport -> lower -> place
```

It also records stage regressions and the `grasp_lost`, `object_dropped`,
`regrasped`, and `recovery_success` events. The first version supports the
single `On(object, target)` goal used by LIBERO-Spatial.

Install the local project and run its tests:

```bash
python -m pip install -e . --no-deps --no-build-isolation
python -m pytest tests/test_stage_tracker.py
```

Smoke-test the privileged-state adapter on a real LIBERO environment:

```bash
python scripts/check_stage_tracker_libero.py
```

Run the normal LeRobot evaluator with stage instrumentation by replacing the
`lerobot-eval` executable with the project script and keeping the usual flags:

```bash
python scripts/eval_libero_with_stages.py \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --seed=1000 \
  --output_dir=outputs/eval/stage_tracker_smoke
```

In addition to the normal `eval_info.json` and videos, this writes one full
timeline per episode under `stages/`, plus `stage_episodes.jsonl` and the
aggregated `stage_summary.json`.
