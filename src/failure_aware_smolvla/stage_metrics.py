"""Aggregate per-episode semantic stage files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .stage_tracker import Stage


def load_stage_episodes(stage_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(stage_dir)
    episodes = []
    for path in sorted(root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["stage_metrics_path"] = str(path)
        episodes.append(payload)
    return episodes


def aggregate_stage_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode["task_group"], int(episode["task_id"]))].append(episode)

    per_task = []
    for (task_group, task_id), task_episodes in sorted(grouped.items()):
        per_task.append(_aggregate_one(task_group, task_id, task_episodes))

    overall = _aggregate_one("overall", -1, episodes)
    overall.pop("task_group")
    overall.pop("task_id")
    return {"per_task": per_task, "overall": overall}


def write_stage_outputs(stage_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    episodes = load_stage_episodes(stage_dir)
    summary = aggregate_stage_episodes(episodes)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_root / "stage_episodes.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for episode in episodes:
            stream.write(json.dumps(episode) + "\n")

    summary_path = output_root / "stage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _aggregate_one(task_group: str, task_id: int, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    n_episodes = len(episodes)
    stage_counts = {stage.name.lower(): 0 for stage in Stage}
    event_counts: Counter[str] = Counter()
    episodes_with_event: Counter[str] = Counter()
    n_success = 0

    for episode in episodes:
        summary = episode["summary"]
        max_stage = Stage[summary["max_stage_reached"].upper()]
        for stage in Stage:
            if max_stage >= stage:
                stage_counts[stage.name.lower()] += 1
        n_success += int(bool(summary["official_success"]))
        names_in_episode = set()
        for event in summary.get("events", []):
            event_counts[event["event"]] += 1
            names_in_episode.add(event["event"])
        episodes_with_event.update(names_in_episode)

    stage_rates = {
        name: (100.0 * count / n_episodes if n_episodes else None) for name, count in stage_counts.items()
    }
    transitions = {}
    ordered = list(Stage)
    for previous, current in zip(ordered[:-1], ordered[1:], strict=True):
        denominator = stage_counts[previous.name.lower()]
        numerator = stage_counts[current.name.lower()]
        transitions[f"{previous.name.lower()}_to_{current.name.lower()}"] = (
            100.0 * numerator / denominator if denominator else None
        )

    regrasp_episodes = episodes_with_event["regrasped"]
    recovery_success_episodes = episodes_with_event["recovery_success"]
    return {
        "task_group": task_group,
        "task_id": task_id,
        "n_episodes": n_episodes,
        "n_success": n_success,
        "pc_success": 100.0 * n_success / n_episodes if n_episodes else None,
        "stage_counts": stage_counts,
        "stage_rates": stage_rates,
        "conditional_transition_rates": transitions,
        "event_counts": dict(sorted(event_counts.items())),
        "episodes_with_event": dict(sorted(episodes_with_event.items())),
        "recovery_success_rate_given_regrasp": (
            100.0 * recovery_success_episodes / regrasp_episodes if regrasp_episodes else None
        ),
    }
