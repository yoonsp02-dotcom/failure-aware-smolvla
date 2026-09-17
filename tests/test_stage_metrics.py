from failure_aware_smolvla.stage_metrics import aggregate_stage_episodes


def episode(task_id, max_stage, success=False, events=()):
    return {
        "task_group": "libero_spatial",
        "task_id": task_id,
        "summary": {
            "max_stage_reached": max_stage,
            "official_success": success,
            "events": [{"event": event} for event in events],
        },
    }


def test_aggregates_stage_and_recovery_metrics():
    result = aggregate_stage_episodes(
        [
            episode(0, "place", success=True),
            episode(0, "lift", events=("grasp_lost", "object_dropped")),
            episode(0, "place", success=True, events=("grasp_lost", "regrasped", "recovery_success")),
        ]
    )

    task = result["per_task"][0]
    assert task["n_episodes"] == 3
    assert task["n_success"] == 2
    assert task["stage_counts"]["grasp"] == 3
    assert task["stage_counts"]["transport"] == 2
    assert task["episodes_with_event"]["grasp_lost"] == 2
    assert task["recovery_success_rate_given_regrasp"] == 100.0
