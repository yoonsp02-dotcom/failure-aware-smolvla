from failure_aware_smolvla.stage_tracker import Stage, StageSignals, StageTracker


def signals(
    step: int,
    *,
    ee=(0.0, 0.0, 0.80),
    obj=(0.30, 0.00, 0.75),
    target=(0.60, 0.00, 0.75),
    grasped=False,
    goal=False,
):
    return StageSignals(
        step=step,
        end_effector_pos=ee,
        object_pos=obj,
        target_pos=target,
        grasped=grasped,
        goal_satisfied=goal,
    )


def test_tracks_complete_pick_and_place_progression():
    tracker = StageTracker()

    assert tracker.reset(signals(0)).current_stage == "start"
    assert tracker.update(signals(1, ee=(0.34, 0.0, 0.75))).current_stage == "reach"
    assert tracker.update(signals(2, ee=(0.30, 0.0, 0.75), grasped=True)).current_stage == "grasp"
    assert tracker.update(
        signals(3, ee=(0.30, 0.0, 0.81), obj=(0.30, 0.0, 0.81), grasped=True)
    ).current_stage == "lift"
    assert tracker.update(
        signals(4, ee=(0.54, 0.0, 0.81), obj=(0.54, 0.0, 0.81), grasped=True)
    ).current_stage == "transport"
    assert tracker.update(
        signals(5, ee=(0.60, 0.0, 0.76), obj=(0.60, 0.0, 0.76), grasped=True)
    ).current_stage == "lower"
    assert tracker.update(signals(6, ee=(0.60, 0.0, 0.76), obj=(0.60, 0.0, 0.76), goal=True)).current_stage == "place"

    summary = tracker.summary()
    assert summary["official_success"] is True
    assert summary["max_stage_reached"] == "place"
    assert summary["first_step_by_stage"] == {
        "start": 0,
        "reach": 1,
        "grasp": 2,
        "lift": 3,
        "transport": 4,
        "lower": 5,
        "place": 6,
    }


def test_tracks_drop_regrasp_and_recovery():
    tracker = StageTracker()
    tracker.reset(signals(0))
    tracker.update(signals(1, ee=(0.30, 0.0, 0.75), grasped=True))
    tracker.update(signals(2, ee=(0.30, 0.0, 0.81), obj=(0.30, 0.0, 0.81), grasped=True))
    tracker.update(signals(3, ee=(0.30, 0.0, 0.81), obj=(0.30, 0.0, 0.81)))
    tracker.update(signals(4, ee=(0.30, 0.0, 0.75)))
    tracker.update(signals(5, ee=(0.30, 0.0, 0.75), grasped=True))
    tracker.update(signals(6, ee=(0.60, 0.0, 0.76), obj=(0.60, 0.0, 0.76), goal=True))

    event_names = [event.event for event in tracker.events]
    assert "grasp_lost" in event_names
    assert "object_dropped" in event_names
    assert "regrasped" in event_names
    assert "recovery_success" in event_names
    assert tracker.max_stage_reached is Stage.PLACE


def test_rejects_non_monotonic_steps():
    tracker = StageTracker()
    tracker.reset(signals(0))

    try:
        tracker.update(signals(0))
    except ValueError as error:
        assert "monotonically" in str(error)
    else:
        raise AssertionError("Expected duplicate step to be rejected")


def test_failed_ground_level_grasp_is_not_counted_as_a_drop():
    tracker = StageTracker()
    tracker.reset(signals(0))
    tracker.update(signals(1, ee=(0.30, 0.0, 0.75), grasped=True))
    tracker.update(signals(2, ee=(0.30, 0.0, 0.75)))

    event_names = [event.event for event in tracker.events]
    assert "grasp_lost" in event_names
    assert "object_dropped" not in event_names
