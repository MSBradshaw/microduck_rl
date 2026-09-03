import math

from mjlab_microduck.tasks.microduck_splits_env_cfg import (
    _LEG_JOINTS,
    EPISODE_LENGTH_S,
    make_microduck_splits_env_cfg,
)


def test_env_builds_train_and_play():
    assert make_microduck_splits_env_cfg() is not None
    assert make_microduck_splits_env_cfg(play=True) is not None


def test_episode_length_matches_constant():
    cfg = make_microduck_splits_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S


def test_leg_joint_indices_match_standup_convention():
    # Same 14-joint layout as every other allcollisions-model task
    # (AGENTS.md: 0-4 left leg, 5-8 neck/head, 9-13 right leg).
    assert _LEG_JOINTS == [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def test_reset_is_standing_only_no_ground_state_mix():
    # Splits starts from standing, not from an on-the-ground recovery mix --
    # no standup-style prone/back spawns.
    cfg = make_microduck_splits_env_cfg()
    params = cfg.events["set_ground_state"].params
    assert params["standing_prob"] == 1.0
    assert params["face_down_prob"] == 0.0
    assert params["face_up_prob"] == 0.0
    assert params["sitting_prob"] == 0.0
    assert "ground_state_mix" not in cfg.curriculum


def test_no_fall_termination_but_nan_state_present():
    cfg = make_microduck_splits_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_roll_and_pitch_terminations_present():
    cfg = make_microduck_splits_env_cfg()
    assert "tipped_sideways" in cfg.terminations
    assert "tipped_forward_or_back" in cfg.terminations


def test_obs_is_still_61d_command_layout():
    cfg = make_microduck_splits_env_cfg()
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert "body_command" in cfg.observations[group].terms


def test_pose_and_height_rewards_present_with_positive_weights():
    cfg = make_microduck_splits_env_cfg()
    for name in ("pose_split", "pose_split_l1", "height_split", "height_split_sharp", "height_split_l1"):
        assert name in cfg.rewards
        assert cfg.rewards[name].weight > 0


def test_orientation_rewards_present():
    cfg = make_microduck_splits_env_cfg()
    assert "roll_split" in cfg.rewards
    assert "pitch_split" in cfg.rewards
    assert cfg.rewards["roll_split"].weight > 0
    assert cfg.rewards["pitch_split"].weight > 0
    # Generous roll std, per explicit design direction.
    assert cfg.rewards["roll_split"].params["std"] > cfg.rewards["pitch_split"].params["std"]


def test_no_limit_proximity_penalty_on_leg_joints():
    cfg = make_microduck_splits_env_cfg()
    assert "dof_pos_limits" not in cfg.rewards
    assert "limit_proximity" not in cfg.rewards


def test_already_negative_penalties_use_positive_weights():
    # Same sign-bug class check as standup/roller_standup's test.
    cfg = make_microduck_splits_env_cfg()
    for name in ("height_split_l1", "pose_split_l1", "gentle_descent"):
        assert cfg.rewards[name].weight > 0, f"{name} calls a function returning negative already"
    # action_rate_l2 is an mjlab-base cost (>=0) so needs a negative weight,
    # and is active (nonzero) from cfg-build time. joint_torque_rate_l2 is
    # the SAME sign class but is deliberately 0.0 at cfg-build time --
    # discovery-vs-polish staging (Task 6 curriculum ramps it negative) --
    # so it's checked separately, not here.
    assert cfg.rewards["action_rate_l2"].weight < 0


def test_settle_damping_and_torque_rate_start_at_zero():
    # Discovery-vs-polish staging (spec §5): must not tax the descent attempt
    # before the skill exists.
    cfg = make_microduck_splits_env_cfg()
    assert cfg.rewards["settle_damping"].weight == 0.0
    assert cfg.rewards["joint_torque_rate_l2"].weight == 0.0


def test_pose_rewards_target_split_joints_via_overrides():
    from mjlab_microduck.tasks.microduck_splits_env_cfg import _LEG_JOINTS, SPLIT_JOINT_OVERRIDES
    cfg = make_microduck_splits_env_cfg()
    for name in ("pose_split", "pose_split_l1"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        assert cfg.rewards[name].params["target_overrides"] == SPLIT_JOINT_OVERRIDES


def test_self_collisions_present():
    cfg = make_microduck_splits_env_cfg()
    assert "self_collisions" in cfg.rewards
    assert cfg.rewards["self_collisions"].weight < 0


def test_split_depth_curriculum_ramps_to_full_target():
    cfg = make_microduck_splits_env_cfg()
    assert "split_depth" in cfg.curriculum
    stages = cfg.curriculum["split_depth"].params["depth_stages"]
    fractions = [s["fraction"] for s in stages]
    assert fractions[0] < 1.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert cfg.curriculum["split_depth"].params["reward_names"] == ("pose_split", "pose_split_l1")


def test_discovery_vs_polish_staging_starts_at_zero():
    cfg = make_microduck_splits_env_cfg()
    for name in ("settle_damping_weight", "torque_rate_weight", "descent_speed_cap_weight"):
        assert name in cfg.curriculum
        first_stage = cfg.curriculum[name].params["weight_stages"][0]
        assert first_stage["step"] == 0
        assert first_stage["weight"] == 0.0


def test_descent_speed_cap_ramps_to_a_positive_weight():
    # descent_speed_cap wraps trunk_downward_velocity_penalty, which returns
    # -clamp(..., min=0.0) -- ALWAYS <= 0, self-negating. A negative final
    # weight here would double-negate it into a reward for violent drops
    # (the exact "bit four envs" bug class) -- this locks the sign in.
    cfg = make_microduck_splits_env_cfg()
    stages = cfg.curriculum["descent_speed_cap_weight"].params["weight_stages"]
    assert stages[-1]["weight"] > 0


def test_action_rate_curriculum_ramps_like_standup():
    cfg = make_microduck_splits_env_cfg()
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert weights[0] == -0.1
    assert weights[-1] == -1.0
    assert weights == sorted(weights, reverse=True)


def test_push_curriculum_ramps_from_zero():
    cfg = make_microduck_splits_env_cfg()
    assert "push_magnitude" in cfg.curriculum
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)


def test_task_registered_with_correct_experiment_name():
    from mjlab_microduck.tasks.microduck_splits_env_cfg import MicroduckSplitsRlCfg
    assert MicroduckSplitsRlCfg.experiment_name == "microduck_splits"


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks
    import mjlab_microduck.tasks  # noqa: F401 (import triggers registration)
    assert "Mjlab-Splits-Flat-MicroDuck" in list_tasks()
    assert "Mjlab-Splits-Flat-Backlash-MicroDuck" in list_tasks()
