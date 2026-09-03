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
