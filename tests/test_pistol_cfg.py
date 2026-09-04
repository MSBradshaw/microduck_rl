import torch

from mjlab_microduck.tasks.microduck_pistol_env_cfg import (
    _STANCE_LEG_JOINTS,
    EPISODE_LENGTH_S,
    POSTURE_DWELL_S,
    POSTURE_RAMP_S,
    PISTOL_STANCE_OVERRIDES,
    PISTOL_RESET_OVERRIDES,
    make_microduck_pistol_env_cfg,
)
from mjlab_microduck.tasks.mdp import AlternatingPostureCommand, AlternatingPostureCommandCfg


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"


def _make_alternating_command(num_envs: int) -> AlternatingPostureCommand:
    cmd = object.__new__(AlternatingPostureCommand)
    cmd._env = _FakeEnv(num_envs)
    cmd.vel_command_b = torch.zeros(num_envs, 3)
    cmd._ever_resampled = torch.zeros(num_envs, dtype=torch.bool)
    return cmd


def test_env_builds_train_and_play():
    assert make_microduck_pistol_env_cfg() is not None
    assert make_microduck_pistol_env_cfg(play=True) is not None


def test_episode_length_covers_multiple_cycles():
    cfg = make_microduck_pistol_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    min_full_cycle = 2 * POSTURE_DWELL_S[0]
    assert EPISODE_LENGTH_S >= 2 * min_full_cycle


def test_dwell_exceeds_ramp_so_a_real_hold_exists():
    assert POSTURE_DWELL_S[0] > POSTURE_RAMP_S


def test_stance_leg_joints_are_left_leg_only():
    # The defining difference from every other commanded-posture task in
    # this family: only the STANCE leg gets a pose-match target.
    assert _STANCE_LEG_JOINTS == [0, 1, 2, 3, 4]


def test_stance_overrides_have_no_right_leg_entries():
    assert set(PISTOL_STANCE_OVERRIDES.keys()).isdisjoint({9, 10, 11, 12, 13})


def test_reset_overrides_include_free_leg_anchor_but_stance_overrides_dont():
    # PISTOL_RESET_OVERRIDES (used only for the reset event) must include
    # BOTH legs so a "start squatted" reset isn't a broken half-pose; the
    # reward target (PISTOL_STANCE_OVERRIDES) must NOT, since the free leg
    # is never a pose-match target (design spec §3.2).
    assert {11, 12, 13}.issubset(PISTOL_RESET_OVERRIDES.keys())
    assert set(PISTOL_STANCE_OVERRIDES.keys()).isdisjoint({11, 12, 13})


def test_command_is_alternating_posture_type():
    cfg = make_microduck_pistol_env_cfg()
    assert isinstance(cfg.commands["twist"], AlternatingPostureCommandCfg)
    assert cfg.commands["twist"].ramp_s == POSTURE_RAMP_S


def test_reset_is_a_standing_squat_mix():
    cfg = make_microduck_pistol_env_cfg()
    params = cfg.events["set_ground_state"].params
    assert params["standing_prob"] == 0.5
    assert params["sitting_prob"] == 0.5
    assert params["face_down_prob"] == 0.0
    assert params["face_up_prob"] == 0.0
    assert params["sitting_joint_overrides"] == PISTOL_RESET_OVERRIDES


def test_tip_terminations_present():
    cfg = make_microduck_pistol_env_cfg()
    assert "nan_state" in cfg.terminations
    assert "tipped_sideways" in cfg.terminations
    assert "tipped_forward_or_back" in cfg.terminations
    assert "fell_over" not in cfg.terminations


def test_obs_is_still_61d_command_layout():
    cfg = make_microduck_pistol_env_cfg()
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert "body_command" in cfg.observations[group].terms


def test_posture_rewards_present_with_positive_weights():
    cfg = make_microduck_pistol_env_cfg()
    for name in (
        "posture_pose_stance", "posture_pose_l1",
        "posture_height", "posture_height_sharp", "posture_height_l1",
        "posture_composite", "posture_stillness", "free_leg_clearance",
    ):
        assert name in cfg.rewards
        assert cfg.rewards[name].weight > 0


def test_posture_pose_and_composite_scoped_to_stance_leg_only():
    cfg = make_microduck_pistol_env_cfg()
    for name in ("posture_pose_stance", "posture_pose_l1", "posture_composite"):
        assert cfg.rewards[name].params["joint_indices"] == _STANCE_LEG_JOINTS


def test_free_leg_clearance_has_no_joint_indices_param():
    # It's a height-based reward, not a pose-match term -- it must not take
    # a joint_indices/sit_overrides param at all (design spec §3.2).
    cfg = make_microduck_pistol_env_cfg()
    params = cfg.rewards["free_leg_clearance"].params
    assert "joint_indices" not in params
    assert "sit_overrides" not in params


def test_no_limit_proximity_penalty_on_leg_joints():
    cfg = make_microduck_pistol_env_cfg()
    assert "dof_pos_limits" not in cfg.rewards
    assert "limit_proximity" not in cfg.rewards


def test_already_negative_penalties_use_positive_weights():
    cfg = make_microduck_pistol_env_cfg()
    for name in ("posture_height_l1", "posture_pose_l1", "gentle_motion", "descent_speed"):
        assert cfg.rewards[name].weight > 0, f"{name} calls a function returning negative already"
    assert cfg.rewards["action_rate_l2"].weight < 0


def test_rise_speed_starts_at_zero_ramps_up():
    cfg = make_microduck_pistol_env_cfg()
    assert cfg.rewards["rise_speed"].weight == 0.0
    stages = cfg.curriculum["rise_speed_weight"].params["weight_stages"]
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] > 0.0


def test_self_collisions_present():
    cfg = make_microduck_pistol_env_cfg()
    assert "self_collisions" in cfg.rewards
    assert cfg.rewards["self_collisions"].weight < 0


def test_stance_depth_curriculum_present_unlike_splits_cycle():
    # Unlike splits_cycle (which dropped an equivalent curriculum), this
    # task's design spec §6 calls for one explicitly.
    cfg = make_microduck_pistol_env_cfg()
    assert "stance_depth" in cfg.curriculum
    stages = cfg.curriculum["stance_depth"].params["depth_stages"]
    assert stages[0]["fraction"] < stages[-1]["fraction"] == 1.0
    assert cfg.curriculum["stance_depth"].params["full_overrides"] == PISTOL_STANCE_OVERRIDES


def test_push_curriculum_ramps_from_zero():
    cfg = make_microduck_pistol_env_cfg()
    assert "push_magnitude" in cfg.curriculum
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)


def test_task_registered_with_correct_experiment_name():
    from mjlab_microduck.tasks.microduck_pistol_env_cfg import MicroduckPistolRlCfg
    assert MicroduckPistolRlCfg.experiment_name == "microduck_pistol"


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks
    import mjlab_microduck.tasks  # noqa: F401 (import triggers registration)
    assert "Mjlab-Pistol-Flat-MicroDuck" in list_tasks()
    assert "Mjlab-Pistol-Rough-MicroDuck" in list_tasks()
    assert "Mjlab-Pistol-Flat-Backlash-MicroDuck" in list_tasks()
