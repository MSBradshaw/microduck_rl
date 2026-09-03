import torch

from mjlab_microduck.tasks.microduck_splits_cycle_env_cfg import (
    _LEG_JOINTS,
    EPISODE_LENGTH_S,
    POSTURE_DWELL_S,
    POSTURE_RAMP_S,
    SPLIT_JOINT_OVERRIDES,
    make_microduck_splits_cycle_env_cfg,
)
from mjlab_microduck.tasks.mdp import AlternatingPostureCommand, AlternatingPostureCommandCfg


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"


def _make_alternating_command(num_envs: int) -> AlternatingPostureCommand:
    # AlternatingPostureCommand._resample_command only touches vel_command_b /
    # num_envs / device / _ever_resampled -- num_envs/device are read-only
    # properties on the mjlab base class backed by `self._env` -- so a bare
    # object (bypassing SitStandCommand.__init__, which needs a real
    # ManagerBasedRlEnv) plus a minimal fake env is enough to unit test the
    # alternation logic in isolation, same CPU-only, no-env philosophy as
    # every other cfg test in this suite.
    cmd = object.__new__(AlternatingPostureCommand)
    cmd._env = _FakeEnv(num_envs)
    cmd.vel_command_b = torch.zeros(num_envs, 3)
    cmd._ever_resampled = torch.zeros(num_envs, dtype=torch.bool)
    return cmd


def test_first_resample_is_random_not_forced():
    torch.manual_seed(0)
    cmd = _make_alternating_command(200)
    cmd._resample_command(torch.arange(200))
    flags = cmd.vel_command_b[:, 0]
    assert set(flags.unique().tolist()) == {0.0, 1.0}
    frac = flags.mean().item()
    assert 0.3 < frac < 0.7  # roughly balanced, not forced to one value


def test_every_resample_after_the_first_flips():
    cmd = _make_alternating_command(50)
    ids = torch.arange(50)
    cmd._resample_command(ids)
    prev = cmd.vel_command_b[:, 0].clone()
    for _ in range(10):
        cmd._resample_command(ids)
        cur = cmd.vel_command_b[:, 0].clone()
        assert torch.all(cur == (1.0 - prev)), "resample must always flip, never repeat"
        prev = cur


def test_env_builds_train_and_play():
    assert make_microduck_splits_cycle_env_cfg() is not None
    assert make_microduck_splits_cycle_env_cfg(play=True) is not None


def test_episode_length_covers_multiple_cycles():
    cfg = make_microduck_splits_cycle_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    # A full cycle is two dwell periods; the episode should fit several.
    min_full_cycle = 2 * POSTURE_DWELL_S[0]
    assert EPISODE_LENGTH_S >= 2 * min_full_cycle


def test_dwell_exceeds_ramp_so_a_real_hold_exists():
    # If dwell <= ramp, the command could flip before the target blend ever
    # finishes traversing STAND<->SPLIT -- no hold at either end.
    assert POSTURE_DWELL_S[0] > POSTURE_RAMP_S


def test_leg_joint_indices_match_standup_convention():
    assert _LEG_JOINTS == [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def test_command_is_alternating_posture_type():
    cfg = make_microduck_splits_cycle_env_cfg()
    assert isinstance(cfg.commands["twist"], AlternatingPostureCommandCfg)
    assert cfg.commands["twist"].ramp_s == POSTURE_RAMP_S


def test_reset_is_a_standing_split_mix():
    cfg = make_microduck_splits_cycle_env_cfg()
    params = cfg.events["set_ground_state"].params
    assert params["standing_prob"] == 0.5
    assert params["sitting_prob"] == 0.5
    assert params["face_down_prob"] == 0.0
    assert params["face_up_prob"] == 0.0
    assert params["sitting_joint_overrides"] == SPLIT_JOINT_OVERRIDES


def test_tip_terminations_present_unlike_sitstand():
    # Unlike sitstand (which drops fall termination entirely), splits-cycle
    # keeps v1's soft tip-band terminations as a safety backstop.
    cfg = make_microduck_splits_cycle_env_cfg()
    assert "nan_state" in cfg.terminations
    assert "tipped_sideways" in cfg.terminations
    assert "tipped_forward_or_back" in cfg.terminations
    assert "fell_over" not in cfg.terminations


def test_obs_is_still_61d_command_layout():
    cfg = make_microduck_splits_cycle_env_cfg()
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert "body_command" in cfg.observations[group].terms


def test_posture_rewards_present_with_positive_weights():
    cfg = make_microduck_splits_cycle_env_cfg()
    for name in (
        "posture_pose_legs", "posture_pose_l1",
        "posture_height", "posture_height_sharp", "posture_height_l1",
        "posture_composite", "posture_stillness",
    ):
        assert name in cfg.rewards
        assert cfg.rewards[name].weight > 0


def test_orientation_rewards_reused_from_splits_v1():
    cfg = make_microduck_splits_cycle_env_cfg()
    assert "roll_split" in cfg.rewards
    assert "pitch_split" in cfg.rewards
    assert cfg.rewards["roll_split"].weight > 0
    assert cfg.rewards["pitch_split"].weight > 0
    assert cfg.rewards["roll_split"].params["std"] > cfg.rewards["pitch_split"].params["std"]


def test_no_limit_proximity_penalty_on_leg_joints():
    cfg = make_microduck_splits_cycle_env_cfg()
    assert "dof_pos_limits" not in cfg.rewards
    assert "limit_proximity" not in cfg.rewards


def test_already_negative_penalties_use_positive_weights():
    cfg = make_microduck_splits_cycle_env_cfg()
    for name in ("posture_height_l1", "posture_pose_l1", "gentle_motion", "descent_speed"):
        assert cfg.rewards[name].weight > 0, f"{name} calls a function returning negative already"
    assert cfg.rewards["action_rate_l2"].weight < 0


def test_rise_speed_starts_at_zero_ramps_up():
    # Attempt-tax lesson (standup/sitstand): a motion-tax active while the
    # rise is still being discovered makes exploratory attempts net-negative.
    cfg = make_microduck_splits_cycle_env_cfg()
    assert cfg.rewards["rise_speed"].weight == 0.0
    stages = cfg.curriculum["rise_speed_weight"].params["weight_stages"]
    assert stages[0]["weight"] == 0.0
    assert stages[-1]["weight"] > 0.0


def test_self_collisions_present():
    cfg = make_microduck_splits_cycle_env_cfg()
    assert "self_collisions" in cfg.rewards
    assert cfg.rewards["self_collisions"].weight < 0


def test_no_split_depth_curriculum():
    # Deliberately dropped vs v1 -- the per-transition alpha ramp already
    # gives repeated graduated depth exposure every cycle.
    cfg = make_microduck_splits_cycle_env_cfg()
    assert "split_depth" not in cfg.curriculum
    assert "settle_damping" not in cfg.rewards


def test_push_curriculum_ramps_from_zero():
    cfg = make_microduck_splits_cycle_env_cfg()
    assert "push_magnitude" in cfg.curriculum
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)


def test_task_registered_with_correct_experiment_name():
    from mjlab_microduck.tasks.microduck_splits_cycle_env_cfg import MicroduckSplitsCycleRlCfg
    assert MicroduckSplitsCycleRlCfg.experiment_name == "microduck_splits_cycle"


def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks
    import mjlab_microduck.tasks  # noqa: F401 (import triggers registration)
    assert "Mjlab-SplitsCycle-Flat-MicroDuck" in list_tasks()
    assert "Mjlab-SplitsCycle-Rough-MicroDuck" in list_tasks()
    assert "Mjlab-SplitsCycle-Flat-Backlash-MicroDuck" in list_tasks()
