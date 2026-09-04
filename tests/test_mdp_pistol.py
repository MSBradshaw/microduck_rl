"""pistol_free_leg_clearance / posture_depth_curriculum: pure-tensor mdp
functions for the pistol squat task. Mock env pattern mirrors
tests/test_mdp_splits.py -- minimal fakes exposing only what each function
actually touches."""

import torch

from mjlab_microduck.tasks.mdp import (
    pistol_free_leg_clearance,
    posture_depth_curriculum,
)


class _SiteData:
    def __init__(self, site_pos_w):
        self.site_pos_w = site_pos_w  # (num_envs, num_sites, 3)


class _SiteAsset:
    def __init__(self, data):
        self.data = data


class _Terrain:
    def __init__(self, num_envs):
        self.env_origins = torch.zeros(num_envs, 3)


class _CommandTerm:
    """No `.alpha` attribute -- forces _posture_blend's fallback branch."""


class _CommandManager:
    def __init__(self, flag):
        self._flag = flag

    def get_term(self, _name):
        return _CommandTerm()

    def get_command(self, _name):
        return self._flag


class _ContactSensorData:
    def __init__(self, found):
        self.found = found  # (num_envs, 2) -- index 0=left foot, 1=right foot


class _ContactSensor:
    def __init__(self, found):
        self.data = _ContactSensorData(found)


class _ClearanceEnv:
    def __init__(self, right_foot_z: float, blend: float, right_contact: float = 0.0):
        num_envs = 1
        site_pos = torch.zeros(num_envs, 1, 3)
        site_pos[0, 0, 2] = right_foot_z
        self._asset = _SiteAsset(_SiteData(site_pos))
        self.scene = self
        self.terrain = _Terrain(num_envs)
        self.command_manager = _CommandManager(torch.tensor([[blend, 0.0, 0.0]]))
        # left foot's contact state is irrelevant to this reward -- 0.0 always.
        found = torch.tensor([[0.0, right_contact]])
        self.sensors = {"feet_ground_contact": _ContactSensor(found)}

    def __getitem__(self, _name):
        return self._asset


class _FakeAssetCfg:
    def __init__(self, site_ids):
        self.name = "robot"
        self.site_ids = site_ids


def test_clearance_full_reward_once_above_margin():
    env = _ClearanceEnv(right_foot_z=0.10, blend=1.0, right_contact=0.0)
    out = pistol_free_leg_clearance(
        env, command_name="twist", margin=0.03, std=0.02,
        asset_cfg=_FakeAssetCfg([0]), sensor_name="feet_ground_contact",
    )
    assert abs(float(out[0]) - 1.0) < 1e-4


def test_clearance_drops_toward_zero_when_touching_ground():
    env = _ClearanceEnv(right_foot_z=0.0, blend=1.0, right_contact=0.0)
    out = pistol_free_leg_clearance(
        env, command_name="twist", margin=0.03, std=0.02,
        asset_cfg=_FakeAssetCfg([0]), sensor_name="feet_ground_contact",
    )
    assert float(out[0]) < 0.15


def test_clearance_gated_to_zero_at_stand_blend():
    # blend=0 (fully standing, not squatting) -- inert regardless of foot
    # height, per spec §3.2 ("gated on posture blend so it's inert during
    # normal standing").
    env = _ClearanceEnv(right_foot_z=0.0, blend=0.0, right_contact=0.0)
    out = pistol_free_leg_clearance(
        env, command_name="twist", margin=0.03, std=0.02,
        asset_cfg=_FakeAssetCfg([0]), sensor_name="feet_ground_contact",
    )
    assert abs(float(out[0])) < 1e-6


def test_clearance_hard_gate_zeroes_reward_even_when_geometrically_clear():
    # The exploit this gate closes: the `right_foot` SITE sits above the
    # foot's actual sole, so a planted-but-tilted foot can read a
    # comfortably-above-margin site_z (here 0.10, same as the "full reward"
    # case above) even with the sole flat on the ground. The height-only
    # version of this reward can't see that -- only the contact sensor can.
    # A foot the sensor reports as touching must score ~0 regardless of how
    # clear it looks geometrically.
    env = _ClearanceEnv(right_foot_z=0.10, blend=1.0, right_contact=1.0)
    out = pistol_free_leg_clearance(
        env, command_name="twist", margin=0.03, std=0.02,
        asset_cfg=_FakeAssetCfg([0]), sensor_name="feet_ground_contact",
    )
    assert abs(float(out[0])) < 1e-4


class _DepthDefaultJointData:
    def __init__(self, default_joint_pos):
        self.default_joint_pos = default_joint_pos


class _DepthAsset:
    def __init__(self, data, num_joints: int):
        self.data = data
        self._num_joints = num_joints

    def find_joints(self, _pattern):
        # _servo_joint_ids calls this; identity mapping on a plain (no
        # passive_* joints) model, same as the real implementation's
        # documented behavior.
        return list(range(self._num_joints)), None


class _TermCfg:
    def __init__(self, params):
        self.params = params


class _RewardManager:
    def __init__(self, term_cfgs):
        self._term_cfgs = term_cfgs

    def get_term_cfg(self, name):
        return self._term_cfgs[name]


class _DepthEnv:
    def __init__(self, step: int, home_row: list[float]):
        self.common_step_counter = step
        default_joint_pos = torch.tensor([home_row])  # (1, num_joints)
        self._asset = _DepthAsset(_DepthDefaultJointData(default_joint_pos), len(home_row))
        self.scene = self
        self.reward_manager = _RewardManager(
            {"posture_pose_legs": _TermCfg({"sit_overrides": {}})}
        )

    def __getitem__(self, _name):
        return self._asset


class _FakeServoAssetCfg:
    name = "robot"


def test_depth_curriculum_interpolates_from_home_not_from_zero():
    # HOME=-0.4579 (index 2), full target=-0.4154 -- at fraction 0.5 the
    # override must sit HALFWAY BETWEEN those two, not at 0.5*(-0.4154).
    # This is the exact bug an earlier draft of this task had: naively
    # scaling the target value alone silently breaks for any joint whose
    # HOME angle isn't ~0 (ankle's HOME=0.453, target=0 is the worst case --
    # naive scaling would never move it at all).
    env = _DepthEnv(step=1000, home_row=[0.0] * 2 + [-0.4579, -0.0049, 0.4530] + [0.0] * 9)
    posture_depth_curriculum(
        env, env_ids=None,
        reward_names=("posture_pose_legs",),
        joint_indices=(2, 3, 4),
        full_overrides={2: -0.4154, 3: 1.1468, 4: 0.0680},
        depth_stages=[{"step": 0, "fraction": 0.0}, {"step": 500, "fraction": 0.5}],
        asset_cfg=_FakeServoAssetCfg(),
    )
    got = env.reward_manager.get_term_cfg("posture_pose_legs").params["sit_overrides"]
    expected_hip = -0.4579 + 0.5 * (-0.4154 - (-0.4579))
    expected_ankle = 0.4530 + 0.5 * (0.0680 - 0.4530)
    assert abs(got[2] - expected_hip) < 1e-4
    assert abs(got[4] - expected_ankle) < 1e-4, "ankle must move even though its target is 0"


def test_depth_curriculum_latest_passed_stage_wins():
    env = _DepthEnv(step=10_000, home_row=[0.0] * 2 + [-0.4579, -0.0049, 0.4530] + [0.0] * 9)
    posture_depth_curriculum(
        env, env_ids=None,
        reward_names=("posture_pose_legs",),
        joint_indices=(2, 3, 4),
        full_overrides={2: -0.4154, 3: 1.1468, 4: 0.0680},
        depth_stages=[
            {"step": 0, "fraction": 0.3},
            {"step": 500, "fraction": 0.6},
            {"step": 1000, "fraction": 1.0},
        ],
        asset_cfg=_FakeServoAssetCfg(),
    )
    got = env.reward_manager.get_term_cfg("posture_pose_legs").params["sit_overrides"]
    assert abs(got[2] - (-0.4154)) < 1e-4, "step 10000 is past every stage -- fraction must be 1.0"
