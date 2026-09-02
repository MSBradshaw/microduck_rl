"""roll_split / pitch_split: pure-tensor reward functions for the splits
task. Mock env/asset pattern mirrors tests/test_descent_speed.py -- these
only ever touch env.scene[name].data.projected_gravity_b, so the fake
objects below only need that one attribute, nothing else.
"""

import torch

from mjlab_microduck.tasks.mdp import pitch_split, roll_split


class _GravityData:
    def __init__(self, gravity_b):
        self.projected_gravity_b = torch.tensor(gravity_b, dtype=torch.float32)


class _GravityAsset:
    def __init__(self, data):
        self.data = data


class _GravityEnv:
    def __init__(self, gravity_b):
        self._a = _GravityAsset(_GravityData(gravity_b))
        self.scene = self

    def __getitem__(self, _k):
        return self._a


def test_roll_split_peaks_at_zero_roll():
    # projected_gravity_b is [x, y, z] in the trunk's body frame. Upright,
    # no roll or pitch: gravity points straight down along body -z, so
    # [x, y] = [0, 0] and z = -1.
    env = _GravityEnv([[0.0, 0.0, -1.0]])
    out = roll_split(env, std=0.45)
    assert abs(float(out[0]) - 1.0) < 1e-5


def test_roll_split_is_generous_to_small_sway():
    # y=0.26 is a real lateral tilt (not upright) -- with the DELIBERATELY
    # generous std=0.45 this should still score high, per the explicit
    # design direction: "so long as it does not fall, tipping side to side
    # a little is fine."
    env = _GravityEnv([[0.0, 0.26, -0.97]])
    out = roll_split(env, std=0.45)
    assert float(out[0]) > 0.7


def test_roll_split_penalizes_large_roll():
    # A much bigger lateral tilt should score noticeably lower than a small
    # one -- the Gaussian still has to distinguish "a little sway" from
    # "actually tipping over," just with a wide std, not an infinitely wide one.
    env = _GravityEnv([[0.0, 0.8, -0.4]])
    out = roll_split(env, std=0.45)
    assert float(out[0]) < 0.2


def test_pitch_split_peaks_at_the_target_not_at_zero():
    # pitch_split tracks a CONFIGURABLE target (SPLIT_PITCH_TARGET, a design
    # default currently 0 -- see the spec), unlike roll_split's fixed
    # target of zero. This test picks a nonzero target specifically to prove
    # the function tracks THAT value, not just "close to vertical."
    target = 0.55
    env_at_target = _GravityEnv([[target, 0.0, -0.8]])
    out_at_target = pitch_split(env_at_target, target_pitch=target, std=0.15)
    assert abs(float(out_at_target[0]) - 1.0) < 1e-5

    env_vertical = _GravityEnv([[0.0, 0.0, -1.0]])
    out_at_vertical = pitch_split(env_vertical, target_pitch=target, std=0.15)
    assert float(out_at_vertical[0]) < float(out_at_target[0])
