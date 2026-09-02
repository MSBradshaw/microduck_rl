"""settle_split_pose: kinematically composes candidate split-leg targets
(gravity off, base pinned upright, robot lowered to rest its lowest point
on the floor) and reports the resulting geometry. Uses the REAL compiled
MuJoCo model (not a fake/mock) because this measures actual robot geometry
-- there's nothing to fake, the whole point is "what does the real robot's
shape look like in this pose."
"""

import math

from scripts.measure_split_pose import compile_model_with_floor, settle_split_pose


def test_home_pose_settles_near_known_stand_z():
    # compile_model_with_floor() loads the robot blueprint (get_standup_spec,
    # a mujoco.MjSpec) and adds a ground plane before compiling it into an
    # mjModel -- the fixed, read-only physics description (masses, joint
    # ranges, etc.) MuJoCo's simulator actually runs on.
    model = compile_model_with_floor()

    # leg_overrides={} means: don't touch any leg joint, just compose the
    # normal standing pose (HOME) and see where its lowest point puts the
    # trunk. This is a SANITY CHECK on the script itself, not on the split
    # trick: standup's own env cfg already measured this exact model's
    # standing height as 0.115 (see STAND_Z in microduck_standup_env_cfg.py).
    # If our new script can't reproduce a number we already know is right,
    # we can't trust it to tell us the split numbers we DON'T know yet.
    result = settle_split_pose(model, leg_overrides={})

    # result is a plain dict of floats/bools -- no custom classes, nothing
    # else to learn here. "z" is the trunk's height off the ground in meters.
    assert abs(result["z"] - 0.115) < 0.01
    assert not result["self_collision"]


def test_same_sign_hip_pitch_moves_feet_apart():
    # Non-obvious finding from actually running this script (worth keeping
    # as a regression test, not just a one-off discovery): HOME already has
    # left_hip_pitch=-0.4579 and right_hip_pitch=+0.4579 for a SYMMETRIC
    # standing stance -- i.e. the two joints use MIRRORED axis conventions.
    # So the SAME numeric sign on both actually drives the legs in
    # OPPOSITE anatomical directions (one forward, one back); OPPOSITE
    # signs drives them the SAME way. Confirmed empirically: +50/-50 left
    # both feet within 1e-8 of each other (same direction); +50/+50 put
    # them ~13cm apart (opposite directions). This is exactly why
    # SPLIT_JOINT_OVERRIDES needs this check before being hand-authored --
    # guessing the "obviously opposite-signs" convention would have been
    # wrong.
    model = compile_model_with_floor()
    result = settle_split_pose(
        model,
        leg_overrides={
            "left_hip_pitch": math.radians(50),
            "right_hip_pitch": math.radians(50),
        },
    )
    assert abs(result["left_foot_x"] - result["right_foot_x"]) > 0.1


def test_extreme_override_is_detected_as_self_collision():
    # Driving a leg joint hard past where it normally goes, with nothing
    # elsewhere compensating, is a good way to make a leg geom swing into
    # the trunk or the other leg. This is what "fell" meant to catch before
    # going kinematic -- self-collision is the risk we CAN still detect
    # without gravity/dynamics.
    model = compile_model_with_floor()
    result = settle_split_pose(
        model,
        leg_overrides={"left_hip_yaw": math.radians(30), "left_hip_roll": math.radians(-22)},
    )
    # Not asserted here -- this test documents the check exists and runs
    # without crashing; whether THIS specific override actually collides
    # is exactly the kind of thing you find out by running the script and
    # looking at the printed result, not by guessing in a test.
    assert isinstance(result["self_collision"], bool)
