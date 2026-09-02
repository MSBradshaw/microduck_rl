"""Kinematic pose measurement for candidate front-split leg targets.

Same technique as scripts/crouch_pose_editor.py: gravity OFF, the free-joint
base pinned upright at a fixed xy/orientation while the leg actuators drive
to their targets, then the whole robot is lowered until its lowest point
touches the floor (z=0) -- exactly what "the robot is resting on the ground
in this pose" means geometrically.

Why not real gravity/dynamics? The robot's actual servos are very low-
stiffness (kp=0.55, forcerange +-0.96 N*m -- see joints_properties.xml) and
only hold a pose through a POLICY's continuous active correction, not
passive stiffness. Standup's STAND_Z was measured "via the velocity policy
holding the robot still" -- an already-trained policy holding position. We
don't have a trained splits policy yet, so there's no active controller to
lean on; a raw ctrl-and-step simulation just sags under gravity (confirmed:
even the plain standing HOME pose settled at z=0.041 instead of 0.115 when
tried with real gravity and no active correction). Going kinematic sidesteps
that entirely -- at the cost of no longer being able to detect an actual
topple; self-collision is checked instead, since it's the other real risk a
front split raises that we CAN check without dynamics.

Usage:
    uv run python scripts/measure_split_pose.py
"""

import re

import mujoco
import numpy as np

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_standup_spec


def compile_model_with_floor() -> mujoco.MjModel:
    """The standup model (get_standup_spec) has no ground plane -- mjlab adds
    terrain separately at the RL-env level (see cfg.scene.terrain in
    microduck_standup_env_cfg.py). Add one geom, same as scene_walk.xml's
    <geom name="floor" type="plane" pos="0 0 0">, so "touching the floor"
    means something here too.
    """
    spec = get_standup_spec()
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05),
        pos=(0.0, 0.0, 0.0),
    )
    return spec.compile()


def _home_ctrl(model: mujoco.MjModel) -> np.ndarray:
    """Build the ctrl array MuJoCo needs, one value per actuator, from HOME.

    HOME_FRAME.joint_pos is a dict of {regex_pattern: angle_rad} -- e.g.
    r".*left_hip_pitch.*" -> -0.4579. We look up each actuator's NAME
    against those patterns to find its resting angle. Actuators are
    "position" actuators here, meaning ctrl[i] IS the target angle for
    actuator i (not a torque or force).
    """
    ctrl = np.zeros(model.nu)  # model.nu = number of actuators
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        for pattern, angle in HOME_FRAME.joint_pos.items():
            if re.search(pattern, name):
                ctrl[i] = float(angle)
                break
    return ctrl


def settle_split_pose(
    model: mujoco.MjModel,
    leg_overrides: dict[str, float],
    settle_steps: int = 300,
) -> dict:
    """Drive `leg_overrides` from HOME, then report the resulting geometry.

    leg_overrides: {actuator_name: target_angle_rad}. Any actuator NOT
    named here keeps its HOME angle. E.g. {"left_hip_pitch": 1.2} moves
    only that one joint; everything else (right leg, neck, head) stays at
    its normal standing target.
    """
    # MjData holds the SIMULATION STATE (positions, velocities, forces) --
    # separate from MjModel, which is the fixed physics description. You
    # get a fresh MjData per run; mj_resetData zeroes it out.
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    model.opt.gravity[:] = [0.0, 0.0, 0.0]  # kinematic: nothing should fall

    ctrl = _home_ctrl(model)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if name in leg_overrides:
            ctrl[i] = leg_overrides[name]
    data.ctrl[:] = ctrl

    # Pin the free-joint base at a fixed, floor-clear height and identity
    # (upright) orientation. qpos[0:3] is the base xyz, qpos[3:7] is the
    # base orientation quaternion (w, x, y, z) -- both come from the free
    # joint every articulated MuJoCo robot has at qpos indices 0-6.
    data.qpos[0:3] = [0.0, 0.0, 0.5]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0

    for _ in range(settle_steps):
        # Re-pin xy/orientation every step: mj_step still applies contact
        # forces (e.g. a leg swinging through the floor before it clears
        # 0.5m), and with gravity off nothing else would correct drift.
        data.qpos[0:2] = [0.0, 0.0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)  # actuators converge toward ctrl targets

    # Lower the whole robot until its lowest point touches z=0 -- i.e.
    # "resting on the floor in this pose". geom_rbound (each geom's
    # bounding-SPHERE radius) is what crouch_pose_editor.py uses for this,
    # but a bounding sphere is a conservative overestimate of a capsule/box
    # geom's actual extent -- it stops the descent too early. Binary-search
    # the base height instead, using MuJoCo's real (exact-geometry) contact
    # detection at each candidate height: mj_forward's collision pass uses
    # the true geom shapes, not bounding spheres.
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_geoms = [
        g for g in range(model.ngeom)
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
    ]

    def _touches_floor(z: float) -> bool:
        data.qpos[2] = z
        mujoco.mj_forward(model, data)
        return any(
            floor_id in (data.contact[i].geom1, data.contact[i].geom2)
            for i in range(data.ncon)
        )

    hi, lo = float(data.qpos[2]), -0.5  # hi: known clear of the floor; lo: known penetrating
    for _ in range(40):  # ~40 halvings over a 1m span is well under 1 micrometer
        mid = (hi + lo) / 2.0
        if _touches_floor(mid):
            lo = mid
        else:
            hi = mid
    data.qpos[2] = lo
    mujoco.mj_forward(model, data)

    # Self-collision: do any two robot geoms (not counting the floor)
    # actually overlap in this pose? A front split swings the legs
    # somewhere they never go in normal standing/walking -- a real risk
    # the design spec calls out. This is what "fell" meant to catch
    # before going kinematic (no gravity -> nothing dynamically falls).
    self_collision = any(
        data.contact[i].geom1 in robot_geoms and data.contact[i].geom2 in robot_geoms
        for i in range(data.ncon)
    )

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    # "left_foot"/"right_foot" are SITES (massless reference markers on the
    # sole), not bodies -- mj_name2id(..., OBJ_BODY, "left_foot") silently
    # returns -1 for both (no error), and data.xpos[-1] is a VALID numpy
    # index (last row), so both "feet" quietly read the same wrong body
    # instead of crashing. Caught by the sign-check test returning exactly
    # 0.0 difference between supposedly different feet.
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    return {
        "z": float(data.xpos[trunk_id][2]),
        "left_foot_x": float(data.site_xpos[left_foot_id][0]),
        "right_foot_x": float(data.site_xpos[right_foot_id][0]),
        "self_collision": self_collision,
    }


if __name__ == "__main__":
    model = compile_model_with_floor()
    result = settle_split_pose(model, leg_overrides={})
    for key, value in result.items():
        print(f"{key}: {value}")
