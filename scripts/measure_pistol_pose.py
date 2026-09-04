"""Kinematic pose measurement for candidate pistol-squat poses (left leg
stance/squat, right leg lifted clear of the ground) -- same technique as
scripts/measure_split_pose.py: gravity off, base pinned upright, then
lowered until the lowest point rests on the floor.

Adds one check splits never needed: does the CoM's xy projection actually
land within the STANCE FOOT's footprint? Splits distributes weight across
two feet on the ground, so this was never the deciding question there --
for single-leg support it's the central one. Computed from the sole mesh's
real vertex bounds transformed to world frame at the settled pose (not a
bounding-sphere approximation), compared against data.subtree_com[0] (the
whole robot's CoM, MuJoCo's own computation, not summed by hand).
"""

import re

import mujoco
import numpy as np

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_standup_spec


def compile_model_with_floor() -> mujoco.MjModel:
    spec = get_standup_spec()
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0.0, 0.0, 0.05), pos=(0.0, 0.0, 0.0),
    )
    return spec.compile()


def _home_ctrl(model: mujoco.MjModel) -> np.ndarray:
    ctrl = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        for pattern, angle in HOME_FRAME.joint_pos.items():
            if re.search(pattern, name):
                ctrl[i] = float(angle)
                break
    return ctrl


def _geom_world_footprint(model, data, geom_name) -> np.ndarray:
    """World-frame XY of every vertex of geom_name's mesh -- the real
    footprint polygon, not a bounding-sphere/box approximation."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[gid]
    vstart, vcount = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    local_verts = model.mesh_vert[vstart:vstart + vcount]  # (N,3), mesh-local frame
    xpos, xmat = data.geom_xpos[gid], data.geom_xmat[gid].reshape(3, 3)
    world_verts = local_verts @ xmat.T + xpos
    return world_verts[:, :2]


def settle_pistol_pose(model: mujoco.MjModel, leg_overrides: dict[str, float],
                        settle_steps: int = 300) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    model.opt.gravity[:] = [0.0, 0.0, 0.0]

    ctrl = _home_ctrl(model)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if name in leg_overrides:
            ctrl[i] = leg_overrides[name]
    data.ctrl[:] = ctrl

    data.qpos[0:3] = [0.0, 0.0, 0.5]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    for _ in range(settle_steps):
        data.qpos[0:2] = [0.0, 0.0]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[0:6] = 0.0
        mujoco.mj_step(model, data)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_geoms = [g for g in range(model.ngeom) if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]

    def _touches_floor(z):
        data.qpos[2] = z
        mujoco.mj_forward(model, data)
        return any(floor_id in (data.contact[i].geom1, data.contact[i].geom2) for i in range(data.ncon))

    hi, lo = float(data.qpos[2]), -0.5
    for _ in range(40):
        mid = (hi + lo) / 2.0
        if _touches_floor(mid):
            lo = mid
        else:
            hi = mid
    data.qpos[2] = lo
    mujoco.mj_forward(model, data)

    self_collision = any(
        data.contact[i].geom1 in robot_geoms and data.contact[i].geom2 in robot_geoms
        for i in range(data.ncon)
    )
    # Which geom(s) actually touch the floor? Should be ONLY left_foot_collision.
    floor_contacts = set()
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 == floor_id:
            floor_contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2))
        elif g2 == floor_id:
            floor_contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1))

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    right_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    left_foot_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")

    com_xy = data.subtree_com[0][:2].copy()  # whole-robot CoM, body 0 = world root
    footprint = _geom_world_footprint(model, data, "left_foot_collision")
    fp_min, fp_max = footprint.min(axis=0), footprint.max(axis=0)
    com_in_footprint = bool((fp_min[0] <= com_xy[0] <= fp_max[0]) and (fp_min[1] <= com_xy[1] <= fp_max[1]))
    # margin: distance from CoM to the nearest footprint edge, signed
    # negative if outside (worst-case per axis, conservative)
    margin_x = min(com_xy[0] - fp_min[0], fp_max[0] - com_xy[0])
    margin_y = min(com_xy[1] - fp_min[1], fp_max[1] - com_xy[1])

    return {
        "trunk_z": float(data.xpos[trunk_id][2]),
        "self_collision": self_collision,
        "floor_contacts": floor_contacts,
        "right_foot_z": float(data.site_xpos[right_foot_site][2]),
        "left_foot_z": float(data.site_xpos[left_foot_site][2]),
        "com_xy": com_xy.tolist(),
        "footprint_x_range": [float(fp_min[0]), float(fp_max[0])],
        "footprint_y_range": [float(fp_min[1]), float(fp_max[1])],
        "com_in_footprint": com_in_footprint,
        "margin_x_mm": float(margin_x * 1000),
        "margin_y_mm": float(margin_y * 1000),
    }


if __name__ == "__main__":
    model = compile_model_with_floor()

    # Left (stance/squat) leg: NOT invented from scratch -- an earlier
    # attempt scaling hip_pitch/knee/ankle together barely lowered the
    # trunk at all (the formula canceled itself out geometrically). This
    # reuses sitstand's actual stability-swept SIT keyframe for the left
    # leg (microduck_sitstand_env_cfg.py's SITTING_TARGET_OVERRIDES,
    # tilt-verified 2026-07-27) as the depth-1.0 endpoint, blended from
    # HOME by depth fraction alpha -- see design spec
    # docs/superpowers/specs/2026-09-04-microduck-pistol-squat-design.md §2.
    HOME_L = {"left_hip_pitch": -0.4579, "left_knee": -0.0049, "left_ankle": 0.4530}
    SIT_L = {"left_hip_pitch": -0.4079, "left_knee": 1.35, "left_ankle": 0.0}

    def blend_left(alpha: float) -> dict:
        return {k: HOME_L[k] + alpha * (SIT_L[k] - HOME_L[k]) for k in HOME_L}

    # Right (free) leg: NOT a reward target (§3.2 of the spec uses a
    # foot-clearance gate, not pose-matching) -- these are candidate
    # RESET/INIT anchor poses only. R3 is the recommended one: stays
    # self-collision-free through alpha=0.85 (the curriculum ceiling --
    # alpha=1.0 self-collides against a lifted right leg regardless of
    # which of these three is used).
    RIGHT_CANDIDATES = {
        "R1 hip.4 knee0 ankle0":    {"right_hip_pitch": 0.4, "right_knee": 0.0, "right_ankle": 0.0},
        "R2 hip.6 knee-.2 ankle0":  {"right_hip_pitch": 0.6, "right_knee": -0.2, "right_ankle": 0.0},
        "R3 hip.8 knee-.4 ankle.2": {"right_hip_pitch": 0.8, "right_knee": -0.4, "right_ankle": 0.2},
    }

    print(f"{'alpha':>6} {'right leg':>26} | {'trunk_z':>8} | {'l_foot_z':>9} | {'r_foot_z':>9} | "
          f"{'floor contact':>24} | {'self-col':>8} | {'CoM in FP':>9} | {'margin x/y (mm)':>16}")
    for alpha in (0.3, 0.6, 0.85, 1.0):
        left = blend_left(alpha)
        for label, right in RIGHT_CANDIDATES.items():
            overrides = dict(left)
            overrides.update(right)
            r = settle_pistol_pose(model, overrides)
            print(f"{alpha:>6.2f} {label:>26} | {r['trunk_z']:>8.4f} | {r['left_foot_z']:>9.4f} | "
                  f"{r['right_foot_z']:>9.4f} | {str(r['floor_contacts']):>24} | "
                  f"{str(r['self_collision']):>8} | {str(r['com_in_footprint']):>9} | "
                  f"{r['margin_x_mm']:>7.1f}/{r['margin_y_mm']:<7.1f}")
