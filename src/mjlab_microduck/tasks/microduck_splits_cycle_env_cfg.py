"""Microduck *splits-cycle* task — commanded, ALTERNATING split <-> stand.

v1 (`microduck_splits_env_cfg.py`) was a single episodic descend-and-hold:
robot starts standing, drops into a front split once, holds it, episode
ends. It worked, but "reach the split once and stop" is a shallow skill.
This is the sequel: one policy that keeps cycling split -> stand -> split
-> ... for the whole episode, commanded like `sitstand`'s sit <-> stand
task, but with the flag DETERMINISTICALLY ALTERNATING every dwell period
(`mdp.AlternatingPostureCommandCfg`) instead of `sitstand`'s independent
50/50 draw -- the whole point here is that it never stops moving, so a
command that could occasionally repeat the same posture back-to-back would
undercut that.

Design lineage -- this file is a splice of two proven pieces, not a fresh
design:
  - FROM `sitstand`: the entire commanded-posture MACHINERY. Its
    `SitStandCommand`/posture_* reward functions in mdp.py are already
    fully generic (parameterized by `sit_overrides`/`sit_z`/`stand_z`, not
    hardcoded to sit semantics) -- "sit" is just "posture B", and SPLIT
    slots into that role unchanged. Reused as-is: posture_pose_match/l1,
    posture_height_gaussian/l1, posture_composite, posture_stillness,
    posture_rise_bootstrap, the descent/rise speed caps + |a_z| shock
    penalty, the reset-state mix mechanics, DR/obs/regularizer set, and
    (new here) `AlternatingPostureCommandCfg` for guaranteed alternation.
  - FROM splits v1: the MEASURED split target (`SPLIT_JOINT_OVERRIDES`,
    `SPLIT_Z`, both from `scripts/measure_split_pose.py` -- never re-guess
    these, see AGENTS.md) and its orientation shaping (`roll_split`/
    `pitch_split` -- generous roll tolerance, tight pitch around the
    measured/design target). These target pitch=0, i.e. the SAME
    orientation goal as standing upright, so unlike sitstand's generic
    posture-independent `upright_linear`/`upright_while_tall` this task
    keeps v1's asymmetric-tolerant pair instead (splits sits closer to a
    real tip-over than sit ever does, so the wider-roll/tighter-pitch
    shaping earned in v1 stays worth keeping). Also kept: v1's soft tip-band
    TERMINATIONS as a safety backstop -- unlike sitstand, which drops fall
    termination entirely because sit's failure modes are gentle, letting a
    bad splits rollout play out repeatedly would poison a lot of the
    cycling data with garbage.

Deliberately DROPPED from v1: the `split_depth` curriculum (ramping the
reward's target fraction from 0.55 -> 1.0 across training). v1 needed it
because a single-shot episode only ever sees the target once; here every
transition already re-runs a 0->1 depth ramp (the `POSTURE_RAMP_S`-second
alpha blend), repeated ~3 times an episode across the whole run -- that
repeated graduated exposure is a reasonable bet to cover the same need.
Revisit (re-add a depth curriculum, this time ramping `sit_overrides`
fractionally) if training telemetry shows the policy stalling short of
full depth.

Also dropped: v1's `settle_damping` (body_ang_vel_at_height, curriculum-
ramped once "arrived"). `posture_stillness` (from sitstand) already pays
for stillness at the commanded posture and is gated on the ramp being
complete, so it covers the same job without a second, redundant damping
term.

Head pose command stays FIXED SMALL range (v1's choice, not sitstand's
widening curriculum) -- commandable head tracking is a separate feature
this task isn't trying to add.

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
"""

import math
from copy import deepcopy

# Symmetry — OFF. Deliberately asymmetric split target (left leg forward,
# right leg back); a mirror loss would train the policy to fight it.
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to splits v1 / sitstand for sim2real parity) ─
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S            = (3.0, 6.0)
VELOCITY_PUSH_RANGE                 = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: long enough for ~3 full split<->stand cycles (see POSTURE_DWELL_S/
# POSTURE_RAMP_S below), so the policy trains SUSTAINED rhythm, not one
# transition.
EPISODE_LENGTH_S = 24.0

# Dwell time before the alternating command may flip again. Must exceed
# POSTURE_RAMP_S so there's a real hold, not a ramp cut short by the next
# flip: 2.0s ramp + 1.0-2.0s hold => ~3.5-4.0s half-cycle, ~7-8s full cycle.
POSTURE_DWELL_S = (3.0, 4.0)
# Seconds for the internal target blend to traverse STAND<->SPLIT. Reused
# from sitstand's proven value as a starting point (comparable single-joint
# travel magnitude to sit's knee/hip swing).
POSTURE_RAMP_S = 2.0

# Empirically-measured standing trunk height (same model as splits v1/sitstand).
STAND_Z = 0.115

# ── Split target — FROM splits v1 Task 1's measure_split_pose.py output ──────
# 75 deg hip_pitch, both legs, SAME numeric sign (HOME already mirrors
# left/right hip_pitch at -0.4579/+0.4579, so a same-sign override pulls the
# legs apart). Knee/ankle left at HOME. See splits v1's docstring / design
# spec for the full derivation.
SPLIT_Z = 0.098
SPLIT_JOINT_OVERRIDES = {
    2:  -1.309,   # left_hip_pitch  (-75 deg)
    11: -1.309,   # right_hip_pitch (-75 deg, same sign as left)
}
SPLIT_PITCH_TARGET = 0.0  # design default, matches STAND's upright target too

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

# Vertical-speed caps (m/s) — BACKSTOPS for overshoot/bounce around the
# slewed target (the ramp is the primary gentleness mechanism, same as
# sitstand). Values reused from sitstand as a starting point.
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED    = 0.08

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlModelCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
    HEAD_BODY_NAMES,
    HEAD_POSE_CMD_RESAMPLE_S,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_splits_cycle_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck splits-cycle environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Rewards: posture-conditioned single-target stack (from sitstand) ──────
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=4.0,
        params={
            "command_name":  "twist",
            "std":           0.5,
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SPLIT_JOINT_OVERRIDES,
        },
    )
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1,
        weight=1.0,
        params={
            "command_name":  "twist",
            "joint_indices": _LEG_JOINTS,
            "sit_overrides": SPLIT_JOINT_OVERRIDES,
        },
    )

    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={"command_name": "twist", "sit_z": SPLIT_Z, "stand_z": STAND_Z, "std": 0.04},
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={"command_name": "twist", "sit_z": SPLIT_Z, "stand_z": STAND_Z, "std": 0.015},
    )
    # Weight 7.5, NOT sitstand's 6.0 — splits v1's own tuned value for this
    # geometry: the split's total height travel (17mm) is much smaller than
    # sit's (55mm), so the L1 term needs to be punchier to stay meaningful at
    # that scale. Reusing the value already validated for THIS target, not
    # the analogous term from a different-scale task.
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=7.5,
        params={"command_name": "twist", "sit_z": SPLIT_Z, "stand_z": STAND_Z},
    )

    cfg.rewards["rise_bootstrap"] = RewardTermCfg(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={"command_name": "twist", "max_height": 0.125, "max_vz": MAX_RISE_SPEED},
    )

    # ── Orientation shaping — FROM splits v1, unchanged ────────────────────────
    # Both commanded postures target pitch=0 (SPLIT_PITCH_TARGET is a design
    # default equal to standing's own upright target), so this pair works
    # identically well in both postures — no posture-conditioning needed.
    # Roll: generous std (mild sideways sway is fine). Pitch: tighter, tracks
    # the measured/design resting pitch, not just "upright" in general —
    # kept instead of sitstand's generic upright_linear/upright_while_tall
    # because splits sits closer to a real tip-over than sit does.
    cfg.rewards["roll_split"] = RewardTermCfg(
        func=microduck_mdp.roll_split,
        weight=0.75,
        params={"std": 0.45, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["pitch_split"] = RewardTermCfg(
        func=microduck_mdp.pitch_split,
        weight=1.5,
        params={
            "target_pitch": SPLIT_PITCH_TARGET,
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Stillness at the commanded posture (sitstand's mechanism) — supersedes
    # v1's separate `settle_damping` curriculum term; no need for both.
    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name": "twist",
            "sit_z":         SPLIT_Z,
            "stand_z":       STAND_Z,
            "band_full":     0.012,
            "band_zero":     0.03,
            "vel_std":       0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )

    # Multiplicative goal score vs the commanded posture — kills partial-sum
    # farming (plank/flop/lean/park-short) in both postures.
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name":  "twist",
            "sit_overrides": SPLIT_JOINT_OVERRIDES,
            "joint_indices": _LEG_JOINTS,
            "sit_z":         SPLIT_Z,
            "stand_z":       STAND_Z,
            "height_std":    0.03,
            "upright_std":   0.40,
            "pose_std":      0.40,
            "head_std":      0.40,
        },
    )

    # ── Gentleness (from sitstand) ─────────────────────────────────────────────
    # ⚠️ POSITIVE weights, deliberately: these functions already return
    # negative values (-clamp(...), -|a_z|) — same sign class as the
    # *_l1_penalty helpers. Double-negating with a negative weight here turns
    # violence into a reward (AGENTS.md's "bit four envs" bug class). After
    # any reward change, check wandb Episode_Reward/<penalty> stays <= 0.
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={
            "max_down_vel": MAX_DESCENT_SPEED,
            "asset_cfg":    SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["rise_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_upward_velocity_penalty,
        weight=0.0,
        params={
            "max_up_vel": MAX_RISE_SPEED,
            "asset_cfg":  SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["gentle_motion"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # ── Sim2real regularisers (velocity's set, matched) ────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)
    # No limit-proximity penalty on the split-leg joints (splits v1's call):
    # the goal legitimately sits near the hip_pitch range limit.
    cfg.rewards.pop("dof_pos_limits", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (identical layout to walking / sitstand / splits v1) ─────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # ── Head pose command (HOME-tracking only, splits v1's choice — not ─────
    # sitstand's widening curriculum; commandable head range is a separate
    # feature this task isn't trying to add).
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),
            (-0.05, 0.05),
            (-0.07, 0.07),
            (-0.015, 0.015),
        ),
    )

    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: alternating split/stand posture flag in the twist slot ──────
    # cmd = [split_flag, 0, 0]; "stand" is the all-zero command (deployment
    # idle parity, same as sitstand/velocity). AlternatingPostureCommand
    # guarantees split -> stand -> split -> ... (see mdp.py docstring) rather
    # than sitstand's independent 50/50 draw — the point of this task is that
    # it never stops moving.
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = POSTURE_DWELL_S
    command.debug_vis = False
    cfg.commands["twist"] = microduck_mdp.AlternatingPostureCommandCfg(
        **{
            **vars(command),
            "ramp_s":   POSTURE_RAMP_S,
            "sit_z":    SPLIT_Z,
            "stand_z":  STAND_Z,
        }
    )

    # ── Terminations — kept from splits v1 (unlike sitstand, which drops fall
    # termination entirely) — a bad splits rollout is a real tip-over risk,
    # so let terminations act as a safety backstop rather than letting many
    # cycles of bad data accumulate.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )
    cfg.terminations["tipped_sideways"] = TerminationTermCfg(
        func=microduck_mdp.gravity_proxy_out_of_band,
        time_out=False,
        params={"axis": 1, "target": 0.0, "band": 0.75},
    )
    cfg.terminations["tipped_forward_or_back"] = TerminationTermCfg(
        func=microduck_mdp.gravity_proxy_out_of_band,
        time_out=False,
        params={"axis": 0, "target": SPLIT_PITCH_TARGET, "band": 0.6},
    )

    # ── Events ──────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Reset-state mix: 50% standing / 50% already in the split (with joint +
    # tilt noise) — trains all four (start-state x command) combinations,
    # unlike v1's standing-only reset (v1 never needed a split-start case,
    # since it never returned to stand).
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":          0.0,
            "face_up_prob":            0.0,
            "sitting_prob":            0.5,
            "standing_prob":           0.5,
            "sitting_joint_overrides": SPLIT_JOINT_OVERRIDES,
            "sitting_joint_noise_std": 0.10,
            "sitting_tilt_max":        math.radians(6),
            "sitting_z_min":           SPLIT_Z - 0.005,
            "sitting_z_max":           SPLIT_Z + 0.005,
            "standing_z_min":          STAND_Z - 0.005,
            "standing_z_max":          STAND_Z + 0.005,
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )
    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
    if ENABLE_VELOCITY_PUSHES:
        interval = (0.5, 1.0) if play else VELOCITY_PUSH_INTERVAL_S
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=interval,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    # ── Terrain ─────────────────────────────────────────────────────────────
    if not rough:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    else:
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = MICRODUCK_ROUGH_TERRAINS_CFG
        if play:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 5
            cfg.scene.terrain.terrain_generator.num_rows = 5

    # ── Curriculum ──────────────────────────────────────────────────────────
    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,   "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                    {"step": 1500 * 24,  "range": 0.015},
                ],
            },
        )
    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,          "range": 0.003},
                    {"step": 500 * 24,   "range": 0.005},
                    {"step": 1000 * 24,  "range": 0.01},
                ],
            },
        )

    # Push curriculum — delayed and staged (sitstand's lesson, extra-relevant
    # here: this task repeats the transition many times per episode, so an
    # early push disrupting one cycle costs more on-policy data than in a
    # single-shot episode).
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,          "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1000 * 24,  "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 1500 * 24,  "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2000 * 24,  "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}},
                    {"step": 2500 * 24,  "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 500 * 24,   "weight": -0.2},
                {"step": 750 * 24,   "weight": -0.4},
                {"step": 1000 * 24,  "weight": -0.6},
                {"step": 1250 * 24,  "weight": -0.8},
                {"step": 1500 * 24,  "weight": -1.0},
            ],
        },
    )

    cfg.curriculum["descent_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "descent_speed",
            "weight_stages": [
                {"step": 0,          "weight": 10.0},
                {"step": 500 * 24,   "weight": 20.0},
            ],
        },
    )
    # Rise-speed cap — introduced only AFTER the rise motion exists (the
    # standup/sitstand attempt-tax lesson): any motion-tax during discovery
    # makes exploratory attempts net-negative and the skill is never found.
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "rise_speed",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1500 * 24,  "weight": 5.0},
                {"step": 2500 * 24,  "weight": 10.0},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 750 * 24,   "weight": -5e-4},
                {"step": 1250 * 24,  "weight": -1e-3},
            ],
        },
    )

    return cfg


# ── RL runner config ───────────────────────────────────────────────────────
MicroduckSplitsCycleRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_splits_cycle",
    run_name="microduck_splits_cycle",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
