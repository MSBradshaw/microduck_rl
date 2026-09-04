"""Microduck *pistol squat* task — commanded, single-leg squat, asymmetric.

Left leg = stance (weight-bearing, squats). Right leg = free (must clear
the ground; no fixed pose target -- its exact shape is left for the policy
to discover, see mdp.pistol_free_leg_clearance). Commanded like sitstand /
splits_cycle: one policy, both directions, driven by a posture flag in the
twist slot (0 = STAND, 1 = SQUAT). Ends in normal two-legged standing, not
a held trick pose -- that's why this reuses the flag-commanded pattern
rather than splits v1's one-way episodic descend-and-hold.

Design lineage -- built on splits_cycle's exact commanded-posture MACHINERY
(itself reused from sitstand): posture_pose_match/l1, posture_height_
gaussian/l1, posture_composite, posture_stillness, posture_rise_bootstrap,
AlternatingPostureCommandCfg (deterministic alternation, not sitstand's
independent draw -- same data-efficiency reasoning splits_cycle used,
arguably stronger here since single-leg balance needs MORE transition
exposure, not less), the descent/rise speed caps + |a_z| shock penalty,
DR/obs/regularizer set, and splits' soft tip-band terminations (single-leg
stance carries real tip-over risk, unlike sitstand's "no fall termination"
choice).

What's NEW here (not in splits_cycle):
  - `pistol_free_leg_clearance` (mdp.py): the free leg gets a HEIGHT-based
    reward, not a joint-angle pose-match target like every other commanded-
    posture task's second leg. Only the STANCE leg (indices 0-4) appears in
    `_LEG_JOINTS`/`joint_indices` for the pose-match/composite terms.
  - `posture_depth_curriculum` (mdp.py): unlike splits_cycle (which dropped
    an equivalent curriculum, judging its own per-transition ramp
    sufficient), this task ramps `PISTOL_STANCE_OVERRIDES` itself from a
    fraction to 100% over TRAINING iterations -- single-leg balance is a
    harder skill (see design spec §6) and gets both the per-transition ramp
    AND a training-time depth ramp.
  - `PISTOL_STANCE_OVERRIDES` is measured (scripts/measure_pistol_pose.py)
    as 85% of the way from HOME toward sitstand's SIT keyframe, NOT
    sitstand's raw full depth -- self-collision against a lifted free leg
    appears at full SIT depth regardless of free-leg shape (design spec §2).
    This dict already IS this task's "100%"; never deepen it without
    re-running the measurement script first.
  - Reset-state mix uses a COMBINED override dict (stance depth + a
    measured free-leg anchor pose) so a "start already squatted" reset
    looks like a plausible squat, not a broken half-pose with the free leg
    still on the ground -- but the reward target itself never includes the
    free-leg anchor (see above).

Full derivation, measurements, and open questions:
docs/superpowers/specs/2026-09-04-microduck-pistol-squat-design.md

Joint layout (14 actuated joints):
    0-4 : left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)  -- STANCE
    5-8 : neck/head (neck_pitch, head_pitch, head_yaw, head_roll)
    9-13: right leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)  -- FREE
"""

import math
from copy import deepcopy

# Symmetry — OFF. Deliberately asymmetric target (left leg stance/squat,
# right leg free); a mirror loss would train the policy to fight it.
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

# Episode: long enough for a few full squat<->stand cycles (mirrors
# splits_cycle's reasoning), somewhat shorter than splits_cycle's since a
# single-leg squat is a slower, more deliberate motion.
EPISODE_LENGTH_S = 20.0

# Dwell time before the alternating command may flip again. Must exceed
# POSTURE_RAMP_S so there's a real hold, not a ramp cut short by the next
# flip.
POSTURE_DWELL_S = (3.5, 5.0)
# Seconds for the internal target blend to traverse STAND<->SQUAT. Slightly
# longer than splits_cycle's 2.0s -- single-leg balance is a slower, more
# deliberate motion than a front split's descent.
POSTURE_RAMP_S = 2.5

# Empirically-measured standing trunk height (same model as every task in
# the family).
STAND_Z = 0.115

# ── Pistol target — FROM scripts/measure_pistol_pose.py, design spec §2 ──────
# Left (stance) leg blended 85% of the way from HOME toward sitstand's
# proven SIT keyframe (SITTING_TARGET_OVERRIDES) -- 85%, not 100%: self-
# collision against a lifted free leg appears at full SIT depth regardless
# of free-leg shape. This dict IS this task's "100%" reward target --
# posture_depth_curriculum (Task 1) ramps a fraction OF THESE values, not
# of sitstand's raw values. Right (free) leg has NO entry here -- it is
# never a reward target (see mdp.pistol_free_leg_clearance).
PISTOL_Z = 0.0995
PISTOL_STANCE_OVERRIDES = {
    2: -0.4154,   # left_hip_pitch  (HOME -0.4579, SIT -0.4079, 85% of the way)
    3:  1.1468,   # left_knee       (HOME -0.0049, SIT  1.35,   85% of the way)
    4:  0.0680,   # left_ankle      (HOME  0.4530, SIT  0.0,    85% of the way)
}
# Right (free) leg anchor -- measurement sweep's "R3" candidate, self-
# collision-free through PISTOL_STANCE_OVERRIDES' depth. Reset-only (Task 2
# Step 12's set_ground_state event); NEVER used as a reward target.
PISTOL_FREE_LEG_ANCHOR = {
    11: 0.8,    # right_hip_pitch
    12: -0.4,   # right_knee
    13: 0.2,    # right_ankle
}
PISTOL_RESET_OVERRIDES = {**PISTOL_STANCE_OVERRIDES, **PISTOL_FREE_LEG_ANCHOR}
PISTOL_PITCH_TARGET = 0.0  # design default, matches STAND's upright target too — open per design spec §7

# Reward-scoping: ONLY the stance leg has a pose-match target. This is the
# one constant most different from splits_cycle's _LEG_JOINTS (which covers
# BOTH legs, since splits targets both symmetrically).
_STANCE_LEG_JOINTS = [0, 1, 2, 3, 4]
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


def make_microduck_pistol_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck pistol-squat environment configuration."""

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
    cfg.rewards["posture_pose_stance"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match,
        weight=4.0,
        params={
            "command_name":  "twist",
            "std":           0.5,
            "joint_indices": _STANCE_LEG_JOINTS,
            "sit_overrides": dict(PISTOL_STANCE_OVERRIDES),
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
            "joint_indices": _STANCE_LEG_JOINTS,
            "sit_overrides": dict(PISTOL_STANCE_OVERRIDES),
        },
    )

    # Free-leg clearance — NEW for this task (mdp.py Task 1). Height-based,
    # not a joint-angle target: the only reward term touching the right leg
    # at all.
    # `asset_cfg` passed explicitly (matches the function's own default
    # SceneEntityCfg("robot", site_names=("right_foot",)) exactly) — mjlab's
    # reward manager only calls .resolve() on SceneEntityCfg objects present
    # in a term's `params` dict (manager_base.py's _resolve_common_term_cfg
    # iterates term_cfg.params.values()), so a SceneEntityCfg left as a bare
    # function default is never resolved and site_ids stays an unresolved
    # slice. Every other site/body-scoped reward in this file (roll_split,
    # pitch_split, body_ang_vel, ...) already follows this explicit-params
    # convention.
    cfg.rewards["free_leg_clearance"] = RewardTermCfg(
        func=microduck_mdp.pistol_free_leg_clearance,
        weight=2.0,
        params={
            "command_name": "twist",
            "margin": 0.03,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot", site_names=("right_foot",)),
        },
    )

    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={"command_name": "twist", "sit_z": PISTOL_Z, "stand_z": STAND_Z, "std": 0.04},
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={"command_name": "twist", "sit_z": PISTOL_Z, "stand_z": STAND_Z, "std": 0.015},
    )
    # Weight 7.5, matching splits_cycle's tuned value for a similarly small
    # total height travel (pistol: STAND_Z-PISTOL_Z = 15.5mm, close to
    # splits' 17mm) rather than sitstand's 6.0 (55mm travel).
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1,
        weight=7.5,
        params={"command_name": "twist", "sit_z": PISTOL_Z, "stand_z": STAND_Z},
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
    # Orientation — reused unchanged from splits/splits_cycle: generous roll
    # tolerance (std=0.45), tighter pitch (std=0.15). Deliberately NOT
    # tightened despite the pistol squat needing more lateral lean than
    # splits does (design spec §2's measured ~22-26mm hip shift, §3.4) --
    # the existing std is already wide enough to not fight a lean of that
    # magnitude; watch Episode_Reward/roll_split in training telemetry
    # rather than pre-tightening blind.
    cfg.rewards["roll_split"] = RewardTermCfg(
        func=microduck_mdp.roll_split,
        weight=0.75,
        params={"std": 0.45, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    cfg.rewards["pitch_split"] = RewardTermCfg(
        func=microduck_mdp.pitch_split,
        weight=1.5,
        params={
            "target_pitch": PISTOL_PITCH_TARGET,
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name": "twist",
            "sit_z":         PISTOL_Z,
            "stand_z":       STAND_Z,
            "band_full":     0.012,
            "band_zero":     0.03,
            "vel_std":       0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )

    # Multiplicative goal score vs the commanded posture — kills partial-sum
    # farming. Scoped to the STANCE leg only (joint_indices), same as
    # posture_pose_stance/posture_pose_l1 above — the free leg has no pose
    # target for this term to score against.
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name":  "twist",
            "sit_overrides": dict(PISTOL_STANCE_OVERRIDES),
            "joint_indices": _STANCE_LEG_JOINTS,
            "sit_z":         PISTOL_Z,
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

    # ── Command: alternating squat/stand posture flag in the twist slot ──────
    # cmd = [squat_flag, 0, 0]; "stand" is the all-zero command (deployment
    # idle parity, same as every other task in the family).
    # AlternatingPostureCommand guarantees squat -> stand -> squat -> ...
    # (see mdp.py docstring) rather than sitstand's independent 50/50 draw —
    # same data-efficiency reasoning splits_cycle used, arguably stronger
    # here since single-leg balance needs MORE transition exposure.
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
            "sit_z":    PISTOL_Z,
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
        params={"axis": 0, "target": PISTOL_PITCH_TARGET, "band": 0.6},
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

    # Reset-state mix: 50% standing / 50% already in the squat. Uses
    # PISTOL_RESET_OVERRIDES (stance depth + the free-leg ANCHOR pose), NOT
    # PISTOL_STANCE_OVERRIDES alone — a "start already squatted" reset with
    # the free leg left at HOME (on the ground) would be a broken half-pose,
    # not a plausible squat. The reward target itself (Step 6/7 above) never
    # includes the free-leg anchor; only this reset event does.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":          0.0,
            "face_up_prob":            0.0,
            "sitting_prob":            0.5,
            "standing_prob":           0.5,
            "sitting_joint_overrides": dict(PISTOL_RESET_OVERRIDES),
            "sitting_joint_noise_std": 0.10,
            "sitting_tilt_max":        math.radians(6),
            "sitting_z_min":           PISTOL_Z - 0.005,
            "sitting_z_max":           PISTOL_Z + 0.005,
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

    # Depth curriculum — NEW vs splits_cycle (design spec §6): single-leg
    # balance is a harder skill, so this task ramps PISTOL_STANCE_OVERRIDES
    # itself over training on TOP OF the per-transition slewed ramp, rather
    # than relying on the per-transition ramp alone. Applied to every reward
    # whose `sit_overrides` targets the stance leg.
    cfg.curriculum["stance_depth"] = CurriculumTermCfg(
        func=microduck_mdp.posture_depth_curriculum,
        params={
            "reward_names":   ("posture_pose_stance", "posture_pose_l1", "posture_composite"),
            "joint_indices":  tuple(PISTOL_STANCE_OVERRIDES.keys()),
            "full_overrides": dict(PISTOL_STANCE_OVERRIDES),
            "depth_stages": [
                {"step": 0,          "fraction": 0.3},
                {"step": 750 * 24,   "fraction": 0.6},
                {"step": 1500 * 24,  "fraction": 1.0},
            ],
        },
    )

    # Push curriculum — delayed and staged, same reasoning as splits_cycle.
    if ENABLE_VELOCITY_PUSHES:
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,          "velocity_range": {"x": (0.0, 0.0),    "y": (0.0, 0.0)}},
                    {"step": 1500 * 24,  "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 2000 * 24,  "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2500 * 24,  "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}},
                    {"step": 3000 * 24,  "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
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
                {"step": 750 * 24,   "weight": 20.0},
            ],
        },
    )
    # Rise-speed cap — introduced only AFTER the rise motion exists (the
    # standup/sitstand attempt-tax lesson): any motion-tax during discovery
    # makes exploratory attempts net-negative and the skill is never found.
    # Delayed further than splits_cycle's (2000/2500*24) since this task's
    # depth curriculum itself doesn't finish ramping until 1500*24.
    cfg.curriculum["rise_speed_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "rise_speed",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": 5.0},
                {"step": 3000 * 24,  "weight": 10.0},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 1000 * 24,  "weight": -5e-4},
                {"step": 1750 * 24,  "weight": -1e-3},
            ],
        },
    )

    return cfg


# ── RL runner config ───────────────────────────────────────────────────────
MicroduckPistolRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_pistol",
    run_name="microduck_pistol",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=8000,
)
