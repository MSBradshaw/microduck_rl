# Microduck Pistol Squat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register a new `Mjlab-Pistol-Flat-MicroDuck` task — a commanded (sitstand-style) single-leg pistol squat, left leg as stance/squat, right leg free (must clear the ground, no fixed pose target) — train a smoke-tested policy and submit it to HF Jobs.

**Architecture:** Copy `microduck_splits_cycle_env_cfg.py` (closest template — same commanded-posture machinery) into a new `microduck_pistol_env_cfg.py`, swap in a left-leg-only depth target, add one new reward function (free-leg clearance, height-based not pose-matched) and one new curriculum function (ramps the stance-leg depth target itself, since — unlike `splits_cycle` — this task's design spec calls for an explicit depth curriculum on top of the command's own per-transition ramp). Register the task, write cfg tests, smoke test, submit to HF Jobs.

**Tech Stack:** mjlab (MuJoCo Warp) + PPO (rsl_rl), Python 3.12, pytest (CPU-only cfg tests).

**Spec:** `docs/superpowers/specs/2026-09-04-microduck-pistol-squat-design.md` (read this first — it explains WHY each number below is what it is, all measured via `scripts/measure_pistol_pose.py`, already committed).

## Global Constraints

- `ENABLE_SYMMETRY = False` — deliberately asymmetric target (left=stance, right=free); the mirror loss would fight it (spec §1).
- Every `*_penalty`/`*_l1` reward function returns ≤ 0 and MUST get a **positive** weight (AGENTS.md's "bit four envs" sign rule) — verify every `Episode_Reward/<penalty>` logs ≤ 0 once training starts.
- The free (right) leg gets NO joint-angle pose-match target anywhere in the reward stack — only a height-based clearance reward (spec §3.2). Do not add one "for symmetry" with the stance leg.
- `PISTOL_STANCE_OVERRIDES` (below) already IS the self-collision-safe depth ceiling — it is NOT sitstand's raw SIT-keyframe depth. Never deepen it without re-running `scripts/measure_pistol_pose.py` first (same rule AGENTS.md states for every measured target: measured, not guessed).

### Measured constants (from `scripts/measure_pistol_pose.py`, spec §2 — do not recompute by hand, these are already the checked values)

```python
# Left (stance) leg — HOME blended 85% toward sitstand's proven SIT keyframe
# (microduck_sitstand_env_cfg.SITTING_TARGET_OVERRIDES: hip_pitch -0.4579->-0.4079,
# knee -0.0049->1.35, ankle 0.4530->0.0) -- 85%, not 100%, because self-collision
# against a lifted free leg appears at the full SIT depth regardless of free-leg
# shape (spec §2). This dict IS the "100%" target for this task's own commanded
# posture -- the curriculum (Task 2) ramps a FRACTION of THESE numbers, not of
# sitstand's raw values.
PISTOL_STANCE_OVERRIDES = {
    2: -0.4154,   # left_hip_pitch  (HOME -0.4579, SIT -0.4079, 85% of the way)
    3:  1.1468,   # left_knee       (HOME -0.0049, SIT  1.35,   85% of the way)
    4:  0.0680,   # left_ankle      (HOME  0.4530, SIT  0.0,    85% of the way)
}

# Right (free) leg anchor -- "R3" candidate from the measurement sweep: stays
# self-collision-free through the depth above, right foot clears the floor.
# Used ONLY as a reset/init anchor pose (Task 2's set_ground_state event) --
# NEVER as a reward target (spec §3.2 -- the policy discovers its own shape).
PISTOL_FREE_LEG_ANCHOR = {
    11: 0.8,   # right_hip_pitch
    12: -0.4,  # right_knee
    13: 0.2,   # right_ankle
}

STAND_Z = 0.115   # same measured standing height every other task in the family uses
PISTOL_Z = 0.0995 # measured trunk_z at the PISTOL_STANCE_OVERRIDES depth with the R3 free leg
```

---

## Task 1: Two new `mdp.py` functions — free-leg clearance reward, stance-depth curriculum

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (append to end of file, under a new `# ── Pistol squat` section header, matching the file's existing per-task section convention)
- Test: `tests/test_mdp_pistol.py` (new)

**Interfaces:**
- Consumes: `_posture_blend(env, command_name)` (existing, private, already used by every `posture_*` function — reads the commanded posture's slewed 0→1 blend), `_servo_default_joint_pos(env, asset)` (existing, returns `(num_envs, num_servo_joints)` HOME angles), `SceneEntityCfg` (from `mjlab.managers.scene_entity_config`, already imported at top of `mdp.py`).
- Produces: `pistol_free_leg_clearance(env, command_name, margin=0.03, std=0.02, asset_cfg=...) -> torch.Tensor` (reward, shape `(num_envs,)`, range `[0, 1]`, positive-weight style like `posture_pose_match`). `posture_depth_curriculum(env, env_ids, reward_names, joint_indices, full_overrides, depth_stages, asset_cfg=...) -> torch.Tensor` (curriculum, mutates named reward terms' `sit_overrides` param in place).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mdp_pistol.py`:

```python
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


class _ClearanceEnv:
    def __init__(self, right_foot_z: float, blend: float):
        num_envs = 1
        site_pos = torch.zeros(num_envs, 1, 3)
        site_pos[0, 0, 2] = right_foot_z
        self._asset = _SiteAsset(_SiteData(site_pos))
        self.scene = self
        self.terrain = _Terrain(num_envs)
        self.command_manager = _CommandManager(torch.tensor([[blend, 0.0, 0.0]]))

    def __getitem__(self, _name):
        return self._asset


class _FakeAssetCfg:
    def __init__(self, site_ids):
        self.name = "robot"
        self.site_ids = site_ids


def test_clearance_full_reward_once_above_margin():
    env = _ClearanceEnv(right_foot_z=0.10, blend=1.0)
    out = pistol_free_leg_clearance(
        env, command_name="twist", margin=0.03, std=0.02,
        asset_cfg=_FakeAssetCfg([0]),
    )
    assert abs(float(out[0]) - 1.0) < 1e-4


def test_clearance_drops_toward_zero_when_touching_ground():
    env = _ClearanceEnv(right_foot_z=0.0, blend=1.0)
    out = pistol_free_leg_clearance(
        env, command_name="twist", margin=0.03, std=0.02,
        asset_cfg=_FakeAssetCfg([0]),
    )
    assert float(out[0]) < 0.05


def test_clearance_gated_to_zero_at_stand_blend():
    # blend=0 (fully standing, not squatting) -- inert regardless of foot
    # height, per spec §3.2 ("gated on posture blend so it's inert during
    # normal standing").
    env = _ClearanceEnv(right_foot_z=0.0, blend=0.0)
    out = pistol_free_leg_clearance(
        env, command_name="twist", margin=0.03, std=0.02,
        asset_cfg=_FakeAssetCfg([0]),
    )
    assert abs(float(out[0])) < 1e-6


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --with pytest pytest tests/test_mdp_pistol.py -v`
Expected: FAIL with `ImportError: cannot import name 'pistol_free_leg_clearance'` (and `posture_depth_curriculum`).

- [ ] **Step 3: Implement the two functions**

Open `src/mjlab_microduck/tasks/mdp.py`, scroll to the end of the file, and append:

```python
# ── Pistol squat ──────────────────────────────────────────────────────────
def pistol_free_leg_clearance(
    env: ManagerBasedRlEnv,
    command_name: str,
    margin: float = 0.03,
    std: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=("right_foot",)),
) -> torch.Tensor:
    """Reward the free (right) foot for clearing the ground during a
    commanded pistol squat.

    Deliberately height-based, NOT a joint-angle pose-match target like the
    stance leg gets (design spec §3.2) -- the free leg's exact shape is left
    for the policy to discover; this only encodes "stays off the ground."
    Full reward (1.0) once the foot is at or above `margin`; smoothly decays
    as it dips below. Gated on the commanded posture blend (0 at STAND, 1 at
    full squat) so it's inert during normal standing -- a foot naturally
    near the ground while standing shouldn't be punished.
    """
    blend = _posture_blend(env, command_name)
    asset = env.scene[asset_cfg.name]
    foot_z = (
        asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]
        - env.scene.terrain.env_origins[:, 2]
    )
    shortfall = torch.clamp(margin - foot_z, min=0.0)
    height_reward = torch.exp(-((shortfall / std) ** 2))
    return blend * height_reward


def posture_depth_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_names: tuple[str, ...],
    joint_indices: tuple[int, ...],
    full_overrides: dict[int, float],
    depth_stages: list[dict],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Ramp a posture_* reward's `sit_overrides` depth from a fraction of
    `full_overrides` to 100%, interpolating FROM EACH JOINT'S OWN HOME ANGLE
    -- not from zero. (`pose_target_depth_curriculum` elsewhere in this file
    scales the raw target value directly, which silently breaks for any
    joint whose HOME angle isn't ~0 -- e.g. ankle's HOME=0.453, target=0
    would never move at all under that scheme. This task's ankle target IS
    exactly that case, so this is a real correctness requirement, not
    pedantry.)

    Unlike the per-transition slewed blend `AlternatingPostureCommand`
    already runs every dwell period (0->1 within EACH transition), this
    ramps what "1" (the command's own SQUAT endpoint) actually targets, over
    TRAINING iterations -- design spec §6's explicit depth curriculum,
    proposed because single-leg balance is a harder skill than splits_cycle
    (which dropped an equivalent curriculum, judging its own per-transition
    ramp sufficient). `depth_stages`: `[{"step": int, "fraction": float}, ...]`;
    latest passed stage wins, applied identically to every reward in
    `reward_names`.
    """
    del env_ids
    fraction = depth_stages[0]["fraction"]
    for stage in depth_stages:
        if env.common_step_counter >= stage["step"]:
            fraction = stage["fraction"]
    asset = env.scene[asset_cfg.name]
    home = _servo_default_joint_pos(env, asset)[0]  # (num_joints,) -- identical across envs
    overrides = {
        idx: float(home[idx]) + fraction * (full_overrides[idx] - float(home[idx]))
        for idx in joint_indices
    }
    for name in reward_names:
        term_cfg = env.reward_manager.get_term_cfg(name)
        merged = dict(term_cfg.params["sit_overrides"])
        merged.update(overrides)
        term_cfg.params["sit_overrides"] = merged
    return torch.tensor([fraction])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --with pytest pytest tests/test_mdp_pistol.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_mdp_pistol.py
git commit -m "feat: add pistol squat free-leg clearance reward + depth curriculum

New mdp.py functions for the pistol squat task (design spec
docs/superpowers/specs/2026-09-04-microduck-pistol-squat-design.md):

- pistol_free_leg_clearance: height-based reward for the free leg staying
  off the ground, deliberately not a joint-angle target (§3.2) -- the free
  leg's shape is left for the policy to discover.
- posture_depth_curriculum: ramps the stance-leg depth target from HOME,
  correctly interpolating per-joint (not scaling the raw target value,
  which silently breaks for ankle's HOME=0.453 -> target=0 case)."
```

---

## Task 2: `microduck_pistol_env_cfg.py` — the env cfg

**Files:**
- Create: `src/mjlab_microduck/tasks/microduck_pistol_env_cfg.py` (copy of `microduck_splits_cycle_env_cfg.py`, then edited per the steps below)

**Interfaces:**
- Consumes: `microduck_mdp.pistol_free_leg_clearance`, `microduck_mdp.posture_depth_curriculum` (Task 1), every other `microduck_mdp.posture_*`/`AlternatingPostureCommandCfg` function used unchanged (already exist).
- Produces: `make_microduck_pistol_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`, `MicroduckPistolRlCfg` (an `RslRlOnPolicyRunnerCfg`) — both consumed by Task 3's task registration.

- [ ] **Step 1: Copy the template**

```bash
cp src/mjlab_microduck/tasks/microduck_splits_cycle_env_cfg.py \
   src/mjlab_microduck/tasks/microduck_pistol_env_cfg.py
```

- [ ] **Step 2: Replace the module docstring**

In `microduck_pistol_env_cfg.py`, replace the entire docstring (everything between the opening `"""` and closing `"""` at the top of the file, lines 1-64 of the copied file) with:

```python
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
```

- [ ] **Step 3: Replace the symmetry comment and constants block**

Find:
```python
# Symmetry — OFF. Deliberately asymmetric split target (left leg forward,
# right leg back); a mirror loss would train the policy to fight it.
ENABLE_SYMMETRY = False
```
Replace with:
```python
# Symmetry — OFF. Deliberately asymmetric target (left leg stance/squat,
# right leg free); a mirror loss would train the policy to fight it.
ENABLE_SYMMETRY = False
```

(The DR block below it — `ENABLE_COM_RANDOMIZATION` through `IMU_ORIENTATION_RANDOMIZATION_ANGLE` — stays byte-for-byte unchanged from splits_cycle. Do not edit those lines.)

- [ ] **Step 4: Replace episode length / dwell / target constants**

Find:
```python
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
```

Replace with:
```python
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
```

- [ ] **Step 5: Rename the function and drop the walking-specific-terms block header (no change needed to its body)**

Find:
```python
def make_microduck_splits_cycle_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck splits-cycle environment configuration."""
```
Replace with:
```python
def make_microduck_pistol_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck pistol-squat environment configuration."""
```

(The `feet_ground_cfg`/`self_collision_cfg`/`foot_frictions_geom_names` block right after this, and the `cfg = make_velocity_env_cfg()` / `cfg.scene.entities` / `cfg.viewer.body_name` / `cfg.episode_length_s` / `cfg.actions["joint_pos"].scale` / the walking-reward-deletion loop right after — ALL stay byte-for-byte unchanged from splits_cycle. Do not edit those lines.)

- [ ] **Step 6: Replace the posture-conditioned reward stack**

Find (the `posture_pose_legs` through `rise_bootstrap` reward blocks):
```python
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
```

Replace with:
```python
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
    cfg.rewards["free_leg_clearance"] = RewardTermCfg(
        func=microduck_mdp.pistol_free_leg_clearance,
        weight=2.0,
        params={"command_name": "twist", "margin": 0.03, "std": 0.02},
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
```

- [ ] **Step 7: Replace orientation shaping, stillness, and composite reward params**

Find:
```python
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
```

Replace with:
```python
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
```

- [ ] **Step 8: Leave the gentleness / sim2real regularizer blocks unchanged**

The `descent_speed`/`rise_speed`/`gentle_motion`/`action_rate_l2`/`joint_torque_rate_l2`/`body_ang_vel`/`angular_momentum`/`dof_pos_limits`-pop/`self_collisions` blocks, and the entire observations block (`del cfg.observations[...]` through the `ENABLE_ENCODER_BIAS` branch and the head-pose-command block) stay byte-for-byte identical to splits_cycle. Do not edit those lines — they don't reference `SPLIT_Z`/`SPLIT_JOINT_OVERRIDES`/`_LEG_JOINTS` at all.

- [ ] **Step 9: Replace the command setup**

Find:
```python
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
```

Replace with:
```python
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
```

- [ ] **Step 10: Replace terminations**

Find:
```python
    cfg.terminations["tipped_forward_or_back"] = TerminationTermCfg(
        func=microduck_mdp.gravity_proxy_out_of_band,
        time_out=False,
        params={"axis": 0, "target": SPLIT_PITCH_TARGET, "band": 0.6},
    )
```

Replace with:
```python
    cfg.terminations["tipped_forward_or_back"] = TerminationTermCfg(
        func=microduck_mdp.gravity_proxy_out_of_band,
        time_out=False,
        params={"axis": 0, "target": PISTOL_PITCH_TARGET, "band": 0.6},
    )
```

(Everything else in the terminations block — the `fell_over` deletion, `nan_state`, `tipped_sideways` — stays unchanged; only the `SPLIT_PITCH_TARGET` reference needed renaming.)

- [ ] **Step 11: Leave the `expand_bam_friction_fields`/`reset_action_history`/`foot_friction` event lines unchanged**

- [ ] **Step 12: Replace the reset-state-mix event**

Find:
```python
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
```

Replace with:
```python
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
```

- [ ] **Step 13: Leave the DR event blocks (`randomize_com` through `push_robot`) and the terrain block unchanged**

- [ ] **Step 14: Replace the curriculum block**

Find (from the `com_range`/`head_com_range` blocks through the end of the curriculum section, i.e. everything from `if ENABLE_COM_RANDOMIZATION:` through the closing of `torque_rate_weight`):

```python
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
```

Replace with:
```python
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
            "joint_indices":  tuple(_STANCE_LEG_JOINTS),
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
```

- [ ] **Step 15: Replace the RL runner config**

Find (the entire block from `# ── RL runner config ──` to the end of the file):
```python
# ── RL runner config ───────────────────────────────────────────────────────
MicroduckSplitsCycleRlCfg = RslRlOnPolicyRunnerCfg(
```
through
```python
    experiment_name="microduck_splits_cycle",
    run_name="microduck_splits_cycle",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6000,
)
```

Replace with (identical `actor`/`critic`/`algorithm` blocks, only `experiment_name`/`run_name`/`max_iterations` differ — this task is a harder balance problem than splits_cycle, and its own curriculum doesn't finish ramping until step 3000*24, so it gets a larger budget than splits_cycle's 6000, per AGENTS.md's "pick a budget that clears the last curriculum stage with reasonable headroom" rule):

```python
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
```

- [ ] **Step 16: Verify the file imports cleanly**

Run: `docker exec microduck bash -c "cd /w && uv run python -c 'from mjlab_microduck.tasks.microduck_pistol_env_cfg import make_microduck_pistol_env_cfg, MicroduckPistolRlCfg; print(make_microduck_pistol_env_cfg())'"`
Expected: prints the cfg object with no traceback. (If it errors on `SPLIT_Z`/`SPLIT_JOINT_OVERRIDES`/`_LEG_JOINTS` being undefined, a copy-edit step above was missed — grep the new file for those three names, every remaining occurrence is a bug.)

- [ ] **Step 17: Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_pistol_env_cfg.py
git commit -m "feat: add microduck_pistol_env_cfg (single-leg squat, commanded)

Copy-and-edit of microduck_splits_cycle_env_cfg.py (design spec
docs/superpowers/specs/2026-09-04-microduck-pistol-squat-design.md):
left leg = stance/squat (pose-matched, depth-curriculum'd), right leg =
free (height-based clearance reward only, no pose target). PISTOL_STANCE_
OVERRIDES already caps at the self-collision-safe depth measured by
scripts/measure_pistol_pose.py -- not sitstand's raw SIT-keyframe depth."
```

---

## Task 3: Register the task

**Files:**
- Modify: `src/mjlab_microduck/tasks/__init__.py`

**Interfaces:**
- Consumes: `make_microduck_pistol_env_cfg`, `MicroduckPistolRlCfg` (Task 2).
- Produces: task ids `Mjlab-Pistol-Flat-MicroDuck`, `Mjlab-Pistol-Rough-MicroDuck`, `Mjlab-Pistol-Flat-Backlash-MicroDuck` resolvable via `mjlab.tasks.registry.list_tasks()`.

- [ ] **Step 1: Add the import**

In `src/mjlab_microduck/tasks/__init__.py`, find:
```python
from .microduck_splits_cycle_env_cfg import (
    make_microduck_splits_cycle_env_cfg,
    MicroduckSplitsCycleRlCfg,
)
from .backlash import make_backlash_variant
```
Replace with:
```python
from .microduck_splits_cycle_env_cfg import (
    make_microduck_splits_cycle_env_cfg,
    MicroduckSplitsCycleRlCfg,
)
from .microduck_pistol_env_cfg import (
    make_microduck_pistol_env_cfg,
    MicroduckPistolRlCfg,
)
from .backlash import make_backlash_variant
```

- [ ] **Step 2: Register the Flat and Rough tasks**

Find:
```python
register_mjlab_task(
    task_id="Mjlab-SplitsCycle-Rough-MicroDuck",
    env_cfg=make_microduck_splits_cycle_env_cfg(rough=True),
    play_env_cfg=make_microduck_splits_cycle_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSplitsCycleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```
Replace with:
```python
register_mjlab_task(
    task_id="Mjlab-SplitsCycle-Rough-MicroDuck",
    env_cfg=make_microduck_splits_cycle_env_cfg(rough=True),
    play_env_cfg=make_microduck_splits_cycle_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSplitsCycleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Pistol squat — commanded, single-leg squat. Left leg stance, right leg
# free (no fixed pose target — see design spec).
register_mjlab_task(
    task_id="Mjlab-Pistol-Flat-MicroDuck",
    env_cfg=make_microduck_pistol_env_cfg(),
    play_env_cfg=make_microduck_pistol_env_cfg(play=True),
    rl_cfg=MicroduckPistolRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Pistol-Rough-MicroDuck",
    env_cfg=make_microduck_pistol_env_cfg(rough=True),
    play_env_cfg=make_microduck_pistol_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckPistolRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

- [ ] **Step 3: Add the Backlash variant**

Find:
```python
    ("Mjlab-SplitsCycle-Flat-Backlash-MicroDuck", make_microduck_splits_cycle_env_cfg, {}, MicroduckSplitsCycleRlCfg, _BL_ALLCOL),
)
```
Replace with:
```python
    ("Mjlab-SplitsCycle-Flat-Backlash-MicroDuck", make_microduck_splits_cycle_env_cfg, {}, MicroduckSplitsCycleRlCfg, _BL_ALLCOL),
    ("Mjlab-Pistol-Flat-Backlash-MicroDuck", make_microduck_pistol_env_cfg, {}, MicroduckPistolRlCfg, _BL_ALLCOL),
)
```

- [ ] **Step 4: Verify registration**

Run: `docker exec microduck bash -c "cd /w && uv run list-envs 2>&1 | grep -i pistol"`
Expected: three lines, `Mjlab-Pistol-Flat-MicroDuck`, `Mjlab-Pistol-Rough-MicroDuck`, `Mjlab-Pistol-Flat-Backlash-MicroDuck`.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/__init__.py
git commit -m "feat: register Mjlab-Pistol-Flat/Rough/Backlash-MicroDuck tasks"
```

---

## Task 4: Cfg tests

**Files:**
- Create: `tests/test_pistol_cfg.py` (mirrors `tests/test_splits_cycle_cfg.py`)

**Interfaces:**
- Consumes: everything `microduck_pistol_env_cfg.py` exports (Task 2), `AlternatingPostureCommand`/`AlternatingPostureCommandCfg` (existing, unchanged).

- [ ] **Step 1: Write the tests**

```python
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
```

- [ ] **Step 2: Run the tests**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_pistol_cfg.py tests/test_mdp_pistol.py -v"`
Expected: all PASS. If any FAIL, fix the corresponding line in `microduck_pistol_env_cfg.py`/`mdp.py` from Task 1/2 — do not weaken a test to make it pass.

- [ ] **Step 3: Run the FULL test suite to make sure nothing else broke**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/ -v"`
Expected: all PASS, including every pre-existing test file (this task only ever ADDED code, never modified an existing function's body — a failure elsewhere means Task 1/2 accidentally touched something shared).

- [ ] **Step 4: Commit**

```bash
git add tests/test_pistol_cfg.py
git commit -m "test: add cfg tests for the pistol squat task"
```

---

## Task 5: Smoke test (mandatory before any real run, per AGENTS.md)

**Files:** none (verification only)

- [ ] **Step 1: Run the smoke test**

```bash
docker exec microduck bash -c "
  cd /w && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m mjlab_microduck.train_cli Mjlab-Pistol-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --gpu-ids None \
    --agent.logger tensorboard
"
```

Expected: exits 0, no NaN warnings, no traceback. Look for iteration log lines showing `Episode_Reward/free_leg_clearance`, `Episode_Reward/posture_pose_stance`, etc. being logged (confirms the new reward terms actually compute, not just that the cfg builds).

- [ ] **Step 2: If it fails, diagnose before retrying**

Common failure classes for a first-time task, per this repo's own history:
- `KeyError`/`AttributeError` mentioning `SPLIT_Z`/`SPLIT_JOINT_OVERRIDES`/`_LEG_JOINTS` → a Task 2 copy-edit step was missed; grep `microduck_pistol_env_cfg.py` for those three names.
- NaN in the first few iterations → check `PISTOL_STANCE_OVERRIDES`' knee value (1.1468 rad ≈ 66°) isn't somehow being fed as a HARD reset target without the reset-state mix's noise/tilt bounds working correctly — re-check Task 2 Step 12 was applied.
- `AssertionError` from `pistol_free_leg_clearance`'s `asset_cfg.site_ids` being empty → the default `SceneEntityCfg("robot", site_names=("right_foot",))` didn't resolve; confirm the robot model in use (`MICRODUCK_STANDUP_ROBOT_CFG`, same as splits_cycle) actually has a `right_foot` site (confirmed present in `scripts/measure_pistol_pose.py`'s own use of `mj_name2id(..., "right_foot")`, so this shouldn't happen, but check first before assuming something else).

- [ ] **Step 3: No commit for this task** — the smoke test is a verification gate, not a code change. Proceed to Task 6 only once Step 1 passes cleanly.

---

## Task 6: Submit to HF Jobs

**Files:** none (this is a training run, not a code change — HOW-TO.md gets updated in Task 7 once the job is submitted)

- [ ] **Step 1: Submit the job**

Follow `HOW-TO.md`'s "Training a new run" section exactly (wandb logging enabled, `--hf-jobs`, `l4x1`):

```bash
docker exec microduck bash -c "
  cd /w && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m mjlab_microduck.train_cli Mjlab-Pistol-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 8000 \
    --hf-jobs --flavor l4x1 \
    --namespace mikeybrad --run-name pistol-run1-2026-09-04 --timeout 12h --detach
"
```

(`--agent.max_iterations 8000` matches `MicroduckPistolRlCfg.max_iterations` from Task 2 Step 15 — the curriculum's own last stage is at step `3000*24 = 72000` env-steps = iteration 3000, so 8000 gives roughly 5000 iterations of headroom past the last curriculum event for the policy to consolidate, more than splits_cycle's ratio since this is explicitly the harder task.)

- [ ] **Step 2: Confirm the job actually started**

```bash
docker exec microduck bash -c "cd /w && uv run python -c \"
from huggingface_hub import HfApi
job = HfApi().inspect_job(job_id='<job-id-from-step-1-output>')
print(job.status)
\""
```

Expected: `RUNNING` or `SCHEDULING`, not an immediate `ERROR` (an immediate error usually means the smoke test's local success didn't carry over to the remote environment — check the job's log via `HfApi().inspect_job` or the HF job page for a fresh `uv sync` failure before assuming the cfg itself is broken).

- [ ] **Step 3: No commit for this task** — record the job id in Task 7's docs update instead.

---

## Task 7: Docs — HOW-TO.md + a tracking doc for this run

**Files:**
- Modify: `HOW-TO.md` ("What's been trained so far" section)
- Create: `~/microduck-results/pistol-run1-2026-09-04/README.md` (outside the repo, same convention as `~/microduck-results/batch1-walking-2026-09-03/README.md` — NOT tracked by git, per HOW-TO.md's "Checkpoints, organized" bullet)

- [ ] **Step 1: Add a "What's been trained so far" entry**

In `HOW-TO.md`, find the `## What's been trained so far (running log)` section and add a new bullet at the end, following the exact style of the existing entries (task name, one-line description, job id, status):

```markdown
- **2026-09-04, `Mjlab-Pistol-Flat-MicroDuck`** — commanded single-leg
  pistol squat (left leg stance, right leg free — no fixed pose target for
  the free leg, only a height-based clearance reward; design spec
  `docs/superpowers/specs/2026-09-04-microduck-pistol-squat-design.md`).
  Target pose measured via `scripts/measure_pistol_pose.py` (self-collision
  caps depth at 85% of sitstand's SIT keyframe, not 100%). Explicit depth
  curriculum on top of the usual per-transition ramp — a harder balance
  problem than splits_cycle, which judged its own per-transition ramp
  sufficient. Job `<job-id-from-task-6>`, 8000 iterations, not yet
  evaluated.
```

- [ ] **Step 2: Create the results tracking doc**

Create `~/microduck-results/pistol-run1-2026-09-04/README.md` (this path is OUTSIDE the git repo — use the `Write` tool with the literal expanded home-directory path, e.g. `/Users/<you>/microduck-results/pistol-run1-2026-09-04/README.md` on the Mac, or the equivalent inside the container's bind mount if running from there):

```markdown
# Pistol squat run 1 — 2026-09-04

First training run for `Mjlab-Pistol-Flat-MicroDuck` (single-leg pistol
squat, commanded, asymmetric — left leg stance, right leg free).

- **Design spec:** `~/microduck_rl/docs/superpowers/specs/2026-09-04-microduck-pistol-squat-design.md`
- **Implementation plan:** `~/microduck_rl/docs/superpowers/plans/2026-09-04-microduck-pistol-squat.md`
- **Job:** `<job-id-from-task-6>` (`pistol-run1-2026-09-04`), 4096 envs,
  `l4x1`, 8000 iterations budgeted (curriculum's last stage at iteration
  3000, so this is ~5000 iterations of consolidation headroom).
- **Checkpoints:** `https://huggingface.co/mikeybrad/pistol-run1-2026-09-04`
- **Not yet evaluated.** Per AGENTS.md: don't trust the aggregate reward
  curve — once this finishes, pull `reward_history.csv`, find the best
  checkpoint (not necessarily the last), and actually watch a headless eval
  / video before believing it works. Specifically worth checking, given
  this task's known-hard physics (design spec §2):
  - Does the free leg actually stay off the ground, or find some contact-
    assist exploit `pistol_free_leg_clearance`'s height check didn't fully
    close off?
  - Does `Episode_Reward/roll_split` stay reasonably high despite the
    ~22-26mm lateral hip shift the task physically requires (design spec
    §3.4) — if it's suppressed, the orientation reward may be fighting the
    necessary lean after all, contrary to this session's assessment that
    the existing std=0.45 was wide enough.
  - Does the stance-leg depth curriculum (`stance_depth`, Task 1) actually
    reach fraction=1.0 cleanly, or does the policy stall at an earlier
    stage (would show up as a wandb metric stepping down right at the
    750*24/1500*24 stage boundaries — AGENTS.md's curriculum-pacing tell).
```

- [ ] **Step 3: Commit the HOW-TO.md change**

```bash
git add HOW-TO.md
git commit -m "docs: add pistol squat run to HOW-TO's training log"
```

- [ ] **Step 4: Push to fork/main**

Per `HOW-TO.md`'s "Pushing your work" section (pushes go to `fork`'s `main`, NOT `fork`'s own `develop` — that branch tracks upstream, not this repo's local history):

```bash
git fetch fork main
git merge-base --is-ancestor fork/main HEAD && echo "fast-forward OK" || echo "STOP -- not a fast-forward, do not push blind"
git push fork develop:main
```
