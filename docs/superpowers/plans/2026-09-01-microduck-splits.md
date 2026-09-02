# Microduck Splits (Front Split) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Mjlab-Splits-Flat-MicroDuck` — an episodic task where the robot
descends from standing into a front split (left leg forward, right leg back) and
holds it.

**Architecture:** New `microduck_splits_env_cfg.py` built on `make_velocity_env_cfg()`,
following the standup template exactly (DR/obs/noise/delay/NaN-guard stack copied
wholesale). Three new pure `mdp.py` functions (`roll_split`, `pitch_split`,
`pose_target_depth_curriculum`). A new headless measurement script determines the
real target-pose numbers before they're wired into rewards.

**Tech Stack:** mjlab (MuJoCo Warp), PPO via rsl_rl, PyTorch, pytest. Runs inside the
`microduck` Docker container per `HOW-TO.md` (Intel Mac, no native torch/mujoco).

**Spec:** `docs/superpowers/specs/2026-09-01-microduck-splits-design.md`

## Global Constraints

- Obs stays 61D actor / matches the shared policy-family layout — never delete a
  command slot, even ones this task doesn't use (AGENTS.md).
- Joint indices: use `_LEG_JOINTS`/`_servo_joint_pos` helpers, never hardcode raw
  indices outside the `_LEG_JOINTS`/`SPLIT_JOINT_OVERRIDES` constants block.
- Sign rule: any reward function that already returns a negative value
  (`*_l1`, `*_penalty` suffix returning ≤ 0) gets a **positive** weight. Every
  `Episode_Reward/<penalty>` must log ≤ 0 — this is the "bit four envs" bug class.
- No limit-proximity penalty on `_LEG_JOINTS` in this task (the split target
  legitimately sits near those joints' hard limits — see spec §3.1).
- `ENABLE_SYMMETRY = False` (asymmetric task).
- Every numeric geometry constant (`SPLIT_Z`, `SPLIT_JOINT_OVERRIDES`, natural
  resting pitch) must come from the Task 1 measurement script's actual output —
  never guessed (AGENTS.md's `STAND_Z` lesson).
- 64-env/5-iteration smoke test is mandatory before any real training run.

---

## Task 1: Settle-test measurement script

**Files:**
- Create: `scripts/measure_split_pose.py`
- Test: `tests/test_measure_split_pose.py`

**Interfaces:**
- Produces: `settle_split_pose(model: mujoco.MjModel, leg_overrides: dict[str, float], sim_seconds: float = 3.0, settle_seconds: float = 1.0) -> dict` — pure
  function, no env/gym dependency, importable from `scripts/measure_split_pose.py`.
  `leg_overrides` keys are actuator names (e.g. `"left_hip_pitch"`); values are
  absolute target angles in radians. Returns
  `{"z": float, "roll_proxy": float, "pitch_proxy": float, "max_tilt_proxy": float, "left_foot_x": float, "right_foot_x": float, "fell": bool}`.
  `roll_proxy`/`pitch_proxy` are the settled `projected_gravity_b[1]`/`[0]`
  components (dimensionless, same convention as the existing `crouch_pose_editor`-
  adjacent code at `mdp.py:1403`, NOT radians). `fell` is `True` if
  `max_tilt_proxy` (computed every step of the transient, not just at the end)
  exceeds 0.85 (roughly full topple) at any point.

- [ ] **Step 1: Write the failing test for the pure settle function**

```python
# tests/test_measure_split_pose.py
"""settle_split_pose: drives candidate split-leg targets from HOME under real
gravity/contacts and reports the settled trunk pose. Pure-model test (CPU,
no gym env) — mirrors the existing get_standup_spec() CPU-compile tests.
"""

import math

from mjlab_microduck.robot.microduck_constants import get_standup_spec
from scripts.measure_split_pose import settle_split_pose


def test_home_pose_settles_near_known_stand_z():
    # Sanity check against the value already measured and documented for
    # standup (STAND_Z = 0.115, microduck_standup_env_cfg.py) — same model,
    # no leg overrides, so this must reproduce that number to trust the script.
    model = get_standup_spec().compile()
    result = settle_split_pose(model, leg_overrides={}, sim_seconds=2.0)
    assert abs(result["z"] - 0.115) < 0.01
    assert abs(result["roll_proxy"]) < 0.05
    assert not result["fell"]


def test_extreme_asymmetric_override_is_detected_as_a_fall():
    # A leg overridden hard against its limit with the other leg untouched is
    # not a plausible split — the robot should topple, and the script must
    # report that instead of silently returning a low-but-"settled" reading.
    model = get_standup_spec().compile()
    result = settle_split_pose(
        model,
        leg_overrides={"left_hip_pitch": math.radians(85)},
        sim_seconds=2.0,
    )
    assert result["fell"]


def test_left_forward_right_back_reports_which_sign_moves_which_foot():
    # Doesn't assert a specific sign (that's what running the script tells a
    # human) — just that the two feet end up at DIFFERENT x positions when
    # driven with opposite-signed hip_pitch targets, proving the measurement
    # actually distinguishes "forward" from "backward" per leg.
    model = get_standup_spec().compile()
    result = settle_split_pose(
        model,
        leg_overrides={
            "left_hip_pitch": math.radians(50),
            "right_hip_pitch": math.radians(-50),
            "left_knee": math.radians(45),
            "right_knee": math.radians(-45),
        },
        sim_seconds=2.5,
    )
    assert abs(result["left_foot_x"] - result["right_foot_x"]) > 0.03
```

- [ ] **Step 2: Run test to verify it fails**

Run (inside the container): `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_measure_split_pose.py -v"`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'scripts.measure_split_pose'`

- [ ] **Step 3: Write `scripts/measure_split_pose.py`**

```python
"""Headless settle-test for candidate front-split leg targets.

Loads the same robot model as standup (robot_allcollisions.xml via
get_standup_spec()), drives the leg actuators toward candidate targets from
the HOME standing pose, steps physics for a few seconds under real gravity
and contacts, and reports the settled trunk pose PLUS the worst tilt seen
during the whole transient — a settle test that only checks the final state
can report a fallen robot as "resting fine" (AGENTS.md).

Usage:
    uv run python scripts/measure_split_pose.py \
        --left-hip-pitch 70 --left-knee 60 --left-ankle 15 \
        --right-hip-pitch -70 --right-knee -60 --right-ankle -15
"""

import argparse
import math
import re

import mujoco
import numpy as np

from mjlab_microduck.robot.microduck_constants import HOME_FRAME, get_standup_spec


def _home_ctrl(model: mujoco.MjModel) -> np.ndarray:
    ctrl = np.zeros(model.nu)
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        for pattern, val in HOME_FRAME.joint_pos.items():
            if re.search(pattern, name):
                ctrl[a] = float(val)
                break
    return ctrl


def settle_split_pose(
    model: mujoco.MjModel,
    leg_overrides: dict[str, float],
    sim_seconds: float = 3.0,
    settle_seconds: float = 1.0,
) -> dict:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    ctrl = _home_ctrl(model)
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        if name in leg_overrides:
            ctrl[a] = leg_overrides[name]
    data.ctrl[:] = ctrl

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")

    n_steps = int(sim_seconds / model.opt.timestep)
    settle_steps = int(settle_seconds / model.opt.timestep)
    max_tilt_proxy = 0.0

    for step in range(n_steps):
        mujoco.mj_step(model, data)
        quat = data.xquat[trunk_id]  # (w, x, y, z)
        tilt_proxy = 2.0 * (quat[1] ** 2 + quat[2] ** 2)  # ~= 1 - cos(tilt)
        if step > settle_steps:  # ignore the initial actuator-drive transient
            max_tilt_proxy = max(max_tilt_proxy, tilt_proxy)

    quat = data.xquat[trunk_id]
    roll_proxy = 2.0 * quat[1] * quat[3] + 2.0 * quat[0] * quat[2]  # gravity_b[1] equiv, small-angle
    pitch_proxy = 2.0 * quat[0] * quat[2] - 2.0 * quat[1] * quat[3]
    # Use MuJoCo's own body xmat to get an exact projected-gravity-style proxy
    # instead of the small-angle quat approximation above:
    rot = data.xmat[trunk_id].reshape(3, 3)
    gravity_b = rot.T @ np.array([0.0, 0.0, -1.0])
    pitch_proxy = float(-gravity_b[0])
    roll_proxy = float(gravity_b[1])

    return {
        "z": float(data.xpos[trunk_id][2]),
        "roll_proxy": roll_proxy,
        "pitch_proxy": pitch_proxy,
        "max_tilt_proxy": float(max_tilt_proxy),
        "left_foot_x": float(data.xpos[left_foot_id][0]),
        "right_foot_x": float(data.xpos[right_foot_id][0]),
        "fell": bool(max_tilt_proxy > 0.85),
    }


def main():
    p = argparse.ArgumentParser()
    for joint in (
        "left-hip-pitch", "left-knee", "left-ankle",
        "right-hip-pitch", "right-knee", "right-ankle",
    ):
        p.add_argument(f"--{joint}", type=float, default=None, help="degrees")
    p.add_argument("--sim-seconds", type=float, default=3.0)
    args = p.parse_args()

    overrides = {}
    for joint in (
        "left_hip_pitch", "left_knee", "left_ankle",
        "right_hip_pitch", "right_knee", "right_ankle",
    ):
        val = getattr(args, joint)
        if val is not None:
            overrides[joint] = math.radians(val)

    model = get_standup_spec().compile()
    result = settle_split_pose(model, overrides, sim_seconds=args.sim_seconds)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_measure_split_pose.py -v"`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/measure_split_pose.py tests/test_measure_split_pose.py
git commit -m "feat: add headless settle-test script for split target poses"
```

- [ ] **Step 6: Run the script to find real target values — RECORD THE OUTPUT**

Try a few candidate depths, starting shallow and increasing, e.g.:

```bash
docker exec microduck bash -c "cd /w && uv run python scripts/measure_split_pose.py \
  --left-hip-pitch 70 --left-knee 55 --left-ankle 15 \
  --right-hip-pitch -70 --right-knee -55 --right-ankle -15 --sim-seconds 3"
```

Check `fell` is `False`, note the actual sign that puts `left_foot_x > right_foot_x`
(swap all six signs together if it's reversed), and increase magnitude toward the
mechanical limits (§2/§3.1 of the spec: leave margin short of the true ±90°) while
`fell` stays `False`. Write down the final chosen values plus the settled `z`,
`pitch_proxy` — Task 5 needs these as `SPLIT_JOINT_OVERRIDES`, `SPLIT_Z`, and the
`pitch_split` target. There is no way to know these numbers before running this
script against the real model; do not guess them.

---

## Task 2: New mdp.py reward functions (roll_split, pitch_split)

**REVISED after Task 1**: `com_downward_velocity` was dropped (YAGNI — splits'
descent is a single-direction reach like "sit down", not standup's discontinuous
prone-recovery problem; `pose_split_l1`/`height_split_l1` already supply a dense
gradient). The existing `trunk_downward_velocity_penalty` is reused as-is instead,
introduced late by curriculum (see Task 5/6). See spec §3.4 for the full reasoning.

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (add after `com_upward_velocity`, ~line 934)
- Test: `tests/test_mdp_splits.py`

**Interfaces:**
- Consumes: nothing new — same `SceneEntityCfg`/`_DEFAULT_ASSET_CFG` pattern every
  other reward function in `mdp.py` uses.
- Produces:
  - `roll_split(env, asset_cfg=_DEFAULT_ASSET_CFG, std=0.45) -> Tensor`
  - `pitch_split(env, target_pitch, asset_cfg=_DEFAULT_ASSET_CFG, std=0.15) -> Tensor`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mdp_splits.py
"""roll_split / pitch_split: pure-tensor reward functions for the splits
task. Mock env/asset pattern mirrors tests/test_descent_speed.py — no real
mjlab env needed.
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
    env = _GravityEnv([[0.0, 0.0, -1.0]])  # upright, no roll
    out = roll_split(env, std=0.45)
    assert abs(float(out[0]) - 1.0) < 1e-5


def test_roll_split_is_generous_to_small_sway():
    # ~15deg-equivalent lateral gravity component should still score high —
    # the whole point of the "generous std" design decision.
    env = _GravityEnv([[0.0, 0.26, -0.97]])
    out = roll_split(env, std=0.45)
    assert float(out[0]) > 0.7


def test_pitch_split_peaks_at_the_target_not_at_zero():
    target = 0.55  # forward-lean proxy, e.g. measured value from Task 1
    env = _GravityEnv([[target, 0.0, -0.8]])
    out = pitch_split(env, target_pitch=target, std=0.15)
    assert abs(float(out[0]) - 1.0) < 1e-5
    out_at_vertical = pitch_split(env, target_pitch=target, std=0.15)
    env_vertical = _GravityEnv([[0.0, 0.0, -1.0]])
    out_at_vertical = pitch_split(env_vertical, target_pitch=target, std=0.15)
    assert float(out_at_vertical[0]) < float(out[0])


- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_mdp_splits.py -v"`
Expected: FAIL with `ImportError: cannot import name 'roll_split'`

- [ ] **Step 3: Implement the two functions in `mdp.py`**

Insert immediately after `com_upward_velocity` (before `fallen_too_long`, ~line 934):

```python
def roll_split(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.45,
) -> torch.Tensor:
    """Gaussian on lateral (roll) gravity-projection, target zero.

    Splits is a sagittal-plane trick — roll is the "you tipped over
    sideways" axis (the failure mode), not a natural part of the motion.
    `std` is DELIBERATELY generous (0.45 in the same gravity_b-projection
    units as mdp.py:1403's `target_pitch` proxy, not radians) so mild sway
    barely costs anything — see spec §3.3.
    """
    asset: Entity = env.scene[asset_cfg.name]
    roll_proxy = asset.data.projected_gravity_b[:, 1]
    return torch.exp(-((roll_proxy / std) ** 2))


def pitch_split(
    env: ManagerBasedRlEnv,
    target_pitch: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.15,
) -> torch.Tensor:
    """Gaussian on forward/back gravity-projection against a target.

    Unlike standup's upright_linear/upright_sharp (target = perfectly
    vertical), a front split plausibly needs the trunk to lean fore/aft as
    it settles — there's no arm counterweight. `target_pitch` = SPLIT_PITCH_TARGET
    (currently 0 -- a DESIGN DEFAULT, not a measurement: Task 1 went
    kinematic, which can't reveal a natural dynamic lean since that only
    exists once an active policy is holding the pose; see spec §2/§3.3).
    """
    asset: Entity = env.scene[asset_cfg.name]
    pitch_proxy = asset.data.projected_gravity_b[:, 0]
    return torch.exp(-(((pitch_proxy - target_pitch) / std) ** 2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_mdp_splits.py -v"`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_mdp_splits.py
git commit -m "feat: add roll_split/pitch_split reward functions"
```

---

## Task 3: New mdp.py curriculum function (pose_target_depth_curriculum)

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (add after `reward_weight`, ~line 3462)
- Test: `tests/test_mdp_splits.py` (append)

**Interfaces:**
- Consumes: same `env.reward_manager.get_term_cfg(...)` pattern as `reward_weight`
  (mdp.py:3442) — mutates the LIVE term cfg, never `env.cfg`.
- Produces: `pose_target_depth_curriculum(env, env_ids, reward_names: tuple[str, ...], joint_indices: tuple[int, ...], full_targets: dict[int, float], depth_stages: list[dict]) -> Tensor`.
  `depth_stages` = `[{"step": int, "fraction": float}, ...]` — at each stage, every
  reward in `reward_names` gets `target_overrides = {idx: fraction * full_targets[idx] for idx in joint_indices}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mdp_splits.py (append)

import torch

from mjlab_microduck.tasks.mdp import pose_target_depth_curriculum


class _TermCfg:
    def __init__(self, params):
        self.params = params


class _RewardManager:
    def __init__(self, term_cfgs):
        self._term_cfgs = term_cfgs

    def get_term_cfg(self, name):
        return self._term_cfgs[name]


class _CurricEnv:
    def __init__(self, step, term_cfgs):
        self.common_step_counter = step
        self.reward_manager = _RewardManager(term_cfgs)


def test_depth_curriculum_scales_targets_by_stage_fraction():
    full_targets = {2: 1.2, 3: 1.0}
    term = _TermCfg(params={"target_overrides": {}})
    env = _CurricEnv(step=1000, term_cfgs={"pose_split": term})
    stages = [
        {"step": 0, "fraction": 0.5},
        {"step": 500, "fraction": 0.75},
        {"step": 1500, "fraction": 1.0},
    ]
    pose_target_depth_curriculum(
        env, torch.tensor([]),
        reward_names=("pose_split",),
        joint_indices=(2, 3),
        full_targets=full_targets,
        depth_stages=stages,
    )
    # step=1000 is past the 500 stage but before 1500 -> fraction 0.75
    assert term.params["target_overrides"][2] == pytest.approx(0.9)
    assert term.params["target_overrides"][3] == pytest.approx(0.75)


def test_depth_curriculum_applies_to_every_named_reward():
    full_targets = {2: 1.0}
    terms = {
        "pose_split": _TermCfg(params={"target_overrides": {}}),
        "pose_split_l1": _TermCfg(params={"target_overrides": {}}),
    }
    env = _CurricEnv(step=0, term_cfgs=terms)
    pose_target_depth_curriculum(
        env, torch.tensor([]),
        reward_names=("pose_split", "pose_split_l1"),
        joint_indices=(2,),
        full_targets=full_targets,
        depth_stages=[{"step": 0, "fraction": 0.4}],
    )
    for name in ("pose_split", "pose_split_l1"):
        assert terms[name].params["target_overrides"][2] == pytest.approx(0.4)
```

Add `import pytest` to the top of `tests/test_mdp_splits.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_mdp_splits.py -v"`
Expected: FAIL with `ImportError: cannot import name 'pose_target_depth_curriculum'`

- [ ] **Step 3: Implement in `mdp.py`**

Insert immediately after `reward_weight` (before `com_range_curriculum`, ~line 3462):

```python
def pose_target_depth_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_names: tuple[str, ...],
    joint_indices: tuple[int, ...],
    full_targets: dict[int, float],
    depth_stages: list[dict],
) -> torch.Tensor:
    """Ramp a pose-target reward's depth from a fraction of the full target to 100%.

    Unlike `reward_weight` (ramps HOW MUCH a term counts) this ramps WHAT
    the term is rewarding — the target angle itself. Needed for targets far
    from the natural resting pose (e.g. a full split): dropping the maximal
    target in from step 0 risks a shallow "good enough" local optimum with
    no pressure to go deeper (spec §5). `depth_stages`:
    `[{"step": int, "fraction": float}, ...]`; latest passed stage wins,
    applied identically to every reward in `reward_names`.
    """
    del env_ids
    fraction = depth_stages[0]["fraction"]
    for stage in depth_stages:
        if env.common_step_counter >= stage["step"]:
            fraction = stage["fraction"]
    overrides = {idx: fraction * full_targets[idx] for idx in joint_indices}
    for name in reward_names:
        term_cfg = env.reward_manager.get_term_cfg(name)
        term_cfg.params["target_overrides"] = dict(overrides)
    return torch.tensor([fraction])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_mdp_splits.py -v"`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_mdp_splits.py
git commit -m "feat: add pose_target_depth_curriculum for the split-depth ramp"
```

---

## Task 4: Env cfg skeleton — reset, termination, observations

**Files:**
- Create: `src/mjlab_microduck/tasks/microduck_splits_env_cfg.py`
- Test: `tests/test_splits_cfg.py`

**Interfaces:**
- Consumes: `mjlab_microduck.tasks.mdp` (as `microduck_mdp`), `mjlab.tasks.velocity.mdp`
  (as `mdp`), `MICRODUCK_STANDUP_ROBOT_CFG` from `microduck_constants`,
  `make_velocity_env_cfg` from `microduck_velocity_env_cfg`, the head-command
  resample constant `HEAD_POSE_CMD_RESAMPLE_S` also imported from
  `microduck_velocity_env_cfg` (same as standup does). No `body_pose` command is
  created — `body_command` obs stay zero-padded (spec §3.6: HOME-tracking only
  for v1), so `BODY_POSE_CMD_RESAMPLE_S` is not imported here.
- Produces: `make_microduck_splits_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg`.
  Rewards/curricula are added in Tasks 5–6 — this task only needs the function to
  build without error.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits_cfg.py
import math

from mjlab_microduck.tasks.microduck_splits_env_cfg import (
    _LEG_JOINTS,
    EPISODE_LENGTH_S,
    make_microduck_splits_env_cfg,
)


def test_env_builds_train_and_play():
    assert make_microduck_splits_env_cfg() is not None
    assert make_microduck_splits_env_cfg(play=True) is not None


def test_episode_length_matches_constant():
    cfg = make_microduck_splits_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S


def test_leg_joint_indices_match_standup_convention():
    # Same 14-joint layout as every other allcollisions-model task
    # (AGENTS.md: 0-4 left leg, 5-8 neck/head, 9-13 right leg).
    assert _LEG_JOINTS == [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def test_reset_is_standing_only_no_ground_state_mix():
    # Splits starts from standing, not from an on-the-ground recovery mix —
    # no standup-style prone/back spawns.
    cfg = make_microduck_splits_env_cfg()
    params = cfg.events["set_ground_state"].params
    assert params["standing_prob"] == 1.0
    assert params["face_down_prob"] == 0.0
    assert params["face_up_prob"] == 0.0
    assert params["sitting_prob"] == 0.0
    assert "ground_state_mix" not in cfg.curriculum


def test_no_fall_termination_but_nan_state_present():
    cfg = make_microduck_splits_env_cfg()
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_roll_and_pitch_terminations_present():
    cfg = make_microduck_splits_env_cfg()
    assert "tipped_sideways" in cfg.terminations
    assert "tipped_forward_or_back" in cfg.terminations


def test_obs_is_still_61d_command_layout():
    cfg = make_microduck_splits_env_cfg()
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert "body_command" in cfg.observations[group].terms
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py -v"`
Expected: FAIL with `ModuleNotFoundError: No module named 'mjlab_microduck.tasks.microduck_splits_env_cfg'`

- [ ] **Step 3: Write the skeleton cfg file**

```python
# src/mjlab_microduck/tasks/microduck_splits_env_cfg.py
"""Microduck front-split task — episodic descend-and-hold.

Robot starts standing (HOME, noisy), lowers into a front split (LEFT leg
forward, RIGHT leg back — fixed, not randomized: see the design spec for
why: ENABLE_SYMMETRY must stay False for an asymmetric target). Single
fixed target pose rewarded from t=0, no waypoints — the descent path is
left for RL to discover, same as standup/sit.

Side splits are not mechanically possible on this robot: hip_roll only has
±22° of range (robot_walk.xml), so the split has to happen fore-aft using
hip_pitch/knee/ankle, which all have a full ±90° range.

This episode covers descent + hold ONLY. Recovery back to standing is an
explicit non-goal — a future companion task (mirroring how standup mirrors
sit), not part of this one.

See docs/superpowers/specs/2026-09-01-microduck-splits-design.md for the
full reward-design reasoning.
"""

import math
from copy import deepcopy

# Symmetry — OFF. This is a deliberately asymmetric target (left leg
# forward, right leg back); the mirror loss would train the policy to
# fight its own target (AGENTS.md: never for asymmetric tasks).
ENABLE_SYMMETRY = False

# ── Domain randomisation (matched to standup for sim2real parity) ─────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = True  # ramped from 0 by curriculum (Task 6)
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

# Episode: gentle descent (~2-3s) + a real hold, so the policy learns to
# STAY in the split, not just touch it and get cut off.
EPISODE_LENGTH_S = 6.0

# Empirically-measured standing trunk height (standup's measured value,
# same model — never guess this, see AGENTS.md's STAND_Z lesson).
STAND_Z = 0.115

# ── Split target — FROM Task 1's measure_split_pose.py output, NOT guessed ────
# TODO(Task 5): replace with the real values recorded in Task 1 Step 6.
SPLIT_Z = 0.115  # placeholder until Task 5 fills in the measured value
SPLIT_JOINT_OVERRIDES = {
    # left leg forward
    2:  0.0,   # left_hip_pitch
    3:  0.0,   # left_knee
    4:  0.0,   # left_ankle
    # right leg back
    11: 0.0,   # right_hip_pitch
    12: 0.0,   # right_knee
    13: 0.0,   # right_ankle
}
SPLIT_PITCH_TARGET = 0.0  # projected_gravity_b[:,0] proxy, from Task 1

_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

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


def make_microduck_splits_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck front-split environment configuration."""

    site_names = ["left_foot", "right_foot"]

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

    # ── Rewards: drop walking-specific terms (rewards added in Task 5) ────────
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

    # ── Observations (identical layout to walking/standup) ────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

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

    # ── Head pose command (HOME-tracking only for v1 — see spec §3.6) ─────────
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

    # ── Command: tiny noise around zero (obs-shape parity only) ──────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    command.resampling_time_range = (EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2)
    command.debug_vis = False
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    # ── Terminations ────────────────────────────────────────────────────────
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": ("feet_ground_contact",)},
    )
    # Roll/pitch failure bands (spec §4). Roll threshold is WIDER than
    # roll_split's reward std (0.45) — the reward is soft, this is the hard
    # backstop, same split standup uses between reward std and termination.
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

    # Reset: standing only (no ground-state mix — splits starts upright,
    # not recovering off the floor). Reuses standup's tested
    # set_random_ground_state rather than writing a new reset function.
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob":  0.0,
            "face_up_prob":    0.0,
            "sitting_prob":    0.0,
            "standing_prob":   1.0,
            "standing_z_min":  STAND_Z - 0.005,
            "standing_z_max":  STAND_Z + 0.005,
            "sitting_tilt_max": math.radians(6),
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

    if not rough:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    return cfg


# ── RL runner config (Task 6 fills in experiment_name / final values) ─────────
MicroduckSplitsRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="microduck_splits",
    run_name="microduck_splits",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=6000,
)
```

- [ ] **Step 4: Add the `gravity_proxy_out_of_band` termination helper it references**

This is a new, small, reusable termination function — add it to `mdp.py` right after
`robot_state_is_nan` (~line 990):

```python
def gravity_proxy_out_of_band(
    env: ManagerBasedRlEnv,
    axis: int,
    target: float,
    band: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when a projected_gravity_b axis strays more than `band` from
    `target`. Generic hard backstop for roll_split/pitch_split (axis 1 / 0) —
    the reward is soft (generous std), this is the actual fall/face-plant
    cutoff, set wider than the reward's std by design.
    """
    asset: Entity = env.scene[asset_cfg.name]
    proxy = asset.data.projected_gravity_b[:, axis]
    return torch.abs(proxy - target) > band
```

Add a matching unit test to `tests/test_mdp_splits.py` using the `_GravityEnv` mock
from Task 2:

```python
from mjlab_microduck.tasks.mdp import gravity_proxy_out_of_band


def test_gravity_proxy_out_of_band_triggers_past_threshold():
    env = _GravityEnv([[0.0, 0.9, -0.3]])
    out = gravity_proxy_out_of_band(env, axis=1, target=0.0, band=0.75)
    assert bool(out[0]) is True


def test_gravity_proxy_out_of_band_stays_off_within_threshold():
    env = _GravityEnv([[0.0, 0.3, -0.9]])
    out = gravity_proxy_out_of_band(env, axis=1, target=0.0, band=0.75)
    assert bool(out[0]) is False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py tests/test_mdp_splits.py -v"`
Expected: PASS (all tests in both files)

- [ ] **Step 6: Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_splits_env_cfg.py src/mjlab_microduck/tasks/mdp.py tests/test_splits_cfg.py tests/test_mdp_splits.py
git commit -m "feat: add splits env cfg skeleton (reset/termination/observations)"
```

---

## Task 5: Wire in reward terms (using Task 1's measured values)

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_splits_env_cfg.py`
- Modify: `tests/test_splits_cfg.py`

**Interfaces:**
- Consumes: `microduck_mdp.pose_target_match`, `pose_l1_penalty`, `height_target_gaussian`,
  `height_l1_penalty`, `roll_split`, `pitch_split`, `trunk_vertical_accel_penalty`,
  `trunk_downward_velocity_penalty`, `body_ang_vel_at_height` (Task 2 + existing
  mdp.py — `trunk_downward_velocity_penalty` reused as-is, no new code).
  `mdp.self_collision_cost`, `mdp.action_rate_l2` (`mjlab.tasks.velocity.mdp`).

- [ ] **Step 1: Update `SPLIT_Z`, `SPLIT_JOINT_OVERRIDES`, `SPLIT_PITCH_TARGET` with the real measured values**

Replace the placeholder block from Task 4 with the actual numbers recorded in
Task 1 Step 6. Example shape (replace every value with what was actually measured —
do not reuse these illustrative numbers):

```python
SPLIT_Z = 0.058  # <- measured trunk_base z from Task 1's settle test
SPLIT_JOINT_OVERRIDES = {
    2:  1.22,   # left_hip_pitch  <- measured
    3:  0.96,   # left_knee       <- measured
    4:  0.26,   # left_ankle      <- measured
    11: -1.22,  # right_hip_pitch <- measured
    12: -0.96,  # right_knee      <- measured
    13: -0.26,  # right_ankle     <- measured
}
SPLIT_PITCH_TARGET = 0.60  # <- measured pitch_proxy from Task 1
```

- [ ] **Step 2: Write the failing cfg tests for the reward set**

Append to `tests/test_splits_cfg.py`:

```python
def test_pose_and_height_rewards_present_with_positive_weights():
    cfg = make_microduck_splits_env_cfg()
    for name in ("pose_split", "pose_split_l1", "height_split", "height_split_sharp", "height_split_l1"):
        assert name in cfg.rewards
        assert cfg.rewards[name].weight > 0

def test_orientation_rewards_present():
    cfg = make_microduck_splits_env_cfg()
    assert "roll_split" in cfg.rewards
    assert "pitch_split" in cfg.rewards
    assert cfg.rewards["roll_split"].weight > 0
    assert cfg.rewards["pitch_split"].weight > 0
    # Generous roll std, per explicit design direction.
    assert cfg.rewards["roll_split"].params["std"] > cfg.rewards["pitch_split"].params["std"]

def test_no_limit_proximity_penalty_on_leg_joints():
    cfg = make_microduck_splits_env_cfg()
    assert "dof_pos_limits" not in cfg.rewards
    assert "limit_proximity" not in cfg.rewards

def test_already_negative_penalties_use_positive_weights():
    # Same sign-bug class check as standup/roller_standup's test.
    cfg = make_microduck_splits_env_cfg()
    for name in ("height_split_l1", "pose_split_l1", "gentle_descent"):
        assert cfg.rewards[name].weight > 0, f"{name} calls a function returning negative already"
    for name in ("action_rate_l2", "joint_torque_rate_l2"):
        assert cfg.rewards[name].weight < 0

def test_settle_damping_and_torque_rate_start_at_zero():
    # Discovery-vs-polish staging (spec §5): must not tax the descent attempt
    # before the skill exists.
    cfg = make_microduck_splits_env_cfg()
    assert cfg.rewards["settle_damping"].weight == 0.0
    assert cfg.rewards["joint_torque_rate_l2"].weight == 0.0

def test_pose_rewards_target_split_joints_via_overrides():
    from mjlab_microduck.tasks.microduck_splits_env_cfg import _LEG_JOINTS, SPLIT_JOINT_OVERRIDES
    cfg = make_microduck_splits_env_cfg()
    for name in ("pose_split", "pose_split_l1"):
        assert cfg.rewards[name].params["joint_indices"] == _LEG_JOINTS
        assert cfg.rewards[name].params["target_overrides"] == SPLIT_JOINT_OVERRIDES

def test_self_collisions_present():
    cfg = make_microduck_splits_env_cfg()
    assert "self_collisions" in cfg.rewards
    assert cfg.rewards["self_collisions"].weight < 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py -v"`
Expected: FAIL — `KeyError: 'pose_split'` etc.

- [ ] **Step 4: Add the reward block**

Insert after the observations block and before the terminations block in
`make_microduck_splits_env_cfg`:

```python
    # ── Rewards: front-split target, descent+hold ──────────────────────────
    # Weights mirror standup's ÷4-rescaled ratios (AGENTS.md: compare reward
    # MASS not absolute weight when copying regularizers between envs — this
    # task's structure is architecturally identical to standup's, just a
    # different geometric target, so the same task/regularizer mass ratio
    # applies as a starting point). Expect the usual whack-a-mole tuning pass.
    cfg.rewards["pose_split"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={"std": 0.5, "joint_indices": _LEG_JOINTS, "target_overrides": SPLIT_JOINT_OVERRIDES},
    )
    cfg.rewards["pose_split_l1"] = RewardTermCfg(
        func=microduck_mdp.pose_l1_penalty,
        weight=1.25,
        params={"joint_indices": _LEG_JOINTS, "target_overrides": SPLIT_JOINT_OVERRIDES},
    )

    cfg.rewards["height_split"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.04,
            "target_height": SPLIT_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_split_sharp"] = RewardTermCfg(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.015,
            "target_height": SPLIT_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.rewards["height_split_l1"] = RewardTermCfg(
        func=microduck_mdp.height_l1_penalty,
        weight=7.5,
        params={
            "target_height": SPLIT_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Roll: generous std (0.45) — mild sideways sway is fine, per explicit
    # design direction. Pitch: tighter (0.15), tracks the MEASURED natural
    # resting pitch, not vertical.
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

    # Speed cap, not a bootstrap reward (com_downward_velocity was dropped,
    # spec §3.4 REVISED note) — existing function, reused as-is. Weight
    # starts at 0, ramped in by curriculum (Task 6) only after descent is
    # discovered, same discovery-vs-polish timing as settle_damping below.
    cfg.rewards["descent_speed_cap"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=0.0,
        params={
            "max_down_vel": 0.15,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # POSITIVE weight — trunk_vertical_accel_penalty already returns -|a_z|
    # (the "bit four envs" sign bug: a negative weight here would double-
    # negate into a reward for hard impacts).
    cfg.rewards["gentle_descent"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Settle-damping (standup's arrival_damping equivalent) — starts at 0,
    # ramped in by curriculum (Task 6) only after descent is discovered.
    cfg.rewards["settle_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low": SPLIT_Z,
            "height_high": SPLIT_Z + 0.02,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Head — HOME-tracking only for v1 (spec §3.6): kept alive for obs-slot
    # parity, lightly weighted.
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=0.75,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # ── Sim2real regularisers (standup's set) ──────────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
```

(`self_collision_cfg` is already in scope from Task 4's sensor definitions earlier
in the function.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py -v"`
Expected: PASS (all tests, including Task 4's)

- [ ] **Step 6: Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_splits_env_cfg.py tests/test_splits_cfg.py
git commit -m "feat: wire split-pose/orientation/motion-quality reward terms"
```

---

## Task 6: Curricula and RL runner cfg finalization

**Files:**
- Modify: `src/mjlab_microduck/tasks/microduck_splits_env_cfg.py`
- Modify: `tests/test_splits_cfg.py`

- [ ] **Step 1: Write the failing curriculum tests**

Append to `tests/test_splits_cfg.py`:

```python
def test_split_depth_curriculum_ramps_to_full_target():
    cfg = make_microduck_splits_env_cfg()
    assert "split_depth" in cfg.curriculum
    stages = cfg.curriculum["split_depth"].params["depth_stages"]
    fractions = [s["fraction"] for s in stages]
    assert fractions[0] < 1.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert cfg.curriculum["split_depth"].params["reward_names"] == ("pose_split", "pose_split_l1")

def test_discovery_vs_polish_staging_starts_at_zero():
    cfg = make_microduck_splits_env_cfg()
    for name in ("settle_damping_weight", "torque_rate_weight", "descent_speed_cap_weight"):
        assert name in cfg.curriculum
        first_stage = cfg.curriculum[name].params["weight_stages"][0]
        assert first_stage["step"] == 0
        assert first_stage["weight"] == 0.0

def test_descent_speed_cap_ramps_to_a_positive_weight():
    # descent_speed_cap wraps trunk_downward_velocity_penalty, which returns
    # -clamp(..., min=0.0) -- ALWAYS <= 0, self-negating. A negative final
    # weight here would double-negate it into a reward for violent drops
    # (the exact "bit four envs" bug class) -- this locks the sign in.
    cfg = make_microduck_splits_env_cfg()
    stages = cfg.curriculum["descent_speed_cap_weight"].params["weight_stages"]
    assert stages[-1]["weight"] > 0

def test_action_rate_curriculum_ramps_like_standup():
    cfg = make_microduck_splits_env_cfg()
    weights = [s["weight"] for s in cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert weights[0] == -0.1
    assert weights[-1] == -1.0
    assert weights == sorted(weights, reverse=True)

def test_push_curriculum_ramps_from_zero():
    cfg = make_microduck_splits_env_cfg()
    assert "push_magnitude" in cfg.curriculum
    stages = cfg.curriculum["push_magnitude"].params["push_stages"]
    assert stages[0]["velocity_range"]["x"] == (0.0, 0.0)

def test_task_registered_with_correct_experiment_name():
    from mjlab_microduck.tasks.microduck_splits_env_cfg import MicroduckSplitsRlCfg
    assert MicroduckSplitsRlCfg.experiment_name == "microduck_splits"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py -v"`
Expected: FAIL — `KeyError: 'split_depth'`

- [ ] **Step 3: Add curricula, right before `return cfg`**

```python
    # ── Curriculum ──────────────────────────────────────────────────────────
    # Split-depth: ramp the target from a modest depth to the full measured
    # target (spec §5) — every difficulty curriculum already in this repo
    # ramps easy->hard rather than starting at max, for the same reason.
    cfg.curriculum["split_depth"] = CurriculumTermCfg(
        func=microduck_mdp.pose_target_depth_curriculum,
        params={
            "reward_names": ("pose_split", "pose_split_l1"),
            "joint_indices": tuple(SPLIT_JOINT_OVERRIDES.keys()),
            "full_targets": dict(SPLIT_JOINT_OVERRIDES),
            "depth_stages": [
                {"step": 0,          "fraction": 0.55},
                {"step": 500 * 24,   "fraction": 0.70},
                {"step": 1000 * 24,  "fraction": 0.85},
                {"step": 1500 * 24,  "fraction": 1.00},
            ],
        },
    )

    # Discovery-vs-polish: identical reasoning to standup's arrival_damping/
    # torque_rate curricula — any attempt-tax active while the descent skill
    # is still being discovered makes "stay standing" the optimum.
    cfg.curriculum["settle_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "settle_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": -0.025},
                {"step": 3000 * 24,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": -1e-3},
            ],
        },
    )
    # trunk_downward_velocity_penalty returns `-clamp(..., min=0.0)` --
    # ALWAYS <= 0, i.e. self-negating (same class as pose_l1_penalty /
    # height_l1_penalty). Per AGENTS.md's sign rule this needs a POSITIVE
    # weight -- a negative weight here would double-negate it into a
    # REWARD for fast, violent drops. (Caught on paper while writing this
    # plan: the first draft had -2.0 here, exactly backwards.)
    cfg.curriculum["descent_speed_cap_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "descent_speed_cap",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2000 * 24,  "weight": 2.0},
            ],
        },
    )
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
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
    if ENABLE_VELOCITY_PUSHES:
        # Ramped from zero, UNLIKE standup (which pushes from step 0) — a
        # push mid-descent is a different perturbation than one while
        # holding a stand (spec §5, left as an explicit open call).
        cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0,          "velocity_range": {"x": (0.0, 0.0),   "y": (0.0, 0.0)}},
                    {"step": 1000 * 24,  "velocity_range": {"x": (-0.1, 0.1),  "y": (-0.1, 0.1)}},
                    {"step": 2000 * 24,  "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )

    return cfg
```

Also update `MicroduckSplitsRlCfg.experiment_name`/`run_name` — already set to
`"microduck_splits"` in Task 4's skeleton, no change needed here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py -v"`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/microduck_splits_env_cfg.py tests/test_splits_cfg.py
git commit -m "feat: add split-depth and discovery-vs-polish curricula"
```

---

## Task 7: Task registration

**Files:**
- Modify: `src/mjlab_microduck/tasks/__init__.py`
- Modify: `tests/test_splits_cfg.py`

**Interfaces:**
- Consumes: `make_microduck_splits_env_cfg`, `MicroduckSplitsRlCfg` from Task 4/6.

- [ ] **Step 1: Write the failing registration test**

Append to `tests/test_splits_cfg.py`:

```python
def test_task_is_registered():
    from mjlab.tasks.registry import list_tasks
    import mjlab_microduck.tasks  # noqa: F401 (import triggers registration)
    assert "Mjlab-Splits-Flat-MicroDuck" in list_tasks()
    assert "Mjlab-Splits-Flat-Backlash-MicroDuck" in list_tasks()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py::test_task_is_registered -v"`
Expected: FAIL — assertion error, task id not in `list_tasks()`

- [ ] **Step 3: Register the task**

In `src/mjlab_microduck/tasks/__init__.py`, add the import near the other task
imports (after the `microduck_roulade_env_cfg` import block, ~line 69):

```python
from .microduck_splits_env_cfg import (
    make_microduck_splits_env_cfg,
    MicroduckSplitsRlCfg,
)
```

Add the registration call after the `Mjlab-StandUp-Rough-MicroDuck` block (~line 121),
before the `SitStand` section:

```python
# Splits — episodic front-split descend-and-hold
register_mjlab_task(
    task_id="Mjlab-Splits-Flat-MicroDuck",
    env_cfg=make_microduck_splits_env_cfg(),
    play_env_cfg=make_microduck_splits_env_cfg(play=True),
    rl_cfg=MicroduckSplitsRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Splits-Rough-MicroDuck",
    env_cfg=make_microduck_splits_env_cfg(rough=True),
    play_env_cfg=make_microduck_splits_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSplitsRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
```

Add an entry to the `_BACKLASH_TASKS` table (find `_BL_ALLCOL` — the same robot-cfg
symbol standup's backlash entries use, since splits uses `MICRODUCK_STANDUP_ROBOT_CFG`
same as standup):

```python
    ("Mjlab-Splits-Flat-Backlash-MicroDuck", make_microduck_splits_env_cfg, {}, MicroduckSplitsRlCfg, _BL_ALLCOL),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/test_splits_cfg.py -v"`
Expected: PASS (all tests)

- [ ] **Step 5: Run the FULL test suite to catch any cross-task regression**

Run: `docker exec microduck bash -c "cd /w && uv run --with pytest pytest tests/ -q"`
Expected: PASS, no failures introduced in unrelated task tests.

- [ ] **Step 6: Commit**

```bash
git add src/mjlab_microduck/tasks/__init__.py tests/test_splits_cfg.py
git commit -m "feat: register Mjlab-Splits-Flat/Rough/Backlash-MicroDuck tasks"
```

---

## Task 8: Smoke test (mandatory before any real training run)

**Files:** none (verification only)

- [ ] **Step 1: Run the 64-env/5-iteration smoke test**

Run:
```bash
docker exec microduck bash -c "
  cd /w && uv run python -m mjlab_microduck.train_cli Mjlab-Splits-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --agent.logger tensorboard --no-wandb
"
```
(per `HOW-TO.md`: `uv run train` is broken in this container's `uv sync` due to an
entry-point collision — invoke `python -m mjlab_microduck.train_cli` directly.)

Expected: completes 5 iterations with no exceptions, no NaN warnings from
`nan_state`/the NaN-safe reward patch, and every reward term in the printed
per-iteration summary has a finite value.

- [ ] **Step 2: Confirm obs shape and ONNX export path both work**

```bash
docker exec microduck bash -c "cd /w && uv run scripts/export.py Mjlab-Splits-Flat-MicroDuck --checkpoint-file <path-to-the-smoke-test-checkpoint>"
```
Expected: exports without error; confirms the 61D obs contract holds and the
normalizer bakes in cleanly (AGENTS.md: export.py is the mandatory ONNX path).

- [ ] **Step 3: Report results, no commit needed**

This task produces no file changes — it's the final go/no-go gate before a real
training run (`--env.scene.num-envs 4096 --agent.max_iterations 6000 --hf-jobs`,
per `HOW-TO.md`'s training-a-new-run recipe). If anything fails here, fix it as a
new commit and re-run this task rather than proceeding to a real run.

---

## Self-review notes

- **Spec coverage:** §1 (task shape) → Task 4. §2 (measurement) → Task 1.
  §3.1 (pose) → Task 5. §3.2 (height) → Task 5. §3.3 (roll/pitch) → Task 2 + Task 5.
  §3.4 (motion quality) → Task 2 + Task 5. §3.5 (regularizers) → Task 5.
  §3.6 (head) → Task 5. §4 (termination) → Task 4. §5 (curricula) → Task 3 + Task 6.
  §6 (testing) → every task's own test file, plus Task 8. §7 open questions are
  explicitly left as starting points with documented reasoning (push-ramp timing, no
  `splits_composite` in v1, `SPLIT_PITCH_TARGET`'s design-default status) — not gaps,
  deliberate v1 scope.
- **Placeholder scan:** none remaining — `SPLIT_Z`/`SPLIT_JOINT_OVERRIDES` were
  resolved by Task 1's actual measurement (75° hip_pitch, z=0.098) before Task 5 was
  written; `SPLIT_PITCH_TARGET=0` is an explicit design default, not a TODO.
- **Type consistency:** `pose_target_depth_curriculum`'s `reward_names`/`joint_indices`/
  `full_targets` signature (Task 3) matches exactly how Task 6 calls it. `roll_split`/
  `pitch_split` signatures (Task 2) match their usage in Task 5's `RewardTermCfg.params`.
  `trunk_downward_velocity_penalty` (existing function) is used with a POSITIVE
  final-stage weight (Task 6), matching its self-negating return value — this was
  caught and fixed during plan-writing, not left for the implementation/review loop
  to find. `gravity_proxy_out_of_band` (Task 4) matches its two termination-cfg call
  sites.
