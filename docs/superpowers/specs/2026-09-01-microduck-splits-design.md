# Microduck splits (front split, episodic pose-landing)

**Date**: 2026-09-01
**New file**: `src/mjlab_microduck/tasks/microduck_splits_env_cfg.py`
**Task id**: `Mjlab-Splits-Flat-MicroDuck` (+ `-Rough-` and `-Backlash-` variants per the
existing registration pattern in `tasks/__init__.py`)
**Template**: standup (`microduck_standup_env_cfg.py`) — closest match per AGENTS.md's
"episodic trick ending in a pose" case. Built on `make_velocity_env_cfg()`, same as
every other task family, so DR/obs/noise/delays/NaN-guard stay in sync for free.

## 1. Objective

Episodic trick: robot starts standing (HOME, with reset noise), lowers itself into a
**front split** (one leg extended forward, one extended back, in the sagittal
plane — the plane the robot walks in) and holds it for the rest of the episode.
Single fixed asymmetric target pose, rewarded from t=0 — no waypoint trajectory, no
episode-progress gating. The descent path is left for RL to discover, matching the
"episodic pose-landing" rule in AGENTS.md (fixed target + L1/Gaussian, not
keyframe/waypoint tracking).

This episode covers **descent + hold only**. Recovery back to standing from the
split is an explicit non-goal here — it would be a separate companion task later
(mirroring how standup mirrors sit), not part of this episode.

### Why front split, not side split

Checked against the actual joint ranges in `robot_walk.xml` / `robot_allcollisions.xml`:

| Joint | Range |
|---|---|
| `hip_roll` (abduction, side-to-side) | **±0.384 rad (±22°)** |
| `hip_pitch` | ±1.571 rad (±90°) |
| `knee` | ±1.571 rad (±90°) |
| `ankle` | ±1.571 rad (±90°) |

A side/straddle split needs hip abduction the hardware doesn't have (±22° max). A
front split only needs hip_pitch/knee/ankle, which all have a full ±90° range — the
only version that's mechanically real. **Fixed leg assignment**: left leg forward,
right leg back, always (not randomized per episode) — chosen for simplicity;
`ENABLE_SYMMETRY = False` follows directly, since AGENTS.md is explicit the mirror
loss is never for asymmetric tasks (it would train the policy to fight its own
lopsided target).

## 2. Target pose — values TBD, must be measured in sim

Per AGENTS.md's standup lesson (a 5mm-wrong `STAND_Z` once made the goal physically
unreachable for days): **none of the numbers below get guessed**. Before locking the
reward config, load the robot in the viewer, drive the candidate leg joint values by
hand, hold for ~3s from a noisy standing init, and read off:

- `SPLIT_Z` — `trunk_base` z once settled (mirrors `STAND_Z`'s derivation)
- natural resting **pitch** of the trunk in the settled pose (front split does not
  necessarily settle trunk-vertical — legs move, but there's no guarantee the trunk
  stays bolt-upright with no arms to counterbalance)
- the actual **safe target angle** for hip_pitch/knee/ankle, short of the ±90° hard
  stop (see §3 for why "short of the limit" is the deliberate choice, not a
  placeholder)

`SPLIT_JOINT_OVERRIDES` (dict, same shape as standup's `SITTING_JOINT_OVERRIDES`)
gets filled in from that measurement, not from CAD geometry or guesswork.

## 3. Reward design

### 3.1 Pose target (legs)

```python
cfg.rewards["pose_split"] = RewardTermCfg(
    func=microduck_mdp.pose_target_match,       # Gaussian, sharp near target
    weight=<TBD>,
    params={"std": <TBD>, "joint_indices": _LEG_JOINTS,
            "target_overrides": SPLIT_JOINT_OVERRIDES},
)
cfg.rewards["pose_split_l1"] = RewardTermCfg(
    func=microduck_mdp.pose_l1_penalty,          # linear, constant gradient
    weight=<TBD>,
    params={"joint_indices": _LEG_JOINTS, "target_overrides": SPLIT_JOINT_OVERRIDES},
)
```

Same Gaussian+L1 pairing as standup's `pose_stand_legs`/`pose_stand_l1`: the
Gaussian is flat (no gradient) far from target, so the L1 term is what bootstraps
the policy toward the target at all; the Gaussian is what polishes the final
approach.

**No limit-proximity penalty on the split-leg joints.** AGENTS.md's stock
limit-proximity rule exists to stop a policy from lazily parking a joint against a
hard limit as a resting crutch elsewhere. Here the *goal* legitimately sits near the
limit (hip_pitch/knee/ankle at or near ±90°) — applying that penalty to these
joints would directly fight the pose reward at the exact point we want the policy to
reach. Resolution: skip the penalty on `_LEG_JOINTS` for this task, and keep a
margin between the reward target and the true mechanical max (§2) so the policy is
never asked to command the actual hard stop. `dof_pos_limits` (mjlab's stock,
last-7.5%-of-range term) stays off for the same reason if it's in the base template.

Optional: a `splits_composite` multiplicative score (pose × height × roll), mirroring
standup's `standing_composite` — AGENTS.md's rule that multiplicative composites
collapse compromise-basin behavior at goal states applies here too (a policy that's
80% split via a half-hearted lean should score far worse than additive would give
it). Add if additive terms show a compromise basin during training; not required
for the first cut.

### 3.2 Height

Two-layer Gaussian, same shape as standup's `height_stand`/`height_stand_sharp`
(wide std for the bootstrap pull down from standing height, narrow std for a sharp
peak at `SPLIT_Z`) plus an L1 layer (`height_split_l1`, mirrors `height_stand_l1`)
so a "stay standing" local optimum isn't net-positive once the L1 cost is applied.

### 3.3 Orientation — roll and pitch handled separately (NOT standup's combined tilt)

Standup's `body_upright_linear`/`upright_sharp` measure total tilt from vertical
(`cos(tilt)`) — right for standing, wrong here. A front split plausibly needs the
trunk to lean forward/back somewhat as it settles (no arms to counterbalance), so a
single "distance from vertical" reward would fight the natural pose. New reward
function reading `projected_gravity`'s x/y components independently instead of the
combined angle:

- **`roll_split`** — Gaussian on roll only, **generous std** (per explicit
  direction: "so long as it does not fall, tipping side to side a little is fine" —
  mild sway should cost almost nothing)
- **`pitch_split`** — Gaussian tracking the *measured* natural resting pitch (§2),
  not pure vertical — same "measure it, don't guess it" rule as height

### 3.4 Motion quality

- **`gentle_descent`** — `trunk_vertical_accel_penalty` on `|a_z|`, small
  **positive** weight (the function already returns `-|a_z|` — a negative weight
  here double-negates into a reward for the shock, the exact "bit four envs" sign
  bug AGENTS.md calls out). Global, not phase-gated, same as standup's `gentle_rise`.
- **`com_downward_velocity`** — mirror of standup's `com_upward_velocity`: rewards
  descending CoM velocity so any lowering attempt pays immediately instead of only
  at the destination (jackpot-avoidance — a pure destination reward lets the policy
  camp standing, which is a stable, high-scoring local optimum on every other term).
  Gate: TBD whether active for the whole descent or only once already fairly low
  (open question, not yet decided — try whole-descent first, tighten if the policy
  learns to farm a fast initial drop).
- **settle-damping** (standup's `arrival_damping` equivalent) — angular-velocity
  penalty gated on height/tilt near the target, **starts at weight 0**, ramped in
  by curriculum only after descent is discovered (§4).

### 3.5 Sim2real regularizers (mostly reused, standup's exact set/weights as starting point)

`action_rate_l2` (curriculum-ramped), `joint_torque_rate_l2` (weight 0 until
curriculum), `body_ang_vel`, `angular_momentum` (both reweighted, not new funcs),
`self_collisions` (`mjlab`'s `self_collision_cost`) — more load-bearing here than in
standup, since the legs swing to extremes past anything walking/standing exercises,
real risk of leg-trunk contact.

### 3.6 Head

Keep `head_pose_tracking` at minimum, for obs-command parity with the rest of the
policy family (61D layout: `[twist(3), head_pose(4), body_pose(6)]` — the slot must
stay live even if lightly weighted, per AGENTS.md's "dead weights" rule). No strong
opinion yet on whether head gets an active command during the split or just tracks
HOME; default to HOME-tracking unless there's a reason to do otherwise.

## 4. Termination

- **Roll beyond a threshold** *wider* than the reward's generous std (reward is
  soft, termination is the hard backstop — same split standup uses between its
  height/tilt reward std and its termination band)
- **Pitch beyond a band** around the measured natural resting pitch (too far
  forward = face-plant, too far back = flip over)
- `nan_state` (`robot_state_is_nan`), same as standup
- No `fell_over` term — that check assumes a standing-height reference that doesn't
  apply once the robot is deliberately low to the ground (same reasoning standup
  uses to drop it)

## 5. Curriculum

**Discovery-vs-polish staging** (direct copy of standup's reasoning): settle-damping
and `joint_torque_rate_l2` start at weight 0, `action_rate_l2` ramps gradually —
introduced only after the descent skill already works. Two of standup's runs proved
that any attempt-tax active during discovery makes "do nothing" the optimum and the
skill never gets found.

**Split-depth curriculum** (new — standup didn't need this because HOME is the
robot's natural resting equilibrium, not a stretch target). A full ±90° split is a
much more extreme ask, closer in kind to `head_pose_range`/`body_pose_range`/
`ground_state_mix`/`com_range` — every difficulty/range curriculum already in this
repo ramps from an easy value to the hard one, never starts at max, because PPO
explores locally around whatever currently pays and an extreme target risks a
shallow "good enough" local optimum with no pressure to go deeper (the same
mechanism behind standup parking 0.18 rad off-HOME until its L1 weight was raised).

Proposed: `SPLIT_JOINT_OVERRIDES` interpolates from a modest depth (~±40–50°) at
step 0 to the full measured target (§2) over roughly the first 1500–2000 iterations,
using the same `pose_command_range_curriculum`-style mechanism already used for
`head_pose_range`/`body_pose_range`, but driving the reward target instead of a
command range.

**DR ramps**: CoM/head-CoM range curricula, push-magnitude curriculum — reuse
standup's shape. Velocity pushes: lean toward **off during discovery, introduced
later** (unlike standup, which keeps them on throughout) — a push mid-descent is a
different kind of perturbation than a push while holding a stand; revisit once
actually training rather than deciding blind now.

## 6. Testing (per AGENTS.md's mandatory workflow)

- **Cfg tests** (`tests/test_splits_cfg.py`, mirroring the existing `test_*_cfg.py`
  pattern): split-leg joint indices resolve correctly on the actual model; every
  `*_penalty`/`*_l1` reward has a positive weight (the "bit four envs" sign check —
  every `Episode_Reward/<penalty>` must log ≤ 0); `roll_split`/`pitch_split` present
  and reading the right projected-gravity components; no limit-proximity term
  active on `_LEG_JOINTS`.
- **Smoke test** (mandatory before any real run): 64 envs, 5 iterations — builds,
  steps NaN-free, obs is still 61D, every reward term computes, ONNX exports.
- **Settle test** (§2, done by hand in the viewer before the numbers above are
  final): hold candidate ctrl for 3s from noisy standing inits, check tilt as well
  as height — a settle test that only checks z can report a fallen state as
  "resting fine."

## 7. Open questions / explicitly out of scope for this doc

- Exact reward weights, stds, and `SPLIT_JOINT_OVERRIDES` values — all deliberately
  left as `<TBD>` above pending the in-sim settle-test measurement (§2, §6). Filling
  these in with guessed numbers would repeat the exact `STAND_Z` mistake AGENTS.md
  warns about.
- `com_downward_velocity`'s height gate (whole descent vs. only-near-target) — noted
  as undecided in §3.4, resolve empirically.
- Whether `splits_composite` (multiplicative) is needed — add only if training shows
  an additive compromise basin (§3.1).
- Recovery-from-split (split → standing) — explicitly a separate future task, not
  designed here.
- Whether the head gets an active command during the split, or just tracks HOME —
  default to HOME-tracking, revisit if there's a reason to command it.
