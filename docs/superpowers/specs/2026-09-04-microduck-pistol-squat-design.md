# Microduck pistol squat (single-leg squat, asymmetric, right leg free)

## 1. Objective

Commanded posture, same shape as `sitstand`/`splits_cycle`: one policy, both
directions, driven by a posture flag (twist slot 0 = STAND, 1 = SQUAT). Not
an episodic one-way trick like splits v1 — it must explicitly return to
normal two-legged standing, which is exactly what the sitstand-style
flag-commanded pattern already does well.

**Fixed leg assignment, not randomized**: left leg = stance (weight-bearing,
squats), right leg = free (lifted, must clear the ground). `ENABLE_SYMMETRY
= False` follows directly (same reasoning as splits: the mirror loss would
train the policy to fight its own deliberately lopsided target).

**The free leg has no fixed target pose** — only "clear of the ground."
(Originally scoped as "right leg fully extended forward"; relaxed during
this session's design pass once the geometry made clear a rigid forward
extension raises self-collision/reach issues without adding anything the
task actually needs.) The exact shape of the free leg is left for the
policy to discover, same as splits leaves the descent path undetermined —
encode *what counts as the maneuver* (stance leg reaches depth, free foot
never touches ground, ends standing) in gates, not a joint trajectory.

## 2. Target pose — measured, not guessed

Per AGENTS.md's standing lesson (a 5mm-wrong `STAND_Z` once made a goal
physically unreachable for days) and splits' own precedent, nothing below
was guessed. Method: `scripts/measure_pistol_pose.py` (new, this session —
same kinematic technique as `measure_split_pose.py`: gravity off, base
pinned upright, robot lowered until its lowest point rests on the floor),
extended with one check splits never needed: whether the whole-robot CoM
(`data.subtree_com[0]`, MuJoCo's own computation) lands within the stance
foot's actual mesh footprint (real vertex bounds transformed to world
frame, not a bounding-sphere approximation).

**Joint ranges**: hip_pitch/knee/ankle are all ≈±90° on every leg — no
kinematic blocker for either a deep squat or a lifted free leg.

**Stance leg depth — reused from a proven pose, not invented.** An early
attempt at inventing hip/knee/ankle angles from scratch barely lowered the
trunk at all (the chosen formula canceled itself out geometrically). The
`sitstand` task's SIT keyframe is a real, stability-swept (2026-07-27,
tilt-verified from noisy resets, documented in
`microduck_sitstand_env_cfg.py`) deep-bend recipe: hip_pitch barely moves
(HOME −0.4579 → −0.4079), ankle goes to 0 (from HOME's 0.4530), and knee
does almost all the work (HOME −0.0049 → 1.35). Reusing that recipe for the
**left leg only** actually lowers the trunk correctly. `STANCE_LEFT_TARGET`
= HOME blended toward SIT's left-leg values by a depth fraction (α).

**Self-collision caps the achievable depth.** Sweeping α with several
candidate free-leg configs: self-collision-free through **α ≈ 0.85**;
α = 1.0 (SIT's full knee bend) self-collides against a lifted right leg
regardless of the right-leg config tried. **Curriculum ceiling: α ≈ 0.85,
not 1.0** — "as low as possible" within what doesn't self-collide, not
literally the deepest the stance leg alone could go.

**Free leg — a workable anchor pose exists, but isn't the reward target.**
Best candidate found (right_hip_pitch ≈ 0.8, right_knee ≈ −0.4,
right_ankle ≈ 0.2 — "R3" in the measurement sweep): self-collision-free
through α = 0.85, right foot clears the floor with margin. Used only as the
reset/init anchor (§6); the actual reward (§3.2) doesn't pin the policy to
these exact numbers.

**CoM requires a real hip shift — expected, not a red flag.** With the base
kinematically pinned at (x=0, y=0) (necessary for the measurement
technique), the CoM sits ~22–26mm outside the stance foot's footprint
laterally, plus a growing 0–30mm fore/aft offset as α increases. This isn't
a flaw in the pinned test — it's the correct physics: a stance foot sits
offset from the trunk's centerline by design (normal stance width), so
balancing on it requires the hips to actually shift over the foot, exactly
like a human pistol squat. Small, plausible magnitudes for a 25cm robot
(nowhere near the leg's kinematic reach limit) — but it means **the reward
design must not penalize trunk xy displacement** the way a symmetric
task's regularizers might assume is always undesirable (§3.4).

`PISTOL_Z` (trunk height at α = 0.85 with the R3 free-leg config):
≈ 0.10–0.11 — to be pinned to an exact measured value once the free-leg
config is finalized during implementation (this session measured a range
across nearby candidates, not yet a single locked-in number).

### 2.1 Correction (post-implementation review)

The initially-locked-in values (`PISTOL_Z = 0.0995`, free-leg anchor
`right_hip_pitch=0.8, right_knee=-0.4, right_ankle=0.2`, "R3" above) were
**wrong**: they were measured from a pose where the FREE (right) foot ended
up touching the floor, not the STANCE (left) foot — the opposite of what
the whole measurement is supposed to characterize. This was an oversight in
reading `scripts/measure_pistol_pose.py`'s own `floor_contacts` output by
eye (the script reports which geom actually touches the floor; that field
was never checked/asserted against, only the trunk height and self-collision
columns were eyeballed).

Re-measured with the same script/technique, checking `floor_contacts`
explicitly this time:

```python
PISTOL_Z = 0.0766  # was 0.0995
PISTOL_FREE_LEG_ANCHOR = {
    11: 1.3,   # right_hip_pitch  (was 0.8)
    12: -0.9,  # right_knee       (was -0.4)
    13: 0.3,   # right_ankle      (was 0.2)
}
```

Verified: at α = 0.85 with this corrected free-leg config,
`floor_contacts == {'left_foot_collision'}` — i.e. the STANCE foot is the
one on the ground, as intended. `scripts/measure_pistol_pose.py`'s
`__main__` block now asserts this explicitly (hard crash instead of a
silent bad measurement) so this class of mistake can't recur unnoticed.
`PISTOL_STANCE_OVERRIDES` (the left-leg depth dict) was never wrong and is
unchanged by this correction.

## 3. Reward design

### 3.1 Stance leg pose target — reuse unchanged

`posture_pose_match`/`posture_pose_l1` (mdp.py) already take a
`joint_indices` param — scope to `[0,1,2,3,4]` (left leg only, per the
existing joint-index convention) with `sit_overrides` = the left-leg-only
target from §2. Same "sit is just posture B" genericity `splits_cycle`
already proved out (its own docstring: `SitStandCommand`'s machinery is
"fully generic... not hardcoded to sit semantics") — reused a second time,
this time scoped to one leg via a parameter that already exists. No core
mdp.py change needed for this piece.

### 3.2 Free leg clearance — NEW reward function

Splits/sitstand have nothing like this (both legs always have a rigid pose
target). Needs one new function, e.g. `pistol_free_leg_clearance`, grouped
under a new "Pistol squat" section in mdp.py per AGENTS.md's convention:

- Hard gate: right foot NOT in contact with the floor (contact-sensor
  check, same category of hard state-based gate AGENTS.md prescribes for
  "what counts as the maneuver" — not a soft penalty nudge).
- Soft shaping: right-foot height above a small margin, Gaussian or
  potential-based, gated on posture blend > 0 (so it's inert during normal
  standing) — gives a gradient toward "lift the foot" without dictating
  *how* the leg gets there.
- Deliberately NO joint-angle pose-match term for the right leg — that's
  the whole point of relaxing "extended forward" to "clear of the ground."

### 3.3 Height — reuse unchanged

`posture_height_gaussian`/`posture_height_l1` with the new `PISTOL_Z`.

### 3.4 CoM / trunk displacement — explicit audit, not a new function

Before training: confirm nothing pulled in from sitstand/splits_cycle's
regularizer set penalizes trunk xy displacement. Every other commanded-
posture task trained in this repo is roughly bilaterally symmetric and
never needed the trunk to translate sideways to succeed — this is the one
task where it does, and it's the single most important thing to get wrong
silently (a policy fighting a necessary weight shift would just never
balance, and the failure would look like "can't learn it" rather than
"reward is fighting the physics").

### 3.5 Motion quality / sim2real regularizers — reuse splits_cycle's set

Descent/rise speed caps, |a_z| shock penalty, DR/obs-noise/delay stack —
proven recipe for gentle posture transitions, same starting point
splits_cycle used from sitstand.

### 3.6 Head

Commandable `head_pose`, zero-padded parity — same as every task in the
family (61D obs invariant).

## 4. Command structure

Reuse `AlternatingPostureCommand`/`AlternatingPostureCommandCfg`
(`splits_cycle`'s class, itself a `SitStandCommand` subclass) rather than
plain `SitStandCommand`'s independent draws. Same training-data-efficiency
reasoning from `splits_cycle`'s own docstring applies here, arguably more
so: independent draws under-expose rare transitions
(`rel_turn_in_place_envs`'s lesson generalizes), and single-leg balance is
a harder skill that needs MORE transition exposure, not less. Deployment
behavior is identical either way (both classes present the same raw-flag
obs contract at inference — a runtime toggling the flag once behaves the
same regardless of which class trained it), so this is purely a
training-efficiency choice.

## 5. Termination

Reuse splits' soft tip-band safety-backstop terminations — NOT sitstand's
"no fall termination" choice. Single-leg stance carries meaningfully higher
real tip-over risk than either sit or a symmetric front split; letting a
genuinely bad single-leg rollout play out repeatedly would poison a lot of
training data the same way splits' own design doc warned about for its own
(lower-risk) case.

## 6. Curriculum

Depth curriculum, starting shallow (per this session's decision, given this
is a harder balance problem than anything trained so far): α stages e.g.
0.3 → 0.55 → 0.85 (capped at the self-collision ceiling from §2, not 1.0).
Reset states: some fraction starting from the free-leg anchor pose (§2) to
give the policy on-policy data near the target from early on, mirroring
splits/sitstand's reset-state-mix pattern.

**Open question**: `splits_cycle` explicitly *dropped* v1's separate depth
curriculum because the alternating command's repeated 0→1 blend every
dwell period already provides graduated exposure across the whole episode.
Whether that same argument applies here (skip a separate α curriculum,
rely on the command's own ramp) or whether single-leg balance is hard
enough to need both, is a real open question — propose starting WITH an
explicit α curriculum (this is a harder task than splits_cycle) and
revisiting if training telemetry shows it wasn't needed.

## 7. Open questions / explicitly out of scope for this doc

- Exact free-leg joint config and final `PISTOL_Z` — pending a locked-in
  choice during implementation (§2 gives a validated range, not yet one
  final number).
- §6's curriculum-vs-alternation-alone question.
- Free-leg clearance reward's margin/std — tune once training starts, not
  guessed here.
- §3.4's CoM-displacement audit must happen as part of implementation, not
  be silently skipped.
- **True dynamic single-leg balance stability cannot be verified before
  training** — same lesson splits already documented: raw physics with no
  active controller just sags (this robot's servos are low-stiffness,
  `kp=0.55`, and only hold a pose through a trained policy's continuous
  correction). The kinematic measurement in §2 is the right pre-training
  check (geometry, self-collision, required CoM shift magnitude) — real
  balance validation comes from training telemetry and eventual video
  review, per AGENTS.md's "measure before theorizing."

## 8. Testing (per AGENTS.md's mandatory workflow)

- **Cfg tests** (`tests/test_pistol_cfg.py`, mirroring `test_splits_cfg.py`):
  joint indices resolve on the actual model; every `*_penalty`/`*_l1`
  reward has a positive weight (the "bit four envs" sign check — every
  `Episode_Reward/<penalty>` must log ≤ 0); free-leg clearance gate present
  and reads the right contact sensor; no accidental trunk-xy penalty term
  active (§3.4, as an explicit assertion, not just a manual check).
- **Smoke test** (mandatory before any real run): 64 envs, 5 iterations —
  builds, steps NaN-free, obs still 61D, every reward term computes, ONNX
  exports.
- **No dynamic settle test** — per splits' own documented lesson, not
  viable pre-training without an active controller. §2's kinematic
  measurement is the substitute.
