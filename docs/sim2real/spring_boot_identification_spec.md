# Spec — spring-boot identification on the RobStride gripper bench

Self-contained. You do not need context from the sprung-foot campaign to
implement this.

## Why

The MicroDuck's sprung foot ("boot") is a Sarrus mechanism with a compression
spring. Simulation predicts a 27 mm hop with it and **zero** without, but that
prediction rests on a **damping ratio that has never been measured** — `zeta`
was chosen to suppress a simulation artifact, not observed. It is worth roughly
a **factor of 2** on predicted hop height (`zeta` 0.05 -> 0.25 spans 10.7 ->
5.2 mm in a design sweep), and it must be re-measured for **every spring and
every mechanism revision**. Hence a bench rather than a one-off.

Output feeds `src/mjlab_microduck/robot/sprung_foot.py`, which accepts an
absolute `damping=` override in N.s/m precisely so a measurement can replace
the estimate.

## What is being measured

**Energy retention per compression-release cycle**, and how it varies with rate.

```
retention = 1 - (loop area) / (area under the loading curve)
zeta  from  retention = exp(-2*pi*zeta / sqrt(1 - zeta^2))
```

**The primary number is a RATIO, so the force axis does not need calibrating.**
An uncalibrated motor-current-as-force axis gives the correct `zeta`. A load
cell (on order) improves absolute stiffness, not this.

Two loss mechanisms must be separated, because they have different fixes:

| mechanism | signature | fix |
|---|---|---|
| **Coulomb friction** (sliding in the Sarrus) | rate-INdependent; present at 0.1 Hz | bearings, surface finish — mechanical |
| **Viscous damping** | grows with rate | intrinsic; this is what `zeta` models |

A hand measurement sees only the first. Separating them is the whole reason to
automate.

## Rig

RobStride QDD motor (~14 N.m) driving a rack-and-pinion **parallel gripper**.
Backdrivable, movable by hand. Parallel jaws give pure linear compression with
no side load — good kinematics for this.

Boot under test: stiffness ~3900 N/m, travel 12 mm, ~0.74 mm preload,
70 g, 30 mm tall.

## Safety limits — the gripper is MASSIVELY over-capable

14 N.m through a rack and pinion is hundreds to thousands of newtons at the
jaw, against **49.7 N** to fully compress the boot. **The risk is crushing the
boot, not stalling the motor.** Enforce in software before the jaw ever touches
one:

- **Force limit 60 N** (~20% over full-travel force). Abort above it.
- **Travel limit 10 mm** of boot compression. Full travel is 12 mm and the
  mechanical hard stop is just past it; do not drive into it.
- Home against a **light** touch (< 5 N) to find zero contact, then work in
  compression from there.

## Protocol

Log at **>= 500 Hz**: timestamp, jaw position [m], motor current [A]
(and load-cell force [N] once available).

### Every measurement is bidirectional and baseline-subtracted

**Bidirectional**: always sweep down AND back up through the same positions.
Friction opposes motion, so it biases the two directions oppositely: the
**mean** of the two force curves is the elastic force and **half the difference**
is the friction. Without this the rig's own friction swamps the boot's.

**Baseline**: run every sweep **with no boot fitted** first, identically. That
records the rig's own friction and inertia; subtract it from the with-boot run.
Jaw inertia otherwise masquerades as damping — both produce a loop in
quadrature with displacement.

### A — quasi-static loop (Coulomb)

Compress 0 -> 10 mm and back at **~2 mm/s**. Gives the stiffness curve and the
rate-independent loss.

### B — frequency sweep (viscous)

Bias to **5 mm** compression, then oscillate **+-3 mm** at
**0.1, 0.5, 1, 2, 5, 10, 15 Hz**. Report loop area at each.

15 Hz is the target because the robot's body-on-springs stance mode is
**15.0 Hz** at k=3900 — that is the operating regime. If the rig cannot reach
it, the trend still extrapolates: **inertia scales as omega^2, viscous as
omega**, so a proper sweep separates them even with an imperfect baseline.

Loss vs frequency: the **intercept is Coulomb**, the **slope is viscous**.

### Do NOT attempt a free-release test

Compressing the boot and releasing it against a mass does not work on this rig.
Friction cancels in a bidirectional sweep but is **uncompensated** in a free
release — the stored energy goes into overcoming rack friction and cogging
before moving any mass, so you measure the rig. (Confirmed by hand: ~400 g will
not back-drive the jaw.) Protocol B gives the same information.

## Expected result

If the current model is right, **energy retention ~0.47**, i.e. `zeta_eff`
~0.12. Higher retention means hop height rises roughly proportionally.

**Mass caveat that has bitten this project once already**: the model sets
`zeta = 0.3` against the **70 g boot**, which is only **`zeta_eff` = 0.12**
against the **877 g robot**. When quoting a damping ratio, always state which
mass it refers to, and convert with `c = 2*zeta*sqrt(k*m)`. The number the
simulation wants is the **absolute damping `c` in N.s/m** — that is unambiguous.

## Deliverable

1. Loop area vs frequency, with the empty-rig baseline subtracted.
2. Coulomb and viscous components separated (intercept and slope).
3. **Absolute damping `c` [N.s/m]**, which goes straight into
   `sprung_foot.py`'s `damping=` override.
4. The measured stiffness curve, as a check against 3900 N/m (the existing
   hand measurement is 3 mm ~ 1500 g, 8 mm ~ 3500 g).
5. Raw logs retained — this will be repeated per spring and per revision.

---

## Amendment, 2026-09-03 — after the first run

Results: `rebot-lerobot/bench/RESULTS.md`.

**The baseline-subtraction instruction above is WRONG for a torque-derived force
axis, and this spec caused a wasted measurement.** Rack friction scales with
transmitted load; the empty-rig baseline carries none (0.108 N against 6.29 N
under load). **A zero-load baseline cannot subtract a load-dependent term.** The
6.29 N "Coulomb" figure was therefore the rack's friction, not the boot's —
falsified by a physical check the bench operator did and this spec did not
suggest: if that force were in the boot's load path there would be a
`6.29/3344` = 1.88 mm stiction dead band, and the boot returns fully to its free
length.

**Consequence: force must be measured DOWNSTREAM of the transmission.** Load
cells on the jaw remove the error rather than correcting for it. Baseline
subtraction remains correct for *inertia* (which is load-independent), and
bidirectional sweeping remains correct for the motor's own friction — but not
for a load-dependent term in series with the specimen.

**Bandwidth was also over-specified.** The rig rolls off with a ~6 Hz corner,
amplitude-limited by ACCELERATION (+-3 mm at 15 Hz needs 26.6 m/s^2), not motor
speed — peak velocity never exceeded 48 of ~155 available rpm. The 15 Hz target
was unreachable, and points above 2 Hz are unusable because the empty and
loaded runs reached different amplitudes. Either accept a 0.1-2 Hz sweep and
extrapolate, or raise the acceleration authority — which is only safe once a
load cell bounds boot force directly.

**What the first run did establish:** `k` = 3344 N/m (not the 3900 hand
figure), loss rate-INdependent to within +5.5% over 20x in frequency, and
`c <= 12.5 N.s/m` as an upper bound. The model's assumed `c` = 9.18 N.s/m sits
inside that bound, so it is not contradicted.
