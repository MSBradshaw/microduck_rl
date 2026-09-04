# Microduck RL — how to train, checkpoint, and view

Working notes for this Mac's setup. This machine is Intel (x86_64), and
`torch`/`onnxruntime`/`mujoco` all dropped Intel-Mac wheels, so everything
`microduck_rl`-related runs inside **Docker** (linux/amd64 containers), not
natively.

## The moving pieces

- **Repo:** `~/microduck_rl` (clone of
  [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)),
  bind-mounted into the containers at `/w`. Two remotes: `origin`
  (pollen-robotics, upstream) and `fork`
  ([MSBradshaw/microduck_rl](https://github.com/MSBradshaw/microduck_rl),
  personal). All active work happens on `develop`; `main` on the fork is kept
  fast-forwarded to `develop` (`git push fork develop:main`) so a fresh clone
  of the fork's default branch has everything without needing to know to
  check out `develop`.
- **Containers:** `microduck` (port 8080) and `microduck2` (port 8081) —
  persistent, `python:3.12-slim` + git/osmesa/uv installed on top. Reuse them
  rather than recreating, since git/osmesa need reinstalling on a fresh
  container (apt packages aren't in a volume) and `uv sync` is slow the
  first time.
- **Shared volumes** (survive container recreation):
  - `microduck-uv-cache` → `/root/.cache/uv` (downloaded wheels)
  - `microduck-warp-cache` → `/root/.cache/warp` (JIT-compiled physics
    kernels — without this, every fresh container takes ~2-3 min to warm up)
- **Checkpoints, organized:** `~/microduck-results/<run-name>/` — see
  `~/microduck-results/run1-2026-09-01/README.md` for that run's specifics
  (best vs. final checkpoint, why, training config).
- **Checkpoints, source of truth:** Hugging Face, `mikeybrad/<run-name>`
  (e.g. `mikeybrad/microduck-run1`) — private model repo, new `.pt` files
  pushed every 60s during training.

## Starting/resuming a container

If `microduck` isn't running (this is the common case — `docker stop`,
Docker Desktop restart, or a reboot don't remove the container, they just
stop it; git/osmesa/uv already installed inside it are still there):

```bash
docker start microduck   # or microduck2
```

If it doesn't exist yet (fresh machine, or you `docker rm`'d it), use the
prebuilt image — no apt-get/pip install step needed, it's already baked in:

```bash
docker run -d --name microduck --platform linux/amd64 \
  -v ~/microduck_rl:/w \
  -v microduck-uv-cache:/root/.cache/uv \
  -v microduck-warp-cache:/root/.cache/warp \
  -w /w -p 8080:8080 \
  microduck-base:latest sleep infinity
```

(For a second simultaneous viewer, repeat with a different name/port, e.g.
`microduck2` / `8081:8080` — see "Viewing two checkpoints at once" below.)

### The `microduck-base` image

Built once from `~/microduck_rl/Dockerfile.dev` (git + osmesa + uv on top of
`python:3.12-slim`):

```bash
docker build --platform linux/amd64 -f Dockerfile.dev -t microduck-base:latest .
```

Stored in Docker's local image store — survives container removal, Docker
Desktop restarts, and reboots. Only gone if you `docker rmi microduck-base`
or fully reset/uninstall Docker Desktop. Rebuild it (same command) if you
ever need to change what's preinstalled.

## Training a new run

With a wandb API key set up (see "Setting up wandb" below), just omit
`--no-wandb` and `--agent.logger` — the RL cfg's own default (`wandb`) takes
over and `--hf-jobs` forwards the key as a job secret automatically:

```bash
docker exec microduck bash -c "
  cd /w && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m mjlab_microduck.train_cli Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 4000 \
    --hf-jobs --flavor l4x1 \
    --namespace mikeybrad --run-name <new-run-name> --timeout 12h --detach
"
```

No wandb key yet, or deliberately skipping it for a run? Add back
`--agent.logger tensorboard --no-wandb` (both flags together — `tensorboard`
alone still lets `hf_jobs.py` try to forward a key if one happens to be
findable; `--no-wandb` is what actually suppresses that).

**Sanity-check `--agent.max_iterations` before submitting** — it's real
money. Every task's `Rl...Cfg` ships its own default, but that default isn't
always right for the specific cfg: `microduck_splits_cycle`'s was copied from
`sitstand` (15,000) without checking that every one of *this* cfg's
curriculum stages actually finishes by iteration 2500 — the other 12,500
would have bought nothing. Skim the cfg's `curriculum[...]["weight_stages"]`/
`"range_stages"`/`"push_stages"` step boundaries (steps = `iteration × 24`)
and pick a budget that clears the last one with reasonable headroom, per
AGENTS.md's rule of thumb (simple tricks ≈1000 iters, gaits/curriculum-heavy
recovery 4000–6000) — don't just reuse another task's number.

Note: `--hf-jobs` runs the actual training on Hugging Face's GPUs, so
`OMP_NUM_THREADS`/`MKL_NUM_THREADS` here only cap the *local* container
process that builds and submits the job — worth keeping anyway so it doesn't
fight a concurrent viewer for CPU.

Note: `uv run train ...` (the plain console script) was **broken** in our
container — `mjlab` and `mjlab_microduck` both register a `train` entry
point, and `mjlab`'s won, silently dropping `--hf-jobs` support (tyro just
reported `--hf-jobs`/`--flavor`/etc as "Unrecognized options"). This
contradicts the repo's own `AGENTS.md`, which documents plain `uv run train
<TASK_ID> --hf-jobs` as the normal command — so this looks like an
environment-specific `uv sync` entry-point collision, not intended
behavior. Worth re-checking after a future `uv sync`/`uv lock` update in
case it's since fixed upstream; until then, invoke
`python -m mjlab_microduck.train_cli` directly to guarantee you get the
right one.

Training runs on Hugging Face's GPUs (HF Jobs), not locally — the container
just builds and submits the job. Check status:

```bash
docker exec microduck bash -c "cd /w && uv run python -c \"
from huggingface_hub import HfApi
job = HfApi().inspect_job(job_id='<job-id>')
print(job.status)
\""
```

## Setting up wandb

One-time, per container (its `~/.netrc` lives in the writable layer, not one
of the named volumes — survives `docker stop`/`start` but not a full
recreate, same as the git/osmesa install):

1. Get an API key from wandb.ai (Settings → API keys); sign up free first if
   you don't have an account.
2. `docker exec -it microduck bash -c "uv run wandb login"`, paste the key.
   (`export WANDB_API_KEY=...` in the container works too —
   `hf_jobs.py`'s `_wandb_api_key()` checks the env var first, then
   `~/.netrc`, which is what `wandb login` writes.)

Every task's `RslRlOnPolicyRunnerCfg` already sets `wandb_project =
"mjlab_microduck"`; wandb logs it under whatever account/entity the API key
belongs to — on a personal key that's your own username, e.g.
`<you>/mjlab_microduck`, not `pollen-robotics/mjlab_microduck`.

**Heads up:** `scripts/wandb_utils.py` hardcodes `WANDB_PROJECT =
"pollen-robotics/mjlab_microduck"` for `play_latest.py`'s "find my latest
run" lookup. Runs logged under your own account won't be found by that
helper until it's pointed at your entity — not yet fixed, hit it if/when
`play_latest.py` comes up empty.

## Local smoke test (always run before submitting a real HF Jobs run)

`AGENTS.md` requires a 5-iteration/64-env smoke test before any real launch.
Locally (CPU, this container) that's:

```bash
docker exec microduck bash -c "
  cd /w && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m mjlab_microduck.train_cli Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --gpu-ids None \
    --agent.logger tensorboard
"
```

Two flags that are easy to get wrong here, both discovered the hard way
(2026-09-03, see `run2-heading-2026-09-03/README.md`):

- **`--gpu-ids None` is required.** This container has no GPU at all, and
  `train`'s default (`gpu_ids=[0]`) assumes one exists — without this flag
  it crashes with `IndexError: list index out of range` in `select_gpus()`
  before training even starts.
- **Don't pass `--no-wandb` to a local smoke test.** It's only valid on the
  `--hf-jobs` submission path (consumed locally before the remote job spec
  is built) — a plain local `train_cli` invocation rejects it with
  "Unrecognized options: --no-wandb". `--agent.logger tensorboard` alone is
  enough locally; no wandb API key needed.

## Finding the best checkpoint (don't assume it's the last one)

Reward isn't guaranteed to improve monotonically. As of 2026-09-03,
`MicroduckOnPolicyRunner` (`src/mjlab_microduck/tasks/__init__.py`) tracks
this automatically: every time a checkpoint is saved (every `save_interval`
iterations, 250 by default) whose mean reward beats every checkpoint saved
before it, it's copied to a canonical `model_best.pt` right next to the
numbered ones. It matches the uploader's `model_*.pt` glob, so it syncs to
the HF model repo like any other checkpoint — no more grepping job logs and
hoping the peak iteration happened to be one that was saved (the
`splits-run1-2026-09-03` problem this fixes). Sanity-check which iteration
it actually is:

```bash
docker exec microduck bash -c "cd /w && uv run python -c \"
import torch
d = torch.load('<path>/model_best.pt', weights_only=False, map_location='cpu')
print('iter:', d['iter'])
\""
```

This is coarser than the TRUE best iteration — it can only ever point at a
checkpoint that was actually saved, same granularity every other checkpoint
already has — but it's a free upgrade over "assume it's the last one." Cross-
reference `reward_history.csv` (see below) for the exact numbers at that
iteration and neighboring ones if you want to double-check nothing better was
one save_interval away.

**Runs from before 2026-09-03 don't have this** — for those, pull the job
logs and parse `Mean reward:` lines per `Learning iteration N/M` to find the
actual peak before picking a checkpoint — see the method in
`run1-2026-09-01/README.md`.

## Per-run diagnostics: save the per-term reward log, not just checkpoints

**Fixed as of 2026-09-03** — `MicroduckOnPolicyRunner` now writes a plain
`reward_history.csv` next to the checkpoints on every `save()` (one row per
`save_interval`, every `Episode_Reward/<term>` / `Curriculum/<term>` mean
included, not just the aggregate), and `scripts/hf/uploader.py` syncs it to
the HF model repo like any other file — durable regardless of logger backend
or job retention. `*.csv` is gitignored (training-run output, never source),
so this never gets checked in; pull it down like a checkpoint:

```bash
docker exec microduck bash -c "cd /w && uv run python -c \"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='mikeybrad/<run-name>', filename='<task>/<timestamped-dir>/reward_history.csv', local_dir='.')
\""
```

**Why this exists:** `--agent.logger tensorboard` only writes tfevents into
the HF Job's *ephemeral* container filesystem, and (before this fix)
`scripts/hf/uploader.py` only watched `model_*.pt` files, so the per-term
reward breakdown was never saved anywhere durable — it only existed as
scrollback in the job's log stream. This bit us investigating run1: the
aggregate reward curve alone couldn't tell us whether a mid-training dip was
a real walking regression or just a curriculum-weight artifact — we needed
the per-term numbers, and had to reconstruct them from raw job logs after
the fact (the method below still works, and is the only option for runs
before this fix).

**While the job record still exists on HF** (retention past completion is
unknown — don't assume it's forever), you can pull the full log and parse
it:

```bash
docker exec microduck bash -c "cd /w && uv run python -c \"
from huggingface_hub import HfApi
with open('/tmp/job_full.log', 'w') as f:
    for line in HfApi().fetch_job_logs(job_id='<job-id>'):
        f.write(line + '\n')
\""
```

Lines matching `Episode_Reward/<term>: <value>` (per-step, logged every
`Learning iteration N/M`) give the full per-term breakdown; `Mean reward:`
gives the aggregate. `Curriculum/<term>` lines show the active curriculum
weight at that iteration — cross-reference against the cfg's
`weight_stages`/`range_stages` step boundaries (steps = `iteration × 24`) to
check whether a reward change lines up with a curriculum stage, same as the
`run1-2026-09-01/README.md` "Investigation" section did.

**Going forward, save this at the time of training, not after:** for every
run, once it completes, fetch and save `job_full.log` (gzip it —
uncompressed is ~10MB for a 4000-iteration run, gzipped ~500KB) into
`~/microduck-results/<run-name>/` alongside the checkpoints and configs.
Don't rely on being able to re-fetch it later from HF.

## Downloading a checkpoint locally

```bash
docker exec microduck bash -c "cd /w && uv run python -c \"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='mikeybrad/<run-name>', filename='velocity/<timestamped-dir>/model_<N>.pt', local_dir='.')
\""
```

Since `/w` is bind-mounted, downloaded files land directly in
`~/microduck_rl/` on the Mac — move them into `~/microduck-results/` to keep
things organized, and copy back into `~/microduck_rl/` (or wherever `/w`
points) when you actually want to load one with `play` (it needs to be
inside the bind mount to be visible in-container).

## Watching a checkpoint walk (the viewer)

```bash
docker exec microduck bash -c "
  cd /w && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MUJOCO_GL=osmesa uv run play Mjlab-Velocity-Flat-MicroDuck \
    --checkpoint-file /w/<checkpoint>.pt --num-envs 1 --viewer viser \
    > /tmp/play.log 2>&1 &
  echo started
"
```

**`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` matters more than it looks.** Without
it, torch/warp's CPU thread pool busy-spins across every core the container
sees (all 16), even while the sim sits paused doing nothing — measured at
~735% CPU (7+ cores) idle for a single `play` process alone in the container,
enough to make the whole Mac feel laggy. Capping it dropped that to ~207%
(~2 cores) idle, a ~3.5x reduction. Still not zero (that remainder is
warp/mujoco's own loop overhead), but noticeably smoother. Always set these
two vars before `uv run play` or `uv run train`/`train_cli`.

**Also watch for duplicate `play` processes.** Since viser's port (8080) is
hardcoded, launching `play` twice without killing the first doesn't error —
both instances keep running and competing for CPU, silently doubling the
load, while the browser only ever talks to whichever one holds the port. If
the viewer still feels laggy after capping threads, check for a stray
process before assuming there's a deeper problem:

```bash
docker exec microduck bash -c "
  for p in /proc/[0-9]*; do
    grep -qa 'bin/play' \$p/cmdline 2>/dev/null && echo \$p: \$(tr '\0' ' ' < \$p/cmdline)
  done
"
```

Kill any extras (`kill -9 <pid>`) before starting a fresh one.

Then open **http://localhost:8080** (or whatever host port that container
publishes). First run after a container restart takes ~2-3 min to
JIT-compile physics kernels even with the warp cache warm for *most*
modules — a handful of solver kernels seem to recompile per-container
regardless. Watch progress with:

```bash
docker exec microduck bash -c "tail -30 /tmp/play.log"
```

Look for `viser (listening *:8080)` in the log — that's your signal it's
ready.

**Requires `MUJOCO_GL=osmesa`** — without it, MuJoCo tries to open a real
OpenGL context and fails (`an OpenGL platform library has not been loaded`),
since there's no display/GPU in the container.

Once loaded, in the browser UI:
- Simulation starts **Paused** — click **Play**.
- To make it actually walk (vs. stand), open **Commands → Twist**, check
  **Enable**, and set `lin_vel_x` (forward speed, m/s). `lin_vel_y` is
  sideways (strafing), `ang_vel_z` is turning rate.
- The **Checkpoints** dropdown only works when the process was launched via
  `--wandb-run-path`, not `--checkpoint-file` — it won't let you swap
  checkpoints live if you loaded a local file. Restart the process instead.

### Viewing two checkpoints at once

`viser`'s port is hardcoded to 8080 inside mjlab (no CLI flag), so run a
**second container** for a second simultaneous viewer rather than trying to
change the port:

```bash
docker run -d --name microduck2 --platform linux/amd64 \
  -v ~/microduck_rl:/w \
  -v microduck-uv-cache:/root/.cache/uv \
  -v microduck-warp-cache:/root/.cache/warp \
  -w /w -p 8081:8080 \
  microduck-base:latest sleep infinity
```

Both containers share the uv/warp cache volumes, so the second one starts
faster than the first did.

## Shutting a viewer down

The process doesn't respond to a plain PID match reliably (multiple
processes can share similar cmdlines) — find it by the socket it holds:

```bash
docker exec microduck bash -c "
  for p in /proc/[0-9]*; do
    for fd in \$p/fd/*; do
      link=\$(readlink \$fd 2>/dev/null)
      [[ \"\$link\" == *'socket:'* ]] || continue
      grep -qa 'bin/play' \$p/cmdline 2>/dev/null && kill -9 \$(basename \$p)
    done
  done
"
```

Or just stop the whole container if you're done for a while:
`docker stop microduck` (keeps it around for next time — `docker rm` if you
actually want to tear it down, but then you lose the installed
git/osmesa/uv and have to redo that setup step).

## Running natively on an M-series Mac (no Docker)

Everything above assumes this Intel Mac, where torch/mujoco/onnxruntime
have no x86_64 wheels and Docker is the only way to get a linux/amd64
Python (see the top of this file). An M-series (Apple Silicon) Mac doesn't
have that problem: `uv.lock` carries `macosx_11_0_arm64` wheels for
`torch`, `mujoco`, and `warp_lang`, and the `[tool.uv.sources]` override
that routes torch to a CUDA index only fires on
`sys_platform == 'linux' and platform_machine == 'aarch64'` (the DGX
Spark/GB10 box) — a Mac never matches that marker regardless of chip. So
on an M-series Mac:

```bash
git clone https://github.com/MSBradshaw/microduck_rl.git
cd microduck_rl
git checkout develop   # main and develop are equivalent as of 2026-09-03
uv sync
```

runs natively — no container, no bind mount, no `docker exec` wrapping
every command — and gets a real GPU-backed window for the native MuJoCo
viewer, which is the reason the two scripts below exist as separate files
rather than folded into the Docker-only workflow above.

### `scripts/infer_policy_statemachine.py` — autonomous walk↔splits state machine

A standalone copy of `scripts/infer_policy.py` (native GLFW viewer,
otherwise byte-for-byte the same starting point) with an autonomous
WALK↔SPLITS state machine layered on top. `infer_policy.py` itself is
**untouched** — this is a separate file so the plain manual-control path
always still works exactly as it did before.

**Where it lives:** the `WalkSplitsStateMachine` class, defined near the
bottom of the file just above `def main()`. Everything else in the file
(the `PolicyInference` class, `TerminalInput`, the keyboard-handling loop
in `main()`) is the same machinery `infer_policy.py` already had — the
state machine just calls into it (`policy.set_vel_cmd(...)`,
`policy.trigger_behavior(...)`), it doesn't reimplement any of it.

**How it works:**

- Two states, **WALK** and **SPLITS**, each with a randomized dwell time
  (`--walk-dwell MIN MAX`, default `5.0 8.0`; `--splits-dwell MIN MAX`,
  default `3.0 5.0` — WALK's range is longer on purpose, so it spends more
  time walking on average, without hard-biasing the transition itself).
- Entering WALK samples a fresh `(lin_vel_x, lin_vel_y)`: `lin_vel_x` is
  always positive (`--vel-x-range MIN MAX`, default `0.10 0.30` m/s — zero
  or negative would just stand there or brake), `lin_vel_y` can be either
  sign (`--vel-y-range MIN MAX`, default `-0.10 0.10` m/s) so the walk
  curves left or right instead of always going straight.
- Entering SPLITS calls `PolicyInference.trigger_behavior("splits",
  duration=...)` — the same timed auto-return machinery
  `--kick-left`/`--kick-right`/`--roulade` already use in `infer_policy.py`
  — but with `trigger_behavior` extended (in this file only) to accept a
  per-call `duration` override, so every SPLITS entry gets a freshly
  randomized dwell instead of one fixed value for the whole run.
- At every dwell boundary, a fair coin flip decides stay (re-roll the same
  kind of state) vs. switch to the other — either state can self-loop, it's
  not a strict alternation.
- **Keyboard always wins.** Any velocity key (arrows, A/E, space) marks the
  state machine "manual" and it stops touching `vel_cmd` until
  `--idle-timeout` seconds (default `5.0`) pass with no further keypress —
  then it resumes with a fresh dwell timer, keeping whatever you last
  commanded rather than snapping back to a random value the instant it
  reactivates.
- **Splits does not know how to stand up.** The trained splits skill
  (`microduck_splits`, v1 — see "What's been trained so far" below) is a
  one-way descend-and-hold; it was never trained to rise. So when its dwell
  timer ends, control swaps straight back to the walking policy from the
  split pose — no scripted recovery, no position reset. That's deliberate
  for this v1: watch what actually happens (it'll likely struggle or fall
  the first few times) rather than paper over a transition nothing has
  actually learned yet.

Run it:

```bash
uv run python scripts/infer_policy_statemachine.py \
  --walking logs/rsl_rl/velocity/2026-09-03_03-59-37_velocity/2026-09-03_03-59-37_velocity.onnx \
  --splits logs/rsl_rl/microduck_splits/2026-09-03_21-29-18_microduck_splits/2026-09-03_21-29-18_microduck_splits.onnx \
  --new-cmd-obs
```

(Those two paths are this repo's own latest walk/splits runs — swap in
whatever checkpoints you're actually using; both need `scripts/export.py`
to have produced them, not a hand-converted checkpoint, per AGENTS.md's obs
normalizer rule.) `--new-cmd-obs` is required — both ONNX files take a 61D
input (13D unified command block), confirmed against each file's actual
`get_inputs()[0].shape` rather than assumed.

Keyboard controls are exactly `infer_policy.py`'s existing set (arrows/A/E
for velocity, SPACE to coast, T to pause inference, Q to quit — full list
prints on startup); the state machine only *watches* those same keypresses
to know when to yield control, it doesn't add any new bindings.

### `scripts/infer_policy_viser.py` — browser viewer for headless Docker

A second standalone copy of `infer_policy.py`, swapping the native
`mujoco.viewer.launch_passive` window for `mjviser`'s `ViserMujocoScene`
(the lower-level piece mjlab's own `play --viewer viser` renders through,
without mjlab's env/policy coupling) — streams to a browser tab on
`--port` (default 8080) instead of opening a native window. This exists
for the headless-Docker path on the Intel Mac above, where there's no
display for a native window to attach to. On an M-series Mac you generally
don't need it — the native window in `infer_policy.py` /
`infer_policy_statemachine.py` just works there directly.

## Pushing your work (personal fork)

`origin` (pollen-robotics) and `fork` (yours) are separate remotes — regular
pushes go to `fork`, not `origin`:

```bash
git push fork develop           # your active branch
git push fork develop:main      # keep the fork's default branch in sync too
```

`fork`'s `main` mirrors upstream `pollen-robotics/microduck_rl`'s `main` at
the moment the fork was created, which (2026-09-03) turned out to be a
strict ancestor of `develop` — so `develop:main` is always a plain
fast-forward, never a merge. If `origin` ever moves in a way that stops being
true, `git merge-base --is-ancestor <fork main sha> develop` will say so
before you push.

To pull this down on a different machine:

```bash
git clone https://github.com/MSBradshaw/microduck_rl.git
cd microduck_rl
git checkout develop   # main and develop are equivalent as of 2026-09-03
```

## What's been trained so far (running log)

- **2026-09-01, `Mjlab-Velocity-Flat-MicroDuck` run1** — walking policy,
  best checkpoint `model_1500_best.pt` (peak reward was mid-run, not final —
  see "Finding the best checkpoint"). Full writeup:
  `~/microduck-results/run1-2026-09-01/README.md`.
- **2026-09-03, `Mjlab-Velocity-Flat-MicroDuck` run2 (heading fix)** — fixed
  `rel_heading_envs: 0.0 → 0.3` (mjlab's own upstream default was disabled,
  so no env ever trained heading-drift correction; see
  `microduck_velocity_env_cfg.py`'s commit for the full diagnosis). Job
  `6a98f15321c5aa7c8364f643`, not yet evaluated.
- **2026-09-03, `Mjlab-Splits-Flat-MicroDuck` (v1)** — episodic front-split
  descend-and-hold. Reaches the split once and stops; no return-to-stand.
  Registered, smoke-tested, HF Jobs submission was blocked for a while on
  missing HF auth in the container (`hf auth login` fixed it).
- **2026-09-03, `Mjlab-SplitsCycle-Flat-MicroDuck` (splits v2)** — v1's
  sequel: commands the robot to keep alternating split → stand → split → ...
  for the whole episode (`AlternatingPostureCommand` in `mdp.py`, a
  deterministic-flip variant of `sitstand`'s posture command), instead of
  reaching the target once and stopping. Reuses `sitstand`'s posture-
  conditioned reward stack pointed at the measured split target, keeps v1's
  orientation shaping and tip-band terminations. Smoke-tested clean. Submitted
  as job `6a99f478259f8e97255dbd41` (`splits-cycle-run1-2026-09-03`,
  6000 iters, 4096 envs, `l4x1`) — first run trained with wandb logging
  enabled (see "Setting up wandb"), checkpoints at
  `mikeybrad/splits-cycle-run1-2026-09-03`. Not yet evaluated.
