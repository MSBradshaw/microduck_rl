# Microduck RL — how to train, checkpoint, and view

Working notes for this Mac's setup. This machine is Intel (x86_64), and
`torch`/`onnxruntime`/`mujoco` all dropped Intel-Mac wheels, so everything
`microduck_rl`-related runs inside **Docker** (linux/amd64 containers), not
natively.

## The moving pieces

- **Repo:** `~/microduck_rl` (clone of
  [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)),
  bind-mounted into the containers at `/w`.
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

```bash
docker exec microduck bash -c "
  cd /w && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run python -m mjlab_microduck.train_cli Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 4096 --agent.max_iterations 4000 \
    --agent.logger tensorboard --hf-jobs --flavor l4x1 \
    --namespace mikeybrad --no-wandb --run-name <new-run-name> --timeout 12h --detach
"
```

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

Reward isn't guaranteed to improve monotonically. Pull the job logs and
parse `Mean reward:` lines per `Learning iteration N/M` to find the actual
peak before picking a checkpoint — see the method in
`run1-2026-09-01/README.md`. Checkpoints save every `save_interval`
iterations (250, in `config/agent.yaml`).

## Per-run diagnostics: save the per-term reward log, not just checkpoints

`--agent.logger tensorboard` only writes tfevents into the HF Job's
*ephemeral* container filesystem — `scripts/hf/uploader.py` only watches and
pushes `model_*.pt` files (see its `CKPT_ROOT`/`logs/rsl_rl` glob), so **the
per-term reward breakdown (`Episode_Reward/<term>` per iteration) is never
saved anywhere durable by default.** It only exists as scrollback in the
job's log stream. This bit us investigating run1: the aggregate reward curve
alone couldn't tell us whether a mid-training dip was a real walking
regression or just a curriculum-weight artifact — we needed the per-term
numbers, and had to reconstruct them from raw job logs after the fact.

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
