import csv
import os
import shutil
import statistics

import torch
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}

        # rsl_rl's Logger only ever writes metrics to its writer (tensorboard/
        # wandb) -- on an HF Jobs run with `--agent.logger tensorboard` and no
        # W&B, that writer lives in the job's ephemeral container filesystem
        # and is discarded when the job exits. scripts/hf/uploader.py only
        # watches for model_*.pt, so the whole per-iteration reward history
        # was otherwise unrecoverable once the job's log dropped out of HF's
        # job-list retention (bit us on the splits-run1-2026-09-03 run: no
        # way to tell which checkpoint was actually best after the fact).
        # Fix: snapshot the per-term extras right before Logger.log() clears
        # them, and append one row per save_interval to a plain CSV next to
        # the checkpoints, which the uploader can then push like any other
        # file -- durable regardless of logger backend or job retention.
        self._reward_history_path = (
            os.path.join(self.logger.log_dir, "reward_history.csv")
            if self.logger.log_dir else None
        )
        self._last_ep_extras: list[dict] = []

        # Best-checkpoint tracking (same "no way to tell which checkpoint was
        # actually best after the fact" problem as reward_history.csv above,
        # this time for the checkpoint file itself, not just the numbers).
        self._best_ckpt_path = (
            os.path.join(self.logger.log_dir, "model_best.pt")
            if self.logger.log_dir else None
        )
        self._best_mean_reward = float("-inf")

    def learn(self, *args, **kwargs):
        # Logger.init_logging_writer() (which sets self.logger.writer) only
        # runs inside the base learn() loop, not __init__ -- so the log()
        # hook has to be installed here, right before delegating, not in
        # __init__ (self.logger.writer doesn't exist yet there).
        orig_log = self.logger.log

        def _snapshot_then_log(*log_args, **log_kwargs):
            self._last_ep_extras = list(self.logger.ep_extras)
            return orig_log(*log_args, **log_kwargs)

        self.logger.log = _snapshot_then_log
        return super().learn(*args, **kwargs)

    def _extras_means(self) -> dict[str, float]:
        """Mirror rsl_rl Logger.log's own per-key averaging over ep_extras
        (Episode_Reward/*, Curriculum/*, ...) so the CSV carries the same
        per-term breakdown AGENTS.md's reward-design checks need, not just
        the aggregate mean reward."""
        means: dict[str, float] = {}
        for key in self._last_ep_extras[0] if self._last_ep_extras else []:
            values = torch.tensor([], device=self.device)
            for ep_info in self._last_ep_extras:
                if key not in ep_info:
                    continue
                v = ep_info[key]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor([v])
                if v.dim() == 0:
                    v = v.unsqueeze(0)
                values = torch.cat((values, v.to(self.device)))
            if values.numel() > 0:
                means[key] = values.mean().item()
        return means

    def save(self, path: str, infos=None):
        super().save(path, infos)
        self._append_reward_history()
        self._maybe_update_best(path)

    def _maybe_update_best(self, path: str) -> None:
        """Copy this checkpoint to a canonical `model_best.pt` whenever its
        mean reward beats every checkpoint saved before it.

        Checked at `save()`'s existing cadence (`save_interval`, plus the
        final save) rather than every iteration: `VelocityOnPolicyRunner.save()`
        re-exports to ONNX (and uploads it to wandb) on every call, so
        tracking at iteration granularity would multiply that cost by
        however many iterations the reward keeps improving for. This is
        coarser than the true best iteration -- it can only ever point at a
        checkpoint that was actually saved -- but that's the same
        granularity every other checkpoint already has. Turns "grep the job
        log for the peak `Mean reward:` line, then hope that exact iteration
        was saved" (the splits-run1-2026-09-03 problem) into "read
        `model_best.pt`'s own `iter` field" -- and since it matches the
        uploader's `model_*.pt` glob, it syncs to the HF model repo
        automatically, re-copied (mtime bumps) every time a new best appears.
        """
        if self._best_ckpt_path is None or not self.logger.rewbuffer:
            return
        mean_reward = statistics.mean(self.logger.rewbuffer)
        if mean_reward <= self._best_mean_reward:
            return
        self._best_mean_reward = mean_reward
        shutil.copy2(path, self._best_ckpt_path)

    def _append_reward_history(self) -> None:
        if self._reward_history_path is None:
            return
        row: dict[str, float] = {"iteration": self.current_learning_iteration}
        if self.logger.rewbuffer:
            row["mean_reward"] = statistics.mean(self.logger.rewbuffer)
        if self.logger.lenbuffer:
            row["mean_episode_length"] = statistics.mean(self.logger.lenbuffer)
        row.update(self._extras_means())

        write_header = not os.path.isfile(self._reward_history_path)
        with open(self._reward_history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .microduck_splits_env_cfg import (
    make_microduck_splits_env_cfg,
    MicroduckSplitsRlCfg,
)
from .microduck_splits_cycle_env_cfg import (
    make_microduck_splits_cycle_env_cfg,
    MicroduckSplitsCycleRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

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

# Splits-cycle — commanded, alternating split <-> stand (v2 of Splits: never
# stops oscillating, unlike v1's single episodic descend-and-hold).
register_mjlab_task(
    task_id="Mjlab-SplitsCycle-Flat-MicroDuck",
    env_cfg=make_microduck_splits_cycle_env_cfg(),
    play_env_cfg=make_microduck_splits_cycle_env_cfg(play=True),
    rl_cfg=MicroduckSplitsCycleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SplitsCycle-Rough-MicroDuck",
    env_cfg=make_microduck_splits_cycle_env_cfg(rough=True),
    play_env_cfg=make_microduck_splits_cycle_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSplitsCycleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand/StandUp → robot_allcollisions_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# allcollisions backlash robot — same model as their base task in each case.
_BL_ALLCOL = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_ALLCOL),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_ALLCOL),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
    ("Mjlab-Splits-Flat-Backlash-MicroDuck", make_microduck_splits_env_cfg, {}, MicroduckSplitsRlCfg, _BL_ALLCOL),
    ("Mjlab-SplitsCycle-Flat-Backlash-MicroDuck", make_microduck_splits_cycle_env_cfg, {}, MicroduckSplitsCycleRlCfg, _BL_ALLCOL),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
