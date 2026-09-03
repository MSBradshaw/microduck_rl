"""Headless eval: how deep does the trained splits checkpoint actually settle?

Loads a checkpoint, runs one full episode (deterministic policy) across many
envs, and reports achieved hip_pitch (both legs) vs. the full measured
target, trunk height, tilt, and how many envs terminated early (fell/NaN).
"""
import sys
from dataclasses import asdict

import numpy as np
import torch

import mjlab.tasks  # noqa: F401 -- populates registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

from mjlab_microduck.tasks.microduck_splits_env_cfg import SPLIT_JOINT_OVERRIDES, SPLIT_Z

TASK_ID = "Mjlab-Splits-Flat-MicroDuck"
CKPT = sys.argv[1]
NUM_ENVS = 64

configure_torch_backends()
device = "cpu"

env_cfg = load_env_cfg(TASK_ID, play=True)
agent_cfg = load_rl_cfg(TASK_ID)
env_cfg.scene.num_envs = NUM_ENVS

env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
runner.load(CKPT, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.scene["robot"]
hip_pitch_ids = list(SPLIT_JOINT_OVERRIDES.keys())
full_targets = np.array([SPLIT_JOINT_OVERRIDES[i] for i in hip_pitch_ids])

obs, _ = wrapped.reset()
ep_len_steps = int(env_cfg.episode_length_s / env.step_dt)
print(f"episode length: {ep_len_steps} steps ({env_cfg.episode_length_s}s @ {1/env.step_dt:.0f}Hz)")
print(f"target hip_pitch (joint idx {hip_pitch_ids}): {full_targets}")
print(f"target trunk z: {SPLIT_Z}")

alive_mask = torch.ones(NUM_ENVS, dtype=torch.bool)
first_done_step = torch.full((NUM_ENVS,), -1, dtype=torch.long)

with torch.no_grad():
    for step in range(ep_len_steps):
        actions = policy(obs)
        obs, _, dones, _ = wrapped.step(actions)
        newly_done = dones.bool() & alive_mask
        first_done_step[newly_done] = step
        alive_mask &= ~dones.bool()

hip_pitch = robot.data.joint_pos[:, hip_pitch_ids].cpu().numpy()
z = (robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]).cpu().numpy()
proj_g = robot.data.projected_gravity_b.cpu().numpy()  # [:,2] ~ -1 upright, roll=[:,1], pitch=[:,0]
tilt_deg = np.degrees(np.arccos(np.clip(-proj_g[:, 2], -1, 1)))

terminated_early = (first_done_step >= 0).numpy()
print(f"\n{terminated_early.sum()}/{NUM_ENVS} envs terminated before episode end "
      f"(mean step {first_done_step[first_done_step>=0].float().mean().item():.0f})" if terminated_early.any() else "\nno early terminations")

survivors = ~terminated_early
print(f"\n--- final state, {survivors.sum()}/{NUM_ENVS} envs that reached episode end ---")
print(f"hip_pitch achieved (mean over survivors): {hip_pitch[survivors].mean(axis=0)}")
print(f"hip_pitch achieved (std):                 {hip_pitch[survivors].std(axis=0)}")
print(f"fraction of full split depth achieved:     {(hip_pitch[survivors].mean(axis=0) / full_targets)}")
print(f"trunk z achieved: mean={z[survivors].mean():.4f} std={z[survivors].std():.4f} (target {SPLIT_Z})")
print(f"tilt (deg from vertical): mean={tilt_deg[survivors].mean():.2f} std={tilt_deg[survivors].std():.2f}")
