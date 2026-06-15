# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

import argparse
import os
import time
import torch

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# -------------------
# CLI arguments
# -------------------
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use a manually specified checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# -------------------
# Launch Isaac Sim first
# -------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------
# 延迟导入依赖模块
# -------------------
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# -------------------
# Main
# -------------------
def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    # 选择 checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = "/home/ma/unitree_clean/unitree_rl_lab/logs/rsl_rl/unitree_g1_29dof_velocity/2026-06-12_11-31-36/model_600.pt"
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 视频录制
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL 包装
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)

    # 获取推理用 policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.actor
    critic_nn = runner.alg.critic if hasattr(runner.alg, "critic") else None

    # 获取归一化器
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # -------------------
    # 导出 TorchScript JIT
    # -------------------
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    # Actor 输入为 obs dict
    example_obs = {"policy": torch.zeros(1, 480, device=env.unwrapped.device)}

    traced_actor = torch.jit.trace(policy_nn, example_obs, strict=False)
    torch.jit.save(traced_actor, os.path.join(export_model_dir, "policy.pt"))
    print(f"[INFO] Exported JIT policy to: {os.path.join(export_model_dir, 'policy.pt')}")

    # -------------------
    # 推理循环
    # -------------------
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()