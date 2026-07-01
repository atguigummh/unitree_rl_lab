import gymnasium as gym


gym.register(
    id="Unitree-G1-29dof-RightWristReach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "unitree_rl_lab.tasks.reach.reach_env_cfg:"
            "RightWristReachEnvCfg",

        "play_env_cfg_entry_point":
            "unitree_rl_lab.tasks.reach.reach_env_cfg:"
            "RightWristReachPlayEnvCfg",

        "rsl_rl_cfg_entry_point":
            "unitree_rl_lab.tasks.reach.agents.rsl_rl_ppo_cfg:"
            "RightWristReachPPORunnerCfg",
    },
)
