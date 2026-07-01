from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import (
    BasePPORunnerCfg,
)


@configclass
class RightWristReachPPORunnerCfg(BasePPORunnerCfg):

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50

    experiment_name = "g1_right_wrist_reach_safe_v2_stage1"

    def __post_init__(self):
        super().__post_init__()

        # 原来通常是1.0，先降低到0.25
        self.policy.init_noise_std = 0.25
