# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from rsl_rl.modules import MLP

@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 12
    max_iterations = 1000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    actor = {
    "class_name": "MLPModel",
    "hidden_dims": [512, 256, 128],
    "activation": "elu",
    "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
    },
}

    critic = {
    "class_name": "MLPModel",
    "hidden_dims": [512, 256, 128],
    "activation": "elu",
}
    obs_groups = {
    "actor": ["policy"],
    "critic": ["policy"],
}
    algorithm = dict(
        actor=actor,
        critic=critic,
        lr=1e-3,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=3,
        num_mini_batches=2,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
