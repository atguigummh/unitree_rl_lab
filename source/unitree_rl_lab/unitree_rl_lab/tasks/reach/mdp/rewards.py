from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import wrist_target_distance

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def wrist_reach_exp(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_wrist_yaw_link"]
    ),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """手腕距离目标越近，奖励越大。

    距离为0时奖励为1。
    """

    distance = wrist_target_distance(
        env=env,
        robot_cfg=robot_cfg,
        target_cfg=target_cfg,
    )

    return torch.exp(-torch.square(distance / std))


def wrist_reach_success(
    env: ManagerBasedRLEnv,
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_wrist_yaw_link"]
    ),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """距离小于阈值时返回1，否则返回0。"""

    distance = wrist_target_distance(
        env=env,
        robot_cfg=robot_cfg,
        target_cfg=target_cfg,
    )

    return (distance < threshold).float()

def wrist_reach_tolerance_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_wrist_yaw_link"]
    ),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """safe_v2 稠密近目标奖励。

    旧版逻辑：
        distance >= threshold 时奖励为 0，
        导致训练早期 wrist_precision_08/05/03/01 全是 0。

    新版逻辑：
        全距离都有奖励；
        distance 越小，奖励越接近 1；
        distance = threshold 时奖励为 0.5；
        distance = 2 * threshold 时奖励为 0.2。

    公式：
        reward = 1 / (1 + (distance / threshold)^2)
    """

    distance = wrist_target_distance(
        env=env,
        robot_cfg=robot_cfg,
        target_cfg=target_cfg,
    )

    return 1.0 / (
        1.0 + torch.square(distance / threshold)
    )


def action_l2_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """惩罚原始 action 幅值，避免策略依赖过大动作。"""

    return torch.sum(
        torch.square(env.action_manager.action),
        dim=1,
    )


def action_l2_near_target(
    env: ManagerBasedRLEnv,
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_wrist_yaw_link"]
    ),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """到达目标附近后，额外惩罚 action 幅值，鼓励静止保持。"""

    distance = wrist_target_distance(
        env=env,
        robot_cfg=robot_cfg,
        target_cfg=target_cfg,
    )

    near_target = (distance < threshold).float()

    action_l2 = torch.sum(
        torch.square(env.action_manager.action),
        dim=1,
    )

    return near_target * action_l2

