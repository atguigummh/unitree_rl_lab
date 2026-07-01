from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def wrist_to_target(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_wrist_yaw_link"]
    ),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """返回机器人根坐标系下，右手腕指向目标球的三维向量。

    输出形状：
        [num_envs, 3]
    """

    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 右手腕在世界坐标系中的位置
    wrist_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :]

    # 目标球在世界坐标系中的位置
    target_pos_w = target.data.root_pos_w

    # 世界坐标系下：手腕指向目标的向量
    error_w = target_pos_w - wrist_pos_w

    # 转换到机器人根坐标系，避免未来机器人朝向变化影响观测
    error_b = quat_apply_inverse(robot.data.root_quat_w, error_w)

    return error_b



def wrist_to_target_scaled(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_wrist_yaw_link"]
    ),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    scale: float = 20.0,
) -> torch.Tensor:
    """返回缩放后的 wrist_to_target，仅用于策略观测。

    注意：
        wrist_to_target() 保持真实米制距离，用于奖励和距离统计。
        这个函数只给 actor/critic 观测使用。

    默认 scale=20：
        1 cm -> 0.2
        5 cm -> 1.0
    """

    return wrist_to_target(
        env=env,
        robot_cfg=robot_cfg,
        target_cfg=target_cfg,
    ) * scale


def clipped_last_action(
    env: ManagerBasedRLEnv,
    clip: float = 1.0,
) -> torch.Tensor:
    """返回裁剪后的上一动作观测，避免 previous_action 正反馈放大。"""

    return torch.clamp(
        env.action_manager.action,
        min=-clip,
        max=clip,
    )


def wrist_target_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["right_wrist_yaw_link"]
    ),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """返回右手腕到目标球的欧氏距离，形状为 [num_envs]。"""

    error = wrist_to_target(
        env=env,
        robot_cfg=robot_cfg,
        target_cfg=target_cfg,
    )

    return torch.linalg.norm(error, dim=-1)