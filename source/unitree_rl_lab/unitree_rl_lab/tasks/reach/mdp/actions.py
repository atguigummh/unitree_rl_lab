from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class RightArmFixedBodyAction(JointPositionAction):
    """策略只控制右肩和右肘，其余关节保持默认姿态。"""

    def process_actions(self, actions: torch.Tensor):
        """先限制网络原始 action，再交给 JointPositionAction 做缩放和默认偏置。

        旧版本只限制最终关节目标，导致策略可以输出 -5、-6 这类过大 action，
        再依赖软关节限位兜底。safe_v2 中先把原始 action 限制在 [-1, 1]。
        """

        safe_actions = torch.clamp(
            actions,
            min=-1.0,
            max=1.0,
        )

        super().process_actions(safe_actions)

    def apply_actions(self):
        # 整个机器人默认关节目标
        full_joint_targets = self._asset.data.default_joint_pos.clone()

        # 策略生成的右臂目标
        right_arm_targets = self.processed_actions

        # 读取右臂软关节限制
        limits = self._asset.data.soft_joint_pos_limits[
            :, self._joint_ids, :
        ]

        lower = limits[..., 0]
        upper = limits[..., 1]

        # 在软限制内部额外保留0.08 rad安全裕量
        margin = 0.08

        safe_lower = lower + margin
        safe_upper = upper - margin

        # 防止策略输出越过关节安全区域
        right_arm_targets = torch.clamp(
            right_arm_targets,
            min=safe_lower,
            max=safe_upper,
        )

        # 仅替换右肩和右肘
        full_joint_targets[:, self._joint_ids] = right_arm_targets

        # 给29个关节发送目标：
        # 右肩和右肘使用策略目标，其余关节使用默认目标
        self._asset.set_joint_position_target(full_joint_targets)


@configclass
class RightArmFixedBodyActionCfg(JointPositionActionCfg):
    class_type: type = RightArmFixedBodyAction