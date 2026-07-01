from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_CFG as G1_CFG,
)
from unitree_rl_lab.tasks.reach import mdp


RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

RIGHT_WRIST_BODY = ["right_wrist_yaw_link"]

STABLE_UPPER_BODY_JOINTS = [
    # 腰部
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    # 左臂
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    # 右腕暂时固定
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# ============================================================
# Scene
# ============================================================

@configclass
class RightWristReachSceneCfg(InteractiveSceneCfg):
    """G1、平地、目标球和灯光。"""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            size=(20.0, 20.0),
        ),
    )

    robot: ArticulationCfg = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(
            # G1前方、右侧、胸口下方附近
            pos=(0.35, -0.25, 1.00),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.05,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.8, 0.8, 0.8),
        ),
    )

    def __post_init__(self):
        # 第一阶段固定G1骨盆，机器人不会摔倒。
        # 以后做全身平衡时再改为 False。
        self.robot.spawn.articulation_props.fix_root_link = True


# ============================================================
# Events
# ============================================================

@configclass
class EventCfg:
    """环境启动和重置事件。"""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # 1.0 × 默认关节位置
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    move_target = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="interval",
        interval_range_s=(2.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

# ============================================================
# Actions
# ============================================================

@configclass
class ActionsCfg:
    """策略仅输出右臂7个关节动作。"""

    right_arm = mdp.RightArmFixedBodyActionCfg(
    asset_name="robot",
    joint_names=RIGHT_ARM_JOINTS,
    preserve_order=True,
    scale={
        "right_shoulder_pitch_joint": 0.18,
        "right_shoulder_roll_joint": 0.15,
        "right_shoulder_yaw_joint": 0.15,
        "right_elbow_joint": 0.20,
    },
    use_default_offset=True,
)


# ============================================================
# Observations
# ============================================================

@configclass
class ObservationsCfg:
    """策略和价值网络的观测。"""

    @configclass
    class PolicyCfg(ObsGroup):

        right_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_ARM_JOINTS,
                    preserve_order=True,
                )
            },
        )

        right_arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_ARM_JOINTS,
                    preserve_order=True,
                )
            },
        )

        wrist_to_target = ObsTerm(
            func=mdp.wrist_to_target_scaled,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=RIGHT_WRIST_BODY,
                ),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        last_action = ObsTerm(
            func=mdp.clipped_last_action,
            params={"clip": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class CriticCfg(ObsGroup):

        right_arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_ARM_JOINTS,
                    preserve_order=True,
                )
            },
        )

        right_arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=RIGHT_ARM_JOINTS,
                    preserve_order=True,
                )
            },
        )

        wrist_to_target = ObsTerm(
            func=mdp.wrist_to_target_scaled,
            params={
                "robot_cfg": SceneEntityCfg(
                    "robot",
                    body_names=RIGHT_WRIST_BODY,
                ),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        last_action = ObsTerm(
            func=mdp.clipped_last_action,
            params={"clip": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ============================================================
# Rewards
# ============================================================

@configclass
class RewardsCfg:
    """奖励函数。"""
    # 腰部、左臂和右腕保持默认姿态
    stable_upper_body_pose = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-1.5,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=STABLE_UPPER_BODY_JOINTS,
        )
    },
)

# 抑制腰部、左臂和右腕晃动
    stable_upper_body_velocity = RewTerm(
    func=mdp.joint_vel_l2,
    weight=-0.05,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=STABLE_UPPER_BODY_JOINTS,
        )
    },
)
    right_arm_pose_deviation = RewTerm(
    func=mdp.joint_deviation_l1,
    weight=-0.05,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=RIGHT_ARM_JOINTS,
        )
    },
)
    # 核心奖励：让手腕接近目标
    wrist_reach = RewTerm(
        func=mdp.wrist_reach_exp,
        weight=5.0,
        params={
            "std": 0.20,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=RIGHT_WRIST_BODY,
            ),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # 距离小于5厘米时额外奖励
    wrist_success = RewTerm(
        func=mdp.wrist_reach_success,
        weight=5.0,
        params={
            "threshold": 0.05,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=RIGHT_WRIST_BODY,
            ),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # safe_v2：8厘米以内的粗到达奖励
    wrist_precision_08 = RewTerm(
        func=mdp.wrist_reach_tolerance_bonus,
        weight=1.0,
        params={
            "threshold": 0.08,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=RIGHT_WRIST_BODY,
            ),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # safe_v2：5厘米以内的精确到达奖励
    wrist_precision_05 = RewTerm(
        func=mdp.wrist_reach_tolerance_bonus,
        weight=2.0,
        params={
            "threshold": 0.05,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=RIGHT_WRIST_BODY,
            ),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # safe_v2：3厘米以内的精细收敛奖励
    wrist_precision_03 = RewTerm(
        func=mdp.wrist_reach_tolerance_bonus,
        weight=3.0,
        params={
            "threshold": 0.03,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=RIGHT_WRIST_BODY,
            ),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # safe_v2：1厘米以内的强收敛奖励
    wrist_precision_01 = RewTerm(
        func=mdp.wrist_reach_tolerance_bonus,
        weight=5.0,
        params={
            "threshold": 0.01,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=RIGHT_WRIST_BODY,
            ),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # safe_v2：全程惩罚过大的原始 action
    action_l2 = RewTerm(
        func=mdp.action_l2_penalty,
        weight=-0.01,
    )

    # safe_v2：到达目标附近后更鼓励小动作保持
    action_l2_near_target = RewTerm(
        func=mdp.action_l2_near_target,
        weight=-0.05,
        params={
            "threshold": 0.05,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=RIGHT_WRIST_BODY,
            ),
            "target_cfg": SceneEntityCfg("target"),
        },
    )

    # 动作不要剧烈跳变
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.08,
    )

    # 抑制右臂关节高速运动
    right_arm_joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.003,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=RIGHT_ARM_JOINTS,
            )
        },
    )

    # 防止撞击关节限位
    right_arm_joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=RIGHT_ARM_JOINTS,
            )
        },
    )


# ============================================================
# Terminations
# ============================================================

@configclass
class TerminationsCfg:
    """第一版仅按照时间结束回合。"""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )


# ============================================================
# Environment
# ============================================================

@configclass
class RightWristReachEnvCfg(ManagerBasedRLEnvCfg):
    """G1右手腕到达目标球任务。"""

    scene: RightWristReachSceneCfg = RightWristReachSceneCfg(
        num_envs=256,
        env_spacing=2.0,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    commands = None

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    curriculum = None

    def __post_init__(self):
        # 物理仿真频率：200 Hz
        self.sim.dt = 0.005

        # 每4个物理步执行一次策略：50 Hz
        self.decimation = 4

        # 每回合5秒
        self.episode_length_s = 20.0

        self.sim.render_interval = self.decimation

        # 摄像机位置
        self.viewer.eye = (2.5, -2.5, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.9)


@configclass
class RightWristReachPlayEnvCfg(RightWristReachEnvCfg):
    """测试时只创建一个环境。"""

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation