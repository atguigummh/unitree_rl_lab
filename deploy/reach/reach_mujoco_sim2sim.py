from __future__ import annotations

import argparse
import time
from pathlib import Path
from contextlib import nullcontext
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


# ============================================================
# 必须与 Isaac Lab Reach 任务完全一致
# ============================================================

POLICY_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

# 按 reach_env_cfg.py 中的值修改
ACTION_SCALE = np.array(
    [0.18, 0.15, 0.15, 0.20],
    dtype=np.float32,
)

# Isaac Lab UNITREE_G1_29DOF_CFG 默认右臂姿态
POLICY_DEFAULT_POS = np.array(
    [0.30, -0.25, 0.00, 0.97],
    dtype=np.float32,
)

# ============================================================
# 随机目标参数
# 当前根节点固定，因此这里直接使用世界坐标
# ============================================================

# 相邻两个目标至少相距6 cm
MIN_TARGET_SHIFT = 0.06
# 目标出现至少0.5秒后才能成功
MIN_TARGET_ACTIVE_TIME = 0.5

TARGET_CENTER_W = np.array(
    [0.35, -0.25, 1.00],
    dtype=np.float64,
)

# 第一轮建议先使用0.05，即±2 cm
TARGET_HALF_RANGE = np.array(
    [0.08, 0.08, 0.08],
    dtype=np.float64,
)

# 手腕距离球心小于2 cm，认为进入成功范围
SUCCESS_DISTANCE = 0.02 

# 连续满足10个策略周期才算成功
# 策略频率为50 Hz，因此约为0.2秒
SUCCESS_HOLD_POLICY_STEPS = 10

# 单个目标最多等待4秒
TARGET_TIMEOUT_SECONDS = 4.0

# Isaac Lab G1-29dof 默认站姿
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.10,
    "left_hip_roll_joint": 0.00,
    "left_hip_yaw_joint": 0.00,
    "left_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.20,
    "left_ankle_roll_joint": 0.00,

    "right_hip_pitch_joint": -0.10,
    "right_hip_roll_joint": 0.00,
    "right_hip_yaw_joint": 0.00,
    "right_knee_joint": 0.30,
    "right_ankle_pitch_joint": -0.20,
    "right_ankle_roll_joint": 0.00,

    "waist_yaw_joint": 0.00,
    "waist_roll_joint": 0.00,
    "waist_pitch_joint": 0.00,

    "left_shoulder_pitch_joint": 0.30,
    "left_shoulder_roll_joint": 0.25,
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "left_wrist_pitch_joint": 0.00,
    "left_wrist_yaw_joint": 0.00,

    "right_shoulder_pitch_joint": 0.30,
    "right_shoulder_roll_joint": -0.25,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.97,
    "right_wrist_roll_joint": -0.15,
    "right_wrist_pitch_joint": 0.00,
    "right_wrist_yaw_joint": 0.00,
}

# 尽量匹配 Isaac Lab 的隐式 PD 参数
PD_GAINS = {
    "hip": (100.0, 2.0),
    "knee": (150.0, 4.0),
    "ankle": (40.0, 2.0),
    "waist_yaw": (200.0, 5.0),
    "waist_other": (40.0, 5.0),
    "arm": (40.0, 1.0),
    "wrist": (40.0, 1.0),
}


def name_to_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"MuJoCo 中找不到：{name}")
    return object_id


def get_joint_id(model: mujoco.MjModel, name: str) -> int:
    return name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def get_body_id(model: mujoco.MjModel, name: str) -> int:
    return name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def create_joint_actuator_map(
    model: mujoco.MjModel,
) -> dict[int, int]:
    """建立 joint_id -> actuator_id 映射。"""
    mapping: dict[int, int] = {}

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id >= 0:
            mapping[joint_id] = actuator_id

    return mapping


def get_pd_gain(joint_name: str) -> tuple[float, float]:
    if "hip" in joint_name:
        return PD_GAINS["hip"]
    if "knee" in joint_name:
        return PD_GAINS["knee"]
    if "ankle" in joint_name:
        return PD_GAINS["ankle"]
    if joint_name == "waist_yaw_joint":
        return PD_GAINS["waist_yaw"]
    if joint_name in ("waist_roll_joint", "waist_pitch_joint"):
        return PD_GAINS["waist_other"]
    if "wrist" in joint_name:
        return PD_GAINS["wrist"]
    return PD_GAINS["arm"]


def compute_safe_joint_limits(
    model: mujoco.MjModel,
    joint_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """近似复制 Isaac Lab soft limit factor=0.9，再保留0.08 rad。"""
    lower = []
    upper = []

    for joint_id in joint_ids:
        hard_lower, hard_upper = model.jnt_range[joint_id]

        center = 0.5 * (hard_lower + hard_upper)
        half_range = 0.5 * (hard_upper - hard_lower)

        soft_lower = center - 0.9 * half_range
        soft_upper = center + 0.9 * half_range

        lower.append(soft_lower + 0.08)
        upper.append(soft_upper - 0.08)

    return (
        np.asarray(lower, dtype=np.float32),
        np.asarray(upper, dtype=np.float32),
    )

def sample_target_position(
    rng: np.random.Generator,
    previous_target: np.ndarray | None = None,
) -> np.ndarray:
    """随机生成目标，并避免连续两个目标距离太近。"""

    lower = TARGET_CENTER_W - TARGET_HALF_RANGE
    upper = TARGET_CENTER_W + TARGET_HALF_RANGE

    candidate = TARGET_CENTER_W.copy()

    for _ in range(100):
        candidate = rng.uniform(
            low=lower,
            high=upper,
        ).astype(np.float64)

        if previous_target is None:
            return candidate

        shift_distance = float(
            np.linalg.norm(candidate - previous_target)
        )

        if shift_distance >= MIN_TARGET_SHIFT:
            return candidate

    print(
        "[WARN] 100次采样后仍未满足最小目标间距，"
        "使用最后一次采样结果。"
    )

    return candidate


def set_mocap_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_id: int,
    target_position: np.ndarray,
) -> None:
    """设置红球的世界坐标位置。"""

    data.mocap_pos[mocap_id] = target_position

    # 单位四元数，不旋转
    data.mocap_quat[mocap_id] = np.array(
        [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    # 立即更新body位置
    mujoco.mj_forward(model, data)
def reset_robot_to_default(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """将机器人关节恢复到训练使用的默认姿态。"""

    # 先恢复MuJoCo初始状态
    mujoco.mj_resetData(model, data)

    # 设置默认关节角
    for joint_name, default_pos in DEFAULT_JOINT_POS.items():
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr] = default_pos

    # 清零速度和控制量
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    mujoco.mj_forward(model, data)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument(
    "--num-targets",
    type=int,
    default=100,
    help="连续测试的随机目标数量。",
)

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子。",
    )

    parser.add_argument(
        "--success-distance",
        type=float,
        default=SUCCESS_DISTANCE,
        help="成功距离阈值，单位m。",
    )

    parser.add_argument(
        "--target-timeout",
        type=float,
        default=TARGET_TIMEOUT_SECONDS,
        help="每个目标最大允许时间，单位s。",
)
    parser.add_argument(
    "--headless",
    action="store_true",
    help="不打开MuJoCo图形窗口，用于批量测试。",
)

    parser.add_argument(
        "--wrist-body",
        default="right_wrist_yaw_link",
    )
    parser.add_argument(
        "--root-body",
        default="pelvis",
    )
    parser.add_argument(
        "--target-body",
        default="reach_target",
    )
    parser.add_argument(
        "--policy-dt",
        type=float,
        default=0.02,
        help="策略周期，Isaac Lab中为0.005*4=0.02秒。",
    )

    parser.add_argument(
    "--reset-robot-each-target",
    action="store_true",
    help="切换目标时将机器人恢复到默认姿态，用于复现训练时的episode重置。",
)

    args = parser.parse_args()

    xml_path = Path(args.xml).expanduser().resolve()
    onnx_path = Path(args.onnx).expanduser().resolve()

   

    if not xml_path.is_file():
        raise FileNotFoundError(f"XML不存在：{xml_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX不存在：{onnx_path}")

    # ========================================================
    # 加载 MuJoCo
    # ========================================================

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    wrist_body_id = get_body_id(model, args.wrist_body)
    root_body_id = get_body_id(model, args.root_body)
    target_body_id = get_body_id(model, args.target_body)

    target_mocap_id = int(
    model.body_mocapid[target_body_id]
)

    if target_mocap_id < 0:
        raise RuntimeError(
            "reach_target 不是 mocap body。"
            "请在 scene_reach.xml 中加入 mocap=\"true\"。"
        )

    print("[INFO] target body id:", target_body_id)
    print("[INFO] target mocap id:", target_mocap_id)

    policy_joint_ids = [
        get_joint_id(model, name)
        for name in POLICY_JOINT_NAMES
    ]

    joint_actuator_map = create_joint_actuator_map(model)

    safe_lower, safe_upper = compute_safe_joint_limits(
        model,
        policy_joint_ids,
    )

    # ========================================================
    # 加载 ONNX
    # ========================================================

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(
            f"当前脚本要求单输入ONNX，实际输入数量={len(inputs)}"
        )
    if len(outputs) < 1:
        raise RuntimeError("ONNX没有输出")

    input_name = inputs[0].name
    output_name = outputs[0].name

    print("[INFO] ONNX input :", input_name, inputs[0].shape)
    print("[INFO] ONNX output:", output_name, outputs[0].shape)

    # ========================================================
    # 初始化姿态
    # ========================================================

    mujoco.mj_resetData(model, data)

     # ========================================================
    # 初始化随机目标
    # ========================================================

    rng = np.random.default_rng(args.seed)

    current_target = sample_target_position(
        rng=rng,
        previous_target=None,
)

    set_mocap_target(
        model=model,
        data=data,
        mocap_id=target_mocap_id,
        target_position=current_target,
    )

    success_hold_count = 0
    success_count = 0
    failure_count = 0
    finished_target_count = 0

    target_start_time = float(data.time)

    print("[INFO] First random target:", current_target)

    for joint_name, default_pos in DEFAULT_JOINT_POS.items():
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if joint_id < 0:
            print(f"[WARN] 跳过不存在的关节：{joint_name}")
            continue

        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr] = default_pos

    mujoco.mj_forward(model, data)

    last_action = np.zeros(4, dtype=np.float32)
    desired_policy_pos = POLICY_DEFAULT_POS.copy()

    physics_dt = float(model.opt.timestep)
    decimation = max(1, int(round(args.policy_dt / physics_dt)))

    print("[INFO] physics_dt =", physics_dt)
    print("[INFO] policy_dt  =", args.policy_dt)
    print("[INFO] decimation =", decimation)
    print("[INFO] policy joints:", POLICY_JOINT_NAMES)

    step_count = 0

    # ========================================================
    # 仿真循环
    # ========================================================

    viewer_context = (
        nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(model, data)
    )

    with viewer_context as viewer:
        while args.headless or viewer.is_running():

            wall_start = time.time()

            # ------------------------------------------------
            # 50 Hz策略推理
            # ------------------------------------------------

            if step_count % decimation == 0:
                joint_pos = np.empty(4, dtype=np.float32)
                joint_vel = np.empty(4, dtype=np.float32)

                for index, joint_id in enumerate(policy_joint_ids):
                    qpos_adr = int(model.jnt_qposadr[joint_id])
                    dof_adr = int(model.jnt_dofadr[joint_id])

                    joint_pos[index] = data.qpos[qpos_adr]
                    joint_vel[index] = data.qvel[dof_adr]

                # 与训练观测一致
                joint_pos_rel = joint_pos - POLICY_DEFAULT_POS
                joint_vel_rel = joint_vel * 0.05

                wrist_pos_w = data.xpos[wrist_body_id].copy()
                target_pos_w = data.xpos[target_body_id].copy()

                # MuJoCo xmat 为 body局部坐标到世界坐标的旋转
                root_rot_w = data.xmat[root_body_id].reshape(3, 3)

                error_w = target_pos_w - wrist_pos_w
                wrist_to_target_b = (
                    root_rot_w.T @ error_w
                ).astype(np.float32)

                obs = np.concatenate(
                    (
                        joint_pos_rel,
                        joint_vel_rel,
                        wrist_to_target_b,
                        last_action,
                    )
                ).astype(np.float32)

                if obs.shape != (15,):
                    raise RuntimeError(
                        f"观测应为15维，实际为{obs.shape}"
                    )

                output = session.run(
                    [output_name],
                    {input_name: obs.reshape(1, 15)},
                )[0]

                action = np.asarray(
                    output,
                    dtype=np.float32,
                ).reshape(-1)

                if action.shape != (4,):
                    raise RuntimeError(
                        f"动作应为4维，实际为{action.shape}"
                    )

                last_action = action.copy()

                desired_policy_pos = (
                    POLICY_DEFAULT_POS
                    + ACTION_SCALE * action
                )

                # 对应自定义 RightArmFixedBodyAction 中的安全限幅
                desired_policy_pos = np.clip(
                    desired_policy_pos,
                    safe_lower,
                    safe_upper,
                )

                # ========================================================
                # 当前手腕到目标的距离
                # ========================================================

                distance = float(
                    np.linalg.norm(wrist_to_target_b)
                )

                # 仿真时间，不使用墙钟时间
                target_elapsed_time = max(
                    0.0,
                    float(data.time - target_start_time),
                )

                # 连续进入成功范围才算成功
                if distance < args.success_distance:
                    success_hold_count += 1
                else:
                    success_hold_count = 0

                # 分别计算成功和超时，避免else缩进错配
                reached_target = (
                    target_elapsed_time >= MIN_TARGET_ACTIVE_TIME
                    and success_hold_count >= SUCCESS_HOLD_POLICY_STEPS
                )

                target_timed_out = (
                    target_elapsed_time
                    >= args.target_timeout
                )

                # 只有真正成功或真正超时才结束当前目标
                if reached_target or target_timed_out:

                    # 必须先加1，再打印、再计算成功率
                    finished_target_count += 1

                    if reached_target:
                        success_count += 1

                        print(
                            "\n"
                            f"[SUCCESS] "
                            f"target={finished_target_count} | "
                            f"time={target_elapsed_time:.3f}s | "
                            f"distance={distance:.4f}m | "
                            f"position={np.round(current_target, 3)}"
                        )

                    else:
                        failure_count += 1

                        print(
                            "\n"
                            f"[TIMEOUT] "
                            f"target={finished_target_count} | "
                            f"time={target_elapsed_time:.3f}s | "
                            f"distance={distance:.4f}m | "
                            f"position={np.round(current_target, 3)}"
                        )

                    # finished_target_count此时至少为1
                    success_rate = (
                        100.0
                        * success_count
                        / finished_target_count
                    )

                    print(
                        f"[STATS] "
                        f"success={success_count} | "
                        f"failure={failure_count} | "
                        f"success_rate={success_rate:.1f}%"
                    )

                    # 完成指定数量后退出仿真循环
                    if finished_target_count >= args.num_targets:
                        print("\n[INFO] Random target test finished.")
                        break

                    # ====================================================
                    # 生成下一个随机目标
                    # ====================================================

                    previous_target = current_target.copy()

                    current_target = sample_target_position(
                        rng,
                        previous_target=previous_target,
                    )

                    if args.reset_robot_each_target:
                        reset_robot_to_default(model, data)

                    target_shift = float(
                        np.linalg.norm(current_target - previous_target)
                    )

                    print(
                        "[INFO] Target moved:",
                        f"{target_shift:.3f} m",
                    )

                    set_mocap_target(
                        model=model,
                        data=data,
                        mocap_id=target_mocap_id,
                        target_position=current_target,
                    )

                    actual_target_pos = data.xpos[target_body_id].copy()

                    print(
                        "[DEBUG] mocap target:",
                        np.round(data.mocap_pos[target_mocap_id], 3),
                    )

                    print(
                        "[DEBUG] body target:",
                        np.round(actual_target_pos, 3),
                    )

                    # 新目标的计时和成功保持状态清零
                    success_hold_count = 0
                    target_start_time = float(data.time)

                    print(
                        "[INFO] New target:",
                        np.round(current_target, 3),
                    )
                    print(
                        "[INFO] Target moved:",
                        f"{target_shift:.3f} m",
                    )
                # ========================================================
                # 实时状态输出
                # ========================================================

                print(
                    "\r"
                    f"target={finished_target_count + 1}/"
                    f"{args.num_targets} | "
                    f"distance={distance:.4f}m | "
                    f"time={target_elapsed_time:.2f}s | "
                    f"success={success_count} | "
                    f"failure={failure_count} | "
                    f"target_pos={np.round(current_target, 3)}",
                    end="",
                    flush=True,
                )
            # ------------------------------------------------
            # MuJoCo每个物理步执行PD力矩控制
            # ------------------------------------------------

            for joint_name, default_pos in DEFAULT_JOINT_POS.items():
                joint_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name,
                )

                if joint_id < 0:
                    continue

                actuator_id = joint_actuator_map.get(joint_id)
                if actuator_id is None:
                    continue

                qpos_adr = int(model.jnt_qposadr[joint_id])
                dof_adr = int(model.jnt_dofadr[joint_id])

                current_pos = float(data.qpos[qpos_adr])
                current_vel = float(data.qvel[dof_adr])

                desired_pos = default_pos

                if joint_name in POLICY_JOINT_NAMES:
                    policy_index = POLICY_JOINT_NAMES.index(
                        joint_name
                    )
                    desired_pos = float(
                        desired_policy_pos[policy_index]
                    )

                kp, kd = get_pd_gain(joint_name)

                torque = (
                    kp * (desired_pos - current_pos)
                    - kd * current_vel
                )

                if model.actuator_ctrllimited[actuator_id]:
                    ctrl_low, ctrl_high = (
                        model.actuator_ctrlrange[actuator_id]
                    )
                    torque = float(
                        np.clip(torque, ctrl_low, ctrl_high)
                    )

                data.ctrl[actuator_id] = torque

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

            step_count += 1

            if viewer is not None:
                sleep_time = physics_dt - (
                    time.time() - wall_start
                )

                if sleep_time > 0:
                    time.sleep(sleep_time)

    print()


if __name__ == "__main__":

    main()
