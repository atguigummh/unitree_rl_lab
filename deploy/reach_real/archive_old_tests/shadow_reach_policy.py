#!/usr/bin/env python3

import argparse
import csv
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

import unitree_sdk2py.core.channel as channel
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


BIND_IP = "192.168.123.100"

# 策略真正使用的4个右臂关节
MODEL_JOINT_INDICES = [22, 23, 24, 25]

# 用于显示的全部右臂7个关节
RIGHT_ARM_INDICES = [22, 23, 24, 25, 26, 27, 28]

JOINT_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

# 训练环境中的默认关节位置
DEFAULT_Q = np.array(
    [0.30, -0.25, 0.00, 0.97],
    dtype=np.float32,
)

# 训练环境中的动作缩放
ACTION_SCALE = np.array(
    [0.18, 0.15, 0.15, 0.20],
    dtype=np.float32,
)

VELOCITY_SCALE = 0.05


class StateBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.q = None
        self.dq = None
        self.receive_time = None
        self.count = 0

    def callback(self, msg):
        try:
            q = np.array(
                [float(msg.motor_state[i].q) for i in RIGHT_ARM_INDICES],
                dtype=np.float32,
            )
            dq = np.array(
                [float(msg.motor_state[i].dq) for i in RIGHT_ARM_INDICES],
                dtype=np.float32,
            )
        except Exception as exc:
            print(f"[回调错误] {exc}")
            return

        with self.lock:
            self.q = q
            self.dq = dq
            self.receive_time = time.monotonic()
            self.count += 1

    def snapshot(self):
        with self.lock:
            if self.q is None:
                return None

            return (
                self.q.copy(),
                self.dq.copy(),
                self.receive_time,
                self.count,
            )


def create_dds_config():
    # 明确固定机器人网络IP，避免双IP时选中172.29.45.144。
    channel.ChannelConfigHasInterface = f"""
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface
          address="{BIND_IP}"
          multicast="true"
        />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="G1右臂ONNX策略纯影子推理，不发送机器人控制指令。"
    )

    parser.add_argument(
        "--interface",
        default="enp3s0",
    )
    parser.add_argument(
        "--model",
        default=(
            "/home/ma/unitree_clean/unitree_rl_lab/"
            "logs/rsl_rl/g1_right_wrist_reach_dynamic_005_v1/"
            "2026-06-25_09-24-34/exported/policy.onnx"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--infer-rate",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=5.0,
    )

    # 直接作为“手腕到目标点”的根坐标系向量输入。
    parser.add_argument("--target-dx", type=float, default=0.0)
    parser.add_argument("--target-dy", type=float, default=0.0)
    parser.add_argument("--target-dz", type=float, default=0.0)

    parser.add_argument(
        "--warning-delta",
        type=float,
        default=0.20,
        help="假想目标角度与真实角度差超过该值时报警，单位rad。",
    )
    parser.add_argument(
        "--log",
        default="",
        help="可选CSV输出路径。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到ONNX模型：{model_path}")

    print("=" * 78)
    print("[纯影子模式] 只订阅 rt/lowstate")
    print("[纯影子模式] 只运行ONNX推理并打印结果")
    print("[纯影子模式] 不向机器人发送任何指令")
    print(f"[DDS绑定地址] {BIND_IP}")
    print(f"[模型] {model_path}")
    print("=" * 78)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        raise RuntimeError(
            f"预期模型只有1个输入，实际为{len(inputs)}个。"
        )

    input_name = inputs[0].name
    output_name = outputs[0].name

    print(
        f"[模型输入] name={input_name}, "
        f"shape={inputs[0].shape}, type={inputs[0].type}"
    )
    print(
        f"[模型输出] name={output_name}, "
        f"shape={outputs[0].shape}, type={outputs[0].type}"
    )

    target_vector = np.array(
        [
            args.target_dx,
            args.target_dy,
            args.target_dz,
        ],
        dtype=np.float32,
    )

    print(
        "[目标相对向量] "
        f"dx={target_vector[0]:+.4f} m, "
        f"dy={target_vector[1]:+.4f} m, "
        f"dz={target_vector[2]:+.4f} m"
    )

    create_dds_config()
    channel.ChannelFactoryInitialize(0, args.interface)

    state = StateBuffer()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(state.callback, 10)

    print("[等待] 正在等待 rt/lowstate……")

    wait_deadline = time.monotonic() + 5.0
    while state.snapshot() is None:
        if time.monotonic() >= wait_deadline:
            raise TimeoutError("5秒内没有收到rt/lowstate。")
        time.sleep(0.05)

    print("[成功] 已收到机器人状态，开始纯影子推理。")

    csv_file = None
    csv_writer = None

    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        csv_file = log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "elapsed_s",
                "state_age_ms",
                "q22",
                "q23",
                "q24",
                "q25",
                "dq22",
                "dq23",
                "dq24",
                "dq25",
                "target_dx",
                "target_dy",
                "target_dz",
                "action0",
                "action1",
                "action2",
                "action3",
                "pred_q22",
                "pred_q23",
                "pred_q24",
                "pred_q25",
                "delta_q22",
                "delta_q23",
                "delta_q24",
                "delta_q25",
                "warning",
            ]
        )

        print(f"[CSV日志] {log_path}")

    previous_action = np.zeros(4, dtype=np.float32)

    infer_period = 1.0 / args.infer_rate
    print_period = 1.0 / args.print_rate

    start_time = time.monotonic()
    next_infer_time = start_time
    next_print_time = start_time

    inference_count = 0

    try:
        while True:
            now = time.monotonic()

            if now - start_time >= args.duration:
                break

            if now < next_infer_time:
                time.sleep(
                    min(
                        next_infer_time - now,
                        0.002,
                    )
                )
                continue

            next_infer_time += infer_period

            snapshot = state.snapshot()
            if snapshot is None:
                continue

            q_all, dq_all, receive_time, _ = snapshot

            q_model = q_all[:4]
            dq_model = dq_all[:4]

            joint_position_observation = q_model - DEFAULT_Q
            joint_velocity_observation = (
                dq_model * VELOCITY_SCALE
            )

            observation = np.concatenate(
                [
                    joint_position_observation,
                    joint_velocity_observation,
                    target_vector,
                    previous_action,
                ]
            ).astype(np.float32)

            if observation.shape != (15,):
                raise RuntimeError(
                    f"观测维数错误：{observation.shape}"
                )

            if not np.all(np.isfinite(observation)):
                raise RuntimeError("观测中出现NaN或Inf。")

            action = session.run(
                [output_name],
                {
                    input_name: observation.reshape(1, 15)
                },
            )[0]

            action = np.asarray(
                action,
                dtype=np.float32,
            ).reshape(-1)

            if action.shape != (4,):
                raise RuntimeError(
                    f"策略输出维数错误：{action.shape}"
                )

            if not np.all(np.isfinite(action)):
                raise RuntimeError("策略输出出现NaN或Inf。")

            # 这里只计算“假想目标角度”，绝不发送给机器人。
            predicted_q_target = (
                DEFAULT_Q + ACTION_SCALE * action
            )

            delta_from_real = predicted_q_target - q_model

            warning = bool(
                np.max(np.abs(delta_from_real))
                > args.warning_delta
            )

            previous_action = action.copy()
            inference_count += 1

            elapsed = now - start_time
            state_age_ms = (
                time.monotonic() - receive_time
            ) * 1000.0

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        elapsed,
                        state_age_ms,
                        *q_model.tolist(),
                        *dq_model.tolist(),
                        *target_vector.tolist(),
                        *action.tolist(),
                        *predicted_q_target.tolist(),
                        *delta_from_real.tolist(),
                        int(warning),
                    ]
                )

            if now >= next_print_time:
                next_print_time += print_period

                status = (
                    "警告：假想动作偏差较大"
                    if warning
                    else "正常"
                )

                print("\n" + "-" * 78)
                print(
                    f"时间={elapsed:6.2f}s  "
                    f"状态延迟={state_age_ms:6.3f}ms  "
                    f"推理次数={inference_count}  "
                    f"状态={status}"
                )
                print(
                    "真实q      = "
                    + np.array2string(
                        q_model,
                        precision=4,
                        floatmode="fixed",
                    )
                )
                print(
                    "原始action = "
                    + np.array2string(
                        action,
                        precision=4,
                        floatmode="fixed",
                    )
                )
                print(
                    "假想目标q  = "
                    + np.array2string(
                        predicted_q_target,
                        precision=4,
                        floatmode="fixed",
                    )
                )
                print(
                    "目标-真实q = "
                    + np.array2string(
                        delta_from_real,
                        precision=4,
                        floatmode="fixed",
                    )
                )

    except KeyboardInterrupt:
        print("\n[退出] 用户按下Ctrl+C。")

    finally:
        if csv_file is not None:
            csv_file.flush()
            csv_file.close()

    print("=" * 78)
    print(f"[完成] 总推理次数：{inference_count}")
    print("[完成] 全程未向机器人发送任何控制指令。")
    print("=" * 78)


if __name__ == "__main__":
    main()
