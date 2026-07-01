#!/usr/bin/env python3
# 作用：读取并打印 Unitree G1 右臂当前关节状态。

import argparse
import math
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


# G1 29DoF右臂关节索引
RIGHT_ARM_JOINTS = [
    ("right_shoulder_pitch", 22),
    ("right_shoulder_roll", 23),
    ("right_shoulder_yaw", 24),
    ("right_elbow", 25),
    ("right_wrist_roll", 26),
    ("right_wrist_pitch", 27),
    ("right_wrist_yaw", 28),
]


class LowStateBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = None
        self.last_receive_time = 0.0
        self.message_count = 0

    def callback(self, msg):
        with self.lock:
            self.state = msg
            self.last_receive_time = time.monotonic()
            self.message_count += 1

    def snapshot(self):
        with self.lock:
            return (
                self.state,
                self.last_receive_time,
                self.message_count,
            )


def main():
    parser = argparse.ArgumentParser(
        description="只读取G1右臂关节状态，不发送任何控制指令。"
    )
    parser.add_argument(
        "interface",
        help="连接G1的网卡，例如 enp3s0",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="运行时间，单位秒",
    )
    parser.add_argument(
        "--print-rate",
        type=float,
        default=5.0,
        help="打印频率，单位Hz",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("[安全模式] 只订阅 rt/lowstate")
    print("[安全模式] 没有创建 ChannelPublisher")
    print("[安全模式] 不会发布 rt/arm_sdk 或 rt/lowcmd")
    print(f"[网络接口] {args.interface}")
    print("=" * 75)

    ChannelFactoryInitialize(0, args.interface)

    buffer = LowStateBuffer()

    subscriber = ChannelSubscriber(
        "rt/lowstate",
        LowState_,
    )
    subscriber.Init(buffer.callback, 10)

    start_time = time.monotonic()
    last_print_time = 0.0
    last_rate_time = start_time
    last_message_count = 0
    receive_rate = 0.0

    while time.monotonic() - start_time < args.duration:
        now = time.monotonic()
        state, last_receive_time, message_count = buffer.snapshot()

        if state is None:
            if now - start_time > 5.0:
                print("\n[错误] 5秒内没有收到 rt/lowstate。")
                print("请检查网卡、机器人状态和DDS通信。")
                return

            print(
                "\r[等待] 正在等待 rt/lowstate 数据……",
                end="",
                flush=True,
            )
            time.sleep(0.05)
            continue

        message_age = now - last_receive_time

        if message_age > 0.5:
            print(
                f"\n[错误] LowState通信超时："
                f"{message_age:.3f} 秒"
            )
            return

        motor_state = state.motor_state

        if len(motor_state) <= 28:
            print(
                f"\n[错误] motor_state长度为 {len(motor_state)}，"
                "不能按G1 29DoF索引读取。"
            )
            return

        if now - last_rate_time >= 1.0:
            elapsed = now - last_rate_time
            receive_rate = (
                message_count - last_message_count
            ) / elapsed

            last_message_count = message_count
            last_rate_time = now

        if now - last_print_time >= 1.0 / args.print_rate:
            last_print_time = now

            print("\033[2J\033[H", end="")
            print("=" * 75)
            print("G1右臂关节状态——只读模式")
            print(f"消息接收频率：{receive_rate:.1f} Hz")
            print(f"最新消息延迟：{message_age * 1000:.1f} ms")
            print("-" * 75)

            for joint_name, joint_index in RIGHT_ARM_JOINTS:
                joint = motor_state[joint_index]

                q = float(joint.q)
                dq = float(joint.dq)

                if not math.isfinite(q) or not math.isfinite(dq):
                    print(
                        f"[错误] {joint_name}出现NaN或Inf："
                        f"q={q}, dq={dq}"
                    )
                    return

                print(
                    f"{joint_name:<23} "
                    f"index={joint_index:2d}  "
                    f"q={q:+8.4f} rad  "
                    f"dq={dq:+8.4f} rad/s"
                )

            print("-" * 75)
            print("程序没有发送控制指令，按Ctrl+C可安全退出。")

        time.sleep(0.002)

    print("\n[完成] 只读测试正常结束。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[退出] 用户按下Ctrl+C，程序已停止。")
