#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Dict, List

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


ARM_JOINTS: Dict[str, int] = {
    "left_shoulder_pitch": 15,
    "left_shoulder_roll": 16,
    "left_shoulder_yaw": 17,
    "left_elbow": 18,
    "left_wrist_roll": 19,
    "left_wrist_pitch": 20,
    "left_wrist_yaw": 21,
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

WAIST_JOINTS: Dict[str, int] = {
    "waist_yaw": 12,
    "waist_roll": 13,
    "waist_pitch": 14,
}

# 官方 g1_arm7_sdk_dds_example.py 里 arm_joints 包含：
# 双臂 15~28 + 腰部 12/13/14
CONTROL_JOINTS = list(range(15, 29)) + [12, 13, 14]

NOT_USED_JOINT = 29  # arm_sdk 权重开关位：q=1 enable, q=0 release


class SafeArmTinyTest:
    def __init__(self, args):
        self.args = args

        self.latest_state = None
        self.latest_time = 0.0
        self.latest_topic = None

        self.crc = CRC()

        self.write_enabled = args.enable_write == "YES"

        if abs(args.delta) > 0.10:
            raise ValueError("delta 过大。当前真机测试限制为 |delta| <= 0.10 rad")

        if args.duration > 5.0:
            raise ValueError("duration 过长。当前真机测试限制为 <= 5.0 s")

        if args.rate > 100:
            raise ValueError("rate 过高。当前真机测试限制为 <= 100 Hz")

        if args.kp > 80.0:
            raise ValueError("kp 过大。当前真机测试限制为 <= 80")

        if args.kd > 5.0:
            raise ValueError("kd 过大。当前真机测试限制为 <= 5")

        if args.joint not in ARM_JOINTS:
            raise ValueError(f"未知关节：{args.joint}")

        self.test_joint_id = ARM_JOINTS[args.joint]

    def setup_cyclonedds(self):
        """强制 Unitree SDK / CycloneDDS 绑定到 192.168.123.100。"""

        xml = f"""<CycloneDDS>
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="{self.args.local_ip}" multicast="true" />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
"""

        # Unitree SDK 内部会对 ChannelConfigHasInterface 调用 .replace(...)
        # 所以这里必须赋值为字符串，不能赋值为函数。
        channel_mod.ChannelConfigHasInterface = xml

        f = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix="cyclonedds_g1_arm_sdk_",
            delete=False,
        )
        f.write(xml)
        f.close()

        os.environ["CYCLONEDDS_URI"] = "file://" + f.name

        print("[DDS] interface =", self.args.interface)
        print("[DDS] local_ip  =", self.args.local_ip)
        print("[DDS] CYCLONEDDS_URI =", os.environ["CYCLONEDDS_URI"])

    def lowstate_callback(self, msg: LowState_):
        self.latest_state = msg
        self.latest_time = time.monotonic()
        self.latest_topic = "rt/lowstate"

    def wait_lowstate(self, timeout: float):
        print("[低状态] 等待 rt/lowstate ...")
        t0 = time.monotonic()

        while time.monotonic() - t0 < timeout:
            if self.latest_state is not None:
                age_ms = (time.monotonic() - self.latest_time) * 1000.0
                print(
                    f"[低状态] 已收到，topic={self.latest_topic}，age={age_ms:.3f} ms"
                )
                return
            time.sleep(0.01)

        raise TimeoutError("未收到 rt/lowstate，停止。不会发送任何控制。")

    def get_current_q(self) -> List[float]:
        state = self.latest_state
        if state is None:
            raise RuntimeError("没有 lowstate")

        q = []
        for i in range(len(state.motor_state)):
            q.append(float(state.motor_state[i].q))
        return q

    def print_arm_q(self, q: List[float], title: str):
        print()
        print("=" * 72)
        print(title)
        print("=" * 72)

        for name, idx in ARM_JOINTS.items():
            print(f"{name:24s} index={idx:2d} q={q[idx]:+.6f} rad")

        print("-" * 72)
        for name, idx in WAIST_JOINTS.items():
            print(f"{name:24s} index={idx:2d} q={q[idx]:+.6f} rad")

        print("=" * 72)

    def make_cmd(self, target_q: Dict[int, float], enable: float):
        cmd = unitree_hg_msg_dds__LowCmd_()

        # q=1：Enable arm_sdk
        # q=0：Release arm_sdk
        cmd.motor_cmd[NOT_USED_JOINT].q = float(enable)

        for joint in CONTROL_JOINTS:
            mc = cmd.motor_cmd[joint]

            # 跟官方 g1_arm7_sdk_dds_example.py 保持一致：
            # 不额外设置 mode，只设置 tau/q/dq/kp/kd。
            mc.tau = 0.0
            mc.q = float(target_q[joint])
            mc.dq = 0.0
            mc.kp = float(self.args.kp)
            mc.kd = float(self.args.kd)

        cmd.crc = self.crc.Crc(cmd)
        return cmd

    def run(self):
        self.setup_cyclonedds()

        ChannelFactoryInitialize(0, self.args.interface)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.lowstate_callback, 10)
        print("[订阅] rt/lowstate")

        pub = None
        if self.write_enabled:
            print("[安全顺序] 已请求写入，但先不创建 Publisher。")
            print("[安全顺序] 先等待 rt/lowstate，确认状态正常后再创建 rt/arm_sdk Publisher。")
        else:
            print("[DRY-RUN] 未启用写入。不会发送任何控制。")
            print("[DRY-RUN] 若确认无误，运行时加：--enable-write YES")

        self.wait_lowstate(timeout=self.args.state_timeout)

        q0 = self.get_current_q()
        self.print_arm_q(q0, "启动时读取到的当前手臂/腰部关节角")

        if self.write_enabled:
            pub = ChannelPublisher(self.args.arm_topic, LowCmd_)
            pub.Init()
            print(f"[发布] 已启用写入：topic={self.args.arm_topic}")

        target0 = {j: q0[j] for j in CONTROL_JOINTS}
        target1 = dict(target0)

        if self.args.mode == "hold":
            print("[模式] hold：只保持当前姿态，不增加目标位移。")
        elif self.args.mode == "tiny":
            target1[self.test_joint_id] = q0[self.test_joint_id] + self.args.delta
            print(
                f"[模式] tiny：{self.args.joint} index={self.test_joint_id} "
                f"目标增加 {self.args.delta:+.6f} rad"
            )
        else:
            raise ValueError("mode 只能是 hold 或 tiny")

        print()
        print("[限制]")
        print(f"duration = {self.args.duration:.3f} s")
        print(f"rate     = {self.args.rate:.1f} Hz")
        print(f"kp       = {self.args.kp:.3f}")
        print(f"kd       = {self.args.kd:.3f}")
        print(f"write    = {self.write_enabled}")
        print()

        if not self.write_enabled:
            q_target_preview = list(q0)
            for j in CONTROL_JOINTS:
                q_target_preview[j] = target1[j]
            self.print_arm_q(q_target_preview, "DRY-RUN 目标手臂/腰部关节角")
            print("[DRY-RUN] 结束。")
            return

        dt = 1.0 / self.args.rate
        t0 = time.monotonic()
        count = 0

        print("[开始] 正在向 rt/arm_sdk 发送动作。按 Ctrl+C 可停止。")

        try:
            while True:
                now = time.monotonic()
                elapsed = now - t0

                if elapsed > self.args.duration:
                    break

                if self.latest_state is None:
                    raise RuntimeError("lowstate 丢失")

                state_age = time.monotonic() - self.latest_time
                if state_age > 0.1:
                    raise RuntimeError(f"lowstate 超时：{state_age:.3f}s")

                # 前 1 秒平滑插值到目标，避免阶跃
                ramp_time = min(1.0, self.args.duration)
                ratio = min(1.0, elapsed / ramp_time)

                target = {}
                for j in CONTROL_JOINTS:
                    target[j] = target0[j] + ratio * (target1[j] - target0[j])

                cmd = self.make_cmd(target, enable=1.0)
                pub.Write(cmd)

                count += 1
                if count % int(max(1, self.args.rate / 5)) == 0:
                    q_live = self.get_current_q()
                    actual = q_live[self.test_joint_id]
                    target_val = target[self.test_joint_id]
                    err = target_val - actual

                    print(
                        f"t={elapsed:.3f}s "
                        f"ratio={ratio:.3f} "
                        f"{self.args.joint} "
                        f"target={target_val:+.6f} "
                        f"actual={actual:+.6f} "
                        f"err={err:+.6f}"
                    )

                time.sleep(dt)

        except KeyboardInterrupt:
            print("\n[停止] 收到 Ctrl+C")
        finally:
            print("[释放] 发送 arm_sdk 释放指令。")

            q_now = self.get_current_q()
            release_target = {j: q_now[j] for j in CONTROL_JOINTS}

            for _ in range(30):
                cmd = self.make_cmd(release_target, enable=0.0)
                pub.Write(cmd)
                time.sleep(dt)

            print("[完成] 已退出。")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interface",
        default="enp3s0",
        help="连接 G1 的网卡名",
    )
    parser.add_argument(
        "--local-ip",
        default="192.168.123.100",
        help="本机连接 G1 的 IP",
    )
    parser.add_argument(
        "--mode",
        choices=["hold", "tiny"],
        default="hold",
        help="hold=只保持当前位置，tiny=小幅动作",
    )
    parser.add_argument(
        "--joint",
        default="right_shoulder_pitch",
        choices=sorted(ARM_JOINTS.keys()),
        help="tiny 模式下要测试的关节",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.087,
        help="tiny 模式下目标增量。5度约等于 0.087 rad",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="运行时长，当前限制 <= 5 秒",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="发布频率，当前限制 <= 100 Hz",
    )
    parser.add_argument(
        "--kp",
        type=float,
        default=30.0,
        help="位置刚度，当前限制 <= 80",
    )
    parser.add_argument(
        "--kd",
        type=float,
        default=1.5,
        help="阻尼，当前限制 <= 5",
    )
    parser.add_argument(
        "--arm-topic",
        default="rt/arm_sdk",
        choices=["rt/arm_sdk", "rt/armsdk"],
        help="高层手臂控制 topic，默认 rt/arm_sdk；如不执行可试 rt/armsdk",
    )

    parser.add_argument(
        "--state-timeout",
        type=float,
        default=30.0,
        help="等待 lowstate 的最长时间",
    )
    parser.add_argument(
        "--enable-write",
        default="NO",
        help='必须写成 YES 才会真正发布到 rt/arm_sdk',
    )

    args = parser.parse_args()

    test = SafeArmTinyTest(args)
    test.run()


if __name__ == "__main__":
    main()
