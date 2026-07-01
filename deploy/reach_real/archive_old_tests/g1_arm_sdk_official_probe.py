#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Dict

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


ARM_JOINTS: Dict[str, int] = {
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}

CONTROL_JOINTS = list(range(15, 29)) + [12, 13, 14]
NOT_USED_JOINT = 29


class OfficialStyleProbe:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        if abs(args.delta) > 0.10:
            raise ValueError("delta 限制为 |delta| <= 0.10 rad")
        if args.duration > 5.0:
            raise ValueError("duration 限制为 <= 5.0 s")
        if args.joint not in ARM_JOINTS:
            raise ValueError(f"未知关节：{args.joint}")

        self.test_joint = ARM_JOINTS[args.joint]

    def setup_dds(self):
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
        channel_mod.ChannelConfigHasInterface = xml

        f = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix="cyclonedds_official_probe_",
            delete=False,
        )
        f.write(xml)
        f.close()
        os.environ["CYCLONEDDS_URI"] = "file://" + f.name

        print("[DDS] interface =", self.args.interface)
        print("[DDS] local_ip  =", self.args.local_ip)
        print("[DDS] CYCLONEDDS_URI =", os.environ["CYCLONEDDS_URI"])

    def cb(self, msg: LowState_):
        self.low_state = msg
        self.low_state_time = time.monotonic()

    def wait_state(self):
        print("[等待] rt/lowstate ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.args.state_timeout:
            if self.low_state is not None:
                print("[收到] rt/lowstate")
                return
            time.sleep(0.01)
        raise TimeoutError("没有收到 rt/lowstate")

    def q(self, idx: int) -> float:
        return float(self.low_state.motor_state[idx].q)

    def write_cmd(self, pub, target, enable: float):
        self.low_cmd.motor_cmd[NOT_USED_JOINT].q = float(enable)

        for j in CONTROL_JOINTS:
            mc = self.low_cmd.motor_cmd[j]
            mc.tau = 0.0
            mc.q = float(target[j])
            mc.dq = 0.0
            mc.kp = float(self.args.kp)
            mc.kd = float(self.args.kd)

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        pub.Write(self.low_cmd)

    def run(self):
        self.setup_dds()
        ChannelFactoryInitialize(0, self.args.interface)

        # 按官方示例：先创建 Publisher，再创建 Subscriber
        pub = ChannelPublisher(self.args.arm_topic, LowCmd_)
        pub.Init()
        print("[发布] topic =", self.args.arm_topic)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.cb, 10)
        print("[订阅] rt/lowstate")

        self.wait_state()

        q0 = {j: self.q(j) for j in CONTROL_JOINTS}
        q1 = dict(q0)
        q1[self.test_joint] = q0[self.test_joint] + self.args.delta

        print()
        print("[起始]")
        print(f"{self.args.joint} index={self.test_joint} q0={q0[self.test_joint]:+.6f}")
        print("[目标]")
        print(f"{self.args.joint} index={self.test_joint} q1={q1[self.test_joint]:+.6f}")
        print("[参数]")
        print(f"delta={self.args.delta:+.6f} rad")
        print(f"duration={self.args.duration:.3f} s")
        print(f"kp={self.args.kp:.3f}, kd={self.args.kd:.3f}")
        print(f"enable_write={self.args.enable_write}")
        print()

        if self.args.enable_write != "YES":
            print("[DRY-RUN] 未发送控制。加 --enable-write YES 才会发送。")
            return

        dt = 1.0 / self.args.rate
        t0 = time.monotonic()
        count = 0

        print("[开始] 官方写法 probe：发送 5° 小动作")
        try:
            while True:
                elapsed = time.monotonic() - t0
                if elapsed > self.args.duration:
                    break

                if time.monotonic() - self.low_state_time > 0.1:
                    raise RuntimeError("lowstate 超时")

                ratio = min(1.0, elapsed / 1.0)

                target = {}
                for j in CONTROL_JOINTS:
                    target[j] = q0[j] + ratio * (q1[j] - q0[j])

                self.write_cmd(pub, target, enable=1.0)

                count += 1
                if count % int(max(1, self.args.rate / 5)) == 0:
                    actual = self.q(self.test_joint)
                    target_val = target[self.test_joint]
                    err = target_val - actual
                    print(
                        f"t={elapsed:.3f}s "
                        f"ratio={ratio:.3f} "
                        f"target={target_val:+.6f} "
                        f"actual={actual:+.6f} "
                        f"err={err:+.6f}"
                    )

                time.sleep(dt)

        finally:
            print("[释放] enable=0")
            q_now = {j: self.q(j) for j in CONTROL_JOINTS}
            for _ in range(30):
                self.write_cmd(pub, q_now, enable=0.0)
                time.sleep(dt)

            print("[完成]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--local-ip", default="192.168.123.100")
    parser.add_argument("--arm-topic", default="rt/arm_sdk", choices=["rt/arm_sdk", "rt/armsdk"])
    parser.add_argument("--joint", default="right_shoulder_pitch", choices=sorted(ARM_JOINTS.keys()))
    parser.add_argument("--delta", type=float, default=0.087)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--kp", type=float, default=30.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--enable-write", default="NO")
    args = parser.parse_args()

    OfficialStyleProbe(args).run()


if __name__ == "__main__":
    main()
