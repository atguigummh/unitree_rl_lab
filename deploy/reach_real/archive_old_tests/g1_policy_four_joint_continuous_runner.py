#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from typing import Dict

import numpy as np
import onnxruntime as ort

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


REAL_JOINT_IDS = {
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
}

POLICY_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

ACTION_SCALE = np.array([0.18, 0.15, 0.15, 0.20], dtype=np.float32)
DEFAULT_RIGHT_ARM_Q = np.array([0.30, -0.25, 0.00, 0.97], dtype=np.float32)

# pitch / roll / yaw / elbow 的最大相对启动姿态偏移
DEFAULT_MAX_DELTA = np.array([0.30, 0.16, 0.18, 0.50], dtype=np.float32)

CONTROL_JOINTS = list(range(15, 29)) + [12, 13, 14]
NOT_USED_JOINT = 29


class FourJointContinuousRunner:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.last_action = np.zeros(4, dtype=np.float32)
        self.cmd_target = None

        if args.real_scale > 3.5:
            raise ValueError("real_scale 限制为 <= 3.5")
        if args.duration > 90.0:
            raise ValueError("duration 限制为 <= 90s")
        if args.max_joint_speed > 0.30:
            raise ValueError("max_joint_speed 限制为 <= 0.30 rad/s")

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
            prefix="cyclonedds_four_joint_continuous_",
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

    def wait_lowstate(self):
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

    def dq(self, idx: int) -> float:
        return float(self.low_state.motor_state[idx].dq)

    def q_policy(self):
        return np.array([self.q(REAL_JOINT_IDS[n]) for n in POLICY_JOINTS], dtype=np.float32)

    def dq_policy(self):
        return np.array([self.dq(REAL_JOINT_IDS[n]) for n in POLICY_JOINTS], dtype=np.float32)

    def continuous_target(self, t: float):
        """
        这里仍然是伪 wrist_to_target，不是真实 FK 计算出来的空间误差。
        目的是先验证：连续变化的目标输入能否驱动 policy 产生连续右臂运动。
        """
        if self.args.path == "circle_xz":
            x = self.args.radius_x * math.sin(2.0 * math.pi * t / self.args.period)
            y = 0.0
            z = self.args.radius_z * math.cos(2.0 * math.pi * t / self.args.period)

        elif self.args.path == "circle_yz":
            x = 0.0
            y = self.args.radius_y * math.sin(2.0 * math.pi * t / self.args.period)
            z = self.args.radius_z * math.cos(2.0 * math.pi * t / self.args.period)

        elif self.args.path == "eight_xz":
            x = self.args.radius_x * math.sin(2.0 * math.pi * t / self.args.period)
            y = 0.0
            z = self.args.radius_z * math.sin(4.0 * math.pi * t / self.args.period)

        elif self.args.path == "up_down":
            x = 0.0
            y = 0.0
            z = self.args.radius_z * math.sin(2.0 * math.pi * t / self.args.period)

        else:
            raise ValueError("path 只能是 circle_xz / circle_yz / eight_xz / up_down")

        return np.array(
            [self.args.base_x + x, self.args.base_y + y, self.args.base_z + z],
            dtype=np.float32,
        )

    def make_obs(self, t: float):
        q = self.q_policy()
        dq = self.dq_policy()

        joint_pos_rel = q - DEFAULT_RIGHT_ARM_Q
        joint_vel_scaled = dq * 0.05
        wrist_to_target = self.continuous_target(t)

        obs = np.concatenate(
            [joint_pos_rel, joint_vel_scaled, wrist_to_target, self.last_action],
            axis=0,
        ).astype(np.float32)

        return obs[None, :], q, dq, wrist_to_target

    def write_cmd(self, pub, target: Dict[int, float], enable: float):
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

        print()
        print("[安全确认]")
        print("本脚本只发布 rt/arm_sdk")
        print("本脚本不发布 rt/lowcmd")
        print("本脚本连续控制 right_shoulder_pitch / roll / yaw / right_elbow")
        print("中途不会回初始位，只在结束时慢慢回收")
        print()

        pub = ChannelPublisher(self.args.arm_topic, LowCmd_)
        pub.Init()
        print("[发布]", self.args.arm_topic)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.cb, 10)
        print("[订阅] rt/lowstate")

        self.wait_lowstate()

        print("[ONNX] 加载 policy:", self.args.policy)
        session = ort.InferenceSession(self.args.policy, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        print("[ONNX] input =", inp.name, inp.shape)
        print("[ONNX] output =", out.name, out.shape)

        q_start = {j: self.q(j) for j in CONTROL_JOINTS}
        policy_real_ids = [REAL_JOINT_IDS[n] for n in POLICY_JOINTS]

        print()
        print("[启动姿态]")
        for name in POLICY_JOINTS:
            print(f"{name:28s} q_start = {self.q(REAL_JOINT_IDS[name]):+.6f}")

        print()
        print("[连续目标]")
        print("path =", self.args.path)
        print("period =", self.args.period)
        print("radius_x =", self.args.radius_x)
        print("radius_y =", self.args.radius_y)
        print("radius_z =", self.args.radius_z)
        print("real_scale =", self.args.real_scale)
        print("squash_temp =", self.args.squash_temp)
        print("max_joint_speed =", self.args.max_joint_speed)
        print("joint gains =",
              self.args.pitch_gain,
              self.args.roll_gain,
              self.args.yaw_gain,
              self.args.elbow_gain)
        print("enable_write =", self.args.enable_write)
        print()

        self.cmd_target = dict(q_start)

        dt = 1.0 / self.args.rate
        total_time = self.args.pre_hold + self.args.ramp_in + self.args.duration
        t0 = time.monotonic()
        count = 0

        try:
            while True:
                elapsed = time.monotonic() - t0
                if elapsed > total_time:
                    break

                if time.monotonic() - self.low_state_time > 0.2:
                    raise RuntimeError("lowstate 超时")

                if elapsed < self.args.pre_hold:
                    phase = "pre_hold"
                    gain = 0.0
                    target_time = 0.0
                elif elapsed < self.args.pre_hold + self.args.ramp_in:
                    phase = "ramp_in"
                    gain = (elapsed - self.args.pre_hold) / self.args.ramp_in
                    gain = max(0.0, min(1.0, gain))
                    target_time = elapsed - self.args.pre_hold
                else:
                    phase = "tracking"
                    gain = 1.0
                    target_time = elapsed - self.args.pre_hold

                obs, q, dq, w2t = self.make_obs(target_time)
                action_raw = session.run([out.name], {inp.name: obs})[0].astype(np.float32)[0]

                # tanh 平滑压缩，避免 raw action 饱和后完全固定
                action_used = np.tanh(action_raw / self.args.squash_temp).astype(np.float32)

                joint_gain = np.array(
                    [
                        self.args.pitch_gain,
                        self.args.roll_gain,
                        self.args.yaw_gain,
                        self.args.elbow_gain,
                    ],
                    dtype=np.float32,
                )

                delta = action_used * ACTION_SCALE * self.args.real_scale * joint_gain * gain
                delta = np.clip(delta, -DEFAULT_MAX_DELTA, DEFAULT_MAX_DELTA)

                desired_target = dict(q_start)
                for i, jid in enumerate(policy_real_ids):
                    desired_target[jid] = q_start[jid] + float(delta[i])

                # 目标关节角限速，避免连续目标变化时突然抽动
                max_step = self.args.max_joint_speed * dt
                target = dict(self.cmd_target)
                for j in CONTROL_JOINTS:
                    diff = desired_target[j] - self.cmd_target[j]
                    diff = float(np.clip(diff, -max_step, max_step))
                    target[j] = self.cmd_target[j] + diff

                self.cmd_target = dict(target)

                if self.args.enable_write == "YES":
                    self.write_cmd(pub, target, enable=1.0)

                self.last_action = action_used.copy()

                count += 1
                if count % int(max(1, self.args.rate / self.args.print_rate)) == 0:
                    actual = np.array([self.q(jid) for jid in policy_real_ids], dtype=np.float32)
                    target_vec = np.array([target[jid] for jid in policy_real_ids], dtype=np.float32)

                    print(
                        f"t={elapsed:05.2f}s "
                        f"{phase:9s} "
                        f"w2t={np.round(w2t, 3)} "
                        f"raw={np.round(action_raw, 2)} "
                        f"used={np.round(action_used, 2)} "
                        f"delta={np.round(delta, 3)} "
                        f"target={np.round(target_vec, 3)} "
                        f"actual={np.round(actual, 3)}"
                    )

                time.sleep(dt)

        finally:
            print("[回收] 连续追踪结束，慢慢回到启动姿态，再释放 arm_sdk")

            current_target = dict(self.cmd_target)
            return_steps = max(1, int(self.args.return_time * self.args.rate))

            for k in range(return_steps):
                ratio = (k + 1) / return_steps
                target = {}
                for j in CONTROL_JOINTS:
                    target[j] = current_target[j] + ratio * (q_start[j] - current_target[j])
                if self.args.enable_write == "YES":
                    self.write_cmd(pub, target, enable=1.0)
                time.sleep(dt)

            post_steps = max(1, int(self.args.post_hold * self.args.rate))
            print(f"[保持] 已回到启动姿态，保持 {self.args.post_hold:.1f}s 后释放")
            for _ in range(post_steps):
                if self.args.enable_write == "YES":
                    self.write_cmd(pub, q_start, enable=1.0)
                time.sleep(dt)

            for _ in range(30):
                if self.args.enable_write == "YES":
                    self.write_cmd(pub, q_start, enable=0.0)
                time.sleep(dt)

            print("[完成]")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--local-ip", default="192.168.123.100")
    parser.add_argument("--arm-topic", default="rt/arm_sdk")
    parser.add_argument(
        "--policy",
        default="/home/ma/unitree_clean/unitree_rl_lab/logs/rsl_rl/g1_right_wrist_reach_dynamic_005_v1/2026-06-25_09-24-34/exported/policy.onnx",
    )

    parser.add_argument("--path", default="circle_xz", choices=["circle_xz", "circle_yz", "eight_xz", "up_down"])
    parser.add_argument("--period", type=float, default=12.0)

    parser.add_argument("--base-x", type=float, default=0.0)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--base-z", type=float, default=0.0)

    parser.add_argument("--radius-x", type=float, default=0.10)
    parser.add_argument("--radius-y", type=float, default=0.08)
    parser.add_argument("--radius-z", type=float, default=0.16)

    parser.add_argument("--pre-hold", type=float, default=2.0)
    parser.add_argument("--ramp-in", type=float, default=6.0)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--return-time", type=float, default=12.0)
    parser.add_argument("--post-hold", type=float, default=5.0)

    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--print-rate", type=float, default=5.0)

    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=3.0)

    parser.add_argument("--real-scale", type=float, default=2.5)
    parser.add_argument("--squash-temp", type=float, default=6.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.12)

    # 四个动作维度的额外增益：
    # pitch, roll, yaw, elbow
    parser.add_argument("--pitch-gain", type=float, default=1.0)
    parser.add_argument("--roll-gain", type=float, default=1.0)
    parser.add_argument("--yaw-gain", type=float, default=1.0)
    parser.add_argument("--elbow-gain", type=float, default=1.0)

    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--enable-write", default="NO")

    args = parser.parse_args()
    FourJointContinuousRunner(args).run()


if __name__ == "__main__":
    main()
