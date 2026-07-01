#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


# 真机 lowstate / arm_sdk 关节编号
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

# deploy.yaml 中的 action scale
ACTION_SCALE = np.array([0.18, 0.15, 0.15, 0.20], dtype=np.float32)

# deploy.yaml/default_joint_pos 中对应右臂四关节的默认角
DEFAULT_RIGHT_ARM_Q = np.array([0.30, -0.25, 0.00, 0.97], dtype=np.float32)

# arm_sdk 控制时一起保持的关节：双臂 + 腰
CONTROL_JOINTS = list(range(15, 29)) + [12, 13, 14]
NOT_USED_JOINT = 29


class PolicyElbowSafeRunner:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.last_action = np.zeros(4, dtype=np.float32)

        if args.real_scale > 3.0:
            raise ValueError("real_scale 目前限制为 <= 3.0。")
        if args.max_delta > 0.55:
            raise ValueError("max_delta 目前限制为 <= 0.55 rad，约31.5度。")
        if args.duration > 20.0:
            raise ValueError("duration 目前限制为 <= 20 s。")

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
            prefix="cyclonedds_policy_elbow_safe_",
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
        return np.array(
            [self.q(REAL_JOINT_IDS[name]) for name in POLICY_JOINTS],
            dtype=np.float32,
        )

    def dq_policy(self):
        return np.array(
            [self.dq(REAL_JOINT_IDS[name]) for name in POLICY_JOINTS],
            dtype=np.float32,
        )

    def make_obs(self):
        q = self.q_policy()
        dq = self.dq_policy()

        joint_pos_rel = q - DEFAULT_RIGHT_ARM_Q
        joint_vel_scaled = dq * 0.05
        wrist_to_target = np.array(
            [self.args.target_x, self.args.target_y, self.args.target_z],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [joint_pos_rel, joint_vel_scaled, wrist_to_target, self.last_action],
            axis=0,
        ).astype(np.float32)

        return obs[None, :], q, dq, joint_pos_rel, joint_vel_scaled, wrist_to_target

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
        print("本脚本只允许 policy 控制 right_elbow 一个关节")
        print("其它手臂/腰部关节保持启动时角度")
        print()

        pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        pub.Init()
        print("[发布] rt/arm_sdk")

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
        elbow_id = REAL_JOINT_IDS["right_elbow_joint"]
        elbow_start = q_start[elbow_id]

        print()
        print("[启动姿态]")
        print("right_elbow q_start =", f"{elbow_start:+.6f}")
        print("real_scale =", self.args.real_scale)
        print("max_delta  =", self.args.max_delta)
        print("enable_write =", self.args.enable_write)
        print()

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
                    command_gain = 0.0
                elif elapsed < self.args.pre_hold + self.args.ramp_in:
                    phase = "ramp_in"
                    command_gain = (elapsed - self.args.pre_hold) / self.args.ramp_in
                    command_gain = max(0.0, min(1.0, command_gain))
                else:
                    phase = "policy"
                    command_gain = 1.0

                obs, q, dq, qrel, dq_scaled, w2t = self.make_obs()
                action_raw = session.run([out.name], {inp.name: obs})[0].astype(np.float32)[0]
                action_clip = np.clip(action_raw, -1.0, 1.0)

                # 只用第4维控制右肘
                elbow_action = float(action_clip[3])
                elbow_delta = elbow_action * float(ACTION_SCALE[3]) * self.args.real_scale * command_gain
                elbow_delta = float(np.clip(elbow_delta, -self.args.max_delta, self.args.max_delta))

                target = dict(q_start)
                target[elbow_id] = elbow_start + elbow_delta

                if self.args.enable_write == "YES":
                    self.write_cmd(pub, target, enable=1.0)

                self.last_action = action_clip.copy()

                count += 1
                if count % int(max(1, self.args.rate / 5)) == 0:
                    actual_elbow = self.q(elbow_id)
                    print(
                        f"t={elapsed:.3f}s "
                        f"phase={phase} "
                        f"gain={command_gain:.3f} "
                        f"raw={np.round(action_raw, 3)} "
                        f"clip={np.round(action_clip, 3)} "
                        f"elbow_delta={elbow_delta:+.5f} "
                        f"target={target[elbow_id]:+.5f} "
                        f"actual={actual_elbow:+.5f} "
                        f"err={target[elbow_id]-actual_elbow:+.5f}"
                    )

                time.sleep(dt)

        finally:
            print("[回收] 慢慢回到启动姿态，再释放 arm_sdk")

            # 先读取当前目标附近，然后缓慢回到启动姿态
            current_target = {j: self.q(j) for j in CONTROL_JOINTS}
            return_steps = max(1, int(self.args.return_time * self.args.rate))

            for k in range(return_steps):
                ratio = (k + 1) / return_steps
                target = {}
                for j in CONTROL_JOINTS:
                    target[j] = current_target[j] + ratio * (q_start[j] - current_target[j])
                if self.args.enable_write == "YES":
                    self.write_cmd(pub, target, enable=1.0)
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
    parser.add_argument(
        "--policy",
        default="/home/ma/unitree_clean/unitree_rl_lab/logs/rsl_rl/g1_right_wrist_reach_dynamic_005_v1/2026-06-25_09-24-34/exported/policy.onnx",
    )

    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.0)

    parser.add_argument("--pre-hold", type=float, default=2.0)
    parser.add_argument("--ramp-in", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--return-time", type=float, default=5.0)

    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--kp", type=float, default=90.0)
    parser.add_argument("--kd", type=float, default=2.5)

    parser.add_argument("--real-scale", type=float, default=0.05)
    parser.add_argument("--max-delta", type=float, default=0.02)

    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--enable-write", default="NO")

    args = parser.parse_args()
    PolicyElbowSafeRunner(args).run()


if __name__ == "__main__":
    main()
