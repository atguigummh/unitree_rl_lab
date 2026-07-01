#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
import time
import math
import numpy as np
import onnxruntime as ort

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


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


class TargetSweepDryRun:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0
        self.last_action = np.zeros(4, dtype=np.float32)

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
            prefix="cyclonedds_target_sweep_",
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

    def q_policy(self):
        return np.array(
            [float(self.low_state.motor_state[REAL_JOINT_IDS[name]].q) for name in POLICY_JOINTS],
            dtype=np.float32,
        )

    def dq_policy(self):
        return np.array(
            [float(self.low_state.motor_state[REAL_JOINT_IDS[name]].dq) for name in POLICY_JOINTS],
            dtype=np.float32,
        )

    def dynamic_target(self, t):
        value = self.args.amplitude * math.sin(2.0 * math.pi * t / self.args.period)
        target = np.array([self.args.base_x, self.args.base_y, self.args.base_z], dtype=np.float32)

        if self.args.axis == "x":
            target[0] += value
        elif self.args.axis == "y":
            target[1] += value
        elif self.args.axis == "z":
            target[2] += value
        else:
            raise ValueError("axis 只能是 x / y / z")

        return target

    def make_obs(self, t):
        q = self.q_policy()
        dq = self.dq_policy()

        joint_pos_rel = q - DEFAULT_RIGHT_ARM_Q
        joint_vel_scaled = dq * 0.05
        wrist_to_target = self.dynamic_target(t)

        obs = np.concatenate(
            [
                joint_pos_rel,
                joint_vel_scaled,
                wrist_to_target,
                self.last_action,
            ],
            axis=0,
        ).astype(np.float32)

        return obs[None, :], q, dq, joint_pos_rel, joint_vel_scaled, wrist_to_target

    def run(self):
        self.setup_dds()
        ChannelFactoryInitialize(0, self.args.interface)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.cb, 10)
        self.wait_lowstate()

        print("[ONNX] 加载 policy:", self.args.policy)
        session = ort.InferenceSession(self.args.policy, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]

        print("[ONNX] input =", inp.name, inp.shape)
        print("[ONNX] output =", out.name, out.shape)

        print()
        print("[安全模式]")
        print("本脚本只订阅 rt/lowstate")
        print("本脚本只运行 policy.onnx")
        print("本脚本不发布 rt/arm_sdk")
        print("本脚本不发布 rt/lowcmd")
        print()
        print("[Sweep]")
        print("axis =", self.args.axis)
        print("amplitude =", self.args.amplitude)
        print("period =", self.args.period)
        print()

        t0 = time.monotonic()
        count = 0

        while time.monotonic() - t0 < self.args.duration:
            t = time.monotonic() - t0

            if time.monotonic() - self.low_state_time > 0.2:
                print("[警告] lowstate 超时")
                time.sleep(0.1)
                continue

            obs, q, dq, qrel, dq_scaled, target = self.make_obs(t)
            action_raw = session.run([out.name], {inp.name: obs})[0].astype(np.float32)[0]
            action_clip = np.clip(action_raw, -1.0, 1.0)
            delta_rad = action_clip * ACTION_SCALE

            self.last_action = action_clip.copy()

            count += 1
            print(
                f"t={t:05.2f}s "
                f"target={np.round(target, 4)} "
                f"raw={np.round(action_raw, 3)} "
                f"clip={np.round(action_clip, 3)} "
                f"delta={np.round(delta_rad, 3)} "
                f"max_raw={float(np.max(np.abs(action_raw))):.3f}"
            )

            time.sleep(1.0 / self.args.rate)

        print("[完成] 动态目标 dry-run 结束，没有控制机器人。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--local-ip", default="192.168.123.100")
    parser.add_argument(
        "--policy",
        default="/home/ma/unitree_clean/unitree_rl_lab/logs/rsl_rl/g1_right_wrist_reach_dynamic_005_v1/2026-06-25_09-24-34/exported/policy.onnx",
    )
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=2.0)
    parser.add_argument("--state-timeout", type=float, default=30.0)

    parser.add_argument("--axis", default="x", choices=["x", "y", "z"])
    parser.add_argument("--amplitude", type=float, default=0.10)
    parser.add_argument("--period", type=float, default=10.0)

    parser.add_argument("--base-x", type=float, default=0.0)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--base-z", type=float, default=0.0)

    args = parser.parse_args()
    TargetSweepDryRun(args).run()


if __name__ == "__main__":
    main()
