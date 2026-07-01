#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
import time
import numpy as np
import onnxruntime as ort

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


JOINT_IDS = {
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

# 来自 deploy.yaml/default_joint_pos，按 POLICY_JOINTS 顺序：
# right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
DEFAULT_RIGHT_ARM_Q = np.array([0.30, -0.25, 0.00, 0.97], dtype=np.float32)


class PolicyDryRunV2:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0
        self.q0 = None
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
            prefix="cyclonedds_policy_dryrun_v2_",
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

    def q_vec(self):
        return np.array(
            [float(self.low_state.motor_state[JOINT_IDS[name]].q) for name in POLICY_JOINTS],
            dtype=np.float32,
        )

    def dq_vec(self):
        return np.array(
            [float(self.low_state.motor_state[JOINT_IDS[name]].dq) for name in POLICY_JOINTS],
            dtype=np.float32,
        )

    def make_obs(self):
        q = self.q_vec()
        dq = self.dq_vec()

        if self.q0 is None:
            self.q0 = q.copy()
            print("[初始化] startup q0 =", np.round(self.q0, 4))
            print("[训练默认] DEFAULT_RIGHT_ARM_Q =", np.round(DEFAULT_RIGHT_ARM_Q, 4))

        # 训练里是 joint_pos_rel：当前关节角 - 训练默认关节角
        joint_pos_rel = q - DEFAULT_RIGHT_ARM_Q

        # 训练配置里 joint_vel scale = 0.05
        joint_vel_scaled = dq * 0.05

        # 真机暂时没有视觉目标/腕部 FK，先手动给 wrist_to_target
        wrist_to_target = np.array(
            [self.args.target_x, self.args.target_y, self.args.target_z],
            dtype=np.float32,
        )

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

        print("[ONNX] input name =", inp.name)
        print("[ONNX] input shape =", inp.shape)
        print("[ONNX] output name =", out.name)
        print("[ONNX] output shape =", out.shape)

        print()
        print("[安全模式]")
        print("本脚本只订阅 rt/lowstate")
        print("本脚本只运行 policy.onnx")
        print("本脚本不发布 rt/arm_sdk")
        print("本脚本不发布 rt/lowcmd")
        print()

        t0 = time.monotonic()
        count = 0

        while time.monotonic() - t0 < self.args.duration:
            if time.monotonic() - self.low_state_time > 0.2:
                print("[警告] lowstate 超时")
                time.sleep(0.1)
                continue

            obs, q, dq, qrel, dq_scaled, w2t = self.make_obs()
            action = session.run([out.name], {inp.name: obs})[0].astype(np.float32)[0]
            action_clip = np.clip(action, -1.0, 1.0)
            delta_rad = action_clip * ACTION_SCALE

            self.last_action = action_clip.copy()

            count += 1
            print("=" * 72)
            print(f"[{count}] q                 =", np.round(q, 4))
            print(f"[{count}] dq                =", np.round(dq, 4))
            print(f"[{count}] joint_pos_rel     =", np.round(qrel, 4))
            print(f"[{count}] joint_vel_scaled  =", np.round(dq_scaled, 4))
            print(f"[{count}] wrist_to_target   =", np.round(w2t, 4))
            print(f"[{count}] obs shape         =", obs.shape)
            print(f"[{count}] action raw        =", np.round(action, 6))
            print(f"[{count}] action clipped    =", np.round(action_clip, 6))
            print(f"[{count}] delta_rad by scale=", np.round(delta_rad, 6))
            print(f"[{count}] max|raw action|   =", float(np.max(np.abs(action))))

            time.sleep(1.0 / self.args.rate)

        print("[完成] dry-run v2 结束，没有控制机器人。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--local-ip", default="192.168.123.100")
    parser.add_argument(
        "--policy",
        default="/home/ma/unitree_clean/unitree_rl_lab/logs/rsl_rl/g1_right_wrist_reach_dynamic_005_v1/2026-06-25_09-24-34/exported/policy.onnx",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--state-timeout", type=float, default=30.0)

    # 手动给一个目标误差，先不要太大
    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.0)

    args = parser.parse_args()
    PolicyDryRunV2(args).run()


if __name__ == "__main__":
    main()
