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


RIGHT_ARM = {
    "right_shoulder_pitch": 22,
    "right_shoulder_roll": 23,
    "right_shoulder_yaw": 24,
    "right_elbow": 25,
    "right_wrist_roll": 26,
    "right_wrist_pitch": 27,
    "right_wrist_yaw": 28,
}


class PolicyDryRun:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0

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
            prefix="cyclonedds_policy_dryrun_",
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

    def get_q(self, idx):
        return float(self.low_state.motor_state[idx].q)

    def get_dq(self, idx):
        return float(self.low_state.motor_state[idx].dq)

    def make_temp_obs(self, obs_dim: int):
        """
        临时 obs：
        现在还不知道训练时完整 obs 顺序，所以这里只做安全 dry-run。
        先构造全 0 obs，然后把右臂 q/dq 放到前面。
        这个 obs 不能直接用于真机控制，只用于看 policy 输出是否异常。
        """
        obs = np.zeros((1, obs_dim), dtype=np.float32)

        q = [
            self.get_q(RIGHT_ARM["right_shoulder_pitch"]),
            self.get_q(RIGHT_ARM["right_shoulder_roll"]),
            self.get_q(RIGHT_ARM["right_shoulder_yaw"]),
            self.get_q(RIGHT_ARM["right_elbow"]),
        ]

        dq = [
            self.get_dq(RIGHT_ARM["right_shoulder_pitch"]),
            self.get_dq(RIGHT_ARM["right_shoulder_roll"]),
            self.get_dq(RIGHT_ARM["right_shoulder_yaw"]),
            self.get_dq(RIGHT_ARM["right_elbow"]),
        ]

        values = q + dq
        n = min(len(values), obs_dim)
        obs[0, :n] = np.asarray(values[:n], dtype=np.float32)

        return obs, q, dq

    def run(self):
        self.setup_dds()
        ChannelFactoryInitialize(0, self.args.interface)

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.cb, 10)

        self.wait_lowstate()

        print("[ONNX] 加载 policy:", self.args.policy)
        session = ort.InferenceSession(
            self.args.policy,
            providers=["CPUExecutionProvider"],
        )

        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]

        print("[ONNX] input name =", inp.name)
        print("[ONNX] input shape =", inp.shape)
        print("[ONNX] output name =", out.name)
        print("[ONNX] output shape =", out.shape)

        shape = inp.shape
        if len(shape) != 2:
            raise RuntimeError(f"暂不支持这个 input shape: {shape}")

        obs_dim = shape[1]
        if not isinstance(obs_dim, int):
            raise RuntimeError(f"obs_dim 不是固定整数，请检查 ONNX input shape: {shape}")

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

            obs, q, dq = self.make_temp_obs(obs_dim)
            action = session.run([out.name], {inp.name: obs})[0]

            count += 1
            print("=" * 72)
            print(f"[{count}] right_arm q  =", np.round(q, 4))
            print(f"[{count}] right_arm dq =", np.round(dq, 4))
            print(f"[{count}] action shape =", action.shape)
            print(f"[{count}] action      =", np.round(action, 6))
            print(f"[{count}] max|action| =", float(np.max(np.abs(action))))

            time.sleep(1.0 / self.args.rate)

        print("[完成] dry-run 结束，没有控制机器人。")


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
    args = parser.parse_args()

    PolicyDryRun(args).run()


if __name__ == "__main__":
    main()
