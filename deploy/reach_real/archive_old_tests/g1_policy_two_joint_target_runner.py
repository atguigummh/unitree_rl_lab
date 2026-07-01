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

CONTROL_JOINTS = list(range(15, 29)) + [12, 13, 14]
NOT_USED_JOINT = 29


class TwoJointTargetRunner:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0
        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.last_action = np.zeros(4, dtype=np.float32)
        self.last_step_id = None
        self.cmd_target = None
        self.current_home_phase = False
        self.current_seq_id = 0

        if args.real_scale > 3.0:
            raise ValueError("real_scale 限制为 <= 3.0")
        if args.max_shoulder_delta > 0.35:
            raise ValueError("max_shoulder_delta 限制为 <= 0.35 rad")
        if args.max_elbow_delta > 0.55:
            raise ValueError("max_elbow_delta 限制为 <= 0.55 rad")
        if args.duration > 60.0:
            raise ValueError("duration 限制为 <= 60 s")

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
            prefix="cyclonedds_two_joint_target_",
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

    def dynamic_target(self, t: float):
        target = np.array([self.args.base_x, self.args.base_y, self.args.base_z], dtype=np.float32)

        self.current_home_phase = False

        if self.args.target_mode == "sine":
            value = self.args.amplitude * math.sin(2.0 * math.pi * t / self.args.period)

            if self.args.axis == "x":
                target[0] += value
            elif self.args.axis == "y":
                target[1] += value
            elif self.args.axis == "z":
                target[2] += value

        elif self.args.target_mode == "step":
            k = int(t // self.args.step_hold)
            value = self.args.amplitude if (k % 2 == 0) else -self.args.amplitude
            self.current_seq_id = k

            if self.args.axis == "x":
                target[0] += value
            elif self.args.axis == "y":
                target[1] += value
            elif self.args.axis == "z":
                target[2] += value

        elif self.args.target_mode == "sequence":
            seq = [
                np.array([ 0.00,  0.00,  0.20], dtype=np.float32),  # 上方
                np.array([ 0.10,  0.00,  0.12], dtype=np.float32),  # 前上
                np.array([ 0.00,  0.00, -0.15], dtype=np.float32),  # 下方
                np.array([-0.10,  0.00,  0.10], dtype=np.float32),  # 后上
                np.array([ 0.00,  0.10,  0.10], dtype=np.float32),  # 侧上
                np.array([ 0.00, -0.10,  0.05], dtype=np.float32),  # 另一侧
            ]

            if self.args.home_between_targets == "YES":
                # 每个目标持续 step_hold 秒，然后强制回初始 home_hold 秒
                cycle = self.args.step_hold + self.args.home_hold
                k = int(t // cycle)
                phase_t = t - k * cycle

                self.current_seq_id = k

                if phase_t >= self.args.step_hold:
                    self.current_home_phase = True
                    # home 阶段返回 base target，这里只是用于日志；真正回初始在主循环里强制 target=q_start
                    return target

                target += seq[k % len(seq)]

            else:
                k = int(t // self.args.step_hold)
                self.current_seq_id = k
                target += seq[k % len(seq)]

        else:
            raise ValueError("target_mode 只能是 sine / step / sequence")

        return target

    def make_obs(self, t: float):
        q = self.q_policy()
        dq = self.dq_policy()

        joint_pos_rel = q - DEFAULT_RIGHT_ARM_Q
        joint_vel_scaled = dq * 0.05
        wrist_to_target = self.dynamic_target(t)

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
        print("本脚本只控制 right_shoulder_pitch + right_elbow")
        print("其它关节保持启动时角度")
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

        shoulder_id = REAL_JOINT_IDS["right_shoulder_pitch_joint"]
        elbow_id = REAL_JOINT_IDS["right_elbow_joint"]

        shoulder_start = q_start[shoulder_id]
        elbow_start = q_start[elbow_id]

        print()
        print("[启动姿态]")
        print("right_shoulder_pitch q_start =", f"{shoulder_start:+.6f}")
        print("right_elbow          q_start =", f"{elbow_start:+.6f}")
        print("axis =", self.args.axis)
        print("amplitude =", self.args.amplitude)
        print("period =", self.args.period)
        print("real_scale =", self.args.real_scale)
        print("max_shoulder_delta =", self.args.max_shoulder_delta)
        print("max_elbow_delta =", self.args.max_elbow_delta)
        print("squash_temp =", self.args.squash_temp)
        print("max_joint_speed =", self.args.max_joint_speed)
        print("enable_write =", self.args.enable_write)
        print()

        dt = 1.0 / self.args.rate
        total_time = self.args.pre_hold + self.args.ramp_in + self.args.duration
        self.cmd_target = dict(q_start)

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
                    phase = "policy"
                    gain = 1.0
                    target_time = elapsed - self.args.pre_hold

                obs, q, dq, w2t = self.make_obs(target_time)
                action_raw = session.run([out.name], {inp.name: obs})[0].astype(np.float32)[0]

                # 不再硬 clip，而是 tanh 平滑压缩，保留 target 引起的 raw action 差异
                action_used = np.tanh(action_raw / self.args.squash_temp).astype(np.float32)

                # home_between_targets 模式下，home 阶段强制回启动姿态，不使用 policy 动作
                if self.current_home_phase:
                    action_used = np.zeros_like(action_used)

                shoulder_delta = float(action_used[0] * ACTION_SCALE[0] * self.args.real_scale * gain)
                elbow_delta = float(action_used[3] * ACTION_SCALE[3] * self.args.real_scale * gain)

                shoulder_delta = float(np.clip(shoulder_delta, -self.args.max_shoulder_delta, self.args.max_shoulder_delta))
                elbow_delta = float(np.clip(elbow_delta, -self.args.max_elbow_delta, self.args.max_elbow_delta))

                desired_target = dict(q_start)
                desired_target[shoulder_id] = shoulder_start + shoulder_delta
                desired_target[elbow_id] = elbow_start + elbow_delta

                # 目标刷新提示
                if self.args.target_mode in ("step", "sequence"):
                    if self.args.target_mode == "sequence":
                        step_id = (self.current_seq_id, self.current_home_phase)
                        step_name = "HOME" if self.current_home_phase else f"SEQ-{self.current_seq_id}"
                    else:
                        sid = int(target_time // self.args.step_hold)
                        step_id = (sid, False)
                        step_name = f"STEP-{sid}"

                    if step_id != self.last_step_id:
                        self.last_step_id = step_id
                        print()
                        print("=" * 80)
                        print(f"[TARGET SWITCH] {step_name}, w2t={np.round(w2t, 3)}")
                        print("=" * 80)

                # 关节目标限速，避免 target step 切换时猛落
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
                if count % int(max(1, self.args.rate / 5)) == 0:
                    actual_shoulder = self.q(shoulder_id)
                    actual_elbow = self.q(elbow_id)

                    print(
                        f"t={elapsed:05.2f}s "
                        f"phase={phase} "
                        f"gain={gain:.2f} "
                        f"w2t={np.round(w2t, 3)} "
                        f"raw={np.round(action_raw, 2)} "
                        f"used={np.round(action_used, 2)} "
                        f"d_sh={shoulder_delta:+.3f} "
                        f"d_el={elbow_delta:+.3f} "
                        f"sh_t={target[shoulder_id]:+.3f} "
                        f"sh_a={actual_shoulder:+.3f} "
                        f"el_t={target[elbow_id]:+.3f} "
                        f"el_a={actual_elbow:+.3f}"
                    )

                time.sleep(dt)

        finally:
            print("[回收] 慢慢回到启动姿态，再释放 arm_sdk")

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

            post_steps = max(1, int(self.args.post_hold * self.args.rate))
            print(f"[保持] 已回到启动姿态，继续保持 {self.args.post_hold:.1f}s 后释放")
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
    parser.add_argument(
        "--policy",
        default="/home/ma/unitree_clean/unitree_rl_lab/logs/rsl_rl/g1_right_wrist_reach_dynamic_005_v1/2026-06-25_09-24-34/exported/policy.onnx",
    )

    parser.add_argument("--axis", default="z", choices=["x", "y", "z"])
    parser.add_argument("--target-mode", default="sine", choices=["sine", "step", "sequence"])
    parser.add_argument("--amplitude", type=float, default=0.10)
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--step-hold", type=float, default=5.0)
    parser.add_argument("--home-between-targets", default="NO", choices=["YES", "NO"])
    parser.add_argument("--home-hold", type=float, default=5.0)
    parser.add_argument("--base-x", type=float, default=0.0)
    parser.add_argument("--base-y", type=float, default=0.0)
    parser.add_argument("--base-z", type=float, default=0.0)

    parser.add_argument("--pre-hold", type=float, default=2.0)
    parser.add_argument("--ramp-in", type=float, default=5.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--return-time", type=float, default=8.0)
    parser.add_argument("--post-hold", type=float, default=5.0)

    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--kp", type=float, default=90.0)
    parser.add_argument("--kd", type=float, default=2.5)

    parser.add_argument("--real-scale", type=float, default=2.0)
    parser.add_argument("--max-shoulder-delta", type=float, default=0.25)
    parser.add_argument("--max-elbow-delta", type=float, default=0.40)
    parser.add_argument("--squash-temp", type=float, default=3.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.10)

    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--enable-write", default="NO")

    args = parser.parse_args()
    TwoJointTargetRunner(args).run()


if __name__ == "__main__":
    main()
