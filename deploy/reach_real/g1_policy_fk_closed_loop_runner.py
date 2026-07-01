#!/usr/bin/env python3
# 作用：Unitree G1 右臂 policy.onnx + URDF FK 闭环真机部署主程序，支持 fixed/circle/live_file/random_ball 等目标模式。
from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from typing import Dict, List

import numpy as np
import onnxruntime as ort

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


REAL_JOINT_IDS = {
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,

    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}

POLICY_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

POLICY_REAL_IDS = [
    REAL_JOINT_IDS["right_shoulder_pitch_joint"],
    REAL_JOINT_IDS["right_shoulder_roll_joint"],
    REAL_JOINT_IDS["right_shoulder_yaw_joint"],
    REAL_JOINT_IDS["right_elbow_joint"],
]

ACTION_SCALE = np.array([0.18, 0.15, 0.15, 0.20], dtype=np.float32)
DEFAULT_RIGHT_ARM_Q = np.array([0.30, -0.25, 0.00, 0.97], dtype=np.float32)

DEFAULT_MAX_DELTA = np.array([0.30, 0.18, 0.20, 0.50], dtype=np.float32)

CONTROL_JOINTS = list(range(15, 29)) + [12, 13, 14]
NOT_USED_JOINT = 29


def rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def axis_angle_R(axis, q):
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return np.eye(3)
    x, y, z = axis / n
    c = math.cos(q)
    s = math.sin(q)
    C = 1.0 - c
    return np.array(
        [
            [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
        ],
        dtype=np.float64,
    )


def make_T(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_to_R(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def make_rot_T(axis, q):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = axis_angle_R(axis, q)
    return T


class URDFFK:
    def __init__(self, urdf_path: str, base_link: str, wrist_link: str):
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.wrist_link = wrist_link

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        self.joints = []
        self.parent_joint_by_child = {}

        for j in root.findall("joint"):
            name = j.attrib["name"]
            jtype = j.attrib.get("type", "fixed")
            parent = j.find("parent").attrib["link"]
            child = j.find("child").attrib["link"]

            origin = j.find("origin")
            if origin is not None:
                xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
            else:
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]

            axis_node = j.find("axis")
            if axis_node is not None:
                axis = [float(v) for v in axis_node.attrib.get("xyz", "1 0 0").split()]
            else:
                axis = [1.0, 0.0, 0.0]

            info = {
                "name": name,
                "type": jtype,
                "parent": parent,
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
            }
            self.joints.append(info)
            self.parent_joint_by_child[child] = info

        self.chain = self.find_chain(base_link, wrist_link)

        print("[URDF]", urdf_path)
        print("[FK] base_link =", base_link)
        print("[FK] wrist_link =", wrist_link)
        print("[FK] chain:")
        for jj in self.chain:
            print("  ", jj["name"], jj["type"], jj["parent"], "->", jj["child"])

    def find_chain(self, base_link, target_link):
        chain_rev = []
        link = target_link
        while link != base_link:
            if link not in self.parent_joint_by_child:
                raise RuntimeError(f"无法从 {target_link} 回溯到 {base_link}，卡在 link={link}")
            j = self.parent_joint_by_child[link]
            chain_rev.append(j)
            link = j["parent"]
        return list(reversed(chain_rev))

    def fk(self, q_by_name: Dict[str, float]):
        T = np.eye(4, dtype=np.float64)

        for j in self.chain:
            T = T @ make_T(j["xyz"], j["rpy"])

            if j["type"] in ("revolute", "continuous"):
                q = float(q_by_name.get(j["name"], 0.0))
                T = T @ make_rot_T(j["axis"], q)
            elif j["type"] == "prismatic":
                q = float(q_by_name.get(j["name"], 0.0))
                trans = np.eye(4, dtype=np.float64)
                trans[:3, 3] = np.asarray(j["axis"], dtype=np.float64) * q
                T = T @ trans

        return T


class FKClosedLoopRunner:
    def __init__(self, args):
        self.args = args
        self.low_state = None
        self.low_state_time = 0.0

        self.crc = CRC()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.last_action = np.zeros(4, dtype=np.float32)
        self.cmd_target = None
        self.wrist_start = None

        if args.real_scale > 6.0:
            raise ValueError("real_scale 限制为 <= 6.0")
        if args.duration > 90.0:
            raise ValueError("duration 限制为 <= 90s")
        if args.max_joint_speed > 0.45:
            raise ValueError("max_joint_speed 限制为 <= 0.45 rad/s")
        if args.err_clip > 0.35:
            raise ValueError("err_clip 限制为 <= 0.35 m")

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
            prefix="cyclonedds_fk_closed_loop_",
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
        return np.array([self.q(i) for i in POLICY_REAL_IDS], dtype=np.float32)

    def dq_policy(self):
        return np.array([self.dq(i) for i in POLICY_REAL_IDS], dtype=np.float32)

    def q_by_name(self):
        out = {}
        for name, idx in REAL_JOINT_IDS.items():
            out[name] = self.q(idx)
        return out

    def target_offset(self, t: float):
        # target_x/y/z 作为所有轨迹的中心偏移
        center = np.array([
            self.args.target_x,
            self.args.target_y,
            self.args.target_z,
        ], dtype=np.float32)

        if self.args.path == "random_ball":
            # random_ball 有两种切换方式：
            # 1) time: 每 random_hold 秒强制换目标；
            # 2) arrive: err_norm 小于阈值并保持一段时间后换目标，超时也会换。
            if not hasattr(self, "_random_rng"):
                seed = None if self.args.random_seed < 0 else self.args.random_seed
                self._random_rng = np.random.default_rng(seed)
                self._random_target = center.copy()
                self._random_next_t = -1.0
                self._random_target_id = -1
                self._random_target_start_t = -1.0
                self._arrive_start_t = None

            def sample_new_target(reason: str):
                radii = np.array([
                    self.args.random_radius_x,
                    self.args.random_radius_y,
                    self.args.random_radius_z,
                ], dtype=np.float32)

                v = np.zeros(3, dtype=np.float32)
                for _ in range(1000):
                    v = self._random_rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
                    if float(np.linalg.norm(v)) <= 1.0:
                        break

                self._random_target = center + v * radii
                self._random_target_id += 1
                self._random_target_start_t = t
                self._arrive_start_t = None

                hold = max(1.0, float(self.args.random_hold))
                self._random_next_t = t + hold

                print(
                    f"[random_ball #{self._random_target_id}] "
                    f"t={t:.1f}s reason={reason} "
                    f"target_offset={np.round(self._random_target, 3)}"
                )

            last_err = getattr(self, "_last_err_norm", None)

            if self._random_target_id < 0:
                sample_new_target("init")

            elif self.args.random_switch_mode == "time":
                if t >= self._random_next_t:
                    sample_new_target("time")

            elif self.args.random_switch_mode == "arrive":
                age = t - self._random_target_start_t

                if age >= self.args.min_target_hold:
                    if last_err is not None and last_err <= self.args.arrive_threshold:
                        if self._arrive_start_t is None:
                            self._arrive_start_t = t
                            print(
                                f"[arrive #{self._random_target_id}] "
                                f"t={t:.1f}s err={last_err:.3f}, hold..."
                            )

                        if t - self._arrive_start_t >= self.args.arrive_hold:
                            sample_new_target("arrived")
                    else:
                        self._arrive_start_t = None

                if age >= self.args.max_target_hold:
                    sample_new_target("timeout")

            else:
                raise ValueError("random_switch_mode 只能是 time 或 arrive")

            return self._random_target

        if self.args.path == "live_file":
            try:
                txt = Path(self.args.target_file).read_text(encoding="utf-8").strip()
                vals = [float(x) for x in txt.replace(",", " ").split()]
                if len(vals) >= 3:
                    return np.array(vals[:3], dtype=np.float32)
            except Exception:
                pass
            return center

        if self.args.path == "fixed":
            return center

        if self.args.path == "circle_yz":
            return center + np.array([
                0.0,
                self.args.radius_y * math.sin(2.0 * math.pi * t / self.args.period),
                self.args.radius_z * math.cos(2.0 * math.pi * t / self.args.period),
            ], dtype=np.float32)

        if self.args.path == "circle_xz":
            return center + np.array([
                self.args.radius_x * math.sin(2.0 * math.pi * t / self.args.period),
                0.0,
                self.args.radius_z * math.cos(2.0 * math.pi * t / self.args.period),
            ], dtype=np.float32)

        if self.args.path == "eight_yz":
            return center + np.array([
                0.0,
                self.args.radius_y * math.sin(2.0 * math.pi * t / self.args.period),
                self.args.radius_z * math.sin(4.0 * math.pi * t / self.args.period),
            ], dtype=np.float32)

        if self.args.path == "up_down":
            return center + np.array([
                0.0,
                0.0,
                self.args.radius_z * math.sin(2.0 * math.pi * t / self.args.period),
            ], dtype=np.float32)

        raise ValueError("path 不支持")

    def make_obs(self, fk: URDFFK, t: float):
        q = self.q_policy()
        dq = self.dq_policy()

        wrist_T = fk.fk(self.q_by_name())
        wrist_pos = wrist_T[:3, 3].astype(np.float32)

        if self.wrist_start is None:
            self.wrist_start = wrist_pos.copy()
            print("[FK] wrist_start =", np.round(self.wrist_start, 4))

        target_pos = self.wrist_start + self.target_offset(t)
        wrist_to_target = target_pos - wrist_pos
        wrist_to_target = np.clip(wrist_to_target, -self.args.err_clip, self.args.err_clip)

        joint_pos_rel = q - DEFAULT_RIGHT_ARM_Q
        joint_vel_scaled = dq * 0.05

        obs = np.concatenate(
            [joint_pos_rel, joint_vel_scaled, wrist_to_target, self.last_action],
            axis=0,
        ).astype(np.float32)

        return obs[None, :], q, dq, wrist_pos, target_pos, wrist_to_target

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
        print("本脚本使用 URDF FK 计算真实 wrist_pos")
        print("wrist_to_target = target_pos - wrist_pos")
        print()

        fk = URDFFK(self.args.urdf, self.args.base_link, self.args.wrist_link)

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
        self.cmd_target = dict(q_start)

        print()
        print("[启动姿态]")
        for name in POLICY_JOINTS:
            print(f"{name:28s} q_start = {self.q(REAL_JOINT_IDS[name]):+.6f}")

        print()
        print("[闭环目标]")
        print("path =", self.args.path)
        print("period =", self.args.period)
        print("radius_x =", self.args.radius_x)
        print("radius_y =", self.args.radius_y)
        print("radius_z =", self.args.radius_z)
        print("real_scale =", self.args.real_scale)
        print("action_base =", self.args.action_base)
        print("squash_temp =", self.args.squash_temp)
        print("max_joint_speed =", self.args.max_joint_speed)
        print("enable_write =", self.args.enable_write)
        print()

        dt = 1.0 / self.args.rate
        total_time = self.args.pre_hold + self.args.ramp_in + self.args.duration
        t0 = time.monotonic()
        count = 0

        joint_gain = np.array(
            [self.args.pitch_gain, self.args.roll_gain, self.args.yaw_gain, self.args.elbow_gain],
            dtype=np.float32,
        )

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

                obs, q, dq, wrist_pos, target_pos, err = self.make_obs(fk, target_time)
                action_raw = session.run([out.name], {inp.name: obs})[0].astype(np.float32)[0]

                action_used = np.tanh(action_raw / self.args.squash_temp).astype(np.float32)

                delta = action_used * ACTION_SCALE * self.args.real_scale * joint_gain * gain

                max_delta = np.array(
                    [
                        self.args.max_pitch_delta,
                        self.args.max_roll_delta,
                        self.args.max_yaw_delta,
                        self.args.max_elbow_delta,
                    ],
                    dtype=np.float32,
                )
                delta = np.clip(delta, -max_delta, max_delta)

                desired_target = dict(q_start)

                for i, jid in enumerate(POLICY_REAL_IDS):
                    if phase == "pre_hold":
                        # pre_hold 阶段必须真正保持启动姿态，不能提前往 default 走
                        desired_target[jid] = q_start[jid]
                        continue

                    if self.args.action_base == "startup":
                        raw_desired = q_start[jid] + float(delta[i])

                    elif self.args.action_base == "default":
                        # default 模式下，ramp_in 阶段从启动姿态平滑过渡到 default+delta
                        base_q = float(DEFAULT_RIGHT_ARM_Q[i])
                        raw_desired = base_q + float(delta[i])
                        raw_desired = q_start[jid] + gain * (raw_desired - q_start[jid])

                    else:
                        raise ValueError("action_base 只能是 startup 或 default")

                    desired_target[jid] = raw_desired

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
                    actual = np.array([self.q(jid) for jid in POLICY_REAL_IDS], dtype=np.float32)
                    target_vec = np.array([target[jid] for jid in POLICY_REAL_IDS], dtype=np.float32)
                    err_norm = float(np.linalg.norm(err))
                    self._last_err_norm = err_norm

                    wrist_move = wrist_pos - self.wrist_start

                    print(
                        f"t={elapsed:05.2f}s "
                        f"{phase:9s} "
                        f"wrist={np.round(wrist_pos, 3)} "
                        f"move={np.round(wrist_move, 3)} "
                        f"target={np.round(target_pos, 3)} "
                        f"err={np.round(err, 3)} "
                        f"|err|={err_norm:.3f} "
                        f"used={np.round(action_used, 2)} "
                        f"delta={np.round(delta, 3)} "
                        f"q_t={np.round(target_vec, 3)} "
                        f"q_a={np.round(actual, 3)}"
                    )

                time.sleep(dt)

        finally:
            print("[回收] FK 闭环结束，慢慢回到启动姿态，再释放 arm_sdk")

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

    parser.add_argument("--urdf", required=True)
    parser.add_argument("--base-link", default="torso_link")
    parser.add_argument("--wrist-link", default="right_wrist_yaw_link")

    parser.add_argument(
        "--policy",
        default="/home/ma/unitree_clean/unitree_rl_lab/logs/rsl_rl/g1_right_wrist_reach_dynamic_005_v1/2026-06-25_09-24-34/exported/policy.onnx",
    )

    parser.add_argument("--path", default="circle_yz", choices=["random_ball", "fixed", "live_file", "circle_yz", "circle_xz", "eight_yz", "up_down"])
    parser.add_argument("--period", type=float, default=12.0)

    parser.add_argument("--radius-x", type=float, default=0.06)
    parser.add_argument("--radius-y", type=float, default=0.08)
    parser.add_argument("--radius-z", type=float, default=0.08)

    parser.add_argument("--target-x", type=float, default=0.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.08)
    parser.add_argument("--target-file", default="/tmp/g1_target.txt")
    parser.add_argument("--random-hold", type=float, default=4.0)
    parser.add_argument("--random-radius-x", type=float, default=0.04)
    parser.add_argument("--random-radius-y", type=float, default=0.12)
    parser.add_argument("--random-radius-z", type=float, default=0.09)
    parser.add_argument("--random-seed", type=int, default=-1)
    parser.add_argument("--random-switch-mode", default="time", choices=["time", "arrive"])
    parser.add_argument("--arrive-threshold", type=float, default=0.16)
    parser.add_argument("--arrive-hold", type=float, default=1.5)
    parser.add_argument("--min-target-hold", type=float, default=2.0)
    parser.add_argument("--max-target-hold", type=float, default=8.0)

    parser.add_argument("--pre-hold", type=float, default=2.0)
    parser.add_argument("--ramp-in", type=float, default=6.0)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--return-time", type=float, default=12.0)
    parser.add_argument("--post-hold", type=float, default=5.0)

    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--print-rate", type=float, default=5.0)

    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--kd", type=float, default=3.0)

    parser.add_argument("--real-scale", type=float, default=2.2)
    parser.add_argument("--action-base", default="startup", choices=["startup", "default"])
    parser.add_argument("--squash-temp", type=float, default=6.0)
    parser.add_argument("--max-joint-speed", type=float, default=0.12)
    parser.add_argument("--err-clip", type=float, default=0.25)

    parser.add_argument("--pitch-gain", type=float, default=0.4)
    parser.add_argument("--roll-gain", type=float, default=2.5)
    parser.add_argument("--yaw-gain", type=float, default=2.5)
    parser.add_argument("--elbow-gain", type=float, default=0.7)

    # 四个 policy 关节的最大偏移，单位 rad
    # 1.05 rad 约等于 60 度
    parser.add_argument("--max-pitch-delta", type=float, default=0.30)
    parser.add_argument("--max-roll-delta", type=float, default=0.18)
    parser.add_argument("--max-yaw-delta", type=float, default=0.20)
    parser.add_argument("--max-elbow-delta", type=float, default=0.50)

    parser.add_argument("--state-timeout", type=float, default=30.0)
    parser.add_argument("--enable-write", default="NO")

    args = parser.parse_args()
    FKClosedLoopRunner(args).run()


if __name__ == "__main__":
    main()
