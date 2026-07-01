#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
import time

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


SAFE_ACTIONS = {
    "release_arm": 99,
    "right_hand_up": 23,
    "wave_above_head": 26,
    "high_five": 18,
    "shake_hand": 27,
    "both_hands_up": 15,
}


def setup_cyclonedds(interface: str, local_ip: str):
    xml = f"""<CycloneDDS>
  <Domain id="any">
    <General>
      <Interfaces>
        <NetworkInterface address="{local_ip}" multicast="true" />
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
        prefix="cyclonedds_g1_arm_action_",
        delete=False,
    )
    f.write(xml)
    f.close()

    os.environ["CYCLONEDDS_URI"] = "file://" + f.name

    print("[DDS] interface =", interface)
    print("[DDS] local_ip  =", local_ip)
    print("[DDS] CYCLONEDDS_URI =", os.environ["CYCLONEDDS_URI"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--local-ip", default="192.168.123.100")

    parser.add_argument(
        "--motion-mode",
        default="ai",
        choices=["ai", "normal", "advanced"],
        help="MotionSwitcher 模式：ai / normal / advanced",
    )

    parser.add_argument(
        "--select-ai",
        default="YES",
        help="兼容旧参数：写 YES 时会执行 SelectMode(--motion-mode)",
    )

    parser.add_argument(
        "--action",
        default="release_arm",
        choices=sorted(SAFE_ACTIONS.keys()),
        help="要执行的高层手臂动作",
    )

    parser.add_argument(
        "--auto-release",
        default="YES",
        help="动作后是否自动 release_arm",
    )

    parser.add_argument(
        "--sleep-after-action",
        type=float,
        default=2.0,
        help="动作执行后等待多久再 release",
    )

    parser.add_argument(
        "--enable-execute",
        default="NO",
        help="必须写 YES 才真正执行动作",
    )

    args = parser.parse_args()

    setup_cyclonedds(args.interface, args.local_ip)
    ChannelFactoryInitialize(0, args.interface)

    print()
    print("[安全确认]")
    print("本脚本不发布 rt/lowcmd")
    print("本脚本不发布 rt/arm_sdk")
    print("本脚本只调用 G1ArmActionClient 高层动作服务")
    print()

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    try:
        code, data = msc.CheckMode()
        print("[MotionSwitcher] 当前模式 code =", code, "data =", data)
    except Exception as e:
        print("[MotionSwitcher] CheckMode 失败：", repr(e))

    if args.select_ai == "YES":
        print(f"[MotionSwitcher] SelectMode({args.motion_mode!r})")
        try:
            ret = msc.SelectMode(args.motion_mode)
            print("[MotionSwitcher] SelectMode ret =", ret)
            time.sleep(0.5)
            code, data = msc.CheckMode()
            print("[MotionSwitcher] SelectMode 后 code =", code, "data =", data)
        except Exception as e:
            print("[MotionSwitcher] SelectMode 失败：", repr(e))
            return

    client = G1ArmActionClient()
    client.SetTimeout(10.0)
    client.Init()

    print()
    print("[动作]")
    print("motion mode =", args.motion_mode)
    print("action name =", args.action)
    print("action id   =", SAFE_ACTIONS[args.action])
    print("execute     =", args.enable_execute)
    print()

    if args.enable_execute != "YES":
        print("[DRY-RUN] 未执行动作。")
        print("[DRY-RUN] 确认安全后加：--enable-execute YES")
        return

    action_id = SAFE_ACTIONS[args.action]

    print("[执行] ExecuteAction:", args.action, action_id)
    code = client.ExecuteAction(action_id)
    print("[结果] ExecuteAction code =", code)

    if args.action != "release_arm" and args.auto_release == "YES":
        print(f"[等待] {args.sleep_after_action:.2f} 秒后自动 release_arm")
        time.sleep(args.sleep_after_action)

        release_id = SAFE_ACTIONS["release_arm"]
        print("[释放] ExecuteAction: release_arm", release_id)
        code = client.ExecuteAction(release_id)
        print("[结果] release_arm code =", code)

    print("[完成]")


if __name__ == "__main__":
    main()
