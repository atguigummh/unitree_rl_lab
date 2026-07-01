#!/usr/bin/env python3
# 作用：检查 motion switcher 状态及 arm_sdk release/接管相关信息。
from __future__ import annotations

import argparse
import os
import tempfile

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient


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
        prefix="cyclonedds_motion_switcher_",
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
    parser.add_argument("--release", default="NO", help="写 YES 才执行 ReleaseMode")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    setup_cyclonedds(args.interface, args.local_ip)

    ChannelFactoryInitialize(0, args.interface)

    msc = MotionSwitcherClient()
    msc.SetTimeout(args.timeout)
    msc.Init()

    print("\n[1] CheckMode before ReleaseMode")
    try:
        code, data = msc.CheckMode()
        print("code =", code)
        print("data =", data)
    except Exception as e:
        print("[ERROR] CheckMode failed:", repr(e))
        return

    if args.release != "YES":
        print("\n[DRY-RUN] 未执行 ReleaseMode。")
        print("[DRY-RUN] 如需释放当前运动模式，请加：--release YES")
        return

    print("\n[2] Execute ReleaseMode")
    try:
        ret = msc.ReleaseMode()
        print("ReleaseMode ret =", ret)
    except Exception as e:
        print("[ERROR] ReleaseMode failed:", repr(e))
        return

    print("\n[3] CheckMode after ReleaseMode")
    try:
        code, data = msc.CheckMode()
        print("code =", code)
        print("data =", data)
    except Exception as e:
        print("[ERROR] CheckMode after release failed:", repr(e))


if __name__ == "__main__":
    main()
