#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
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
        prefix="cyclonedds_g1_arm_action_list_",
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
    parser.add_argument("--select-ai", default="YES")
    args = parser.parse_args()

    setup_cyclonedds(args.interface, args.local_ip)
    ChannelFactoryInitialize(0, args.interface)

    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()

    print("\n[MotionSwitcher] before")
    code, data = msc.CheckMode()
    print("code =", code)
    print("data =", data)

    if args.select_ai == "YES":
        print("\n[MotionSwitcher] SelectMode('ai')")
        print("ret =", msc.SelectMode("ai"))

        code, data = msc.CheckMode()
        print("after code =", code)
        print("after data =", data)

    client = G1ArmActionClient()
    client.SetTimeout(10.0)
    client.Init()

    print("\n[ArmAction] GetActionList")
    ret = client.GetActionList()
    print("ret =", ret)


if __name__ == "__main__":
    main()
