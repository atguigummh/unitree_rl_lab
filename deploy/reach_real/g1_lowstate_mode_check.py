#!/usr/bin/env python3
# 作用：检查 Unitree G1 lowstate 数据和当前机器人模式状态。
import argparse
import os
import tempfile
import time

import unitree_sdk2py.core.channel as channel_mod
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_


latest = None


def cb(msg):
    global latest
    latest = msg


def setup(interface, local_ip):
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

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", prefix="cyclonedds_lowstate_mode_", delete=False)
    f.write(xml)
    f.close()
    os.environ["CYCLONEDDS_URI"] = "file://" + f.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="enp3s0")
    parser.add_argument("--local-ip", default="192.168.123.100")
    args = parser.parse_args()

    setup(args.interface, args.local_ip)
    ChannelFactoryInitialize(0, args.interface)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(cb, 10)

    print("[等待] rt/lowstate ...")
    t0 = time.time()
    while latest is None and time.time() - t0 < 5:
        time.sleep(0.01)

    if latest is None:
        print("[失败] 没收到 lowstate")
        return

    print("[收到] lowstate")
    print()

    for name in dir(latest):
        if name.startswith("_"):
            continue
        lname = name.lower()
        if "mode" in lname or "fsm" in lname or "state" in lname or "machine" in lname:
            try:
                print(f"{name} = {getattr(latest, name)}")
            except Exception as e:
                print(f"{name} = <读取失败 {e}>")

    print()
    print("[常见字段尝试]")
    for name in ["mode_pr", "mode_machine", "tick", "wireless_remote"]:
        if hasattr(latest, name):
            print(f"{name} = {getattr(latest, name)}")


if __name__ == "__main__":
    main()
