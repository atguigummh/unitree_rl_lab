#!/usr/bin/env python3

"""
G1右臂状态只读入口。

作用：
1. 将CycloneDDS明确绑定到192.168.123.100；
2. 调用原有read_g1_right_arm.py；
3. 不创建Publisher，不发送任何机器人控制命令。
"""

from pathlib import Path
import runpy

import unitree_sdk2py.core.channel as channel


BIND_IP = "192.168.123.100"

# 覆盖Unitree SDK默认的“只按网卡名称选择地址”配置。
# 避免enp3s0同时存在两个IPv4时选中172.29.45.144。
channel.ChannelConfigHasInterface = f"""
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface
          address="{BIND_IP}"
          multicast="true"
        />
      </Interfaces>
      <AllowMulticast>true</AllowMulticast>
    </General>
  </Domain>
</CycloneDDS>
"""

target_script = Path(__file__).with_name("read_g1_right_arm.py")

if not target_script.is_file():
    raise FileNotFoundError(
        f"找不到原始只读脚本：{target_script}"
    )

print(f"[DDS绑定地址] {BIND_IP}")

# 在当前进程中运行原有只读脚本。
runpy.run_path(str(target_script), run_name="__main__")
