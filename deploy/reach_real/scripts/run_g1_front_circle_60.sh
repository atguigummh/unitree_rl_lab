#!/usr/bin/env bash
# 作用：运行 G1 右臂前方小圆 60° 连续追踪展示版。
set -e

cd /home/ma/unitree_clean/unitree_rl_lab

export URDF=/home/ma/unitree_clean/g1_description/urdf/g1_29dof.urdf

python deploy/reach_real/g1_policy_fk_closed_loop_runner.py \
  --interface enp3s0 \
  --local-ip 192.168.123.100 \
  --urdf "$URDF" \
  --base-link torso_link \
  --wrist-link right_wrist_yaw_link \
  --path circle_yz \
  --target-x 0.15 \
  --target-y 0.0 \
  --target-z 0.0 \
  --period 20 \
  --radius-y 0.10 \
  --radius-z 0.08 \
  --action-base default \
  --real-scale 5.0 \
  --squash-temp 5.0 \
  --max-joint-speed 0.14 \
  --max-pitch-delta 1.05 \
  --max-roll-delta 1.05 \
  --max-yaw-delta 1.05 \
  --max-elbow-delta 1.05 \
  --pre-hold 2.0 \
  --ramp-in 14.0 \
  --duration 40 \
  --return-time 14 \
  --post-hold 5 \
  --kp 55 \
  --kd 8.0 \
  --pitch-gain 1.8 \
  --roll-gain 4.5 \
  --yaw-gain 4.5 \
  --elbow-gain -2.2 \
  --print-rate 2 \
  --enable-write YES
