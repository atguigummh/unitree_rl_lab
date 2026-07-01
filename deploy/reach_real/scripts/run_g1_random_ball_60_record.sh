#!/usr/bin/env bash
# 作用：运行 G1 右臂 60° 随机小球追踪实验，按固定时间切换目标并保存日志。
set -euo pipefail

cd /home/ma/unitree_clean/unitree_rl_lab

export URDF=/home/ma/unitree_clean/g1_description/urdf/g1_29dof.urdf

LOG_DIR=logs/real_g1
mkdir -p "$LOG_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/random_ball_60_${TS}.log"

echo "[开始] G1 随机小球 60° 追踪实验"
echo "[日志] $LOG_FILE"

python deploy/reach_real/g1_policy_fk_closed_loop_runner.py \
  --interface enp3s0 \
  --local-ip 192.168.123.100 \
  --urdf "$URDF" \
  --base-link torso_link \
  --wrist-link right_wrist_yaw_link \
  --path random_ball \
  --target-x 0.18 \
  --target-y 0.0 \
  --target-z 0.0 \
  --random-radius-x 0.04 \
  --random-radius-y 0.18 \
  --random-radius-z 0.12 \
  --random-hold 5.0 \
  --random-seed -1 \
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
  --duration 90 \
  --return-time 14 \
  --post-hold 5 \
  --kp 55 \
  --kd 8.0 \
  --pitch-gain 1.8 \
  --roll-gain 4.5 \
  --yaw-gain 4.5 \
  --elbow-gain -2.2 \
  --print-rate 2 \
  --enable-write YES | tee "$LOG_FILE"

echo "[完成] 原始日志已保存到: $LOG_FILE"
