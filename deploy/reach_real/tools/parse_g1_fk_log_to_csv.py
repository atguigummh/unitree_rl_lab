#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 作用：解析 G1 FK 闭环运行日志，将 target/wrist/move/err/action/joint 数据转换为 CSV。

import argparse
import csv
import re
from pathlib import Path


def parse_vec(text):
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    return [float(x) for x in nums]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if args.out is None:
        out_path = log_path.with_suffix(".csv")
    else:
        out_path = Path(args.out)

    rows = []
    current_target_offset = None
    target_id = -1

    random_re = re.compile(
        r"\[random_ball\]\s+t=([0-9.]+)s\s+target_offset=\[([^\]]+)\]"
    )

    line_re = re.compile(
        r"t=([0-9.]+)s\s+(\w+)\s+"
        r"wrist=\[([^\]]+)\]\s+"
        r"move=\[([^\]]+)\]\s+"
        r"target=\[([^\]]+)\]\s+"
        r"err=\[([^\]]+)\]\s+"
        r"\|err\|=([0-9.]+)\s+"
        r"used=\[([^\]]+)\]\s+"
        r"delta=\[([^\]]+)\]\s+"
        r"q_t=\[([^\]]+)\]\s+"
        r"q_a=\[([^\]]+)\]"
    )

    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = random_re.search(raw)
        if m:
            target_id += 1
            current_target_offset = parse_vec(m.group(2))
            continue

        m = line_re.search(raw)
        if not m:
            continue

        t = float(m.group(1))
        phase = m.group(2)
        wrist = parse_vec(m.group(3))
        move = parse_vec(m.group(4))
        target = parse_vec(m.group(5))
        err = parse_vec(m.group(6))
        err_norm = float(m.group(7))
        used = parse_vec(m.group(8))
        delta = parse_vec(m.group(9))
        q_t = parse_vec(m.group(10))
        q_a = parse_vec(m.group(11))

        toff = current_target_offset or [None, None, None]

        rows.append({
            "t": t,
            "phase": phase,
            "target_id": target_id,
            "target_offset_x": toff[0],
            "target_offset_y": toff[1],
            "target_offset_z": toff[2],
            "wrist_x": wrist[0],
            "wrist_y": wrist[1],
            "wrist_z": wrist[2],
            "move_x": move[0],
            "move_y": move[1],
            "move_z": move[2],
            "target_x": target[0],
            "target_y": target[1],
            "target_z": target[2],
            "err_x": err[0],
            "err_y": err[1],
            "err_z": err[2],
            "err_norm": err_norm,
            "used_pitch": used[0],
            "used_roll": used[1],
            "used_yaw": used[2],
            "used_elbow": used[3],
            "delta_pitch": delta[0],
            "delta_roll": delta[1],
            "delta_yaw": delta[2],
            "delta_elbow": delta[3],
            "q_t_pitch": q_t[0],
            "q_t_roll": q_t[1],
            "q_t_yaw": q_t[2],
            "q_t_elbow": q_t[3],
            "q_a_pitch": q_a[0],
            "q_a_roll": q_a[1],
            "q_a_yaw": q_a[2],
            "q_a_elbow": q_a[3],
        })

    if not rows:
        raise RuntimeError("没有解析到有效 tracking 日志，请检查日志格式。")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[完成] CSV 已保存: {out_path}")
    print(f"[统计] 共解析 {len(rows)} 行")


if __name__ == "__main__":
    main()
