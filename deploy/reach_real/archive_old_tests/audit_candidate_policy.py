#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


DEFAULT_Q = np.array(
    [0.30, -0.25, 0.00, 0.97],
    dtype=np.float32,
)

ACTION_SCALE = np.array(
    [0.18, 0.15, 0.15, 0.20],
    dtype=np.float32,
)

# 最近一次真机稳定状态的前4个右臂关节
REAL_Q = np.array(
    [0.058855, 0.000264, -0.011553, 0.95305],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX右臂策略离线安全审计，不连接机器人。"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mark-on-fail",
        action="store_true",
        help="测试失败时在模型目录写入UNSAFE标记。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    print("=" * 78)
    print("[离线安全审计]")
    print("[安全] 不连接机器人")
    print("[安全] 不初始化DDS")
    print("[安全] 不创建Publisher")
    print(f"[模型] {model_path}")
    print("=" * 78)

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            f"预期1个输入和1个输出，实际为"
            f"{len(inputs)}个输入、{len(outputs)}个输出。"
        )

    input_info = inputs[0]
    output_info = outputs[0]

    print(
        f"[输入] name={input_info.name}, "
        f"shape={input_info.shape}, type={input_info.type}"
    )
    print(
        f"[输出] name={output_info.name}, "
        f"shape={output_info.shape}, type={output_info.type}"
    )

    input_shape = input_info.shape
    output_shape = output_info.shape

    shape_ok = (
        len(input_shape) == 2
        and input_shape[-1] == 15
        and len(output_shape) == 2
        and output_shape[-1] == 4
    )

    if not shape_ok:
        print("[失败] 模型输入输出维度不是15维输入、4维输出。")

    def infer(obs):
        obs = np.asarray(
            obs,
            dtype=np.float32,
        ).reshape(1, 15)

        result = session.run(
            [output_info.name],
            {input_info.name: obs},
        )[0]

        action = np.asarray(
            result,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape != (4,):
            raise RuntimeError(
                f"动作输出维度异常：{action.shape}"
            )

        return action

    def show_case(name, obs, reference_q):
        action = infer(obs)
        q_target = DEFAULT_Q + ACTION_SCALE * action
        delta = q_target - reference_q

        print("\n" + "-" * 78)
        print(name)
        print(
            "action   =",
            np.array2string(
                action,
                precision=5,
                floatmode="fixed",
            ),
        )
        print(
            "q_target =",
            np.array2string(
                q_target,
                precision=5,
                floatmode="fixed",
            ),
        )
        print(
            "delta_q  =",
            np.array2string(
                delta,
                precision=5,
                floatmode="fixed",
            ),
        )

        return action, q_target, delta

    # A：训练原点
    zero_obs = np.zeros(15, dtype=np.float32)

    action_zero, _, delta_zero = show_case(
        "A. 全零观测：训练原点保持测试",
        zero_obs,
        DEFAULT_Q,
    )

    # B：当前真机姿态，目标就在当前手腕位置
    real_obs = np.zeros(15, dtype=np.float32)
    real_obs[0:4] = REAL_Q - DEFAULT_Q

    action_real, _, delta_real = show_case(
        "B. 当前真机姿态保持测试",
        real_obs,
        REAL_Q,
    )

    # C/D/E：1厘米目标变化
    target_actions = []

    for name, vector in [
        ("C. X方向目标偏移1厘米", [0.01, 0.00, 0.00]),
        ("D. Y方向目标偏移1厘米", [0.00, 0.01, 0.00]),
        ("E. Z方向目标偏移1厘米", [0.00, 0.00, 0.01]),
    ]:
        obs = np.zeros(15, dtype=np.float32)
        obs[8:11] = np.asarray(vector, dtype=np.float32)

        action, _, _ = show_case(
            name,
            obs,
            DEFAULT_Q,
        )
        target_actions.append(action)

    # F：上一动作递推
    previous_action = np.zeros(4, dtype=np.float32)
    recursive_actions = []

    print("\n" + "-" * 78)
    print("F. 仅递推上一动作20次")

    for step in range(1, 21):
        obs = np.zeros(15, dtype=np.float32)
        obs[11:15] = previous_action

        action = infer(obs)
        recursive_actions.append(action.copy())
        previous_action = action.copy()

        if step in (1, 2, 3, 5, 10, 20):
            print(
                f"第{step:2d}次：",
                np.array2string(
                    action,
                    precision=5,
                    floatmode="fixed",
                ),
            )

    final_recursive_action = recursive_actions[-1]
    final_recursive_delta = (
        ACTION_SCALE * final_recursive_action
    )

    all_arrays = [
        action_zero,
        delta_zero,
        action_real,
        delta_real,
        final_recursive_action,
        final_recursive_delta,
        *target_actions,
    ]

    finite_ok = all(
        np.all(np.isfinite(array))
        for array in all_arrays
    )

    zero_delta_max = float(
        np.max(np.abs(delta_zero))
    )
    real_delta_max = float(
        np.max(np.abs(delta_real))
    )
    recursive_delta_max = float(
        np.max(np.abs(final_recursive_delta))
    )

    target_sensitivity = max(
        float(np.max(np.abs(action - action_zero)))
        for action in target_actions
    )

    print("\n" + "=" * 78)
    print("[审计指标]")
    print(f"数值全部有限：{finite_ok}")
    print(
        "训练原点最大假想关节变化："
        f"{zero_delta_max:.6f} rad"
    )
    print(
        "当前真机姿态最大假想关节变化："
        f"{real_delta_max:.6f} rad"
    )
    print(
        "20次上一动作递推后的最大关节变化："
        f"{recursive_delta_max:.6f} rad"
    )
    print(
        "1厘米目标引起的最大动作变化："
        f"{target_sensitivity:.6f}"
    )

    # 本项目采用的保守离线门槛。
    # 通过只表示可以进入影子推理，不代表可以直接控制真机。
    failures = []

    if not shape_ok:
        failures.append("模型输入输出维度不符合15→4")

    if not finite_ok:
        failures.append("模型输出存在NaN或Inf")

    if zero_delta_max > 0.05:
        failures.append(
            "全零观测下最大关节偏移超过0.05 rad"
        )

    if real_delta_max > 0.05:
        failures.append(
            "当前姿态保持误差超过0.05 rad"
        )

    if recursive_delta_max > 0.10:
        failures.append(
            "上一动作递推后关节偏移超过0.10 rad"
        )

    passed = len(failures) == 0

    if passed:
        print("\n[离线结论] 通过离线门槛。")
        print(
            "[注意] 仅允许进入纯影子推理，"
            "尚未获得真机控制资格。"
        )
    else:
        print("\n[离线结论] 未通过，禁止真机控制。")

        for index, failure in enumerate(
            failures,
            start=1,
        ):
            print(f"  {index}. {failure}")

        if args.mark_on_fail:
            marker_path = (
                model_path.parent
                / "UNSAFE_FOR_REAL_ROBOT.txt"
            )

            marker_text = f"""该策略不得用于真实机器人控制。

模型：
{model_path}

离线审计失败原因：
{chr(10).join(f"- {item}" for item in failures)}

主要指标：
- 训练原点最大关节变化：{zero_delta_max:.6f} rad
- 当前真机姿态最大关节变化：{real_delta_max:.6f} rad
- 上一动作递推最大关节变化：{recursive_delta_max:.6f} rad
- 1厘米目标动作敏感度：{target_sensitivity:.6f}

允许用途：
- IsaacLab仿真
- ONNX离线分析
- 纯影子推理
- 作为下一版本训练修改的基线

禁止用途：
- rt/arm_sdk真实控制
- rt/lowcmd真实控制
- 任何真实电机指令发布
"""

            marker_path.write_text(
                marker_text,
                encoding="utf-8",
            )

            print(f"[已写入危险标记] {marker_path}")

    print("=" * 78)

    # 失败时返回非零退出码，便于脚本判断。
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
