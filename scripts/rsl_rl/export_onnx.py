# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert exported JIT policy to ONNX format for deployment."""

import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert JIT policy to ONNX.")
    parser.add_argument("--jit_path", type=str, required=True, help="Path to the exported policy.pt file")
    parser.add_argument("--obs_dim", type=int, default=480, help="Observation dimension")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path (default: same dir as jit)")
    args = parser.parse_args()

    jit_path = os.path.abspath(args.jit_path)
    if not os.path.exists(jit_path):
        print(f"[ERROR] JIT model not found: {jit_path}")
        return

    if args.output:
        onnx_path = args.output
    else:
        onnx_path = os.path.join(os.path.dirname(jit_path), "policy.onnx")

    print(f"[INFO] Loading JIT model from: {jit_path}")
    model = torch.jit.load(jit_path, map_location="cpu")
    model.eval()

    # Rebuild as a pure nn.Module to avoid ScriptModule export issues.
    # Model structure: obs_normalizer (Identity, no-op) → mlp (4-layer ELU MLP)
    #   mlp: Linear(480,512) → ELU → Linear(512,256) → ELU → Linear(256,128) → ELU → Linear(128,29)
    class RebuiltModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(480, 512),
                torch.nn.ELU(),
                torch.nn.Linear(512, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 29),
            )

        def forward(self, obs):
            return self.mlp(obs)

    rebuilt = RebuiltModel()
    rebuilt.eval()

    # Copy weights from ScriptModule to rebuilt model
    for (_, src), (_, dst) in zip(
        model.mlp.named_parameters(), rebuilt.mlp.named_parameters()
    ):
        dst.data.copy_(src.data)

    # Verify output matches
    example_input = torch.randn(1, args.obs_dim, dtype=torch.float32)
    with torch.no_grad():
        orig_out = model({"policy": example_input})
        rebuilt_out = rebuilt(example_input)
    diff = (orig_out - rebuilt_out).abs().max().item()
    print(f"[INFO] Max output diff (should be 0.0): {diff}")

    print(f"[INFO] Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        rebuilt,
        example_input,
        onnx_path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
    )

    print(f"[INFO] Successfully exported ONNX model to: {onnx_path}")

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[INFO] ONNX model verification passed.")


if __name__ == "__main__":
    main()