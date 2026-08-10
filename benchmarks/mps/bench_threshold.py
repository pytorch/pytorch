#!/usr/bin/env python3
"""A/B benchmark for MPS threshold/threshold_backward migrated from MPSGraph
to Metal (threshold_backward is also relu's backward).

Run once with a stock nightly wheel (MPSGraph baseline) and once with this
build (Metal); each cell amortizes K iterations behind a single
torch.mps.synchronize() so the ~110us sync floor does not mask kernel cost.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


K = 20
SHAPES = [(4096,), (64, 4096), (512, 768), (2048, 2048), (16, 128, 1024)]
DTYPES = [torch.float32, torch.float16]


def bench_cell(stmt, glb):
    t = Timer(stmt=stmt, globals=glb)
    return t.blocked_autorange(min_run_time=0.4).median / K * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    print(f"label={args.label} torch={torch.__version__}")

    for dt in DTYPES:
        for shape in SHAPES:
            x = torch.randn(*shape, device="mps", dtype=dt)
            g = torch.randn(*shape, device="mps", dtype=dt)
            glb = {"torch": torch, "F": F, "x": x, "g": g, "K": K}
            cells = {
                "threshold_fwd": "\nfor _ in range(K): F.threshold(x, 0.5, 0.0)\ntorch.mps.synchronize()",
                "threshold_bwd": "\nfor _ in range(K): torch.ops.aten.threshold_backward(g, x, 0.5)\ntorch.mps.synchronize()",
                "relu_bwd": "\nfor _ in range(K): torch.ops.aten.threshold_backward(g, x, 0)\ntorch.mps.synchronize()",
            }
            for name, stmt in cells.items():
                ms = bench_cell(stmt, glb)
                print(
                    f"{name} {str(dt).replace('torch.', ''):<9} {str(shape):<17} {ms:10.5f}"
                )
        # integral dtype leg (exact scalar_t params path)
        xi = torch.randint(-100, 100, (2048, 2048), device="mps", dtype=torch.int64)
        glb = {"torch": torch, "F": F, "xi": xi, "K": K}
        ms = bench_cell(
            "\nfor _ in range(K): F.threshold(xi, 3, -1)\ntorch.mps.synchronize()", glb
        )
        print(f"threshold_fwd int64     (2048, 2048)      {ms:10.5f}")


if __name__ == "__main__":
    main()
