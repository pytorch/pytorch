#!/usr/bin/env python3
"""A/B benchmark for MPS nll_loss migrated from MPSGraph to Metal.

Run once with a stock nightly wheel (MPSGraph baseline) and once with this
build (Metal); each cell amortizes K iterations behind a single
torch.mps.synchronize() so the ~110us sync floor does not mask kernel cost.
"""

import argparse

import torch
from torch.utils.benchmark import Timer


K = 20
# (N, C) class problems spanning small-batch/large-vocab and the reverse,
# tied to real model output shapes (classification heads and LM heads).
CASES = [(4096, 10), (128, 32000), (4096, 1000), (65536, 10)]
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
        for N, C in CASES:
            x = torch.randn(N, C, device="mps", dtype=dt).log_softmax(-1)
            t = torch.randint(0, C, (N,), device="mps")
            w = torch.rand(C, device="mps", dtype=dt) + 0.5
            g = torch.tensor(1.0, device="mps", dtype=dt)
            tw = torch.tensor(float(N), device="mps", dtype=dt)
            glb = {
                "torch": torch,
                "F": torch.nn.functional,
                "x": x,
                "t": t,
                "w": w,
                "g": g,
                "tw": tw,
                "K": K,
            }
            cells = {
                "fwd_mean": "\nfor _ in range(K): F.nll_loss(x, t)\ntorch.mps.synchronize()",
                "fwd_none": "\nfor _ in range(K): F.nll_loss(x, t, reduction='none')\ntorch.mps.synchronize()",
                "fwd_mean_w": "\nfor _ in range(K): F.nll_loss(x, t, weight=w)\ntorch.mps.synchronize()",
                "bwd_mean": (
                    "\nfor _ in range(K): torch.ops.aten.nll_loss_backward(g, x, t, None, 1, -100, tw)"
                    "\ntorch.mps.synchronize()"
                ),
            }
            for name, stmt in cells.items():
                ms = bench_cell(stmt, glb)
                print(
                    f"{name} {str(dt).replace('torch.', ''):<9} N{N}xC{C:<7} {ms:10.5f}"
                )
        # spatial (nll_loss2d) case
        xi = torch.randn(16, 5, 64, 64, device="mps", dtype=dt).log_softmax(1)
        ti = torch.randint(0, 5, (16, 64, 64), device="mps")
        glb = {"torch": torch, "F": torch.nn.functional, "xi": xi, "ti": ti, "K": K}
        ms = bench_cell(
            "\nfor _ in range(K): F.nll_loss(xi, ti)\ntorch.mps.synchronize()", glb
        )
        print(f"fwd_2d_mean {str(dt).replace('torch.', ''):<9} 16x5x64x64 {ms:10.5f}")


if __name__ == "__main__":
    main()
