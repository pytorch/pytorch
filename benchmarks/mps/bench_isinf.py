#!/usr/bin/env python3
"""A/B benchmark for MPS isposinf/isneginf migrated from MPSGraph to Metal.

Run once with a stock nightly wheel (MPSGraph baseline) and once with this
build (Metal); each cell amortizes K iterations behind a single
torch.mps.synchronize() so the ~110us sync floor does not mask kernel cost.
"""

import argparse

import torch
from torch.utils.benchmark import Timer


K = 20
SHAPES = [(4096,), (512, 768), (2048, 2048), (16, 128, 1024), (32, 4096)]
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
            x.view(-1)[:: max(1, x.numel() // 64)] = float("inf")
            glb = {"torch": torch, "x": x, "K": K}
            cells = {
                "isposinf": "\nfor _ in range(K): torch.isposinf(x)\ntorch.mps.synchronize()",
                "isneginf": "\nfor _ in range(K): torch.isneginf(x)\ntorch.mps.synchronize()",
            }
            for name, stmt in cells.items():
                ms = bench_cell(stmt, glb)
                print(
                    f"{name} {str(dt).replace('torch.', ''):<9} {str(shape):<17} {ms:10.5f}"
                )


if __name__ == "__main__":
    main()
