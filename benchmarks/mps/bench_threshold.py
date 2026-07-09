#!/usr/bin/env python3
"""A/B benchmark for MPS threshold/relu fwd+bwd (Metal vs MPSGraph baseline).

Run once with a stock nightly (label=mpsgraph) and once with this build
(label=metal); Timer.blocked_autorange median per cell.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


parser = argparse.ArgumentParser()
parser.add_argument("--label", default="run")
parser.add_argument(
    "--dtype", default="float32", choices=["float32", "float16", "bfloat16"]
)
args = parser.parse_args()
dtype = getattr(torch, args.dtype)

print(f"label={args.label} dtype={args.dtype} torch={torch.__version__}")
print(f"{'config':<36} {'median (ms)':>12}")
shapes = [(1024,), (64, 4096), (512, 768), (2048, 2048), (16, 128, 1024)]
for shape in shapes:
    x = torch.randn(*shape, device="mps", dtype=dtype, requires_grad=True)
    t = Timer(
        stmt="y = F.threshold(x, 0.5, 0.0); y.backward(g); x.grad = None",
        globals={"F": F, "x": x, "g": torch.ones(shape, device="mps", dtype=dtype)},
    )
    m = t.blocked_autorange(min_run_time=0.5).median * 1e3
    print(f"threshold {str(shape):<26} {m:>12.3f}")
    xr = torch.randn(*shape, device="mps", dtype=dtype, requires_grad=True)
    tr = Timer(
        stmt="y = torch.relu(xr); y.backward(g); xr.grad = None",
        globals={
            "torch": torch,
            "xr": xr,
            "g": torch.ones(shape, device="mps", dtype=dtype),
        },
    )
    m = tr.blocked_autorange(min_run_time=0.5).median * 1e3
    print(f"relu      {str(shape):<26} {m:>12.3f}")
