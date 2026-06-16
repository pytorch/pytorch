"""mse_loss: fused square-and-reduce kernel timing.

Measures F.mse_loss on the current build. The fused kernel computes (a-b)^2 and
reduces in one pass (no materialized squared-diff tensor). For the A/B vs the
MPSGraph path, run this on a parent (pre-migration) checkout for the baseline and
on this checkout for the fused kernel; the PR body carries that comparison.
Methodology: torch.utils.benchmark.Timer.blocked_autorange over an amortized
inner loop (default: 100 mse calls followed by one device sync), repeated 5
times after warmup. The reported value is mean microseconds per mse call.

Defaults to the MPS device (this kernel's target) but takes --device, so the
same script profiles CUDA/CPU too. Covers contiguous and non-contiguous inputs,
all three reductions, and forward as well as forward+backward (the backward
path is migrated too: scalar broadcast grad_output for mean/sum, full-size
grad_output for none).
"""

import argparse
import json
import os
import statistics

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
SHAPES = [
    ("8x2048x4096", (8, 2048, 4096)),
    ("1024x1024", (1024, 1024)),
    ("4x65536", (4, 65536)),
    ("65536x128", (65536, 128)),
    ("256x1024", (256, 1024)),
]
INNER_ITERS = int(os.environ.get("INNER_ITERS", "100"))
WARMUP_BLOCKS = int(os.environ.get("WARMUP_BLOCKS", "30"))
REPEATS = int(os.environ.get("REPEATS", "5"))
MIN_RUN_TIME = float(os.environ.get("MIN_RUN_TIME", "0.2"))


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(
    device,
    shape,
    dt,
    reduction,
    layout,
    backward,
    inner_iters,
    warmup_blocks,
    repeats,
    min_run_time,
):
    x = torch.randn(shape, device=device, dtype=dt)
    t = torch.randn(shape, device=device, dtype=dt)
    if layout == "noncontig" and x.dim() >= 2:
        # Transposed views sharing one dense layout: mean/sum reductions take
        # the physical-order dense walk (no materialization); reduction=none
        # and the backward take the strided binary/ternary iterator paths.
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        t = t.transpose(-1, -2).contiguous().transpose(-1, -2)
    x.requires_grad_(backward)
    # Non-scalar loss (reduction="none") needs an explicit grad_output; a
    # full-size ones tensor also exercises the elementwise backward, while the
    # scalar mean/sum backward covers the broadcast (stride-0) grad_output.
    g = torch.ones(shape, device=device, dtype=dt) if reduction == "none" else None

    def run_block():
        for _ in range(inner_iters):
            loss = F.mse_loss(x, t, reduction=reduction)
            if backward:
                x.grad = None
                loss.backward(g)
        synchronize(device)

    for _ in range(warmup_blocks):
        run_block()
    synchronize(device)

    timer = Timer(stmt="run_block()", globals={"run_block": run_block})
    samples_us = [
        timer.blocked_autorange(min_run_time=min_run_time).mean * 1e6 / inner_iters
        for _ in range(repeats)
    ]
    return {
        "mean_us": round(statistics.mean(samples_us), 1),
        "stdev_us": (
            round(statistics.stdev(samples_us), 1) if len(samples_us) > 1 else 0.0
        ),
        "samples_us": [round(v, 1) for v in samples_us],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inner-iters", type=int, default=INNER_ITERS)
    parser.add_argument("--warmup-blocks", type=int, default=WARMUP_BLOCKS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--min-run-time", type=float, default=MIN_RUN_TIME)
    args = parser.parse_args()
    device = torch.device(args.device)

    out = {
        "metadata": {
            "device": str(device),
            "inner_iters": args.inner_iters,
            "warmup_blocks": args.warmup_blocks,
            "repeats": args.repeats,
            "min_run_time_s": args.min_run_time,
            "method": "Timer.blocked_autorange mean of repeated amortized blocks",
        },
        "results": {},
    }
    for nm, sh in SHAPES:
        for dn, dt in DTYPES.items():
            for red in ("none", "mean", "sum"):
                for layout in ("contig", "noncontig"):
                    if layout == "noncontig" and len(sh) < 2:
                        continue
                    for mode in ("fwd", "fwd+bwd"):
                        out["results"][f"{nm}|{dn}|{red}|{layout}|{mode}"] = bench(
                            device,
                            sh,
                            dt,
                            red,
                            layout,
                            mode == "fwd+bwd",
                            args.inner_iters,
                            args.warmup_blocks,
                            args.repeats,
                            args.min_run_time,
                        )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
