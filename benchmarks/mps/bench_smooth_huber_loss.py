"""smooth_l1_loss / huber_loss: native Metal kernel timing.

Same methodology as bench_mse_loss.py: run this script on the PR parent for
the MPSGraph baseline and on this checkout for the native kernels; the PR body
carries the comparison. Timer.blocked_autorange over an amortized inner loop
(100 calls per device sync), 5 repeats after warmup, mean microseconds per
call. Covers both ops, all three reductions, forward and forward+backward,
and contiguous / non-contiguous (same-layout transposed) operands.
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
OPS = {
    "smooth_l1": (F.smooth_l1_loss, {"beta": 1.0}),
    "huber": (F.huber_loss, {"delta": 1.0}),
}
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
    op,
    reduction,
    layout,
    backward,
    inner_iters,
    warmup_blocks,
    repeats,
    min_run_time,
):
    fn, kw = OPS[op]
    x = torch.randn(shape, device=device, dtype=dt)
    t = torch.randn(shape, device=device, dtype=dt)
    if layout == "noncontig" and x.dim() >= 2:
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        t = t.transpose(-1, -2).contiguous().transpose(-1, -2)
    x.requires_grad_(backward)
    g = torch.ones(shape, device=device, dtype=dt) if reduction == "none" else None

    def run_block():
        for _ in range(inner_iters):
            loss = fn(x, t, reduction=reduction, **kw)
            if backward:
                x.grad = None
                loss.backward(g)
        synchronize(device)

    # The MPSGraph-era smooth_l1/huber raise on non-contiguous inputs; record
    # the error so a baseline run still produces a full report.
    try:
        for _ in range(warmup_blocks):
            run_block()
        synchronize(device)
    except RuntimeError as e:
        return {"error": str(e).splitlines()[0]}

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
            for op in OPS:
                for red in ("none", "mean", "sum"):
                    for layout in ("contig", "noncontig"):
                        if layout == "noncontig" and len(sh) < 2:
                            continue
                        for mode in ("fwd", "fwd+bwd"):
                            key = f"{nm}|{dn}|{op}|{red}|{layout}|{mode}"
                            out["results"][key] = bench(
                                device,
                                sh,
                                dt,
                                op,
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
