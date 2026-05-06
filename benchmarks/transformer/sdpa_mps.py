"""Benchmark MPS scaled_dot_product_attention forward+backward.

Sweeps (B, H, S, D, dtype, causal) across the math and efficient_attention
backends and reports per-iter wall-clock and peak driver memory as CSV.

Usage:
    python benchmarks/transformer/sdpa_mps.py                 # full sweep
    python benchmarks/transformer/sdpa_mps.py --quick         # short sanity sweep
    python benchmarks/transformer/sdpa_mps.py --out file.csv  # write to file

Output columns: B,H,S,D,dtype,causal,backend,time_ms,peak_mb

To find regression shapes, sort the CSV by time_ms ratio between backends for
matching (B,H,S,D,dtype,causal) rows. To see the memory wins, look at peak_mb
at large S.

Depends only on torch. Intended as the reproducibility artifact behind the
MPS efficient SDPA backward PR; CSVs from other M-series chips are welcome.
"""
import argparse
import csv
import gc
import itertools
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def measure(B, H, S, D, dtype, causal, backend, iters):
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    base = torch.mps.driver_allocated_memory()

    q = torch.randn(B, H, S, D, device="mps", dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    g = torch.randn_like(q)

    # Warmup
    for _ in range(3):
        with sdpa_kernel([backend]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out.backward(g, retain_graph=True)
        torch.mps.synchronize()
        q.grad = k.grad = v.grad = None

    peak = torch.mps.driver_allocated_memory() - base

    t0 = time.perf_counter()
    for _ in range(iters):
        with sdpa_kernel([backend]):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out.backward(g, retain_graph=True)
        torch.mps.synchronize()
        q.grad = k.grad = v.grad = None
    dt = (time.perf_counter() - t0) / iters

    del q, k, v, g, out
    return dt * 1e3, peak / 1e6


def shapes(quick):
    if quick:
        return [(1, 4, 256, 64), (1, 4, 1024, 64), (1, 16, 512, 128)]
    return list(
        itertools.product(
            [1, 2, 4],                   # B
            [4, 8, 16, 32],              # H
            [128, 256, 512, 1024, 2048], # S
            [32, 64, 96, 128],           # D
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="Short sanity sweep instead of full grid")
    ap.add_argument("--out", default=None, help="Write CSV here instead of stdout")
    ap.add_argument("--iters", type=int, default=10, help="Timed iterations per measurement")
    args = ap.parse_args()

    if not torch.backends.mps.is_available():
        print("MPS is not available; this benchmark only runs on Apple Silicon.", file=sys.stderr)
        sys.exit(1)

    out = open(args.out, "w", newline="") if args.out else sys.stdout
    w = csv.writer(out)
    w.writerow(["B", "H", "S", "D", "dtype", "causal", "backend", "time_ms", "peak_mb"])

    grid = list(
        itertools.product(
            shapes(args.quick),
            [torch.float16, torch.float32],
            [False, True],
            [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION],
        )
    )
    for (B, H, S, D), dtype, causal, backend in grid:
        dtype_name = str(dtype).split(".")[-1]
        try:
            ms, mb = measure(B, H, S, D, dtype, causal, backend, args.iters)
        except Exception as e:  # pragma: no cover - benchmark only
            ms, mb = float("nan"), float("nan")
            print(
                f"# skipped B={B} H={H} S={S} D={D} {dtype_name} causal={causal} {backend.name}: {e}",
                file=sys.stderr,
            )
        w.writerow([B, H, S, D, dtype_name, causal, backend.name, f"{ms:.3f}", f"{mb:.1f}"])
        out.flush()

    if args.out:
        out.close()


if __name__ == "__main__":
    main()
