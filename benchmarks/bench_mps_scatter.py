#!/usr/bin/env python3
"""Benchmark scatter/gather ops on MPS — Metal kernel vs MPSGraph baseline.

Measures:
  1. Fixed-shape throughput (steady-state, no recompilation)
  2. Variable-shape throughput (many unique shapes — cache swell test)
  3. Gather throughput

Usage: python bench_mps_scatter.py
"""

import random
import time

import torch


def bench_scatter(N, D, dim, mode, dtype, num_warmup=20, num_runs=200):
    self_t = torch.zeros(N, D, dtype=dtype, device="mps")
    if dtype.is_floating_point:
        src = torch.randn(N, D, dtype=dtype, device="mps")
    else:
        src = torch.ones(N, D, dtype=dtype, device="mps")
    index = torch.randint(0, N, (N, D), dtype=torch.long, device="mps")

    def run():
        out = self_t.clone()
        if mode == "set":
            out.scatter_(dim, index, src)
        elif mode == "add":
            out.scatter_add_(dim, index, src)
        else:
            out.scatter_reduce_(dim, index, src, reduce=mode)
        torch.mps.synchronize()

    for _ in range(num_warmup):
        run()

    start = time.perf_counter()
    for _ in range(num_runs):
        run()
    return (time.perf_counter() - start) / num_runs * 1000


def bench_gather(N, D, dim, dtype, num_warmup=20, num_runs=200):
    self_t = torch.randn(N, D, dtype=dtype, device="mps")
    index = torch.randint(0, N, (N, D), dtype=torch.long, device="mps")

    for _ in range(num_warmup):
        torch.gather(self_t, dim, index)
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        torch.gather(self_t, dim, index)
        torch.mps.synchronize()
    return (time.perf_counter() - start) / num_runs * 1000


def bench_variable_shapes(num_shapes, D, mode, dtype):
    """Many unique (N, D) shapes in sequence — MPSGraph recompiles each one."""
    random.seed(42)
    shapes = [(random.randint(100, 10000), D) for _ in range(num_shapes)]

    # single warmup
    w = torch.zeros(100, D, dtype=dtype, device="mps")
    ws = (
        torch.randn(100, D, dtype=dtype, device="mps")
        if dtype.is_floating_point
        else torch.ones(100, D, dtype=dtype, device="mps")
    )
    wi = torch.randint(0, 100, (100, D), dtype=torch.long, device="mps")
    w.scatter_add_(0, wi, ws)
    torch.mps.synchronize()

    start = time.perf_counter()
    for N, d in shapes:
        self_t = torch.zeros(N, d, dtype=dtype, device="mps")
        if dtype.is_floating_point:
            src = torch.randn(N, d, dtype=dtype, device="mps")
        else:
            src = torch.ones(N, d, dtype=dtype, device="mps")
        index = torch.randint(0, N, (N, d), dtype=torch.long, device="mps")
        self_t.scatter_add_(0, index, src)
        torch.mps.synchronize()
    total = time.perf_counter() - start
    return total, total / num_shapes * 1000


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    # ── Fixed-shape scatter ───────────────────────────────────────────────
    print("=" * 70)
    print("Fixed-shape scatter (ms per op)")
    print("=" * 70)
    configs = [
        (1024, 64, 0, "add", torch.float32),
        (4096, 64, 0, "add", torch.float32),
        (4096, 128, 0, "add", torch.float32),
        (10000, 64, 0, "add", torch.float32),
        (4096, 64, 0, "add", torch.float16),
        (4096, 64, 0, "add", torch.bfloat16),
        (4096, 64, 0, "prod", torch.float32),
        (4096, 64, 0, "amax", torch.float32),
        (4096, 64, 0, "amin", torch.float32),
        (4096, 64, 0, "set", torch.float32),
        (4096, 64, 0, "add", torch.int64),
        (4096, 64, 0, "prod", torch.int64),
        (4096, 64, 0, "amax", torch.int64),
    ]
    print(f"{'Config':<50} {'ms':>8}")
    print("-" * 60)
    for N, D, dim, mode, dtype in configs:
        ms = bench_scatter(N, D, dim, mode, dtype)
        label = f"N={N:<6} D={D:<4} {mode:<5} {str(dtype):<14}"
        print(f"{label:<50} {ms:>8.3f}")

    # ── Gather ────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("Gather (ms per op)")
    print("=" * 70)
    gather_cfgs = [
        (1024, 64, 0, torch.float32),
        (4096, 64, 0, torch.float32),
        (4096, 128, 0, torch.float32),
        (10000, 64, 0, torch.float32),
        (4096, 64, 0, torch.float16),
    ]
    print(f"{'Config':<50} {'ms':>8}")
    print("-" * 60)
    for N, D, dim, dtype in gather_cfgs:
        ms = bench_gather(N, D, dim, dtype)
        label = f"N={N:<6} D={D:<4} {str(dtype):<14}"
        print(f"{label:<50} {ms:>8.3f}")

    # ── Variable-shape (cache swell test) ─────────────────────────────────
    print()
    print("=" * 70)
    print("Variable-shape scatter_add — 50 unique shapes (cache swell test)")
    print("=" * 70)
    for dtype in [torch.float32, torch.float16]:
        total, avg = bench_variable_shapes(50, 64, "add", dtype)
        print(f"{str(dtype):<16} total={total:.3f}s  avg={avg:.3f}ms/shape")
