#!/usr/bin/env python
"""Benchmark comparison: C++ vs Python deserialization for torch.export.load().

Produces a comparison grid showing:
  - torch.jit.load() (baseline, fully C++)
  - torch.export.load() (now using C++ JSON parse + schema construction)
  - Sequential vs parallel (threaded) loading
"""

import os
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn


class BenchmarkModel(nn.Module):
    def __init__(self, hidden=256, layers=24):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=8,
                    dim_feedforward=hidden * 4,
                    batch_first=True,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def time_fn(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return time.perf_counter() - t0, result


def benchmark_sequential(load_fn, path, n=4, warmup=1):
    """Load n models sequentially."""
    for _ in range(warmup):
        load_fn(path)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            load_fn(path)
        times.append(time.perf_counter() - t0)
    return times


def benchmark_parallel(load_fn, path, n=4, warmup=1):
    """Load n models in parallel threads."""
    for _ in range(warmup):
        load_fn(path)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(load_fn, [path] * n))
        times.append(time.perf_counter() - t0)
    return times


def benchmark_single(load_fn, path, warmup=1, iters=5):
    """Single load, multiple iterations."""
    for _ in range(warmup):
        load_fn(path)

    times = []
    for _ in range(iters):
        elapsed, _ = time_fn(load_fn, path)
        times.append(elapsed)
    return times


def median(times):
    return statistics.median(times)


def main():
    n_parallel = 4

    print("=" * 72)
    print("  torch.export.load() C++ Deserialization Benchmark")
    print("=" * 72)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  CPUs:    {os.cpu_count()}")
    print(f"  Threads: {n_parallel} (for parallel tests)")
    print()

    model = BenchmarkModel()
    model.eval()
    example_input = (torch.randn(1, 16, 256),)

    with tempfile.TemporaryDirectory() as tmp:
        # Save models
        print("Preparing models...")
        ep = torch.export.export(model, example_input, strict=False)
        pt2_path = os.path.join(tmp, "model.pt2")
        torch.export.save(ep, pt2_path)
        pt2_size = os.path.getsize(pt2_path) / (1024 * 1024)

        with torch.no_grad():
            traced = torch.jit.trace(model, example_input, check_trace=False)
        ts_path = os.path.join(tmp, "model_ts.pt")
        torch.jit.save(traced, ts_path)
        ts_size = os.path.getsize(ts_path) / (1024 * 1024)

        print(f"  export model: {pt2_size:.1f} MB")
        print(f"  jit model:    {ts_size:.1f} MB")
        print()

        # Define loaders
        def jit_load(p):
            return torch.jit.load(p, map_location="cpu")

        def export_load(p):
            return torch.export.load(p)

        # --- Single load benchmarks ---
        print("Running single-load benchmarks...")
        jit_single = benchmark_single(jit_load, ts_path)
        export_single = benchmark_single(export_load, pt2_path)

        # --- Sequential N loads ---
        print(f"Running sequential {n_parallel}x load benchmarks...")
        jit_seq = benchmark_sequential(jit_load, ts_path, n=n_parallel)
        export_seq = benchmark_sequential(export_load, pt2_path, n=n_parallel)

        # --- Parallel N loads ---
        print(f"Running parallel {n_parallel}x load benchmarks...")
        jit_par = benchmark_parallel(jit_load, ts_path, n=n_parallel)
        try:
            export_par = benchmark_parallel(export_load, pt2_path, n=n_parallel)
        except Exception as e:
            print(f"  export parallel failed: {e}")
            export_par = None

    # --- Results grid ---
    print()
    print("=" * 72)
    print("  RESULTS (median wall-clock seconds)")
    print("=" * 72)
    print()

    header = f"{'Scenario':<30} {'jit.load':<12} {'export.load':<12} {'ratio':<8}"
    print(header)
    print("-" * len(header))

    def row(label, jit_times, export_times):
        j = median(jit_times)
        if export_times is None:
            print(f"{label:<30} {j:<12.3f} {'N/A':<12} {'N/A':<8}")
            return
        e = median(export_times)
        r = e / j if j > 0 else float("inf")
        print(f"{label:<30} {j:<12.3f} {e:<12.3f} {r:<8.1f}x")

    row("Single load", jit_single, export_single)
    row(f"Sequential {n_parallel}x", jit_seq, export_seq)
    row(f"Parallel {n_parallel}x (threads)", jit_par, export_par)

    # Thread scaling
    print()
    print("-" * len(header))
    jit_speedup = median(jit_seq) / median(jit_par)
    print(f"{'Thread scaling (seq/par)':<30} {jit_speedup:<12.2f}", end="")
    if export_par is not None:
        export_speedup = median(export_seq) / median(export_par)
        print(f" {export_speedup:<12.2f}", end="")
    else:
        print(f" {'N/A':<12}", end="")
    print()
    if export_par is None:
        print()
        print("  NOTE: Parallel export.load() failed due to pre-existing")
        print("  thread-safety issue in GraphModuleDeserializer (global state).")

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
