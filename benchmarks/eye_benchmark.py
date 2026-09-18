"""Benchmark for torch.eye showing performance cliff fix at d=182.

Before fix: ~10x slowdown at d=182 due to zero_() dispatching through
parallel TensorIterator when numel >= GRAIN_SIZE (32768).

After fix: Consistent performance across the threshold.

Usage:
    python benchmarks/eye_benchmark.py
"""

import time

import torch


def benchmark_eye(sizes, warmup=10, iterations=1000):
    results = {}
    for d in sizes:
        # Warmup
        for _ in range(warmup):
            torch.eye(d)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            torch.eye(d)
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / iterations) * 1e6
        results[d] = avg_us
    return results


if __name__ == "__main__":
    sizes = [
        100, 128, 150, 170, 180, 181, 182, 183,
        190, 200, 256, 300, 500, 512, 1024, 2048,
        4096, 8192,
    ]
    print(f"{'Size':>6} {'Time (us)':>12} {'Ratio vs d=181':>16}")
    print("-" * 38)

    results = benchmark_eye(sizes)
    baseline = results.get(181, results.get(180, 1.0))

    for d, t in results.items():
        ratio = t / baseline
        marker = " <<<" if d == 182 else ""
        print(f"{d:>6} {t:>12.2f} {ratio:>16.2f}x{marker}")
