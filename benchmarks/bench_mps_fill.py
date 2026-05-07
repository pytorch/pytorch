"""Benchmark MPS fill/zero operations: UMA memset/fill vs Metal compute.

Measures: zero_(), fill_(0), fill_(1.0), torch.zeros()
Across sizes and dtypes.

Usage:
    PYTHONPATH=~/pytorch python benchmarks/bench_mps_fill.py
"""
import math
import time

import torch

BASELINE_US = {
    # Measured on parent commit (Metal compute only), Apple M1 Max
    "scalar float32 zero_()": 176, "scalar float32 fill_(0)": 124,
    "scalar float32 fill_(1.0)": 108, "scalar float32 torch.zeros": 158,
    "1 KB float32 zero_()": 310, "1 KB float32 fill_(0)": 251,
    "1 KB float32 fill_(1.0)": 104, "1 KB float32 torch.zeros": 89,
    "64 KB float32 zero_()": 93, "64 KB float32 fill_(0)": 134,
    "64 KB float32 fill_(1.0)": 104, "64 KB float32 torch.zeros": 96,
    "1 MB float32 zero_()": 95, "1 MB float32 fill_(0)": 220,
    "1 MB float32 fill_(1.0)": 194, "1 MB float32 torch.zeros": 101,
    "16 MB float32 zero_()": 252, "16 MB float32 fill_(0)": 176,
    "16 MB float32 fill_(1.0)": 621, "16 MB float32 torch.zeros": 169,
    "1 MB float16 zero_()": 91, "1 MB float16 fill_(0)": 638,
    "1 MB float16 fill_(1.0)": 98, "1 MB float16 torch.zeros": 102,
    "1 MB bfloat16 zero_()": 100, "1 MB bfloat16 fill_(0)": 166,
    "1 MB bfloat16 fill_(1.0)": 210, "1 MB bfloat16 torch.zeros": 98,
    "1 MB int32 zero_()": 98, "1 MB int32 fill_(0)": 277,
    "1 MB int32 fill_(1.0)": 556, "1 MB int32 torch.zeros": 144,
    "1 MB int64 zero_()": 88, "1 MB int64 fill_(0)": 94,
    "1 MB int64 fill_(1.0)": 104, "1 MB int64 torch.zeros": 147,
    "1 MB bool zero_()": 100, "1 MB bool fill_(0)": 94,
    "1 MB bool fill_(1.0)": 94, "1 MB bool torch.zeros": 224,
}

def bench(fn, warmup=30, iters=100):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    mu = sum(times) / len(times)
    sigma = math.sqrt(sum((t - mu)**2 for t in times) / len(times))
    return mu, sigma

def speedup_str(label, after_us):
    base = BASELINE_US.get(label)
    if base is None:
        return ""
    return f"  [{base:.0f} -> {after_us:.1f} µs, {base/after_us:.1f}x]"

def run_fill(shape, dtype=torch.float32, label=""):
    t = torch.empty(shape, dtype=dtype, device="mps")

    # zero_()
    mu, sigma = bench(lambda: t.zero_())
    sp = speedup_str(f"{label} zero_()", mu)
    print(f"  {label+' zero_()':35s}  {mu:8.2f} ± {sigma:5.2f} µs{sp}")

    # fill_(0)
    mu, sigma = bench(lambda: t.fill_(0))
    sp = speedup_str(f"{label} fill_(0)", mu)
    print(f"  {label+' fill_(0)':35s}  {mu:8.2f} ± {sigma:5.2f} µs{sp}")

    # fill_(1.0)
    mu, sigma = bench(lambda: t.fill_(1.0))
    sp = speedup_str(f"{label} fill_(1.0)", mu)
    print(f"  {label+' fill_(1.0)':35s}  {mu:8.2f} ± {sigma:5.2f} µs{sp}")

    # torch.zeros (alloc + fill)
    mu, sigma = bench(lambda: torch.zeros(shape, dtype=dtype, device="mps"))
    sp = speedup_str(f"{label} torch.zeros", mu)
    print(f"  {label+' torch.zeros()':35s}  {mu:8.2f} ± {sigma:5.2f} µs{sp}")

print(f"PyTorch {torch.__version__}, MPS available: {torch.backends.mps.is_available()}")
print()

sizes = [
    ((1,), "scalar"),
    ((256,), "1 KB"),
    ((16384,), "64 KB"),
    ((262144,), "1 MB"),
    ((4194304,), "16 MB"),
]

print("MPS fill/zero benchmark (mean ± σ of 100 runs, 30 warmup):")
print("-" * 80)
for shape, label in sizes:
    run_fill(shape, label=f"{label} float32")
    print()

print("dtype coverage (1 MB tensors):")
print("-" * 80)
for dtype, dlabel in [
    (torch.float32, "float32"),
    (torch.float16, "float16"),
    (torch.bfloat16, "bfloat16"),
    (torch.int32, "int32"),
    (torch.int64, "int64"),
    (torch.bool, "bool"),
]:
    n = 1048576 // torch.tensor([], dtype=dtype).element_size()
    run_fill((n,), dtype=dtype, label=f"1 MB {dlabel}")
    print()
