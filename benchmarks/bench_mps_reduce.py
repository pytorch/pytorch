"""Benchmark Metal reduce ops (var, std, prod, argmax, argmin, max, min) on MPS.

Uses torch.utils.benchmark.Timer + blocked_autorange.

Usage:
    python benchmarks/bench_mps_reduce.py
    python benchmarks/bench_mps_reduce.py --variable-shapes
"""

import argparse

import torch
from torch.utils.benchmark import Timer


def fmt(m):
    return f"{m.median * 1e6:>8.1f} +/- {m.iqr * 1e6:>5.1f}"


def bench(stmt, glob, min_run_time=2.0):
    t = Timer(stmt=stmt, globals=glob)
    return t.blocked_autorange(min_run_time=min_run_time)


def run_fixed(args):
    print("=" * 72)
    print("FIXED-SHAPE REDUCTIONS")
    print("=" * 72)

    cases = [
        ("prod dim=0", (128, 256), "torch.prod(x, dim=0); torch.mps.synchronize()"),
        ("prod dim=1", (128, 256), "torch.prod(x, dim=1); torch.mps.synchronize()"),
        ("var dim=1", (128, 256), "torch.var(x, dim=1); torch.mps.synchronize()"),
        ("var dim=0", (128, 256), "torch.var(x, dim=0); torch.mps.synchronize()"),
        ("std dim=1", (128, 256), "torch.std(x, dim=1); torch.mps.synchronize()"),
        (
            "var_mean dim=1",
            (128, 256),
            "torch.var_mean(x, dim=1); torch.mps.synchronize()",
        ),
        (
            "std_mean dim=1",
            (128, 256),
            "torch.std_mean(x, dim=1); torch.mps.synchronize()",
        ),
        ("argmax dim=0", (128, 256), "torch.argmax(x, dim=0); torch.mps.synchronize()"),
        ("argmax dim=1", (128, 256), "torch.argmax(x, dim=1); torch.mps.synchronize()"),
        ("argmin dim=1", (128, 256), "torch.argmin(x, dim=1); torch.mps.synchronize()"),
        ("max dim=0", (128, 256), "torch.max(x, dim=0); torch.mps.synchronize()"),
        ("max dim=1", (128, 256), "torch.max(x, dim=1); torch.mps.synchronize()"),
        ("min dim=0", (128, 256), "torch.min(x, dim=0); torch.mps.synchronize()"),
    ]

    header = f"{'op':<24} {'shape':>14} {'med +/- iqr (us)':>22}"
    print(header)
    print("-" * len(header))

    for name, shape, stmt in cases:
        x = torch.randn(shape, device="mps")
        result = bench(stmt, {"x": x, "torch": torch}, min_run_time=args.min_time)
        print(f"{name:<24} {str(shape):>14} {fmt(result):>22}")


def run_large(args):
    print()
    print("=" * 72)
    print("LARGE REDUCTIONS")
    print("=" * 72)

    cases = [
        ("var dim=1", (1024, 4096), "torch.var(x, dim=1); torch.mps.synchronize()"),
        (
            "argmax dim=1",
            (1024, 4096),
            "torch.argmax(x, dim=1); torch.mps.synchronize()",
        ),
        ("max dim=1", (1024, 4096), "torch.max(x, dim=1); torch.mps.synchronize()"),
        ("prod dim=1", (1024, 4096), "torch.prod(x, dim=1); torch.mps.synchronize()"),
    ]

    header = f"{'op':<24} {'shape':>14} {'med +/- iqr (us)':>22}"
    print(header)
    print("-" * len(header))

    for name, shape, stmt in cases:
        x = torch.randn(shape, device="mps")
        result = bench(stmt, {"x": x, "torch": torch}, min_run_time=args.min_time)
        print(f"{name:<24} {str(shape):>14} {fmt(result):>22}")


def run_variable(args):
    print()
    print("=" * 72)
    print("VARIABLE-SHAPE (20 different shapes)")
    print("=" * 72)

    import random

    random.seed(42)
    shapes = [(random.randint(32, 512), random.randint(64, 1024)) for _ in range(20)]

    for op_name, op_fn_str in [
        ("var dim=0", "torch.var(x, dim=0)"),
        ("argmax dim=0", "torch.argmax(x, dim=0)"),
    ]:
        tensors = [torch.randn(s, device="mps") for s in shapes]
        stmt = f"for x in tensors:\n    {op_fn_str}\ntorch.mps.synchronize()"
        result = bench(
            stmt, {"tensors": tensors, "torch": torch}, min_run_time=args.min_time
        )
        per_call = result.median * 1e6 / len(shapes)
        iqr_per = result.iqr * 1e6 / len(shapes)
        print(
            f"{op_name:<24} 20 shapes     {per_call:>8.1f} +/- {iqr_per:>5.1f} us/call"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MPS reduce ops (Metal vs MPSGraph)"
    )
    parser.add_argument("--min-time", type=float, default=2.0)
    parser.add_argument(
        "--variable-shapes",
        action="store_true",
        help="Include variable-shape cache-swelling test",
    )
    args = parser.parse_args()

    print(f"PyTorch {torch.__version__}")
    print("Benchmark: torch.utils.benchmark.Timer + blocked_autorange")
    print()

    run_fixed(args)
    run_large(args)
    if args.variable_shapes:
        run_variable(args)


if __name__ == "__main__":
    main()
