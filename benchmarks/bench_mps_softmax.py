"""Benchmark Metal softmax, log_softmax, and cross-entropy kernels on Apple Silicon.

Uses torch.utils.benchmark.Timer (the official PyTorch benchmarking tool)
with blocked_autorange for proper statistical measurement.

Usage:
    python benchmarks/bench_mps_softmax.py
    python benchmarks/bench_mps_softmax.py --dtype float16
    python benchmarks/bench_mps_softmax.py --backward
    python benchmarks/bench_mps_softmax.py --section ce
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


def fmt_result(measurement):
    med = measurement.median * 1e6
    iqr = measurement.iqr * 1e6
    return f"{med:>8.1f} +/- {iqr:>5.1f}"


def bench(stmt, glob, min_run_time=2.0):
    t = Timer(stmt=stmt, globals=glob)
    return t.blocked_autorange(min_run_time=min_run_time)


def run_softmax(args, dtype):
    print("=" * 72)
    print("SOFTMAX")
    print("=" * 72)

    cases = [
        ((1, 32000), -1),
        ((1, 50257), -1),
        ((1, 128256), -1),
        ((1, 256000), -1),
        ((4, 128256), -1),
        ((8, 128256), -1),
        ((32, 128256), -1),
        ((128, 128256), -1),
        ((32, 1, 4096), -1),
        ((32, 1, 32768), -1),
        ((32, 512, 512), -1),
        ((256, 1024), -1),
        ((1024, 256), -1),
        ((32, 4096, 1), 1),
        ((8, 128256, 4), 1),
        ((32, 512, 512), 0),
        ((4, 256, 256), 1),
        ((4, 21, 256, 256), 1),
        ((2, 21, 512, 512), 1),
        ((4, 19, 512, 1024), 1),
        ((2, 150, 128, 128), 1),
    ]

    header = f"{'shape':<24} {'dim':>4} {'fwd med +/- iqr (us)':>22}"
    if args.backward:
        header += f"  {'fwd+bwd med +/- iqr (us)':>22}"
    print(header)
    print("-" * len(header))

    for shape, dim in cases:
        x = torch.randn(shape, device="mps", dtype=dtype)
        fwd = bench(
            "torch.softmax(x, dim=dim); torch.mps.synchronize()",
            {"x": x, "dim": dim, "torch": torch},
            min_run_time=args.min_time,
        )
        line = f"{str(shape):<24} {dim:>4} {fmt_result(fwd):>22}"

        if args.backward:
            fb = bench(
                "xr = x.detach().requires_grad_(True); "
                "torch.softmax(xr, dim=dim).sum().backward(); "
                "torch.mps.synchronize()",
                {"x": x, "dim": dim, "torch": torch},
                min_run_time=args.min_time,
            )
            line += f"  {fmt_result(fb):>22}"
        print(line)


def run_log_softmax(args, dtype):
    print()
    print("=" * 72)
    print("LOG-SOFTMAX")
    print("=" * 72)

    cases = [
        ((1, 128256), -1),
        ((8, 128256), -1),
        ((32, 128256), -1),
        ((128, 128256), -1),
        ((8, 50257), -1),
        ((8, 4096), -1),
        ((256, 1024), -1),
        ((8, 128256, 4), 1),
        ((4, 21, 256, 256), 1),
        ((2, 150, 128, 128), 1),
    ]

    header = f"{'shape':<24} {'dim':>4} {'fwd med +/- iqr (us)':>22}"
    if args.backward:
        header += f"  {'fwd+bwd med +/- iqr (us)':>22}"
    print(header)
    print("-" * len(header))

    for shape, dim in cases:
        x = torch.randn(shape, device="mps", dtype=dtype)
        fwd = bench(
            "F.log_softmax(x, dim=dim); torch.mps.synchronize()",
            {"x": x, "dim": dim, "torch": torch, "F": F},
            min_run_time=args.min_time,
        )
        line = f"{str(shape):<24} {dim:>4} {fmt_result(fwd):>22}"

        if args.backward:
            fb = bench(
                "xr = x.detach().requires_grad_(True); "
                "F.log_softmax(xr, dim=dim).sum().backward(); "
                "torch.mps.synchronize()",
                {"x": x, "dim": dim, "torch": torch, "F": F},
                min_run_time=args.min_time,
            )
            line += f"  {fmt_result(fb):>22}"
        print(line)


def run_cross_entropy(args, dtype):
    print()
    print("=" * 72)
    print("CROSS-ENTROPY (fused vs decomposed)")
    print("=" * 72)

    cases = [(1, 128256), (8, 128256), (32, 128256), (8, 50257), (8, 1000), (256, 1000)]

    header = f"{'shape':<18} {'fused med +/- iqr (us)':>22}  {'decomp med +/- iqr (us)':>22}"
    print(header)
    print("-" * len(header))

    for B, V in cases:
        x = torch.randn(B, V, device="mps", dtype=dtype, requires_grad=True)
        t = torch.randint(0, V, (B,), device="mps")

        ft = bench(
            "loss = F.cross_entropy(x, t); loss.backward(); "
            "x.grad = None; torch.mps.synchronize()",
            {"x": x, "t": t, "F": F, "torch": torch},
            min_run_time=args.min_time,
        )
        dt = bench(
            "log_p = F.log_softmax(x, dim=-1); "
            "loss = F.nll_loss(log_p, t); loss.backward(); "
            "x.grad = None; torch.mps.synchronize()",
            {"x": x, "t": t, "F": F, "torch": torch},
            min_run_time=args.min_time,
        )

        print(f"{str((B, V)):<18} {fmt_result(ft):>22}  {fmt_result(dt):>22}")

    print()
    print("Cross-entropy with label_smoothing=0.1:")
    for B, V in [(8, 128256), (32, 128256)]:
        x = torch.randn(B, V, device="mps", dtype=dtype, requires_grad=True)
        t = torch.randint(0, V, (B,), device="mps")
        st = bench(
            "loss = F.cross_entropy(x, t, label_smoothing=0.1); "
            "loss.backward(); x.grad = None; torch.mps.synchronize()",
            {"x": x, "t": t, "F": F, "torch": torch},
            min_run_time=args.min_time,
        )
        print(f"  {str((B, V)):<18} {fmt_result(st):>22}")

    print()
    print("Cross-entropy with ignore_index:")
    for B, V in [(8, 128256), (32, 128256)]:
        x = torch.randn(B, V, device="mps", dtype=dtype, requires_grad=True)
        t = torch.randint(0, V, (B,), device="mps")
        t[0] = -100; t[B // 2] = -100
        it = bench(
            "loss = F.cross_entropy(x, t, ignore_index=-100); "
            "loss.backward(); x.grad = None; torch.mps.synchronize()",
            {"x": x, "t": t, "F": F, "torch": torch},
            min_run_time=args.min_time,
        )
        print(f"  {str((B, V)):<18} {fmt_result(it):>22}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MPS softmax, log_softmax, and cross-entropy kernels"
    )
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--backward", action="store_true",
                        help="Include fwd+bwd for softmax/log_softmax")
    parser.add_argument("--section", default="all",
                        choices=["all", "softmax", "log_softmax", "ce"])
    parser.add_argument("--min-time", type=float, default=2.0)
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    print(f"PyTorch {torch.__version__}")
    print(f"dtype: {args.dtype}")
    print(f"Benchmark: torch.utils.benchmark.Timer + blocked_autorange")
    print()

    if args.section in ("all", "softmax"):
        run_softmax(args, dtype)
    if args.section in ("all", "log_softmax"):
        run_log_softmax(args, dtype)
    if args.section in ("all", "ce"):
        run_cross_entropy(args, dtype)


if __name__ == "__main__":
    main()
