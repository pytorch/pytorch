"""Benchmark Metal softmax kernels on Apple Silicon.

Measures forward and forward+backward wall-clock time across LLM-realistic
shapes (vocab logits, attention scores) for both last-dim and non-last-dim.

Methodology:
- 300-iteration GPU warmup to reach steady-state clock frequency
- 5 independent trials, each timed for at least 3 seconds
- torch.mps.synchronize() inside every measurement window
- Reports min / median / max across trials to expose thermal noise
- Shapes run in fixed order (logits → generation → prefill) to avoid
  thermal bias from heavy shapes warming the GPU for lighter ones

Usage:
    python benchmarks/bench_mps_softmax.py
    python benchmarks/bench_mps_softmax.py --dtype float16
    python benchmarks/bench_mps_softmax.py --dtype bfloat16 --backward
"""
import argparse
import time
import statistics
import torch


def time_forward(x, dim, n_iters):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        torch.softmax(x, dim=dim)
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / n_iters


def time_fwd_bwd(x_template, dim, n_iters):
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        x = x_template.detach().requires_grad_(True)
        y = torch.softmax(x, dim=dim)
        y.sum().backward()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / n_iters


def bench(shape, dim, dtype, include_backward, min_time=3.0, n_trials=5, warmup=300):
    x = torch.randn(shape, device="mps", dtype=dtype)

    # Warmup
    for _ in range(warmup):
        torch.softmax(x, dim=dim)
    if include_backward:
        for _ in range(warmup):
            xw = x.detach().requires_grad_(True)
            torch.softmax(xw, dim=dim).sum().backward()
    torch.mps.synchronize()

    # Forward trials
    fwd_times = []
    for _ in range(n_trials):
        n_iters = 10
        elapsed = time_forward(x, dim, n_iters)
        n_iters = max(10, int(min_time / (elapsed + 1e-9)))
        fwd_times.append(time_forward(x, dim, n_iters) * 1e6)

    result = {"fwd": fwd_times}

    # Forward+backward trials
    if include_backward:
        fb_times = []
        for _ in range(n_trials):
            n_iters = 10
            elapsed = time_fwd_bwd(x, dim, n_iters)
            n_iters = max(10, int(min_time / (elapsed + 1e-9)))
            fb_times.append(time_fwd_bwd(x, dim, n_iters) * 1e6)
        result["fwd_bwd"] = fb_times

    return result


def fmt_times(times):
    mn, md, mx = min(times), statistics.median(times), max(times)
    return f"{md:>8.1f}  ({mn:.1f} / {mx:.1f})"


def main():
    parser = argparse.ArgumentParser(description="Benchmark MPS softmax kernels")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type (default: float32)",
    )
    parser.add_argument(
        "--backward", action="store_true", help="Include forward+backward timing"
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=3.0,
        help="Minimum seconds per trial (default: 3.0)",
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="Number of trials (default: 5)"
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # (shape, dim) pairs — ordered light to heavy to avoid thermal bias
    cases = [
        # Vocab logits (autoregressive generation), dim=-1
        ((1, 32000), -1),
        ((1, 50257), -1),
        ((1, 128256), -1),
        ((1, 256000), -1),
        # Batched logits
        ((4, 128256), -1),
        ((8, 128256), -1),
        # Attention scores (generation: batch * n_heads, 1, seq_len)
        ((32, 1, 4096), -1),
        ((32, 1, 32768), -1),
        ((32, 1, 131072), -1),
        # Attention scores (prefill: batch * n_heads, seq_len, seq_len)
        ((32, 512, 512), -1),
        ((32, 2048, 2048), -1),
        # Non-last-dim softmax (uses permute path)
        ((32, 4096, 1), 1),
        ((8, 128256, 4), 1),
        ((32, 512, 512), 0),
    ]

    print(f"PyTorch {torch.__version__}")
    print(f"dtype: {args.dtype}")
    print(f"Trials: {args.trials} x {args.min_time}s min each")
    print()

    header = f"{'shape':<24} {'dim':>4} {'fwd med (min/max) us':>28}"
    if args.backward:
        header += f"  {'fwd+bwd med (min/max) us':>28}"
    print(header)
    print("-" * len(header))

    for shape, dim in cases:
        result = bench(
            shape,
            dim,
            dtype,
            include_backward=args.backward,
            min_time=args.min_time,
            n_trials=args.trials,
        )
        line = f"{str(shape):<24} {dim:>4} {fmt_times(result['fwd']):>28}"
        if args.backward:
            line += f"  {fmt_times(result['fwd_bwd']):>28}"
        print(line)


if __name__ == "__main__":
    main()
