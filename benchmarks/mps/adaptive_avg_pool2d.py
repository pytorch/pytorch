"""Compare adaptive_avg_pool2d on CPU and MPS.

Run from a checkout with the matching PyTorch build installed:
    python benchmarks/mps/adaptive_avg_pool2d.py --num-threads 1
    python benchmarks/mps/adaptive_avg_pool2d.py --memory-format channels_last

Forward uses inputs without autograd. Backward uses a precomputed graph and
does not accumulate gradients. Transfers and correctness checks are untimed.
MPS is synchronized at timing block boundaries; times include dispatch costs.
"""

import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


CASES = (
    ("small_downsample", (1, 8, 31, 29), (7, 5)),
    ("large_downsample", (16, 64, 127, 119), (31, 29)),
    ("small_upsample", (1, 8, 7, 5), (13, 11)),
    ("large_upsample", (8, 64, 31, 29), (63, 61)),
    ("small_mixed", (1, 8, 31, 7), (13, 15)),
    ("large_mixed", (8, 64, 127, 31), (61, 63)),
    ("small_divisible", (1, 8, 32, 32), (8, 8)),
    ("large_divisible", (16, 64, 128, 128), (32, 32)),
)


def mps_timer():
    torch.mps.synchronize()
    return time.perf_counter()


def measure(fn, device, args):
    for _ in range(args.warmup):
        fn()
    return Timer(
        stmt="fn()",
        globals={"fn": fn},
        timer=mps_timer if device == "mps" else time.perf_counter,
        num_threads=args.num_threads,
    ).blocked_autorange(min_run_time=args.min_run_time)


def benchmark_case(shape, output_size, args):
    memory_format = (
        torch.channels_last
        if args.memory_format == "channels_last"
        else torch.contiguous_format
    )
    generator = torch.Generator().manual_seed(args.seed)
    cpu_input = torch.randn(shape, generator=generator).contiguous(
        memory_format=memory_format
    )
    cpu_grad = torch.randn((*shape[:2], *output_size), generator=generator).contiguous(
        memory_format=memory_format
    )
    reference_input = cpu_input.detach().requires_grad_()
    reference_output = F.adaptive_avg_pool2d(reference_input, output_size)
    reference_grad = torch.autograd.grad(reference_output, reference_input, cpu_grad)[0]

    results = {}
    for device in ("cpu",) if args.cpu_only else ("cpu", "mps"):
        x = cpu_input.to(device).detach().requires_grad_()
        grad = cpu_grad.to(device)
        y = F.adaptive_avg_pool2d(x, output_size)
        actual_grad = torch.autograd.grad(y, x, grad, retain_graph=True)[0]
        torch.testing.assert_close(y.detach().cpu(), reference_output.detach())
        torch.testing.assert_close(actual_grad.cpu(), reference_grad)
        forward = partial(F.adaptive_avg_pool2d, x.detach(), output_size)
        backward = partial(torch.autograd.grad, y, x, grad, retain_graph=True)
        results[device] = {
            "forward": measure(forward, device, args),
            "backward": measure(backward, device, args),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run only the CPU baseline."
    )
    parser.add_argument(
        "--memory-format", choices=("contiguous", "channels_last"), default="contiguous"
    )
    parser.add_argument("--num-threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_threads < 1 or args.min_run_time <= 0 or args.warmup < 0:
        parser.error("Require num-threads > 0, min-run-time > 0 and warmup >= 0")
    if not args.cpu_only and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --cpu-only for the CPU baseline")
    if not args.cpu_only and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Unset PYTORCH_ENABLE_MPS_FALLBACK to benchmark native MPS")

    torch.set_num_threads(args.num_threads)
    cpu_name = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        cpu_name = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    print(f"# torch={torch.__version__}, git={torch.version.git_version}")
    print(f"# platform={platform.platform()}")
    print(f"# cpu={cpu_name}, logical_cpus={os.cpu_count()}")
    if not args.cpu_only:
        print(
            f"# mps={torch.backends.mps.get_name()}, "
            f"gpu_cores={torch.backends.mps.get_core_count()}"
        )
    print(
        f"# dtype=float32, memory_format={args.memory_format}, "
        f"cpu_threads={torch.get_num_threads()}, seed={args.seed}, "
        f"warmup={args.warmup}, min_run_time={args.min_run_time}s"
    )
    print("# Outputs and input gradients checked against CPU before timing.")
    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "case",
            "input_shape",
            "output_size",
            "phase",
            "cpu_median_us",
            "cpu_iqr_us",
            "mps_median_us",
            "mps_iqr_us",
            "cpu_over_mps",
        )
    )
    for name, shape, output_size in CASES:
        results = benchmark_case(shape, output_size, args)
        for phase in ("forward", "backward"):
            cpu = results["cpu"][phase]
            row = [
                name,
                "x".join(map(str, shape)),
                "x".join(map(str, output_size)),
                phase,
                f"{cpu.median * 1e6:.3f}",
                f"{cpu.iqr * 1e6:.3f}",
            ]
            if args.cpu_only:
                row.extend(("", "", ""))
            else:
                mps = results["mps"][phase]
                row.extend(
                    (
                        f"{mps.median * 1e6:.3f}",
                        f"{mps.iqr * 1e6:.3f}",
                        f"{cpu.median / mps.median:.3f}",
                    )
                )
            writer.writerow(row)


if __name__ == "__main__":
    main()
