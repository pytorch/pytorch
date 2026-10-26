"""Compare resident CPU, native MPS, and explicit MPS-to-CPU-to-MPS pooling.

    python benchmarks/mps/adaptive_avg_pool2d.py --num-threads 4 --min-run-time 0
    python benchmarks/mps/adaptive_avg_pool2d.py --dtype float16 --case large_mixed

Forward disables autograd; backward reuses a graph; forward_backward builds a
fresh graph. The round trip includes transfers, including gradients returning to
the original MPS leaf. Other paths keep tensors resident. Correctness and warmup
are untimed. Paths rotate order with identical fixed-size timing blocks;
MPS synchronizes at block boundaries. Set --min-run-time 0 for exactly --repeats
blocks per path. Times include dispatch, allocation, and Python overhead.
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F


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
PHASES = ("forward", "backward", "forward_backward")


def cpu_roundtrip(x, output_size):
    return F.adaptive_avg_pool2d(x.cpu(), output_size).to("mps")


def training_step(pool, x, output_size, grad):
    return torch.autograd.grad(pool(x, output_size), x, grad)[0]


def check(actual, expected):
    actual = actual.detach().cpu()
    error = None
    try:
        torch.testing.assert_close(actual, expected)
    except AssertionError as exception:
        error = str(exception)
    return {
        "passed": error is None,
        "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
        "error": error,
    }


def measure(functions, args):
    paths = list(functions)
    for path, fn in functions.items():
        for _ in range(args.warmup):
            fn()
        if path != "cpu":
            torch.mps.synchronize()
    samples = {path: [] for path in paths}
    orders = []
    started = time.perf_counter()
    repeat = 0
    while repeat < args.repeats or time.perf_counter() - started < args.min_run_time:
        order = paths[repeat % len(paths) :] + paths[: repeat % len(paths)]
        orders.append(order)
        for path in order:
            if path != "cpu":
                torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                output = functions[path]()
            if path != "cpu":
                torch.mps.synchronize()
            samples[path].append((time.perf_counter() - start) * 1e6 / args.iterations)
            del output
        repeat += 1
    results = {}
    for path, values in samples.items():
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        results[path] = {
            "median_us": statistics.median(values),
            "iqr_us": quartiles[2] - quartiles[0],
            "samples_us": values,
        }
    return {"paths": results, "orders": orders, "iterations": args.iterations}


def benchmark_case(shape, output_size, args):
    memory_format = (
        torch.channels_last
        if args.memory_format == "channels_last"
        else torch.contiguous_format
    )
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    cpu_input = (
        torch.randn(shape, generator=generator)
        .to(dtype)
        .contiguous(memory_format=memory_format)
    )
    cpu_grad = torch.randn((*shape[:2], *output_size), generator=generator).to(dtype)
    cpu_grad = cpu_grad.contiguous(memory_format=memory_format)
    reference_input = cpu_input.detach().requires_grad_()
    reference_output = F.adaptive_avg_pool2d(reference_input, output_size)
    reference_grad = torch.autograd.grad(reference_output, reference_input, cpu_grad)[0]
    functions = {phase: {} for phase in PHASES}
    checks, strides = {}, {}
    paths = ("cpu",) if args.cpu_only else ("cpu", "mps", "cpu_roundtrip")
    for path in paths:
        device = "cpu" if path == "cpu" else "mps"
        pool = cpu_roundtrip if path == "cpu_roundtrip" else F.adaptive_avg_pool2d
        x = cpu_input.to(device).detach().requires_grad_()
        grad = cpu_grad.to(device)
        y = pool(x, output_size)
        actual_grad = torch.autograd.grad(y, x, grad, retain_graph=True)[0]
        functions["forward"][path] = partial(pool, x.detach(), output_size)
        functions["backward"][path] = partial(
            torch.autograd.grad, y, x, grad, retain_graph=True
        )
        functions["forward_backward"][path] = partial(
            training_step, pool, x, output_size, grad
        )
        checks[path] = {
            "forward": check(y, reference_output.detach()),
            "backward": check(actual_grad, reference_grad),
            "forward_backward": check(
                functions["forward_backward"][path](), reference_grad
            ),
        }
        strides[path] = {"input": list(x.stride()), "output": list(y.stride())}
    timings = {phase: measure(functions[phase], args) for phase in PHASES}
    return {"checks": checks, "strides": strides, "timings": timings}


def metadata(args):
    cpu_name = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        cpu_name = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    result = {
        "torch_version": torch.__version__,
        "torch_git": torch.version.git_version,
        "platform": platform.platform(),
        "cpu": cpu_name,
        "logical_cpus": os.cpu_count(),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_LOG_PROFILE_INFO",
                "PYTORCH_MPS_TRACE_SIGNPOSTS",
            )
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": vars(args),
        "correctness": "torch.testing.assert_close default dtype tolerances; failures recorded, not hidden",
        "timing": "Interleaved equal-size blocks; per-call synchronized wall times when iterations=1",
        "min_run_time": "Minimum combined elapsed time per phase; 0 fixes repeat count",
    }
    if not args.cpu_only:
        result.update(
            mps=torch.backends.mps.get_name(),
            gpu_cores=torch.backends.mps.get_core_count(),
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument(
        "--memory-format", choices=("contiguous", "channels_last"), default="contiguous"
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--case", action="append", choices=[name for name, _, _ in CASES]
    )
    parser.add_argument("--num-threads", type=int, default=torch.get_num_threads())
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json", help="Save metadata, raw samples, and correctness results."
    )
    parser.add_argument(
        "--expect-git", help="Require this installed torch git version."
    )
    args = parser.parse_args()
    if (
        args.num_threads < 1
        or args.min_run_time < 0
        or args.warmup < 0
        or args.repeats < 2
        or args.iterations < 1
    ):
        parser.error(
            "Require positive threads/iterations, repeats >= 2, and nonnegative runtime/warmup"
        )
    if args.expect_git and torch.version.git_version != args.expect_git:
        parser.error(
            f"Expected build {args.expect_git}, got {torch.version.git_version}"
        )
    if not args.cpu_only and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --cpu-only for the CPU baseline")
    if not args.cpu_only and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (
        None,
        "",
        "0",
    ):
        parser.error("Unset PYTORCH_ENABLE_MPS_FALLBACK to benchmark native MPS")
    torch.set_num_threads(args.num_threads)
    report = {"metadata": metadata(args), "cases": {}}
    print("# " + json.dumps(report["metadata"]), flush=True)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        (
            "case",
            "phase",
            "path",
            "median_us",
            "iqr_us",
            "repeats",
            "correctness_passed",
        )
    )
    for name, shape, output_size in CASES:
        if args.case and name not in args.case:
            continue
        results = benchmark_case(shape, output_size, args)
        report["cases"][name] = {"shape": shape, "output_size": output_size, **results}
        for phase, timing in results["timings"].items():
            for path, values in timing["paths"].items():
                passed = results["checks"][path][phase]["passed"]
                writer.writerow(
                    (
                        name,
                        phase,
                        path,
                        f"{values['median_us']:.3f}",
                        f"{values['iqr_us']:.3f}",
                        len(values["samples_us"]),
                        passed,
                    )
                )
                if not passed:
                    print(
                        f"Correctness mismatch: {name}/{phase}/{path}; details available with --json",
                        file=sys.stderr,
                    )
        sys.stdout.flush()
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    report["completed"] = True
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
