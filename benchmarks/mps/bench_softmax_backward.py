"""Compare separate baseline/proposal builds; time backward without forward/autograd.

Run this same file with each build's Python in a fresh process. Record --label
and --patch-sha for an uncommitted build. Memory needs both --shapes fixed and
--shapes varying runs; RSS/driver measurements are evidence, not leak attribution.
Each memory mode uses a separate process. Current RSS and Darwin's ru_maxrss
high-water bytes are labelled separately, with live-reference and cleanup stages.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import timeit
from pathlib import Path


torch = None


CASES = [
    ((512, 768), -1),
    ((4, 4096), -1),
    ((4, 4097), -1),
    ((4, 32768), -1),
    ((4, 65536), -1),
    ((128, 65536), -1),
    ((4, 65536), 0),
    ((8, 256, 1024), 0),
    ((257, 33), 0),
    ((65536, 4), 0),
    # Keep old cases first so their seeded inputs remain unchanged.
    ((128, 257), 0),
    ((129, 257), 0),
    ((128, 127), 0),
    ((128, 128), 0),
    ((63, 65536), -1),
    ((64, 65536), -1),
    ((4, 32000), -1),
    ((8, 50257), -1),
    ((1, 128256), -1),
    ((8, 128256), -1),
    ((128, 128256), -1),
]


def inputs(shape, dim, dtype, layout):
    # Both tensors are rounded on CPU; this does not run MPS forward or RNG kernels.
    output = torch.randn(shape).softmax(dim).to(dtype=dtype).to("mps")
    grad = torch.randn(shape).to(dtype=dtype).to("mps")
    if layout == "transposed":
        grad = grad.transpose(-1, -2).contiguous().transpose(-1, -2)
        output = output.transpose(-1, -2).contiguous().transpose(-1, -2)
    return grad, output


def backward(grad, output, dim):
    return torch.ops.aten._softmax_backward_data(grad, output, dim, output.dtype)


def memory_sample(stage, repeat):
    import psutil

    rss_bytes = psutil.Process().memory_info().rss
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    darwin = platform.system() == "Darwin"
    return {
        "stage": stage,
        "pass": repeat,
        "current_rss_bytes": rss_bytes,
        "current_rss_source": "psutil.Process().memory_info().rss (bytes)",
        "psutil_version": psutil.__version__,
        "peak_rss_native": peak,
        "peak_rss_native_unit": "bytes" if darwin else "KiB",
        "peak_rss_bytes": peak if darwin else peak * 1024,
        "peak_rss_source": "resource.ru_maxrss; high-water mark, never current RSS",
        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        "mps_live_tensor_bytes": torch.mps.current_allocated_memory(),
    }


def check_parity(grad, output, dim, dtype_name):
    expected = torch.ops.aten._softmax_backward_data(
        grad.cpu().float(), output.cpu().float(), dim, torch.float32
    )
    actual = backward(grad, output, dim).cpu().float()
    rtol, atol = {
        "float32": (1e-5, 1e-6),
        "float16": (1e-3, 1e-5),
        "bfloat16": (1.6e-2, 1e-5),
    }[dtype_name]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return {
        "passed": True,
        "reference": "fp32 CPU using the same dtype-rounded inputs",
        "max_absolute_error": (actual - expected).abs().max().item(),
        "rtol": rtol,
        "atol": atol,
        "limit": "One deterministic sample, not full semantic validation.",
    }


def tensor_details(grad, output):
    return {
        name: {
            "shape": list(t.shape),
            "stride": list(t.stride()),
            "storage_offset": t.storage_offset(),
            "dtype": str(t.dtype),
        }
        for name, t in (("grad", grad), ("output", output))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument(
        "--expected-sha", help="Exact torch.version.git_version of the intended build"
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--patch-sha", default="")
    parser.add_argument("--mode", choices=("timing", "memory"), default="timing")
    parser.add_argument("--family", choices=("last", "nonlast", "all"), default="all")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--layout", choices=("contiguous", "transposed"), default="contiguous"
    )
    parser.add_argument(
        "--shapes",
        choices=("fixed", "varying"),
        help="Required in memory mode; one fresh process per mode",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--min-run-time", type=float, default=0.3)
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps(CASES, indent=2))
        return
    if not args.label or not args.expected_sha:
        parser.error("runtime measurements require --label and --expected-sha")
    if args.mode == "memory" and (args.family == "all" or args.shapes is None):
        parser.error(
            "memory mode requires --family last|nonlast and explicit --shapes fixed|varying"
        )
    if (
        min(args.steps, args.passes, args.block_size, args.trials) <= 0
        or args.warmup < 0
        or args.min_run_time <= 0
    ):
        parser.error(
            "counts must be positive, warmup nonnegative, and min-run-time positive"
        )
    global torch
    import torch
    from torch.utils.benchmark import Timer

    if torch.version.git_version != args.expected_sha:
        raise RuntimeError(
            f"Expected build {args.expected_sha}, found {torch.version.git_version} at {torch.__file__}"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("An MPS build and device are required")
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    record = {
        "arguments": vars(args),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_git_version": torch.version.git_version,
        "torch_version": torch.__version__,
        "torch_file": torch.__file__,
        "device": "mps",
        "pid": os.getpid(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_brand": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "macos_version": platform.mac_ver()[0],
        "identity_limit": "git_version verifies the imported revision; a supplied dirty patch hash is attribution, not compiled-source proof.",
        "measurement_scope": "Raw backward excludes MPS forward and autograd; no automatic leak or regression acceptance conclusion.",
        "timing_statistic": "Arithmetic mean of Timer.blocked_autorange trial means",
        "timing_boundaries": "Custom timer synchronizes MPS before each timing boundary",
        "results": [],
    }
    if args.mode == "timing":
        for shape, dim in CASES:
            if args.family != "all" and (dim == -1) != (args.family == "last"):
                continue
            grad, output = inputs(shape, dim, dtype, args.layout)
            try:
                parity = check_parity(grad, output, dim, args.dtype)
            except AssertionError as error:
                record["results"].append(
                    {
                        "shape": shape,
                        "dim": dim,
                        "status": "failed_no_timing_claim",
                        "error": str(error),
                    }
                )
                continue

            def block():
                for _ in range(args.block_size):
                    backward(grad, output, dim)

            def timer():
                torch.mps.synchronize()
                return timeit.default_timer()

            for _ in range(args.warmup):
                block()
            torch.mps.synchronize()
            measurements = [
                Timer(
                    stmt="block()", globals={"block": block}, timer=timer
                ).blocked_autorange(min_run_time=args.min_run_time)
                for _ in range(args.trials)
            ]
            samples = [m.mean * 1e6 / args.block_size for m in measurements]
            record["results"].append(
                {
                    "shape": shape,
                    "dim": dim,
                    "status": "measured",
                    "samples_us": samples,
                    "mean_us_per_call": statistics.mean(samples),
                    "cpu_parity": parity,
                    "tensors": tensor_details(grad, output),
                    "raw_trials": [
                        {
                            "raw_batch_times_seconds": list(m.raw_times),
                            "number_per_run": m.number_per_run,
                        }
                        for m in measurements
                    ],
                }
            )
    else:
        dim = -1 if args.family == "last" else 0
        shapes = [
            (32 + index, 48 + index) if args.shapes == "varying" else (32, 48)
            for index in range(args.steps)
        ]
        record.update(
            shape_schedule_per_pass=shapes,
            dim=dim,
            setup_scope="CPU input generation/rounding and transfer repeated in both controls",
            warmup_scope="Backward calls on the initial shape; no MPS forward/autograd",
        )

        def checkpoint(stage, repeat):
            record["results"].append(memory_sample(stage, repeat))

        def cleanup_checkpoints(repeat):
            checkpoint("after_del", repeat)
            gc.collect()
            checkpoint("after_gc", repeat)
            torch.mps.synchronize()
            checkpoint("after_synchronize", repeat)
            torch.mps.empty_cache()
            checkpoint("after_empty_cache", repeat)
            torch.mps.synchronize()
            checkpoint("after_empty_cache_synchronize", repeat)

        checkpoint("before_setup", 0)
        grad, output = inputs((32, 48), dim, dtype, args.layout)
        record["cpu_parity_initial_shape"] = check_parity(grad, output, dim, args.dtype)
        record["initial_tensors"] = tensor_details(grad, output)
        result = None
        for _ in range(args.warmup):
            result = backward(grad, output, dim)
        torch.mps.synchronize()
        checkpoint("warmup_inputs_and_output_live", 0)
        del grad, output, result
        cleanup_checkpoints(0)
        for repeat in range(1, args.passes + 1):
            checkpoint("before_pass", repeat)
            for index, shape in enumerate(shapes):
                grad, output = inputs(shape, dim, dtype, args.layout)
                result = backward(grad, output, dim)
                torch.mps.synchronize()
                if index == len(shapes) - 1:
                    checkpoint("last_shape_inputs_and_output_live", repeat)
                    record["last_shape_tensors"] = tensor_details(grad, output)
                del grad, output, result
            cleanup_checkpoints(repeat)
    print(json.dumps(record, indent=2))
    if any(row.get("status") == "failed_no_timing_claim" for row in record["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
