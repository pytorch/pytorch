"""Compare eager/compiled dispatchless workloads on CPU and CUDA.

All variants run the same pointwise pipeline, with optional nested pytree
arguments/results and repeated operator calls. CUDA host timings exclude the
final synchronization; end-to-end timings include it. Bounded batches limit
queue buildup, but host timings can still include backpressure.

Compilation timings use fresh and then warm local caches, not fresh processes.
Generated kernel counts exclude eager kernels inside opaque custom operators.
The clone microbenchmark in dispatchless_custom_op.py isolates eager overhead.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import config as inductor_config, metrics
from torch._inductor.utils import clear_caches, fresh_cache, run_and_get_code
from torch._library.dispatchless import dispatchless_custom_op


def pointwise(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x.sin() * scale


def nested_pointwise(payload, *, scale):
    return {"output": pointwise(payload["tensors"][0], scale), "metadata": (None, scale)}


def make_variants(workload: str, custom) -> dict[str, Callable]:
    if workload == "flat":
        return {
            "plain": pointwise,
            "custom_op": custom,
            "dispatchless_custom_op": dispatchless_custom_op(pointwise),
        }

    def nested_custom(payload, *, scale):
        return {"output": custom(payload["tensors"][0], scale), "metadata": (None, scale)}

    return {
        "plain": nested_pointwise,
        "custom_op": nested_custom,
        "dispatchless_custom_op": dispatchless_custom_op(nested_pointwise),
    }


def make_pipeline(op: Callable, workload: str, calls: int) -> Callable:
    def pipeline(x):
        x = x + 1
        for _ in range(calls):
            if workload == "flat":
                x = op(x, 2.0)
            else:
                x = op({"tensors": [x]}, scale=2.0)["output"]
        return x.cos() + 3

    return pipeline


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(samples: list[float]) -> dict[str, float | int]:
    quartiles = (
        statistics.quantiles(samples, n=4, method="inclusive")
        if len(samples) > 1
        else [samples[0]] * 3
    )
    return {
        "median_us": statistics.median(samples) * 1e6,
        "iqr_us": (quartiles[2] - quartiles[0]) * 1e6,
        "blocks": len(samples),
    }


def measure(fn: Callable, x: torch.Tensor, args: argparse.Namespace) -> dict[str, Any]:
    for _ in range(10):
        fn(x)
    host_samples, synchronized_samples = [], []
    for _ in range(args.repeats):
        deadline = time.perf_counter() + args.min_run_time
        while True:
            synchronize(x.device)
            start = time.perf_counter()
            for _ in range(args.batch_size):
                fn(x)
            host_end = time.perf_counter()
            synchronize(x.device)
            end = time.perf_counter()
            host_samples.append((host_end - start) / args.batch_size)
            synchronized_samples.append((end - start) / args.batch_size)
            if end >= deadline:
                break
    result = {"host": summarize(host_samples)}
    if x.device.type == "cuda":
        result["synchronized"] = summarize(synchronized_samples)
    return result


def cache_counters() -> dict[str, dict[str, int]]:
    return {
        group: {key: counters[group][key] for key in keys}
        for group, keys in {
            "aot_autograd": (
                "autograd_cache_hit",
                "autograd_cache_miss",
                "autograd_cache_bypass",
                "autograd_cache_saved",
            ),
            "inductor": (
                "fxgraph_cache_hit",
                "fxgraph_cache_miss",
                "fxgraph_cache_bypass",
            ),
        }.items()
    }


def compile_phase(fn: Callable, x: torch.Tensor) -> tuple[Callable, dict[str, Any]]:
    torch._dynamo.reset()
    clear_caches()
    counters.clear()
    metrics.reset()
    compiled = torch.compile(fn, fullgraph=True)
    synchronize(x.device)
    start = time.perf_counter()
    result = compiled(x)
    synchronize(x.device)
    elapsed = time.perf_counter() - start
    report = {"first_call_s": elapsed, "cache_counters": cache_counters()}
    torch.testing.assert_close(result, fn(x))
    return compiled, report


def benchmark_variant(
    fn: Callable, x: torch.Tensor, args: argparse.Namespace
) -> dict[str, Any]:
    eager = measure(fn, x, args)
    with fresh_cache():
        _, cold = compile_phase(fn, x)
        compiled, warm = compile_phase(fn, x)
        value = x + 0.125
        torch.testing.assert_close(compiled(value), fn(value))
        compiled_times = measure(compiled, x, args)
        metrics.reset()
        with (
            inductor_config.patch(fx_graph_cache=False),
            functorch_config.patch(enable_autograd_cache=False),
        ):
            result, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        if not code:
            raise AssertionError("No generated code was captured for fusion diagnostics")
        torch.testing.assert_close(result, fn(x))
        fusion = {
            "generated_kernel_count": metrics.generated_kernel_count,
            "opaque_custom_op_calls": sum(
                source.count("torch.ops.dispatchless_workloads.pointwise.default(")
                for source in code
            ),
            "dispatchless_call_remains": any(
                name in source
                for source in code
                for name in ("flat_dispatchless_call(", "flat_dispatchless_graph_call(")
            ),
        }
    return {
        "eager": eager,
        "compiled": compiled_times,
        "fresh_local_cache_compile": cold,
        "warm_local_cache_recompile": warm,
        "fusion": fusion,
    }


def benchmark_case(
    custom, device: str, workload: str, size: int, calls: int, args: argparse.Namespace
) -> dict[str, Any]:
    # Citrine C3: Allocate workload inputs directly on the measured device.
    x = torch.linspace(-1, 1, size, device=device, dtype=torch.float32)
    variants = make_variants(workload, custom)
    expected = make_pipeline(variants["plain"], workload, calls)(x)
    results = {}
    for name, op in variants.items():
        fn = make_pipeline(op, workload, calls)
        torch.testing.assert_close(fn(x), expected)
        results[name] = benchmark_variant(fn, x, args)
        fusion = results[name]["fusion"]
        if name == "custom_op" and fusion["opaque_custom_op_calls"] != calls:
            raise AssertionError(f"Expected {calls} opaque operator calls: {fusion}")
        if name != "custom_op" and (
            fusion["generated_kernel_count"] != 1
            or fusion["opaque_custom_op_calls"] != 0
            or fusion["dispatchless_call_remains"]
        ):
            raise AssertionError(f"Expected a single fused kernel for {name}: {fusion}")
    for name, result in results.items():
        for execution in ("eager", "compiled"):
            for timing, summary in result[execution].items():
                baseline = results["plain"][execution][timing]["median_us"]
                custom_us = results["custom_op"][execution][timing]["median_us"]
                summary["overhead_vs_plain_us"] = summary["median_us"] - baseline
                summary["speedup_vs_custom_op"] = custom_us / summary["median_us"]
    return {
        "device": device,
        "workload": workload,
        "numel": size,
        "calls": calls,
        "variants": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", choices=("cpu", "cuda"), nargs="+", default=["cpu"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 1024])
    parser.add_argument("--calls", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--workloads", choices=("flat", "nested"), nargs="+", default=["flat", "nested"]
    )
    parser.add_argument("--min-run-time", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if (
        not math.isfinite(args.min_run_time)
        or args.min_run_time <= 0
        or args.repeats < 1
        or args.batch_size < 1
        or any(value < 1 for value in args.sizes + args.calls)
    ):
        parser.error("timing parameters, sizes, and call counts must be positive")
    if "cuda" in args.devices and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.output_json and Path(args.output_json).exists():
        parser.error("output file already exists; choose a new path")
    return args


def run_benchmarks(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    custom = torch.library.custom_op(
        "dispatchless_workloads::pointwise", pointwise, mutates_args=()
    )
    custom.register_fake(pointwise)
    results = []
    with (
        torch.no_grad(),
        inductor_config.patch(fx_graph_cache=True, fx_graph_remote_cache=False),
        functorch_config.patch(
            enable_autograd_cache=True, enable_remote_autograd_cache=False
        ),
    ):
        for device, workload, size, calls in product(
            args.devices, args.workloads, args.sizes, args.calls
        ):
            print(
                f"Benchmarking {device} {workload} numel={size} calls={calls}",
                file=sys.stderr,
            )
            results.append(benchmark_case(custom, device, workload, size, calls, args))
    return results


def main() -> None:
    args = parse_args()
    results = run_benchmarks(args)
    report = {
        "environment": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "host": platform.node(),
            "threads": torch.get_num_threads(),
            "cuda_device": (
                torch.cuda.get_device_name() if "cuda" in args.devices else None
            ),
            "dtype": "float32",
            "grad_mode": "no_grad",
            "parameters": vars(args),
        },
        "notes": [
            "Timings are per whole pipeline, not per operator; registration, correctness "
            "checks, and compilation are excluded from steady-state timing.",
            "Nested custom_op uses a pytree adapter because dispatcher schemas cannot "
            "represent arbitrary nested pytrees.",
            "CUDA synchronized timing is amortized end-to-end batch latency, "
            "not CUDA-event kernel time.",
            "Generated kernel counts exclude kernels launched inside opaque custom_op calls.",
            "Fresh/warm local-cache compilations share a process; cache hit or bypass "
            "outcomes are reported rather than assumed.",
            "The first fresh-cache call can include one-time framework/compiler startup.",
            "IQR describes pooled block timings, not confidence intervals; "
            "host drift and CUDA backpressure may affect comparisons.",
        ],
        "results": results,
    }
    if args.output_json:
        with open(args.output_json, "x", encoding="utf-8", newline="\n") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
