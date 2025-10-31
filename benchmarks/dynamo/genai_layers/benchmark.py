#!/usr/bin/env python3
"""
Benchmark runner for various kernel implementations.

This script provides a command-line interface to run benchmarks for different
kernel implementations including CrossEntropy, Softmax, RMSNorm, and LayerNorm
kernels in both forward and backward directions.
"""

import argparse
import sys

from kernels import (
    BenchmarkKernel,
    CrossEntropyBackward,
    CrossEntropyForward,
    LayerNormBackward,
    LayerNormForward,
    RMSNormBackward,
    RMSNormForward,
    SoftmaxBackward,
    SoftmaxForward,
)

import torch


torch._dynamo.config.automatic_dynamic_shapes = False
# Needed since changing args to function causes recompiles
torch._dynamo.config.recompile_limit = 1000000


# Registry of all available benchmarks
BENCHMARK_REGISTRY: dict[str, type[BenchmarkKernel]] = {
    "cross_entropy_forward": CrossEntropyForward,
    "cross_entropy_backward": CrossEntropyBackward,
    "softmax_forward": SoftmaxForward,
    "softmax_backward": SoftmaxBackward,
    "rmsnorm_forward": RMSNormForward,
    "rmsnorm_backward": RMSNormBackward,
    "layernorm_forward": LayerNormForward,
    "layernorm_backward": LayerNormBackward,
}


def show_environment_info():
    """Show environment information."""
    print("Environment information:")
    print(f"  Python version: {sys.version}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")


def list_benchmarks():
    """List all available benchmarks."""
    print(f"Available benchmarks: {list(BENCHMARK_REGISTRY.keys())}")


def _run_benchmark(
    benchmark_cls,
    script_args,
):
    benchmark = benchmark_cls(script_args)
    benchmark.benchmark()
    benchmark.report_geomean_speedup()
    if script_args.print_benchmark_result:
        print(f"Benchmarking results {benchmark.name}:")
        print(benchmark.profiling_results)
    if script_args.visualize:
        benchmark.visualize()


def run_benchmark(
    benchmark_name: str,
    script_args,
):
    """Run a specific benchmark."""
    if benchmark_name not in BENCHMARK_REGISTRY:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print("Use --list to see available benchmarks")
        return False

    print(f"Running benchmark: {benchmark_name}")
    print(f"Torch compile mode: {script_args.compile_mode}")
    print("=" * 60)

    benchmark_class = BENCHMARK_REGISTRY[benchmark_name]
    _run_benchmark(benchmark_class, script_args)

    return True


def run_all_benchmarks(script_args):
    """Run all available benchmarks."""
    print("Running all benchmarks...")
    print(f"Torch compile mode: {script_args.compile_mode}")
    print("=" * 60)

    for name, cls in BENCHMARK_REGISTRY.items():
        print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")
        _run_benchmark(cls, script_args)
        print()


def main():
    show_environment_info()

    parser = argparse.ArgumentParser(
        description="Benchmark runner for kernel implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --list                    # List all available benchmarks
  python benchmark.py --all                     # Run all benchmarks
  python benchmark.py cross_entropy_forward     # Run specific benchmark
  python benchmark.py softmax_forward softmax_backward  # Run multiple benchmarks
        """,
    )

    parser.add_argument(
        "benchmarks",
        nargs="*",
        help="Names of benchmarks to run (use --list to see available options)",
    )

    parser.add_argument(
        "--list", action="store_true", help="List all available benchmarks"
    )

    parser.add_argument(
        "--all", action="store_true", help="Run all available benchmarks"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize results after running benchmarks",
    )

    parser.add_argument(
        "--compile-mode",
        choices=["default", "max-autotune-no-cudagraphs"],
        default="max-autotune-no-cudagraphs",
        help="Torch compile mode to use (default: default)",
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Tolerance for the accuracy check",
    )

    parser.add_argument(
        "--exit-on-accuracy-failure",
        action="store_true",
        help="Whether to exit with an error message for accuracy failure",
    )

    parser.add_argument(
        "--print-benchmark-result",
        action="store_true",
        help="Whether to print the raw benchmarking result. Easier to quickly check the benchmark results on a server without GUI",
    )

    parser.add_argument(
        "--custom-compile-name",
        type=str,
        default=None,
        help="Name for the curve with customized compilation options",
    )

    parser.add_argument(
        "--custom-compile-options",
        type=str,
        default=None,
        help="Json string for the custom compile options.",
    )

    args = parser.parse_args()

    if args.custom_compile_options:
        import json

        try:
            args.custom_compile_options = json.loads(args.custom_compile_options)
        except json.decoder.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid json string for --custom-compile-options: {args.custom_compile_options}"
            ) from e

        if not args.custom_compile_options:
            raise RuntimeError("Found no options for --custom-compile-options")
        if not args.custom_compile_name:
            raise RuntimeError("Missing label name for the custom compilation")

    # Handle list option
    if args.list:
        list_benchmarks()
        return

    # Handle all option
    if args.all:
        run_all_benchmarks(args)
        return

    # Handle specific benchmarks
    if not args.benchmarks:
        print("Error: No benchmarks specified")
        print("Use --list to see available benchmarks or --all to run all benchmarks")
        parser.print_help()
        sys.exit(1)

    for benchmark_name in args.benchmarks:
        run_benchmark(benchmark_name, args)
        print()  # Add spacing between benchmarks


if __name__ == "__main__":
    main()
