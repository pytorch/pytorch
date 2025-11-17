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


def run_benchmark(
    benchmark_name: str,
    should_visualize: bool = False,
    compile_mode: str = "max-autotune-no-cudagraphs",
):
    """Run a specific benchmark."""
    if benchmark_name not in BENCHMARK_REGISTRY:
        print(f"Error: Unknown benchmark '{benchmark_name}'")
        print("Use --list to see available benchmarks")
        return False

    print(f"Running benchmark: {benchmark_name}")
    print(f"Torch compile mode: {compile_mode}")
    print("=" * 60)

    benchmark_class = BENCHMARK_REGISTRY[benchmark_name]
    benchmark = benchmark_class(compile_mode)
    benchmark.benchmark()
    if should_visualize:
        benchmark.visualize()

    return True


def run_all_benchmarks(should_visualize: bool = False, compile_mode: str = "default"):
    """Run all available benchmarks."""
    print("Running all benchmarks...")
    print(f"Torch compile mode: {compile_mode}")
    print("=" * 60)

    for name, cls in BENCHMARK_REGISTRY.items():
        print(f"\n{'=' * 20} {name.upper()} {'=' * 20}")
        benchmark = cls(compile_mode)
        benchmark.benchmark()
        if should_visualize:
            benchmark.visualize()
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

    args = parser.parse_args()

    # Handle list option
    if args.list:
        list_benchmarks()
        return

    # Handle all option
    if args.all:
        run_all_benchmarks(args.visualize, args.compile_mode)
        return

    # Handle specific benchmarks
    if not args.benchmarks:
        print("Error: No benchmarks specified")
        print("Use --list to see available benchmarks or --all to run all benchmarks")
        parser.print_help()
        sys.exit(1)

    for benchmark_name in args.benchmarks:
        run_benchmark(benchmark_name, args.visualize, args.compile_mode)
        print()  # Add spacing between benchmarks


if __name__ == "__main__":
    main()
