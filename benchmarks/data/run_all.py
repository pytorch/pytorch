#!/usr/bin/env python3
# coding: utf-8

"""
Script to run all benchmarks in the data directory.

This script runs the unified samplers_bench.py script with different
sampler configurations to benchmark all PyTorch sampler implementations.
"""

import argparse
import subprocess
import sys


def run_samplers_benchmark(samplers, avg_times=10, max_combinations=5):
    """Run the unified samplers benchmark with specified parameters."""
    print(f"Running benchmark for samplers: {samplers}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "-m",
        "benchmarks.data.samplers_bench",
        "--samplers",
        samplers,
        "--avg-times",
        str(avg_times),
        "--max-combinations",
        str(max_combinations),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running samplers benchmark: {e}")

    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmarks in the data directory"
    )
    parser.add_argument(
        "--samplers",
        type=str,
        default="BatchSampler,RandomSampler,SequentialSampler,SubsetRandomSampler,WeightedRandomSampler",
        help="Comma-separated list of sampler classes to benchmark",
    )
    parser.add_argument(
        "--avg-times",
        type=int,
        default=10,
        help="Number of times to run each benchmark for averaging",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=5,
        help="Maximum number of parameter combinations to test per sampler",
    )
    args = parser.parse_args()

    # Run unified samplers benchmark with all sampler configurations
    print("\n=== Running Sampler Benchmarks ===\n")

    # Run benchmarks for all specified samplers
    for sampler in args.samplers.split(","):
        run_samplers_benchmark(sampler, args.avg_times, args.max_combinations)


if __name__ == "__main__":
    main()
