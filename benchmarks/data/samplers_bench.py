#!/usr/bin/env python3
# coding: utf-8

import argparse
import importlib
import inspect
import time
from collections.abc import Iterable, Iterator, Sized
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch
from tabulate import tabulate
from torch.utils.data import (
    BatchSampler,
    RandomSampler,
    Sampler,
    SequentialSampler,
    SubsetRandomSampler,
    WeightedRandomSampler,
)

# Dictionary mapping original sampler classes to their alternative implementations
ALTERNATIVE_IMPLEMENTATIONS = {
    RandomSampler: SequentialSampler,
    # Add more alternative implementations as they are developed
}


def get_sampler_params(sampler_class: Type[Sampler]) -> Dict[str, Dict[str, Any]]:
    """Get default parameter configurations for different sampler classes.

    Returns a dictionary mapping parameter names to dictionaries of test values.
    """
    # Common dataset size for all samplers
    data_size = 1000000

    if sampler_class == SequentialSampler:
        return {
            "data_source": {"values": [range(data_size)]},
        }

    elif sampler_class == RandomSampler:
        return {
            "data_source": {"values": [range(data_size)]},
            "replacement": {"values": [True, False]},
            "num_samples": {"values": [1000, 10000, 100000, None]},
        }

    elif sampler_class == BatchSampler:
        return {
            "sampler": {"values": [SequentialSampler(range(data_size))]},
            "batch_size": {"values": [4, 8, 64, 640, 6400]},
            "drop_last": {"values": [True, False]},
        }

    elif sampler_class == SubsetRandomSampler:
        return {
            "indices": {"values": [list(range(1000)), list(range(10000))]},
        }

    elif sampler_class == WeightedRandomSampler:
        return {
            "weights": {"values": [torch.ones(1000), torch.rand(10000)]},
            "num_samples": {"values": [1000, 10000]},
            "replacement": {"values": [True, False]},
        }

    else:
        # For custom samplers, provide a basic configuration
        # Users can extend this function to add specific configurations
        sig = inspect.signature(sampler_class.__init__)
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param_name == "data_source" or param_name == "dataset":
                params[param_name] = {"values": [range(data_size)]}
            elif param_name == "batch_size":
                params[param_name] = {"values": [4, 64, 640]}
            elif param_name == "drop_last":
                params[param_name] = {"values": [True, False]}
            elif param_name == "replacement":
                params[param_name] = {"values": [True, False]}
            elif param_name == "num_samples":
                params[param_name] = {"values": [1000, 10000]}
            else:
                # For unknown parameters, provide a default value if available
                if param.default is not inspect.Parameter.empty:
                    params[param_name] = {"values": [param.default]}

        return params


def generate_param_combinations(
    params: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters for testing."""
    if not params:
        return [{}]

    # Start with the first parameter
    param_name = next(iter(params))
    param_values = params[param_name]["values"]

    # Generate combinations for the rest of the parameters
    rest_params = {k: v for k, v in params.items() if k != param_name}
    rest_combinations = generate_param_combinations(rest_params)

    # Combine with the current parameter
    combinations = []
    for value in param_values:
        for combo in rest_combinations:
            new_combo = {param_name: value, **combo}
            combinations.append(new_combo)

    return combinations


def benchmark_sampler(
    sampler_class: Type[Sampler],
    params: Dict[str, Any],
    avg_times: int = 10,
    alternative_class: Optional[Type[Sampler]] = None,
) -> Dict[str, Any]:
    """Benchmark a sampler with given parameters.

    Args:
        sampler_class: The sampler class to benchmark
        params: Parameters to pass to the sampler constructor
        avg_times: Number of times to run the benchmark for averaging
        alternative_class: Optional alternative implementation to compare against

    Returns:
        Dictionary with benchmark results
    """
    # Create a readable parameter string for display
    param_str = ", ".join(
        f"{k}={v}"
        for k, v in params.items()
        if k not in ["data_source", "sampler", "indices", "weights"]
    )

    print(f"Benchmarking {sampler_class.__name__}({param_str})")

    # Benchmark original implementation
    original_times = []
    for _ in range(avg_times):
        try:
            start = time.perf_counter()
            sampler = sampler_class(**params)
            # Consume the iterator
            list(iter(sampler))
            end = time.perf_counter()
            original_times.append(end - start)
            time.sleep(0.1)  # Small delay to reduce system load
        except Exception as e:
            print(f"Error with original implementation: {e}")
            return {
                "sampler_class": sampler_class.__name__,
                "params": params,
                "original_avg": "ERROR",
                "alternative_avg": "N/A",
                "speedup": "N/A",
                "error": str(e),
            }

    original_avg = float(np.mean(original_times))
    print(
        f"Original {sampler_class.__name__}: {original_avg:.4f}s (raw times: {original_times})"
    )

    # If no alternative implementation is provided, return results for original only
    if alternative_class is None:
        return {
            "sampler_class": sampler_class.__name__,
            "params": params,
            "original_avg": original_avg,
            "alternative_avg": "N/A",
            "speedup": "N/A",
        }

    # Benchmark alternative implementation
    alternative_times = []
    for _ in range(avg_times):
        try:
            start = time.perf_counter()
            sampler = alternative_class(**params)
            # Consume the iterator
            list(iter(sampler))
            end = time.perf_counter()
            alternative_times.append(end - start)
            time.sleep(0.1)  # Small delay to reduce system load
        except Exception as e:
            print(f"Error with alternative implementation: {e}")
            return {
                "sampler_class": sampler_class.__name__,
                "params": params,
                "original_avg": original_avg,
                "alternative_avg": "ERROR",
                "speedup": "N/A",
                "error": str(e),
            }

    alternative_avg = float(np.mean(alternative_times))
    print(
        f"Alternative {alternative_class.__name__}: {alternative_avg:.4f}s (raw times: {alternative_times})"
    )

    # Calculate speedup
    if original_avg > 0 and alternative_avg > 0:
        speedup = (original_avg - alternative_avg) / original_avg * 100
        speedup_str = f"{speedup:.2f}%"
    else:
        speedup_str = "N/A"

    print(f"Speedup: {speedup_str}\n")

    return {
        "sampler_class": sampler_class.__name__,
        "params": params,
        "original_avg": original_avg,
        "alternative_avg": alternative_avg,
        "speedup": speedup_str,
    }


def run_benchmarks(
    sampler_classes: List[Type[Sampler]],
    avg_times: int = 10,
    max_combinations: int = 10,
) -> List[Dict[str, Any]]:
    """Run benchmarks for multiple sampler classes.

    Args:
        sampler_classes: List of sampler classes to benchmark
        avg_times: Number of times to run each benchmark for averaging
        max_combinations: Maximum number of parameter combinations to test per sampler

    Returns:
        List of benchmark results
    """
    results = []

    for sampler_class in sampler_classes:
        print(f"\n{'='*80}\nBenchmarking {sampler_class.__name__}\n{'='*80}")

        # Get parameter configurations for this sampler
        params = get_sampler_params(sampler_class)

        # Generate parameter combinations
        combinations = generate_param_combinations(params)

        # Limit the number of combinations to avoid excessive benchmarking
        if len(combinations) > max_combinations:
            print(
                f"Limiting to {max_combinations} parameter combinations out of {len(combinations)}"
            )
            combinations = combinations[:max_combinations]

        # Get alternative implementation if available
        alternative_class = ALTERNATIVE_IMPLEMENTATIONS.get(sampler_class)

        # Run benchmarks for each parameter combination
        for combo in combinations:
            result = benchmark_sampler(
                sampler_class=sampler_class,
                params=combo,
                avg_times=avg_times,
                alternative_class=alternative_class,
            )
            results.append(result)

    return results


def format_results(results: List[Dict[str, Any]]) -> List[List[str]]:
    """Format benchmark results for tabular display."""
    formatted = []

    for result in results:
        # Format parameters for display
        params_str = ", ".join(
            f"{k}={v}"
            for k, v in result["params"].items()
            if k not in ["data_source", "sampler", "indices", "weights"]
        )

        # Format times
        original_time = (
            f"{result['original_avg']:.4f}"
            if isinstance(result["original_avg"], float)
            else result["original_avg"]
        )
        alternative_time = (
            f"{result['alternative_avg']:.4f}"
            if isinstance(result["alternative_avg"], float)
            else result["alternative_avg"]
        )

        formatted.append(
            [
                result["sampler_class"],
                params_str,
                original_time,
                alternative_time,
                result["speedup"],
            ]
        )

    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark for torch.utils.data samplers"
    )
    parser.add_argument(
        "--samplers",
        type=str,
        default="BatchSampler,RandomSampler,SequentialSampler",
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
        default=10,
        help="Maximum number of parameter combinations to test per sampler",
    )
    parser.add_argument(
        "--custom-module",
        type=str,
        default="",
        help="Optional Python module path containing custom sampler implementations",
    )

    args = parser.parse_args()

    # Load custom module if provided
    if args.custom_module:
        try:
            custom_module = importlib.import_module(args.custom_module)
            print(f"Loaded custom module: {args.custom_module}")
        except ImportError as e:
            print(f"Error loading custom module: {e}")
            custom_module = None
    else:
        custom_module = None

    # Get sampler classes to benchmark
    sampler_classes = []
    for sampler_name in args.samplers.split(","):
        sampler_name = sampler_name.strip()

        # Try to find the sampler class in torch.utils.data
        if hasattr(torch.utils.data, sampler_name):
            sampler_classes.append(getattr(torch.utils.data, sampler_name))
        # Try to find in the custom module
        elif custom_module and hasattr(custom_module, sampler_name):
            sampler_classes.append(getattr(custom_module, sampler_name))
        # Try to find in the current module (for alternative implementations)
        elif sampler_name in globals():
            sampler_classes.append(globals()[sampler_name])
        else:
            print(f"Warning: Sampler class '{sampler_name}' not found, skipping")

    if not sampler_classes:
        print("No valid sampler classes specified. Exiting.")
        return

    # Run benchmarks
    results = run_benchmarks(
        sampler_classes=sampler_classes,
        avg_times=args.avg_times,
        max_combinations=args.max_combinations,
    )

    # Format and display results
    formatted_results = format_results(results)
    headers = [
        "Sampler Class",
        "Parameters",
        "Original (s)",
        "Alternative (s)",
        "Speedup",
    ]
    print("\nBenchmark Results:")
    print(tabulate(formatted_results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
