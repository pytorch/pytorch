#!/usr/bin/env python3
"""
SHRT Prototype: Enumerate all possible sharding placement combinations for opinfos.

This script counts how many placement combinations exist when you enumerate:
1. All sample inputs from opinfo for a given operator (e.g., add)
2. All possible 1-D mesh placements for each tensor input
"""

from collections import defaultdict

import torch
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

# Partial reduce ops to enumerate (excluding NormPartial and MaskPartial)
PARTIAL_REDUCE_OPS = ["sum", "avg", "min", "max"]
from torch.testing._internal.common_methods_invocations import op_db


def get_opinfo_by_name(name: str):
    """Find an OpInfo by operator name."""
    matches = [op for op in op_db if op.name == name]
    if not matches:
        raise ValueError(f"No OpInfo found for operator: {name}")
    return matches


def get_1d_placements_for_tensor(t: torch.Tensor):
    """
    Get all possible 1-D mesh placements for a tensor.
    For a tensor with ndim dimensions, we can:
    - Shard on any dimension: Shard(0), Shard(1), ..., Shard(ndim-1)
    - Replicate
    - Partial with each reduce op: Partial("sum"), Partial("avg"), Partial("min"), Partial("max")
    """
    placements = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))
    for reduce_op in PARTIAL_REDUCE_OPS:
        placements.append(Partial(reduce_op))
    return placements


def extract_tensors_from_sample(sample_input):
    """
    Extract all tensor arguments from a SampleInput.
    Returns a list of (name, tensor) pairs.
    """
    tensors = []

    # The primary input
    if isinstance(sample_input.input, torch.Tensor):
        tensors.append(("input", sample_input.input))
    elif isinstance(sample_input.input, (list, tuple)):
        for i, t in enumerate(sample_input.input):
            if isinstance(t, torch.Tensor):
                tensors.append((f"input[{i}]", t))

    # Positional args
    for i, arg in enumerate(sample_input.args):
        if isinstance(arg, torch.Tensor):
            tensors.append((f"args[{i}]", arg))
        elif isinstance(arg, (list, tuple)):
            for j, t in enumerate(arg):
                if isinstance(t, torch.Tensor):
                    tensors.append((f"args[{i}][{j}]", t))

    # Keyword args
    for key, val in sample_input.kwargs.items():
        if isinstance(val, torch.Tensor):
            tensors.append((f"kwargs[{key}]", val))
        elif isinstance(val, (list, tuple)):
            for j, t in enumerate(val):
                if isinstance(t, torch.Tensor):
                    tensors.append((f"kwargs[{key}][{j}]", t))

    return tensors


def count_placement_combinations(op_name: str, device: str = "cpu", dtype=torch.float32, verbose: bool = False):
    """
    Count all possible placement combinations for an operator's sample inputs.
    """
    opinfos = get_opinfo_by_name(op_name)
    print(f"Found {len(opinfos)} OpInfo(s) for '{op_name}'")

    total_samples = 0
    total_combinations = 0
    samples_by_tensor_count = defaultdict(int)
    combinations_by_tensor_count = defaultdict(int)

    for opinfo in opinfos:
        variant = opinfo.variant_test_name or "(default)"
        print(f"\n  Variant: {variant}")

        # Generate sample inputs
        try:
            samples = list(opinfo.sample_inputs(device, dtype))
        except Exception as e:
            print(f"    Error generating samples: {e}")
            continue

        print(f"    Sample inputs: {len(samples)}")

        for sample_idx, sample in enumerate(samples):
            tensors = extract_tensors_from_sample(sample)
            num_tensors = len(tensors)

            if num_tensors == 0:
                print(f"      Sample {sample_idx}: No tensors found")
                continue

            # For each tensor, compute how many placements are possible
            placement_counts = []
            tensor_info = []
            for name, t in tensors:
                placements = get_1d_placements_for_tensor(t)
                placement_counts.append(len(placements))
                tensor_info.append(f"{name}: shape={list(t.shape)}, placements={len(placements)}")

            # Total combinations = product of all placement counts
            num_combinations = 1
            for count in placement_counts:
                num_combinations *= count

            total_samples += 1
            total_combinations += num_combinations
            samples_by_tensor_count[num_tensors] += 1
            combinations_by_tensor_count[num_tensors] += num_combinations

            if verbose or sample_idx < 5:  # Only show first 5 samples unless verbose
                print(f"      Sample {sample_idx}: {num_tensors} tensor(s), {num_combinations} combinations")
                for info in tensor_info:
                    print(f"        - {info}")
            elif sample_idx == 5:
                print(f"      ... (more samples)")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total sample inputs: {total_samples}")
    print(f"Total placement combinations: {total_combinations}")
    print()
    print("Breakdown by number of tensor arguments:")
    for num_tensors in sorted(samples_by_tensor_count.keys()):
        num_samples = samples_by_tensor_count[num_tensors]
        num_combos = combinations_by_tensor_count[num_tensors]
        print(f"  {num_tensors} tensor(s): {num_samples} samples, {num_combos} combinations")

    return total_samples, total_combinations


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enumerate sharding combinations for opinfo")
    parser.add_argument("--op", default="add", help="Operator name to analyze")
    parser.add_argument("--device", default="cpu", help="Device to use for sample generation")
    parser.add_argument("--dtype", default="float32", help="Dtype to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all samples")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    print(f"Analyzing operator: {args.op}")
    print(f"Device: {args.device}, Dtype: {dtype}")
    print("=" * 60)

    count_placement_combinations(args.op, args.device, dtype, args.verbose)
