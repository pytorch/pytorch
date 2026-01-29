#!/usr/bin/env python3
"""
SHRT Prototype: Validate sharding rules by testing placement combinations.

This script:
1. Runs the operator on full tensors to get ground truth
2. For each placement combination, shards inputs and runs the operator
3. Reassembles outputs and compares with ground truth
4. Prunes invalid combinations to avoid redundant testing
"""

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

# Override common size variables to ensure even sharding across world_size=2
# This matches what test_dtensor_ops.py does
# Must modify both core and common_ops BEFORE importing op_db
from torch.testing._internal.opinfo import core as opinfo_core
opinfo_core.L = 24
opinfo_core.M = 12
opinfo_core.S = 4
opinfo_core.XS = 2

import torch.testing._internal.common_methods_invocations as common_ops
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2

from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate
from torch.distributed.tensor.placement_types import Partial, Shard
from torch.testing._internal.common_methods_invocations import op_db
from torch.utils import _pytree as pytree


# Partial reduce ops to enumerate for outputs (all types)
PARTIAL_REDUCE_OPS = ["sum", "avg", "min", "max"]

# Partial reduce ops to enumerate for inputs
# We include min/max but use tensor index to vary the "winning rank" pattern
# so that P(max) + P(max) -> P(max) is correctly detected as invalid.
PARTIAL_INPUT_REDUCE_OPS = ["sum", "avg", "min", "max"]


def get_opinfo_by_name(name: str):
    """Find an OpInfo by operator name."""
    matches = [op for op in op_db if op.name == name]
    if not matches:
        raise ValueError(f"No OpInfo found for operator: {name}")
    return matches


def get_1d_input_placements_for_tensor(t: torch.Tensor, include_partial: bool = False):
    """
    Get all possible 1-D mesh placements for an INPUT tensor.

    Args:
        t: The tensor to get placements for
        include_partial: If True, include Partial placements for inputs.
            Only includes sum/avg (not min/max) since those compose well.
    """
    placements = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))
    if include_partial:
        for reduce_op in PARTIAL_INPUT_REDUCE_OPS:
            placements.append(Partial(reduce_op))
    return placements


def get_1d_output_placements_for_tensor(t: torch.Tensor):
    """
    Get all possible 1-D mesh placements for an OUTPUT tensor.
    Outputs can be Replicate, Shard, or Partial.
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

    if isinstance(sample_input.input, torch.Tensor):
        tensors.append(("input", sample_input.input))
    elif isinstance(sample_input.input, (list, tuple)):
        for i, t in enumerate(sample_input.input):
            if isinstance(t, torch.Tensor):
                tensors.append((f"input[{i}]", t))

    for i, arg in enumerate(sample_input.args):
        if isinstance(arg, torch.Tensor):
            tensors.append((f"args[{i}]", arg))
        elif isinstance(arg, (list, tuple)):
            for j, t in enumerate(arg):
                if isinstance(t, torch.Tensor):
                    tensors.append((f"args[{i}][{j}]", t))

    for key, val in sample_input.kwargs.items():
        if isinstance(val, torch.Tensor):
            tensors.append((f"kwargs[{key}]", val))
        elif isinstance(val, (list, tuple)):
            for j, t in enumerate(val):
                if isinstance(t, torch.Tensor):
                    tensors.append((f"kwargs[{key}][{j}]", t))

    return tensors


def placement_tuple_to_str(placements: tuple) -> str:
    """Convert a tuple of placements to a readable string."""
    parts = []
    for p in placements:
        if isinstance(p, Shard):
            parts.append(f"S{p.dim}")
        elif isinstance(p, Replicate):
            parts.append("R")
        elif isinstance(p, Partial):
            parts.append(f"P({p.reduce_op})")
        else:
            parts.append(str(p))
    return "(" + ", ".join(parts) + ")"


@dataclass
class PlacementCombination:
    """Represents a combination of input and output placements."""
    input_placements: tuple  # One placement per input tensor
    output_placement: Any    # Placement for the output tensor

    def __hash__(self):
        return hash((self.input_placements, str(self.output_placement)))

    def __eq__(self, other):
        return (self.input_placements == other.input_placements and
                str(self.output_placement) == str(other.output_placement))

    def __str__(self):
        return f"inputs={placement_tuple_to_str(self.input_placements)}, output={placement_tuple_to_str((self.output_placement,))}"


def is_fully_replicated(placements: tuple) -> bool:
    """Check if all placements are Replicate."""
    return all(isinstance(p, Replicate) for p in placements)


def generate_placement_combinations(tensors: list, output_tensor: torch.Tensor):
    """
    Generate all possible (input_placements, output_placement) combinations.
    Skip the fully-replicated case since it's trivially correct.
    """
    # Get placements for each input tensor (Replicate or Shard only)
    input_placement_options = [get_1d_input_placements_for_tensor(t) for _, t in tensors]

    # Get placements for output tensor (Replicate, Shard, or Partial)
    output_placement_options = get_1d_output_placements_for_tensor(output_tensor)

    # Generate Cartesian product
    for input_combo in itertools.product(*input_placement_options):
        for output_placement in output_placement_options:
            # Skip fully replicated case (trivially correct, not interesting to test)
            if is_fully_replicated(input_combo) and isinstance(output_placement, Replicate):
                continue

            yield PlacementCombination(input_combo, output_placement)


def _create_partial_input(
    tensor: torch.Tensor, placement: Partial, world_size: int, tensor_idx: int = 0
) -> LocalTensor:
    """
    Create a LocalTensor with values that reduce to the original tensor.

    We use asymmetric splits/offsets to avoid coincidental matches when
    combining different Partial types.

    For Partial(sum): tensor_idx-dependent split (60/40, 70/30, 80/20, ...)
    For Partial(avg): tensor_idx-dependent split scaled by world_size
    For Partial(min): Element-wise variation based on tensor_idx
    For Partial(max): Element-wise variation based on tensor_idx

    Using tensor_idx-dependent ratios prevents coincidental cancellation
    when dividing or multiplying Partial tensors (e.g., P(avg)/P(avg) would
    incorrectly appear valid if both used the same 60/40 split).

    Args:
        tensor: The original tensor value
        placement: The Partial placement to create
        world_size: Number of ranks
        tensor_idx: Index of this tensor among inputs (0, 1, 2, ...).
            Used to vary the "winning rank" pattern for min/max so that
            different inputs have different patterns, correctly detecting
            invalid rules like P(max) + P(max) -> P(max).
    """
    reduce_op = placement.reduce_op

    if reduce_op == "sum":
        # Split unevenly with tensor_idx-dependent ratios to prevent
        # coincidental cancellation when dividing P(sum) tensors.
        # tensor_idx=0: 60%/40%, tensor_idx=1: 70%/30%, etc.
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)  # 0.6, 0.7, 0.8, 0.6, ...
        local_tensors = {}
        for r in range(world_size):
            if r == 0:
                local_tensors[r] = tensor.clone() * base_ratio
            else:
                # Distribute remaining across other ranks
                local_tensors[r] = tensor.clone() * ((1 - base_ratio) / (world_size - 1))
        return LocalTensor(local_tensors)

    elif reduce_op == "avg":
        # For avg: local values should average to original
        # avg = (sum of locals) / world_size = original
        # So sum of locals = original * world_size
        # Use tensor_idx-dependent split to prevent cancellation in division
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)  # 0.6, 0.7, 0.8, 0.6, ...
        local_tensors = {}
        for r in range(world_size):
            if r == 0:
                local_tensors[r] = tensor.clone() * base_ratio * world_size
            else:
                local_tensors[r] = tensor.clone() * ((1 - base_ratio) / (world_size - 1)) * world_size
        return LocalTensor(local_tensors)

    elif reduce_op == "min":
        # Create element-wise variation: which rank holds the min varies
        # by element index AND tensor_idx. This ensures different input
        # tensors have different patterns.
        local_tensors = {}
        flat = tensor.flatten()
        # Use tensor_idx to shift the pattern
        mask = (torch.arange(flat.numel(), device=tensor.device) + tensor_idx) % 2 == 0
        for r in range(world_size):
            if r == 0:
                # Rank 0: add offset where mask is False (min on rank 1)
                r_offset = torch.where(mask, torch.zeros_like(flat), torch.full_like(flat, 0.7))
            else:
                # Rank 1: add offset where mask is True (min on rank 0)
                r_offset = torch.where(mask, torch.full_like(flat, 0.7), torch.zeros_like(flat))
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)
        return LocalTensor(local_tensors)

    elif reduce_op == "max":
        # Create element-wise variation based on tensor_idx
        local_tensors = {}
        flat = tensor.flatten()
        mask = (torch.arange(flat.numel(), device=tensor.device) + tensor_idx) % 2 == 0
        for r in range(world_size):
            if r == 0:
                # Rank 0: subtract offset where mask is False (max on rank 1)
                r_offset = torch.where(mask, torch.zeros_like(flat), torch.full_like(flat, -1.3))
            else:
                # Rank 1: subtract offset where mask is True (max on rank 0)
                r_offset = torch.where(mask, torch.full_like(flat, -1.3), torch.zeros_like(flat))
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)
        return LocalTensor(local_tensors)

    else:
        local_tensors = {r: tensor.clone() for r in range(world_size)}
        return LocalTensor(local_tensors)


def validate_combination(
    op,
    sample_input,
    tensors: list,
    combination: PlacementCombination,
    ground_truth: torch.Tensor,
    world_size: int = 2,
    mesh=None,
) -> tuple[bool, str]:
    """
    Validate a single placement combination.

    The validation logic:
    1. Shard inputs according to input placements to get local tensors
    2. Run the raw op on local tensors (bypassing DTensor dispatch)
    3. Wrap the local output in a DTensor with the claimed output placement
    4. Redistribute to Replicate and compare with ground truth

    For Partial inputs, we create inputs with different local values per rank
    to properly test the mathematical properties of the operation.

    Args:
        op: The operator function
        sample_input: The SampleInput with original arguments
        tensors: List of (name, tensor) pairs extracted from sample
        combination: The placement combination to validate
        ground_truth: Expected output tensor
        world_size: Number of simulated ranks
        mesh: Optional pre-created device mesh (for performance).
            If None, creates one internally.

    Returns:
        (is_valid, error_message)
    """
    try:
        # Use provided mesh or create one
        if mesh is None:
            device = tensors[0][1].device.type if tensors else "cpu"
            mesh = init_device_mesh(device, (world_size,))

        # Distribute input tensors according to placements, then get local tensors
        local_tensors = []
        for tensor_idx, ((name, tensor), placement) in enumerate(
            zip(tensors, combination.input_placements)
        ):
            if isinstance(placement, Partial):
                # For Partial inputs, create truly different local values
                # Pass tensor_idx so different inputs get different patterns
                local_tensor = _create_partial_input(
                    tensor, placement, world_size, tensor_idx
                )
                local_tensors.append(local_tensor)
            elif isinstance(placement, Replicate):
                # For Replicate inputs, all ranks have the same value
                local_tensor = LocalTensor({r: tensor.clone() for r in range(world_size)})
                local_tensors.append(local_tensor)
            else:
                # For Shard inputs, use distribute_tensor
                dt = distribute_tensor(tensor.clone(), mesh, (placement,))
                local_tensors.append(dt.to_local())

        # Build args with local tensors
        local_idx = 0

        def _replace_with_local(a):
            nonlocal local_idx
            if isinstance(a, torch.Tensor):
                local = local_tensors[local_idx]
                local_idx += 1
                return local
            return a

        # Replace tensors in args with local tensors
        if isinstance(sample_input.input, torch.Tensor):
            local_input = _replace_with_local(sample_input.input)
        else:
            local_input = pytree.tree_map(_replace_with_local, sample_input.input)

        local_args = pytree.tree_map(_replace_with_local, sample_input.args)
        local_kwargs = pytree.tree_map(_replace_with_local, sample_input.kwargs)

        # Run the operator on local tensors (raw operation, not DTensor dispatch)
        local_output = op(local_input, *local_args, **local_kwargs)

        if not isinstance(local_output, torch.Tensor):
            return False, f"Local output is not a tensor: {type(local_output)}"

        # All inputs are LocalTensors, so output should be LocalTensor
        if not isinstance(local_output, LocalTensor):
            return False, f"LocalTensor inputs produced non-LocalTensor output: {type(local_output)}"

        # Wrap local output in DTensor with the claimed output placement
        # Pass the expected global shape for correct handling of uneven sharding
        output_dt = DTensor.from_local(
            local_output, mesh, (combination.output_placement,),
            shape=ground_truth.shape, stride=ground_truth.stride()
        )

        # For Replicate outputs, verify that local values are identical
        if isinstance(combination.output_placement, Replicate):
            local_values = [local_output._local_tensors[r] for r in range(world_size)]
            all_same = all(torch.allclose(local_values[0], lv, atol=1e-5, rtol=1e-5) for lv in local_values[1:])
            if not all_same:
                return False, "Replicate output but local values differ across ranks"

        # Redistribute to replicate to compare
        full_output = output_dt.redistribute(mesh, (Replicate(),)).to_local()

        # If full_output is a LocalTensor, extract rank 0's value for comparison
        # (after redistribution to Replicate, all ranks should have the same value)
        if isinstance(full_output, LocalTensor):
            full_output = full_output._local_tensors[0]

        # Compare with ground truth
        if ground_truth.shape != full_output.shape:
            return False, f"Shape mismatch: expected {ground_truth.shape}, got {full_output.shape}"

        if not torch.allclose(ground_truth, full_output, atol=1e-5, rtol=1e-5):
            max_diff = (ground_truth - full_output).abs().max().item()
            return False, f"Value mismatch: max_diff={max_diff:.6f}"

        return True, ""

    except Exception as e:
        return False, f"Exception: {type(e).__name__}: {e}"


def validate_operator(
    op_name: str,
    device: str = "cpu",
    dtype=torch.float32,
    world_size: int = 2,
    max_samples: int = None,
    verbose: bool = False,
):
    """
    Validate sharding rules for an operator.
    """
    # Initialize fake process group for LocalTensorMode
    if not dist.is_initialized():
        dist.init_process_group("fake", rank=0, world_size=world_size)

    # Clear sharding propagation cache to avoid stale references
    from torch.distributed.tensor.debug import _clear_sharding_prop_cache
    _clear_sharding_prop_cache()

    start_time = time.time()

    opinfos = get_opinfo_by_name(op_name)
    print(f"Found {len(opinfos)} OpInfo(s) for '{op_name}'")
    print(f"World size: {world_size}")

    # Track combinations that have been validated
    # For each combination, track how many times it was valid vs tested
    # Also track input shapes for debugging
    combination_stats: dict[PlacementCombination, dict[str, Any]] = defaultdict(
        lambda: {"valid": 0, "tested": 0, "valid_shapes": [], "invalid_shapes": [],
                 "valid_degenerate": 0}  # count of valid results from degenerate inputs
    )

    total_samples = 0
    total_tests = 0

    for opinfo in opinfos:
        variant = opinfo.variant_test_name or "(default)"
        print(f"\n  Variant: {variant}")

        # Get the operator function
        op = opinfo.op

        # Generate sample inputs
        try:
            samples = list(opinfo.sample_inputs(device, dtype))
        except Exception as e:
            print(f"    Error generating samples: {e}")
            continue

        if max_samples:
            samples = samples[:max_samples]

        print(f"    Processing {len(samples)} sample inputs...")

        for sample_idx, sample in enumerate(samples):
            tensors = extract_tensors_from_sample(sample)

            if len(tensors) == 0:
                if verbose:
                    print(f"      Sample {sample_idx}: No tensors found, skipping")
                continue

            total_samples += 1

            # Run operator on full tensors to get ground truth
            try:
                if isinstance(sample.input, torch.Tensor):
                    ground_truth = op(sample.input, *sample.args, **sample.kwargs)
                else:
                    ground_truth = op(*sample.input, *sample.args, **sample.kwargs)

                if not isinstance(ground_truth, torch.Tensor):
                    if verbose:
                        print(f"      Sample {sample_idx}: Output is not a tensor ({type(ground_truth)}), skipping")
                    continue
            except Exception as e:
                if verbose:
                    print(f"      Sample {sample_idx}: Error computing ground truth: {e}")
                continue

            # Generate all placement combinations for this sample
            all_combinations = list(generate_placement_combinations(tensors, ground_truth))

            if verbose:
                print(f"      Sample {sample_idx}: {len(tensors)} tensor(s), "
                      f"{len(all_combinations)} combinations to test")

            # Build shape signature for debugging
            shape_sig = tuple(t.shape for _, t in tensors)

            # Check for potentially degenerate cases (zero-size tensors)
            has_zero_size = any(0 in t.shape for _, t in tensors)

            sample_valid = 0
            sample_invalid = 0

            for combo in all_combinations:
                total_tests += 1
                is_valid, error_msg = validate_combination(
                    op, sample, tensors, combo, ground_truth, world_size
                )

                combination_stats[combo]["tested"] += 1
                if is_valid:
                    combination_stats[combo]["valid"] += 1
                    combination_stats[combo]["valid_shapes"].append(shape_sig)
                    if has_zero_size:
                        combination_stats[combo]["valid_degenerate"] += 1
                    sample_valid += 1
                else:
                    combination_stats[combo]["invalid_shapes"].append(shape_sig)
                    sample_invalid += 1

                    if verbose:
                        print(f"        INVALID: {combo}")
                        print(f"          Reason: {error_msg}")

            if verbose:
                print(f"        Results: {sample_valid} valid, {sample_invalid} invalid")

    # Categorize combinations
    always_valid = []  # Valid for ALL samples where tested
    sometimes_valid = []  # Valid for SOME but not all samples
    never_valid = []  # Never valid for any sample
    only_degenerate = []  # Only valid for degenerate (zero-size) inputs

    for combo, stats in combination_stats.items():
        if stats["valid"] == stats["tested"]:
            always_valid.append((combo, stats))
        elif stats["valid"] > 0:
            # Check if all valid results came from degenerate inputs
            if stats["valid"] == stats["valid_degenerate"]:
                only_degenerate.append((combo, stats))
            else:
                sometimes_valid.append((combo, stats))
        else:
            never_valid.append((combo, stats))

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {total_samples}")
    print(f"Total combination tests: {total_tests}")
    print(f"Elapsed time: {elapsed_time:.2f}s ({total_tests / elapsed_time:.1f} tests/sec)" if elapsed_time > 0 else f"Elapsed time: {elapsed_time:.2f}s")
    print()
    print(f"Always valid (all samples): {len(always_valid)}")
    print(f"Sometimes valid (some samples): {len(sometimes_valid)}")
    print(f"Only valid for degenerate inputs (suspicious): {len(only_degenerate)}")
    print(f"Never valid: {len(never_valid)}")

    if always_valid:
        print("\n--- ALWAYS VALID (reliable rules) ---")
        for combo, stats in sorted(always_valid, key=lambda x: str(x[0])):
            print(f"  {combo}  [{stats['valid']}/{stats['tested']} samples]")

    if sometimes_valid:
        print("\n--- SOMETIMES VALID (input-dependent) ---")
        for combo, stats in sorted(sometimes_valid, key=lambda x: str(x[0])):
            print(f"  {combo}  [{stats['valid']}/{stats['tested']} samples]")
            if verbose:
                # Show unique shapes that passed
                unique_valid = set(stats["valid_shapes"])
                unique_invalid = set(stats["invalid_shapes"])
                print(f"    Valid for shapes: {unique_valid}")
                print(f"    Invalid for shapes: {unique_invalid}")

    if only_degenerate:
        print("\n--- ONLY DEGENERATE (likely false positives) ---")
        for combo, stats in sorted(only_degenerate, key=lambda x: str(x[0])):
            print(f"  {combo}  [{stats['valid']}/{stats['tested']} samples, all zero-size inputs]")

    # Cleanup
    _clear_sharding_prop_cache()
    try:
        dist.destroy_process_group()
    except Exception:
        pass

    return always_valid, sometimes_valid, only_degenerate, never_valid


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate sharding rules for opinfo")
    parser.add_argument("--op", default="add", help="Operator name to validate")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--dtype", default="float32", help="Dtype to use")
    parser.add_argument("--world-size", type=int, default=2, help="Simulated world size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    print(f"Validating operator: {args.op}")
    print(f"Device: {args.device}, Dtype: {dtype}")
    print("=" * 70)

    validate_operator(
        args.op,
        args.device,
        dtype,
        args.world_size,
        args.max_samples,
        args.verbose,
    )
