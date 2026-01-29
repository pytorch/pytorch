#!/usr/bin/env python3
"""
SHRT Prototype: Compare DTensor's sharding rules against ground truth validation.

This script:
1. Uses shrt_validate's ground truth validation for each placement combination
2. Directly queries DTensor's registered sharding strategies (not dispatch)
3. Compares the registered rules against ground truth
4. Reports false positives (DTensor has rule, ground truth says invalid) and
   false negatives (ground truth says valid, DTensor has no rule)
"""

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

# Override common size variables to ensure even sharding across world_size=2
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

from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.placement_types import Partial, Shard
from torch.testing._internal.common_methods_invocations import op_db

# Import validation logic from shrt_validate
from shrt_validate import (
    get_opinfo_by_name,
    extract_tensors_from_sample,
    validate_combination,
    PlacementCombination,
    placement_tuple_to_str,
    get_1d_input_placements_for_tensor,
    get_1d_output_placements_for_tensor,
    is_fully_replicated,
)


@dataclass
class Discrepancy:
    """Represents a discrepancy between ground truth and DTensor's rules."""
    input_placements: tuple
    output_placement: Any
    sample_idx: int
    input_shapes: tuple
    discrepancy_type: str  # "false_positive" or "false_negative"
    error_msg: str = ""
    scalar_args: tuple = ()  # Non-tensor args
    scalar_kwargs: dict = field(default_factory=dict)  # Non-tensor kwargs


@dataclass
class ComparisonStats:
    """Statistics for comparing ground truth vs DTensor rules."""
    true_positives: int = 0  # Both agree valid
    true_negatives: int = 0  # Both agree invalid
    false_positives: list = field(default_factory=list)  # DTensor has rule, ground truth says invalid
    false_negatives: list = field(default_factory=list)  # Ground truth valid, DTensor has no rule


def get_dtensor_strategies_for_op(op_overload, input_specs, mesh):
    """
    Query DTensor's registered strategies for an operator.

    Returns:
        List of (input_placements, output_placement) tuples that DTensor supports.
    """
    from torch.distributed.tensor._api import DTensor
    from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, DTensorSpec
    from torch.distributed.tensor._dtensor_spec import TensorMeta

    propagator = DTensor._op_dispatcher.sharding_propagator

    # Check which type of strategy is registered
    if op_overload in propagator.op_to_rules:
        # Has explicit propagation rule - harder to enumerate
        return None, "propagation_rule"

    if op_overload in propagator.op_strategy_funcs:
        strategy_func = propagator.op_strategy_funcs[op_overload]

        # Build OpSchema
        args_schema = tuple(input_specs)
        op_schema = OpSchema(op_overload, args_schema, {})

        try:
            strategy = strategy_func(op_schema)
            if isinstance(strategy, OpStrategy):
                # Extract all (input_placements, output_placement) pairs
                results = []
                for spec in strategy.strategies:
                    input_plcs = tuple(s.placements for s in spec.input_specs)
                    output_plc = spec.output_spec.placements
                    results.append((input_plcs, output_plc))
                return results, "op_strategy"
        except Exception as e:
            return None, f"strategy_error: {e}"

    if op_overload in propagator.op_single_dim_strategy_funcs:
        # Has single-dim strategy - we can query this directly
        return None, "single_dim_strategy"

    return None, "not_registered"


def get_aten_op_for_opinfo(opinfo):
    """
    Get the aten op that corresponds to this opinfo.
    Uses the opinfo's aten_name or falls back to name mapping.
    """
    aten = torch.ops.aten

    # Check if opinfo has aten_name attribute
    if hasattr(opinfo, 'aten_name') and opinfo.aten_name:
        name = opinfo.aten_name
    else:
        name = opinfo.name

    # Common op name mappings to aten ops
    op_mappings = {
        'add': aten.add.Tensor,
        'sub': aten.sub.Tensor,
        'mul': aten.mul.Tensor,
        'div': aten.div.Tensor,
        'matmul': aten.matmul.default,
        'mm': aten.mm.default,
        'bmm': aten.bmm.default,
        'addmm': aten.addmm.default,
        'neg': aten.neg.default,
        'abs': aten.abs.default,
        'exp': aten.exp.default,
        'log': aten.log.default,
        'sqrt': aten.sqrt.default,
        'relu': aten.relu.default,
        'sigmoid': aten.sigmoid.default,
        'tanh': aten.tanh.default,
        'sum': aten.sum.default,
        'mean': aten.mean.default,
        'max': aten.max.default,
        'min': aten.min.default,
        'cat': aten.cat.default,
        'stack': aten.stack.default,
        'view': aten.view.default,
        'reshape': aten.reshape.default,
        'transpose': aten.transpose.int,
        'permute': aten.permute.default,
        'contiguous': aten.contiguous.default,
        'clone': aten.clone.default,
        't': aten.t.default,
    }

    if name in op_mappings:
        return op_mappings[name]

    # Try to find it dynamically
    if hasattr(aten, name):
        op_packet = getattr(aten, name)
        # Try common overloads
        for overload in ['default', 'Tensor', 'out']:
            if hasattr(op_packet, overload):
                return getattr(op_packet, overload)

    return None


def query_single_dim_strategy(op_overload, tensors, mesh):
    """
    Query DTensor's single-dim strategy for given input tensors.

    Returns list of [output_placement, *input_placements] rules.
    Expands _ShardingPlaceholder to concrete Shard types.
    """
    from torch.distributed.tensor._api import DTensor
    from torch.distributed.tensor._dtensor_spec import TensorMeta
    from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder

    propagator = DTensor._op_dispatcher.sharding_propagator

    if op_overload not in propagator.op_single_dim_strategy_funcs:
        return None

    strategy_func = propagator.op_single_dim_strategy_funcs[op_overload]

    # Build args as TensorMeta objects (what the strategy function expects)
    args_meta = tuple(
        TensorMeta(shape=t.shape, stride=t.stride(), dtype=t.dtype)
        for _, t in tensors
    )

    try:
        # Call the single-dim strategy function
        # It returns list of [output_placement, *input_placements]
        result = strategy_func(op_overload, args_meta, {})

        # Expand _ShardingPlaceholder to concrete Shard types
        expanded_result = []
        for combo in result:
            expanded_combo = []
            for p in combo:
                if isinstance(p, _ShardingPlaceholder):
                    # Convert placeholder to Shard
                    expanded_combo.append(Shard(p.dim))
                else:
                    expanded_combo.append(p)
            expanded_result.append(expanded_combo)

        return expanded_result
    except Exception as e:
        return None


def compare_operator(
    op_name: str,
    device: str = "cpu",
    dtype=torch.float32,
    world_size: int = 2,
    max_samples: int = None,
    verbose: bool = False,
):
    """
    Compare DTensor's sharding rules against ground truth for an operator.
    """
    # Initialize fake process group for LocalTensorMode
    if not dist.is_initialized():
        dist.init_process_group("fake", rank=0, world_size=world_size)

    # Clear sharding propagation cache
    from torch.distributed.tensor.debug import _clear_sharding_prop_cache
    _clear_sharding_prop_cache()

    start_time = time.time()

    opinfos = get_opinfo_by_name(op_name)
    print(f"Found {len(opinfos)} OpInfo(s) for '{op_name}'")
    print(f"World size: {world_size}")

    stats = ComparisonStats()
    total_samples = 0
    total_combinations = 0
    ground_truth_time = 0.0
    strategy_query_time = 0.0

    for opinfo in opinfos:
        variant = opinfo.variant_test_name or "(default)"
        print(f"\n  Variant: {variant}")

        op = opinfo.op

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
                continue

            # Skip degenerate inputs
            has_zero_size = any(0 in t.shape for _, t in tensors)
            if has_zero_size:
                continue

            total_samples += 1

            # Get ground truth output
            try:
                if isinstance(sample.input, torch.Tensor):
                    ground_truth = op(sample.input, *sample.args, **sample.kwargs)
                else:
                    ground_truth = op(*sample.input, *sample.args, **sample.kwargs)

                if not isinstance(ground_truth, torch.Tensor):
                    continue
            except Exception:
                continue

            input_shapes = tuple(t.shape for _, t in tensors)

            # Extract non-tensor args and kwargs for context in discrepancy reports
            scalar_args = tuple(
                a for a in sample.args if not isinstance(a, torch.Tensor)
            )
            scalar_kwargs = {
                k: v for k, v in sample.kwargs.items() if not isinstance(v, torch.Tensor)
            }

            # Get all possible input placement combinations (including Partial)
            input_placement_options = [
                get_1d_input_placements_for_tensor(t, include_partial=True)
                for _, t in tensors
            ]
            output_placement_options = get_1d_output_placements_for_tensor(ground_truth)

            # Query DTensor's single-dim strategy (if available)
            aten_op = get_aten_op_for_opinfo(opinfo)

            from torch.distributed.tensor._api import DTensor
            propagator = DTensor._op_dispatcher.sharding_propagator

            dtensor_rules = set()  # Set of (input_placements, output_placement) strings

            strategy_start = time.time()
            if aten_op and aten_op in propagator.op_single_dim_strategy_funcs:
                strategy_result = query_single_dim_strategy(aten_op, tensors, None)
                if strategy_result:
                    # Parse strategy result to get valid combinations
                    # The result is a list of placement combinations
                    # Each combo is: [output_placement, *input_placements]
                    for combo in strategy_result:
                        if len(combo) >= len(tensors) + 1:
                            output_plc = combo[0]
                            input_plcs = tuple(combo[1:len(tensors)+1])

                            dtensor_rules.add((
                                tuple(str(p) for p in input_plcs),
                                str(output_plc)
                            ))

            elif aten_op and aten_op in propagator.op_strategy_funcs:
                # Query op_strategy_funcs for ops like mm, bmm
                # These take OpSchema with input OpStrategies and return output OpStrategy
                from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, OpSpec, DTensorSpec
                from torch.distributed.tensor._dtensor_spec import TensorMeta

                try:
                    # Create a mesh for building specs
                    mesh = init_device_mesh("cpu", (world_size,))

                    # Build input OpStrategies with all possible placements (including Partial)
                    input_strategies = []
                    for name, t in tensors:
                        input_placements = get_1d_input_placements_for_tensor(t, include_partial=True)
                        specs = []
                        for p in input_placements:
                            spec = DTensorSpec(
                                mesh=mesh,
                                placements=(p,),
                                tensor_meta=TensorMeta(shape=t.shape, stride=t.stride(), dtype=t.dtype),
                            )
                            specs.append(OpSpec(output_specs=spec, input_specs=tuple()))
                        input_strategies.append(OpStrategy(specs))

                    # Build OpSchema
                    op_schema = OpSchema(aten_op, tuple(input_strategies), {})

                    # Call strategy function
                    strategy_func = propagator.op_strategy_funcs[aten_op]
                    output_strategy = strategy_func(op_schema)

                    if isinstance(output_strategy, OpStrategy):
                        for spec in output_strategy.strategies:
                            output_plc = spec.output_spec.placements[0]
                            input_plcs = tuple(s.placements[0] for s in spec.input_specs)

                            # Skip fully replicated (trivially correct, not tested)
                            if is_fully_replicated(input_plcs) and isinstance(output_plc, Replicate):
                                continue

                            dtensor_rules.add((
                                tuple(str(p) for p in input_plcs),
                                str(output_plc)
                            ))
                except Exception as e:
                    if verbose:
                        print(f"        Error querying op_strategy: {e}")
            strategy_query_time += time.time() - strategy_start

            # Compute ground truth for all combinations
            ground_truth_valid = set()  # Set of (input_placements, output_placement) strings

            gt_start = time.time()
            # Create LocalTensorMode and mesh once per sample for performance
            device = tensors[0][1].device.type if tensors else "cpu"
            with LocalTensorMode(frozenset(range(world_size))):
                mesh = init_device_mesh(device, (world_size,))

                for input_placements in itertools.product(*input_placement_options):
                    if is_fully_replicated(input_placements):
                        continue

                    for output_placement in output_placement_options:
                        total_combinations += 1
                        combo = PlacementCombination(input_placements, output_placement)

                        # Validate using ground truth, passing pre-created mesh
                        is_valid, error_msg = validate_combination(
                            op, sample, tensors, combo, ground_truth, world_size, mesh
                        )

                        combo_key = (
                            tuple(str(p) for p in input_placements),
                            str(output_placement)
                        )

                        if is_valid:
                            ground_truth_valid.add(combo_key)
            ground_truth_time += time.time() - gt_start

            # Compare ground truth vs DTensor rules
            if dtensor_rules:
                for combo_key in ground_truth_valid:
                    if combo_key in dtensor_rules:
                        stats.true_positives += 1
                    else:
                        # Ground truth says valid, DTensor doesn't have rule
                        stats.false_negatives.append(Discrepancy(
                            input_placements=combo_key[0],
                            output_placement=combo_key[1],
                            sample_idx=sample_idx,
                            input_shapes=input_shapes,
                            discrepancy_type="false_negative",
                            scalar_args=scalar_args,
                            scalar_kwargs=scalar_kwargs,
                        ))

                for combo_key in dtensor_rules:
                    if combo_key not in ground_truth_valid:
                        # DTensor has rule, ground truth says invalid
                        stats.false_positives.append(Discrepancy(
                            input_placements=combo_key[0],
                            output_placement=combo_key[1],
                            sample_idx=sample_idx,
                            input_shapes=input_shapes,
                            discrepancy_type="false_positive",
                            scalar_args=scalar_args,
                            scalar_kwargs=scalar_kwargs,
                        ))
                    # (true positives already counted above)

                # True negatives are implicit (not in either set)

            if verbose:
                print(f"      Sample {sample_idx}: shapes={input_shapes}")
                print(f"        Ground truth valid: {len(ground_truth_valid)}")
                print(f"        DTensor rules: {len(dtensor_rules)}")

    # Summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {total_samples}")
    print(f"Total combinations tested: {total_combinations}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    if elapsed_time > 0:
        print(f"  - Strategy query time: {strategy_query_time:.2f}s ({100*strategy_query_time/elapsed_time:.1f}%)")
        print(f"  - Ground truth time: {ground_truth_time:.2f}s ({100*ground_truth_time/elapsed_time:.1f}%)")
    print()
    print(f"True positives (both agree valid): {stats.true_positives}")
    print(f"False positives (DTensor has rule, ground truth invalid): {len(stats.false_positives)}")
    print(f"False negatives (ground truth valid, DTensor missing): {len(stats.false_negatives)}")

    if stats.false_positives:
        print("\n--- FALSE POSITIVES (DTensor has incorrect rules) ---")
        by_combo = defaultdict(list)
        for d in stats.false_positives:
            key = (d.input_placements, d.output_placement)
            by_combo[key].append(d)

        for (inp, out), discrepancies in sorted(by_combo.items(), key=str):
            print(f"\n  inputs={inp}, output={out}")
            for d in discrepancies[:3]:
                extra = ""
                if d.scalar_args or d.scalar_kwargs:
                    parts = []
                    if d.scalar_args:
                        parts.append(f"args={d.scalar_args}")
                    if d.scalar_kwargs:
                        parts.append(f"kwargs={d.scalar_kwargs}")
                    extra = f", {', '.join(parts)}"
                print(f"    Sample {d.sample_idx}: shapes={d.input_shapes}{extra}")
            if len(discrepancies) > 3:
                print(f"    ... and {len(discrepancies) - 3} more")

    if stats.false_negatives:
        print("\n--- FALSE NEGATIVES (DTensor missing valid rules) ---")
        by_combo = defaultdict(list)
        for d in stats.false_negatives:
            key = (d.input_placements, d.output_placement)
            by_combo[key].append(d)

        for (inp, out), discrepancies in sorted(by_combo.items(), key=str):
            print(f"\n  inputs={inp}, output={out}")
            for d in discrepancies[:3]:
                extra = ""
                if d.scalar_args or d.scalar_kwargs:
                    parts = []
                    if d.scalar_args:
                        parts.append(f"args={d.scalar_args}")
                    if d.scalar_kwargs:
                        parts.append(f"kwargs={d.scalar_kwargs}")
                    extra = f", {', '.join(parts)}"
                print(f"    Sample {d.sample_idx}: shapes={d.input_shapes}{extra}")
            if len(discrepancies) > 3:
                print(f"    ... and {len(discrepancies) - 3} more")

    # Cleanup
    _clear_sharding_prop_cache()
    try:
        dist.destroy_process_group()
    except Exception:
        pass

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare DTensor rules against ground truth")
    parser.add_argument("--op", default="add", help="Operator name to compare")
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

    print(f"Comparing operator: {args.op}")
    print(f"Device: {args.device}, Dtype: {dtype}")
    print("=" * 70)

    compare_operator(
        args.op,
        args.device,
        dtype,
        args.world_size,
        args.max_samples,
        args.verbose,
    )
