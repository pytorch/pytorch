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

# Ops to skip in validation because ground truth comparison is not meaningful.
# These produce non-deterministic or uninitialized outputs.
SKIP_OPS = frozenset([
    "bernoulli",      # Random sampling
    "empty_like",     # Uninitialized memory
    "new_empty",      # Uninitialized memory
    "new_empty_strided",  # Uninitialized memory
    "normal",         # Random sampling
    "rand_like",      # Random sampling
    "randint_like",   # Random sampling
    "randn_like",     # Random sampling
    "uniform",        # Random sampling
])


def has_scalar_tensors(tensors: list) -> bool:
    """Check if any tensor in the list is a scalar (0-dim)."""
    return any(t.dim() == 0 for _, t in tensors)


def has_pmin_pmax(input_placements, output_placement) -> bool:
    """Check if any placement is Partial(min) or Partial(max)."""
    for p in input_placements:
        if isinstance(p, Partial) and p.reduce_op in ("min", "max"):
            return True
    if isinstance(output_placement, Partial) and output_placement.reduce_op in ("min", "max"):
        return True
    return False


def has_any_partial(input_placements, output_placement) -> bool:
    """Check if any placement is Partial (any reduce op)."""
    for p in input_placements:
        if isinstance(p, Partial):
            return True
    if isinstance(output_placement, Partial):
        return True
    return False


def _parse_placement(s: str):
    """Parse a placement string back to a placement object.

    Placement strings are: R, S(dim), P(reduce_op)
    """
    import re
    s = s.strip()
    if s == "R":
        return Replicate()
    elif s.startswith("S("):
        m = re.match(r"S\((\d+)\)", s)
        if m:
            return Shard(int(m.group(1)))
    elif s.startswith("P("):
        m = re.match(r"P\((\w+)\)", s)
        if m:
            return Partial(m.group(1))
    return None


def negate_scalar_tensors(tensors: list) -> list:
    """Return a new list with scalar tensors negated."""
    result = []
    for name, t in tensors:
        if t.dim() == 0:
            result.append((name, -t))
        else:
            result.append((name, t))
    return result


def negate_all_tensors(tensors: list) -> list:
    """Return a new list with all tensors negated."""
    return [(name, -t) for name, t in tensors]


def create_negated_sample(sample, tensors: list):
    """Create a sample with scalar tensors negated."""
    from torch.testing._internal.common_methods_invocations import SampleInput
    from torch.utils import _pytree as pytree

    # Track which tensors are scalars by their data_ptr
    scalar_ptrs = {t.data_ptr() for _, t in tensors if t.dim() == 0}

    def maybe_negate(x):
        if isinstance(x, torch.Tensor) and x.data_ptr() in scalar_ptrs:
            return -x
        return x

    new_input = pytree.tree_map(maybe_negate, sample.input)
    new_args = pytree.tree_map(maybe_negate, sample.args)
    new_kwargs = pytree.tree_map(maybe_negate, sample.kwargs)

    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)


def create_fully_negated_sample(sample, tensors: list):
    """Create a sample with ALL tensors negated (for P(min)/P(max) sign testing)."""
    from torch.testing._internal.common_methods_invocations import SampleInput
    from torch.utils import _pytree as pytree

    tensor_ptrs = {t.data_ptr() for _, t in tensors}

    def negate_tensor(x):
        if isinstance(x, torch.Tensor) and x.data_ptr() in tensor_ptrs:
            return -x
        return x

    new_input = pytree.tree_map(negate_tensor, sample.input)
    new_args = pytree.tree_map(negate_tensor, sample.args)
    new_kwargs = pytree.tree_map(negate_tensor, sample.kwargs)

    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)


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
    aten_op: Any = None  # The actual aten op overload
    variant: str = ""  # OpInfo variant name (e.g., "trunc_rounding")


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


class _CaptureAtenOp(torch.utils._python_dispatch.TorchDispatchMode):
    """Dispatch mode that captures aten ops called and their args."""

    def __init__(self, target_op_name: str = ""):
        self.target_op_name = target_op_name.lower()
        self.all_ops = []  # List of (op, args, kwargs)
        self.best_match = None
        self.best_match_args = None
        self.best_match_kwargs = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func.namespace == "aten":
            self.all_ops.append((func, args, kwargs))
            # Check if this op matches the target name
            op_name = func.name().split("::")[1].split(".")[0].lower()
            if self.target_op_name and self.target_op_name in op_name:
                if self.best_match is None:
                    self.best_match = func
                    self.best_match_args = args
                    self.best_match_kwargs = kwargs
        return func(*args, **kwargs)


def get_aten_op_for_sample(op, sample, op_name: str = ""):
    """
    Determine the actual aten op that will be dispatched for a given sample.

    Runs the operation through a capture mode to see which aten overload
    is actually used (e.g., sum.default vs sum.dim_IntList).

    Args:
        op: The operator callable
        sample: The sample input
        op_name: Optional op name to prefer when multiple ops are called

    Returns (aten_op, non_tensor_args, non_tensor_kwargs) where the args/kwargs
    are the actual values passed to the aten op (with tensors removed).
    """
    with _CaptureAtenOp(op_name) as capture:
        try:
            if isinstance(sample.input, torch.Tensor):
                op(sample.input, *sample.args, **sample.kwargs)
            else:
                op(*sample.input, *sample.args, **sample.kwargs)
        except Exception:
            pass

    # Use best match if we found one, otherwise fall back to first op
    if capture.best_match is not None:
        captured_op = capture.best_match
        captured_args = capture.best_match_args
        captured_kwargs = capture.best_match_kwargs
    elif capture.all_ops:
        captured_op, captured_args, captured_kwargs = capture.all_ops[0]
    else:
        return None, (), {}

    # Extract non-tensor args and kwargs from what was actually passed to aten op
    non_tensor_args = tuple(
        a for a in captured_args if not isinstance(a, torch.Tensor)
    )
    non_tensor_kwargs = {
        k: v for k, v in captured_kwargs.items()
        if not isinstance(v, torch.Tensor)
    }

    return captured_op, non_tensor_args, non_tensor_kwargs


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
    incorrect_only: bool = False,
):
    """
    Compare DTensor's sharding rules against ground truth for an operator.

    Args:
        op_name: Name of the operator to test
        device: Device to run on
        dtype: Data type for tensors
        world_size: Simulated world size
        max_samples: Maximum number of samples to test per OpInfo
        verbose: Print detailed output
        incorrect_only: If True, only test DTensor's claimed rules for correctness.
            Skips exhaustive search for missing rules (much faster).
    """
    # Check if op should be skipped
    if op_name in SKIP_OPS:
        print(f"Skipping '{op_name}' (non-deterministic or uninitialized output)")
        return ComparisonStats()

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
        variant = opinfo.variant_test_name
        if variant:
            print(f"\n  OpInfo variant: {variant}")

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

            # For P(min)/P(max) combinations, create a fully negated variant to test
            # sign-dependent behavior (e.g., R / P(max) -> P(max) only works for certain signs)
            try:
                fully_negated_sample = create_fully_negated_sample(sample, tensors)
                fully_negated_tensors = negate_all_tensors(tensors)

                if isinstance(fully_negated_sample.input, torch.Tensor):
                    fully_negated_ground_truth = op(
                        fully_negated_sample.input,
                        *fully_negated_sample.args,
                        **fully_negated_sample.kwargs
                    )
                else:
                    fully_negated_ground_truth = op(
                        *fully_negated_sample.input,
                        *fully_negated_sample.args,
                        **fully_negated_sample.kwargs
                    )

                if not isinstance(fully_negated_ground_truth, torch.Tensor):
                    fully_negated_sample = None
            except Exception:
                fully_negated_sample = None
                fully_negated_tensors = None
                fully_negated_ground_truth = None

            # For samples with rounding_mode, create a non-rounded variant to detect
            # rounding-induced false positives (where trunc/floor masks real differences)
            has_rounding_mode = "rounding_mode" in sample.kwargs
            non_rounded_sample = None
            non_rounded_ground_truth = None
            non_rounded_negated_sample = None
            non_rounded_negated_tensors = None
            non_rounded_negated_ground_truth = None

            if has_rounding_mode:
                from torch.testing._internal.common_methods_invocations import SampleInput
                try:
                    non_rounded_kwargs = {k: v for k, v in sample.kwargs.items() if k != "rounding_mode"}
                    non_rounded_sample = SampleInput(
                        sample.input, args=sample.args, kwargs=non_rounded_kwargs
                    )

                    if isinstance(non_rounded_sample.input, torch.Tensor):
                        non_rounded_ground_truth = op(
                            non_rounded_sample.input,
                            *non_rounded_sample.args,
                            **non_rounded_sample.kwargs
                        )
                    else:
                        non_rounded_ground_truth = op(
                            *non_rounded_sample.input,
                            *non_rounded_sample.args,
                            **non_rounded_sample.kwargs
                        )

                    if not isinstance(non_rounded_ground_truth, torch.Tensor):
                        non_rounded_sample = None
                    else:
                        # Also create non-rounded negated sample to catch sign-dependent
                        # behavior that rounding masks
                        non_rounded_negated_sample = create_fully_negated_sample(
                            non_rounded_sample, tensors
                        )
                        non_rounded_negated_tensors = negate_all_tensors(tensors)

                        if isinstance(non_rounded_negated_sample.input, torch.Tensor):
                            non_rounded_negated_ground_truth = op(
                                non_rounded_negated_sample.input,
                                *non_rounded_negated_sample.args,
                                **non_rounded_negated_sample.kwargs
                            )
                        else:
                            non_rounded_negated_ground_truth = op(
                                *non_rounded_negated_sample.input,
                                *non_rounded_negated_sample.args,
                                **non_rounded_negated_sample.kwargs
                            )

                        if not isinstance(non_rounded_negated_ground_truth, torch.Tensor):
                            non_rounded_negated_sample = None
                except Exception:
                    non_rounded_sample = None
                    non_rounded_ground_truth = None
                    non_rounded_negated_sample = None

            # Get all possible input placement combinations (including Partial)
            input_placement_options = [
                get_1d_input_placements_for_tensor(t, include_partial=True)
                for _, t in tensors
            ]
            output_placement_options = get_1d_output_placements_for_tensor(ground_truth)

            # Query DTensor's single-dim strategy (if available)
            aten_op, non_tensor_args, non_tensor_kwargs = get_aten_op_for_sample(
                op, sample, opinfo.name
            )

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
                # Query op_strategy_funcs for ops like mm, bmm, sum
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

                    # Build args_schema: tensor strategies + captured non-tensor args
                    args_schema = list(input_strategies) + list(non_tensor_args)

                    # Build OpSchema with captured non-tensor kwargs
                    op_schema = OpSchema(aten_op, tuple(args_schema), non_tensor_kwargs)

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

            # Compute ground truth validation
            ground_truth_valid = set()  # Set of (input_placements, output_placement) strings

            gt_start = time.time()
            # Create LocalTensorMode and mesh once per sample for performance
            device = tensors[0][1].device.type if tensors else "cpu"
            with LocalTensorMode(frozenset(range(world_size))):
                mesh = init_device_mesh(device, (world_size,))

                if incorrect_only:
                    # Fast mode: only test DTensor's claimed rules
                    combinations_to_test = []
                    for combo_key in dtensor_rules:
                        input_plc_strs, output_plc_str = combo_key
                        # Parse placement strings back to objects
                        input_plcs = tuple(_parse_placement(s) for s in input_plc_strs)
                        output_plc = _parse_placement(output_plc_str)
                        if input_plcs and output_plc:
                            combinations_to_test.append((input_plcs, output_plc, combo_key))
                else:
                    # Full mode: test all possible combinations
                    combinations_to_test = []
                    for input_placements in itertools.product(*input_placement_options):
                        if is_fully_replicated(input_placements):
                            continue
                        for output_placement in output_placement_options:
                            combo_key = (
                                tuple(str(p) for p in input_placements),
                                str(output_placement)
                            )
                            combinations_to_test.append((input_placements, output_placement, combo_key))

                for input_placements, output_placement, combo_key in combinations_to_test:
                    total_combinations += 1
                    combo = PlacementCombination(input_placements, output_placement)

                    # Validate using ground truth, passing pre-created mesh
                    is_valid, error_msg = validate_combination(
                        op, sample, tensors, combo, ground_truth, world_size, mesh
                    )

                    # For P(min)/P(max) combinations, also test with fully negated inputs
                    # to catch sign-dependent behavior (e.g., R / P(max) -> P(max))
                    if is_valid and fully_negated_sample and has_pmin_pmax(input_placements, output_placement):
                        negated_combo = PlacementCombination(input_placements, output_placement)
                        negated_valid, _ = validate_combination(
                            op, fully_negated_sample, fully_negated_tensors, negated_combo,
                            fully_negated_ground_truth, world_size, mesh
                        )
                        is_valid = is_valid and negated_valid

                    # For samples with rounding_mode, check if the non-rounded version also passes
                    # If rounded passes but non-rounded fails, the rounding is masking real differences
                    if is_valid and non_rounded_sample and has_any_partial(input_placements, output_placement):
                        non_rounded_combo = PlacementCombination(input_placements, output_placement)
                        non_rounded_valid, _ = validate_combination(
                            op, non_rounded_sample, tensors, non_rounded_combo,
                            non_rounded_ground_truth, world_size, mesh
                        )
                        is_valid = is_valid and non_rounded_valid

                    # Also check non-rounded negated to catch rounding-masked sign issues
                    if is_valid and non_rounded_negated_sample and has_pmin_pmax(input_placements, output_placement):
                        non_rounded_negated_combo = PlacementCombination(input_placements, output_placement)
                        non_rounded_negated_valid, _ = validate_combination(
                            op, non_rounded_negated_sample, non_rounded_negated_tensors,
                            non_rounded_negated_combo, non_rounded_negated_ground_truth,
                            world_size, mesh
                        )
                        is_valid = is_valid and non_rounded_negated_valid

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
                            aten_op=aten_op,
                            variant=variant,
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
                            aten_op=aten_op,
                            variant=variant,
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

    # Count distinct rules (unique placement combinations)
    fp_rules = set((d.input_placements, d.output_placement) for d in stats.false_positives)
    fn_rules = set((d.input_placements, d.output_placement) for d in stats.false_negatives)

    print(f"True positives (both agree valid): {stats.true_positives}")
    if stats.false_positives:
        print(f"DTensor incorrect: {len(fp_rules)} rules over {len(stats.false_positives)} samples")
    else:
        print(f"DTensor incorrect: 0")
    if stats.false_negatives:
        print(f"DTensor missing: {len(fn_rules)} rules over {len(stats.false_negatives)} samples")
    else:
        print(f"DTensor missing: 0")

    if stats.false_positives:
        print("\n--- DTENSOR INCORRECT (has rule but ground truth invalid) ---")
        # Group by aten_op first, then by placement combo
        by_op = defaultdict(lambda: defaultdict(list))
        for d in stats.false_positives:
            op_name = str(d.aten_op) if d.aten_op else "(unknown)"
            key = (d.input_placements, d.output_placement)
            by_op[op_name][key].append(d)

        for op_name in sorted(by_op.keys()):
            print(f"\n  [{op_name}]")
            for (inp, out), discrepancies in sorted(by_op[op_name].items(), key=str):
                inp_str = ", ".join(inp)
                print(f"    {inp_str} -> {out}")
                for d in discrepancies[:3]:
                    # Format shapes concisely: [4, 1], [4] instead of torch.Size([4, 1]), torch.Size([4])
                    shapes_str = ", ".join(str(list(s)) for s in d.input_shapes)
                    extra = ""
                    if d.scalar_kwargs:
                        extra = f", {d.scalar_kwargs}"
                    print(f"      Sample {d.sample_idx}: [{shapes_str}]{extra}")
                if len(discrepancies) > 3:
                    print(f"      ... and {len(discrepancies) - 3} more")

    if stats.false_negatives:
        print("\n--- DTENSOR MISSING (ground truth valid but no rule) ---")
        # Group by aten_op first, then by placement combo
        by_op = defaultdict(lambda: defaultdict(list))
        for d in stats.false_negatives:
            op_name = str(d.aten_op) if d.aten_op else "(unknown)"
            key = (d.input_placements, d.output_placement)
            by_op[op_name][key].append(d)

        for op_name in sorted(by_op.keys()):
            print(f"\n  [{op_name}]")
            for (inp, out), discrepancies in sorted(by_op[op_name].items(), key=str):
                inp_str = ", ".join(inp)
                print(f"    {inp_str} -> {out}")
                for d in discrepancies[:3]:
                    shapes_str = ", ".join(str(list(s)) for s in d.input_shapes)
                    extra = ""
                    if d.scalar_kwargs:
                        extra = f", {d.scalar_kwargs}"
                    print(f"      Sample {d.sample_idx}: [{shapes_str}]{extra}")
                if len(discrepancies) > 3:
                    print(f"      ... and {len(discrepancies) - 3} more")

    # Cleanup
    _clear_sharding_prop_cache()
    try:
        dist.destroy_process_group()
    except Exception:
        pass

    return stats


def get_registered_op_names():
    """Get all op names that have DTensor sharding rules and also have OpInfo."""
    from torch.distributed.tensor._api import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator

    # Get all registered aten ops
    all_registered = (
        set(propagator.op_single_dim_strategy_funcs.keys()) |
        set(propagator.op_strategy_funcs.keys())
    )

    # Extract base names (aten.mul.Tensor -> mul)
    base_names = set()
    for op in all_registered:
        parts = str(op).split('.')
        if len(parts) >= 2:
            base_names.add(parts[1])

    # Find which ones have OpInfo
    opinfo_names = set(op.name for op in op_db)
    return sorted(base_names & opinfo_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare DTensor rules against ground truth")
    parser.add_argument("--op", default=None, help="Operator name to compare")
    parser.add_argument("--all-registered", action="store_true",
                        help="Test all ops with DTensor sharding rules registered")
    parser.add_argument("--incorrect-only", action="store_true",
                        help="Only test DTensor's claimed rules (faster, skips missing detection)")
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

    if args.all_registered:
        # Test all ops with registered sharding rules
        op_names = get_registered_op_names()
        print(f"Testing {len(op_names)} ops with DTensor sharding rules")
        if args.incorrect_only:
            print("Mode: incorrect-only (fast)")
        print(f"Device: {args.device}, Dtype: {dtype}")
        print("=" * 70)

        total_stats = ComparisonStats()
        ops_with_errors = []

        for i, op_name in enumerate(op_names):
            if op_name in SKIP_OPS:
                continue

            print(f"\n[{i+1}/{len(op_names)}] {op_name}")
            try:
                stats = compare_operator(
                    op_name,
                    args.device,
                    dtype,
                    args.world_size,
                    args.max_samples,
                    args.verbose,
                    args.incorrect_only,
                )
                total_stats.true_positives += stats.true_positives
                total_stats.false_positives.extend(stats.false_positives)
                total_stats.false_negatives.extend(stats.false_negatives)

                if stats.false_positives:
                    ops_with_errors.append((op_name, len(stats.false_positives)))
            except Exception as e:
                print(f"  Error: {e}")

        # Final summary
        print("\n" + "=" * 70)
        print("OVERALL SUMMARY")
        print("=" * 70)
        print(f"Total ops tested: {len(op_names)}")
        print(f"Total true positives: {total_stats.true_positives}")
        print(f"Total incorrect rules: {len(total_stats.false_positives)}")
        if not args.incorrect_only:
            print(f"Total missing rules: {len(total_stats.false_negatives)}")

        if ops_with_errors:
            print(f"\nOps with INCORRECT rules ({len(ops_with_errors)}):")
            for op_name, count in sorted(ops_with_errors, key=lambda x: -x[1]):
                print(f"  {op_name}: {count}")

    else:
        # Test a single operator
        op_name = args.op or "add"
        print(f"Comparing operator: {op_name}")
        if args.incorrect_only:
            print("Mode: incorrect-only (fast)")
        print(f"Device: {args.device}, Dtype: {dtype}")
        print("=" * 70)

        compare_operator(
            op_name,
            args.device,
            dtype,
            args.world_size,
            args.max_samples,
            args.verbose,
            args.incorrect_only,
        )
