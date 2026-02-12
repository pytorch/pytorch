# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Strategy validation for DTensor sharding rules.

This module provides utilities to validate DTensor's sharding strategies by:
1. Running operators on full tensors to get ground truth
2. Simulating sharding with various placement combinations
3. Comparing redistributed outputs against ground truth
4. Reporting incorrect rules (DTensor claims valid but wrong) and
   missing rules (ground truth valid but DTensor has no rule)

Run as a module to compare DTensor rules against ground truth:
    python -m torch.distributed.tensor._ops.strategy_validation --op add
    python -m torch.distributed.tensor._ops.strategy_validation --all-registered
    python -m torch.distributed.tensor._ops.strategy_validation --op div --incorrect-only
"""

import argparse
import itertools
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import (
    DTensorSpec,
    OpSchema,
    OpSpec,
    OpStrategy,
)
from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder
from torch.distributed.tensor.placement_types import Partial, Placement, Shard
from torch.testing._internal.common_methods_invocations import op_db, SampleInput
from torch.testing._internal.opinfo import core as opinfo_core
from torch.utils import _pytree as pytree


# Partial reduce ops to enumerate
PARTIAL_REDUCE_OPS = ["sum", "avg", "min", "max"]

# Ops to skip in validation because ground truth comparison is not meaningful.
# These produce non-deterministic or uninitialized outputs.
SKIP_OPS = frozenset(
    [
        "bernoulli",  # Random sampling
        "empty_like",  # Uninitialized memory
        "new_empty",  # Uninitialized memory
        "new_empty_strided",  # Uninitialized memory
        "normal",  # Random sampling
        "rand_like",  # Random sampling
        "randint_like",  # Random sampling
        "randn_like",  # Random sampling
        "uniform",  # Random sampling
    ]
)


@dataclass
class PlacementCombination:
    """Represents a combination of input and output placements."""

    input_placements: tuple  # One placement per input tensor
    output_placement: Any  # Placement for the output tensor

    def __hash__(self):
        return hash(
            (tuple(str(p) for p in self.input_placements), str(self.output_placement))
        )

    def __eq__(self, other):
        if not isinstance(other, PlacementCombination):
            return NotImplemented
        return tuple(str(p) for p in self.input_placements) == tuple(
            str(p) for p in other.input_placements
        ) and str(self.output_placement) == str(other.output_placement)

    def __str__(self):
        return f"inputs={placement_tuple_to_str(self.input_placements)}, output={placement_tuple_to_str((self.output_placement,))}"


@dataclass
class Discrepancy:
    """Represents a discrepancy between ground truth and DTensor's rules."""

    input_placements: tuple
    output_placement: Any
    sample_idx: int
    input_shapes: tuple
    discrepancy_type: str  # "false_positive" or "false_negative"
    error_msg: str = ""
    scalar_args: tuple = ()
    scalar_kwargs: dict = field(default_factory=dict)
    aten_op: Any = None
    variant: str = ""


@dataclass
class ComparisonStats:
    """Statistics for comparing ground truth vs DTensor rules."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: list = field(
        default_factory=list
    )  # DTensor has rule, ground truth says invalid
    false_negatives: list = field(
        default_factory=list
    )  # Ground truth valid, DTensor has no rule


@dataclass
class _FalsePositiveMitigations:
    """Bundle of sample variants used to detect false positive validations.

    Contains negated and non-rounded variants of a sample. These are used to
    re-test combinations that pass the primary validation, catching cases where
    sign patterns or rounding modes mask real differences.
    """

    negated_sample: Any = None
    negated_tensors: list | None = None
    negated_ground_truth: torch.Tensor | None = None
    non_rounded_sample: Any = None
    non_rounded_ground_truth: torch.Tensor | None = None
    non_rounded_negated_sample: Any = None
    non_rounded_negated_tensors: list | None = None
    non_rounded_negated_ground_truth: torch.Tensor | None = None


def placement_tuple_to_str(placements: tuple) -> str:
    """Convert a tuple of placements to a readable string."""
    parts: list[str] = []
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


def parse_placement(s: str) -> Placement | None:
    """
    Parse a placement string back to a placement object.
    Placement strings are: R, S(dim), P(reduce_op)
    """
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


def is_fully_replicated(placements: tuple) -> bool:
    """Check if all placements are Replicate."""
    return all(isinstance(p, Replicate) for p in placements)


def is_trivial_shard(p, shape: tuple[int, ...]) -> bool:
    """Check if placement is a Shard on a size-1 dimension (equivalent to Replicate)."""
    return isinstance(p, Shard) and p.dim < len(shape) and shape[p.dim] == 1


def normalize_placement(p, shape: tuple[int, ...]):
    """
    Normalize a placement for a given tensor shape.

    Converts trivial Shard (on size-1 dimension) to Replicate, since they
    are semantically equivalent.
    """
    if is_trivial_shard(p, shape):
        return Replicate()
    return p


def normalize_placement_str(p_str: str, shape: tuple[int, ...]) -> str:
    """Normalize a placement string, converting trivial shards to Replicate."""
    p = parse_placement(p_str)
    if p is None:
        return p_str
    normalized = normalize_placement(p, shape)
    if isinstance(normalized, Replicate):
        return "R"
    return p_str


def normalize_combo_key(
    combo_key: tuple,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
) -> tuple:
    """
    Normalize a combo_key by converting trivial shards to Replicate.

    This deduplicates equivalent placement combinations, e.g.:
    - P(max) -> S(0) on output [1,1,1] becomes P(max) -> R
    - S(0), R -> R on input [1,4] becomes R, R -> R

    Args:
        combo_key: (input_placement_strs, output_placement_str) tuple
        input_shapes: Shapes of input tensors
        output_shape: Shape of output tensor

    Returns:
        Normalized combo_key with trivial shards converted to Replicate
    """
    input_placement_strs, output_placement_str = combo_key

    # Normalize input placements
    normalized_inputs = tuple(
        normalize_placement_str(p_str, shape)
        for p_str, shape in zip(input_placement_strs, input_shapes)
    )

    # Normalize output placement
    normalized_output = normalize_placement_str(output_placement_str, output_shape)

    return (normalized_inputs, normalized_output)


def get_1d_input_placements_for_tensor(
    t: torch.Tensor, include_partial: bool = False
) -> list:
    """
    Get all possible 1-D mesh placements for an INPUT tensor.

    Args:
        t: The tensor to get placements for
        include_partial: If True, include Partial placements for inputs.
    """
    placements: list[Placement] = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))
    if include_partial and t.dtype != torch.bool:
        for reduce_op in PARTIAL_REDUCE_OPS:
            placements.append(Partial(reduce_op))
    return placements


def get_1d_output_placements_for_tensor(t: torch.Tensor) -> list:
    """
    Get all possible 1-D mesh placements for an OUTPUT tensor.

    For integer outputs, only P(min) and P(max) are included since
    P(avg) requires division (which truncates for integers) and P(sum)
    can overflow discrete-valued outputs where summation is unexpected.
    """
    placements: list[Placement] = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))

    is_integer = not t.dtype.is_floating_point and not t.dtype.is_complex
    if t.dtype != torch.bool:
        for reduce_op in PARTIAL_REDUCE_OPS:
            if is_integer and reduce_op in ("sum", "avg"):
                continue
            placements.append(Partial(reduce_op))
    return placements


def extract_tensors_from_sample(sample_input) -> list:
    """
    Extract all tensor arguments from a SampleInput.
    Returns a list of (name, tensor) pairs.

    Uses pytree traversal to match the same order as _replace_with_local
    in validate_combination, which uses pytree.tree_map on the same structures.
    """
    tensors: list[tuple[str, torch.Tensor]] = []
    idx = 0

    def _collect(x):
        nonlocal idx
        if isinstance(x, torch.Tensor):
            tensors.append((f"tensor_{idx}", x))
            idx += 1
        return x

    pytree.tree_map(_collect, sample_input.input)
    pytree.tree_map(_collect, sample_input.args)
    pytree.tree_map(_collect, sample_input.kwargs)

    return tensors


def _create_partial_input(
    tensor: torch.Tensor, placement: Partial, world_size: int, tensor_idx: int = 0
) -> LocalTensor:
    """
    Create a LocalTensor with values that reduce to the original tensor.

    Uses asymmetric splits to avoid coincidental matches when combining
    different Partial types.
    """
    reduce_op = placement.reduce_op
    local_tensors: dict[int, torch.Tensor] = {}

    if reduce_op in ("sum", "avg"):
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)

        # Add a sign-varying offset so local values have mixed signs.
        # Pure proportional splits (tensor * ratio) preserve sign patterns,
        # causing non-linear ops like abs to falsely validate P(sum)->P(sum).
        flat = tensor.flatten()
        offset_mag = flat.abs() + 1.0
        signs = torch.ones_like(flat)
        signs[
            (torch.arange(flat.numel(), device=tensor.device) + tensor_idx) % 2 == 0
        ] = -1.0
        offset = (offset_mag * signs).reshape(tensor.shape)

        scale = world_size if reduce_op == "avg" else 1
        for r in range(world_size):
            if r == 0:
                local_tensors[r] = tensor.clone() * base_ratio * scale + offset
            else:
                local_tensors[r] = tensor.clone() * (
                    (1 - base_ratio) / (world_size - 1)
                ) * scale - offset / (world_size - 1)

    elif reduce_op == "min":
        # For P(min): on each element, one rank holds the true value (offset=0)
        # and the other holds value+0.7. min() selects the unmodified value.
        # The mask alternates which rank holds the true value (shifted by tensor_idx).
        # 0.7 is arbitrary; any positive value works. Using a different magnitude
        # than max's 1.3 prevents accidental cancellation when min/max are combined.
        flat = tensor.flatten()
        mask = (torch.arange(flat.numel(), device=tensor.device) + tensor_idx) % 2 == 0
        for r in range(world_size):
            if r == 0:
                r_offset = torch.where(
                    mask, torch.zeros_like(flat), torch.full_like(flat, 0.7)
                )
            else:
                r_offset = torch.where(
                    mask, torch.full_like(flat, 0.7), torch.zeros_like(flat)
                )
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)

    elif reduce_op == "max":
        # For P(max): on each element, one rank holds the true value (offset=0)
        # and the other holds value-1.3. max() selects the unmodified value.
        # 1.3 is arbitrary; any positive magnitude works.
        flat = tensor.flatten()
        mask = (torch.arange(flat.numel(), device=tensor.device) + tensor_idx) % 2 == 0
        for r in range(world_size):
            if r == 0:
                r_offset = torch.where(
                    mask, torch.zeros_like(flat), torch.full_like(flat, -1.3)
                )
            else:
                r_offset = torch.where(
                    mask, torch.full_like(flat, -1.3), torch.zeros_like(flat)
                )
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)

    else:
        for r in range(world_size):
            local_tensors[r] = tensor.clone()

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    return LocalTensor(local_tensors)


def validate_combination(
    op: Callable,
    sample_input,
    tensors: list,
    combination: PlacementCombination,
    ground_truth: torch.Tensor,
    world_size: int = 2,
    mesh=None,
) -> tuple[bool, str]:
    """
    Validate a single placement combination against ground truth.

    The validation logic:
    1. Shard inputs according to input placements to get local tensors
    2. Run the raw op on local tensors (bypassing DTensor dispatch)
    3. Wrap the local output in a DTensor with the claimed output placement
    4. Redistribute to Replicate and compare with ground truth

    Args:
        op: The operator function
        sample_input: The SampleInput with original arguments
        tensors: List of (name, tensor) pairs extracted from sample
        combination: The placement combination to validate
        ground_truth: Expected output tensor
        world_size: Number of simulated ranks
        mesh: Optional pre-created device mesh (for performance)

    Returns:
        (is_valid, error_message)
    """
    try:
        if mesh is None:
            device = tensors[0][1].device.type if tensors else "cpu"
            mesh = init_device_mesh(device, (world_size,))

        local_tensors = []
        for tensor_idx, ((name, tensor), placement) in enumerate(
            zip(tensors, combination.input_placements)
        ):
            if isinstance(placement, Partial):
                local_tensor = _create_partial_input(
                    tensor, placement, world_size, tensor_idx
                )
                local_tensors.append(local_tensor)
            elif isinstance(placement, Replicate):
                _tmp = {r: tensor.clone() for r in range(world_size)}
                # pyrefly: ignore [bad-argument-type, bad-argument-count]
                local_tensors.append(LocalTensor(_tmp))
            elif isinstance(placement, Shard):
                # Create sharded LocalTensor directly to work in LocalTensorMode
                shard_dim = placement.dim
                chunks = tensor.tensor_split(world_size, dim=shard_dim)
                _tmp = {r: chunks[r].clone().contiguous() for r in range(world_size)}
                # pyrefly: ignore [bad-argument-type, bad-argument-count]
                local_tensors.append(LocalTensor(_tmp))
            else:
                # Fallback for other placement types
                dt = distribute_tensor(tensor.clone(), mesh, (placement,))
                local_tensors.append(dt.to_local())

        local_idx = 0

        def _replace_with_local(a):
            nonlocal local_idx
            if isinstance(a, torch.Tensor):
                local = local_tensors[local_idx]
                local_idx += 1
                return local
            return a

        if isinstance(sample_input.input, torch.Tensor):
            local_input = _replace_with_local(sample_input.input)
        else:
            local_input = pytree.tree_map(_replace_with_local, sample_input.input)

        local_args = pytree.tree_map(_replace_with_local, sample_input.args)
        local_kwargs = pytree.tree_map(_replace_with_local, sample_input.kwargs)

        local_output = op(local_input, *local_args, **local_kwargs)

        if not isinstance(local_output, torch.Tensor):
            return False, f"Local output is not a tensor: {type(local_output)}"

        if not isinstance(local_output, LocalTensor):
            return False, "LocalTensor inputs produced non-LocalTensor output"

        output_dt = DTensor.from_local(
            local_output,
            mesh,
            (combination.output_placement,),
            shape=ground_truth.shape,
            stride=ground_truth.stride(),
        )

        if isinstance(combination.output_placement, Replicate):
            local_values = [local_output._local_tensors[r] for r in range(world_size)]
            all_same = all(
                torch.allclose(local_values[0], lv, atol=1e-5, rtol=1e-5)
                for lv in local_values[1:]
            )
            if not all_same:
                return False, "Replicate output but local values differ across ranks"

        full_output = output_dt.redistribute(mesh, (Replicate(),)).to_local()

        if isinstance(full_output, LocalTensor):
            full_output = full_output._local_tensors[0]

        if ground_truth.shape != full_output.shape:
            return (
                False,
                f"Shape mismatch: expected {ground_truth.shape}, got {full_output.shape}",
            )

        if not torch.allclose(
            ground_truth, full_output, atol=1e-5, rtol=1e-5, equal_nan=True
        ):
            max_diff = (ground_truth - full_output).abs().max().item()
            return False, f"Value mismatch: max_diff={max_diff:.6f}"

        return True, ""

    except Exception as e:
        # TODO: This is too broad. Consider: (1) explicit checks for shard dim
        # validity and shape compatibility before calling tensor_split/from_local,
        # (2) scoped try/except around op() and redistribute() that raise specific
        # exceptions (e.g., UnsupportedRedistribute, OpError), and (3) only
        # catching those here, letting real bugs propagate.
        return False, f"Exception: {type(e).__name__}: {e}"


def has_pmin_pmax(input_placements, output_placement) -> bool:
    """Check if any placement is Partial(min) or Partial(max)."""
    for p in input_placements:
        if isinstance(p, Partial) and p.reduce_op in ("min", "max"):
            return True
    if isinstance(output_placement, Partial) and output_placement.reduce_op in (
        "min",
        "max",
    ):
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


def negate_all_tensors(tensors: list) -> list:
    """Return a new list with all tensors negated."""
    return [(name, -t) for name, t in tensors]


def create_fully_negated_sample(sample):
    """Create a sample with ALL tensors negated (for P(min)/P(max) sign testing)."""

    def negate_tensor(x):
        if isinstance(x, torch.Tensor):
            return -x
        return x

    new_input = pytree.tree_map(negate_tensor, sample.input)
    new_args = pytree.tree_map(negate_tensor, sample.args)
    new_kwargs = pytree.tree_map(negate_tensor, sample.kwargs)

    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)


def _run_op_on_sample(op, sample):
    """Run an operator on a SampleInput, handling both tensor and tuple inputs."""
    if isinstance(sample.input, torch.Tensor):
        return op(sample.input, *sample.args, **sample.kwargs)
    return op(*sample.input, *sample.args, **sample.kwargs)


def _extract_rules_from_op_strategy(op_strategy, input_shapes, output_shape):
    """Extract normalized sharding rules from an OpStrategy."""
    rules = set()
    if not isinstance(op_strategy, OpStrategy):
        return rules
    for spec in op_strategy.strategies:
        if spec.input_specs is None:
            continue
        output_plc = spec.output_spec.placements[0]
        input_plcs = tuple(s.placements[0] for s in spec.input_specs)
        rule_key = (tuple(str(p) for p in input_plcs), str(output_plc))
        normalized_rule = normalize_combo_key(rule_key, input_shapes, output_shape)
        if not is_fully_replicated(
            tuple(parse_placement(p) or Replicate() for p in normalized_rule[0])
        ):
            rules.add(normalized_rule)
    return rules


class _CaptureAtenOp(torch.utils._python_dispatch.TorchDispatchMode):
    """Dispatch mode that captures aten ops called and their args."""

    def __init__(self, target_op_name: str = ""):
        self.target_op_name = target_op_name.lower()
        self.all_ops: list[tuple] = []
        self.best_match: Any = None
        self.best_match_args: tuple | None = None
        self.best_match_kwargs: dict | None = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func.namespace == "aten":
            self.all_ops.append((func, args, kwargs))
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
    """
    with _CaptureAtenOp(op_name) as capture:
        try:
            if isinstance(sample.input, torch.Tensor):
                op(sample.input, *sample.args, **sample.kwargs)
            else:
                op(*sample.input, *sample.args, **sample.kwargs)
        except Exception:
            pass

    if capture.best_match is not None:
        captured_op = capture.best_match
        captured_args = capture.best_match_args
        captured_kwargs = capture.best_match_kwargs
    elif capture.all_ops:
        captured_op, captured_args, captured_kwargs = capture.all_ops[0]
    else:
        return None, (), {}

    non_tensor_args = tuple(a for a in captured_args if not isinstance(a, torch.Tensor))
    non_tensor_kwargs = {
        k: v for k, v in captured_kwargs.items() if not isinstance(v, torch.Tensor)
    }

    return captured_op, non_tensor_args, non_tensor_kwargs


def query_single_dim_strategy(op_overload, tensors, mesh, kwargs=None):
    """
    Query DTensor's single-dim strategy for given input tensors.
    Returns list of [output_placement, *input_placements] rules.
    """
    propagator = DTensor._op_dispatcher.sharding_propagator

    if op_overload not in propagator.op_single_dim_strategy_funcs:
        return None

    strategy_func = propagator.op_single_dim_strategy_funcs[op_overload]

    args_meta = tuple(
        TensorMeta(shape=t.shape, stride=t.stride(), dtype=t.dtype) for _, t in tensors
    )

    try:
        result = strategy_func(op_overload, args_meta, kwargs or {})

        expanded_result = []
        for combo in result:
            expanded_combo = []
            for p in combo:
                if isinstance(p, _ShardingPlaceholder):
                    expanded_combo.append(Shard(p.dim))
                else:
                    expanded_combo.append(p)
            expanded_result.append(expanded_combo)

        return expanded_result
    except Exception:
        return None


def get_opinfo_by_name(name: str):
    """Find OpInfo entries by operator name."""
    matches = [op for op in op_db if op.name == name]
    if not matches:
        raise ValueError(f"No OpInfo found for operator: {name}")
    return matches


def _prepare_false_positive_mitigations(op, sample, tensors):
    """Create negated and non-rounded sample variants for false positive detection."""
    m = _FalsePositiveMitigations()

    try:
        m.negated_sample = create_fully_negated_sample(sample)
        m.negated_tensors = negate_all_tensors(tensors)
        result = _run_op_on_sample(op, m.negated_sample)
        if isinstance(result, torch.Tensor):
            m.negated_ground_truth = result
        else:
            m.negated_sample = None
    except Exception:
        m.negated_sample = None
        m.negated_tensors = None

    if "rounding_mode" not in sample.kwargs:
        return m

    try:
        non_rounded_kwargs = {
            k: v for k, v in sample.kwargs.items() if k != "rounding_mode"
        }
        m.non_rounded_sample = SampleInput(
            sample.input, args=sample.args, kwargs=non_rounded_kwargs
        )
        result = _run_op_on_sample(op, m.non_rounded_sample)
        if not isinstance(result, torch.Tensor):
            m.non_rounded_sample = None
        else:
            m.non_rounded_ground_truth = result
            m.non_rounded_negated_sample = create_fully_negated_sample(
                m.non_rounded_sample
            )
            m.non_rounded_negated_tensors = negate_all_tensors(tensors)
            nr_neg_result = _run_op_on_sample(op, m.non_rounded_negated_sample)
            if isinstance(nr_neg_result, torch.Tensor):
                m.non_rounded_negated_ground_truth = nr_neg_result
            else:
                m.non_rounded_negated_sample = None
    except Exception:
        m.non_rounded_sample = None
        m.non_rounded_ground_truth = None
        m.non_rounded_negated_sample = None

    return m


def _query_dtensor_rules(
    aten_op,
    tensors,
    non_tensor_args,
    non_tensor_kwargs,
    input_shapes,
    output_shape,
    world_size,
    verbose,
):
    """Query DTensor's claimed sharding rules via single-dim, op_strategy, or decomp paths.

    TODO: This reimplements strategy resolution logic from ShardingPropagator.
    Refactor ShardingPropagator to expose a public API for querying sharding
    rules given an op and tensor metadata, so this function can be replaced
    with a single call.
    """
    if not aten_op:
        return set()

    propagator = DTensor._op_dispatcher.sharding_propagator
    rules: set = set()

    if aten_op in propagator.op_single_dim_strategy_funcs:
        strategy_result = query_single_dim_strategy(
            aten_op, tensors, None, kwargs=non_tensor_kwargs
        )
        if strategy_result:
            for combo in strategy_result:
                if len(combo) >= len(tensors) + 1:
                    output_plc = combo[0]
                    input_plcs = tuple(combo[1 : len(tensors) + 1])
                    rule_key = (
                        tuple(str(p) for p in input_plcs),
                        str(output_plc),
                    )
                    normalized_rule = normalize_combo_key(
                        rule_key, input_shapes, output_shape
                    )
                    if not is_fully_replicated(
                        tuple(
                            parse_placement(p) or Replicate()
                            for p in normalized_rule[0]
                        )
                    ):
                        rules.add(normalized_rule)

    elif aten_op in propagator.op_strategy_funcs:
        try:
            mesh = init_device_mesh("cpu", (world_size,))
            input_strategies = []
            for name, t in tensors:
                input_placements = get_1d_input_placements_for_tensor(
                    t, include_partial=True
                )
                specs = []
                for p in input_placements:
                    spec = DTensorSpec(
                        mesh=mesh,
                        placements=(p,),
                        tensor_meta=TensorMeta(
                            shape=t.shape, stride=t.stride(), dtype=t.dtype
                        ),
                    )
                    specs.append(OpSpec(output_specs=spec, input_specs=tuple()))
                input_strategies.append(OpStrategy(specs))
            args_schema = list(input_strategies) + list(non_tensor_args)
            op_schema = OpSchema(aten_op, tuple(args_schema), non_tensor_kwargs)
            strategy_func = propagator.op_strategy_funcs[aten_op]
            output_strategy = strategy_func(op_schema)
            rules |= _extract_rules_from_op_strategy(
                output_strategy, input_shapes, output_shape
            )
        except Exception as e:
            if verbose:
                print(f"        Error querying op_strategy: {e}")

    else:
        # Decomp-based strategy: only discovers rules reachable from a single
        # seed (Shard(0) on the first input). Rules requiring other input
        # placements (e.g., Shard(1), Partial, or sharding on non-first inputs)
        # will not be found, so this under-reports DTensor's capabilities.
        try:
            from torch.distributed.tensor._decompositions import DecompShardingStrategy
        except ImportError:
            DecompShardingStrategy = None  # type: ignore[assignment, misc]
        if DecompShardingStrategy is not None and DecompShardingStrategy.has_decomp(
            aten_op
        ):
            try:
                mesh = init_device_mesh("cpu", (world_size,))
                args_schema = []
                for i, (_, t) in enumerate(tensors):
                    # First tensor gets Shard(0) to seed candidate
                    # placement generation in _get_candidate_placements
                    plc = Shard(0) if i == 0 else Replicate()
                    spec = DTensorSpec(
                        mesh=mesh,
                        placements=(plc,),
                        tensor_meta=TensorMeta(
                            shape=t.shape, stride=t.stride(), dtype=t.dtype
                        ),
                    )
                    args_schema.append(spec)
                args_schema.extend(non_tensor_args)
                op_schema = OpSchema(aten_op, tuple(args_schema), non_tensor_kwargs)
                DecompShardingStrategy.ensure_schema_info(aten_op, propagator)
                output_strategy = DecompShardingStrategy.propagate_strategy(
                    op_schema, propagator
                )
                if output_strategy is not None:
                    rules |= _extract_rules_from_op_strategy(
                        output_strategy, input_shapes, output_shape
                    )
            except Exception as e:
                if verbose:
                    print(f"        Error querying decomp strategy: {e}")

    return rules


def _validate_with_mitigations(
    op,
    sample,
    tensors,
    input_placements,
    output_placement,
    ground_truth,
    world_size,
    mesh,
    mitigations,
):
    """Validate a combination, including false positive mitigation re-checks."""
    combo = PlacementCombination(input_placements, output_placement)
    is_valid, _ = validate_combination(
        op, sample, tensors, combo, ground_truth, world_size, mesh
    )

    if (
        is_valid
        and mitigations.negated_sample
        and has_pmin_pmax(input_placements, output_placement)
    ):
        assert mitigations.negated_tensors is not None
        assert mitigations.negated_ground_truth is not None
        negated_combo = PlacementCombination(input_placements, output_placement)
        negated_valid, _ = validate_combination(
            op,
            mitigations.negated_sample,
            mitigations.negated_tensors,
            negated_combo,
            mitigations.negated_ground_truth,
            world_size,
            mesh,
        )
        is_valid = is_valid and negated_valid

    if (
        is_valid
        and mitigations.non_rounded_sample
        and has_any_partial(input_placements, output_placement)
    ):
        assert mitigations.non_rounded_ground_truth is not None
        non_rounded_combo = PlacementCombination(input_placements, output_placement)
        non_rounded_valid, _ = validate_combination(
            op,
            mitigations.non_rounded_sample,
            tensors,
            non_rounded_combo,
            mitigations.non_rounded_ground_truth,
            world_size,
            mesh,
        )
        is_valid = is_valid and non_rounded_valid

    if (
        is_valid
        and mitigations.non_rounded_negated_sample
        and has_pmin_pmax(input_placements, output_placement)
    ):
        assert mitigations.non_rounded_negated_tensors is not None
        assert mitigations.non_rounded_negated_ground_truth is not None
        nr_negated_combo = PlacementCombination(input_placements, output_placement)
        nr_negated_valid, _ = validate_combination(
            op,
            mitigations.non_rounded_negated_sample,
            mitigations.non_rounded_negated_tensors,
            nr_negated_combo,
            mitigations.non_rounded_negated_ground_truth,
            world_size,
            mesh,
        )
        is_valid = is_valid and nr_negated_valid

    return is_valid


def _assert_keys_normalized(keys, input_shapes, output_shape):
    """Assert all combo keys have trivial shards already normalized to Replicate."""
    for key in keys:
        assert key == normalize_combo_key(key, input_shapes, output_shape), (
            f"Key {key} contains un-normalized trivial shards; "
            f"call normalize_combo_key before _compare_rules"
        )


def _compare_rules(
    ground_truth_valid,
    dtensor_rules,
    input_shapes,
    output_shape,
    sample_idx,
    scalar_args,
    scalar_kwargs,
    aten_op,
    variant,
    stats,
):
    """Compare ground truth valid rules against DTensor claimed rules, updating stats."""
    if not dtensor_rules:
        return

    _assert_keys_normalized(ground_truth_valid, input_shapes, output_shape)
    _assert_keys_normalized(dtensor_rules, input_shapes, output_shape)

    for combo_key in ground_truth_valid:
        if combo_key in dtensor_rules:
            stats.true_positives += 1
        else:
            stats.false_negatives.append(
                Discrepancy(
                    input_placements=combo_key[0],
                    output_placement=combo_key[1],
                    sample_idx=sample_idx,
                    input_shapes=input_shapes,
                    discrepancy_type="false_negative",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    aten_op=aten_op,
                    variant=variant,
                )
            )

    for combo_key in dtensor_rules:
        if combo_key not in ground_truth_valid:
            stats.false_positives.append(
                Discrepancy(
                    input_placements=combo_key[0],
                    output_placement=combo_key[1],
                    sample_idx=sample_idx,
                    input_shapes=input_shapes,
                    discrepancy_type="false_positive",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    aten_op=aten_op,
                    variant=variant,
                )
            )


def _print_discrepancy_section(title, discrepancies):
    """Print grouped discrepancies for a section (incorrect or missing)."""
    if not discrepancies:
        return
    print(f"\n--- {title} ---")
    by_op: dict = defaultdict(lambda: defaultdict(list))
    for d in discrepancies:
        op_str = str(d.aten_op) if d.aten_op else "(unknown)"
        key = (d.input_placements, d.output_placement)
        by_op[op_str][key].append(d)

    for op_str in sorted(by_op.keys()):
        print(f"\n  [{op_str}]")
        for (inp, out), discs in sorted(  # pyrefly: ignore[not-iterable]
            by_op[op_str].items(), key=str
        ):
            inp_str = ", ".join(inp)
            print(f"    {inp_str} -> {out}")
            for d in discs[:3]:
                shapes_str = ", ".join(str(list(s)) for s in d.input_shapes)
                extra = ""
                if d.scalar_kwargs:
                    extra = f", {d.scalar_kwargs}"
                print(f"      Sample {d.sample_idx}: [{shapes_str}]{extra}")
            if len(discs) > 3:
                print(f"      ... and {len(discs) - 3} more")


def _print_comparison_summary(
    stats,
    total_samples,
    total_combinations,
    elapsed_time,
    strategy_query_time,
    ground_truth_time,
):
    """Print the comparison summary report."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Total samples processed: {total_samples}")
    print(f"Total combinations tested: {total_combinations}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    if elapsed_time > 0:
        print(
            f"  - Strategy query time: {strategy_query_time:.2f}s ({100 * strategy_query_time / elapsed_time:.1f}%)"
        )
        print(
            f"  - Ground truth time: {ground_truth_time:.2f}s ({100 * ground_truth_time / elapsed_time:.1f}%)"
        )
    print()

    fp_rules = {(d.input_placements, d.output_placement) for d in stats.false_positives}
    fn_rules = {(d.input_placements, d.output_placement) for d in stats.false_negatives}

    print(f"True positives (both agree valid): {stats.true_positives}")
    if stats.false_positives:
        print(
            f"DTensor incorrect: {len(fp_rules)} rules over {len(stats.false_positives)} samples"
        )
    else:
        print("DTensor incorrect: 0")
    if stats.false_negatives:
        print(
            f"DTensor missing: {len(fn_rules)} rules over {len(stats.false_negatives)} samples"
        )
    else:
        print("DTensor missing: 0")

    _print_discrepancy_section(
        "DTENSOR INCORRECT (has rule but ground truth invalid)", stats.false_positives
    )
    _print_discrepancy_section(
        "DTENSOR MISSING (ground truth valid but no rule)", stats.false_negatives
    )


def compare_operator(
    op_name: str,
    device: str = "cpu",
    dtype=torch.float32,
    world_size: int = 2,
    max_samples: int | None = None,
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
    if op_name in SKIP_OPS:
        print(f"Skipping '{op_name}' (non-deterministic or uninitialized output)")
        return ComparisonStats()

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

            if any(0 in t.shape for _, t in tensors):
                continue

            total_samples += 1

            try:
                ground_truth = _run_op_on_sample(op, sample)
                if not isinstance(ground_truth, torch.Tensor):
                    continue
                if ground_truth.numel() > 0 and (ground_truth == 0).all():
                    total_samples -= 1
                    continue
                if ground_truth.numel() > 0 and ground_truth.isnan().all():
                    total_samples -= 1
                    continue
            except Exception:
                continue

            input_shapes = tuple(t.shape for _, t in tensors)
            output_shape = tuple(ground_truth.shape)

            scalar_args = tuple(
                a for a in sample.args if not isinstance(a, torch.Tensor)
            )
            scalar_kwargs = {
                k: v
                for k, v in sample.kwargs.items()
                if not isinstance(v, torch.Tensor)
            }

            mitigations = _prepare_false_positive_mitigations(op, sample, tensors)

            input_placement_options = [
                get_1d_input_placements_for_tensor(t, include_partial=True)
                for _, t in tensors
            ]
            output_placement_options = get_1d_output_placements_for_tensor(ground_truth)

            aten_op, non_tensor_args, non_tensor_kwargs = get_aten_op_for_sample(
                op, sample, opinfo.name
            )

            strategy_start = time.time()
            dtensor_rules = _query_dtensor_rules(
                aten_op,
                tensors,
                non_tensor_args,
                non_tensor_kwargs,
                input_shapes,
                output_shape,
                world_size,
                verbose,
            )
            strategy_query_time += time.time() - strategy_start

            ground_truth_valid: set = set()

            gt_start = time.time()
            tensor_device = tensors[0][1].device.type if tensors else "cpu"
            with LocalTensorMode(frozenset(range(world_size))):
                mesh = init_device_mesh(tensor_device, (world_size,))

                if incorrect_only:
                    combinations_to_test = []
                    for combo_key in dtensor_rules:
                        input_plc_strs, output_plc_str = combo_key
                        input_plcs = tuple(parse_placement(s) for s in input_plc_strs)
                        output_plc = parse_placement(output_plc_str)
                        if (
                            not any(p is None for p in input_plcs)
                            and output_plc is not None
                        ):
                            combinations_to_test.append(
                                (input_plcs, output_plc, combo_key)
                            )
                else:
                    combinations_to_test = []
                    for input_placements in itertools.product(*input_placement_options):
                        if is_fully_replicated(input_placements):
                            continue
                        for output_placement in output_placement_options:
                            combo_key = (
                                tuple(str(p) for p in input_placements),
                                str(output_placement),
                            )
                            combinations_to_test.append(
                                (input_placements, output_placement, combo_key)
                            )

                for (
                    input_placements,
                    output_placement,
                    combo_key,
                ) in combinations_to_test:
                    total_combinations += 1
                    is_valid = _validate_with_mitigations(
                        op,
                        sample,
                        tensors,
                        input_placements,
                        output_placement,
                        ground_truth,
                        world_size,
                        mesh,
                        mitigations,
                    )

                    if is_valid:
                        normalized_key = normalize_combo_key(
                            combo_key, input_shapes, output_shape
                        )
                        if not is_fully_replicated(
                            tuple(
                                parse_placement(p) or Replicate()
                                for p in normalized_key[0]
                            )
                        ):
                            ground_truth_valid.add(normalized_key)
            ground_truth_time += time.time() - gt_start

            _compare_rules(
                ground_truth_valid,
                dtensor_rules,
                input_shapes,
                output_shape,
                sample_idx,
                scalar_args,
                scalar_kwargs,
                aten_op,
                variant,
                stats,
            )

            if verbose:
                print(f"      Sample {sample_idx}: shapes={input_shapes}")
                print(f"        Ground truth valid: {len(ground_truth_valid)}")
                print(f"        DTensor rules: {len(dtensor_rules)}")

    _print_comparison_summary(
        stats,
        total_samples,
        total_combinations,
        time.time() - start_time,
        strategy_query_time,
        ground_truth_time,
    )

    return stats


def get_registered_op_names():
    """Get all op names that have DTensor sharding rules and also have OpInfo."""

    propagator = DTensor._op_dispatcher.sharding_propagator

    # Get all registered aten ops
    all_registered = set(propagator.op_single_dim_strategy_funcs.keys()) | set(
        propagator.op_strategy_funcs.keys()
    )

    # Extract base names (aten.mul.Tensor -> mul)
    base_names = set()
    for op in all_registered:
        parts = str(op).split(".")
        if len(parts) >= 2:
            base_names.add(parts[1])

    # Find which ones have OpInfo
    opinfo_names = {op.name for op in op_db}
    return sorted(base_names & opinfo_names)


if __name__ == "__main__":
    # Override common size variables to ensure even sharding across world_size=2.
    # These are process-global mutations, but this is a CLI entry point so the
    # process exits after running.
    opinfo_core.L = 24  # pyrefly: ignore[bad-assignment]
    opinfo_core.M = 12  # pyrefly: ignore[bad-assignment]
    opinfo_core.S = 4  # pyrefly: ignore[bad-assignment]
    opinfo_core.XS = 2  # pyrefly: ignore[bad-assignment]

    common_ops.L = 24  # pyrefly: ignore[bad-assignment]
    common_ops.M = 12  # pyrefly: ignore[bad-assignment]
    common_ops.S = 4  # pyrefly: ignore[bad-assignment]
    common_ops.XS = 2  # pyrefly: ignore[bad-assignment]

    parser = argparse.ArgumentParser(
        description="Compare DTensor rules against ground truth"
    )
    parser.add_argument("--op", default=None, help="Operator name to compare")
    parser.add_argument(
        "--all-registered",
        action="store_true",
        help="Test all ops with DTensor sharding rules registered",
    )
    parser.add_argument(
        "--incorrect-only",
        action="store_true",
        help="Only test DTensor's claimed rules (faster, skips missing detection)",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="float32", help="Dtype to use")
    parser.add_argument(
        "--world-size", type=int, default=2, help="Simulated world size"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to test"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    dist.init_process_group("fake", rank=0, world_size=args.world_size)
    try:
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

                print(f"\n[{i + 1}/{len(op_names)}] {op_name}")
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
    finally:
        dist.destroy_process_group()
