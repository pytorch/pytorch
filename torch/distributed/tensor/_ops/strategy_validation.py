# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Strategy validation for DTensor sharding rules.

This module provides utilities to validate DTensor's sharding strategies by:
1. Running operators on full tensors to get ground truth
2. Simulating sharding with various placement combinations
3. Comparing redistributed outputs against ground truth
4. Reporting incorrect rules (DTensor claims valid but wrong) and
   missing rules (ground truth valid but DTensor has no rule)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed._local_tensor import LocalTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate
from torch.distributed.tensor.placement_types import Partial, Shard
from torch.utils import _pytree as pytree


if TYPE_CHECKING:
    from collections.abc import Callable


# Partial reduce ops to enumerate
PARTIAL_REDUCE_OPS = ["sum", "avg", "min", "max"]


@dataclass
class PlacementCombination:
    """Represents a combination of input and output placements."""

    input_placements: tuple  # One placement per input tensor
    output_placement: Any  # Placement for the output tensor

    def __hash__(self):
        return hash((self.input_placements, str(self.output_placement)))

    def __eq__(self, other):
        return self.input_placements == other.input_placements and str(
            self.output_placement
        ) == str(other.output_placement)

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


def parse_placement(s: str):
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


def placements_equivalent(p1, p2, shape: tuple[int, ...]) -> bool:
    """
    Check if two placements are equivalent for a given tensor shape.

    Shard(dim) is equivalent to Replicate() when shape[dim] == 1, because
    sharding a size-1 dimension produces the same result as replicating.
    """

    def is_trivial_shard(p):
        return isinstance(p, Shard) and p.dim < len(shape) and shape[p.dim] == 1

    # Check if both are trivial shards (equivalent to each other and to Replicate)
    if is_trivial_shard(p1) and is_trivial_shard(p2):
        return True

    # Check Shard vs Replicate equivalence for size-1 dims
    if isinstance(p1, Replicate) and is_trivial_shard(p2):
        return True
    if isinstance(p2, Replicate) and is_trivial_shard(p1):
        return True

    # Same type comparisons
    if type(p1) is type(p2):
        if isinstance(p1, Shard):
            return p1.dim == p2.dim
        if isinstance(p1, Partial):
            return p1.reduce_op == p2.reduce_op
        return True  # Both Replicate

    return False


def has_equivalent_rule(
    combo_key: tuple,
    rules: set,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shape: tuple[int, ...],
) -> bool:
    """
    Check if any rule in the set is equivalent to the given combo.

    Args:
        combo_key: (input_placements, output_placement) tuple where placements are strings
        rules: Set of (input_placements, output_placement) tuples where placements are strings
        input_shapes: Shapes of input tensors
        output_shape: Shape of output tensor
    """
    input_placements_strs, output_placement_str = combo_key

    # Parse combo placements
    combo_input_placements = [parse_placement(s) for s in input_placements_strs]
    combo_output_placement = parse_placement(output_placement_str)

    if not all(combo_input_placements) or not combo_output_placement:
        return False

    for rule_key in rules:
        rule_input_strs, rule_output_str = rule_key
        if len(rule_input_strs) != len(input_placements_strs):
            continue

        # Parse rule placements
        rule_input_placements = [parse_placement(s) for s in rule_input_strs]
        rule_output_placement = parse_placement(rule_output_str)

        if not all(rule_input_placements) or not rule_output_placement:
            continue

        # Check all input placements match (considering equivalence)
        inputs_match = all(
            placements_equivalent(p1, p2, shape)
            for p1, p2, shape in zip(
                combo_input_placements, rule_input_placements, input_shapes
            )
        )
        if not inputs_match:
            continue

        # Check output placement matches (considering equivalence)
        if placements_equivalent(
            combo_output_placement, rule_output_placement, output_shape
        ):
            return True

    return False


def get_1d_input_placements_for_tensor(
    t: torch.Tensor, include_partial: bool = False
) -> list:
    """
    Get all possible 1-D mesh placements for an INPUT tensor.

    Args:
        t: The tensor to get placements for
        include_partial: If True, include Partial placements for inputs.
    """
    placements = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))
    if include_partial and t.ndim > 0:
        for reduce_op in PARTIAL_REDUCE_OPS:
            placements.append(Partial(reduce_op))
    return placements


def get_1d_output_placements_for_tensor(t: torch.Tensor) -> list:
    """
    Get all possible 1-D mesh placements for an OUTPUT tensor.

    For integer outputs, only P(min) and P(max) are included since
    P(sum) and P(avg) don't make semantic sense for discrete values.
    """
    placements = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))

    if t.ndim == 0:
        return placements

    is_integer = not t.dtype.is_floating_point and not t.dtype.is_complex
    for reduce_op in PARTIAL_REDUCE_OPS:
        if is_integer and reduce_op in ("sum", "avg"):
            continue
        placements.append(Partial(reduce_op))
    return placements


def extract_tensors_from_sample(sample_input) -> list:
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


def _create_partial_input(
    tensor: torch.Tensor, placement: Partial, world_size: int, tensor_idx: int = 0
) -> LocalTensor:
    """
    Create a LocalTensor with values that reduce to the original tensor.

    Uses asymmetric splits to avoid coincidental matches when combining
    different Partial types.
    """
    reduce_op = placement.reduce_op

    if reduce_op == "sum":
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)
        local_tensors = {}
        for r in range(world_size):
            if r == 0:
                local_tensors[r] = tensor.clone() * base_ratio
            else:
                local_tensors[r] = tensor.clone() * (
                    (1 - base_ratio) / (world_size - 1)
                )
        return LocalTensor(local_tensors)

    elif reduce_op == "avg":
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)
        local_tensors = {}
        for r in range(world_size):
            if r == 0:
                local_tensors[r] = tensor.clone() * base_ratio * world_size
            else:
                local_tensors[r] = (
                    tensor.clone() * ((1 - base_ratio) / (world_size - 1)) * world_size
                )
        return LocalTensor(local_tensors)

    elif reduce_op == "min":
        local_tensors = {}
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
        return LocalTensor(local_tensors)

    elif reduce_op == "max":
        local_tensors = {}
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
        return LocalTensor(local_tensors)

    else:
        local_tensors = {r: tensor.clone() for r in range(world_size)}
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
                local_tensor = LocalTensor(
                    {r: tensor.clone() for r in range(world_size)}
                )
                local_tensors.append(local_tensor)
            elif isinstance(placement, Shard):
                # Create sharded LocalTensor directly to work in LocalTensorMode
                shard_dim = placement.dim
                chunks = tensor.tensor_split(world_size, dim=shard_dim)
                local_tensor = LocalTensor(
                    {r: chunks[r].clone().contiguous() for r in range(world_size)}
                )
                local_tensors.append(local_tensor)
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

        if not torch.allclose(ground_truth, full_output, atol=1e-5, rtol=1e-5):
            max_diff = (ground_truth - full_output).abs().max().item()
            return False, f"Value mismatch: max_diff={max_diff:.6f}"

        return True, ""

    except Exception as e:
        return False, f"Exception: {type(e).__name__}: {e}"


class _CaptureAtenOp(torch.utils._python_dispatch.TorchDispatchMode):
    """Dispatch mode that captures aten ops called and their args."""

    def __init__(self, target_op_name: str = ""):
        self.target_op_name = target_op_name.lower()
        self.all_ops = []
        self.best_match = None
        self.best_match_args = None
        self.best_match_kwargs = None

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


def query_single_dim_strategy(op_overload, tensors, mesh):
    """
    Query DTensor's single-dim strategy for given input tensors.
    Returns list of [output_placement, *input_placements] rules.
    """
    from torch.distributed.tensor._dtensor_spec import TensorMeta
    from torch.distributed.tensor._ops.single_dim_strategy import _ShardingPlaceholder

    propagator = DTensor._op_dispatcher.sharding_propagator

    if op_overload not in propagator.op_single_dim_strategy_funcs:
        return None

    strategy_func = propagator.op_single_dim_strategy_funcs[op_overload]

    args_meta = tuple(
        TensorMeta(shape=t.shape, stride=t.stride(), dtype=t.dtype) for _, t in tensors
    )

    try:
        result = strategy_func(op_overload, args_meta, {})

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


def create_fully_negated_sample(sample, tensors: list):
    """Create a sample with ALL tensors negated (for P(min)/P(max) sign testing)."""
    from torch.testing._internal.common_methods_invocations import SampleInput

    tensor_ptrs = {t.data_ptr() for _, t in tensors}

    def negate_tensor(x):
        if isinstance(x, torch.Tensor) and x.data_ptr() in tensor_ptrs:
            return -x
        return x

    new_input = pytree.tree_map(negate_tensor, sample.input)
    new_args = pytree.tree_map(negate_tensor, sample.args)
    new_kwargs = pytree.tree_map(negate_tensor, sample.kwargs)

    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)


def get_opinfo_by_name(name: str):
    """Find OpInfo entries by operator name."""
    from torch.testing._internal.common_methods_invocations import op_db

    matches = [op for op in op_db if op.name == name]
    if not matches:
        raise ValueError(f"No OpInfo found for operator: {name}")
    return matches
