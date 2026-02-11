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

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.placement_types import Partial, Placement, Shard
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


def placements_equivalent(p1, p2, shape: tuple[int, ...]) -> bool:
    """
    Check if two placements are equivalent for a given tensor shape.

    Shard(dim) is equivalent to Replicate() when shape[dim] == 1, because
    sharding a size-1 dimension produces the same result as replicating.
    """

    # Check if both are trivial shards (equivalent to each other and to Replicate)
    if is_trivial_shard(p1, shape) and is_trivial_shard(p2, shape):
        return True

    # Check Shard vs Replicate equivalence for size-1 dims
    if isinstance(p1, Replicate) and is_trivial_shard(p2, shape):
        return True
    if isinstance(p2, Replicate) and is_trivial_shard(p1, shape):
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

    if any(p is None for p in combo_input_placements) or combo_output_placement is None:
        return False

    for rule_key in rules:
        rule_input_strs, rule_output_str = rule_key
        if len(rule_input_strs) != len(input_placements_strs):
            continue

        # Parse rule placements
        rule_input_placements = [parse_placement(s) for s in rule_input_strs]
        rule_output_placement = parse_placement(rule_output_str)

        if (
            any(p is None for p in rule_input_placements)
            or rule_output_placement is None
        ):
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
