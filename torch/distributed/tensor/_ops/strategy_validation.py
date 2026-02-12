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
from torch.distributed._local_tensor import LocalTensor
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate
from torch.distributed.tensor.placement_types import Partial, Placement, Shard
from torch.testing._internal.common_methods_invocations import SampleInput
from torch.utils import _pytree as pytree


# Partial reduce ops to enumerate
PARTIAL_REDUCE_OPS = ["sum", "avg", "min", "max"]

# Ops to skip in validation because ground truth comparison is not meaningful.
# These produce non-deterministic or uninitialized outputs.
# TODO: can we get this list directly from opinfo tags, or something?
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

    input_placements: tuple[Placement]  # One placement per input tensor
    output_placement: Placement  # Placement for the output tensor

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


def is_trivial_shard(p, tensor_shape: tuple[int, ...]) -> bool:
    """Check if placement is a Shard on a size-1 dimension."""
    return (
        isinstance(p, Shard)
        and p.dim < len(tensor_shape)
        and tensor_shape[p.dim] == 1
    )


def normalize_placement(p, tensor_shape: tuple[int, ...]):
    """
    Normalize a placement for a given tensor shape.

    Converts Shard on a size-1 dimension to Replicate for deduplication.
    Shard(0) on a [1, 4] tensor puts all data on rank 0 and an empty [0, 4]
    on rank 1. Rank 1's empty computation is vacuous (contributes nothing
    after redistribution), so the validation outcome is determined entirely
    by rank 0, which has the full data — same as Replicate. We gain no
    signal from testing S(0) on a size-1 dim beyond what R already provides,
    so we normalize to R to avoid spurious "missing rule" noise when ground
    truth and DTensor use different forms for size-1 dims.
    """
    if is_trivial_shard(p, tensor_shape):
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
    """
    placements: list[Placement] = [Replicate()]
    for dim in range(t.ndim):
        placements.append(Shard(dim))

    if t.dtype != torch.bool:
        for reduce_op in PARTIAL_REDUCE_OPS:
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
