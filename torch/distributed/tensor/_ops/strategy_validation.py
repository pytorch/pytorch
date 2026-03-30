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
    python -m torch.distributed.tensor._ops.strategy_validation --op relu,mul,add
    python -m torch.distributed.tensor._ops.strategy_validation --op "nn.functional.*"
    python -m torch.distributed.tensor._ops.strategy_validation --all-registered
    python -m torch.distributed.tensor._ops.strategy_validation --op div --incorrect-only
    python -m torch.distributed.tensor._ops.strategy_validation --op add --show-repro
"""

import argparse
import fnmatch
import itertools
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch._ops import OpOverload
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate
from torch.distributed.tensor._decompositions import DecompShardingStrategy
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


# A combo key is (input_placement_strs, output_placement_strs)
# For single-output ops: (("S(0)",), ("P(min)",))
# For multi-output ops:  (("S(0)",), ("P(min)", "P(min)"))
ComboKey = tuple[tuple[str, ...], tuple[str, ...]]

# Partial reduce ops to enumerate
PARTIAL_REDUCE_OPS = ["sum", "avg", "min", "max"]

SKIP_OPS: dict[str, str] = {
    "bernoulli": "non-deterministic (random sampling)",
    "empty_like": "uninitialized memory",
    "new_empty": "uninitialized memory",
    "new_empty_strided": "uninitialized memory",
    "nn.functional.dropout": "non-deterministic (random masking)",
    "normal": "non-deterministic (random sampling)",
    "rand_like": "non-deterministic (random sampling)",
    "randint_like": "non-deterministic (random sampling)",
    "randn_like": "non-deterministic (random sampling)",
    "uniform": "non-deterministic (random sampling)",
}


PlacementCombination = tuple[tuple[Placement, ...], tuple[Placement, ...]]


@dataclass
class Discrepancy:
    """Represents a discrepancy between ground truth and DTensor's rules."""

    input_placements: tuple[str, ...]
    output_placements: tuple[str, ...]
    sample_idx: int
    input_shapes: tuple[tuple[int, ...], ...]
    discrepancy_type: str  # "false_positive" or "false_negative"
    error_msg: str = ""
    scalar_args: tuple[Any, ...] = ()
    scalar_kwargs: dict[str, Any] = field(default_factory=dict)
    aten_op: OpOverload | None = None
    variant: str = ""
    sample: SampleInput | None = None


@dataclass
class ComparisonStats:
    """Statistics for comparing ground truth vs DTensor rules."""

    true_positives: int = 0
    true_negatives: int = 0
    false_positives: list[Discrepancy] = field(
        default_factory=list
    )  # DTensor has rule, ground truth says invalid
    false_negatives: list[Discrepancy] = field(
        default_factory=list
    )  # Ground truth valid, DTensor has no rule
    total_samples: int = 0
    total_combinations: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    no_dtensor_support: bool = False
    # Per aten op variant breakdown (e.g. "aten.min.dim" -> 5)
    true_positives_by_op: dict[str, int] = field(default_factory=dict)


@dataclass
class _FalsePositiveMitigations:
    """Bundle of sample variants used to detect false positive validations.

    Contains negated and non-rounded variants of a sample. These are used to
    re-test combinations that pass the primary validation, catching cases where
    sign patterns or rounding modes mask real differences.
    """

    negated_sample: SampleInput | None = None
    negated_tensors: list[tuple[str, torch.Tensor]] | None = None
    negated_ground_truth: torch.Tensor | list[torch.Tensor] | None = None
    non_rounded_sample: SampleInput | None = None
    non_rounded_ground_truth: torch.Tensor | list[torch.Tensor] | None = None
    non_rounded_negated_sample: SampleInput | None = None
    non_rounded_negated_tensors: list[tuple[str, torch.Tensor]] | None = None
    non_rounded_negated_ground_truth: torch.Tensor | list[torch.Tensor] | None = None


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


def is_fully_replicated(placements: tuple[Placement, ...]) -> bool:
    """Check if all placements are Replicate."""
    return all(isinstance(p, Replicate) for p in placements)


def is_trivial_shard(p: Placement, tensor_shape: tuple[int, ...]) -> bool:
    """Check if placement is a Shard on a size-1 dimension."""
    return (
        isinstance(p, Shard) and p.dim < len(tensor_shape) and tensor_shape[p.dim] == 1
    )


def normalize_placement(p: Placement, tensor_shape: tuple[int, ...]) -> Placement:
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
    combo_key: ComboKey,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> ComboKey:
    """
    Normalize a combo_key by converting trivial shards to Replicate.

    This deduplicates equivalent placement combinations, e.g.:
    - P(max) -> S(0) on output [1,1,1] becomes P(max) -> R
    - S(0), R -> R on input [1,4] becomes R, R -> R

    Args:
        combo_key: (input_placement_strs, output_placement_strs) tuple
        input_shapes: Shapes of input tensors
        output_shapes: Shapes of output tensors

    Returns:
        Normalized combo_key with trivial shards converted to Replicate
    """
    input_placement_strs, output_placement_strs = combo_key

    normalized_inputs = tuple(
        normalize_placement_str(p_str, shape)
        for p_str, shape in zip(input_placement_strs, input_shapes)
    )

    normalized_outputs = tuple(
        normalize_placement_str(p_str, shape)
        for p_str, shape in zip(output_placement_strs, output_shapes)
    )

    return (normalized_inputs, normalized_outputs)


def get_1d_input_placements_for_tensor(
    t: torch.Tensor, include_partial: bool = False
) -> list[Placement]:
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


def get_1d_output_placements_for_tensor(t: torch.Tensor) -> list[Placement]:
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


def extract_tensors_from_sample(
    sample_input: SampleInput,
) -> list[tuple[str, torch.Tensor]]:
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


def _checkerboard_mask(
    tensor: torch.Tensor, tensor_idx: int = 0, mask_shift: int = 0
) -> torch.Tensor:
    """Checkerboard mask that alternates in every dimension.

    Unlike flat-index % 2, which can be uniform along even-stride dimensions
    (causing all elements in a reduction group to get the same offset), the
    checkerboard uses sum-of-coordinates mod 2 so adjacent elements differ
    along every axis.

    Returns a flat bool tensor of shape (numel,).
    """
    if tensor.ndim == 0:
        return torch.tensor([(tensor_idx + mask_shift) % 2 == 0], device=tensor.device)
    coords = [torch.arange(s, device=tensor.device) for s in tensor.shape]
    grids = torch.meshgrid(*coords, indexing="ij")
    coord_sum = grids[0].clone()
    for g in grids[1:]:
        coord_sum += g
    return ((coord_sum + tensor_idx + mask_shift) % 2 == 0).flatten()


def _create_partial_input(
    tensor: torch.Tensor,
    placement: Partial,
    world_size: int,
    tensor_idx: int = 0,
    mask_shift: int = 0,
) -> LocalTensor:
    """
    Create a LocalTensor with values that reduce to the original tensor.

    Uses asymmetric splits to avoid coincidental matches when combining
    different Partial types.

    For each placement combination, it creates local tensors that would
    reduce to the original (e.g., for P(sum), splits values across ranks so
    they sum back), runs the op on those local tensors, wraps the output as
    a DTensor, redistributes to Replicate, and compares against ground
    truth.

    The main challenge is avoiding false positives where a rule appears
    valid on a specific input but is actually incorrect. Several techniques
    are used:

    Asymmetric splits for P(sum)/P(avg): instead of splitting evenly
    (tensor/2 per rank), uses a 60/40 ratio (varied by tensor index) so
    that ops which are not truly linear don't accidentally produce
    matching outputs.

    Sign-varying offsets for P(sum)/P(avg): adds an offset that
    alternates sign across elements, so local tensors have mixed positive
    and negative values. Without this, proportional splits preserve the
    sign pattern of the original tensor, causing non-linear ops like abs
    to falsely validate P(sum)->P(sum).

    Distinct magnitudes for P(min) vs P(max): P(min) offsets non-holding
    ranks by +(range*2+1) while P(max) offsets by -(range*2+1), where
    range is the tensor's value range. Using adaptive offsets that exceed
    the value range ensures that index-returning ops (argmin/argmax)
    produce different results on different ranks, correctly rejecting
    P(min)/P(max) inputs for those ops. Using different signs for min vs
    max prevents accidental cancellation when both appear in the same
    combination.

    Alternating rank ownership for P(min)/P(max): a multi-dimensional
    checkerboard mask (sum of coordinates mod 2) controls which rank holds
    the true value vs the offset value. Unlike a flat-index mask which can
    have uniform parity along an even-stride dimension, the checkerboard
    guarantees alternation along EVERY dimension. The mask_shift parameter
    allows re-validation with the complementary mask to catch ops where
    the result coincidentally matches.

    """
    reduce_op = placement.reduce_op
    local_tensors: dict[int, torch.Tensor] = {}

    if reduce_op in ("sum", "avg"):
        base_ratio = 0.6 + 0.1 * (tensor_idx % 3)

        # See docstring above: "Sign-varying offsets"
        flat = tensor.flatten()
        offset_mag = flat.abs() + 1.0
        signs = torch.ones_like(flat)
        # Use checkerboard mask so offset sign alternates in every dimension,
        # not just along flat index (which can be uniform along even-stride dims).
        signs[_checkerboard_mask(tensor, tensor_idx, mask_shift)] = -1.0
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
        # See docstring above: "Distinct Magnitudes" and "Alternating Rank Ownership"
        flat = tensor.flatten()
        # Offset must exceed the tensor's value range so that the mask pattern
        # determines argmin/argmax, not the original values.
        value_range = (flat.max() - flat.min()).item()
        min_offset = value_range * 2 + 1
        mask = _checkerboard_mask(tensor, tensor_idx, mask_shift)
        for r in range(world_size):
            if r == 0:
                r_offset = torch.where(
                    mask, torch.zeros_like(flat), torch.full_like(flat, min_offset)
                )
            else:
                r_offset = torch.where(
                    mask, torch.full_like(flat, min_offset), torch.zeros_like(flat)
                )
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)

    elif reduce_op == "max":
        # See docstring above: "Distinct Magnitudes" and "Alternating Rank Ownership"
        flat = tensor.flatten()
        value_range = (flat.max() - flat.min()).item()
        max_offset = -(value_range * 2 + 1)
        mask = _checkerboard_mask(tensor, tensor_idx, mask_shift)
        for r in range(world_size):
            if r == 0:
                r_offset = torch.where(
                    mask, torch.zeros_like(flat), torch.full_like(flat, max_offset)
                )
            else:
                r_offset = torch.where(
                    mask, torch.full_like(flat, max_offset), torch.zeros_like(flat)
                )
            local_tensors[r] = (flat + r_offset).reshape(tensor.shape)

    else:
        for r in range(world_size):
            local_tensors[r] = tensor.clone()

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    return LocalTensor(local_tensors)


def _shard_tensors(
    tensors: list[tuple[str, torch.Tensor]],
    input_placements: tuple[Placement, ...],
    world_size: int,
    mesh: DeviceMesh,
    mask_shift: int = 0,
) -> list[LocalTensor | torch.Tensor]:
    """Create sharded LocalTensors from tensors according to placements."""
    local_tensors: list[LocalTensor | torch.Tensor] = []
    for tensor_idx, ((name, tensor), placement) in enumerate(
        zip(tensors, input_placements)
    ):
        if isinstance(placement, Partial):
            local_tensor = _create_partial_input(
                tensor, placement, world_size, tensor_idx, mask_shift
            )
        elif isinstance(placement, Replicate):
            _tmp = {r: tensor.clone() for r in range(world_size)}
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            local_tensor = LocalTensor(_tmp)
        elif isinstance(placement, Shard):
            shard_dim = placement.dim
            chunks = tensor.tensor_split(world_size, dim=shard_dim)
            _tmp = {
                r: chunks[r].clone(memory_format=torch.contiguous_format)
                for r in range(world_size)
            }
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            local_tensor = LocalTensor(_tmp)
        else:
            dt = distribute_tensor(tensor.clone(), mesh, (placement,))
            local_tensor = dt.to_local()
        local_tensors.append(local_tensor)
    return local_tensors


def _compare_outputs(
    local_output: Any,
    ground_truth: torch.Tensor | list[torch.Tensor],
    output_placements: tuple[Placement, ...],
    mesh: DeviceMesh,
    world_size: int,
) -> tuple[bool, str]:
    """Compare op output (wrapped as DTensor) against ground truth."""
    if isinstance(local_output, (list, tuple)):
        local_outputs = list(local_output)
    else:
        local_outputs = [local_output]

    if isinstance(ground_truth, list):
        ground_truths = ground_truth
    else:
        ground_truths = [ground_truth]

    if len(local_outputs) != len(ground_truths):
        return (
            False,
            f"Output count mismatch: got {len(local_outputs)}, "
            f"expected {len(ground_truths)}",
        )

    if len(local_outputs) != len(output_placements):
        return (
            False,
            f"Output count mismatch with placements: "
            f"got {len(local_outputs)}, expected {len(output_placements)}",
        )

    for i, (local_out, gt, out_plc) in enumerate(
        zip(local_outputs, ground_truths, output_placements)
    ):
        if not isinstance(local_out, torch.Tensor):
            return False, f"Local output[{i}] is not a tensor: {type(local_out)}"

        if not isinstance(local_out, LocalTensor):
            return False, f"LocalTensor inputs produced non-LocalTensor output[{i}]"

        output_dt = DTensor.from_local(
            local_out,
            mesh,
            (out_plc,),
            shape=gt.shape,
            stride=gt.stride(),
        )

        if isinstance(out_plc, Replicate):
            local_values = [local_out._local_tensors[r] for r in range(world_size)]
            all_same = all(
                torch.allclose(local_values[0], lv, atol=1e-5, rtol=1e-5)
                for lv in local_values[1:]
            )
            if not all_same:
                return (
                    False,
                    f"Replicate output[{i}] but local values differ across ranks",
                )

        full_output = output_dt.redistribute(mesh, (Replicate(),)).to_local()

        if isinstance(full_output, LocalTensor):
            full_output = full_output._local_tensors[0]

        if gt.shape != full_output.shape:
            return (
                False,
                f"Shape mismatch[{i}]: expected {gt.shape}, got {full_output.shape}",
            )

        if not torch.allclose(gt, full_output, atol=1e-5, rtol=1e-5, equal_nan=True):
            max_diff = (gt - full_output).abs().max().item()
            return False, f"Value mismatch[{i}]: max_diff={max_diff:.6f}"

    return True, ""


def validate_combination(
    op: Callable[..., Any],
    sample_input: SampleInput,
    tensors: list[tuple[str, torch.Tensor]],
    combination: PlacementCombination,
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int = 2,
    mesh: DeviceMesh | None = None,
    mask_shift: int = 0,
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
        ground_truth: Expected output tensor(s). For multi-output ops, a list
            of tensors where each element is validated independently against
            the same output placement.
        world_size: Number of simulated ranks
        mesh: Optional pre-created device mesh (for performance)

    Returns:
        (is_valid, error_message)
    """
    try:
        if mesh is None:
            device = tensors[0][1].device.type if tensors else "cpu"
            mesh = init_device_mesh(device, (world_size,))

        local_tensors = _shard_tensors(
            tensors, combination[0], world_size, mesh, mask_shift
        )

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

        return _compare_outputs(
            local_output, ground_truth, combination[1], mesh, world_size
        )

    except Exception as e:
        # TODO: This is too broad. Consider: (1) explicit checks for shard dim
        # validity and shape compatibility before calling tensor_split/from_local,
        # (2) scoped try/except around op() and redistribute() that raise specific
        # exceptions (e.g., UnsupportedRedistribute, OpError), and (3) only
        # catching those here, letting real bugs propagate.
        return False, f"Exception: {type(e).__name__}: {e}"


def extract_tensors_from_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> list[tuple[str, torch.Tensor]]:
    """Extract tensor arguments from captured aten args/kwargs.

    Unlike extract_tensors_from_sample which walks SampleInput pytrees,
    this walks the flat aten-level args and kwargs directly.
    """
    tensors: list[tuple[str, torch.Tensor]] = []
    idx = 0

    def _collect(x: Any) -> Any:
        nonlocal idx
        if isinstance(x, torch.Tensor):
            tensors.append((f"tensor_{idx}", x))
            idx += 1
        return x

    pytree.tree_map(_collect, args)
    pytree.tree_map(_collect, kwargs)
    return tensors


def validate_aten_combination(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    ground_truth: torch.Tensor | list[torch.Tensor],
    combination: PlacementCombination,
    world_size: int,
    mesh: DeviceMesh,
    mask_shift: int = 0,
) -> tuple[bool, str]:
    """Validate a placement combination using aten-level captured args.

    Works directly with aten op args/kwargs instead of SampleInput pytrees.
    Replaces tensors in the flat args/kwargs with sharded LocalTensors,
    calls the aten op, and compares output.
    """
    try:
        tensors = extract_tensors_from_args(captured_args, captured_kwargs)
        if not tensors:
            return False, "No tensor args in captured aten call"

        local_tensors = _shard_tensors(
            tensors, combination[0], world_size, mesh, mask_shift
        )

        local_idx = 0

        def _replace_with_local(a: Any) -> Any:
            nonlocal local_idx
            if isinstance(a, torch.Tensor):
                local = local_tensors[local_idx]
                local_idx += 1
                return local
            return a

        local_args = pytree.tree_map(_replace_with_local, captured_args)
        local_kwargs = pytree.tree_map(_replace_with_local, captured_kwargs)

        local_output = aten_op(*local_args, **local_kwargs)

        return _compare_outputs(
            local_output, ground_truth, combination[1], mesh, world_size
        )

    except Exception as e:
        return False, f"Exception: {type(e).__name__}: {e}"


def has_pmin_pmax(
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
) -> bool:
    """Check if any placement is Partial(min) or Partial(max)."""
    for p in (*input_placements, *output_placements):
        if isinstance(p, Partial) and p.reduce_op in ("min", "max"):
            return True
    return False


def has_any_partial(
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
) -> bool:
    """Check if any placement is Partial (any reduce op)."""
    for p in (*input_placements, *output_placements):
        if isinstance(p, Partial):
            return True
    return False


def negate_all_tensors(
    tensors: list[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    """Return a new list with all tensors negated."""
    return [(name, -t) for name, t in tensors]


def create_fully_negated_sample(sample: SampleInput) -> SampleInput:
    """Create a sample with ALL tensors negated (for P(min)/P(max) sign testing)."""

    def negate_tensor(x):
        if isinstance(x, torch.Tensor):
            return -x
        return x

    new_input = pytree.tree_map(negate_tensor, sample.input)
    new_args = pytree.tree_map(negate_tensor, sample.args)
    new_kwargs = pytree.tree_map(negate_tensor, sample.kwargs)

    return SampleInput(new_input, args=new_args, kwargs=new_kwargs)


def _run_op_on_sample(op: Callable[..., Any], sample: SampleInput) -> Any:
    """Run an operator on a SampleInput, handling both tensor and tuple inputs."""
    if isinstance(sample.input, torch.Tensor):
        return op(sample.input, *sample.args, **sample.kwargs)
    return op(*sample.input, *sample.args, **sample.kwargs)


def _extract_rules_from_op_strategy(
    op_strategy: Any,
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> set[ComboKey]:
    """Extract normalized sharding rules from an OpStrategy.

    Called during rule comparison to collect DTensor's claimed-valid placement
    combinations. These are compared against ground truth (brute-force
    validation) to find false positives (DTensor claims valid but wrong) and
    false negatives (valid but DTensor has no rule).
    """
    rules: set[ComboKey] = set()
    if not isinstance(op_strategy, OpStrategy):
        return rules
    for spec in op_strategy.strategies:
        if spec.input_specs is None:
            continue
        if isinstance(spec.output_specs, tuple):
            output_plcs: list[Placement] = []
            has_none = False
            for out_spec in spec.output_specs:
                if out_spec is None:
                    # None means the output placement is undefined for this
                    # strategy (e.g. indices under P(max) reduction). Skip it.
                    has_none = True
                    break
                output_plcs.append(out_spec.placements[0])
            if has_none:
                continue
        else:
            # Single DTensorSpec — the propagator duplicates it for all
            # outputs of multi-output ops, so we do the same here.
            output_plcs = [spec.output_spec.placements[0]] * len(output_shapes)
        input_plcs = tuple(s.placements[0] for s in spec.input_specs)
        rule_key: ComboKey = (
            tuple(str(p) for p in input_plcs),
            tuple(str(p) for p in output_plcs),
        )
        normalized_rule = normalize_combo_key(rule_key, input_shapes, output_shapes)
        if not is_fully_replicated(
            tuple(parse_placement(p) or Replicate() for p in normalized_rule[0])
        ):
            rules.add(normalized_rule)
    return rules


class _CaptureAtenOp(torch.utils._python_dispatch.TorchDispatchMode):
    """Dispatch mode that captures aten ops called, their args, and return values."""

    def __init__(self, target_op_name: str = ""):
        self.target_op_name = target_op_name.lower()
        self.all_ops: list[tuple[OpOverload, tuple[Any, ...], dict[str, Any], Any]] = []
        self.best_match: OpOverload | None = None
        self.best_match_args: tuple[Any, ...] | None = None
        self.best_match_kwargs: dict[str, Any] | None = None
        self.best_match_result: Any = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        result = func(*args, **kwargs)
        if func.namespace == "aten":
            self.all_ops.append((func, args, kwargs, result))
            op_name = func.name().split("::")[1].split(".")[0].lower()
            if self.target_op_name and self.target_op_name in op_name:
                if self.best_match is None:
                    self.best_match = func
                    self.best_match_args = args
                    self.best_match_kwargs = kwargs
                    self.best_match_result = result
        return result


def get_aten_op_for_sample(
    op: Callable[..., Any], sample: SampleInput, op_name: str = ""
) -> _CaptureAtenOp:
    """
    Capture aten ops dispatched for a given sample.

    Returns the _CaptureAtenOp object containing all captured ops with their
    args, kwargs, and return values. Use best_match for the primary op or
    all_ops for exhaustive iteration.
    """
    with _CaptureAtenOp(op_name) as capture:
        try:
            if isinstance(sample.input, torch.Tensor):
                op(sample.input, *sample.args, **sample.kwargs)
            else:
                op(*sample.input, *sample.args, **sample.kwargs)
        except Exception:
            pass

    # Populate best_match from first op if target match wasn't found
    if capture.best_match is None and capture.all_ops:
        first_op, first_args, first_kwargs, first_result = capture.all_ops[0]
        capture.best_match = first_op
        capture.best_match_args = first_args
        capture.best_match_kwargs = first_kwargs
        capture.best_match_result = first_result

    return capture


def query_single_dim_strategy(
    op_overload: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> list[list[Placement]] | None:
    """
    Query DTensor's single-dim strategy for given input tensors.
    Returns list of [output_placement, *input_placements] rules.
    """
    propagator = DTensor._op_dispatcher.sharding_propagator

    if op_overload not in propagator.op_single_dim_strategy_funcs:
        return None

    strategy_func = propagator.op_single_dim_strategy_funcs[op_overload]

    args_meta = tuple(
        TensorMeta(shape=a.shape, stride=a.stride(), dtype=a.dtype)
        if isinstance(a, torch.Tensor)
        else a
        for a in captured_args
    )
    kwargs_meta = {
        k: TensorMeta(shape=v.shape, stride=v.stride(), dtype=v.dtype)
        if isinstance(v, torch.Tensor)
        else v
        for k, v in captured_kwargs.items()
    }

    try:
        result = strategy_func(op_overload, args_meta, kwargs_meta)

        expanded_result: list[list[Placement]] = []
        for combo in result:
            expanded_combo: list[Placement] = []
            for p in combo:
                if isinstance(p, _ShardingPlaceholder):
                    expanded_combo.append(Shard(p.dim))
                else:
                    expanded_combo.append(p)
            expanded_result.append(expanded_combo)

        return expanded_result
    except Exception:
        return None


def get_opinfo_by_name(name: str) -> list[opinfo_core.OpInfo]:
    """Find OpInfo entries by exact operator name."""
    matches = [op for op in op_db if op.name == name]
    if matches:
        return matches

    # Suggest alternatives
    candidates = _find_opinfo_candidates(name)
    if candidates:
        suggestions = ", ".join(f'"{c}"' for c in candidates)
        raise ValueError(f'No OpInfo found for "{name}", did you mean: {suggestions}?')
    raise ValueError(
        f'No OpInfo found for "{name}". OpInfo is required as it provides '
        f"sample inputs for the operator."
    )


def _find_opinfo_candidates(name: str) -> list[str]:
    """Find OpInfo names that plausibly match a short/incorrect name."""
    candidates: list[str] = []
    seen: set[str] = set()
    # Match on aten_name (e.g., "relu" -> OpInfo with aten_name="relu")
    for op in op_db:
        if op.aten_name == name and op.name not in seen:
            candidates.append(op.name)
            seen.add(op.name)
    # Suffix match: "relu" matches "nn.functional.relu"
    suffix = "." + name
    for op in op_db:
        if op.name.endswith(suffix) and op.name not in seen:
            candidates.append(op.name)
            seen.add(op.name)
    return candidates


def resolve_op_names(patterns: list[str]) -> list[str]:
    """Resolve user-provided op patterns to exact OpInfo names.

    Supports exact names, comma separation, and glob patterns (e.g.,
    "nn.functional.*"). Short names like "relu" are resolved unambiguously
    or an error is raised with suggestions.
    """
    all_opinfo_names = sorted({op.name for op in op_db})
    resolved: list[str] = []
    seen: set[str] = set()

    for pattern in patterns:
        # Glob pattern
        if "*" in pattern or "?" in pattern:
            matches = fnmatch.filter(all_opinfo_names, pattern)
            if not matches:
                raise ValueError(f'No OpInfo names match pattern "{pattern}".')
            for m in matches:
                if m not in seen:
                    resolved.append(m)
                    seen.add(m)
            continue

        # Exact match
        if pattern in {op.name for op in op_db}:
            if pattern not in seen:
                resolved.append(pattern)
                seen.add(pattern)
            continue

        # Try to resolve shorthand
        candidates = _find_opinfo_candidates(pattern)
        if len(candidates) == 1:
            name = candidates[0]
            if name not in seen:
                resolved.append(name)
                seen.add(name)
        elif len(candidates) > 1:
            suggestions = ", ".join(f'"{c}"' for c in candidates)
            raise ValueError(
                f'"{pattern}" is ambiguous, matching: {suggestions}. '
                f"Use the fully qualified name."
            )
        else:
            raise ValueError(f'No OpInfo found for "{pattern}".')

    return resolved


def _is_tensor_output(result: Any) -> bool:
    """Check if a result is a tensor or list/tuple of tensors."""
    if isinstance(result, torch.Tensor):
        return True
    if isinstance(result, (list, tuple)):
        has_tensor = any(isinstance(t, torch.Tensor) for t in result)
        all_tensor = all(isinstance(t, torch.Tensor) for t in result)
        if has_tensor and not all_tensor:
            raise NotImplementedError(
                f"Mixed tensor/non-tensor tuple outputs are not supported by the "
                f"validator. Got types: {[type(t).__name__ for t in result]}"
            )
        return all_tensor
    return False


def _to_ground_truth(result: Any) -> torch.Tensor | list[torch.Tensor]:
    """Convert an op result to the ground truth format (tensor or list of tensors)."""
    if isinstance(result, torch.Tensor):
        return result
    return list(result)


def _prepare_false_positive_mitigations(
    op: Callable[..., Any],
    sample: SampleInput,
    tensors: list[tuple[str, torch.Tensor]],
) -> _FalsePositiveMitigations:
    """Create negated and non-rounded sample variants for false positive detection."""
    m = _FalsePositiveMitigations()

    try:
        m.negated_sample = create_fully_negated_sample(sample)
        m.negated_tensors = negate_all_tensors(tensors)
        result = _run_op_on_sample(op, m.negated_sample)
        if _is_tensor_output(result):
            m.negated_ground_truth = _to_ground_truth(result)
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
        if not _is_tensor_output(result):
            m.non_rounded_sample = None
        else:
            m.non_rounded_ground_truth = _to_ground_truth(result)
            m.non_rounded_negated_sample = create_fully_negated_sample(
                m.non_rounded_sample
            )
            m.non_rounded_negated_tensors = negate_all_tensors(tensors)
            nr_neg_result = _run_op_on_sample(op, m.non_rounded_negated_sample)
            if _is_tensor_output(nr_neg_result):
                m.non_rounded_negated_ground_truth = _to_ground_truth(nr_neg_result)
            else:
                m.non_rounded_negated_sample = None
    except Exception:
        m.non_rounded_sample = None
        m.non_rounded_ground_truth = None
        m.non_rounded_negated_sample = None

    return m


def _query_dtensor_rules(
    aten_op: OpOverload | None,
    tensors: list[tuple[str, torch.Tensor]],
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    world_size: int,
    verbose: bool,
) -> set[ComboKey]:
    """Query DTensor's claimed sharding rules via single-dim, op_strategy, or decomp paths.

    TODO: This reimplements strategy resolution logic from ShardingPropagator.
    Refactor ShardingPropagator to expose a public API for querying sharding
    rules given an op and tensor metadata, so this function can be replaced
    with a single call.
    """
    if not aten_op:
        return set()

    num_tensors = len(tensors)
    non_tensor_kwargs = {
        k: v for k, v in captured_kwargs.items() if not isinstance(v, torch.Tensor)
    }
    n_outputs = len(output_shapes)
    propagator = DTensor._op_dispatcher.sharding_propagator
    rules: set[ComboKey] = set()

    if aten_op in propagator.op_single_dim_strategy_funcs:
        strategy_result = query_single_dim_strategy(
            aten_op, captured_args, captured_kwargs
        )
        if strategy_result:
            for combo in strategy_result:
                if len(combo) >= n_outputs + num_tensors:
                    output_plcs = combo[:n_outputs]
                    input_plcs = tuple(combo[n_outputs : n_outputs + num_tensors])
                    rule_key: ComboKey = (
                        tuple(str(p) for p in input_plcs),
                        tuple(str(p) for p in output_plcs),
                    )
                    normalized_rule = normalize_combo_key(
                        rule_key, input_shapes, output_shapes
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
            # Build OpStrategy objects for each tensor, keyed by identity
            tensor_to_strategy: dict[int, OpStrategy] = {}
            for _, t in tensors:
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
                tensor_to_strategy[id(t)] = OpStrategy(specs)
            # Interleave strategies and non-tensor args at original positions
            args_schema = [
                tensor_to_strategy[id(a)] if isinstance(a, torch.Tensor) else a
                for a in captured_args
            ]
            op_schema = OpSchema(aten_op, tuple(args_schema), non_tensor_kwargs)
            strategy_func = propagator.op_strategy_funcs[aten_op]
            output_strategy = strategy_func(op_schema)
            rules |= _extract_rules_from_op_strategy(
                output_strategy, input_shapes, output_shapes
            )
        except Exception as e:
            if verbose:
                print(f"        Error querying op_strategy: {e}")

    else:
        # Decomp-based strategy: only discovers rules reachable from a single
        # seed (Shard(0) on the first input). Rules requiring other input
        # placements (e.g., Shard(1), Partial, or sharding on non-first inputs)
        # will not be found, so this under-reports DTensor's capabilities.
        if DecompShardingStrategy.has_decomp(aten_op):
            try:
                mesh = init_device_mesh("cpu", (world_size,))
                # Interleave DTensorSpec and non-tensor args at original positions
                tensor_idx = 0
                args_schema: list[Any] = []
                for a in captured_args:
                    if isinstance(a, torch.Tensor):
                        # First tensor gets Shard(0) to seed candidate
                        # placement generation in _get_candidate_placements
                        plc = Shard(0) if tensor_idx == 0 else Replicate()
                        spec = DTensorSpec(
                            mesh=mesh,
                            placements=(plc,),
                            tensor_meta=TensorMeta(
                                shape=a.shape, stride=a.stride(), dtype=a.dtype
                            ),
                        )
                        args_schema.append(spec)
                        tensor_idx += 1
                    else:
                        args_schema.append(a)
                op_schema = OpSchema(aten_op, tuple(args_schema), non_tensor_kwargs)
                propagator.decomp_strategy.ensure_schema_info(aten_op)
                output_strategy = propagator.decomp_strategy.propagate_strategy(
                    op_schema,
                )
                if output_strategy is not None:
                    rules |= _extract_rules_from_op_strategy(
                        output_strategy, input_shapes, output_shapes
                    )
            except Exception as e:
                if verbose:
                    print(f"        Error querying decomp strategy: {e}")

    return rules


def _validate_with_mitigations(
    op: Callable[..., Any],
    sample: SampleInput,
    tensors: list[tuple[str, torch.Tensor]],
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int,
    mesh: DeviceMesh,
    mitigations: _FalsePositiveMitigations,
) -> bool:
    """Validate a combination, including false positive mitigation re-checks."""
    combo: PlacementCombination = (input_placements, output_placements)
    is_valid, _ = validate_combination(
        op, sample, tensors, combo, ground_truth, world_size, mesh
    )

    # Flipped-mask mitigation: the checkerboard mask that controls offset
    # signs (for P(sum)/P(avg)) or rank ownership (for P(min)/P(max)) is
    # deterministic per tensor_idx. Re-validate with the complementary mask
    # to catch index-returning ops (argmin/argmax) where the result
    # coincidentally matches because the dominant value happens to land on
    # a position where both mask orientations preserve argmin/argmax.
    if is_valid and has_any_partial(input_placements, output_placements):
        is_valid, _ = validate_combination(
            op,
            sample,
            tensors,
            combo,
            ground_truth,
            world_size,
            mesh,
            mask_shift=1,
        )

    if (
        is_valid
        and mitigations.negated_sample
        and has_pmin_pmax(input_placements, output_placements)
    ):
        if mitigations.negated_tensors is None:
            raise AssertionError("negated_tensors is None")
        if mitigations.negated_ground_truth is None:
            raise AssertionError("negated_ground_truth is None")
        is_valid, _ = validate_combination(
            op,
            mitigations.negated_sample,
            mitigations.negated_tensors,
            combo,
            mitigations.negated_ground_truth,
            world_size,
            mesh,
        )

    if (
        is_valid
        and mitigations.non_rounded_sample
        and has_any_partial(input_placements, output_placements)
    ):
        if mitigations.non_rounded_ground_truth is None:
            raise AssertionError("non_rounded_ground_truth is None")
        is_valid, _ = validate_combination(
            op,
            mitigations.non_rounded_sample,
            tensors,
            combo,
            mitigations.non_rounded_ground_truth,
            world_size,
            mesh,
        )

    if (
        is_valid
        and mitigations.non_rounded_negated_sample
        and has_pmin_pmax(input_placements, output_placements)
    ):
        if mitigations.non_rounded_negated_tensors is None:
            raise AssertionError("non_rounded_negated_tensors is None")
        if mitigations.non_rounded_negated_ground_truth is None:
            raise AssertionError("non_rounded_negated_ground_truth is None")
        is_valid, _ = validate_combination(
            op,
            mitigations.non_rounded_negated_sample,
            mitigations.non_rounded_negated_tensors,
            combo,
            mitigations.non_rounded_negated_ground_truth,
            world_size,
            mesh,
        )

    return is_valid


@dataclass
class _AtenFalsePositiveMitigations:
    """Bundle of negated variants for aten-level false positive detection.

    Unlike _FalsePositiveMitigations, this works with captured aten args/kwargs
    directly, without SampleInput. The rounding_mode mitigation is skipped
    since it's an OpInfo-level concept that doesn't appear in captured aten kwargs.
    """

    negated_args: tuple[Any, ...] | None = None
    negated_kwargs: dict[str, Any] | None = None
    negated_ground_truth: torch.Tensor | list[torch.Tensor] | None = None


def _negate_tensors_in_tree(tree: Any) -> Any:
    """Negate all tensors in a pytree structure."""

    def _negate(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return -x
        return x

    return pytree.tree_map(_negate, tree)


def _prepare_aten_mitigations(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> _AtenFalsePositiveMitigations:
    """Create negated variants for aten-level false positive detection."""
    m = _AtenFalsePositiveMitigations()
    try:
        m.negated_args = _negate_tensors_in_tree(captured_args)
        m.negated_kwargs = _negate_tensors_in_tree(captured_kwargs)
        result = aten_op(*m.negated_args, **m.negated_kwargs)
        if _is_tensor_output(result):
            m.negated_ground_truth = _to_ground_truth(result)
        else:
            m.negated_args = None
            m.negated_kwargs = None
    except Exception:
        m.negated_args = None
        m.negated_kwargs = None
    return m


def _validate_aten_with_mitigations(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placements: tuple[Placement, ...],
    output_placements: tuple[Placement, ...],
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int,
    mesh: DeviceMesh,
    mitigations: _AtenFalsePositiveMitigations,
) -> bool:
    """Validate an aten-level combination with false positive mitigations."""
    combo: PlacementCombination = (input_placements, output_placements)
    is_valid, _ = validate_aten_combination(
        aten_op,
        captured_args,
        captured_kwargs,
        ground_truth,
        combo,
        world_size,
        mesh,
    )

    if is_valid and has_any_partial(input_placements, output_placements):
        is_valid, _ = validate_aten_combination(
            aten_op,
            captured_args,
            captured_kwargs,
            ground_truth,
            combo,
            world_size,
            mesh,
            mask_shift=1,
        )

    if (
        is_valid
        and mitigations.negated_args is not None
        and has_pmin_pmax(input_placements, output_placements)
    ):
        if mitigations.negated_kwargs is None:
            raise AssertionError("negated_kwargs must not be None")
        if mitigations.negated_ground_truth is None:
            raise AssertionError("negated_ground_truth must not be None")
        is_valid, _ = validate_aten_combination(
            aten_op,
            mitigations.negated_args,
            mitigations.negated_kwargs,
            mitigations.negated_ground_truth,
            combo,
            world_size,
            mesh,
        )

    return is_valid


def _assert_keys_normalized(
    keys: set[ComboKey],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
) -> None:
    """Assert all combo keys have trivial shards already normalized to Replicate."""
    for key in keys:
        if key != normalize_combo_key(key, input_shapes, output_shapes):
            raise AssertionError(
                f"Key {key} contains un-normalized trivial shards; "
                f"call normalize_combo_key before _compare_rules"
            )


def _compare_rules(
    ground_truth_valid: set[ComboKey],
    dtensor_rules: set[ComboKey],
    input_shapes: tuple[tuple[int, ...], ...],
    output_shapes: tuple[tuple[int, ...], ...],
    sample_idx: int,
    scalar_args: tuple[Any, ...],
    scalar_kwargs: dict[str, Any],
    aten_op: OpOverload | None,
    variant: str,
    stats: ComparisonStats,
    sample: SampleInput | None = None,
) -> None:
    """Compare ground truth valid rules against DTensor claimed rules, updating stats."""
    if not dtensor_rules:
        return

    _assert_keys_normalized(ground_truth_valid, input_shapes, output_shapes)
    _assert_keys_normalized(dtensor_rules, input_shapes, output_shapes)

    op_str = str(aten_op)
    for combo_key in ground_truth_valid:
        if combo_key in dtensor_rules:
            stats.true_positives += 1
            stats.true_positives_by_op[op_str] = (
                stats.true_positives_by_op.get(op_str, 0) + 1
            )
        else:
            stats.false_negatives.append(
                Discrepancy(
                    input_placements=combo_key[0],
                    output_placements=combo_key[1],
                    sample_idx=sample_idx,
                    input_shapes=input_shapes,
                    discrepancy_type="false_negative",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    aten_op=aten_op,
                    variant=variant,
                    sample=sample,
                )
            )

    for combo_key in dtensor_rules:
        if combo_key not in ground_truth_valid:
            stats.false_positives.append(
                Discrepancy(
                    input_placements=combo_key[0],
                    output_placements=combo_key[1],
                    sample_idx=sample_idx,
                    input_shapes=input_shapes,
                    discrepancy_type="false_positive",
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    aten_op=aten_op,
                    variant=variant,
                    sample=sample,
                )
            )


def _format_op_name(opinfo_name: str) -> str:
    """Format an OpInfo name for display, adding aten. prefix for simple names."""
    if "." not in opinfo_name:
        return f"aten.{opinfo_name}"
    return opinfo_name


def _format_sample_repro(sample: SampleInput, aten_op: OpOverload | None = None) -> str:
    """Format a SampleInput's values for repro output, using schema arg names."""
    # Get argument names from the aten op schema if available.
    # Schema args are ordered: the first is the input, then positional args,
    # then keyword-only args (which appear in sample.kwargs).
    arg_names: list[str] = []
    if aten_op is not None:
        try:
            arg_names = [a.name for a in aten_op._schema.arguments]
        except Exception:
            pass

    parts = []
    # input is the first positional arg
    name = arg_names[0] if arg_names else "input"
    parts.append(f"{name}={sample.input!r}")
    # remaining positional args
    for i, arg in enumerate(sample.args):
        name = arg_names[1 + i] if 1 + i < len(arg_names) else f"args[{i}]"
        parts.append(f"{name}={arg!r}")
    # kwargs — use their actual key names (already named)
    for k, v in sample.kwargs.items():
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _print_discrepancy_section(
    title: str, discrepancies: list[Discrepancy], show_repro: int = 0
) -> None:
    """Print grouped discrepancies for a section (incorrect or missing)."""
    if not discrepancies:
        return
    print(f"\n{title}")
    by_op: dict[str, dict[ComboKey, list[Discrepancy]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for d in discrepancies:
        op_str = str(d.aten_op)
        key = (d.input_placements, d.output_placements)
        by_op[op_str][key].append(d)

    for op_str in sorted(by_op.keys()):
        print(f"\n  [{op_str}]")
        for (inp, out), discs in sorted(by_op[op_str].items(), key=str):
            inp_str = ", ".join(inp)
            out_str = out[0] if len(out) == 1 else "(" + ", ".join(out) + ")"
            print(f"    {inp_str} -> {out_str}")
            if show_repro:
                limit = len(discs) if show_repro < 0 else show_repro
                for d in discs[:limit]:
                    if d.sample is not None:
                        print(
                            f"      Repro: {_format_sample_repro(d.sample, d.aten_op)}"
                        )
                if len(discs) > limit:
                    print(f"      ... and {len(discs) - limit} more")


def _print_comparison_summary(
    stats: ComparisonStats,
    show_repro: int = 0,
) -> None:
    """Print discrepancy details for an operator."""
    # Per aten op variant breakdown
    fp_by_op: dict[str, set[ComboKey]] = defaultdict(set)
    for d in stats.false_positives:
        op_str = str(d.aten_op)
        fp_by_op[op_str].add((d.input_placements, d.output_placements))
    fn_by_op: dict[str, set[ComboKey]] = defaultdict(set)
    for d in stats.false_negatives:
        op_str = str(d.aten_op)
        fn_by_op[op_str].add((d.input_placements, d.output_placements))

    all_ops = sorted(set(stats.true_positives_by_op) | set(fp_by_op) | set(fn_by_op))
    if len(all_ops) > 1:
        for op_str in all_ops:
            tp = stats.true_positives_by_op.get(op_str, 0)
            fp = len(fp_by_op.get(op_str, set()))
            fn = len(fn_by_op.get(op_str, set()))
            print(f"  {op_str}: {tp} correct, {fp} incorrect, {fn} missing")

    _print_discrepancy_section(
        "Incorrect (has rule but ground truth invalid)",
        stats.false_positives,
        show_repro,
    )
    _print_discrepancy_section(
        "Possibly missing (valid in ground truth but no DTensor rule)",
        stats.false_negatives,
        show_repro,
    )


def _has_dtensor_support(aten_op: OpOverload) -> bool:
    """Check if an aten op has any DTensor sharding strategy registered."""
    propagator = DTensor._op_dispatcher.sharding_propagator
    if aten_op in propagator.op_single_dim_strategy_funcs:
        return True
    if aten_op in propagator.op_strategy_funcs:
        return True
    return DecompShardingStrategy.has_decomp(aten_op)


def _discover_aten_op(
    opinfos: list[opinfo_core.OpInfo],
    device: str,
    dtype: torch.dtype,
) -> OpOverload | None:
    """Discover the aten op dispatched by the first valid sample."""
    for opinfo in opinfos:
        try:
            samples = list(opinfo.sample_inputs(device, dtype))
        except Exception:
            continue
        for sample in samples:
            tensors = extract_tensors_from_sample(sample)
            if not tensors or any(0 in t.shape for _, t in tensors):
                continue
            capture = get_aten_op_for_sample(opinfo.op, sample, opinfo.name)
            aten_op = capture.best_match
            if aten_op is not None:
                return aten_op
    return None


def _check_ground_truth(
    result: Any,
) -> torch.Tensor | list[torch.Tensor] | None:
    """Validate an op result is suitable as ground truth.

    Returns the ground truth tensor(s) or None if the result should be skipped.
    """
    if isinstance(result, (list, tuple)):
        if not all(isinstance(t, torch.Tensor) for t in result):
            return None
        gt = list(result)
    elif isinstance(result, torch.Tensor):
        gt = result
    else:
        return None

    first_gt = gt[0] if isinstance(gt, list) else gt
    if first_gt.numel() == 0:
        return None
    if (first_gt == 0).all():
        return None
    if first_gt.isnan().all():
        return None
    return gt


def _validate_aten_op_for_sample(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int,
    incorrect_only: bool,
    verbose: bool,
    sample_idx: int,
    variant: str,
    stats: ComparisonStats,
    sample: SampleInput | None = None,
) -> tuple[int, int]:
    """Validate a single aten op with captured args against ground truth.

    Shared logic used by both default (1:1) and allow_composite modes in
    compare_operator. Returns (samples_counted, combinations_counted).
    """
    tensors = extract_tensors_from_args(captured_args, captured_kwargs)
    if not tensors:
        return 0, 0
    if any(0 in t.shape for _, t in tensors):
        return 0, 0

    input_shapes = tuple(t.shape for _, t in tensors)
    gt_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    output_shapes = tuple(tuple(gt.shape) for gt in gt_list)
    n_outputs = len(gt_list)
    first_gt = gt_list[0]

    scalar_args = tuple(a for a in captured_args if not isinstance(a, torch.Tensor))
    scalar_kwargs = {
        k: v for k, v in captured_kwargs.items() if not isinstance(v, torch.Tensor)
    }

    mitigations = _prepare_aten_mitigations(aten_op, captured_args, captured_kwargs)

    input_placement_options = [
        get_1d_input_placements_for_tensor(t, include_partial=True) for _, t in tensors
    ]
    output_placement_options = get_1d_output_placements_for_tensor(first_gt)

    dtensor_rules = _query_dtensor_rules(
        aten_op,
        tensors,
        captured_args,
        captured_kwargs,
        input_shapes,
        output_shapes,
        world_size,
        verbose,
    )

    ground_truth_valid: set[ComboKey] = set()
    total_combinations = 0

    tensor_device = tensors[0][1].device.type if tensors else "cpu"
    with LocalTensorMode(frozenset(range(world_size))):
        mesh = init_device_mesh(tensor_device, (world_size,))

        if incorrect_only:
            combinations_to_test = []
            for combo_key in dtensor_rules:
                input_plc_strs, output_plc_strs = combo_key
                input_plcs_list: list[Placement] = []
                all_valid = True
                for s in input_plc_strs:
                    p = parse_placement(s)
                    if p is None:
                        all_valid = False
                        break
                    input_plcs_list.append(p)
                output_plcs_list: list[Placement] = []
                for s in output_plc_strs:
                    p = parse_placement(s)
                    if p is None:
                        all_valid = False
                        break
                    output_plcs_list.append(p)
                if not all_valid:
                    continue
                combinations_to_test.append(
                    (
                        tuple(input_plcs_list),
                        tuple(output_plcs_list),
                        combo_key,
                    )
                )
        else:
            combinations_to_test = []
            for input_placements in itertools.product(*input_placement_options):
                if is_fully_replicated(input_placements):
                    continue
                for output_placement in output_placement_options:
                    output_placements = tuple(
                        output_placement for _ in range(n_outputs)
                    )
                    combo_key = (
                        tuple(str(p) for p in input_placements),
                        tuple(str(p) for p in output_placements),
                    )
                    combinations_to_test.append(
                        (input_placements, output_placements, combo_key)
                    )

        for (
            input_placements,
            output_placements,
            combo_key,
        ) in combinations_to_test:
            total_combinations += 1
            is_valid = _validate_aten_with_mitigations(
                aten_op,
                captured_args,
                captured_kwargs,
                input_placements,
                output_placements,
                ground_truth,
                world_size,
                mesh,
                mitigations,
            )

            if is_valid:
                normalized_key = normalize_combo_key(
                    combo_key, input_shapes, output_shapes
                )
                if not is_fully_replicated(
                    tuple(parse_placement(p) or Replicate() for p in normalized_key[0])
                ):
                    ground_truth_valid.add(normalized_key)

    _compare_rules(
        ground_truth_valid,
        dtensor_rules,
        input_shapes,
        output_shapes,
        sample_idx,
        scalar_args,
        scalar_kwargs,
        aten_op,
        variant,
        stats,
        sample,
    )

    if verbose:
        print(f"      Sample {sample_idx} [{aten_op}]: shapes={input_shapes}")
        print(f"        Ground truth valid: {len(ground_truth_valid)}")
        print(f"        DTensor rules: {len(dtensor_rules)}")

    return 1, total_combinations


def compare_operator(
    op_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    world_size: int = 2,
    max_samples: int | None = None,
    verbose: bool = False,
    incorrect_only: bool = False,
    allow_composite: bool = False,
) -> ComparisonStats:
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
            Skips search for missing rules (much faster).
        allow_composite: If True, validate each supported aten op individually for
            samples that decompose into multiple aten calls. Default (False)
            skips samples where the OpInfo doesn't map 1:1 to a single aten op.
    """
    if op_name in SKIP_OPS:
        return ComparisonStats()

    opinfos = get_opinfo_by_name(op_name)

    stats = ComparisonStats()

    if not allow_composite:
        aten_op = _discover_aten_op(opinfos, device, dtype)
        if aten_op is None or not _has_dtensor_support(aten_op):
            if verbose:
                print(f"  ATEN_OP_MAP: {op_name} -> {aten_op} [no_support]")
            stats.no_dtensor_support = True
            return stats
        if verbose:
            print(f"  ATEN_OP_MAP: {op_name} -> {aten_op} [supported]")

    total_samples = 0
    total_combinations = 0
    skip_reasons: dict[str, int] = defaultdict(int)

    for opinfo in opinfos:
        variant = opinfo.variant_test_name
        if variant and verbose:
            print(f"\n  OpInfo variant: {variant}")

        op = opinfo.op

        try:
            samples = list(opinfo.sample_inputs(device, dtype))
        except Exception as e:
            if verbose:
                print(f"    Error generating samples: {e}")
            continue

        if max_samples:
            samples = samples[:max_samples]

        for sample_idx, sample in enumerate(samples):
            # Check that SampleInput has tensor inputs and no zero-sized tensors
            sample_tensors = extract_tensors_from_sample(sample)
            if len(sample_tensors) == 0:
                skip_reasons["no tensor inputs"] += 1
                continue
            if any(0 in t.shape for _, t in sample_tensors):
                skip_reasons["zero-sized tensor"] += 1
                continue

            # Capture all aten ops dispatched for this sample
            capture = get_aten_op_for_sample(op, sample, opinfo.name)
            if capture.best_match is None:
                skip_reasons["no aten op captured"] += 1
                continue

            # Count supported aten ops in the capture
            supported_ops = [
                (func, args, kwargs, result)
                for func, args, kwargs, result in capture.all_ops
                if _has_dtensor_support(func)
            ]
            num_supported = len(supported_ops)

            if allow_composite:
                # Validate each supported aten op individually
                if num_supported == 0:
                    skip_reasons["no supported aten ops"] += 1
                    continue

                for func, args, kwargs, result in supported_ops:
                    gt = _check_ground_truth(result)
                    if gt is None:
                        skip_reasons["non-tensor/degenerate aten output"] += 1
                        continue
                    n_samples, n_combos = _validate_aten_op_for_sample(
                        func,
                        args,
                        kwargs,
                        gt,
                        world_size,
                        incorrect_only,
                        verbose,
                        sample_idx,
                        variant,
                        stats,
                        sample,
                    )
                    total_samples += n_samples
                    total_combinations += n_combos
            else:
                # Default: only validate samples with a single supported aten op
                if num_supported != 1:
                    skip_reasons["non-1:1 aten mapping"] += 1
                    continue

                func, args, kwargs, result = supported_ops[0]
                gt = _check_ground_truth(result)
                if gt is None:
                    skip_reasons["non-tensor/degenerate aten output"] += 1
                    continue

                n_samples, n_combos = _validate_aten_op_for_sample(
                    func,
                    args,
                    kwargs,
                    gt,
                    world_size,
                    incorrect_only,
                    verbose,
                    sample_idx,
                    variant,
                    stats,
                    sample,
                )
                total_samples += n_samples
                total_combinations += n_combos

    stats.total_samples = total_samples
    stats.total_combinations = total_combinations
    stats.skip_reasons = dict(skip_reasons)

    # In allow_composite mode, check DTensor support after processing
    if allow_composite and total_samples == 0 and not skip_reasons:
        stats.no_dtensor_support = True

    return stats


def get_registered_op_names() -> list[str]:
    """Get all op names that have DTensor sharding rules and also have OpInfo.

    Returns OpInfo names (which may differ from aten base names, e.g.,
    "nn.functional.relu" instead of "relu").
    """

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

    # Build mappings from OpInfo: both by name and by aten_name
    opinfo_by_name = {}
    opinfo_by_aten_name: dict[str, list[str]] = {}
    for op in op_db:
        opinfo_by_name[op.name] = True
        opinfo_by_aten_name.setdefault(op.aten_name, []).append(op.name)

    result = set()
    for base_name in base_names:
        if base_name in opinfo_by_name:
            # Direct match (e.g., "add" -> OpInfo named "add")
            result.add(base_name)
        elif base_name in opinfo_by_aten_name:
            # Match via aten_name (e.g., "relu" -> OpInfo "nn.functional.relu"
            # which has aten_name="relu")
            result.update(opinfo_by_aten_name[base_name])

    return sorted(result)


def report_registrations(verbose: bool = False) -> None:
    """Report the number (and optionally full list) of ops by registration method.

    Directly registered categories (mutually exclusive):
      - rule: ops registered via register_prop_rule
      - op_strategy: ops registered via register_op_strategy
      - single_dim_strategy: ops registered via register_single_dim_strategy

    Ops not in any of the above may still be supported at runtime via
    DecompShardingStrategy, which traces through the op's decomposition and
    propagates placements through the decomposed sub-ops.  Whether this
    actually works depends on every sub-op having a registered strategy.
    We report the decomposition_table entries as a separate (untested) count.
    """
    from torch._decomp import decomposition_table

    propagator = DTensor._op_dispatcher.sharding_propagator

    rule_ops = sorted(propagator.op_to_rules.keys(), key=str)
    strategy_ops = sorted(propagator.op_strategy_funcs.keys(), key=str)
    single_dim_ops = sorted(propagator.op_single_dim_strategy_funcs.keys(), key=str)

    directly_registered = (
        set(propagator.op_to_rules.keys())
        | set(propagator.op_strategy_funcs.keys())
        | set(propagator.op_single_dim_strategy_funcs.keys())
    )

    # Ops from the explicit decomposition table that aren't directly registered.
    # These *may* work via DecompShardingStrategy if all their sub-ops are
    # supported, but we can't verify that without tracing each one.
    decomp_only_ops = sorted(
        (op for op in decomposition_table if op not in directly_registered),
        key=str,
    )

    print("=" * 70)
    print("DTensor operator registration report")
    print("=" * 70)

    print("\nDirectly registered:")
    print(f"  rule (register_prop_rule):            {len(rule_ops):>4}")
    print(f"  op_strategy (register_op_strategy):   {len(strategy_ops):>4}")
    print(f"  single_dim_strategy:                  {len(single_dim_ops):>4}")
    print(f"  total:                                {len(directly_registered):>4}")

    print(f"\nDecomposition table (not directly registered): {len(decomp_only_ops)}")
    print(
        "  These ops have entries in torch._decomp.decomposition_table but no\n"
        "  direct DTensor strategy. They may work at runtime via\n"
        "  DecompShardingStrategy if all decomposed sub-ops are supported.\n"
        "  Additional ops beyond this count may also be reachable via CIA\n"
        "  (CompositeImplicitAutograd) decompositions."
    )

    if verbose:

        def _print_ops(label: str, ops: list) -> None:
            print(f"\n{label} ({len(ops)}):")
            for op in ops:
                print(f"  {op}")

        _print_ops("rule", rule_ops)
        _print_ops("op_strategy", strategy_ops)
        _print_ops("single_dim_strategy", single_dim_ops)
        _print_ops("decomp table (not directly registered)", decomp_only_ops)


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
    parser.add_argument(
        "--op",
        default=None,
        help="Operator name(s) to compare (comma-separated, supports glob "
        'patterns, e.g., "relu,add" or "nn.functional.*")',
    )
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
    parser.add_argument(
        "--allow-composite",
        action="store_true",
        help="Validate each supported aten op individually for decomposed ops "
        "(default skips non-1:1 aten mappings)",
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="float32", help="Dtype to use")
    parser.add_argument(
        "--world-size", type=int, default=2, help="Simulated world size"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Max samples to test"
    )
    parser.add_argument(
        "--show-repro",
        nargs="?",
        const=1,
        type=int,
        default=0,
        metavar="N",
        help="Show N sample repros per rule (default 1 if flag given, -1 for all)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Show registration statistics (op counts per registration method)",
    )
    parser.add_argument(
        "--report-full",
        action="store_true",
        help="Show full op lists with registration statistics (implies --report)",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    dist.init_process_group("fake", rank=0, world_size=args.world_size)

    if args.report or args.report_full:
        report_registrations(verbose=args.report_full)
        if not args.all_registered and not args.op:
            dist.destroy_process_group()
            sys.exit(0)
        print()

    try:
        if args.all_registered:
            op_names = get_registered_op_names()
        elif args.op:
            patterns = [p.strip() for p in args.op.split(",")]
            op_names = resolve_op_names(patterns)
        else:
            op_names = ["add"]

        # Preamble
        display_names = [_format_op_name(n) for n in op_names]
        print(f"Testing ops: {', '.join(display_names)}")
        if args.allow_composite:
            print(
                "Mode: allow-composite (validates each aten op in decomposed samples)"
            )
        if args.incorrect_only:
            print("Mode: incorrect-only (fast)")
        print(f"Device: {args.device}, Dtype: {dtype}, World size: {args.world_size}")

        op_results: list[tuple[str, ComparisonStats, float]] = []

        for i, op_name in enumerate(op_names):
            display = _format_op_name(op_name)

            if op_name in SKIP_OPS:
                print(
                    f"\n[{i + 1}/{len(op_names)}] {display}"
                    f" — skipped: {SKIP_OPS[op_name]}"
                )
                continue

            try:
                op_start = time.time()
                stats = compare_operator(
                    op_name,
                    args.device,
                    dtype,
                    args.world_size,
                    args.max_samples,
                    verbose=True,
                    incorrect_only=args.incorrect_only,
                    allow_composite=args.allow_composite,
                )
                elapsed = time.time() - op_start

                if stats.no_dtensor_support:
                    print(
                        f"\n[{i + 1}/{len(op_names)}] {display}"
                        f" — skipped: no DTensor support"
                    )
                    continue

                if stats.total_samples == 0 and stats.skip_reasons:
                    reasons = ", ".join(
                        f"{r} ({n})" for r, n in stats.skip_reasons.items()
                    )
                    print(f"\n[{i + 1}/{len(op_names)}] {display} — skipped: {reasons}")
                    continue

                op_results.append((op_name, stats, elapsed))

                skipped = sum(stats.skip_reasons.values())
                skipped_str = f" ({skipped} skipped)" if skipped else ""
                print(
                    f"\n[{i + 1}/{len(op_names)}] {display}"
                    f" — Samples: {stats.total_samples}{skipped_str},"
                    f" Combinations: {stats.total_combinations}"
                )
                print("-" * 70)
                _print_comparison_summary(stats, show_repro=args.show_repro)
            except Exception as e:
                print(f"\n[{i + 1}/{len(op_names)}] {display} — Error: {e}")

        if len(op_results) >= 1:
            # Summary table
            print("\n" + "=" * 70)
            print("Summary")
            print("=" * 70)

            formatted_results = [
                (_format_op_name(name), stats, elapsed)
                for name, stats, elapsed in op_results
            ]

            col_widths = {
                "op": max(len(name) for name, _, _ in formatted_results),
                "tp": 7,
                "fp": 9,
                "fn": 7,
                "time": 6,
            }
            col_widths["op"] = max(col_widths["op"], 2)  # min width

            header = (
                f"{'Op':<{col_widths['op']}}  "
                f"{'Correct':>{col_widths['tp']}}  "
                f"{'Incorrect':>{col_widths['fp']}}  "
                f"{'Missing':>{col_widths['fn']}}  "
                f"{'Time':>{col_widths['time']}}"
            )
            print(header)
            print("-" * len(header))

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_time = 0.0

            for name, stats, elapsed in formatted_results:
                fp_rules = {
                    (d.input_placements, d.output_placements)
                    for d in stats.false_positives
                }
                fn_rules = {
                    (d.input_placements, d.output_placements)
                    for d in stats.false_negatives
                }
                n_fp = len(fp_rules)
                n_fn = len(fn_rules)
                total_tp += stats.true_positives
                total_fp += n_fp
                total_fn += n_fn
                total_time += elapsed

                fp_str = str(n_fp) if n_fp else "0"
                fn_str = str(n_fn) if n_fn else "0"

                print(
                    f"{name:<{col_widths['op']}}  "
                    f"{stats.true_positives:>{col_widths['tp']}}  "
                    f"{fp_str:>{col_widths['fp']}}  "
                    f"{fn_str:>{col_widths['fn']}}  "
                    f"{elapsed:>{col_widths['time']}.1f}s"
                )

            print("-" * len(header))
            fp_str = str(total_fp) if total_fp else "0"
            fn_str = str(total_fn) if total_fn else "0"
            print(
                f"{'Total':<{col_widths['op']}}  "
                f"{total_tp:>{col_widths['tp']}}  "
                f"{fp_str:>{col_widths['fp']}}  "
                f"{fn_str:>{col_widths['fn']}}  "
                f"{total_time:>{col_widths['time']}.1f}s"
            )
    finally:
        dist.destroy_process_group()
