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

Adversarial audits of the rules DTensor claims (special values, integer
extremes, Scalar-argument variants, uneven/empty shards, world sizes 2-4):
    python -m torch.distributed.tensor._ops.strategy_validation --op sign --stress
    python -m torch.distributed.tensor._ops.strategy_validation --all-registered \
        --incorrect-only --stress --dtype float32,int64
"""

import argparse
import fnmatch
import itertools
import math
import re
import sys
import time
import unittest.mock
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch._ops import OpOverload
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode
from torch.distributed._local_tensor._c10d import _local_reduce
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor._decompositions import DecompShardingStrategy
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import DTensorSpec, OpSchema, OpStrategy
from torch.distributed.tensor._sharding_prop import (
    _propagate_use_strided_shard_flag,
    _select_min_cost_strategy,
)
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
    "exponential": "non-deterministic (random sampling)",
    "new_empty": "uninitialized memory",
    "new_empty_strided": "uninitialized memory",
    "nn.functional.dropout": "non-deterministic (random masking)",
    "normal": "non-deterministic (random sampling)",
    "rand_like": "non-deterministic (random sampling)",
    "randint_like": "non-deterministic (random sampling)",
    "randn_like": "non-deterministic (random sampling)",
    "uniform": "non-deterministic (random sampling)",
}

# Tensor operands that carry structure rather than values (indices, pivots,
# sorted boundaries). Value variants never modify them.
CONSTRAINED_TENSOR_ARGS = frozenset(
    {
        "batch_sizes",
        "boundaries",
        "index",
        "indices",
        "lengths",
        "offsets",
        "pivots",
        "sorted_sequence",
        "sorter",
        "target",
    }
)

# OpInfo (S, M, L, XS) settings used to generate samples. "even" makes S, M
# and L divide evenly on 2 and 4 ranks (XS=2 still leaves empty shards on 4);
# OpInfo's defaults are "base". "base" and "small" give uneven and empty shards.
SIZE_PROFILES: dict[str, tuple[int, int, int, int]] = {
    "even": (4, 12, 24, 2),
    "base": (5, 10, 20, 3),
    "small": (3, 7, 13, 2),
}

SPECIAL_VALUES = (math.nan, math.inf, -math.inf, -0.0)

# In incorrect-only mode, claims are queried for every input placement
# combination up to this many; larger products are searched (see
# claimed_input_combinations).
_MAX_ENUMERATED_COMBINATIONS = 1024

_DTYPE_CONVERSION_OPS = frozenset(
    {
        torch.ops.aten._to_copy.default,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
        torch.ops.aten.to.other,
    }
)

# Scalar/float arguments that are sizes, scale factors or numerical
# constants rather than values the op applies elementwise; fuzzing them only
# leaves the op's domain.
CONSTRAINED_SCALAR_ARGS = frozenset({"eps", "momentum", "scale", "scales", "size"})

# Directed values for Scalar/float arguments; Scalar arguments also take the
# Python ints. Pairs of crossing values (0, 2) / (2, 0) cover conditions such
# as threshold's value > threshold.
SCALAR_FLOAT_VALUES = (0.0, 2.0, -1.0, 0.5, -2.0, 1.0)
SCALAR_INT_VALUES = (0, 2, -1, 1, -2, 3)


@dataclass(frozen=True)
class ValidationOptions:
    """Opt-in adversarial variants used to audit DTensor-claimed rules.

    special_values: also run claimed (and, outside incorrect-only mode,
        finite-valid) combinations on inputs with NaN/+-inf/-0.0, adversarial
        Partial pieces and integer extremes/wraparound. A missing rule is only
        reported if it passes every variant.
    scalar_variants: number of scalar-argument variants per sample (0
        disables scalar fuzzing). Directed values come first, then seeded
        random draws.
    seed: base seed for every generated variant.
    """

    special_values: bool = False
    scalar_variants: int = 0
    seed: int = 0


class _UntestableInput(Exception):
    """The validator cannot construct inputs for this placement combination."""


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
    # Input variant that exposed an incorrect rule (e.g. "finite",
    # "special-nan", "nan-partial", "int-wraparound", "rewrite"), possibly
    # prefixed by "scalar:<args> " and "empty-shard " or suffixed "(alternative)".
    trigger: str = ""
    world_size: int = 0
    dtype: torch.dtype | None = None


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
    # Claimed rules the validator could not run or could not decide
    # ("unresolved: ..."), by reason
    untestable_claims: dict[str, int] = field(default_factory=dict)
    # Samples whose claimed input combinations were found by search, which can
    # miss some (see claimed_input_combinations)
    incomplete_discovery: int = 0


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
    # NOTE: isinstance(_, Shard) does not match _StridedShard; see _is_shard_like().
    return (
        isinstance(p, Shard) and p.dim < len(tensor_shape) and tensor_shape[p.dim] == 1
    )


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
    Returns a list of (name, tensor) pairs, in pytree traversal order.
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


def _partial_pieces(
    tensor: torch.Tensor,
    placement: Partial,
    world_size: int,
    tensor_idx: int = 0,
    mask_shift: int = 0,
    adversarial: bool = False,
    wraparound: bool = False,
) -> list[torch.Tensor] | None:
    """
    Per-rank pieces that reduce to ``tensor`` element-wise, or None if the
    validator cannot construct them (custom Partial subclasses, P(product),
    bool, integer P(avg), complex P(min)/P(max)).

    The main challenge is avoiding false positives where a rule appears valid
    on a specific input but is actually incorrect:

    Asymmetric splits for P(sum)/P(avg): rank 0 gets a 60/70/80% share (varied
    by tensor index) so that ops which are not truly linear don't accidentally
    produce matching outputs. A sign-varying offset (checkerboard mask, so it
    alternates along every dim) gives local tensors mixed signs; otherwise
    proportional splits preserve the sign pattern and e.g. abs would falsely
    validate P(sum)->P(sum).

    Distinct magnitudes for P(min)/P(max): one rank holds the true value and
    the others hold it offset by +-(range*2+1), where range is the range of the
    finite values, so index-returning ops (argmin/argmax) see different winners
    on different ranks. The checkerboard mask decides which rank holds the true
    value; mask_shift flips it to re-validate with the complementary pattern.

    Non-finite values: NaN, +inf and -inf positions are held by one rank and
    the other ranks get finite values on both sides of zero (P(max) at a -inf
    position and P(min) at a +inf position need -inf/+inf on every rank).
    These are exact under LocalTensor's reductions (functools.reduce with
    torch.maximum/minimum, which propagate NaN, and in-dtype addition).

    adversarial: non-holding ranks of P(max)/P(min) get -inf/+inf at finite
    positions, and NaN positions of P(sum)/P(avg) become a +inf/-inf pair.

    Integer pieces stay in the input dtype and never overflow: P(sum) splits
    x into near-equal parts with small sign-varying perturbations, and the
    P(min)/P(max) losers stay within half the dtype's range. With wraparound,
    integer P(sum) pieces are chosen so that their true sum leaves the dtype's
    range while their in-dtype (wrapping) sum is x; None if no element can.
    """
    reduce_op = placement.reduce_op
    if type(placement) is not Partial or reduce_op not in PARTIAL_REDUCE_OPS:
        return None
    if tensor.dtype == torch.bool or (
        tensor.is_complex() and reduce_op in ("min", "max")
    ):
        return None
    mask = _checkerboard_mask(tensor, tensor_idx, mask_shift)
    flat = tensor.flatten()
    if not tensor.dtype.is_floating_point and not tensor.is_complex():
        pieces = _integer_partial_pieces(flat, reduce_op, world_size, mask, wraparound)
    elif reduce_op in ("sum", "avg"):
        pieces = _float_sum_pieces(
            flat, reduce_op, world_size, mask, tensor_idx, adversarial
        )
    else:
        pieces = _float_minmax_pieces(flat, reduce_op, world_size, mask, adversarial)
    if pieces is None:
        return None
    return [p.reshape(tensor.shape) for p in pieces]


def _float_sum_pieces(
    flat: torch.Tensor,
    reduce_op: str,
    world_size: int,
    mask: torch.Tensor,
    tensor_idx: int,
    adversarial: bool,
) -> list[torch.Tensor] | None:
    finite = flat.isfinite()
    base_ratio = 0.6 + 0.1 * (tensor_idx % 3)
    scale = world_size if reduce_op == "avg" else 1
    max_offset = torch.finfo(flat.dtype).max / 4
    finite_flat = torch.where(finite, flat, torch.zeros_like(flat))
    sign = torch.where(mask, -1.0, 1.0).to(finite_flat.real.dtype)
    offset = (finite_flat.abs().clamp(max=max_offset) + 1.0) * sign
    others = world_size - 1
    pieces = [finite_flat * base_ratio * scale + offset]
    for _ in range(others):
        pieces.append(
            finite_flat * ((1 - base_ratio) / others) * scale - offset / others
        )
    if not all(bool((p.isfinite() | ~finite).all()) for p in pieces):
        return None
    if finite.all():
        return pieces
    split_nan = flat.isnan() & adversarial
    for r in range(world_size):
        special = flat if r == 0 else offset.to(flat.dtype)
        if r < 2:
            pair = torch.full_like(flat, math.inf if r == 0 else -math.inf)
            special = torch.where(split_nan, pair, special)
        pieces[r] = torch.where(finite, pieces[r], special)
    return pieces


def _float_minmax_pieces(
    flat: torch.Tensor,
    reduce_op: str,
    world_size: int,
    mask: torch.Tensor,
    adversarial: bool,
) -> list[torch.Tensor]:
    finite = flat.isfinite()
    finite_vals = flat[finite].double()
    value_range = (
        (finite_vals.max() - finite_vals.min()).item() if finite_vals.numel() else 0.0
    )
    magnitude = finite_vals.abs().max().item() + 1.0 if finite_vals.numel() else 1.0
    finfo = torch.finfo(flat.dtype)
    # Losing pieces sit below the true value for P(max) and above it for P(min).
    direction = -1.0 if reduce_op == "max" else 1.0
    winning_inf = -direction * math.inf
    loser = (flat.double() + direction * (value_range * 2 + 1)).clamp(
        finfo.min, finfo.max
    )
    loser = loser.to(flat.dtype)
    # Where any value would lose, alternate the sign across the non-finite
    # positions so ops that map NaN or inf to a finite value meet both larger
    # and smaller values on the losing ranks.
    special_rank = (~finite).cumsum(0)
    filler = torch.where(special_rank % 2 == 1, magnitude, -magnitude).to(flat.dtype)
    loser = torch.where(flat.isnan() | (flat == winning_inf), filler, loser)
    loser = torch.where(flat == -winning_inf, flat, loser)
    if adversarial:
        loser = torch.where(finite, torch.full_like(flat, -winning_inf), loser)
    return [
        torch.where(mask if r == 0 else ~mask, flat, loser) for r in range(world_size)
    ]


def _integer_partial_pieces(
    flat: torch.Tensor,
    reduce_op: str,
    world_size: int,
    mask: torch.Tensor,
    wraparound: bool = False,
) -> list[torch.Tensor] | None:
    if reduce_op == "avg" or flat.dtype in (torch.uint16, torch.uint32, torch.uint64):
        return None
    info = torch.iinfo(flat.dtype)
    if wraparound and reduce_op == "sum":
        # Ranks > 0 hold a value at the far end of the range (opposite in
        # sign to x) and rank 0 holds x minus their in-dtype sum. Elements
        # whose true sum stays in range (e.g. x == -1 on two ranks) cannot wrap.
        if info.min < 0:
            big = torch.where(flat >= 0, info.min, info.max)
            big = torch.where(mask, big, big - big.sign())
        else:
            big = torch.where(mask, info.max, info.max - 1)
        pieces = [flat.clone()] + [
            big.to(flat.dtype).clone() for _ in range(world_size - 1)
        ]
        for other in pieces[1:]:
            pieces[0] = pieces[0] - other
        acc, crossed = pieces[0], torch.zeros_like(mask)
        for p in pieces[1:]:
            total = acc + p
            crossed |= torch.where(p >= 0, total < acc, total > acc)
            acc = total
        return pieces if bool(crossed.any()) else None
    wide = flat.to(torch.int64)
    if reduce_op in ("min", "max"):
        # Losing pieces stay within half the dtype's range so that ops do not
        # overflow on them; the int-extreme variant covers the extremes.
        offset = min(2 * (int(wide.max()) - int(wide.min())) + 1, 2**62)
        if reduce_op == "max":
            floor = info.min // 2
            loser = torch.where(
                wide >= floor + offset, wide - offset, torch.clamp(wide, max=floor)
            )
        else:
            ceil = info.max // 2
            loser = torch.where(
                wide <= ceil - offset, wide + offset, torch.clamp(wide, min=ceil)
            )
        pieces = [
            torch.where(mask if r == 0 else ~mask, wide, loser)
            for r in range(world_size)
        ]
        return [p.to(flat.dtype) for p in pieces]
    # P(sum): rank r>0 holds q - d and rank 0 holds q + rem + (world_size - 1) * d,
    # where q = x // world_size and |d| <= 1. Partial sums move monotonically
    # from rank 0's piece to x, so none of them leaves the dtype's range.
    q = torch.div(wide, world_size, rounding_mode="floor")
    first = wide - q * (world_size - 1)
    if info.min < 0:
        d = torch.where(mask, 1, -1)
    else:
        d = torch.where(mask & (q > 0), 1, 0)
    spread = world_size - 1
    safe = (first + spread <= info.max) & (first - spread >= info.min)
    safe &= (q + 1 <= info.max) & (q - 1 >= info.min)
    d = torch.where(safe, d, 0)
    pieces = [first + spread * d] + [q - d for _ in range(spread)]
    return [p.to(flat.dtype) for p in pieces]


_VALUE_MISMATCH = "Value mismatch"
# Prefix of untestable reasons for value mismatches that vanish on the same
# pieces in 64-bit, or whose 64-bit re-run still computes in narrow floats:
# rounding, overflow and a branch flip look alike there, never counted correct.
_UNRESOLVED = "unresolved: "


_TOLERANCES = {
    torch.float16: (1e-3, 1e-3),
    torch.complex32: (1e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1e-2),
}


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """(rtol, atol) used to compare a sharded result against the reference."""
    return _TOLERANCES.get(dtype, (1e-5, 1e-5))


def _mismatch(got: torch.Tensor, want: torch.Tensor) -> str:
    """Empty if ``got`` matches ``want``, else a description.

    Dtypes must be equal, integers and bools identical, NaN positions (a
    complex value is NaN if either component is) and inf signs identical;
    other values must be within atol + rtol * |want| as in torch.isclose,
    per component for complex values with an infinite component.
    """
    if got.dtype != want.dtype:
        return f"dtype {got.dtype}, expected {want.dtype}"
    if want.is_floating_point() or want.is_complex():
        rtol, atol = _tolerances(want.dtype)
        bad = ~torch.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True)
        if want.is_complex():
            g = torch.view_as_real(got.resolve_conj())
            w = torch.view_as_real(want.resolve_conj())
            close = torch.isclose(g, w, rtol=rtol, atol=atol, equal_nan=True).all(-1)
            infinite = (got.isinf() | want.isinf()) & ~(got.isnan() & want.isnan())
            bad = torch.where(infinite, ~close, bad)
    else:
        bad = got != want
    if not bad.any():
        return ""
    idx = tuple(bad.nonzero()[0].tolist())
    return (
        f"{int(bad.sum())}/{want.numel()} elements differ, e.g. at {list(idx)}: "
        f"got {got[idx].item()}, expected {want[idx].item()}"
    )


def _compare_outputs(
    local_output: Any,
    ground_truth: torch.Tensor | list[torch.Tensor],
    output_placements: tuple[Placement, ...],
    mesh: DeviceMesh,
    world_size: int,
) -> tuple[bool, str]:
    """Compare op output (wrapped as DTensor) against ground truth.

    Value mismatches are reported with the _VALUE_MISMATCH prefix.
    """
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
            if any(
                lv.shape != local_values[0].shape or _mismatch(lv, local_values[0])
                for lv in local_values[1:]
            ):
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

        if full_output.dtype != gt.dtype:
            got, want = full_output.dtype, gt.dtype
            return False, f"Dtype mismatch[{i}]: got {got}, expected {want}"
        mismatch = _mismatch(full_output, gt)
        if mismatch:
            return False, f"{_VALUE_MISMATCH}[{i}]: {mismatch}"

    return True, ""


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


@dataclass(frozen=True)
class _Variant:
    """One way of perturbing the captured inputs before validating a combination.

    specials: "" (none), "nan" / "inf" / "-inf" (written into a few elements),
        "random" (NaN/+-inf/-0.0 in random elements) or "int" (integer dtype
        min/max).
    target: which operands receive specials: "partial", "other" (non-Partial)
        or "all"; see _materialize_inputs for the policy.
    adversarial: losing ranks of P(max)/P(min) hold -inf/+inf; NaN under
        P(sum)/P(avg) becomes a +inf/-inf pair.
    wraparound: integer P(sum) pieces whose true sum leaves the dtype's range.
    """

    name: str
    mask_shift: int = 0
    negate: bool = False
    specials: str = ""
    target: str = "all"
    adversarial: bool = False
    wraparound: bool = False
    seed: int = 0


def _value_variants(
    input_placements: tuple[Placement, ...],
    options: ValidationOptions,
    float_specials: bool = True,
) -> list[_Variant]:
    """Input variants for a placement combination, in the order they are tried.

    The finite variant is mitigated against coincidental passes by re-running
    with the complementary checkerboard mask (any Partial input) and with
    negated inputs (P(min)/P(max) inputs). float_specials=False drops the
    NaN/inf variants (e.g. for float -> integer conversions, where they are
    undefined behavior).
    """
    partial = [p for p in input_placements if isinstance(p, Partial)]
    variants = [_Variant("finite")]
    if partial:
        variants.append(_Variant("finite", mask_shift=1))
    if any(p.reduce_op in ("min", "max") for p in partial):
        variants.append(_Variant("finite", negate=True))
    if options.special_values and float_specials:
        for kind, name in (("nan", "nan"), ("inf", "posinf"), ("-inf", "neginf")):
            if partial:
                variants.append(
                    _Variant(f"special-{name}", specials=kind, target="partial")
                )
            variants.append(_Variant(f"special-{name}", specials=kind, target="other"))
        if partial:
            variants.append(
                _Variant(
                    "adversarial", specials="nan", target="partial", adversarial=True
                )
            )
        variants.append(
            _Variant("special-random", specials="random", seed=options.seed)
        )
    if options.special_values:
        variants.append(_Variant("int-extreme", specials="int"))
        if any(p.reduce_op == "sum" for p in partial):
            variants.append(_Variant("int-wraparound", wraparound=True))
    return variants


def _constrained_operands(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
) -> list[bool]:
    """Per tensor operand (extract_tensors_from_args order): True if values must not change.

    Structural operands (indices, pivots, lengths, sorted boundaries, targets)
    are recognized by schema argument name (CONSTRAINED_TENSOR_ARGS).
    """
    schema_args = aten_op._schema.arguments
    names: list[str] = []
    for i, arg in enumerate(captured_args):
        name = schema_args[i].name if i < len(schema_args) else ""
        names += [name] * sum(
            isinstance(x, torch.Tensor) for x in pytree.tree_leaves(arg)
        )
    for name, value in captured_kwargs.items():
        names += [name] * sum(
            isinstance(x, torch.Tensor) for x in pytree.tree_leaves(value)
        )
    return [any(s in name for s in CONSTRAINED_TENSOR_ARGS) for name in names]


def _inject_specials(t: torch.Tensor, kind: str, gen: torch.Generator) -> torch.Tensor:
    """Copy of ``t`` with special values written into some elements (see _Variant)."""
    if t.numel() == 0:
        return t
    flat = t.flatten().clone()
    positions = sorted({0, flat.numel() // 2})
    if kind == "int":
        if t.is_floating_point() or t.is_complex() or t.dtype == torch.bool:
            return t
        info = torch.iinfo(t.dtype)
        flat[positions[0]] = info.min
        flat[positions[-1]] = info.max
    elif not t.is_floating_point():
        return t
    elif kind == "random":
        values = torch.tensor(SPECIAL_VALUES, dtype=t.dtype, device=t.device)
        replace = torch.rand(flat.numel(), generator=gen) < 0.2
        choice = torch.randint(len(SPECIAL_VALUES), (flat.numel(),), generator=gen)
        flat = torch.where(replace.to(t.device), values[choice.to(t.device)], flat)
    else:
        flat[positions] = float(kind)
    return flat.reshape(t.shape)


def _materialize_inputs(
    tensors: list[tuple[str, torch.Tensor]],
    input_placements: tuple[Placement, ...],
    world_size: int,
    variant: _Variant,
    constrained: list[bool],
) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
    """Global inputs and their per-rank pieces for one combination and variant.

    The global value of a Partial input is the reduction of its pieces under
    LocalTensor semantics, so the reference never depends on how exactly a
    decomposition reproduces the captured value. Constrained operands are
    never modified. Specials go into non-Partial operands only if no input is
    P(sum)/P(avg) and the variant is not adversarial: P(sum) with mixed-sign
    pieces cannot commute with a non-finite operand even for linear ops
    (pieces [-1, 2] times inf give NaN), which is a limit of Partial semantics
    rather than a rule bug. Raises _UntestableInput if the variant does not
    apply to this combination.
    """
    linear_partial = any(
        isinstance(p, Partial) and p.reduce_op in ("sum", "avg")
        for p in input_placements
    )
    gen = torch.Generator().manual_seed(variant.seed)
    changed = not (variant.specials or variant.wraparound)
    global_inputs: list[torch.Tensor] = []
    rank_inputs: list[list[torch.Tensor]] = []
    for idx, ((_, tensor), placement, fixed) in enumerate(
        zip(tensors, input_placements, constrained)
    ):
        value = tensor.clone()
        is_partial = isinstance(placement, Partial)
        if not fixed and variant.negate and value.dtype != torch.bool:
            if (
                value.is_floating_point()
                or value.is_complex()
                or torch.iinfo(value.dtype).min < 0
            ):
                value = -value
        if not fixed and variant.specials:
            if is_partial:
                inject = variant.target in ("partial", "all")
            else:
                inject = variant.target in ("other", "all") and not (
                    linear_partial or variant.adversarial
                )
            if inject:
                injected = _inject_specials(value, variant.specials, gen)
                changed |= injected is not value
                value = injected
        if is_partial:
            pieces = _partial_pieces(
                value,
                placement,
                world_size,
                idx,
                variant.mask_shift,
                variant.adversarial,
                variant.wraparound,
            )
            changed |= (
                variant.wraparound
                and pieces is not None
                and not (value.is_floating_point() or value.is_complex())
            )
            if pieces is None:
                raise _UntestableInput(
                    f"cannot construct {placement} input of {value.dtype}"
                )
            value = _local_reduce(placement.reduce_op, pieces)
        elif isinstance(placement, Shard):
            chunks, _ = placement._split_tensor(value, world_size, with_padding=False)
            pieces = [c.clone(memory_format=torch.contiguous_format) for c in chunks]
        elif isinstance(placement, Replicate):
            pieces = [value.clone() for _ in range(world_size)]
        else:
            raise _UntestableInput(f"cannot construct {placement} input")
        global_inputs.append(value)
        rank_inputs.append(pieces)
    if not changed:
        raise _UntestableInput(f"variant {variant.name} does not apply")
    return global_inputs, rank_inputs


@dataclass
class _RuntimeClaim:
    """An output placement DTensor produces, without redistributing, for the tested inputs.

    local_op/local_args/local_kwargs are the call DTensor makes on the local
    tensors: non-tensor args that refer to global sizes are rewritten
    (rewritten=True), and tensors are DTensorSpec placeholders.
    """

    output_placements: tuple[Placement, ...]
    local_op: OpOverload
    local_args: tuple[Any, ...]
    local_kwargs: dict[str, Any]
    selected: bool
    rewritten: bool = False
    # Set if DTensor's rewrite of the local call raises; dispatch raises the
    # same error when it selects this candidate.
    error: str = ""


def _runtime_schema(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placements: Iterable[Placement],
    mesh: DeviceMesh,
) -> OpSchema:
    """OpSchema the runtime builds for the captured call with the given input placements."""
    propagator = DTensor._op_dispatcher.sharding_propagator
    placements = iter(input_placements)

    def _to_spec(a: Any) -> Any:
        if isinstance(a, torch.Tensor):
            meta = TensorMeta(shape=a.shape, stride=a.stride(), dtype=a.dtype)
            return DTensorSpec(mesh, (next(placements),), tensor_meta=meta)
        return a

    schema_info = propagator.op_to_schema_info.get(aten_op)
    if schema_info is None:
        schema_info = propagator.op_to_schema_info_for_single_dim_strategy.get(aten_op)
    return OpSchema(
        aten_op,
        pytree.tree_map(_to_spec, captured_args),
        pytree.tree_map(_to_spec, captured_kwargs),
        schema_info=schema_info,
    )


def _output_meta(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    mesh: DeviceMesh,
) -> Any:
    """Global output tensor meta of the captured call (it does not depend on the input placements), or None."""
    replicated = itertools.repeat(Replicate())
    schema = _runtime_schema(aten_op, captured_args, captured_kwargs, replicated, mesh)
    propagator = DTensor._op_dispatcher.sharding_propagator
    try:
        return propagator._propagate_tensor_meta_non_cached(schema)
    except Exception:
        return None


def _runtime_strategy(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placements: tuple[Placement, ...],
    mesh: DeviceMesh,
    *,
    out_meta: Any = None,
) -> tuple[OpSchema, OpStrategy, Any] | None:
    """Runtime OpSchema, candidate OpStrategy and output meta for the given input placements.

    Mirrors ShardingPropagator.propagate_op_sharding_non_cached up to strategy
    selection; out_meta (see _output_meta) is computed if not given.
    Returns None if the op has no strategy for these inputs and raises
    _UntestableInput if propagation fails or the strategy is not an
    OpStrategy (e.g. TupleStrategy for foreach ops).
    """
    propagator = DTensor._op_dispatcher.sharding_propagator
    call = (aten_op, captured_args, captured_kwargs)
    op_schema = _runtime_schema(*call, input_placements, mesh)
    try:
        if out_meta is None:
            out_meta = propagator._propagate_tensor_meta_non_cached(op_schema)
        op_strategy, decomp_error = propagator._get_op_strategy(op_schema, out_meta)
    except Exception as e:
        raise _UntestableInput(f"sharding propagation raised {type(e).__name__}") from e
    if decomp_error is not None:
        raise _UntestableInput(f"decomposition raised {type(decomp_error).__name__}")
    if op_strategy is None:
        return None
    if not isinstance(op_strategy, OpStrategy):
        raise _UntestableInput(f"{type(op_strategy).__name__} is not supported")
    if not op_strategy.strategies:
        return None
    _propagate_use_strided_shard_flag(op_strategy, op_schema)
    return op_schema, op_strategy, out_meta


def _desired_input_placements(
    op_schema: OpSchema, spec: Any
) -> tuple[Placement, ...] | None:
    """Placements a candidate strategy wants for the positional tensor args (as the runtime compares them)."""
    if spec.input_specs is None:
        if not isinstance(spec.output_specs, DTensorSpec):
            return None
        return tuple(spec.output_spec.placements[0] for _ in op_schema.args_spec)
    return tuple(
        spec.input_specs[i].placements[0] for i in range(len(op_schema.args_spec))
    )


def claimed_input_combinations(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placement_options: list[list[Placement]],
    mesh: DeviceMesh,
    *,
    out_meta: Any = None,
) -> tuple[list[tuple[Placement, ...]], bool]:
    """Non-replicated input placement combinations to query for claims, and whether the list is complete.

    Every combination is returned up to _MAX_ENUMERATED_COMBINATIONS (a
    sixteenth of that for decompositions, whose queries trace the
    decomposition). Beyond that, a search starts from combinations with a
    single non-replicated input and follows the input placements every
    candidate strategy asks for; it can miss claims, since candidates can
    depend on the actual input placements (e.g. under uneven sharding).
    """
    n = len(input_placement_options)
    propagator = DTensor._op_dispatcher.sharding_propagator
    limit = _MAX_ENUMERATED_COMBINATIONS
    if (
        aten_op not in propagator.op_strategy_funcs
        and aten_op not in propagator.op_single_dim_strategy_funcs
    ):
        limit //= 16
    n_combos = math.prod(len(o) for o in input_placement_options)
    if n_combos <= limit:
        product = itertools.product(*input_placement_options)
        return [c for c in product if not is_fully_replicated(c)], True
    replicated = tuple(Replicate() for _ in range(n))
    queue = [
        replicated[:i] + (p,) + replicated[i + 1 :]
        for i, options in enumerate(input_placement_options)
        for p in options
        if not isinstance(p, Replicate)
    ]
    seen: set[tuple[Placement, ...]] = set()
    found: list[tuple[Placement, ...]] = []
    while queue:
        combo = queue.pop()
        if combo in seen:
            continue
        seen.add(combo)
        found.append(combo)
        try:
            call = (aten_op, captured_args, captured_kwargs)
            runtime = _runtime_strategy(*call, combo, mesh, out_meta=out_meta)
        except _UntestableInput:
            continue
        if runtime is None:
            continue
        op_schema, op_strategy, _ = runtime
        n_args = len(op_schema.args_spec)
        if n_args > n:
            continue
        for spec in op_strategy.strategies:
            desired = _desired_input_placements(op_schema, spec)
            if desired is None:
                continue
            # Tensor kwargs are not in args_spec; they keep their placements.
            desired = desired + combo[n_args:]
            if (
                desired not in seen
                and not is_fully_replicated(desired)
                and all(
                    p in options for p, options in zip(desired, input_placement_options)
                )
            ):
                queue.append(desired)
    return found, False


def query_runtime_claims(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placements: tuple[Placement, ...],
    mesh: DeviceMesh,
    *,
    out_meta: Any = None,
) -> list[_RuntimeClaim]:
    """Rules DTensor applies to inputs whose placements are exactly ``input_placements``.

    Builds the runtime OpSchema and candidate strategies the way
    ShardingPropagator.propagate_op_sharding_non_cached does, and returns
    every candidate whose desired input placements equal the actual ones
    (the one the runtime selector picks from the full candidate set first).
    Raises _UntestableInput if the strategies cannot be computed (see
    _runtime_strategy).
    """
    call = (aten_op, captured_args, captured_kwargs)
    runtime = _runtime_strategy(*call, input_placements, mesh, out_meta=out_meta)
    if runtime is None:
        return []
    op_schema, op_strategy, out_tensor_meta = runtime
    propagator = DTensor._op_dispatcher.sharding_propagator
    selected = _select_min_cost_strategy(op_strategy, op_schema)
    args_specs = op_schema.args_spec
    claims: list[_RuntimeClaim] = []
    for spec in op_strategy.strategies:
        if spec.input_specs is None and not isinstance(spec.output_specs, DTensorSpec):
            continue
        desired = [
            spec.output_spec if spec.input_specs is None else spec.input_specs[i]
            for i in range(len(args_specs))
        ]
        if any(d.placements != a.placements for d, a in zip(desired, args_specs)):
            continue
        output_specs = spec.output_specs
        if isinstance(output_specs, DTensorSpec):
            output_specs = (output_specs,)
        if not isinstance(output_specs, tuple) or None in output_specs:
            # None marks an output whose placement is undefined for this
            # strategy (e.g. indices under P(max) reduction).
            continue
        outputs = tuple(o.placements[0] for o in output_specs if o is not None)
        is_selected = spec is selected
        expected = [
            d.shallow_copy_with_tensor_meta(a.tensor_meta)
            for d, a in zip(desired, args_specs)
        ]
        try:
            rewritten = propagator._rewrite_args_for_local(
                op_schema, None, spec, expected, out_tensor_meta
            )
        except Exception as e:
            error = f"local arg rewrite raised {type(e).__name__}: {e}"
            claims.append(
                _RuntimeClaim(outputs, aten_op, (), {}, is_selected, error=error)
            )
            continue
        local_schema = rewritten or op_schema
        claims.append(
            _RuntimeClaim(
                outputs,
                local_schema.op,
                local_schema.args_schema,
                local_schema.kwargs_schema,
                is_selected,
                rewritten is not None,
            )
        )
    claims.sort(key=lambda c: not c.selected)
    return claims


def validate_aten_combination(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    combination: PlacementCombination,
    world_size: int,
    mesh: DeviceMesh,
    variant: _Variant | None = None,
    claim: _RuntimeClaim | None = None,
) -> tuple[bool | None, str]:
    """Validate a placement combination on one input variant of a captured aten call.

    Builds the per-rank inputs (see _materialize_inputs), computes the
    reference by running the op on the global inputs, runs the op on the
    per-rank inputs (as DTensor would for ``claim``: rewritten op and args),
    redistributes the claimed output placements to Replicate and compares.

    Returns (True, ""), (False, error_msg), or (None, reason) if the variant
    cannot be built, the reference op rejects it, or a mismatch is unresolved
    (see _validate_variant).
    """
    valid, msg, _ = _validate_variant(
        aten_op,
        captured_args,
        captured_kwargs,
        combination,
        world_size,
        mesh,
        variant or _Variant("finite"),
        claim,
    )
    return valid, msg


def _mixed_nan(pieces: list[torch.Tensor]) -> bool:
    """True if some element is NaN on some ranks but not on all of them.

    A MAX/MIN reduction of such pieces depends on the backend: LocalTensor
    propagates the NaN, while NCCL drops it and Gloo keeps it only on rank 0.
    """
    if not (pieces[0].is_floating_point() or pieces[0].is_complex()):
        return False
    if any(p.shape != pieces[0].shape for p in pieces):
        return False
    nan = torch.stack([p.isnan() for p in pieces])
    return bool((nan.any(0) & ~nan.all(0)).any())


def _is_narrow_float(dtype: torch.dtype) -> bool:
    floating = dtype.is_floating_point or dtype.is_complex
    return floating and torch.finfo(dtype).bits < 64


def _to_64bit(a: Any) -> Any:
    """Floating tensors and float dtypes (e.g. a dtype= argument) upcast to 64-bit; anything else unchanged."""
    if isinstance(a, torch.Tensor):
        return a.to(_to_64bit(a.dtype)) if _is_narrow_float(a.dtype) else a
    if isinstance(a, torch.dtype) and _is_narrow_float(a):
        return torch.complex128 if a.is_complex else torch.float64
    return a


def _run_variant(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    input_placements: tuple[Placement, ...],
    world_size: int,
    variant: _Variant,
    local_call: tuple[OpOverload, tuple[Any, ...], dict[str, Any]],
    widen: bool = False,
) -> tuple[str, Any]:
    """Reference and per-rank outputs of one input variant, independent of the output placements.

    widen: run on the same pieces upcast (exactly) to 64-bit floats, with
    float dtype arguments upcast too.
    Returns ("untestable", reason), ("error", message) if the per-rank call
    raises, or ("ok", (reference, local_output, mixed NaN in a P(min)/P(max)
    input)).
    """
    tensors = extract_tensors_from_args(captured_args, captured_kwargs)
    constrained = _constrained_operands(aten_op, captured_args, captured_kwargs)
    try:
        global_inputs, rank_inputs = _materialize_inputs(
            tensors, input_placements, world_size, variant, constrained
        )
    except _UntestableInput as e:
        return "untestable", str(e)
    if widen:
        try:
            rank_inputs = [[_to_64bit(t) for t in pieces] for pieces in rank_inputs]
        except Exception as e:
            return "untestable", f"64-bit conversion raised {type(e).__name__}"
        global_inputs = [
            _local_reduce(p.reduce_op, pieces)
            if isinstance(p, Partial)
            else _to_64bit(g)
            for g, p, pieces in zip(global_inputs, input_placements, rank_inputs)
        ]
        wide = pytree.tree_map(_to_64bit, (captured_args, captured_kwargs, local_call))
        captured_args, captured_kwargs, local_call = wide

    def _substitute(tree: Any, values: list[Any], types: tuple[type, ...]) -> Any:
        it = iter(values)
        return pytree.tree_map(lambda a: next(it) if isinstance(a, types) else a, tree)

    ref_args, ref_kwargs = _substitute(
        (captured_args, captured_kwargs), global_inputs, (torch.Tensor,)
    )
    try:
        reference = _check_ground_truth(aten_op(*ref_args, **ref_kwargs))
    except Exception as e:
        return "untestable", f"reference raised {type(e).__name__}"
    if reference is None:
        return "untestable", "reference has no tensor output"
    nan_inputs = any(
        isinstance(p, Partial) and p.reduce_op in ("min", "max") and _mixed_nan(pieces)
        for p, pieces in zip(input_placements, rank_inputs)
    )
    local_op, local_args, local_kwargs = local_call
    try:
        with LocalTensorMode(frozenset(range(world_size))):
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            local_tensors = [LocalTensor(dict(enumerate(p))) for p in rank_inputs]
            run_args, run_kwargs = _substitute(
                (local_args, local_kwargs), local_tensors, (torch.Tensor, DTensorSpec)
            )
            local_output = local_op(*run_args, **run_kwargs)
    except Exception as e:
        return "error", f"Exception: {type(e).__name__}: {e}"
    return "ok", (reference, local_output, nan_inputs)


def _validate_variant(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    combination: PlacementCombination,
    world_size: int,
    mesh: DeviceMesh,
    variant: _Variant,
    claim: _RuntimeClaim | None,
    cache: dict[Any, tuple[str, Any]] | None = None,
    *,
    recheck: bool = True,
) -> tuple[bool | None, str, bool]:
    """validate_aten_combination, also returning, for a failure, whether NaN met numbers in a P(min)/P(max) input or output.

    cache keeps _run_variant results across the output placements of one
    input placement combination. With recheck, a value mismatch when an
    input is a float of 32 bits or fewer is re-run on the same pieces upcast
    exactly to 64-bit (see _run_variant): if it persists, the rule is wrong;
    if it vanishes (e.g. rounding of large cancelling P(sum) pieces, or
    overflow), or the re-run still computes in narrow floats, it is
    unresolved (an untestable reason with the _UNRESOLVED prefix), since a
    branch flip can vanish there too. If the re-run cannot be done, the
    mismatch stands; so it does for bitcasts (view.dtype), which do not round
    and whose dtype argument must not be widened.
    """
    if claim is None or not claim.rewritten:
        local_call = (aten_op, captured_args, captured_kwargs)
    else:
        local_call = (claim.local_op, claim.local_args, claim.local_kwargs)

    def _run(widen: bool) -> tuple[str, Any]:
        key = (combination[0], variant, widen, local_call[0], *map(id, local_call[1:]))
        if cache is not None and key in cache:
            return cache[key]
        result = _run_variant(
            aten_op,
            captured_args,
            captured_kwargs,
            combination[0],
            world_size,
            variant,
            local_call,
            widen,
        )
        if cache is not None:
            cache[key] = result
        return result

    def _compare(reference: Any, local_output: Any) -> tuple[bool, str]:
        try:
            with LocalTensorMode(frozenset(range(world_size))):
                return _compare_outputs(
                    local_output, reference, combination[1], mesh, world_size
                )
        except Exception as e:
            return False, f"Exception: {type(e).__name__}: {e}"

    status, payload = _run(False)
    if status == "untestable":
        return None, payload, False
    if status == "error":
        return False, payload, False
    reference, local_output, nan_inputs = payload
    valid, msg = _compare(reference, local_output)
    if valid:
        return True, "", False
    outs = local_output if isinstance(local_output, (list, tuple)) else [local_output]
    mixed_nan = nan_inputs or any(
        isinstance(p, Partial)
        and p.reduce_op in ("min", "max")
        and isinstance(o, LocalTensor)
        and _mixed_nan([o._local_tensors[r] for r in range(world_size)])
        for p, o in zip(combination[1], outs)
    )
    tensors = extract_tensors_from_args(captured_args, captured_kwargs)
    narrow_inputs = any(_is_narrow_float(t.dtype) for _, t in tensors)
    bitcast = aten_op in (torch.ops.aten.view.dtype, torch.ops.aten.view_copy.dtype)
    if bitcast or not (recheck and narrow_inputs and msg.startswith(_VALUE_MISMATCH)):
        return False, msg, mixed_nan
    wide_status, wide_payload = _run(True)
    if wide_status != "ok":
        return False, f"{msg} (64-bit re-run: {wide_payload})", mixed_nan
    wide_reference, wide_output, _ = wide_payload
    wide_refs = wide_reference if isinstance(wide_reference, list) else [wide_reference]
    if any(_is_narrow_float(r.dtype) for r in wide_refs):
        return None, f"{_UNRESOLVED}64-bit re-run computes in narrow floats", mixed_nan
    wide_valid, wide_msg = _compare(wide_reference, wide_output)
    if not wide_valid:
        return False, f"{msg} (64-bit: {wide_msg})", mixed_nan
    return None, f"{_UNRESOLVED}mismatch vanishes in 64-bit", mixed_nan


def _audit_claim(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    combination: PlacementCombination,
    world_size: int,
    mesh: DeviceMesh,
    claim: _RuntimeClaim | None,
    options: ValidationOptions,
    float_specials: bool = True,
    cache: dict[Any, tuple[str, Any]] | None = None,
    *,
    recheck: bool = True,
) -> tuple[bool | None, str, str]:
    """Check a combination on every applicable input variant.

    Returns (valid, trigger, message): valid is None if no variant could be
    built or one was unresolved (see _validate_variant) and none failed;
    trigger names the first failing variant. recheck=False (for the
    missing-rule search, which only acts on True) skips the 64-bit re-run
    of mismatches. A failure where NaN met numbers
    in the P(min)/P(max) pieces of an input or output is labelled
    "nan-partial": the result follows LocalTensor's NaN-propagating
    reduction, while real backends reduce such pieces differently.
    """
    reason = ""
    unresolved = ""
    tested = False
    for variant in _value_variants(combination[0], options, float_specials):
        is_valid, msg, mixed_nan = _validate_variant(
            aten_op,
            captured_args,
            captured_kwargs,
            combination,
            world_size,
            mesh,
            variant,
            claim,
            cache,
            recheck=recheck,
        )
        if is_valid is None:
            if msg.startswith(_UNRESOLVED):
                unresolved = unresolved or msg
            else:
                reason = reason or msg
            continue
        tested = True
        if not is_valid:
            return False, "nan-partial" if mixed_nan else variant.name, msg
    if unresolved or not tested:
        return None, "", unresolved or reason
    return True, "", ""


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
    untestable: set[ComboKey] | None = None,
    failures: dict[ComboKey, tuple[str, str]] | None = None,
    world_size: int = 0,
    dtype: torch.dtype | None = None,
) -> None:
    """Compare ground truth valid rules against DTensor claimed rules, updating stats.

    failures maps an incorrect rule to the (trigger, message) of the input
    variant that exposed it.
    """
    if not dtensor_rules:
        return
    if untestable is None:
        untestable = set()
    if failures is None:
        failures = {}

    op_str = str(aten_op)
    for combo_key in ground_truth_valid:
        if combo_key in dtensor_rules:
            stats.true_positives += 1
            stats.true_positives_by_op[op_str] = (
                stats.true_positives_by_op.get(op_str, 0) + 1
            )
        elif combo_key not in untestable:
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
                    world_size=world_size,
                    dtype=dtype,
                )
            )

    for combo_key in dtensor_rules:
        if combo_key not in ground_truth_valid and combo_key not in untestable:
            trigger, msg = failures.get(combo_key, ("", ""))
            stats.false_positives.append(
                Discrepancy(
                    input_placements=combo_key[0],
                    output_placements=combo_key[1],
                    sample_idx=sample_idx,
                    input_shapes=input_shapes,
                    discrepancy_type="false_positive",
                    error_msg=msg,
                    scalar_args=scalar_args,
                    scalar_kwargs=scalar_kwargs,
                    aten_op=aten_op,
                    variant=variant,
                    sample=sample,
                    trigger=trigger,
                    world_size=world_size,
                    dtype=dtype,
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
            triggers = sorted({d.trigger for d in discs if d.trigger})
            sizes = sorted({d.world_size for d in discs if d.world_size})
            dtypes = sorted(
                {str(d.dtype).removeprefix("torch.") for d in discs if d.dtype}
            )
            details = []
            if triggers:
                details.append("trigger: " + ", ".join(triggers))
            if sizes:
                details.append("world size: " + ",".join(str(s) for s in sizes))
            if dtypes:
                details.append(",".join(dtypes))
            suffix = f"  [{'; '.join(details)}]" if details else ""
            print(f"    {inp_str} -> {out_str}{suffix}")
            if show_repro:
                limit = len(discs) if show_repro < 0 else show_repro
                for d in discs[:limit]:
                    if d.sample is not None:
                        print(
                            f"      OpInfo sample: {_format_sample_repro(d.sample, d.aten_op)}"
                        )
                    if d.trigger or d.error_msg:
                        scalars = [
                            *d.scalar_args,
                            *(f"{k}={v}" for k, v in d.scalar_kwargs.items()),
                        ]
                        print(
                            f"      Variant: {d.trigger} (world size {d.world_size}, "
                            f"{d.dtype}, shapes {[list(s) for s in d.input_shapes]}, "
                            f"scalars {scalars}): {d.error_msg}"
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

    if stats.untestable_claims:
        reasons = ", ".join(
            f"{r} ({n})" for r, n in sorted(stats.untestable_claims.items())
        )
        print(f"  Untestable claimed rules: {reasons}")
    if stats.incomplete_discovery:
        print(
            f"  Claims found by search, may be incomplete: "
            f"{stats.incomplete_discovery} samples"
        )

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

    Returns the ground truth tensor(s), or None if the result has no
    non-empty tensor output. Degenerate outputs (all zero / all NaN) are kept:
    they still exercise claimed rules (e.g. sign(NaN) == 0); see _is_degenerate.
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
    return gt


def _is_degenerate(gt: torch.Tensor) -> bool:
    """All-zero or all-NaN outputs make many wrong rules pass; skip them when searching for missing rules."""
    if (gt == 0).all():
        return True
    return bool((gt.is_floating_point() or gt.is_complex()) and gt.isnan().all())


def _shards_nonempty(
    tensors: list[tuple[str, torch.Tensor]],
    input_placements: tuple[Placement, ...],
    world_size: int,
) -> bool:
    """True if no Shard input leaves a rank with an empty shard.

    torch.chunk can leave trailing ranks empty even when the dim is at least
    world_size long (4 elements on 3 ranks split as 2, 2, 0).
    """
    last = world_size - 1
    return not any(
        isinstance(p, Shard)
        and Shard.local_shard_size_and_offset(t.size(p.dim), world_size, last)[0] == 0
        for (_, t), p in zip(tensors, input_placements)
    )


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
    options: ValidationOptions | None = None,
    trigger_prefix: str = "",
) -> tuple[int, int]:
    """Validate a single aten op with captured args against ground truth.

    For every non-replicated input placement combination, each output
    placement DTensor produces for it (query_runtime_claims) is audited on all
    input variants. Unless incorrect_only, the remaining output placements are
    searched for missing rules, skipping combinations with empty shards.
    Shared by default (1:1) and allow_composite modes in compare_operator.
    Returns (samples_counted, combinations_counted).
    """
    options = options or ValidationOptions()
    tensors = extract_tensors_from_args(captured_args, captured_kwargs)
    if not tensors:
        return 0, 0
    if any(0 in t.shape for _, t in tensors):
        return 0, 0
    out = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
    # Converting NaN/inf to an integer dtype is undefined behavior.
    float_specials = aten_op not in _DTYPE_CONVERSION_OPS or (
        out.is_floating_point() or out.is_complex() or out.dtype == torch.bool
    )

    input_shapes = tuple(tuple(t.shape) for _, t in tensors)
    gt_list = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    output_shapes = tuple(tuple(gt.shape) for gt in gt_list)
    n_outputs = len(gt_list)

    scalar_args = tuple(a for a in captured_args if not isinstance(a, torch.Tensor))
    scalar_kwargs = {
        k: v for k, v in captured_kwargs.items() if not isinstance(v, torch.Tensor)
    }
    # Non-finite scalars only meet combinations without P(sum)/P(avg) inputs;
    # see _materialize_inputs for why.
    nonfinite_scalars = any(
        isinstance(v, float) and not math.isfinite(v)
        for v in pytree.tree_leaves((scalar_args, scalar_kwargs))
    )
    constrained = _constrained_operands(aten_op, captured_args, captured_kwargs)
    input_placement_options = [
        get_1d_input_placements_for_tensor(t, include_partial=not fixed)
        for (_, t), fixed in zip(tensors, constrained)
    ]
    output_placement_options = get_1d_output_placements_for_tensor(gt_list[0])
    search_missing = not incorrect_only and not _is_degenerate(gt_list[0])

    # Claims are queried under LocalTensorMode, like DTensor dispatch: shape
    # arguments are then rewritten per rank (uneven and empty shards differ).
    local_mode = LocalTensorMode(frozenset(range(world_size)))
    with local_mode:
        mesh = init_device_mesh(tensors[0][1].device.type, (world_size,))
        out_meta = _output_meta(aten_op, captured_args, captured_kwargs, mesh)

    dtensor_rules: set[ComboKey] = set()
    ground_truth_valid: set[ComboKey] = set()
    untestable: set[ComboKey] = set()
    failures: dict[ComboKey, tuple[str, str]] = {}
    total_combinations = 0
    if incorrect_only:
        with local_mode:
            input_combinations, complete = claimed_input_combinations(
                aten_op,
                captured_args,
                captured_kwargs,
                input_placement_options,
                mesh,
                out_meta=out_meta,
            )
        stats.incomplete_discovery += not complete
    else:
        input_combinations = list(itertools.product(*input_placement_options))
    for input_placements in input_combinations:
        if is_fully_replicated(input_placements):
            continue
        if nonfinite_scalars and any(
            isinstance(p, Partial) and p.reduce_op in ("sum", "avg")
            for p in input_placements
        ):
            continue
        input_key = tuple(str(p) for p in input_placements)
        claimed: set[tuple[str, ...]] = set()
        cache: dict[Any, tuple[str, Any]] = {}
        nonempty = _shards_nonempty(tensors, input_placements, world_size)
        try:
            with local_mode:
                claims = query_runtime_claims(
                    aten_op,
                    captured_args,
                    captured_kwargs,
                    input_placements,
                    mesh,
                    out_meta=out_meta,
                )
        except _UntestableInput as e:
            reason = f"query: {e}"
            stats.untestable_claims[reason] = stats.untestable_claims.get(reason, 0) + 1
            claims = []
        for claim in claims:
            outputs = claim.output_placements
            if len(outputs) == 1:
                outputs = outputs * n_outputs
            output_key = tuple(str(p) for p in outputs)
            if len(outputs) != n_outputs or output_key in claimed:
                continue
            claimed.add(output_key)
            combo_key: ComboKey = (input_key, output_key)
            dtensor_rules.add(combo_key)
            total_combinations += 1
            if claim.error and not claim.selected:
                is_valid, trigger, msg = None, "", "local arg rewrite raised"
            elif claim.error:
                is_valid, trigger, msg = False, "rewrite", claim.error
            else:
                is_valid, trigger, msg = _audit_claim(
                    aten_op,
                    captured_args,
                    captured_kwargs,
                    (input_placements, outputs),
                    world_size,
                    mesh,
                    claim,
                    options,
                    float_specials,
                    cache,
                )
            if is_valid is None:
                untestable.add(combo_key)
                stats.untestable_claims[msg] = stats.untestable_claims.get(msg, 0) + 1
            elif is_valid:
                ground_truth_valid.add(combo_key)
            else:
                if not nonempty:
                    trigger = f"empty-shard {trigger}"
                if not claim.selected:
                    # Exact-input candidate that the runtime selector does not
                    # pick for these inputs on a 1-D mesh.
                    trigger = f"{trigger} (alternative)"
                failures[combo_key] = (trigger_prefix + trigger, msg)

        if not search_missing or not nonempty:
            continue
        for output_placement in output_placement_options:
            outputs = tuple(output_placement for _ in range(n_outputs))
            output_key = tuple(str(p) for p in outputs)
            if output_key in claimed or any(
                is_trivial_shard(p, s) for p, s in zip(outputs, output_shapes)
            ):
                continue
            total_combinations += 1
            is_valid, _, _ = _audit_claim(
                aten_op,
                captured_args,
                captured_kwargs,
                (input_placements, outputs),
                world_size,
                mesh,
                None,
                options,
                float_specials,
                cache,
                recheck=False,
            )
            if is_valid:
                ground_truth_valid.add((input_key, output_key))

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
        untestable,
        failures,
        world_size,
        tensors[0][1].dtype,
    )

    if verbose:
        print(f"      Sample {sample_idx} [{aten_op}]: shapes={input_shapes}")
        print(f"        Ground truth valid: {len(ground_truth_valid)}")
        print(f"        DTensor rules: {len(dtensor_rules)}")

    return 1, total_combinations


def _validate_captured_call(
    func: OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ground_truth: torch.Tensor | list[torch.Tensor],
    world_size: int,
    incorrect_only: bool,
    verbose: bool,
    sample_idx: int,
    variant: str,
    stats: ComparisonStats,
    sample: SampleInput | None,
    options: ValidationOptions,
) -> tuple[int, int]:
    """Validate a captured aten call, then audit its claimed rules on each scalar-argument variant.

    Scalar variants add incorrect and untestable claims to ``stats`` but not
    correct ones, so a rule is counted once per sample. Returns
    (samples_counted, combinations_counted).
    """
    n_samples, n_combos = _validate_aten_op_for_sample(
        func,
        args,
        kwargs,
        ground_truth,
        world_size,
        incorrect_only,
        verbose,
        sample_idx,
        variant,
        stats,
        sample,
        options,
    )
    for label, fuzzed_args, fuzzed_kwargs in _scalar_variants(
        func, args, kwargs, options
    ):
        ref_args, ref_kwargs = pytree.tree_map(
            lambda a: a.clone() if isinstance(a, torch.Tensor) else a,
            (fuzzed_args, fuzzed_kwargs),
        )
        try:
            fuzzed_gt = _check_ground_truth(func(*ref_args, **ref_kwargs))
        except Exception:
            continue
        if fuzzed_gt is None:
            continue
        variant_stats = ComparisonStats()
        _, n = _validate_aten_op_for_sample(
            func,
            fuzzed_args,
            fuzzed_kwargs,
            fuzzed_gt,
            world_size,
            True,
            verbose,
            sample_idx,
            variant,
            variant_stats,
            sample,
            options,
            trigger_prefix=f"{label} ",
        )
        n_combos += n
        stats.false_positives += variant_stats.false_positives
        stats.incomplete_discovery += variant_stats.incomplete_discovery
        for reason, count in variant_stats.untestable_claims.items():
            stats.untestable_claims[reason] = (
                stats.untestable_claims.get(reason, 0) + count
            )
    return n_samples, n_combos


def _scalar_variants(
    aten_op: OpOverload,
    captured_args: tuple[Any, ...],
    captured_kwargs: dict[str, Any],
    options: ValidationOptions,
) -> list[tuple[str, tuple[Any, ...], dict[str, Any]]]:
    """Copies of a captured call with its Scalar/float arguments replaced.

    Covers arguments typed Scalar, Scalar?, float and float? that hold a
    number or were omitted. Directed values come first, with
    neighbouring arguments taking different values so that conditions like
    threshold's value > threshold are hit; then seeded random draws. Scalar
    arguments take Python ints when every tensor operand is integral and
    otherwise alternate between floats and ints. Returns (label, args, kwargs)
    triples.
    """
    if options.scalar_variants <= 0:
        return []
    schema_args = aten_op._schema.arguments
    slots: list[tuple[str, int | None, bool]] = []
    for i, arg in enumerate(schema_args):
        arg_type = arg.type
        optional = isinstance(arg_type, torch.OptionalType)
        if optional:
            arg_type = arg_type.getElementType()
        if not isinstance(arg_type, (torch.NumberType, torch.FloatType)):
            continue
        if any(s in arg.name for s in CONSTRAINED_SCALAR_ARGS):
            continue
        if arg.kwarg_only:
            value, position = captured_kwargs.get(arg.name), None
        else:
            value = captured_args[i] if i < len(captured_args) else None
            position = i
        if isinstance(value, bool) or not isinstance(value, (int, float, type(None))):
            continue
        if value is None and not (optional or arg.has_default_value()):
            continue
        slots.append((arg.name, position, isinstance(arg_type, torch.NumberType)))
    if not slots:
        return []

    int_data = all(
        not (t.is_floating_point() or t.is_complex())
        for _, t in extract_tensors_from_args(captured_args, captured_kwargs)
    )
    choices: list[list[float | int]] = []
    for j in range(len(SCALAR_FLOAT_VALUES)):
        row = []
        for k, (_, _, is_scalar) in enumerate(slots):
            use_ints = is_scalar and (int_data or j % 2 == 1)
            pool = SCALAR_INT_VALUES if use_ints else SCALAR_FLOAT_VALUES
            row.append(pool[(j + k) % len(pool)])
        choices.append(row)
    rng = torch.Generator().manual_seed(options.seed)
    while len(choices) < options.scalar_variants:
        row: list[float | int] = []
        for _, _, is_scalar in slots:
            pool = list(SCALAR_FLOAT_VALUES) + (
                list(SCALAR_INT_VALUES) if is_scalar else []
            )
            if options.special_values:
                pool += list(SPECIAL_VALUES)
            row.append(pool[int(torch.randint(len(pool), (1,), generator=rng))])
        choices.append(row)

    variants = []
    for row in choices[: options.scalar_variants]:
        args, kwargs = list(captured_args), dict(captured_kwargs)
        for (name, position, _), value in zip(slots, row):
            if position is None:
                kwargs[name] = value
                continue
            # The dispatcher passes positional args positionally, so omitted
            # ones are filled in with their defaults rather than bound by name.
            while len(args) <= position:
                args.append(schema_args[len(args)].default_value)
            args[position] = value
        label = ",".join(f"{name}={value}" for (name, _, _), value in zip(slots, row))
        variants.append((f"scalar:{label}", tuple(args), kwargs))
    return variants


@contextmanager
def _opinfo_sizes(profile: str) -> Iterator[None]:
    """Temporarily set the OpInfo size constants (S, M, L, XS) to a SIZE_PROFILES entry.

    Patches every loaded torch.testing._internal module that holds them (the
    OpInfo definitions import them by value); sizes already bound as default
    arguments are not reached. These are process-wide globals, so only the
    single-threaded CLI uses this.
    """
    sizes = dict(zip(("S", "M", "L", "XS"), SIZE_PROFILES[profile]))
    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if module is None or not name.startswith("torch.testing._internal"):
                continue
            for key, size in sizes.items():
                if type(getattr(module, key, None)) is int:
                    stack.enter_context(unittest.mock.patch.object(module, key, size))
        yield


def compare_operator(
    op_name: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    world_size: int = 2,
    max_samples: int | None = None,
    verbose: bool = False,
    incorrect_only: bool = False,
    allow_composite: bool = False,
    options: ValidationOptions | None = None,
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
        options: Opt-in special-value and scalar-argument variants.
    """
    if op_name in SKIP_OPS:
        return ComparisonStats()

    options = options or ValidationOptions()
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

    def _validate(func, args, kwargs, gt, sample_idx, variant, sample) -> None:
        nonlocal total_samples, total_combinations
        n_samples, n_combos = _validate_captured_call(
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
            options,
        )
        total_samples += n_samples
        total_combinations += n_combos

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

            # Count supported aten ops in the capture; random ops cannot be
            # compared against a reference run.
            supported_ops = [
                (func, args, kwargs, result)
                for func, args, kwargs, result in capture.all_ops
                if _has_dtensor_support(func)
            ]
            if any(
                torch.Tag.nondeterministic_seeded in f.tags for f, *_ in supported_ops
            ):
                skip_reasons["nondeterministic op"] += 1
                continue
            num_supported = len(supported_ops)

            if allow_composite:
                # Validate each supported aten op individually
                if num_supported == 0:
                    skip_reasons["no supported aten ops"] += 1
                    continue

                for func, args, kwargs, result in supported_ops:
                    gt = _check_ground_truth(result)
                    if gt is None:
                        skip_reasons["non-tensor/empty aten output"] += 1
                        continue
                    _validate(func, args, kwargs, gt, sample_idx, variant, sample)
            else:
                # Default: only validate samples with a single supported aten op
                if num_supported != 1:
                    skip_reasons["non-1:1 aten mapping"] += 1
                    continue

                func, args, kwargs, result = supported_ops[0]
                gt = _check_ground_truth(result)
                if gt is None:
                    skip_reasons["non-tensor/empty aten output"] += 1
                    continue
                _validate(func, args, kwargs, gt, sample_idx, variant, sample)

    stats.total_samples = total_samples
    stats.total_combinations = total_combinations
    stats.skip_reasons = dict(skip_reasons)

    # In allow_composite mode, check DTensor support after processing
    if allow_composite and total_samples == 0 and not skip_reasons:
        stats.no_dtensor_support = True

    return stats


def _merge_stats(into: ComparisonStats, stats: ComparisonStats) -> None:
    """Accumulate one configuration's stats into an operator's combined stats."""
    into.true_positives += stats.true_positives
    into.true_negatives += stats.true_negatives
    into.false_positives += stats.false_positives
    into.false_negatives += stats.false_negatives
    into.total_samples += stats.total_samples
    into.total_combinations += stats.total_combinations
    into.no_dtensor_support = into.no_dtensor_support and stats.no_dtensor_support
    into.incomplete_discovery += stats.incomplete_discovery
    for src, dst in (
        (stats.skip_reasons, into.skip_reasons),
        (stats.true_positives_by_op, into.true_positives_by_op),
        (stats.untestable_claims, into.untestable_claims),
    ):
        for k, v in src.items():
            dst[k] = dst.get(k, 0) + v


def get_registered_op_names() -> list[str]:
    """Get all op names that have DTensor sharding rules and also have OpInfo.

    Returns OpInfo names (which may differ from aten base names, e.g.,
    "nn.functional.relu" instead of "relu"), matched by name, by aten_name or
    by the single supported aten op their first float32 CPU sample dispatches
    to. OpInfos that need other dtypes or decompose into several supported
    ops are only covered by name.
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

    # Match by dispatch: OpInfos whose first non-empty sample runs exactly one
    # supported aten op under a different name (e.g. "byte" -> aten._to_copy,
    # "nn.functional.threshold" -> aten.threshold).
    for opinfo in op_db:
        if opinfo.name in result or opinfo.name in SKIP_OPS:
            continue
        try:
            for sample in opinfo.sample_inputs("cpu", torch.float32):
                tensors = extract_tensors_from_sample(sample)
                if not tensors or any(0 in t.shape for _, t in tensors):
                    continue
                capture = get_aten_op_for_sample(opinfo.op, sample, opinfo.name)
                supported = [f for f, *_ in capture.all_ops if _has_dtensor_support(f)]
                if len(supported) == 1 and (
                    torch.Tag.nondeterministic_seeded not in supported[0].tags
                ):
                    result.add(opinfo.name)
                break
        except Exception:
            continue

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
    dtype_map = {
        name: getattr(torch, name)
        for name in (
            "float32",
            "float64",
            "float16",
            "bfloat16",
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
            "bool",
        )
    }

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
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help=f"Comma-separated dtypes: {', '.join(dtype_map)}",
    )
    parser.add_argument(
        "--world-size",
        default=None,
        help="Comma-separated simulated world sizes (default 2)",
    )
    parser.add_argument(
        "--size-profile",
        default=None,
        help="Comma-separated OpInfo size profiles (default even): "
        + ", ".join(f"{k}={v}" for k, v in SIZE_PROFILES.items())
        + " as (S, M, L, XS)",
    )
    parser.add_argument(
        "--special-values",
        action="store_true",
        help="Also check rules on NaN/+-inf/-0.0 inputs, adversarial Partial pieces "
        "and integer extremes/wraparound",
    )
    parser.add_argument(
        "--scalar-variants",
        type=int,
        default=None,
        metavar="N",
        help="Audit claimed rules on N variants of each sample's Scalar/float args",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for generated input variants"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Preset: --special-values --scalar-variants 6 --world-size 2,3,4 "
        "--size-profile even,base,small (explicit --scalar-variants, --world-size "
        "and --size-profile take precedence)",
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

    world_size_arg = args.world_size or ("2,3,4" if args.stress else "2")
    profile_arg = args.size_profile or ("even,base,small" if args.stress else "even")
    scalar_variants = args.scalar_variants
    if scalar_variants is None:
        scalar_variants = 6 if args.stress else 0
    dtype_names = [d.strip() for d in args.dtype.split(",")]
    if not set(dtype_names) <= set(dtype_map):
        parser.error(f"--dtype: expected some of {list(dtype_map)}")
    dtypes = [dtype_map[d] for d in dtype_names]
    try:
        world_sizes = [int(w) for w in world_size_arg.split(",")]
    except ValueError:
        parser.error(
            f"--world-size: expected comma-separated integers, got {world_size_arg!r}"
        )
    if min(world_sizes) < 2:
        parser.error("--world-size: world sizes must be at least 2")
    profiles = [p.strip() for p in profile_arg.split(",")]
    if not set(profiles) <= set(SIZE_PROFILES):
        parser.error(f"--size-profile: expected some of {list(SIZE_PROFILES)}")
    if scalar_variants < 0:
        parser.error("--scalar-variants: must be non-negative")
    options = ValidationOptions(
        special_values=args.special_values or args.stress,
        scalar_variants=scalar_variants,
        seed=args.seed,
    )

    dist.init_process_group("fake", rank=0, world_size=max(world_sizes))

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
        print(
            f"Device: {args.device}, Dtype: {', '.join(str(d) for d in dtypes)}, "
            f"World size: {','.join(str(w) for w in world_sizes)}, "
            f"Size profile: {','.join(profiles)}"
        )
        if options.special_values or options.scalar_variants:
            print(
                f"Variants: special values={options.special_values}, "
                f"scalar variants={options.scalar_variants}, seed={options.seed}"
            )

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
                stats = ComparisonStats(no_dtensor_support=True)
                for world_size, dtype, profile in itertools.product(
                    world_sizes, dtypes, profiles
                ):
                    with _opinfo_sizes(profile):
                        config_stats = compare_operator(
                            op_name,
                            args.device,
                            dtype,
                            world_size,
                            args.max_samples,
                            verbose=True,
                            incorrect_only=args.incorrect_only,
                            allow_composite=args.allow_composite,
                            options=options,
                        )
                    _merge_stats(stats, config_stats)
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
                "un": 10,
                "time": 6,
            }
            col_widths["op"] = max(col_widths["op"], 2)  # min width

            header = (
                f"{'Op':<{col_widths['op']}}  "
                f"{'Correct':>{col_widths['tp']}}  "
                f"{'Incorrect':>{col_widths['fp']}}  "
                f"{'Missing':>{col_widths['fn']}}  "
                f"{'Untestable':>{col_widths['un']}}  "
                f"{'Time':>{col_widths['time']}}"
            )
            print(header)
            print("-" * len(header))

            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_un = 0
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
                n_un = sum(stats.untestable_claims.values())
                total_un += n_un
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
                    f"{n_un:>{col_widths['un']}}  "
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
                f"{total_un:>{col_widths['un']}}  "
                f"{total_time:>{col_widths['time']}.1f}s"
            )
    finally:
        dist.destroy_process_group()
