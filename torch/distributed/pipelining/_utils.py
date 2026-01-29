# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import fx


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Placement

logger = logging.getLogger(__name__)


def flatten_args_detach(args):
    """
    Flatten the args into a list form and detach the tensors from computational graph.
    """
    flat_detached_args = []

    def extract_tensor_args(a):
        nonlocal flat_detached_args
        if isinstance(a, torch.Tensor):
            val = a.detach().requires_grad_(a.requires_grad)
            flat_detached_args.append(val)
            return val
        else:
            flat_detached_args.append(a)
            return a

    new_args = fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return new_args, flat_detached_args


def flatten_args(args):
    """
    Flatten the args into a list form.
    """
    flat_args = []

    def extract_tensor_args(a):
        nonlocal flat_args
        flat_args.append(a)
        return a

    fx.node.map_aggregate(
        args,
        extract_tensor_args,
    )

    return flat_args


class PipeliningShapeError(RuntimeError):
    """Shape mismatch between configured and runtime values."""


class PipeliningDTensorError(RuntimeError):
    """DTensor metadata mismatch between configured and runtime values."""


@dataclass
class _DTensorMeta:
    """
    Metadata needed to reconstruct a DTensor from a local tensor.
    """

    # Global tensor properties
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype

    # DTensor distribution properties
    placements: tuple["Placement", ...]  # e.g., (Shard(0), Replicate())

    @staticmethod
    def from_dtensor(dtensor: "DTensor") -> "_DTensorMeta":
        """Extract metadata from a DTensor."""
        spec = dtensor._spec
        return _DTensorMeta(
            global_shape=spec.shape,
            global_stride=spec.stride,
            dtype=dtensor.dtype,
            placements=spec.placements,
        )


def validate_dtensor_metadata(
    desc: str,
    expected: _DTensorMeta,
    given_dtensor: "DTensor",
) -> None:
    """Validate DTensor metadata matches expected configuration."""
    from torch.distributed.tensor import DTensor

    if not isinstance(given_dtensor, DTensor):
        raise PipeliningDTensorError(
            f"{desc}: expected DTensor, got {type(given_dtensor)}"
        )

    given_meta = _DTensorMeta.from_dtensor(given_dtensor)

    if expected.global_shape != given_meta.global_shape:
        raise PipeliningDTensorError(
            f"{desc} has a global shape mismatch: "
            f"expected {expected.global_shape} actual {given_meta.global_shape}"
        )

    if expected.placements != given_meta.placements:
        raise PipeliningDTensorError(
            f"{desc} has a placements mismatch: "
            f"expected {expected.placements} actual {given_meta.placements}"
        )


def validate_tensor_metadata(desc, expected, given):
    from torch.distributed.tensor import DTensor

    # For DTensors, compare local shapes since that's what's communicated
    if isinstance(given, DTensor):
        given_local = given.to_local()
    else:
        given_local = given

    if not expected.shape == given_local.shape:
        raise PipeliningShapeError(
            f"{desc} has a shape mismatch: expected {expected.shape} actual {given_local.shape}"
        )
    if not expected.dtype == given_local.dtype:
        raise PipeliningShapeError(
            f"{desc} has a dtype mismatch: expected {expected.dtype} actual {given_local.dtype}"
        )
    if not expected.stride() == given_local.stride():
        raise PipeliningShapeError(
            f"{desc} has a stride mismatch: expected {expected.stride()} actual {given_local.stride()}"
        )


def validate_tensors_metadata(
    desc,
    expected_tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
    actual_tensors: list[torch.Tensor] | tuple[torch.Tensor, ...],
):
    if len(expected_tensors) != len(actual_tensors):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(actual_tensors)}) does not match expected number ({len(expected_tensors)})"
        )
    for i in range(len(expected_tensors)):
        validate_tensor_metadata(
            f"{desc}: value {i}", expected_tensors[i], actual_tensors[i]
        )


def generate_stage_to_rank_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, int]:
    """
    Compute the stage id to rank mapping for either a looped or V-style schedule.

    Most commonly num_stages == pp_size * 2, but this function can be used to
    compute the mapping for any number of stages per rank.
    """
    mapping = {}
    if style == "loop":
        for stage_index in range(num_stages):
            mapping[stage_index] = stage_index % pp_size
    elif style == "v":
        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages {num_stages} must be evenly divisible by pp_size {pp_size} for V schedules"
            )

        rank_index = 0
        for stage_index in range(num_stages):
            mapping[stage_index] = rank_index
            # dont change rank if we are on the border (to keep v shape)
            if (stage_index + 1) % pp_size == 0:
                continue
            if (stage_index // pp_size) % 2 == 0:
                rank_index += 1
            else:
                rank_index -= 1
    else:
        raise ValueError(f"Style {style} is not supported.")
    return mapping


def generate_rank_to_stage_mapping(
    pp_size: int, num_stages: int, style: str = "loop"
) -> dict[int, list[int]]:
    """
    Compute the rank to stage id mapping for either a looped or V-style schedule.

    This function inverts the stage_to_rank_mapping to get which stages are assigned to each rank.

    Returns a dictionary mapping rank -> list of stage indices assigned to that rank.
    """
    stage_to_rank = generate_stage_to_rank_mapping(pp_size, num_stages, style)

    # Invert the mapping: rank -> list of stages
    rank_to_stages: dict[int, list[int]] = {}
    for stage_id, rank in stage_to_rank.items():
        if rank not in rank_to_stages:
            rank_to_stages[rank] = []
        rank_to_stages[rank].append(stage_id)

    # Sort the stage lists for each rank to ensure consistent ordering
    for stages in rank_to_stages.values():
        stages.sort()

    return rank_to_stages


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool
