# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from dataclasses import dataclass
from typing import List, Tuple, Union

import torch
from torch import fx


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


def validate_tensor_metadata(desc, expected, given):
    if not expected.shape == given.shape:
        raise PipeliningShapeError(
            f"{desc} has a shape mismatch: expected {expected.shape} actual {given.shape}"
        )
    if not expected.dtype == given.dtype:
        raise PipeliningShapeError(
            f"{desc} has a dtype mismatch: expected {expected.dtype} actual {given.dtype}"
        )
    if not expected.stride() == given.stride():
        raise PipeliningShapeError(
            f"{desc} has a stride mismatch: expected {expected.stride()} actual {given.stride()}"
        )


def validate_tensors_metadata(
    desc,
    expected_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
    actual_tensors: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
):
    if len(expected_tensors) != len(actual_tensors):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(actual_tensors)}) does not match expected number ({len(expected_tensors)})"
        )
    for i in range(len(expected_tensors)):
        validate_tensor_metadata(
            f"{desc}: value {i}", expected_tensors[i], actual_tensors[i]
        )


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool
