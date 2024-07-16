# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

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


def validate_tensor_metadata(desc, expected: torch.Tensor, given: torch.Tensor):
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


def validate_arguments(
    desc,
    expected: Union[List[Any], Tuple[Any, ...]],
    actual: Union[List[Any], Tuple[Any, ...]],
):
    if len(expected) != len(actual):
        raise PipeliningShapeError(
            f"{desc}: Number of values ({len(expected)}) does not match expected number ({len(actual)})"
        )

    for i in range(len(expected)):
        expected_item = expected[i]
        actual_item = actual[i]
        # Check if both items are tensors and validate them
        if isinstance(expected_item, torch.Tensor) and isinstance(
            actual_item, torch.Tensor
        ):
            validate_tensor_metadata(f"{desc}: value {i}", expected_item, actual_item)
        # Add more type checks as needed for other data types
        elif isinstance(expected_item, (list, tuple)) and isinstance(
            actual_item, (list, tuple)
        ):
            if len(expected_item) != len(actual_item):
                raise PipeliningShapeError(
                    f"{desc}: value {i} list/tuple length mismatch: expected {len(expected_item)}, actual {len(actual_item)}"
                )
            # Recursively validate list/tuple elements
            validate_arguments(f"{desc}: value {i}", expected_item, actual_item)
        elif type(expected_item) != type(actual_item):
            raise PipeliningShapeError(
                f"{desc}: value {i} type mismatch: expected {type(expected_item)}, actual {type(actual_item)}"
            )


@dataclass
class PipeInfo:
    """
    Captures information for a pipeline (`Pipe` object).
    """

    graph: fx.Graph
    num_stages: int
    has_loss_and_backward: bool
