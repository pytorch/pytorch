import logging
from typing import List, Optional

import torch
import torch.distributed.distributed_c10d as c10d

"""
Moved eager kernel implementations to a separate file partly for readability and partly as it is currently
easier in dynamo to set tracing policy on a file-by-file level.

Do not put code in this file that Dynamo is expected to trace into, as dynamo may disallow this whole file.

DEBUG/TESTING HELPERS:

This module includes some helpers that are quite useful when debugging or testing functional collectives:

_tensor_needs_wait
_outstanding_wait_count
_wait_all

"""

logger = logging.getLogger(__name__)

"""
Kernel implementations (for eager runtime only) - should never be traced by torch.compile

These functions should all be bound to dispatcher ops.  During tracing, the op itself should be
captured in the graph and the backend should implement the op however it prefers.
"""


def _broadcast(input, src, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.broadcast(
        input,
        group_name,
    )


def _all_reduce(input, reduce_op, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce(
        input,
        reduce_op,
        group_name,
    )


def _all_reduce_coalesced(inputs, reduce_op, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_reduce_coalesced(
        inputs,
        reduce_op,
        group_name,
    )


def _all_gather_into_tensor(input, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor(
        input,
        group_size,
        group_name,
    )


def _all_gather_into_tensor_coalesced(input, tag, ranks, group_size):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        input,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor(
    input: torch.Tensor,
    reduce_op: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor(
        input,
        reduce_op,
        group_size,
        group_name,
    )


def _reduce_scatter_tensor_coalesced(
    inputs: List[torch.Tensor],
    reduce_op: str,
    tag: str,
    ranks: List[int],
    group_size: int,
):
    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        inputs,
        reduce_op,
        group_size,
        group_name,
    )


def _all_to_all_single(
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    tag: str,
    ranks: List[int],
    group_size: int,
):
    if output_split_sizes is None or input_split_sizes is None:
        assert output_split_sizes is None and input_split_sizes is None, (
            "output_split_sizes and input_split_sizes must either be "
            "specified together or both set to None"
        )
        output_split_sizes = [input.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes

    group_name = c10d._resolve_group_name_by_ranks_and_tag(ranks, tag)
    return torch.ops._c10d_functional.all_to_all_single(
        input,
        output_split_sizes,
        input_split_sizes,
        group_name,
    )


def _wait_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.ops._c10d_functional.wait_tensor(tensor)
