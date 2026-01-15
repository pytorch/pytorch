# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Example 2: Simulating Collective Operations
============================================

This module contains core patterns for collective operations.
The functions below are designed to be:
1. Included in the tutorial via literalinclude
2. Directly tested to ensure tutorial correctness

Each function returns a tuple of (actual, expected) for testing.
"""

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensor, LocalTensorMode


# [core_all_reduce]
def all_reduce_sum(process_group):
    """Simulate all_reduce with SUM across ranks.

    Returns: (result, expected)
    """
    tensors = {
        0: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        1: torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        2: torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
    }

    expected = sum(tensors.values())

    with LocalTensorMode(frozenset(tensors.keys())):
        lt = LocalTensor({k: v.clone() for k, v in tensors.items()})
        dist.all_reduce(lt, op=dist.ReduceOp.SUM, group=process_group)
        result = lt.reconcile()

    return result, expected


# [end_core_all_reduce]


# [core_broadcast]
def broadcast_from_rank(process_group, src_rank: int = 0):
    """Simulate broadcast from a source rank.

    Returns: (result, expected)
    """
    tensors = {
        0: torch.tensor([10.0, 20.0, 30.0]),
        1: torch.tensor([40.0, 50.0, 60.0]),
        2: torch.tensor([70.0, 80.0, 90.0]),
    }

    expected = tensors[src_rank].clone()

    with LocalTensorMode(frozenset(tensors.keys())):
        lt = LocalTensor({k: v.clone() for k, v in tensors.items()})
        dist.broadcast(lt, src=src_rank, group=process_group)
        result = lt.reconcile()

    return result, expected


# [end_core_broadcast]


# [core_all_gather]
def all_gather_tensors(process_group):
    """Simulate all_gather to collect tensors from all ranks.

    Returns: (results_list, expected_list)
    """
    tensors = {
        0: torch.tensor([[1.0, 2.0]]),
        1: torch.tensor([[3.0, 4.0]]),
        2: torch.tensor([[5.0, 6.0]]),
    }
    num_ranks = len(tensors)

    expected = [tensors[i].clone() for i in range(num_ranks)]

    with LocalTensorMode(frozenset(tensors.keys())):
        lt = LocalTensor(tensors)
        output_list = [torch.zeros_like(lt) for _ in range(num_ranks)]
        dist.all_gather(output_list, lt, group=process_group)
        results = [out.reconcile() for out in output_list]

    return results, expected


# [end_core_all_gather]


# [core_reduce_scatter]
def reduce_scatter_tensors(process_group):
    """Simulate reduce_scatter: reduce then scatter results.

    Returns: (rank_outputs_dict, expected_dict)
    """
    tensors = {
        0: torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        1: torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        2: torch.tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]]),
    }
    num_ranks = len(tensors)

    total_sum = sum(tensors.values())
    rows_per_rank = total_sum.size(0) // num_ranks
    expected = {
        rank: total_sum[rank * rows_per_rank : (rank + 1) * rows_per_rank]
        for rank in range(num_ranks)
    }

    with LocalTensorMode(frozenset(tensors.keys())):
        lt_input = LocalTensor({k: v.clone() for k, v in tensors.items()})
        output_shape = (tensors[0].size(0) // num_ranks, tensors[0].size(1))
        lt_output = torch.zeros(output_shape, dtype=lt_input.dtype)

        dist.reduce_scatter_tensor(lt_output, lt_input, group=process_group)

        rank_outputs = {
            rank: lt_output._local_tensors[rank].clone() for rank in range(num_ranks)
        }

    return rank_outputs, expected


# [end_core_reduce_scatter]


if __name__ == "__main__":
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("fake", rank=0, world_size=3)
    pg = dist.distributed_c10d._get_default_group()

    try:
        print("=== all_reduce_sum ===")
        result, expected = all_reduce_sum(pg)
        print(f"Result: {result}\nExpected: {expected}")

        print("\n=== broadcast_from_rank ===")
        result, expected = broadcast_from_rank(pg, src_rank=0)
        print(f"Result: {result}, Expected: {expected}")

        print("\n=== all_gather_tensors ===")
        results, expected = all_gather_tensors(pg)
        for i, (r, e) in enumerate(zip(results, expected)):
            print(f"  [{i}] {torch.equal(r, e)}")

        print("\n=== reduce_scatter_tensors ===")
        results, expected = reduce_scatter_tensors(pg)
        for rank in results:
            print(f"  Rank {rank}: {torch.equal(results[rank], expected[rank])}")
    finally:
        dist.destroy_process_group()
