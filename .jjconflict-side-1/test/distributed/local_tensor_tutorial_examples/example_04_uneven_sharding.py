# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Example 4: Handling Uneven Sharding
====================================

This module contains core patterns for uneven sharding.
The functions below are designed to be:
1. Included in the tutorial via literalinclude
2. Directly tested to ensure tutorial correctness

Each function returns a tuple of (actual, expected) for testing.
"""

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalIntNode, LocalTensor, LocalTensorMode
from torch.distributed.tensor import distribute_tensor, init_device_mesh, Shard


# [core_uneven_shards]
def create_uneven_shards():
    """Create LocalTensor with different sizes per rank.

    Returns: ((local_tensor, is_symint), expected_shapes_dict)
    """
    tensors = {
        0: torch.tensor([[1.0, 2.0, 3.0, 4.0]]),  # 1 row
        1: torch.tensor([[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]),  # 2 rows
        2: torch.tensor([[13.0, 14.0, 15.0, 16.0]]),  # 1 row
    }

    lt = LocalTensor(tensors)
    is_symint = isinstance(lt.shape[0], torch.SymInt)

    expected_shapes = {rank: t.shape for rank, t in tensors.items()}
    return (lt, is_symint), expected_shapes


# [end_core_uneven_shards]


# [core_local_int_node]
def local_int_node_arithmetic():
    """LocalIntNode for per-rank integer values.

    Returns: ((add_result, mul_result), (expected_add, expected_mul))
    """
    values_a = {0: 10, 1: 20, 2: 30}
    values_b = {0: 1, 1: 2, 2: 3}
    local_a = LocalIntNode(values_a)
    local_b = LocalIntNode(values_b)

    result_add = local_a.add(local_b)
    result_mul = local_a.mul(local_b)

    expected_add = {k: values_a[k] + values_b[k] for k in values_a}
    expected_mul = {k: values_a[k] * values_b[k] for k in values_a}

    return (
        (dict(result_add._local_ints), dict(result_mul._local_ints)),
        (expected_add, expected_mul),
    )


# [end_core_local_int_node]


# [core_dtensor_uneven]
def dtensor_uneven_sharding(world_size: int = 3):
    """DTensor with unevenly divisible tensor dimension.

    Returns: ((rows_per_rank, matches), expected_total_rows)
    """
    total_rows = 10

    with LocalTensorMode(world_size):
        mesh = init_device_mesh("cpu", (world_size,))

        tensor = torch.arange(total_rows * 4).reshape(total_rows, 4).float()
        dt = distribute_tensor(tensor, mesh, [Shard(0)])

        local = dt.to_local()
        rows_per_rank = {
            rank: local._local_tensors[rank].shape[0] for rank in range(world_size)
        }

        reconstructed = dt.full_tensor().reconcile()
        matches = torch.equal(reconstructed, tensor)

    return (rows_per_rank, matches), total_rows


# [end_core_dtensor_uneven]


if __name__ == "__main__":
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("fake", rank=0, world_size=3)

    try:
        print("=== create_uneven_shards ===")
        (lt, is_symint), exp_shapes = create_uneven_shards()
        print(f"Is SymInt: {is_symint}, Shapes: {exp_shapes}")

        print("\n=== local_int_node_arithmetic ===")
        (add_res, mul_res), (exp_add, exp_mul) = local_int_node_arithmetic()
        print(f"Add: {add_res}={exp_add}, Mul: {mul_res}={exp_mul}")

        print("\n=== dtensor_uneven_sharding ===")
        (rows, matches), exp_total = dtensor_uneven_sharding()
        print(
            f"Rows: {rows}, Total: {sum(rows.values())}={exp_total}, Matches: {matches}"
        )
    finally:
        dist.destroy_process_group()
