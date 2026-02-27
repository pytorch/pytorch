# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Example 3: Working with DTensor
================================

This module contains core patterns for DTensor integration.
The functions below are designed to be:
1. Included in the tutorial via literalinclude
2. Directly tested to ensure tutorial correctness

Each function returns a tuple of (actual, expected) for testing.
"""

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.tensor import (
    distribute_tensor,
    init_device_mesh,
    Replicate,
    Shard,
)


# [core_dtensor_distribute]
def distribute_and_verify(world_size: int = 4):
    """Distribute a tensor and verify reconstruction.

    Returns: ((sharded_actual, replicated_actual), (sharded_expected, replicated_expected))
    """
    with LocalTensorMode(world_size):
        mesh = init_device_mesh("cpu", (world_size,))
        tensor = torch.arange(16).reshape(4, 4).float()

        dt_sharded = distribute_tensor(tensor, mesh, [Shard(0)])
        dt_replicated = distribute_tensor(tensor, mesh, [Replicate()])

        sharded_actual = dt_sharded.full_tensor().reconcile()
        replicated_actual = dt_replicated.to_local().reconcile()

    return (sharded_actual, replicated_actual), (tensor, tensor)


# [end_core_dtensor_distribute]


# [core_dtensor_matmul]
def dtensor_matmul(world_size: int = 4):
    """Perform matrix multiplication with DTensors.

    Returns: (actual, expected)
    """
    with LocalTensorMode(world_size):
        mesh = init_device_mesh("cpu", (world_size,))

        a = torch.randn(8, 4)
        b = torch.randn(4, 6)

        da = distribute_tensor(a, mesh, [Shard(0)])
        db = distribute_tensor(b, mesh, [Replicate()])

        dc = da @ db

        expected = a @ b
        actual = dc.full_tensor().reconcile()

    return actual, expected


# [end_core_dtensor_matmul]


# [core_dtensor_nn_layer]
def dtensor_linear_layer(world_size: int = 4):
    """Simulate a distributed linear layer forward pass.

    Returns: (actual, expected)
    """
    batch_size, in_features, out_features = 16, 8, 4

    with LocalTensorMode(world_size):
        mesh = init_device_mesh("cpu", (world_size,))

        x = torch.randn(batch_size, in_features)
        w = torch.randn(in_features, out_features)
        b = torch.randn(out_features)

        dx = distribute_tensor(x, mesh, [Shard(0)])
        dw = distribute_tensor(w, mesh, [Replicate()])
        db = distribute_tensor(b, mesh, [Replicate()])

        dy = torch.relu(dx @ dw + db)

        expected = torch.relu(x @ w + b)
        actual = dy.full_tensor().reconcile()

    return actual, expected


# [end_core_dtensor_nn_layer]


if __name__ == "__main__":
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group("fake", rank=0, world_size=4)

    try:
        print("=== distribute_and_verify ===")
        (sharded, replicated), (exp_sharded, exp_replicated) = distribute_and_verify()
        print(f"Sharded matches: {torch.equal(sharded, exp_sharded)}")
        print(f"Replicated matches: {torch.equal(replicated, exp_replicated)}")

        print("\n=== dtensor_matmul ===")
        actual, expected = dtensor_matmul()
        print(f"Matmul matches: {torch.allclose(actual, expected, atol=1e-5)}")

        print("\n=== dtensor_linear_layer ===")
        actual, expected = dtensor_linear_layer()
        print(f"Linear matches: {torch.allclose(actual, expected, atol=1e-5)}")
    finally:
        dist.destroy_process_group()
