# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

"""
Example 5: Rank-Specific Computations
======================================

This module contains core patterns for rank-specific operations.
The functions below are designed to be:
1. Included in the tutorial via literalinclude
2. Directly tested to ensure tutorial correctness

Each function returns a tuple of (actual, expected) for testing.
"""

import torch
from torch.distributed._local_tensor import (
    LocalTensorMode,
    maybe_disable_local_tensor_mode,
)


# [core_rank_map]
def use_rank_map(world_size: int = 4):
    """Create LocalTensors with per-rank values using rank_map.

    Returns: (values_dict, expected_dict)
    """
    with LocalTensorMode(world_size) as mode:
        lt = mode.rank_map(lambda rank: torch.full((2, 3), float(rank)))
        values = {
            rank: lt._local_tensors[rank][0, 0].item() for rank in range(world_size)
        }

    expected = {rank: float(rank) for rank in range(world_size)}
    return values, expected


# [end_core_rank_map]


# [core_tensor_map]
def use_tensor_map(world_size: int = 4):
    """Transform each shard differently using tensor_map.

    Returns: (values_dict, expected_dict)
    """
    with LocalTensorMode(world_size) as mode:
        lt = mode.rank_map(lambda rank: torch.ones(2, 2) * (rank + 1))

        def scale_by_rank(rank: int, tensor: torch.Tensor) -> torch.Tensor:
            return tensor * (rank + 1)

        scaled = mode.tensor_map(lt, scale_by_rank)
        values = {
            rank: scaled._local_tensors[rank][0, 0].item() for rank in range(world_size)
        }

    # (rank + 1) * (rank + 1) = (rank + 1)^2
    expected = {rank: float((rank + 1) ** 2) for rank in range(world_size)}
    return values, expected


# [end_core_tensor_map]


# [core_disable_mode]
def disable_mode_temporarily(world_size: int = 4):
    """Temporarily exit LocalTensorMode for regular tensor ops.

    Returns: ((inside_type, disabled_type), (expected_inside, expected_disabled))
    """
    with LocalTensorMode(world_size) as mode:
        lt = torch.ones(2, 2)
        inside_type = type(lt).__name__

        with mode.disable():
            regular = torch.ones(2, 2)
            disabled_type = type(regular).__name__

    return (inside_type, disabled_type), ("LocalTensor", "Tensor")


# [end_core_disable_mode]


# [core_maybe_disable]
def use_maybe_disable():
    """Use maybe_disable_local_tensor_mode() for portable code.

    Returns: ((outside_type, inside_type), (expected_outside, expected_inside))
    """

    def create_tensor():
        with maybe_disable_local_tensor_mode():
            return torch.tensor([1.0, 2.0, 3.0])

    t1 = create_tensor()
    outside_type = type(t1).__name__

    with LocalTensorMode(4):
        t2 = create_tensor()
        inside_type = type(t2).__name__

    return (outside_type, inside_type), ("Tensor", "Tensor")


# [end_core_maybe_disable]


if __name__ == "__main__":
    print("=== use_rank_map ===")
    values, expected = use_rank_map()
    print(f"Values: {values}={expected}")

    print("\n=== use_tensor_map ===")
    values, expected = use_tensor_map()
    print(f"Values: {values}={expected}")

    print("\n=== disable_mode_temporarily ===")
    (inside, disabled), (exp_in, exp_dis) = disable_mode_temporarily()
    print(f"Inside: {inside}={exp_in}, Disabled: {disabled}={exp_dis}")

    print("\n=== use_maybe_disable ===")
    (outside, inside), (exp_out, exp_in) = use_maybe_disable()
    print(f"Outside: {outside}={exp_out}, Inside: {inside}={exp_in}")
