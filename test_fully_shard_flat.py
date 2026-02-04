#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for fully_shard_flat API with uneven sharding.

Tests parameter shapes that don't evenly divide by world_size (2, 4).

Usage:
    torchrun --nproc_per_node=2 test_fully_shard_flat.py
    torchrun --nproc_per_node=4 test_fully_shard_flat.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import ChunkedStorage, fully_shard_flat
from torch.distributed.tensor import DTensor, Shard, Replicate, distribute_tensor


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg):
    """Print only on rank 0."""
    if dist.get_rank() == 0:
        print(msg)


class UnevenModel(nn.Module):
    """
    Model with parameter shapes that don't evenly divide by 2 or 4.
    Uses mixed dtypes: bfloat16 for main layers, float32 for output layer.

    Shapes chosen to create uneven sharding:
    - 5: doesn't divide evenly by 2 or 4
    - 7: doesn't divide evenly by 2 or 4
    - 11: doesn't divide evenly by 2 or 4
    - 13: doesn't divide evenly by 2 or 4
    - 3: doesn't divide evenly by 2 or 4, smaller than world_size 4
    """

    def __init__(self):
        super().__init__()
        # bfloat16 layers (main compute)
        # Layer 1: (5, 16) - 5 rows, uneven for world_size 2, 4
        self.fc1 = nn.Linear(16, 5, bias=True).to(torch.bfloat16)

        # Layer 2: (7, 5) - 7 rows, uneven for world_size 2, 4
        self.fc2 = nn.Linear(5, 7, bias=True).to(torch.bfloat16)

        # Layer 3: (11, 7) - 11 rows, uneven for world_size 2, 4
        self.fc3 = nn.Linear(7, 11, bias=True).to(torch.bfloat16)

        # float32 layers (for numerical stability)
        # Layer 4: (13, 11) - 13 rows, uneven for world_size 2, 4
        self.fc4 = nn.Linear(11, 13, bias=True)  # float32

        # Layer 5: (3, 13) - 3 rows, smaller than world_size 4
        self.fc5 = nn.Linear(13, 3, bias=True)  # float32

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # Convert to float32 for final layers
        x = x.to(torch.float32)
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def test_uneven_sharding():
    """Test with uneven parameter shapes and mixed dtypes."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())

    mesh = init_device_mesh("cuda", (world_size,))

    model = UnevenModel().to(device)

    print_rank0(f"\n{'='*70}")
    print_rank0(f"Testing UNEVEN sharding + MIXED DTYPE with world_size={world_size}")
    print_rank0(f"{'='*70}")

    print_rank0("\n=== Original Parameter Shapes & Dtypes ===")
    for name, param in model.named_parameters():
        print_rank0(f"  {name}: {tuple(param.shape)}, dtype={param.dtype}")

    # Apply fully_shard_flat
    storage = fully_shard_flat(model, mesh)

    print_rank0(f"\n=== After fully_shard_flat (world_size={world_size}) ===")
    print_rank0(f"{'Parameter':<20} | {'Dtype':^10} | {'Global':^12} | {'Local':^12} | {'Sharding':<20}")
    print_rank0("-" * 85)

    for name, param in model.named_parameters():
        global_shape = tuple(param.shape)
        local_shape = tuple(param._local_tensor.shape)
        dim0_global = global_shape[0]
        dim0_local = local_shape[0]
        dtype_str = "bf16" if param.dtype == torch.bfloat16 else "fp32"

        # Describe the sharding
        if dim0_global < world_size:
            sharding = f"{dim0_global} rows / {world_size} ranks (sparse)"
        elif dim0_global % world_size == 0:
            sharding = f"{dim0_global}/{world_size} = {dim0_local} (even)"
        else:
            sharding = f"{dim0_global}/{world_size} ≈ {dim0_local} (uneven)"

        if rank == 0:
            print(f"  {name:<18} | {dtype_str:^10} | {str(global_shape):^12} | {str(local_shape):^12} | {sharding}")

    # Show unified byte storage info
    print_rank0(f"\n=== Unified Byte Storage ===")
    print_rank0(f"  Total bytes: {storage.total_bytes}")
    print_rank0(f"  Storage dtype: {storage.byte_storage.dtype}")
    print_rank0(f"  Storage shape: {storage.byte_storage.shape}")

    # Show byte offsets
    print_rank0(f"\n=== Parameter Byte Offsets ===")
    for fqn, info in storage.param_infos.items():
        num_bytes = info.local_numel * info.dtype.itemsize
        dtype_str = "bf16" if info.dtype == torch.bfloat16 else "fp32"
        print_rank0(f"  {fqn}: offset={info.byte_offset}, bytes={num_bytes}, dtype={dtype_str}")

    # Verify all parameters share SINGLE unified storage
    storage_ptrs = set()
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor"
        storage_ptrs.add(param._local_tensor.untyped_storage().data_ptr())

    assert len(storage_ptrs) == 1, f"Expected 1 unified storage, got {len(storage_ptrs)}"
    print_rank0(f"\n✓ ALL parameters (bf16 + fp32) share SINGLE unified byte storage")

    # Test forward/backward
    print_rank0("\n=== Forward/Backward Test ===")

    x = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
    x_dt = distribute_tensor(x, mesh, (Replicate(),))

    y = model(x_dt)
    print_rank0(f"  Input:  {tuple(x_dt.shape)}, dtype={x_dt.dtype}")
    print_rank0(f"  Output: {tuple(y.shape)}, dtype={y.dtype}")

    loss = y.sum()
    loss.backward()

    grad_count = sum(1 for _, p in model.named_parameters() if p.grad is not None)
    num_params = sum(1 for _ in model.parameters())
    assert grad_count == num_params, f"Expected {num_params} gradients, got {grad_count}"
    print_rank0(f"  ✓ All {num_params} parameters have gradients")

    # Show local shapes on all ranks
    print_rank0(f"\n=== Local Shapes Per Rank ===")

    param_names = [name for name, _ in model.named_parameters()]

    # Gather local shapes from all ranks
    for name, param in model.named_parameters():
        local_shape = param._local_tensor.shape
        local_numel = param._local_tensor.numel()

        # Collect info from all ranks
        all_shapes = [None] * world_size
        all_numels = [None] * world_size

        dist.all_gather_object(all_shapes, tuple(local_shape))
        dist.all_gather_object(all_numels, local_numel)

        if rank == 0:
            shapes_str = ", ".join([f"r{i}:{s[0]}" for i, s in enumerate(all_shapes)])
            print(f"  {name:<18}: [{shapes_str}]")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED: Uneven sharding + mixed dtype test")
    print_rank0(f"{'='*70}")

    return True


def main():
    """Run test."""
    rank, world_size = setup_distributed()

    print_rank0(f"Running test with world_size={world_size}")

    try:
        test_uneven_sharding()
        print_rank0("\n=== All tests passed! ===")
        success = True
    except Exception as e:
        print_rank0(f"FAILED: {e}")
        import traceback
        if rank == 0:
            traceback.print_exc()
        success = False

    cleanup_distributed()
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
