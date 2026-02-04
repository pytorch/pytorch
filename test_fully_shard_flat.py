#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for fully_shard_flat API with Transformer model.

Tests uneven sharding (dim % world_size != 0) and sparse sharding (dim < world_size).

Usage:
    torchrun --nproc_per_node=2 test_fully_shard_flat.py
    torchrun --nproc_per_node=4 test_fully_shard_flat.py
"""

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import fully_shard_flat
from torch.distributed.tensor import DTensor
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


def setup_distributed():
    """Initialize distributed environment."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=20))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    return rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        torch.cuda.synchronize()  # Ensure all GPU work is done
        dist.destroy_process_group()


def print_rank0(msg):
    """Print only on rank 0."""
    if dist.get_rank() == 0:
        print(msg)


def run():
    """Test fully_shard_flat with Transformer model."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    # Configure Transformer with dimensions for uneven/sparse sharding:
    # - vocab_size=3: sparse sharding for world_size=4 (some ranks get 0 rows)
    # - max_seq_len=5: uneven for world_size=2,4
    # - dim=6: even for ws=2, uneven for ws=4
    # - n_heads=2: head_dim=3
    # - hidden_dim=4*dim=24: even for both
    args = ModelArgs(
        n_layers=1,
        vocab_size=3,  # sparse for ws=4
        max_seq_len=5,  # uneven for ws=2,4
        dim=6,
        n_heads=2,
        dropout_p=0.0,  # disable dropout for deterministic testing
        weight_tying=False,  # separate output weight from embeddings
    )
    model = Transformer(args).to(device).to(torch.bfloat16)

    print_rank0(f"\n{'='*70}")
    print_rank0(f"Testing fully_shard_flat with Transformer, world_size={world_size}")
    print_rank0(f"{'='*70}")
    print_rank0(f"  vocab_size={args.vocab_size}, max_seq_len={args.max_seq_len}, dim={args.dim}")

    # Store original parameter shapes for verification
    original_shapes = {name: param.shape for name, param in model.named_parameters()}

    print_rank0("\n=== Original Parameter Shapes & Dtypes ===")
    for name, param in model.named_parameters():
        dtype_str = "bf16" if param.dtype == torch.bfloat16 else "fp32"
        print_rank0(f"  {name}: {tuple(param.shape)}, {dtype_str}")

    # Apply fully_shard_flat
    storage = fully_shard_flat(model, mesh)

    # === Test 1: Sharding ===
    print_rank0(f"\n=== After fully_shard_flat ===")
    print_rank0(f"{'Parameter':<40} | {'Dtype':^6} | {'Global':^15} | {'Local':^15} | {'Sharding':<20}")
    print_rank0("-" * 110)

    for name, param in model.named_parameters():
        global_shape = tuple(param.shape)
        local_shape = tuple(param._local_tensor.shape)
        dim0_global = global_shape[0]
        dim0_local = local_shape[0]
        dtype_str = "bf16" if param.dtype == torch.bfloat16 else "fp32"

        if dim0_global < world_size:
            sharding = f"{dim0_global}/{world_size} (sparse)"
        elif dim0_global % world_size == 0:
            sharding = f"{dim0_global}/{world_size}={dim0_local} (even)"
        else:
            sharding = f"{dim0_global}/{world_size}≈{dim0_local} (uneven)"

        if rank == 0:
            print(f"  {name:<38} | {dtype_str:^6} | {str(global_shape):^15} | {str(local_shape):^15} | {sharding}")

    # Verify unified byte storage
    print_rank0(f"\n=== Unified Byte Storage ===")
    print_rank0(f"  Sharded bytes: {storage.total_bytes}")
    print_rank0(f"  Unsharded bytes: {storage.total_unsharded_bytes}")
    print_rank0(f"  Storage dtype: {storage.byte_storage.dtype}")

    storage_ptrs = set()
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor"
        storage_ptrs.add(param._local_tensor.untyped_storage().data_ptr())

    assert len(storage_ptrs) == 1, f"Expected 1 unified storage, got {len(storage_ptrs)}"
    print_rank0(f"  ✓ ALL parameters share SINGLE unified byte storage")

    # Show local shapes per rank (first few params only)
    print_rank0(f"\n=== Local Shapes Per Rank (sample) ===")
    sample_params = ["tok_embeddings.weight", "pos_embeddings.weight", "output.weight"]
    for name, param in model.named_parameters():
        if name in sample_params:
            all_shapes = [None] * world_size
            dist.all_gather_object(all_shapes, tuple(param._local_tensor.shape))
            if rank == 0:
                shapes_str = ", ".join([f"r{i}:{s[0]}" for i, s in enumerate(all_shapes)])
                print(f"  {name:<25}: [{shapes_str}]")

    # === Test 2: Unshard/Reshard ===
    print_rank0(f"\n=== Testing unshard() ===")
    print_rank0(f"  State before: {storage.state}")
    storage.unshard()
    print_rank0(f"  State after:  {storage.state}")

    for name, param in model.named_parameters():
        assert not isinstance(param, DTensor), f"{name} should NOT be DTensor after unshard"
        assert param.shape == original_shapes[name], f"{name} shape mismatch"
    print_rank0(f"  ✓ All parameters are unsharded with original shapes")

    # Forward with unsharded params
    batch_size = 2
    seq_len = args.max_seq_len
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    y = model(tokens)
    print_rank0(f"  ✓ Forward pass works (input: {tuple(tokens.shape)}, output: {tuple(y.shape)})")

    # === Test 3: Reshard ===
    print_rank0(f"\n=== Testing reshard() ===")
    storage.reshard()
    print_rank0(f"  State: {storage.state}")

    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor after reshard"
    print_rank0(f"  ✓ All parameters restored to DTensors")

    # === Test 4: Context Manager ===
    print_rank0(f"\n=== Testing context manager ===")
    with storage.unsharded():
        for name, param in model.named_parameters():
            assert not isinstance(param, DTensor), f"{name} should be unsharded in context"
        y = model(tokens)
        print_rank0(f"  ✓ Forward in context manager works")

    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor after context"
    print_rank0(f"  ✓ Parameters restored after context")

    # === Test 5: Forward + Backward ===
    print_rank0(f"\n=== Testing forward + backward ===")

    # Clear any existing gradients
    for param in model.parameters():
        param.grad = None

    # Unshard before forward
    storage.unshard()
    print_rank0(f"  State after unshard: {storage.state}")

    # Forward pass
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    y = model(tokens)
    loss = y.sum()
    print_rank0(f"  Forward: input={tuple(tokens.shape)}, output={tuple(y.shape)}, loss={loss.item():.4f}")

    # Backward pass (with unsharded params)
    loss.backward()
    print_rank0(f"  Backward complete")

    # Verify gradients exist
    grads_ok = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print_rank0(f"  ERROR: {name} has no gradient")
            grads_ok = False
    assert grads_ok, "Some parameters missing gradients"
    print_rank0(f"  ✓ All {len(list(model.parameters()))} parameters have gradients")

    # Reshard after backward
    storage.reshard()
    print_rank0(f"  State after reshard: {storage.state}")

    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor after reshard"
    print_rank0(f"  ✓ Parameters restored to DTensors after backward")

    # Check if DTensor params have gradients after reshard
    print_rank0(f"\n=== Checking gradients on DTensor params after reshard ===")
    for name, param in model.named_parameters():
        has_grad = param.grad is not None
        if has_grad:
            grad_is_dtensor = isinstance(param.grad, DTensor)
            grad_global_shape = tuple(param.grad.shape)
            grad_local_shape = tuple(param.grad._local_tensor.shape) if grad_is_dtensor else "N/A"
            print_rank0(f"  {name}: DTensor={grad_is_dtensor}, global={grad_global_shape}, local={grad_local_shape}")
        else:
            print_rank0(f"  {name}: grad=NO")

    # Verify all grads are DTensors
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} missing gradient"
        assert isinstance(param.grad, DTensor), f"{name} gradient should be DTensor"
    print_rank0(f"  ✓ All gradients are DTensors with sharded local tensors")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED: All tests")
    print_rank0(f"{'='*70}")

    # Synchronize GPU operations before exiting
    torch.cuda.synchronize()

    return True


def main():
    """Run test."""
    rank, world_size = setup_distributed()

    print_rank0(f"Running tests with world_size={world_size}")

    try:
        run()
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
