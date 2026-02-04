#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Test script for fully_shard_flat API with Transformer model.

Tests uneven sharding (dim % world_size != 0) and nested wrapping.

Usage:
    torchrun --nproc_per_node=2 test_fully_shard_flat.py
    torchrun --nproc_per_node=4 test_fully_shard_flat.py
"""

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import fully_shard_flat, get_chunked_storage
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
    """Test fully_shard_flat with nested wrapping and automatic hooks."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    # Configure Transformer with dimensions for uneven sharding:
    # - vocab_size=5: uneven for world_size=2,4
    # - max_seq_len=7: uneven for world_size=2,4
    # - dim=6: even for ws=2, uneven for ws=4
    args = ModelArgs(
        n_layers=2,
        vocab_size=5,
        max_seq_len=7,
        dim=6,
        n_heads=2,
        dropout_p=0.0,
        weight_tying=False,
    )
    model = Transformer(args).to(device).to(torch.bfloat16)

    print_rank0(f"\n{'='*70}")
    print_rank0(f"Testing fully_shard_flat with world_size={world_size}")
    print_rank0(f"{'='*70}")

    # Count params per component
    layer_param_names = set()
    for i, layer in enumerate(model.layers):
        for name, _ in layer.named_parameters():
            layer_param_names.add(f"layers.{i}.{name}")
    root_param_names = set()
    for name, _ in model.named_parameters():
        if name not in layer_param_names:
            root_param_names.add(name)

    print_rank0(f"\n=== Original Parameters ===")
    print_rank0(f"  Total: {sum(1 for _ in model.parameters())}, "
                f"Layers: {len(layer_param_names)}, Root: {len(root_param_names)}")

    # === Step 1: Nested wrapping (layers first, then root) ===
    print_rank0(f"\n=== Applying nested fully_shard_flat ===")
    layer_storages = []
    for i, layer in enumerate(model.layers):
        storage = fully_shard_flat(layer, mesh)
        layer_storages.append(storage)
        print_rank0(f"  Layer {i}: {len(storage.param_infos)} params, {storage.total_bytes} bytes")
        assert get_chunked_storage(layer) is storage

    root_storage = fully_shard_flat(model, mesh)
    print_rank0(f"  Root: {len(root_storage.param_infos)} params, {root_storage.total_bytes} bytes")
    assert get_chunked_storage(model) is root_storage

    # Verify root storage excludes layer params
    root_storage_fqns = set(root_storage.param_infos.keys())
    assert root_storage_fqns == root_param_names, (
        f"Root storage should only have root params.\n"
        f"Expected: {sorted(root_param_names)}\nGot: {sorted(root_storage_fqns)}"
    )
    print_rank0(f"  ✓ Root storage correctly excludes layer parameters")

    # === Step 2: Verify sharding ===
    print_rank0(f"\n=== After fully_shard_flat ===")
    print_rank0(f"{'Parameter':<40} | {'Global':^12} | {'Local':^12} | {'Sharding':<15}")
    print_rank0("-" * 85)

    for name, param in model.named_parameters():
        global_shape = tuple(param.shape)
        local_shape = tuple(param._local_tensor.shape)
        dim0_global, dim0_local = global_shape[0], local_shape[0]

        if dim0_global < world_size:
            sharding = "sparse"
        elif dim0_global % world_size == 0:
            sharding = "even"
        else:
            sharding = "uneven"

        if rank == 0:
            print(f"  {name:<38} | {str(global_shape):^12} | {str(local_shape):^12} | {sharding}")

    # Verify unified storage per module
    for i, storage in enumerate(layer_storages):
        ptrs = set()
        for fqn in storage.param_infos:
            param = dict(model.layers[i].named_parameters())[fqn]
            ptrs.add(param._local_tensor.untyped_storage().data_ptr())
        assert len(ptrs) == 1, f"Layer {i}: expected 1 unified storage"
    print_rank0(f"  ✓ Each module has unified byte storage")

    # === Step 3: Forward (automatic unshard via hooks) ===
    print_rank0(f"\n=== Forward pass (automatic hooks) ===")
    batch_size, seq_len = 2, args.max_seq_len
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    output = model(tokens)
    print_rank0(f"  input={tuple(tokens.shape)}, output={tuple(output.shape)}")

    # After forward, params are unsharded (kept for backward)
    for name, param in model.named_parameters():
        assert not isinstance(param, DTensor), f"{name} should be unsharded after forward"
    print_rank0(f"  ✓ All parameters unsharded after forward")

    # === Step 4: Forward + Backward (automatic hooks) ===
    print_rank0(f"\n=== Forward + Backward (automatic hooks) ===")
    for param in model.parameters():
        param.grad = None

    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    output = model(tokens)
    loss = output.sum()
    loss.backward()
    torch.cuda.synchronize()
    print_rank0(f"  loss={loss.item():.4f}")

    # Verify params are DTensors with DTensor gradients
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor after backward"
        if param._local_tensor.numel() > 0:
            assert param.grad is not None, f"{name} missing gradient"
            assert isinstance(param.grad, DTensor), f"{name} gradient should be DTensor"
    print_rank0(f"  ✓ All parameters are DTensors with DTensor gradients")

    # === Step 5: Multiple iterations ===
    print_rank0(f"\n=== Multiple iterations ===")
    for iteration in range(3):
        for param in model.parameters():
            param.grad = None
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
        loss = model(tokens).sum()
        loss.backward()
        torch.cuda.synchronize()

        for name, param in model.named_parameters():
            assert isinstance(param, DTensor), f"Iter {iteration}: {name} should be DTensor"
            if param._local_tensor.numel() > 0:
                assert param.grad is not None, f"Iter {iteration}: {name} missing gradient"

    print_rank0(f"  ✓ 3 iterations completed")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED")
    print_rank0(f"{'='*70}")

    torch.cuda.synchronize()
    return True


def main():
    """Run tests."""
    rank, world_size = setup_distributed()

    print_rank0(f"Running tests with world_size={world_size}")

    try:
        run()
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
