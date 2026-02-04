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

import copy

import torch
import torch.distributed as dist
from datetime import timedelta
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import fully_shard_flat, get_chunked_storage
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP
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

    # Configure Transformer with dimensions for uneven sharding
    torch.manual_seed(42)
    args = ModelArgs(
        n_layers=2,
        vocab_size=33,  # uneven for ws=2,4
        max_seq_len=7,  # uneven for ws=2,4
        dim=16,
        n_heads=2,
        dropout_p=0.0,
        weight_tying=False,
    )
    model = Transformer(args).to(device)

    # Create reference model before sharding (DDP for comparison)
    ref_model = copy.deepcopy(model)
    ref_model = DDP(ref_model, device_ids=[rank])
    ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

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

    print_rank0(f"\n=== Parameters ===")
    print_rank0(f"  Total: {sum(1 for _ in model.parameters())}, "
                f"Layers: {len(layer_param_names)}, Root: {len(root_param_names)}")

    # === Apply nested fully_shard_flat ===
    print_rank0(f"\n=== Applying nested fully_shard_flat ===")
    for i, layer in enumerate(model.layers):
        storage = fully_shard_flat(layer, mesh)
        print_rank0(f"  Layer {i}: {len(storage.param_infos)} params, {storage.total_bytes} bytes")
        assert get_chunked_storage(layer) is storage

    root_storage = fully_shard_flat(model, mesh)
    print_rank0(f"  Root: {len(root_storage.param_infos)} params, {root_storage.total_bytes} bytes")
    assert get_chunked_storage(model) is root_storage

    # Verify root storage excludes layer params
    root_storage_fqns = set(root_storage.param_infos.keys())
    assert root_storage_fqns == root_param_names

    # Create optimizer for FSDP model
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    # === Verify sharding ===
    print_rank0(f"\n=== Sharding ===")
    for name, param in model.named_parameters():
        global_shape = tuple(param.shape)
        local_shape = tuple(param._local_tensor.shape)
        dim0_global = global_shape[0]
        sharding = "uneven" if dim0_global % world_size != 0 else "even"
        if rank == 0:
            print(f"  {name:<38} | {str(global_shape):^12} | {str(local_shape):^12} | {sharding}")

    # === Training loop with numerics verification ===
    print_rank0(f"\n=== Training loop (comparing with DDP reference) ===")
    batch_size, seq_len = 2, args.max_seq_len
    num_iters = 5

    # Verify weights match after sharding (before any training)
    root_storage.unshard()
    for layer_storage in [get_chunked_storage(layer) for layer in model.layers]:
        if layer_storage:
            layer_storage.unshard()

    max_weight_diff = 0.0
    for (name1, p1), (name2, p2) in zip(
        ref_model.module.named_parameters(), model.named_parameters()
    ):
        diff = (p1 - p2).abs().max().item()
        max_weight_diff = max(max_weight_diff, diff)

    if max_weight_diff > 1e-6:
        raise AssertionError(f"Weight mismatch after sharding: max_diff={max_weight_diff}")
    print_rank0(f"  ✓ Weights match after unshard (max_diff={max_weight_diff})")

    # Reshard for training loop
    root_storage.reshard()
    for layer_storage in [get_chunked_storage(layer) for layer in model.layers]:
        if layer_storage:
            layer_storage.reshard()

    torch.manual_seed(42 + rank)  # Same seed across models, different per rank
    for iteration in range(num_iters):
        inp = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)

        # Reference model (DDP)
        ref_optim.zero_grad()
        ref_loss = ref_model(inp).sum()
        ref_loss.backward()
        ref_optim.step()

        # FSDP model
        optim.zero_grad()
        fsdp_loss = model(inp).sum()
        fsdp_loss.backward()
        torch.cuda.synchronize()  # Wait for post-backward callback
        optim.step()

        # Verify params are DTensors after backward
        for name, param in model.named_parameters():
            assert isinstance(param, DTensor), f"Iter {iteration}: {name} should be DTensor"
            if param._local_tensor.numel() > 0:
                assert param.grad is not None, f"Iter {iteration}: {name} missing gradient"

        # Compare losses (use torch.testing.assert_close like FSDP2's assertEqual)
        torch.testing.assert_close(ref_loss, fsdp_loss)
        print_rank0(f"  Iter {iteration}: ref_loss={ref_loss.item():.6f}, "
                    f"fsdp_loss={fsdp_loss.item():.6f} ✓")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED: Numerics match DDP reference")
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
