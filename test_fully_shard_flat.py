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
    """Test fully_shard_flat with Transformer model."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    # Configure Transformer with dimensions for uneven sharding:
    # - vocab_size=5: uneven for world_size=2,4 (no sparse sharding)
    # - max_seq_len=7: uneven for world_size=2,4
    # - dim=6: even for ws=2, uneven for ws=4
    # - n_heads=2: head_dim=3
    # - hidden_dim=4*dim=24: even for both
    args = ModelArgs(
        n_layers=1,
        vocab_size=5,  # uneven for ws=2,4 (avoids sparse)
        max_seq_len=7,  # uneven for ws=2,4
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

    # Apply fully_shard_flat (without hooks for manual control)
    storage = fully_shard_flat(model, mesh, register_hooks=False)

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

    # Show local shapes per rank (rank 0 only, no collective)
    print_rank0(f"\n=== Local Shapes on Rank 0 (sample) ===")
    sample_params = ["tok_embeddings.weight", "pos_embeddings.weight", "output.weight"]
    for name, param in model.named_parameters():
        if name in sample_params:
            print_rank0(f"  {name:<25}: local={tuple(param._local_tensor.shape)}")

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

    # Verify all grads are DTensors (skip for ranks with 0 elements)
    for name, param in model.named_parameters():
        if param._local_tensor.numel() > 0:
            assert param.grad is not None, f"{name} missing gradient"
            assert isinstance(param.grad, DTensor), f"{name} gradient should be DTensor"
    print_rank0(f"  ✓ All gradients are DTensors with sharded local tensors")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED: All tests")
    print_rank0(f"{'='*70}")

    # Synchronize GPU operations before exiting
    torch.cuda.synchronize()

    return True


def run_nested():
    """Test nested wrapping: layers first, then root."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    # Create Transformer with 2 layers
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
    print_rank0(f"Testing NESTED wrapping with Transformer, world_size={world_size}")
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

    print_rank0(f"\n=== Parameter Distribution ===")
    print_rank0(f"  Total params: {sum(1 for _ in model.parameters())}")
    print_rank0(f"  Layer params: {len(layer_param_names)}")
    print_rank0(f"  Root params:  {len(root_param_names)}")
    print_rank0(f"  Root param names: {sorted(root_param_names)}")

    # === Step 1: Apply fully_shard_flat to each layer (without hooks for manual control) ===
    print_rank0(f"\n=== Step 1: Wrapping layers ===")
    layer_storages = []
    for i, layer in enumerate(model.layers):
        storage = fully_shard_flat(layer, mesh, register_hooks=False)
        layer_storages.append(storage)
        print_rank0(f"  Layer {i}: {len(storage.param_infos)} params, {storage.total_bytes} bytes")
        # Verify storage is attached to layer
        assert get_chunked_storage(layer) is storage, f"Layer {i} storage not attached"

    # === Step 2: Apply fully_shard_flat to root (should exclude layer params) ===
    print_rank0(f"\n=== Step 2: Wrapping root (excludes layers) ===")
    root_storage = fully_shard_flat(model, mesh, register_hooks=False)
    print_rank0(f"  Root: {len(root_storage.param_infos)} params, {root_storage.total_bytes} bytes")
    assert get_chunked_storage(model) is root_storage, "Root storage not attached"

    # Verify root storage only has non-layer params
    root_storage_fqns = set(root_storage.param_infos.keys())
    print_rank0(f"  Root storage FQNs: {sorted(root_storage_fqns)}")
    assert root_storage_fqns == root_param_names, (
        f"Root storage should only have root params.\n"
        f"Expected: {sorted(root_param_names)}\n"
        f"Got: {sorted(root_storage_fqns)}"
    )
    print_rank0(f"  ✓ Root storage correctly excludes layer parameters")

    # Print the model structure after applying fully_shard_flat
    print_rank0(f"\n=== Model after fully_shard_flat ===")
    print_rank0(str(model))

    # Show that parameters are DTensors
    print_rank0(f"\n=== Sample Parameters (showing DTensor) ===")
    for name in ["tok_embeddings.weight", "layers.0.attention.wq.weight", "output.weight"]:
        param = dict(model.named_parameters())[name]
        print_rank0(f"  {name}:")
        print_rank0(f"    type: {type(param.data).__name__}")
        print_rank0(f"    global_shape: {tuple(param.shape)}")
        print_rank0(f"    local_shape:  {tuple(param._local_tensor.shape)}")
        print_rank0(f"    placements:   {param.placements}")

    # === Step 3: Test unshard/reshard for all storages ===
    print_rank0(f"\n=== Step 3: Testing unshard/reshard ===")

    # Unshard all (layers first, then root)
    for i, storage in enumerate(layer_storages):
        storage.unshard()
    root_storage.unshard()
    print_rank0(f"  ✓ All storages unsharded")

    # Verify all params are unsharded
    for name, param in model.named_parameters():
        assert not isinstance(param, DTensor), f"{name} should be unsharded"
    print_rank0(f"  ✓ All parameters are unsharded tensors")

    # Forward pass
    batch_size = 2
    seq_len = args.max_seq_len
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    y = model(tokens)
    print_rank0(f"  ✓ Forward pass: input={tuple(tokens.shape)}, output={tuple(y.shape)}")

    # Reshard all (root first, then layers - reverse order)
    root_storage.reshard()
    for storage in layer_storages:
        storage.reshard()
    print_rank0(f"  ✓ All storages resharded")

    # Verify all params are DTensors again
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor"
    print_rank0(f"  ✓ All parameters are DTensors")

    # === Step 4: Test forward + backward with nested wrapping ===
    print_rank0(f"\n=== Step 4: Testing forward + backward ===")

    # Clear gradients
    for param in model.parameters():
        param.grad = None

    # Unshard all
    for storage in layer_storages:
        storage.unshard()
    root_storage.unshard()

    # Forward + backward
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    y = model(tokens)
    loss = y.sum()
    loss.backward()
    print_rank0(f"  Forward + backward complete, loss={loss.item():.4f}")

    # Verify all grads exist
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} missing gradient"
    print_rank0(f"  ✓ All parameters have gradients")

    # Reshard all (reduce-scatter grads)
    root_storage.reshard()
    for storage in layer_storages:
        storage.reshard()
    print_rank0(f"  ✓ All storages resharded with gradients")

    # Verify all grads are DTensors
    for name, param in model.named_parameters():
        if param._local_tensor.numel() > 0:
            assert param.grad is not None, f"{name} missing gradient"
            assert isinstance(param.grad, DTensor), f"{name} gradient should be DTensor"
    print_rank0(f"  ✓ All gradients are DTensors")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED: Nested wrapping test")
    print_rank0(f"{'='*70}")

    torch.cuda.synchronize()
    return True


def run_auto_hooks():
    """Test automatic hook-based unshard/reshard scheduling."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    mesh = init_device_mesh("cuda", (world_size,))

    # Create Transformer with 2 layers
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
    print_rank0(f"Testing AUTOMATIC HOOKS with Transformer, world_size={world_size}")
    print_rank0(f"{'='*70}")

    # Apply nested wrapping (hooks are registered automatically)
    print_rank0(f"\n=== Applying fully_shard_flat with automatic hooks ===")
    layer_storages = []
    for i, layer in enumerate(model.layers):
        storage = fully_shard_flat(layer, mesh)
        layer_storages.append(storage)
        print_rank0(f"  Layer {i}: {len(storage.param_infos)} params, hooks registered")

    root_storage = fully_shard_flat(model, mesh)
    print_rank0(f"  Root: {len(root_storage.param_infos)} params, hooks registered")

    # Verify all params are sharded DTensors initially
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor initially"
    print_rank0(f"  ✓ All parameters are sharded DTensors")

    # === Test 1: Forward with automatic hooks ===
    print_rank0(f"\n=== Test 1: Forward with automatic hooks ===")
    batch_size = 2
    seq_len = args.max_seq_len
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)

    # Just call forward - hooks should handle unshard automatically
    output = model(tokens)
    print_rank0(f"  Forward: input={tuple(tokens.shape)}, output={tuple(output.shape)}")

    # After forward, params are UNSHARDED (we keep them unsharded for backward)
    for name, param in model.named_parameters():
        assert not isinstance(param, DTensor), f"{name} should be unsharded after forward"
    print_rank0(f"  ✓ All parameters are unsharded after forward (ready for backward)")

    # === Test 2: Forward + Backward with automatic hooks ===
    print_rank0(f"\n=== Test 2: Forward + Backward with automatic hooks ===")

    # Clear gradients
    for param in model.parameters():
        param.grad = None

    # Forward + backward - hooks handle everything
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    output = model(tokens)
    loss = output.sum()
    loss.backward()
    print_rank0(f"  Forward + backward complete, loss={loss.item():.4f}")

    # Wait for post_backward callback to complete
    torch.cuda.synchronize()

    # Debug: Check storage states after sync
    print_rank0(f"  Root storage state: {root_storage.state}")
    for i, s in enumerate(layer_storages):
        print_rank0(f"  Layer {i} storage state: {s.state}")

    # Verify params are DTensors with DTensor gradients
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), f"{name} should be DTensor after backward"
        if param._local_tensor.numel() > 0:
            assert param.grad is not None, f"{name} should have gradient"
            assert isinstance(param.grad, DTensor), f"{name} gradient should be DTensor"
    print_rank0(f"  ✓ All parameters are DTensors with DTensor gradients")

    # === Test 3: Multiple forward/backward iterations ===
    print_rank0(f"\n=== Test 3: Multiple iterations ===")
    for iteration in range(3):
        # Clear gradients
        for param in model.parameters():
            param.grad = None

        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
        output = model(tokens)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

        # Verify state after each iteration
        for name, param in model.named_parameters():
            assert isinstance(param, DTensor), f"Iter {iteration}: {name} should be DTensor"

    print_rank0(f"  ✓ 3 iterations completed successfully")

    # === Test 4: Verify multiple consecutive forward+backward cycles ===
    print_rank0(f"\n=== Test 4: Consecutive forward+backward cycles ===")

    for cycle in range(2):
        # Clear gradients
        for param in model.parameters():
            param.grad = None

        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
        output = model(tokens)
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()

        # Verify state after each cycle
        for name, param in model.named_parameters():
            assert isinstance(param, DTensor), f"Cycle {cycle}: {name} should be DTensor"
            if param._local_tensor.numel() > 0:
                assert param.grad is not None, f"Cycle {cycle}: {name} should have gradient"

    print_rank0(f"  ✓ 2 consecutive cycles completed successfully")

    print_rank0(f"\n{'='*70}")
    print_rank0("PASSED: Automatic hooks test")
    print_rank0(f"{'='*70}")

    torch.cuda.synchronize()
    return True


def main():
    """Run tests."""
    rank, world_size = setup_distributed()

    print_rank0(f"Running tests with world_size={world_size}")

    try:
        # Test 1: Basic flat sharding (manual unshard/reshard)
        run()

        # Test 2: Nested wrapping (manual unshard/reshard)
        run_nested()

        # Test 3: Automatic hook-based scheduling
        run_auto_hooks()

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
