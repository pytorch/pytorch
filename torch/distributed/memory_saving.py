# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Memory-saving distributed tensor utilities.

This module provides APIs for reducing per-device memory usage by physically
sharding tensor storage across devices in a process group. Unlike DTensor's
logical sharding (which affects the tensor's logical view), these utilities
partition the underlying storage itself.

Key APIs:
    scatter_tensor_storage: Scatter a single tensor's storage across ranks
    scatter_tensor_group: Scatter a group of tensors across ranks
    MemoryShardedDTensor: DTensor subclass with sharded storage
    TensorGroupStorage: Manage groups of sharded tensors
"""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._memory_sharded import (
    _validate_dtensor_for_storage_sharding,
    BlockStorageShardingSpec,
    MemoryShardedDTensor,
    ShardingBoundary,
    TensorGroupStorage,
)


def scatter_tensor_storage(
    dtensor: DTensor,
    dim: int | list[int] | tuple[int, ...],
    mesh_dim: int | str | list[int | str] | tuple[int | str, ...] | None = None,
    weights: list[int | float] | list[list[int | float]] | None = None,
) -> MemoryShardedDTensor:
    """
    Scatter a DTensor's storage across ranks.

    This function takes a DTensor with replicated data and physically shards its
    underlying storage across devices, reducing per-device memory usage. The
    DTensor must be replicated on the target mesh dimension(s).

    Args:
        dtensor: The input DTensor to shard. Must be replicated on the target
            mesh dimension(s).
        dim: The tensor dimension(s) to shard. Can be a single int for 1D
            sharding, or a list/tuple of ints for multi-dimensional block
            sharding.
        mesh_dim: The mesh dimension(s) to shard across. For single-dim sharding,
            can be an int or str. For multi-dim sharding, should be a list/tuple
            matching the length of dim. If None and dim is a sequence, uses mesh
            dimensions 0, 1, 2, ... (first len(dim) mesh dims).
        weights: Optional weights for weighted sharding. Each weight specifies
            the relative proportion of elements each rank should receive.
            For single-dim sharding: list of weights, one per rank.
            For multi-dim sharding: list of lists, one list per sharded dimension.
            The dimension size must be divisible by the sum of weights.
            Example: weights=[1, 2, 1, 1] means rank 1 gets 2x the elements.

    Returns:
        A MemoryShardedDTensor with storage scattered across ranks.

    Raises:
        ValueError: If dim or mesh_dim is out of range, if mesh_dim doesn't
            exist, if the DTensor is not replicated on the target mesh dim,
            if weights are invalid, or if dimension size is not divisible
            by the sum of weights.

    Example:
        >>> # Single-dim sharding (FSDP-style)
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> param = distribute_tensor(torch.randn(16, 8), mesh, [Replicate()])
        >>> sharded = scatter_tensor_storage(param, dim=0, mesh_dim="dp")
        >>> sharded.shape  # Local shape: (4, 8)
        >>> sharded.full_shape  # Original shape: (16, 8)

        >>> # Weighted sharding
        >>> param = distribute_tensor(torch.randn(20, 8), mesh, [Replicate()])
        >>> sharded = scatter_tensor_storage(param, dim=0, mesh_dim="dp", weights=[1, 2, 1, 1])
        >>> # Rank 1 gets 8 elements (2x), others get 4 each

        >>> # Multi-dim block sharding
        >>> mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("dp", "tp"))
        >>> param = distribute_tensor(
        ...     torch.randn(8, 4), mesh, [Replicate(), Replicate()]
        ... )
        >>> sharded = scatter_tensor_storage(param, dim=[0, 1], mesh_dim=["dp", "tp"])
        >>> sharded.shape  # (2, 2)
        >>> sharded.full_shape  # (8, 4)
    """
    from torch.distributed.tensor.placement_types import Replicate

    device_mesh = dtensor.device_mesh
    ndim = dtensor.ndim

    # Normalize dim to a tuple
    if isinstance(dim, int):
        shard_dims = (dim,)
        is_single_dim = True
    else:
        shard_dims = tuple(dim)
        is_single_dim = False

    # Normalize mesh_dim to a tuple
    if mesh_dim is None:
        # Default: use first len(shard_dims) mesh dimensions
        mesh_dims_input: tuple[int | str, ...] = tuple(range(len(shard_dims)))
    elif isinstance(mesh_dim, (int, str)):
        mesh_dims_input = (mesh_dim,)
    else:
        mesh_dims_input = tuple(mesh_dim)

    # Validate lengths match
    if len(shard_dims) != len(mesh_dims_input):
        raise ValueError(
            f"dim and mesh_dim must have same length, "
            f"got {len(shard_dims)} and {len(mesh_dims_input)}"
        )

    # Normalize negative dims and validate
    normalized_shard_dims = []
    for d in shard_dims:
        if d < 0:
            d = d + ndim
        if d < 0 or d >= ndim:
            raise ValueError(f"dim {d} is out of range for tensor with {ndim} dimensions")
        normalized_shard_dims.append(d)
    shard_dims = tuple(normalized_shard_dims)

    # Resolve mesh_dims to indices and names
    mesh_dim_indices = []
    mesh_dim_names_list = []
    for md in mesh_dims_input:
        if isinstance(md, str):
            names = device_mesh.mesh_dim_names
            if names is None or md not in names:
                raise ValueError(
                    f"mesh_dim '{md}' not found in device mesh. "
                    f"Available dimensions: {names}"
                )
            mesh_dim_names_list.append(md)
            mesh_dim_indices.append(names.index(md))
        else:
            if md < 0 or md >= device_mesh.ndim:
                raise ValueError(
                    f"mesh_dim {md} is out of range for mesh with "
                    f"{device_mesh.ndim} dimensions"
                )
            mesh_dim_indices.append(md)
            names = device_mesh.mesh_dim_names
            mesh_dim_names_list.append(names[md] if names else f"dim_{md}")

    mesh_dim_indices_tuple = tuple(mesh_dim_indices)
    mesh_dim_names_tuple = tuple(mesh_dim_names_list)

    # Validate that the DTensor is replicated on target mesh dimensions
    for mesh_dim_idx in mesh_dim_indices_tuple:
        _validate_dtensor_for_storage_sharding(dtensor, device_mesh, mesh_dim_idx)

    # Normalize and validate weights
    normalized_weights: list[list[float]] | None = None
    if weights is not None:
        # Normalize weights to list of lists (one per sharded dimension)
        if is_single_dim:
            if not isinstance(weights[0], (int, float)):
                raise ValueError(
                    "For single-dim sharding, weights must be a flat list of numbers"
                )
            normalized_weights = [list(weights)]  # type: ignore[arg-type]
        else:
            # Multi-dim: should be list of lists
            if not isinstance(weights[0], (list, tuple)):
                raise ValueError(
                    "For multi-dim sharding, weights must be a list of lists"
                )
            normalized_weights = [list(w) for w in weights]  # type: ignore[union-attr]

        # Validate each dimension's weights
        for dim_idx, (mesh_dim_idx, dim_weights) in enumerate(
            zip(mesh_dim_indices_tuple, normalized_weights)
        ):
            world_size = device_mesh.size(mesh_dim_idx)
            if len(dim_weights) != world_size:
                raise ValueError(
                    f"weights length {len(dim_weights)} must match "
                    f"world_size {world_size} for mesh dimension {mesh_dim_idx}"
                )
            for i, w in enumerate(dim_weights):
                if w <= 0:
                    raise ValueError(
                        f"All weights must be positive, got weight {w} at index {i}"
                    )

    # Get the full tensor data (replicated on all ranks)
    full_tensor = dtensor.to_local()

    # Compute shard sizes and extract local block
    padded_shard_sizes = []
    actual_shard_sizes = []
    local_slices = [slice(None)] * ndim
    all_per_rank_sizes: list[tuple[int, ...]] = []

    for dim_loop_idx, (tensor_dim, mesh_dim_idx) in enumerate(
        zip(shard_dims, mesh_dim_indices_tuple)
    ):
        world_size = device_mesh.size(mesh_dim_idx)
        local_rank = device_mesh.get_local_rank(mesh_dim_idx)
        full_size = full_tensor.size(tensor_dim)

        if normalized_weights is not None:
            # Weighted sharding
            dim_weights = normalized_weights[dim_loop_idx]
            total_weight = sum(dim_weights)

            # Compute per-rank sizes using proportions
            # Validate that each size is a whole number (no truncation)
            per_rank_sizes_float = [full_size * w / total_weight for w in dim_weights]
            per_rank_sizes = []
            for i, size_f in enumerate(per_rank_sizes_float):
                if size_f != int(size_f):
                    raise ValueError(
                        f"Weight {dim_weights[i]} at index {i} does not produce an integer "
                        f"shard size. Computed {size_f} elements, but must be a whole number. "
                        f"Ensure dimension size {full_size} is evenly divisible by weights."
                    )
                per_rank_sizes.append(int(size_f))
            per_rank_sizes = tuple(per_rank_sizes)

            all_per_rank_sizes.append(per_rank_sizes)

            # Compute this rank's start/end
            start_idx = sum(per_rank_sizes[:local_rank])
            actual_shard_size = per_rank_sizes[local_rank]
            end_idx = start_idx + actual_shard_size
            padded_shard_size = max(per_rank_sizes)  # For all-gather padding
        else:
            # Even sharding (existing logic)
            padded_shard_size = (full_size + world_size - 1) // world_size
            start_idx = local_rank * padded_shard_size
            end_idx = min(start_idx + padded_shard_size, full_size)
            actual_shard_size = max(0, end_idx - start_idx)

        padded_shard_sizes.append(padded_shard_size)
        actual_shard_sizes.append(actual_shard_size)

        # Build slice for this dimension
        if actual_shard_size > 0:
            local_slices[tensor_dim] = slice(start_idx, end_idx)
        else:
            local_slices[tensor_dim] = slice(0, 0)

    # Extract local shard/block
    local_shard = full_tensor[tuple(local_slices)].contiguous()

    # Preserve requires_grad
    if full_tensor.requires_grad:
        local_shard = local_shard.requires_grad_(True)

    # Create storage sharding spec
    storage_spec = BlockStorageShardingSpec(
        orig_size=full_tensor.size(),
        orig_stride=full_tensor.stride(),
        shard_dims=shard_dims,
        mesh_dims=mesh_dim_names_tuple,
        padded_shard_sizes=tuple(padded_shard_sizes),
        actual_shard_sizes=tuple(actual_shard_sizes),
        mesh_dim_indices=mesh_dim_indices_tuple,
        per_rank_shard_sizes=tuple(all_per_rank_sizes) if all_per_rank_sizes else None,
    )

    # Use the first mesh dimension's process group as primary
    primary_pg = device_mesh.get_group(mesh_dim_indices_tuple[0])

    # Create placements - replicated on all dimensions
    placements = tuple(Replicate() for _ in range(device_mesh.ndim))

    # For scatter_tensor_storage, the global shape is the full tensor shape (no TP involved)
    # We pass this as global_tensor_meta so that full_tensor() returns the correct shape
    return MemoryShardedDTensor._create(
        local_tensor=local_shard,
        device_mesh=device_mesh,
        storage_spec=storage_spec,
        process_group=primary_pg,
        placements=placements,
        global_tensor_meta=dtensor._spec.tensor_meta,
    )


def scatter_tensor_group(
    tensors: list[torch.Tensor | DTensor],
    device_mesh: DeviceMesh,
    mesh_dim: int | str,
    boundary: ShardingBoundary = ShardingBoundary.ELEMENT,
    weights: list[int | float] | None = None,
) -> list[MemoryShardedDTensor]:
    """
    Scatter a group of tensors across ranks.

    This convenience function creates a TensorGroupStorage and shards the tensors.

    Args:
        tensors: List of tensors to scatter. Can be plain torch.Tensor or
            DTensor. DTensor inputs must have Replicate placement on the
            target mesh dimension.
        device_mesh: The DeviceMesh for distributed operations.
        mesh_dim: The mesh dimension (name or index) to shard across.
        boundary: Sharding boundary - ShardingBoundary.ELEMENT (FSDP v1 style)
            or ShardingBoundary.TENSOR (whole tensors per rank).
        weights: Optional weights for weighted sharding. Each weight specifies
            the relative proportion each rank should receive.

    Returns:
        List of MemoryShardedDTensor, one per input tensor.

    Example:
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> tensors = [torch.randn(8, 4), torch.randn(10, 6)]

        >>> # Element boundary (FSDP v1 style)
        >>> sharded = scatter_tensor_group(tensors, mesh, "dp", ShardingBoundary.ELEMENT)

        >>> # Tensor boundary (each tensor whole on one rank)
        >>> sharded = scatter_tensor_group(tensors, mesh, "dp", ShardingBoundary.TENSOR)

        >>> # Weighted element boundary
        >>> sharded = scatter_tensor_group(
        ...     tensors, mesh, "dp", ShardingBoundary.ELEMENT, weights=[1, 2, 1, 1]
        ... )
    """
    group = TensorGroupStorage(tensors, device_mesh, mesh_dim, boundary=boundary, weights=weights)
    return group.shard()


__all__ = [
    "scatter_tensor_storage",
    "scatter_tensor_group",
    "MemoryShardedDTensor",
    "ShardingBoundary",
    "TensorGroupStorage",
]

# Set proper module names for public APIs
scatter_tensor_storage.__module__ = "torch.distributed.memory_saving"
scatter_tensor_group.__module__ = "torch.distributed.memory_saving"
MemoryShardedDTensor.__module__ = "torch.distributed.memory_saving"
ShardingBoundary.__module__ = "torch.distributed.memory_saving"
TensorGroupStorage.__module__ = "torch.distributed.memory_saving"
