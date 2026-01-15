# Copyright (c) Meta Platforms, Inc. and affiliates
"""
MemoryShardedDTensor: A DTensor variant that shards storage across devices.

This module provides a memory-efficient DTensor implementation where the tensor's
storage is sharded across devices in a process group, reducing per-device memory
usage. Unlike regular DTensor sharding which affects the logical tensor view,
MemoryShardedDTensor physically partitions the underlying storage.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


class ShardingBoundary(Enum):
    """
    Specifies how tensors are distributed across ranks in TensorGroupStorage.

    ELEMENT: Elements are distributed across ranks (FSDP v1 style).
        All tensors are flattened, concatenated, and the resulting buffer
        is sharded by elements. A single tensor may span multiple ranks.

    TENSOR: Whole tensors are distributed across ranks (Muon style).
        Each tensor stays complete on one rank. No tensor is split.
    """

    ELEMENT = "element"
    TENSOR = "tensor"


@dataclass
class BlockStorageShardingSpec:
    """
    Unified specification for storage sharding (single or multi-dimensional).

    This spec supports both single-dimension sharding (FSDP v2 style) and
    multi-dimensional block sharding. Single-dim sharding is a special case
    where len(shard_dims) == 1.

    Attributes:
        orig_size: Original (full) tensor size before sharding.
        orig_stride: Original tensor stride before sharding.
        shard_dims: Tuple of tensor dimensions that are sharded.
        mesh_dims: Tuple of mesh dimension names corresponding to shard_dims.
        padded_shard_sizes: Per-dimension padded shard sizes for even division.
        actual_shard_sizes: Per-dimension actual shard sizes on this rank.
        mesh_dim_indices: Cached mesh dimension indices for performance.
        per_rank_shard_sizes: Per-rank shard sizes for weighted sharding.
            Shape: (num_shard_dims, world_size). None for even sharding.
    """

    orig_size: torch.Size
    orig_stride: tuple[int, ...]
    shard_dims: tuple[int, ...]
    mesh_dims: tuple[str, ...]
    padded_shard_sizes: tuple[int, ...]
    actual_shard_sizes: tuple[int, ...]
    mesh_dim_indices: tuple[int, ...]
    per_rank_shard_sizes: Optional[tuple[tuple[int, ...], ...]] = None


@dataclass
class ShardParamInfo:
    """
    Tracks how a parameter maps to a rank's shard in flattened storage mode.

    In FSDP v1-style flattening, multiple parameters are concatenated into a
    single flat buffer and sharded across ranks. This class tracks which portion
    of each original parameter is present in each rank's shard.

    Attributes:
        in_shard: Whether any part of this parameter is in this rank's shard.
        offset_in_shard: Start offset within the local shard (None if not in shard).
        numel_in_shard: Number of elements from this param in the shard (None if not in shard).
        intra_param_start: Start index within the original parameter (None if not in shard).
        intra_param_end: End index (exclusive) within the original parameter (None if not in shard).
    """

    in_shard: bool
    offset_in_shard: Optional[int] = None
    numel_in_shard: Optional[int] = None
    intra_param_start: Optional[int] = None
    intra_param_end: Optional[int] = None


@dataclass
class FlattenedStorageShardingSpec:
    """
    Specification for FSDP v1-style flattened storage sharding.

    In this mode, multiple parameters are flattened to 1D, concatenated into
    a single buffer, and that buffer is sharded across ranks. Each rank holds
    a contiguous slice of the flattened buffer.

    Attributes:
        param_shapes: Original shapes of all parameters in the group.
        param_strides: Original strides of all parameters.
        param_numels: Number of elements in each parameter.
        total_numel: Total number of elements across all parameters.
        padded_total_numel: Total elements after padding for even division.
        mesh_dim: The mesh dimension name used for sharding.
        local_offset: Offset into the (unpadded) concatenated buffer for this rank.
        local_numel: Number of elements in this rank's shard.
        shard_param_infos: Per-parameter shard mapping information.
        param_index: Which parameter this MemoryShardedDTensor instance represents.
    """

    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_numels: tuple[int, ...]
    total_numel: int
    padded_total_numel: int
    mesh_dim: str
    local_offset: int
    local_numel: int
    shard_param_infos: tuple[ShardParamInfo, ...]
    param_index: int
    # Per-rank shard sizes for weighted sharding (None for even sharding)
    per_rank_shard_sizes: Optional[tuple[int, ...]] = None


@dataclass
class TensorGroupShardingSpec:
    """
    Specification for tensor-group sharding where each tensor is fully on one rank.

    In this mode, a group of tensors is distributed across ranks such that each
    tensor stays WHOLE on one rank (not split across ranks). This differs from
    FlattenedStorageShardingSpec where elements can span ranks.

    Distribution is contiguous: first N/world_size tensors to rank 0, etc.

    Attributes:
        param_shapes: Original shapes of all tensors in the group.
        param_strides: Original strides of all tensors.
        param_numels: Number of elements in each tensor.
        total_params: Total number of tensors in the group.
        mesh_dim: The mesh dimension name used for sharding.
        mesh_dim_idx: The mesh dimension index.
        param_to_rank: Mapping from param_index to the owning rank.
        rank_to_params: Mapping from rank to list of owned param indices.
        param_index: Which tensor this MemoryShardedDTensor instance represents.
        owns_tensor: Whether this rank owns this specific tensor.
    """

    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_numels: tuple[int, ...]
    total_params: int
    mesh_dim: str
    mesh_dim_idx: int
    param_to_rank: tuple[int, ...]
    rank_to_params: tuple[tuple[int, ...], ...]
    param_index: int
    owns_tensor: bool


# Union type for storage specs
StorageSpec = Union[
    BlockStorageShardingSpec, FlattenedStorageShardingSpec, TensorGroupShardingSpec
]


def _validate_dtensor_for_storage_sharding(
    dtensor: DTensor,
    device_mesh: DeviceMesh,
    mesh_dim_idx: int,
) -> None:
    """
    Validate that a DTensor can be storage-sharded on the given mesh dimension.

    FSDP-style storage sharding requires the data to be replicated on the target
    mesh dimension. If data is already sharded or has pending reductions, we
    cannot properly distribute the storage.

    Args:
        dtensor: The DTensor to validate.
        device_mesh: The target device mesh for storage sharding.
        mesh_dim_idx: The mesh dimension index to shard on.

    Raises:
        ValueError: If the DTensor cannot be storage-sharded on the given mesh dim.
    """
    from torch.distributed.tensor.placement_types import Partial, Shard

    # Check same mesh
    if dtensor.device_mesh != device_mesh:
        raise ValueError(
            f"DTensor device_mesh {dtensor.device_mesh} does not match "
            f"target device_mesh {device_mesh}"
        )

    # Check placement on target mesh_dim is Replicate (not Shard or Partial)
    placement = dtensor.placements[mesh_dim_idx]
    if isinstance(placement, Shard):
        raise ValueError(
            f"Cannot shard storage on mesh_dim {mesh_dim_idx}: "
            f"DTensor already has Shard placement on this dimension"
        )
    if isinstance(placement, Partial):
        raise ValueError(
            f"Cannot shard storage on mesh_dim {mesh_dim_idx}: "
            f"DTensor has Partial placement (reduction pending)"
        )


class MemoryShardedDTensor(DTensor):
    """
    A DTensor subclass that physically shards storage across devices.

    MemoryShardedDTensor reduces per-device memory by partitioning the tensor's
    storage along a specified dimension. Each device holds only its local shard.
    The full tensor can be reconstructed via the unshard() method which performs
    an all-gather collective.

    This is useful for FSDP-style memory savings where parameters are sharded
    during forward/backward and gathered only when needed.

    Attributes:
        _storage_spec: StorageSpec describing the sharding configuration.
        _process_group: The process group used for collective operations.
        _padded_local: 1D flattened tensor with padding for even all-gather.
        _flat_buffer: For flattened mode, the shared flat buffer across params.
    """

    _storage_spec: StorageSpec
    # TODO(ailzhang): do we need to store the process group or can we infer it from the device mesh?
    _process_group: dist.ProcessGroup
    _padded_local: torch.Tensor
    _flat_buffer: Optional[torch.Tensor]
    __slots__ = ["_storage_spec", "_process_group", "_padded_local", "_flat_buffer"]

    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
        *,
        requires_grad: bool,
        flat_buffer: Optional[torch.Tensor] = None,
    ) -> "MemoryShardedDTensor":
        # Create the DTensor base
        r = super().__new__(
            cls,
            local_tensor,
            spec,
            requires_grad=requires_grad,
        )
        r._storage_spec = storage_spec
        r._process_group = process_group
        r._padded_local = padded_local
        r._flat_buffer = flat_buffer
        return r

    def __init__(
        self,
        local_tensor: torch.Tensor,
        spec: "DTensor._spec",  # type: ignore[name-defined]
        storage_spec: StorageSpec,
        process_group: dist.ProcessGroup,
        padded_local: torch.Tensor,
        *,
        requires_grad: bool,
        flat_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

    def __repr__(self) -> str:
        if self.is_flattened_mode():
            spec = self._storage_spec
            return (
                f"MemoryShardedDTensor(flattened_mode=True, "
                f"local_shape={self.shape}, "
                f"full_shape={self.full_shape}, "
                f"param_index={spec.param_index}, "
                f"device_mesh={self._spec.mesh})"
            )
        elif self.is_tensor_group_mode():
            spec = self._storage_spec
            return (
                f"MemoryShardedDTensor(tensor_group_mode=True, "
                f"local_shape={self.shape}, "
                f"full_shape={self.full_shape}, "
                f"param_index={spec.param_index}, "
                f"owns_tensor={spec.owns_tensor}, "
                f"device_mesh={self._spec.mesh})"
            )
        spec = self._storage_spec
        return (
            f"MemoryShardedDTensor(local_shape={self.shape}, "
            f"full_shape={self.full_shape}, "
            f"shard_dims={spec.shard_dims}, "
            f"device_mesh={self._spec.mesh})"
        )

    def is_flattened_mode(self) -> bool:
        """
        Returns True if this tensor uses flattened storage sharding (FSDP v1 style).
        """
        return isinstance(self._storage_spec, FlattenedStorageShardingSpec)

    def is_tensor_group_mode(self) -> bool:
        """
        Returns True if this tensor uses tensor-group sharding.

        In tensor-group mode, each tensor in the group is fully on one rank
        (not split across ranks).
        """
        return isinstance(self._storage_spec, TensorGroupShardingSpec)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        """
        Intercept tensor operations and raise an error.

        MemoryShardedDTensor does not support direct tensor operations. Users must
        either:
        1. Call .local() to get the local shard, operate on it, then use
           from_local_shard() to create a new MemoryShardedDTensor
        2. Call .unshard() to get the full tensor, operate on it, then use
           scatter_tensor_storage() to re-shard

        This explicit error prevents silent loss of sharding information that would
        occur if operations returned regular DTensor.
        """
        raise NotImplementedError(
            f"MemoryShardedDTensor does not support tensor operations (got {func.__name__}). "
            f"To perform operations:\n"
            f"  1. Use .local() to get the local shard, operate on it, then use "
            f"MemoryShardedDTensor.from_local_shard() to create a new MemoryShardedDTensor\n"
            f"  2. Or use .unshard() to reconstruct the full tensor, operate on it, "
            f"then use scatter_tensor_storage() to re-shard"
        )

    def _raise_op_not_supported(self, op_name: str):
        """Raise error for unsupported operations."""
        raise NotImplementedError(
            f"MemoryShardedDTensor does not support tensor operations (got {op_name}). "
            f"To perform operations:\n"
            f"  1. Use .local() to get the local shard, operate on it, then use "
            f"MemoryShardedDTensor.from_local_shard() to create a new MemoryShardedDTensor\n"
            f"  2. Or use .unshard() to reconstruct the full tensor, operate on it, "
            f"then use scatter_tensor_storage() to re-shard"
        )

    # TODO(ailzhang): Currently only dunder methods (__add__, __mul__, etc.) are
    # blocked. Other tensor ops like torch.relu(msdt), msdt.sum(), msdt.view()
    # still go through DTensor's C++ fast path and silently return regular DTensor.
    # To block ALL ops, we need to either:
    # 1. Not inherit from DTensor (use composition) so __torch_function__ works
    # 2. Make C++ changes to check for MemoryShardedDTensor in the dispatch path
    # Override arithmetic operations to raise explicit errors
    def __add__(self, other):
        self._raise_op_not_supported("__add__")

    def __radd__(self, other):
        self._raise_op_not_supported("__radd__")

    def __sub__(self, other):
        self._raise_op_not_supported("__sub__")

    def __rsub__(self, other):
        self._raise_op_not_supported("__rsub__")

    def __mul__(self, other):
        self._raise_op_not_supported("__mul__")

    def __rmul__(self, other):
        self._raise_op_not_supported("__rmul__")

    def __truediv__(self, other):
        self._raise_op_not_supported("__truediv__")

    def __rtruediv__(self, other):
        self._raise_op_not_supported("__rtruediv__")

    def __floordiv__(self, other):
        self._raise_op_not_supported("__floordiv__")

    def __rfloordiv__(self, other):
        self._raise_op_not_supported("__rfloordiv__")

    def __matmul__(self, other):
        self._raise_op_not_supported("__matmul__")

    def __rmatmul__(self, other):
        self._raise_op_not_supported("__rmatmul__")

    def __pow__(self, other):
        self._raise_op_not_supported("__pow__")

    def __rpow__(self, other):
        self._raise_op_not_supported("__rpow__")

    def __neg__(self):
        self._raise_op_not_supported("__neg__")

    def __pos__(self):
        self._raise_op_not_supported("__pos__")

    def __abs__(self):
        self._raise_op_not_supported("__abs__")

    @classmethod
    def _create(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        storage_spec: StorageSpec,
        process_group: dist.ProcessGroup,
        placements: tuple,
        padded_local: Optional[torch.Tensor] = None,
        flat_buffer: Optional[torch.Tensor] = None,
        global_tensor_meta: Optional["TensorMeta"] = None,
    ) -> "MemoryShardedDTensor":
        """
        Factory method to create a MemoryShardedDTensor.

        Args:
            local_tensor: The local shard of the tensor on this rank.
            device_mesh: The DeviceMesh for this distributed tensor.
            storage_spec: StorageSpec describing sharding configuration.
            process_group: Process group for collective operations.
            placements: DTensor placements tuple.
            padded_local: Optional pre-computed 1D padded tensor. If None,
                will be computed from local_tensor and storage_spec.
            flat_buffer: For flattened mode, the shared flat buffer.
            global_tensor_meta: Optional global tensor metadata. If provided,
                used for the DTensor spec to represent the full tensor shape.
                Required for TP+FSDP case where the local tensor has been
                sharded by both TP and FSDP.

        Returns:
            A new MemoryShardedDTensor instance.
        """
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

        # Compute padded_local if not provided (only for non-flattened mode)
        if padded_local is None:
            if isinstance(storage_spec, BlockStorageShardingSpec):
                padded_local = cls._compute_padded_local(local_tensor, storage_spec)
            else:
                # For flattened mode, use flat_buffer as padded_local
                padded_local = (
                    flat_buffer if flat_buffer is not None else local_tensor.view(-1)
                )

        # Use provided global_tensor_meta or compute from local tensor
        # Note: For TP+FSDP case, global_tensor_meta should be provided to
        # represent the full tensor shape before any sharding.
        if global_tensor_meta is not None:
            tensor_meta = global_tensor_meta
        else:
            # Fall back to local tensor's metadata (for backward compatibility)
            tensor_meta = TensorMeta(
                shape=local_tensor.shape,
                stride=local_tensor.stride(),
                dtype=local_tensor.dtype,
            )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return cls(
            local_tensor,
            dtensor_spec,
            storage_spec,
            process_group,
            padded_local,
            requires_grad=local_tensor.requires_grad,
            flat_buffer=flat_buffer,
        )

    @property
    def full_shape(self) -> torch.Size:
        """
        Returns the original (full) shape of the tensor before sharding.
        """
        if self.is_flattened_mode() or self.is_tensor_group_mode():
            spec = self._storage_spec
            return spec.param_shapes[spec.param_index]
        return self._storage_spec.orig_size

    @property
    def shape(self) -> torch.Size:
        """
        Returns the local (sharded) shape of the tensor.

        Unlike regular DTensor where shape returns the global shape,
        MemoryShardedDTensor returns the local tensor shape because
        the storage is physically sharded across devices.

        Use full_shape to get the original (unsharded) shape.
        """
        return self._local_tensor.shape

    def size(self, dim: Optional[int] = None) -> int | torch.Size:
        """
        Returns the local (sharded) size of the tensor.

        Args:
            dim: If specified, returns the size of that dimension.
                 If None, returns the full local shape.

        Returns:
            Size of the specified dimension, or local shape if dim is None.
        """
        if dim is None:
            return self.shape
        return self.shape[dim]

    def full_size(self, dim: Optional[int] = None) -> int | torch.Size:
        """
        Returns the original (full) size of the tensor.

        Args:
            dim: If specified, returns the size of that dimension.
                 If None, returns the full shape.

        Returns:
            Size of the specified dimension, or full shape if dim is None.
        """
        full_shape = self.full_shape
        if dim is None:
            return full_shape
        return full_shape[dim]

    def local(self) -> torch.Tensor:
        """
        Returns the local shard as a torch.Tensor.

        Returns:
            The underlying local tensor shard.
        """
        return self._local_tensor

    @property
    def storage_spec(self) -> StorageSpec:
        """
        Returns the storage sharding specification.
        """
        return self._storage_spec

    @property
    def process_group(self) -> dist.ProcessGroup:
        """
        Returns the process group used for collective operations.
        """
        return self._process_group

    @staticmethod
    def _compute_padded_local(
        local_tensor: torch.Tensor,
        storage_spec: BlockStorageShardingSpec,
    ) -> torch.Tensor:
        """
        Compute the 1D padded tensor from a local shard and storage spec.

        For multi-dimensional block sharding, pads each sharded dimension to
        its padded_shard_size, then flattens to 1D.

        Args:
            local_tensor: The local shard tensor.
            storage_spec: BlockStorageShardingSpec with padding info.

        Returns:
            1D flattened tensor with padding on all sharded dimensions.
        """
        # Check if any dimension needs padding
        needs_padding = any(
            actual != padded
            for actual, padded in zip(
                storage_spec.actual_shard_sizes,
                storage_spec.padded_shard_sizes,
            )
        )

        if not needs_padding:
            # No padding needed - just flatten
            return local_tensor.view(-1)

        # Build padded shape - local_tensor.shape has actual_shard_sizes on sharded dims
        padded_shape = list(local_tensor.shape)
        for i, shard_dim in enumerate(storage_spec.shard_dims):
            padded_shape[shard_dim] = storage_spec.padded_shard_sizes[i]

        # Create padded buffer and copy actual data
        padded = local_tensor.new_zeros(padded_shape)

        # Build slices to copy actual data
        slices = [slice(None)] * len(padded_shape)
        for i, shard_dim in enumerate(storage_spec.shard_dims):
            slices[shard_dim] = slice(0, storage_spec.actual_shard_sizes[i])

        padded[tuple(slices)] = local_tensor

        return padded.view(-1)

    def detach(self) -> "MemoryShardedDTensor":
        """
        Returns a detached MemoryShardedDTensor.

        This is required for nn.Parameter compatibility - the detach() method
        must return an instance of the same type.

        Returns:
            A new MemoryShardedDTensor with detached local tensor.
        """
        detached_local = self._local_tensor.detach()
        detached_padded = self._padded_local.detach()
        return self._create(
            local_tensor=detached_local,
            device_mesh=self._spec.mesh,
            storage_spec=self._storage_spec,
            process_group=self._process_group,
            placements=self._spec.placements,
            padded_local=detached_padded,
            global_tensor_meta=self._spec.tensor_meta,
        )

    def _get_padded_local(self) -> torch.Tensor:
        """
        Returns the local tensor padded to the padded_shard_sizes as ND tensor.

        For uneven sharding, some ranks may have smaller shards than the
        padded size. This method returns an ND view of the padded local tensor
        for operations that need multi-dimensional access (like unshard).

        Returns:
            Local tensor padded on all sharded dimensions.
        """
        spec = self._storage_spec
        local_tensor = self._local_tensor

        # Build padded shape from local tensor shape with padded shard sizes
        padded_shape = list(local_tensor.shape)
        for i, shard_dim in enumerate(spec.shard_dims):
            padded_shape[shard_dim] = spec.padded_shard_sizes[i]

        return self._padded_local.view(padded_shape)

    def get_all_gather_input(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Returns a 1D flattened tensor suitable for all-gather collective operations.

        This method is used by FSDP to get the input tensor for batched all-gather.
        Returns the pre-computed padded tensor for O(1) access.

        For flattened mode, returns the shared flat buffer (single all-gather for
        all params in the group).

        Args:
            dtype: If provided, convert the tensor to this dtype before returning.
                   Used for mixed precision training where storage dtype differs
                   from compute dtype.

        Returns:
            A 1D flattened plain torch.Tensor (not DTensor) containing the padded
            local shard, suitable for passing to all-gather collectives.

        Example:
            >>> sharded = distribute_storage(dtensor, dim=0, mesh_dim="dp")
            >>> all_gather_input = sharded.get_all_gather_input(torch.float16)
            >>> # Use all_gather_input in batched all-gather collective
        """
        if self.is_flattened_mode():
            # In flattened mode, return the shared flat buffer
            result = self._flat_buffer
        else:
            # _padded_local is already 1D and padded - O(1) access
            result = self._padded_local

        # Apply dtype conversion if needed
        if dtype is not None and result.dtype != dtype:
            result = result.to(dtype)

        return result

    @classmethod
    def from_local_shard(
        cls,
        local_shard: torch.Tensor,
        full_shape: torch.Size,
        shard_dim: int,
        device_mesh: DeviceMesh,
        mesh_dim: int | str,
        *,
        requires_grad: bool = False,
        placements: tuple | None = None,
        padded_local: Optional[torch.Tensor] = None,
        global_tensor_meta: Optional["TensorMeta"] = None,
        weights: list[int | float] | None = None,
    ) -> "MemoryShardedDTensor":
        """
        Create a MemoryShardedDTensor from an already-sharded local tensor.

        This factory method is used by FSDP when it has already computed the
        local shard and needs to wrap it in a MemoryShardedDTensor. Unlike
        distribute_storage() which shards a full tensor, this method takes
        a pre-sharded local tensor.

        Args:
            local_shard: The local shard tensor on this rank.
            full_shape: The original (full) shape of the tensor before sharding.
            shard_dim: The dimension along which the tensor is sharded.
            device_mesh: The DeviceMesh for this distributed tensor.
            mesh_dim: The mesh dimension (name or index) used for sharding.
            requires_grad: Whether the tensor requires gradient computation.
            placements: Optional DTensor placements tuple. If None, defaults to
                all-Replicate placements. For TP+FSDP case, pass the combined
                SPMD placements (e.g., (Shard(dim), Shard(dim)) for TP sharding).
            padded_local: Optional pre-computed 1D padded tensor. If None,
                will be computed from local_shard.
            global_tensor_meta: Optional global tensor metadata (shape, stride, dtype)
                for the full tensor before any sharding. For TP+FSDP case, this should
                be the original tensor's metadata before TP and FSDP sharding.
            weights: Optional weights for weighted sharding. Each weight specifies
                the relative proportion of elements each rank should receive.
                The dimension size must be divisible by the sum of weights.

        Returns:
            A MemoryShardedDTensor wrapping the local shard.

        Example:
            >>> # FSDP has already computed the local shard
            >>> local_shard = full_param.narrow(0, start, length).contiguous()
            >>> sharded = MemoryShardedDTensor.from_local_shard(
            ...     local_shard=local_shard,
            ...     full_shape=full_param.shape,
            ...     shard_dim=0,
            ...     device_mesh=mesh,
            ...     mesh_dim="dp",
            ... )
        """
        from torch.distributed.tensor.placement_types import Replicate

        # Resolve mesh_dim to index and name
        if isinstance(mesh_dim, str):
            mesh_dim_names = device_mesh.mesh_dim_names
            if mesh_dim_names is None or mesh_dim not in mesh_dim_names:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in device mesh. "
                    f"Available dimensions: {mesh_dim_names}"
                )
            mesh_dim_name = mesh_dim
            mesh_dim_idx = mesh_dim_names.index(mesh_dim)
        else:
            mesh_dim_idx = mesh_dim
            if mesh_dim_idx < 0 or mesh_dim_idx >= device_mesh.ndim:
                raise ValueError(
                    f"mesh_dim {mesh_dim_idx} is out of range for mesh with "
                    f"{device_mesh.ndim} dimensions"
                )
            mesh_dim_names = device_mesh.mesh_dim_names
            mesh_dim_name = (
                mesh_dim_names[mesh_dim_idx] if mesh_dim_names else "default"
            )

        # Get process group and world size
        process_group = device_mesh.get_group(mesh_dim_idx)
        world_size = device_mesh.size(mesh_dim_idx)

        # Compute shard sizes
        full_size_on_dim = full_shape[shard_dim]
        actual_shard_size = local_shard.size(shard_dim)
        per_rank_shard_sizes: Optional[tuple[tuple[int, ...], ...]] = None

        if weights is not None:
            # Validate weights
            if len(weights) != world_size:
                raise ValueError(
                    f"weights length {len(weights)} must match "
                    f"world_size {world_size}"
                )
            for i, w in enumerate(weights):
                if w <= 0:
                    raise ValueError(
                        f"All weights must be positive, got weight {w} at index {i}"
                    )

            total_weight = sum(weights)

            # Compute per-rank sizes using proportions
            # Validate that each size is a whole number (no truncation)
            per_rank_sizes_float = [full_size_on_dim * w / total_weight for w in weights]
            per_rank_sizes = []
            for i, size_f in enumerate(per_rank_sizes_float):
                if size_f != int(size_f):
                    raise ValueError(
                        f"Weight {weights[i]} at index {i} does not produce an integer "
                        f"shard size. Computed {size_f} elements, but must be a whole number. "
                        f"Ensure dimension size {full_size_on_dim} is evenly divisible by weights."
                    )
                per_rank_sizes.append(int(size_f))
            per_rank_sizes = tuple(per_rank_sizes)

            per_rank_shard_sizes = (per_rank_sizes,)
            padded_shard_size = max(per_rank_sizes)
        else:
            # Even sharding (existing logic)
            padded_shard_size = (full_size_on_dim + world_size - 1) // world_size

        # Compute original stride (assume contiguous layout for full tensor)
        orig_stride = []
        stride = 1
        for i in range(len(full_shape) - 1, -1, -1):
            orig_stride.insert(0, stride)
            stride *= full_shape[i]
        orig_stride = tuple(orig_stride)

        # Create storage sharding spec (single-dim is a special case of block sharding)
        storage_spec = BlockStorageShardingSpec(
            orig_size=full_shape,
            orig_stride=orig_stride,
            shard_dims=(shard_dim,),
            mesh_dims=(mesh_dim_name,),
            padded_shard_sizes=(padded_shard_size,),
            actual_shard_sizes=(actual_shard_size,),
            mesh_dim_indices=(mesh_dim_idx,),
            per_rank_shard_sizes=per_rank_shard_sizes,
        )

        # Use provided placements or default to all-Replicate
        if placements is None:
            placements = tuple(Replicate() for _ in range(device_mesh.ndim))

        # Ensure requires_grad is set correctly
        if requires_grad and not local_shard.requires_grad:
            local_shard = local_shard.requires_grad_(True)

        return cls._create(
            local_tensor=local_shard,
            device_mesh=device_mesh,
            storage_spec=storage_spec,
            process_group=process_group,
            placements=placements,
            padded_local=padded_local,
            global_tensor_meta=global_tensor_meta,
        )

    def unshard(self) -> DTensor:
        """
        Reconstruct the full tensor via all-gather collective.

        Performs an all-gather operation to collect all shards from all ranks
        in the process group, then reconstructs the original tensor shape.

        For flattened mode, this unshards this specific parameter by all-gathering
        the shared flat buffer and extracting this parameter's view.

        Returns:
            A DTensor containing the full (unsharded) tensor data replicated
            across all ranks.

        Example:
            >>> sharded = distribute_storage(dtensor, dim=0, mesh_dim="dp")
            >>> sharded.shape  # (4, 8) - local shard
            >>> full = sharded.unshard()
            >>> full.shape  # (16, 8) - full tensor
        """
        if self.is_flattened_mode():
            return self._unshard_flattened()
        elif self.is_tensor_group_mode():
            return self._unshard_tensor_group()
        return self._unshard_per_param()

    def full_tensor(
        self, *, grad_placements: Optional[tuple] = None
    ) -> torch.Tensor:
        """
        Return the full tensor by unsharding the storage and then handling
        any remaining DTensor placements (e.g., TP sharding).

        This overrides DTensor.full_tensor() to first unshard the storage via
        all-gather, then delegates to DTensor.full_tensor() to handle any
        remaining placements like tensor parallelism.

        Returns:
            The full (unsharded) tensor as a plain torch.Tensor.
        """
        # First unshard the storage (FSDP sharding)
        unsharded = self.unshard()
        # Then handle any remaining placements (e.g., TP sharding)
        return unsharded.full_tensor(grad_placements=grad_placements)

    def _unshard_flattened(self) -> DTensor:
        """
        Unshard a parameter in flattened storage mode.

        All-gathers the shared flat buffer, then extracts this parameter's
        portion and reshapes it to the original shape.
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        spec = self._storage_spec
        world_size = dist.get_world_size(self._process_group)

        # Get the shared flat buffer
        flat_buffer = self._flat_buffer

        # Detach for all-gather to avoid autograd issues
        orig_requires_grad = flat_buffer.requires_grad
        if orig_requires_grad:
            flat_buffer = flat_buffer.detach()

        if spec.per_rank_shard_sizes is not None:
            # Weighted sharding: variable sizes per rank
            per_rank_sizes = spec.per_rank_shard_sizes
            max_size = max(per_rank_sizes)

            # Pad local buffer to max_size for uniform all-gather
            if flat_buffer.numel() < max_size:
                padded_buffer = flat_buffer.new_zeros(max_size)
                padded_buffer[: flat_buffer.numel()] = flat_buffer
            else:
                padded_buffer = flat_buffer

            # All-gather padded buffers
            gathered = padded_buffer.new_empty(max_size * world_size)
            dist.all_gather_into_tensor(gathered, padded_buffer, group=self._process_group)

            # Extract actual contributions from each rank
            chunks = []
            for rank_idx, actual_size in enumerate(per_rank_sizes):
                start = rank_idx * max_size
                end = start + actual_size
                chunks.append(gathered[start:end])
            gathered_buffer = torch.cat(chunks, dim=0)
        else:
            # Even sharding: all ranks have same size
            gathered_buffer = flat_buffer.new_empty(flat_buffer.numel() * world_size)
            dist.all_gather_into_tensor(
                gathered_buffer,
                flat_buffer,
                group=self._process_group,
            )

            # Slice to remove padding
            gathered_buffer = gathered_buffer[: spec.total_numel]

        # Extract this parameter's portion
        param_idx = spec.param_index
        offset = sum(spec.param_numels[:param_idx])
        numel = spec.param_numels[param_idx]
        param_data = gathered_buffer[offset : offset + numel]

        # Reshape to original shape
        param_shape = spec.param_shapes[param_idx]
        param_tensor = param_data.view(param_shape).contiguous()

        # Preserve requires_grad
        if orig_requires_grad:
            param_tensor = param_tensor.requires_grad_(True)

        # Create DTensor with Replicate placement
        device_mesh = self._spec.mesh
        placements = tuple(Replicate() for _ in range(device_mesh.ndim))

        tensor_meta = TensorMeta(
            shape=param_tensor.shape,
            stride=param_tensor.stride(),
            dtype=param_tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return DTensor(
            param_tensor,
            dtensor_spec,
            requires_grad=param_tensor.requires_grad,
        )

    def _unshard_tensor_group(self) -> DTensor:
        """
        Unshard a tensor in tensor-group mode via broadcast.

        In tensor-group mode, each tensor is fully on one rank. Unsharding
        broadcasts the tensor from its owning rank to all ranks.
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        spec = self._storage_spec
        device_mesh = self._spec.mesh
        owning_rank = spec.param_to_rank[spec.param_index]
        shape = spec.param_shapes[spec.param_index]
        stride = spec.param_strides[spec.param_index]

        # Get or create tensor for broadcast
        if spec.owns_tensor:
            # This rank owns the tensor - use local data
            tensor = self._local_tensor.clone()
        else:
            # This rank doesn't own the tensor - create empty buffer
            tensor = self._local_tensor.new_zeros(shape)

        # Broadcast from owning rank (use group_src since owning_rank is mesh-local rank)
        dist.broadcast(tensor, group=self._process_group, group_src=owning_rank)

        # Ensure correct shape and contiguity
        if tensor.shape != shape:
            tensor = tensor.view(shape)
        tensor = tensor.contiguous()

        # Create DTensor with Replicate placements
        placements = tuple(Replicate() for _ in range(device_mesh.ndim))
        tensor_meta = TensorMeta(
            shape=tensor.shape,
            stride=tensor.stride(),
            dtype=tensor.dtype,
        )
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return DTensor(
            tensor,
            dtensor_spec,
            requires_grad=self.requires_grad,
        )

    def _unshard_per_param(self) -> DTensor:
        """
        Unshard using block sharding mode via sequential all-gathers.

        This method handles both single-dimension sharding (FSDP v2 style) and
        multi-dimensional block sharding by performing all-gathers on each
        sharded dimension in reverse order (innermost to outermost).
        """
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        spec = self._storage_spec
        device_mesh = self._spec.mesh

        # Get padded local tensor for even all-gather
        padded_local = self._get_padded_local()

        # Detach for all-gather to avoid autograd issues, track original requires_grad
        orig_requires_grad = padded_local.requires_grad
        if orig_requires_grad:
            padded_local = padded_local.detach()

        # Perform all-gathers in reverse order (innermost to outermost)
        current_tensor = padded_local
        for i in range(len(spec.shard_dims) - 1, -1, -1):
            shard_dim = spec.shard_dims[i]
            mesh_dim_idx = spec.mesh_dim_indices[i]

            process_group = device_mesh.get_group(mesh_dim_idx)
            world_size = device_mesh.size(mesh_dim_idx)

            # Check if we have weighted sharding for this dimension
            if spec.per_rank_shard_sizes is not None:
                per_rank_sizes = spec.per_rank_shard_sizes[i]
                current_tensor = self._all_gather_weighted(
                    current_tensor, shard_dim, world_size, process_group, per_rank_sizes
                )
            else:
                current_tensor = self._all_gather_on_dim(
                    current_tensor, shard_dim, world_size, process_group
                )

        # Slice to original size (remove padding on all sharded dimensions)
        # For weighted sharding, the tensor is already correctly sized after
        # _all_gather_weighted, but we still slice to be safe
        slices = [slice(None)] * current_tensor.ndim
        for i, shard_dim in enumerate(spec.shard_dims):
            slices[shard_dim] = slice(0, spec.orig_size[shard_dim])

        gathered_tensor = current_tensor[tuple(slices)].contiguous()

        # Preserve requires_grad
        if orig_requires_grad:
            gathered_tensor = gathered_tensor.requires_grad_(True)

        # Compute placements for the unsharded tensor
        # Start with the original DTensor placements, then replace storage-sharded
        # dimensions with Replicate while preserving other placements (e.g., TP)
        original_placements = list(self._spec.placements)
        for mesh_dim_idx in spec.mesh_dim_indices:
            original_placements[mesh_dim_idx] = Replicate()
        placements = tuple(original_placements)

        # Use the original global tensor metadata to preserve the full tensor shape
        # This is important for TP+FSDP case where the global shape includes
        # all sharding dimensions (TP sharding is still active after FSDP unshard)
        tensor_meta = self._spec.tensor_meta
        dtensor_spec = DTensorSpec(
            mesh=device_mesh,
            placements=placements,
            tensor_meta=tensor_meta,
        )

        return DTensor(
            gathered_tensor,
            dtensor_spec,
            requires_grad=gathered_tensor.requires_grad,
        )

    @staticmethod
    def _all_gather_weighted(
        tensor: torch.Tensor,
        gather_dim: int,
        world_size: int,
        process_group: dist.ProcessGroup,
        per_rank_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """
        All-gather with weighted (uneven) shard sizes.

        For weighted sharding, each rank has a different shard size. We pad all
        tensors to the max size, all-gather, then extract each rank's actual
        contribution and concatenate.

        Args:
            tensor: Input tensor (already padded to max_size on gather_dim).
            gather_dim: Dimension to gather along.
            world_size: Number of ranks in the process group.
            process_group: The process group for collective.
            per_rank_sizes: Tuple of actual sizes for each rank.

        Returns:
            Gathered tensor with correct size (sum of per_rank_sizes on gather_dim).
        """
        ndim = tensor.ndim
        max_size = max(per_rank_sizes)

        # Move gather_dim to position 0 for simpler all-gather
        if gather_dim != 0:
            perm = [gather_dim] + [i for i in range(ndim) if i != gather_dim]
            tensor = tensor.permute(perm).contiguous()

        # Compute output shape for all-gather (with padding)
        padded_output_shape = list(tensor.shape)
        padded_output_shape[0] = max_size * world_size

        padded_output = tensor.new_empty(padded_output_shape)

        # All-gather with padding
        dist.all_gather_into_tensor(padded_output, tensor, group=process_group)

        # Extract each rank's actual contribution and concatenate
        slices = []
        for rank, actual_size in enumerate(per_rank_sizes):
            start = rank * max_size
            end = start + actual_size
            slices.append(padded_output[start:end])

        output = torch.cat(slices, dim=0)

        # Permute back if needed
        if gather_dim != 0:
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            output = output.permute(inv_perm).contiguous()

        return output

    @staticmethod
    def _all_gather_on_dim(
        tensor: torch.Tensor,
        gather_dim: int,
        world_size: int,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Perform all-gather on a specific dimension of the tensor.

        Args:
            tensor: Input tensor to gather.
            gather_dim: Dimension to gather along.
            world_size: Number of ranks in the process group.
            process_group: The process group for collective.

        Returns:
            Gathered tensor with gather_dim size multiplied by world_size.
        """
        ndim = tensor.ndim

        # Move gather_dim to position 0 for simpler all-gather
        if gather_dim != 0:
            perm = [gather_dim] + [i for i in range(ndim) if i != gather_dim]
            tensor = tensor.permute(perm).contiguous()

        # Compute output shape
        output_shape = list(tensor.shape)
        output_shape[0] = output_shape[0] * world_size

        output = tensor.new_empty(output_shape)

        # All-gather
        dist.all_gather_into_tensor(output, tensor, group=process_group)

        # Permute back if needed
        if gather_dim != 0:
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            output = output.permute(inv_perm).contiguous()

        return output


class TensorGroupStorage:
    """
    Manages a group of tensors distributed across ranks.

    This class supports two sharding boundaries:

    - ShardingBoundary.ELEMENT (FSDP v1 style): Tensors are flattened to 1D,
      concatenated into a single buffer, and that buffer is sharded element-wise
      across ranks. A single tensor may span multiple ranks.

    - ShardingBoundary.TENSOR (Muon style): Each tensor is assigned to exactly
      one rank (not split). Distribution is contiguous: first N/world_size
      tensors to rank 0, etc.

    Example:
        >>> mesh = DeviceMesh("cuda", list(range(4)))
        >>> tensors = [
        ...     torch.randn(8, 4),
        ...     torch.randn(10, 6),
        ...     torch.randn(4,),
        ... ]

        >>> # Element boundary (FSDP v1 style)
        >>> group = TensorGroupStorage(tensors, mesh, mesh_dim=0, boundary=ShardingBoundary.ELEMENT)
        >>> sharded = group.shard()

        >>> # Tensor boundary (each tensor whole on one rank)
        >>> group = TensorGroupStorage(tensors, mesh, mesh_dim=0, boundary=ShardingBoundary.TENSOR)
        >>> sharded = group.shard()
    """

    def __init__(
        self,
        params: list[torch.Tensor | DTensor],
        device_mesh: DeviceMesh,
        mesh_dim: int | str,
        boundary: ShardingBoundary = ShardingBoundary.ELEMENT,
        weights: list[int | float] | None = None,
    ):
        """
        Initialize a TensorGroupStorage.

        Args:
            params: List of tensors to distribute. Can be plain torch.Tensor or
                DTensor. DTensor inputs must have Replicate placement on the
                target mesh dimension.
            device_mesh: The DeviceMesh for distributed operations.
            mesh_dim: The mesh dimension (name or index) to shard across.
            boundary: Sharding boundary - ShardingBoundary.ELEMENT (FSDP v1 style)
                or ShardingBoundary.TENSOR (whole tensors per rank).
            weights: Optional weights for weighted sharding. Each weight specifies
                the relative proportion of elements/tensors each rank should receive.
                For element boundary: distribution of flattened elements.
                For tensor boundary: distribution of number of tensors per rank.
        """
        if not isinstance(boundary, ShardingBoundary):
            raise ValueError(
                f"boundary must be a ShardingBoundary enum, got {type(boundary).__name__}"
            )

        self._device_mesh = device_mesh
        self._mesh_dim = mesh_dim
        self._boundary = boundary
        self._weights = weights
        self._flat_buffer: Optional[torch.Tensor] = None
        self._sharded_dtensors: list[MemoryShardedDTensor] = []
        self._process_group: Optional[dist.ProcessGroup] = None
        self._mesh_dim_name: Optional[str] = None

        # Resolve mesh_dim to index and name first (needed for validation)
        if isinstance(mesh_dim, str):
            mesh_dim_names = device_mesh.mesh_dim_names
            if mesh_dim_names is None or mesh_dim not in mesh_dim_names:
                raise ValueError(
                    f"mesh_dim '{mesh_dim}' not found in device mesh. "
                    f"Available dimensions: {mesh_dim_names}"
                )
            self._mesh_dim_name = mesh_dim
            self._mesh_dim_idx = mesh_dim_names.index(mesh_dim)
        else:
            self._mesh_dim_idx = mesh_dim
            if mesh_dim < 0 or mesh_dim >= device_mesh.ndim:
                raise ValueError(
                    f"mesh_dim {mesh_dim} is out of range for mesh with "
                    f"{device_mesh.ndim} dimensions"
                )
            mesh_dim_names = device_mesh.mesh_dim_names
            self._mesh_dim_name = (
                mesh_dim_names[mesh_dim] if mesh_dim_names else "default"
            )

        # Validate weights if provided
        if weights is not None:
            world_size = device_mesh.size(self._mesh_dim_idx)
            if len(weights) != world_size:
                raise ValueError(
                    f"weights length {len(weights)} must match "
                    f"world_size {world_size}"
                )
            for i, w in enumerate(weights):
                if w <= 0:
                    raise ValueError(
                        f"All weights must be positive, got weight {w} at index {i}"
                    )

        # Validate and extract local tensors from DTensor inputs
        local_params = []
        for p in params:
            if isinstance(p, DTensor):
                _validate_dtensor_for_storage_sharding(
                    p, device_mesh, self._mesh_dim_idx
                )
                local_params.append(p.to_local())
            else:
                local_params.append(p)
        self._params = local_params

    def _get_process_group(self) -> dist.ProcessGroup:
        """Get or cache the process group for this mesh dimension."""
        if self._process_group is None:
            self._process_group = self._device_mesh.get_group(self._mesh_dim_idx)
        return self._process_group

    def _compute_shard_param_infos(
        self,
        numels: list[int],
        local_offset: int,
        shard_size: int,
    ) -> list[ShardParamInfo]:
        """
        Compute per-parameter shard mapping information.

        For each parameter, determines which portion (if any) falls within
        this rank's shard of the flattened buffer.

        Args:
            numels: Number of elements in each parameter.
            local_offset: Start offset of this rank's shard in the concatenated buffer.
            shard_size: Size of each rank's shard.

        Returns:
            List of ShardParamInfo, one per parameter.
        """
        shard_param_infos = []
        local_end = local_offset + shard_size
        param_start = 0

        for numel in numels:
            param_end = param_start + numel

            # Check if this param overlaps with local shard
            overlap_start = max(param_start, local_offset)
            overlap_end = min(param_end, local_end)

            if overlap_start < overlap_end:
                # Parameter is (partially) in this shard
                offset_in_shard = overlap_start - local_offset
                numel_in_shard = overlap_end - overlap_start
                intra_param_start = overlap_start - param_start
                intra_param_end = overlap_end - param_start

                shard_param_infos.append(
                    ShardParamInfo(
                        in_shard=True,
                        offset_in_shard=offset_in_shard,
                        numel_in_shard=numel_in_shard,
                        intra_param_start=intra_param_start,
                        intra_param_end=intra_param_end,
                    )
                )
            else:
                # Parameter not in this shard
                shard_param_infos.append(ShardParamInfo(in_shard=False))

            param_start = param_end

        return shard_param_infos

    def _extract_param_local(
        self,
        param_index: int,
        shard_param_info: ShardParamInfo,
    ) -> torch.Tensor:
        """
        Extract this parameter's portion from the local shard.

        Args:
            param_index: Index of the parameter in the group.
            shard_param_info: ShardParamInfo for this parameter.

        Returns:
            A tensor containing this parameter's local portion, or empty tensor
            if the parameter is not in this rank's shard.
        """
        if not shard_param_info.in_shard:
            # Parameter not in this shard - return empty tensor
            return self._flat_buffer.new_empty(0)

        # Extract from flat buffer
        offset = shard_param_info.offset_in_shard
        numel = shard_param_info.numel_in_shard
        return self._flat_buffer[offset : offset + numel]

    def shard(self) -> list[MemoryShardedDTensor]:
        """
        Distribute tensors across ranks based on the sharding mode.

        For "element" mode (FSDP v1 style):
        - Flattens each parameter to 1D
        - Concatenates them into a single buffer
        - Pads for even distribution across ranks
        - Takes this rank's shard of the buffer

        For "tensor" mode:
        - Assigns whole tensors to ranks (contiguous chunks)
        - Each tensor stays fully on one rank

        Returns:
            List of MemoryShardedDTensor, one per input tensor.
        """
        if self._boundary == ShardingBoundary.ELEMENT:
            return self._shard_element()
        else:
            return self._shard_tensor()

    def _shard_element(self) -> list[MemoryShardedDTensor]:
        """Shard using element mode (FSDP v1 style)."""
        from torch.distributed.tensor._dtensor_spec import TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        # 1. Collect metadata
        shapes = [p.shape for p in self._params]
        strides = [p.stride() for p in self._params]
        numels = [p.numel() for p in self._params]
        total_numel = sum(numels)

        # 2. Flatten and concatenate
        flat_tensors = [p.view(-1) for p in self._params]
        concatenated = torch.cat(flat_tensors, dim=0)

        # 3. Compute shard info
        world_size = self._device_mesh.size(self._mesh_dim_idx)
        rank = self._device_mesh.get_local_rank(self._mesh_dim_idx)

        if self._weights is not None:
            # Weighted sharding - compute sizes using proportions
            # Validate that each size is a whole number (no truncation)
            total_weight = sum(self._weights)
            per_rank_sizes_float = [total_numel * w / total_weight for w in self._weights]
            per_rank_sizes: list[int] | None = []
            for i, size_f in enumerate(per_rank_sizes_float):
                if size_f != int(size_f):
                    raise ValueError(
                        f"Weight {self._weights[i]} at index {i} does not produce an integer "
                        f"shard size. Computed {size_f} elements, but must be a whole number. "
                        f"Ensure total numel {total_numel} is evenly divisible by weights."
                    )
                per_rank_sizes.append(int(size_f))

            shard_size = per_rank_sizes[rank]
            local_offset = sum(per_rank_sizes[:rank])
            padded_numel = total_numel  # No padding needed for weighted
        else:
            # Even sharding (existing logic)
            per_rank_sizes = None
            padded_numel = math.ceil(total_numel / world_size) * world_size
            shard_size = padded_numel // world_size
            local_offset = rank * shard_size

        # 4. Pad if needed
        if total_numel < padded_numel:
            concatenated = F.pad(concatenated, [0, padded_numel - total_numel])

        # 5. Take this rank's shard
        self._flat_buffer = concatenated[
            local_offset : local_offset + shard_size
        ].clone()

        # 6. Compute per-param shard mappings
        shard_param_infos = self._compute_shard_param_infos(
            numels, local_offset, shard_size
        )

        # 7. Create MemoryShardedDTensor with FlattenedStorageShardingSpec for each
        placements = tuple(Replicate() for _ in range(self._device_mesh.ndim))
        process_group = self._get_process_group()

        for i, (shape, stride, spi) in enumerate(
            zip(shapes, strides, shard_param_infos)
        ):
            spec = FlattenedStorageShardingSpec(
                param_shapes=tuple(shapes),
                param_strides=tuple(strides),
                param_numels=tuple(numels),
                total_numel=total_numel,
                padded_total_numel=padded_numel,
                mesh_dim=self._mesh_dim_name,
                local_offset=local_offset,
                local_numel=shard_size,
                shard_param_infos=tuple(shard_param_infos),
                param_index=i,
                per_rank_shard_sizes=tuple(per_rank_sizes) if per_rank_sizes else None,
            )

            # Extract this param's portion of the local shard
            local_tensor = self._extract_param_local(i, spi)

            # Create global tensor metadata for this parameter
            global_tensor_meta = TensorMeta(
                shape=shape,
                stride=stride,
                dtype=local_tensor.dtype,
            )

            dtensor = MemoryShardedDTensor._create(
                local_tensor=local_tensor,
                device_mesh=self._device_mesh,
                storage_spec=spec,
                process_group=process_group,
                placements=placements,
                flat_buffer=self._flat_buffer,
                global_tensor_meta=global_tensor_meta,
            )
            self._sharded_dtensors.append(dtensor)

        return self._sharded_dtensors

    def _shard_tensor(self) -> list[MemoryShardedDTensor]:
        """Shard using tensor mode (each tensor fully on one rank)."""
        from torch.distributed.tensor._dtensor_spec import TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        # 1. Collect metadata
        shapes = tuple(p.shape for p in self._params)
        strides = tuple(p.stride() for p in self._params)
        numels = tuple(p.numel() for p in self._params)
        total_params = len(self._params)

        # 2. Compute tensor-to-rank assignment (contiguous chunks)
        world_size = self._device_mesh.size(self._mesh_dim_idx)
        rank = self._device_mesh.get_local_rank(self._mesh_dim_idx)

        param_to_rank, rank_to_params = self._compute_tensor_assignment(
            total_params, world_size, self._weights
        )

        # 3. Create MemoryShardedDTensor for each tensor
        placements = tuple(Replicate() for _ in range(self._device_mesh.ndim))
        process_group = self._get_process_group()

        for i, (shape, stride, numel) in enumerate(zip(shapes, strides, numels)):
            owns_tensor = param_to_rank[i] == rank

            spec = TensorGroupShardingSpec(
                param_shapes=shapes,
                param_strides=strides,
                param_numels=numels,
                total_params=total_params,
                mesh_dim=self._mesh_dim_name,
                mesh_dim_idx=self._mesh_dim_idx,
                param_to_rank=param_to_rank,
                rank_to_params=rank_to_params,
                param_index=i,
                owns_tensor=owns_tensor,
            )

            # Get local tensor (actual tensor if owned, empty placeholder if not)
            if owns_tensor:
                local_tensor = self._params[i]
            else:
                # Create empty placeholder with correct shape
                local_tensor = self._params[i].new_empty(0)

            # Create global tensor metadata for this parameter
            global_tensor_meta = TensorMeta(
                shape=shape,
                stride=stride,
                dtype=self._params[i].dtype,
            )

            dtensor = MemoryShardedDTensor._create(
                local_tensor=local_tensor,
                device_mesh=self._device_mesh,
                storage_spec=spec,
                process_group=process_group,
                placements=placements,
                global_tensor_meta=global_tensor_meta,
            )
            self._sharded_dtensors.append(dtensor)

        return self._sharded_dtensors

    @staticmethod
    def _compute_tensor_assignment(
        total_params: int,
        world_size: int,
        weights: list[int | float] | None = None,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """
        Compute tensor-to-rank assignment using contiguous chunks.

        For even distribution: first N/world_size tensors go to rank 0, etc.
        For weighted distribution: tensors are assigned proportionally to weights.

        Args:
            total_params: Total number of parameters/tensors.
            world_size: Number of ranks.
            weights: Optional weights for weighted distribution.

        Returns:
            param_to_rank: Tuple mapping param_index to owning rank.
            rank_to_params: Tuple mapping rank to tuple of owned param indices.
        """
        if total_params == 0:
            return (), tuple(() for _ in range(world_size))

        if weights is not None:
            # Weighted distribution - compute counts using proportions
            # Validate that each count is a whole number (no truncation)
            total_weight = sum(weights)
            per_rank_counts_float = [total_params * w / total_weight for w in weights]
            per_rank_counts = []
            for i, count_f in enumerate(per_rank_counts_float):
                if count_f != int(count_f):
                    raise ValueError(
                        f"Weight {weights[i]} at index {i} does not produce an integer "
                        f"tensor count. Computed {count_f} tensors, but must be a whole number. "
                        f"Ensure total params {total_params} is evenly divisible by weights."
                    )
                per_rank_counts.append(int(count_f))

            param_to_rank = []
            rank_to_params: list[list[int]] = [[] for _ in range(world_size)]

            param_idx = 0
            for r in range(world_size):
                count = per_rank_counts[r]
                for _ in range(count):
                    param_to_rank.append(r)
                    rank_to_params[r].append(param_idx)
                    param_idx += 1
        else:
            # Even distribution (existing logic)
            base_count = total_params // world_size
            extra = total_params % world_size

            param_to_rank = []
            rank_to_params = [[] for _ in range(world_size)]

            param_idx = 0
            for r in range(world_size):
                # First 'extra' ranks get base_count + 1 tensors
                count = base_count + (1 if r < extra else 0)
                for _ in range(count):
                    param_to_rank.append(r)
                    rank_to_params[r].append(param_idx)
                    param_idx += 1

        return tuple(param_to_rank), tuple(tuple(p) for p in rank_to_params)

    def unshard_all(self) -> list[DTensor]:
        """
        Reconstruct all tensors and return them replicated across ranks.

        For "element" mode: Performs a single all-gather on the flat buffer,
        then extracts each tensor's view.

        For "tensor" mode: Broadcasts each tensor from its owning rank.

        Returns:
            List of DTensor, one per tensor, with Replicate placements.
        """
        if not self._sharded_dtensors:
            raise RuntimeError("Must call shard() before unshard_all()")

        if self._boundary == ShardingBoundary.ELEMENT:
            return self._unshard_all_element()
        else:
            return self._unshard_all_tensor()

    def _unshard_all_element(self) -> list[DTensor]:
        """Unshard all tensors in element mode (single all-gather)."""
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
        from torch.distributed.tensor.placement_types import Replicate

        # Single all-gather for entire flat buffer
        world_size = self._device_mesh.size(self._mesh_dim_idx)
        process_group = self._get_process_group()

        # Detach for all-gather
        flat_buffer = self._flat_buffer
        orig_requires_grad = flat_buffer.requires_grad
        if orig_requires_grad:
            flat_buffer = flat_buffer.detach()

        # Get spec from first sharded tensor
        spec = self._sharded_dtensors[0]._storage_spec

        if spec.per_rank_shard_sizes is not None:
            # Weighted sharding: variable sizes per rank
            per_rank_sizes = spec.per_rank_shard_sizes
            max_size = max(per_rank_sizes)

            # Pad local buffer to max_size for uniform all-gather
            if flat_buffer.numel() < max_size:
                padded_buffer = flat_buffer.new_zeros(max_size)
                padded_buffer[: flat_buffer.numel()] = flat_buffer
            else:
                padded_buffer = flat_buffer

            # All-gather padded buffers
            gathered = padded_buffer.new_empty(max_size * world_size)
            dist.all_gather_into_tensor(gathered, padded_buffer, group=process_group)

            # Extract actual contributions from each rank
            chunks = []
            for rank_idx, actual_size in enumerate(per_rank_sizes):
                start = rank_idx * max_size
                end = start + actual_size
                chunks.append(gathered[start:end])
            full_buffer = torch.cat(chunks, dim=0)
        else:
            # Even sharding: all ranks have same size
            full_buffer = flat_buffer.new_empty(flat_buffer.numel() * world_size)
            dist.all_gather_into_tensor(full_buffer, flat_buffer, group=process_group)

            # Slice to remove padding
            full_buffer = full_buffer[: spec.total_numel]

        # Create views for each param
        unsharded = []
        placements = tuple(Replicate() for _ in range(self._device_mesh.ndim))
        offset = 0

        for i, numel in enumerate(spec.param_numels):
            shape = spec.param_shapes[i]
            param_data = full_buffer[offset : offset + numel]
            param_tensor = param_data.view(shape).contiguous()

            if orig_requires_grad:
                param_tensor = param_tensor.requires_grad_(True)

            tensor_meta = TensorMeta(
                shape=param_tensor.shape,
                stride=param_tensor.stride(),
                dtype=param_tensor.dtype,
            )
            dtensor_spec = DTensorSpec(
                mesh=self._device_mesh,
                placements=placements,
                tensor_meta=tensor_meta,
            )

            unsharded.append(
                DTensor(
                    param_tensor,
                    dtensor_spec,
                    requires_grad=param_tensor.requires_grad,
                )
            )
            offset += numel

        return unsharded

    def _unshard_all_tensor(self) -> list[DTensor]:
        """Unshard all tensors in tensor mode (broadcast from each owner)."""
        # Simply call unshard() on each sharded tensor
        return [msdt.unshard() for msdt in self._sharded_dtensors]

    def get_flat_buffer(self) -> torch.Tensor:
        """
        Returns the shared flat buffer.

        This can be used for FSDP integration where the all-gather input
        is needed directly.

        Returns:
            The 1D flat buffer containing this rank's shard of all parameters.
        """
        if self._flat_buffer is None:
            raise RuntimeError("Must call shard() before get_flat_buffer()")
        return self._flat_buffer
