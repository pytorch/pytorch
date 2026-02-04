# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Placement

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


__all__ = ["ChunkedStorage", "fully_shard_flat"]


class ShardedState(Enum):
    """State of the parameters in ChunkedStorage."""

    SHARDED = auto()  # Parameters are sharded DTensors
    UNSHARDED = auto()  # Parameters are unsharded for forward/backward


@dataclass
class ParamInfo:
    """Metadata for a parameter in chunked storage."""

    fqn: str
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    placements: tuple[Placement, ...]
    local_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    local_numel: int = 0
    byte_offset: int = 0  # byte offset into the sharded storage
    global_numel: int = 0  # total elements in unsharded param
    unsharded_byte_offset: int = 0  # byte offset into the unsharded storage


def _get_dtype_alignment(dtype: torch.dtype) -> int:
    """Get alignment requirement in bytes for a dtype."""
    # Most dtypes need alignment equal to their element size
    return dtype.itemsize


def _align_offset(offset: int, alignment: int) -> int:
    """Round up offset to next aligned boundary."""
    return (offset + alignment - 1) // alignment * alignment


class ChunkedStorage:
    """
    Manages a unified byte buffer that backs multiple DTensor parameters.

    All parameters, regardless of dtype, are stored in a single contiguous
    byte buffer. Each parameter's local shard is a typed view into this buffer
    at the appropriate byte offset with proper alignment.

    This enables a single all-gather operation for the entire parameter group.

    Lifecycle:
        1. SHARDED state: Parameters are sharded DTensors, model.parameters() returns DTensors
        2. unshard(): All-gather byte buffer, register unsharded params for forward
        3. UNSHARDED state: Parameters are full tensors for forward/backward
        4. reshard(): Free unsharded buffer, restore sharded DTensor params
    """

    def __init__(
        self,
        byte_storage: torch.Tensor,
        param_infos: dict[str, ParamInfo],
        mesh: DeviceMesh,
        total_bytes: int,
        total_unsharded_bytes: int,
        module: nn.Module,
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes
        self._total_unsharded_bytes = total_unsharded_bytes
        self._module = module
        self._state = ShardedState.SHARDED

        # Unsharded buffer (allocated on demand)
        self._unsharded_byte_storage: torch.Tensor | None = None

        # Cache sharded DTensor parameters for reshard
        self._sharded_params: dict[str, nn.Parameter] = {}
        for fqn in param_infos:
            parts = fqn.split(".")
            mod = module
            for part in parts[:-1]:
                mod = getattr(mod, part)
            self._sharded_params[fqn] = getattr(mod, parts[-1])

    @property
    def byte_storage(self) -> torch.Tensor:
        """The underlying unified byte storage tensor (sharded)."""
        return self._byte_storage

    @property
    def flat_storage(self) -> torch.Tensor:
        """Alias for byte_storage for backwards compatibility."""
        return self._byte_storage

    @property
    def total_bytes(self) -> int:
        """Total bytes in the sharded storage."""
        return self._total_bytes

    @property
    def total_unsharded_bytes(self) -> int:
        """Total bytes needed for unsharded storage."""
        return self._total_unsharded_bytes

    @property
    def numel(self) -> int:
        """Total number of bytes (for compatibility, returns byte count)."""
        return self._byte_storage.numel()

    @property
    def param_infos(self) -> dict[str, ParamInfo]:
        """Metadata for each parameter."""
        return self._param_infos

    @property
    def state(self) -> ShardedState:
        """Current state (SHARDED or UNSHARDED)."""
        return self._state

    @property
    def world_size(self) -> int:
        """World size of the mesh."""
        return self._mesh.size()

    def get_local_view(self, fqn: str) -> torch.Tensor:
        """Get the local tensor view for a parameter by FQN (from sharded storage)."""
        info = self._param_infos[fqn]
        num_bytes = info.local_numel * info.dtype.itemsize
        byte_view = self._byte_storage[info.byte_offset : info.byte_offset + num_bytes]
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.local_shape)

    def get_unsharded_view(self, fqn: str) -> torch.Tensor:
        """Get the unsharded tensor view for a parameter by FQN."""
        if self._unsharded_byte_storage is None:
            raise RuntimeError("Unsharded storage not allocated. Call unshard() first.")
        info = self._param_infos[fqn]
        num_bytes = info.global_numel * info.dtype.itemsize
        byte_view = self._unsharded_byte_storage[
            info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
        ]
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.global_shape)

    def all_gather(self) -> torch.Tensor:
        """
        All-gather the sharded byte buffer to get the full unsharded buffer.

        Returns:
            Unsharded byte buffer containing all parameter data.
        """
        # Allocate unsharded buffer if needed
        if self._unsharded_byte_storage is None:
            self._unsharded_byte_storage = torch.empty(
                self._total_unsharded_bytes,
                dtype=torch.uint8,
                device=self._byte_storage.device,
            )

        # Get the process group from mesh
        pg = self._mesh.get_group()
        world_size = self.world_size
        my_rank = self._mesh.get_local_rank()

        # For each parameter, all-gather its chunks and reassemble
        for fqn, info in self._param_infos.items():
            # Get shard dimension
            shard_dim = 0
            for placement in info.placements:
                if isinstance(placement, Shard):
                    shard_dim = placement.dim
                    break

            # Get local chunk for this rank
            local_view = self.get_local_view(fqn)

            # Compute max local numel across all ranks for padding
            max_local_numel = 0
            local_numels = []
            for rank in range(world_size):
                local_shape_for_rank = self._compute_local_shape_for_rank(
                    info.global_shape, info.placements, rank
                )
                numel = 1
                for d in local_shape_for_rank:
                    numel *= d
                local_numels.append(numel)
                max_local_numel = max(max_local_numel, numel)

            # Pad local tensor to max size
            if local_view.numel() < max_local_numel:
                padded = torch.zeros(
                    max_local_numel, dtype=info.dtype, device=self._byte_storage.device
                )
                if local_view.numel() > 0:
                    padded[: local_view.numel()] = local_view.view(-1)
            else:
                padded = local_view.view(-1)

            # All-gather the padded tensors
            all_gather_output = torch.empty(
                world_size * max_local_numel,
                dtype=info.dtype,
                device=self._byte_storage.device,
            )
            dist.all_gather_into_tensor(all_gather_output, padded.contiguous(), group=pg)

            # Extract and concatenate non-padded chunks
            chunks = []
            for rank in range(world_size):
                rank_numel = local_numels[rank]
                if rank_numel > 0:
                    start = rank * max_local_numel
                    chunk = all_gather_output[start : start + rank_numel]
                    local_shape_for_rank = self._compute_local_shape_for_rank(
                        info.global_shape, info.placements, rank
                    )
                    chunks.append(chunk.view(local_shape_for_rank))

            # Concatenate along shard dimension
            if chunks:
                unsharded_param = torch.cat(chunks, dim=shard_dim)
            else:
                unsharded_param = torch.empty(
                    info.global_shape,
                    dtype=info.dtype,
                    device=self._byte_storage.device,
                )

            # Copy into unsharded buffer
            num_bytes = info.global_numel * info.dtype.itemsize
            dest = self._unsharded_byte_storage[
                info.unsharded_byte_offset : info.unsharded_byte_offset + num_bytes
            ]
            dest.copy_(unsharded_param.view(-1).view(torch.uint8))

        return self._unsharded_byte_storage

    def _compute_local_shape_for_rank(
        self, global_shape: torch.Size, placements: tuple[Placement, ...], rank: int
    ) -> torch.Size:
        """Compute local shape for a specific rank."""
        # This is a simplified version - for proper implementation we'd use DTensor utils
        shard_dim = 0
        for placement in placements:
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                break

        world_size = self.world_size
        dim_size = global_shape[shard_dim]

        # Compute chunk size for this rank (handles uneven sharding)
        base_size = dim_size // world_size
        remainder = dim_size % world_size

        if rank < remainder:
            local_dim_size = base_size + 1
        else:
            local_dim_size = base_size

        local_shape = list(global_shape)
        local_shape[shard_dim] = local_dim_size
        return torch.Size(local_shape)

    def unshard(self) -> None:
        """
        All-gather the byte buffer and register unsharded parameters on the module.

        After calling this, model.parameters() returns unsharded tensors for forward/backward.
        """
        if self._state == ShardedState.UNSHARDED:
            return  # Already unsharded

        # All-gather the byte buffer
        self.all_gather()

        # Register unsharded parameters on the module
        for fqn, info in self._param_infos.items():
            unsharded_view = self.get_unsharded_view(fqn)
            unsharded_param = nn.Parameter(unsharded_view, requires_grad=info.requires_grad)
            _set_param_on_module(self._module, fqn, unsharded_param)

        self._state = ShardedState.UNSHARDED

    def reshard(self) -> None:
        """
        Reduce-scatter gradients, free unsharded buffer, and restore sharded DTensor parameters.

        Gradients from the unsharded parameters are reduce-scattered across ranks,
        and the resulting sharded gradients are stored on the DTensor parameters.

        After calling this, model.parameters() returns sharded DTensors with gradients.
        """
        if self._state == ShardedState.SHARDED:
            return  # Already sharded

        # Reduce-scatter gradients before swapping parameters
        self._reduce_scatter_grads()

        # Restore sharded DTensor parameters
        for fqn, sharded_param in self._sharded_params.items():
            _set_param_on_module(self._module, fqn, sharded_param)

        # Free unsharded buffer
        if self._unsharded_byte_storage is not None:
            self._unsharded_byte_storage = None

        self._state = ShardedState.SHARDED

    def _reduce_scatter_grads(self) -> None:
        """
        Reduce-scatter gradients from unsharded params to sharded DTensor params.

        For each parameter:
        1. Get gradient from current (unsharded) param on module
        2. Reduce-scatter: sum gradients across ranks, then each rank keeps its shard
        3. Store sharded gradient as DTensor on the cached DTensor param
        """
        pg = self._mesh.get_group()
        world_size = self.world_size
        my_rank = self._mesh.get_local_rank()

        for fqn, info in self._param_infos.items():
            # Get current unsharded param from module
            unsharded_param = _get_param_from_module(self._module, fqn)

            if unsharded_param.grad is None:
                continue

            grad = unsharded_param.grad

            # Get shard dimension
            shard_dim = 0
            for placement in info.placements:
                if isinstance(placement, Shard):
                    shard_dim = placement.dim
                    break

            # Compute local sizes for each rank (for uneven sharding)
            local_numels = []
            for rank in range(world_size):
                local_shape = self._compute_local_shape_for_rank(
                    info.global_shape, info.placements, rank
                )
                numel = 1
                for d in local_shape:
                    numel *= d
                local_numels.append(numel)

            max_local_numel = max(local_numels)
            my_local_numel = local_numels[my_rank]

            # Pad gradient to be evenly divisible for reduce_scatter
            # We need to split into world_size equal chunks
            padded_size = max_local_numel * world_size
            if grad.numel() < padded_size:
                padded_grad = torch.zeros(
                    padded_size, dtype=info.dtype, device=self._byte_storage.device
                )
                padded_grad[: grad.numel()] = grad.view(-1)
            else:
                padded_grad = grad.view(-1)[:padded_size].contiguous()

            # Reduce-scatter: each rank gets chunk[rank] after reduction
            output = torch.empty(
                max_local_numel, dtype=info.dtype, device=self._byte_storage.device
            )
            dist.reduce_scatter_tensor(output, padded_grad, group=pg)

            # Extract only the valid (non-padded) portion for this rank
            if my_local_numel > 0:
                sharded_grad = output[:my_local_numel].view(info.local_shape)
                # Create DTensor gradient and assign to sharded param
                grad_dtensor = _create_dtensor_from_view(sharded_grad, info, self._mesh)
                self._sharded_params[fqn].grad = grad_dtensor

    @contextmanager
    def unsharded(self):
        """
        Context manager for automatic unshard/reshard around forward.

        Usage:
            with storage.unsharded():
                output = model(input)
        """
        self.unshard()
        try:
            yield
        finally:
            self.reshard()


def _compute_local_info(
    global_shape: torch.Size,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[torch.Size, int]:
    """Compute local shape and numel for a parameter on current rank."""
    local_shape, _ = compute_local_shape_and_global_offset(
        global_shape, mesh, placements, skip_offset=True
    )
    local_numel = 1
    for dim in local_shape:
        local_numel *= dim
    return torch.Size(local_shape), local_numel


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[dict[str, ParamInfo], int, int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    Parameters are laid out sequentially in the byte buffer with proper alignment.

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
        total_unsharded_bytes: total bytes needed for the unsharded buffer
    """
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0
    current_unsharded_byte_offset = 0

    for fqn, param in named_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(global_shape, mesh, placements)
        dtype = param.dtype

        # Compute global numel
        global_numel = 1
        for dim in global_shape:
            global_numel *= dim

        # Align offset for this dtype (sharded buffer)
        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)

        # Align offset for unsharded buffer
        aligned_unsharded_offset = _align_offset(current_unsharded_byte_offset, alignment)

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=aligned_offset,
            global_numel=global_numel,
            unsharded_byte_offset=aligned_unsharded_offset,
        )
        param_infos[fqn] = info

        # Move offsets past this parameter's bytes
        param_bytes = local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

        unsharded_param_bytes = global_numel * dtype.itemsize
        current_unsharded_byte_offset = aligned_unsharded_offset + unsharded_param_bytes

    return param_infos, current_byte_offset, current_unsharded_byte_offset


def _create_dtensor_from_view(
    local_view: torch.Tensor,
    info: ParamInfo,
    mesh: DeviceMesh,
) -> DTensor:
    """Create a DTensor whose _local_tensor is the given view."""
    spec = DTensorSpec(
        mesh=mesh,
        placements=info.placements,
        tensor_meta=TensorMeta(
            shape=info.global_shape,
            stride=info.global_stride,
            dtype=info.dtype,
        ),
    )
    return DTensor(
        local_view,
        spec,
        requires_grad=info.requires_grad,
    )


def _set_param_on_module(
    root_module: nn.Module,
    fqn: str,
    param: nn.Parameter,
) -> None:
    """Navigate to submodule by FQN and set parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], param)


def _get_param_from_module(
    root_module: nn.Module,
    fqn: str,
) -> nn.Parameter:
    """Navigate to submodule by FQN and get parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    return getattr(module, parts[-1])


def _copy_original_data_to_flat_storage(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh: DeviceMesh,
) -> None:
    """
    Copy original parameter data into the unified byte storage, sharding as needed.

    For each parameter:
    1. If already on correct device, chunk and copy the local shard
    2. If on meta device, leave uninitialized (caller must initialize later)
    """
    if mesh.ndim != 1:
        raise NotImplementedError("Only 1D mesh (pure FSDP) is supported currently")

    shard_rank = mesh.get_local_rank()
    shard_world_size = mesh.size()

    for fqn, param in named_params:
        info = param_infos[fqn]

        if param.device.type == "meta":
            continue

        # Determine shard dimension from placements
        shard_dim = 0
        for placement in info.placements:
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                break

        # Chunk the parameter and get this rank's shard
        chunks = list(torch.chunk(param.data, shard_world_size, dim=shard_dim))
        while len(chunks) < shard_world_size:
            chunks.append(chunks[0].new_empty(0))
        local_shard = chunks[shard_rank]

        # Copy into byte storage
        if local_shard.numel() > 0:
            num_bytes = info.local_numel * info.dtype.itemsize
            byte_view = byte_storage[info.byte_offset : info.byte_offset + num_bytes]
            # View as the correct dtype, then copy
            typed_dest = byte_view.view(info.dtype)
            typed_dest.copy_(local_shard.view(-1))


def fully_shard_flat(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...] | None = None,
) -> ChunkedStorage:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects all parameters from the module
    2. Creates a single unified byte buffer for all parameters (regardless of dtype)
    3. Replaces each parameter with a DTensor whose _local_tensor is a typed view
       into the byte buffer at the appropriate offset
    4. Returns ChunkedStorage for managing the unified buffer

    The unified byte buffer enables a single all-gather operation for all parameters.

    Args:
        module: The module to shard. Can have real or meta device parameters.
        mesh: The device mesh for sharding. Currently only 1D mesh is supported.
        placements: The sharding placements. Defaults to (Shard(0),).

    Returns:
        ChunkedStorage instance containing the unified byte buffer and parameter metadata.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,))
        >>> model = nn.Linear(1024, 1024)
        >>> storage = fully_shard_flat(model, mesh)
        >>> # model.weight and model.bias are now DTensors backed by storage.byte_storage

    Note:
        - Parameters of different dtypes are supported in a single unified buffer
        - Proper alignment is maintained for each dtype
        - Parameters on meta device will have uninitialized storage
        - The returned ChunkedStorage holds the byte buffer; keep it alive
    """
    if placements is None:
        placements = (Shard(0),)

    # Collect all parameters
    named_params = list(module.named_parameters())
    if not named_params:
        raise ValueError("Module has no parameters to shard")

    # Determine device
    device = mesh.device_type
    if device == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)

    # Create parameter infos with local shapes and byte offsets
    param_infos, total_bytes, total_unsharded_bytes = _create_param_infos(
        named_params, mesh, placements
    )

    # Allocate unified byte storage
    byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    # Copy original data into byte storage (handles sharding)
    _copy_original_data_to_flat_storage(byte_storage, named_params, param_infos, mesh)

    # Replace each parameter with DTensor view (before creating ChunkedStorage)
    for fqn, info in param_infos.items():
        local_view = byte_storage[info.byte_offset : info.byte_offset + info.local_numel * info.dtype.itemsize]
        typed_view = local_view.view(info.dtype).view(info.local_shape)
        dtensor = _create_dtensor_from_view(typed_view, info, mesh)
        new_param = nn.Parameter(dtensor, requires_grad=info.requires_grad)
        _set_param_on_module(module, fqn, new_param)

    # Create ChunkedStorage (after DTensor params are registered)
    storage = ChunkedStorage(
        byte_storage, param_infos, mesh, total_bytes, total_unsharded_bytes, module
    )

    return storage
