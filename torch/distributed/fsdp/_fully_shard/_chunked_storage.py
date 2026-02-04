# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Placement

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


__all__ = ["ChunkedStorage", "fully_shard_flat"]


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
    byte_offset: int = 0  # byte offset into the unified storage


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
    """

    def __init__(
        self,
        byte_storage: torch.Tensor,
        param_infos: dict[str, ParamInfo],
        mesh: DeviceMesh,
        total_bytes: int,
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes

    @property
    def byte_storage(self) -> torch.Tensor:
        """The underlying unified byte storage tensor."""
        return self._byte_storage

    @property
    def flat_storage(self) -> torch.Tensor:
        """Alias for byte_storage for backwards compatibility."""
        return self._byte_storage

    @property
    def total_bytes(self) -> int:
        """Total bytes in the unified storage."""
        return self._total_bytes

    @property
    def numel(self) -> int:
        """Total number of bytes (for compatibility, returns byte count)."""
        return self._byte_storage.numel()

    @property
    def param_infos(self) -> dict[str, ParamInfo]:
        """Metadata for each parameter."""
        return self._param_infos

    def get_local_view(self, fqn: str) -> torch.Tensor:
        """Get the local tensor view for a parameter by FQN."""
        info = self._param_infos[fqn]
        num_bytes = info.local_numel * info.dtype.itemsize
        byte_view = self._byte_storage[info.byte_offset : info.byte_offset + num_bytes]
        # Reinterpret bytes as the correct dtype, then reshape
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.local_shape)


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
) -> tuple[dict[str, ParamInfo], int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    Parameters are laid out sequentially in the byte buffer with proper alignment.

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the unified buffer
    """
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0

    for fqn, param in named_params:
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(global_shape, mesh, placements)
        dtype = param.dtype

        # Align offset for this dtype
        alignment = _get_dtype_alignment(dtype)
        aligned_offset = _align_offset(current_byte_offset, alignment)

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
        )
        param_infos[fqn] = info

        # Move offset past this parameter's bytes
        param_bytes = local_numel * dtype.itemsize
        current_byte_offset = aligned_offset + param_bytes

    return param_infos, current_byte_offset


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
    param_infos, total_bytes = _create_param_infos(named_params, mesh, placements)

    # Allocate unified byte storage
    byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)

    # Copy original data into byte storage (handles sharding)
    _copy_original_data_to_flat_storage(byte_storage, named_params, param_infos, mesh)

    # Create ChunkedStorage
    storage = ChunkedStorage(byte_storage, param_infos, mesh, total_bytes)

    # Replace each parameter with DTensor view
    for fqn, info in param_infos.items():
        local_view = storage.get_local_view(fqn)
        dtensor = _create_dtensor_from_view(local_view, info, mesh)
        new_param = nn.Parameter(dtensor, requires_grad=info.requires_grad)
        _set_param_on_module(module, fqn, new_param)

    return storage
