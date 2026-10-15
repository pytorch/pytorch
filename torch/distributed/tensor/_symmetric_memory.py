# Copyright (c) Meta Platforms, Inc. and affiliates
from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import torch
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.config as dist_config
from torch._guards import detect_fake_mode
from torch.distributed.tensor.placement_types import _StridedShard, Placement, Shard


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.distributed.device_mesh import DeviceMesh


def is_symmetric_memory_enabled() -> bool:
    return bool(dist_config.dtensor_use_symmetric_memory)


def _max_shard_dim_size(
    dim_size: int,
    mesh_dim_size: int,
    placement: Shard | _StridedShard,
) -> int:
    max_size = 0
    for rank in range(mesh_dim_size):
        shard_size, _ = placement._local_shard_size_and_offset(  # type: ignore[attr-defined]
            dim_size,
            mesh_dim_size,
            rank,
        )
        max_size = max(max_size, int(shard_size))
    return max_size


def _compute_max_local_numel(
    global_shape: torch.Size,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> int:
    max_shape = list(global_shape)
    for mesh_dim, placement in enumerate(placements):
        if not isinstance(placement, (Shard, _StridedShard)):
            continue
        shard_dim = placement.dim
        if shard_dim < 0:
            shard_dim += len(max_shape)
        if shard_dim < 0 or shard_dim >= len(max_shape):
            raise AssertionError(
                f"Sharding dim {placement.dim} out of range for tensor ndim {len(max_shape)}"
            )
        max_shape[shard_dim] = _max_shard_dim_size(
            max_shape[shard_dim],
            device_mesh.size(mesh_dim),
            placement,
        )
    return prod(max_shape)


@torch._disable_dynamo
def _allocate_symmetric_memory(
    global_shape: torch.Size,
    local_shape: torch.Size,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
    dtype: torch.dtype | None,
    device: torch.device,
) -> torch.Tensor:
    if device.index is None:
        device = torch.device(device.type, torch.cuda.current_device())
    # Rendezvous needs a nonempty backing allocation, even for empty tensors.
    max_numel = max(1, _compute_max_local_numel(global_shape, device_mesh, placements))
    base = symm_mem.empty(max_numel, dtype=dtype, device=device)
    return base.narrow(0, 0, prod(local_shape)).view(local_shape).detach()


def _is_eligible_device(device: torch.device) -> bool:
    return device.type == "cuda"


def _should_allocate_symmetric(
    device: torch.device,
    device_mesh: DeviceMesh,
    local_tensor: torch.Tensor | None = None,
) -> bool:
    if not is_symmetric_memory_enabled() or not _is_eligible_device(device):
        return False
    if torch.compiler.is_compiling() or detect_fake_mode(local_tensor) is not None:
        raise RuntimeError(
            "SymmetricMemory-backed DTensor construction is not supported under "
            "torch.compile or FakeTensorMode. Construct DTensors eagerly before tracing."
        )
    if (
        symm_mem.get_backend(device) == "NVSHMEM"
        and device_mesh.size() != torch.distributed.get_world_size()
    ):
        raise RuntimeError(
            "SymmetricMemory-backed DTensor construction with NVSHMEM requires a "
            "mesh containing every rank in the default process group; submeshes "
            "are not supported because NVSHMEM allocation is collective."
        )
    return device_mesh._is_current_rank_part_of_mesh()


def empty_symmetric_memory_local_tensor(
    global_shape: torch.Size,
    local_shape: torch.Size,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
    *,
    dtype: torch.dtype | None,
    device: torch.device,
) -> torch.Tensor | None:
    """Return a SymmetricMemory-backed local tensor, or None if ineligible."""
    if not _should_allocate_symmetric(device, device_mesh):
        return None

    return _allocate_symmetric_memory(
        global_shape, local_shape, device_mesh, placements, dtype, device
    )


def copy_to_symmetric_memory(
    local_tensor: torch.Tensor,
    global_shape: torch.Size,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> torch.Tensor:
    if not _should_allocate_symmetric(local_tensor.device, device_mesh, local_tensor):
        return local_tensor
    if not local_tensor.is_contiguous():
        return local_tensor

    out = _allocate_symmetric_memory(
        global_shape,
        local_tensor.shape,
        device_mesh,
        placements,
        local_tensor.dtype,
        local_tensor.device,
    )
    with torch.no_grad():
        out.copy_(local_tensor)
    return out
