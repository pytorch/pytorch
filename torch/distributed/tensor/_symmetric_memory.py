# Copyright (c) Meta Platforms, Inc. and affiliates
from __future__ import annotations

from functools import reduce
from operator import mul
from typing import TYPE_CHECKING

import torch
import torch.distributed._symmetric_memory as symm_mem
import torch.distributed.config as dist_config
from torch.distributed.tensor.placement_types import _StridedShard, Placement, Shard


if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.distributed.device_mesh import DeviceMesh


def is_symmetric_memory_enabled() -> bool:
    return bool(dist_config.dtensor_use_symmetric_memory)


def _numel(shape: Sequence[int]) -> int:
    return reduce(mul, shape, 1)


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
                f"Sharding dim {placement.dim} greater than tensor ndim {len(max_shape)}"
            )
        max_shape[shard_dim] = _max_shard_dim_size(
            max_shape[shard_dim],
            device_mesh.size(mesh_dim),
            placement,
        )
    return _numel(max_shape)


def _narrow_and_view(base: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return base.narrow(0, 0, _numel(shape)).view(shape).detach()


def _is_eligible_device(device: torch.device) -> bool:
    return device.type == "cuda"


def _should_allocate_symmetric(
    device: torch.device,
    device_mesh: DeviceMesh,
) -> bool:
    return (
        is_symmetric_memory_enabled()
        and _is_eligible_device(device)
        and device_mesh._is_current_rank_part_of_mesh()
    )


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

    max_numel = _compute_max_local_numel(global_shape, device_mesh, placements)
    base = symm_mem.empty(max_numel, dtype=dtype, device=device)
    return _narrow_and_view(base, local_shape)


def copy_to_symmetric_memory(
    local_tensor: torch.Tensor,
    global_shape: torch.Size,
    device_mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> torch.Tensor:
    if not _should_allocate_symmetric(local_tensor.device, device_mesh):
        return local_tensor
    if not local_tensor.is_contiguous():
        return local_tensor

    max_numel = _compute_max_local_numel(global_shape, device_mesh, placements)
    base = symm_mem.empty(
        max_numel, dtype=local_tensor.dtype, device=local_tensor.device
    )
    out = _narrow_and_view(base, local_tensor.shape)
    with torch.no_grad():
        out.copy_(local_tensor)
    return out
