from typing import cast, List, Sequence, Tuple

import torch
from torch._prims_common import ShapeType
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import (
    _Partial,
    Placement,
    Replicate,
    Shard,
)


def compute_local_shape(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[int, ...]:
    """
    Compute the shape of a local shard of the given DTensor on its current
    coordinate of the mesh.
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty shape
        return ()
    else:
        local_shape = list(global_shape)  # start with global shape
        ndim = len(global_shape)
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert (
                    shard_dim < ndim
                ), f"Sharding dim {shard_dim} greater than tensor ndim {ndim}"
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim], mesh_dim_size, my_coordinate[idx]
                )
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size

        return tuple(local_shape)


def compute_local_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[int, ...]:
    """
    Compute the offsets of a local shard of the given DTensor on its current
    global rank. This is mostly used by distributed checkpointing to know the
    exact offsets of the local shard.
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ()
    else:
        local_offsets = [0] * len(global_shape)
        local_shape = list(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )
                local_shape[shard_dim] = shard_size
                local_offsets[shard_dim] = shard_offset
        return tuple(local_offsets)


def compute_global_tensor_info(
    tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[List[int], List[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.
    The local size is multiplited by `world_size` per Sharding dim.
    The local stride is multiplited by `world_size` per Sharding dim, as long as the
    dimension is outside sharding dim.

    For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Local tensor which DTensor will be constructed from.
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: A List of int which specifies the size of DTensor which build
            on top of the local tensor.
        tensor_stride: A List of int which specifies the stride of DTensor.
    """
    tensor_shape = list(tensor.size())
    tensor_stride = list(tensor.stride())
    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            local_dim_size = tensor_shape[shard_dim]
            tensor_shape[shard_dim] = local_dim_size * mesh_dim_size

            # recover tensor stride by modifying the stride that larger than
            # the current stride on the shard_dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    # rescale the stride by the shard size
                    tensor_stride[i] = tensor_stride[i] * mesh_dim_size
        elif not isinstance(placement, (Replicate, _Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")
    return tensor_shape, tensor_stride
