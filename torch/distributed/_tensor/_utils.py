from typing import Sequence, Tuple

from torch._prims_common import ShapeType
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Placement, Shard


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
