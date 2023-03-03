from typing import Optional, Sequence

import torch
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed._tensor.placement_types import Placement, Replicate, Shard


def compute_local_tensor_size(
    size: torch.Size, device_mesh: DeviceMesh, placements: Sequence[Placement]
) -> Optional[torch.Size]:
    """
    Args:
        size(torch.Size): define the shape of the whole Dtensor.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placement: a sequence of :class:`Placement` type: Shard, Replicate

    Returns:
        A :class:`torch.Size` for the local tensor on the device_mesh
    """
    if device_mesh.get_coordinate() is None:
        return None
    else:
        local_size = list(size)
        rank_coordinates = device_mesh.get_coordinate()
        if rank_coordinates is None:
            return None
        for idx, placement in enumerate(placements):
            if isinstance(placement, Replicate):
                continue
            elif isinstance(placement, Shard):
                curr_coordinate = rank_coordinates[idx]
                shard_dim = placement.dim
                len_before_shard = local_size[shard_dim]
                num_chucks = device_mesh.size(idx)

                len_after_shard, _ = placement._local_shard_size_on_dim(
                    len_before_shard, num_chucks, curr_coordinate
                )
                local_size[shard_dim] = len_after_shard
            else:
                raise RuntimeError(f"placement type {type(placement)} not supported!")

        return torch.Size(local_size)
