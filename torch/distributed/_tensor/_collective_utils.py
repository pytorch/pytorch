import logging

from typing import List, Optional

import torch
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import (
    all_to_all,
    broadcast,
    get_global_rank,
    get_rank,
    get_world_size,
    GroupMember,
    ProcessGroup,
    scatter,
    Work,
)

logger = logging.getLogger(__name__)


# TODO: we need to migrate these APIs to be functional collectives


def mesh_scatter(
    output: torch.Tensor,
    scatter_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    """
    scatter a list of tensors to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
    2 to rank 2/3.

    Args:
        output (torch.Tensor): the tensor to receive the scattered list.
        scatter_list (List[torch.Tensor]): the tensor list to be scattered.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Returns:
        A :class:`Work` object
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    if output.is_meta:
        return None
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    # src need to be global rank
    src_for_dim = 0

    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)

    if src_for_dim == get_rank():
        fut = scatter(
            output,
            scatter_list=scatter_list,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )
    else:
        fut = scatter(
            output,
            scatter_list=None,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )

    return fut


def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    """
    broadcast the tensor to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
    broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
    to rank 2/3.

    Args:
        tensor (torch.Tensor): tensor to broadcast.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Returns:
        A :class:`Work` object
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    if tensor.is_meta:
        return None
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    # src need to be global rank
    src_for_dim = 0
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)

    return broadcast(tensor, src=src_for_dim, group=dim_group, async_op=async_op)


# TODO: test uneven split on GLOO and NCCL
def mesh_all_to_all(
    output_tensor_list: List[torch.Tensor],
    input_tensor_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)

    work = None
    # no direct dist.all_to_all support on 'gloo' so we manually do scatters
    if mesh.device_type == "cpu":
        logger.warning(
            "ProcessGroupGloo does not support all_to_all, falling back with scatters!"
        )
        # TODO: pull the handle of uneven case in #492
        dim_group_size = get_world_size(dim_group)
        for i in range(dim_group_size):
            # src need to be global rank
            src_for_dim = i
            if dim_group is not GroupMember.WORLD:
                src_for_dim = get_global_rank(dim_group, i)

            work = scatter(
                output_tensor_list[i],
                input_tensor_list if mesh.get_rank() == src_for_dim else [],
                group=dim_group,
                src=src_for_dim,
                async_op=async_op,
            )
    else:
        work = all_to_all(
            output_tensor_list,
            input_tensor_list,
            dim_group,
            async_op=async_op,
        )
    return work
