# mypy: allow-untyped-defs
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._dtensor_spec as dtensor_spec
from torch._C._distributed_c10d import _resolve_process_group
from torch._logging import warning_once
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import (
    _get_group_size_by_name,
    broadcast,
    get_group_rank,
    get_rank,
    ProcessGroup,
    scatter,
    Work,
)


logger = logging.getLogger(__name__)


@torch.library.register_fake("_dtensor::shard_dim_alltoall")
def _shard_dim_alltoall_meta(input, gather_dim, shard_dim, group_name):
    group_size = _get_group_size_by_name(group_name)
    stacked_list = [torch.empty_like(input) for _ in range(group_size)]
    group = _resolve_process_group(group_name)
    group_rank = get_group_rank(group, get_rank())

    return (
        torch.cat(stacked_list, dim=gather_dim)
        .chunk(group_size, dim=shard_dim)[group_rank]
        .contiguous()
    )


def shard_dim_alltoall(input, gather_dim, shard_dim, mesh, mesh_dim):
    if mesh.device_type == "cpu":
        # Gloo does not support alltoall, so falling back to allgather + chunk
        warning_once(
            logger,
            "CPU process group does not support alltoall yet, falling back with allgather + chunk!",
        )
        out = funcol.all_gather_tensor(input, gather_dim, (mesh, mesh_dim))
        if isinstance(out, funcol.AsyncCollectiveTensor):
            # stick to the same behavior for the alltoall case, remove this once we enable alltoall async
            out = out.wait()
        out = torch.chunk(out, mesh.size(mesh_dim), dim=shard_dim)[
            mesh.get_local_rank(mesh_dim)
        ]
        return out.contiguous()

    group_name = funcol._resolve_group_name((mesh, mesh_dim))
    # TODO: enable async op for shard_dim_alltoall
    return torch.ops._dtensor.shard_dim_alltoall(
        input, gather_dim, shard_dim, group_name
    )


def mesh_scatter(
    output: torch.Tensor,
    scatter_list: list[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
    *,
    group_src: int = 0,
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

    Keyword args:
        group_src (int, optional): the group rank of the source data for the
        logical/global tensor, on the specific mesh dimension. By default, we
        use ``group_rank=0`` on each DeviceMesh dimension as the source data
        to preserve the single-device semantic. If passing ``None`` explicitly,
        this method simply uses its local data with no communication.

    Returns:
        A :class:`Work` object
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    if output.is_meta:
        return None
    dim_group = mesh.get_group(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)

    if group_src == get_rank(dim_group):
        fut = scatter(
            output,
            scatter_list=scatter_list,
            group=dim_group,
            async_op=async_op,
            group_src=group_src,
        )
    else:
        fut = scatter(
            output,
            scatter_list=None,
            group=dim_group,
            async_op=async_op,
            group_src=group_src,
        )

    return fut


def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
    *,
    group_src: int = 0,
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

    Keyword args:
        group_src (int, optional): the group rank of the source data for the
        logical/global tensor, on the specific mesh dimension. By default, we
        use ``group_rank=0`` on each DeviceMesh dimension as the source data
        to preserve the single-device semantic. If passing ``None`` explicitly,
        this method simply uses its local data with no communication.

    Returns:
        A :class:`Work` object
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    if tensor.is_meta:
        return None
    dim_group = mesh.get_group(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)

    return broadcast(tensor, group=dim_group, async_op=async_op, group_src=group_src)


def pad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor:
    if pad_size == 0:
        return tensor
    pad = [0, 0] * (tensor.ndim - pad_dim)
    pad[-1] = pad_size
    return torch.nn.functional.pad(tensor, pad)


def unpad_tensor(tensor: torch.Tensor, pad_dim: int, pad_size: int) -> torch.Tensor:
    if pad_size == 0:
        return tensor
    return tensor.narrow(
        pad_dim,
        start=0,
        length=tensor.size(pad_dim) - pad_size,
    )


def fill_empty_tensor_to_shards(
    shards: list[torch.Tensor], shard_dim: int, num_empty_tensors: int
) -> list[torch.Tensor]:
    if num_empty_tensors == 0:
        return shards
    tensor_size = list(shards[0].size())
    tensor_size[shard_dim] = 0
    tensor = shards[0].new_zeros(tensor_size)
    shards.extend(tensor for _ in range(num_empty_tensors))
    return shards


def check_tensor_meta(
    local_tensor, check_shape_stride=False
) -> Optional["dtensor_spec.TensorMeta"]:
    local_metadata = {
        "dtype": local_tensor.dtype,
        "requires_grad": local_tensor.requires_grad,
    }

    if check_shape_stride:
        local_metadata.update(
            {"shape": local_tensor.shape, "stride": local_tensor.stride()}
        )

    gathered_metadata = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered_metadata, local_metadata)

    # Check if metadata is consistent across ranks
    if not all(meta == local_metadata for meta in gathered_metadata):
        raise ValueError(
            "Inconsistent tensor metadata (including shape and stride) across ranks."
        )
    return None


def spec_to_bytes(spec: "dtensor_spec.DTensorSpec") -> int:
    assert spec.tensor_meta is not None, "spec should have tensor meta defined!"
    return spec.tensor_meta.dtype.itemsize * math.prod(spec.shape)


@dataclass
class MeshTopoInfo:
    """
    Mesh information for collective cost estimation
    """

    mesh: DeviceMesh
    mesh_dim_devices: list[int]
    mesh_dim_bandwidth: list[float]
    mesh_dim_latency: list[float]

    @staticmethod
    @lru_cache(None)
    def build_from_mesh(mesh: DeviceMesh) -> "MeshTopoInfo":
        # Generate mesh topology info for intra-host/inter-host communication pattern
        # Note that we made bunch of assumptions for simplicity:
        # 1. we assume the mesh is homogeneous, and it's gpu/nccl model
        # 2. we assume gpu arch is Ampere or Hopper
        # 3. we assume collectives are all ring base algo for now
        num_devices_per_host = _mesh_resources.num_devices_per_host(mesh.device_type)
        # the base bw number (intra-node), GB/s
        base_bw = 87.7
        mesh_dim_bandwidth = [base_bw] * mesh.ndim
        # the latency in terms of us (intra-node, nv-link)
        mesh_dim_latency = [0.6] * mesh.ndim
        mesh_dim_devices = [1] * mesh.ndim

        total_num_devices = 1
        for mesh_dim in reversed(range(mesh.ndim)):
            num_devices = mesh.size(mesh_dim)
            mesh_dim_devices[mesh_dim] = num_devices
            total_num_devices *= num_devices
            if total_num_devices > num_devices_per_host:
                # magic number for inter-host communication bandwidth/latency factor
                # This number assumes latest GPU arch, i.e. Ampere or Hopper
                # TODO: see if we need to tweak this or offer a way for user
                # to specify the bandwidths/latency
                mesh_dim_bandwidth[mesh_dim] *= 0.22
                # set to ethernet latency for inter-host
                mesh_dim_latency[mesh_dim] = 2.7

        return MeshTopoInfo(
            mesh, mesh_dim_devices, mesh_dim_bandwidth, mesh_dim_latency
        )


def allgather_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    num_hops = num_devices_on_mesh_dim - 1
    # base latency + comm latency
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]  # us
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth  # s
    return latency + bw * 1e6  # rescale to us


def allreduce_cost(bytes_gb: float, mesh_topo: MeshTopoInfo, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    # allreduce have almost 2x comm bytes compare to allgather/reduce_scatter
    num_hops = 2 * (num_devices_on_mesh_dim - 1)

    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth
    return latency + bw * 1e6


def reduce_scatter_cost(
    bytes_gb: float,
    mesh_topo: MeshTopoInfo,
    mesh_dim: int,
) -> float:
    num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[mesh_dim]
    mesh_dim_bandwidth = mesh_topo.mesh_dim_bandwidth[mesh_dim]
    num_hops = num_devices_on_mesh_dim - 1
    # base latency + comm latency
    latency = 6.6 + num_hops * mesh_topo.mesh_dim_latency[mesh_dim]
    bw = (bytes_gb * num_hops / num_devices_on_mesh_dim) / mesh_dim_bandwidth
    return latency + bw * 1e6


def redistribute_cost(
    current_spec: "dtensor_spec.DTensorSpec",
    target_spec: "dtensor_spec.DTensorSpec",
) -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trival (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
    if current_spec.mesh != target_spec.mesh:
        # make infinite cost if meshes are not same
        # TODO: see if we want to support this once there's cross mesh communication
        return float("inf")

    if current_spec.is_replicated():
        # short-cut:
        # comm cost is 0 if current spec is already full replication
        return 0.0

    mesh_topo = MeshTopoInfo.build_from_mesh(current_spec.mesh)
    cost = 0.0
    comm_bytes_gb = (
        spec_to_bytes(current_spec) / current_spec.num_shards / 1024 / 1024 / 1024
    )
    # Transformation that considered for redistribute cost:
    # 1. allgather 2. alltoall
    # 3. allreduce 4. reduce_scatter
    for i, (current, target) in enumerate(
        zip(current_spec.placements, target_spec.placements)
    ):
        if current == target:
            continue

        num_devices_on_mesh_dim = mesh_topo.mesh_dim_devices[i]
        if current.is_shard() and target.is_replicate():
            # allgather gives larger comm bytes
            comm_bytes_gb *= num_devices_on_mesh_dim
            # add up allgather comm cost
            cost += allgather_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_shard() and target.is_shard():
            # should be alltoall comm, since we haven't implement it yet, add penalty
            # to favor allgather instead
            cost += allgather_cost(comm_bytes_gb, mesh_topo, i) + 1.0
        elif current.is_partial() and target.is_replicate():
            # add up allreduce comm cost
            cost += allreduce_cost(comm_bytes_gb, mesh_topo, i)
        elif current.is_partial() and target.is_shard():
            # add up reduce_scatter comm cost
            cost += reduce_scatter_cost(comm_bytes_gb, mesh_topo, i)
            # after reduce_scatter the comm bytes for further collectives halved.
            comm_bytes_gb /= num_devices_on_mesh_dim
        elif current.is_shard() and target.is_partial():
            # ban shard -> partial as it does not make sense to perform
            # this redistribute
            return float("inf")

    return cost
