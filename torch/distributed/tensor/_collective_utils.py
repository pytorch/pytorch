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
from torch.distributed._local_tensor import local_tensor_mode
from torch.distributed._pycute.int_tuple import crd2idx, idx2crd
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
    if mesh.device_type == "cpu" and local_tensor_mode() is None:
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


def all_permute_mesh_dim(
    input: torch.Tensor,
    full_tensor_shape: torch.Size,
    src_distribution: "dtensor_spec.ShardOrder",
    tgt_distribution: "dtensor_spec.ShardOrder",
    mesh: DeviceMesh,
):
    """
    Perform an AllPermute operation to redistribute a tensor from source to target sharding.

    AllPermute is a collective communication primitive that can transform any tensor distribution
    τ1 to τ2 if their local and global shapes match. Unlike traditional redistributions that may
    require multiple AllGather operations, AllPermute achieves the transformation through a single
    permutation-based communication, significantly reducing memory overhead.

    Mathematical Background:
    ------------------------
    Given a DeviceMesh with dimensions {X, Y, Z, ...} and a tensor with shape [D0, D1, D2, ...],
    sharding annotations specify how tensor dimensions are distributed across mesh dimensions.

    Notation (from https://arxiv.org/pdf/2112.01075 "section 2.1 Distributed array types"):
        - [32{X,Y}512, 128]: Tensor of shape [32*512, 128] sharded on mesh dims X,Y for dim 0
        - [128{Y}512, 32{X}128]: First tensor dim sharded on mesh Y, second dim on mesh X

    Examples:
    ---------
    Given mesh with size {X:4, Y:4, Z:16}:

    Example 1: Swap mesh dimension order for the same tensor dimension
        Source: [32{X,Y}512, 128]  → Target: [32{Y,X}512, 128]
        This swaps the order in which mesh dimensions X and Y shard the first tensor dimension.

    Example 2: Swap which tensor dimensions are sharded on which mesh dimensions
        Source: [128{Y}512, 32{X}128] → Target: [128{X}512, 32{Y}128]
        Tensor dimension 0 moves from mesh dim Y to X, dimension 1 moves from X to Y.

    Example 3: Change mesh dimensions entirely while preserving shapes
        Source: [32{X,Y}512, 128] → Target: [32{Z}512, 128]
        Re-shard from mesh dimensions X,Y to mesh dimension Z.

    Algorithm:
    ----------
    1. For each source rank, compute its position in the global tensor based on src_distribution
    2. Determine which target rank should receive this data based on tgt_distribution
    3. Build a permutation mapping: src_rank → tgt_rank
    4. Execute permute_tensor to swap data according to the mapping

    The key insight is that each rank knows exactly where its data should go in the target
    distribution without requiring intermediate gather operations.

    Args:
        input (torch.Tensor): Local tensor shard to redistribute. Must be the actual data
            held by the current rank according to src_distribution.
        full_tensor_shape (torch.Size): Shape of the global (unsharded) tensor before any
            distribution. Used to compute correct offsets and mappings.
        src_distribution (ShardOrder): Description of how the input tensor is currently
            distributed across the mesh. Specifies which tensor dimensions are sharded
            on which mesh dimensions.
        tgt_distribution (ShardOrder): Target distribution specification. Describes the
            desired sharding pattern after the AllPermute operation.
        mesh (DeviceMesh): The device mesh over which the tensor is distributed. Defines
            the process group topology for communication.

    Returns:
        torch.Tensor: The redistributed local tensor shard according to tgt_distribution.
            The returned tensor represents this rank's portion of the global tensor under
            the new distribution scheme.

    Preconditions:
        - Source and target distributions must have matching local and global tensor shapes
        - Input must be a local tensor (not a DTensor wrapper)
        - The mesh topology must match both src_distribution and tgt_distribution

    Example Usage:
        >>> # 2D mesh [4, 4] with tensor shape [64, 128]
        >>> mesh = DeviceMesh(
        ...     "cuda", [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        ... )
        >>> # Redistribute from [64{mesh_dim=0}, 128] to [64, 128{mesh_dim=1}]
        >>> src_dist = ShardOrder([ShardEntry(tensor_dim=0, mesh_dims=[0])])
        >>> tgt_dist = ShardOrder([ShardEntry(tensor_dim=1, mesh_dims=[1])])
        >>> output = all_permute_mesh_dim(
        ...     local_shard, (64, 128), src_dist, tgt_dist, mesh
        ... )
    """
    orig_tensor_shape = list(input.shape)
    for entry in src_distribution:
        tensor_dim = entry.tensor_dim
        shard_tensor_on_mesh_dims = entry.mesh_dims
        for mesh_dim in shard_tensor_on_mesh_dims:
            orig_tensor_shape[tensor_dim] *= mesh.size(mesh_dim)

    permuted_rank = list(range(0, torch.distributed.get_world_size()))
    for src_rank in range(mesh.size()):
        src_coordinate = idx2crd(src_rank, mesh.shape)
        tgt_coordinate = [0] * mesh.ndim
        src_distr_idx, tgt_distr_idx = 0, 0
        for tensor_dim in range(len(full_tensor_shape)):
            src_distribute_to_mesh_dims: tuple[int, ...] = ()
            if (
                src_distr_idx < len(src_distribution)
                and src_distribution[src_distr_idx].tensor_dim == tensor_dim
            ):
                src_distribute_to_mesh_dims = src_distribution[src_distr_idx].mesh_dims
                src_distr_idx += 1
            tgt_distribute_to_mesh_dims: tuple[int, ...] = ()
            if (
                tgt_distr_idx < len(tgt_distribution)
                and tgt_distribution[tgt_distr_idx].tensor_dim == tensor_dim
            ):
                tgt_distribute_to_mesh_dims = tgt_distribution[tgt_distr_idx].mesh_dims
                tgt_distr_idx += 1

            # check if local size and global size match between src and tgt
            # distribution for this tensor dim. In addition, make sure there is
            # no padding needed with the sharding specification.
            if (
                src_prod := math.prod(
                    [mesh.size(i) for i in src_distribute_to_mesh_dims]
                )
            ) != (
                tgt_prod := math.prod(
                    [mesh.size(i) for i in tgt_distribute_to_mesh_dims]
                )
            ):
                raise ValueError(
                    f"local size mismatch between source (shard into: {src_prod} parts) "
                    f"and target (shard into: {tgt_prod} parts) distribution for tensor dim {tensor_dim}."
                )
            for mesh_dim in src_distribute_to_mesh_dims:
                if full_tensor_shape[tensor_dim] % mesh.size(mesh_dim) != 0:
                    raise ValueError(
                        f"tensor dim {tensor_dim} (size: {full_tensor_shape[tensor_dim]}) "
                        f"is not divisible by mesh dim {mesh_dim} (size: {mesh.size(mesh_dim)}) "
                        f"based on source distribution."
                    )
            for mesh_dim in tgt_distribute_to_mesh_dims:
                if full_tensor_shape[tensor_dim] % mesh.size(mesh_dim) != 0:
                    raise ValueError(
                        f"tensor dim {tensor_dim} (size: {full_tensor_shape[tensor_dim]}) "
                        f"is not divisible by mesh dim {mesh_dim} (size: {mesh.size(mesh_dim)}) "
                        f"based on target distribution."
                    )

            # main algorithm to find which rank to send data to
            dividend_size = 0
            prev_size = full_tensor_shape[tensor_dim]
            for mesh_dim in src_distribute_to_mesh_dims:
                cur_size = math.ceil(prev_size / mesh.size(mesh_dim))
                dividend_size += src_coordinate[mesh_dim] * cur_size  # type: ignore[index, operator]

                prev_size = cur_size
            divisor_size = full_tensor_shape[tensor_dim]
            for mesh_dim in tgt_distribute_to_mesh_dims:
                divisor_size = math.ceil(divisor_size / mesh.size(mesh_dim))
                tgt_coordinate[mesh_dim], dividend_size = divmod(
                    dividend_size, divisor_size
                )
        tgt_rank = crd2idx(tuple(tgt_coordinate), mesh.shape)
        # indicate to send src_rank data to tgt_rank
        permuted_rank[src_rank] = tgt_rank
    assert sorted(permuted_rank) == list(range(len(permuted_rank)))
    output = funcol.permute_tensor(input, permuted_rank, mesh)
    if isinstance(output, funcol.AsyncCollectiveTensor):
        output = output.wait()
    return output


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
       are quite trivial (i.e. we only need to narrow or simple division)
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
