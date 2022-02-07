import copy
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
from .sharding_spec import (
    ChunkShardingSpec,
    ShardingSpec,
)
from torch.distributed._shard.sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
)
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardMetadata,
    ShardedTensor,
)

def _shard_tensor(
    tensor: torch.Tensor, sharding_spec: ShardingSpec, src_rank=0, process_group=None
):
    """
    Given a :class:`torch.Tensor`, it shards that tensor according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    Args:
        tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        A :class:`ShardedTensor` sharded from the given tensor.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
    if not isinstance(sharding_spec, ChunkShardingSpec):
        raise NotImplementedError('Only ChunkShardingspec is supported.')
    if not tensor.is_contiguous():
        raise ValueError('input tensor is not a contiguous Tensor')

    pg = process_group if process_group is not None else distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    # Validate src_rank and sharding_spec are same across all ranks.
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, (src_rank, sharding_spec), group=pg)

    for idx, entry in enumerate(gathered_list):
        if src_rank != entry[0]:  # type: ignore[index]
            raise ValueError(
                f'src_rank={src_rank} on rank: {rank} does not '  # type: ignore[index]
                f'match with src_rank={entry[0]} on rank: {idx}')
        if sharding_spec != entry[1]:  # type: ignore[index]
            raise ValueError(
                f'sharding_spec={sharding_spec} on rank: {rank} does not '  # type: ignore[index]
                f'match with sharding_spec={entry[1]} on rank: {idx}')

    # Rearrange chunks according to placement.
    local_metadata = None
    current_offsets = [0] * len(tensor.size())
    shards_metadata = []
    sharding_dim_size = tensor.size(sharding_spec.dim)  # type: ignore[arg-type]
    split_size = get_split_size(sharding_dim_size, world_size)
    tensor_sizes = list(tensor.size())
    for idx, placement in enumerate(sharding_spec.placements):
        chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        shard_size = copy.deepcopy(tensor_sizes)
        shard_size[sharding_spec.dim] = chunked_dim_size  # type: ignore[index]

        shard_metadata = ShardMetadata(
            shard_offsets=copy.deepcopy(current_offsets),
            shard_sizes=shard_size,
            placement=placement,
        )
        shards_metadata.append(shard_metadata)

        if rank == placement.rank():  # type: ignore[union-attr]
            local_metadata = shard_metadata

        current_offsets[sharding_spec.dim] += chunked_dim_size  # type: ignore[index]

    # Scatter the shards (use broadcast since NCCL doesn't support scatter, this is very inefficient).
    dist.broadcast(tensor, src=src_rank, group=pg)

    # Reshape to get shard for this rank and we don't want autograd
    # recording here for the narrow op and 'local_shard' should be a
    # leaf variable in the autograd graph.
    local_shard = tensor.narrow(
        sharding_spec.dim,  # type: ignore[arg-type]
        local_metadata.shard_offsets[sharding_spec.dim],  # type: ignore[union-attr, arg-type, index]
        local_metadata.shard_sizes[sharding_spec.dim],  # type: ignore[union-attr, index]
    ).clone().detach().contiguous()

    # Sync requires_grad to local_shard.
    local_shard.requires_grad = tensor.requires_grad

    # Create ShardedTensor based on local shards.
    local_shards = [
        Shard(
            tensor=local_shard,
            metadata=local_metadata,  # type: ignore[arg-type]
        )
    ]

    st = ShardedTensor._init_from_local_shards(local_shards, tensor.size(), process_group=pg)

    # Manually set sharding_spec
    st._sharding_spec = sharding_spec

    return st

def shard_parameter(
        module: torch.nn.Module,
        param_name: str,
        sharding_spec: ShardingSpec,
        src_rank=0,
        process_group=None):
    """
    Given a :class:`torch.nn.Module`, a ``param_name`` for a parameter in that
    module, it shards that parameter according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    This method replaces ``module.param_name`` with a
    :class:`torch.distributed._sharded_tensor.ShardedTensor`

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter needs to be sharded.
        param_name (str): Name of the parameter of ``module`` that needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
    # Perform some validation first.
    if not hasattr(module, param_name):
        raise ValueError(f'module: {module} does not have parameter with name: {param_name}')

    tensor = getattr(module, param_name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}')

    if not tensor.is_contiguous():
        raise ValueError(f'param: {param_name} is not a contiguous Tensor')

    st = _shard_tensor(tensor, sharding_spec, src_rank, process_group)

    # Replace param with ShardedTensor.

    # Need to delete the attribute first since param_name might be
    # torch.nn.Parameter and can't be replaced with ShardedTensor which is
    # not torch.nn.Parameter.
    delattr(module, param_name)

    # Now we can set the attribute appropriately.
    setattr(module, param_name, st)
