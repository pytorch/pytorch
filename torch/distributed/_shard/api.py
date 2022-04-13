from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
)
from .sharding_spec import (
    ShardingSpec,
)
from .replicated_tensor import ReplicatedTensor

def _shard_tensor(
    tensor: torch.Tensor, sharding_spec: ShardingSpec, src_rank=0, process_group=None
) -> ShardedTensor:
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
    if not tensor.is_contiguous():
        raise ValueError('input tensor is not a contiguous Tensor')

    pg = process_group if process_group is not None else distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    current_rank = dist.get_rank(pg)

    # Validate src_rank and sharding_spec are same across all ranks.
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, (src_rank, sharding_spec), group=pg)

    for idx, entry in enumerate(gathered_list):
        if src_rank != entry[0]:  # type: ignore[index]
            raise ValueError(
                f'src_rank={src_rank} on rank: {current_rank} does not '  # type: ignore[index]
                f'match with src_rank={entry[0]} on rank: {idx}')
        if sharding_spec != entry[1]:  # type: ignore[index]
            raise ValueError(
                f'sharding_spec={sharding_spec} on rank: {current_rank} does not '  # type: ignore[index]
                f'match with sharding_spec={entry[1]} on rank: {idx}')

    st = sharding_spec.shard(tensor, src_rank=src_rank, process_group=process_group)

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


def _replicate_tensor(tensor: torch.Tensor, process_group=None) -> ReplicatedTensor:
    """
    Given a :class:`torch.Tensor`, mark it as a ReplicatedTensor where all
    ranks have the same value.

    Args:
        tensor (:class:`torch.Tensor`): the tensor to be marked as replicated.
    Keyword args:
        process_group (ProcessGroup, optional): The process group to replicate on.
            If None, the default process group will be used.
    Returns:
        A :class:`ReplicatedTensor` from the given tensor.

    """
    return ReplicatedTensor(tensor, process_group=process_group)

# Tracks the current process group in the load context manager.
_CURRENT_PROCESS_GROUP = None

@contextmanager
def load_with_process_group(process_group):
    """
    Context manager to set the process group with which to load a ShardedTensor/ReplicatedTensor.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is not None:
        raise RuntimeError(
            'ProcessGroup already set by previous "load_with_process_group" '
            'context manager')
    _CURRENT_PROCESS_GROUP = process_group
    try:
        yield process_group
    finally:
        _CURRENT_PROCESS_GROUP = None

def _get_current_process_group():
    """
    Retrieves the current process group set by ``load_with_process_group``.
    If not set, it just returns the default group.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is None:
        return distributed_c10d._get_default_group()
    else:
        return _CURRENT_PROCESS_GROUP
