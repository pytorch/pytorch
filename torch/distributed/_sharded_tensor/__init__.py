# coding=utf-8

import copy
import torch
from torch.distributed.nn.functional import (
    reduce_scatter,
)
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
    ShardingSpec,
)
from torch.distributed._sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
)
from typing import List

from .api import (
    CreateOp,
    Shard,
    ShardMetadata,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorInitParams,
    TensorProperties,
)
from .utils import load_with_process_group
import torch.distributed as dist
from torch.distributed import distributed_c10d


def empty(sharding_spec: ShardingSpec,
          *size,
          dtype=None,
          layout=torch.strided,
          requires_grad=False,
          pin_memory=False,
          memory_format=torch.contiguous_format,
          process_group=None,
          init_rrefs=False):
    """
    Returns a :class:`ShardedTensor` filled with uninitialized data.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    tensor_properties = TensorProperties(dtype=dtype, layout=layout,
                                         requires_grad=requires_grad,
                                         pin_memory=pin_memory, memory_format=memory_format, )
    tensor_init_params = TensorInitParams(create_op=CreateOp.EMPTY, tensor_properties=tensor_properties, )
    return ShardedTensor(
        sharding_spec,
        *size,
        tensor_init_params=tensor_init_params,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )

def ones(sharding_spec: ShardingSpec,
         *size,
         dtype=None,
         layout=torch.strided,
         requires_grad=False,
         pin_memory=False,
         memory_format=torch.contiguous_format,
         process_group=None,
         init_rrefs=False):
    """
    Returns a :class:`ShardedTensor` with the scalar value 1.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    tensor_properties = TensorProperties(dtype=dtype, layout=layout,
                                         requires_grad=requires_grad,
                                         pin_memory=pin_memory, memory_format=memory_format, )
    tensor_init_params = TensorInitParams(create_op=CreateOp.ONES, tensor_properties=tensor_properties)
    return ShardedTensor(
        sharding_spec,
        *size,
        tensor_init_params=tensor_init_params,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def rand(sharding_spec: ShardingSpec,
         *size,
         dtype=None,
         layout=torch.strided,
         requires_grad=False,
         pin_memory=False,
         memory_format=torch.contiguous_format,
         process_group=None,
         init_rrefs=False):
    """
    Returns a :class:`ShardedTensor` filled with random numbers from a uniform distribution on the
        interval :math:`[0, 1)`. Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    tensor_properties = TensorProperties(
        dtype=dtype, layout=layout, requires_grad=requires_grad,
        pin_memory=pin_memory, memory_format=memory_format
    )
    tensor_init_params = TensorInitParams(create_op=CreateOp.RAND, tensor_properties=tensor_properties, )
    return ShardedTensor(
        sharding_spec,
        *size,
        tensor_init_params=tensor_init_params,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def zeros(sharding_spec: ShardingSpec,
          *size,
          dtype=None,
          layout=torch.strided,
          requires_grad=False,
          pin_memory=False,
          memory_format=torch.contiguous_format,
          process_group=None,
          init_rrefs=False):
    """
    Returns a :class:`ShardedTensor` filled with the scalar value 0.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    tensor_properties = TensorProperties(
        dtype=dtype, layout=layout, requires_grad=requires_grad,
        pin_memory=pin_memory, memory_format=memory_format,
    )
    tensor_init_params = TensorInitParams(create_op=CreateOp.ZEROS, tensor_properties=tensor_properties, )
    return ShardedTensor(
        sharding_spec,
        *size,
        tensor_init_params=tensor_init_params,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def full(sharding_spec: ShardingSpec,
         size,
         fill_value=torch.types.Number,
         dtype=None,
         layout=torch.strided,
         requires_grad=False,
         pin_memory=False,
         memory_format=torch.contiguous_format,
         process_group=None,
         init_rrefs=False):
    """
    Creates a :class:`ShardedTensor` filled with fill_value. The tensor’s dtype
        is inferred from fill_value. If dtype is specified, it will override the
        inferred type from fill_value. Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.
        fill_value (Scalar) – the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    tensor_properties = TensorProperties(
        dtype=dtype, layout=layout, requires_grad=requires_grad,
        pin_memory=pin_memory, memory_format=memory_format,
    )
    tensor_init_params = TensorInitParams(
        create_op=CreateOp.FULL, fill_value=fill_value, tensor_properties=tensor_properties)
    return ShardedTensor(
        sharding_spec,
        *size,
        tensor_init_params=tensor_init_params,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def init_from_local_shards(
        local_shards: List[Shard],
        *global_size,
        process_group=None,
        init_rrefs=False):
    """
    Creates an :class:`ShardedTensor` from local shards and the global metadata.
    Needs to be called on all ranks in an SPMD fashion.

    Args:
        local_shards (List[:class `torch.distributed._sharded_tensor.Shard`]): A list
            of shards that represent the local shards on this rank.
        global_size (int...):  a list, tuple, or `torch.Size` of integers defining the
            shape of the overall sharded tensor.

    Keyword args:
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object handle on this rank


    Examples:
      Suppose we want construct a sharded tensor on two ranks, global size = (10, 5),
      each shard have a (5, 5) local tensor, we can do it like below:

      on rank 0:
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[0, 0]
        >>>     shard_lengths=[5, 5]
        >>>     placement="rank:0/cuda:0"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])

      on rank 1:
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[5, 0]
        >>>     shard_lengths=[5, 5]
        >>>     placement="rank:1/cuda:1"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])
    """
    return ShardedTensor._init_from_local_shards(
        local_shards,
        *global_size,
        process_group=process_group,
        init_rrefs=init_rrefs
    )

def state_dict_hook(module, destination, prefix, local_metadata):
    """
    Hook to add ShardedTensor to Module's ``state_dict``. Needs to be
    registered to the Module using
    :meth:`torch.nn.Module._register_state_dict_hook`.
    """
    _recurse_update_dict(module, destination, prefix)

def pre_load_state_dict_hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    """
    Pre-load state dict hook to add ShardedTensor to the module.
    """
    _recurse_update_module(module, state_dict, prefix)

def _recurse_update_module(module, state_dict, prefix):
    for attr_name, attr in module.__dict__.items():
        key = prefix + attr_name
        if key in state_dict:
            if isinstance(state_dict[key], ShardedTensor):
                setattr(module, attr_name, state_dict[key])

    for submodule_name, submodule in module.named_modules():
        key = prefix + submodule_name
        if submodule_name:
            _recurse_update_module(submodule, state_dict, key + '.')


def _recurse_update_dict(module, destination, prefix):
    for attr_name, attr in module.__dict__.items():
        if isinstance(attr, ShardedTensor):
            destination[prefix + attr_name] = attr

    for submodule_name, submodule in module.named_modules():
        if submodule_name != '':
            _recurse_update_dict(submodule, destination, prefix + submodule_name + '.')

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
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    .. warning::
        Only :class:`torch.distributed._sharding_spec.ShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
    # Perform some validation first.
    if not isinstance(sharding_spec, ChunkShardingSpec):
        raise ValueError('Only ChunkShardingspec is supported.')

    if not hasattr(module, param_name):
        raise ValueError(f'module: {module} does not have parameter with name: {param_name}')

    tensor = getattr(module, param_name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}')

    if not tensor.is_contiguous():
        raise ValueError(f'param: {param_name} is not a contiguous Tensor')

    pg = process_group if process_group is not None else distributed_c10d._get_default_group()
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    # Validate src_rank and sharding_spec are same across all ranks.
    gathered_list = [None] * world_size
    with torch.cuda.device(tensor.device):
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

        if rank == placement.rank():  # type: ignore[union-attr]
            local_metadata = shard_metadata

        current_offsets[sharding_spec.dim] += chunked_dim_size  # type: ignore[index]

    # Scatter the shards (use broadcast since NCCL doesn't support scatter, this is very inefficient).
    dist.broadcast(tensor, src=src_rank, group=pg)

    # We don't want autograd recording here for the narrow op and
    # 'local_shard' should be a leaf variable in the autograd graph
    with torch.no_grad():
        # Reshape to get shard for this rank.
        local_shard = tensor.narrow(
            sharding_spec.dim,  # type: ignore[arg-type]
            local_metadata.shard_offsets[sharding_spec.dim],  # type: ignore[union-attr, arg-type, index]
            local_metadata.shard_sizes[sharding_spec.dim],  # type: ignore[union-attr, index]
        ).contiguous()

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

    # Replace param with ShardedTensor.

    # Need to delete the attribute first since param_name might be
    # torch.nn.Parameter and can't be replaced with ShardedTensor which is
    # not torch.nn.Parameter.
    delattr(module, param_name)

    # Now we can set the attribute appropriately.
    setattr(module, param_name, st)


def init_sharded_tensor_from_local_result(
    sharded_tensor,
    local_result,
    tensor_shard_dim,
    result_shard_dim,
    world_size,
    pg,
):
    """
    Given a sharded tensor and local_result from an op on top of it. We want
    to create a new sharded tensor from the local_result so that the the next
    op can be performed on the basis of the new sharded tensor. This can seen
    as the last step of the first phase of the Megatron-LM style model(tensor)
    parallelism.

    Args:
        sharded_tensor: Sharded tensor which the op was performed on.
        local_result: A tensor which is from the op performed on the local_shard of
            the sharded_tensor.
        tensor_shard_dim: Dim which the tensor is sharded on.
        result_shard_dim: Dim which the new sharded tensor will be sharded on.
        world_size: number of ranks.
        pg (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Return: new sharded tensor from the local_result.
    """
    sharded_weight_metadata = sharded_tensor.local_shards()[0].metadata
    current_offsets = [0] * len(local_result.size())
    current_offsets[result_shard_dim] = sharded_weight_metadata.shard_offsets[tensor_shard_dim]
    local_shard_metadata = ShardMetadata(
        shard_offsets=current_offsets,
        shard_sizes=list(local_result.size()),
        placement=sharded_weight_metadata.placement,
    )
    local_shards = [Shard(local_result, local_shard_metadata)]
    global_size = list(local_result.size())
    global_size[result_shard_dim] = sharded_tensor.size(tensor_shard_dim)

    new_st = ShardedTensor._init_from_local_shards(local_shards, tuple(global_size), process_group=pg)

    # Manually set sharding_spec
    new_st._sharding_spec = copy.deepcopy(sharded_tensor._sharding_spec)
    new_st._sharding_spec.dim = result_shard_dim
    return new_st


def merge_sharded_local_results(
    sharded_local_result,
    world_size,
    pg=None,
):
    """
    Given a local result sharded tensor, we want to merge it into a local tensor.

    Args:
        sharded_local_result: Sharded tensor which we need to merge.
        world_size: number of ranks.
        pg (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Return: new sharded tensor from the local_result.
    """
    from torch.distributed._sharded_tensor.ops._common import (
        _result_distribute_with_col_rearrange,
    )
    local_shard = sharded_local_result.local_shards()[0].tensor
    local_results = []
    for i in range(local_shard.size(0)):
        local_results.append(local_shard[i].t().contiguous())

    # Distribute results to each rank with col rearrangement.
    output = _result_distribute_with_col_rearrange(
        local_results,
        local_shard,
        world_size,
        sharded_local_result,
        pg
    )

    # transpose the output and return result.
    return output.t().contiguous()


def aggregate_partial_tensor_list(
    tensor_list,
    rank,
    pg=None,
):
    """
    Given parital result from op on a sharded tensor. We want to aggregate the
    partial result to create a fully synced local tensor. This can seen as the
    last step of the second phase of the Megatron-LM style model(tensor) parallelism.

    Args:
        tensor_list: List of partial results to be aggergated on.
        rank: # of cuda process.
        pg (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Return: new sharded tensor from the local_result.
    """
    local_result = torch.empty_like(tensor_list[rank])
    return reduce_scatter(local_result, tensor_list, group=pg)
