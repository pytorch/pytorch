# coding=utf-8

import copy
import torch
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
    load_with_process_group,
)
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
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False):
    """
    Creates an :class:`ShardedTensor` from local shards and the global metadata.
    Needs to be called on all ranks in an SPMD fashion.

    Args:
        local_shards (List[:class `torch.distributed._sharded_tensor.Shard`]): A list
            of shards that represent the local shards on this rank.
        sharded_tensor_metadata (:class:`torch.distributed._sharded_tensor.ShardedTensorMetadata`)
            The ShardedTensorMetadata that created manually, represents the global metadata
            of the ShardedTensor, must comply with `local_shards` defined in each rank.
            Note that `sharded_tensor_metadata` must be valid and should also contain
            local shards metadata.

    Keyword args:
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object handle on this rank
    """
    return ShardedTensor._init_from_local_shards(
        local_shards,
        sharded_tensor_metadata,
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
            shard_lengths=shard_size,
            placement=placement,
        )
        shards_metadata.append(shard_metadata)

        if rank == placement.rank():  # type: ignore[union-attr]
            local_metadata = shard_metadata

        current_offsets[sharding_spec.dim] += chunked_dim_size  # type: ignore[index]

    # Scatter the shards (use broadcast since NCCL doesn't support scatter, this is very inefficient).
    dist.broadcast(tensor, src=src_rank, group=pg)

    # Reshape to get shard for this rank.
    local_shard = tensor.narrow(
        sharding_spec.dim,  # type: ignore[arg-type]
        local_metadata.shard_offsets[sharding_spec.dim],  # type: ignore[union-attr, arg-type, index]
        local_metadata.shard_lengths[sharding_spec.dim],  # type: ignore[union-attr, index]
    ).contiguous()

    # Create ShardedTensor based on local shards.
    local_shards = [
        Shard(
            tensor=local_shard,
            metadata=local_metadata,  # type: ignore[arg-type]
        )
    ]
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shards_metadata,
        size=tensor.size(),
        tensor_properties=TensorProperties(
            dtype=local_shard.dtype,
            layout=local_shard.layout,
            requires_grad=local_shard.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=local_shard.is_pinned(),
        )
    )

    st = ShardedTensor._init_from_local_shards(local_shards, sharded_tensor_metadata, process_group=pg)

    # Manually set sharding_spec
    st._sharding_spec = sharding_spec

    # Replace param with ShardedTensor.

    # Need to delete the attribute first since param_name might be
    # torch.nn.Parameter and can't be replaced with ShardedTensor which is
    # not torch.nn.Parameter.
    delattr(module, param_name)

    # Now we can set the attribute appropriately.
    setattr(module, param_name, st)
