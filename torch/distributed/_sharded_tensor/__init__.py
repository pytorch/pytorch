from typing import List

import torch
from torch.distributed._sharding_spec import ShardingSpec
from .api import (
    CreateOp,
    Shard,
    ShardedTensor,
    ShardedTensorMetadata,
    TensorInitParams,
    TensorProperties,
    load_with_process_group,
)


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
    Creates an empty :class:`ShardedTensor`. Needs to be called on all ranks in an SPMD fashion.

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
    Creates a ones :class:`ShardedTensor`. Needs to be called on all ranks in an SPMD fashion.

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


def normal_(sharded_tensor: ShardedTensor, mean: float = 0., std: float = 1.) -> ShardedTensor:
    r"""
    Fills the Tensors in sharded_tensor.local_shards with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        sharded_tensor: tensor sharded across devices
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    """
    for i in range(len(sharded_tensor.local_shards())):
        torch.nn.init.normal_(sharded_tensor.local_shards()[i].tensor, mean=mean, std=std)
    return sharded_tensor


def uniform_(sharded_tensor: ShardedTensor, a: float = 0., b: float = 1.) -> ShardedTensor:
    r"""
    Fills the Tensor in sharded_tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        sharded_tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    for i in range(len(sharded_tensor.local_shards())):
        torch.nn.init.uniform_(sharded_tensor.local_shards()[i].tensor, a=a, b=b)
    return sharded_tensor


def kaiming_uniform_(sharded_tensor: ShardedTensor,
                     a=0, mode='fan_in',
                     nonlinearity='leaky_relu') -> ShardedTensor:
    r"""
    Fills the Tensors in sharded_tensor.local_shards with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        sharded_tensor: tensor sharded across devices
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
    for i in range(len(sharded_tensor.local_shards())):
        torch.nn.init.kaiming_uniform_(sharded_tensor.local_shards()[i].tensor,
                                       a=a, mode=mode, nonlinearity=nonlinearity)
    return sharded_tensor


def init_from_local_shards(local_shards: List[Shard],
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
