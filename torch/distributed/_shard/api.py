import abc
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
    ShardedTensor,
    _PartialTensor
)
from .sharding_spec import (
    ShardingSpec,
    ChunkShardingSpec
)
from .sharding_plan import (
    ShardingPlan
)

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


def _reshard_output(
        module: torch.nn.Module,
        resharding_spec: ShardingSpec) -> torch.nn.Module:
    """
    Hook a module with local shards collection in the forward pass according
    to the given ``resharding_spec``.

    Args:
        module (:class:`torch.nn.Module`): Module whose output needs to be resharded.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            The specification describing how the output of the module will be resharded.

    Returns:
        A :class:`torch.nn.Module` object with collection API hooked.
    """
    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor) or isinstance(output, _PartialTensor):
            return output.reshard(resharding_spec)
        return output
    module.register_forward_hook(hook_func)
    return module


def _collect_local_shard(module: torch.nn.Module) -> torch.nn.Module:
    """
    Hook a module with local shards collection in the forward pass.

    This API is typically used to convert a sharded representation back to data parallel
    representation. In particular, it returns the local tensor for this Shard. If the
    size along the sharding dimension for the local tensor is 1, this dimension is removed
    from the final result. For example a [4, 16] ShardedTensor across 4 ranks is typically
    a local Tensor of size [16] across each rank and not [1, 16] across each rank.

    Args:
        module (:class:`torch.nn.Module`): Module whose output needs to be resharded.

    Returns:
        A :class:`torch.nn.Module` object with collection API hooked.
    """

    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor):
            local_tensor = output.local_tensor()
            # Squeeze the # of dimensions manually, only applicable to ChunkShardingSpec
            sharding_spec = output._sharding_spec
            if isinstance(sharding_spec, ChunkShardingSpec) \
               and local_tensor.size(sharding_spec.dim) == 1:  # type: ignore[attr-defined]
                local_tensor = local_tensor.squeeze(
                    output._sharding_spec.dim  # type: ignore[attr-defined]
                )
            return local_tensor
    module.register_forward_hook(hook_func)
    return module

class ShardedModuleSwapper(abc.ABC):
    @abc.abstractmethod
    def process(self, module: nn.Module) -> nn.Module:
        """
        Processes a module and if needed swaps it with a custom sharded
        Implementation. Should return ``None`` if no swapping should be
        performed.

        The Sharder would produce ShardedTensors for the module based on
        ShardingPlan, and then call the ShardedModuleSwapper. The passed
        in module would consist of ShardedTensors and a common way to
        perform module swapping would be to use the state_dict of the passed
        in module and apply it to the new sharded module via its
        load_state_dict method.
        """
        pass

def shard_module(
    module: nn.Module,
    plan: ShardingPlan,
    sharded_module_swapper: ShardedModuleSwapper = None,
    src_rank=0,
    process_group=None
):
    """
    Shards a given module according to the provided sharding_plan. This method
    first shards all the parameters according to the given sharding_plan. Next,
    If sharded_module_swapper is specified, it recursively traverses the module
    tree and calls ``sharded_module_swapper.process()`` for each submodule.
    If ``sharded_module_swapper.process()`` returns ``None``, the method continues
    to recurse further down the tree. If ``sharded_module_swapper.process()``
    returns an ``nn.Module``, then the current module is replaced with the return
    value and we don't recurse further.

    Args:
        module (:class:`torch.nn.Module`): The module to apply sharding to
        sharding_plan (:class:`torch.distributed._shard.sharding_plan.ShardingPlan`):
            The ShardingPlan which specified param name to ShardingSpec to apply to
            each parameter.

    Keyword args:
        sharded_module_swapper (:class:`torch.distributed._shard.ShardModuleSwapper`, optional):
            Implementation of ShardedModuleSwapper for module swapping.
         src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the module that would be sharded and scattered across the rest
            of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    """
    if sharded_module_swapper is not None:
        raise NotImplementedError("custom module swapping not implemented yet!")

    for mod_prefix, mod in module.named_modules():
        # memo is used to avoid duplicate params between modules, this logic
        # is mostly copied from the module._named_members(), with adaptions
        # to handle parameter sharding.
        memo = set()
        # create a list from the dict keys, because we are mutating the
        # parameters on the fly, we need to use a separate list instead.
        param_keys = list(mod._parameters.keys())
        for k in param_keys:
            v = mod._parameters[k]
            name = mod_prefix + ('.' if mod_prefix else '') + k
            if v is None or v in memo:
                continue
            memo.add(v)
            if name in plan.plan:
                shard_parameter(
                    mod,
                    k,
                    plan.plan[name],
                    src_rank=src_rank,
                    process_group=process_group
                )

        # reshard output if there's an entry in `reshard_output` for this module
        if plan.output_plan is not None and mod_prefix in plan.output_plan:
            _reshard_output(mod, plan.output_plan[mod_prefix])
        # convert the output back to data parallel by calling `_collect_local_shard`
        # if it's specified in the plan.
        if plan.collect_local_shards is not None and mod_prefix in plan.collect_local_shards:
            _collect_local_shard(mod)
