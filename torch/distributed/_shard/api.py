# mypy: allow-untyped-defs
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import ShardedTensor

from .sharder import Sharder
from .sharding_plan import ShardingPlan
from .sharding_spec import ChunkShardingSpec, ShardingSpec


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
        raise ValueError("input tensor is not a contiguous Tensor")

    pg = (
        process_group
        if process_group is not None
        else distributed_c10d._get_default_group()
    )
    world_size = dist.get_world_size(pg)
    current_rank = dist.get_rank(pg)

    # Validate src_rank and sharding_spec are same across all ranks.
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, (src_rank, sharding_spec), group=pg)

    for idx, entry in enumerate(gathered_list):
        if src_rank != entry[0]:  # type: ignore[index]
            raise ValueError(
                f"src_rank={src_rank} on rank: {current_rank} does not "  # type: ignore[index]
                f"match with src_rank={entry[0]} on rank: {idx}"  # type: ignore[index]
            )
        if sharding_spec != entry[1]:  # type: ignore[index]
            raise ValueError(
                f"sharding_spec={sharding_spec} on rank: {current_rank} does not "  # type: ignore[index]
                f"match with sharding_spec={entry[1]} on rank: {idx}"  # type: ignore[index]
            )

    st = sharding_spec.shard(tensor, src_rank=src_rank, process_group=pg)

    return st


def shard_parameter(
    module: torch.nn.Module,
    param_name: str,
    sharding_spec: ShardingSpec,
    src_rank=0,
    process_group=None,
):
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
        raise AttributeError(f"{module._get_name()} has no attribute `{param_name}`")

    tensor = getattr(module, param_name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}"
        )

    if not tensor.is_contiguous():
        raise ValueError(f"param: {param_name} is not a contiguous Tensor")

    st = _shard_tensor(tensor, sharding_spec, src_rank, process_group)

    # Replace param with ShardedTensor.
    module.register_parameter(param_name, nn.Parameter(st))


# Tracks the current process group in the load context manager.
_CURRENT_PROCESS_GROUP: Optional[dist.ProcessGroup] = None


@contextmanager
def load_with_process_group(process_group):
    """
    Context manager to set the process group with which to load a ShardedTensor.
    """
    global _CURRENT_PROCESS_GROUP
    if _CURRENT_PROCESS_GROUP is not None:
        raise RuntimeError(
            'ProcessGroup already set by previous "load_with_process_group" '
            "context manager"
        )
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
    if _CURRENT_PROCESS_GROUP is None:
        return distributed_c10d._get_default_group()
    else:
        return _CURRENT_PROCESS_GROUP


def _reshard_output(
    module: torch.nn.Module, resharding_spec: ShardingSpec
) -> torch.nn.Module:
    """
    Hook a module with output resharding in the forward pass according
    to the given ``resharding_spec``.

    Args:
        module (:class:`torch.nn.Module`): Module whose output needs to be resharded.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
            The specification describing how the output of the module will be resharded.

    Returns:
        A :class:`torch.nn.Module` object with reshard API hooked.
    """

    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor):
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
        module (:class:`torch.nn.Module`): Module whose output is ShardedTensor and the
            local tensor value needs to be returned.

    Returns:
        A :class:`torch.nn.Module` object with collection API hooked.
    """

    def hook_func(_module, _input, output):
        if isinstance(output, ShardedTensor):
            local_tensor = output.local_tensor()
            # Squeeze the # of dimensions manually, only applicable to ChunkShardingSpec
            sharding_spec = output._sharding_spec
            if (
                isinstance(sharding_spec, ChunkShardingSpec)
                and local_tensor.size(sharding_spec.dim) == 1  # type: ignore[attr-defined, arg-type]
            ):
                local_tensor = local_tensor.squeeze(
                    output._sharding_spec.dim  # type: ignore[attr-defined]
                )
            return local_tensor

    module.register_forward_hook(hook_func)
    return module


def shard_module(module: nn.Module, plan: ShardingPlan, src_rank=0, process_group=None):
    """
    Shards a given module according to the provided sharding `plan`. This method
    first shards all the parameters according to the given sharding `plan`. Then if
    `output_plan` and `return_local_tensor` are specified in the sharding `plan`, it
    will tag the output of modules according `output_plan`, convert the module's
    output back to data parallel according to `return_local_tensor`.

    Needs to be called on all ranks in an SPMD fashion.

    Args:
        module (:class:`torch.nn.Module`): The module to apply sharding to
        plan (:class:`torch.distributed._shard.sharding_plan.ShardingPlan`):
            The ShardingPlan which specified param name to ShardingSpec to apply to
            each parameter.

    Keyword args:
         src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the module that would be sharded and scattered across the rest
            of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    """
    # record Sharder paths for sanity check on the plan to ensure items in the plan
    # does not conflict with the submodule tree that the Sharder is working with
    sharder_paths = []
    for name, spec in plan.plan.items():
        if isinstance(spec, Sharder):
            sharder_paths.append(name)

    # shard the parameter according to the ShardingPlan
    for name, spec in plan.plan.items():
        if isinstance(spec, ShardingSpec):
            # if found a sharding spec, try to shard the parameter
            module_path, _, param_name = name.rpartition(".")

            for sharder_path in sharder_paths:
                if module_path.startswith(sharder_path):
                    raise RuntimeError(
                        f"ShardingPlan is in-valid, trying to shard a parameter: {name},"
                        f" but there's already a Sharder entry for module {sharder_path},"
                        f" parameter sharding should not conflict with the submodule tree"
                        f" that a Sharder is working with!"
                    )

            mod = module.get_submodule(module_path)
            shard_parameter(
                mod, param_name, spec, src_rank=src_rank, process_group=process_group
            )
        elif isinstance(spec, Sharder):
            parent_mod_path, _, _mod_name = name.rpartition(".")
            if name == "":
                raise KeyError("Module path must not be empty for custom sharder!")
            mod = module.get_submodule(name)
            parent_mod = module.get_submodule(parent_mod_path)
            sharded_mod = spec.shard(mod)
            # swap this submodule with the sharded module
            parent_mod.mod_name = sharded_mod
        else:
            raise TypeError(
                f"Only `ShardingSpec` and `Sharder` are supported to shard '{name}'"
            )

    # reshard output if there's an entry in `reshard_output` for this module
    if plan.output_plan is not None:
        for module_path, output_spec in plan.output_plan.items():
            if isinstance(output_spec, ShardingSpec):
                mod = module.get_submodule(module_path)
                _reshard_output(mod, output_spec)
            else:
                raise TypeError(
                    f"Only `ShardingSpec` is supported as output_plan for '{module_path}'"
                )
    # convert the output back to data parallel for the modules appears in
    # `return_local_tensor` of the plan, we will call `_collect_local_shard`
    # to collect the local tensor for output of modules
    if plan.return_local_tensor is not None:
        for module_path in plan.return_local_tensor:
            mod = module.get_submodule(module_path)
            _collect_local_shard(mod)
