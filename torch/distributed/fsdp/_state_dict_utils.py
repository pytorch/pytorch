import contextlib
import logging
import math
import warnings
from typing import Any, Callable, cast, Dict, Generator, Iterator, no_type_check, Tuple

import torch
import torch.distributed as dist

import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper

import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import (
    init_from_local_shards,
    Shard,
    ShardedTensor,
)
from torch.distributed._tensor import DTensor, Replicate

from torch.distributed.distributed_c10d import _get_pg_default_device
from torch.distributed.fsdp._common_utils import (
    _FSDPState,
    _get_module_fsdp_state_if_fully_sharded_module,
    _has_fsdp_params,
    _is_composable,
    _module_handle,
    clean_tensor_name,
    FSDP_PREFIX,
    FSDP_WRAPPED_MODULE,
)
from torch.distributed.fsdp._runtime_utils import (
    _cast_buffers_to_dtype_and_device,
    _get_orig_buffer_dtypes,
    _lazy_init,
    _reset_flat_param_grad_info_if_needed,
)
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.utils import _replace_by_prefix

from ._fsdp_extensions import (
    _ext_chunk_dtensor,
    _ext_chunk_tensor,
    _ext_pre_load_state_dict_transform,
)
from ._unshard_param_utils import _unshard_fsdp_state_params, FLAT_PARAM


logger = logging.getLogger(__name__)


def _convert_to_wrapped_module_name(module_name: str) -> str:
    module_name = module_name.replace(f"{FSDP_PREFIX}", "")
    module_name = module_name.replace(f"{FSDP_WRAPPED_MODULE}", "")
    if module_name:
        module_name = f"{module_name}."
    # `CheckpointWrapper` adds a prefix that has to be removed as well.
    module_name = module_name.replace(checkpoint_wrapper._CHECKPOINT_PREFIX, "")
    return module_name


def _param_name_infos(
    module: nn.Module, fsdp_state: _FSDPState
) -> Iterator[Tuple[str, str, str]]:
    if not _has_fsdp_params(fsdp_state, module):
        return
    for param_name, module_name in _module_handle(
        fsdp_state, module
    ).param_module_names():
        module_name = _convert_to_wrapped_module_name(module_name)
        fqn = f"{module_name}{param_name}"
        yield fqn, param_name, module_name


def _shared_param_name_infos(
    module: nn.Module, fsdp_state
) -> Iterator[Tuple[str, str, str]]:
    for param_name, module_name in _module_handle(
        fsdp_state, module
    ).shared_param_module_names():
        module_name = _convert_to_wrapped_module_name(module_name)
        fqn = f"{module_name}{param_name}"
        yield fqn, param_name, module_name


@no_type_check
def _enter_unshard_params_ctx(
    module: nn.Module,
    fsdp_state: _FSDPState,
    writeback: bool = False,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
) -> None:
    """
    state_dict hooks cannot use the pure context call as the checkpoint flow
    requires to enter the context in the pre-hook but leave the context in the
    post-hook. This API enters the context of ``_unshard_fsdp_state_params``.
    """
    assert module not in fsdp_state._unshard_params_ctx, (
        "Entering the ``_unshard_fsdp_state_params`` context but _unshard_params_ctx[module] "
        "is not None."
    )
    fsdp_state._unshard_params_ctx[module] = _unshard_fsdp_state_params(
        module,
        fsdp_state,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    )
    fsdp_state._unshard_params_ctx[module].__enter__()


@no_type_check
def _exit_unshard_params_ctx(module: nn.Module, fsdp_state: _FSDPState) -> None:
    """A helper function to exit ``_unshard_fsdp_state_params`` context."""
    fsdp_state._unshard_params_ctx[module].__exit__(None, None, None)
    fsdp_state._unshard_params_ctx.pop(module)


def _common_pre_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
) -> None:
    """Performs the pre-state_dict tasks shared by all state_dict types."""
    if fsdp_state._device_handle.is_available():
        fsdp_state._device_handle.synchronize()
    # TODO: need to check if this is always correct for composable FSDP.
    _lazy_init(fsdp_state, module)
    if fsdp_state._is_root:
        _reset_flat_param_grad_info_if_needed(fsdp_state._all_handles)


def _common_unshard_pre_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    offload_to_cpu: bool,
    rank0_only: bool,
) -> None:
    """
    Performs the pre-state_dict tasks shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this hook.
    """
    # For composable `fully_shard`, it does not need to unshard parameters for `NO_SHARD` cases.
    if (
        _is_composable(fsdp_state)
        and fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
    ):
        return
    _enter_unshard_params_ctx(
        module,
        fsdp_state,
        writeback=False,
        offload_to_cpu=offload_to_cpu,
        rank0_only=rank0_only,
    )


@no_type_check
def _common_unshard_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
    param_hook: Callable,
) -> Dict[str, Any]:
    """
    The post-state_dict flow that shared by all state_dict types that require
    ``_unshard_fsdp_state_params()``. FULL_STATE_DICT and SHARDED_STATE_DICT use this
    hook.
    """
    _replace_by_prefix(state_dict, prefix + f"{FSDP_PREFIX}", prefix)
    # Return early for trivial cases
    if not state_dict or not _has_fsdp_params(fsdp_state, module):
        if not (
            _is_composable(fsdp_state)
            and fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
        ):
            _exit_unshard_params_ctx(module, fsdp_state)
        return state_dict

    # If a rank does not have unsharded parameters(when `rank0_only=True`
    # and `rank != 0`), then the rank only needed to participate in the
    # all-gather and does not need to save the # state dict. We simply check
    # rank0_only to ensure this issue.
    rank0_only = (
        fsdp_state._state_dict_type == StateDictType.FULL_STATE_DICT
        and cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only
    )
    # no_fsdp_return means the state_dict returned by this rank should contain
    # only non-FSDP controlled parameters and buffers.
    no_fsdp_return = rank0_only and fsdp_state.rank != 0
    if no_fsdp_return and not fsdp_state._use_orig_params:
        for clean_key in fsdp_state._buffer_names:
            # This is a hack to support activation checkpoint.
            clean_key = clean_key.replace(
                f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
            )
            state_dict.pop(f"{prefix}{clean_key}", None)
        # Non-zero ranks have flat_param key when rank0_only=True, because rank0_only=True is
        # passed in to unshard context, but nonzero ranks reshard early, causing this flat_param
        # to appear in state_dict.
        state_dict.pop(f"{prefix}{FLAT_PARAM}")
        _exit_unshard_params_ctx(module, fsdp_state)
        return state_dict

    # Loop only the parameters saved in this instance's wrapped module to
    # avoid processing buffers.
    for fqn, param_name, module_name in _param_name_infos(module, fsdp_state):
        fqn = f"{prefix}{fqn}"
        if no_fsdp_return:
            state_dict.pop(fqn)
            continue
        assert fqn in state_dict, (
            f"FSDP assumes {fqn} is in the state_dict but the state_dict only "
            f"has {state_dict.keys()}. "
            f"prefix={prefix}, module_name={module_name}, "
            f"param_name={param_name} rank={fsdp_state.rank}."
        )

        param_hook(state_dict, prefix, fqn)

    if not (
        _is_composable(fsdp_state)
        and fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
    ):
        _exit_unshard_params_ctx(module, fsdp_state)

    cpu_device = torch.device("cpu")
    buffer_clean_fqns = []
    buffers = []
    for clean_key in fsdp_state._buffer_names:
        # This is a hack to support activation checkpoint.
        clean_key = clean_tensor_name(clean_key)
        fqn = f"{prefix}{clean_key}"
        if fqn not in state_dict:
            # A buffer can be registered as non-persistent.
            continue
        if no_fsdp_return:
            state_dict.pop(fqn)
        else:
            buffer = state_dict[fqn]
            if (
                fsdp_state._state_dict_config.offload_to_cpu
                and buffer.device != cpu_device
            ):
                state_dict[fqn] = buffer.to(cpu_device)
            # skip upcasting for ignored buffers
            if clean_key not in fsdp_state._ignored_buffer_names:
                buffer_clean_fqns.append(clean_key)
                buffers.append(state_dict[fqn])

    if buffers:
        mixed_precision_enabled_for_buffers = (
            fsdp_state._mixed_precision_enabled_for_buffers()
            if not _is_composable(fsdp_state)
            else (fsdp_state.mixed_precision.buffer_dtype is not None)
        )
        if mixed_precision_enabled_for_buffers:
            buffer_dtypes = _get_orig_buffer_dtypes(fsdp_state, buffer_clean_fqns)
            _cast_buffers_to_dtype_and_device(
                buffers, buffer_dtypes, fsdp_state.compute_device
            )
            for buffer, clean_fqn in zip(buffers, buffer_clean_fqns):
                fqn = f"{prefix}{clean_fqn}"
                logger.info("FSDP is casting the dtype of %s to %s", fqn, buffer.dtype)
                state_dict[fqn] = buffer.clone()
    return state_dict


@no_type_check
def _full_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    Hook that runs before model.state_dict() is called. pre-state_dict hook is
    not actually supported by ``nn.Module``. As a result, this API is called
    from ``_full_post_state_dict_hook()`` to simulate the case. Once pre-state_dict
    is supported in ``nn.Module``, this hook will be registered as a hook in
    ``nn.Module``.
    """
    _common_pre_state_dict_hook(module, fsdp_state)
    _common_unshard_pre_state_dict_hook(
        module,
        fsdp_state,
        offload_to_cpu=fsdp_state._state_dict_config.offload_to_cpu,
        rank0_only=cast(FullStateDictConfig, fsdp_state._state_dict_config).rank0_only,
    )


@no_type_check
def _full_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    Hook that runs after model.state_dict() is called before returning result to
    user. For FSDP, we may have to clone the tensors in state_dict as params go
    back to sharded version after _unshard_fsdp_state_params ends, and also remove
    the ``FSDP_WRAPPED_MODULE`` prefix.
    """

    def param_hook(
        state_dict: Dict[str, Any],
        prefix: str,
        fqn: str,
    ) -> None:
        clean_key = fqn
        clean_prefix = clean_tensor_name(prefix)
        # Strip prefix out of key if needed as buffer names and param names
        # do not have prefix considered as they are not computed in `state_dict`
        # call.
        if clean_key.startswith(clean_prefix):
            clean_key = clean_key[len(clean_prefix) :]

        # Clone parameters before exiting the `_unshard_fsdp_state_params()` context.
        if not getattr(state_dict[fqn], "_has_been_cloned", False):
            try:
                state_dict[fqn] = state_dict[fqn].clone().detach()
                state_dict[fqn]._has_been_cloned = True  # type: ignore[attr-defined]
            except BaseException as e:
                warnings.warn(
                    f"Failed to clone() tensor with name {fqn} on rank {fsdp_state.rank}. "
                    "This may mean that this state_dict entry could point to invalid "
                    "memory regions after returning from state_dict() call if this "
                    "parameter is managed by FSDP. Please check clone "
                    f"implementation of {fqn}. Error: {str(e)}"
                )

    return _common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )


def _full_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    _lazy_init(fsdp_state, module)
    if not (
        _is_composable(fsdp_state)
        and fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
    ):
        _enter_unshard_params_ctx(module, fsdp_state, writeback=True)
    # Add FSDP_PREFIX only for wrapper-based FSDP.
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")


def _full_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    if not (
        _is_composable(fsdp_state)
        and fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
    ):
        _exit_unshard_params_ctx(module, fsdp_state)


def _local_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    Hook that runs before model.state_dict() is called. Right now, pre-state_dict
    hook is not supported by the PyTorch core. So this API is called from
    `_local_post_state_dict_hook()` to simulate the case.
    """
    if (
        _has_fsdp_params(fsdp_state, module)
        and not _module_handle(fsdp_state, module).uses_sharded_strategy
    ):
        raise RuntimeError(
            "``local_state_dict`` can only be used when parameters are flatten "
            "and sharded."
        )
    _common_pre_state_dict_hook(module, fsdp_state)


@no_type_check
def _local_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    This hook create a ShardedTensor from the local flat_param and replace
    the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
    will happen. The underlying storage is the same.
    """

    _replace_by_prefix(state_dict, f"{prefix}{FSDP_PREFIX}", prefix)
    if not _has_fsdp_params(fsdp_state, module):
        return state_dict

    # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
    # value as the flat_param but it is a pure Tensor because
    # nn.Module.state_dict() will detach the parameter. Therefore, we need
    # to get flat_param to get the metadata.
    assert _module_handle(fsdp_state, module), "Should have returned early"
    flat_param = _module_handle(fsdp_state, module).flat_param
    # Constructs a ShardedTensor from the flat_param "without" padding.
    # Removing the padding allows users to change the number of ranks
    # when loading the local_state_dict.
    full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
    shard_offset = flat_param.numel() * fsdp_state.rank
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    if valid_data_size > 0:
        # If FlatParameter is returned, FlatParameter._local_shard cause a
        # pickling issue (can be torch.save but not torch.load). Since there
        # is no benefit for state_dict to return the actual FlatParameter class,
        # a view (which is a tensor) of the FlatParameter will be returned.
        flat_param = flat_param[:valid_data_size].view(valid_data_size)
        local_shards = [
            Shard.from_tensor_and_offsets(flat_param, [shard_offset], fsdp_state.rank)
        ]
    else:
        local_shards = []
    sharded_tensor = init_from_local_shards(
        local_shards, full_numel, process_group=fsdp_state.process_group
    )  # type: ignore[assignment]
    # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
    if fsdp_state._state_dict_config.offload_to_cpu:
        sharded_tensor = sharded_tensor.cpu()
    state_dict[f"{prefix}{FLAT_PARAM}"] = sharded_tensor
    return state_dict


def _local_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    pass


def _local_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
    _lazy_init(fsdp_state, module)
    _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")
    fqn = f"{prefix}{FSDP_PREFIX}{FLAT_PARAM}"
    if fqn not in state_dict:
        assert not _has_fsdp_params(fsdp_state, module), (
            "No `FlatParameter` in `state_dict` for this FSDP instance "
            "but it has parameters"
        )
        return
    load_tensor = state_dict[fqn]
    assert isinstance(
        load_tensor, ShardedTensor
    ), "Tensors in local_state_dict should be ShardedTensor."

    # Convert the ShardedTensor to a Tensor.
    flat_param = _module_handle(fsdp_state, module).flat_param
    assert flat_param is not None
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    shards = load_tensor.local_shards()
    if valid_data_size > 0:
        assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
        load_tensor = shards[0].tensor

        # Get the metadata of the flat_param to decide whether to pad the loaded
        # tensor.
        if flat_param._shard_numel_padded > 0:
            assert load_tensor.numel() < flat_param.numel(), (
                f"Local shard size = {flat_param.numel()} and the tensor in "
                f"the state_dict is {load_tensor.numel()}."
            )
            load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
    else:
        load_tensor = flat_param
    # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
    state_dict[fqn] = load_tensor


def _sharded_pre_state_dict_hook(
    fsdp_state: _FSDPState,
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    Hook that runs before model.state_dict() is called. Check
    ``_full_pre_load_state_dict_hook`` for the detail.
    """
    if (
        _has_fsdp_params(fsdp_state, module)
        and not _module_handle(fsdp_state, module).uses_sharded_strategy
    ):
        raise RuntimeError(
            "``sharded_state_dict`` can only be used when parameters are flatten "
            "and sharded."
        )
    _common_pre_state_dict_hook(module, fsdp_state)
    # Setting offload_to_cpu here does not work even if offload_to_cpu is True.
    # We have to create ShardedTensor first then move it to CPU.
    _common_unshard_pre_state_dict_hook(
        module,
        fsdp_state,
        offload_to_cpu=False,
        rank0_only=False,
    )


@no_type_check
def _sharded_post_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    The hook replaces the unflattened, unsharded parameter in the state_dict
    with a unflattened, sharded parameter (a ShardedTensor).
    """

    def param_hook(state_dict: Dict[str, Any], prefix: str, fqn: str):
        param = state_dict[fqn]
        if not fsdp_state._state_dict_config.use_dtensor:
            sharded_tensor = _ext_chunk_tensor(
                tensor=param,
                rank=fsdp_state.rank,
                world_size=fsdp_state.world_size,
                num_devices_per_node=fsdp_state._device_handle.device_count(),
                pg=fsdp_state.process_group,
            )
        else:
            sharded_tensor = _ext_chunk_dtensor(
                tensor=param,
                rank=fsdp_state.rank,
                device_mesh=fsdp_state._device_mesh,
            )
        if fsdp_state._state_dict_config.offload_to_cpu:
            sharded_tensor = sharded_tensor.cpu()
        state_dict[fqn] = sharded_tensor

    return _common_unshard_post_state_dict_hook(
        module, fsdp_state, state_dict, prefix, param_hook
    )


@no_type_check
def _sharded_post_load_state_dict_hook(
    module: nn.Module, fsdp_state: _FSDPState, *args, **kwargs
) -> None:
    if _has_fsdp_params(fsdp_state, module):
        _exit_unshard_params_ctx(module, fsdp_state)


@no_type_check
def _sharded_pre_load_state_dict_hook(
    module: nn.Module,
    fsdp_state: _FSDPState,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    """
    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
    _lazy_init(fsdp_state, module)
    if not _is_composable(fsdp_state):
        _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP_PREFIX}")
    if not _has_fsdp_params(fsdp_state, module):
        return

    handle = _module_handle(fsdp_state, module)
    if not handle.uses_sharded_strategy:
        raise RuntimeError(
            "load_sharded_state_dict can only be called when parameters "
            "are flattened and sharded."
        )

    device = fsdp_state.compute_device
    for fqn, _, _ in _param_name_infos(module, fsdp_state):
        if not _is_composable(fsdp_state):
            fqn_from_global_root = f"{prefix}{FSDP_PREFIX}{fqn}"
        else:
            fqn_from_global_root = f"{prefix}{fqn}"
        param = state_dict.pop(fqn_from_global_root)

        if not fsdp_state._state_dict_config.use_dtensor:
            # All-gather the param (ShardedTensor)
            param, shards = _ext_pre_load_state_dict_transform(param)

            assert len(shards) < 2, (
                "Expects 0 or 1 shard per rank "
                f"but got {len(shards)} shards on rank {fsdp_state.rank}."
            )
            param_numel = param.size().numel()
            dim_0_size = param.size()[0]
            chunk_size = (
                math.ceil(dim_0_size / fsdp_state.world_size)
                * param_numel
                // dim_0_size
            )
            if len(shards) == 1:
                local_tensor = shards[0].tensor.flatten()
                pg_device = _get_pg_default_device(fsdp_state.process_group)
                if local_tensor.device.type != pg_device.type:
                    local_tensor = local_tensor.to(pg_device)
                num_padding = chunk_size - local_tensor.numel()
                if num_padding > 0:
                    local_tensor = F.pad(local_tensor, [0, num_padding])
            else:
                local_tensor = torch.zeros(chunk_size, dtype=param.dtype, device=device)
            tensor = torch.empty(
                chunk_size * fsdp_state.world_size,
                dtype=local_tensor.dtype,
                device=device,
            )
            if local_tensor.is_cpu:
                tensor_list = list(
                    torch.chunk(tensor, dist.get_world_size(fsdp_state.process_group))
                )
                dist.all_gather(
                    tensor_list, local_tensor, group=fsdp_state.process_group
                )
            else:
                dist.all_gather_into_tensor(
                    tensor, local_tensor, group=fsdp_state.process_group
                )
            tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
            state_dict[fqn_from_global_root] = tensor
        else:
            if param.device != fsdp_state._device_mesh.device_type:
                param = param.to(fsdp_state._device_mesh.device_type)

            param = param.redistribute(
                device_mesh=param.device_mesh, placements=[Replicate()]
            )
            state_dict[fqn_from_global_root] = param.to_local()

    _enter_unshard_params_ctx(module, fsdp_state, writeback=True)


@contextlib.contextmanager
def _replace_with_full_state_dict_type(fsdp_state: _FSDPState) -> Generator:
    old_state_dict_config = fsdp_state._state_dict_config
    old_state_dict_type = fsdp_state._state_dict_type
    fsdp_state._state_dict_config = FullStateDictConfig()
    fsdp_state._state_dict_type = StateDictType.FULL_STATE_DICT
    yield
    fsdp_state._state_dict_config = old_state_dict_config
    fsdp_state._state_dict_type = old_state_dict_type


@no_type_check
@torch.no_grad()
def _post_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> Dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() of this
    FSDP module is executed. ``fsdp_state._state_dict_type`` is used to decide
    what postprocessing will be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        context = contextlib.nullcontext()

    with context:
        _post_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_post_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_post_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_post_state_dict_hook,
        }
        processed_state_dict = _post_state_dict_hook_fn[fsdp_state._state_dict_type](
            module, fsdp_state, state_dict, prefix
        )

    if fsdp_state._is_root:
        logger.info("FSDP finished processing state_dict(), prefix=%s", prefix)
        for key, tensor in sorted(processed_state_dict.items()):
            if key.startswith(prefix) and isinstance(tensor, torch.Tensor):
                local_shape = tensor.shape
                if isinstance(tensor, ShardedTensor):
                    local_shape = None
                    shards = tensor.local_shards()
                    if shards:
                        local_shape = shards[0].tensor.shape
                elif isinstance(tensor, DTensor):
                    local_shape = tensor.to_local().shape
                logger.info(
                    "FQN=%s: type=%s, shape=%s, local_shape=%s, dtype=%s, device=%s",
                    key,
                    type(tensor),
                    tensor.shape,
                    local_shape,
                    tensor.dtype,
                    tensor.device,
                )

    return processed_state_dict


@no_type_check
@torch.no_grad()
def _pre_state_dict_hook(
    module: nn.Module,
    *args,
    **kwargs,
) -> None:
    """
    This is called before the core state dict saving logic of ``module``.
    ``fsdp_state._state_dict_type`` is used to decide what postprocessing will
    be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        context = contextlib.nullcontext()

    with context:
        _pre_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_pre_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_pre_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_pre_state_dict_hook,
        }
        _pre_state_dict_hook_fn[fsdp_state._state_dict_type](
            fsdp_state,
            module,
            *args,
            **kwargs,
        )


@no_type_check
@torch.no_grad()
def _pre_load_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    """
    This is called before ``module._load_from_state_dict()``.
    ``fsdp_state._state_dict_type`` is used to decide what preprocessing will
    be done.
    """
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        context = contextlib.nullcontext()

    with context:
        _pre_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_pre_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_pre_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_pre_load_state_dict_hook,
        }
        # Code that is common for all state_dict impls
        if fsdp_state._device_handle.is_available():
            fsdp_state._device_handle.synchronize()
        # Dispatch into state_dict specific implementation of pre-hook.
        _pre_load_state_dict_hook_fn[fsdp_state._state_dict_type](
            module, fsdp_state, state_dict, prefix
        )


@no_type_check
@torch.no_grad()
def _post_load_state_dict_hook(
    module: nn.Module,
    *args: Any,
) -> None:
    fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(module)
    if fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD:
        context = _replace_with_full_state_dict_type(fsdp_state)
        warnings.warn(
            "When using ``NO_SHARD`` for ``ShardingStrategy``, full_state_dict will"
            "be returned."
        )
    else:
        context = contextlib.nullcontext()

    with context:
        _post_load_state_dict_hook_fn = {
            StateDictType.FULL_STATE_DICT: _full_post_load_state_dict_hook,
            StateDictType.LOCAL_STATE_DICT: _local_post_load_state_dict_hook,
            StateDictType.SHARDED_STATE_DICT: _sharded_post_load_state_dict_hook,
        }
        # Code that is common for all state_dict impls
        # Dispatch into state_dict type specific implementation of post-hook for
        # loading state_dict.
        _post_load_state_dict_hook_fn[fsdp_state._state_dict_type](module, fsdp_state)


def _register_all_state_dict_hooks(state: _FSDPState):
    """
    Registers pre-save, post-save, pre-load, and post-load state dict hooks.
    """
    for hook_registration_fn_str, hook, hook_registration_fn_kwargs in (
        ("register_state_dict_pre_hook", _pre_state_dict_hook, {}),
        ("_register_state_dict_hook", _post_state_dict_hook, {}),
        (
            "_register_load_state_dict_pre_hook",
            _pre_load_state_dict_hook,
            {"with_module": True},
        ),
        ("register_load_state_dict_post_hook", _post_load_state_dict_hook, {}),
    ):
        _register_state_dict_hooks_base(
            state, hook_registration_fn_str, hook, hook_registration_fn_kwargs
        )


@no_type_check
def _register_state_dict_hooks_base(
    state: _FSDPState,
    hook_registration_fn_name: str,
    hook: Callable,
    hook_registration_fn_kwargs: Dict[str, Any],
) -> None:
    """Registers ``hook`` using ``hook_registration_fn``."""
    if not _is_composable(state):
        getattr(state, hook_registration_fn_name)(hook, **hook_registration_fn_kwargs)
    else:
        handle = state._handle
        if handle:
            getattr(handle._fully_sharded_module, hook_registration_fn_name)(
                hook, **hook_registration_fn_kwargs
            )
