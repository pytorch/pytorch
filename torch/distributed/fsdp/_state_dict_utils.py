import functools
import math
import warnings
from typing import Any, cast, Dict

import torch
import torch.distributed as dist
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as checkpoint_wrapper
# Import the entire FSDP file to avoid circular imports
import torch.distributed.fsdp.fully_sharded_data_parallel as FSDP
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensor,
    init_from_local_shards,
)
from torch.distributed.utils import (
    _replace_by_prefix,
)

from ._fsdp_extensions import _ext_chunk_tensor, _ext_pre_load_state_dict_transform
from .flat_param import (
    FlatParamHandle,
)
from .flatten_params_wrapper import (
    FLAT_PARAM,
)


def _full_post_state_dict_hook(
    self,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    Hook that runs after model.state_dict() is called before returning result to
    user. For FSDP, we may have to clone the tensors in state_dict as params go
    back to sharded version after _summon_full_params ends, and also remove
    "_fsdp_wrapped_module" prefix.
    """
    _replace_by_prefix(state_dict, prefix + f"{FSDP.FSDP_WRAPPED_MODULE}.", prefix)
    self._assert_state([FSDP.TrainingState_.SUMMON_FULL_PARAMS])
    # Return early for trivial cases
    if not state_dict or not self._fsdp_wrapped_module.has_params:
        return state_dict

    # If a rank has already exited the `summon_full_params()` context here
    # (e.g. when `rank0_only=True` and `rank != 0`), then the rank only
    # needed to participate in the all-gather and does not need to save the
    # state dict. For `use_orig_params=False`, we can check this via
    # `FlatParameter` registration.
    # TODO: For `use_orig_params=True`, we check for the reshard upon
    # exiting `summon_full_params()` via the parameter shape. However, for
    # `NO_SHARD`, we cannot tell from the shape, so we do not return early.
    if (
        (
            not self._use_orig_params
            and "flat_param" in self._fsdp_wrapped_module._parameters
        )
        or (
            self._use_orig_params
            and self._handles
            and self._handles[0].uses_sharded_strategy
            and self._handles[0].is_sharded(self._handles[0].flat_param)
        )
    ):
        return state_dict

    offload_to_cpu = self._state_dict_config.offload_to_cpu
    cpu_device = torch.device("cpu")

    # Loop only the parameters saved in self._fsdp_wrapped_module to avoid
    # processing buffers.
    for fqn, param_name, module_name in self._param_fqns:
        fqn = f"{prefix}{fqn}"
        clean_key = fqn
        clean_prefix = FSDP.clean_tensor_name(prefix)
        # Strip prefix out of key if needed as buffer names and param names
        # do not have prefix considered as they are not computed in `state_dict`
        # call.
        if clean_key.startswith(clean_prefix):
            clean_key = clean_key[len(clean_prefix):]

        # Clone non-ignored parameters before exiting the
        # `_summon_full_params()` context
        assert fqn in state_dict, (
            f"FSDP assumes {fqn} is in the state_dict but the state_dict "
            f"only has {state_dict.keys()}. prefix={prefix}, "
            f"module_name={module_name} param_name={param_name} rank={self.rank}."
        )
        if clean_key not in self._ignored_param_names and \
                not getattr(state_dict[fqn], "_has_been_cloned", False):
            try:
                state_dict[fqn] = state_dict[fqn].clone().detach()
                state_dict[fqn]._has_been_cloned = True  # type: ignore[attr-defined]
            except BaseException as e:
                warnings.warn(
                    f"Failed to clone() tensor with name {fqn} on rank {self.rank}. "
                    "This may mean that this state_dict entry could point to invalid "
                    "memory regions after returning from state_dict() call if this "
                    "parameter is managed by FSDP. Please check clone "
                    f"implementation of {fqn}. Error: {str(e)}"
                )

    # Offload the buffer to CPU if needed -- we do not do this in
    # `_summon_full_params()` since without care, that would free
    # the original buffer's GPU memory and require reallocating
    # that memory later; this only affects the state dict's buffer
    # variable and leaves the original buffer's GPU memory intact
    if offload_to_cpu:
        for clean_key in self._buffer_names:
            # This is a hack to support activation checkpoint.
            clean_key = clean_key.replace(
                f"{checkpoint_wrapper._CHECKPOINT_PREFIX}.", ""
            )
            fqn = f"{prefix}{clean_key}"
            if fqn not in state_dict:
                # A buffer can be registered as non-persistent.
                continue
            if state_dict[fqn].device != cpu_device:
                state_dict[fqn] = state_dict[fqn].to(cpu_device)
    return state_dict


def _full_pre_load_state_dict_hook(
    self,
    state_dict: Dict[str, Any],
    prefix: str,
) -> None:
    # We do not expect to be calling pre-hooks twice without post-hook
    # call in between.
    assert getattr(self, '_full_param_ctx', None) is None
    # Note that it needs writeback=True to persist.
    self._full_param_ctx = self._summon_full_params(
        recurse=False, writeback=True
    )
    self._full_param_ctx.__enter__()
    _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP.FSDP_WRAPPED_MODULE}.")


def _full_post_load_state_dict_hook(self, *args, **kwargs) -> None:
    # We should exit summon_full_params context.
    self._assert_state([FSDP.TrainingState_.SUMMON_FULL_PARAMS])
    assert getattr(self, '_full_param_ctx', None) is not None
    self._full_param_ctx.__exit__(None, None, None)
    self._full_param_ctx = None


def _local_post_state_dict_hook(
    self,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    This hook create a ShardedTensor from the local flat_param and replace
    the state_dict[f"{prefix}{FLAT_PARAM}] with the ShardedTensor. No copy
    will happen. The underlying storage is the same.
    """
    _replace_by_prefix(state_dict, f"{prefix}{FSDP.FSDP_WRAPPED_MODULE}.", prefix)
    if not self._fsdp_wrapped_module.has_params:
        return state_dict

    # state_dict[f"{prefix}{FLAT_PARAM}"] exists and has the same tensor
    # value as the flat_param but it is a pure Tensor because
    # nn.Module.state_dict() will detach the parameter. Therefore, we need
    # to get flat_param from the FlattenParamsWrapper to get the metadata.
    flat_param = getattr(self._fsdp_wrapped_module, FLAT_PARAM, None)
    assert flat_param is not None
    # Construct a ShardedTensor from the flat_param.
    full_numel = flat_param._unpadded_unsharded_size.numel()  # type: ignore[attr-defined]
    shard_offset = flat_param.numel() * self.rank
    valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
    if valid_data_size > 0 and flat_param._shard_numel_padded > 0:
        flat_param = flat_param.narrow(0, 0, valid_data_size)
    local_shards = [
        Shard.from_tensor_and_offsets(flat_param, [shard_offset], self.rank)
    ]
    sharded_tensor = init_from_local_shards(
        local_shards, full_numel, process_group=self.process_group
    )  # type: ignore[assignment]
    if self._state_dict_config.offload_to_cpu:
        sharded_tensor = sharded_tensor.cpu()
    state_dict[f"{prefix}{FLAT_PARAM}"] = sharded_tensor
    return state_dict


def _local_post_load_state_dict_hook(self, *args, **kwargs) -> None:
    pass


def _local_pre_load_state_dict_hook(
    self, state_dict: Dict[str, Any], prefix: str,
) -> None:
    """
    This hook finds the local flat_param for this FSDP module from the
    state_dict. The flat_param should be a ShardedTensor. This hook converts
    the ShardedTensor to a tensor. No copy happen unless padding is required.
    """
    _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP.FSDP_WRAPPED_MODULE}.")
    fqn = f"{prefix}{FSDP.FSDP_WRAPPED_MODULE}.{FLAT_PARAM}"
    if fqn not in state_dict:
        assert getattr(self._fsdp_wrapped_module, FLAT_PARAM, None) is None, (
            "No flat parameter in state_dict but "
            "self._fsdp_wrapped_module.flat_param is not None."
        )
        return
    load_tensor = state_dict[fqn]
    assert isinstance(
        load_tensor, ShardedTensor
    ), "Tensors in local_state_dict should be ShardedTensor."

    # Convert the ShardedTensor to a Tensor.
    shards = load_tensor.local_shards()
    assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
    load_tensor = cast(torch.Tensor, shards[0].tensor)

    # Get the metada of the flat_param to decide whether to pad the loaded
    # tensor.
    flat_param = self._fsdp_wrapped_module.flat_param
    assert flat_param is not None
    if flat_param._shard_numel_padded not in (0, flat_param.numel()):
        assert load_tensor.numel() < flat_param.numel(), (
            f"Local shard size = {flat_param.numel()} and the tensor in "
            f"the state_dict is {load_tensor.numel()}."
        )
        load_tensor = F.pad(load_tensor, [0, flat_param._shard_numel_padded])
    state_dict[fqn] = load_tensor


def _sharded_post_state_dict_hook(
    self,
    state_dict: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    The hook replaces the unflattened, unsharded parameter in the state_dict
    with a unflattened, sharded parameter (a ShardedTensor).
    """
    _replace_by_prefix(state_dict, f"{prefix}{FSDP.FSDP_WRAPPED_MODULE}.", prefix)
    if not self._fsdp_wrapped_module.has_params:
        return state_dict

    assert self.training_state != FSDP.TrainingState_.SUMMON_FULL_PARAMS, (
        "Inside _sharded_post_state_dict_hook, the training_state must "
        "not be SUMMON_FULL_PARAMS."
    )
    with self._summon_full_params(recurse=False, writeback=False):
        for fqn, _, _ in self._param_fqns:
            # Create a ShardedTensor for the unflattened, non-sharded parameter.
            param = functools.reduce(getattr, fqn.split("."), self.module)
            sharded_tensor = _ext_chunk_tensor(
                tensor=param,
                rank=self.rank,
                world_size=self.world_size,
                num_devices_per_node=torch.cuda.device_count(),
                pg=self.process_group
            )
            if self._state_dict_config.offload_to_cpu:
                sharded_tensor = sharded_tensor.cpu()
            state_dict[f"{prefix}{fqn}"] = sharded_tensor
    # For `use_orig_params=True`, the `FlatParameter` is not registered, so
    # there is no entry in the state dict for it to pop.
    if not self._use_orig_params:
        state_dict.pop(f"{prefix}{FLAT_PARAM}")
    return state_dict


def _sharded_post_load_state_dict_hook(self, *args, **kwargs) -> None:
    if self._use_orig_params:
        self._register_orig_params()


def _sharded_pre_load_state_dict_hook(
    self, state_dict: Dict[str, Any], prefix: str,
) -> None:
    """
    The hook combines the unflattened, sharded parameters (ShardedTensor) to
    a new FlatParameter and shards the new FlatParameter to the local chunk.
    """
    _replace_by_prefix(state_dict, prefix, prefix + f"{FSDP.FSDP_WRAPPED_MODULE}.")
    if not self._fsdp_wrapped_module.has_params:
        return

    if not self._fsdp_wrapped_module.handle.uses_sharded_strategy:
        raise RuntimeError(
            "load_sharded_state_dict can only be called when parameters "
            "are flatten and sharded."
        )

    nonsharded_tensors = []
    shared_fqns = [fqn for fqn, _, _ in self._shared_param_fqns]
    for fqn, _, _ in self._param_fqns:
        full_fqn = f"{prefix}{FSDP.FSDP_WRAPPED_MODULE}.{fqn}"
        param = state_dict.pop(full_fqn)
        if fqn in shared_fqns:
            continue
        # All-gather the param (ShardedTensor)
        param, shards = _ext_pre_load_state_dict_transform(param)
        assert len(shards) < 2, (
            "Expects 0 or 1 shard per rank "
            f"but got {len(shards)} shards on rank {self.rank}."
        )
        param_numel = param.size().numel()
        dim_0_size = param.size()[0]
        chunk_size = (
            math.ceil(dim_0_size / self.world_size) * param_numel // dim_0_size
        )
        if len(shards) == 1:
            local_tensor = cast(torch.Tensor, shards[0].tensor).flatten()
            if not local_tensor.is_cuda:
                local_tensor = local_tensor.cuda()
            num_padding = chunk_size - local_tensor.numel()
            if num_padding > 0:
                local_tensor = F.pad(local_tensor, [0, num_padding])
        else:
            local_tensor = torch.zeros(chunk_size, dtype=param.dtype).cuda()
        tensor = torch.empty(
            chunk_size * self.world_size, dtype=local_tensor.dtype
        ).cuda()
        dist.all_gather_into_tensor(tensor, local_tensor, group=self.process_group)
        tensor = tensor.narrow(0, 0, param_numel).reshape(param.size())
        nonsharded_tensors.append(tensor)

    # Create a new flat_param from the loaded, non-sharded tensors.
    flat_param = self._fsdp_wrapped_module.flat_param
    loaded_flat_param = FlatParamHandle.flatten_params(
        nonsharded_tensors, requires_grad=False
    )

    # Get the chunk from the loaded flat_param for the local rank.
    loaded_flat_param, num_to_pad = FlatParamHandle._get_shard(
        loaded_flat_param, self.rank, self.world_size,
    )
    loaded_flat_param.to(flat_param.device)
    assert flat_param.numel() == loaded_flat_param.numel(), (
        f"The loaded local chunk has different numel({loaded_flat_param.numel()}) "
        f"from the local chunk {flat_param.numel()}."
    )
    assert flat_param._shard_numel_padded == num_to_pad, (
        f"The loaded local chunk has different padding({num_to_pad}) "
        f"from the local chunk {flat_param._shard_numel_padded}."
    )
    state_dict[f"{prefix}_fsdp_wrapped_module.flat_param"] = loaded_flat_param
    if self._use_orig_params:
        self._deregister_orig_params()


@torch.no_grad()
def _post_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> Dict[str, Any]:
    """
    _post_state_dict_hook() is called after the state_dict() of this
    FSDP module is executed. ``self._state_dict_type`` is used to decide
    what postprocessing will be done.
    """
    _post_state_dict_hook_fn = {
        FSDP.StateDictType.FULL_STATE_DICT: _full_post_state_dict_hook,
        FSDP.StateDictType.LOCAL_STATE_DICT: _local_post_state_dict_hook,
        FSDP.StateDictType.SHARDED_STATE_DICT: _sharded_post_state_dict_hook,
    }
    self = cast(FSDP.FullyShardedDataParallel, module)
    processed_state_dict = _post_state_dict_hook_fn[self._state_dict_type](
        self, state_dict, prefix
    )
    # Restore buffers, which currently are in their full precision type,
    # back to their mixed precision type. This is because buffers are cast
    # during lazy_init() and stay at their mixed precision type before/after
    # forward/backward. As a result state_dict() should maintain this.
    if (
        self._is_root
        and self._mixed_precision_enabled_for_buffers()
    ):
        self._cast_buffers(recurse=True)
    return processed_state_dict


@torch.no_grad()
def _pre_load_state_dict_hook(
    module: nn.Module,
    state_dict: Dict[str, Any],
    prefix: str,
    *args: Any,
) -> None:
    """
    ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()``
    is called. ``self._state_dict_type`` is used to decide what preprocessing
    will be done.
    """
    _pre_load_state_dict_hook_fn = {
        FSDP.StateDictType.FULL_STATE_DICT: _full_pre_load_state_dict_hook,
        FSDP.StateDictType.LOCAL_STATE_DICT: _local_pre_load_state_dict_hook,
        FSDP.StateDictType.SHARDED_STATE_DICT: _sharded_pre_load_state_dict_hook,
    }
    # Code that is common for all state_dict impls
    self = cast(FSDP.FullyShardedDataParallel, module)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Dispatch into state_dict specific implementation of pre-hook.
    _pre_load_state_dict_hook_fn[self._state_dict_type](self, state_dict, prefix)


@torch.no_grad()
def _post_load_state_dict_hook(module: nn.Module, *args: Any) -> None:
    _post_load_state_dict_hook_fn = {
        FSDP.StateDictType.FULL_STATE_DICT: _full_post_load_state_dict_hook,
        FSDP.StateDictType.LOCAL_STATE_DICT: _local_post_load_state_dict_hook,
        FSDP.StateDictType.SHARDED_STATE_DICT: _sharded_post_load_state_dict_hook,
    }
    # Code that is common for all state_dict impls
    self = cast(FSDP.FullyShardedDataParallel, module)
    # Dispatch into state_dict type specific implementation of post-hook for
    # loading state_dict.
    _post_load_state_dict_hook_fn[self._state_dict_type](self)
