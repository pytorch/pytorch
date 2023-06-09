from typing import Callable, Iterable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import contract
from torch.distributed._composable_state import _get_module_state, _insert_module_state
from torch.distributed.fsdp._common_utils import _FSDPState

from torch.distributed.fsdp._init_utils import (
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_ignored_module_states,
    _init_param_handles_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
)
from torch.distributed.fsdp._runtime_utils import (
    _register_post_forward_hooks,
    _register_pre_forward_hooks,
    _register_root_pre_forward_hook,
)
from torch.distributed.fsdp._state_dict_utils import _register_all_state_dict_hooks
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import _FSDPPolicy


@contract(state_cls=_FSDPState)
def fully_shard(
    module: nn.Module,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    policy: Optional[_FSDPPolicy] = None,
    strategy: Optional[ShardingStrategy] = None,
    mixed_precision: Optional[MixedPrecision] = None,
    cpu_offload: Optional[CPUOffload] = None,
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
    device_id: Optional[Union[int, torch.device]] = None,
    param_init_fn: Optional[Callable[[nn.Module], None]] = None,
    sync_module_states: bool = False,
    forward_prefetch: bool = False,
    ignored_states: Union[
        Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
    ] = None,
) -> nn.Module:
    """
    Applies ``FullyShardedDataParallel` (FSDP) semantics to ``module``.
    """
    torch._C._log_api_usage_once("torch.distributed.fully_shard")
    # Enforce the new auto wrap policy
    if policy is not None and not isinstance(policy, _FSDPPolicy):
        raise ValueError(f"Expects an `_FSDPPolicy` but got {policy}")
    state = fully_shard.state(module)
    state = _init_ignored_module_states(state, module, ignored_modules, ignored_states)
    state = _init_device_handle(state, module, state._ignored_params, device_id)
    state = _init_process_group_state(
        state, process_group, ShardingStrategy.FULL_SHARD, policy
    )
    limit_all_gathers = True
    use_orig_params = True
    backward_prefetch_limit = 1
    forward_prefetch_limit = 1
    state = _init_core_state(
        state,
        strategy or ShardingStrategy.FULL_SHARD,
        mixed_precision,
        cpu_offload,
        limit_all_gathers,
        use_orig_params,
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    state = _init_runtime_state(state)
    state = _init_prefetching_state(
        state, BackwardPrefetch.BACKWARD_PRE, forward_prefetch=forward_prefetch
    )
    state = _init_buffer_state(state, module)
    state = _init_param_handles_from_module(
        state,
        module,
        policy,
        device_id,
        param_init_fn,
        sync_module_states,
    )
    state = _init_state_dict_state(state)
    _register_all_state_dict_hooks(state)
    modules = list(module.modules())
    _register_pre_forward_hooks(state, modules)
    _register_post_forward_hooks(state, modules)
    _register_root_pre_forward_hook(state, module)  # prepend last
    for submodule in module.modules():
        if (
            submodule in state._fully_sharded_module_to_handles
            and _get_module_state(submodule) is None
        ):
            _insert_module_state(submodule, state)
    return module
