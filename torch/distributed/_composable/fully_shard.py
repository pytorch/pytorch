from typing import Callable, cast, Iterable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, ComposableFSDPState
from torch.distributed.fsdp._init_utils import (
    _init_buffer_state,
    _init_core_state,
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
)
from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)


def fully_shard(
    module: nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
    sharding_strategy: Optional[ShardingStrategy] = None,
    mixed_precision: Optional[MixedPrecision] = None,
    cpu_offload: Optional[CPUOffload] = None,
    auto_wrap_policy: Optional[Callable] = None,
    ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
    device_id: Optional[Union[int, torch.device]] = None,
    param_init_fn: Optional[Callable[[nn.Module], None]] = None,
    sync_module_states: bool = False,
) -> ComposableFSDPState:
    """
    Applies ``FullyShardedDataParallel` (FSDP) semantics to ``module``.
    """
    state = cast(_FSDPState, ComposableFSDPState())
    state = _init_ignored_module_states(state, module, ignored_modules)
    state = _init_process_group_state(state, process_group)
    limit_all_gathers = True
    use_orig_params = True
    backward_prefetch_limit = 1
    forward_prefetch_limit = 1
    state = _init_core_state(
        state,
        sharding_strategy,
        mixed_precision,
        cpu_offload,
        limit_all_gathers,
        use_orig_params,
        backward_prefetch_limit,
        forward_prefetch_limit,
    )
    state = _init_runtime_state(state)
    state = _init_prefetching_state(state, BackwardPrefetch.BACKWARD_PRE, False)
    state = _init_buffer_state(state, module)
    state = _init_param_handles_from_module(
        state,
        module,
        auto_wrap_policy,
        device_id,
        param_init_fn,
        sync_module_states,
    )
    state = _init_state_dict_state(state)
    modules = list(module.modules())
    _register_pre_forward_hooks(state, modules)
    _register_post_forward_hooks(state, modules)
    return cast(ComposableFSDPState, state)
