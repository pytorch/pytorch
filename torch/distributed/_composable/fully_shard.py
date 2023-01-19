import contextlib
from typing import Callable, Generator, Iterable, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.contract import contract
from torch.distributed._composable_state import _get_module_state, _insert_module_state
from torch.distributed.fsdp._common_utils import _FSDPState

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
) -> nn.Module:
    """
    Applies ``FullyShardedDataParallel` (FSDP) semantics to ``module``.
    """
    # Enforce the new auto wrap policy
    if policy is not None and not isinstance(policy, _FSDPPolicy):
        raise ValueError(f"Expects an `_FSDPPolicy` but got {policy}")
    state = fully_shard.state(module)
    state = _init_ignored_module_states(state, module, ignored_modules)
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
    state = _init_prefetching_state(state, BackwardPrefetch.BACKWARD_PRE, False)
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
            submodule not in state._ignored_modules
            and _get_module_state(submodule) is None
        ):
            _insert_module_state(submodule, state)
    return module


@contextlib.contextmanager
def unshard_params(
    module: nn.Module,
    recurse: bool = True,
    writeback: bool = True,
    rank0_only: bool = False,
    offload_to_cpu: bool = False,
    with_grads: bool = False,
) -> Generator:
    """
    This context manager unshards FSDP-managed parameters.

    Args:
        module (nn.Module): Root module whose module tree to which this
            parameter unsharding logic applies.
        writeback (bool): Whether writes to the parameters persist after
            exiting the context.
        rank0_only (bool): Whether to unshard parameters on rank 0 only or on
            all ranks.
        offload_to_cpu (bool): Whether to offload the unsharded parameters to
            CPU while inside the context.
        with_grads (bool): Whether to additionally unshard gradients along with
            the parameters.
    """
    # TODO (awgu): IMHO, `recurse=False` does not present meaningful semantics.
    # In what case would the user want to only unshard the parameters of the
    # root FSDP modules (where there may be multiple)? To me, the non-recursive
    # case should directly target a module annotated with `fully_shard` or be a
    # no-op otherwise.
    # TODO (awgu): I plan to unify documentation with `summon_full_params()` in
    # a follow-up.
    # TODO (awgu): The current implementation relies on traversal utils that
    # do not traverse through incompatible composable APIs. This may not be the
    # desired behavior for some functions (like this one), in which case we
    # need an option to continue traversing.
    with torch.distributed.fsdp._unshard_param_utils._unshard_params(
        module=module,
        recurse=recurse,
        writeback=writeback,
        rank0_only=rank0_only,
        offload_to_cpu=offload_to_cpu,
        with_grads=with_grads,
    ):
        yield
