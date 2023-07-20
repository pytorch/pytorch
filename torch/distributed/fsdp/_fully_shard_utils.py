import functools

import torch.nn as nn

from torch.distributed.fsdp._common_utils import _FSDPState

from torch.distributed.fsdp._runtime_utils import (
    _post_forward,
    _post_forward_reshard,
    _pre_forward,
    _pre_forward_unshard,
    _root_pre_forward,
)


def _register_root_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    """
    Registers root pre-forward hook on ``module``, which should be the local
    FSDP root.

    NOTE: For the current composable FSDP design, we have each application of
    ``fully_shard()`` to a module to indicate that that module is the local
    FSDP root. We may remove this assumption in the future, in which case we
    will need to register this root pre-forward hook on any candidate module
    that may be the local FSDP root.
    """
    for forward_handle in state._root_pre_forward_handles:
        forward_handle.remove()
    state._root_pre_forward_handles.clear()
    hook = functools.partial(_root_pre_forward, state)
    state._root_pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


def _register_pre_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    """
    Registers a pre-forward hook on ``module``.
    """
    for forward_handle in state._pre_forward_handles:
        forward_handle.remove()
    state._pre_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(
        _pre_forward, state, module_param_handle, _pre_forward_unshard
    )
    state._pre_forward_handles.append(
        module.register_forward_pre_hook(hook, prepend=True, with_kwargs=True)
    )


def _register_post_forward_hook(
    state: _FSDPState,
    module: nn.Module,
) -> None:
    """
    Registers a post-forward hook on ``module``. Even if the module has no
    handles, we should register the hook since it will register the module's
    pre-backward hook.
    """
    for forward_handle in state._post_forward_handles:
        forward_handle.remove()
    state._post_forward_handles.clear()
    module_param_handle = state._fully_sharded_module_to_handle.get(module, None)
    hook = functools.partial(
        _post_forward,
        state,
        module_param_handle,
        _post_forward_reshard,
    )
    state._post_forward_handles.append(module.register_forward_hook(hook))
