import collections
import functools
import warnings
from typing import Any, Deque, Dict, List, NamedTuple, Set, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened
from torch.distributed.fsdp._shared_param_utils import get_shared_param_info_to_lca
from torch.distributed.fsdp._utils import (
    _contains_batchnorm,
    _override_batchnorm_mixed_precision,
)
from torch.distributed.fsdp.wrap import (
    _FSDPPolicy,
    _or_policy,
    _recursive_wrap,
    _wrap_batchnorm_individually,
)


class FullyShardedModuleState(NamedTuple):
    """
    Module state for ``_get_fully_sharded_module_to_states()``, representing
    a logical grouping (e.g. parameters to be flattened together).
    """

    params: List[nn.Parameter]
    buffers: List[torch.Tensor]


def _auto_wrap(
    auto_wrap_kwargs: Dict[str, Any],
    fsdp_kwargs: Dict[str, Any],
    module_wrapper_cls: Any,  # e.g. `FullyShardedDataParallel`
) -> None:
    """
    Recursively auto wraps the root module given by the key "module" in
    ``auto_wrap_kwargs`` with the arguments in ``auto_wrap_kwargs`` and
    ``fsdp_kwargs``.

    Precondition: ``auto_wrap_policy`` contains the arguments expected by
    ``_recursive_wrap()``, where ``auto_wrap_policy`` is not ``None``.
    ``fsdp_kwargs`` contains all FSDP arguments except ``module``.
    """
    auto_wrap_policy = auto_wrap_kwargs["auto_wrap_policy"]
    # Support new way to pass an auto wrap policy
    if isinstance(auto_wrap_policy, _FSDPPolicy):
        auto_wrap_policy = auto_wrap_policy.policy
    root_module = auto_wrap_kwargs["module"]
    assert auto_wrap_policy is not None
    # For auto wrapping, submodules should not already be wrapped with FSDP
    # since double wrapping is not supported
    for module_name, module in root_module.named_modules():
        if isinstance(module, module_wrapper_cls):
            raise ValueError(
                f"Expected {module_name} to NOT be FullyShardedDataParallel "
                "if using an `auto_wrap_policy`"
            )
    mixed_precision = fsdp_kwargs["mixed_precision"]
    if mixed_precision is not None and _contains_batchnorm(root_module):
        _override_batchnorm_mixed_precision(root_module)
        auto_wrap_policy = functools.partial(
            _or_policy, policies=[_wrap_batchnorm_individually, auto_wrap_policy]
        )
        warnings.warn(
            "Both mixed precision and an `auto_wrap_policy` were specified "
            "for FSDP, where the wrapped module has batch norm submodules. "
            "The batch norm submodules will be wrapped as separate FSDP "
            "instances with mixed precision disabled since some batch norm "
            "kernels do not support low precision."
        )
    auto_wrap_kwargs["auto_wrap_policy"] = auto_wrap_policy
    _recursive_wrap(**auto_wrap_kwargs, **fsdp_kwargs)


def _get_fully_sharded_module_to_states(
    root_module: nn.Module,
    auto_wrap_policy: _FSDPPolicy,
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
) -> Tuple[Dict[nn.Module, FullyShardedModuleState], Dict[nn.Parameter, nn.Module]]:
    """
    Returns two data structures: (1) is a mapping from fully sharded module to
    its parameters and buffers, where each entry logically represents a
    grouping according to the given auto wrap policy and ignored
    modules/parameters. However, this method does not actually perform any
    module wrapping. (2) is a mapping from shared parameter to its lowest
    common ancestor (LCA) module.

    For (1), the mapped-to values are the states from the subtree rooted at the
    corresponding fully sharded module key, excluding child submodules in the
    mapping and ignored state. The parameter and buffer names are prefixed
    starting from the fully sharded module.

    Each non-ignored parameter and buffer appears exactly once in (1), and for
    simplicity, (1) does not provide any ordering guarantees for its keys. The
    function consuming the dict should ensure to iterate over it in an
    appropraite order. A mapped-to parameter list may be empty if the fully
    sharded module has no parameters or if its parameters were assigned to a
    parent fully sharded module instead.
    """
    # Record the modules to wrap without actually wrapping
    wrapped_modules: List[nn.Module] = []  # these are only logically wrapped
    wrapper_cls = functools.partial(_record_module_wrapper_cls, wrapped_modules)
    if auto_wrap_policy is not None:
        _recursive_wrap(
            root_module,
            auto_wrap_policy=auto_wrap_policy.policy,
            wrapper_cls=wrapper_cls,
            ignored_modules=ignored_modules,
            ignored_params=ignored_params,
            only_wrap_children=False,
        )
    # Always include the root module even if not wrapped by the given policy
    if root_module not in wrapped_modules:
        wrapped_modules.append(root_module)

    fully_sharded_module_to_states = collections.OrderedDict()
    visited_params = set(ignored_params)  # shallow copy
    visited_buffers = set()
    visited_modules = set(ignored_modules)  # shallow copy

    shared_param_info_to_lca = get_shared_param_info_to_lca(root_module, ignored_params)
    shared_params: Set[nn.Parameter] = set()
    lca_module_to_shared_params = collections.defaultdict(list)
    shared_param_to_lca_module: Dict[nn.Parameter, nn.Module] = {}
    for shared_param_info, lca_module in shared_param_info_to_lca.items():
        shared_param = shared_param_info.param
        shared_params.add(shared_param)
        lca_module_to_shared_params[lca_module].append(shared_param)
        shared_param_to_lca_module[shared_param] = lca_module
    visited_params.update(shared_params)  # finished handling shared parameters

    # Constructing `wrapped_modules` with `_recursive_wrap()` follows a
    # post-order traversal (~bottom up). We iterate following this order so
    # that each shared parameter is assigned to the *lowest* module in
    # `wrapped_modules` that is a parent of or equal to the shared parameter's
    # LCA module.
    wrapped_modules_set = set(wrapped_modules)
    for fully_sharded_module in wrapped_modules:
        # Perform a DFS from `fully_sharded_module` and record all unvisited
        # state that is not already associated with another module in
        # `wrapped_modules`. A BFS also works since we do not enforce ordering
        # guarantees in the returned dict.
        deque: Deque[Tuple[nn.Module, str]] = collections.deque()
        deque.append((fully_sharded_module, ""))
        params: List[nn.Parameter] = []
        buffers: List[torch.Tensor] = []
        while len(deque) > 0:
            module, prefix = deque.popleft()
            visited_modules.add(module)
            # Reverse `named_children()`, use `appendleft()`, and add to the
            # deque before processing to run left-to-right non-recursive DFS
            for child_module_name, child_module in reversed(
                list(module.named_children())
            ):
                if child_module not in wrapped_modules_set:
                    deque.appendleft((child_module, prefix + child_module_name + "."))
            for param in module.parameters(recurse=False):
                if param not in visited_params and not _is_fsdp_flattened(param):
                    params.append(param)
                    visited_params.add(param)
            for buffer in module.buffers(recurse=False):
                if buffer not in visited_buffers:
                    buffers.append(buffer)
                    visited_buffers.add(buffer)
            # Assign a shared parameter to this module if the walk visits its
            # LCA module, and remove the entry to be sure to not also assign to
            # another module.
            params.extend(lca_module_to_shared_params.get(module, []))
            lca_module_to_shared_params.pop(module, None)
        fully_sharded_module_to_states[fully_sharded_module] = FullyShardedModuleState(
            params, buffers
        )
    return fully_sharded_module_to_states, shared_param_to_lca_module


def _record_module_wrapper_cls(
    wrapped_modules: List[nn.Module],
    module: nn.Module,
    **kwargs,
) -> nn.Module:
    """
    This defines a pseudo-wrapper class to be passed to ``_recursive_wrap()``
    that records the wrapped module to the input ``wrapped_modules`` without
    actually wrapping with a class.
    """
    wrapped_modules.append(module)
    return module
