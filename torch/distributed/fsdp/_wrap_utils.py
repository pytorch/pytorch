import collections
import functools
import warnings
from functools import partial
from typing import Any, Deque, Dict, List, NamedTuple, Set, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _is_fsdp_flattened
from torch.distributed.fsdp._utils import _override_module_mixed_precision

from torch.distributed.fsdp.wrap import (
    _FSDPPolicy,
    _or_policy,
    _recursive_wrap,
    _wrap_module_cls_individually,
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
    if mixed_precision is not None:
        for mp_module_to_override in mixed_precision._module_classes_to_ignore:
            # Make modules of this particular type run in fp32 by wrapping them in their own
            # FSDP unit.
            _override_module_mixed_precision(root_module, mp_module_to_override)

        auto_wrap_policy = functools.partial(
            _or_policy,
            policies=[
                auto_wrap_policy,
                partial(
                    _wrap_module_cls_individually,
                    module_classes=mixed_precision._module_classes_to_ignore,
                ),
            ],
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
) -> Dict[nn.Module, FullyShardedModuleState]:
    """
    Returns a mapping from fully sharded module to its parameters, buffers,
    parameter names, and buffer names, where each entry logically represents a
    grouping according to the given auto wrap policy and ignored
    modules/parameters. However, this method does not actually perform any
    module wrapping.

    The mapped-to values are the states from the subtree rooted at the
    corresponding submodule key, excluding child submodules in the mapping and
    ignored state. Sibling submodules cannot be grouped together. The parameter
    and buffer names are prefixed starting from the submodule.

    Each non-ignored parameter and buffer appears exactly once in the returned
    ``dict``, and the ``dict`` is ordered by increasing tree depth. A mapped-to
    parameter list may be empty if the fully sharded module has no parameters
    or if its parameters were assigned to a parent fully sharded module
    instead.
    """
    # Record the modules to wrap without actually wrapping
    wrapped_modules_set: Set[nn.Module] = set()  # these are only logically wrapped
    wrapper_cls = functools.partial(_record_module_wrapper_cls, wrapped_modules_set)
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
    wrapped_modules_set.add(root_module)

    fully_sharded_module_to_states = collections.OrderedDict()
    visited_params = set()
    for ignored_param in ignored_params:
        visited_params.add(ignored_param)
    visited_buffers = set()
    # Construct `wrapped_modules` to follow `.modules()` order to ensure that
    # downstream data structures (`._handles`) match those of the wrapper path.
    # NOTE: Since `.modules()` follows a depth-first order, which is a
    # topological sort, and we iterate over `wrapped_modules` following that
    # order, parent-child shared parameters are assigned to the parent module.
    wrapped_modules: List[nn.Module] = []
    for module in root_module.modules():
        if module in wrapped_modules_set:
            wrapped_modules.append(module)
    for submodule in wrapped_modules:
        # Perform a DFS from `submodule` and record all unvisited state that is
        # not already associated with another module in `wrapped_modules`. We
        # use DFS to follow the `.modules()` order.
        deque: Deque[Tuple[nn.Module, str]] = collections.deque()
        deque.append((submodule, ""))
        params: List[nn.Parameter] = []
        buffers: List[torch.Tensor] = []
        while len(deque) > 0:
            module, prefix = deque.popleft()
            # Reverse `named_children()`, use `appendleft()`, and add to the
            # deque before processing to perform non-recursive DFS
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
        fully_sharded_module_to_states[submodule] = FullyShardedModuleState(
            params, buffers
        )
    return fully_sharded_module_to_states


def _record_module_wrapper_cls(
    wrapped_modules_set: Set[nn.Module],
    module: nn.Module,
    **kwargs,
) -> nn.Module:
    """
    This defines a pseudo-wrapper class to be passed to ``_recursive_wrap()``
    that records the wrapped module to the input ``wrapped_modules_set``
    without actually wrapping with a class.
    """
    wrapped_modules_set.add(module)
    return module
