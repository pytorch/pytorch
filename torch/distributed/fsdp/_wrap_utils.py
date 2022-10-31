import collections
import functools
import warnings
from typing import Any, Callable, Deque, Dict, List, Set, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp._utils import (
    _contains_batchnorm,
    _override_batchnorm_mixed_precision,
)
from torch.distributed.fsdp.wrap import (
    _or_policy,
    _recursive_wrap,
    _wrap_batchnorm_individually,
)


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


def _get_submodule_to_states(
    root_module: nn.Module,
    auto_wrap_policy: Callable,
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
) -> Dict[
    nn.Module, Tuple[List[nn.Parameter], List[torch.Tensor], List[str], List[str]]
]:
    """
    Returns a mapping from submodule to its parameters, buffers, parameter
    names, and buffer names, where each entry logically represents a wrapping
    according to the given auto wrap policy and ignored modules/parameters.
    However, this method does not actually perform any explicit module
    wrapping.

    The mapping values are aligned to module tree boundaries, meaning that they
    are exactly the parameters of the subtree rooted at the corresponding
    submodule key. Sibling submodules cannot be grouped together. Each
    parameter and each buffer in the module tree appears exactly once in the
    returned dict. The returned dict is ordered by increasing tree depth.

    If a parameter is shared among multiple wrapped submodules, then it is
    assigned to the lowest common ancestor (LCA) wrapped module. A mapped-to
    list may be empty, either because the wrapped module truly has no
    parameters or because its parameters were assigned to the LCA.
    """
    # Record the modules to wrap without actually wrapping
    wrapped_modules: List[nn.Module] = []  # these are only logically wrapped
    wrapper_cls = functools.partial(_record_module_wrapper_cls, wrapped_modules)
    _recursive_wrap(
        root_module,
        auto_wrap_policy=auto_wrap_policy,
        wrapper_cls=wrapper_cls,
        ignored_modules=ignored_modules,
        ignored_params=ignored_params,
        only_wrap_children=False,
    )
    # Always include the root module even if not wrapped by the given policy
    if root_module not in wrapped_modules:
        wrapped_modules.append(root_module)

    submodule_to_states = collections.OrderedDict()
    visited_params = set()
    visited_buffers = set()
    # Constructing `wrapped_modules` with `_recursive_wrap()` orders the
    # modules following a post-order traversal. We record parameters in
    # `params_per_wrapped_module` using a reverse post-ordering, which is a
    # topological sort, so that each shared parameter is guaranteed to be
    # grouped with its lowest common ancestor module's parameters.
    wrapped_modules.reverse()
    wrapped_modules_set = set(wrapped_modules)
    for submodule in wrapped_modules:
        # Perform a BFS from `module_to_wrap` and record all untraversed
        # parameters that are not already associated with another module in
        # `wrapped_modules`.
        queue: Deque[Tuple[nn.Module, str]] = collections.deque()
        queue.append((submodule, ""))
        params: List[nn.Parameter] = []
        param_names: List[str] = []
        buffers: List[torch.Tensor] = []
        buffer_names: List[str] = []
        while len(queue) > 0:
            module, prefix = queue.popleft()
            for param_name, param in module.named_parameters(recurse=False):
                if param not in visited_params:
                    params.append(param)
                    visited_params.add(param)
                    param_names.append(prefix + param_name)
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer not in visited_buffers:
                    buffers.append(buffer)
                    visited_buffers.add(buffer)
                    buffer_names.append(prefix + buffer_name)
            for child_module_name, child_module in module.named_children():
                if child_module not in wrapped_modules_set:
                    queue.append((child_module, prefix + child_module_name + "."))
        submodule_to_states[submodule] = (params, buffers, param_names, buffer_names)
    return submodule_to_states


def _record_module_wrapper_cls(
    wrapped_modules: List[nn.Module],
    module: nn.Module,
    **kwargs,
) -> nn.Module:
    """
    This defines a wrapper class to be passed to ``_recursive_wrap()`` that
    records the wrapped module to the input ``wrapped_modules``.
    """
    wrapped_modules.append(module)
    return module
