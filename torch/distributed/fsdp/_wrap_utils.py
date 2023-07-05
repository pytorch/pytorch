import collections
import functools
import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Set, Tuple, Type, Union

import torch.nn as nn
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.fsdp._utils import _override_module_mixed_precision

from torch.distributed.fsdp.wrap import (
    _construct_wrap_fn,
    _FSDPPolicy,
    _or_policy,
    _post_order_apply,
    _recursive_wrap,
    _run_mixed_precision_override_policy,
    _run_module_wrap_policy,
    _wrap_module_cls_individually,
    ModuleWrapPolicy,
)


def _auto_wrap(
    root_module: nn.Module,
    policy: Union[Callable, _FSDPPolicy],
    ignored_modules: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    fsdp_kwargs: Dict[str, Any],
    fsdp_fn: Callable,  # `FullyShardedDataParallel` or `fully_shard`
):
    """
    Auto wraps modules in ``root_module`` 's tree according to ``policy``
    following a post-order traversal.

    Precondition: ``fsdp_kwargs`` should contain all FSDP arguments except
    ``module``. This function accepts the kwargs dict directly since it gets
    forwarded into the post-order traversal function.
    """
    mixed_precision = fsdp_kwargs["mixed_precision"]
    is_wrapper = inspect.isclass(fsdp_fn)
    # TODO: We may relax this no-nested-wrapping constraint to support manual
    # wrapping followed by auto wrapping.
    _check_nested_wrapping(root_module)

    # TODO: Start migration to refactored auto wrapping with `ModuleWrapPolicy`
    if isinstance(policy, ModuleWrapPolicy):
        module_classes = policy._module_classes
        fsdp_kwargs["auto_wrap_policy" if is_wrapper else "policy"] = None
        target_module_to_kwargs = _run_module_wrap_policy(
            root_module, module_classes, ignored_modules, fsdp_kwargs
        )
        if mixed_precision is not None:
            target_module_to_kwargs = _run_mixed_precision_override_policy(
                root_module,
                mixed_precision._module_classes_to_ignore,
                ignored_modules,
                fsdp_kwargs,
                target_module_to_kwargs,
            )
            overridden_module_classes = _override_module_mixed_precision(
                root_module, mixed_precision._module_classes_to_ignore
            )
            _warn_on_overridden_mixed_precision(overridden_module_classes)
        wrap_fn = _construct_wrap_fn(root_module, target_module_to_kwargs, fsdp_fn)
        _post_order_apply(root_module, wrap_fn)
        return

    # Support new way to pass an auto wrap policy
    if isinstance(policy, _FSDPPolicy):
        policy = policy.policy
    assert policy is not None
    recursive_wrap_kwargs = {
        "module": root_module,
        "auto_wrap_policy": policy,
        "wrapper_cls": fsdp_fn,
        "ignored_modules": ignored_modules,
        "ignored_params": ignored_params,
        "only_wrap_children": True,
    }
    if mixed_precision is not None:
        # Wrap modules of the ignored types separately and register forward
        # hooks to cast to fp32 and back to the original dtype, respectively
        overridden_module_classes = _override_module_mixed_precision(
            root_module, mixed_precision._module_classes_to_ignore
        )
        policy = functools.partial(
            _or_policy,
            policies=[
                policy,
                partial(
                    _wrap_module_cls_individually,
                    module_classes=mixed_precision._module_classes_to_ignore,
                ),
            ],
        )
        recursive_wrap_kwargs["auto_wrap_policy"] = policy
        _warn_on_overridden_mixed_precision(overridden_module_classes)
    _recursive_wrap(**recursive_wrap_kwargs, **fsdp_kwargs)


def _check_nested_wrapping(root_module: nn.Module):
    for module_name, module in root_module.named_modules():
        if _get_module_fsdp_state(module) is not None:
            raise ValueError(
                "FSDP auto wrapping requires modules to not already have "
                f"FSDP applied but found {module_name} in\n{root_module}"
            )


def _warn_on_overridden_mixed_precision(
    overridden_module_classes: Set[Type[nn.Module]],
):
    if len(overridden_module_classes) == 0:
        return
    warnings.warn(
        "Both mixed precision and an auto_wrap_policy were specified to FSDP, "
        f"where the wrapped module has submodules of type:\n{overridden_module_classes}\n"
        "These modules will be wrapped as separate FSDP instacnes with mixed "
        "precision disabled."
    )


def _validate_frozen_params(
    root_module: nn.Module,
    modules_to_wrap: Set[nn.Module],
    ignored_params: Set[nn.Parameter],
    strict: bool,
):
    """
    This checks that, given ``modules_to_wrap``, each module would manage
    parameters that are uniformly frozen or non-frozen. This uniformity
    requirement is strict for ``use_orig_params=False`` (hard error) and highly
    recommended for ``use_orig_params=True`` (user warning).
    """
    visited_modules = {root_module}
    stack = [("", root_module)]
    topo_sorted_named_modules: List[Tuple[str, nn.Module]] = []
    while stack:
        module_name, module = stack.pop()
        topo_sorted_named_modules.append((module_name, module))
        for child_module_name, child_module in module.named_children():
            if child_module is None:  # only for overrides of `named_children()`
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                if module_name != "":
                    child_module_name = module_name + "." + child_module_name
                stack.append((child_module_name, child_module))
    reverse_topo_sorted_modules = reversed(topo_sorted_named_modules)
    visited_modules.clear()
    for module_name, module in reverse_topo_sorted_modules:
        if module in modules_to_wrap:
            param_to_fqn = _get_param_to_fqn(
                module, ignored_params, visited_modules, module_name
            )
            frozen_param_fqns: List[str] = []
            nonfrozen_param_fqns: List[str] = []
            for param, fqn in param_to_fqn.items():
                if param.requires_grad:
                    nonfrozen_param_fqns.append(fqn)
                else:
                    frozen_param_fqns.append(fqn)
            if len(frozen_param_fqns) > 0 and len(nonfrozen_param_fqns) > 0:
                msg = f"{module_name} has both parameters with requires_grad=True and False."
                if strict:
                    msg += " FSDP does not support wrapping such modules.\n"
                else:
                    msg += (
                        " FSDP does not recommend wrapping such modules since "
                        "the gradient memory usage will be higher than expected.\n"
                    )
                msg += (
                    f"The following parameters have requires_grad=True:\n{nonfrozen_param_fqns}\n"
                    f"The following parameters have requires_grad=False:\n{frozen_param_fqns}"
                )
                if strict:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)


def _get_param_to_fqn(
    root_module: nn.Module,
    ignored_params: Set[nn.Parameter],
    visited_modules: Set[nn.Module],
    root_prefix: str,
) -> Dict[nn.Parameter, str]:
    """
    NOTE: This function differs from the ``_get_param_to_fqn()`` used for
    ``rekey_optim_state_dict()``. Here, we rely on the keys in the dict
    being exactly the parameters that would be managed by ``root_module`` given
    ``visited_modules``, which depends on the target modules to wrap.
    """
    param_to_fqn: Dict[nn.Parameter, str] = {}
    # Run BFS (or any tree traversal works)
    queue = collections.deque([(root_module, root_prefix)])
    visited_modules.add(root_module)
    while queue:
        module, prefix = queue.popleft()
        for param_name, param in module.named_parameters(recurse=False):
            if param not in ignored_params:
                fqn = param_name if prefix == "str" else prefix + "." + param_name
                param_to_fqn[param] = fqn
        for child_module_name, child_module in module.named_children():
            if child_module is None:  # only for overrides of `named_children()`
                continue
            if child_module not in visited_modules:
                visited_modules.add(child_module)
                child_prefix = (
                    child_module_name
                    if prefix == "str"
                    else prefix + "." + child_module_name
                )
                queue.append((child_module, child_prefix))
    return param_to_fqn
