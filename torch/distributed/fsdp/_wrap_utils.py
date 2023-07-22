import functools
import inspect
import warnings
from functools import partial
from typing import Any, Callable, Dict, Set, Type, Union

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
    _recursive_wrap(**recursive_wrap_kwargs, **fsdp_kwargs)  # type: ignore[arg-type]


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
