import collections
import functools
import warnings
from typing import Any, Callable, Deque, Dict, List, Set

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
