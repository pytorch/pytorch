from collections import OrderedDict
from typing import Any, Callable, Dict, List, Set, Tuple, Union

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from torch.nn.utils.rnn import PackedSequence

"""Useful functions to deal with tensor types with other python container types."""

def _contains_batchnorm(module):
    return any(
        isinstance(mod, _BatchNorm) for mod in module.modules()
    )

def _override_batchnorm_mixed_precision(module):
    for mod in module.modules():
        if isinstance(mod, _BatchNorm):
            mod._wrap_overrides = {"mixed_precision": None}  # type: ignore[assignment]

def _apply_to_tensors(
    fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set, OrderedDict, PackedSequence]
) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x: Union[torch.Tensor, Dict, List, Tuple, Set, OrderedDict, PackedSequence]) -> Any:
        if torch.is_tensor(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)

def _apply_to_modules(
    root_module: torch.nn.Module,
    module_fn: Callable,
    return_fn: Callable,
    *args,
    **kwargs,
):
    """
    Performs a pre-order traversal of the modules in the hierarchy rooted at
    ``root_module``, applying ``module_fn`` at each module and finally
    returning a value using ``return_fn``. The traversal constructs the full
    module prefix name (e.g. "module.submodule." just like in model state dict)
    and makes that available to ``module_fn``.
    """
    def f(module: torch.nn.Module, prefix: str, *args, **kwargs):
        # Call the module function before recursing over children (pre-order)
        module_fn(module, prefix, *args, **kwargs)
        for submodule_name, submodule in module.named_children():
            if submodule is not None:
                new_prefix = prefix + submodule_name + "."
                f(submodule, new_prefix, *args, **kwargs)

    f(root_module, "", *args, **kwargs)
    return return_fn(*args, **kwargs)
