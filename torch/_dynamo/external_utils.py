# This module contains functions that *will be allowed* by dynamo

import functools
from typing import List

import torch
import torch.utils._pytree as pytree

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


def is_compiling() -> bool:
    """
    Indicates whether we are tracing/compiling with torch.compile() or torch.export().

    If need to check specifically that TorchDynamo is used, then use
    torch.compiler.is_dynamo_compiling().

    TODO(khabinov): we should deprecate this function and use one of these two:
    * torch.compiler.is_compiling(),
    * torch.compiler.is_dynamo_compiling().
    It will depend on the context where to use what.
    """
    return torch.compiler.is_compiling()


def wrap_inline(fn):
    """
    Create an extra frame around fn that is not in skipfiles
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def call_hook(hook, *args):
    """
    Used by compiled autograd to handle hook returning None
    """
    result = hook(*args)
    if result is None:
        return args[0]
    return result


def wrap_numpy(f):
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.
    """
    if not np:
        return f

    @functools.wraps(f)
    def wrap(*args, **kwargs):
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, lambda x: x.numpy(), (args, kwargs)
        )
        out = f(*args, **kwargs)
        return pytree.tree_map_only(np.ndarray, lambda x: torch.as_tensor(x), out)

    return wrap


class FakeBackwardCFunction:
    # duck type
    def __init__(
        self,
        other: torch.autograd.function.BackwardCFunction,
        saved_tensors: List[torch.Tensor],
    ):
        blocklist = {
            "saved_tensors",
            "saved_variables",
            "_raw_saved_tensors",
            "materialize_grads",
        }
        for attr_name in dir(other):
            if attr_name.startswith("__") or attr_name in blocklist:
                continue
            setattr(self, attr_name, getattr(other, attr_name))

        self._forward_cls = other._forward_cls  # type: ignore[attr-defined]
        self.saved_tensors = saved_tensors


def call_backward(backward_c_function, saved_tensors, *args):
    # TODO: speculate instead of always graph breaking
    @torch._dynamo.disable()
    def run_eager():
        fake = FakeBackwardCFunction(backward_c_function, saved_tensors)
        grads = fake._forward_cls.backward(fake, *args)  # type: ignore[attr-defined]

        # in eager, we wrap in a tuple when there's only one grad output
        if type(grads) is not tuple:
            grads = (grads,)

        return grads

    return run_eager()


def untyped_storage_size(x: torch.Tensor):
    return x.untyped_storage().size()


def call_hook_from_backward_state(*args, bw_state, hook_name: str, **kwargs):
    return getattr(bw_state, hook_name)(*args, **kwargs)


def call_module_hooks_from_backward_state(
    _, result, *args, bw_state, hooks_name: str, module_name: str
):
    module = getattr(bw_state, module_name)
    hooks = getattr(bw_state, hooks_name)
    for hook in hooks:
        new_result = hook(module, result, *args)
        if new_result is not None:
            result = new_result
    return result
