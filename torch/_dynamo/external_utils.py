# This module contains functions that *will be allowed* by dynamo

import functools
import warnings
from typing import Any, Callable, List, Optional, Union

import torch
import torch.utils._pytree as pytree


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


def is_compiling() -> bool:
    """
    Indicates whether we are tracing/compiling with torch.compile() or torch.export().
    """
    return torch.compiler.is_compiling()


def wrap_inline(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Create an extra frame around fn that is not in skipfiles.
    """

    @functools.wraps(fn)
    def inner(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return inner


def call_hook(
    hook: Callable[..., Optional[torch.Tensor]], *args: Any, **kwargs: Any
) -> torch.Tensor:
    """
    Used by compiled autograd to handle hook returning None.
    """
    result = hook(*args)
    if result is None:
        return args[0]
    elif kwargs.get("hook_type") == "post_acc_grad_hook":
        raise RuntimeError("Tensor post accumulate grad hooks should return None.")
    return result


def wrap_numpy(f: Callable[..., Any]) -> Callable[..., Any]:
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.
    """
    if not np:
        return f

    @functools.wraps(f)
    def wrap(*args: Any, **kwargs: Any) -> Any:
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, lambda x: x.numpy(), (args, kwargs)
        )
        out = f(*args, **kwargs)
        return pytree.tree_map_only(np.ndarray, lambda x: torch.as_tensor(x), out)

    return wrap


class FakeBackwardCFunction:
    def __init__(
        self,
        real: torch.autograd.function.BackwardCFunction,
        saved_tensors: List[torch.Tensor],
    ) -> None:
        self.real = real
        self.saved_tensors = saved_tensors
        self.aot_symints = real._get_compiled_autograd_symints()
        self.bw_module = real._forward_cls._lazy_backward_info.bw_module

    def __getattr__(self, name: str) -> Any:
        if name == "saved_variables":
            warnings.warn(
                "'saved_variables' is deprecated; use 'saved_tensors'",
                DeprecationWarning,
            )
            return self.saved_tensors

        return getattr(self.real, name)


def create_fake_ctx(
    ctx: torch.autograd.function.BackwardCFunction, saved_tensors: List[torch.Tensor]
) -> FakeBackwardCFunction:
    return FakeBackwardCFunction(ctx, saved_tensors)


# def call_backward_prologue(
#     ctx: torch.autograd.function.BackwardCFunction, saved_tensors: List[torch.Tensor],
#     # fakectx: FakeBackwardCFunction,
#     *args: Any,
# ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
#     fakectx = FakeBackwardCFunction(ctx, saved_tensors)
#     return fakectx._forward_cls._backward_prologue(fakectx, *args)  # type: ignore[attr-defined]


# def call_backward_impl(
#     ctx: torch.autograd.function.BackwardCFunction, saved_tensors: List[torch.Tensor],
#     # fakectx: FakeBackwardCFunction,
#     *args: Any,
# ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
#     fakectx = FakeBackwardCFunction(ctx, saved_tensors)
#     return fakectx._forward_cls._backward_impl(fakectx, *args)  # type: ignore[attr-defined]

# def call_backward_impl(
#     # ctx: torch.autograd.function.BackwardCFunction, saved_tensors: List[torch.Tensor],
#     fakectx: FakeBackwardCFunction,
#     *args: Any,
# ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
#     torch._dynamo.comptime.comptime.breakpoint()

#     # fakectx = FakeBackwardCFunction(ctx, saved_tensors)
#     return fakectx._forward_cls._backward_impl(fakectx, *args)  # type: ignore[attr-defined]


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]

def call_backward_impl(
    # ctx: torch.autograd.function.BackwardCFunction, saved_tensors: List[torch.Tensor],
    ctx: FakeBackwardCFunction,
    *args: Any,
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    # fakectx = FakeBackwardCFunction(ctx, saved_tensors)
    # assert len(args) == 1  # data-dependent jump? 
    all_args = args[0]
    # return fakectx._forward_cls._backward_impl(fakectx, *args)  # type: ignore[attr-defined]
    # bw_module = ctx._forward_cls._lazy_backward_info.bw_module
    bw_module = ctx.bw_module
    symints = ctx.aot_symints
    # assert len(symints) == ctx.symints  # data-dependent jump?
    # backward_state_indices
    # context = torch._C._DisableAutocast if disable_amp else nullcontext
    return normalize_as_list(bw_module(*all_args))


# def call_backward_epilogue(
#     ctx: torch.autograd.function.BackwardCFunction, saved_tensors: List[torch.Tensor],
#     # fakectx: FakeBackwardCFunction,
#     *args: Any,
# ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
#     fakectx = FakeBackwardCFunction(ctx, saved_tensors)
#     return fakectx._forward_cls._backward_epilogue(fakectx, *args)  # type: ignore[attr-defined]


def untyped_storage_size(x: torch.Tensor) -> int:
    return x.untyped_storage().size()


class FakeCompiledAutogradEngine:
    @staticmethod
    def queue_callback(
        final_callbacks: List[Callable[[], None]], cb: Callable[[], None]
    ) -> None:
        final_callbacks.append(cb)

    @staticmethod
    def exec_final_callbacks(final_callbacks: List[Callable[[], None]]) -> None:
        i = 0
        while i < len(final_callbacks):
            cb = final_callbacks[i]
            cb()
            i += 1
        final_callbacks.clear()

    @staticmethod
    def _exec_final_callbacks_stub() -> None:
        pass


def call_hook_from_backward_state(
    *args: Any, bw_state: Any, hook_name: str, **kwargs: Any
) -> Any:
    return getattr(bw_state, hook_name)(*args, **kwargs)


def call_module_hooks_from_backward_state(
    _: Any, result: Any, *args: Any, bw_state: Any, hooks_name: str, module_name: str
) -> Any:
    module = getattr(bw_state, module_name)
    hooks = getattr(bw_state, hooks_name)
    for hook in hooks:
        new_result = hook(module, result, *args)
        if new_result is not None:
            result = new_result
    return result

class CustomObj:
    pass
