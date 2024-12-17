# This module contains functions that *will be allowed* by dynamo

import functools
import warnings
from typing import Any, Callable, List, Optional, TYPE_CHECKING, Union
from typing_extensions import deprecated

import torch
import torch.utils._pytree as pytree


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:
    # TorchScript does not support `@deprecated`
    # This is a workaround to avoid breaking TorchScript
    @deprecated(
        "`torch._dynamo.external_utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
        category=FutureWarning,
    )
    def is_compiling() -> bool:
        return torch.compiler.is_compiling()

else:

    def is_compiling() -> bool:
        """
        Indicates whether we are tracing/compiling with torch.compile() or torch.export().
        """
        # NOTE: With `@torch.compile(backend="eager")`, torch._dynamo.is_compiling() will get traced
        # and return true. torch.compiler.is_compiling() is skipped and will return false.
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

    def __getattr__(self, name: str) -> Any:
        if name == "saved_variables":
            warnings.warn(
                "'saved_variables' is deprecated; use 'saved_tensors'",
                DeprecationWarning,
            )
            return self.saved_tensors

        return getattr(self.real, name)


def call_backward(
    backward_c_function: torch.autograd.function.BackwardCFunction,
    saved_tensors: List[torch.Tensor],
    *args: Any,
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    fake = FakeBackwardCFunction(backward_c_function, saved_tensors)
    grads = fake._forward_cls.backward(fake, *args)  # type: ignore[attr-defined]

    if not isinstance(grads, tuple):
        grads = (grads,)

    return grads


def normalize_as_list(x: Any) -> List[Any]:
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


def call_aot_bwd_impl(
    ctx: torch.autograd.function.BackwardCFunction,
    saved_tensors: List[torch.Tensor],
    all_args: List[
        Union[torch.Tensor, torch.fx.experimental._backward_state.BackwardState]
    ],
    backward_state: Optional[torch.fx.experimental._backward_state.BackwardState],
) -> List[torch.Tensor]:
    fakectx = FakeBackwardCFunction(ctx, saved_tensors)
    bw_module = fakectx._bw_module
    if backward_state is not None:
        all_args.append(backward_state)
    out = bw_module(*all_args)
    return normalize_as_list(out)


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
