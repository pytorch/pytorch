"""
This module contains utility functions that are explicitly allowed to be called during
TorchDynamo compilation. These functions are carefully vetted to ensure they work
correctly within the TorchDynamo tracing and compilation process.

Key functionality groups:

- Compilation State:
  Functions for checking compilation state (is_compiling)

- Function Wrapping:
  Utilities for wrapping functions (wrap_inline, wrap_numpy) to work with
  TorchDynamo compilation

- Autograd Hooks:
  Functions and classes for handling autograd hooks and backward passes
  (call_hook, FakeBackwardCFunction, etc.)

- Tensor Operations:
  Utility functions for tensor operations and transformations
"""

import functools
import warnings
from typing import Any, Callable, Optional, TYPE_CHECKING, TypeVar, Union
from typing_extensions import deprecated, ParamSpec

import torch
import torch.utils._pytree as pytree


try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

_P = ParamSpec("_P")
_R = TypeVar("_R")

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


def wrap_inline(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Create an extra frame around fn that is not in skipfiles.
    """

    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
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


def wrap_numpy(f: Callable[_P, _R]) -> Callable[_P, _R]:
    r"""Decorator that turns a function from ``np.ndarray``s to ``np.ndarray``s into a function
    from ``torch.Tensor``s to ``torch.Tensor``s.
    """
    if not np:
        return f

    @functools.wraps(f)
    def wrap(*args: _P.args, **kwargs: _P.kwargs) -> pytree.PyTree:
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, lambda x: x.numpy(), (args, kwargs)
        )
        # pyrefly: ignore  # invalid-param-spec
        out = f(*args, **kwargs)
        # pyrefly: ignore  # missing-attribute
        return pytree.tree_map_only(np.ndarray, lambda x: torch.as_tensor(x), out)

    return wrap


class FakeBackwardCFunction:
    def __init__(
        self,
        real: torch.autograd.function.BackwardCFunction,
        saved_tensors: list[torch.Tensor],
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
    saved_tensors: list[torch.Tensor],
    *args: Any,
) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
    fake = FakeBackwardCFunction(backward_c_function, saved_tensors)
    grads = fake._forward_cls.backward(fake, *args)  # type: ignore[attr-defined]

    if not isinstance(grads, tuple):
        grads = (grads,)

    return grads


def normalize_as_list(x: Any) -> list[Any]:
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


def untyped_storage_size(x: torch.Tensor) -> int:
    return x.untyped_storage().size()


class FakeCompiledAutogradEngine:
    @staticmethod
    def queue_callback(
        final_callbacks: list[Callable[[], None]], cb: Callable[[], None]
    ) -> None:
        final_callbacks.append(cb)

    @staticmethod
    def exec_final_callbacks(final_callbacks: list[Callable[[], None]]) -> None:
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


# used for torch._dynamo.disable(recursive=False)
def get_nonrecursive_disable_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    # wrap function to get the right error message
    # this function is in external_utils so that convert_frame doesn't skip it.
    @functools.wraps(fn)
    def nonrecursive_disable_wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return fn(*args, **kwargs)

    return nonrecursive_disable_wrapper


def wrap_dunder_call_ctx_manager(self: Any, func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Apply self as a ctx manager around a call to func
    """

    # NOTE: do not functools.wraps(func) because we don't ever want this frame to be skipped!
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self:
            return func(*args, **kwargs)

    return inner


# Use only on ints marked dynamic via torch.empty(0, integer)
# Currently only way to mark ints as dynamic: https://github.com/pytorch/pytorch/issues/129623
def unwrap_maybe_dynamic_int(x: Union[torch.Tensor, int]) -> int:
    if isinstance(x, torch.Tensor):
        # x.size() is expected to be [0, dynamic_int]
        return x.size(1)
    return x


def call_accumulate_grad(
    variable: torch.Tensor, grad: torch.Tensor, has_post_hooks: bool
) -> None:
    updated_grad = torch._dynamo.compiled_autograd.ops.AccumulateGrad(  # type: ignore[attr-defined]
        [grad], variable, variable.grad, has_post_hooks
    )
    variable.grad = updated_grad[0]


def wrap_inline_with_error_on_graph_break(
    fn: Callable[_P, _R], error_on_graph_break: bool
) -> Callable[_P, _R]:
    # NB: need multiple definitions in order to prevent `fullgraph` from
    # being a freevar of wrapper
    # NOTE: do not functools.wraps(fn) because we don't ever want these wrappers to be skipped!
    if error_on_graph_break:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with torch._dynamo.error_on_graph_break(True):
                return fn(*args, **kwargs)

    else:

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with torch._dynamo.error_on_graph_break(False):
                return fn(*args, **kwargs)

    return wrapper


def filter_out_const_values(tup: tuple[Any, ...], masks: list[bool]) -> tuple[Any, ...]:
    """
    masks is a list of bools, where True means the corresponding element in tup
    is a const value. Filter out the const values.
    """
    out = []
    for mask_idx, mask in enumerate(masks):
        if not mask:
            out.append(tup[mask_idx])
    return tuple(out)


def insert_const_values_with_mask(
    tup: tuple[Any, ...], masks: list[bool], values: tuple[Any, ...]
) -> tuple[Any, ...]:
    """
    masks and values are of same length. For indices where the mask is True, use
    the const_values to fill in.
    """
    out = []
    idx = 0
    for mask_idx, mask in enumerate(masks):
        if mask:
            out.append(values[mask_idx])
        else:
            out.append(tup[idx])
            idx += 1
    return tuple(out)
