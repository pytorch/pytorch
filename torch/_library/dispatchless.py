from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Generic, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

from torch._ops import HigherOrderOperator


if TYPE_CHECKING:
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


_P = ParamSpec("_P")
_R = TypeVar("_R")


@cache
def _proxy_mode_getter() -> Callable[[], ProxyTorchDispatchMode | None]:
    from torch.fx.experimental.proxy_tensor import get_proxy_mode

    return get_proxy_mode


def _trace_through_transforms(args) -> bool:
    import torch
    import torch.utils._pytree as pytree

    # The C++ any-autocast helper does not yet include MPS or MAIA.
    return (
        torch._C._are_functorch_transforms_active()
        or torch._C._is_any_autocast_enabled()
        or torch.is_autocast_enabled("mps")
        or torch.is_autocast_enabled("maia")
        or (
            torch.is_grad_enabled()
            and any(
                isinstance(arg, torch.Tensor) and arg.requires_grad
                for arg in pytree.tree_leaves(args)
            )
        )
    )


def _call_with_tensor_checks(fn, args, kwargs):
    import torch
    import torch.utils._pytree as pytree
    from torch._library.utils import zip_schema
    from torch.utils._python_dispatch import TorchDispatchMode

    def unwrap(tensor):
        functorch = torch._C._functorch
        while functorch.is_batchedtensor(tensor) or functorch.is_gradtrackingtensor(tensor):
            tensor = functorch.get_unwrapped(tensor)
        return tensor

    inputs = [
        unwrap(value)
        for value in pytree.tree_leaves((args, kwargs))
        if isinstance(value, torch.Tensor)
    ]

    class MutationGuard(TorchDispatchMode):
        supports_higher_order_operators = True

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if isinstance(func, torch._ops.OpOverload) and func._schema.is_mutable:
                for info, value in zip_schema(func._schema, args, kwargs):
                    if info.is_write and any(
                        isinstance(tensor, torch.Tensor)
                        and any(torch._C._is_alias_of(unwrap(tensor), x) for x in inputs)
                        for tensor in pytree.tree_leaves(value)
                    ):
                        raise RuntimeError(
                            "dispatchless_custom_op must not mutate input tensors "
                            f"during tracing: {func} writes to argument {info.name}"
                        )
            return func(*args, **kwargs)

    with MutationGuard():
        result = fn(*args, **kwargs)
    outputs = []
    for value in pytree.tree_leaves(result):
        if isinstance(value, torch.Tensor):
            tensor = unwrap(value)
            if any(torch._C._is_alias_of(tensor, x) for x in (*inputs, *outputs)):
                raise RuntimeError(
                    "dispatchless_custom_op outputs must not alias inputs or "
                    "other outputs during tracing; return a clone instead"
                )
            outputs.append(tensor)
    return result


@dataclass(frozen=True, eq=False)
class _DispatchlessCustomOp(Generic[_P, _R]):
    __wrapped__: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        mode = _proxy_mode_getter()()
        if mode is None:
            return self.__wrapped__(*args, **kwargs)
        if _trace_through_transforms((args, kwargs)):
            return self.__wrapped__(*args, **kwargs)

        import torch.utils._pytree as pytree
        from torch._higher_order_ops.flat_apply import func_to_graphable, to_graphable
        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

        # Dispatch normally removes the active mode before invoking its handler.
        with disable_proxy_modes_tracing():
            flat_args, input_spec = to_graphable((args, kwargs))
            _, func_spec = func_to_graphable(self.__wrapped__)
            flat_out, output_spec = to_graphable(
                _call_with_tensor_checks(self.__wrapped__, args, kwargs)
            )
            flat_out = _trace_flat_call(
                mode, func_spec, input_spec, output_spec, flat_args, flat_out
            )
        return pytree.tree_unflatten(flat_out, output_spec)


def dispatchless_custom_op(fn: Callable[_P, _R]) -> _DispatchlessCustomOp[_P, _R]:
    """Wrap a functional Python function as an experimental dispatchless operator.

    Eager calls invoke ``fn`` directly. ``make_fx`` and Dynamo preserve a flat
    operator call with pytree inputs and outputs; Inductor traces through its
    implementation. Tensor dependencies must be explicit arguments, and ``fn``
    must work with FakeTensor inputs. Registered pytree containers and constants
    are supported, with a stable output structure for each captured call.

    The function must not mutate inputs or Python state, or return tensor aliases.
    These purity requirements are preconditions, not eager runtime checks.
    Capturing an opaque call diagnoses dispatched input writes and output aliases.
    These checks are not a proof of purity: Python state, writes outside dispatch,
    opaque higher-order operator bodies, and storage-less tensor subclasses are
    not fully checked. Transform trace-through paths retain their native behavior.
    Schema-marked writes to input aliases are conservatively rejected, including
    metadata-only changes to local views of inputs.
    Autograd follows the backing function. AOTAutograd decomposes the operator
    before differentiating it. Proxy tracing also traces through the backing
    function under autograd, torch.func transforms, and autocast. Custom gradient
    registration and export serialization are not supported. Compiling
    ``torch.func.functionalize(fn)`` has the same restrictions as for plain
    functions; eager functionalization and tracing its result are supported.
    """
    if not callable(fn):
        raise TypeError("dispatchless_custom_op expects a callable")
    return _DispatchlessCustomOp(fn)


def _flat_call(func_spec, input_spec, output_spec, *flat_args):
    import torch.utils._pytree as pytree
    from torch._higher_order_ops.flat_apply import to_graphable

    fn = pytree._retrieve_constant(func_spec)
    args, kwargs = pytree.tree_unflatten(flat_args, input_spec)
    flat_out, actual_spec = to_graphable(fn(*args, **kwargs))
    if actual_spec != output_spec:
        raise RuntimeError(
            "Dispatchless custom operators must return the same output pytree "
            f"structure for each captured call: expected {output_spec}, "
            f"got {actual_spec}"
        )
    return flat_out


def _trace_flat_call(
    mode, func_spec, input_spec, output_spec, flat_args, flat_out=None
):
    import torch.utils._pytree as pytree
    from torch.fx.experimental.proxy_tensor import (
        maybe_handle_decomp,
        track_tensor_tree,
    )

    node_args = (func_spec, input_spec, output_spec, *flat_args)
    result = maybe_handle_decomp(mode, flat_dispatchless_call, node_args, {})
    if result is not NotImplemented:
        return result
    if flat_out is None:
        flat_out = _call_with_tensor_checks(_flat_call, node_args, {})
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, node_args)
    proxy = mode.tracer.create_proxy(
        "call_function", flat_dispatchless_call, proxy_args, {}
    )
    return track_tensor_tree(flat_out, proxy, constant=None, tracer=mode.tracer)


class _FlatDispatchlessCall(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_dispatchless_call")

    def __call__(self, func_spec, input_spec, output_spec, *flat_args):
        mode = _proxy_mode_getter()()
        if mode is None:
            return _flat_call(func_spec, input_spec, output_spec, *flat_args)
        if _trace_through_transforms(flat_args):
            return _flat_call(func_spec, input_spec, output_spec, *flat_args)

        from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

        with disable_proxy_modes_tracing():
            return _trace_flat_call(mode, func_spec, input_spec, output_spec, flat_args)


flat_dispatchless_call = _FlatDispatchlessCall()
