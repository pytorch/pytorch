from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar
from typing_extensions import ParamSpec

from torch._ops import HigherOrderOperator


_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True, eq=False)
class _DispatchlessCustomOp(Generic[_P, _R]):
    __wrapped__: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        from torch.fx.experimental.proxy_tensor import (
            disable_proxy_modes_tracing,
            get_proxy_mode,
        )

        mode = get_proxy_mode()
        if mode is None:
            return self.__wrapped__(*args, **kwargs)

        import torch.utils._pytree as pytree
        from torch._higher_order_ops.flat_apply import func_to_graphable, to_graphable

        # Dispatch normally removes the active mode before invoking its handler.
        with disable_proxy_modes_tracing():
            flat_args, input_spec = to_graphable((args, kwargs))
            _, func_spec = func_to_graphable(self.__wrapped__)
            flat_out, output_spec = to_graphable(self.__wrapped__(*args, **kwargs))
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
    Autograd follows the backing function eagerly and with Inductor. Other AOT
    backends must decompose the operator before differentiating it. Custom
    gradient registration and export serialization are not supported.
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
        flat_out = _flat_call(*node_args)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, node_args)
    proxy = mode.tracer.create_proxy(
        "call_function", flat_dispatchless_call, proxy_args, {}
    )
    return track_tensor_tree(flat_out, proxy, constant=None, tracer=mode.tracer)


class _FlatDispatchlessCall(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flat_dispatchless_call")

    def __call__(self, func_spec, input_spec, output_spec, *flat_args):
        from torch.fx.experimental.proxy_tensor import (
            disable_proxy_modes_tracing,
            get_proxy_mode,
        )

        mode = get_proxy_mode()
        if mode is None:
            return _flat_call(func_spec, input_spec, output_spec, *flat_args)
        with disable_proxy_modes_tracing():
            return _trace_flat_call(mode, func_spec, input_spec, output_spec, flat_args)


flat_dispatchless_call = _FlatDispatchlessCall()
