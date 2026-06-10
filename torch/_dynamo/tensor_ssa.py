from __future__ import annotations

import operator
import time
from collections.abc import Callable, Sequence  # noqa: TC003
from typing import Any, NamedTuple, TYPE_CHECKING

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensor

from . import config
from .utils import (
    _disable_saved_tensors_hooks_during_tracing,
    ensure_graph_fake,
    proxy_args_kwargs,
    set_current_node,
    wrap_fake_exception,
)


if TYPE_CHECKING:
    from .symbolic_convert import InstructionTranslator

_BINARY_TENSOR_FNS: set[Callable[..., object]] = {
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.matmul,
}

_UNARY_TENSOR_FNS: set[Callable[..., object]] = {
    operator.neg,
    operator.pos,
}

_TENSOR_METHODS: set[str] = {
    "abs",
    "acos",
    "asin",
    "atan",
    "ceil",
    "clone",
    "cos",
    "cosh",
    "erf",
    "exp",
    "expm1",
    "floor",
    "log",
    "log1p",
    "neg",
    "reciprocal",
    "relu",
    "round",
    "sigmoid",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
}


class NormalizedArgs(NamedTuple):
    vt_args: list[Any]
    vt_kwargs: dict[str, Any]
    fake_args: list[Any]
    fake_kwargs: dict[str, Any]
    has_tensor_arg: bool


def _realize_lazy_tensor(value: Any) -> Any:
    from .variables.lazy import LazyVariableTracker

    if not isinstance(value, LazyVariableTracker):
        return value
    if value.is_realized():
        return value.realize()

    try:
        value_type = value.peek_type()
    except Exception:
        return value

    if isinstance(value_type, type) and issubclass(value_type, torch.Tensor):
        return value.realize()
    return value


def _normalize_arg(
    tx: InstructionTranslator, value: Any
) -> tuple[Any, Any, bool] | None:
    from .variables.constant import ConstantVariable
    from .variables.tensor import SymNodeVariable, TensorVariable

    value = _realize_lazy_tensor(value)
    if type(value) is TensorVariable and value.class_type is torch.Tensor:
        example_value = value.as_proxy().node.meta.get("example_value")
        if (
            isinstance(example_value, FakeTensor)
            and example_value.fake_mode is tx.fake_mode
            and not example_value.is_sparse
            and not example_value.is_nested
        ):
            return value, example_value, True
        return None

    if isinstance(value, ConstantVariable):
        constant = value.as_python_constant()
        if constant is None or type(constant) in (bool, int, float):
            return value, constant, False
        return None

    if isinstance(value, SymNodeVariable):
        return value, value.sym_num, False

    return None


def _normalize_args(
    tx: InstructionTranslator,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> NormalizedArgs | None:
    normalized_args = []
    fake_args = []
    has_tensor_arg = False

    for arg in args:
        normalized = _normalize_arg(tx, arg)
        if normalized is None:
            return None
        proxy_arg, fake_arg, is_tensor_arg = normalized
        normalized_args.append(proxy_arg)
        fake_args.append(fake_arg)
        has_tensor_arg = has_tensor_arg or is_tensor_arg

    normalized_kwargs = {}
    fake_kwargs = {}
    for key, value in kwargs.items():
        normalized = _normalize_arg(tx, value)
        if normalized is None:
            return None
        proxy_arg, fake_arg, is_tensor_arg = normalized
        normalized_kwargs[key] = proxy_arg
        fake_kwargs[key] = fake_arg
        has_tensor_arg = has_tensor_arg or is_tensor_arg

    return NormalizedArgs(
        normalized_args,
        normalized_kwargs,
        fake_args,
        fake_kwargs,
        has_tensor_arg,
    )


def _compute_fake_value(
    tx: InstructionTranslator,
    node: torch.fx.Node,
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any:
    from .exc import Unsupported

    _t0 = time.time_ns()
    fake_mode = tx.fake_mode
    try:
        if fake_mode is None:
            return None

        with (
            _disable_saved_tensors_hooks_during_tracing(),
            set_current_node(node),
            fake_mode,
            enable_python_dispatcher(),
        ):
            return wrap_fake_exception(lambda: fn(*args, **kwargs))
    except Unsupported:
        raise
    except (RuntimeError, TypeError):
        return None
    finally:
        tx.output.bytecode_tracing_timings.get_fake_value_ns += time.time_ns() - _t0


def _collect_mutation_inputs(
    tx: InstructionTranslator,
    node: torch.fx.Node,
) -> tuple[list[Any], dict[int, int]] | None:
    if not (config.use_graph_deduplication or config.track_nodes_for_deduplication):
        return None

    from .graph_utils import _get_flat_args
    from .utils import get_fake_values_from_nodes, is_fake

    flat_args_kwargs = get_fake_values_from_nodes(tx, _get_flat_args(node, {}), False)
    id_to_initial_version = {
        id(arg): arg._version for arg in flat_args_kwargs if is_fake(arg)
    }
    return flat_args_kwargs, id_to_initial_version


def _track_node_mutations(
    tx: InstructionTranslator,
    node: torch.fx.Node,
    mutation_inputs: tuple[list[Any], dict[int, int]] | None,
) -> None:
    if mutation_inputs is None:
        return

    flat_args_kwargs, id_to_initial_version = mutation_inputs
    tx.output.region_tracker.track_node_mutations(
        node,
        flat_args_kwargs,
        id_to_initial_version,
    )


def _can_use_fastpath(
    tx: InstructionTranslator, args: Sequence[Any], kwargs: dict[str, Any]
) -> bool:
    if not config.enable_tensor_ssa_fastpath:
        return False

    from .variables.torch_function import can_dispatch_torch_function

    return not can_dispatch_torch_function(tx, args, kwargs)


def maybe_fastpath_tensor_stack_op(
    tx: InstructionTranslator,
    fn: Callable[..., object],
    args: Sequence[Any],
) -> Any | None:
    if fn not in _BINARY_TENSOR_FNS and fn not in _UNARY_TENSOR_FNS:
        return None
    if not _can_use_fastpath(tx, args, {}):
        return None

    normalized = _normalize_args(tx, args, {})
    if normalized is None:
        return None

    if not normalized.has_tensor_arg:
        return None

    proxy = tx.output.create_proxy(
        "call_function",
        fn,
        *proxy_args_kwargs(normalized.vt_args, normalized.vt_kwargs),
    )
    try:
        mutation_inputs = _collect_mutation_inputs(tx, proxy.node)
        example_value = _compute_fake_value(
            tx, proxy.node, fn, normalized.fake_args, normalized.fake_kwargs
        )
        if isinstance(example_value, torch.Tensor):
            example_value = ensure_graph_fake(example_value, tx)
    except Exception:
        tx.output.remove_node(proxy.node)
        raise
    if not isinstance(example_value, torch.Tensor):
        tx.output.remove_node(proxy.node)
        return None

    _track_node_mutations(tx, proxy.node, mutation_inputs)

    from .variables.builder import wrap_fx_proxy

    return wrap_fx_proxy(tx, proxy, precomputed_example_value=example_value)


def maybe_fastpath_tensor_method(
    tx: InstructionTranslator,
    tensor: Any,
    name: str,
    args: Sequence[Any],
    kwargs: dict[str, Any],
) -> Any | None:
    if name not in _TENSOR_METHODS:
        return None

    all_args = [tensor, *args]
    if not _can_use_fastpath(tx, all_args, kwargs):
        return None

    normalized = _normalize_args(tx, all_args, kwargs)
    if normalized is None:
        return None

    if not normalized.has_tensor_arg:
        return None

    proxy = tx.output.create_proxy(
        "call_method",
        name,
        *proxy_args_kwargs(normalized.vt_args, normalized.vt_kwargs),
    )
    fake_self, *fake_method_args = normalized.fake_args
    try:
        mutation_inputs = _collect_mutation_inputs(tx, proxy.node)
        example_value = _compute_fake_value(
            tx,
            proxy.node,
            lambda *method_args, **method_kwargs: getattr(fake_self, name)(
                *method_args, **method_kwargs
            ),
            fake_method_args,
            normalized.fake_kwargs,
        )
        if isinstance(example_value, torch.Tensor):
            example_value = ensure_graph_fake(example_value, tx)
    except Exception:
        tx.output.remove_node(proxy.node)
        raise
    if not isinstance(example_value, torch.Tensor):
        tx.output.remove_node(proxy.node)
        return None

    _track_node_mutations(tx, proxy.node, mutation_inputs)

    from .variables.builder import wrap_fx_proxy

    return wrap_fx_proxy(tx, proxy, precomputed_example_value=example_value)
