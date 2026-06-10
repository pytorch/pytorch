from __future__ import annotations

import functools
import threading
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
from torch._ops import OpOverload


if TYPE_CHECKING:
    from collections.abc import Callable


_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")

aten = torch._ops.ops.aten

# pyrefly: ignore [implicit-any]
op_implementations_dict = {}
# pyrefly: ignore [implicit-any]
op_implementations_checks = []

_fake_impls_loaded = False
_fake_impls_loading = False
_fake_impls_lock = threading.RLock()


def ordered_set(*items: _T) -> dict[_T, bool]:
    return dict.fromkeys(items, True)


def ensure_fake_impls_loaded() -> None:
    global _fake_impls_loaded, _fake_impls_loading

    if _fake_impls_loaded:
        return

    with _fake_impls_lock:
        if _fake_impls_loaded:
            return
        if _fake_impls_loading:
            return

        # The lock keeps concurrent callers from observing partially registered
        # fake impls while Python's import lock serializes module execution.
        # _fake_impls_loading only handles same-thread reentry through the
        # fake_tensor -> registry -> fake_impls -> fake_tensor import cycle.
        dict_snapshot = op_implementations_dict.copy()
        checks_snapshot = list(op_implementations_checks)
        _fake_impls_loading = True
        try:
            import torch._subclasses.fake_impls  # noqa: F401
        except Exception:
            op_implementations_dict.clear()
            op_implementations_dict.update(dict_snapshot)
            op_implementations_checks[:] = checks_snapshot
            raise
        else:
            _fake_impls_loaded = True
        finally:
            _fake_impls_loading = False


def get_op_implementations_checks() -> list[tuple[Callable[[OpOverload], bool], Any]]:
    ensure_fake_impls_loaded()
    return op_implementations_checks


def get_fast_op_impls() -> dict[OpOverload, Callable[..., Any]]:
    ensure_fake_impls_loaded()
    # The fast implementations depend on FakeTensor helpers and still live in
    # fake_impls; after ensure_fake_impls_loaded(), this import is a sys.modules
    # lookup that preserves the lazy-loading boundary for fake_tensor.py.
    from torch._subclasses.fake_impls import get_fast_op_impls as _get_fast_op_impls

    return _get_fast_op_impls()


def contains_tensor_types(type_: Any) -> bool:
    tensor_type = torch._C.TensorType.get()
    return type_.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type_.containedTypes()
    )


@functools.cache
def _is_tensor_constructor(func: OpOverload) -> bool:
    if not isinstance(func, OpOverload):
        raise AssertionError(f"func must be an OpOverload, got {type(func)}")
    schema = func._schema
    if any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return False
    # TODO: no real reason to restrict multiple outputs
    return (
        len(schema.returns) == 1 and schema.returns[0].type is torch._C.TensorType.get()
    )


def register_op_impl(
    run_impl_check: Callable[[OpOverload], bool]
    | OpOverload
    | list[OpOverload]
    | tuple[OpOverload, ...],
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def impl_decorator(op_impl: Callable[_P, _R]) -> Callable[_P, _R]:
        if isinstance(run_impl_check, OpOverload):
            if run_impl_check in op_implementations_dict:
                raise AssertionError(f"duplicate registration: {run_impl_check}")
            op_implementations_dict[run_impl_check] = op_impl
        elif isinstance(run_impl_check, (list, tuple)):
            for op in run_impl_check:
                register_op_impl(op)(op_impl)
        else:
            if not callable(run_impl_check):
                raise AssertionError(
                    f"run_impl_check must be callable, got {type(run_impl_check)}"
                )
            op_implementations_checks.append((run_impl_check, op_impl))

        return op_impl

    return impl_decorator


def _is_op_registered_to_fake_rule(op: OpOverload) -> bool:
    ensure_fake_impls_loaded()
    return op in op_implementations_dict


def _deregister_op_impl(op: OpOverload) -> None:
    op_implementations_dict.pop(op, None)
    for check, impl in op_implementations_checks:
        if check is op:
            op_implementations_checks.remove((check, impl))
            break


_like_tensor_constructors = ordered_set(
    aten.empty_like.default,
    aten.empty_like.out,
    aten.full_like.default,
    aten.full_like.out,
    aten.ones_like.default,
    aten.ones_like.out,
    aten.rand_like.default,
    aten.rand_like.generator,
    aten.rand_like.out,
    aten.rand_like.generator_out,
    aten.randn_like.default,
    aten.randn_like.generator,
    aten.randn_like.out,
    aten.randn_like.generator_out,
    aten.randint_like.default,
    aten.randint_like.generator,
    aten.randint_like.Tensor,
    aten.randint_like.Tensor_generator,
    aten.randint_like.Tensor_out,
    aten.randint_like.Tensor_generator_out,
    aten.randint_like.out,
    aten.randint_like.generator_out,
    aten.randint_like.low_dtype,
    aten.randint_like.low_generator_dtype,
    aten.randint_like.low_dtype_out,
    aten.randint_like.low_generator_dtype_out,
    aten.zeros_like.default,
    aten.zeros_like.out,
    aten.new_empty.default,
    aten.new_empty.out,
    aten.new_empty_strided.default,
    aten.new_empty_strided.out,
    aten.new_full.default,
    aten.new_full.out,
    aten.new_zeros.default,
    aten.new_zeros.out,
    aten.new_ones.default,
    aten.new_ones.out,
)


_device_not_kwarg_ops = ordered_set(
    aten._resize_output_.default,
    aten._nested_tensor_from_tensor_list.default,
    aten._nested_tensor_from_tensor_list.out,
    aten.pin_memory.default,
    aten.to.device,
    aten.to.prim_Device,
    aten.is_pinned.default,
    aten._pin_memory.default,
    aten._pin_memory.out,
    aten._resize_output.default,
    aten._resize_output.out,
)


def stride_incorrect_op(op: OpOverload) -> bool:
    return False


def has_meta(func: OpOverload) -> bool:
    return torch._C._dispatch_has_computed_kernel_for_dispatch_key(func.name(), "Meta")
