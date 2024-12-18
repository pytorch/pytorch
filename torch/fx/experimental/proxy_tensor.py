# mypy: allow-untyped-decorators
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import inspect
import logging
import operator
import traceback
import typing
import typing_extensions
import warnings
import weakref
from collections import defaultdict
from contextlib import _GeneratorContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    overload,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import Concatenate, ParamSpec, Self
from weakref import WeakKeyDictionary

import torch
import torch._ops
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import SymBool, SymInt, Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._library.fake_class_registry import FakeScriptObject
from torch._logging import trace_structured
from torch._subclasses.fake_impls import fast_detach
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    is_fake,
    unset_fake_temporarily,
)
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.graph_module import _assign_attr
from torch.fx.node import _side_effectful_need_to_be_preserved_pre_dispatch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
    _disable_infra_mode,
    _push_mode,
    _unset_infra_mode,
    TorchDispatchMode,
)
from torch.utils._stats import count
from torch.utils._thunk import Thunk
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary, WeakTensorKeyDictionary

from ._backward_state import BackwardState
from .sym_node import SymNode


if TYPE_CHECKING:
    import types
    from collections.abc import MutableMapping

    import sympy

    from torch._ops import OpOverload
    from torch.fx._symbolic_trace import PHBase
    from torch.types import IntLikeType

__all__ = [
    "PythonKeyTracer",
    "dispatch_trace",
    "make_fx",
    "DecompositionInterpreter",
    "py_sym_types",
    "get_innermost_proxy_mode",
    "get_proxy_mode",
    "handle_sym_dispatch",
    "maybe_enable_thunkify",
    "maybe_disable_thunkify",
]

_ProxyTracer = Union["PythonKeyTracer", "_GraphAppendingTracerEx"]

_AnyScriptObject = (torch.ScriptObject, FakeScriptObject)
_AnyScriptObjectType = Union[torch.ScriptObject, FakeScriptObject]

aten = torch.ops.aten
prim = torch.ops.prim

log = logging.getLogger(__name__)
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

CURRENT_DECOMPOSITION_TABLE: Mapping[OpOverload, Callable] = {}

CONSTANT_NUMEL_LIMIT = 1

T = TypeVar("T")
U = TypeVar("U")
_P = ParamSpec("_P")
R = TypeVar("R")

null_ctx_type = type(nullcontext)
# We currently convert all SymInt to proxies before we use them.
# This could plausibly be handled at the Dynamo level.
pytree.register_pytree_node(
    torch.Size,
    lambda xs: (list(xs), None),
    lambda xs, _: tuple(xs),
    flatten_with_keys_fn=lambda xs: (
        [(pytree.SequenceKey(i), x) for i, x in enumerate(xs)],
        None,
    ),
    serialized_type_name="torch.Size",
)


def fake_signature(fn: Callable[_P, R], nargs: int) -> Callable[_P, R]:
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})


@contextmanager
def decompose(
    decomposition_table: Optional[Mapping[OpOverload, Callable]]
) -> Generator[Mapping[OpOverload, Callable], None, None]:
    global CURRENT_DECOMPOSITION_TABLE
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table or {}
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table


# ensure we cannot collide with other properties
proxy_slot = object()


class _NoDefault:
    pass


no_default = _NoDefault()

from torch.types import py_sym_types, PySymType


class _HasMeta(Protocol):
    meta: Dict[str, PySymType]


def is_sym_node(node: _HasMeta) -> bool:
    assert hasattr(node, "meta"), "All nodes traced with proxy_tensor should have meta"
    return "val" in node.meta and isinstance(node.meta["val"], py_sym_types)


@overload
def set_proxy_slot(obj: Tensor, tracer: _ProxyTracer, proxy: _ProxyTensor) -> None:
    ...


@overload
def set_proxy_slot(
    obj: _AnyScriptObjectType, tracer: _ProxyTracer, proxy: Proxy
) -> None:
    ...


@overload
def set_proxy_slot(
    obj: PySymType, tracer: _ProxyTracer, proxy: _PySymProxyType
) -> None:
    ...


def set_proxy_slot(
    obj: Union[PySymType, _AnyScriptObjectType, Tensor],
    tracer: _ProxyTracer,
    proxy: object,
) -> None:
    log.debug("set_proxy_slot %s (%s) %s", obj, id(obj), proxy)
    if isinstance(obj, Tensor):
        # We DO want to clobber proxies whenever we run an inplace operation
        # on a tensor, and it affects the metadata on the proxy.
        assert isinstance(proxy, _ProxyTensor)
        tracer.tensor_tracker[obj] = proxy
    elif isinstance(obj, (_AnyScriptObject)):
        # We DO want to clobber proxies, with a similar rationale as for tensors.
        assert isinstance(proxy, Proxy)
        tracer.script_object_tracker[obj] = proxy
    else:
        # NB: Never clobber pre-existing proxy.  Although the proxies
        # are in principle equivalent, when we do graph partitioning
        # we need there not to be spurious dependencies on tangent inputs.
        # This works because primals get their SymInts set first, and
        # THEN later we allocate tangent inputs.  Make sure if a SymInt
        # is derivable from a primal that we use that.
        assert isinstance(obj, py_sym_types), type(obj)
        if obj not in tracer.symnode_tracker:
            tracer.symnode_tracker[obj] = typing.cast(_PySymProxyType, proxy)

            # WAR: python test/dynamo/test_subclasses.py
            # TestNestedTensor.test_basic_autograd
            #
            # AOTAutograd doesn't pass the "outer sizes" as an actual argument
            # to make_fx, but it is made use of internally in AOTAutograd's
            # call to tensor unflatten.  Because the outer sizes isn't passed
            # as an argument, it is therefore untracked.  However, it turns
            # out you luck out, because *Dynamo* will manually add the outer
            # sizes as an argument so you can fix up the proxy'ness.
            #
            # This is probably fixed in
            # https://github.com/pytorch/pytorch/pull/125941/
            import sympy

            if isinstance(obj.node.expr, sympy.Symbol):
                tracer.sympy_expr_tracker[obj.node.expr] = proxy


def has_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> bool:
    assert isinstance(obj, (Tensor, SymNode)), type(obj)
    return bool(get_proxy_slot(obj, tracer, False, lambda _: True))


_PySymProxyType = Thunk[Proxy]


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
) -> _ProxyTensor:
    ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
) -> Union[_ProxyTensor, U]:
    ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_ProxyTensor], R],
) -> Union[R, U]:
    ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
) -> Proxy:
    ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
) -> Union[Proxy, U]:
    ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[Proxy], R],
) -> Union[R, U]:
    ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
) -> _PySymProxyType:
    ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: T,
) -> Union[T, _PySymProxyType]:
    ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_PySymProxyType], R],
) -> Union[R, U]:
    ...


# the default argument is what to return if the slot is not set.
# the transform argument is handy if you need to extract a subfield from
# the successfully looked up result (but NOT the default.)
def get_proxy_slot(
    obj: Union[Tensor, _AnyScriptObjectType, PySymType],
    tracer: _ProxyTracer,
    default: object = no_default,
    transform: Callable = lambda x: x,
) -> object:
    tracker: Any
    if isinstance(obj, Tensor):
        tracker = tracer.tensor_tracker
    elif isinstance(obj, _AnyScriptObject):
        tracker = tracer.script_object_tracker
    else:
        assert isinstance(obj, py_sym_types), type(obj)
        tracker = tracer.symnode_tracker

    if obj not in tracker:
        # Last ditch
        if isinstance(obj, py_sym_types) and obj.node.expr in tracer.sympy_expr_tracker:
            value = tracer.sympy_expr_tracker[obj.node.expr]
        else:
            if isinstance(default, _NoDefault):
                raise RuntimeError(
                    f"{obj} ({id(obj)})is not tracked with proxy for {tracer}"
                )
            return default
    else:
        value = tracker[obj]
    res = transform(value)
    return res


def snapshot_fake(val: Tensor) -> Optional[Tensor]:
    # val.detach() will also eventually call fast_detach(),
    # but this saves us a full trip into __torch_dispatch__
    # (snapshot_fake is called a lot)
    if isinstance(val, FakeTensor):
        return fast_detach(val.fake_mode, val)
    else:
        return val.detach()


_ExtractValType = Optional[
    Union[
        PySymType,
        _AnyScriptObjectType,
        BackwardState,
        List["_ExtractValType"],
        Tuple["_ExtractValType", ...],
        Dict[str, "_ExtractValType"],
        Tensor,
        int,
        float,
        bool,
    ]
]


def extract_val(val: _ExtractValType) -> _ExtractValType:
    if is_fake(val):
        return snapshot_fake(val)
    elif isinstance(val, py_sym_types):
        return val
    elif isinstance(val, _AnyScriptObject):
        return val
    elif isinstance(val, BackwardState):
        return val
    elif isinstance(val, (list, tuple)):
        return val.__class__([extract_val(x) for x in val])
    elif isinstance(val, dict):
        return {k: extract_val(v) for k, v in val.items()}
    elif isinstance(val, Tensor):
        if not val.is_sparse:
            # NB: Kinda hacky, but we should try to get val as the metadata
            # everywhere
            # TODO: This doesn't properly track storages.  A more robust
            # approach would be to maintain a per-trace FakeTensorMode and
            # from_real_tensor to create fake values (don't forget to
            # snapshot_fake)
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            with fake_tensor_mode:
                return torch.empty_strided(
                    val.shape, val.stride(), device=val.device, dtype=val.dtype
                )
        else:
            return None
    elif isinstance(val, (int, float, bool)):
        return val
    elif val is None:
        return None

    typing_extensions.assert_never(val)


@contextmanager
def _enable_thunkify(
    tracer: _ProxyTracer, *, enable: bool = True
) -> Generator[None, None, None]:
    """
    Enable thunkification inside the context manager.  Thunkification prevents
    SymNode computation from directly being traced into an FX graph; instead,
    the compute is only added to the graph if it is actually used.  This helps
    us track SymNode compute when it is computed (since we need /something/
    to put in the tracker) even if it is unlikely to be used.
    """
    old = tracer.enable_thunkify
    tracer.enable_thunkify = enable
    try:
        yield
    finally:
        tracer.enable_thunkify = old


@contextmanager
def maybe_disable_thunkify() -> Generator[None, None, None]:
    """Within a context, disable thunkification.  See :func:`maybe_enable_thunkify`
    for more details.  This is helpful if you have a wrapper function which
    you want to enable thunkification on, but in some segment on the inside (say,
    the original user function), you want to disable thunkification as you know
    it is not needed there.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer, enable=False):
            yield
    else:
        yield


@contextmanager
def maybe_enable_thunkify() -> Generator[None, None, None]:
    """Within this context manager, if you are doing make_fx tracing, we will thunkify
    all SymNode compute and avoid tracing it into the graph unless it is actually needed.
    You should prefer to avoid using this as much as possible, as lazy evaluation of
    SymNode tracing can lead to long chains of thunks which will stack overflow
    if you evaluate them.  However, this is currently sometimes necessary as there
    are buggy parts of PT2 which will fail with "s0 is not tracked with proxy" error
    due to insufficient tracing of SymNode computation.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer):
            yield
    else:
        yield


# Note [invariants for node meta 'val']
# What invariants do we have for the 'val' set on the FX node?  It has accurate
# metadata... but only for metadata that exists "below" all other subsystems
# (most notably autograd, but also vmap, functorch transforms, etc).  This means
# you can get the dtype, shape, stride, storage, but you CANNOT get requires_grad,
# grad_fn, _base (_base actually may be set due to recursive call to
# ADInplaceOrView, but you shouldn't rely on it.)
def set_meta(proxy: Proxy, val: _ExtractValType) -> Proxy:
    proxy.node.meta["val"] = extract_val(val)

    with _enable_thunkify(proxy.tracer):  # type: ignore[arg-type]
        # Best effort tensor_meta setting; prefer using val!
        if is_fake(val):
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
        elif isinstance(val, Tensor) and not val.is_sparse:
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
    return proxy


def thunkify(
    tracer: _ProxyTracer, f: Callable[_P, R], *args: _P.args, **kwargs: _P.kwargs
) -> Thunk[R]:
    """
    Delays computation of f until it's called again
    Also caches the result
    """
    if tracer.enable_thunkify:
        return Thunk(functools.partial(f, *args, **kwargs))
    else:
        r = f(*args, **kwargs)
        return Thunk(lambda: r)


def track_tensor(
    tensor: Tensor, proxy: Proxy, *, constant: Optional[Tensor], tracer: _ProxyTracer
) -> None:
    def try_set_proxy_slot(
        outer_s: IntLikeType,
        proxy_callable: Callable[Concatenate[PySymType, _P], Proxy],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> None:
        assert callable(proxy_callable)
        if isinstance(outer_s, SymInt):
            with _enable_thunkify(tracer):
                set_proxy_slot(
                    outer_s,
                    tracer,
                    thunkify(tracer, proxy_callable, outer_s, *args, **kwargs),
                )

    # The basic idea is that we need to associate each tensor/SymInt
    # with a Proxy.  How do we setup this association?  We just store
    # the proxy on the proxy slot of the object, keyed on the tracer
    # (so that if we have multiple tracers at the same time, they
    # don't clobber each other.)
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(
            s,
            lambda x, i: set_meta(
                tracer.create_proxy(
                    "call_function", torch.ops.aten.sym_size.int, (proxy, i), {}
                ),
                x,
            ),
            i,
        )

    if not is_sparse_any(tensor):
        for i, s in enumerate(tensor.stride()):
            try_set_proxy_slot(
                s,
                lambda x, i: set_meta(
                    tracer.create_proxy(
                        "call_function", torch.ops.aten.sym_stride.int, (proxy, i), {}
                    ),
                    x,
                ),
                i,
            )

    try_set_proxy_slot(
        tensor.numel(),
        lambda x: set_meta(
            tracer.create_proxy(
                "call_function", torch.ops.aten.sym_numel.default, (proxy,), {}
            ),
            x,
        ),
    )
    if not is_sparse_any(tensor):
        try_set_proxy_slot(
            tensor.storage_offset(),
            lambda x: set_meta(
                tracer.create_proxy(
                    "call_function",
                    torch.ops.aten.sym_storage_offset.default,
                    (proxy,),
                    {},
                ),
                x,
            ),
        )
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))


_NestedProxys = Union[
    Proxy, Sequence["_NestedProxys"], Mapping[object, "_NestedProxys"]
]
_NestedTensors = Union[
    Tensor, Sequence["_NestedTensors"], Mapping[object, "_NestedTensors"]
]


def track_tensor_tree(
    inner_res: T,
    proxy_res: _NestedProxys,
    *,
    constant: Optional[_NestedTensors],
    tracer: _ProxyTracer,
) -> T:
    # NB: We call set_unbacked_bindings only on the *topmost* call to
    # track_tensor_tree, not recursive calls.  This is because there must
    # be only ONE unbacked_binding proxy call, and it should be the one
    # where all of the unbacked SymInts actually first come into existence.
    # If you call this again on the inner proxies for the tuple projections,
    # you will have multiple unbacked_bindings for the same symbol, but
    # they're not going to show up anywhere.
    #
    # I was briefly deceived into setting unbacked bindings recursively when
    # working on https://github.com/pytorch/pytorch/pull/133585 because I
    # observed that some extra unbacked bindings were needed to handle some
    # higher order operator code.  But actually it looks like this was
    # just an unrelated bug that needed to be fixed separately.
    _set_unbacked_bindings(inner_res, proxy_res)

    def wrap_with_proxy(
        e: object, proxy: _NestedProxys, constant: Optional[_NestedTensors]
    ) -> None:
        if isinstance(e, Tensor):
            assert isinstance(proxy, Proxy)
            assert constant is None or isinstance(constant, Tensor)
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            assert isinstance(proxy, Proxy)
            # NB: eagerly set meta here, so that the numbering is in order
            set_meta(proxy, e)
            set_proxy_slot(e, tracer, thunkify(tracer, lambda: proxy))
        elif isinstance(e, _AnyScriptObject):
            assert isinstance(proxy, Proxy)
            set_proxy_slot(e, tracer, proxy)
            set_meta(proxy, e)
        elif isinstance(e, (tuple, list)):
            # example use case: allreduce_ returns ([tensor], work)
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            def get_constant(
                c: Optional[_NestedTensors], idx: int
            ) -> Optional[_NestedTensors]:
                if c is None:
                    return None
                else:
                    assert isinstance(c, (list, tuple))
                    return c[idx]

            for idx, ee in enumerate(e):
                # Use an indexer here - if proxy is a List then it will unwrap
                # it. If it's a Proxy then it will proxy the getelem.
                wrap_with_proxy(ee, proxy[idx], get_constant(constant, idx))  # type: ignore[index]

        elif isinstance(e, dict):
            # example use case: triton_kernel_wrapper takes arguments as kwargs

            # In theory we could support const-prop when proxy-tensor-tracing
            # operators that returns dicts of tensors, but we have no use case
            # for it today (since the only op we currently trace that can
            # return a dict is triton_kernel_wrapper_functional/mutation,
            # which does not participate in const-prop)
            assert constant is None

            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            for key, val in e.items():
                wrap_with_proxy(val, proxy[key], None)  # type: ignore[index]

        elif isinstance(e, BackwardState):
            assert isinstance(proxy, Proxy)
            set_meta(proxy, e)
            e.proxy = proxy
        else:
            # intentionally pass on primitives
            pass

    wrap_with_proxy(inner_res, proxy_res, constant)

    return inner_res


@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Optional[Tensor]


def fetch_sym_proxy(
    tracer: _ProxyTracer,
) -> Callable[[PySymType], Union[bool, int, float, Proxy]]:
    def inner(e: PySymType) -> Union[int, bool, float, Proxy]:
        n = e.node
        if n.constant is not None:
            return n.constant
        if e.node.expr.is_number:
            if isinstance(e, SymBool):
                return bool(e.node.expr)
            elif isinstance(e, SymInt):
                return int(e.node.expr)
            return float(e.node.expr)
        else:
            assert isinstance(e, py_sym_types)
            # NB: we REQUIRE all symints to be tracked
            return get_proxy_slot(e, tracer).force()

    return inner


@overload
def fetch_object_proxy(tracer: _ProxyTracer, t: Tensor) -> Union[_ProxyTensor, Tensor]:
    ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: _AnyScriptObjectType
) -> Union[Proxy, _AnyScriptObjectType]:
    ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: PySymType
) -> Union[_PySymProxyType, PySymType]:
    ...


def fetch_object_proxy(
    tracer: _ProxyTracer, t: Union[Tensor, _AnyScriptObjectType, PySymType]
) -> object:
    return get_proxy_slot(t, tracer, t)


HANDLED_TYPES = (Tensor, torch.nn.Parameter, FakeTensor)


def _maybe_record_pointwise_barrier(
    func: object, proxy_mode: ProxyTorchDispatchMode
) -> None:
    """
    Records pointwise operators in user program (non decomposed) that were output in fp16/bf16
    """
    if proxy_mode.decomp_layers or not proxy_mode.emulate_precision_casts:
        return

    if (
        not isinstance(func, torch._ops.OpOverload)
        or torch.Tag.pointwise not in func.tags
    ):
        return

    last_node = next(iter(reversed(proxy_mode.tracer.graph.nodes)))
    t = last_node.meta.get("val")
    if not isinstance(t, torch.Tensor) or t.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return

    last_node.meta["low_precision_pointwise_barrier"] = True


def proxy_call(
    proxy_mode: ProxyTorchDispatchMode,
    func: OpOverload,
    pre_dispatch: bool,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    unrecognized_types: List[Type] = []
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))

    def can_handle_tensor(x: Tensor) -> bool:
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) in (torch._subclasses.FakeTensor,)
        if not r:
            unrecognized_types.append(type(x))
        return r

    # If there are any tensor subclasses, we need to handle those tensor subclasses first
    # TODO: we could use types to test this
    if not all(can_handle_tensor(x) for x in flat_args_kwargs if isinstance(x, Tensor)):
        not_implemented_log.debug(
            "ProxyTensorMode tensors without proxy had unrecognized subclasses: %s",
            unrecognized_types,
        )
        return NotImplemented

    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        _maybe_record_pointwise_barrier(func, proxy_mode)
        return r

    # For pre-autograd tracing, we do not want to run CompositeImplicit decomps.
    if not pre_dispatch and func not in [
        torch.ops.aten.size.default,
        torch.ops.aten.stride.default,
        torch.ops.aten.storage_offset.default,
    ]:
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    if func is torch.ops.aten.is_nonzero.default:
        with proxy_mode:
            return (args[0] != 0).item()  # type: ignore[attr-defined]

    tracer = proxy_mode.tracer
    f_flat_args_kwargs = [
        (
            fetch_object_proxy(tracer, x)
            if isinstance(x, (Tensor, _AnyScriptObject))
            else x
        )
        for x in flat_args_kwargs
    ]

    # If there are SymInts, we also should not consider this constant.
    # However, fake tensor handling of SymInts is sufficiently broken that
    # I couldn't write a test for this case
    all_constant = (
        not any(
            t.constant is None
            for t in f_flat_args_kwargs
            if isinstance(t, _ProxyTensor)
        )
        # TODO: maybe constant SymInts should also be allowed?  Not sure if
        # this can happen
        and not any(isinstance(x, py_sym_types) for x in flat_args_kwargs)
    )

    if torch.Tag.data_dependent_output in func.tags:
        # Check if all of the Tensor inputs are constants
        if all_constant:
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            with unset_fake_temporarily():
                return func(*const_args, **const_kwargs)
        # If any of the Tensor inputs are "real" (not FakeTensor), we may
        # incorrectly burn in constants by allowing this access.  Raise
        # an error in this case
        if proxy_mode._error_on_data_dependent_ops and pytree.tree_all_only(
            Tensor, lambda t: not is_fake(t), (args, kwargs)
        ):
            raise RuntimeError(
                f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! "
                "It's likely that this is caused by data-dependent control flow or similar.  "
                "It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' "
                "in your make_fx call."
            )

    proxy_flat_args_kwargs = [
        e.proxy if isinstance(e, _ProxyTensor) else e for e in f_flat_args_kwargs
    ]
    proxy_flat_args_kwargs = [
        (fetch_sym_proxy(proxy_mode.tracer)(e) if isinstance(e, py_sym_types) else e)
        for e in proxy_flat_args_kwargs
    ]
    proxy_args, proxy_kwargs = pytree.tree_unflatten(proxy_flat_args_kwargs, spec)

    # When we trace through a torch.tensor invocation, you never actually
    # see a torch.ops.aten.tensor call. Instead, the way this function is
    # implemented internally is that we allocate a plain tensor (this is
    # *guaranteed* to be a plain tensor, we disable all modes when doing
    # so), and then call at::lift_fresh on it (to give modes a chance to do
    # their stuff).  Furthermore, the tensor argument to lift_fresh is guaranteed
    # to be freshly allocated, so we want lift_fresh to be a no-op (directly
    # returning the input argument).
    #
    # Here is the basic problem: when we trace this sequence of executions
    # into an FX graph, what happens to this call sequence?  Traditionally,
    # tensor constants get interned as buffers on the FX GraphModule.  But
    # this is dangerous.  Consider:
    #
    #       x = torch.tensor(1)
    #       x.add_(2)
    #
    # Naively, this traces into:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh(t)
    #       x.add_(2)
    #
    # If lift_fresh returns t directly, the subsequent add_ call will
    # modify the tensor constant. Really, the problem is we've violated
    # the invariant the argument to lift is fresh.  So what we should
    # preserve the invariant by replacing lift_fresh with lift_fresh_copy:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh_copy(t)
    #       x.add_(2)
    #
    # This is what the overload modification does.
    if func is torch.ops.aten.lift_fresh.default:
        func = torch.ops.aten.lift_fresh_copy.default

    proxy_out = proxy_mode.tracer.create_proxy(
        "call_function",
        func,
        proxy_args,
        proxy_kwargs,
        name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__),
    )

    with _enable_thunkify(proxy_mode.tracer):
        out = func(*args, **kwargs)

    # In some circumstances, we will be tracing in a situation where a tensor
    # is *statically* known to be a constant (currently, this only happens if
    # you run torch.tensor; deterministic factory functions like torch.arange
    # don't get this treatment).  When the tensor in question is small, it's
    # helpful to due constant propagation in case we call item() (in which
    # case we can return the constant value that is known, rather than give
    # an error.)  The logic here tests if constant propagation is possible
    # (because all of the inputs are constant).  If so, we disable fake tensor
    # mode (if it is on) and do true compute on the constant.
    #
    # It's worth highlighting that we're making a policy decision here.
    # There is a potential that the tensor is actually quite large, and we
    # don't actually want to run the compute.  The tensor being quite large
    # is one of the reasons why factory functions don't get this treatment
    # (since they can be quite large; if a parameter is initialized to a
    # constant value it will be!)  Similarly, there is also a potential
    # to run an operator that blows up the size of a small tensor; we don't
    # protect against this case, but we could force, e.g., only single
    # element constant computation by testing the numel of the result before
    # propagating const-ness.  Similarly, we don't require the constant to
    # live on CPU, but we could.
    any_constant = any(
        t.constant is not None
        for t in f_flat_args_kwargs
        if isinstance(t, _ProxyTensor)
    )

    constant = None

    def tensor_numel_in_limit(t: Tensor) -> bool:
        return t.numel() <= CONSTANT_NUMEL_LIMIT

    # If this is a lift, the input tensor is guaranteed to be a
    # constant, so we keep a copy of the original argument along so
    # we can query it if we're asked to item() it at some later point
    if (
        func is torch.ops.aten.lift_fresh_copy.default
        and out.numel() <= CONSTANT_NUMEL_LIMIT
    ):
        with unset_fake_temporarily():
            assert isinstance(args[0], (Proxy, Tensor)), type(args[0])
            constant = args[0].clone()
    elif (
        torch.Tag.nondeterministic_seeded not in func.tags
        and all_constant
        and any_constant
        and pytree.tree_all_only(Tensor, tensor_numel_in_limit, out)
    ):
        # NB: do NOT include factories as constants
        with unset_fake_temporarily():
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None

    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    _maybe_record_pointwise_barrier(func, proxy_mode)
    return out


class _SymNodeDict:
    """
    Wrapper around a dictionary that will hash SymInts with their nodes
    """

    def __init__(self) -> None:
        self.sym_node_dict: Dict[PySymType, _PySymProxyType] = {}

    def __setitem__(self, key: PySymType, value: _PySymProxyType) -> None:
        self.sym_node_dict[key.node] = value

    def __getitem__(self, key: PySymType) -> _PySymProxyType:
        return self.sym_node_dict[key.node]

    def __contains__(self, key: PySymType) -> bool:
        return key.node in self.sym_node_dict

    def get(
        self, key: PySymType, default: Optional[_PySymProxyType] = None
    ) -> _PySymProxyType:
        # dict.get()'s annotation doesn't accept `None` when the value type
        # isn't Optional.
        return self.sym_node_dict.get(key.node, default)  # type: ignore[arg-type]

    def __iter__(self) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sym_node_dict)


class PythonKeyTracer(Tracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: Dict[sympy.Symbol, object]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    torch_fn_counts: Dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self) -> None:
        super().__init__(autowrap_modules=())  # type: ignore[arg-type]
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.symnode_tracker = _SymNodeDict()
        self.script_object_tracker = WeakIdKeyDictionary(
            dict=None, ref_type=_WeakHashRef
        )
        self.sympy_expr_tracker = dict()

        # Stores the torch function that was called during tracing
        self.torch_fn_metadata = None
        # Stores the counts for every torch function called. This is to help
        # distinguish between different calls to the same torch function.
        self.torch_fn_counts = {}
        self.enable_thunkify = False

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
        self,
        m: Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        return forward(*args, **kwargs)

    # We don't want to turn getattr calls into proxies. So we just return the actual value.
    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: Dict[str, Proxy]
    ) -> object:
        return attr_val

    def create_arg(self, a: object) -> fx.node.Node:
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})

            qualname = self.get_fresh_qualname("_param_constant")
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})
        elif isinstance(a, py_sym_types):
            assert a.node.constant is not None
            return a.node.constant
        return super().create_arg(a)  # type: ignore[return-value]

    @overload
    def unwrap_proxy(self, e: Tensor) -> Union[Proxy, Tensor]:
        ...

    @overload
    def unwrap_proxy(self, e: PySymType) -> Union[Proxy, PySymType]:
        ...

    @overload
    def unwrap_proxy(
        self, e: _AnyScriptObjectType
    ) -> Union[Proxy, _AnyScriptObjectType]:
        ...

    def unwrap_proxy(self, e: T) -> object:
        if isinstance(e, Tensor):
            return get_proxy_slot(e, self, e, lambda x: x.proxy)
        elif isinstance(e, py_sym_types):
            return get_proxy_slot(e, self, e, lambda e: e.force())
        elif isinstance(e, _AnyScriptObject):
            return get_proxy_slot(e, self, e)
        else:
            return e


def _make_temp_remove_mode_context_manager(
    mode_ty: Type[TorchFunctionMode],
) -> Callable[[], _GeneratorContextManager[Optional[TorchFunctionMode]]]:
    @contextmanager
    def context_manager_fn() -> Generator[Optional[TorchFunctionMode], None, None]:
        from torch.overrides import _len_torch_function_stack, _pop_mode, _push_mode

        temp_elements = []
        removed_mode = None

        while _len_torch_function_stack() > 0:
            mode = _pop_mode()
            if isinstance(mode, mode_ty):
                removed_mode = mode
                break
            else:
                temp_elements.append(mode)

        for mode in reversed(temp_elements):
            _push_mode(mode)

        try:
            yield removed_mode

        finally:
            if removed_mode is not None:
                count = len(temp_elements)
                while count > 0:
                    mode = _pop_mode()
                    count -= 1

                temp_elements.append(removed_mode)

                for mode in reversed(temp_elements):
                    _push_mode(mode)

    return context_manager_fn


@torch._disable_dynamo
def dispatch_trace(
    root: Union[Module, Callable],
    tracer: Tracer,
    concrete_args: Optional[Tuple[Any, ...]] = None,
) -> GraphModule:
    graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]

    # NB: be careful not to DCE .item() calls
    def impure_pred(n: fx.Node) -> bool:
        from .symbolic_shapes import is_accessor_node

        # Always defer to the built-in notion of impure
        if n.is_impure():
            return True

        # Accessors always OK to DCE
        if is_accessor_node(n):
            return False

        # If the operator in question takes SymInt args to SymInt output,
        # we assume it's pure and OK to DCE
        if (
            isinstance(n.meta.get("val"), py_sym_types)
            and
            # NB: constant args ok
            all(
                isinstance(a.meta.get("val"), py_sym_types)
                for a in n.args
                if isinstance(a, fx.Node)
            )
        ):
            return False

        # No idea, just assume it's not OK
        return True

    graph.eliminate_dead_code(impure_pred)
    from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints

    dedupe_symints(graph)
    name = root.__class__.__name__ if isinstance(root, Module) else root.__name__
    return fx._lazy_graph_module._make_graph_module(tracer.root, graph, name)


def wrap_key(
    f: Callable[_P, R], tensors: _P.args, tracer: _ProxyTracer, pre_dispatch: bool
) -> Callable[_P, R]:
    flat_tensors, _tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies: _P.args, **_unused: _P.kwargs) -> R:
        flat_proxies, _proxies_spec = pytree.tree_flatten(proxies)
        assert len(flat_proxies) == len(flat_tensors)
        with disable_proxy_modes_tracing() as m:
            assert isinstance(m, ProxyTorchDispatchMode)
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)

        def get_tensor_proxy_slot(t: Tensor) -> Union[Tensor, Proxy]:
            return get_proxy_slot(t, tracer, t, lambda x: x.proxy)

        out = f(*tensors)  # type:ignore[call-arg]
        out = pytree.tree_map_only(Tensor, get_tensor_proxy_slot, out)
        out = pytree.tree_map_only(
            _AnyScriptObject, lambda t: get_proxy_slot(t, tracer, t, lambda x: x), out
        )

        def get_sym_proxy_slot(t: PySymType) -> Proxy:
            return get_proxy_slot(t, tracer).force()

        out = pytree.tree_map_only(py_sym_types, get_sym_proxy_slot, out)
        return out

    return wrapped


# TODO: Make downstream users of this work with OperatorBase
ORIGINAL_ATEN: Optional[object] = None


@contextmanager
def set_original_aten_op(func: OpOverload) -> Generator[None, None, None]:
    global ORIGINAL_ATEN
    if ORIGINAL_ATEN is None and fx_traceback.has_preserved_node_meta():
        ORIGINAL_ATEN = func
        fx_traceback.current_meta["original_aten"] = func
        try:
            yield
        finally:
            ORIGINAL_ATEN = None
            fx_traceback.current_meta["original_aten"] = None
    else:
        yield


class TorchFunctionMetadataMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None:
        self.tracer = tracer

    def __torch_function__(
        self,
        func: OpOverload,
        types: Tuple[torch._C._TensorMeta, ...],
        args: Tuple[object, ...] = (),
        kwargs: Optional[Dict[str, object]] = None,
    ) -> object:
        kwargs = kwargs or {}
        self.tracer.torch_fn_metadata = func
        self.tracer.torch_fn_counts[func] = self.tracer.torch_fn_counts.get(func, 0) + 1
        return func(*args, **kwargs)


_temp_remove_metadata_torch_function_mode = _make_temp_remove_mode_context_manager(
    TorchFunctionMetadataMode
)


# This mode is **only** used for pre_dispatch tracing.
# In particular, we need to make sure that autograd/autocast API's
# that do not desugar into dispatcher operators stay in the graph.
class PreDispatchTorchFunctionMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None:
        self.tracer = tracer
        # The input to torch.amp.autocast_mode._exit_autocast graph node should be the
        # enter_autocast node. So we have to save the enter autocast node here, and assign it
        # to the exit_autocast call_function node.
        self.enter_autocast_nodes: List[torch.fx.Node] = []

    def __torch_function__(
        self,
        func: Union[OpOverload, Callable],
        types: Tuple[torch._C._TensorMeta, ...],
        args: Tuple[object, ...] = (),
        kwargs: Optional[Dict[str, object]] = None,
    ) -> object:
        kwargs = kwargs or {}
        from torch.utils._python_dispatch import (
            is_traceable_wrapper_subclass,
        )
        import builtins
        if func in _side_effectful_need_to_be_preserved_pre_dispatch:
            # It's for passing the export verifier which needs to verify the meta['val']
            # TODO(tmanlaibaatar): we should systematically couple it with expoert verifier,
            # instead of hardcoding it here.
            # T203648563
            if func == torch.amp.autocast_mode._exit_autocast:
                enter_node = self.enter_autocast_nodes.pop()
                args = (enter_node,)
            node = self.tracer.create_node("call_function", func, args, {})  # type: ignore[arg-type]
            if func == torch.amp.autocast_mode._enter_autocast:
                self.enter_autocast_nodes.append(node)
            if func in [
                torch._C._set_grad_enabled,
                torch.amp.autocast_mode._enter_autocast,
                torch.amp.autocast_mode._exit_autocast,
            ]:
                node.meta["val"] = None
            return node
            # Don't actually run the function! We just want to trace the calls
            # into a graph. We don't actualy want to change global autograd state.
        
        if func == builtins.getattr:
            assert len(args) == 2 
            potential_tensor_subclass, attr = args
            if is_traceable_wrapper_subclass(potential_tensor_subclass):
                assert isinstance(potential_tensor_subclass, torch.Tensor)
                assert isinstance(attr, str)
                proxy = get_proxy_slot(potential_tensor_subclass, self.tracer).proxy
                inner_proxy = self.tracer.create_proxy("call_function", func, (proxy, attr), {})
                track_tensor_tree(getattr(potential_tensor_subclass, attr), inner_proxy, constant=None, tracer=self.tracer)
                return inner_proxy.node
        return func(*args, **kwargs)


_temp_remove_pre_dispatch_torch_function_mode = _make_temp_remove_mode_context_manager(
    PreDispatchTorchFunctionMode
)


class ProxyTorchDispatchMode(TorchDispatchMode):
    # Ensure this is read-only; this exists only for legacy reasons
    @property
    def enable_tracing(self) -> bool:
        return True

    def __init__(
        self,
        tracer: _ProxyTracer,
        tracing_mode: str,
        pre_dispatch: bool = False,
        _allow_fake_constant: bool = False,
        _error_on_data_dependent_ops: bool = True,
    ) -> None:
        dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None
        super().__init__(dk)
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.pre_dispatch = pre_dispatch
        self._allow_fake_constant = _allow_fake_constant
        self._error_on_data_dependent_ops = _error_on_data_dependent_ops
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.PROXY
        # Every time we enter a mode, we maintain a stack telling us what the previous
        # ProxyTorchDispatchMode state was (if there was any).
        # This lets us properly reset the state on exit.
        self.enter_stack: List[Optional[ProxyTorchDispatchMode]] = []
        self.decomp_layers = 0
        from torch._inductor import config

        self.emulate_precision_casts = config.emulate_precision_casts

    @count
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Tuple[torch._C._TensorMeta, ...],
        args: Tuple[object, ...] = (),
        kwargs: Optional[Dict[str, object]] = None,
    ) -> object:
        with set_original_aten_op(func):
            kwargs = kwargs or {}

            if func in (prim.device.default,):
                return func(*args, **kwargs)

            return proxy_call(self, func, self.pre_dispatch, args, kwargs)

    def __enter__(self) -> Self:
        # Stash and store the previous proxy mode (there may or may not be one)
        maybe_prev_proxy_mode = _unset_infra_mode(torch._C._TorchDispatchModeKey.PROXY)
        self.enter_stack.append(maybe_prev_proxy_mode)
        return super().__enter__()

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> Optional[bool]:
        b = super().__exit__(exc_type, exc_value, traceback)

        # Re-enable the previous proxy mode, if there was one.
        mb_previous_proxy_mode = self.enter_stack.pop()
        if mb_previous_proxy_mode is not None:
            _push_mode(mb_previous_proxy_mode)

        return b

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True

    def _compute_proxy(
        self, func: OpOverload, args: Tuple[object, ...], out: PySymType
    ) -> Proxy:
        # Handle torch.sym_sum
        n_args: Tuple[object, ...]
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            n_args = (
                tuple(
                    get_proxy_slot(a, self.tracer).force().node
                    if isinstance(a, py_sym_types)
                    else a
                    for a in args[0]
                ),
            )
        else:
            n_args = tuple(
                get_proxy_slot(a, self.tracer).force().node
                if isinstance(a, py_sym_types)
                else a
                for a in args
            )

        # func doesn't have a __torch_function__ that Proxy can interpose, so
        # we gotta do it manually
        n_out = self.tracer.create_node("call_function", func, n_args, {})  # type: ignore[arg-type]
        p_out = fx.Proxy(n_out, self.tracer)
        set_meta(p_out, out)
        return p_out

    def __sym_dispatch__(
        self,
        func: OpOverload,
        types: Tuple[torch._C._TensorMeta, ...],
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        # Peephole optimize multiply by one
        # NB: be careful not to trigger guards here!
        if func == operator.mul:
            if isinstance(args[1], int) and args[1] == 1:
                return args[0]
            elif isinstance(args[0], int) and args[0] == 1:
                return args[1]

        # For speed, we assume there are no nested data structures
        # (otherwise we could use tree_map)
        # We also assume there are no keyword arguments.
        assert not kwargs
        out = func(*args, **kwargs)

        # If func returned a constant, we don't need to trace; we have
        # determined that the result is constant (no matter if the inputs
        # were symbolic) and it is no longer necessary to trace the
        # computation.  This could occur if func triggered some guards.
        if isinstance(out, py_sym_types):
            p_out_thunk = thunkify(
                self.tracer, self._compute_proxy, func=func, args=args, out=out
            )
            set_proxy_slot(out, self.tracer, p_out_thunk)

        return out


class _GraphAppendingTracerEx(fx.proxy.GraphAppendingTracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: MutableMapping[PySymType, _PySymProxyType]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    sympy_expr_tracker: Dict[sympy.Symbol, object]
    torch_fn_metadata: Optional[OpOverload]
    torch_fn_counts: Dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self, graph: fx.graph.Graph) -> None:
        super().__init__(graph)
        self.symnode_tracker = weakref.WeakKeyDictionary()
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.sympy_expr_tracker = {}
        self.script_object_tracker = WeakIdKeyDictionary(
            dict=None, ref_type=_WeakHashRef
        )
        # Stores the torch function that was called during tracing
        self.torch_fn_metadata = None
        # Stores the counts for every torch function called. This is to help
        # distinguish between different calls to the same torch function.
        self.torch_fn_counts = {}


# TODO: I'm not sure what the point of this class is; you can just
# make_fx through a regular Interpreter
class DecompositionInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        new_graph: fx.Graph,
        decomposition_table: Optional[Mapping[OpOverload, Callable]] = None,
        **kwargs: object,
    ) -> None:
        super().__init__(module, **kwargs)  # type: ignore[arg-type]
        self.new_graph = new_graph
        self.tracer = _GraphAppendingTracerEx(self.new_graph)
        # Blegh
        self.decomposition_table = decomposition_table or {}
        self.mode = ProxyTorchDispatchMode(self.tracer, tracing_mode="real")

    def placeholder(
        self, target: str, args: Tuple[object, ...], kwargs: Dict[str, object]  # type: ignore[override]
    ) -> object:
        out = super().placeholder(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        # TODO handle case where the first character of target is '*'
        return out

    def get_attr(
        self, target: str, args: Tuple[object, ...], kwargs: Dict[str, object]  # type: ignore[override]
    ) -> object:
        out = super().get_attr(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    # call_function, call_method, call_module get traced automatically by the outer mode.

    def output(
        self, target: str, args: Tuple[object, ...], kwargs: Dict[str, object]  # type: ignore[override]
    ) -> object:
        out = super().output(target, args, kwargs)  # type: ignore[arg-type]

        def get_proxy_node(x: _ProxyTensor) -> fx.node.Node:
            return x.proxy.node

        def unwrap(e: Tensor) -> Union[Tensor, fx.Node]:
            return get_proxy_slot(e, self.tracer, e, get_proxy_node)

        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args: object, **kwargs: object) -> object:
        # Should enter the mode at least once for being able to restore it later
        # See: https://github.com/pytorch/pytorch/pull/82549#discussion_r934782025
        with decompose(self.decomposition_table), self.mode:
            return super().run(*args, **kwargs)  # type: ignore[arg-type]


def wrapper_and_args_for_make_fx(
    func: Callable[..., R], args: Tuple[object, ...], kwargs: Dict[str, object]
) -> Tuple[Callable[[List[object]], R], List[object]]:
    # make_fx doesn't support kwargs, so we need to do this flattening
    # and then unflatten the args before calling func
    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args: List[object]) -> R:
        fn_args, fn_kwargs = pytree.tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)

    return wrapped, flat_args


@contextmanager
def disable_autocast_cache() -> Generator[None, None, None]:
    old_value = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(old_value)


class _ModuleNotInstalledAsSubmoduleError(NameError):
    pass


# Base class for inline _ModuleStackTracer.__init__.AttrProxy
class _AttrProxy:
    def reset_proxy_mapping(self, base: Module, path: str) -> None:
        pass


class _ModuleStackTracer(PythonKeyTracer):
    r"""Customized version of PythonKeyTracer that retains module stack
    information in node.meta["nn_module_stack"].

    FX symbolic trace actually does this already, but it relies on `self.root`
    being the actual module being traced. Since make_fx traces a lambda of our
    creation, things don't work properly.

    So for this version we hold onto a reference to the original module
    (scope_root) and use that to match the path. Also when we see,
            A
           / \
          B   C
           \ /
            D
    we want to record the path as A.B.D by recording only one path.
    See Note [Preserving the nn module stack metadata during export non-strict mode]  # noqa: W605
    """

    def __init__(self, scope_root: GraphModule) -> None:
        super().__init__()
        self.scope_root = scope_root
        self.enable_attr_proxy = False
        self.submodule_paths = {}
        for name, m in self.scope_root.named_modules(remove_duplicate=False):
            if m in self.submodule_paths:
                self.enable_attr_proxy = True
            else:
                self.submodule_paths[m] = name

        self.proxy_paths: WeakKeyDictionary[_AttrProxy, str] = WeakKeyDictionary()
        self.attr_proxy_map: WeakKeyDictionary[Module, _AttrProxy] = WeakKeyDictionary()
        self.proxy_modules: WeakKeyDictionary[_AttrProxy, Module] = WeakKeyDictionary()
        self.counter = 0

        self.module_id_cache = defaultdict(list)
        for name, mod in self.scope_root.named_modules(remove_duplicate=False):
            self.module_id_cache[id(mod)].append(name)

        # Build a wrapper around _AttrProxy to provide the tracer. We can't
        # store it on _AttrProxy itself beceause we mimic the underlying class
        # (including its attributes).
        tracer = self

        class AttrProxy(_AttrProxy):
            def __init__(self, base: Module, path: str) -> None:
                # Class is modified to be a subclass of torch.nn.Module
                # Warning: We blow away our own attributes here to mimic the base class
                # - so don't expect `self.x` to do anything useful.
                self.__class__ = type(
                    base.__class__.__name__,
                    (self.__class__, base.__class__),
                    {},
                )
                self.__dict__ = base.__dict__
                self.__class__.__module__ = base.__class__.__module__
                self.__class__.__qualname__ = base.__class__.__qualname__
                self.reset_proxy_mapping(base, path)

            def reset_proxy_mapping(self, base: Module, path: str) -> None:
                tracer.proxy_paths[self] = path
                tracer.proxy_modules[self] = base

            def __getattr__(self, name: str) -> AttrProxy:
                assert isinstance(self, Module)
                # Calling into torch.nn.Module.__getattr__ with super(),
                # That __getattr__ is patched to be module_getattr_wrapper in _symbolic_trace.py.
                # which then calls into _ModuleStackTracer.getattr
                attr_val = super().__getattr__(name)  # type: ignore[misc]
                if isinstance(attr_val, AttrProxy):
                    attr_val = tracer.proxy_modules[attr_val]
                elif not isinstance(attr_val, Module):
                    return attr_val
                if attr_val not in tracer.attr_proxy_map:
                    tracer.attr_proxy_map[attr_val] = AttrProxy(
                        attr_val, tracer.proxy_paths[self] + "." + name
                    )
                else:
                    # NOTE [caching AttrProxy]. Caching ensures a 1-1 mapping between AttrProxy and the actual attr_val.
                    # 1. We reset the proxy_mapping to solve the diamond shape reference problem: we want to record the
                    # path as A.B.D instead of A.C.D (the purpose of _ModuleStackTracer).
                    # 2. Instead of creating a new AttrProxy, we just reset the proxy_mapping of existing one. This is to avoid
                    # dynamo creating multiple guards for the same attr_val but different AttrProxy when exporting
                    # a model that calls torch.compile (e.g when a model uses torch.cond.)
                    tracer.attr_proxy_map[attr_val].reset_proxy_mapping(
                        attr_val, tracer.proxy_paths[self] + "." + name
                    )
                return tracer.attr_proxy_map[attr_val]

            def get_base(self) -> Module:
                return tracer.proxy_modules[self]

            @property
            def _modules(self) -> Dict[str, AttrProxy]:
                assert "_modules" in self.__dict__
                submodules = self.__dict__["_modules"]
                assert isinstance(submodules, dict)
                return {
                    key: (
                        AttrProxy(value, tracer.proxy_paths[self] + "." + str(key))  # type: ignore[misc]
                        if value is not None
                        else value
                    )
                    for key, value in submodules.items()
                }

        self.proxy_type = AttrProxy

    def path_of_module(self, mod: Module) -> str:
        """
        Use tracked access path during tracing instead of the default BFS behavior.
        Still use all the possible module paths to verify the result.
        """
        if mod is self.scope_root:
            return ""

        if isinstance(mod, _AttrProxy):
            return self.proxy_paths[mod]

        try:
            return Tracer.path_of_module(self, mod)
        except NameError as e:
            raise _ModuleNotInstalledAsSubmoduleError from e

    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: Dict[str, Proxy]
    ) -> object:
        if (
            not isinstance(attr_val, Module)
            or isinstance(attr_val, fx.GraphModule)
            or not self.enable_attr_proxy
        ):
            return super().getattr(attr, attr_val, parameter_proxy_cache)
        if isinstance(attr_val, _AttrProxy):
            return attr_val

        # See NOTE [caching AttrProxy].
        if attr_val not in self.attr_proxy_map:
            self.attr_proxy_map[attr_val] = self.proxy_type(attr_val, attr)
        else:
            self.attr_proxy_map[attr_val].reset_proxy_mapping(attr_val, attr)
        return self.attr_proxy_map[attr_val]

    def trace(  # type: ignore[override]
        self, root: Union[Module, Callable], concrete_args: Optional[Dict[str, object]]
    ) -> fx.Graph:
        res = super().trace(root, concrete_args)

        # Since we are making _AttrProxy mimic the original
        # submodule, when someone registers a module directly
        # to the tracer while tracing, the proxy object gets registered
        # first. So we need to replace the proxy modules with the real ones
        # This can happen during HOO tracing
        proxy_module_names_to_be_replaced: List[Tuple[str, _AttrProxy]] = []
        for name, module in self.root.named_modules():
            if module in self.proxy_modules:
                proxy_module_names_to_be_replaced.append((name, module))

        def _delete_proxy_attr(obj: Module, target: str) -> bool:
            # Copied from fx/graph_module.py
            # Customized it for proxy type
            atoms = target.split(".")
            path, target_submod = atoms[:-1], atoms[-1]
            assert isinstance(obj, Module)
            mod = obj

            # Get the parent module
            for item in path:
                if not hasattr(mod, item):
                    return False

                mod = getattr(mod, item)

                if not isinstance(mod, (_AttrProxy, Module)):
                    return False

            if not hasattr(mod, target_submod):
                return False

            # At least the leaf module should be proxy type.
            if not isinstance(getattr(mod, target_submod), _AttrProxy):
                return False

            delattr(mod, target_submod)
            return True

        for proxy_module_name, proxy_module in proxy_module_names_to_be_replaced:
            _delete_proxy_attr(self.root, proxy_module_name)
            actual_module = self.proxy_modules[proxy_module]
            _assign_attr(actual_module, self.root, proxy_module_name)

        return res

    def call_module(
        self,
        m: Module,
        forward: Callable,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> None:
        """PythonKeyTracer overrides call_module to avoid the scope handling,
        but we actually want it.
        """
        from torch._dynamo import OptimizedModule

        # FIXME (tmanlaibaatar)
        # When we call torch.compile inside HOO, we will end up
        # invoking a module that is not registered on the root. For
        # now, we just inline them. But once we start supporting
        # mark_strict in export, we do need to properly handle this.
        # Right now, it doesn't matter because current non-strict
        # use cases don't need to work with HOO.
        if isinstance(m, (OptimizedModule, GraphModule)):
            return forward(*args, **kwargs)

        try:
            return Tracer.call_module(self, m, forward, args, kwargs)
        except _ModuleNotInstalledAsSubmoduleError:
            warnings.warn(
                f"Unable to find the path of the module {m}. "
                "This might be because the module was not properly registered "
                "as a submodule, which is not good practice. We will trace "
                "through the module without recording stack information."
            )
            return forward(*args, **kwargs)

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        return False

    def create_node(self, *args: object, **kwargs: object) -> fx.node.Node:
        """
        Create node and add on metadata.
        Add nn_module_stack here instead of TracerBase,
        since calls to make_fx() might not want to record module stack metadata.
        Add torch_fn by looking at torch_fn_metadata and torch_fn_counts.
        Add stack_trace by filtering out forward() stack frames.
        """
        node = super().create_node(*args, **kwargs)  # type: ignore[arg-type]

        # nn_module_stack
        if node.op not in ["placeholder", "output"]:
            if "nn_module_stack" not in node.meta:
                node.meta["nn_module_stack"] = self.module_stack
            # convert nn_module_stack from Dict[key, (FQN, class)] -> Dict[str, Tuple[str, str]]
            for key, (fqn, mod_cls) in node.meta["nn_module_stack"].items():
                if isinstance(mod_cls, type):
                    node.meta["nn_module_stack"][key] = (
                        fqn,
                        mod_cls.__module__ + "." + mod_cls.__qualname__,
                    )

        # torch_fn
        if (
            node.op == "call_function"
            and self.torch_fn_metadata is not None
            and "torch_fn" not in node.meta
        ):
            node.meta["torch_fn"] = (
                f"{self.torch_fn_metadata.__name__}_{self.torch_fn_counts[self.torch_fn_metadata]}",
                f"{self.torch_fn_metadata.__class__.__name__}.{self.torch_fn_metadata.__name__}",
            )

        # stack_trace
        if "stack_trace" not in node.meta and node.op not in ["placeholder", "output"]:
            user_frame_summary = CapturedTraceback.extract().summary()
            if user_frame_summary:
                # we retain frames from forward() calls, or ops
                # located in torch/__init__.py (e.g. sym_int, sym_constrain_range, vmap)
                stack_trace = [
                    frame
                    for frame in user_frame_summary
                    if (
                        frame.name == "forward"
                        or frame.filename.endswith("torch/__init__.py")
                    )
                ]
                # filter out forward() frames from fx/_symbolic_trace.py, export/_trace.py
                # this is hardcoded, but leads to a much cleaner stack trace
                stack_trace = [
                    frame
                    for frame in stack_trace
                    if not (
                        frame.filename.endswith("fx/_symbolic_trace.py")
                        or frame.filename.endswith("export/_trace.py")
                    )
                ]
                if (
                    stack_trace
                ):  # empty list for strict mode, dynamo should handle stack_trace
                    stack_trace = traceback.StackSummary.from_list(stack_trace)
                    node.meta["stack_trace"] = "".join(stack_trace.format()).strip()

        return node


class _MakefxTracer:
    def __init__(
        self,
        decomposition_table: Optional[Mapping[OpOverload, Callable]],
        tracing_mode: str,
        _allow_non_fake_inputs: bool,
        pre_dispatch: bool,
        record_module_stack: bool,
        _allow_fake_constant: bool,
        _error_on_data_dependent_ops: bool,
    ) -> None:
        # Configurations that are used to initialize the context managers and their states.
        # Should not modify them during tracing.
        self.decomposition_table: Dict[OpOverload, Callable] = dict(
            decomposition_table or {}
        )
        self.decomposition_table.setdefault(
            torch.ops.aten.sym_numel.default, torch._decomp.decompositions.sym_numel
        )
        self.tracing_mode: str = tracing_mode
        self._allow_non_fake_inputs: bool = _allow_non_fake_inputs
        self.pre_dispatch: bool = pre_dispatch
        self.record_module_stack: bool = record_module_stack
        self._allow_fake_constant: bool = _allow_fake_constant
        self._error_on_data_dependent_ops: bool = _error_on_data_dependent_ops

        # All context managers and their states should be initialized before tracing based on the inputs
        # and configurations. After tracing, their states should be cleaned except for shape_env.
        # Remember to specify how to intialize it from user inputs and from parent tracer whenever
        # adding new modes in _MakefxTracer.
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self.proxy_mode: Union[nullcontext, ProxyTorchDispatchMode] = nullcontext()
        self.proxy_function_mode: Union[
            nullcontext, PreDispatchTorchFunctionMode
        ] = nullcontext()
        self.fx_tracer: Optional[PythonKeyTracer] = None
        self.python_dispatcher_mode: Union[nullcontext, Any] = nullcontext()
        self.torch_fn_metadata_mode: Union[
            nullcontext, TorchFunctionMetadataMode
        ] = nullcontext()

    def _checkpoint_modes(self) -> List[Any]:
        return [
            self.fake_tensor_mode,
            self.proxy_mode,
            self.proxy_function_mode,
            self.fx_tracer,
            self.python_dispatcher_mode,
            self.torch_fn_metadata_mode,
        ]

    def _restore_modes(
        self,
        prev_fake_tensor_mode: Optional[FakeTensorMode],
        prev_proxy_mode: Union[nullcontext, ProxyTorchDispatchMode],
        prev_proxy_function_mode: Union[nullcontext, PreDispatchTorchFunctionMode],
        prev_fx_tracer: Optional[PythonKeyTracer],
        prev_python_dispatcher_mode: Union[nullcontext, Any],
        prev_torch_fn_metadata_mode: Union[nullcontext, TorchFunctionMetadataMode],
    ) -> None:
        self.fake_tensor_mode = prev_fake_tensor_mode
        self.proxy_mode = prev_proxy_mode
        self.proxy_function_mode = prev_proxy_function_mode
        self.fx_tracer = prev_fx_tracer
        self.python_dispatcher_mode = prev_python_dispatcher_mode
        self.torch_fn_metadata_mode = prev_torch_fn_metadata_mode

    @contextmanager
    def _init_modes_from_inputs(
        self, f: Callable, args: Tuple[object, ...]
    ) -> Generator[None, None, None]:
        prev_modes = self._checkpoint_modes()
        try:
            # Avoid importing sympy at a module level
            from .symbolic_shapes import ShapeEnv

            if hasattr(f, "_orig_mod") and self.record_module_stack:
                scope_root = f._orig_mod
                self.fx_tracer = _ModuleStackTracer(scope_root)
            else:
                self.fx_tracer = PythonKeyTracer()

            if self.tracing_mode == "fake":
                import torch._dynamo

                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    import torch._functorch.config as _config

                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=True,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=ShapeEnv(),
                            static_shapes=True,
                        )
                self.fake_tensor_mode = fake_tensor_mode
            elif self.tracing_mode == "symbolic":
                import torch._dynamo

                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    shape_env = ShapeEnv()
                    import torch._functorch.config as _config

                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=False,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=shape_env,
                        )
                assert (
                    fake_tensor_mode.shape_env is not None
                ), "shape_env should be set if tracing with 'symbolic'"
                self.fake_tensor_mode = fake_tensor_mode
            else:
                if not self.tracing_mode == "real":
                    raise AssertionError(
                        f"Unexpected tracing type: {self.tracing_mode}"
                    )

            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)

    def _construct_modes_with_fx_tracer(self, fx_tracer: _ProxyTracer) -> None:
        self.proxy_mode = ProxyTorchDispatchMode(
            fx_tracer,
            self.tracing_mode,
            pre_dispatch=self.pre_dispatch,
            _allow_fake_constant=self._allow_fake_constant,
            _error_on_data_dependent_ops=self._error_on_data_dependent_ops,
        )

        if self.pre_dispatch:
            self.proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        if self.tracing_mode == "symbolic" or self.pre_dispatch:
            self.python_dispatcher_mode = enable_python_dispatcher()

        self.torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)

    @contextmanager
    def _init_modes_from_parent(
        self, parent_tracer: _MakefxTracer
    ) -> Generator[None, None, None]:
        # By default, subtracer creates new modes based on parent tracer's config.
        # However, there are cases where we want to share the same modes with parent tracer
        # For example, fake_tensor_mode, we want the example value's fake_mode of parent graph and subgraphs to be the same.
        prev_modes = self._checkpoint_modes()
        try:
            self.fake_tensor_mode = parent_tracer.fake_tensor_mode

            def _create_sub_fx_tracer(parent_tracer: _ProxyTracer) -> PythonKeyTracer:
                if type(parent_tracer) == PythonKeyTracer:
                    return PythonKeyTracer()
                elif type(parent_tracer) == _ModuleStackTracer:
                    return _ModuleStackTracer(parent_tracer.scope_root)
                else:
                    raise RuntimeError(
                        f"Unexpected tracer type: {type(parent_tracer)}."
                    )

            assert parent_tracer.fx_tracer is not None
            self.fx_tracer = _create_sub_fx_tracer(parent_tracer.fx_tracer)
            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)

    def _trace_inner(self, f: Callable, *args: object) -> GraphModule:
        # TODO: We need to explicitly import torch._dynamo before calling dispatch_trace,
        # because dispatch_trace will introduce the lazy import of torch._dynamo,
        # and some contexts set before calling dispatch_trace will cause problems with the import of torch._dynamo,
        # such as some torch API(torch.ones and so on) in populate_builtin_to_tensor_fn_map() will be affected
        # by the context set before dispatch_trace.
        import torch._dynamo

        phs = pytree.tree_map(lambda _: torch.fx._symbolic_trace.PH, args)

        def _wrap_fake(args: T) -> T:
            arg_count = 0

            def inner_wrap_fake(x: object) -> object:
                nonlocal arg_count
                # TODO: it would be nice to line these up with the names
                # FX will choose for the placeholders, but we don't
                # actually know what the names will be at this point yet
                # NB: the Source here is actually meaningless
                from torch._dynamo.source import ConstantSource

                assert self.fake_tensor_mode is not None
                source = ConstantSource(f"input{arg_count}")
                if isinstance(x, Tensor):
                    arg_count += 1
                    return self.fake_tensor_mode.from_tensor(x, source=source)
                # NB: don't match on bools
                elif type(x) is int and self.tracing_mode == "symbolic":
                    assert (
                        self.fake_tensor_mode.shape_env is not None
                    ), "shape_env should be set if tracing with 'symbolic'"
                    return self.fake_tensor_mode.shape_env.create_symintnode(
                        self.fake_tensor_mode.shape_env.create_symbol(
                            x, source, positive=None
                        ),
                        hint=x,
                        source=source,
                    )
                elif isinstance(x, torch.ScriptObject):
                    return torch._library.fake_class_registry.maybe_to_fake_obj(
                        self.fake_tensor_mode, x
                    )

                assert not isinstance(
                    x, FakeScriptObject
                ), f"ScriptObject {x} has been fakified. Cannot wrap_fake it again."
                return x

            wrap_fn_map = {
                "real": lambda x: x,
                "fake": inner_wrap_fake,
                "symbolic": inner_wrap_fake,
            }
            return pytree.tree_map(wrap_fn_map[self.tracing_mode], args)

        def _wrap_func(f: Callable[_P, R], phs: Sequence[PHBase]) -> Callable[_P, R]:
            if (
                not hasattr(inspect.unwrap(f), "__code__")
                or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS
            ):
                # FX doesn't support varargs, so we gotta fake up a wrapper
                # TODO: Would be nice to fix this at the source...
                return fake_signature(f, len(phs))
            return f

        args = _wrap_fake(args)
        func = _wrap_func(f, phs)
        # We disable the autocast cache as the autocast cache causes type conversions on parameters to
        # check a cache, which introduces untracked tensors into the graph
        #
        # We also disable tracing by any other tensor proxy-based tracers except the current. The
        # purpose of `make_fx` is to produce graphmodules as a side effect; its internal execution is
        # thus irrelevant to any external functional trace.
        proxy_mode: ProxyTorchDispatchMode = typing.cast(
            ProxyTorchDispatchMode, self.proxy_mode
        )
        with ExitStack() as stack:
            stack.enter_context(decompose(self.decomposition_table))
            if self.fake_tensor_mode:
                stack.enter_context(self.fake_tensor_mode)
            stack.enter_context(self.python_dispatcher_mode)
            stack.enter_context(self.proxy_function_mode)
            stack.enter_context(self.torch_fn_metadata_mode)
            stack.enter_context(proxy_mode)
            stack.enter_context(disable_autocast_cache())
            stack.enter_context(_set_make_fx_tracer(self))

            assert self.fx_tracer is not None
            try:
                t = dispatch_trace(
                    wrap_key(func, args, self.fx_tracer, self.pre_dispatch),
                    tracer=self.fx_tracer,
                    concrete_args=tuple(phs),
                )
            except Exception:
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "make_fx_fail_partial",
                        "encoding": "string",
                    },
                    payload_fn=lambda: self.fx_tracer.graph.python_code(  # type: ignore[union-attr]
                        root_module="self",
                        verbose=True,
                        include_stride=True,
                        include_device=True,
                    ).src,
                )
                raise

        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if self.tracing_mode == "symbolic":
            assert self.fake_tensor_mode is not None
            t.shape_env = self.fake_tensor_mode.shape_env
        return t

    def trace(self, f: Callable, *args: object) -> fx.GraphModule:
        with self._init_modes_from_inputs(f, args):
            return self._trace_inner(f, *args)

    def trace_subgraph(self, f: Callable, *args: object) -> GraphModule:
        # Create a new tracer based on parent's config
        sub_tracer = _MakefxTracer(
            self.decomposition_table,
            "real",
            self._allow_non_fake_inputs,
            self.pre_dispatch,
            self.record_module_stack,
            self._allow_fake_constant,
            self._error_on_data_dependent_ops,
        )
        with sub_tracer._init_modes_from_parent(self):
            return sub_tracer._trace_inner(f, *args)


_CURRENT_MAKE_FX_TRACER: Optional[_MakefxTracer] = None


@contextmanager
def _set_make_fx_tracer(tracer: _MakefxTracer) -> Generator[None, None, None]:
    global _CURRENT_MAKE_FX_TRACER
    prev_tracer = _CURRENT_MAKE_FX_TRACER
    try:
        _CURRENT_MAKE_FX_TRACER = tracer
        yield
    finally:
        _CURRENT_MAKE_FX_TRACER = prev_tracer


def make_fx(
    f: Callable,
    decomposition_table: Optional[Mapping[OpOverload, Callable]] = None,
    tracing_mode: str = "real",
    _allow_non_fake_inputs: bool = False,
    *,
    pre_dispatch: bool = False,
    record_module_stack: bool = False,
    _allow_fake_constant: bool = False,
    _error_on_data_dependent_ops: bool = True,
) -> Callable[..., GraphModule]:
    """
    Given a function f, return a new function which when executed with valid
    arguments to f, returns an FX GraphModule representing the set of operations that
    were executed during the course of execution.
    """

    assert tracing_mode in ["real", "fake", "symbolic"]

    make_fx_tracer = _MakefxTracer(
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops,
    )

    @functools.wraps(f)
    def wrapped(*args: object) -> GraphModule:
        return make_fx_tracer.trace(f, *args)

    return wrapped


def get_torch_dispatch_modes() -> List[TorchDispatchMode]:
    return torch.utils._python_dispatch._get_current_dispatch_mode_stack()


# TODO: this is a legacy name, there is only ever one proxy mode as it's an
# infra mode
def get_innermost_proxy_mode() -> Optional[ProxyTorchDispatchMode]:
    return get_proxy_mode()


def get_proxy_mode() -> Optional[ProxyTorchDispatchMode]:
    """
    Current the currently active proxy tracing mode, or None if
    we are not currently tracing.  This includes pre-dispatch proxy
    tracing.
    """
    pre_dispatch_mode = torch._ops._get_dispatch_mode_pre_dispatch(
        torch._C._TorchDispatchModeKey.PROXY
    )
    mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    assert (
        pre_dispatch_mode is None or mode is None
    ), f"pre_dispatch_mode={pre_dispatch_mode}, mode={mode}"
    return pre_dispatch_mode or mode


def handle_sym_dispatch(func: Callable[_P, R], args: _P.args, kwargs: _P.kwargs) -> R:
    """
    Call into the currently active proxy tracing mode to do a
    SymInt/SymFloat/SymBool dispatch trace on a function that operates on
    these arguments.
    """
    mode = get_proxy_mode()
    assert mode
    # Have to do it manually, because we're not doing the normal torch
    # dispatch machinery which disables it for us
    with disable_proxy_modes_tracing():
        # TODO: properly compute types
        types: List[Type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)  # type: ignore[arg-type, return-value]


@contextmanager
def disable_proxy_modes_tracing() -> Generator[ProxyTorchDispatchMode, None, None]:
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.PROXY)


def maybe_handle_decomp(
    proxy_mode: ProxyTorchDispatchMode,
    op: OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> object:
    from torch._inductor.compiler_bisector import CompilerBisector

    if op in CURRENT_DECOMPOSITION_TABLE:
        if CompilerBisector.disable_subsystem(
            "aot_eager_decomp_partition", "decomposition", lambda: repr(op)
        ):
            return NotImplemented

        with proxy_mode:
            proxy_mode.decomp_layers += 1
            out = CURRENT_DECOMPOSITION_TABLE[op](*args, **kwargs)
            proxy_mode.decomp_layers -= 1
            return out

    return NotImplemented


def get_isolated_graphmodule(
    func: Callable,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
    tracing_mode: str = "real",
    decomposition_table: Optional[Mapping[OpOverload, Callable]] = None,
) -> GraphModule:
    """A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
    wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)

    with disable_proxy_modes_tracing():
        gm = make_fx(
            wrapped, decomposition_table=decomposition_table, tracing_mode=tracing_mode
        )(all_args)
    return gm


def _set_unbacked_bindings(out: object, out_proxy: _NestedProxys) -> None:
    """A helper function for setting up unbacked_bindings on the destination FX graph."""
    from .symbolic_shapes import compute_unbacked_bindings

    # Can't use detect_fake_mode here,
    #
    # python test/distributed/_tensor/test_dtensor_compile.py -k
    # test_tp_compile_fullgraph_is_seq_parallel_False
    #
    # will fail.  Very strange, it probably isn't right for them to be using
    # two fake modes there...
    fake_mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    if fake_mode and fake_mode.shape_env:
        if symbol_to_path := compute_unbacked_bindings(fake_mode.shape_env, out):
            assert isinstance(out_proxy, Proxy), out_proxy
            out_proxy.node.meta["unbacked_bindings"] = symbol_to_path
