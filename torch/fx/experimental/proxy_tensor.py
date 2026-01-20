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
import threading
import typing
import typing_extensions
import weakref
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    Concatenate,
    Optional,
    overload,
    Protocol,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec, Self, TypeVarTuple, Unpack
from weakref import WeakKeyDictionary

import torch
import torch._ops
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import SymBool, SymInt, Tensor
from torch._dispatch.python import enable_python_dispatcher
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_type
from torch._logging import trace_structured
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_impls import fast_detach
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    get_plain_tensors,
    is_fake,
    unset_fake_temporarily,
)
from torch._subclasses.functional_tensor import FunctionalTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.graph_module import _assign_attr
from torch.fx.node import (
    _side_effectful_need_to_be_preserved_pre_dispatch,
    Argument,
    Target,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
    _disable_infra_mode,
    _push_mode,
    _unset_infra_mode,
    autograd_would_have_decomposed,
    TorchDispatchMode,
)
from torch.utils._stats import count
from torch.utils._thunk import Thunk
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary, WeakTensorKeyDictionary

from ._backward_state import BackwardState
from .sym_node import SymNode


if TYPE_CHECKING:
    import types
    from collections.abc import MutableMapping

    import sympy

    from torch._ops import OpOverload
    from torch.fx._symbolic_trace import PHBase
    from torch.types import BoolLikeType, FloatLikeType, IntLikeType

__all__ = [
    "PythonKeyTracer",
    "dispatch_trace",
    "make_fx",
    "DecompositionInterpreter",
    "selective_decompose",
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
_Ts = TypeVarTuple("_Ts")

null_ctx_type = type(nullcontext)
# We currently convert all SymInt to proxies before we use them.
# This could plausibly be handled at the Dynamo level.
pytree.register_pytree_node(
    torch.Size,
    lambda xs: (list(xs), None),
    lambda xs, _: tuple(xs),
    # pyrefly: ignore [bad-argument-type]
    flatten_with_keys_fn=lambda xs: (
        [(pytree.SequenceKey(i), x) for i, x in enumerate(xs)],
        None,
    ),
    serialized_type_name="torch.Size",
)
# Ideally unflattening should not lose info, but we unflatten
# torch.Size to tuple (see above). This is necessary because the
# torch.Size constructor only accepts ints whereas our infra often
# transforms them to non-ints, e.g. symint proxies. Anyway, losing
# such info can cause pytree mapping or spec matching to fail, so
# work around this problem using the following dict as needed.
_pytree_subclasses_that_lose_info = {torch.Size: tuple}


def fake_signature(fn: Callable[_P, R], nargs: int) -> Callable[_P, R]:
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})


@contextmanager
def decompose(
    decomposition_table: Optional[Mapping[OpOverload, Callable]],
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
    meta: dict[str, PySymType]


def is_sym_node(node: _HasMeta) -> bool:
    assert hasattr(node, "meta"), "All nodes traced with proxy_tensor should have meta"
    return "val" in node.meta and isinstance(node.meta["val"], py_sym_types)


@overload  # type: ignore[no-overload-impl]
def set_proxy_slot(obj: Tensor, tracer: _ProxyTracer, proxy: _ProxyTensor) -> None: ...


@overload
def set_proxy_slot(
    obj: _AnyScriptObjectType, tracer: _ProxyTracer, proxy: Proxy
) -> None: ...


@overload
def set_proxy_slot(
    obj: PySymType, tracer: _ProxyTracer, proxy: _PySymProxyType
) -> None: ...


class _DisableUpdateTensorTracker(threading.local):
    value: bool = False


_disable_update_tensor_tracker_tls = _DisableUpdateTensorTracker()


_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT: dict[int, torch.fx.Node] = {}


def _is_proxy_tensor_update_tensor_tracker_disabled() -> bool:
    """
    Returns current state of disabling update tensor tracker.
    """
    return _disable_update_tensor_tracker_tls.value


@contextmanager
def _proxy_tensor_disable_update_tensor_tracker() -> Generator[None, None, None]:
    """
    NOTE "Do not clobber inplace ops"
    By default tensor_tracker is updated every time.
    This leads to chaining every operation by the FakeTensor.
    For example for mutable ops if we have several consecutive mutable operations:

    def f(x, y, z):
        x.copy_(y)
        x.copy_(z)
        return x

    Default graph result:
    def f_graph(x, y, z)
        x_1 = x.copy_(y)
        x_2 = x_1.copy_(z)
        return x_2

    This chaining simplifies the fx passes and helps to prevent the reordering.
    But in some cases, we want those nodes to be disconnected.
    E.g. in case of splitting joint graph into forward and backward.
    If first inplace op happened in forward, second in backward,
    we want them after split to be properly placed.

    Enabling this context manager for copy_ will result in:
    def f_graph_2(x, y, z):
        x_1 = x.copy_(y)
        x_2 = x.copy_(z)
        return x

    Results of copy_ x1 and x2 will have empty users in the graph.
    The reason why this behavior is not enabled for all inplace ops is that
    some fx passes (e.g. fx quantization) rely on chaining inplace ops like add_
    in their fusions passes.
    We could revisit enabling this logic for all inplace ops in future.
    """
    orig_value = _disable_update_tensor_tracker_tls.value
    _disable_update_tensor_tracker_tls.value = True
    try:
        yield
    finally:
        _disable_update_tensor_tracker_tls.value = orig_value


def set_proxy_slot(  # type: ignore[no-redef]
    obj: Union[PySymType, _AnyScriptObjectType, Tensor],
    tracer: _ProxyTracer,
    proxy: object,
) -> None:
    log.debug("set_proxy_slot %s (%s) %s", obj, id(obj), proxy)
    if isinstance(obj, Tensor):
        # We DO want to clobber proxies whenever we run an inplace operation
        # on a tensor, and it affects the metadata on the proxy.
        assert isinstance(proxy, _ProxyTensor)
        # see NOTE [Do not clobber inplace ops]
        if (
            obj not in tracer.tensor_tracker
            or not _is_proxy_tensor_update_tensor_tracker_disabled()
        ):
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
            proxy = typing.cast(_PySymProxyType, proxy)
            tracer.symnode_tracker[obj] = proxy

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
                tracer.sympy_expr_tracker[obj.node.expr] = _SympyExprTrackerValue(
                    proxy, obj
                )


def has_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> bool:
    assert isinstance(obj, (Tensor, SymNode)), type(obj)

    return bool(get_proxy_slot(obj, tracer, False, lambda _: True))


_PySymProxyType = Thunk[Proxy]


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
) -> _ProxyTensor: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
) -> Union[_ProxyTensor, U]: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_ProxyTensor], R],
) -> Union[R, U]: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
) -> Proxy: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
) -> Union[Proxy, U]: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[Proxy], R],
) -> Union[R, U]: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
) -> _PySymProxyType: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: T,
) -> Union[T, _PySymProxyType]: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_PySymProxyType], R],
) -> Union[R, U]: ...


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

    # pyrefly: ignore [index-error]
    # pyrefly: ignore [no-matching-overload, bad-argument-type]
    value = tracker.get(obj)

    if value is None and isinstance(obj, py_sym_types):
        if obj.node.is_symbolic():
            # Last ditch - we found a SymInt (SymBool, etc) we don't know
            # about.
            if (tmp := tracer.sympy_expr_tracker.get(obj.node.expr)) is not None:
                value = tmp.proxy

            else:
                # Attempt to build it from first principles.
                _build_proxy_for_sym_expr(tracer, obj.node.expr, obj)
                # pyrefly: ignore [no-matching-overload]
                value = tracker.get(obj)

    if value is None:
        # We don't know this value - return the default.
        if isinstance(default, _NoDefault):
            raise RuntimeError(
                f"{obj} ({type(obj)}, {id(obj)})is not tracked with proxy for {tracer}"
            )
        return default

    res = transform(value)
    return res


# Recursively traverses tensor subclasses,
# returnining an (unordered) list of Proxy objects that are tracked
# for all inner tensors, given the current extant proxy mode.
# Returns an empty list if no proxy mode is active.
def _get_proxies(t: torch.Tensor) -> list[Proxy]:
    proxies = []
    mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    if mode is None:
        return proxies
    assert isinstance(mode, ProxyTorchDispatchMode)
    tracer = mode.tracer
    for t_inner in get_plain_tensors(t, out=[]):
        if isinstance(t_inner, FunctionalTensor):
            t_inner = torch._from_functional_tensor(t_inner.elem)
        if not isinstance(t_inner, torch.Tensor):
            continue
        proxy_tensor = get_proxy_slot(t_inner, tracer)
        if proxy_tensor is not None:
            proxies.append(proxy_tensor.proxy)
    return proxies


@functools.cache
def _sympy_handlers() -> dict[type[sympy.Expr], Callable[..., Any]]:
    """
    Returns a dict converting sympy functions to python operators
    (i.e. `sympy.Mul` -> `operator.mul`)
    """
    import torch.utils._sympy.interp

    handlers = {}
    for k, v in torch.utils._sympy.interp.handlers().items():
        op = getattr(operator, v, None)
        if op is not None:
            handlers[k] = op
    return handlers


def _build_proxy_for_sym_expr(
    tracer: _ProxyTracer, expr: sympy.Expr, out: PySymType | None = None
) -> IntLikeType | FloatLikeType | BoolLikeType | None:
    """
    Decompose `expr` and look for the pieces as inputs. If `out` is provided
    then that will be the resulting SymNode (and `out.expr` must be the same as
    `expr`).

    This function is used when the ProxyTorchDispatchMode sees a SymNode
    that it hasn't seen before to try to associate it with traced inputs.

    How can this happen?

    First thing to remember is that although sympy.Exprs are interned (so
    `sympy.Expr("s3*s4")` will always have the same `id` and will always compare
    equal) SymNode does not (so doing `SymNode("s3")*SymNode("s4")` twice in a
    row will give two unique SymNodes).

    - On way for this to happen is if we turn off tracing to compute an
      intermediate value and then USE that value with tracing turned on - for
      example if we turn off tracing to do some FakeTensor propagation to
      compute a size (dtensor does this) but then turn tracing back on and use
      that computed size.

    - Another way is if we compute a size in one graph and stash it somewhere
      hidden (such as in some meta-data) and later use it in a different graph
      (dtensor does this too). Since the size was computed in the first graph
      and it's not an official input to the second graph it's not tracked
      properly. This is often going to show up as it usually works in fullgraph
      but a graph break causes a failure.

    To handle this we decompose the sympy.Expr and look for the pieces as
    inputs. But there are problems with this approach:

    - We lose operation provanance: We end up figuring out where to get the
      inputs - but those may not actually be correct. If we have "s1" coming in
      from both tensor1 and tensor2 and we pick the wrong one we could end up
      keeping a tensor alive longer than intended.

    - There's no guarantee that those values are inputs to the graph: If we have
      "s1*s2" computed in a graph #1 and used in graph #2 there's no guarantee
      that the input that holds "s1" is actually an input on graph #2.

    - The decomposition isn't guaranteed to be the same: Sympy can "simplify"
      expressions so it's possible that our inputs are "s1*s2" and "s3" but we
      decompose it into "s1" and "s2*s3" - which wouldn't be found.

    Other ways we could handle this:

    - Don't: Just require that all inputs are tracked properly. This is the
      "correct" solution but harder because you need to track down each
      potential problem one by one and fix them. And when it fails it's a lot of
      work to figure out both why it's failing and the right way to fix it. This
      is complicated by the fact that a stashed value could be incorrect but
      work fine until we happen to get an graph break in the wrong place - so it
      may be a while before the bug is found. (Maybe we need a "dynamo abuse
      mode" where we run tests with as many graph breaks inserted as possible?)

    - Track SymNode ops separately from proxy tracing: Right now SymNode
      operations are tracked as part of the proxy tracing - so when we disable
      proxy tracing we also disable SymNode tracing. But we don't have to do
      that - we could instead always have SymNodes track where they came from
      and just use that when needed. This solves the problem of tracing being
      temporarily turned off but doesn't help if an input isn't present after a
      graph break.

    - Better decomposition: Right now the decomposition is pretty simple. We do
      have a sat-solver available to us so we could theoretically do a better
      job figuring out a "correct" decomposition. But that still relies on
      having the inputs available at all - which isn't a guarantee.
    """

    if (value := tracer.sympy_expr_tracker.get(expr)) is not None:
        assert not out
        return value.value

    if isinstance(expr, (int, float, bool)):
        return expr
    if expr.is_Integer:
        return int(expr)
    if expr.is_Float:
        return float(expr)

    args = []
    for arg in expr.args:
        if (arg_value := _build_proxy_for_sym_expr(tracer, arg)) is None:
            return None
        args.append(arg_value)
    args = tuple(args)

    func: OpOverload | None = _sympy_handlers().get(expr.func)  # type: ignore[assignment]
    if not func:
        # Handler not found
        return None

    if out is None:
        out = func(*args)
    else:
        _sym_register(tracer, func, args, out)
    return out


def snapshot_fake(val: Tensor, include_real: bool = False) -> Optional[Tensor]:
    # val.detach() will also eventually call fast_detach(),
    # but this saves us a full trip into __torch_dispatch__
    # (snapshot_fake is called a lot)
    if isinstance(val, FakeTensor):
        return fast_detach(val.fake_mode, val, include_real)
    else:
        return val.detach()


_ExtractValType = Optional[
    Union[
        PySymType,
        _AnyScriptObjectType,
        BackwardState,
        list["_ExtractValType"],
        tuple["_ExtractValType", ...],
        dict[str, "_ExtractValType"],
        Tensor,
        int,
        float,
        bool,
    ]
]


def extract_val(val: _ExtractValType, include_real: bool = False) -> _ExtractValType:
    if is_fake(val):
        return snapshot_fake(val, include_real=include_real)
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
            from torch._guards import detect_fake_mode

            fake_tensor_mode = detect_fake_mode(val)
            if not fake_tensor_mode:
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
    proxy.node.meta["val"] = extract_val(
        val, include_real=(proxy.node.op == "placeholder")
    )

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
                    # pyrefly: ignore [bad-return]
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
def fetch_object_proxy(
    tracer: _ProxyTracer, t: Tensor
) -> Union[_ProxyTensor, Tensor]: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: _AnyScriptObjectType
) -> Union[Proxy, _AnyScriptObjectType]: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: PySymType
) -> Union[_PySymProxyType, PySymType]: ...


def fetch_object_proxy(
    tracer: _ProxyTracer, t: Union[Tensor, _AnyScriptObjectType, PySymType]
) -> object:
    return get_proxy_slot(t, tracer, t)


HANDLED_TYPES = (Tensor, torch.nn.Parameter, FakeTensor)


def _maybe_record_pointwise_barrier(
    func: object, proxy_mode: ProxyTorchDispatchMode
) -> None:
    """
    Records operators whose tensor outputs or inputs are fp16/bf16 so downstream pointwise code can
    emulate eager's rounding behavior when emulate_precision_casts is enabled.
    """
    if proxy_mode.decomp_layers or not proxy_mode.emulate_precision_casts:
        return

    if not isinstance(func, torch._ops.OpOverload):
        return

    last_node = next(iter(reversed(proxy_mode.tracer.graph.nodes)))
    t = last_node.meta.get("val")
    low_pr_fp = (torch.bfloat16, torch.float16)

    output_low_precision = isinstance(t, torch.Tensor) and t.dtype in low_pr_fp

    if not output_low_precision:
        for input_node in last_node.all_input_nodes:
            val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
            if isinstance(val, torch.Tensor) and val.dtype in low_pr_fp:
                output_low_precision = True
                break

    if not output_low_precision:
        return

    last_node.meta["low_precision_pointwise_barrier"] = True


def _fetch_proxies_and_all_constant_flag(
    flat_args_kwargs: Union[list[object], tuple[object, ...]], tracer: _ProxyTracer
) -> tuple[list[object], tuple[object, ...], bool]:
    """
    Given flat arguments, fetch the proxies and whether they are all constants.
    This is later used in proxy_call or when someone is trying to stitch together
    graph node in tf or td modes.
    """
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

    proxy_flat_args_kwargs = [
        e.proxy if isinstance(e, _ProxyTensor) else e for e in f_flat_args_kwargs
    ]

    proxy_flat_args_kwargs = [
        (fetch_sym_proxy(tracer)(e) if isinstance(e, py_sym_types) else e)
        for e in proxy_flat_args_kwargs
    ]

    return f_flat_args_kwargs, tuple(proxy_flat_args_kwargs), all_constant


def proxy_call(
    proxy_mode: ProxyTorchDispatchMode,
    func: OpOverload,
    pre_dispatch: bool,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    unrecognized_types: list[type] = []
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))

    def can_handle_tensor(x: Tensor) -> bool:
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) is torch._subclasses.FakeTensor
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
    if (
        not pre_dispatch
        and func
        not in [
            torch.ops.aten.size.default,
            torch.ops.aten.stride.default,
            torch.ops.aten.storage_offset.default,
        ]
        and autograd_would_have_decomposed(func, flat_args_kwargs)
    ):
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    if func is torch.ops.aten.is_nonzero.default:
        with proxy_mode:
            torch._check(
                args[0].numel() == 1,  # type: ignore[attr-defined]
                lambda: "Boolean value of Tensor with more than one value is ambiguous",
            )
            return (args[0] != 0).item()  # type: ignore[attr-defined]

    tracer = proxy_mode.tracer
    f_flat_args_kwargs, proxy_flat_args_kwargs, all_constant = (
        _fetch_proxies_and_all_constant_flag(flat_args_kwargs, tracer)
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
        self.sym_node_dict: dict[PySymType, _PySymProxyType] = {}

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
        return self.sym_node_dict.get(key.node, default)  # type: ignore[arg-type, return-value]

    def __iter__(self) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sym_node_dict)


@dataclass
class _SympyExprTrackerValue:
    proxy: _PySymProxyType
    value: PySymType


class PythonKeyTracer(Tracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: dict[sympy.Symbol, _SympyExprTrackerValue]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self) -> None:
        super().__init__(autowrap_modules=())  # type: ignore[arg-type]
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.symnode_tracker = _SymNodeDict()
        self.script_object_tracker = WeakIdKeyDictionary(
            dict=None, ref_type=_WeakHashRef
        )
        self.sympy_expr_tracker = {}

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
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return forward(*args, **kwargs)

    # We don't want to turn getattr calls into proxies. So we just return the actual value.
    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
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
    def unwrap_proxy(self, e: Tensor) -> Union[Proxy, Tensor]: ...

    @overload
    def unwrap_proxy(self, e: PySymType) -> Union[Proxy, PySymType]: ...

    @overload
    def unwrap_proxy(
        self, e: _AnyScriptObjectType
    ) -> Union[Proxy, _AnyScriptObjectType]: ...

    def unwrap_proxy(self, e: T) -> object:
        if isinstance(e, Tensor):
            return get_proxy_slot(e, self, e, lambda x: x.proxy)  # type: ignore[attr-defined]
        elif isinstance(e, py_sym_types):
            return get_proxy_slot(e, self, e, lambda e: e.force())
        elif isinstance(e, _AnyScriptObject):
            return get_proxy_slot(e, self, e)
        else:
            return e

    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)  # type: ignore[arg-type]

        if node.op in ["placeholder", "output"] and "stack_trace" in node.meta:
            del node.meta["stack_trace"]

        if kind == "get_attr":
            assert isinstance(target, str)
            attr = getattr(self.root, target)
            if isinstance(attr, torch.Tensor):
                with disable_proxy_modes_tracing():
                    node.meta["val"] = extract_val(attr)

        def map_fn(v: Any) -> Optional[_ExtractValType]:
            if not isinstance(v, torch.fx.Node) or "val" not in v.meta:
                return None
            val = v.meta["val"]
            # other subclasses like FunctionalTensor error on `extract_val`
            # "Attempting to use FunctionalTensor on its own." just store FakeTensors for now
            if isinstance(val, torch.Tensor) and not isinstance(val, FakeTensor):
                return None
            return extract_val(v.meta["val"])

        if _should_save_eager_input_vals(target, (args, kwargs)):
            # NOTE "eager_input_vals"
            # We save the original (args, kwargs) FakeTensor values for nodes
            # that have exact stride requirements. This is useful downstream.
            # We use this information inside Inductor to ensure that inputs to
            # stride-sensitive operators have the correct strides.
            arg_inp, kwarg_inp = torch.fx.node.map_aggregate((args, kwargs), map_fn)  # type: ignore[misc, arg-type]
            node.meta["eager_input_vals"] = (arg_inp, kwarg_inp)

        return node


def _should_save_eager_input_vals(
    target: Any,
    args_kwargs: Optional[tuple[tuple[Argument, ...], dict[str, Argument]]] = None,
) -> bool:
    from torch._higher_order_ops.invoke_subgraph import InvokeSubgraphHOP

    if not callable(target):
        return False
    if isinstance(
        target,
        (
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperMutation,
            InvokeSubgraphHOP,
        ),
    ):
        return True
    if args_kwargs is not None and (
        target is torch.ops.higher_order.auto_functionalized
        or target is torch.ops.higher_order.auto_functionalized_v2
    ):
        args = args_kwargs[0]
        assert isinstance(
            args[0], (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        )
        return _should_save_eager_input_vals(args[0], None)
    if target is torch.ops.higher_order.with_effects:
        # TODO: inductor lowering for with_effects needs to be updated to propagate
        # the arg_kwarg_vals
        return False
    if isinstance(target, torch._ops.HigherOrderOperator):
        if pytree.tree_any(_should_save_eager_input_vals, args_kwargs):
            raise RuntimeError(
                f"NYI: The HOP {target} has an input that is an OpOverload that "
                f"needs exact strides. We probably need special logic to "
                f"propagate the FakeTensor vals. Please file an issue."
            )
    if isinstance(target, torch._ops.OpOverload):
        from torch._library.utils import get_layout_constraint_tag

        return get_layout_constraint_tag(target) == torch._C.Tag.needs_exact_strides
    return False


def _make_temp_remove_mode_context_manager(
    mode_ty: type[TorchFunctionMode],
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
    concrete_args: Optional[tuple[Any, ...]] = None,
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
    f: Callable[[Unpack[_Ts]], R],
    tensors: tuple[Unpack[_Ts]],
    tracer: _ProxyTracer,
    pre_dispatch: bool,
) -> Callable[_P, R]:
    flat_tensors, _tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies: _P.args, **_unused: _P.kwargs) -> R:
        nonlocal tensors

        flat_proxies, _proxies_spec = pytree.tree_flatten(proxies)
        assert len(flat_proxies) == len(flat_tensors)
        with disable_proxy_modes_tracing() as m:
            assert isinstance(m, ProxyTorchDispatchMode)
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)

        if getattr(tracer, "proxy_module_inputs", False):
            tensors = [  # type: ignore[assignment, var-annotated]
                p if isinstance(t, torch.nn.Module) else t
                for t, p in zip(tensors, proxies)  # type: ignore[arg-type]
            ]

        def get_tensor_proxy_slot(t: Tensor) -> Union[Tensor, Proxy]:
            return get_proxy_slot(t, tracer, t, lambda x: x.proxy)  # type: ignore[attr-defined]

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
def set_original_aten_op(
    func: OpOverload | torch._ops.HigherOrderOperator,
) -> Generator[None, None, None]:
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
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: Optional[dict[str, object]] = None,
    ) -> object:
        kwargs = kwargs or {}
        # pyrefly: ignore [bad-assignment]
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
        self.enter_autocast_nodes: list[torch.fx.Node] = []

    def __torch_function__(
        self,
        func: Union[OpOverload, Callable],
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: Optional[dict[str, object]] = None,
    ) -> object:
        kwargs = kwargs or {}
        if func in _side_effectful_need_to_be_preserved_pre_dispatch:
            # It's for passing the export verifier which needs to verify the meta['val']
            # TODO(tmanlaibaatar): we should systematically couple it with export verifier,
            # instead of hardcoding it here.
            # T203648563
            if func is torch.amp.autocast_mode._exit_autocast:
                enter_node = self.enter_autocast_nodes.pop()
                args = (enter_node,)
            node = self.tracer.create_node("call_function", func, args, {})  # type: ignore[arg-type]
            if func is torch.amp.autocast_mode._enter_autocast:
                self.enter_autocast_nodes.append(node)
            if func in [
                torch._C._set_grad_enabled,
                torch.amp.autocast_mode._enter_autocast,
                torch.amp.autocast_mode._exit_autocast,
            ]:
                node.meta["val"] = None
            # For autocast, the python APIs run so we don't have to run them again
            # here.
            if func is torch._C._set_grad_enabled:
                # pyrefly: ignore [bad-argument-type]
                func(*args, **kwargs)
            return node

        # We need more complicated handling here because the inputs
        # to these functions are sometimes tensors or symints where
        # we need to fetch the proxies properly.
        if func in [
            torch._functorch.predispatch._add_batch_dim,
            torch._functorch.predispatch._remove_batch_dim,
            torch._functorch.predispatch._vmap_increment_nesting,
            torch._functorch.predispatch._vmap_decrement_nesting,
            torch._functorch.vmap.lazy_load_decompositions,
        ]:
            _, proxies, _ = _fetch_proxies_and_all_constant_flag(args, self.tracer)
            out_proxy = self.tracer.create_proxy(
                "call_function",
                func,
                proxies,
                {},
            )
            res = func(*args, **kwargs)
            track_tensor_tree(res, out_proxy, constant=None, tracer=self.tracer)
            return res
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
        self.enter_stack: list[Optional[ProxyTorchDispatchMode]] = []
        self.decomp_layers: int = 0
        from torch._inductor import config

        self.emulate_precision_casts: bool = config.emulate_precision_casts

    @count
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: Optional[dict[str, object]] = None,
    ) -> object:
        with set_original_aten_op(func):
            kwargs = kwargs or {}

            if func == prim.device.default:
                return func(*args, **kwargs)

            return proxy_call(self, func, self.pre_dispatch, args, kwargs)

    def __enter__(self) -> Self:
        # Stash and store the previous proxy mode (there may or may not be one)
        maybe_prev_proxy_mode = _unset_infra_mode(torch._C._TorchDispatchModeKey.PROXY)
        self.enter_stack.append(maybe_prev_proxy_mode)
        return super().__enter__()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
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

    def __sym_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        # Peephole optimize multiply by one
        # NB: be careful not to trigger guards here!
        if func is operator.mul:
            if isinstance(args[1], int) and args[1] == 1:
                return args[0]
            elif isinstance(args[0], int) and args[0] == 1:
                return args[1]

        # For speed, we assume there are no nested data structures
        # (otherwise we could use tree_map)
        # We also assume there are no keyword arguments.
        assert not kwargs
        out = func(*args, **kwargs)
        _sym_register(self.tracer, func, args, out)
        return out


def _sym_register(
    tracer: _ProxyTracer, func: OpOverload, args: tuple[object, ...], out: object
) -> None:
    # If func returned a constant, we don't need to trace; we have
    # determined that the result is constant (no matter if the inputs
    # were symbolic) and it is no longer necessary to trace the
    # computation.  This could occur if func triggered some guards.
    if isinstance(out, py_sym_types):
        p_out_thunk = thunkify(
            tracer, _compute_proxy, tracer, func=func, args=args, out=out
        )
        set_proxy_slot(out, tracer, p_out_thunk)


def _compute_proxy(
    tracer: _ProxyTracer, func: OpOverload, args: tuple[object, ...], out: PySymType
) -> Proxy:
    # Handle torch.sym_sum
    n_args: tuple[object, ...]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        n_args = (
            tuple(
                (
                    get_proxy_slot(a, tracer).force().node
                    if isinstance(a, py_sym_types)
                    else a
                )
                for a in args[0]
            ),
        )
    else:
        n_args = tuple(
            (
                get_proxy_slot(a, tracer).force().node
                if isinstance(a, py_sym_types)
                else a
            )
            for a in args
        )

    # func doesn't have a __torch_function__ that Proxy can interpose, so
    # we gotta do it manually
    n_out = tracer.create_node("call_function", func, n_args, {})  # type: ignore[arg-type]
    p_out = fx.Proxy(n_out, tracer)
    set_meta(p_out, out)
    return p_out


class _GraphAppendingTracerEx(fx.proxy.GraphAppendingTracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: MutableMapping[PySymType, _PySymProxyType]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    sympy_expr_tracker: dict[sympy.Symbol, _SympyExprTrackerValue]
    torch_fn_metadata: Optional[OpOverload]
    torch_fn_counts: dict[OpOverload, int]
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

    # pyrefly: ignore [bad-override]
    def placeholder(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        out = super().placeholder(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        # TODO handle case where the first character of target is '*'
        return out

    # pyrefly: ignore [bad-override]
    def get_attr(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        out = super().get_attr(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    # call_function, call_method, call_module get traced automatically by the outer mode.

    # pyrefly: ignore [bad-override]
    def output(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
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


class _SelectiveDecomposeInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        should_decompose: Callable[[fx.Node], bool],
        decomposition_table: Mapping[OpOverload, Callable],
        **kwargs: object,
    ) -> None:
        """
        For all nodes in `module`, selectively decompose if is `should_decompose`,
        following the given `decomposition_table`.
        """
        super().__init__(module, **kwargs)  # type: ignore[arg-type]
        self.should_decompose = should_decompose
        self.decomposition_table = decomposition_table

    @staticmethod
    def recursive_wrap(
        gm: fx.GraphModule,
        should_decompose: Callable[[fx.Node], bool],
        decomposition_table: Mapping[OpOverload, Callable],
        **kwargs: object,
    ) -> _SelectiveDecomposeInterpreter:
        """
        Recursively wrap gm and its sub graph modules. Specifically, HOP takes
        sub graph module as args. We may not want to decompose all nodes within
        these sub graph modules. So we also need to wrap these sub graph modules.
        As a result:
        - if should_decompose(hop) is True, we decompose all nodes within the hop.
        - if should_decompose(hop) is False, we check each node within the hop
            and decide whether decompose or not.
        """
        for node in gm.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, HigherOrderOperator
            ):
                new_args = []
                for arg in node.args:
                    if isinstance(arg, fx.GraphModule):
                        new_arg = _SelectiveDecomposeInterpreter.recursive_wrap(
                            arg, should_decompose, decomposition_table, **kwargs
                        )
                    else:
                        new_arg = arg
                    new_args.append(new_arg)
                node.args = tuple(new_args)

        return _SelectiveDecomposeInterpreter(
            gm, should_decompose, decomposition_table, **kwargs
        )

    def run_node(self, n):
        if self.should_decompose(n):
            with decompose(self.decomposition_table):
                result = super().run_node(n)
        else:
            result = super().run_node(n)
        return result


def selective_decompose(
    joint_gm: fx.GraphModule,
    *args,
    decomposition,
    should_decompose,
    trace_joint_graph: bool,
) -> fx.GraphModule:
    """Retrace a joint graph module and selectively apply decomposition."""

    if trace_joint_graph:
        # the arg name, primals and tangents, are important.
        # make_fx keeps the name in the traced graph and partitioner later relies
        # on the name to partition joint graph correctly.
        def wrap_fn(primals: list[Any], tangents: list[Any]):
            return _SelectiveDecomposeInterpreter.recursive_wrap(
                joint_gm, should_decompose, decomposition
            ).run(*args)
    else:

        def wrap_fn(*args):
            return _SelectiveDecomposeInterpreter.recursive_wrap(
                joint_gm, should_decompose, decomposition
            ).run(*args)

    return make_fx(wrap_fn, decomposition_table={})(*args)


def wrapper_and_args_for_make_fx(
    func: Callable[..., R], args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[Callable[[list[object]], R], list[object]]:
    # make_fx doesn't support kwargs, so we need to do this flattening
    # and then unflatten the args before calling func
    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args: list[object]) -> R:
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
        self.record_stack_traces = True
        self._record_forward_stack_traces_only = True
        self.scope_root = scope_root
        self.enable_attr_proxy = False
        self.submodule_paths = {}
        for name, m in self.scope_root.named_modules(remove_duplicate=False):
            if m in self.submodule_paths:
                log.info(
                    "Shared module found between %s and %s, AttrProxy is enabled.",
                    self.submodule_paths[m],
                    name,
                )
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
            def __init__(self, base: Union[Module, _AttrProxy], path: str) -> None:
                if isinstance(base, _AttrProxy):
                    base = base.get_base()  # type: ignore[attr-defined]

                assert isinstance(base, Module)
                # Class is modified to be a subclass of torch.nn.Module
                # Warning: We blow away our own attributes here to mimic the base class
                # - so don't expect `self.x` to do anything useful.
                # pyrefly: ignore [no-matching-overload]

                self.__class__ = type(
                    base.__class__.__name__,
                    (self.__class__, base.__class__),
                    {},
                )
                self.__dict__ = base.__dict__
                self.__class__.__module__ = base.__class__.__module__
                self.__class__.__qualname__ = base.__class__.__qualname__

                # This overwrites any existing paths if `base` is an AttrProxy
                tracer.proxy_paths[self] = path
                tracer.proxy_modules[self] = base

            def __getattr__(self, name: str) -> AttrProxy:
                assert isinstance(self, Module)
                # Calling into torch.nn.Module.__getattr__ with super(),
                # That __getattr__ is patched to be module_getattr_wrapper in _symbolic_trace.py.
                # which then calls into _ModuleStackTracer.getattr
                attr_val = super().__getattr__(name)  # type: ignore[misc]
                if not isinstance(attr_val, Module):
                    return attr_val

                return AttrProxy(attr_val, tracer.proxy_paths[self] + "." + name)

            def get_base(self) -> Module:
                return tracer.proxy_modules[self]

            def __getitem__(self, idx: Union[int, slice]) -> AttrProxy:
                if isinstance(idx, slice):
                    if isinstance(self, torch.nn.Sequential):
                        # Copied from nn/modules/container.py
                        res = torch.nn.Sequential(
                            OrderedDict(list(self._modules.items())[idx])
                        )

                        return AttrProxy(res, f"{tracer.proxy_paths[self]}.{idx}")
                    elif isinstance(self, torch.nn.ModuleList):
                        # Copied from nn/modules/container.py
                        res = torch.nn.ModuleList(list(self._modules.values())[idx])

                        return AttrProxy(res, f"{tracer.proxy_paths[self]}.{idx}")

                return super().__getitem__(idx)  # type: ignore[misc]

            @property
            def _modules(self) -> dict[str, AttrProxy]:
                assert "_modules" in self.__dict__
                submodules = self.__dict__["_modules"]
                assert isinstance(submodules, dict)
                # pyrefly: ignore [bad-return]
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
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
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
        self, root: Union[Module, Callable], concrete_args: Optional[dict[str, object]]
    ) -> fx.Graph:
        res = super().trace(root, concrete_args)

        # NOTE [export non-strict fake tensor leak detection]
        # In non-strict export, we don't have dynamo's side effect
        # tracking logic which makes some cases hard to detect.
        # In general, our detecting strategy is:
        #  (1) We instrument fake tensor creation to log all the fake tensors created during export.
        #  (2) We dump the proxy to fake tensor map from make_fx tracer (_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT))
        #  (3) Filter out fake tensors that are logged during (1):
        #      (1) Associated with TrackedFake (input tracking thing in symbolic_shapes)
        #      (2) Associated with gm.meta
        #  (4) Do ID match with the proxies

        global _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT
        _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT.clear()

        for key, val in self.tensor_tracker.items():
            _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT[id(key)] = val.proxy.node

        # Since we are making _AttrProxy mimic the original
        # submodule, when someone registers a module directly
        # to the tracer while tracing, the proxy object gets registered
        # first. So we need to replace the proxy modules with the real ones
        # This can happen during HOO tracing
        proxy_module_names_to_be_replaced: list[tuple[str, _AttrProxy]] = []
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
            # pyrefly: ignore [bad-assignment]
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
        args: tuple[object, ...],
        kwargs: dict[str, object],
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
            log.debug(
                "Unable to find the path of the module %s. "
                "This might be because the module was not properly registered "
                "as a submodule, which is not good practice. We will trace "
                "through the module without recording stack information.",
                str(m),
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
            if node.meta.get("nn_module_stack") is None:
                node.meta["nn_module_stack"] = self.module_stack.copy()
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
        record_stack_traces: bool = False,
        parent_tracer: Optional[_MakefxTracer] = None,
        proxy_module_inputs: bool = False,
        _disable_torch_fn_metadata_mode: bool = False,
    ) -> None:
        # Configurations that are used to initialize the context managers and their states.
        # Should not modify them during tracing.
        self.decomposition_table: dict[OpOverload, Callable] = dict(
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
        # Remember to specify how to initialize it from user inputs and from parent tracer whenever
        # adding new modes in _MakefxTracer.
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self.proxy_mode: Union[nullcontext, ProxyTorchDispatchMode] = nullcontext()
        self.proxy_function_mode: Union[nullcontext, PreDispatchTorchFunctionMode] = (
            nullcontext()
        )
        self.fx_tracer: Optional[PythonKeyTracer] = None
        self.python_dispatcher_mode: Union[nullcontext, Any] = nullcontext()
        self.torch_fn_metadata_mode: Union[nullcontext, TorchFunctionMetadataMode] = (
            nullcontext()
        )
        self.record_stack_traces = record_stack_traces
        self.parent_tracer: Optional[_MakefxTracer] = parent_tracer
        self.proxy_module_inputs = proxy_module_inputs
        self._disable_torch_fn_metadata_mode = _disable_torch_fn_metadata_mode

    def _checkpoint_modes(self) -> list[Any]:
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
        self, f: Callable, args: tuple[object, ...]
    ) -> Generator[None, None, None]:
        prev_modes = self._checkpoint_modes()
        try:
            # Avoid importing sympy at a module level
            from .symbolic_shapes import ShapeEnv

            if hasattr(f, "_orig_mod") and self.record_module_stack:
                scope_root = f._orig_mod
                # _ModuleStackTracer always try to preserve stack trace
                # in forward functions
                self.fx_tracer = _ModuleStackTracer(scope_root)
            else:
                self.fx_tracer = PythonKeyTracer()
                self.fx_tracer.record_stack_traces = self.record_stack_traces
                if self.record_stack_traces:
                    self.fx_tracer._record_forward_stack_traces_only = True

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
                assert fake_tensor_mode.shape_env is not None, (
                    "shape_env should be set if tracing with 'symbolic'"
                )
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

        if not self._disable_torch_fn_metadata_mode:
            self.torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)
        fx_tracer.proxy_module_inputs = self.proxy_module_inputs  # type: ignore[union-attr]

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
                if type(parent_tracer) is PythonKeyTracer:
                    return PythonKeyTracer()
                elif type(parent_tracer) is _ModuleStackTracer:
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
                    assert self.fake_tensor_mode.shape_env is not None, (
                        "shape_env should be set if tracing with 'symbolic'"
                    )
                    return self.fake_tensor_mode.shape_env.create_symintnode(
                        self.fake_tensor_mode.shape_env.create_symbol(
                            x, source, positive=None
                        ),
                        hint=x,
                        source=source,
                    )
                elif isinstance(x, torch.ScriptObject) or is_opaque_type(type(x)):
                    return torch._library.fake_class_registry.maybe_to_fake_obj(
                        self.fake_tensor_mode, x
                    )

                assert not isinstance(x, FakeScriptObject), (
                    f"ScriptObject {x} has been fakified. Cannot wrap_fake it again."
                )
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

        if (
            self.is_hop_subgraph_tracer()
            and (fake_mode := torch._guards.detect_fake_mode(args))
            and fake_mode.shape_env is not None
        ):
            from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

            insert_deferred_runtime_asserts(t, fake_mode.shape_env, "reenter_make_fx")
            t.recompile()
        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if self.tracing_mode == "symbolic":
            assert self.fake_tensor_mode is not None
            t.shape_env = self.fake_tensor_mode.shape_env  # type: ignore[assignment]
        return t

    def trace(self, f: Callable, *args: object) -> fx.GraphModule:
        with self._init_modes_from_inputs(f, args):
            return self._trace_inner(f, *args)

    def is_hop_subgraph_tracer(self) -> bool:
        return self.parent_tracer is not None

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
            parent_tracer=self,
        )
        with sub_tracer._init_modes_from_parent(self):
            return sub_tracer._trace_inner(f, *args)

    def trace_subgraph_custom_decomp(
        self, f: Callable, decomp_table: Mapping[OpOverload, Callable], *args
    ) -> GraphModule:
        assert isinstance(decomp_table, Mapping)
        # Create a new tracer based on parent's config, but use a different decomposition table
        sub_tracer = _MakefxTracer(
            decomp_table,
            "real",
            self._allow_non_fake_inputs,
            self.pre_dispatch,
            self.record_module_stack,
            self._allow_fake_constant,
            self._error_on_data_dependent_ops,
            parent_tracer=self,
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
    record_stack_traces: bool = False,
    proxy_module_inputs: bool = False,
    _disable_torch_fn_metadata_mode: bool = False,
) -> Callable[..., GraphModule]:
    """
    Given a function f, return a new function which when executed with valid
    arguments to f, returns an FX GraphModule representing the set of operations that
    were executed during the course of execution.

    If record_stack_traces is True, the stack trace will be preserved on node.meta["stack_trace"]
    """

    assert tracing_mode in ["real", "fake", "symbolic"]

    from torch._inductor import config

    make_fx_tracer = _MakefxTracer(
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops,
        record_stack_traces=record_stack_traces
        or config.trace.provenance_tracking_level == 1,
        proxy_module_inputs=proxy_module_inputs,
        _disable_torch_fn_metadata_mode=_disable_torch_fn_metadata_mode,
    )

    @functools.wraps(f)
    def wrapped(*args: object) -> GraphModule:
        return make_fx_tracer.trace(f, *args)

    return wrapped


def get_torch_dispatch_modes() -> list[TorchDispatchMode]:
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
    assert pre_dispatch_mode is None or mode is None, (
        f"pre_dispatch_mode={pre_dispatch_mode}, mode={mode}"
    )
    return pre_dispatch_mode or mode


def handle_sym_dispatch(
    func: Callable[_P, R],
    args: _P.args,  # type: ignore[valid-type]  # not allowed to use _P.args here
    kwargs: _P.kwargs,  # type: ignore[valid-type]  # not allowed to use _P.kwargs here
) -> R:
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
        types: list[type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)  # type: ignore[arg-type, return-value]


@contextmanager
def disable_proxy_modes_tracing() -> Generator[ProxyTorchDispatchMode, None, None]:
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.PROXY)


def maybe_handle_decomp(
    proxy_mode: ProxyTorchDispatchMode,
    op: OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
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
    args: tuple[object, ...],
    kwargs: dict[str, object],
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
