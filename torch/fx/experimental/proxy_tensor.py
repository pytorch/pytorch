# mypy: ignore-errors

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
from torch.fx.graph_module import _assign_attr
from weakref import WeakKeyDictionary
from collections import defaultdict
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, unset_fake_temporarily, is_fake
from torch._dispatch.python import enable_python_dispatcher
import torch.fx as fx
from torch.fx.node import _side_effectful_need_to_be_preserved_pre_dispatch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass
import weakref
import operator
import traceback
from torch.utils._stats import count
from torch.utils._traceback import CapturedTraceback
import logging
from torch._library.fake_class_registry import FakeScriptObject
import warnings

from torch.overrides import TorchFunctionMode

from torch.utils._python_dispatch import (
    TorchDispatchMode,
    _disable_infra_mode,
    _push_mode,
    _unset_infra_mode,
)

from ._backward_state import BackwardState
from .sym_node import SymNode
from ._sym_dispatch_mode import SymDispatchMode
from torch.fx import Proxy
import torch.fx.traceback as fx_traceback
from torch import SymInt, SymFloat, SymBool
from torch.utils.weak import WeakTensorKeyDictionary, WeakIdKeyDictionary, _WeakHashRef

__all__ = ["PythonKeyTracer", "dispatch_trace", "make_fx", "DecompositionInterpreter", "py_sym_types", "get_innermost_proxy_mode"]

aten = torch.ops.aten
prim = torch.ops.prim

log = logging.getLogger(__name__)
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

CURRENT_DECOMPOSITION_TABLE: Dict[torch._ops.OperatorBase, Callable] = {}

CONSTANT_NUMEL_LIMIT = 1

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
)
def fake_signature(fn, nargs):
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})

@contextmanager
def decompose(decomposition_table):
    global CURRENT_DECOMPOSITION_TABLE
    old_decomposition_table = CURRENT_DECOMPOSITION_TABLE
    CURRENT_DECOMPOSITION_TABLE = decomposition_table
    try:
        yield CURRENT_DECOMPOSITION_TABLE
    finally:
        CURRENT_DECOMPOSITION_TABLE = old_decomposition_table

# ensure we cannot collide with other properties
proxy_slot = object()
no_default = object()

py_sym_types = (SymInt, SymFloat, SymBool)

def is_sym_node(node):
    assert hasattr(node, 'meta'), "All nodes traced with proxy_tensor should have meta"
    return "val" in node.meta and isinstance(node.meta['val'], py_sym_types)

def set_proxy_slot(obj, tracer, proxy):
    if isinstance(obj, torch.Tensor):
        # We DO want to clobber proxies whenever we run an inplace operation
        # on a tensor, and it affects the metadata on the proxy.
        tracer.tensor_tracker[obj] = proxy
    elif isinstance(obj, (torch.ScriptObject, FakeScriptObject)):
        # We DO want to clobber proxies, with a similar rationale as for tensors.
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
            tracer.symnode_tracker[obj] = proxy

def has_proxy_slot(obj, tracer):
    assert isinstance(obj, (torch.Tensor, SymNode)), type(obj)
    return get_proxy_slot(obj, tracer, False, lambda _: True)

# the default argument is what to return if the slot is not set.
# the transform argument is handy if you need to extract a subfield from
# the successfully looked up result (but NOT the default.)
def get_proxy_slot(obj, tracer, default=no_default, transform=lambda x: x):
    if isinstance(obj, torch.Tensor):
        tracker = tracer.tensor_tracker
    elif isinstance(obj, (torch.ScriptObject, FakeScriptObject)):
        tracker = tracer.script_object_tracker
    else:
        assert isinstance(obj, py_sym_types), type(obj)
        tracker = tracer.symnode_tracker

    if obj not in tracker:
        if default is no_default:
            raise RuntimeError(f"{obj} is not tracked with proxy for {tracer}")
        return default
    return transform(tracker[obj])

def snapshot_fake(val):
    return val.detach()

def extract_val(val):
    if is_fake(val):
        return snapshot_fake(val)
    elif isinstance(val, py_sym_types):
        return val
    elif isinstance(val, (torch.ScriptObject, FakeScriptObject)):
        return val
    elif isinstance(val, BackwardState):
        return val
    elif isinstance(val, (list, tuple)):
        return val.__class__([extract_val(x) for x in val])
    elif isinstance(val, torch.Tensor):
        if not val.is_sparse:
            # NB: Kinda hacky, but we should try to get val as the metadata
            # everywhere
            # TODO: This doesn't properly track storages.  A more robust
            # approach would be to maintain a per-trace FakeTensorMode and
            # from_real_tensor to create fake values (don't forget to
            # snapshot_fake)
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            with fake_tensor_mode:
                return torch.empty_strided(val.shape, val.stride(), device=val.device, dtype=val.dtype)
        else:
            return None
    elif isinstance(val, (int, float, bool)):
        return val

# What invariants do we have for the 'val' set on the FX node?  It has accurate
# metadata... but only for metadata that exists "below" all other subsystems
# (most notably autograd, but also vmap, functorch transforms, etc).  This means
# you can get the dtype, shape, stride, storage, but you CANNOT get requires_grad,
# grad_fn, _base (_base actually may be set due to recursive call to
# ADInplaceOrView, but you shouldn't rely on it.)
def set_meta(proxy, val):
    proxy.node.meta['val'] = extract_val(val)

    # Best effort tensor_meta setting; prefer using val!
    if is_fake(val):
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
    elif isinstance(val, torch.Tensor) and not val.is_sparse:
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(val)
    return proxy

def thunkify(f, *args, **kwargs):
    """
    Delays computation of f until it's called again
    Also caches the result
    """
    return functools.lru_cache(1)(functools.partial(f, *args, **kwargs))

def track_tensor(tensor, proxy, *, constant, tracer):
    def try_set_proxy_slot(outer_s, proxy_callable, *args):
        assert callable(proxy_callable)
        if isinstance(outer_s, SymInt):
            set_proxy_slot(outer_s, tracer, thunkify(proxy_callable, outer_s, *args))
    # The basic idea is that we need to associate each tensor/SymInt
    # with a Proxy.  How do we setup this association?  We just store
    # the proxy on the proxy slot of the object, keyed on the tracer
    # (so that if we have multiple tracers at the same time, they
    # don't clobber each other.)
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(s, lambda x, i: set_meta(
            tracer.create_proxy('call_function', torch.ops.aten.sym_size.int, (proxy, i), {}), x), i)

    for i, s in enumerate(tensor.stride()):
        try_set_proxy_slot(s, lambda x, i: set_meta(
            tracer.create_proxy('call_function', torch.ops.aten.sym_stride.int, (proxy, i), {}), x), i)

    try_set_proxy_slot(tensor.numel(), lambda x: set_meta(
        tracer.create_proxy('call_function', torch.ops.aten.sym_numel.default, (proxy,), {}), x))
    try_set_proxy_slot(tensor.storage_offset(), lambda x: set_meta(
        tracer.create_proxy('call_function', torch.ops.aten.sym_storage_offset.default, (proxy,)), x))
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))

def track_tensor_tree(inner_res, proxy_res, *, constant, tracer):
    _set_unbacked_bindings(inner_res, proxy_res)

    def wrap_with_proxy(e, proxy, constant):
        if isinstance(e, torch.Tensor):
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            # NB: eagerly set meta here, so that the numbering is in order
            set_meta(proxy, e)
            set_proxy_slot(e, tracer, lambda: proxy)
        elif isinstance(e, (torch.ScriptObject, FakeScriptObject)):
            set_proxy_slot(e, tracer, proxy)
            set_meta(proxy, e)
        elif isinstance(e, (tuple, list)):
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            # example use case: allreduce_ returns ([tensor], work)
            for idx, ee in enumerate(e):
                wrap_with_proxy(ee, proxy[idx], get_constant(idx))
        elif isinstance(e, dict):
            # In theory we could support const-prop when proxy-tensor-tracing
            # operators that returns dicts of tensors, but we have no use case
            # for it today (since the only op we currently trace that can
            # return a dict is triton_kernel_wrapper_functional/mutation,
            # which does not participate in const-prop)
            assert constant is None

            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            # example use case: triton_kernel_wrapper takes arguments as kwargs
            for key, val in e.items():
                wrap_with_proxy(val, proxy[key], None)
        elif isinstance(e, BackwardState):
            set_meta(proxy, e)
            e.proxy = proxy
        else:
            # intentionally pass on primitives
            pass


    def get_constant(idx):
        if constant is None:
            return None
        else:
            return constant[idx]

    wrap_with_proxy(inner_res, proxy_res, constant)

    return inner_res


def maybe_disable_fake_tensor_mode():
    # TODO: figure out if this API generally makes sense and bake it into the
    # library
    return unset_fake_temporarily()


@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Optional[torch.Tensor]


def fetch_sym_proxy(tracer):
    def inner(e):
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
            # NB: we REQUIRE all symints to be tracked
            return get_proxy_slot(e, tracer)()
    return inner


def fetch_object_proxy(tracer):
    return lambda t: get_proxy_slot(t, tracer, t)

HANDLED_TYPES = (torch.Tensor, torch.nn.Parameter, FakeTensor)

def proxy_call(proxy_mode, func, pre_dispatch, args, kwargs):
    unrecognized_types = []
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))

    def can_handle_tensor(x):
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) in (torch._subclasses.FakeTensor,)
        if not r:
            unrecognized_types.append(type(x))
        return r

    # If there are any tensor subclasses, we need to handle those tensor subclasses first
    # TODO: we could use types to test this
    if not all(
        can_handle_tensor(x) for x in flat_args_kwargs if isinstance(x, torch.Tensor)
    ):
        not_implemented_log.debug(
            "ProxyTensorMode tensors without proxy had unrecognized subclasses: %s",
            unrecognized_types,
        )
        return NotImplemented

    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
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

    tracer = proxy_mode.tracer
    f_flat_args_kwargs = [
        (
            fetch_object_proxy(tracer)(x)
            if isinstance(x, (torch.Tensor, torch.ScriptObject, FakeScriptObject))
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
        and not any(
            isinstance(x, (SymInt, SymFloat, SymBool)) for x in flat_args_kwargs
        )
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
            with maybe_disable_fake_tensor_mode():
                return func(*const_args, **const_kwargs)
        # If any of the Tensor inputs are "real" (not FakeTensor), we may
        # incorrectly burn in constants by allowing this access.  Raise
        # an error in this case
        if proxy_mode._error_on_data_dependent_ops and pytree.tree_all_only(
            torch.Tensor, lambda t: not is_fake(t), (args, kwargs)
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
        (
            fetch_sym_proxy(proxy_mode.tracer)(e)
            if isinstance(e, (SymInt, SymFloat, SymBool))
            else e
        )
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

    # This makes DCE marginally less likely to DCE inplace operations.
    # It is not strictly necessary
    # Kind of a hacky way to test if an op is in-place or not
    if (
        func.overloadpacket.__name__[-1] == "_"
        and func.overloadpacket.__name__[0] != "_"
    ):
        if isinstance(args[0], List):
            # e.g., c10d::allreduce_ returns a list of tensors as the first element
            # in the output.
            for i, a in enumerate(args[0]):
                a.proxy = proxy_out[0][i]
        else:
            args[0].proxy = proxy_out

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

    # If this is a lift, the input tensor is guaranteed to be a
    # constant, so we keep a copy of the original argument along so
    # we can query it if we're asked to item() it at some later point
    if (
        func is torch.ops.aten.lift_fresh_copy.default
        and out.numel() <= CONSTANT_NUMEL_LIMIT
    ):
        with maybe_disable_fake_tensor_mode():
            constant = args[0].clone()
    elif (
        torch.Tag.nondeterministic_seeded not in func.tags
        and all_constant
        and any_constant
        and pytree.tree_all_only(
            torch.Tensor, lambda t: t.numel() <= CONSTANT_NUMEL_LIMIT, out
        )
    ):
        # NB: do NOT include factories as constants
        with maybe_disable_fake_tensor_mode():
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
    return out

class _SymNodeDict:
    """
    Wrapper around a dictionary that will hash SymInts with their nodes
    """
    def __init__(self):
        self.sym_node_dict = {}

    def __setitem__(self, key: py_sym_types, value: Any):
        self.sym_node_dict[key.node] = value

    def __getitem__(self, key: py_sym_types):
        return self.sym_node_dict[key.node]

    def __contains__(self, key: py_sym_types):
        return key.node in self.sym_node_dict

    def get(self, key: py_sym_types, default: Any = None):
        return self.sym_node_dict.get(key.node, default)

class PythonKeyTracer(Tracer):
    def __init__(self):
        super().__init__(autowrap_modules=())
        self.tensor_tracker = WeakTensorKeyDictionary()
        self.symnode_tracker = _SymNodeDict()  # type: ignore[var-annotated]
        self.script_object_tracker = WeakIdKeyDictionary(dict=None, ref_type=_WeakHashRef)

        # Stores the torch function that was called during tracing
        self.torch_fn_metadata = None
        # Stores the counts for every torch function called. This is to help
        # distinguish between different calls to the same torch function.
        self.torch_fn_counts = {}

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
            self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        return forward(*args, **kwargs)

    # We don't want to turn getattr calls into proxies. So we just return the actual value.
    def getattr(self, attr, attr_val, parameter_proxy_cache):
        return attr_val

    def create_arg(self, a: Any):
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            qualname: Optional[str] = None

            if not qualname:
                qualname = self.get_fresh_qualname("_param_constant")
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})
        elif isinstance(a, (SymInt, SymFloat, SymBool)):
            assert a.node.constant is not None
            return a.node.constant
        return super().create_arg(a)

    def unwrap_proxy(self, e):
        if isinstance(e, torch.Tensor):
            return get_proxy_slot(e, self, e, lambda e: e.proxy)
        elif isinstance(e, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return get_proxy_slot(e, self, e, lambda e: e())
        elif isinstance(e, (torch.ScriptObject, FakeScriptObject)):
            return get_proxy_slot(e, self, e)
        else:
            return e


@contextmanager
def _temp_remove_pre_dispatch_torch_function_mode():
    from torch.overrides import _len_torch_function_stack, _pop_mode, _push_mode
    temp_elements = []
    pre_dispatch_mode = None

    while _len_torch_function_stack() > 0:
        mode = _pop_mode()
        if isinstance(mode, PreDispatchTorchFunctionMode):
            pre_dispatch_mode = mode
            break
        else:
            temp_elements.append(mode)

    for mode in reversed(temp_elements):
        _push_mode(mode)

    try:
        yield

    finally:
        if pre_dispatch_mode is not None:
            count = len(temp_elements)
            while count > 0:
                mode = _pop_mode()
                count -= 1

            temp_elements.append(pre_dispatch_mode)

            for mode in reversed(temp_elements):
                _push_mode(mode)


@torch._disable_dynamo
def dispatch_trace(
        root: Union[torch.nn.Module, Callable],
        tracer: Tracer,
        concrete_args: Optional[Tuple[Any, ...]] = None,
) -> GraphModule:
    graph = tracer.trace(root, concrete_args)
    from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints
    dedupe_symints(graph)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return fx._lazy_graph_module._make_graph_module(tracer.root, graph, name)


def wrap_key(f, tensors, tracer, pre_dispatch: bool):
    flat_tensors, tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies):
        flat_proxies, proxies_spec = pytree.tree_flatten(proxies)
        assert len(flat_proxies) == len(flat_tensors)
        with disable_proxy_modes_tracing() as m:
            assert isinstance(m, ProxyTorchDispatchMode)
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)

        out = f(*tensors)
        out = pytree.tree_map_only(
            torch.Tensor,
            lambda t: get_proxy_slot(t, tracer, t, lambda x: x.proxy),
            out
        )
        out = pytree.tree_map_only(
            (torch.ScriptObject, FakeScriptObject),
            lambda t: get_proxy_slot(t, tracer, t, lambda x: x),
            out
        )
        out = pytree.tree_map_only(
            (SymInt, SymFloat, SymBool),
            lambda t: get_proxy_slot(t, tracer)(),
            out
        )
        return out

    return wrapped

ORIGINAL_ATEN = None
@contextmanager
def set_original_aten_op(func):
    global ORIGINAL_ATEN
    if ORIGINAL_ATEN is None and fx_traceback.has_preserved_node_meta():
        ORIGINAL_ATEN = func
        fx_traceback.current_meta['original_aten'] = func
        try:
            yield
        finally:
            ORIGINAL_ATEN = None
            fx_traceback.current_meta['original_aten'] = None
    else:
        yield


class TorchFunctionMetadataMode(TorchFunctionMode):

    def __init__(self, tracer):
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.tracer.torch_fn_metadata = func
        self.tracer.torch_fn_counts[func] = self.tracer.torch_fn_counts.get(func, 0) + 1
        return func(*args, **kwargs)


# This mode is **only** used for pre_dispatch tracing.
# In particular, we need to make sure that autograd/autocast API's
# that do not desugar into dispatcher operators stay in the graph.
class PreDispatchTorchFunctionMode(TorchFunctionMode):

    def __init__(self, tracer):
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _side_effectful_need_to_be_preserved_pre_dispatch:
            # It's for passing the export verifier which needs to verify the meta['val']
            # TODO(tmanlaibaatar): we should systematically couple it with expoert verifier,
            # instead of hardcoding it here.
            node = self.tracer.create_node("call_function", func, args, {})
            if func is torch._C._set_grad_enabled:
                node.meta['val'] = None
            return node
            # Don't actually run the function! We just want to trace the calls
            # into a graph. We don't actualy want to change global autograd state.
        return func(*args, **kwargs)


class ProxyTorchDispatchMode(TorchDispatchMode):
    def __init__(self, tracer, tracing_mode, pre_dispatch=False, _allow_fake_constant=False, _error_on_data_dependent_ops=True):
        dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None
        super().__init__(dk)
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.enable_tracing = True
        self.pre_dispatch = pre_dispatch
        self._allow_fake_constant = _allow_fake_constant
        self._error_on_data_dependent_ops = _error_on_data_dependent_ops
        self.sym_mode = ProxySymDispatchMode(tracer)
        self.trace_state = {}
        self._managers = []
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.PROXY
        # Every time we enter a mode, we maintain a stack telling us what the previous
        # ProxyTorchDispatchMode state was (if there was any).
        # This lets us properly reset the state on exit.
        self.enter_stack: List[Optional[ProxyTorchDispatchMode]] = []

    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        with self.sym_mode.enable(False), set_original_aten_op(func):
            return self.inner_torch_dispatch(func, types, args, kwargs)

    def __enter__(self):
        # sym mode first, then us...
        m = self.sym_mode.enable(True)
        self._managers.append(m)
        m.__enter__()
        # Stash and store the previous proxy mode (there may or may not be one)
        maybe_prev_proxy_mode = _unset_infra_mode(torch._C._TorchDispatchModeKey.PROXY)
        self.enter_stack.append(maybe_prev_proxy_mode)
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        m = self._managers.pop()
        # ...exit us first, then sym mode
        b = super().__exit__(exc_type, exc_value, traceback)

        # Re-enable the previous proxy mode, if there was one.
        mb_previous_proxy_mode = self.enter_stack.pop()
        if mb_previous_proxy_mode is not None:
            _push_mode(mb_previous_proxy_mode)

        if not b:
            return m.__exit__(exc_type, exc_value, traceback)
        else:
            return m.__exit__(None, None, None)


    def inner_torch_dispatch(self, func, types, args=(), kwargs=None):
        if not self.enable_tracing:
            return func(*args, **kwargs)

        if func in [prim.device.default]:
            return func(*args, **kwargs)

        return proxy_call(self, func, self.pre_dispatch, args, kwargs)


class ProxySymDispatchMode(SymDispatchMode):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer
        # When false, we don't trace operations.  If you do this, you MUST
        # call track_tensor/track_tensor_tree on all results of the operation
        # to ensure we can adequately track the results
        self.enable_tracing = True

    @contextmanager
    def enable(self, b):
        old = self.enable_tracing
        self.enable_tracing = b
        try:
            yield
        finally:
            self.enable_tracing = old

    def _compute_proxy(self, func, args, out: Union[SymInt, SymFloat, SymBool]):
        n_args = tuple(
            get_proxy_slot(a, self.tracer)().node if isinstance(a, py_sym_types) else a
            for a in args
        )

        # func doesn't have a __torch_function__ that Proxy can interpose, so
        # we gotta do it manually
        n_out = self.tracer.create_node("call_function", func, n_args, {})
        p_out = fx.Proxy(n_out, self.tracer)
        set_meta(p_out, out)
        return p_out

    def __sym_dispatch__(self, func, types, args, kwargs):
        if not self.enable_tracing:
            return func(*args, **kwargs)

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
            # Delays tracing out the proxies on this op until we actually need it
            p_out_thunk = thunkify(self._compute_proxy, func=func, args=args, out=out)
            set_proxy_slot(out, self.tracer, p_out_thunk)

        return out


# TODO: I'm not sure what the point of this class is; you can just
# make_fx through a regular Interpreter
class DecompositionInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule, new_graph: torch.fx.Graph, decomposition_table=None, **kwargs):
        super().__init__(module, **kwargs)
        self.new_graph = new_graph
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)
        # Blegh
        self.tracer.tensor_tracker = WeakTensorKeyDictionary()  # type: ignore[attr-defined]
        self.tracer.symnode_tracker = weakref.WeakKeyDictionary()  # type: ignore[attr-defined]
        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}
        self.mode = ProxyTorchDispatchMode(self.tracer, tracing_mode="real")

        # Stores the torch function that was called during tracing
        self.tracer.torch_fn_metadata = None
        # Stores the counts for every torch function called. This is to help
        # distinguish between different calls to the same torch function.
        self.tracer.torch_fn_counts = {}

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        # TODO handle case where the first character of target is '*'
        return out

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        proxy = torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    # call_function, call_method, call_module get traced automatically by the outer mode.

    def output(self, target, args, kwargs):
        out = super().output(target, args, kwargs)

        def unwrap(e):
            return get_proxy_slot(e, self.tracer, e, lambda x: x.proxy.node)
        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args, **kwargs):
        # Should enter the mode at least once for being able to restore it later
        # See: https://github.com/pytorch/pytorch/pull/82549#discussion_r934782025
        with decompose(self.decomposition_table), self.mode:
            return super().run(*args, **kwargs)


def wrapper_and_args_for_make_fx(func, args, kwargs):
    # make_fx doesn't support kwargs, so we need to do this flattening
    # and then unflatten the args before calling func
    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args):
        fn_args, fn_kwargs = pytree.tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)
    return wrapped, flat_args

@contextmanager
def disable_autocast_cache():
    old_value = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(old_value)


class _ModuleNotInstalledAsSubmoduleError(NameError):
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

    def __init__(self, scope_root):
        super().__init__()
        self.scope_root = scope_root
        self.proxy_paths = WeakKeyDictionary()
        self.proxy_modules = WeakKeyDictionary()
        self.counter = 0

        self.module_id_cache = defaultdict(list)
        for name, mod in self.scope_root.named_modules(remove_duplicate=False):
            self.module_id_cache[id(mod)].append(name)

        self_ = self

        class AttrProxy:
            def __init__(self, base, path):
                self.__class__ = type(
                    base.__class__.__name__,
                    (self.__class__, base.__class__),
                    {},
                )
                self.__dict__ = base.__dict__
                self.__class__.__module__ = base.__class__.__module__
                self.__class__.__qualname__ = base.__class__.__qualname__
                self_.proxy_paths[self] = path
                self_.proxy_modules[self] = base

            def __getattr__(self, name):
                assert isinstance(self, torch.nn.Module)
                attr_val = super().__getattr__(name)
                if isinstance(attr_val, AttrProxy):
                    attr_val = self_.proxy_modules[attr_val]
                elif not isinstance(attr_val, torch.nn.Module):
                    return attr_val
                return AttrProxy(attr_val, self_.proxy_paths[self] + "." + name)

            @property
            def _modules(self):
                assert "_modules" in self.__dict__
                submodules = self.__dict__["_modules"]
                assert isinstance(submodules, dict)
                return {
                    key: AttrProxy(value, self_.proxy_paths[self] + "." + str(key))
                    for key, value in submodules.items()
                }

        self.proxy_type = AttrProxy

    def path_of_module(self, mod: torch.nn.Module) -> str:
        """
        Use tracked access path during tracing instead of the default BFS behavior.
        Still use all the possible module paths to verify the result.
        """
        if mod is self.scope_root:
            return ""

        if isinstance(mod, self.proxy_type):
            return self.proxy_paths[mod]

        try:
            return Tracer.path_of_module(self, mod)
        except NameError as e:
            raise _ModuleNotInstalledAsSubmoduleError from e

    def getattr(self, attr, attr_val, parameter_proxy_cache):
        if not isinstance(attr_val, torch.nn.Module) or isinstance(attr_val, torch.fx.GraphModule):
            return super().getattr(attr, attr_val, parameter_proxy_cache)
        if isinstance(attr_val, self.proxy_type):
            return attr_val
        return self.proxy_type(attr_val, attr)

    def trace(self, root, concrete_args):
        res = super().trace(root, concrete_args)
        # Since we are making AttrProxy mimic the original
        # submodule, when someone registers a module directly
        # to the tracer while tracing, the proxy object gets registered
        # first. So we need to replace the proxy modules with the real ones
        # This can happen during HOO tracing
        proxy_module_names_to_be_replaced = []
        for name, module in self.root.named_modules():
            if module in self.proxy_modules:
                proxy_module_names_to_be_replaced.append((name, module))

        def _delete_proxy_attr(obj, target):
            # Copied from fx/graph_module.py
            # Customized it for proxy type
            atoms = target.split(".")
            path, target_submod = atoms[:-1], atoms[-1]
            assert isinstance(obj, torch.nn.Module)
            mod = obj

            # Get the parent module
            for item in path:

                if not hasattr(mod, item):
                    return False

                mod = getattr(mod, item)

                if not isinstance(mod, (self.proxy_type, torch.nn.Module)):
                    return False

            if not hasattr(mod, target_submod):
                return False

            # At least the leaf module should be proxy type.
            if not isinstance(getattr(mod, target_submod), self.proxy_type):
                return False

            delattr(mod, target_submod)
            return True

        for (proxy_module_name, proxy_module) in proxy_module_names_to_be_replaced:
            _delete_proxy_attr(self.root, proxy_module_name)
            actual_module = self.proxy_modules[proxy_module]
            _assign_attr(actual_module, self.root, proxy_module_name)

        return res


    def call_module(self, m, forward, args, kwargs):
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
        except _ModuleNotInstalledAsSubmoduleError as e:
            warnings.warn(
                f"Unable to find the path of the module {m}. "
                "This might be because the module was not properly registered "
                "as a submodule, which is not good practice. We will trace "
                "through the module without recording stack information."
            )
            return forward(*args, **kwargs)


    def is_leaf_module(self, m, module_qualified_name):
        return False

    def create_node(self, *args, **kwargs):
        '''
        Create node and add on metadata.
        Add nn_module_stack here instead of TracerBase,
        since calls to make_fx() might not want to record module stack metadata.
        Add torch_fn by looking at torch_fn_metadata and torch_fn_counts.
        Add stack_trace by filtering out forward() stack frames.
        '''
        node = super().create_node(*args, **kwargs)

        # nn_module_stack
        if node.op not in ["placeholder", "output"]:
            if "nn_module_stack" not in node.meta:
                node.meta["nn_module_stack"] = self.module_stack
            # convert nn_module_stack from Dict[key, (FQN, class)] -> Dict[str, Tuple[str, str]]
            for key, (fqn, mod_cls) in node.meta["nn_module_stack"].items():
                if isinstance(mod_cls, type):
                    node.meta["nn_module_stack"][key] = (fqn, mod_cls.__module__ + "." + mod_cls.__qualname__)

        # torch_fn
        if node.op == "call_function" and self.torch_fn_metadata is not None and "torch_fn" not in node.meta:
            node.meta["torch_fn"] = (
                f"{self.torch_fn_metadata.__name__}_{self.torch_fn_counts[self.torch_fn_metadata]}",
                f"{self.torch_fn_metadata.__class__.__name__}.{self.torch_fn_metadata.__name__}"
            )

        # stack_trace
        if 'stack_trace' not in node.meta and node.op not in ["placeholder", "output"]:
            user_frame_summary = CapturedTraceback.extract().summary()
            if user_frame_summary:
                # we retain frames from forward() calls, or ops
                # located in torch/__init__.py (e.g. sym_int, sym_constrain_range, vmap)
                stack_trace = [frame for frame in user_frame_summary if (
                    frame.name == 'forward'
                    or frame.filename.endswith('torch/__init__.py')
                )]
                # filter out forward() frames from fx/_symbolic_trace.py, export/_trace.py
                # this is hardcoded, but leads to a much cleaner stack trace
                stack_trace = [
                    frame for frame in stack_trace if not (
                        frame.filename.endswith('fx/_symbolic_trace.py')
                        or frame.filename.endswith('export/_trace.py')
                    )
                ]
                if stack_trace:  # empty list for strict mode, dynamo should handle stack_trace
                    stack_trace = traceback.StackSummary.from_list(stack_trace)
                    node.meta["stack_trace"] = ''.join(stack_trace.format()).strip()

        return node

class _MakefxTracer:

    def __init__(
        self,
        decomposition_table: Optional[Dict[Callable, Callable]],
        tracing_mode: str,
        _allow_non_fake_inputs: bool,
        pre_dispatch: bool,
        record_module_stack: bool,
        _allow_fake_constant: bool,
        _error_on_data_dependent_ops: bool
    ):
        # Configurations that are used to initialize the context managers and their states.
        # Should not modify them during tracing.
        self.decomposition_table: Dict[Callable, Callable] = decomposition_table or {}
        self.decomposition_table.setdefault(torch.ops.aten.sym_numel.default, torch._decomp.decompositions.sym_numel)
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
        self.fake_tensor_mode: Union[null_ctx_type, FakeTensorMode] = nullcontext()
        self.proxy_mode: Union[null_ctx_type, ProxyTorchDispatchMode] = nullcontext()
        self.proxy_function_mode: Union[null_ctx_type, PreDispatchTorchFunctionMode] = nullcontext()
        self.fx_tracer: Union[null_ctx_type, Tracer] = nullcontext()
        self.python_dispatcher_mode: Union[null_ctx_type, Any] = nullcontext()
        self.torch_fn_metadata_mode: Union[null_ctx_type, TorchFunctionMetadataMode] = nullcontext()

    def _checkpoint_modes(self) -> List[Any]:
        return [
            self.fake_tensor_mode,
            self.proxy_mode,
            self.proxy_function_mode,
            self.fx_tracer,
            self.python_dispatcher_mode,
            self.torch_fn_metadata_mode
        ]

    def _restore_modes(
        self,
        prev_fake_tensor_mode: Union[null_ctx_type, FakeTensorMode],
        prev_proxy_mode: Union[null_ctx_type, ProxyTorchDispatchMode],
        prev_proxy_function_mode: Union[null_ctx_type, PreDispatchTorchFunctionMode],
        prev_fx_tracer: Union[null_ctx_type, Tracer],
        prev_python_dispatcher_mode: Union[null_ctx_type, Any],
        prev_torch_fn_metadata_mode : Union[null_ctx_type, TorchFunctionMetadataMode],
    ) -> None:
        self.fake_tensor_mode = prev_fake_tensor_mode
        self.proxy_mode = prev_proxy_mode
        self.proxy_function_mode = prev_proxy_function_mode
        self.fx_tracer = prev_fx_tracer
        self.python_dispatcher_mode = prev_python_dispatcher_mode
        self.torch_fn_metadata_mode = prev_torch_fn_metadata_mode

    @contextmanager
    def _init_modes_from_inputs(self, f, args):
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
                            shape_env=shape_env)
                assert fake_tensor_mode.shape_env is not None, "shape_env should be set if tracing with 'symbolic'"
                self.fake_tensor_mode = fake_tensor_mode
            else:
                if not self.tracing_mode == "real":
                    raise AssertionError(f"Unexpected tracing type: {self.tracing_mode}")

            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)

    def _construct_modes_with_fx_tracer(self, fx_tracer):
        self.proxy_mode = ProxyTorchDispatchMode(
            fx_tracer,
            self.tracing_mode,
            pre_dispatch=self.pre_dispatch,
            _allow_fake_constant=self._allow_fake_constant,
            _error_on_data_dependent_ops=self._error_on_data_dependent_ops
        )

        if self.pre_dispatch:
            self.proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        if self.tracing_mode == "symbolic" or self.pre_dispatch:
            self.python_dispatcher_mode = enable_python_dispatcher()

        self.torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)

    @contextmanager
    def _init_modes_from_parent(self, parent_tracer):
        # By default, subtracer creates new modes based on parent tracer's config.
        # However, there are cases where we want to share the same modes with parent tracer
        # For example, fake_tensor_mode, we want the example value's fake_mode of parent graph and subgraphs to be the same.
        prev_modes = self._checkpoint_modes()
        try:
            self.fake_tensor_mode = parent_tracer.fake_tensor_mode

            def _create_sub_fx_tracer(parent_tracer):
                if type(parent_tracer) == PythonKeyTracer:
                    sub_tracer = PythonKeyTracer()
                elif type(parent_tracer) == _ModuleStackTracer:
                    sub_tracer = _ModuleStackTracer(parent_tracer.scope_root)
                else:
                    raise RuntimeError(f"Unexpected tracer type: {type(parent_tracer)}.")

                return sub_tracer

            self.fx_tracer = _create_sub_fx_tracer(parent_tracer.fx_tracer)
            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)


    def _trace_inner(self, f, *args):
        phs = pytree.tree_map(lambda _: fx.PH, args)  # type: ignore[attr-defined]

        def _wrap_fake(args: Tuple[Any]) -> Tuple[Any]:
            arg_count = 0

            def inner_wrap_fake(x):
                nonlocal arg_count
                # TODO: it would be nice to line these up with the names
                # FX will choose for the placeholders, but we don't
                # actually know what the names will be at this point yet
                # NB: the Source here is actually meaningless
                from torch._dynamo.source import ConstantSource
                source = ConstantSource(f"input{arg_count}")
                if isinstance(x, torch.Tensor):
                    arg_count += 1
                    return self.fake_tensor_mode.from_tensor(x, source=source)  # type: ignore[attr-defined]
                # NB: don't match on bools
                elif type(x) is int and self.tracing_mode == "symbolic":
                    return self.fake_tensor_mode.shape_env.create_symintnode(
                        self.fake_tensor_mode.shape_env.create_symbol(x, source, positive=None),
                        hint=x,
                        source=source
                    )
                elif isinstance(x, torch.ScriptObject):
                    return torch._library.fake_class_registry.to_fake_obj(self.fake_tensor_mode, x)

                assert not isinstance(x, FakeScriptObject), f"ScriptObject {x} has been fakified. Cannot wrap_fake it again."
                return x

            wrap_fn_map = {
                "real": lambda x: x,
                "fake": inner_wrap_fake,
                "symbolic": inner_wrap_fake,
            }
            return pytree.tree_map(wrap_fn_map[self.tracing_mode], args)

        def _wrap_func(f, phs):
            if not hasattr(inspect.unwrap(f), '__code__') or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS:
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
        with decompose(self.decomposition_table), self.fake_tensor_mode, self.python_dispatcher_mode, self.proxy_function_mode, \
             self.proxy_mode.sym_mode, self.torch_fn_metadata_mode, \
             self.proxy_mode, disable_autocast_cache(), _set_make_fx_tracer(self):
            t = dispatch_trace(
                wrap_key(func, args, self.fx_tracer, self.pre_dispatch),
                tracer=self.fx_tracer,
                concrete_args=tuple(phs)
            )

        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if self.tracing_mode == "symbolic":
            t.shape_env = self.fake_tensor_mode.shape_env  # type: ignore[assignment]
        return t

    def trace(self, f, *args) -> torch.fx.GraphModule:
        with self._init_modes_from_inputs(f, args):
            return self._trace_inner(f, *args)

    def trace_subgraph(self, f, *args):
        # Create a new tracer based on parent's config
        sub_tracer = _MakefxTracer(
            self.decomposition_table,
            self.tracing_mode,
            self._allow_non_fake_inputs,
            self.pre_dispatch,
            self.record_module_stack,
            self._allow_fake_constant,
            self._error_on_data_dependent_ops
        )
        with sub_tracer._init_modes_from_parent(self):
            return sub_tracer._trace_inner(f, *args)

_CURRENT_MAKE_FX_TRACER : Optional[_MakefxTracer] = None

@contextmanager
def _set_make_fx_tracer(tracer: _MakefxTracer) -> None:
    global _CURRENT_MAKE_FX_TRACER
    prev_tracer = _CURRENT_MAKE_FX_TRACER
    try:
        _CURRENT_MAKE_FX_TRACER = tracer
        yield
    finally:
        _CURRENT_MAKE_FX_TRACER = prev_tracer

def make_fx(
        f,
        decomposition_table=None,
        tracing_mode="real",
        _allow_non_fake_inputs=False,
        *,
        pre_dispatch=False,
        record_module_stack=False,
        _allow_fake_constant=False,
        _error_on_data_dependent_ops=True):

    assert tracing_mode in ["real", "fake", "symbolic"]


    make_fx_tracer = _MakefxTracer(
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops
    )

    @functools.wraps(f)
    def wrapped(*args):
        return make_fx_tracer.trace(f, *args)

    return wrapped

def get_torch_dispatch_modes():
    return torch.utils._python_dispatch._get_current_dispatch_mode_stack()


def get_innermost_proxy_mode():
    return torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)


@contextlib.contextmanager
def disable_proxy_modes_tracing():
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.PROXY)


def maybe_handle_decomp(proxy_mode, op, args, kwargs):
    if op in CURRENT_DECOMPOSITION_TABLE:
        with proxy_mode:
            return CURRENT_DECOMPOSITION_TABLE[op](*args, **kwargs)
    return NotImplemented


def get_isolated_graphmodule(func, args, kwargs, tracing_mode="real"):
    """A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
    wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)

    with disable_proxy_modes_tracing():
        gm = make_fx(wrapped, tracing_mode=tracing_mode)(all_args)
    return gm


def _set_unbacked_bindings(out, out_proxy):
    """A helper function for setting up unbacked_bindings on the destination FX graph."""
    from .symbolic_shapes import compute_unbacked_bindings

    # Can't use detect_fake_mode here,
    #
    # python test/distributed/_tensor/test_dtensor_compile.py -k
    # test_tp_compile_fullgraph_is_seq_parallel_False
    #
    # will fail.  Very strange, it probably isn't right for them to be using
    # two fake modes there...
    fake_mode = torch._C._get_dispatch_mode(
        torch._C._TorchDispatchModeKey.FAKE
    )
    if fake_mode and fake_mode.shape_env:
        if symbol_to_path := compute_unbacked_bindings(fake_mode.shape_env, out):
            out_proxy.node.meta["unbacked_bindings"] = symbol_to_path
