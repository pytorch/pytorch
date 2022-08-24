# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import functools
from typing import Any, Dict, Optional, Tuple, Callable, Union
import torch
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
from torch._subclasses.fake_tensor import FakeTensorMode
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager, nullcontext
import inspect
from dataclasses import dataclass
import weakref

from torch.utils._python_dispatch import TorchDispatchMode, enable_torch_dispatch_mode
from torch._subclasses import FakeTensor
from .symbolic_shapes import ShapeEnv, SymDispatchMode, PySymInt
import torch.fx.experimental.symbolic_shapes as symbolic_shapes
from torch.fx import Proxy

__all__ = ["PythonKeyTracer", "dispatch_trace", "make_fx", "DecompositionInterpreter"]
aten = torch.ops.aten
prim = torch.ops.prim

CURRENT_DECOMPOSITION_TABLE: Dict[torch._ops.OpOverload, Callable] = {}

CONSTANT_NUMEL_LIMIT = 1


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

def set_proxy_slot(obj, tracer, proxy):
    d = obj.__dict__.setdefault(proxy_slot, weakref.WeakKeyDictionary())
    assert isinstance(d, weakref.WeakKeyDictionary)
    d[tracer] = proxy

def has_proxy_slot(obj, tracer):
    return get_proxy_slot(obj, tracer, False, lambda _: True)

# the default argument is what to return if the slot is not set.
# the transform argument is handy if you need to extract a subfield from
# the successfully looked up result (but NOT the default.)
def get_proxy_slot(obj, tracer, default=no_default, transform=lambda x: x):
    d = obj.__dict__.get(proxy_slot)
    if not d:
        if default is no_default:
            raise KeyError(f"{obj} is not tracked with proxy for {tracer}")
        return default
    assert isinstance(d, weakref.WeakKeyDictionary)
    if tracer not in d:
        if default is no_default:
            raise KeyError(f"{obj} is not tracked with proxy for {tracer}")
        else:
            return default
    return transform(d[tracer])


def get_proxy_slots(obj):
    return obj.__dict__.get(proxy_slot)


def track_tensor(tensor, proxy, *, constant, tracer):
    # The basic idea is that we need to associate each tensor/SymInt
    # with a Proxy.  How do we setup this association?  We just store
    # the proxy on the proxy slot of the object, keyed on the tracer
    # (so that if we have multiple tracers at the same time, they
    # don't clobber each other.)
    for i, s in enumerate(tensor.shape):
        if isinstance(s, SymInt):
            inner_s = s.get_pyobj()
            assert isinstance(inner_s, PySymInt)
            # TODO: improve naming
            # TODO: lazily insert this into the graph only on first
            # use?  Maybe complicated and DCE is a better idea
            set_proxy_slot(inner_s, tracer, torch.ops.aten.sym_size(proxy, i))

        # TODO: also do stride/numel
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))

def track_tensor_tree(inner_res, proxy_res, *, constant, tracer):
    def wrap_with_proxy(e, proxy, constant):
        if isinstance(e, torch.Tensor):
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            if not e.is_sparse:
                proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(e)

    def get_constant(idx):
        if constant is None:
            return None
        else:
            return constant[idx]

    # Unfortunately, tree_map cannot directly be used here. As the resulting
    # object may be a proxy that represents a tuple, we may need to
    # explicitly unwrap the proxy by simulating the flattening operations.
    if isinstance(inner_res, tuple) or isinstance(inner_res, list):
        for idx, e in enumerate(inner_res):
            wrap_with_proxy(e, proxy_res[idx], get_constant(idx))
    elif isinstance(inner_res, torch.Tensor):
        wrap_with_proxy(inner_res, proxy_res, constant)

    return inner_res


def maybe_disable_fake_tensor_mode():
    # TODO: figure out if this API generally makes sense and bake it into the
    # library
    mb_fake_mode = torch._C._get_torch_dispatch_mode()
    if isinstance(mb_fake_mode, FakeTensorMode):
        return enable_torch_dispatch_mode(mb_fake_mode.inner, replace=mb_fake_mode)
    else:
        return nullcontext()


@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Optional[torch.Tensor]


def fetch_symint_proxy(tracer):
    def inner(e):
        n = e.get_pyobj()
        if n.constant is not None:
            return n.constant
        else:
            # NB: we REQUIRE all symints to be tracked
            return get_proxy_slot(n, tracer)
    return inner


def fetch_tensor_proxy(tracer):
    return lambda t: get_proxy_slot(t, tracer, t)


def proxy_call(proxy_mode, func_overload, args, kwargs=None):
    if kwargs is None:
        kwargs = {}

    func = func_overload.overloadpacket
    if func_overload in CURRENT_DECOMPOSITION_TABLE:
        with proxy_mode.restore():
            r = CURRENT_DECOMPOSITION_TABLE[func_overload](*args, **kwargs)
            if r is not NotImplemented:
                return r

    # Some of these are not "real" aten ops and will fail if we
    # call _dispatch_has_kernel_for_dispatch_key on them.
    # This list is probably incomplete
    if func_overload not in [torch.ops.aten.size.default]:
        with proxy_mode.restore():
            r = func_overload.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    tracer = proxy_mode.tracer

    f_args, f_kwargs = pytree.tree_map_only(torch.Tensor, fetch_tensor_proxy(tracer), (args, kwargs))

    # If there are SymInts, we also should not consider this constant.
    # However, fake tensor handling of SymInts is sufficiently broken that
    # I couldn't write a test for this case
    all_constant = (
        pytree.tree_all_only(_ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs))
        # TODO: maybe constant SymInts should also be allowed?  Not sure if
        # this can happen
        and pytree.tree_all_only(SymInt, lambda _: False, (args, kwargs))
    )

    if torch.Tag.data_dependent_output in func_overload.tags:  # type: ignore[attr-defined]
        # Check if all of the Tensor inputs are constants
        if all_constant:
            const_args, const_kwargs = pytree.tree_map_only(
                _ProxyTensor, lambda t: t.constant, (f_args, f_kwargs)
            )
            with maybe_disable_fake_tensor_mode():
                return func_overload(*const_args, **const_kwargs)
        raise RuntimeError(
            "It appears that you're trying to get value out of a tracing tensor - erroring out! "
            "It's likely that this is caused by data-dependent control flow or similar."
        )

    proxy_args, proxy_kwargs = pytree.tree_map_only(
        SymInt,
        fetch_symint_proxy(proxy_mode.tracer),
        pytree.tree_map_only(_ProxyTensor, lambda e: e.proxy, (f_args, f_kwargs))
    )
    proxy_out = func_overload(*proxy_args, **proxy_kwargs)

    # Kind of a hacky way to test if an op is in-place or not
    if func.__name__[-1] == "_" and func.__name__[0] != "_":
        # This makes DCE marginally less likely to DCE inplace operations.
        # It is not strictly necessary
        args[0].proxy = proxy_out
        proxy_out.node.meta['tensor_meta'] = _extract_tensor_metadata(args[0])

    out = func_overload(*args, **kwargs)

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
    any_constant = pytree.tree_any_only(_ProxyTensor, lambda t: t.constant is not None, (f_args, f_kwargs))

    constant = None
    # NB: do NOT include factories as constants
    if (
        torch.Tag.nondeterministic_seeded not in func_overload.tags  # type: ignore[attr-defined]
        and all_constant
        and any_constant
        and pytree.tree_all_only(torch.Tensor, lambda t: t.numel() <= CONSTANT_NUMEL_LIMIT, out)
    ):
        with maybe_disable_fake_tensor_mode():
            const_args, const_kwargs = pytree.tree_map_only(
                _ProxyTensor, lambda t: t.constant, (f_args, f_kwargs)
            )
            constant = func_overload(*const_args, **const_kwargs)

    track_tensor_tree(out, proxy_out, constant=constant, tracer=tracer)
    return out


class PythonKeyTracer(Tracer):
    def __init__(self):
        super().__init__()

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
            self, m: torch.nn.Module, forward: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> Any:
        return forward(*args, **kwargs)

    def create_arg(self, a: Any):
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node('get_attr', n, (), {})
            qualname: Optional[str] = None

            if not qualname:
                i = 0
                while True:
                    qualname = f'_param_constant{i}'
                    if not hasattr(self.root, qualname):
                        break
                    i += 1
                setattr(self.root, qualname, a)

            return self.create_node('get_attr', qualname, (), {})
        elif isinstance(a, SymInt):
            assert a.get_pyobj().constant is not None
            return a.get_pyobj().constant
        return super().create_arg(a)


def dispatch_trace(
        root: Union[torch.nn.Module, Callable],
        tracer: Tracer,
        concrete_args: Optional[Tuple[Any, ...]] = None,
) -> GraphModule:
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)


def wrap_key(f, tensors, tracer):
    flat_tensors, tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies):
        flat_proxies, proxies_spec = pytree.tree_flatten(proxies)
        assert len(flat_proxies) == len(flat_tensors)
        track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)

        out = f(*tensors)
        return pytree.tree_map_only(
            torch.Tensor,
            lambda t: get_proxy_slot(t, tracer, t, lambda x: x.proxy),
            out
        )

    return wrapped


class ProxyTorchDispatchMode(TorchDispatchMode):
    def __init__(self, tracer):
        self.tracer = tracer
        self.enable_tracing = True
        self.sym_mode = ProxySymDispatchMode(tracer)
        self.trace_state = {}

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        with self.sym_mode.enable(False):
            return self.inner_torch_dispatch(func_overload, types, args, kwargs)

    @contextmanager
    def restore(self):
        with self.sym_mode.enable(True):
            with super().restore():
                yield

    def inner_torch_dispatch(self, func_overload, types, args=(), kwargs=None):
        if not self.enable_tracing:
            return func_overload(*args, **kwargs)

        if symbolic_shapes.is_symbolic_op(func_overload):
            with self.restore():
                return symbolic_shapes.handle_symbolic_op(func_overload, args, kwargs)

        func = func_overload.overloadpacket
        # We don't want to convert torch.tensor constants into tracing objects.
        if func_overload == aten.lift.default:
            return args[0]

        if func in [prim.device]:
            return func_overload(*args, **kwargs)

        if pytree.tree_any_only(
            torch.Tensor,
            lambda t: has_proxy_slot(t, self.tracer),
            (args, kwargs)
        ):
            out = proxy_call(self, func_overload, args, kwargs)
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
        # the invariant the the argument to lift is fresh.  So what we should
        # preserve the invariant by replacing lift_fresh with lift_fresh_copy:
        #
        #       t = self._tensor_constant0  # initialized to torch.tensor(1)
        #       x = torch.ops.aten.lift_fresh_copy(t)
        #       x.add_(2)
        #
        # This is what the overload modification does.
        else:
            flat_args = pytree.tree_flatten((args, kwargs))[0]
            handled_types = [torch.Tensor, _ProxyTensor, torch.nn.Parameter]

            # If there are any tensor subclasses, we need to handle those tensor subclasses first
            # TODO: we could use types to test this
            if any(isinstance(arg, torch.Tensor) and type(arg) not in handled_types for arg in flat_args):
                return NotImplemented

            if func_overload is torch.ops.aten.lift_fresh.default:
                func_overload = torch.ops.aten.lift_fresh_copy.default

            n_args, n_kwargs = pytree.tree_map_only(SymInt, fetch_symint_proxy(self.tracer), (args, kwargs))

            proxy_out = self.tracer.create_proxy('call_function', func_overload, n_args, n_kwargs,
                                                 name=self.tracer.graph._target_to_str(func.__name__))

            out = func_overload(*args, **kwargs)

            # If this is a lift, the input tensor is guaranteed to be a
            # constant, so we keep a copy of the original argument along so
            # we can query it if we're asked to item() it at some later point
            is_lift = func_overload is torch.ops.aten.lift_fresh_copy.default
            if is_lift and out.numel() <= CONSTANT_NUMEL_LIMIT:
                with maybe_disable_fake_tensor_mode():
                    constant = args[0].clone()
            else:
                constant = None
            track_tensor_tree(out, proxy_out, constant=constant, tracer=self.tracer)

        def assert_proxy_tensor(e):
            assert has_proxy_slot(e, self.tracer), \
                f"Internal Error: make_fx is incorrectly baking a tensor constant into the graph: {str(e)}"

        # When we trace factory functions, we expect that tensor outputs are *always* tracked.
        # (Except for torch.tensor() constants handled through lift(), which is handled
        # specially further up).
        pytree.tree_map_only(torch.Tensor, assert_proxy_tensor, out)
        return out


SymInt = torch._C.SymIntNode


class ProxySymDispatchMode(SymDispatchMode):
    def __init__(self, tracer):
        super().__init__()
        self.tracer = tracer
        # When false, we don't trace operations.  If you do this, you MUST
        # call track_tensor/track_tensor_tree on all results of the operation
        # to ensure we can adeduately track the results
        self.enable_tracing = True

    @contextmanager
    def enable(self, b):
        old = self.enable_tracing
        self.enable_tracing = b
        try:
            yield
        finally:
            self.enable_tracing = old

    def __sym_dispatch__(self, func, types, args, kwargs):
        if not self.enable_tracing:
            return func(*args, **kwargs)
        p_args, p_kwargs = pytree.tree_map_only(
            PySymInt,
            lambda s: get_proxy_slot(s, self.tracer) if s.constant is None else s.constant,
            (args, kwargs)
        )
        # func doesn't have a __torch_function__ that Proxy can interpose, so
        # we gotta do it manually
        n_args, n_kwargs = pytree.tree_map_only(fx.Proxy, lambda p: p.node, (p_args, p_kwargs))

        n_out = self.tracer.create_node("call_function", func, n_args, n_kwargs)
        p_out = fx.Proxy(n_out, self.tracer)
        out = func(*args, **kwargs)
        assert isinstance(out, PySymInt), f"{func}(*{args}, **{kwargs}) = {out}"
        set_proxy_slot(out, self.tracer, p_out)
        return out


# TODO: I'm not sure what the point of this class is; you can just
# make_fx through a regular Interpreter
class DecompositionInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule, new_graph: torch.fx.Graph, decomposition_table=None, **kwargs):
        super().__init__(module, **kwargs)
        self.new_graph = new_graph
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)
        self.decomposition_table = decomposition_table
        if self.decomposition_table is None:
            self.decomposition_table = {}
        self.mode = ProxyTorchDispatchMode(self.tracer)

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


def make_fx(f, decomposition_table=None, tracing_mode="real"):
    assert tracing_mode in ["real", "fake", "symbolic"]

    if decomposition_table is None:
        decomposition_table = {}

    @functools.wraps(f)
    def wrapped(*args):
        phs = pytree.tree_map(lambda _: fx.PH, args)  # type: ignore[attr-defined]
        fx_tracer = PythonKeyTracer()
        fx_tracer.record_stack_traces = True
        fake_tensor_mode: Any = nullcontext()
        if tracing_mode == "real":
            fake_tensor_mode = nullcontext()
        elif tracing_mode == "fake":
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
        elif tracing_mode == "symbolic":
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False)
        else:
            raise AssertionError(f"Unexpected tracing type: {tracing_mode}")

        proxy_mode = ProxyTorchDispatchMode(fx_tracer)

        def wrap_fake_concrete(x):
            if isinstance(x, torch.Tensor):
                return fake_tensor_mode.from_tensor(x)  # type: ignore[attr-defined]

            return x

        shape_env = ShapeEnv()
        sym_mode = proxy_mode.sym_mode

        # todo: Figure out a more informative name for symints
        def wrap_fake_symbolic(x, sym_shape):
            if isinstance(x, torch.Tensor):
                val = FakeTensor(fake_tensor_mode, torch.empty(sym_shape, device="meta", requires_grad=x.requires_grad), x.device)
                return val
            return x

        wrap_fn_map = {
            "real": lambda x: x,
            "fake": wrap_fake_concrete,
        }
        if tracing_mode == "symbolic":
            flat_shapes = shape_env.create_shapes_for_args(args)
            flat_args, spec = pytree.tree_flatten(args)
            args = pytree.tree_unflatten(list(map(lambda a: wrap_fake_symbolic(a[0], a[1]), zip(flat_args, flat_shapes))), spec)
        else:
            args = pytree.tree_map(wrap_fn_map[tracing_mode], args)

        if not hasattr(f, '__code__') or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS:
            # FX doesn't support varargs, so we gotta fake up a wrapper
            # TODO: Would be nice to fix this at the source...
            func = fake_signature(f, len(phs))
        else:
            func = f

        with decompose(decomposition_table), fake_tensor_mode, sym_mode, proxy_mode:  # type: ignore[attr-defined]
            t = dispatch_trace(wrap_key(func, args, fx_tracer), tracer=fx_tracer, concrete_args=tuple(phs))

        # TODO: kind of a bad way to do it, should maybe figure out a better way
        t.shape_env = shape_env  # type: ignore[assignment]
        return t

    return wrapped


def get_torch_dispatch_modes():
    modes = [torch._C._get_torch_dispatch_mode()]
    if modes[-1] is None:
        return list()
    while modes[-1].inner is not None:
        modes.append(modes[-1].inner)
    return modes


@contextlib.contextmanager
def disable_proxy_modes_tracing():
    # TODO: This probably doesn't correctly also disable ProxySymDispatchMode
    modes = get_torch_dispatch_modes()
    proxy_tensor_modes = [m for m in modes if isinstance(m, ProxyTorchDispatchMode)]
    olds = [m.enable_tracing for m in proxy_tensor_modes]
    for proxy_mode in proxy_tensor_modes:
        proxy_mode.enable_tracing = False
    try:
        yield
    finally:
        for proxy_mode, old in zip(proxy_tensor_modes, olds):
            proxy_mode.enable_tracing = old


def get_isolated_graphmodule(func, args, kwargs):
    """A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
    wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)

    with disable_proxy_modes_tracing():
        gm = make_fx(wrapped)(all_args)
    return gm
