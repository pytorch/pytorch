# mypy: allow-untyped-defs
"""Compiles pure blocks of the torch._rust CFG executor with Inductor.

The executor calls ``trace_region`` on a run of a pure block's version (see
torch/_rust/src/exec/region.rs) with the block's params and inputs. It runs
the body once under ``make_fx`` on the real inputs, so the run is also the
trace, and compiles the aten graph with ``compile_fx``. Tensors the body read
that aren't inputs (module parameters and buffers) become extra graph inputs,
held by the region, so autograd reaches them.

The executor adds a version calling the region's ``func`` on the block's
params and inputs, guarded on what the trace specialized on: the guard checks
top-level tensors, scalars and objects; ``func`` checks the containers (lists,
tuples, dicts) and returns None when one doesn't match, which runs the
version the trace came from instead.

What the body read from anything else (a global, an attribute of a module, a
closure variable, at any call depth) and the thread state the graph is
specialized on are recorded while it is traced and checked by the executor
before ``func`` is called (torch/_rust/src/exec/reads.rs): a region traced in
training mode isn't entered in evaluation mode, which gets one of its own.
"""

import contextlib
import dataclasses
import functools
import importlib
import inspect
import logging
import operator
import os
import re
import types
import weakref
from collections.abc import Callable
from typing import Any

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_mode,
    make_fx,
    PythonKeyTracer,
    track_tensor_tree,
)
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


log = logging.getLogger(__name__)


def _rust() -> Any:
    """The torch._rust extension (it has no stubs, and a build may not have
    it: see torch/__init__.py)."""
    return importlib.import_module("torch._rust")


class _Null:
    def __repr__(self) -> str:
        return "NULL"


# Stands for CPython's NULL (an empty LoadMethod self slot) in args and
# outputs crossing into Python.
NULL = _Null()

MIN_OPS = int(os.environ.get("TORCH_RUST_REGION_MIN_OPS", "4"))
DEBUG = bool(os.environ.get("TORCH_RUST_REGION_DEBUG"))
COUNT = bool(os.environ.get("TORCH_RUST_REGION_COUNT"))
counts: dict[str, int] = {}
# Set by torch.vm: the block passes of the function it's running (a
# torch.vm._pipeline.Pipeline); None compiles with Inductor.
block_pipeline: Any = None

_PLAIN = (torch.Tensor, torch.nn.Parameter)
_SCALARS = (bool, int, float, type(None))
_IMMUTABLE = (
    bool,
    int,
    float,
    type(None),
    str,
    torch.dtype,
    torch.device,
    torch.layout,
    torch.memory_format,
)
_SKIP_OPS = {torch.ops.aten._local_scalar_dense.default, torch.ops.aten.item.default}


def _cpu_read(n: torch.fx.Node) -> bool:
    """A read of a CPU tensor's value (no sync), which torch.vm traces."""
    v = getattr(n.args[0], "meta", {}).get("val")
    return isinstance(v, torch.Tensor) and v.device.type == "cpu"


_CONTAINERS = (list, tuple, dict)


class _Scalar(float):
    """A float a region takes as a graph input (in torch.vm): while the region
    is traced, the Python arithmetic on it is recorded as graph nodes (the
    value is computed as on plain floats, so a graph run reproduces it
    exactly), so it may differ on the next call. Branching on it, comparing
    or hashing it specializes the region on the value of every input it came
    from. Builtins that read the C double (``math.sqrt``) aren't seen: the
    value they return is baked into the graph."""

    # expr: ("input", k) for the region's k-th float input, else (op, args)
    # with args _Scalars of the same trace or numbers. trace: the
    # _scalar_state of the trace that made it.
    __slots__ = ("expr", "deps", "trace")
    expr: tuple[Any, ...]
    deps: frozenset[int]
    trace: dict[str, Any] | None


# While a region is traced in torch.vm: the float inputs (by position among
# the graph's float inputs) it specialized on, and _DATA if it branched on a
# value it read from a tensor.
_scalar_state: dict[str, Any] | None = None
_DATA = -1


def _new_scalar(value: float, expr: tuple[Any, ...], deps: frozenset) -> _Scalar:
    s = _Scalar(value)
    s.expr = expr
    s.deps = deps
    s.trace = _scalar_state
    return s


def _plain(x):
    return float.__float__(x) if type(x) is _Scalar else x


def _tracer() -> PythonKeyTracer:
    """The tracer of the make_fx trace running (the region's)."""
    mode = get_proxy_mode()
    if mode is None or not isinstance(mode.tracer, PythonKeyTracer):
        raise AssertionError("no make_fx trace is running")
    return mode.tracer


def _plain_result(x, depth=0):
    """x as the program is to see it once a trace is over: each _Scalar in it
    replaced by the float it is. Exact tuples are rebuilt, exact lists and
    dicts changed in place (the run that was traced made them). The executor
    calls this on what a traced run leaves behind (see exec/region.rs)."""
    t = type(x)
    if t is _Scalar:
        return float.__float__(x)
    if depth < 16:
        if t is tuple:
            new = tuple(_plain_result(y, depth + 1) for y in x)
            return new if any(a is not b for a, b in zip(new, x)) else x
        if t is list or t is dict:
            for k, y in list(x.items()) if t is dict else list(enumerate(x)):
                z = _plain_result(y, depth + 1)
                if z is not y:
                    x[k] = z
    return x


def _live(x) -> bool:
    """Whether x is a _Scalar of the trace running (one an earlier trace left
    in a frame is a plain float)."""
    return type(x) is _Scalar and _scalar_state is not None and x.trace is _scalar_state


def _arith(op, reflected=False):
    def method(self, other):
        if type(other) is not _Scalar and type(other) not in (int, float):
            return NotImplemented
        a, b = (other, self) if reflected else (self, other)
        value = op(_plain(a), _plain(b))
        if type(value) is not float or not (_live(a) or _live(b)):
            return value
        deps = frozenset().union(*(x.deps for x in (a, b) if _live(x)))
        return _new_scalar(
            value, (op, tuple(x if _live(x) else _plain(x) for x in (a, b))), deps
        )

    return method


def _unary(op):
    def method(self):
        value = op(float.__float__(self))
        if not _live(self):
            return value
        return _new_scalar(value, (op, (self,)), self.deps)

    return method


def _specializing(name):
    base = getattr(float, name)

    def method(self, *args):
        if _scalar_state is not None:
            _scalar_state["specialized"].update(
                *(a.deps for a in (self, *args) if _live(a))
            )
        return base(float.__float__(self), *map(_plain, args))

    return method


for _name, _op in (
    ("add", operator.add),
    ("sub", operator.sub),
    ("mul", operator.mul),
    ("truediv", operator.truediv),
    ("floordiv", operator.floordiv),
    ("mod", operator.mod),
    ("pow", operator.pow),
):
    setattr(_Scalar, f"__{_name}__", _arith(_op))
    setattr(_Scalar, f"__r{_name}__", _arith(_op, reflected=True))
for _name, _op in (("neg", operator.neg), ("pos", operator.pos), ("abs", operator.abs)):
    setattr(_Scalar, f"__{_name}__", _unary(_op))
for _name in (
    "__bool__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__int__",
    "__hash__",
    "__round__",
    "__trunc__",
    "__floor__",
    "__ceil__",
    "__format__",
    "__repr__",
    "__str__",
    "__divmod__",
    "__rdivmod__",
    "is_integer",
    "as_integer_ratio",
    "hex",
):
    setattr(_Scalar, _name, _specializing(_name))


def _has_floats(p) -> bool:
    return p is not None and (
        p[0] == "float" or (p[0] == "tree" and any(c[0] == _FLOAT for c in p[2]))
    )


# Whether _skip_tracer_frames has run.
_tracer_frames_skipped = False


def _skip_tracer_frames() -> None:
    """While a region's body is traced the executor runs the frames under it,
    so that what they read is recorded (torch/_rust/src/exec/reads.rs). The
    trace's own Python is not the body's. The executor leaves the frames of
    this file, of torch.fx and of torch.overrides to CPython and takes the
    frames they enter back (a function mode's handler calls the body's
    function). Those listed here run on CPython with everything they call:
    the float subclass, the handlers of the dispatch modes, and the stand-in
    for ``autograd.Function.apply``, whose forward the graph calls anew on
    every run."""
    global _tracer_frames_skipped
    if _tracer_frames_skipped:
        return
    from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode

    _cfg_skip = _rust()._cfg_skip

    own = [
        *vars(_Scalar).values(),
        _RefuseImpure.__torch_dispatch__,
        ProxyTorchDispatchMode.__torch_dispatch__,
        _tracing_apply,
        _refusing_init,
        _opaque_call,
    ]
    for f in own:
        if hasattr(f, "__code__"):
            _cfg_skip(f, True)
    _tracer_frames_skipped = True


def _plain_tree(x) -> bool:
    t = type(x)
    if t is list or t is tuple:
        return all(map(_plain_tree, x))
    if t is dict:
        return all(map(_plain_tree, x.values()))
    if _made_of(x) is not None:
        # A float in a cell can't be swapped for the trace: it is a value.
        return False
    return not _is_tree(x) and (not isinstance(x, float) or t is float or t is _Scalar)


def _substitute(x, scalars, restore, n, locs, where):
    """x with each float leaf (in _walk's order) replaced by the next of
    `scalars`: lists and dicts changed in place (undone from `restore`),
    tuples rebuilt. `n` counts leaves; `locs` gets (arg, leaf index) per
    float."""
    t = type(x)
    if t is float or t is _Scalar:
        locs.append((where[0], n[0] if where[1] else None))
        n[0] += 1
        return next(scalars)
    if t is list or t is dict:
        items = list(x.items()) if t is dict else list(enumerate(x))
        for k, y in items:
            z = _substitute(y, scalars, restore, n, locs, (where[0], True))
            if z is not y:
                restore.append((x, k, y))
                x[k] = z
        return x
    if t is tuple:
        new = tuple(
            _substitute(y, scalars, restore, n, locs, (where[0], True)) for y in x
        )
        return new if any(a is not b for a, b in zip(new, x)) else x
    n[0] += 1
    return x


def _floats_in(x, out):
    """Appends the floats in x (args, kwargs: nested lists, tuples, dicts) to
    out, in order."""
    if isinstance(x, float):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _floats_in(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _floats_in(y, out)
    return out


class _ScalarArgs(TorchFunctionMode):
    """Puts the _Scalars a torch call is given back into the node the call
    traced: the dispatcher hands make_fx new Python floats. The node's float
    args are matched to the call's, in order; when they don't line up (the
    call traced several nodes, or made a scalar a tensor), the region is
    specialized on those _Scalars instead."""

    def __init__(self, graph: torch.fx.Graph) -> None:
        super().__init__()
        self.graph = graph

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if (
            func is torch.Tensor.item
            and not kwargs
            and len(args) == 1
            and _scalar_state is not None
        ):
            t = args[0]
            if (
                isinstance(t, torch.Tensor)
                and t.device.type == "cpu"
                and t.numel() == 1
            ):
                # A CPU scalar's value (no sync): a graph node, so the region
                # reads it on every call.
                with torch.utils._python_dispatch._disable_current_modes():
                    value = t.item()
                tracer = _tracer()
                proxy = tracer.create_proxy(
                    "call_function",
                    torch.ops.aten._local_scalar_dense.default,
                    (tracer.unwrap_proxy(t),),
                    {},
                )
                proxy.node.meta["val"] = value
                if type(value) is float:
                    return _new_scalar(value, ("node", proxy.node), frozenset([_DATA]))
                _scalar_state["specialized"].add(_DATA)
                return value
        given = _floats_in([args, kwargs], [])
        live = [x for x in given if _live(x)]
        if not live:
            return func(*args, **kwargs)
        last = self.graph._root.prev
        out = func(*args, **kwargs)
        made = []
        node = last.next
        while node is not self.graph._root:
            if node.op == "call_function":
                made.append(node)
            node = node.next
        traced = [n for n in made if _floats_in([list(n.args), dict(n.kwargs)], [])]
        if len(traced) == 1 and [
            float.__float__(x)
            for x in _floats_in([list(traced[0].args), dict(traced[0].kwargs)], [])
        ] == [float.__float__(x) for x in given]:
            it = iter(given)

            def put_back(a):
                return next(it) if isinstance(a, float) else a

            traced[0].args = torch.fx.node.map_aggregate(traced[0].args, put_back)
            traced[0].kwargs = torch.fx.node.map_aggregate(traced[0].kwargs, put_back)
        elif _scalar_state is not None:
            _scalar_state["specialized"].update(*(x.deps for x in live))
        return out


def _scalars_to_nodes(
    gm: torch.fx.GraphModule, inputs: list[torch.fx.Node], state: dict[str, Any]
) -> None:
    """Replaces the _Scalars of the trace (`state`) in gm's nodes by nodes
    computing them from the float `inputs`, each made before its first use.
    Others (an earlier trace's) are their values."""
    made: dict[int, Any] = {}

    def build(a, before):
        if type(a) is not _Scalar:
            return a
        if a.trace is not state:
            return float.__float__(a)
        if id(a) not in made:
            if a.expr[0] == "input":
                made[id(a)] = inputs[a.expr[1]]
            elif a.expr[0] == "node":
                made[id(a)] = a.expr[1]
            else:
                op, args = a.expr
                args = tuple(build(x, before) for x in args)
                with gm.graph.inserting_before(before):
                    made[id(a)] = gm.graph.call_function(op, args)
                made[id(a)].meta["val"] = float.__float__(a)
        return made[id(a)]

    for node in list(gm.graph.nodes):
        node.args = torch.fx.node.map_aggregate(node.args, lambda a: build(a, node))
        node.kwargs = torch.fx.node.map_aggregate(node.kwargs, lambda a: build(a, node))


def _paths(args, plan) -> dict[int, tuple[int, tuple[Any, ...]]]:
    """The lists and dicts inside args the plan reads as trees (whose
    structure func checks), by id: (arg, path to it)."""
    out: dict[int, tuple[int, tuple[Any, ...]]] = {}
    for i, (a, p) in enumerate(zip(args, plan)):
        if p is None or p[0] != "tree":
            continue
        todo: list[tuple[Any, tuple[Any, ...]]] = [(a, ())]
        while todo:
            x, path = todo.pop()
            t = type(x)
            if t is list or t is tuple:
                if t is list and path:
                    out.setdefault(id(x), (i, path))
                todo.extend((y, (*path, k)) for k, y in enumerate(x))
            elif t is dict:
                if path:
                    out.setdefault(id(x), (i, path))
                todo.extend((y, (*path, k)) for k, y in x.items())
    return out


def _at_path(path, items):
    x = items[0]
    for k in path:
        x = x[k]
    return x


# Pytree node types (by qualified name: torch doesn't depend on torchrec)
# whose flattening keeps all of their state, so a region can take and make
# them by their leaves. Others (KeyedJaggedTensor caches Python lists) are
# leaves, checked by identity.
_TREES = {
    "torchrec.sparse.jagged_tensor.KeyedTensor",
    "torchrec.sparse.jagged_tensor.JaggedTensor",
}


_tree_types: dict[type, bool] = {}


def _is_tree(x) -> bool:
    t = type(x)
    tree = _tree_types.get(t)
    if tree is None:
        tree = _tree_types[t] = (
            issubclass(t, _CONTAINERS) or f"{t.__module__}.{t.__qualname__}" in _TREES
        )
    return tree


def _flatten(x):
    return pytree.tree_flatten(x, is_leaf=lambda y: not _is_tree(y))


# The functions being walked (a closure can hold itself).
_walking: set[int] = set()


def _made_of(x):
    """``(shape, parts)`` of a function that is made of more than its code,
    None for anything else: a function with a closure or defaults (its
    defaults, then what its cells hold), a bound method (its function and its
    self), a ``functools.partial`` (its function, its args, its keyword
    values). One made anew on every call can't be told by identity, so a
    region guards it as it guards a container: by the shape, and each part by
    the rule of its type. A tensor in a cell is then an input of the graph:
    the region computes with the one this call's closure holds. A function
    with neither closure nor defaults is what it is, by identity."""
    t = type(x)
    if t is types.FunctionType:
        cells = x.__closure__ or ()
        defaults = x.__defaults__ or ()
        named = x.__kwdefaults__ or {}
        if not (cells or defaults or named) or id(x) in _walking:
            return None
        held = []
        for c in cells:
            try:
                held.append(c.cell_contents)
            except ValueError:
                # A cell nothing is bound to yet.
                held.append(NULL)
        shape = (
            t,
            x.__code__,
            id(x.__globals__),
            len(defaults),
            tuple(named),
            len(cells),
        )
        return shape, [*defaults, *named.values(), *held]
    if t is types.MethodType:
        return (t,), [x.__func__, x.__self__]
    if t is functools.partial:
        return (t, tuple(x.keywords)), [x.func, *x.args, *x.keywords.values()]
    return None


def _walk(x, leaves, shape):
    """Appends x's leaves to `leaves` and its structure to `shape`: as
    _flatten, without pytree for exact lists, tuples and dicts (run on every
    region call)."""
    t = type(x)
    if t is list or t is tuple:
        shape.append((t, len(x)))
        for y in x:
            _walk(y, leaves, shape)
    elif t is dict:
        shape.append((dict, tuple(x)))
        for y in x.values():
            _walk(y, leaves, shape)
    elif (made := _made_of(x)) is not None:
        shape.append(made[0])
        _walking.add(id(x))
        try:
            for y in made[1]:
                _walk(y, leaves, shape)
        finally:
            _walking.discard(id(x))
    elif _is_tree(x):
        node = pytree.SUPPORTED_NODES.get(t)
        if node is None:
            more, spec = _flatten(x)
            shape.append(spec)
            leaves.extend(more)
            return
        # A registered pytree node, one level at a time: no TreeSpec to build
        # and compare on every call.
        children, context = node.flatten_fn(x)
        shape.append((t, context))
        for y in children:
            _walk(y, leaves, shape)
    else:
        leaves.append(x)


_ITERATORS = tuple(
    type(iter(x)) for x in ([], (), {}, {}.values(), {}.items(), range(0))
)
# Objects an output may be read from a region's pinned objects as: those that
# a run doesn't make anew with its tensors in them.
_PINNABLE = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.MethodWrapperType,
    types.MethodDescriptorType,
    types.WrapperDescriptorType,
    types.ClassMethodDescriptorType,
    types.ModuleType,
    type,
    torch.nn.Module,
    torch.Size,
)

# "reads": the entries of the regions' checks of what their traces read.
stats = {
    "traced": 0,
    "compiled": 0,
    "reused": 0,
    "small": 0,
    "refused": 0,
    "misses": 0,
    "reads": 0,
}


@dataclasses.dataclass
class Trace:
    """What a trace of a block or a span came to (see torch.vm.regions)."""

    # "QUALNAME:bbN lines A-B", or "QUALNAME:span bbN".
    name: str
    # "region", "too small", "refused" (the trace depends on something a
    # region cannot hold or check) or "gave up" (the traced run raised, or
    # left the span by a way no earlier run took).
    state: str
    # Why it is no region.
    why: str | None = None
    ops: int = 0
    inputs: int = 0
    constants: int = 0
    # The check of what the region's trace read (exec/reads.rs), if any.
    reads: Any = None
    # How often the region ran, and the calls its own checks turned down, by
    # what did not match.
    runs: int = 0
    misses: dict[str, int] = dataclasses.field(default_factory=dict)


# The traces of this process, in the order they were made.
traces: list[Trace] = []


def _traced(name: str, state: str, why: str | None = None, **fields: Any) -> Trace:
    record = Trace(name, state, why, **fields)
    traces.append(record)
    return record


# Compiled graphs by (code, input metadata, grad mode): clones of a code (one
# per call site) trace the same graphs.
_compiled: dict[Any, Any] = {}


class _Refused(Exception):
    pass


class _Bound:
    """A region's lowering that took the region's constants once (see
    torch.vm.lower_to's ``bind``): ``call`` takes the other inputs. The
    executor calls ``rebind`` before an entry at which the memory of a
    constant is somewhere else (exec/reads.rs)."""

    def __init__(self, bind: Callable, consts: list[torch.Tensor]) -> None:
        self.bind = bind
        self.consts = consts
        self.call = bind(consts)

    def rebind(self) -> None:
        self.call = self.bind(self.consts)


# Makes autograd.Function.apply raise, before the forward runs, while a region
# is traced, unless it's _tracing_apply's: the trace would have the forward's
# ops and lose the backward, and the forward may have effects (torchrec's
# launch collectives). apply makes the Function's ctx (a BackwardCFunction)
# first.
def _refusing_init(ctx, *a, **kw):
    raise _Refused(f"it calls the autograd.Function {ctx._forward_cls.__name__}")


def autocast_on():
    """The autocast dispatch keys that are on, on this thread: those the guard
    that turns autocast off adds to the excluded keys."""
    excluded = torch._C._dispatch_tls_local_exclude_set()
    with torch._C._DisableAutocast():
        return torch._C._dispatch_tls_local_exclude_set() - excluded


def _under(autocast, fn):
    """``fn``, run with the autocast keys ``autocast`` on again. A region's
    graph runs with the autocast keys of its trace off (see trace_region); a
    node that calls back into the program runs what the program ran, the way
    the program ran it."""
    if not autocast.raw_repr():
        return fn

    def call(*args, **kwargs):
        included = torch._C._dispatch_tls_local_include_set()
        excluded = torch._C._dispatch_tls_local_exclude_set() - autocast
        with torch._C._ForceDispatchKeyGuard(included, excluded):
            return fn(*args, **kwargs)

    return call


_function_apply = torch.autograd.Function.__dict__["apply"]
# The graph's target for each autograd.Function's apply (one per class and
# set of autocast keys it is called under, so that graphs of the same code
# are the same).
_appliers: dict[tuple[type, int], Any] = {}


def _applier(cls, autocast):
    fn = _appliers.get((cls, autocast.raw_repr()))
    if fn is None:
        apply = _under(autocast, _function_apply.__get__(None, cls))

        def fn(*args):
            return apply(*args)

        name = f"apply_{cls.__name__}"
        if name in globals():
            name = f"{name}_{len(_appliers)}"
        fn.__name__ = fn.__qualname__ = name
        # A graph's code calls it as torch._rust_inductor.<name>, as it does
        # anything from torch.
        globals()[name] = fn
        _appliers[cls, autocast.raw_repr()] = fn
    return fn


def _tracing_apply(cls, *args, **kwargs):
    """autograd.Function.apply while a region is traced: one node calling the
    real apply, which the backend's tracing then runs with autograd, so the
    graph keeps the Function's own backward (as Dynamo's
    autograd_function_apply does). Purity has judged its forward."""
    mode = get_proxy_mode()
    if mode is None:
        # Not tracing (a call site that recorded this as its callee while a
        # trace ran, or one inside the Function's own forward): the real apply.
        return _function_apply.__get__(None, cls)(*args, **kwargs)
    if kwargs:
        raise _Refused(f"it calls the autograd.Function {cls.__name__} with keywords")
    target = _applier(cls, autocast_on())
    Backward = torch.autograd.function.BackwardCFunction
    with disable_proxy_modes_tracing():
        del Backward.__init__
        try:
            out = target(*args)
        finally:
            Backward.__init__ = _refusing_init
    tracer = _tracer()
    proxy = tracer.create_proxy(
        "call_function", target, tuple(pytree.tree_map(tracer.unwrap_proxy, args)), {}
    )
    return track_tensor_tree(out, proxy, constant=None, tracer=tracer)


class _Effect:
    """The target of a graph node calling `fn`, a call torch.vm.Cfg.merge_across
    put in a region: the node runs it as it is (an effect the graph keeps)."""

    def __init__(self, fn: Any, name: str, autocast: Any) -> None:
        self.fn = fn
        self.call = _under(autocast, fn)
        self.__name__ = self.__qualname__ = name

    def __call__(self, *args, **kwargs):
        # Traced into another graph (a lowering re-tracing the region, as AOT
        # Autograd does), it would run once, at trace time, and the tensors it
        # reads itself (a module's weights) would be baked in, with no grads.
        if (
            get_proxy_mode() is not None
            or torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
            is not None
        ):
            raise RuntimeError(
                f"{self!r} can't be traced into another graph: it must run as eager on every call"
            )
        return self.call(*args, **kwargs)

    def __repr__(self) -> str:
        return f"opaque({getattr(self.fn, '__qualname__', self.fn)!r})"


_effects: dict[tuple[int, int], _Effect] = {}


def _opaque_call(fn, *args, **kwargs):
    """A call of `fn` (see torch.vm.Cfg.merge_across) while a region is
    traced: run as it is, and recorded as one graph node calling it."""
    mode = get_proxy_mode()
    if mode is None:
        return fn(*args, **kwargs)
    autocast = autocast_on()
    target = _effects.get((id(fn), autocast.raw_repr()))
    if target is None:
        name = f"opaque_{getattr(fn, '__name__', 'call')}_{len(_effects)}"
        target = _effects[id(fn), autocast.raw_repr()] = _Effect(fn, name, autocast)
        # A graph's code calls it as torch._rust_inductor.<name>.
        globals()[name] = target
        torch.fx.node._side_effectful_functions.add(target)
    # As eager runs it: no tracing modes, its floats plain; the tensors it
    # writes in place backed up, as the trace's own (the executor runs the
    # block again if the trace gives up).
    global _scalar_state
    state, _scalar_state = _scalar_state, None
    try:
        with (
            torch.utils._python_dispatch._disable_current_modes(),
            torch._C.DisableTorchFunction(),
        ):
            with (
                _RefuseImpure(_trace_backups, refuse=False)
                if _trace_backups is not None
                else contextlib.nullcontext()
            ):
                out = fn(*args, **kwargs)
    finally:
        _scalar_state = state
    tracer = _tracer()
    proxy = tracer.create_proxy(
        "call_function",
        target,
        tuple(pytree.tree_map(tracer.unwrap_proxy, args)),
        pytree.tree_map(tracer.unwrap_proxy, kwargs),
    )
    return track_tensor_tree(out, proxy, constant=None, tracer=tracer)


# The tensors the trace running wrote in place, with their copies (see
# _RefuseImpure).
_trace_backups: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None


class _RefuseImpure(TorchDispatchMode):
    """Raises before an op a region can't hold runs: a collective, or one
    reading a tensor's value. The executor then runs the block as usual,
    which is only sound if nothing it did under tracing had an effect: so
    each tensor an op writes in place is copied into `backups` first, for
    the caller to put back."""

    def __init__(
        self, backups: dict[int, tuple[torch.Tensor, torch.Tensor]], refuse: bool = True
    ) -> None:
        super().__init__()
        self.backups = backups
        self.refuse = refuse

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.refuse and (
            func.namespace in ("c10d", "_c10d_functional") or func in _SKIP_OPS
        ):
            raise _Refused(f"it runs {func}")
        kwargs = kwargs or {}
        if func._schema.is_mutable:
            for k, a in enumerate(func._schema.arguments):
                if a.alias_info is None or not a.alias_info.is_write:
                    continue
                x = args[k] if k < len(args) else kwargs.get(a.name)
                for t in x if isinstance(x, (list, tuple)) else [x]:
                    if isinstance(t, torch.Tensor) and id(t) not in self.backups:
                        with torch.utils._python_dispatch._disable_current_modes():
                            self.backups[id(t)] = (t, t.detach().clone())
        return func(*args, **kwargs)


def _meta(t: torch.Tensor):
    return (type(t), t.dtype, t.device, t.shape, t.stride(), t.requires_grad)


# How func reads a leaf of a container arg: as a graph input (a tensor,
# checked by metadata, or a float, checked by value once the trace
# specialized on it), or checked by value or identity.
_TENSOR, _VALUE, _IS, _FLOAT = 0, 1, 2, 3


def _leaf_check(leaf, where, scalars=False):
    if scalars and isinstance(leaf, float):
        return (_FLOAT, None)
    if isinstance(leaf, torch.Tensor):
        if type(leaf) not in _PLAIN:
            raise _Refused(f"{where} is a {type(leaf).__name__}")
        return (_TENSOR, _meta(leaf))
    if type(leaf) in _IMMUTABLE or isinstance(leaf, torch.Size):
        return (_VALUE, (type(leaf), leaf))
    if isinstance(leaf, (list, dict, set)):
        raise _Refused(f"{where} holds a {type(leaf).__name__}")
    return (_IS, leaf)


def _plan(args, unused, span=False, scalars=False):
    """What the guard checks (Rust) and what func reads and checks (Python),
    by arg: ("tensor",), ("float", value or None), ("equal", type, value) for
    an immutable value that isn't a number, ("tree", spec, leaf
    checks, a TreeChecks or None: see rust.TreeChecks) or None (nothing). Args at `unused` the block doesn't read, or
    only passes on. With `scalars`, floats (and those in plain containers)
    are graph inputs."""
    checks: list[tuple[Any, ...]] = [("grad_enabled", torch.is_grad_enabled())]
    is_checks: list[tuple[Any, ...]] = []
    plan: list[Any] = []
    for i, a in enumerate(args):
        if i in unused:
            plan.append(None)
        elif a is NULL:
            checks.append(("null", i))
            plan.append(None)
        elif isinstance(a, torch.Tensor):
            if type(a) not in _PLAIN:
                raise _Refused(f"arg {i} is a {type(a).__name__}")
            checks.append(("tensor", i, a))
            checks.append(("requires_grad", i, a.requires_grad))
            plan.append(("tensor",))
        elif scalars and isinstance(a, float):
            plan.append(("float", None))
        elif type(a) in _SCALARS:
            checks.append(("value", i, a))
            plan.append(None)
        elif type(a) is tuple and all(type(x) is int for x in a):
            checks.append(("ints", i, True, list(a)))
            plan.append(None)
        elif type(a) in _IMMUTABLE:
            # A str, a device, a dtype: what it is, not which object it is
            # (`tensor.device` is a new object on every read). func checks it.
            plan.append(("equal", type(a), a))
        elif _is_tree(a) or _made_of(a) is not None:
            leaves, shape = [], []
            _walk(a, leaves, shape)
            plain = scalars and _plain_tree(a)
            leaf_checks = [_leaf_check(x, f"arg {i}", plain) for x in leaves]
            fast = None
            if all(c[0] != _FLOAT for c in leaf_checks):
                try:
                    fast = _rust().TreeChecks(a)
                except TypeError:
                    pass
            plan.append(("tree", shape, leaf_checks, fast))
        elif type(a) is set and span and all(type(x) in _IMMUTABLE for x in a):
            # Only a span's region checks it (in Python).
            checks.append(("set", i, frozenset(a)))
            plan.append(None)
        elif isinstance(a, set):
            raise _Refused(f"arg {i} is a set")
        else:
            # Cheap and the most selective: checked first.
            is_checks.append(("is", i, a))
            plan.append(None)
    return is_checks + checks, plan


def _inputs(args, plan, why=None):
    """The graph's inputs among args (tensors, and tensor leaves of
    containers), then the float inputs, or None when a container or a
    specialized float doesn't match the trace (with the mismatch appended to
    `why`)."""
    inputs = []
    floats = []
    for i, (a, p) in enumerate(zip(args, plan)):
        if p is None:
            continue
        if p[0] == "tensor":
            inputs.append(a)
            continue
        if p[0] == "float":
            if not isinstance(a, float) or (
                p[1] is not None and float.__float__(a) != p[1]
            ):
                if why is not None:
                    why.append(f"arg {i} (float)")
                return None
            floats.append(float.__float__(a))
            continue
        if p[0] == "equal":
            if type(a) is not p[1] or a != p[2]:
                if why is not None:
                    why.append(f"arg {i} ({p[1].__name__}, checked by value)")
                return None
            continue
        if p[3] is not None:
            got = p[3].leaves(a)
            if got is not None:
                inputs.extend(got)
                continue
        leaves, shape = [], []
        _walk(a, leaves, shape)
        if shape != p[1]:
            if why is not None:
                why.append(f"arg {i}'s structure")
            return None
        for k, (leaf, (kind, want)) in enumerate(zip(leaves, p[2])):
            if kind == _TENSOR:
                ok = isinstance(leaf, torch.Tensor) and _meta(leaf) == want
                if ok:
                    inputs.append(leaf)
            elif kind == _VALUE:
                ok = type(leaf) is want[0] and leaf == want[1]
            elif kind == _FLOAT:
                ok = isinstance(leaf, float) and (
                    want is None or float.__float__(leaf) == want
                )
                if ok:
                    floats.append(float.__float__(leaf))
            else:
                ok = leaf is want
            if not ok:
                if why is not None:
                    why.append(
                        f"arg {i} leaf {k} ({type(leaf).__name__}, checked by {('meta', 'value', 'identity', 'float value')[kind]})"
                    )
                return None
    return inputs + floats


@functools.cache
def _fx_wrapper_code():
    from torch.fx._symbolic_trace import _create_wrapped_func

    return _create_wrapped_func(len).__code__


def _template(o, args, tensor_outs, objects, old):
    if o is NULL:
        return ("null",)
    for i, a in enumerate(args):
        if o is a:
            return ("arg", i)
    if type(o) is _Scalar:
        tensor_outs.append(o)
        return ("out", len(tensor_outs) - 1)
    if isinstance(o, (list, dict, set)) and id(o) in old:
        # A container that was there before the run, which the version reads
        # (from an object the guard pins) rather than makes: the same one.
        # One an arg holds that no check pins (an unread arg's) may be another
        # next run; one inside an arg func reads as a tree is read from it.
        where = old[id(o)]
        if type(where) is tuple:
            objects.append(functools.partial(_at_path, where[1]))
            return ("call", len(objects) - 1, [("arg", where[0])])
        if not where:
            raise _Refused(f"an output is a {type(o).__name__} an unpinned arg holds")
        objects.append(o)
        return ("obj", len(objects) - 1)
    if isinstance(o, torch.Tensor):
        tensor_outs.append(o)
        return ("out", len(tensor_outs) - 1)
    if type(o) in _IMMUTABLE:
        objects.append(o)
        return ("obj", len(objects) - 1)
    if type(o) in (tuple, list):
        kind = "tuple" if type(o) is tuple else "list"
        return (kind, [_template(x, args, tensor_outs, objects, old) for x in o])
    if type(o) is dict:
        keys = []
        for k in o:
            if type(k) not in _IMMUTABLE:
                raise _Refused(f"an output dict has a {type(k).__name__} key")
            objects.append(k)
            keys.append(len(objects) - 1)
        return (
            "dict",
            keys,
            [_template(v, args, tensor_outs, objects, old) for v in o.values()],
        )
    if type(o) is enumerate:
        it, count = o.__reduce__()[1]
        objects.append(_enumerate)
        return (
            "call",
            len(objects) - 1,
            [
                _template(it, args, tensor_outs, objects, old),
                _template(count, args, tensor_outs, objects, old),
            ],
        )
    if type(o) is zip:
        objects.append(_zip)
        return (
            "call",
            len(objects) - 1,
            [
                _template(it, args, tensor_outs, objects, old)
                for it in o.__reduce__()[1]
            ],
        )
    if type(o) in _ITERATORS:
        # A loop's iterator: rebuilt over what's left of it (the trace's run
        # consumed the rest), which misses changes to the container later.
        reduced = o.__reduce__()
        start = reduced[2] if len(reduced) > 2 and reduced[2] is not None else 0
        rest = list(reduced[1][0])[start:]
        return ("iter", _template(rest, args, tensor_outs, objects, old))
    if (names := _init_fields(o)) is not None:
        objects.append(functools.partial(_dataclass, type(o), names))
        return (
            "call",
            len(objects) - 1,
            [_template(getattr(o, n), args, tensor_outs, objects, old) for n in names],
        )
    if _is_tree(o):
        leaves, spec = _flatten(o)
        objects.append(functools.partial(_unflatten, spec))
        return (
            "call",
            len(objects) - 1,
            [_template(x, args, tensor_outs, objects, old) for x in leaves],
        )
    if isinstance(o, types.FunctionType) and o.__code__ is _fx_wrapper_code():
        # A function fx.wrap registered, which make_fx patches while tracing.
        o = inspect.unwrap(o, stop=lambda f: f.__code__ is not _fx_wrapper_code())
    if isinstance(o, types.FunctionType) and "<locals>" in o.__code__.co_qualname:
        # Maybe made by the run (a new closure each time), not pinnable.
        raise _Refused(f"an output is a local function {o.__qualname__}")
    if not isinstance(o, _PINNABLE):
        raise _Refused(f"an output is a {type(o).__name__}")
    # Read from the objects the guard pins (a submodule, a bound method), which
    # are assumed not to be reassigned, like the parameters a region holds.
    objects.append(o)
    return ("obj", len(objects) - 1)


def _old_containers(args, depth: int = 3, limit: int = 20000) -> dict[int, Any]:
    """The lists, dicts and sets reachable from args (through containers and
    objects' attributes) a few levels deep, by id."""
    seen: dict[int, Any] = {}
    todo = [(a, 0) for a in args]
    while todo and len(seen) < limit:
        x, d = todo.pop()
        if isinstance(x, torch.Tensor) or type(x) in _IMMUTABLE:
            continue
        if isinstance(x, (list, dict, set)):
            if id(x) in seen:
                continue
            seen[id(x)] = x
        if d >= depth:
            continue
        if isinstance(x, dict):
            todo.extend((v, d + 1) for v in x.values())
        elif isinstance(x, (list, tuple, set)):
            todo.extend((v, d + 1) for v in x)
        else:
            attrs = getattr(x, "__dict__", None)
            if isinstance(attrs, dict):
                todo.extend((v, d + 1) for v in attrs.values())
    return seen


def _contents(args) -> dict[int, Any]:
    """The lists, dicts and sets among args (and in them), with what each holds
    (by identity)."""
    out: dict[int, Any] = {}
    todo = list(args)
    while todo:
        x = todo.pop()
        if isinstance(x, (list, dict, set)) and id(x) not in out:
            items = list(x.items()) if isinstance(x, dict) else list(x)
            out[id(x)] = (
                x,
                [(k, id(v)) for k, v in items]
                if isinstance(x, dict)
                else [id(v) for v in items],
            )
            todo.extend(x.values() if isinstance(x, dict) else x)
        elif type(x) is tuple:
            todo.extend(x)
    return out


def _unflatten(spec, leaves):
    return pytree.tree_unflatten(leaves, spec)


def _enumerate(items):
    return enumerate(items[0], items[1])


def _zip(items):
    return zip(*items)


def _dataclass(cls, names, items):
    return cls(**dict(zip(names, items)))


def _init_fields(o) -> list[str] | None:
    """The fields of dataclass instance o, if its class's generated __init__
    sets only those."""
    cls = type(o)
    if not dataclasses.is_dataclass(cls) or hasattr(cls, "__post_init__"):
        return None
    code = getattr(cls.__init__, "__code__", None)
    if code is None or "__create_fn__" not in code.co_qualname:
        return None
    names = [f.name for f in dataclasses.fields(cls)]
    if any(not f.init for f in dataclasses.fields(cls)) or set(
        getattr(o, "__dict__", ())
    ) != set(names):
        return None
    return names


def _lift_constants(gm: torch.fx.GraphModule) -> list[torch.Tensor]:
    """Turns the graph's tensor constants into trailing placeholders."""
    consts = []
    lifted: dict[str, torch.fx.Node] = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    last = placeholders[-1] if placeholders else None
    for node in list(gm.graph.nodes):
        if node.op != "get_attr":
            continue
        value = getattr(gm, node.target)
        if not isinstance(value, torch.Tensor):
            continue
        if value.grad_fn is not None:
            # Made by the autograd graph of one run, not held across runs.
            raise _Refused(
                f"it reads {node.target} ({tuple(value.shape)}), which has a grad_fn ({value.grad_fn.name()})"
            )
        if node.target not in lifted:
            if last is None:
                with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
                    ph = gm.graph.placeholder(f"const_{len(consts)}")
            else:
                with gm.graph.inserting_after(last):
                    ph = gm.graph.placeholder(f"const_{len(consts)}")
            ph.meta = dict(node.meta)
            last = ph
            lifted[node.target] = ph
            consts.append(value)
        node.replace_all_uses_with(lifted[node.target])
        gm.graph.erase_node(node)
    for target in lifted:
        delattr(gm, target)
    gm.graph.lint()
    gm.recompile()
    return consts


def _split_host(gm: torch.fx.GraphModule, n_inputs: int):
    """Splits a graph that reads CPU tensors' values (`.item()`) into a host
    prologue and the rest: the prologue runs the reads, what they read from
    (CPU ops, and the in-place ops before them on what they read) and the
    Python arithmetic on their values, and the rest takes the floats it needs
    from the prologue as inputs, after the graph's first `n_inputs` (before
    the constants). So block passes, which trace with fake tensors, see no
    reads. Returns (prologue, rest, the float values of the trace) or None if
    a read can't move first (it reads a GPU tensor's copy, or another op
    reads what the prologue writes before it does)."""

    def cpu(n):
        v = n.meta.get("val")
        return not isinstance(v, torch.Tensor) or v.device.type == "cpu"

    def written(n):
        if not isinstance(n.target, torch._ops.OpOverload):
            return []
        out = []
        for i, a in enumerate(n.target._schema.arguments):
            if a.alias_info is not None and a.alias_info.is_write:
                x = n.args[i] if i < len(n.args) else n.kwargs.get(a.name)
                out.extend(
                    y for y in pytree.tree_leaves(x) if isinstance(y, torch.fx.Node)
                )
        return out

    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    writers: dict[torch.fx.Node, list[torch.fx.Node]] = {}
    for n in gm.graph.nodes:
        for x in written(n):
            writers.setdefault(x, []).append(n)
    reads = [n for n in gm.graph.nodes if n.target in _SKIP_OPS]
    host: set[torch.fx.Node] = set()
    todo = [(n, len(order)) for n in reads]
    while todo:
        n, before = todo.pop()
        todo.extend(
            (w, order[w])
            for w in writers.get(n, ())
            if order[w] < before and w not in host
        )
        if n.op != "call_function" or n in host:
            continue
        if not cpu(n) or not all(cpu(a) for a in n.all_input_nodes):
            return None
        host.add(n)
        todo.extend((a, order[n]) for a in n.all_input_nodes)
    for n in gm.graph.nodes:
        if (
            n.op == "call_function"
            and n not in host
            and not isinstance(n.meta.get("val"), torch.Tensor)
            and n.all_input_nodes
            and all(a in host or a.op == "placeholder" for a in n.all_input_nodes)
            and any(a in host for a in n.all_input_nodes)
        ):
            host.add(n)
    for w in host:
        for x in written(w):
            if any(
                u not in host and u.op != "output" and order[u] < order[w]
                for u in x.users
            ):
                return None
    shared: dict[torch.fx.Node, None] = {}
    for n in gm.graph.nodes:
        if n not in host:
            for a in n.all_input_nodes:
                if a in host:
                    shared[a] = None
    if any(type(a.meta.get("val")) is not float for a in shared):
        return None
    pro = torch.fx.Graph()
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    for n in gm.graph.nodes:
        if n.op == "placeholder" or n in host:
            env[n] = pro.node_copy(n, env.__getitem__)
            if n.target in _SKIP_OPS:
                # Tensor.item skips OpOverload.__call__.
                env[n].op, env[n].target = "call_method", "item"
            elif isinstance(
                n.target, torch._ops.OpOverload
            ) and n.target.overloadpacket.__name__.startswith("_foreach_"):
                # torch's own binding converts a tensor list faster than
                # OpOverload's (it picks the same overload from the args).
                env[n].target = getattr(
                    torch, n.target.overloadpacket.__name__, n.target
                )
    pro.output(tuple(env[a] for a in shared))
    _dedupe_reads(pro)
    _memo_chains(pro)
    placeholders = list(gm.graph.find_nodes(op="placeholder"))
    anchor = placeholders[n_inputs - 1] if n_inputs else None
    for i, a in enumerate(shared):
        if anchor is None:
            with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
                ph = gm.graph.placeholder(f"host_{i}")
        else:
            with gm.graph.inserting_after(anchor):
                ph = gm.graph.placeholder(f"host_{i}")
        ph.meta["val"] = a.meta["val"]
        anchor = ph
        a.replace_all_uses_with(ph)
    for n in reversed(list(gm.graph.nodes)):
        if n in host:
            gm.graph.erase_node(n)
    gm.graph.lint()
    gm.recompile()
    return torch.fx.GraphModule(gm, pro), gm, [a.meta["val"] for a in shared]


def _dedupe_reads(pro: torch.fx.Graph) -> None:
    """Has a host prologue's `.item()` of a tensor reuse an earlier read of
    the same tensor when nothing wrote in between (Adam reads each step count
    twice)."""
    seen: dict[torch.fx.Node, torch.fx.Node] = {}
    for n in list(pro.nodes):
        if n.op == "call_method" and n.target == "item":
            first = seen.setdefault(n.args[0], n)
            if first is not n:
                n.replace_all_uses_with(first)
                pro.erase_node(n)
        elif (
            isinstance(n.target, torch._ops.OpOverload)
            and n.target._schema.is_mutable
            or callable(n.target)
            and getattr(n.target, "__name__", "").endswith("_")
        ):
            seen.clear()
    pro.lint()


_SCALAR_OPS = {
    v for k, v in vars(operator).items() if not k.startswith("_") and callable(v)
}


def _run_chains(template, reads, *shared):
    """`template(r, *shared)` (a tuple) for each of `reads`, flattened, run
    once per distinct value (Python floats: the same value, the same
    result; -0.0 and NaN, which a dict can't tell apart, run each time)."""
    out = []
    seen: dict[Any, tuple[Any, ...]] = {}
    for r in reads:
        if r == 0 or r != r:
            out.extend(template(r, *shared))
            continue
        got = seen.get(r)
        if got is None:
            got = seen[r] = template(r, *shared)
        out.extend(got)
    return out


def _chains_target(template: torch.fx.GraphModule) -> Callable[..., list[Any]]:
    """`_run_chains` of `template`, as a function a graph's code calls by its
    qualified name (torch._rust_inductor._chains_N)."""
    name = f"_chains_{len(_chain_templates)}"
    _chain_templates.append(template)

    def run(reads, *shared):
        return _run_chains(template, reads, *shared)

    run.__name__ = run.__qualname__ = name
    run.__module__ = __name__
    globals()[name] = run
    return run


_chain_templates: list[torch.fx.GraphModule] = []


def _memo_chains(pro: torch.fx.Graph) -> None:
    """Replaces the host prologue's arithmetic chains that are the same for
    each value read (Adam's bias corrections: one chain per parameter, on
    step counts that are mostly equal) by one `_run_chains` per group."""
    reads = [n for n in pro.nodes if n.op == "call_method" and n.target == "item"]
    if len(reads) < 8:
        return
    order = {n: i for i, n in enumerate(pro.nodes)}
    tainted = set(reads)
    for n in pro.nodes:
        if any(a in tainted for a in n.all_input_nodes):
            tainted.add(n)
    owner: dict[torch.fx.Node, torch.fx.Node] = {r: r for r in reads}
    for n in pro.nodes:
        if n.op != "call_function" or n.target not in _SCALAR_OPS or n in owner:
            continue
        chain_deps = {owner[a] for a in n.all_input_nodes if a in owner}
        # Its other inputs must not depend on a read at all (shared values).
        if len(chain_deps) == 1 and all(
            a in owner or a not in tainted for a in n.all_input_nodes
        ):
            owner[n] = chain_deps.pop()
    chains: dict[torch.fx.Node, list[torch.fx.Node]] = {r: [] for r in reads}
    for n in pro.nodes:
        if n in owner and n is not owner[n]:
            chains[owner[n]].append(n)
    groups: dict[
        Any,
        list[
            tuple[
                torch.fx.Node,
                list[torch.fx.Node],
                list[torch.fx.Node],
                list[torch.fx.Node],
            ]
        ],
    ] = {}
    for r, nodes in chains.items():
        members = {r, *nodes}
        index = {x: i for i, x in enumerate(nodes)}
        shared: list[torch.fx.Node] = []

        def canon(a):
            if a is r:
                return ("R",)
            if isinstance(a, torch.fx.Node):
                if a in index:
                    return ("C", index[a])
                if a not in shared:
                    shared.append(a)
                return ("S", a)
            if type(a) is float:
                return ("F", a.hex())
            if type(a) in (int, bool, type(None)):
                return ("L", type(a), a)
            return None

        sig = []
        for x in nodes:
            args = tuple(canon(a) for a in x.args)
            if x.kwargs or any(c is None for c in args):
                break
            sig.append((x.target, args))
        else:
            exports = [x for x in (r, *nodes) if any(u not in members for u in x.users)]
            key = (tuple(sig), tuple((r, *nodes).index(x) for x in exports))
            groups.setdefault(key, []).append((r, nodes, exports, shared))
    for (sig, export_pos), group in groups.items():
        if len(group) < 4 or not sig:
            continue
        r0, nodes0, _, shared0 = group[0]
        template = torch.fx.Graph()
        env: dict[torch.fx.Node, torch.fx.Node] = {r0: template.placeholder("r")}
        for i, x in enumerate(shared0):
            env[x] = template.placeholder(f"s{i}")
        for x in nodes0:
            env[x] = template.node_copy(x, env.__getitem__)
        template.output(tuple(env[(r0, *nodes0)[k]] for k in export_pos))
        fn = _chains_target(torch.fx.GraphModule(torch.nn.Module(), template))
        users = [
            u
            for _, nodes, exports, _ in group
            for x in exports
            for u in x.users
            if u not in owner
        ]
        first_use = min(users, key=order.__getitem__)
        with pro.inserting_before(first_use):
            run = pro.call_function(fn, ([r for r, *_ in group], *shared0))
            k = 0
            for r, nodes, exports, _ in group:
                for x in exports:
                    g = pro.call_function(operator.getitem, (run, k))
                    g.meta = dict(x.meta)
                    x.replace_all_uses_with(g, delete_user_cb=lambda u: u not in owner)
                    k += 1
        for r, nodes, _, _ in group:
            for x in reversed(nodes):
                if not x.users:
                    pro.erase_node(x)
        stats["chains"] = stats.get("chains", 0) + len(group)
    pro.lint()


def _const_homes(args, consts):
    """Where a region's constants (the tensors its body read that aren't
    inputs) are kept by what it was given: (dict, key, tensor) for each one
    that is a parameter, a buffer or an attribute of a module among ``args``
    (or of a submodule), or an attribute of another object among them. The
    region only runs while each of those still holds its tensor: they are
    added to its reads (see trace_region)."""
    want = {id(c): c for c in consts}
    homes = []
    if not want:
        return homes

    def look(d):
        if type(d) is dict or isinstance(d, dict):
            for key, value in d.items():
                if id(value) in want:
                    homes.append((d, key, value))

    seen = set()
    for a in args:
        if isinstance(a, torch.nn.Module):
            for m in a.modules():
                if id(m) not in seen:
                    seen.add(id(m))
                    look(m._parameters)
                    look(m._buffers)
                    look(m.__dict__)
        elif not isinstance(a, torch.Tensor):
            look(getattr(a, "__dict__", None))
    return homes


def _tree_holders(args, plan):
    """The containers, pytree nodes and functions made of parts (see
    _made_of; with their cells) inside the args a region reads as trees (with
    the nodes' own ``__dict__``): what `func` checks by structure and leaf on
    every call."""
    out = []
    seen: set[int] = set()
    todo = [a for a, p in zip(args, plan) if p is not None and p[0] == "tree"]
    while todo:
        x = todo.pop()
        t = type(x)
        if id(x) in seen:
            continue
        seen.add(id(x))
        if t is list or t is tuple:
            todo.extend(x)
        elif t is dict:
            todo.extend(x.values())
        elif (made := _made_of(x)) is not None:
            # The body reads the cells: the region is handed what they hold.
            if t is types.FunctionType:
                out.extend(x.__closure__ or ())
            todo.extend(made[1])
        elif _is_tree(x):
            node = pytree.SUPPORTED_NODES.get(t)
            attrs = getattr(x, "__dict__", None)
            if attrs is not None:
                out.append(attrs)
            todo.extend(node.flatten_fn(x)[0] if node is not None else _flatten(x)[0])
        else:
            continue
        if t is not tuple:
            out.append(x)
    return out


# The tensors the traced regions gave the program, by id: (a weak reference,
# its region's name, which output). An input of a later trace that is one of
# them says so in its type (TensorType.made_by). The references have no
# callback: one would run, as Python code, at whichever instruction of the
# program drops the tensor.
_made: dict[int, tuple[Any, str, int]] = {}


def _made_by(x: torch.Tensor) -> tuple[str, int] | None:
    entry = _made.get(id(x))
    return entry[1:] if entry is not None and entry[0]() is x else None


def _input_types(args, plan, types, consts):
    """torch.vm Types of the graph's inputs: tensor args' from what the VM
    recorded (``types``, by arg), container leaves', floats' and constants'
    exact."""
    from torch.vm import _types

    out = []
    floats = []
    for a, p, t in zip(args, plan, types or [None] * len(args)):
        if p is None:
            continue
        if p[0] == "tensor":
            out.append(_types.from_rust(t, a))
            out[-1].made_by = _made_by(a)
            continue
        if p[0] == "float":
            floats.append(_types.FloatType(p[1]))
            continue
        if p[0] == "equal":
            continue
        leaves: list[Any] = []
        _walk(a, leaves, [])
        for x in leaves:
            if isinstance(x, torch.Tensor):
                out.append(_types.exact(x))
                out[-1].made_by = _made_by(x)
        floats.extend(_types.FloatType(c[1]) for c in p[2] if c[0] == _FLOAT)
    return out + floats + [_types.exact(c, const=True) for c in consts]


def _compile(name, gm, example, consts, args, plan, types):
    from torch._inductor.compile_fx import compile_fx

    if block_pipeline is None:
        key = (
            gm.code,
            tuple(_meta(t) for t in example + consts),
            torch.is_grad_enabled(),
        )
        if key not in _compiled:
            _compiled[key] = compile_fx(gm, example + consts)
            stats["compiled"] += 1
        else:
            stats["reused"] += 1
        return _compiled[key]
    split = (
        _split_host(gm, len(example))
        if any(n.target in _SKIP_OPS for n in gm.graph.nodes)
        else None
    )
    if split is not None:
        prologue, gm, floats = split
        k = len(example)
        types = _input_types(args, plan, types, consts)
        rest = _compile_vm(
            name,
            gm,
            example + floats,
            consts,
            types[:k] + [_vm_types().FloatType() for _ in floats] + types[k:],
        )

        def run(*inputs):
            return rest(*inputs[:k], *prologue(*inputs), *inputs[k:])

        return run
    return _compile_vm(
        name, gm, example, consts, _input_types(args, plan, types, consts)
    )


def _vm_types():
    from torch.vm import _types

    return _types


def _compile_vm(name, gm, example, consts, input_types):
    # Regions with the same graph and input metadata share the passes' output,
    # unless a pass read a weight's value. Those traced in the same autocast
    # state: a lowering compiles for the state it is called in (AOT Autograd
    # turns autocast off around its backward only if it compiled under it).
    key = (
        block_pipeline,
        gm.code,
        tuple(
            _meta(t) if isinstance(t, torch.Tensor) else float for t in example + consts
        ),
        torch.is_grad_enabled(),
        autocast_on().raw_repr(),
    )
    own = (*key, tuple(map(id, consts)))
    for k in (key, own):
        if k in _compiled:
            stats["reused"] += 1
            return _compiled[k][0]
    compiled, const_read = block_pipeline(name, gm, example + consts, input_types)
    # Keeps the constants alive while their ids key the entry.
    _compiled[own if const_read else key] = (compiled, consts)
    stats["compiled"] += 1
    return compiled


def check(name: str, mine, theirs) -> None:
    """Debugging: compares a region's outputs with its version's (see
    TORCH_RUST_REGION_CHECK)."""

    def same(a, b, where):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                return (
                    f"{where}: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
                )
            if not torch.allclose(
                a.detach().float(), b.detach().float(), rtol=1e-3, atol=1e-3
            ):
                return f"{where}: values differ by {(a.detach().float() - b.detach().float()).abs().max().item()}"
            return None
        if type(a) is not type(b):
            return f"{where}: {type(a).__name__} vs {type(b).__name__}"
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return f"{where}: length {len(a)} vs {len(b)}"
            return next(
                (
                    m
                    for m in (
                        same(x, y, f"{where}[{i}]")
                        for i, (x, y) in enumerate(zip(a, b))
                    )
                    if m
                ),
                None,
            )
        if isinstance(a, dict):
            if list(a) != list(b):
                return f"{where}: keys {list(a)} vs {list(b)}"
            return next(
                (m for m in (same(a[k], b[k], f"{where}[{k!r}]") for k in a) if m), None
            )
        if type(a) in _IMMUTABLE and a != b:
            return f"{where}: {a!r} vs {b!r}"
        if type(a) in _ITERATORS or type(a) in (enumerate, zip):
            return None
        if type(a) not in _IMMUTABLE and a is not b and not _is_tree(a):
            return f"{where}: different {type(a).__name__} objects: {getattr(a, '__qualname__', a)!r} closure={getattr(a, '__closure__', None) is not None}"
        return None

    for i, (a, b) in enumerate(zip(mine, theirs)):
        m = same(a, b, f"output {i}")
        if m:
            print(f"region {name}: MISMATCH {m}", flush=True)


def _span_checks(checks):
    """A span's checks (no guard makes them) as (kind, arg, expected); its
    tensor args checked in one TreeChecks call."""
    out: list[tuple[str, Any, Any]] = []
    tensors: list[tuple[int, torch.Tensor]] = []
    for c in checks:
        kind = c[0]
        if kind == "grad_enabled":
            out.append((kind, None, c[1]))
        elif kind == "null":
            out.append((kind, c[1], None))
        elif kind == "tensor":
            tensors.append((c[1], c[2]))
        elif kind == "value":
            out.append((kind, c[1], (type(c[2]), c[2])))
        elif kind == "ints":
            out.append((kind, c[1], tuple(c[3])))
        elif kind in ("is", "set"):
            out.append((kind, c[1], c[2]))
        elif kind != "requires_grad":
            raise _Refused(f"a span can't check {kind}")
    if tensors:
        out.append(
            (
                "tensors",
                tuple(i for i, _ in tensors),
                _rust().TreeChecks(tuple(t for _, t in tensors)),
            )
        )
    return out


def _quick(args, plan, own_checks):
    """What the checks of a region's args imply that the executor can ask
    before anything else of the region runs (region::call), so that a place
    with many regions (one per layer of a loop, one per input size) passes
    over the others for nanoseconds each: ("arg", i, by identity, what) and
    ("tensors", args, TreeChecks) from a span's own checks, and ("item", i,
    j, by identity, what) for the items of an exact tuple or list arg that
    are leaves checked by value or by identity, or ("tree", i, TreeChecks)
    for a container arg of tensors and plain values, whole (its tensors'
    metadata too: a region for another input size held in a tuple is passed
    over, not entered and turned down). Each is one of the checks func makes
    anyway, so a region passed over here is one func would turn down."""
    simple = (int, bool, str, type(None))
    out: list[tuple[Any, ...]] = []
    for kind, i, want in own_checks:
        if kind == "is":
            out.append(("arg", i, True, want))
        elif kind == "value" and want[0] in simple:
            out.append(("arg", i, False, want[1]))
        elif kind == "tensors":
            out.append(("tensors", list(i), want))
    for i, (a, p) in enumerate(zip(args, plan)):
        if p is None or p[0] != "tree":
            continue
        if p[3] is not None and p[3].quick:
            out.append(("tree", i, p[3]))
            continue
        if type(a) not in (tuple, list):
            continue
        for j, x in enumerate(a):
            if type(x) in simple:
                out.append(("item", i, j, False, x))
            elif (
                type(x) not in (list, tuple, dict)
                and _made_of(x) is None
                and not _is_tree(x)
                and _leaf_check(x, "")[0] is _IS
            ):
                out.append(("item", i, j, True, x))
    return out


def _keys(args, plan, checks):
    """What the checks of a region's args pin to one object or to one small
    value, as (arg, item or None, what): an arg checked by identity, an int,
    bool, None or str arg checked by value, a bound method whose ``__self__``
    is checked by identity, and the items of an exact tuple or list arg that
    are any of these. A place with many regions
    (one per layer of a loop) finds the one for a call by these (see
    exec/region.rs's Variants) instead of asking each. Every entry is implied
    by a check the region makes anyway (the guard's, a quick one, func's), so
    a call that differs in one is a call the region would turn down."""
    simple = (int, bool, str, type(None))

    def by_identity(x) -> bool:
        return (
            type(x) not in (list, tuple, dict)
            and _made_of(x) is None
            and not _is_tree(x)
            and _leaf_check(x, "")[0] is _IS
        )

    def pinned(x) -> bool:
        # A bound method made anew on every read counts by what it is bound
        # to (the key takes a method object that way).
        return (
            type(x) in simple
            or by_identity(x)
            or (type(x) is types.MethodType and by_identity(x.__self__))
        )

    out: list[tuple[int, int | None, Any]] = []
    for c in checks:
        if c[0] == "is" or (c[0] == "value" and type(c[2]) in simple):
            out.append((c[1], None, c[2]))
    for i, (a, p) in enumerate(zip(args, plan)):
        if p is None:
            continue
        if p[0] == "equal" and p[1] is str:
            out.append((i, None, a))
        elif p[0] == "tree" and type(a) is types.MethodType:
            if by_identity(a.__self__):
                out.append((i, None, a))
        elif p[0] == "tree" and type(a) in (tuple, list):
            out.extend((i, j, x) for j, x in enumerate(a) if pinned(x))
    return out


def _dropped(func) -> None:
    """The region whose function is ``func`` was dropped by its place (see
    exec/region.rs's Variants::room): it never ran, and neither did the
    regions traced after it, each for another object or value where the
    place tells its regions apart. Its trace is no region any more, and what
    the trace read is let go."""
    record = func.record
    record.state = "refused"
    record.why = "it is found by which object (or value) an argument is, and that was another on every call: its regions never ran, so they were dropped and the place takes no new one"
    record.reads = None
    stats["refused"] += 1


def _span_holds(checks, args) -> bool:
    for kind, i, want in checks:
        if kind == "grad_enabled":
            ok = torch.is_grad_enabled() == want
        elif kind == "null":
            ok = args[i] is NULL
        elif kind == "tensors":
            ok = want.leaves(tuple(args[j] for j in i)) is not None
        elif kind == "value":
            ok = type(args[i]) is want[0] and args[i] == want[1]
        elif kind == "ints":
            ok = type(args[i]) is tuple and args[i] == want
        elif kind == "set":
            ok = type(args[i]) is set and args[i] == want
        else:
            ok = args[i] is want
        if not ok:
            return False
    return True


def _shared_container(outs) -> bool:
    """Whether a list, dict or set is reachable twice from outs (a region
    makes each anew, so they'd no longer be the same)."""
    seen: set[int] = set()
    todo = list(outs)
    while todo:
        x = todo.pop()
        if isinstance(x, (list, dict, set)):
            if id(x) in seen:
                return True
            seen.add(id(x))
        if isinstance(x, (list, tuple, set)):
            todo.extend(x)
        elif isinstance(x, dict):
            todo.extend(x.values())
        elif (names := _init_fields(x)) is not None:
            todo.extend(getattr(x, n) for n in names)
    return False


def trace_region(
    name: str,
    args: tuple[Any, ...],
    unused: list[int],
    span: bool = False,
    types: list[Any] | None = None,
):
    """Runs the block being traced (via torch._rust._region_body) and returns
    (func, outputs, objects, checks, reads, quick checks, keys) for its
    region, or None. A None before
    the body ran leaves running it to the executor. A span's region (`span`)
    makes the checks itself. ``types``: what the VM recorded at each arg over
    the runs before (see torch.vm.Type)."""
    _region_body, _region_reads = _rust()._region_body, _rust()._region_reads

    allow = os.environ.get("TORCH_RUST_REGION_ALLOW")
    if allow is not None and not re.search(allow, name):
        return None
    try:
        checks, plan = _plan(
            args, set(unused), span, scalars=block_pipeline is not None
        )
        example = _inputs(args, plan)
        own_checks = _span_checks(checks) if span else []
    except _Refused as e:
        stats["refused"] += 1
        _traced(name, "refused", str(e))
        if DEBUG:
            print(f"region {name}: refused: {e}")
        return None
    if example is None:
        raise AssertionError(f"region {name}: the inputs don't match their own trace")
    if os.environ.get("TORCH_RUST_REGION_DEBUG") == "2":
        kinds = [
            p[0] if p else ("unread" if i in unused else "checked")
            for i, p in enumerate(plan)
        ]
        print(
            f"region {name}: args {[(type(a).__name__, k) for a, k in zip(args, kinds)]}"
        )

    result: dict[str, Any] = {}
    Backward = torch.autograd.function.BackwardCFunction
    if "__init__" in Backward.__dict__:
        raise AssertionError("expected BackwardCFunction to inherit __init__")

    pinned = {c[1] for c in checks if c[0] == "is"}
    unpinned = _old_containers([a for i, a in enumerate(args) if i not in pinned])
    # By id: whether only args the guard pins hold it.
    old: dict[int, Any] = {
        k: k not in unpinned for k in _old_containers([args[i] for i in pinned])
    }
    old.update(dict.fromkeys(unpinned, False))
    old.update(_paths(args, plan))
    n_floats = sum(1 for p in plan if p is not None and p[0] == "float") + sum(
        1
        for p in plan
        if p is not None and p[0] == "tree"
        for c in p[2]
        if c[0] == _FLOAT
    )
    n_tensors = len(example) - n_floats
    backups: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    locs: list[Any] = []
    state = {"specialized": set()}

    in_vm = block_pipeline is not None
    _skip_tracer_frames()
    # The trace records below autocast: the graph holds the casts autocast
    # made, and what an operator autocast treated ran with it off. Run with
    # autocast on, its operators would be treated a second time: func runs it
    # with the autocast keys that are on now turned off. (A region is only
    # entered in the autocast state of its trace: see exec/reads.rs.)
    autocast = autocast_on()
    below_autocast = bool(autocast.raw_repr())

    def body(*xs):
        global _scalar_state, _trace_backups
        # make_fx tracks the input tensors by identity, wherever args hold them;
        # floats are swapped for _Scalars of their placeholders.
        call_args, restore = args, []
        if in_vm:
            _scalar_state = state
        if n_floats:
            scalars = iter(
                [
                    _new_scalar(v, ("input", k), frozenset([k]))
                    for k, v in enumerate(xs[n_tensors:])
                ]
            )
            call_args = tuple(
                _substitute(a, scalars, restore, [0], locs, (i, p[0] == "tree"))
                if _has_floats(p)
                else a
                for i, (a, p) in enumerate(zip(args, plan))
            )
        try:
            before = _contents(call_args)
            scalar_args = (
                _ScalarArgs(_tracer().graph) if in_vm else contextlib.nullcontext()
            )
            with scalar_args, _RefuseImpure(backups):
                _trace_backups = backups
                try:
                    outs = _region_body(*call_args)
                finally:
                    _trace_backups = None
            # The executor goes on from where the body left off: what it did
            # stands even if no region comes of it.
            backups.clear()
            # A comprehension's loop body adds to a container it was given
            # (which purity doesn't see): a region would skip that.
            for x, items in before.values():
                if _contents([x])[id(x)][1] != items:
                    result["refused"] = f"it changes a {type(x).__name__} it was given"
                    return []
            tensor_outs: list[Any] = []
            objects: list[Any] = []
            if span and _shared_container(outs):
                result["refused"] = "its outputs share a container"
                return []
            try:
                result["outputs"] = [
                    _template(o, call_args, tensor_outs, objects, old) for o in outs
                ]
            except _Refused as e:
                result["refused"] = str(e)
                return []
            result["objects"] = objects
            result["tensors"] = tensor_outs
            return tensor_outs
        finally:
            _scalar_state = None
            for container, k, v in reversed(restore):
                container[k] = v

    Backward.__init__ = _refusing_init
    torch.autograd.Function.apply = classmethod(_tracing_apply)
    try:
        # An explicit (empty) table: without one make_fx turns `detach` into
        # an alias, and the region's graph, which autograd differentiates
        # again when it runs, would let gradients through where the program
        # cut them.
        gm = make_fx(body, decomposition_table={}, tracing_mode="real")(*example)
    except Exception as e:
        # The executor runs the block again from its start.
        with torch.no_grad():
            for t, saved in backups.values():
                t.copy_(saved)
        if isinstance(e, _Refused):
            _traced(name, "refused", str(e))
        else:
            _traced(name, "gave up", f"{type(e).__name__}: {e}")
        raise
    finally:
        del Backward.__init__
        torch.autograd.Function.apply = _function_apply
    stats["traced"] += 1
    # What the body read from outside its args (see exec/reads.rs).
    reads, unchecked = _region_reads()
    if unchecked is not None:
        result.setdefault("refused", unchecked)
    if _DATA in state["specialized"]:
        result["refused"] = "it branches on a value it reads from a tensor"
    elif in_vm:
        _scalars_to_nodes(
            gm, list(gm.graph.find_nodes(op="placeholder"))[n_tensors:], state
        )
        for k in state["specialized"]:
            (i, leaf), value = locs[k], example[n_tensors + k]
            if leaf is None:
                plan[i] = ("float", value)
            else:
                plan[i][2][leaf] = (_FLOAT, value)
    if "refused" in result:
        stats["refused"] += 1
        _traced(name, "refused", result["refused"])
        if DEBUG:
            print(f"region {name}: refused: {result['refused']}")
        return None
    # A parameter only dead code reads would still be an input whose
    # AccumulateGrad the backward runs (with no gradient), which DDP counts as
    # the parameter being used, unlike eager: dead code's parameters are
    # dropped (AOT lifts every parameter of the module), and an input nothing
    # reads goes in detached.
    gm.graph.eliminate_dead_code()
    gm.recompile()
    read = {n.target for n in gm.graph.find_nodes(op="get_attr")}
    for k in [*gm._parameters, *gm._buffers]:
        if k not in read:
            delattr(gm, k)
    unread = [
        i
        for i, n in enumerate(gm.graph.find_nodes(op="placeholder"))
        if not n.users and isinstance(n.meta.get("val"), torch.Tensor)
    ]
    ops = [n for n in gm.graph.nodes if n.op == "call_function"]
    if any(n.target in _SKIP_OPS and not (in_vm and _cpu_read(n)) for n in ops):
        stats["refused"] += 1
        _traced(name, "refused", "it reads a tensor's value")
        if DEBUG:
            print(f"region {name}: refused: reads a tensor's value")
        return None
    n_ops = sum(1 for n in ops if n.target is not operator.getitem)
    if n_ops < MIN_OPS:
        stats["small"] += 1
        _traced(name, "too small", f"{n_ops} ops, fewer than {MIN_OPS}", ops=n_ops)
        if DEBUG:
            print(f"region {name}: {n_ops} ops, too small")
        return None
    try:
        consts = _lift_constants(gm)
    except _Refused as e:
        stats["refused"] += 1
        _traced(name, "refused", str(e))
        if DEBUG:
            print(f"region {name}: refused: {e}")
        return None
    compiled = _compile(name, gm, example, consts, args, plan, types)
    bound: _Bound | None = None
    if reads is not None:
        # A constant the program binds anew (`m.seen = m.seen + y`) is no
        # longer what the body would read: its home is one of the reads,
        # whichever way the body came by it.
        for home, key, const in _const_homes(args, consts):
            reads.pin(home, key, const)
        # The containers among the args are the region's own to check, by
        # structure, on every call.
        reads.forget(_tree_holders(args, plan))
        stats["reads"] += len(reads)
        # The graph is for the constants' shapes and dtypes as well: checked
        # with the reads. Where every constant is checked, a lowering that
        # asked for it is bound to them, and hears when one's memory moved.
        bind = getattr(compiled, "bind", None)
        if reads.constants(consts) == len(consts) and bind is not None:
            bound = _Bound(bind, consts)
            reads.bound(bound.rebind)
    record = _traced(
        name,
        "region",
        ops=n_ops,
        inputs=len(example),
        constants=len(consts),
        reads=reads,
    )
    for key in [key for key, entry in _made.items() if entry[0]() is None]:
        del _made[key]
    for k, t in enumerate(result["tensors"]):
        if isinstance(t, torch.Tensor):
            _made[id(t)] = (weakref.ref(t), name, k)
    if DEBUG:
        print(
            f"region {name}: {n_ops} ops, {len(example)} inputs + {len(consts)} constants"
        )
        if reads is not None and os.environ.get("TORCH_RUST_REGION_DEBUG") == "2":
            print(f"region {name}: checks {len(reads)} reads")
            print("\n".join(f"  read: {line}" for line in reads.describe()))

    if COUNT:
        name = f"{name} {n_ops}ops"

    def func(region_args):
        if own_checks and not _span_holds(own_checks, region_args):
            record.misses["an argument"] = record.misses.get("an argument", 0) + 1
            if COUNT:
                counts[f"{name} CHECK"] = counts.get(f"{name} CHECK", 0) + 1
            return None
        why = [] if DEBUG else None
        inputs = _inputs(region_args, plan, why)
        for i in unread if inputs is not None else ():
            inputs[i] = inputs[i].detach()
        if COUNT:
            key = name if inputs is not None else f"{name} MISS"
            counts[key] = counts.get(key, 0) + 1
        if inputs is None:
            stats["misses"] += 1
            if why is None:
                why = []
                _inputs(region_args, plan, why)
            what = why[0] if why else "an argument"
            record.misses[what] = record.misses.get(what, 0) + 1
            if DEBUG:
                print(f"region {name}: miss on {what}")
            return None
        record.runs += 1
        call, rest = (compiled, consts) if bound is None else (bound.call, ())
        if not below_autocast:
            return call(*inputs, *rest)
        with torch._C._ExcludeDispatchKeyGuard(autocast):
            return call(*inputs, *rest)

    # For _dropped.
    func.record = record  # type: ignore[attr-defined]
    quick = _quick(args, plan, own_checks)
    keys = _keys(args, plan, checks)
    return func, result["outputs"], result["objects"], checks, reads, quick, keys
