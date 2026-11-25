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
"""

import dataclasses
import functools
import logging
import operator
import os
import re
import types
from typing import Any

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, get_proxy_mode, make_fx, track_tensor_tree
from torch.utils._python_dispatch import TorchDispatchMode


log = logging.getLogger(__name__)


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

_PLAIN = (torch.Tensor, torch.nn.Parameter)
_SCALARS = (bool, int, float, type(None))
_IMMUTABLE = (bool, int, float, type(None), str, torch.dtype, torch.device, torch.layout, torch.memory_format)
_SKIP_OPS = {torch.ops.aten._local_scalar_dense.default, torch.ops.aten.item.default}
_CONTAINERS = (list, tuple, dict)
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
        tree = _tree_types[t] = issubclass(t, _CONTAINERS) or f"{t.__module__}.{t.__qualname__}" in _TREES
    return tree


def _flatten(x):
    return pytree.tree_flatten(x, is_leaf=lambda y: not _is_tree(y))


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
    elif _is_tree(x):
        more, spec = _flatten(x)
        shape.append(spec)
        leaves.extend(more)
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

stats = {"traced": 0, "compiled": 0, "reused": 0, "small": 0, "refused": 0, "misses": 0}

# Compiled graphs by (code, input metadata, grad mode): clones of a code (one
# per call site) trace the same graphs.
_compiled: dict[Any, Any] = {}


class _Refused(Exception):
    pass


# Makes autograd.Function.apply raise, before the forward runs, while a region
# is traced, unless it's _tracing_apply's: the trace would have the forward's
# ops and lose the backward, and the forward may have effects (torchrec's
# launch collectives). apply makes the Function's ctx (a BackwardCFunction)
# first.
def _refusing_init(ctx, *a, **kw):
    raise _Refused(f"it calls the autograd.Function {ctx._forward_cls.__name__}")


_function_apply = torch.autograd.Function.__dict__["apply"]
# The graph's target for each autograd.Function's apply (one per class, so
# that graphs of the same code are the same).
_appliers: dict[type, Any] = {}


def _applier(cls):
    fn = _appliers.get(cls)
    if fn is None:

        def fn(*args):
            return _function_apply.__get__(None, cls)(*args)

        fn.__name__ = fn.__qualname__ = f"apply_{cls.__name__}"
        _appliers[cls] = fn
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
    target = _applier(cls)
    Backward = torch.autograd.function.BackwardCFunction
    with disable_proxy_modes_tracing():
        del Backward.__init__
        try:
            out = target(*args)
        finally:
            Backward.__init__ = _refusing_init
    proxy = mode.tracer.create_proxy("call_function", target, tuple(pytree.tree_map(mode.tracer.unwrap_proxy, args)), {})
    return track_tensor_tree(out, proxy, constant=None, tracer=mode.tracer)


class _RefuseImpure(TorchDispatchMode):
    """Raises before an op a region can't hold runs: a collective, or one
    reading a tensor's value. The executor then runs the block as usual, which
    is only sound if nothing it did under tracing had an effect."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.namespace in ("c10d", "_c10d_functional") or func in _SKIP_OPS:
            raise _Refused(f"it runs {func}")
        return func(*args, **(kwargs or {}))


def _meta(t: torch.Tensor):
    return (type(t), t.dtype, t.device, t.shape, t.stride(), t.requires_grad)


# How func reads a leaf of a container arg: as a graph input (a tensor,
# checked by metadata), or checked by value or identity.
_TENSOR, _VALUE, _IS = 0, 1, 2


def _leaf_check(leaf, where):
    if isinstance(leaf, torch.Tensor):
        if type(leaf) not in _PLAIN:
            raise _Refused(f"{where} is a {type(leaf).__name__}")
        return (_TENSOR, _meta(leaf))
    if type(leaf) in _IMMUTABLE or isinstance(leaf, torch.Size):
        return (_VALUE, (type(leaf), leaf))
    if isinstance(leaf, (list, dict, set)):
        raise _Refused(f"{where} holds a {type(leaf).__name__}")
    return (_IS, leaf)


def _plan(args, unused, span=False):
    """What the guard checks (Rust) and what func reads and checks (Python),
    by arg: ("tensor",), ("tree", spec, leaf checks) or None (nothing). Args
    at `unused` the block doesn't read, or only passes on."""
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
        elif type(a) in _SCALARS:
            checks.append(("value", i, a))
            plan.append(None)
        elif type(a) is tuple and all(type(x) is int for x in a):
            checks.append(("ints", i, True, list(a)))
            plan.append(None)
        elif _is_tree(a):
            leaves, shape = [], []
            _walk(a, leaves, shape)
            plan.append(("tree", shape, [_leaf_check(x, f"arg {i}") for x in leaves]))
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
    containers), or None when a container doesn't match the trace (with the
    mismatch appended to `why`)."""
    inputs = []
    for i, (a, p) in enumerate(zip(args, plan)):
        if p is None:
            continue
        if p[0] == "tensor":
            inputs.append(a)
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
            else:
                ok = leaf is want
            if not ok:
                if why is not None:
                    why.append(f"arg {i} leaf {k} ({type(leaf).__name__}, checked by {('meta', 'value', 'identity')[kind]})")
                return None
    return inputs


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
    if isinstance(o, (list, dict, set)) and id(o) in old:
        # A container that was there before the run, which the version reads
        # (from an object the guard pins) rather than makes: the same one.
        # One an arg holds that no check pins (a tree's, or an unread arg's)
        # may be another next run.
        if not old[id(o)]:
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
        return ("dict", keys, [_template(v, args, tensor_outs, objects, old) for v in o.values()])
    if type(o) is enumerate:
        it, count = o.__reduce__()[1]
        objects.append(_enumerate)
        return ("call", len(objects) - 1, [_template(it, args, tensor_outs, objects, old), _template(count, args, tensor_outs, objects, old)])
    if type(o) is zip:
        objects.append(_zip)
        return ("call", len(objects) - 1, [_template(it, args, tensor_outs, objects, old) for it in o.__reduce__()[1]])
    if type(o) in _ITERATORS:
        # A loop's iterator: rebuilt over what's left of it (the trace's run
        # consumed the rest), which misses changes to the container later.
        reduced = o.__reduce__()
        start = reduced[2] if len(reduced) > 2 and reduced[2] is not None else 0
        rest = list(reduced[1][0])[start:]
        return ("iter", _template(rest, args, tensor_outs, objects, old))
    if (names := _init_fields(o)) is not None:
        objects.append(functools.partial(_dataclass, type(o), names))
        return ("call", len(objects) - 1, [_template(getattr(o, n), args, tensor_outs, objects, old) for n in names])
    if _is_tree(o):
        leaves, spec = _flatten(o)
        objects.append(functools.partial(_unflatten, spec))
        return ("call", len(objects) - 1, [_template(x, args, tensor_outs, objects, old) for x in leaves])
    if isinstance(o, types.FunctionType) and o.__code__ is _fx_wrapper_code():
        # A function fx.wrap registered, which make_fx patches while tracing.
        o = o.__wrapped__
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
            out[id(x)] = (x, [(k, id(v)) for k, v in items] if isinstance(x, dict) else [id(v) for v in items])
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
    if any(not f.init for f in dataclasses.fields(cls)) or set(getattr(o, "__dict__", ())) != set(names):
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
            raise _Refused(f"it reads {node.target} ({tuple(value.shape)}), which has a grad_fn ({value.grad_fn.name()})")
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


def _compile(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    from torch._inductor.compile_fx import compile_fx

    # aot_module_simplified's forward takes the inputs unboxed (whatever its
    # copied _boxed_call says).
    return compile_fx(gm, example_inputs)


def check(name: str, mine, theirs) -> None:
    """Debugging: compares a region's outputs with its version's (see
    TORCH_RUST_REGION_CHECK)."""

    def same(a, b, where):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                return f"{where}: {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
            if not torch.allclose(a.detach().float(), b.detach().float(), rtol=1e-3, atol=1e-3):
                return f"{where}: values differ by {(a.detach().float() - b.detach().float()).abs().max().item()}"
            return None
        if type(a) is not type(b):
            return f"{where}: {type(a).__name__} vs {type(b).__name__}"
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return f"{where}: length {len(a)} vs {len(b)}"
            return next((m for m in (same(x, y, f"{where}[{i}]") for i, (x, y) in enumerate(zip(a, b))) if m), None)
        if isinstance(a, dict):
            if list(a) != list(b):
                return f"{where}: keys {list(a)} vs {list(b)}"
            return next((m for m in (same(a[k], b[k], f"{where}[{k!r}]") for k in a) if m), None)
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
    """A span's checks (no guard makes them) as (kind, arg, expected)."""
    out = []
    for c in checks:
        kind = c[0]
        if kind == "grad_enabled":
            out.append((kind, None, c[1]))
        elif kind == "null":
            out.append((kind, c[1], None))
        elif kind == "tensor":
            out.append((kind, c[1], _meta(c[2])))
        elif kind == "value":
            out.append((kind, c[1], (type(c[2]), c[2])))
        elif kind == "ints":
            out.append((kind, c[1], tuple(c[3])))
        elif kind in ("is", "set"):
            out.append((kind, c[1], c[2]))
        elif kind != "requires_grad":
            raise _Refused(f"a span can't check {kind}")
    return out


def _span_holds(checks, args) -> bool:
    for kind, i, want in checks:
        if kind == "grad_enabled":
            ok = torch.is_grad_enabled() == want
        elif kind == "null":
            ok = args[i] is NULL
        elif kind == "tensor":
            ok = isinstance(args[i], torch.Tensor) and _meta(args[i]) == want
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
        elif _init_fields(x) is not None:
            todo.extend(getattr(x, n) for n in _init_fields(x))
    return False


def trace_region(name: str, args: tuple[Any, ...], unused: list[int], span: bool = False):
    """Runs the block being traced (via torch._rust._region_body) and returns
    (func, outputs, objects, checks) for its region, or None. A None before
    the body ran leaves running it to the executor. A span's region (`span`)
    makes the checks itself."""
    from torch._rust import _region_body

    allow = os.environ.get("TORCH_RUST_REGION_ALLOW")
    if allow is not None and not re.search(allow, name):
        return None
    try:
        checks, plan = _plan(args, set(unused), span)
        example = _inputs(args, plan)
        own_checks = _span_checks(checks) if span else []
    except _Refused as e:
        stats["refused"] += 1
        if DEBUG:
            print(f"region {name}: refused: {e}")
        return None
    if example is None:
        raise AssertionError(f"region {name}: the inputs don't match their own trace")
    if os.environ.get("TORCH_RUST_REGION_DEBUG") == "2":
        kinds = [p[0] if p else ("unread" if i in unused else "checked") for i, p in enumerate(plan)]
        print(f"region {name}: args {[(type(a).__name__, k) for a, k in zip(args, kinds)]}")

    result: dict[str, Any] = {}
    Backward = torch.autograd.function.BackwardCFunction
    if "__init__" in Backward.__dict__:
        raise AssertionError("expected BackwardCFunction to inherit __init__")

    pinned = {c[1] for c in checks if c[0] == "is"}
    unpinned = _old_containers([a for i, a in enumerate(args) if i not in pinned])
    # By id: whether only args the guard pins hold it.
    old = {k: k not in unpinned for k in _old_containers([args[i] for i in pinned])}
    old.update(dict.fromkeys(unpinned, False))
    before = _contents(args)

    def body(*tensors):
        # make_fx tracks the input tensors by identity, wherever args hold them.
        with _RefuseImpure():
            outs = _region_body(*args)
        # A comprehension's loop body adds to a container it was given (which
        # purity doesn't see): a region would skip that.
        for x, items in before.values():
            if _contents([x])[id(x)][1] != items:
                result["refused"] = f"it changes a {type(x).__name__} it was given"
                return []
        tensor_outs: list[torch.Tensor] = []
        objects: list[Any] = []
        if span and _shared_container(outs):
            result["refused"] = "its outputs share a container"
            return []
        try:
            result["outputs"] = [_template(o, args, tensor_outs, objects, old) for o in outs]
        except _Refused as e:
            result["refused"] = str(e)
            return []
        result["objects"] = objects
        return tensor_outs

    Backward.__init__ = _refusing_init
    torch.autograd.Function.apply = classmethod(_tracing_apply)
    try:
        gm = make_fx(body, tracing_mode="real")(*example)
    finally:
        del Backward.__init__
        torch.autograd.Function.apply = _function_apply
    stats["traced"] += 1
    if "refused" in result:
        stats["refused"] += 1
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
    unread = [i for i, n in enumerate(gm.graph.find_nodes(op="placeholder")) if not n.users and isinstance(n.meta.get("val"), torch.Tensor)]
    ops = [n for n in gm.graph.nodes if n.op == "call_function"]
    if any(n.target in _SKIP_OPS for n in ops):
        stats["refused"] += 1
        if DEBUG:
            print(f"region {name}: refused: reads a tensor's value")
        return None
    n_ops = sum(1 for n in ops if n.target is not operator.getitem)
    if n_ops < MIN_OPS:
        stats["small"] += 1
        if DEBUG:
            print(f"region {name}: {n_ops} ops, too small")
        return None
    try:
        consts = _lift_constants(gm)
    except _Refused as e:
        stats["refused"] += 1
        if DEBUG:
            print(f"region {name}: refused: {e}")
        return None
    key = (gm.code, tuple(_meta(t) for t in example + consts), torch.is_grad_enabled())
    compiled = _compiled.get(key)
    if compiled is None:
        compiled = _compiled[key] = _compile(gm, example + consts)
        stats["compiled"] += 1
        if DEBUG:
            print(f"region {name}: compiled {n_ops} ops, {len(example)} inputs + {len(consts)} constants")
    else:
        stats["reused"] += 1

    if COUNT:
        name = f"{name} {n_ops}ops"

    def func(region_args):
        if own_checks and not _span_holds(own_checks, region_args):
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
            if DEBUG:
                print(f"region {name}: miss on {why[0]}")
            return None
        return compiled(*inputs, *consts)

    return func, result["outputs"], result["objects"], checks
