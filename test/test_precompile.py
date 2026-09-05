# Owner(s): ["oncall: pt2"]
import ast
import base64
import functools
import hashlib
import importlib
import io
import linecache
import os
import pickle
import sys
import tempfile
import types
import typing
from unittest import mock

import torch
import torch.utils._pytree as _pytree
from torch._precompile import PrecompileError
from torch.compiler.precompile import DynamoTracer, MakeFxTracer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


def _make_tracer(tracer, tracer_kwargs):
    """Build a tracer object from the string/None spelling most of these tests
    use, routing loose per-tracer kwargs onto the right tracer. capture()'s real
    default is DynamoTracer(); these tests predate that flip, so a bare capture
    here stays make_fx (the historical default)."""
    if isinstance(tracer, (MakeFxTracer, DynamoTracer)):
        if tracer_kwargs:
            raise TypeError(
                f"tracer object passed with extra kwargs {sorted(tracer_kwargs)}"
            )
        return tracer
    if tracer in (None, "make_fx"):
        return MakeFxTracer(**tracer_kwargs)
    if tracer == "dynamo":
        return DynamoTracer(**tracer_kwargs)
    raise ValueError("tracer must be 'make_fx' or 'dynamo'")


def _load_pair(*args, artifact_path=None, cache_path=None, fn=None):
    """Load either an on-disk pair (artifact_path=/cache_path=, the real load API)
    or an in-memory (python_code, cache) pair, staging the two halves to files
    since torch.compiler.precompile.load is paths-only."""
    if artifact_path is not None:
        return torch.compiler.precompile.load(artifact_path, cache_path, fn=fn)
    code, cache = args
    d = tempfile.mkdtemp()
    ap = os.path.join(d, "artifact.py")
    cp = os.path.join(d, "artifact.cache")
    with open(ap, "w", encoding="utf-8") as f:
        f.write(code)
    with open(cp, "wb") as f:
        f.write(cache)
    return torch.compiler.precompile.load(ap, cp, fn=fn)


# A module-level (global) model + a function referencing it, to exercise the
# constant-tensor guard against a baked global.
_GLOBAL_TENSOR = torch.randn(3)

# A plain scalar global folded into the output must be baked by the dynamo
# tracer, not left dangling as an uncovered external reference.
_GLOBAL_SCALE = 10


class _PrecompileTwoBreakModule(torch.nn.Module):
    """Two breaks, so a capture yields three frames and the varying dim reaches
    every one of them."""

    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        z = y + 1
        torch._dynamo.graph_break()
        return z.sum()


class _PrecompileLateVaryingModule(torch.nn.Module):
    """``x`` never varies; ``z`` does, and is only read AFTER the break, so only
    the continuation should be promoted to dynamic."""

    def forward(self, x, z):
        y = x * 2
        torch._dynamo.graph_break()
        return y.sum() + z.sum()


_PRECOMPILE_FIXED_INPUT = torch.randn(4)

# Rows: model, the example call for a batch size, per-frame variant counts.
_AUTO_DYNAMIC_CASES = {
    "every_frame": (_PrecompileTwoBreakModule, lambda n: (torch.randn(n),), [2, 2, 2]),
    "only_the_frame_that_varied": (
        _PrecompileLateVaryingModule,
        lambda n: (_PRECOMPILE_FIXED_INPUT, torch.randn(n)),
        [1, 2],
    ),
}


class _PrecompileBreakingModule(torch.nn.Module):
    """One graph break, so a capture yields an entry frame and a continuation."""

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(8, 4)

    def forward(self, x):
        y = self.l(x)
        torch._dynamo.graph_break()
        return y.sum() + 1


def _precompile_unreachable_helper(y):
    z = y * 3
    torch._dynamo.graph_break()
    return z.sum()


def _precompile_unreachable_helper_caller_with_opts(x, opts):
    return _precompile_unreachable_helper_caller(x) * opts["scale"]


def _compiled_subgraph_count(code: str) -> int:
    """Compiled subgraphs a dynamo artifact carries, in either serving mode."""
    from torch._precompile import _read_literal

    tree = ast.parse(code)
    if _read_literal(tree, "SERVING_MODE") == "installed":
        package = pickle.loads(base64.b64decode(_read_literal(tree, "_PACKAGE")))
        return len(package.backends)
    pickled = pickle.loads(base64.b64decode(_read_literal(tree, "_BACKENDS")))
    return len(pickled) + code.count("_SUBGRAPHS[")


def _precompile_unreachable_helper_caller(x):
    return _precompile_unreachable_helper(x * 2)


def _precompile_closure_entry_factory():
    scale = 3.0

    def entry(x):
        return _precompile_unreachable_helper_caller(x) * scale

    return entry


def _precompile_closure_lambda_factory():
    scale, cfg = 3.0, {"bias": 1.0}
    return lambda model, xx: model(xx) * scale + cfg["bias"]


# Entries that close over a cell, with the example call each takes: a standalone
# lambda and an entry over an un-inlinable helper (the installed shape).
_PRECOMPILE_CLOSURE_ENTRIES = {
    "lambda": (
        _precompile_closure_lambda_factory,
        lambda: (torch.nn.Linear(4, 3).eval(), torch.randn(5, 4)),
    ),
    "installed_entry": (
        _precompile_closure_entry_factory,
        lambda: (torch.randn(4),),
    ),
}


@torch._dynamo.disable
def _brk_disabled_fn(t):
    """A disabled callee: calling it breaks the graph (gb0098)."""
    return t * 1.0


class _BrkEagerHelper:
    @torch.compiler.disable
    def helper(self, t):
        """A disabled METHOD: reported differently from a free function."""
        return t + 0.0


_BRK_HELPER = _BrkEagerHelper()


class _BrkDisabledCallee(torch.nn.Module):
    """Break from calling a torch._dynamo.disable'd function."""

    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(4, 4)

    def forward(self, x):
        h = self.l(x)
        h = _brk_disabled_fn(h)
        return self.l(h).sum()


class _BrkDisabledMethod(torch.nn.Module):
    """Break from a torch.compiler.disable'd bound method."""

    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(4, 4)

    def forward(self, x):
        return _BRK_HELPER.helper(self.l(x)).sum()


class _BrkExplicit(torch.nn.Module):
    """Two explicit graph breaks, so there are two continuations in one frame."""

    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(4, 4)

    def forward(self, x):
        h = self.l(x)
        torch._dynamo.graph_break()
        h = h * 2
        torch._dynamo.graph_break()
        return h.sum()


class _BrkDataDependent(torch.nn.Module):
    """A break from .item(), the classic data-dependent one."""

    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(4, 4)

    def forward(self, x):
        h = self.l(x)
        scale = h.abs().max().item()
        return (h * scale).sum()


class _BrkNested(torch.nn.Module):
    """The break lives in a CHILD module, so the artifact must install."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = _BrkDisabledCallee()

    def forward(self, x):
        return self.inner(x) * 2


class _BrkInLoop(torch.nn.Module):
    """A break inside a loop -- Dynamo may skip the whole frame to eager."""

    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(4, 4)

    def forward(self, x):
        acc = x
        for _ in range(3):
            acc = self.l(acc)
            acc = _brk_disabled_fn(acc)
        return acc.sum()


class _BrkBranchy(torch.nn.Module):
    """Recompiles on a bool flag AND breaks, so variants x continuations."""

    def __init__(self) -> None:
        super().__init__()
        self.l = torch.nn.Linear(4, 4)

    def forward(self, x, flag):
        h = self.l(x)
        h = _brk_disabled_fn(h)
        return (h * 3).sum() if flag else (h + 1).sum()


_BREAKING_MODELS = {
    "disabled_fn": _BrkDisabledCallee,
    "disabled_method": _BrkDisabledMethod,
    "explicit_breaks": _BrkExplicit,
    "data_dependent": _BrkDataDependent,
    "nested_child": _BrkNested,
    "break_in_loop": _BrkInLoop,
}


def _maybe_scoped(loaded):
    """Installed artifacts scope their install; standalone ones have nothing to."""
    import contextlib

    return loaded if loaded.installed else contextlib.nullcontext()


class _PrecompileUnguardedAttr(torch.nn.Module):
    """Holds an interned value no guard reads -- the pruning-collision shape."""

    def __init__(self, junk) -> None:
        super().__init__()
        self.l = torch.nn.Linear(8, 8)
        self.junk = junk

    def forward(self, x):
        return self.l(x).relu().sum()


class _PrecompilePipeline:
    """A guarded object that is NOT an nn.Module and holds unpicklable state."""

    def __init__(self, model) -> None:
        self.model = model
        self.it = (n for n in range(3))


def _precompile_via_pipeline(pipeline, x):
    return pipeline.model(x).relu().sum()


def _precompile_scale_sum(x):
    return torch.relu(x * 2.0).sum()


_PRECOMPILE_PUBLIC_METHODS = [
    name
    for name in dir(torch.compiler.precompile)
    if not name.startswith("_") and callable(getattr(torch.compiler.precompile, name))
]


def _brk_call(model, x):
    return model(x)


def _brk_call_flag(model, x, flag):
    return model(x, flag)


class _PrecompilePlusOneMode(torch.overrides.TorchFunctionMode):
    """Adds one to a scalar addend, so a doubly-applied mode is visible."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.add and not isinstance(args[1], torch.Tensor):
            return func(args[0], args[1] + 1, **kwargs)
        return func(*args, **kwargs)


def _precompile_add_one(xx):
    return torch.add(xx, 1.0)


def _precompile_scaled_model(model, xx, k=1.0):
    return model(xx) * k


# Argument-validation errors a capture raises before any call runs.
_PRECOMPILE_BAD_CALLS = {
    "capture_option_needs_dynamo": (
        TypeError,
        "unexpected keyword argument 'dynamic'",
        (_precompile_add_one,),
        {"dynamic": False},
    ),
    "partial": (
        PrecompileError,
        "cannot capture a partial",
        (functools.partial(_precompile_scaled_model, k=3.0),),
        {"tracer": "dynamo"},
    ),
}


def _precompile_with_defaults(model, xx, scale=2.0, bias=1.0, *, gain=1.0):
    return (model(xx) * scale + bias) * gain


class _PrecompileTiedWeights(torch.nn.Module):
    """Two Linears sharing one weight tensor, for the tied-weight round trip."""

    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(4, 4, bias=False)
        self.b = torch.nn.Linear(4, 4, bias=False)
        self.b.weight = self.a.weight

    def forward(self, xx):
        return self.b(self.a(xx))


class _PrecompileFoldsAGlobal(torch.nn.Module):
    """fn IS the module, and its forward folds in a module-level constant."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)

    def forward(self, xx):
        return self.lin(xx), _GLOBAL_SCALE


def _precompile_two_modules(ma, mb, xx):
    return ma(xx) + mb(xx)


def _precompile_call_model(model, x):
    return model(x)


# Round-trip entries: the entry (None when the module itself is the entry), the
# modules it is called with, and the input shape.
_PRECOMPILE_ROUNDTRIP_CASES = {
    "sequential": (
        _precompile_call_model,
        lambda: (
            torch.nn.Sequential(
                torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3)
            ).eval(),
        ),
        (5, 4),
    ),
    "two_modules": (
        _precompile_two_modules,
        lambda: (torch.nn.Linear(4, 3).eval(), torch.nn.Linear(4, 3).eval()),
        (5, 4),
    ),
    "tied_weights": (
        _precompile_call_model,
        lambda: (_PrecompileTiedWeights().eval(),),
        (5, 4),
    ),
    # The spelling the bare-module refusal points at.
    "module_behind_a_function": (
        _brk_call,
        lambda: (torch.nn.Linear(8, 4).eval(),),
        (3, 8),
    ),
    # When fn IS an nn.Module, Dynamo traces fn.forward, whose globals live in the
    # traced code's f_globals -- not fn.__globals__ (an nn.Module has none); a
    # folded module-level constant must still reach the artifact from there.
    "module_is_the_entry": (
        None,
        lambda: (_PrecompileFoldsAGlobal().eval(),),
        (5, 4),
    ),
}


def _precompile_out_nested(model, xx):
    y = model(xx)
    return y, y * 2, {"k": y + 1}


def _global_helper_with_attr():
    return None


# Pickled BY REFERENCE, so the attached tensor is NOT baked and must not be rejected.
_global_helper_with_attr.cache = torch.randn(3)


# Output structures the transformed bytecode reassembles; the non-tensor values
# are baked as defaults.
_PRECOMPILE_OUTPUT_SHAPES = {
    "global_constant": lambda model, xx: (model(xx), _GLOBAL_SCALE),
    "nested_multi_tensor": _precompile_out_nested,
    "float": lambda model, xx, b=3.14: (model(xx), b),
    "complex": lambda model, xx, b=2 + 3j: (model(xx), b),
    "str": lambda model, xx, b="hi": (model(xx), b),
    "by_reference_callable": lambda model, xx: (model(xx), _global_helper_with_attr),
}


class _PrecompileTrainMod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(8, 8)
        self.b = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.b(torch.relu(self.a(x))))


# Training captures: model, entry, input width; the second row breaks the graph.
_PRECOMPILE_TRAINING_CASES = {
    "plain": (_PrecompileTrainMod, _precompile_call_model, 8),
    "across_a_graph_break": (_BrkDisabledCallee, _brk_call, 4),
}


class _PrecompileUnpicklableHolder:
    def __init__(self, bad):
        self.bad = bad


def _precompile_reads_holder(obj, x):
    return x * 2 if obj.bad is not None else x


def _precompile_reads_holder_in_list(objs, y):
    return y * 2 if objs[0].bad is not None else y


class _PrecompileClassA:
    def f(self, x):
        return x * 2


class _PrecompileClassB:
    def f(self, x):
        return x * 100


def _precompile_calls_method(obj, x, k):
    return obj.f(x) + k


class _PrecompileStepCounter(torch.nn.Module):
    """Its own forward advances a value the guards will be built from."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.step = 0

    def forward(self, x):
        self.step += 1
        return self.lin(x) * self.step


def _precompile_scaled(x, k):
    return x * k


def _precompile_branchy(x, flag):
    if flag:
        return (x * 2).sum()
    return (x + 1).sum()


_PRECOMPILE_CLASS_A = _PrecompileClassA()

_PRECOMPILE_X4 = torch.randn(4)

_PRECOMPILE_X28 = torch.randn(2, 8)

# A guard whose value never varied discriminates nothing and is not serialized --
# EXCEPT the ones that pin a value, a shape or a type, which are kept regardless.
# Rows: entry, example calls, calls the artifact serves, calls it must refuse.
_PRECOMPILE_GUARD_POLICY_CASES = {
    # k pins a value, so it is checked even though it never varied.
    "value_pin": (
        _precompile_scaled,
        [(torch.randn(3), 2), (torch.randn(5), 2)],
        [(torch.randn(3), 2)],
        [(torch.randn(3), 5)],
    ),
    # A value that DID vary selects between the variants, and both serve.
    "discriminating_flag": (
        _precompile_branchy,
        [(_PRECOMPILE_X4, False), (_PRECOMPILE_X4, True)],
        [(_PRECOMPILE_X4, False), (_PRECOMPILE_X4, True)],
        [],
    ),
    # With ONE example nothing can discriminate, so the rule would drop every
    # input guard; shape-bearing guards are therefore never policy-dropped.
    "shape_from_a_single_example": (
        _precompile_scale_sum,
        [(_PRECOMPILE_X28,)],
        [(_PRECOMPILE_X28,)],
        [
            (torch.randn(3, 8),),
            (torch.randn(2, 9),),
            (torch.randn(16),),
            (torch.randn(2, 8, dtype=torch.float64),),
        ],
    ),
    # TYPE_MATCH used to be droppable: a graph traced for one class was served
    # another. Two variants differing only in k, so obj's type never varies.
    "type_match": (
        _precompile_calls_method,
        [
            (_PRECOMPILE_CLASS_A, _PRECOMPILE_X4, 1.0),
            (_PRECOMPILE_CLASS_A, _PRECOMPILE_X4, 2.0),
        ],
        [(_PRECOMPILE_CLASS_A, _PRECOMPILE_X4, 1.0)],
        [(_PrecompileClassB(), _PRECOMPILE_X4, 1.0)],
    ),
}


class _PrecompileOptsModule(torch.nn.Module):
    """Branches on a module-owned dict, so the membership guard is environment-rooted."""

    def __init__(self, opts):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.opts = opts

    def forward(self, x):
        y = self.lin(x)
        if "flag" in self.opts:
            return y * 2
        return y * 100


def _precompile_dict_flag_branch(x, d):
    if "flag" in d:
        return x * 2
    return x * 100


def _precompile_only_disabled(x):
    return _brk_disabled_fn(x)


_PRECOMPILE_GRAD_MODES_SEEN: list[bool] = []


def _precompile_observe_grad_mode(model, x):
    _PRECOMPILE_GRAD_MODES_SEEN.append(torch.is_grad_enabled())
    return model(x)


def _precompile_multi_graph(x):
    x = x * 2
    torch._dynamo.graph_break()
    x = x + 3
    torch._dynamo.graph_break()
    return x.sum()


def _precompile_empty_resume(x, flag):
    y = x + 1
    torch._dynamo.graph_break()
    if flag:
        return y
    return y.cos() * 100


def _precompile_single_graph(x):
    return x.sin()


def _precompile_module_arg(module, x):
    return module(x)


def _precompile_raises_on_flag(x, fail):
    if fail:
        raise KeyError("automatic example failed")
    return x + 1


# A custom pytree node whose context (a set) is not JSON-dumpable and which has no
# to_dumpable_context serializer, so treespec_dumps raises TypeError (distinct from the
# unregistered-namedtuple NotImplementedError path). Registered once at module load and
# used by test_unserializable_context_in_spec_still_compiles.
class _UnserializableCtxInput:
    def __init__(self, a, b):
        self.a = a
        self.b = b


_pytree.register_pytree_node(
    _UnserializableCtxInput,
    lambda n: ([n.a, n.b], {"ctx"}),
    lambda children, _ctx: _UnserializableCtxInput(children[0], children[1]),
    serialized_type_name="test_precompile._UnserializableCtxInput",
)


# An eager-backend graph is carried as a pickled GraphModule, and GraphModule's
# reduction re-derives the Graph by symbolically re-tracing the generated source.
# cond, while_loop and vmap do not survive that (their operands reject Proxies), and
# autocast is worse: its enter/exit take no Proxy at all, so the retrace EXECUTES
# them and leaves no node behind. checkpoint DOES survive it, and is here for the
# other half of the fix -- its body is run through fx.Interpreter, so it is the case
# that proves a nested body must keep a real Graph. Each is also placed behind an
# un-inlinable helper, because the installed serving mode re-records its backends and
# only that path copies them.
def _eager_rt_cond(x):
    return torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))


def _eager_rt_while_loop(x):
    return torch._higher_order_ops.while_loop(
        lambda i, t: i < 3, lambda i, t: (i + 1, t + 1.0), (torch.tensor(0), x)
    )[1]


def _eager_rt_checkpoint(x):
    return torch.utils.checkpoint.checkpoint(
        lambda t: t.sin().cos(), x, use_reentrant=False
    )


def _eager_rt_vmap(x):
    return torch.vmap(lambda t: t * 2.0)(x)


def _eager_rt_autocast(x):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return x @ x


def _eager_rt_no_grad_region(x):
    y = x * 2.0
    with torch.no_grad():
        return y.sin()


_EAGER_ROUND_TRIP = {
    "cond": _eager_rt_cond,
    "while_loop": _eager_rt_while_loop,
    "checkpoint": _eager_rt_checkpoint,
    "vmap": _eager_rt_vmap,
    "autocast": _eager_rt_autocast,
    "no_grad_region": _eager_rt_no_grad_region,
}


def _eager_rt_helper(key, x):
    y = x * 2.0
    torch._dynamo.graph_break()
    return _EAGER_ROUND_TRIP[key](y)


def _eager_rt_broken(key, x):
    return _eager_rt_helper(key, x)


def _strip_artifact(cache: bytes) -> bytes:
    """Return the cache envelope with its compiled artifact removed, forcing load()
    onto the inlined (no-cache) path that JIT-compiles from python_code. Many tests
    reload the same artifact both cache-primed and stripped to check they agree."""
    blob = torch.load(io.BytesIO(cache), weights_only=True)
    blob["artifact"] = None
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _default_and_inlined_loaders(code: str, cache: bytes, backend: str):
    """Yield (label, loaded_fn) for the load paths a backend exposes: the default
    (cache-primed) path always, plus -- on inductor only -- the inlined path that
    strips the artifact to force JIT from python_code. The eager backend has a single
    driver, so it yields the default path alone."""
    yield "default", _load_pair(code, cache)
    if backend == "inductor":
        yield "inlined", _load_pair(code, _strip_artifact(cache))


_LUT_MODULE = None


_precompile_reads_shadowed = {
    "pytype": lambda x: x * x.pytype,
    "fake_mode": lambda x: x * x.fake_mode,
    "dispatch_keys": lambda x: x * x.dispatch_keys,
    "_fake_device": lambda x: x * x._fake_device,
}


# A user global the rendered inductor source shadows with a name of its own.
BACKEND = 5.0


_DRIFT_MODULE = None


_PRECOMPILE_ACCUM_RAN: list[str] = []


_PRECOMPILE_RECURSIVE_CAPTURE: list = []


_PRECOMPILE_CLOSING_CAPTURE: list = []


# precompile drives make_fx internally, which cannot symbolically trace a
# dynamo-optimized function; the whole suite is therefore incompatible with
# PYTORCH_TEST_WITH_DYNAMO (dynamo_wrapped CI), so skip it there.
def _multigraph_step(m, x, scale=2.0):
    y = m(x)
    torch._dynamo.graph_break()
    return y * scale


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompile(TestCase):
    def test_load_invalid_python_code_rejected(self):
        # load() surfaces a clear PrecompileError (not a raw SyntaxError) when
        # python_code is not valid Python.
        buf = io.BytesIO()
        torch.save({"artifact": None}, buf)
        with self.assertRaisesRegex(PrecompileError, "not valid Python"):
            _load_pair("def (:::", buf.getvalue())

    def test_summary_types_pickle(self):
        # A capture summary or invariants report is the kind of value users
        # stash next to an artifact (torch.save of a diagnostics record, a
        # multiprocessing capture farm). A previous revision pointed these
        # classes' __module__ at torch.compiler, which does not export them,
        # so pickle could not resolve the class and every instance raised.
        from torch.compiler._precompile_types import (
            FrameInvariants,
            GuardFact,
            PrecompileSummary,
        )

        fact = GuardFact("TYPE_MATCH", "L['x']", ("code",), "is int", True)
        inv = FrameInvariants("f", "f.py", 1, 2, (fact,), (), ())
        summary = PrecompileSummary(1, 0, 1, 1, ())
        for obj in (fact, inv, summary):
            self.assertEqual(pickle.loads(pickle.dumps(obj)), obj)

    @parametrize("name", _PRECOMPILE_PUBLIC_METHODS)
    def test_precompile_public_members_resolve(self, name):
        typing.get_type_hints(getattr(torch.compiler.precompile, name))

    @staticmethod
    def _module_with(src: str, name: str):
        """A real module whose globals are exactly what the source binds."""
        mod = types.ModuleType(name)
        mod.__file__ = f"{name}.py"
        linecache.cache[mod.__file__] = (
            len(src),
            None,
            src.splitlines(True),
            mod.__file__,
        )
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        sys.modules[name] = mod
        return mod

    def _multigraph_frames(self, code):
        from torch._precompile import _parse_artifact_metadata

        return _parse_artifact_metadata(code)["FRAMES"]

    def test_no_dispatchable_graph_names_the_cause(self):
        # An entry frame with no variants has two very different causes. If
        # Dynamo BYPASSED the frame it recorded why, and saying so beats the
        # thin-wrapper advice, which in that case is simply wrong. Only the
        # ENTRY's own bypassed codes count: an unrelated bypassed helper frame
        # must not relabel a thin-wrapper entry as a bypass.
        from torch._dynamo.package import SerializedCode
        from torch._precompile import _reject_uninstallable_entry

        def fwd_loss_bwd():
            pass

        def helper():
            pass

        def bypassed_code(fn):
            return types.SimpleNamespace(
                bypassed=True,
                bypass_reason="cannot pickle 'generator' object",
                install_to_global=False,
                python_code=SerializedCode.from_code_object(fn.__code__),
            )

        entry = types.SimpleNamespace(
            fn_name="fwd_loss_bwd", codes=[bypassed_code(fwd_loss_bwd)]
        )
        # The state a bypassed ENTRY actually arrives in: _multigraph_frames
        # DROPS bypassed codes, so there is no entry frame at all -- the
        # diagnostic must fire from the empty list, not from a variant-less
        # entry frame it would never see.
        with self.assertRaisesRegex(PrecompileError, "were BYPASSED during capture"):
            _reject_uninstallable_entry([], entry)
        with self.assertRaisesRegex(PrecompileError, "cannot pickle 'generator'"):
            _reject_uninstallable_entry([], entry)
        # An entry frame that compiled but produced no variants, with a
        # bypassed sibling code of the same name, reports the bypass too.
        frames = [{"is_entry": True, "variants": []}]
        with self.assertRaisesRegex(PrecompileError, "were BYPASSED during capture"):
            _reject_uninstallable_entry(frames, entry)
        foreign = types.SimpleNamespace(
            fn_name="fwd_loss_bwd", codes=[bypassed_code(helper)]
        )
        with self.assertRaisesRegex(PrecompileError, "thin wrapper"):
            _reject_uninstallable_entry(frames, foreign)
        # No entry frame and only a FOREIGN bypassed code: neither diagnostic
        # applies, so neither may fire as a guess.
        _reject_uninstallable_entry([], foreign)
        with self.assertRaisesRegex(PrecompileError, "thin wrapper"):
            _reject_uninstallable_entry(
                frames, types.SimpleNamespace(fn_name="step", codes=[])
            )

    def test_multigraph_artifact_round_trips_a_graph_break(self):
        from torch._dynamo.package import CompilePackage, SerializedCode
        from torch._dynamo.precompile_context import EagerCacheArtifact
        from torch._dynamo.precompile_package import default_guard_filter_fn
        from torch._precompile import _build_multigraph_artifact, _multigraph_frames

        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        package = CompilePackage(
            _multigraph_step,
            explicit_capture=True,
            serialization_guard_filter_fn=default_guard_filter_fn,
        )
        compiled = torch._dynamo.optimize(backend="eager", package=package)(
            _multigraph_step
        )
        compiled(m, x)
        compiled(m, x, 3.0)
        entry = package.cache_entry()
        # The entry frame and the continuation after the graph break.
        frames = _multigraph_frames(entry)
        names = [SerializedCode.to_code_object(f["code"]).co_name for f in frames]
        self.assertEqual(len(frames), 2, names)
        backends = {
            backend_id: EagerCacheArtifact(key=backend_id, content=fn)
            for backend_id, fn in package.cached_backends.items()
        }
        summary = types.SimpleNamespace(
            dropped_guards=(),
            risky_dropped_guards=(),
            policy_dropped_guards=(),
            wont_generalize=(),
        )
        code, cache = _build_multigraph_artifact(
            entry, backends, summary, "eager", entry_fn=_multigraph_step
        )
        torch._dynamo.reset()
        f = _load_pair(code, cache)
        # The default travels with the artifact; the second variant pins 3.0.
        self.assertEqual(f(m, x), _multigraph_step(m, x))
        self.assertEqual(f(m, x, 3.0), _multigraph_step(m, x, 3.0))
        with self.assertRaisesRegex(PrecompileError, "no captured variant"):
            f(m, torch.randn(7, 4))

        # The artifact is locked to the Python it was produced on.
        current = f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}"
        self.assertIn(current, code)
        foreign = code.replace(current, "_DYNAMO_PYTHON_VERSION = (3, 99)", 1)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["code_hash"] = hashlib.sha256(foreign.encode("utf-8")).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "produced on Python 3.99"):
            _load_pair(foreign, buf.getvalue())

    def test_missing_calling_convention_metadata_rejected(self):
        # Syntactically valid python_code that lacks a required metadata global is not a
        # precompile artifact; load() raises a clear PrecompileError naming the gap.
        buf = io.BytesIO()
        torch.save(
            {
                "format": "torch.compiler.precompile",
                "version": 1,
                "backend": "inductor",
                "artifact": None,
            },
            buf,
        )
        with self.assertRaisesRegex(
            PrecompileError, "missing calling-convention metadata"
        ):
            _load_pair("x = 1\n", buf.getvalue())

    def test_precompile_module_identity(self):
        # torch.compiler.precompile is a submodule: re-importing it resolves to the
        # SAME module object, and its name is the stable public path.
        p = torch.compiler.precompile
        self.assertIs(importlib.import_module("torch.compiler.precompile"), p)
        self.assertIs(sys.modules["torch.compiler.precompile"], p)
        self.assertEqual(p.__name__, "torch.compiler.precompile")

    def _spy_on_guard_drift(self, stack, drift):
        """Patch _report_guard_drift to append each newly recorded drift to ``drift``."""
        from torch._dynamo.precompile_package import PrecompileSession

        real = PrecompileSession._report_guard_drift

        def spy(session, code_entry, rebuilt, live_key):
            before = set(session._drifted_guards)
            real(session, code_entry, rebuilt, live_key)
            drift.extend(session._drifted_guards - before)

        stack.enter_context(
            mock.patch.object(PrecompileSession, "_report_guard_drift", spy)
        )

    def test_precompile_artifact_write_leaves_the_previous_pair_on_failure(self):
        # The two halves only load together -- the cache carries a sha256 of
        # exactly the python_code it was emitted with -- so truncating them in
        # place puts a new artifact next to a stale cache for as long as the
        # write takes. An accumulating capture rewrites on every call and sells
        # exactly that crash as the thing it protects against.
        import builtins

        from torch._precompile import _write_artifact

        with tempfile.TemporaryDirectory() as d:
            artifact_path = os.path.join(d, "a.py")
            cache_path = os.path.join(d, "a.cache")
            _write_artifact(artifact_path, cache_path, "GOOD = 1\n", b"goodcache")

            real_open = builtins.open
            seen = []

            def flaky(path, *args, **kwargs):
                if str(path).endswith(".tmp"):
                    seen.append(path)
                    if len(seen) == 2:
                        raise OSError("disk full")
                return real_open(path, *args, **kwargs)

            with mock.patch.object(builtins, "open", flaky):
                with self.assertRaisesRegex(OSError, "disk full"):
                    _write_artifact(artifact_path, cache_path, "NEW = 2\n", b"newcache")
            with open(artifact_path) as f:
                self.assertEqual(f.read(), "GOOD = 1\n")
            with open(cache_path, "rb") as f:
                self.assertEqual(f.read(), b"goodcache")
            self.assertEqual([f for f in os.listdir(d) if f.endswith(".tmp")], [])

    def test_precompile_artifact_write_honours_the_umask(self):
        # mkstemp creates its file 0600 and the rename carried that onto the
        # artifact, so nobody else on a shared directory could read it. The
        # pair has to land with the mode a plain open() gives under the umask.
        import stat

        from torch._precompile import _write_artifact

        umask = os.umask(0)
        os.umask(umask)
        with tempfile.TemporaryDirectory() as d:
            artifact_path = os.path.join(d, "a.py")
            cache_path = os.path.join(d, "a.cache")
            _write_artifact(artifact_path, cache_path, "X = 1\n", b"cache")
            for path in (artifact_path, cache_path):
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o666 & ~umask)
            self.assertEqual([f for f in os.listdir(d) if f.endswith(".tmp")], [])

    def test_precompile_artifact_write_restores_the_previous_pair_on_rename_failure(
        self,
    ):
        # An OSError on the second rename left the new .py beside the old
        # cache: a pair load refuses on code_hash, for as long as it takes a
        # later call to succeed. The previous artifact is moved aside first
        # and put back, so the named files stay the last good pair.
        from torch._precompile import _write_artifact

        with tempfile.TemporaryDirectory() as d:
            artifact_path = os.path.join(d, "a.py")
            cache_path = os.path.join(d, "a.cache")
            _write_artifact(artifact_path, cache_path, "GOOD = 1\n", b"goodcache")
            real_replace = os.replace

            def flaky(src, dst):
                if dst == cache_path:
                    raise OSError("cache rename failed")
                return real_replace(src, dst)

            with mock.patch.object(os, "replace", flaky):
                with self.assertRaisesRegex(OSError, "cache rename failed"):
                    _write_artifact(artifact_path, cache_path, "NEW = 2\n", b"newcache")
            with open(artifact_path) as f:
                self.assertEqual(f.read(), "GOOD = 1\n")
            with open(cache_path, "rb") as f:
                self.assertEqual(f.read(), b"goodcache")
            self.assertEqual(sorted(os.listdir(d)), ["a.cache", "a.py"])
        # A first write whose cache target cannot be renamed over (a directory)
        # leaves no half artifact behind either.
        with tempfile.TemporaryDirectory() as d:
            artifact_path = os.path.join(d, "a.py")
            cache_dir = os.path.join(d, "a.cache")
            os.mkdir(cache_dir)
            with self.assertRaises(OSError):
                _write_artifact(artifact_path, cache_dir, "NEW = 2\n", b"newcache")
            self.assertEqual(os.listdir(d), ["a.cache"])

    def test_missing_backend_error_reports_the_recorded_split(self):
        # One missing id is fatal, so the message has to say how many of the
        # capture actually landed. Reading "their compiled backends were never
        # recorded" off a capture that recorded 41 of 56 sends the reader after
        # a total failure that did not happen.
        from torch._dynamo.precompile_package import _missing_backends_message

        message = _missing_backends_message(56, [f"b{i}" for i in range(15)])
        self.assertIn("recorded 41 of 56", message)
        self.assertIn("15 graph(s)", message)
        self.assertNotIn("never recorded", message)
        # Long lists get truncated rather than pasting 15 opaque ids.
        self.assertIn("... (7 more)", message)
        self.assertNotIn("b8", message)
        # The cache no longer gates recording, so it must not be the headline.
        self.assertIn("training=True", message)

        one = _missing_backends_message(2, ["b0"])
        self.assertIn("recorded 1 of 2", one)
        self.assertNotIn("more)", one)

    @parametrize("name", _PRECOMPILE_PUBLIC_METHODS)
    def test_precompile_member_module_and_qualname_resolve_to_it(self, name):
        # Nothing hung off the singleton rewrites __module__/__qualname__: the
        # docs place these under torch.compiler.precompile.<name>, but only a
        # name torch.compiler.__all__ exports may claim torch.compiler, or
        # pickle cannot resolve the class and inspect cannot find its source.
        member = getattr(torch.compiler.precompile, name)
        target = sys.modules[member.__module__]
        for part in member.__qualname__.split("."):
            target = getattr(target, part)
        self.assertIs(target, getattr(member, "__func__", member))


def _graph_devices_literal(code: str) -> str:
    """The GRAPH_DEVICES line the artifact records, for tests that assert on it."""
    for line in code.splitlines():
        if line.startswith("GRAPH_DEVICES"):
            return line
    raise AssertionError("artifact has no GRAPH_DEVICES line")


if __name__ == "__main__":
    run_tests()
