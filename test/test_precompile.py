# Owner(s): ["oncall: pt2"]
import ast
import base64
import contextlib
import copy
import functools
import gc
import hashlib
import importlib
import inspect
import io
import linecache
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import threading
import types
import typing
import unittest
import warnings
import weakref
from unittest import mock

import torch
import torch.utils._pytree as _pytree
from torch._dynamo.decorators import mark_dynamic, mark_unbacked
from torch._dynamo.precompile_context import PrecompileContext
from torch._precompile import PrecompileError
from torch.compiler.precompile import DynamoTracer, MakeFxTracer
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_GPU


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


class _CaptureToFiles:
    """Test adapter over torch.compiler.precompile.capture, which writes the
    artifact to disk when the block exits. Accepts the string/None tracer spelling
    and loose per-tracer kwargs the bulk of these tests use, and exposes
    result() -> (python_code, cache) by reading the two files back."""

    def __init__(
        self, fn, *, tracer=None, backend="inductor", training=False, **tracer_kwargs
    ):
        self._dir = tempfile.mkdtemp()
        self._artifact_path = os.path.join(self._dir, "artifact.py")
        self._cache_path = os.path.join(self._dir, "artifact.cache")
        self._cap = torch.compiler.precompile.capture(
            fn,
            artifact_path=self._artifact_path,
            cache_path=self._cache_path,
            tracer=_make_tracer(tracer, tracer_kwargs),
            backend=backend,
            training=training,
        )

    def __enter__(self):
        self._cap.__enter__()
        return self

    def __exit__(self, *exc):
        return self._cap.__exit__(*exc)

    def __call__(self, *args, **kwargs):
        return self._cap(*args, **kwargs)

    def result(self):
        if os.path.exists(self._artifact_path):
            with open(self._artifact_path, encoding="utf-8") as f:
                code = f.read()
            with open(self._cache_path, "rb") as f:
                cache = f.read()
            return code, cache
        rendered = getattr(self._cap, "_rendered", None)
        if rendered is None:
            raise PrecompileError("nothing was captured")
        return rendered

    def summary(self):
        return self._cap.summary()

    def invariants(self):
        return self._cap.invariants()

    def calls(self):
        return self._cap.calls()


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


class _PrecompileAliasedMutationModule(torch.nn.Module):
    """Mutates one of two aliased inputs: the synthetic-base shape the training
    composer refuses to render as source."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4))

    def forward(self, a, b):
        a.mul_(2)
        return ((a + b) * self.w).sum()


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


def _precompile_capture(fn, **kwargs):
    """Drive a caller-driven dynamo capture directly."""
    return _CaptureToFiles(fn, tracer="dynamo", **kwargs)


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


def _precompile_backward_step(model, x):
    model(x).sum().backward()


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


class _PrecompileRebound:
    def forward(self, x):
        return x * 2


class _PrecompileReboundModule(torch.nn.Module):
    def forward(self, x):
        return x * 2


def _precompile_rebound_shadow(x):
    return x * 3


class _PrecompileForwardHolder:
    """The torchrec TrainPipelineSparseDist shape.

    It keeps each module's ORIGINAL bound forward in a list, and optionally
    rebinds the attribute the way TrainPipelineSparseDist swaps in a
    PipelinedForward. Both halves matter: the saved method's receiver is what
    pruning replaces with the sentinel, and whether the attribute still
    resolves to the saved function is what used to pick the reducer's branch.
    """

    def __init__(self, inner, shadow):
        self.inner = inner
        self._original_forwards = [inner.forward]
        self.scale = torch.zeros(4)
        if shadow:
            inner.forward = _precompile_rebound_shadow


def _precompile_rebound_entry(pipeline, x):
    return pipeline._original_forwards[0](x) + x


def _precompile_rebound_unread_entry(pipeline, x):
    # The holder is passed through and guarded, but nothing reads the saved
    # method -- the shape the real model has.
    return x * 2 + pipeline.scale


class _PrecompileReadsAttr(torch.nn.Module):
    def forward(self, x):
        companion = getattr(x, "_cpu_copy", None)
        if companion is not None:
            return x * 2 + companion.to(x.device)
        return x * 2


def _precompile_attr_helper(model, x):
    # x itself must cross the break: rebinding it to an intermediate would drop
    # the attribute and the guard under test would never be built.
    torch._dynamo.graph_break()
    return model(x).sum()


def _precompile_attr_entry(model, x):
    return _precompile_attr_helper(model, x)


_LUT_MODULE = None


def _precompile_reads_module_global(x):
    return x * _LUT_MODULE.LUT.sum()


def _precompile_reads_flag(x):
    return x * getattr(x, "my_flag", 1)


_precompile_reads_shadowed = {
    "pytype": lambda x: x * x.pytype,
    "fake_mode": lambda x: x * x.fake_mode,
    "dispatch_keys": lambda x: x * x.dispatch_keys,
    "_fake_device": lambda x: x * x._fake_device,
}


class _PrecompileClassAttrCfg:
    # A CLASS attribute, so the instance __dict__ lacks it and Dynamo guards
    # its absence with NOT_PRESENT_IN_GENERIC_DICT -- a type no never-drop
    # list covers.
    mode = "a"


def _precompile_class_attr_branch(cfg, x):
    return x * (2.0 if cfg.mode == "a" else 100.0)


class _PrecompileHasattrCfg:
    """Branched on by hasattr, so the branch taken depends on a HASATTR guard."""


def _precompile_hasattr_branch(cfg, x):
    if hasattr(cfg, "fast"):
        return x * 2.0
    return x * 100.0


def _precompile_dict_len(d, x):
    # len(d) rides on the same Guard as the key check, as a DERIVED type.
    return d["a"] * len(d)


class _PrecompileAccumModel(torch.nn.Module):
    """Three branches behind a graph break, so each mode is its own variant."""

    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(8, 4)

    def forward(self, x, mode):
        y = self.l(x)
        torch._dynamo.graph_break()
        if mode == "a":
            return y.sum() * 2
        if mode == "b":
            return y.sum() + 1
        return y.sum() - 3


def _precompile_accum_forward(model, x, mode):
    return model(x, mode)


def _precompile_attr_probe(x):
    tmp = x.side_note  # noqa: F841 -- installs HASATTR without a value guard
    return x + 1


def _serialized_guard_names(code):
    """Every guard name actually present in a shipped artifact's guard state."""
    from torch._dynamo.package import load_guards_state
    from torch._precompile import _read_literal

    names = []
    for frame in pickle.loads(
        base64.b64decode(_read_literal(ast.parse(code), "_FRAMES"))
    ):
        for variant in frame["variants"]:
            state = load_guards_state(variant["guards_state"])
            names += [g.name for g in state.output_graph.guards]
    return " ".join(names)


def _precompile_defaulted_helper(model, x, scale):
    # A nested frame no name in the entry reaches, which is what puts the
    # capture into the installed serving mode.
    torch._dynamo.graph_break()
    return model(x).sum() * scale


def _precompile_defaulted_entry(model, x, scale=3.0):
    """An entry with a default, which a code object does not carry."""
    return _precompile_defaulted_helper(model, x, scale)


# A user global the rendered inductor source shadows with a name of its own.
BACKEND = 5.0


def _precompile_shadowed_global_entry(t):
    return t * BACKEND


_DRIFT_MODULE = None


def _precompile_drift_entry(x):
    return _DRIFT_MODULE.scaled(x).sum()


class _PrecompileDeadResultModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(4, 4)


def _precompile_dead_result(model, x):
    # The result is unused, so the graph's output prunes to nothing -- the
    # shape upstream short-circuits past the backend.
    model.l(x)
    return None


def _precompile_grad_step(model, x):
    loss = model(x).sum()
    loss.backward()
    return loss


@torch._dynamo.allow_in_graph
def _precompile_unkeyable(t):
    """Not on AOTAutogradCache's allowlist, so it refuses to key the graph --
    standing in for the get_external_object_by_index a sharded model emits."""
    return t


def _precompile_calls_unkeyable(model, x):
    # On BOTH sides of the break, so the capture produces two graphs the cache
    # will not key -- which is what makes their fallback keys collide.
    y = _precompile_unkeyable(x)
    torch._dynamo.graph_break()
    return model(_precompile_unkeyable(y)).sum()


def _precompile_mixed_keyability(model, x):
    # Only SOME graphs are unkeyable, which is the shape that used to leave a
    # capture with most of its backends recorded and the rest missing.
    y = model(x).relu()
    torch._dynamo.graph_break()
    y = _precompile_unkeyable(y)
    torch._dynamo.graph_break()
    y = y.sin()
    torch._dynamo.graph_break()
    return _precompile_unkeyable(y).cos()


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
    def test_decompositions_kwarg(self):
        # The decompositions table is threaded into make_fx during capture; a
        # custom decomposition is invoked and the result still matches eager.
        called = []

        def my_relu_decomp(x):
            called.append(True)
            return (x > 0) * x

        decomps = {torch.ops.aten.relu.default: my_relu_decomp}
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x), decompositions=decomps) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertTrue(called)  # the table was used during capture

        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_constant_tensor_is_rejected(self):
        captured = torch.randn(3)
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            with _CaptureToFiles(lambda x: x + captured) as cap:
                cap(torch.randn(3))

    def test_global_tensor_rejected_unlike_make_fx(self):
        # Vanilla make_fx silently bakes a referenced global tensor into the
        # GraphModule as a get_attr constant; precompile must instead error.
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(x):
            return x + _GLOBAL_TENSOR

        gm = make_fx(f)(torch.randn(3))
        baked = [
            n.target
            for n in gm.graph.nodes
            if n.op == "get_attr"
            and isinstance(getattr(gm, n.target, None), torch.Tensor)
        ]
        self.assertTrue(baked, "expected vanilla make_fx to bake a tensor constant")

        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            with _CaptureToFiles(f) as cap:
                cap(torch.randn(3))

    def test_unregistered_module_tensor_attr_is_rejected(self):
        # A plain tensor attribute (not a registered parameter/buffer) is not
        # lifted, so referencing it would bake it in -- this must error.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))
                self.scale = torch.randn(4)  # plain attr, NOT a buffer/parameter

            def forward(self, x):
                return (x @ self.weight) * self.scale

        m = M().eval()
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            with _CaptureToFiles(lambda model, x: model(x)) as cap:
                cap(m, torch.randn(2, 4))

    def test_export_and_reload_roundtrip(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()

        self.assertIn("Inductor output code", code)
        self.assertIn("def forward(", code)
        self.assertIn("PARAM_NAMES = ['lin.weight', 'lin.bias']", code)

        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_self_contained_exec_needs_no_cache(self):
        # python_code runs standalone with NO cache: exec it and call forward().
        # The default eager backend has no kernels; the captured graph is
        # interpreted directly from the inlined source and the cache is always
        # empty (artifact=None), so python_code is fully self-contained.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, _cache = cap.result()

        ns = {"__name__": "_artifact"}
        exec(compile(code, "<artifact>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    @unittest.skipUnless(
        torch.cuda.is_available(), "needs CUDA + Triton for the kernel cache"
    )
    @torch._inductor.config.patch({"compile_threads": 1})
    def test_cache_reload_without_eager_static_launcher_rehydration(self):
        # A cold load should use JIT instead of eagerly rehydrating the static launcher.
        import torch._inductor.config as ind_config

        if ind_config.force_disable_caches or not ind_config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        if not ind_config.use_static_cuda_launcher:
            self.skipTest("requires the static CUDA launcher")
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        m = (
            torch.nn.Sequential(
                torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
            )
            .eval()
            .cuda()
        )
        x = torch.randn(3, 8, device="cuda")
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIsInstance(cache, bytes)

        with fresh_cache():
            counters.clear()
            f_c = _load_pair(code, cache)
            self.assertEqual(f_c(m, x), m(x))
            self.assertEqual(
                counters["inductor"]["triton_bundler_load_static_autotuner"], 0
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for Triton autotuning")
    def test_cache_bundles_autotune_artifacts(self):
        from torch._inductor.utils import fresh_cache

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(512, 512)
                self.l2 = torch.nn.Linear(512, 512)

            def forward(self, x):
                return torch.softmax(self.l2(torch.relu(self.l1(x))), dim=-1)

        m = M().cuda().eval()
        x = torch.randn(128, 512, device="cuda")
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        with fresh_cache():
            f_c = _load_pair(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_dtensor_subclass(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        # Use a free port (a hardcoded one flakes on shared CI) and restore the
        # env afterwards so we do not leak MASTER_ADDR/MASTER_PORT to later tests.
        saved_env = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            for name, p in list(m.named_parameters()):
                setattr(
                    m,
                    name,
                    torch.nn.Parameter(
                        distribute_tensor(p.detach(), mesh, [Replicate()])
                    ),
                )
            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            ref = m(x)

            with _CaptureToFiles(lambda model, x: model(x)) as cap:
                cap(m, x)
            code, cache = cap.result()
            # Subclass handling is via our own protocol-based driver, not embedded
            # AOTAutograd wrapper source.
            self.assertIn("__tensor_unflatten__", code)
            self.assertNotIn("subclass_wrapper", code)

            # load() takes the bundled-artifact path (real AOTAutograd runtime).
            f_c = _load_pair(code, cache)
            self.assertEqual(f_c(m, x).to_local(), ref.to_local())

            # Also exercise the standalone driver (the generated python, no cache):
            # subclass inputs/outputs handled by the inlined recipes via
            # __tensor_flatten__/__tensor_unflatten__.
            ns = {"__name__": "_dt"}
            exec(compile(code, "<dt>", "exec"), ns)
            self.assertEqual(ns["forward"](m, x).to_local(), ref.to_local())
        finally:
            dist.destroy_process_group()
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_cache_holds_only_artifact(self):
        # The cache is purely an acceleration: the only COMPILED blob it carries is the
        # ``artifact`` (no weights, no calling-convention metadata -- that lives in
        # python_code, the single source of truth, and load() parses it back from
        # there). The envelope additionally carries a lightweight format/version/backend
        # integrity tag (plain str/int), which load() verifies.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()

        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        # The artifact is the only compiled blob; the rest is the integrity tag (the
        # format/version/backend/tracer tag plus a code_hash binding the cache to its
        # python_code).
        self.assertEqual(
            set(blob),
            {"artifact", "format", "version", "backend", "tracer", "code_hash"},
        )
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "inductor")
        self.assertEqual(blob["tracer"], "make_fx")
        self.assertIsInstance(blob["artifact"], bytes)
        # The calling convention is recoverable from python_code alone.
        from torch._precompile import _parse_artifact_metadata

        meta = _parse_artifact_metadata(code)
        self.assertEqual(meta["BACKEND"], "inductor")
        self.assertEqual(meta["MODULE_POSITIONS"], [0])

        # load() works using metadata from python_code + artifact from the cache.
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_inlined_fallback_when_artifact_absent(self):
        # When the cache holds no serialized artifact, load() falls back to
        # executing the inlined python (recompiling kernels). Force that branch by
        # stripping the artifact and check it still matches eager; this also
        # exercises the self-contained inlined path (JIT from inlined source).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        self.assertIsNotNone(blob["artifact"])

        f_c = _load_pair(code, _strip_artifact(cache))
        self.assertEqual(f_c(m, x), m(x))

    def test_cache_envelope_is_weights_only_safe(self):
        # The cache is a plain {"artifact": bytes, "format"/"version"/"backend": ...}
        # envelope of only str/int/bytes: it loads with the safe unpickler
        # (weights_only=True). The executable part is the inner artifact bytes, fed to
        # load_cache_artifacts inside load() to prime the inductor cache -- that (plus the
        # subsequent exec of python_code) is the code-execution step, not this outer load.
        # The integrity tag is present and correct (and itself weights_only-safe).
        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        _code, cache = cap.result()
        blob = torch.load(io.BytesIO(cache), weights_only=True)  # must not raise
        self.assertEqual(
            set(blob),
            {"artifact", "format", "version", "backend", "tracer", "code_hash"},
        )
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "inductor")
        # code_hash is a plain str (sha256 hexdigest), so the envelope stays
        # weights_only-safe even with this added key.
        self.assertIsInstance(blob["code_hash"], str)

    def test_wrong_param_count_model_rejected(self):
        # Invariant 2: a runtime model whose param/buffer count differs from the
        # traced model is rejected with a clear error rather than an opaque inner
        # failure. This exercises the default eager load path, which execs
        # python_code (the eager cache carries no artifact).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)

        bigger = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)
        ).eval()
        with self.assertRaisesRegex(PrecompileError, "structurally identical"):
            f_c(bigger, x)

    def test_wrong_param_count_rejected_inlined(self):
        # The same guard fires on the inlined (no-cache) path with the same exception
        # type as the cached path (PrecompileError): strip the artifact so load()
        # execs python_code, then call with a structurally different model.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, _strip_artifact(cache))

        bigger = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)
        ).eval()
        with self.assertRaisesRegex(PrecompileError, "structurally identical"):
            f_c(bigger, x)

    def test_runtime_input_structure_mismatch_rejected(self):
        # Invariant 3: a runtime input whose pytree structure differs from the traced
        # example (here a list where a bare tensor was traced) is rejected via the
        # IN_SPEC check, rather than silently flattening to the wrong leaves.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "different structure"):
            f_c(m, [x, x])

    def test_unserializable_in_spec_still_compiles(self):
        # A runtime input whose pytree TreeSpec is not JSON-serializable (an unregistered
        # collections.namedtuple) must still compile/run on the default eager backend:
        # IN_SPEC degrades to None and the structure check is skipped rather than
        # hard-failing.
        import collections

        P = collections.namedtuple("P", ["x", "y"])
        m = torch.nn.Linear(4, 3).eval()
        inp = P(torch.randn(5, 4), torch.randn(5, 4))
        with _CaptureToFiles(lambda model, p: model(p.x + p.y)) as cap:
            cap(m, inp)
        code, cache = cap.result()
        self.assertIn("IN_SPEC = None", code)
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.x + inp.y))

    def test_unserializable_context_in_spec_still_compiles(self):
        # A registered pytree node whose context is not JSON-dumpable makes
        # treespec_dumps raise TypeError (not NotImplementedError); IN_SPEC must still
        # degrade to None rather than crashing precompile.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        with _CaptureToFiles(lambda model, h: model(h.a + h.b)) as cap:
            cap(m, inp)
        code, cache = cap.result()
        self.assertIn("IN_SPEC = None", code)
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.a + inp.b))

    def test_unserializable_out_spec_hard_fails(self):
        # OUT_SPEC is load-bearing (the driver rebuilds fn's output via tree_unflatten),
        # so unlike IN_SPEC it CANNOT degrade to None. An fn that RETURNS an unregistered
        # collections.namedtuple has a non-JSON-serializable output TreeSpec and must
        # raise a clear PrecompileError rather than leaking a raw pytree error.
        import collections

        Out = collections.namedtuple("Out", ["a", "b"])
        with self.assertRaisesRegex(
            PrecompileError, "cannot serialize the output structure"
        ):
            with _CaptureToFiles(lambda x: Out(x + 1, x + 2)) as cap:
                cap(torch.randn(4))
            cap.result()

    def test_input_leaf_count_mismatch_rejected_when_spec_unserializable(self):
        # When IN_SPEC degrades to None the structural in_spec check is skipped; a runtime
        # input flattening to a DIFFERENT leaf count must still raise a clean
        # PrecompileError (not a raw zip/unpack error) on the live and eager-inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        for backend in ("inductor", "eager"):
            with _CaptureToFiles(
                lambda model, h: model(h.a + h.b), backend=backend
            ) as cap:
                cap(m, inp)
            code, cache = cap.result()
            self.assertIn("IN_SPEC = None", code)
            f = _load_pair(code, cache)
            with self.assertRaisesRegex(PrecompileError, "flattened to"):
                f(m, torch.randn(5, 4))  # one leaf vs the traced two

    def test_user_input_error_precedes_structural_error(self):
        # All three load paths run the user-input checks BEFORE the structural model-name
        # check, so a call violating BOTH (wrong dtype and a different model) reports the
        # user-input (dtype) error, keeping the first-reported error consistent.
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)

        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 3)

            def forward(self, t):
                return self.l1(self.l0(t))

        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        f_i = _load_pair(code, _strip_artifact(cache))
        with _CaptureToFiles(lambda mm, t: mm(t), backend="eager") as cap:
            cap(m, x)
        code_e, cache_e = cap.result()
        f_e = _load_pair(code_e, cache_e)
        for f in (f_c, f_i, f_e):
            with self.assertRaisesRegex(PrecompileError, "dtype"):
                f(
                    B(), x.double()
                )  # wrong model AND wrong dtype -> dtype reported first

    def test_unserializable_out_spec_rejected(self):
        # OUT_SPEC is load-bearing (the driver rebuilds fn's output via tree_unflatten),
        # so unlike IN_SPEC it cannot degrade to None: a fn returning an unregistered
        # namedtuple must fail with a clear PrecompileError, not a raw pytree error, on
        # both backends. A registered namedtuple output round-trips fine.
        import collections

        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        NT = collections.namedtuple("NT", ["p", "q"])
        for backend in ("inductor", "eager"):
            with self.assertRaisesRegex(PrecompileError, "output structure"):
                with _CaptureToFiles(
                    lambda model, xx: NT(model(xx), model(xx) + 1), backend=backend
                ) as cap:
                    cap(m, x)
                cap.result()
        # A registered namedtuple output serializes and round-trips on both backends.
        # Registration mutates the process-global pytree registry, so deregister it on
        # cleanup rather than leaking the node into later tests.
        RNT = collections.namedtuple("RNT", ["p", "q"])
        _pytree._register_namedtuple(RNT, serialized_type_name="test_precompile.RNT")
        self.addCleanup(_pytree._deregister_pytree_node, RNT)
        ref = (m(x), m(x) + 1)
        for backend in ("inductor", "eager"):
            with _CaptureToFiles(
                lambda model, xx: RNT(model(xx), model(xx) + 1), backend=backend
            ) as cap:
                cap(m, x)
            code, cache = cap.result()
            out = _load_pair(code, cache)(m, x)
            self.assertEqual((out.p, out.q), ref)

    def test_cached_and_inlined_paths_agree(self):
        # Both load paths exec the SAME inlined driver in python_code; the only difference
        # is whether the cache primed the kernels first (warm) or not (cold JIT). They must
        # produce identical results -- cross-check via identical scattered grads from a
        # cache-primed load and a cache-stripped (artifact=None) load of the SAME artifact,
        # with multiple modules AND a tied weight across two of them (the case where an
        # ordering divergence in the embedded _extract_param_buffers would show).
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4, bias=False)
        b = torch.nn.Linear(4, 4, bias=False)
        b.weight = a.weight  # tie across two distinct module args
        c = torch.nn.Linear(4, 3)
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)

        def step(ma, mb, mc, x, target):
            loss_fn(mc(mb(torch.relu(ma(x)))), target).backward()

        with _CaptureToFiles(step, training=True) as cap:
            cap(a, b, c, x, target)
        code, cache = cap.result()

        def grads(ms):
            return [p.grad for m in ms for p in m.parameters()]

        # deepcopy the three together so the a/b weight tie is preserved.
        ca, cb, cc = copy.deepcopy((a, b, c))
        _load_pair(code, cache)(ca, cb, cc, x, target)  # cached path

        ia, ib, ic = copy.deepcopy((a, b, c))
        _load_pair(code, _strip_artifact(cache))(ia, ib, ic, x, target)  # inlined

        for cg, ig in zip(grads((ca, cb, cc)), grads((ia, ib, ic))):
            self.assertEqual(cg, ig)

    def test_eager_param_ordering_agrees_with_inductor(self):
        # Both backends now emit the same _extract_param_buffers (from
        # torch._precompile_driver), which must stay in sync with
        # torch._precompile._intern_param_buffers. The test above cross-checks only the
        # cached vs inductor-inlined paths; cross-check the EAGER backend too, on the same
        # multi-module + tied-weight + backward step, so an ordering divergence in the
        # shared driver shows as a scattered-grad mismatch against the inductor cached path.
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4, bias=False)
        b = torch.nn.Linear(4, 4, bias=False)
        b.weight = a.weight  # tie across two distinct module args
        c = torch.nn.Linear(4, 3)
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)

        def step(ma, mb, mc, x, target):
            loss_fn(mc(mb(torch.relu(ma(x)))), target).backward()

        def grads(ms):
            return [p.grad for m in ms for p in m.parameters()]

        # deepcopy the three together so the a/b weight tie is preserved.
        with _CaptureToFiles(step, training=True) as cap:
            cap(a, b, c, x, target)
        icode, icache = cap.result()
        ia, ib, ic = copy.deepcopy((a, b, c))
        _load_pair(icode, icache)(ia, ib, ic, x, target)  # inductor cached path

        with _CaptureToFiles(step, backend="eager", training=True) as cap:
            cap(a, b, c, x, target)
        ecode, ecache = cap.result()
        ea, eb, ec = copy.deepcopy((a, b, c))
        _load_pair(ecode, ecache)(ea, eb, ec, x, target)  # eager path

        ind_grads = grads((ia, ib, ic))
        eager_grads = grads((ea, eb, ec))
        self.assertEqual(len(ind_grads), len(eager_grads))
        for ig, eg in zip(ind_grads, eager_grads):
            self.assertEqual(ig, eg)

    def test_non_module_at_module_position_rejected(self):
        # Passing a non-nn.Module where the traced fn took a module yields a clear
        # PrecompileError citing invariant 2, not a bare AttributeError.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "must be the nn.Module"):
            f_c(x, x)  # tensor at the module slot

    def test_wrong_arg_count_rejected(self):
        # A runtime call with the wrong number of positional args raises a clear
        # PrecompileError (invariant 2) -- not a raw IndexError -- on all three load
        # paths, including when a module is at a non-zero position (where args[i] would
        # otherwise index past the short args tuple).
        m = torch.nn.Linear(4, 3)
        x = torch.randn(2, 4)
        # Module at position 1 (so a missing trailing arg would index past args).
        with _CaptureToFiles(lambda xx, model: model(xx)) as cap:
            cap(x, m)
        code, cache = cap.result()
        inlined_cache = _strip_artifact(cache)  # force the inlined path
        with _CaptureToFiles(lambda xx, model: model(xx), backend="eager") as cap:
            cap(x, m)
        ecode, ecache = cap.result()
        loaders = {
            "cached": _load_pair(code, cache),
            "inlined": _load_pair(code, inlined_cache),
            "eager": _load_pair(ecode, ecache),
        }
        for label, f_c in loaders.items():
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "expected 2 positional"):
                    f_c(x)  # too few (omits the module arg)
                with self.assertRaisesRegex(PrecompileError, "expected 2 positional"):
                    f_c(x, m, x)  # too many
                self.assertEqual(f_c(x, m), m(x))  # correct arity still works

    def test_buffer_requiring_grad_rejected(self):
        # A registered buffer with requires_grad=True that receives a gradient is not
        # harvested (only params are), so precompile rejects it rather than silently
        # dropping the grad.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.randn(4, requires_grad=True))

            def forward(self, x):
                return (x * self.b).sum()

        m = M()
        x = torch.randn(4)
        with self.assertRaisesRegex(PrecompileError, "buffer received a gradient"):
            with _CaptureToFiles(
                lambda model, x: model(x).backward(), training=True
            ) as cap:
                cap(m, x)

    def test_user_input_requiring_grad_rejected(self):
        # Sibling of the buffer guard: a requires_grad USER INPUT (not a param) that
        # receives a gradient during the traced backward is not harvested (only params
        # are), so precompile rejects it rather than silently dropping the grad.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaisesRegex(PrecompileError, "user input received a gradient"):
            with _CaptureToFiles(
                lambda t: (t * t).sum().backward(), training=True
            ) as cap:
                cap(x)

    def test_control_flow_subgraph_rejected(self):
        # torch.cond captures as a HOP with get_attr subgraph submodules, which the
        # standalone artifact cannot inline; reject it at capture with a clear message.
        def f(x):
            return torch.cond(x.sum() > 0, lambda t: t + 1, lambda t: t - 1, (x,))

        with self.assertRaisesRegex(PrecompileError, "control-flow subgraph"):
            with _CaptureToFiles(f) as cap:
                cap(torch.randn(4))

    def test_load_falls_back_when_cache_unreconstructable(self):
        # The cache is only an acceleration; python_code always runs standalone. A
        # corrupt / stale cache must degrade to the inlined JIT path, not crash.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertIsNotNone(blob["artifact"])
        blob["artifact"] = b"corrupt-not-a-real-artifact"
        buf = io.BytesIO()
        torch.save(blob, buf)

        f_c = _load_pair(code, buf.getvalue())  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_falls_back_on_corrupt_cache_envelope(self):
        # Not just a bad inner artifact -- a corrupt/truncated cache ENVELOPE (not even
        # a valid torch.save blob) must also degrade to the inlined python_code path,
        # since the cache is purely an acceleration.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, _cache = cap.result()
        f_c = _load_pair(code, b"not-a-torch-save-blob")  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_invalid_python_code_rejected(self):
        # load() surfaces a clear PrecompileError (not a raw SyntaxError) when
        # python_code is not valid Python.
        buf = io.BytesIO()
        torch.save({"artifact": None}, buf)
        with self.assertRaisesRegex(PrecompileError, "not valid Python"):
            _load_pair("def (:::", buf.getvalue())

    def test_untrusted_input_warning_fires_per_load(self):
        # The trust warning is emitted PER load (not warning_once) via log.warning on the
        # torch._precompile logger: load() always execs python_code (through
        # _make_inlined_forward), which warns before the exec, whether or not the cache
        # primed the kernels first. Calling load() TWICE must fire the untrusted-input
        # warning on BOTH calls, locking in per-load behavior rather than once-per-process.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        # Cached path (inductor): the exec of python_code warns about untrusted input.
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                _load_pair(code, cache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"cached load did not warn about untrusted input: {cm.output}",
            )
        # Eager backend (empty cache, nothing to prime): load() still EXECs python_code
        # via _make_inlined_forward, which warns about exec'ing untrusted code every load.
        with _CaptureToFiles(lambda model, t: model(t), backend="eager") as cap:
            cap(m, x)
        ecode, ecache = cap.result()
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                _load_pair(ecode, ecache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"inlined load did not warn about untrusted input: {cm.output}",
            )
            self.assertTrue(
                any("EXEC" in line for line in cm.output),
                f"inlined load did not warn about exec'ing python_code: {cm.output}",
            )

    def test_no_compute_graph_rejected_inductor(self):
        # The inductor backend produces no runnable module for a graph with no compute
        # to lower -- one that returns inputs or Python constants unchanged (a constant,
        # a bare passthrough, or an alias like .detach()). Reject with a clear
        # PrecompileError rather than a raw "found 0 runnable modules" RuntimeError. The
        # eager backend handles these (the contract is otherwise identical).
        x = torch.randn(4)
        for fn in (lambda xx: 7, lambda xx: xx, lambda xx: xx.detach()):
            with self.assertRaisesRegex(PrecompileError, "no compute"):
                with _CaptureToFiles(fn) as cap:
                    cap(x)
        # The eager backend handles a passthrough and a constant fn.
        with _CaptureToFiles(lambda xx: xx, backend="eager") as cap:
            cap(x)
        code, cache = cap.result()
        self.assertEqual(_load_pair(code, cache)(x), x)
        with _CaptureToFiles(lambda xx: 7, backend="eager") as cap:
            cap(x)
        code, cache = cap.result()
        self.assertEqual(_load_pair(code, cache)(x), 7)

    def test_same_count_different_structure_rejected(self):
        # Invariant 2: the structural check now compares the baked PARAM_NAMES /
        # BUFFER_NAMES against the runtime model's extracted param/buffer names, so a
        # same-count-but-different-structure (here, differently-NAMED submodules) model
        # is REJECTED rather than silently running the traced graph with the wrong
        # weights. Both the cached and the inlined (artifact-stripped) load paths fire.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        with _CaptureToFiles(lambda m, x: m(x)) as cap:
            cap(a, x)
        code, cache = cap.result()
        # The traced names come from the Sequential (``0.weight``, ``1.weight`` ...).
        self.assertIn(
            "PARAM_NAMES = ['0.weight', '0.bias', '1.weight', '1.bias']", code
        )

        class B(torch.nn.Module):  # same 4 params (same count/shapes), different names
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        b = B().eval()
        loaders = {
            "cached": _load_pair(code, cache),
            "inlined": _load_pair(code, _strip_artifact(cache)),
        }
        for label, f_c in loaders.items():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, "do not match the traced model"
                ):
                    f_c(b, x)

    def test_same_count_different_structure_rejected_eager(self):
        # The eager driver's _check_structure rejects a same-param-COUNT but
        # different-NAME model (here differently-named submodules) rather than
        # silently running the traced graph with the wrong weights (invariant 2).
        # What's distinct from test_wrong_param_count_model_rejected above is the
        # INPUT -- same count / different name, not a count mismatch.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        with _CaptureToFiles(lambda m, x: m(x), backend="eager") as cap:
            cap(a, x)
        code, cache = cap.result()
        self.assertIn(
            "PARAM_NAMES = ['0.weight', '0.bias', '1.weight', '1.bias']", code
        )

        class B(torch.nn.Module):  # same 4 params (same count/shapes), different names
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        b = B().eval()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(b, x)

    # Input mutation, output aliasing, tensor subclasses, and functionalized RNG are
    # SUPPORTED: the inductor backend lowers through aot_autograd.compile_to_python,
    # which composes AOTAutograd's own codegen'd prelude/epilogue into the artifact.
    # Only effectful ops are rejected up front (see test_effectful_op_unsupported).

    def test_effectful_op_unsupported(self):
        # Effectful custom ops are rejected up front by _assert_supported, which
        # detects the with_effects HOP in the captured graph -- the effect cannot
        # be lowered to standalone source, so capture fails cleanly.
        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
        from torch.library import _scoped_library

        with _scoped_library("mlprecompile", "FRAGMENT") as lib:
            lib.define("eff(Tensor x) -> Tensor")
            lib.impl("eff", lambda x: x + 1.0, "CompositeExplicitAutograd")
            lib.impl("eff", lambda x: torch.empty_like(x), "Meta")
            op = torch.ops.mlprecompile.eff.default
            _register_effectful_op(op, _EffectType.ORDERED)
            try:
                with self.assertRaisesRegex(
                    PrecompileError, "effectful op.*not supported yet"
                ):
                    with _CaptureToFiles(
                        lambda a: torch.ops.mlprecompile.eff(a)
                    ) as cap:
                        cap(torch.randn(4))
            finally:
                _register_effectful_op(op, None)

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

    def test_dynamo_artifact_version_lock_raises_precompile_error(self):
        # Note [precompile programming model] promises the driver's
        # Python-version lock surfaces as a clean PrecompileError, so an
        # ``except torch.compiler.precompile.PrecompileError`` handler written
        # to the docs catches it -- not a raw ValueError.
        def fn(x):
            return x + 1

        with _CaptureToFiles(fn, tracer="dynamo", backend="eager") as cap:
            cap(torch.randn(3))
        python_code, cache = cap.result()
        cur = f"_DYNAMO_PYTHON_VERSION = {tuple(sys.version_info[:2])!r}"
        self.assertIn(cur, python_code)
        skewed = python_code.replace(cur, "_DYNAMO_PYTHON_VERSION = (3, 9)")
        # exec, not load(): load()'s cache-pairing hash check would reject the
        # edited text first, and python_code is documented as self-contained.
        with self.assertRaisesRegex(PrecompileError, "produced on Python 3.9"):
            exec(skewed, {})

    @parametrize("name", _PRECOMPILE_PUBLIC_METHODS)
    def test_precompile_public_members_resolve(self, name):
        typing.get_type_hints(getattr(torch.compiler.precompile, name))

    def test_precompile_public_result_types(self):
        # precompile is a module; capture() returns a Capture and load() a
        # PrecompiledRunnable.
        self.assertFalse(callable(torch.compiler.precompile))
        self.assertIs(
            typing.get_type_hints(torch.compiler.precompile.capture)["return"],
            torch.compiler.precompile.Capture,
        )
        # The guard/variant knobs live on the tracer, not on capture/accumulate.
        self.assertIn(
            "guard_filter_fn", inspect.getdoc(torch.compiler.precompile.DynamoTracer)
        )
        # The risky-drop lint is the rail that is ON by default. Requiring NO
        # dropped guards at all is not, and must not be: every model drops the
        # identity guards precompile cannot serialize, so it would refuse
        # essentially every real artifact.
        params = inspect.signature(torch.compiler.precompile.DynamoTracer).parameters
        self.assertTrue(params["require_no_risky_drops"].default)
        self.assertFalse(params["require_no_dropped_guards"].default)
        # Every shape load() returns is a torch.compiler.PrecompiledRunnable, so
        # one isinstance check and one enter/unload protocol cover them all.
        x = torch.randn(4)
        with _CaptureToFiles(_precompile_single_graph, backend="eager") as cap:
            cap(x)
        code, cache = cap.result()
        loaded = _load_pair(code, cache)
        self.assertIn("PrecompiledRunnable", torch.compiler.__all__)
        self.assertIsInstance(loaded, torch.compiler.PrecompiledRunnable)
        self.assertFalse(loaded.installed)
        with loaded:
            self.assertEqual(loaded(x), _precompile_single_graph(x))

    @parametrize("case", list(_PRECOMPILE_BAD_CALLS))
    def test_precompile_rejects_a_malformed_call(self, case):
        exc, regex, args, kwargs = _PRECOMPILE_BAD_CALLS[case]
        with self.assertRaisesRegex(exc, regex):
            _CaptureToFiles(*args, backend="eager", **kwargs)

    def test_make_fx_refuses_a_second_call(self):
        x = torch.randn(3)
        with _CaptureToFiles(_precompile_add_one, backend="eager") as cap:
            cap(x)
            with self.assertRaisesRegex(
                torch.compiler.PrecompileError, "captures a single call"
            ):
                cap(x)

    @parametrize("case", list(_PRECOMPILE_ROUNDTRIP_CASES))
    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_roundtrip(self, case, backend):
        # The dynamo tracer captures via Dynamo, inlines the transformed bytecode, and
        # (like make_fx) lowers the subgraph through the chosen backend. The reload runs
        # the same computation as eager, and a different but structurally identical model
        # swapped in at runtime works (invariant 2) -- no weights are baked in.
        fn, make_modules, shape = _PRECOMPILE_ROUNDTRIP_CASES[case]
        modules, x = make_modules(), torch.randn(*shape)
        # When fn IS the module, the loaded artifact still takes the receiver
        # explicitly: its entry frame is forward(self, xx).
        entry, examples = (modules[0], (x,)) if fn is None else (fn, (*modules, x))
        reference = _precompile_call_model if fn is None else fn
        with _CaptureToFiles(
            entry, training=True, tracer="dynamo", backend=backend
        ) as cap:
            cap(*examples)
        code, cache = cap.result()
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            self.assertIsInstance(f_c, torch.compiler.PrecompiledRunnable)
            self.assertFalse(f_c.installed)
            self.assertEqual(f_c(*modules, x), reference(*modules, x))
            swapped = make_modules()
            self.assertEqual(f_c(*swapped, x), reference(*swapped, x))

    def test_tracer_dynamo_autograd_grad_returned(self):
        # torch.autograd.grad is captured too, and (unlike .backward()) it only RETURNS the
        # grads: nothing is scattered, so the runtime model's .grad stays None.
        # Seeding is a capture-time mutation of the caller's model, so fn can SEE it:
        # `p.grad is not None` traced as True where eager reads False. When the
        # re-capture turns out to need no accumulate at all -- autograd.grad only
        # returns grads -- the seeds bought nothing, and the first (unseeded) capture
        # is the one that has to ship.
        x, t = torch.randn(5, 4), torch.randn(5, 3)

        def grad_step(model, xx, tt):
            seen = model.weight.grad is not None
            loss = torch.nn.functional.mse_loss(model(xx), tt)
            return seen, torch.autograd.grad(loss, list(model.parameters()))

        def fresh():
            torch.manual_seed(0)
            return torch.nn.Linear(4, 3)

        with _CaptureToFiles(grad_step, training=True, tracer="dynamo") as cap:
            cap(fresh(), x, t)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        run, ref = fresh(), fresh()
        self.assertEqual(f_c(run, x, t), grad_step(ref, x, t))
        self.assertTrue(all(p.grad is None for p in run.parameters()))

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

    @parametrize("decompose", [False, True])
    @parametrize("backend", ["eager", "inductor"])
    def test_capture_under_a_torch_function_mode_applies_it_once(
        self, decompose, backend
    ):
        # make_fx only. The dynamo tracer mirrors torch.compile, which applies an
        # ambient torch_function mode TWICE (measured: eager 2.0, torch.compile 3.0)
        # because lowering re-traces the torch-level graph with the modes still live.
        # make_fx clears the stack around lowering and stays eager-correct, which is
        # the guarantee this pins.
        tracer = "make_fx"
        # Capture clears the caller's torch_function modes so Dynamo can apply them
        # SYMBOLICALLY while tracing. The captured graph is torch-level Python, so
        # lowering it with the modes restored re-traces through every one of them a
        # second time and bakes a doubly-transformed kernel. Nothing catches that:
        # the artifact needs no mode at all to reproduce the wrong number, so there
        # is no guard to drop and no error to raise -- it is simply wrong forever.
        fn = _precompile_add_one
        x = torch.zeros(3)
        with _PrecompilePlusOneMode():
            expected = fn(x).clone()
            with _CaptureToFiles(
                fn,
                tracer=tracer,
                backend=backend,
                **{"decompositions": {}} if decompose else {},
            ) as cap:
                cap(x)
            code, cache = cap.result()
        # Served with NO mode: the artifact must already carry the one application.
        self.assertEqual(_load_pair(code, cache)(x), expected)

    @parametrize("guard_filter", ["keep_all", "portable", "drop_all"])
    @parametrize("caching_precompile", [False, True])
    def test_guard_filter_composes_with_the_default(
        self, guard_filter, caching_precompile
    ):
        # A custom filter COMPOSES with the default rather than replacing it:
        # keep_all cannot re-admit the unserializable identity guards, a portable
        # filter's second trip through the guard pickle still loads, and a drop
        # the filter ADDED is risky by construction, so drop_all is refused at
        # the default gates. caching_precompile's own filter must not leak in.
        filters = {
            "keep_all": lambda entries: [True] * len(entries),
            "portable": torch.compiler.keep_portable_guards_unsafe,
            "drop_all": lambda entries: [False] * len(entries),
        }
        xs = [torch.randn(n, 8) for n in (2, 3)]
        with torch._dynamo.config.patch(caching_precompile=caching_precompile):
            if guard_filter == "drop_all":
                with self.assertRaisesRegex(
                    PrecompileError, r"can affect dispatch on \[.*'x'"
                ):
                    with _CaptureToFiles(
                        _precompile_multi_graph,
                        backend="eager",
                        dynamic=False,
                        tracer="dynamo",
                        guard_filter_fn=filters[guard_filter],
                    ) as cap:
                        with torch.no_grad():
                            for _ex_args in [(x,) for x in xs]:
                                cap(*_ex_args)
                    cap.result()
                return
            with _CaptureToFiles(
                _precompile_multi_graph,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                guard_filter_fn=filters[guard_filter],
                require_no_risky_drops=False,
            ) as cap:
                with torch.no_grad():
                    for _ex_args in [(x,) for x in xs]:
                        cap(*_ex_args)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            for x in xs:
                self.assertEqual(loaded(x), _precompile_multi_graph(x))
            # The shape guards were kept: an uncovered shape is refused.
            with self.assertRaisesRegex(PrecompileError, "no captured variant"):
                loaded(torch.randn(9, 8))

    @parametrize("how", ["raise_in_block", "caught_call"])
    def test_a_failed_capture_is_incomplete(self, how):
        x = torch.randn(4)
        entry = (
            _precompile_multi_graph
            if how == "raise_in_block"
            else _precompile_raises_on_flag
        )

        def run(session):
            # raise_in_block escapes the block with a KeyError, so nothing is
            # rendered (a block that raised leaves the files untouched);
            # caught_call catches its per-call failure and exits cleanly, so the
            # render+gate runs on the way out and the default gates refuse the
            # incomplete capture at exit.
            if how == "raise_in_block":
                with self.assertRaisesRegex(KeyError, "capture failed"):
                    with session as compiled:
                        compiled(torch.randn(4, 8))
                        raise KeyError("capture failed")
            else:
                with session as compiled:
                    compiled(x, False)
                    with self.assertRaisesRegex(KeyError, "automatic example failed"):
                        compiled(x, True)

        session = _precompile_capture(entry, backend="eager", dynamic=False)
        if how == "caught_call":
            with self.assertRaises(PrecompileError):
                run(session)
        else:
            run(session)
        summary = session.summary()
        self.assertGreater(summary.guarded_codes, 0)
        self.assertFalse(summary.complete)
        self.assertEqual(len(summary.capture_errors), 1)
        # No artifact was written for the incomplete capture.
        with self.assertRaises(PrecompileError):
            session.result()
        if how == "caught_call":
            # The block exits cleanly, so relaxing the gates at construction lets
            # the same drive render the partial capture on the way out.
            relaxed = _precompile_capture(
                entry,
                backend="eager",
                dynamic=False,
                require_complete=False,
                require_no_dropped_guards=False,
            )
            run(relaxed)
            relaxed.result()

    def test_sessions_restore_the_capture_config(self):
        # Entering a session flips two config flags; leaving it must restore
        # them whether sessions nest on one thread, overlap across threads, or
        # fail in __enter__ (an invalid backend).
        import torch._functorch.config as functorch_config
        from torch._dynamo.precompile_package import precompile_capture

        def flags():
            return (
                functorch_config.bundled_autograd_cache,
                torch._dynamo.config.allow_empty_graphs,
            )

        with (
            functorch_config.patch("bundled_autograd_cache", False),
            torch._dynamo.config.patch(allow_empty_graphs=False),
        ):
            first = precompile_capture(_precompile_multi_graph, backend="eager")
            second = precompile_capture(_precompile_multi_graph, backend="eager")
            first.__enter__()
            second.__enter__()
            first.__exit__(None, None, None)
            self.assertEqual(flags(), (True, True))
            second.__exit__(None, None, None)
            self.assertEqual(flags(), (False, False))

            entered, release, errors = threading.Event(), threading.Event(), []

            def hold():
                try:
                    held = precompile_capture(_precompile_single_graph, backend="eager")
                    held.__enter__()
                    entered.set()
                    self.assertTrue(release.wait(10))
                    held.__exit__(None, None, None)
                except BaseException as error:
                    errors.append(error)

            holder = threading.Thread(target=hold)
            holder.start()
            self.assertTrue(entered.wait(10))
            session = precompile_capture(_precompile_single_graph, backend="eager")
            session.__enter__()
            self.assertEqual(flags(), (True, True))
            release.set()
            holder.join(10)
            self.assertFalse(holder.is_alive())
            self.assertEqual(errors, [])
            # The earlier session left on its own thread while this one is open.
            self.assertEqual(flags(), (True, True))
            session.__exit__(None, None, None)
            self.assertEqual(flags(), (False, False))

            session = precompile_capture(
                _precompile_multi_graph,
                backend="definitely_missing_backend",
                dynamic=False,
            )
            with self.assertRaisesRegex(
                torch._dynamo.exc.InvalidBackend, "Invalid backend"
            ):
                with session:
                    pass
            self.assertEqual(flags(), (False, False))
            self.assertFalse(session.summary().complete)
            self.assertIn("InvalidBackend", session.summary().capture_errors[0])
            from torch._dynamo.exc import PackageError

            with self.assertRaises(PackageError):
                session.artifact()

    def test_exit_waits_for_a_worker_thread_call(self):
        # A worker thread's call inherits the capture, and __exit__ waits for it.
        from torch._dynamo.backends.registry import register_backend
        from torch._dynamo.eval_frame import _get_total_cache_entry_count
        from torch._dynamo.precompile_package import precompile_capture

        entered = threading.Event()
        release = threading.Event()
        worker_done = threading.Event()
        errors = []
        outputs = []

        def blocking_backend(gm, example_inputs):
            entered.set()
            self.assertTrue(release.wait(20))
            return gm.forward

        backend_name = f"precompile_exit_waits_{id(entered)}"
        register_backend(blocking_backend, name=backend_name)
        session = precompile_capture(
            _precompile_single_graph, backend=backend_name, dynamic=False
        )
        compiled = session.__enter__()
        x = torch.randn(2)

        def run():
            try:
                outputs.append(compiled(x))
            except Exception as e:
                errors.append(e)
            finally:
                worker_done.set()

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(entered.wait(20))

        def release_call():
            self.assertFalse(worker_done.wait(0.1))
            release.set()

        releaser = threading.Thread(target=release_call)
        releaser.start()
        session.__exit__(None, None, None)
        worker.join(20)
        releaser.join(20)
        self.assertFalse(worker.is_alive())
        self.assertFalse(releaser.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(outputs, [_precompile_single_graph(x)])
        self.assertTrue(session.summary().complete)
        self.assertEqual(session.summary().guarded_codes, 1)
        self.assertEqual(
            _get_total_cache_entry_count(_precompile_single_graph.__code__), 0
        )

    def test_session_is_one_shot_and_releases_its_examples(self):
        example = torch.randn(1024)
        example_ref = weakref.ref(example)
        session = _precompile_capture(
            _precompile_multi_graph, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(example)
        self.assertTrue(session.summary().complete)
        with self.assertRaisesRegex(RuntimeError, "not active"):
            compiled(example)
        with self.assertRaisesRegex(RuntimeError, "already been entered"):
            with session:
                pass
        del example
        torch._dynamo.reset()
        gc.collect()
        self.assertIsNone(example_ref())

        # A failed call's traceback must not keep its example alive either.
        failed = torch.randn(1024)
        failed_ref = weakref.ref(failed)
        session = _precompile_capture(
            _precompile_raises_on_flag, backend="eager", dynamic=False
        )
        # The only call fails, so the capture is incomplete and the default
        # gates refuse to render it when the block exits cleanly.
        with self.assertRaises(PrecompileError):
            with session as compiled:
                with self.assertRaisesRegex(KeyError, "automatic example failed"):
                    compiled(failed, True)
        del failed
        torch._dynamo.reset()
        gc.collect()
        self.assertIsNone(failed_ref())

    def _multigraph_frames(self, code):
        from torch._precompile import _parse_artifact_metadata

        return _parse_artifact_metadata(code)["FRAMES"]

    @parametrize("backend", ("eager", "inductor"))
    def test_multi_graph_artifact_follows_the_code_cache_contract(self, backend):
        from torch._C._dynamo.eval_frame import _debug_get_precompile_entries
        from torch._precompile import _parse_artifact_metadata
        from torch.compiler._cache import CacheArtifactManager

        # A capture with graph breaks and several variants returns the same
        # (python_code, cache) pair the single-graph forms do, and load() takes
        # it back. python_code is standalone: it installs nothing onto the
        # callable's code objects, so serving it mutates no global state.
        PrecompileContext.clear()
        self.addCleanup(PrecompileContext.clear)
        model = _PrecompileBreakingModule().eval()
        shapes = [(3, 8), (5, 8)]
        inputs = [torch.randn(*shape) for shape in shapes]
        with torch.no_grad():
            expected = [model(x) for x in inputs]

        # The capture serializes the process-global cache-artifact list into the
        # bundle. An artifact an unrelated earlier compile left pending must be
        # neither folded into this bundle nor dropped from that list.
        with CacheArtifactManager.with_fresh_cache():
            CacheArtifactManager.record_artifact("pgo", "unrelated_pending", b"x")
            with _CaptureToFiles(
                model, backend=backend, dynamic=False, tracer="dynamo"
            ) as cap:
                with torch.no_grad():
                    for _ex_args in [(x,) for x in inputs]:
                        cap(*_ex_args)
            code, cache = cap.result()
            pending = CacheArtifactManager._new_cache_artifacts["pgo"]
            self.assertEqual([a.key for a in pending], ["unrelated_pending"])
        self.assertIsInstance(code, str)
        self.assertIsInstance(cache, bytes)
        bundle = torch.load(io.BytesIO(cache), weights_only=True)["artifact"]
        if bundle is not None:
            artifacts = CacheArtifactManager.deserialize(bundle)
            keys = [a.key for group in artifacts.values() for a in group]
            self.assertNotIn("unrelated_pending", keys)
        # The artifact carries its own backends; neither the capture nor a load
        # may file them into the process-global context the transparent cache uses.
        self.assertEqual(PrecompileContext._backend_artifacts_by_key, {})

        # The readable half says what is in the opaque half.
        meta = _parse_artifact_metadata(code)
        self.assertEqual(meta["TRACER"], "dynamo")
        self.assertIn("DROPPED_GUARDS", meta)
        # Guard trees and bytecode have no source form and stay opaque, but a
        # compiled subgraph is Inductor output, which does: on inductor the
        # kernels are emitted as readable source, and only eager (whose
        # "backend" is an fx graph with nothing to render) stays pickled.
        if backend == "inductor":
            self.assertIn("_SUBGRAPHS[", code)
        else:
            self.assertNotIn("_SUBGRAPHS[", code)

        # It reports one entry frame plus one continuation, two variants each.
        frames = meta["FRAMES"]
        self.assertEqual([count for _, count in frames], [2, 2])
        self.assertTrue(
            any(name.startswith("torch_dynamo_resume_in") for name, _ in frames)
        )

        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        self.assertIsInstance(loaded, torch.compiler.PrecompiledRunnable)
        self.assertFalse(loaded.installed)
        with torch.no_grad():
            for x, want in zip(inputs, expected):
                self.assertEqual(loaded(model, x), want)
        self.assertEqual(PrecompileContext._backend_artifacts_by_key, {})
        # Nothing was installed, so the model still compiles normally.
        self.assertEqual(
            len(_debug_get_precompile_entries(type(model).forward.__code__)), 0
        )
        # A call no variant covers raises rather than silently recompiling.
        with self.assertRaisesRegex(PrecompileError, "no captured variant"):
            with torch.no_grad():
                loaded(model, torch.randn(9, 8))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_training_step_with_backward_in_fn(self, backend):
        # The Note's headline training step: .backward() inside fn graph-breaks,
        # the continuation runs it through the live autograd engine, and the
        # served step accumulates the same grads eager does. Capture is
        # caller-driven, so the captured call runs the backward for real and
        # accumulates onto the model exactly as the served step later will.
        from torch._dynamo.utils import counters

        def train_step(model, x, t):
            torch.nn.functional.mse_loss(model(x), t).backward()

        torch.manual_seed(0)
        model = _PrecompileTrainMod()
        x, t = torch.randn(4, 8), torch.randn(4, 8)
        # An eager backward from clean grads is the reference the served step
        # must reproduce.
        train_step(model, x, t)
        eager = [p.grad.clone() for p in model.parameters()]
        for p in model.parameters():
            p.grad = None
        with _CaptureToFiles(
            train_step,
            backend=backend,
            dynamic=False,
            tracer="dynamo",
            training=True,
            require_complete=False,
            require_no_risky_drops=False,
        ) as cap:
            cap(model, x, t)
        code, cache = cap.result()
        for want, p in zip(eager, model.parameters()):
            self.assertEqual(p.grad, want)  # the captured call ran the backward
        for p in model.parameters():
            p.grad = None
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        counters.clear()
        with _maybe_scoped(loaded):
            self.assertIsNone(loaded(model, x, t))
        for want, p in zip(eager, model.parameters()):
            self.assertEqual(p.grad, want)
        self.assertEqual(counters["stats"]["unique_graphs"], 0)

    @parametrize("case", list(_AUTO_DYNAMIC_CASES))
    def test_automatic_dynamic_promotes_the_frames_that_varied(self, case):
        # A dim that varies has to be detected separately in every frame that
        # reads it -- otherwise the artifact serves a new shape up to the first
        # break and then misses -- and ONLY in those: an entry frame whose input
        # never varies stays specialized while the continuation that did see
        # variation is promoted.
        model_cls, make_args, counts = _AUTO_DYNAMIC_CASES[case]
        model = model_cls().eval()
        captured = [make_args(n) for n in (3, 5)]
        unseen = [make_args(n) for n in (7, 11)]

        with _CaptureToFiles(
            model,
            backend="eager",
            dynamic=None,
            tracer="dynamo",
            require_no_risky_drops=False,
        ) as cap:
            with torch.no_grad():
                for _ex_args in captured:
                    cap(*_ex_args)
        code, cache = cap.result()
        frames = self._multigraph_frames(code)
        # One static compile per frame that varied, then one promoted to dynamic.
        self.assertEqual([count for _, count in frames], counts)
        self.assertEqual(
            sum(1 for name, _ in frames if name.startswith("torch_dynamo_resume_in")),
            len(counts) - 1,
        )

        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with torch.no_grad():
            for args in captured + unseen:
                self.assertEqual(loaded(model, *args), model(*args))

        if case != "every_frame":
            return
        # The contrast that makes the assertion above meaningful: with automatic
        # dynamic off, the same capture serves only what it saw.
        torch._dynamo.reset()
        with _CaptureToFiles(
            model,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
            require_no_risky_drops=False,
        ) as cap:
            with torch.no_grad():
                for _ex_args in captured:
                    cap(*_ex_args)
        static_code, static_cache = cap.result()
        torch._dynamo.reset()
        static = _load_pair(static_code, static_cache)
        with torch.no_grad():
            for args in unseen:
                with self.assertRaisesRegex(PrecompileError, "no captured variant"):
                    static(model, *args)

    def test_multi_graph_unreachable_frame_is_served_by_installing(self):
        # A frame Dynamo compiled that is entered by an ORDINARY call -- an
        # un-inlinable helper -- is reached only through the frame evaluator. A
        # source artifact does not use one, so such a capture is served by
        # installing onto the live code objects instead, and the frame that a
        # source artifact would have run eager is named in the header.
        from torch._dynamo.utils import counters
        from torch._precompile import _parse_artifact_metadata

        with _CaptureToFiles(
            _precompile_unreachable_helper_caller,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
        ) as cap:
            with torch.no_grad():
                cap(torch.randn(4))
        code, cache = cap.result()
        meta = _parse_artifact_metadata(code)
        self.assertEqual(meta["SERVING_MODE"], "installed")
        self.assertIn(
            _precompile_unreachable_helper.__name__,
            meta["UNREACHABLE_WITHOUT_INSTALL"],
        )

        x = torch.randn(4)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        # The installed shape shares the runnable base with the standalone one.
        self.assertIsInstance(loaded, torch.compiler.PrecompiledRunnable)
        self.assertIsInstance(loaded, torch.compiler.PrecompiledCallable)
        self.assertTrue(loaded.installed)
        counters.clear()
        with loaded, torch.no_grad():
            self.assertEqual(loaded(x), _precompile_unreachable_helper_caller(x))
            # Served, not recompiled: the whole point of installing.
            self.assertEqual(counters["stats"]["unique_graphs"], 0)

    def test_a_subgraph_whose_lowering_fails_keeps_the_bundle(self):
        # Rendering is per subgraph: one that compile_to_python refuses stays
        # pickled in _BACKENDS, the header counts it, and the artifact serves.
        from torch._functorch import aot_autograd
        from torch._precompile import _read_literal

        real_compile_to_python = aot_autograd.compile_to_python
        refused = []

        def refuse_first(gm, example_inputs, **kwargs):
            if not refused:
                refused.append(gm)
                raise RuntimeError("synthetic lowering failure")
            return real_compile_to_python(gm, example_inputs, **kwargs)

        x = torch.randn(4, 8)
        with mock.patch.object(aot_autograd, "compile_to_python", refuse_first):
            with _CaptureToFiles(
                _precompile_multi_graph,
                backend="inductor",
                dynamic=False,
                tracer="dynamo",
            ) as cap:
                with torch.no_grad():
                    cap(x)
            code, cache = cap.result()
        self.assertEqual(len(refused), 1)
        self.assertIn("1 could not be rendered and stay in _BACKENDS", code)
        pickled = pickle.loads(
            base64.b64decode(_read_literal(ast.parse(code), "_BACKENDS"))
        )
        self.assertEqual(len(pickled), 1)
        # Two breaks make three subgraphs; the other two rendered as source.
        self.assertEqual(code.count("_SUBGRAPHS["), 2)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with torch.no_grad():
            self.assertEqual(loaded(x), _precompile_multi_graph(x))

    def test_installed_artifact_fn_must_match_the_capture(self):
        # load(fn=...) installs onto whatever function it is given, so it has
        # to refuse one the artifact was not captured from -- the same check
        # the path form makes -- or the wrong callable serves the captured
        # graphs. The function it WAS captured from is accepted.
        with _CaptureToFiles(
            _precompile_unreachable_helper_caller,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
        ) as cap:
            cap(torch.randn(4))
        code, cache = cap.result()
        x = torch.randn(4)
        torch._dynamo.reset()
        with self.assertRaisesRegex(
            PrecompileError, "was captured from .* but is being loaded onto"
        ):
            _load_pair(code, cache, fn=_precompile_unreachable_helper)
        loaded = _load_pair(code, cache, fn=_precompile_unreachable_helper_caller)
        with loaded, torch.no_grad():
            self.assertEqual(loaded(x), _precompile_unreachable_helper_caller(x))

    def test_installed_artifact_unload_releases_its_recorded_backends(self):
        # Installing files the artifact's backends into the process-global
        # PrecompileContext. unload() has to take exactly those out again, or
        # every install/unload cycle leaks them for the life of the process.
        with _CaptureToFiles(
            _precompile_unreachable_helper_caller,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
        ) as cap:
            cap(torch.randn(4))
        code, cache = cap.result()
        x = torch.randn(4)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        registry = PrecompileContext._backend_artifacts_by_key
        baseline = len(registry)
        for _ in range(2):
            with loaded, torch.no_grad():
                self.assertEqual(loaded(x), _precompile_unreachable_helper_caller(x))
                self.assertGreater(len(registry), baseline)
            self.assertEqual(len(registry), baseline)

    def test_installed_artifact_handle_lifecycle(self):
        # Installing mutates process-global code objects, so a call after
        # unload() must raise rather than silently re-install; a fresh load() is
        # the way to serve again. Two threads racing the FIRST call install once,
        # and a first call that reaches the install lock only after unload()
        # returned finds the handle retired and must not install.
        with _CaptureToFiles(
            _precompile_unreachable_helper_caller,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
        ) as cap:
            cap(torch.randn(4))
        code, cache = cap.result()
        x = torch.randn(4)
        expected = _precompile_unreachable_helper_caller(x)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with torch.no_grad():
            self.assertEqual(loaded(x), expected)
            loaded.unload()
            loaded.unload()  # idempotent
            with self.assertRaisesRegex(PrecompileError, "unloaded"):
                loaded(x)
            # Re-entering is the explicit way to install again; the handle is
            # reusable across scopes.
            with loaded:
                self.assertEqual(loaded(x), expected)
            with self.assertRaisesRegex(PrecompileError, "unloaded"):
                loaded(x)
            fresh = _load_pair(code, cache)
            try:
                self.assertEqual(fresh(x), expected)
            finally:
                fresh.unload()

        loaded = _load_pair(code, cache)
        handle = loaded._compiled
        original_serve = handle._serve
        serves = []

        def counting_serve(fn, **kwargs):
            serves.append(threading.current_thread().name)
            return original_serve(fn, **kwargs)

        handle._serve = counting_serve
        barrier = threading.Barrier(2)
        outcomes = []

        def first_call():
            barrier.wait()
            try:
                with torch.no_grad():
                    outcomes.append(loaded(x))
            except Exception as e:
                outcomes.append(e)

        threads = [threading.Thread(target=first_call) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        try:
            self.assertEqual(len(serves), 1)
            for out in outcomes:
                self.assertEqual(out, expected)
        finally:
            loaded.unload()

        loaded = _load_pair(code, cache)
        handle = loaded._compiled
        real_lock = handle._install_lock
        unloaded = threading.Event()

        class _GatedLock:
            def __enter__(self):
                if threading.current_thread().name == "racer":
                    unloaded.wait()
                return real_lock.__enter__()

            def __exit__(self, *exc):
                return real_lock.__exit__(*exc)

        handle._install_lock = _GatedLock()
        errors = []

        def racing_call():
            try:
                with torch.no_grad():
                    loaded(torch.randn(4))
            except PrecompileError as e:
                errors.append(e)

        racer = threading.Thread(target=racing_call, name="racer")
        racer.start()
        loaded.unload()
        unloaded.set()
        racer.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("unloaded", str(errors[0]))
        self.assertIsNone(handle._inner)

    def test_uncovered_call_outside_serving_tolerates_an_unpicklable_local(self):
        # An installed artifact recompiles an uncovered call outside serving()
        # the way torch.compile does. That frame is served, not captured, so
        # its guards are not serialized: a value they could not pickle (a lock
        # in the dict the function reads) is not an error.
        x = torch.randn(4)
        with _CaptureToFiles(
            _precompile_unreachable_helper_caller_with_opts,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
        ) as cap:
            cap(x, {"scale": 2})
        code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        uncovered = {"scale": 3, "lock": threading.Lock()}
        with loaded, torch.no_grad():
            self.assertEqual(
                loaded(x, {"scale": 2}), _precompile_unreachable_helper_caller(x) * 2
            )
            self.assertEqual(
                loaded(x, uncovered), _precompile_unreachable_helper_caller(x) * 3
            )

    def test_missing_captured_module_raises_precompile_error(self):
        # Both drivers import the modules the capture came from. A module that
        # is not importable here surfaces as the documented PrecompileError, not
        # as a bare ModuleNotFoundError from inside the exec'd driver.
        x = torch.randn(4)
        # Standalone: the frames' import aliases.
        with _CaptureToFiles(
            _precompile_single_graph, backend="eager", dynamic=False, tracer="dynamo"
        ) as cap:
            cap(x)
        code, cache = cap.result()
        from torch._precompile import _read_literal

        frames = pickle.loads(
            base64.b64decode(_read_literal(ast.parse(code), "_FRAMES"))
        )
        missing = next(iter(frames[0]["import_sources"].values()))
        real_import = importlib.import_module

        def import_without(name, *args, **kwargs):
            if name == missing:
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(importlib, "import_module", import_without):
            with self.assertRaisesRegex(PrecompileError, "not importable here"):
                _load_pair(code, cache)
        # Installed: the module each captured frame was defined in.
        name = "_precompile_missing_module_fixture"
        mod = self._module_with(
            textwrap.dedent(
                """
                import torch


                class Child(torch.nn.Module):
                    def forward(self, x):
                        y = x + 1
                        torch._dynamo.graph_break()
                        return y * 2


                def entry(child, x):
                    return child(x)
                """
            ),
            name,
        )
        try:
            with _CaptureToFiles(
                mod.entry,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(mod.Child(), x)
            code, cache = cap.result()
        finally:
            del sys.modules[name]
        from torch._precompile import _parse_artifact_metadata

        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], "installed")
        torch._dynamo.reset()
        with self.assertRaisesRegex(PrecompileError, "not importable here"):
            _load_pair(code, cache)

    @parametrize("case", list(_PRECOMPILE_TRAINING_CASES))
    @parametrize("backend", ("eager", "inductor"))
    def test_training_capture_serves_a_backward_without_a_loss(self, case, backend):
        # The capture never sees a loss and never calls .backward(). The joint
        # trace synthesizes tangents from the forward outputs, and the backward
        # is lowered eagerly, so a served output is still wired to
        # AOTAutograd's CompiledFunction and .backward() runs precompiled code.
        # The parameters carry a .grad into the capture: no guard reads it, so
        # the pickler prunes it to _Missing, which load must drop, not assign.
        model_cls, entry, width = _PRECOMPILE_TRAINING_CASES[case]
        model = model_cls()
        xs = [torch.randn(n, width) for n in (4, 6)]
        expected = []
        for x in xs:
            model.zero_grad(set_to_none=True)
            entry(model, x).sum().backward()
            expected.append([p.grad.clone() for p in model.parameters()])

        with _CaptureToFiles(
            entry,
            backend=backend,
            dynamic=False,
            tracer="dynamo",
            training=True,
            require_complete=False,
            require_no_risky_drops=False,
        ) as cap:
            for _ex_args in [(model, x) for x in xs]:
                cap(*_ex_args)
        code, cache = cap.result()
        # The example calls left the caller's gradients exactly as they were.
        for want, param in zip(expected[-1], model.parameters()):
            self.assertEqual(want, param.grad)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        from torch._dynamo.utils import counters

        counters.clear()
        with _maybe_scoped(loaded):
            for x, grads in zip(xs, expected):
                model.zero_grad(set_to_none=True)
                out = loaded(model, x)
                self.assertTrue(out.requires_grad)
                if backend == "inductor" and case == "plain":
                    # AOTAutograd's compiled backward, not an eager autograd node.
                    self.assertIn("CompiledFunction", type(out.grad_fn).__name__)
                out.sum().backward()
                for want, param in zip(grads, model.parameters()):
                    self.assertEqual(want, param.grad)
        # Served, not recompiled: forward and backward alike.
        self.assertEqual(counters["stats"]["unique_graphs"], 0)

    def test_training_capture_does_not_read_grad_off_non_leaf_tensors(self):
        # The pickler reads .grad off the tensors it serializes. On a non-leaf
        # that is always None and warns; a training capture guards such tensors.
        model = _PrecompileTrainMod()
        x = torch.randn(4, 8, requires_grad=True) * 2
        self.assertFalse(x.is_leaf)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with _CaptureToFiles(
                _precompile_call_model,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                training=True,
            ) as cap:
                cap(model, x)
        leaf_warnings = [str(w.message) for w in caught if "non-leaf" in str(w.message)]
        self.assertEqual(leaf_warnings, [])

    @parametrize("tracer", ("make_fx", "dynamo"))
    @parametrize("training", (False, True))
    def test_example_call_runs_in_the_grad_mode_training_selects(
        self, tracer, training
    ):
        # Capture is caller-driven, so the caller picks the grad mode and
        # precompile runs the call in it -- no_grad for inference, enable_grad
        # for a training capture -- rather than choosing one itself.
        _PRECOMPILE_GRAD_MODES_SEEN.clear()
        m = torch.nn.Linear(4, 3).eval()
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with _CaptureToFiles(
            _precompile_observe_grad_mode,
            backend="eager",
            tracer=tracer,
            training=training,
        ) as cap:
            with grad_ctx:
                cap(m, torch.randn(2, 4))
        self.assertEqual(set(_PRECOMPILE_GRAD_MODES_SEEN), {training})

    def test_inference_capture_stays_grad_free(self):
        # The default is unchanged: examples run under no_grad, so a served
        # output carries no autograd history.
        model = _PrecompileTrainMod()
        x = torch.randn(4, 8)
        with _CaptureToFiles(
            _precompile_call_model, backend="eager", dynamic=False, tracer="dynamo"
        ) as cap:
            with torch.no_grad():
                cap(model, x)
        code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with torch.no_grad():
            self.assertFalse(loaded(model, x).requires_grad)

    @parametrize("shape", list(_BREAKING_MODELS))
    @parametrize("backend", ["eager", "inductor"])
    def test_dynamo_tracer_serves_each_graph_break_shape(self, shape, backend):
        # The dynamo tracer against graph breaks and recompilations. Every case
        # asserts the artifact serves, and that it agrees with torch.compile --
        # parity with torch.compile is the contract, so torch.compile is the
        # reference rather than eager.
        from torch._dynamo.utils import counters

        torch._dynamo.reset()
        model = _BREAKING_MODELS[shape]().eval()
        x = torch.randn(4, 4)
        with torch.no_grad():
            reference = torch.compile(_brk_call, backend=backend)(model, x)

        torch._dynamo.reset()
        if shape == "break_in_loop":
            # Dynamo skips a frame that breaks inside a loop, so torch.compile
            # runs this shape eager and there is no graph to serve. The capture
            # is refused as empty rather than shipped as an eager artifact.
            with (
                torch.no_grad(),
                self.assertRaisesRegex(PrecompileError, "compiled no graph"),
            ):
                with _CaptureToFiles(
                    _brk_call,
                    backend=backend,
                    tracer="dynamo",
                    require_no_risky_drops=False,
                ) as cap:
                    cap(model, x)
                cap.result()
                # cap.result() runs after the capture block exits (its render
                # error is raised there, not at __exit__).
            return
        with torch.no_grad():
            with _CaptureToFiles(
                _brk_call,
                backend=backend,
                tracer="dynamo",
                require_complete=False,
                require_no_risky_drops=False,
            ) as cap:
                cap(model, x)
            code, cache = cap.result()
        self.assertGreater(_compiled_subgraph_count(code), 0)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        counters.clear()
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(model, x), reference)
        # Nothing compiled while serving, and an uncovered call would have
        # raised (standalone) or compiled (installed): a captured variant served.
        self.assertEqual(counters["stats"]["unique_graphs"], 0)

    @parametrize("backend", ["eager", "inductor"])
    def test_dynamo_tracer_serves_every_captured_recompilation(self, backend):
        # Two axes vary -- a bool that changes the branch and a shape that
        # changes the specialization -- so the capture holds several guarded
        # variants of the same frames, on both sides of the break. An UNCOVERED
        # call is never answered wrong: a STANDALONE artifact has no compiler
        # behind it and refuses; an INSTALLED one is on the frame evaluator and
        # recompiles exactly as torch.compile would.
        from torch._dynamo.utils import counters
        from torch._precompile import _parse_artifact_metadata

        torch._dynamo.reset()
        model = _BrkBranchy().eval()
        calls = [
            (model, torch.randn(n, 4), flag) for n in (4, 6) for flag in (False, True)
        ]
        uncovered = (model, torch.randn(5, 4), True)
        torch._dynamo.reset()
        compiled = torch.compile(_brk_call_flag, backend=backend, dynamic=False)
        with torch.no_grad():
            reference = [compiled(*c) for c in calls + [uncovered]]

        torch._dynamo.reset()
        with torch.no_grad():
            with _CaptureToFiles(
                _brk_call_flag,
                backend=backend,
                dynamic=False,
                tracer="dynamo",
                require_complete=False,
                require_no_risky_drops=False,
            ) as cap:
                for _ex_args in calls:
                    cap(*_ex_args)
            code, cache = cap.result()
        meta = _parse_artifact_metadata(code)
        # The capture really did hold several variants, and really did break.
        self.assertTrue(any(v > 1 for _, v in meta["FRAMES"]))
        self.assertTrue(any("resume_in" in n for n, _ in meta["FRAMES"]))

        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        counters.clear()
        with _maybe_scoped(loaded), torch.no_grad():
            for call, want in zip(calls, reference):
                self.assertEqual(loaded(*call), want)
            # Served, not recompiled: every one of those calls was captured.
            self.assertEqual(counters["stats"]["unique_graphs"], 0)
            if meta["SERVING_MODE"] == "installed":
                self.assertEqual(loaded(*uncovered), reference[-1])
            else:
                with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                    loaded(*uncovered)

    @parametrize("case", list(_PRECOMPILE_GUARD_POLICY_CASES))
    def test_guard_policy_keeps_what_discriminates(self, case):
        from torch._precompile import _parse_artifact_metadata, _read_literal

        fn, examples, served, refused = _PRECOMPILE_GUARD_POLICY_CASES[case]
        with torch.no_grad():
            expected = [fn(*args) for args in served]
        torch._dynamo.reset()
        with _CaptureToFiles(
            fn, backend="eager", dynamic=False, tracer="dynamo"
        ) as cap:
            with torch.no_grad():
                for _ex_args in examples:
                    cap(*_ex_args)
        code, cache = cap.result()
        self.assertEqual(_parse_artifact_metadata(code)["TRACER"], "dynamo")
        # The policy drops guards that could have been serialized, so what it
        # discarded has to stay visible in the header even though it is now
        # applied after the capture rather than during it.
        dropped = _read_literal(ast.parse(code), "POLICY_DROPPED_GUARDS")
        self.assertIn(["AUTOGRAD_SAVED_TENSORS_HOOKS", ""], dropped)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            for args, want in zip(served, expected):
                self.assertEqual(loaded(*args), want)
            for args in refused:
                with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                    loaded(*args)

    def test_invariants_report_lists_policy_dropped_slots(self):
        # The report header promises a dropped line is "a precondition NOTHING
        # checks". A policy-dropped slot is exactly that, but the policy used
        # to DELETE those facts from the report's source data rather than
        # re-mark them, so an auditor reading the file saw a validity domain
        # far wider than the artifact's true one.
        import re as re_mod

        from torch._precompile import _read_literal

        # With hook guards on, a module argument's EMPTY_NN_MODULE_HOOKS_DICT
        # slots are droppable invariants with real source names, so the
        # per-name check below is not vacuous.
        model = _PrecompileTrainMod()
        with (
            tempfile.TemporaryDirectory() as d,
            torch._dynamo.config.patch(skip_nnmodule_hook_guards=False),
        ):
            path = os.path.join(d, "invariants.txt")
            with _CaptureToFiles(
                _precompile_call_model,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                invariants=path,
            ) as cap:
                for _ex_args in [(model, torch.randn(n, 8)) for n in (3, 5)]:
                    cap(*_ex_args)
            code, _ = cap.result()
            dropped = _read_literal(ast.parse(code), "POLICY_DROPPED_GUARDS")
            with open(path) as f:
                report = f.read()
        named = [name for _guard_type, name in dropped if name]
        self.assertTrue(named)
        for name in named:
            self.assertRegex(report, rf"\[dropped \][^\n]*{re_mod.escape(name)}")

    def test_a_capture_without_compiled_code_is_not_complete(self):
        # allow_empty_graphs keeps a frame that compiled nothing as one guarded
        # code, so guarded_codes alone cannot tell a real capture from one whose
        # whole body sits behind torch._dynamo.disable. Such a capture carries
        # no compiled compute: complete says so, and require_complete gates
        # exactly complete.
        from torch._precompile import _parse_artifact_metadata

        session = _precompile_capture(
            _precompile_only_disabled, backend="eager", dynamic=False
        )
        # The block exits cleanly, so the default require_complete gate refuses
        # to render the graphless capture when it does.
        with self.assertRaisesRegex(PrecompileError, "compiled no graph"):
            with session as compiled, torch.no_grad():
                compiled(torch.randn(3))
        summary = session.summary()
        self.assertEqual(summary.guarded_codes, 1)
        self.assertEqual(summary.backend_graphs, 0)
        self.assertFalse(summary.complete)
        relaxed = _precompile_capture(
            _precompile_only_disabled,
            backend="eager",
            dynamic=False,
            require_complete=False,
        )
        with relaxed as compiled, torch.no_grad():
            compiled(torch.randn(3))
        code, _cache = relaxed.result()
        self.assertEqual(_parse_artifact_metadata(code)["TRACER"], "dynamo")

    @parametrize("root", ["local", "module"])
    def test_a_dict_membership_guard_is_never_dropped(self, root):
        # `"flag" in d` is a branch, and the guard pinning it (DICT_NOT_CONTAINS)
        # is a Python fact about a container's contents. Whatever the policy
        # makes of the root, dropping it serves the captured branch to a caller
        # who holds the key -- so the artifact must refuse, not answer. The same
        # guard on `self.opts` is MODULE-rooted, which the invariant policy
        # classes as environment -- and used to drop: M({"flag": 1}) was
        # answered with M({})'s branch, no error, at the default gates.
        from torch._precompile import _read_literal

        x = torch.ones(2)
        if root == "local":
            fn, served, refused = (
                _precompile_dict_flag_branch,
                (x, {}),
                (x, {"flag": 1}),
            )
        else:
            captured, other = (
                _PrecompileOptsModule({}),
                _PrecompileOptsModule({"flag": 1}),
            )
            other.load_state_dict(captured.state_dict())
            fn, served, refused = _precompile_call_model, (captured, x), (other, x)
        with torch.no_grad():
            expected = fn(*served)
            with _CaptureToFiles(
                fn, backend="eager", dynamic=False, tracer="dynamo"
            ) as cap:
                cap(*served)
            code, cache = cap.result()
        dropped = _read_literal(ast.parse(code), "POLICY_DROPPED_GUARDS")
        self.assertNotIn("DICT_NOT_CONTAINS", {guard_type for guard_type, _ in dropped})
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(*served), expected)
            with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                loaded(*refused)

    @parametrize("where", ["object", "in_a_list"])
    def test_unpicklable_guard_value_names_where_it_lives(self, where):
        # The type in a pickle error says WHAT failed and never WHERE, which on a
        # large model means bisecting by hand. A lock is the archetypal offender
        # and the one the type-name match used to miss, because CPython reports
        # it as '_thread.lock' while type(...).__name__ is 'lock'.
        if where == "object":
            entry = _precompile_reads_holder
            args = (_PrecompileUnpicklableHolder(threading.Lock()), torch.randn(4))
            expected = r"reached via: local_scope\['obj'\].bad"
        else:
            entry = _precompile_reads_holder_in_list
            args = ([_PrecompileUnpicklableHolder(threading.Lock())], torch.randn(4))
            expected = r"reached via: local_scope\['objs'\]\[0\].bad"
        with self.assertRaisesRegex(PrecompileError, expected), torch.no_grad():
            with _CaptureToFiles(
                entry,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(*args)

    def test_serialized_guards_carry_no_bookkeeping(self):
        # Guard.code_list and .guard_types are rebuilt by create_fn at load, and
        # set_export_info EXTENDS them on every guard build, so shipping them
        # means shipping each code part once per build. Applying the policy
        # re-serializes guard state whose tensors are now the FAKES the first
        # pass wrote, and empty_like() under a live FakeTensorMode hands back
        # another fake -- which pickles the mode, its converters and their
        # weakrefs along with it.
        from torch._dynamo.package import load_guards_state
        from torch._precompile import _read_literal

        model = torch.nn.Linear(8, 4).eval()
        xs = [torch.randn(n, 8) for n in (3, 5)]
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_call_model, backend="eager", dynamic=False, tracer="dynamo"
            ) as cap:
                for _ex_args in [(model, x) for x in xs]:
                    cap(*_ex_args)
            code, cache = cap.result()
        blob = base64.b64decode(_read_literal(ast.parse(code), "_FRAMES"))
        for name in (b"FakeTensorMode", b"MetaTensorDescriber", b"WeakIdRef"):
            self.assertNotIn(name, blob)
        for frame in pickle.loads(blob):
            for variant in frame["variants"]:
                state = load_guards_state(variant["guards_state"])
                for guard in state.output_graph.guards:
                    self.assertIsNone(guard.code_list)
                    self.assertIsNone(guard.guard_types)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            for x in xs:
                self.assertEqual(loaded(model, x), model(x))

    def test_a_mutating_module_is_guarded_on_what_the_capture_saw(self):
        # Learning the guard policy from a throwaway first capture would run
        # every example twice. A counter advanced by the capture itself is baked
        # into the guards. It has to be the value the ONE pass saw, or a fresh
        # model never matches: a discarded first pass would leave every variant
        # pinned to a step the served model has not reached.
        torch.manual_seed(0)
        model = _PrecompileStepCounter()
        xs = [torch.randn(n, 8) for n in (2, 3, 4)]
        with torch.no_grad():
            with _CaptureToFiles(
                _brk_call,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                for _ex_args in [(model, x) for x in xs]:
                    cap(*_ex_args)
            code, cache = cap.result()
        torch._dynamo.reset()
        torch.manual_seed(0)
        cold = _PrecompileStepCounter()
        torch.manual_seed(0)
        reference = _PrecompileStepCounter()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            for x in xs:
                self.assertEqual(loaded(cold, x), _brk_call(reference, x))

    @parametrize("junk", ["dtype", "int", "str", "device"])
    def test_unguarded_interned_attribute_does_not_poison_the_artifact(self, junk):
        # Value pruning is keyed by id(), which asks "same OBJECT" when it means
        # "same REFERENCE". For an interned value those come apart: an unguarded
        # module attribute holding torch.float32 registers that id as missing,
        # and every OTHER reference to that dtype -- including ones the artifact
        # needs -- then resolves to the sentinel, so it fails to load with
        # "empty_strided(): argument 'dtype' must be torch.dtype, not _Missing".
        value = {
            "dtype": torch.float32,
            "int": 8,
            "str": "cuda",
            "device": torch.device("cpu"),
        }[junk]
        model = _PrecompileUnguardedAttr(value)
        x = torch.randn(4, 8)
        with torch.no_grad():
            expected = model(x)
            with _CaptureToFiles(
                _brk_call, backend="inductor", dynamic=False, tracer="dynamo"
            ) as cap:
                cap(model, x)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(model, x), expected)

    @parametrize("entry", list(_PRECOMPILE_CLOSURE_ENTRIES))
    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_closure_entry_is_refused(self, entry, backend):
        factory, make_args = _PRECOMPILE_CLOSURE_ENTRIES[entry]
        with self.assertRaisesRegex(PrecompileError, "closes over"):
            with _CaptureToFiles(factory(), tracer="dynamo", backend=backend) as cap:
                with torch.no_grad():
                    cap(*make_args())
            cap.result()

    @parametrize("case", ["inference", "training", "two_variants_of_one_frame"])
    def test_dynamo_tracer_renders_kernels_as_source(self, case):
        # A compiled subgraph is Inductor output, which has a source form -- so
        # the dynamo tracer emits it rather than pickling it, leaving only the
        # guard trees and bytecode opaque. This holds for a TRAINING capture too:
        # its forward and backward are both rendered and bridged by an emitted
        # autograd.Function. Two variants of one frame render the SAME names into
        # one namespace, so without per-subgraph renaming the first variant would
        # silently run the second's code.
        training = case == "training"
        if case == "two_variants_of_one_frame":
            fn, xs = _precompile_scale_sum, [torch.randn(2, 8), torch.randn(4, 8)]
        else:
            fn, xs = (
                _PrecompileBreakingModule().eval(),
                [torch.randn(3, 8), torch.randn(5, 8)],
            )
        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            with _CaptureToFiles(
                fn,
                backend="inductor",
                dynamic=False,
                tracer="dynamo",
                training=training,
            ) as cap:
                for _ex_args in [(x,) for x in xs]:
                    cap(*_ex_args)
            code, cache = cap.result()
        self.assertIn("_SUBGRAPHS[", code)

        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        # Serve in the mode it was captured in: grad mode is a GLOBAL_STATE
        # guard, and it is checked.
        with _maybe_scoped(loaded), ctx:
            for x in xs:
                # entry frame is forward, so the receiver is passed explicitly
                args = (fn, x) if isinstance(fn, torch.nn.Module) else (x,)
                self.assertEqual(loaded(*args), fn(x))

    def test_dynamo_tracer_names_why_a_subgraph_stayed_pickled(self):
        # A subgraph the composer refuses stays a pickled bundle, and the
        # artifact says so in its header, with the reason, rather than shipping
        # base64 under a bare OPAQUE banner; the same reason is warned at capture.
        model = _PrecompileAliasedMutationModule()
        base = torch.arange(4, dtype=torch.float32) + 1
        with self.assertLogs(
            "torch._dynamo.precompile_package", level="WARNING"
        ) as logs:
            with _CaptureToFiles(
                model, backend="inductor", dynamic=False, tracer="dynamo", training=True
            ) as cap:
                cap(base[:], base)
            code, _cache = cap.result()
        self.assertTrue(any("synthetic_base_wrapper" in m for m in logs.output))
        self.assertNotIn("_SUBGRAPHS[", code)
        header = [line for line in code.splitlines() if "stays pickled:" in line]
        self.assertEqual(len(header), 1)
        self.assertTrue(header[0].startswith("#    __compiled_fn_"), header[0])
        self.assertIn("NotImplementedError", header[0])
        self.assertIn("synthetic_base_wrapper", header[0])

    @parametrize("construct", sorted(_EAGER_ROUND_TRIP))
    @parametrize("broken", [False, True])
    def test_eager_backend_graph_survives_serialization(self, construct, broken):
        # An eager subgraph ships as a pickled GraphModule, whose reduction keeps
        # only the generated source and re-derives the Graph by re-tracing it. A HOP
        # explodes on the Proxy; autocast's enter/exit take no Proxy at all, so the
        # retrace RUNS them and drops the nodes -- served output was fp32; the same
        # for _set_grad_enabled, which also leaked into the LOADER's grad mode.
        from torch._precompile import _parse_artifact_metadata

        entry, args = (
            (_eager_rt_broken, (construct,))
            if broken
            else (_EAGER_ROUND_TRIP[construct], ())
        )
        training = construct == "no_grad_region"
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        x = torch.randn(4, 4, requires_grad=training)
        with grad_ctx:
            expected = torch.compile(entry, backend="eager")(*args, x)
        self.assertFalse(expected.requires_grad)
        torch._dynamo.reset()
        with _CaptureToFiles(
            entry,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
            training=training,
            require_no_risky_drops=False,
        ) as cap:
            with grad_ctx:
                cap(*args, x)
        code, cache = cap.result()
        self.assertEqual(
            _parse_artifact_metadata(code)["SERVING_MODE"],
            "installed" if broken else "standalone",
        )
        torch._dynamo.reset()
        with torch.no_grad():
            loaded = _load_pair(code, cache)
            self.assertFalse(torch.is_grad_enabled())
        with _maybe_scoped(loaded), grad_ctx:
            served = loaded(*args, x)
        self.assertFalse(served.requires_grad)
        self.assertEqual(served, expected)

    def test_multi_graph_bare_module_capture_is_refused(self):
        # Handing precompile a bare nn.Module compiles Dynamo's OWN wrapper
        # frame (wrap_inline's `inner`), which closes over the module: the
        # entry frame holds no graph, and no artifact can rebuild that closure.
        # Refuse, and name the spelling that works.
        model = torch.nn.Linear(8, 4).eval()
        x = torch.randn(3, 8)
        with self.assertRaisesRegex(
            torch._precompile.PrecompileError, "captured no dispatchable graph"
        ):
            with _CaptureToFiles(
                model, backend="eager", dynamic=False, tracer="dynamo"
            ) as cap:
                with torch.no_grad():
                    cap(x)
            cap.result()

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

    def test_make_fx_artifact_ignores_ambient_autocast(self):
        # The graph was traced with autocast off, so the artifact runs it that
        # way: a caller's autocast region must not change what it computes.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        expected = m(x)
        with _CaptureToFiles(lambda model, xx: model(xx)) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn("GRAPH_DEVICES = ('cpu',)", code)
        f = _load_pair(code, cache)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = f(m, x)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out, expected)

    def test_guarded_user_object_prunes_its_unguarded_attributes(self):
        # Pruning used to apply only to nn.Module, so a guarded object that is
        # NOT a module -- a train pipeline holding a dataloader -- was pickled
        # whole and one unguarded attribute several levels down took the frame
        # with it ("cannot pickle 'generator' object"), which surfaced only as
        # "the entry frame produced no guarded code".
        x = torch.randn(4, 8)
        pipeline = _PrecompilePipeline(torch.nn.Linear(8, 8))
        with torch.no_grad():
            expected = _precompile_via_pipeline(pipeline, x)
            with _CaptureToFiles(
                _precompile_via_pipeline,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
            ) as cap:
                cap(pipeline, x)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(pipeline, x), expected)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_capture_limits_and_frame_state_are_scoped_to_the_session(self):
        # Neither recompile limits nor automatic-dynamic frame state cross
        # between a session and ordinary compiles of the same code.
        from torch._C._dynamo.eval_frame import get_code_exec_strategy
        from torch._dynamo.types import FrameAction

        def ordinary_frame_count():
            counter = torch._dynamo.testing.CompileCounter()
            compiled = torch.compile(
                _precompile_single_graph, backend=counter, dynamic=None
            )
            for n in (4, 5):
                x = torch.randn(n)
                self.assertEqual(compiled(x), _precompile_single_graph(x))
            return counter.frame_count

        torch._dynamo.reset()
        self.assertEqual(ordinary_frame_count(), 2)
        torch._dynamo.reset()
        with torch._dynamo.config.patch(
            accumulated_recompile_limit=2, recompile_limit=8
        ):
            inputs = [torch.randn(n) for n in (2, 3, 4)]
            session = _precompile_capture(
                _precompile_single_graph,
                backend="eager",
                dynamic=False,
                recompile_limit=20,
            )
            with session as captured:
                for x in inputs:
                    self.assertEqual(captured(x), _precompile_single_graph(x))
                during = torch._dynamo.testing.CompileCounter()
                ordinary = torch.compile(
                    _precompile_single_graph, backend=during, dynamic=False
                )
                x = torch.randn(5)
                self.assertEqual(ordinary(x), _precompile_single_graph(x))
                self.assertEqual(during.frame_count, 1)
            summary = session.summary()
            self.assertTrue(summary.complete)
            self.assertEqual(summary.truncated, ())
            self.assertEqual(summary.guarded_codes, len(inputs))
            after = torch._dynamo.testing.CompileCounter()
            ordinary = torch.compile(
                _precompile_single_graph, backend=after, dynamic=False
            )
            x = torch.randn(6)
            self.assertEqual(ordinary(x), _precompile_single_graph(x))
            self.assertEqual(after.frame_count, 1)
            strategy = get_code_exec_strategy(_precompile_single_graph.__code__)
            self.assertEqual(strategy.cur_action, FrameAction.DEFAULT)
            self.assertEqual(strategy.recursive_action, FrameAction.DEFAULT)

        torch._dynamo.reset()
        warm = torch.compile(_precompile_multi_graph, backend="eager", dynamic=False)
        for n in (2, 3):
            x = torch.randn(n, 8)
            self.assertEqual(warm(x), _precompile_multi_graph(x))
        x = torch.randn(4, 8)
        session = _precompile_capture(
            _precompile_multi_graph,
            backend="eager",
            dynamic=False,
            recompile_limit=2,
        )
        with session as compiled:
            self.assertEqual(compiled(x), _precompile_multi_graph(x))
        summary = session.summary()
        self.assertTrue(summary.complete)
        self.assertEqual(summary.truncated, ())
        self.assertEqual(summary.uncovered_frames, ())
        self.assertEqual(summary.guarded_codes, 3)

        session = _precompile_capture(
            _precompile_single_graph, backend="eager", dynamic=None
        )
        with session as compiled:
            for n in (2, 3):
                x = torch.randn(n)
                self.assertEqual(compiled(x), _precompile_single_graph(x))
        self.assertEqual(ordinary_frame_count(), 2)

    @parametrize("backend", ["eager", "inductor"])
    def test_multi_graph_round_trip_exercised_empty_resume(self, backend):
        x = torch.arange(3.0)
        expected = [_precompile_empty_resume(x, flag) for flag in (False, True)]
        session = _precompile_capture(
            _precompile_empty_resume,
            backend=backend,
            dynamic=False,
            require_no_dropped_guards=False,
        )
        with session as compiled:
            with torch.no_grad():
                compiled(x, False)
                compiled(x, True)
        self.assertTrue(session.summary().complete)

        code, cache = session.result()
        torch._dynamo.reset()
        with self.assertLogs("torch._precompile", level="WARNING"):
            loaded = _load_pair(code, cache)
        with torch.no_grad():
            for flag, want in zip((False, True), expected):
                self.assertEqual(loaded(x, flag), want)

    def test_dynamo_captures_a_keyword_call(self):
        # The dynamo tracer takes keyword arguments, so the loaded artifact serves
        # the same keywords rather than positional arguments only.
        from torch._precompile import _parse_artifact_metadata

        x = torch.randn(3)
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_scaled,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
            ) as cap:
                cap(x, 1)
                cap(x, k=2)
            code, cache = cap.result()
        frames = _parse_artifact_metadata(code)["FRAMES"]
        self.assertEqual([count for _, count in frames], [2])
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with torch.no_grad():
            self.assertEqual(loaded(x, k=2), x * 2)
            self.assertEqual(loaded(x, 2), x * 2)
            self.assertEqual(loaded(x, 1), x)

    @parametrize("bounds", [None, (4, 16)])
    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_mark_unbacked_runs_across_sizes(self, bounds, backend):
        # Dynamic shapes are opt-in via mark_unbacked for the dynamo tracer too: Dynamo
        # captures the marked dim as an UNBACKED symint, so the ONE artifact serves any
        # runtime size of that dim on either backend (the make_fx tracer's eager backend
        # rejects dynamic dims; the dynamo subgraph carries its own runtime asserts, so it
        # does not have to). mark_unbacked's min/max become ShapeEnv runtime asserts,
        # which Dynamo emits into the subgraph itself -- so the artifact rejects an
        # out-of-range runtime size even though the thin dynamo driver has no bounds
        # check of its own (the make_fx tracer instead re-checks the bounds in its driver).
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).eval()
        x = torch.randn(8, 4)
        if bounds is None:
            mark_unbacked(x, 0)
        else:
            mark_unbacked(x, 0, min=bounds[0], max=bounds[1])
        with _CaptureToFiles(
            lambda model, xx: model(xx), training=True, tracer="dynamo", backend=backend
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            for bs in (8, 16, 1, 0) if bounds is None else (8, 16, 6):
                xt = torch.randn(bs, 4)
                self.assertEqual(f_c(m, xt), m(xt))
            if bounds is not None:
                with self.assertRaisesRegex(PrecompileError, "no captured variant"):
                    f_c(m, torch.randn(2, 4))

    def test_tracer_dynamo_cross_tracer_cache_rejected(self):
        # A cache from the make_fx tracer paired with dynamo python_code is a mismatched
        # pairing and must be rejected (the code_hash and the tracer tag both catch it),
        # not run under foreign metadata.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, xx: model(xx), tracer="dynamo") as cap:
            with torch.no_grad():
                cap(m, x)
        dyn_code, _ = cap.result()
        with _CaptureToFiles(lambda model, xx: model(xx)) as cap:
            cap(m, x)
        _, mf_cache = cap.result()
        with self.assertRaisesRegex(PrecompileError, "tracer"):
            _load_pair(dyn_code, mf_cache)

    @parametrize("output", list(_PRECOMPILE_OUTPUT_SHAPES))
    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_output_structures_round_trip(self, output, backend):
        # Under dynamo the transformed bytecode (NOT the driver's OUT_SPEC path make_fx
        # uses) reassembles fn's output, so each leaf is checked on both backends: a
        # mis-ordered flatten or a dropped dict leaf would silently corrupt the output
        # structure, a global constant left dangling would NameError at load, and a
        # non-tensor output is a DIVERGENCE from make_fx, which REJECTS it on inductor
        # (test_nontensor_output_inductor_clean_error).
        fn = _PRECOMPILE_OUTPUT_SHAPES[output]
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(
            fn, training=True, tracer="dynamo", backend=backend
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            out, ref = f_c(m, x), fn(m, x)
            self.assertEqual(out, ref)
            if output == "by_reference_callable":
                self.assertIs(out[1], ref[1])

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_dtensor_subclass(self, backend):
        # The dynamo-tracer analog of test_dtensor_subclass: a DTensor param/input must
        # round-trip on both backends. The dynamo eager emit inlines the subgraph against an
        # empty _GraphSelf and splats flat args, and inductor lowers via AOTAutograd's subclass
        # wrap/unwrap -- neither is exercised by the dense-input dynamo tests, so cover both.
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        saved_env = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            for name, p in list(m.named_parameters()):
                setattr(
                    m,
                    name,
                    torch.nn.Parameter(
                        distribute_tensor(p.detach(), mesh, [Replicate()])
                    ),
                )
            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            ref = m(x)
            with _CaptureToFiles(
                lambda model, xx: model(xx),
                training=True,
                tracer="dynamo",
                backend=backend,
            ) as cap:
                cap(m, x)
            code, cache = cap.result()
            # Exercise BOTH reload paths on the DTensor artifact: load() takes the
            # bundled-artifact path (primes the cache, then execs python_code), while the
            # direct exec below runs the self-contained python_code with no cache.
            f_c = _load_pair(code, cache)
            self.assertEqual(f_c(m, x).to_local(), ref.to_local())
            ns = {"__name__": "_dt"}
            exec(compile(code, "<dt>", "exec"), ns)
            self.assertEqual(ns["forward"](m, x).to_local(), ref.to_local())
        finally:
            dist.destroy_process_group()
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @parametrize("call", ["omit_all_defaults", "pass_the_first_positionally"])
    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_defaults_roundtrip(self, call, backend):
        # fn with positional defaults and a keyword-only default drives the driver's
        # argdefs / kwdefaults restoration; the defaults must be honored at the runtime
        # call (omitting them would TypeError / use a wrong value if they were dropped).
        # __defaults__ covers the LAST len(defaults) parameters: a call passing the
        # first defaulted argument positionally must still see the second one's real
        # default in the guard check, not the first one's.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        args = (m, x) if call == "omit_all_defaults" else (m, x, 3.0)
        with _CaptureToFiles(
            _precompile_with_defaults, training=True, tracer="dynamo", backend=backend
        ) as cap:
            cap(*args)
        code, cache = cap.result()
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(f_c(*args), _precompile_with_defaults(*args))

    def test_tracer_dynamo_static_under_dynamic_config(self):
        # dynamic=None is the ambient config (test_automatic_dynamic_promotes_the_frames_that_varied
        # relies on that), so a STATIC capture is spelled dynamic=False, which pins
        # assume_static_by_default and automatic_dynamic_shapes for the capture
        # (eval_frame.make_set_enable_dynamic). The eager subgraph ships as a pickle, so the
        # only observable of "static" is dispatch: an unseen batch size must be refused.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with torch._dynamo.config.patch(assume_static_by_default=False):
            with _CaptureToFiles(
                lambda model, xx: model(xx),
                training=True,
                tracer="dynamo",
                backend="eager",
                dynamic=False,
            ) as cap:
                cap(m, x)
            code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        with self.assertRaisesRegex(PrecompileError, "no captured variant"):
            f_c(m, torch.randn(6, 4))

        # Shape history is per capture: precompiling the SAME fn (one code object) at a
        # second shape does not promote the batch dim, because each session opens its own
        # isolated region with fresh frame state. Both artifacts serve only what they saw.
        def f(model, xx):
            return model(xx)

        with _CaptureToFiles(f, training=True, tracer="dynamo", backend="eager") as cap:
            cap(m, torch.randn(5, 4))
        c1, _ = cap.result()
        with _CaptureToFiles(f, training=True, tracer="dynamo", backend="eager") as cap:
            cap(m, torch.randn(7, 4))
        c2, _ = cap.result()
        self.assertNotIn("SymInt", c1)
        self.assertNotIn("SymInt", c2)

    def test_load_cache_without_tracer_key(self):
        # BC: a cache produced before the dynamo tracer existed carries no "tracer" key in
        # its envelope. load() must accept such an envelope. (It does not read the key at
        # this commit; the tracer="dynamo" wiring adds the read and this test then guards
        # its "make_fx" default.) Simulate a legacy envelope by deleting the key. Assert
        # the cache envelope was actually CONSUMED (no "could not read the cache envelope"
        # warning): a KeyError would be swallowed by load()'s except and fall back to JIT,
        # which still returns the right answer -- so an output-only assertion would not
        # guard the default.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, xx: model(xx)) as cap:
            cap(m, x)
        code, cache = cap.result()
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        del blob["tracer"]
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            f_c = _load_pair(code, buf.getvalue())
        self.assertFalse(
            any("could not read the cache envelope" in msg for msg in cm.output),
            f"legacy cache envelope was not consumed (fell back to JIT): {cm.output}",
        )
        self.assertEqual(f_c(m, x), m(x))

    def test_backend_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(
            ValueError, "backend must be 'inductor' or 'eager'"
        ):
            with _CaptureToFiles(lambda x, y: x + y, backend="nope") as cap:
                cap(a, b)

    def test_tracer_default_and_explicit(self):
        # capture()'s tracer defaults to DynamoTracer(); MakeFxTracer is the
        # single-trace alternative. Drive the real API (not the string shim) so
        # the default and both explicit tracers are locked in.
        self.assertIsInstance(
            inspect.signature(torch.compiler.precompile.capture)
            .parameters["tracer"]
            .default,
            DynamoTracer,
        )
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        for tracer in (None, DynamoTracer(), MakeFxTracer()):
            kwargs = {} if tracer is None else {"tracer": tracer}
            with tempfile.TemporaryDirectory() as d:
                ap, cp = os.path.join(d, "m.py"), os.path.join(d, "m.cache")
                with (
                    torch.no_grad(),
                    torch.compiler.precompile.capture(
                        lambda model, xx: model(xx),
                        artifact_path=ap,
                        cache_path=cp,
                        backend="eager",
                        **kwargs,
                    ) as cap,
                ):
                    cap(m, x)
                loaded = torch.compiler.precompile.load(ap, cp)
                with _maybe_scoped(loaded), torch.no_grad():
                    self.assertEqual(loaded(m, x), m(x))

    def test_tracer_invalid_raises(self):
        # capture() takes a tracer OBJECT; a wrong type is a TypeError naming the
        # accepted tracers.
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(TypeError, "MakeFxTracer or DynamoTracer"):
                torch.compiler.precompile.capture(
                    lambda x, y: x + y,
                    artifact_path=os.path.join(d, "m.py"),
                    cache_path=os.path.join(d, "m.cache"),
                    tracer="nope",
                )

    def test_backend_default_is_inductor(self):
        # The default lowers through Inductor: the generated code inlines the Inductor
        # output module. Use a graph_partition-agnostic marker (the ``call = runner.call``
        # form is only emitted when config.graph_partition is on, which is off in fbcode).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, _ = cap.result()
        self.assertIn("Inductor output code", code)

    def test_inductor_graph_partition_off(self):
        # graph_partition defaults off in fbcode; the Inductor output module then exposes
        # a top-level ``def call(args):`` instead of ``call = runner.call``. The source
        # extractor must still find it (regression: it previously matched only the
        # runner.call form, so torch.compiler.precompile crashed in fbcode).
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(graph_partition=False):
            with _CaptureToFiles(lambda model, xx: model(xx)) as cap:
                cap(m, x)
            code, cache = cap.result()
            self.assertNotIn("call = runner.call", code)  # non-partition form
            f_c = _load_pair(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_inductor_caches_disabled(self):
        # Source is captured off codegen (GraphLowering.save_output_code), not the cache
        # bundle, so precompile must work even when caching is disabled -- producing a
        # runnable python_code with an empty cache, not a misleading "non-cacheable HOP"
        # error. Covers force_disable_caches and fx_graph_cache=False.
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        for patch in (
            {"force_disable_caches": True},
            {"fx_graph_cache": False},
        ):
            with ind_config.patch(**patch):
                with _CaptureToFiles(lambda model, xx: model(xx)) as cap:
                    cap(m, x)
                code, cache = cap.result()
                # No saveable artifact when caches are off; the cache is empty.
                blob = torch.load(io.BytesIO(cache), weights_only=True)
                self.assertIsNone(blob["artifact"], patch)
                # python_code still runs standalone (JITs from inlined source).
                ns = {"__name__": "_a"}
                exec(compile(code, "<a>", "exec"), ns)
                self.assertEqual(ns["forward"](m, x), m(x), patch)
                # ...and load() falls back to the inlined path.
                self.assertEqual(_load_pair(code, cache)(m, x), m(x), patch)

    def test_inductor_cpp_wrapper_pinned_off(self):
        # cpp_wrapper would make Inductor emit a C++ ``call`` (no python module); a
        # python artifact cannot come from it, so compile_to_python pins it off. With
        # cpp_wrapper=True ambient, precompile must still produce a working python artifact.
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(cpp_wrapper=True):
            with _CaptureToFiles(lambda model, xx: model(xx)) as cap:
                cap(m, x)
            code, cache = cap.result()
            f_c = _load_pair(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_example_grad_restored_when_fn_raises(self):
        # If fn runs a backward then raises during the make_fx trace, the example
        # model's .grad must be restored (the snapshot/restore is in a finally), not
        # left clobbered -- precompile does not mutate the example model's grads.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        for p in m.parameters():
            self.assertIsNone(p.grad)

        def boom(model, xx):
            model(xx).sum().backward()  # populates .grad on the lifted example params
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            with _CaptureToFiles(boom, training=True) as cap:
                cap(m, x)
        for n, p in m.named_parameters():
            self.assertIsNone(p.grad, f"{n}: example .grad must be restored on failure")

    def test_unbacked_capture_with_preexisting_grad(self):
        # Regression: in the mark_unbacked path the example params are fakeified BEFORE
        # the grad clear. A model with a pre-existing .grad (the warmup-step-then-
        # precompile flow) plus a backward in fn must still capture -- the clear must
        # precede fakeify so the fakes inherit no grad. The trace itself leaves the real
        # .grad untouched; capture then runs the step once for real (caller-driven
        # contract), accumulating a second identical backward onto the warmup grad.
        from torch._dynamo.decorators import mark_unbacked

        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(8, 4)
        m(x).sum().backward()  # warmup: populate .grad before precompile
        saved = {n: p.grad.clone() for n, p in m.named_parameters()}
        mark_unbacked(x, 0)
        with _CaptureToFiles(
            lambda mm, t: mm(t).sum().backward(), training=True
        ) as cap:
            cap(m, x)
        code, _ = cap.result()
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 is dynamic
        # warmup + capture's identical backward
        for n, p in m.named_parameters():
            self.assertEqual(p.grad, saved[n] * 2)

    def test_backend_eager_no_inductor_lowering(self):
        # backend="eager" skips Inductor: the generated code has no inductor ``call``
        # entry point, and instead embeds the readable captured ATen graph and the
        # eager driver. The eager backend has no kernels to accelerate, so the cache
        # is empty -- python_code is the whole artifact.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x), backend="eager") as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn('backend="eager"', code)
        self.assertNotIn("call = runner.call", code)
        self.assertIn("torch.ops.aten", code)  # readable captured graph

        # The cache holds no artifact (eager caches nothing); the backend tag lives in
        # python_code (the single source of truth). The envelope still carries the
        # integrity tag, with backend='eager' to match python_code.
        self.assertIn("BACKEND = 'eager'", code)
        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        self.assertEqual(
            set(blob),
            {"artifact", "format", "version", "backend", "tracer", "code_hash"},
        )
        self.assertIsNone(blob["artifact"])  # eager has no compiled blob to bundle
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "eager")

    def test_backend_eager_self_contained_exec(self):
        # The eager python_code execs standalone with NO cache (the captured graph
        # is inlined) and runs, matching eager.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, x: model(x), backend="eager") as cap:
            cap(m, x)
        code, _cache = cap.result()

        ns = {"__name__": "_eager"}
        exec(compile(code, "<eager>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    def test_preexisting_param_grad_capture_succeeds(self):
        # Precompiling a backward fn on a model whose params already carry a .grad (the
        # common warmup-step-then-precompile flow) must capture cleanly: the pre-existing
        # grad must be cleared before tracing, not baked as a constant (invariant 1).
        # Eager simply accumulates a second backward, so precompile must too.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        m(x).sum().backward()  # warmup: params now carry a .grad
        self.assertIsNotNone(m.weight.grad)
        grad_before = m.weight.grad.clone()

        with _CaptureToFiles(
            lambda model, xx: model(xx).sum().backward(), training=True
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        # Capture runs the step once for real (caller-driven contract), so it
        # accumulates a second identical backward onto the warmup grad -- exactly what a
        # second eager model(x).sum().backward() would do (same weights, same x).
        self.assertEqual(m.weight.grad, grad_before * 2)

        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        _load_pair(code, cache)(run, x)  # run.grad starts None
        ref = torch.nn.Linear(4, 3)
        ref.load_state_dict(m.state_dict())
        ref(x).sum().backward()
        for (n, p), (_, rp) in zip(run.named_parameters(), ref.named_parameters()):
            self.assertEqual(p.grad, rp.grad, n)

    def test_nontensor_output_inductor_clean_error(self):
        # A non-tensor python value (float, complex, str, ...) in fn's output trips the
        # inductor backend's codegen assert; surface a clear PrecompileError (not a raw
        # InductorError) pointing to backend="eager". int / None outputs lower fine, and
        # eager handles the non-tensor value.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        for bad in (3.14, 2 + 3j, "hi"):
            with self.assertRaisesRegex(PrecompileError, "non-tensor Python value"):
                with _CaptureToFiles(lambda model, t, b=bad: (model(t), b)) as cap:
                    cap(m, x)
        for extra in (7, None):
            with _CaptureToFiles(lambda model, t, e=extra: (model(t), e)) as cap:
                cap(m, x)
            code, cache = cap.result()
            self.assertEqual(_load_pair(code, cache)(m, x)[1], extra)
        with _CaptureToFiles(lambda model, t: (model(t), 3.14), backend="eager") as cap:
            cap(m, x)
        ecode, ecache = cap.result()
        self.assertEqual(_load_pair(ecode, ecache)(m, x)[1], 3.14)

    def test_input_layout_mismatch_inductor_clean_error(self):
        # The inductor backend bakes each input's stride / memory format (invariant 6);
        # a same-shape input with a different layout must raise a clear PrecompileError
        # (not a raw assert_size_stride AssertionError) on BOTH the cached and inlined
        # paths. The eager backend is layout-flexible and accepts it.
        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(
            8, 6
        ).t()  # example: shape (6, 8), non-contiguous stride (1, 6)
        self.assertFalse(xex.is_contiguous())
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, xex)
        code, cache = cap.result()
        self.assertIn("assert_size_stride", code)  # the layout guard we convert
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            _load_pair(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            _load_pair(code, _strip_artifact(cache))(m, xrt)  # inlined path
        # A matching (same-stride) input still works on inductor.
        xmatch = torch.randn(8, 6).t()
        self.assertEqual(_load_pair(code, cache)(m, xmatch), m(xmatch))
        # The eager backend accepts the differently-strided input.
        with _CaptureToFiles(lambda model, t: model(t), backend="eager") as cap:
            cap(m, xex)
        ecode, ecache = cap.result()
        self.assertEqual(_load_pair(ecode, ecache)(m, xrt), m(xrt))

    def test_input_layout_mismatch_enforced_without_size_asserts(self):
        # The layout guard must be a PROACTIVE driver check, not a reliance on inductor's
        # assert_size_stride: with size_asserts=False the assert is elided, so a naive
        # try/except would silently read wrong strides. Both load paths must still raise.
        import torch._inductor.config as ind_config

        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(8, 6).t()  # non-contiguous example, shape (6, 8)
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with ind_config.patch(size_asserts=False):
            with _CaptureToFiles(lambda model, t: model(t)) as cap:
                cap(m, xex)
            code, cache = cap.result()
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                _load_pair(code, cache)(m, xrt)  # cached path
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                _load_pair(code, _strip_artifact(cache))(m, xrt)  # inlined

    def test_input_shape_mismatch_clean_error(self):
        # A same-structure but wrong-SHAPE input is an invariant-3 (shape) mismatch, NOT
        # an invariant-6 layout one: the driver must say "shape" / invariant 3 and not
        # misadvise a no-op .contiguous() (both inputs here are already contiguous).
        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(6, 8)  # contiguous example
        xrt = torch.randn(7, 8)  # contiguous, different shape (same pytree structure)
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, xex)
        code, cache = cap.result()
        with self.assertRaisesRegex(PrecompileError, "shape"):
            _load_pair(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "shape"):
            _load_pair(code, _strip_artifact(cache))(m, xrt)  # inlined path
        # The error must NOT mislabel a pure shape mismatch as a memory-format one.
        try:
            _load_pair(code, cache)(m, xrt)
        except PrecompileError as e:
            self.assertNotIn("memory format", str(e))

    def test_size1_dim_stride_exempt_like_inductor(self):
        # A size-1 dim's stride is irrelevant (one element); inductor's assert_size_stride
        # ignores it (guards.cpp), so the proactive layout check must too -- a kept-dim
        # slice x[i:i+1] (size-1 dim with a wider stride) must RUN, not raise.
        m = torch.nn.Linear(4, 3).eval()
        xex = torch.randn(1, 4)  # contiguous, stride (4, 1)
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, xex)
        code, cache = cap.result()
        row = torch.randn(2, 8)[
            0:1, :4
        ]  # shape (1, 4), stride (8, 1): size-1 dim differs
        self.assertEqual(tuple(row.shape), (1, 4))
        self.assertNotEqual(row.stride(), xex.stride())
        self.assertEqual(_load_pair(code, cache)(m, row), m(row))
        self.assertEqual(
            _load_pair(code, _strip_artifact(cache))(m, row),
            m(row),
        )

    def test_empty_input_shape_is_still_checked(self):
        # The numel==0 exemption must relax ONLY the (meaningless) stride check, not the
        # shape check: an empty runtime input whose shape differs from the example must
        # still raise invariant 3, not silently return the traced-shape output.
        with _CaptureToFiles(lambda t: t.sum(0)) as cap:
            cap(torch.randn(0, 4))
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(torch.randn(0, 6))
        # A matching empty input runs (shape matches; stride is not checked).
        self.assertEqual(f_c(torch.randn(0, 4)), torch.randn(0, 4).sum(0))

    def test_shape_only_input_is_layout_flexible(self):
        # An input used only for its .shape (not its data) is not stride-consumed by the
        # kernel, so inductor emits no assert_size_stride for it; a transposed version
        # (same shape) must RUN, not be wrongly rejected as a memory-format mismatch.
        class M(torch.nn.Module):
            def forward(self, x, y):
                return y * x.shape[0]

        m = M().eval()
        x = torch.randn(4, 4)  # square so .t() keeps shape (4, 4)
        y = torch.randn(4, 4)
        with _CaptureToFiles(lambda mm, a, b: mm(a, b)) as cap:
            cap(m, x, y)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        xt = x.t()  # same shape, different stride; only x.shape is consumed
        self.assertNotEqual(xt.stride(), x.stride())
        self.assertEqual(f_c(m, xt, y), m(xt, y))
        # A different x SHAPE is still rejected (x.shape[0] is baked).
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(5, 4), y)

    def test_dynamic_shapes_static_dim_still_checked(self):
        # The non-marked (feature) dim stays specialized: a mismatch on it is rejected,
        # while the marked (batch) dim is free.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, torch.randn(16, 4)).shape, (16, 3))  # dynamic dim free
        with self.assertRaisesRegex(PrecompileError, "dynamic dim"):
            f_c(m, torch.randn(16, 5))  # static feature dim mismatched

    def test_dynamic_shapes_guard_required_rejected(self):
        # A graph that must guard on the dynamic dim fails LOUDLY at capture (the unbacked
        # dim cannot be guarded), as a clear PrecompileError rather than a silent artifact.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)

        def needs_guard(mm, t):
            if t.shape[0] > 4:
                return mm(t)
            return mm(t) + 1

        with self.assertRaisesRegex(PrecompileError, "guard on a dim marked with"):
            with _CaptureToFiles(needs_guard) as cap:
                cap(m, x)

    def test_dynamic_shapes_eager_rejected(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(
            NotImplementedError, "only supported with backend='inductor'"
        ):
            with _CaptureToFiles(lambda mm, t: mm(t), backend="eager") as cap:
                cap(m, x)

    @parametrize("path", ("cached", "inlined"))
    def test_dtype_mismatch_rejected(self, path):
        # Each dense input's dtype is baked at capture (invariant 6); a runtime input of
        # a different dtype is rejected up front on BOTH the cached and inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # float32 example
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    @parametrize("path", ("cached", "inlined"))
    def test_device_mismatch_rejected(self, path):
        # Each dense input's device is baked at capture (invariant 6); a cpu-traced
        # artifact rejects a cuda input up front on BOTH load paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "device"):
            f_c(m, x.cuda())

    def test_mark_dynamic_backed_rejected(self):
        # Backed dynamic marks (mark_dynamic) have no analogue in the static/unbacked
        # capture path; precompile rejects them loudly rather than silently dropping
        # them and baking a wrong artifact (invariant 3).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_dynamic(x, 0)
        with self.assertRaisesRegex(PrecompileError, "mark_dynamic"):
            with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
                cap(m, x)

    def test_mark_unbacked_hint_override_honored(self):
        # A mark_unbacked hint_override is a perf-only autotuning size hint (never a
        # guard), so precompile does NOT reject it; the single artifact is valid for any
        # runtime size and the hint is threaded onto the capture ShapeEnv's symbol.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, hint_override=16)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        x2 = torch.randn(32, 4)
        self.assertEqual(f_c(m, x2), m(x2))

    def test_mark_unbacked_specialize_on_rejected(self):
        # A mark_unbacked specialize_on list cannot be honored (precompile produces a
        # single artifact, not per-value specializations); it is rejected at capture.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, specialize_on=[lambda t: t.shape[0] == 8])
        with self.assertRaisesRegex(PrecompileError, "specialize_on"):
            with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
                cap(m, x)

    def test_mark_unbacked_subclass_rejected(self):
        # A mark_unbacked dim on a tensor subclass (DTensor) cannot be honored: the
        # dynamic capture refakes a marked leaf via torch.empty, which drops the subclass
        # and would trace on a plain dense tensor. mark_unbacked stamps its marks on the
        # OUTER DTensor too (the decorator's DTensor branch falls through), so precompile
        # sees the mark and must reject it LOUDLY rather than silently tracing a
        # subclass-stripped tensor (invariant 3).
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        saved_env = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            x = distribute_tensor(torch.randn(8, 4), mesh, [Replicate()])
            mark_unbacked(x, 0)
            with self.assertRaisesRegex(PrecompileError, "tensor subclass"):
                with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
                    cap(m, x)
        finally:
            dist.destroy_process_group()
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @parametrize("path", ("cached", "inlined"))
    def test_shape_id_mismatched_sizes_rejected(self, path):
        # Two inputs sharing a shape_id reuse ONE unbacked symbol, so their marked dims
        # are equal by construction. A runtime call passing MISMATCHED sizes for those
        # dims violates the baked equality and is rejected with a clear PrecompileError.
        # The cached path catches it via the reconstructed artifact's assert_size_stride;
        # the inlined (artifact-stripped) path catches it via the inlined driver's own
        # assert_size_stride relabel -- exercise both so the inlined driver copy is covered.
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="b")
        mark_unbacked(y, 0, shape_id="b")
        with _CaptureToFiles(lambda mm, a, b: mm(a) + b) as cap:
            cap(m, x, y)
        code, cache = cap.result()
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape or memory format"):
            f_c(m, torch.randn(8, 4), torch.randn(16, 4))

    @parametrize("path", ("cached", "inlined"))
    def test_shape_id_bounds_from_both_occurrences_enforced(self, path):
        # Bounds from BOTH occurrences of a shared shape_id are applied to the single
        # shared symbol at capture: a min on one input and a max on the other are each
        # threaded onto the same unbacked symbol (see _fakeify_with_unbacked) AND baked as
        # a runtime USER_INPUT_BOUNDS guard. mark_unbacked's docstring promises a runtime
        # min/max check; this asserts it actually fires. An OUT-OF-BOUNDS size (< 2 or
        # > 64) is rejected with a PrecompileError naming the bound, while in-bounds sizes
        # (including the boundaries 2 and 64) still run and match eager. Both load paths.
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="b", min=2)
        mark_unbacked(y, 0, shape_id="b", max=64)
        with _CaptureToFiles(lambda mm, a, b: mm(a) + b) as cap:
            cap(m, x, y)
        code, cache = cap.result()
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = _load_pair(code, cache)
        for bs in (2, 8, 64):  # min boundary, an interior size, max boundary
            xt = torch.randn(bs, 4)
            yt = torch.randn(bs, 4)
            self.assertEqual(f_c(m, xt, yt), m(xt) + yt)
        # Below the declared min on the first occurrence's dim is rejected.
        with self.assertRaisesRegex(PrecompileError, "min=2"):
            f_c(m, torch.randn(1, 4), torch.randn(1, 4))
        # Above the declared max (from the second occurrence) is rejected.
        with self.assertRaisesRegex(PrecompileError, "max=64"):
            f_c(m, torch.randn(65, 4), torch.randn(65, 4))

    @parametrize("path", ("cached", "inlined"))
    def test_mark_unbacked_min_enforced_at_runtime(self, path):
        # mark_unbacked(x, 0, min=4) promises (in its docstring) a runtime check that the
        # dim is >= min. The capture-time torch._check on the unbacked symint never becomes
        # a runtime guard, so precompile bakes USER_INPUT_BOUNDS and the driver enforces it:
        # running the artifact at batch 2 raises a PrecompileError naming the bound on BOTH
        # load paths, while batch 8 runs and matches eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, min=4)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn("USER_INPUT_BOUNDS = [{0: (4, None)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "size 2.*min=4"):
            f_c(m, torch.randn(2, 4))
        xt = torch.randn(8, 4)
        self.assertEqual(f_c(m, xt), m(xt))

    def test_eager_backend_wrong_static_shape_rejected(self):
        # The eager driver now checks USER_INPUT_SHAPES too: a wrong static shape is
        # rejected (invariant 3).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t), backend="eager") as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(7, 4))

    def test_eager_backend_dtype_mismatch_rejected(self):
        # The eager driver checks USER_INPUT_DTYPES too: a dtype mismatch is rejected
        # (invariant 6).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t), backend="eager") as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    def test_cache_integrity_tampered_backend_rejected(self):
        # The cache envelope's backend tag is an integrity check: a tampered backend
        # (here flipped to a value that does not match python_code's BACKEND) makes
        # load() raise a clear PrecompileError rather than reconstruct a foreign cache.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["backend"] = "eager"  # python_code says inductor
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "backend"):
            _load_pair(code, buf.getvalue())

    @parametrize("tag", ("format", "version"))
    def test_cache_format_version_mismatch_degrades(self, tag):
        # The cache is acceleration-only, so a FORMAT or VERSION mismatch (a foreign or
        # different-build envelope) is NOT fatal: load() DEGRADES to JIT'ing from
        # python_code rather than hard-failing. The reloaded callable must still run and
        # match eager, and load() must emit a degrade WARNING on the torch._precompile
        # logger. (A BACKEND or CODE_HASH mismatch still hard-fails -- see
        # test_cache_integrity_tampered_backend_rejected and
        # test_load_rejects_mismatched_code_cache_pair.)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        # Tamper either the format string or bump the version to a foreign value.
        blob[tag] = "not-a-precompile-cache" if tag == "format" else 999
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            f_c = _load_pair(code, buf.getvalue())  # must not raise
        self.assertTrue(
            any("different torch build" in line for line in cm.output),
            f"expected a format/version degrade warning, got: {cm.output}",
        )
        self.assertEqual(f_c(m, x), m(x))  # JIT fallback runs and is correct

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

    def test_nonliteral_calling_convention_metadata_rejected(self):
        with _CaptureToFiles(lambda x: x.sin(), backend="eager") as cap:
            cap(torch.randn(2))
        code, cache = cap.result()
        bad_code = code.replace("BACKEND = 'eager'", "BACKEND = object()", 1)
        with self.assertRaisesRegex(
            PrecompileError, "BACKEND.*calling-convention metadata"
        ):
            _load_pair(bad_code, cache)

    def test_precompile_module_identity(self):
        # torch.compiler.precompile is a submodule: re-importing it resolves to the
        # SAME module object, and its name is the stable public path.
        p = torch.compiler.precompile
        self.assertIs(importlib.import_module("torch.compiler.precompile"), p)
        self.assertIs(sys.modules["torch.compiler.precompile"], p)
        self.assertEqual(p.__name__, "torch.compiler.precompile")

    def test_standalone_runtime_artifact_execs_in_fresh_process(self):
        # A generated artifact that imports a standalone_runtime helper (here output-
        # aliasing, which emits ``from ...standalone_runtime import gen_alias_from_base``)
        # must EXEC in a FRESH process whose only prior import is ``torch`` -- a
        # regression for the runtime_wrappers <-> _dynamo circular import that a cold
        # exec used to hit. We write python_code to a temp file and exec it in a
        # subprocess that imports only torch, then runs forward().
        x = torch.randn(3, 4)
        with _CaptureToFiles(lambda a: a.t()) as cap:
            cap(x)
        code, _cache = cap.result()
        self.assertIn("standalone_runtime import gen_alias_from_base", code)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as artifact_file:
            artifact_file.write(code)
            artifact_path = artifact_file.name
        driver = textwrap.dedent(
            f"""
            import torch  # the ONLY pre-import; the artifact must self-bootstrap
            ns = {{"__name__": "_fresh_artifact"}}
            with open({artifact_path!r}) as fh:
                exec(compile(fh.read(), {artifact_path!r}, "exec"), ns)
            x = torch.randn(3, 4)
            out = ns["forward"](x)
            assert torch.equal(out, x.t()), "fresh-process artifact output mismatch"
            print("FRESH_OK")
            """
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", driver],
                capture_output=True,
                text=True,
                timeout=300,
            )
        finally:
            if os.path.exists(artifact_path):
                os.remove(artifact_path)
        self.assertEqual(
            proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        self.assertIn("FRESH_OK", proc.stdout)

    def test_load_rejects_mismatched_code_cache_pair(self):
        # The cache envelope's code_hash (sha256 of python_code) binds a cache to the
        # EXACT python_code it accelerates. Two artifacts from the SAME backend but
        # DIFFERENT fn produce different python_code (hence different code_hash), so
        # pairing one's code with the other's cache must fail loudly rather than
        # silently run the cache's compiled graph under foreign metadata (the core
        # silent-wrong-result guard). The MATCHED pair still runs and is correct.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda mm, t: mm(t) * 2) as cap:
            cap(m, x)
        codeA, cacheA = cap.result()
        with _CaptureToFiles(lambda mm, t: mm(t) + 100) as cap:
            cap(m, x)
        codeB, cacheB = cap.result()
        self.assertNotEqual(codeA, codeB)
        with self.assertRaisesRegex(PrecompileError, "code_hash|does not match"):
            _load_pair(codeA, cacheB)
        f_a = _load_pair(codeA, cacheA)
        self.assertEqual(f_a(m, x), m(x) * 2)

    def test_non_size_stride_assertion_propagates_unchanged(self):
        # The inductor driver's forward() wraps the inlined ``call`` in a try/except
        # AssertionError that relabels ONLY inductor's own assert_size_stride failure
        # (a layout/shape mismatch) as a "shape or memory format" PrecompileError. A
        # NON-size-stride AssertionError (e.g. a user torch._assert or an internal
        # invariant) must propagate with its ORIGINAL message, not be mislabeled. A
        # call() that raises a non-layout AssertionError is hard to trigger from a real
        # compiled artifact, so doctor a real artifact's call() to raise a custom
        # assertion and re-pair its code_hash, exercising the inlined relabel guard.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        head = code[: code.index("\ndef call(")]
        banner = code.rindex(
            "# " + "=" * 70, 0, code.index("# 2. Calling-convention metadata")
        )
        new_call = (
            '\n\ndef call(args):\n    assert False, "my custom user assertion"\n\n\n'
        )
        new_code = head + new_call + code[banner:]
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = None  # force the inlined path so the doctored call() runs
        import hashlib

        blob["code_hash"] = hashlib.sha256(new_code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        f = _load_pair(new_code, buf.getvalue())
        with self.assertRaisesRegex(AssertionError, "my custom user assertion"):
            f(m, x)
        # The original assertion must NOT be relabeled as a layout error.
        try:
            f(m, x)
        except AssertionError as e:
            self.assertNotIn("shape or memory format", str(e))

    @parametrize("backend", ("inductor", "eager"))
    def test_renamed_buffer_structural_mismatch_rejected(self, backend):
        # The BUFFER_NAMES half of the structural check (invariant 2): a runtime model
        # whose PARAM names match exactly but a BUFFER is renamed (same count and shape)
        # must be rejected, since the buffer name list is part of the baked structure.
        # The cached/inlined inductor driver and the eager driver each have their own
        # _check_structure, so cover both backends.
        class WithBuf(torch.nn.Module):
            def __init__(self, bufname):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer(bufname, torch.randn(3))
                self._bn = bufname

            def forward(self, x):
                return self.lin(x) + getattr(self, self._bn)

        m = WithBuf("buf").eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda mm, t: mm(t), backend=backend) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn("BUFFER_NAMES = ['buf']", code)
        renamed = WithBuf("buf2").eval()  # same params, buffer renamed (same shape)
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(renamed, x)

    def test_inplace_input_mutation_not_restored(self):
        # Capture EXECUTES fn once on the call's inputs (invariant 3), so an in-place
        # mutation fn performs on a user input happens at capture time and is NOT undone.
        # Pin this surprising contract so it stays covered: the tensor the caller passed
        # reflects the mutation afterward.
        scratch = torch.zeros(4)
        with _CaptureToFiles(lambda a: a.add_(1.0)) as cap:
            cap(scratch)
        self.assertEqual(scratch, torch.ones(4))

    @parametrize("path", ("cached", "inlined", "eager"))
    def test_wrong_dtype_rejected_across_all_paths(self, path):
        # The same wrong-dtype input is rejected on ALL load paths -- cached (artifact),
        # inlined (artifact stripped), and eager -- each with its own driver copy of the
        # dtype check (invariant 6). Loading the SAME inductor artifact via cached and
        # inlined, plus a separate eager artifact, keeps the three drivers in agreement.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        if path == "eager":
            with _CaptureToFiles(lambda mm, t: mm(t), backend="eager") as cap:
                cap(m, x)
            code, cache = cap.result()
        else:
            with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
                cap(m, x)
            code, cache = cap.result()
            if path == "inlined":
                cache = _strip_artifact(cache)
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    def test_eager_device_mismatch_rejected(self):
        # The eager driver bakes each input's device (invariant 6): a cpu-traced eager
        # artifact rejects a cuda input up front, like the inductor backend.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        with _CaptureToFiles(lambda mm, t: mm(t), backend="eager") as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "device"):
            f_c(m, x.cuda())

    def test_unserializable_in_spec_accepts_distinct_structures(self):
        # When IN_SPEC degrades to None (the input pytree spec was not serializable) the
        # structural in_spec check is SKIPPED -- a documented best-effort limit. Two
        # SAME-leaf-count, same-per-leaf-shape but STRUCTURALLY DISTINCT runtime inputs
        # are therefore both accepted without error (the only check left is leaf count /
        # per-leaf shape). Make that best-effort gap explicit.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        with _CaptureToFiles(lambda model, h: model(h.a + h.b)) as cap:
            cap(m, inp)
        code, cache = cap.result()
        self.assertIn("IN_SPEC = None", code)
        f_c = _load_pair(code, cache)
        t = torch.randn(5, 4)
        # The traced structure (the custom node) and a plain list of the same two leaves
        # have distinct pytree structures but the same flattened leaves/shapes; both run.
        out_node = f_c(m, _UnserializableCtxInput(t, t))
        out_list = f_c(m, [t, t])
        self.assertEqual(out_node, m(t + t))
        self.assertEqual(out_list, m(t + t))

    @parametrize("path", ("cached", "inlined"))
    def test_mark_unbacked_max_enforced_at_runtime(self, path):
        # The max-only mirror of test_mark_unbacked_min_enforced_at_runtime:
        # mark_unbacked(x, 0, max=16) records USER_INPUT_BOUNDS = [{0: (None, 16)}] and
        # the driver rejects an ABOVE-max runtime size on BOTH load paths (the capture-time
        # torch._check never becomes a runtime guard on an unbacked symint), while an
        # in-bounds size runs and matches eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, max=16)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn("USER_INPUT_BOUNDS = [{0: (None, 16)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "max"):
            f_c(m, torch.randn(32, 4))
        xt = torch.randn(8, 4)
        self.assertEqual(f_c(m, xt), m(xt))

    @unittest.skipUnless(TEST_CUDA, "functionalize_rng_ops seeds via CUDA rng state")
    def test_functionalized_rng_matches_eager_cpu(self):
        # Under functionalized RNG the dropout draw is seeded from the global generator,
        # so seeding torch.manual_seed identically before the artifact run and before eager
        # makes both draw the SAME dropout mask: the artifact output is numerically EQUAL
        # to eager (a stronger check than structure-only). This runs on CPU tensors, but
        # functionalize_rng_ops still seeds via CUDARngStateHelper.get_torch_state_as_tuple,
        # which raises unless CUDA is available, so the whole test is gated on TEST_CUDA
        # (mirroring test_functionalized_rng_supported). The CUDA functionalized path uses
        # different Philox offset bookkeeping than eager, so this numeric equivalence is
        # CPU-tensor-only (see test_functionalized_rng_supported for the device-generic
        # structural check).
        import torch._functorch.config as functorch_config

        x = torch.randn(64)
        with functorch_config.patch(functionalize_rng_ops=True):
            with _CaptureToFiles(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True)
            ) as cap:
                cap(x)
            code, cache = cap.result()
            f_c = _load_pair(code, cache)
            torch.manual_seed(0)
            out = f_c(x)
        torch.manual_seed(0)
        ref = torch.nn.functional.dropout(x, 0.5, training=True)
        self.assertTrue((out == 0).any())  # dropout zeroed some elements
        self.assertEqual(out, ref)  # same mask under the same seed

    @parametrize("backend", ("inductor", "eager"))
    def test_param_shape_mismatch_rejected(self, backend):
        # The headline silent-wrong-result fix: the structural check (invariant 2) now
        # compares each runtime param's SHAPE against the baked example, not just its
        # name/count. A runtime model with the SAME param names but a different param
        # SHAPE (here Linear(4, K) for the traced Linear(4, M), K != M) is rejected with a
        # PrecompileError naming the offending param -- on BOTH backends, and on the
        # inductor backend's cached AND inlined load paths. Before the fix the eager
        # backend (no assert_size_stride backstop) silently returned a wrong-shaped tensor.
        m = torch.nn.Linear(4, 3).eval()  # M = 3
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t), backend=backend) as cap:
            cap(m, x)
        code, cache = cap.result()
        bad = torch.nn.Linear(4, 7).eval()  # K = 7 != 3, same param names

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "weight.*shape"):
                    f_c(bad, x)

    @parametrize("backend", ("inductor", "eager"))
    def test_param_dtype_mismatch_rejected(self, backend):
        # The dtype half of the structural shape/dtype check (invariant 2): a runtime
        # model with the SAME param names and shapes but a different param DTYPE (a
        # .half() copy of the traced float32 model) is rejected with a PrecompileError
        # naming the param, on both backends, AND -- on the inductor backend -- on the
        # cached (artifact) AND inlined (artifact-stripped) load paths. The inlined
        # inductor driver has its own _check_structure dtype branch, so cover it the
        # same way test_param_shape_mismatch_rejected covers the shape branch.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t), backend=backend) as cap:
            cap(m, x)
        code, cache = cap.result()
        bad = torch.nn.Linear(4, 3).eval().half()  # same shape, different dtype

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "weight.*dtype"):
                    f_c(bad, x)

    @parametrize("backend", ("inductor", "eager"))
    def test_buffer_shape_dtype_mismatch_rejected(self, backend):
        # The BUFFER half of the structural SHAPE/DTYPE check (invariant 2): the
        # structural loop iterates PARAM_NAMES then BUFFER_NAMES, but only the param
        # branch was exercised elsewhere. A runtime model whose PARAMS match exactly but
        # whose registered BUFFER (same name, same count) has a different SHAPE or DTYPE
        # must be rejected naming that buffer. Cover both backends, and -- on inductor --
        # the cached AND inlined driver copies (each has its own _check_structure).
        class WithBuf(torch.nn.Module):
            def __init__(self, size, dtype):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                # A plain buffer the graph READS, so it is lifted to a graph input and
                # survives to the structural check (a buffer never read might be elided).
                self.register_buffer("b", torch.randn(size).to(dtype))

            def forward(self, x):
                return self.lin(x) + self.b.sum()

        m = WithBuf(3, torch.float32).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t), backend=backend) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn("BUFFER_NAMES = ['b']", code)
        # Same buffer name and count, but a different SHAPE / DTYPE.
        bad_shape = WithBuf(5, torch.float32).eval()
        bad_dtype = WithBuf(3, torch.float64).eval()

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, r"'b'.*shape"):
                    f_c(bad_shape, x)
                with self.assertRaisesRegex(PrecompileError, r"'b'.*dtype"):
                    f_c(bad_dtype, x)

    def test_param_layout_specialization_rejected_inductor(self):
        # MAJOR2 (invariant 2 inductor caveat / invariant 6): the inductor backend bakes
        # each param/buffer's LAYOUT (memory format) too, since it emits assert_size_stride
        # on every weight the graph reads. A runtime model whose weight has the SAME
        # shape+dtype but a DIFFERENT memory format (a non-contiguous view) is rejected,
        # with the broadened relabel that names a model PARAMETER/BUFFER layout. The eager
        # backend is layout-flexible and ACCEPTS the same non-contiguous weight.
        m = torch.nn.Linear(8, 5).eval()
        x = torch.randn(4, 8)
        with _CaptureToFiles(lambda model, t: model(t)) as cap:
            cap(m, x)
        code, cache = cap.result()

        def with_noncontig_weight():
            run = torch.nn.Linear(8, 5).eval()
            run.load_state_dict(m.state_dict())
            # A non-contiguous view of the same data: same shape+dtype, different layout.
            nc = run.weight.data.t().contiguous().t()
            self.assertFalse(nc.is_contiguous())
            self.assertEqual(tuple(nc.shape), tuple(m.weight.shape))
            run.weight = torch.nn.Parameter(nc)
            return run

        def loaders():
            yield "cached", _load_pair(code, cache)
            yield (
                "inlined",
                _load_pair(code, _strip_artifact(cache)),
            )

        for label, f_c in loaders():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, r"memory format.*PARAMETER/BUFFER.*layout"
                ):
                    f_c(with_noncontig_weight(), x)
        # The eager backend accepts the same non-contiguous weight (layout-flexible).
        with _CaptureToFiles(lambda model, t: model(t), backend="eager") as cap:
            cap(m, x)
        ecode, ecache = cap.result()
        run = with_noncontig_weight()
        self.assertEqual(_load_pair(ecode, ecache)(run, x), run(x))

    def test_unbacked_equality_shared_vs_independent_shape_id(self):
        # MAJOR1 (invariant 3 DANGER note): two mark_unbacked dims that the graph requires
        # to be EQUAL behave differently depending on shape_id. (a) A SHARED shape_id binds
        # them to ONE symbol, so they are equal by construction AND a runtime size mismatch
        # is LOUDLY rejected. (b) Two INDEPENDENTLY marked dims (no shared shape_id)
        # combined elementwise bake a SILENT equal-size assumption: unlike eager, a runtime
        # mismatch is NOT loudly rejected -- NOT because the constraint is unrecoverable, but
        # because precompile does not harvest it: the capture ShapeEnv DOES record the
        # equality as a deferred runtime assert (Eq(u0, u1)), yet only the decorator's
        # min/max feed USER_INPUT_BOUNDS, so the driver never enforces the relational assert.
        # The artifact runs and returns the FIRST input's shape. This documents the "give
        # equal-must-be-equal dims a shared shape_id" limitation (and would flip to a loud
        # failure if that harvesting gap is later closed) rather than asserting silent-wrong
        # is correct.
        m = torch.nn.Linear(4, 4).eval()
        # (a) shared shape_id -> equality enforced.
        xs = torch.randn(8, 4)
        ys = torch.randn(8, 4)
        mark_unbacked(xs, 0, shape_id="b")
        mark_unbacked(ys, 0, shape_id="b")
        with _CaptureToFiles(lambda mm, a, b: mm(a) + b) as cap:
            cap(m, xs, ys)
        code_s, cache_s = cap.result()
        f_s = _load_pair(code_s, cache_s)
        xt, yt = torch.randn(8, 4), torch.randn(8, 4)
        self.assertEqual(f_s(m, xt, yt), m(xt) + yt)  # matched sizes work
        with self.assertRaisesRegex(PrecompileError, "shape or memory format"):
            f_s(m, torch.randn(8, 4), torch.randn(16, 4))  # mismatch rejected
        # (b) independent marks -> the documented silent equal-size limitation. A matched
        # call works; a mismatched call does NOT raise and returns the first input's shape.
        xi = torch.randn(8, 4)
        yi = torch.randn(8, 4)
        mark_unbacked(xi, 0)
        mark_unbacked(yi, 0)
        with _CaptureToFiles(lambda mm, a, b: mm(a) + b) as cap:
            cap(m, xi, yi)
        code_i, cache_i = cap.result()
        f_i = _load_pair(code_i, cache_i)
        xm, ym = torch.randn(10, 4), torch.randn(10, 4)
        self.assertEqual(f_i(m, xm, ym), m(xm) + ym)  # matched sizes work
        out = f_i(m, torch.randn(10, 4), torch.randn(12, 4))  # mismatch NOT rejected
        self.assertEqual(tuple(out.shape), (10, 4))  # broadcasts to the first input

    def test_grad_identity_preserved_across_precompile(self):
        # Capture snapshots and restores the example model's .grad by the SAME object (no
        # clone), so a caller holding a prior p.grad reference -- or optimizer state keyed
        # on grad identity -- is not invalidated. Warm up a backward to populate .grad,
        # snapshot the object identity, precompile a backward step on the same model, and
        # assert p.grad is still the SAME object afterward.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        m(x).sum().backward()  # warmup populates .grad
        g = m.weight.grad
        self.assertIsNotNone(g)
        with _CaptureToFiles(
            lambda mm, t: mm(t).sum().backward(), training=True
        ) as cap:
            cap(m, x)
        self.assertIs(m.weight.grad, g)  # same object, not a clone

    def test_precompile_error_public_binding(self):
        # PrecompileError is a single public type reachable two ways
        # (torch.compiler.PrecompileError and torch.compiler.precompile.PrecompileError),
        # is a real exception type, is advertised in torch.compiler.__all__, and a raised
        # instance is catchable via the public torch.compiler.PrecompileError alias.
        self.assertIs(
            torch.compiler.PrecompileError, torch.compiler.precompile.PrecompileError
        )
        self.assertIsInstance(torch.compiler.PrecompileError, type)
        self.assertIn("PrecompileError", torch.compiler.__all__)
        # A real PrecompileError (here the invariant-1 constant-tensor guard) is catchable
        # via the public torch.compiler.PrecompileError alias.
        captured = torch.randn(3)
        with self.assertRaisesRegex(torch.compiler.PrecompileError, "hard-coded"):
            with _CaptureToFiles(lambda x: x + captured) as cap:
                cap(torch.randn(3))

    def test_single_trust_warning_on_inlined_load(self):
        # On the inlined load path (an eager artifact has an empty cache, so there is
        # nothing to prime and load() just EXECs python_code) the untrusted-input / EXEC
        # warning must fire EXACTLY ONCE -- only _make_inlined_forward warns. Asserting
        # "exactly once" guards against the EXEC warning being duplicated on this load.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with _CaptureToFiles(lambda model, t: model(t), backend="eager") as cap:
            cap(m, x)
        code, cache = cap.result()
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            _load_pair(code, cache)
        exec_warnings = [line for line in cm.output if "EXEC" in line]
        self.assertEqual(
            len(exec_warnings), 1, f"expected one EXEC warning, got: {cm.output}"
        )
        self.assertTrue(any("untrusted" in line.lower() for line in cm.output))

    def test_tied_weights_single_input_single_grad(self):
        # Invariants 1/2/5: a weight tied across two layers is interned by identity to a
        # SINGLE graph input (PARAM_NAMES lists the first name once) and accumulates ONE
        # grad -- the sum of both uses -- matching an eager backward, not one grad per name.
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(4, 4, bias=False)
                self.l2 = torch.nn.Linear(4, 4, bias=False)
                self.l2.weight = self.l1.weight  # tie: same tensor, two names

            def forward(self, x):
                return self.l2(self.l1(x))

        m = Tied()
        t = torch.randn(5, 4)
        with _CaptureToFiles(
            lambda model, t: model(t).sum().backward(), training=True
        ) as cap:
            cap(m, t)
        code, cache = cap.result()
        self.assertIn("PARAM_NAMES = ['l1.weight']", code)  # tie collapsed to one
        # Capture runs the step once for real (caller-driven contract); reset so the
        # deepcopy'd reference and the loaded artifact each see exactly one backward.
        m.zero_grad(set_to_none=True)

        ref = copy.deepcopy(m)  # deepcopy preserves the tie within the object graph
        ref(t).sum().backward()

        _load_pair(code, cache)(m, t)  # one call: tied grad
        self.assertEqual(m.l1.weight.grad, ref.l1.weight.grad)
        self.assertIs(m.l1.weight, m.l2.weight)  # still one tensor at runtime

    def test_multiple_module_args_all_lifted(self):
        # The multi=True naming branch: two DIFFERENT nn.Module args are BOTH lifted, their
        # positions recorded in MODULE_POSITIONS, and their params disambiguated as m0.* /
        # m1.* (per-module prefixes). Loaded artifact matches eager m2(m1(t)).
        torch.manual_seed(0)
        m1 = torch.nn.Linear(4, 4)
        m2 = torch.nn.Linear(4, 3)
        t = torch.randn(5, 4)
        with _CaptureToFiles(lambda a, b, t: b(a(t))) as cap:
            cap(m1, m2, t)
        code, cache = cap.result()
        self.assertIn("MODULE_POSITIONS = [0, 1]", code)
        self.assertIn("m0.weight", code)  # first module's params prefixed m0.*
        self.assertIn("m1.weight", code)  # second module's params prefixed m1.*
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m1, m2, t), m2(m1(t)))

    def test_frozen_param_keeps_none_grad(self):
        # Invariant 5 with a mix: only params that received a gradient are harvested
        # (recorded in GRAD_PARAM_INDICES), so a frozen (requires_grad=False) param keeps
        # .grad is None while a trainable param gets a grad matching an eager backward.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Linear(4, 4)
                self.trainable = torch.nn.Linear(4, 4)
                for p in self.frozen.parameters():
                    p.requires_grad_(False)

            def forward(self, x):
                return self.trainable(self.frozen(x))

        m = M()
        t = torch.randn(5, 4)
        with _CaptureToFiles(
            lambda model, t: model(t).sum().backward(), training=True
        ) as cap:
            cap(m, t)
        code, cache = cap.result()
        # Capture runs the step once for real (caller-driven contract); reset so the
        # deepcopy'd reference and the loaded artifact each see exactly one backward.
        m.zero_grad(set_to_none=True)

        ref = copy.deepcopy(m)
        ref(t).sum().backward()

        _load_pair(code, cache)(m, t)
        for p in m.frozen.parameters():
            self.assertIsNone(p.grad)  # frozen: never harvested
        for p in m.trainable.parameters():
            self.assertIsNotNone(p.grad)
        for (n, p), (_, rp) in zip(
            m.trainable.named_parameters(), ref.trainable.named_parameters()
        ):
            self.assertEqual(p.grad, rp.grad, n)

    def test_requires_grad_flip_is_noop(self):
        # Which params get a scattered grad is fixed at CAPTURE time from the example
        # model's requires_grad (invariant 5); flipping a runtime param's requires_grad
        # does NOT change what the artifact computes. Capture with params requiring grad,
        # set requires_grad=False on the runtime model, and assert the grad is STILL
        # scattered (and matches eager) -- locking the documented contract.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)  # params require grad at capture
        x = torch.randn(5, 4)
        with _CaptureToFiles(
            lambda mm, t: mm(t).sum().backward(), training=True
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        for p in run.parameters():
            p.requires_grad_(False)  # flip OFF at runtime -- must be a no-op
        _load_pair(code, cache)(run, x)
        self.assertIsNotNone(run.weight.grad)  # still scattered despite the flip
        ref = torch.nn.Linear(4, 3)
        ref.load_state_dict(m.state_dict())
        ref(x).sum().backward()
        self.assertEqual(run.weight.grad, ref.weight.grad)

    @parametrize("module_receiver", [True, False])
    @parametrize("shadow", [True, False])
    @parametrize("read", [True, False])
    def test_precompile_serves_a_bound_method_whose_receiver_was_pruned(
        self, module_receiver, shadow, read
    ):
        # The bound-method reducer EMITS a bound method, so re-serializing its
        # own output reaches it again with a receiver the first pass replaced
        # with the _Missing sentinel. Deciding the branch by reading that
        # receiver then raised "'_Missing' object has no attribute forward".
        # Only the public path re-serializes -- the guard policy rebuilds each
        # frame from the pickle capture already made -- so only it aborted.
        #
        # Serving, not merely capturing, is the assertion that matters: an
        # earlier attempt at this fix degraded the method itself to the
        # sentinel, which captured fine and then re-pinned the TYPE_MATCH on it
        # to _Missing, so the artifact could never match a live receiver.
        inner = _PrecompileReboundModule() if module_receiver else _PrecompileRebound()
        holder = _PrecompileForwardHolder(inner, shadow)
        entry = _precompile_rebound_entry if read else _precompile_rebound_unread_entry
        x = torch.randn(4)
        with torch.no_grad():
            with _CaptureToFiles(
                entry,
                tracer="dynamo",
                backend="eager",
                require_no_risky_drops=False,
                require_complete=False,
            ) as cap:
                cap(holder, x)
            code, cache = cap.result()
            expected = entry(holder, x)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(holder, x), expected)

    def test_precompile_pruned_receiver_identity_guard_is_risky_not_silent(self):
        # The CLOSURE_MATCH identity slot on the saved function is dropped, as
        # it is in every artifact -- but as RISKY, which the default refuses.
        # Both entry points must agree on that, or require_no_risky_drops means
        # something different depending on which one you called.
        from torch._dynamo.precompile_package import precompile_capture

        x = torch.randn(4)
        with (
            self.assertRaisesRegex(
                torch.compiler.precompile.PrecompileError,
                r"_original_forwards\[0\]",
            ),
            torch.no_grad(),
        ):
            with _CaptureToFiles(
                _precompile_rebound_entry,
                tracer="dynamo",
                backend="eager",
                require_complete=False,
            ) as cap:
                cap(_PrecompileForwardHolder(_PrecompileRebound(), True), x)
            cap.result()

        session = precompile_capture(_precompile_rebound_entry, backend="eager")
        with session as compiled:
            compiled(_PrecompileForwardHolder(_PrecompileRebound(), True), x)
        session.artifact(require_no_risky_drops=False, require_complete=False)
        risky = [name for _, name in session.summary().risky_dropped_guards]
        self.assertIn("pipeline._original_forwards[0].__func__", risky)

    @parametrize(
        "attr", ["plain", "across_break", "large_then_rebound", "self_referential"]
    )
    def test_guard_through_a_tensor_attribute_round_trips(self, attr):
        # A guard rooted at a Python attribute of a tensor (x._cpu_copy) is
        # REBUILT at load, so the attribute travels in the pickle's STATE slot:
        # after the tensor is memoized, so an attribute pointing back at its
        # own tensor terminates, and by value, so rebinding it between capture
        # and serve changes the answer and a large one does not land in the
        # artifact. Both serving modes, because the standalone one raised at
        # load() while the installed one deferred the same AttributeError into
        # the first served call.
        if attr == "self_referential":
            x = torch.randn(4)
            x.my_flag = x
            entry, args = _precompile_reads_flag, (x,)
        else:
            model = _PrecompileReadsAttr()
            x = torch.randn(8)
            big = attr == "large_then_rebound"
            x._cpu_copy = torch.zeros(2048, 8)[0] if big else torch.randn(8)
            entry = (
                _precompile_attr_entry
                if attr == "across_break"
                else _precompile_call_model
            )
            args = (model, x)
        with torch.no_grad():
            expected = entry(*args)
            with _CaptureToFiles(
                entry,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(*args)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(*args), expected)
            if attr == "large_then_rebound":
                self.assertLess(len(code), 200_000)
                for _ in range(2):
                    x._cpu_copy = torch.randn(8)
                    self.assertEqual(loaded(model, x), model(x))

    @parametrize(
        "marking",
        [
            "unbacked",
            "unbacked_bounds",
            "unbacked_shape_id",
            "static",
            "dynamic",
            "dynamic_bounds",
        ],
    )
    def test_marked_artifact_serves_the_tensor_it_captured(self, marking):
        # The dimension-marking guard reads its attributes off the example value
        # rather than through a source, so nothing registers them in the guard
        # tree and value pruning drops them -- leaving the rebuilt guard
        # comparing against nothing and the artifact refusing the very tensor it
        # was captured on.
        #
        # Serving the SAME marked tensor is the whole point. Every other marking
        # test in these suites marks one tensor and serves a fresh unmarked one,
        # which passes with the bug fully present.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        {
            "unbacked": lambda t: mark_unbacked(t, 0),
            "unbacked_bounds": lambda t: mark_unbacked(t, 0, min=4, max=16),
            "unbacked_shape_id": lambda t: mark_unbacked(t, 0, shape_id="b"),
            "static": lambda t: torch._dynamo.decorators.mark_static(t, 0),
            "dynamic": lambda t: mark_dynamic(t, 0),
            "dynamic_bounds": lambda t: mark_dynamic(t, 0, min=4, max=16),
        }[marking](x)
        with _CaptureToFiles(
            _precompile_call_model,
            training=True,
            tracer="dynamo",
            require_no_risky_drops=False,
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        loaded = _load_pair(code, cache)
        self.assertEqual(loaded(m, x), m(x))
        if marking == "dynamic_bounds":
            # A differently bounded tensor is a different capture: the rebuilt
            # _dynamo_dynamic_range guard only survives if _has_dynamo_dim_marking
            # was carried, so this refusal is what proves the gate round-tripped.
            y = torch.randn(8, 4)
            mark_dynamic(y, 0, min=5, max=17)
            with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                loaded(m, y)

    def test_attribute_shadowing_fake_tensor_state_is_refused(self):
        # A reconstructed tensor IS a FakeTensor, so an attribute of the same
        # name as one a FakeTensor keeps its own state in cannot be carried.
        # Refuse by name rather than fail somewhere inside the rebuild.
        for name, fn in _precompile_reads_shadowed.items():
            x = torch.randn(4)
            x.__dict__[name] = 3
            with self.assertRaisesRegex(PrecompileError, f"a guard reads '{name}'"):
                with _CaptureToFiles(
                    fn,
                    backend="eager",
                    dynamic=False,
                    tracer="dynamo",
                    require_no_risky_drops=False,
                ) as cap:
                    cap(x)

    def test_guard_on_a_module_global_tensor_round_trips(self):
        # A TENSOR_MATCH carries its own subject inside its create_fn partial.
        # When the source root round-trips by ALIAS -- a module global comes
        # back live -- that carried copy is a different object from the one the
        # source walk reaches, so id-keyed value pruning replaced a KEPT guard's
        # own subject with the _Missing sentinel and load died in
        # _dispatch_keys. Any model reading a module-global tensor hits it.
        global _LUT_MODULE
        _LUT_MODULE = self._module_with(
            "import torch\n\nLUT = torch.tensor([0.0, 0.5, 1.0])\n",
            "torch.test_precompile_lut",
        )
        x = torch.randn(4)
        with torch.no_grad():
            want = _precompile_reads_module_global(x)
            with _CaptureToFiles(
                _precompile_reads_module_global,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(x)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(x), want)
            # Keeping the guard is only worth anything if it still checks.
            _LUT_MODULE.LUT = _LUT_MODULE.LUT.double()
            with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                loaded(x)

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

    @parametrize("api", ["session", "public"])
    def test_guards_that_cannot_be_rebuilt_are_dropped_not_refused(self, api):
        # A guard that cannot be rebuilt is dropped and recorded, not grounds
        # for refusing the other frames. Whether it is worth keeping is a
        # separate question with an answer the caller already controls, so the
        # drop is RISKY: the default rail still refuses, and only a caller who
        # said it accepts unchecked slots gets an artifact.
        # Fault-injected, because the reachable causes are fixed.
        from torch._dynamo.exc import PackageError
        from torch._dynamo.guards import GuardsStatePickler
        from torch._dynamo.precompile_package import precompile_capture
        from torch._precompile import _parse_artifact_metadata

        model = _PrecompileReadsAttr()
        x = torch.randn(8)
        x._cpu_copy = torch.randn(8)
        # Stop carrying ONE attribute, leaving the rest intact, so this can tell
        # a dropped guard apart from a wrecked artifact. Bound before patching,
        # or the replacement calls itself.
        real_carry = GuardsStatePickler._carried_tensor_attributes

        def drop_only_cpu_copy(self, obj):
            carried = real_carry(self, obj)
            if carried:
                carried = {k: v for k, v in carried.items() if k != "_cpu_copy"}
            return carried or None

        drop = mock.patch.object(
            GuardsStatePickler, "_carried_tensor_attributes", drop_only_cpu_copy
        )
        if api == "session":
            with drop, torch.no_grad():
                session = precompile_capture(_precompile_call_model, backend="eager")
                with session as compiled:
                    compiled(model, x)
                with self.assertRaisesRegex(PackageError, "can affect dispatch"):
                    session.artifact(require_complete=False)
                summary = session.summary()
            self.assertTrue(
                any("_cpu_copy" in name for _, name in summary.risky_dropped_guards)
            )
            return
        drift = []
        with contextlib.ExitStack() as stack, drop, torch.no_grad():
            self._spy_on_guard_drift(stack, drift)
            with self.assertRaisesRegex(PrecompileError, "can affect dispatch"):
                with _CaptureToFiles(
                    _precompile_call_model,
                    backend="eager",
                    dynamic=False,
                    tracer="dynamo",
                ) as cap:
                    cap(model, x)
                cap.result()
            with self.assertLogs("torch._dynamo.precompile_package", "WARNING") as cm:
                with _CaptureToFiles(
                    _precompile_call_model,
                    backend="eager",
                    dynamic=False,
                    tracer="dynamo",
                    require_no_risky_drops=False,
                ) as cap:
                    cap(model, x)
                code, cache = cap.result()
        meta = _parse_artifact_metadata(code)
        self.assertIn("_cpu_copy", str(meta["RISKY_DROPPED_GUARDS"]))
        # Recorded is not enough: the pickle is what the serving machine
        # rebuilds from, so a dropped guard still in it fails there exactly
        # as it failed here.
        self.assertNotIn("_cpu_copy", _serialized_guard_names(code))
        # drop_failed_guards() removes the guard and its HASATTR companion
        # before any filter entry is built, so the filter's recording never
        # saw them; they are recorded where they leave the pickle, so the
        # RISKY section, DROPPED_GUARD_CODE and the drop warning agree on what
        # was removed. The companion's inverted HASATTR is not in the tree that
        # ships, so it is not drift either; and each slot warns once, not once
        # per Guard object per rebuild.
        self.assertIn(["HASATTR", "L['x']"], meta["RISKY_DROPPED_GUARDS"])
        checks = [c for _, _, c in meta["DROPPED_GUARD_CODE"] if "_cpu_copy" in c]
        self.assertTrue(checks, meta["DROPPED_GUARD_CODE"])
        self.assertEqual(drift, [])
        dropping = [m for m in cm.output if "dropping guard HASATTR on L['x']" in m]
        self.assertEqual(len(dropping), 1, cm.output)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        # Serving is the point: dropping the guard is only useful if what
        # is left still matches the call it was captured on.
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(model, x), _precompile_call_model(model, x))

    def test_guard_drift_is_reported(self):
        # The loud half of a lossy reconstruction is a guard that will not
        # rebuild at all. The quiet half is one that rebuilds into a DIFFERENT
        # check, which serializes, loads, and then matches the wrong branch.
        # Injected for real: the reconstructed scope the rebuild reads is
        # edited on the way in, so the EQUALS_MATCH on ``mode`` rebakes to a
        # constant no live build made, and that guard is what ships.
        import torch._dynamo.package as package

        real_load = package.load_guards_state

        def rebake(guards_state):
            loaded = real_load(guards_state)
            scope = loaded.output_graph.local_scope
            if scope.get("mode") == "a":
                scope["mode"] = "z"
            return loaded

        model = _PrecompileAccumModel()
        x = torch.randn(3, 8)
        drift = []
        with contextlib.ExitStack() as stack, torch.no_grad():
            self._spy_on_guard_drift(stack, drift)
            with _CaptureToFiles(
                _precompile_accum_forward,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(model, x, "a")
            self.assertEqual(drift, [])
            with mock.patch.object(package, "load_guards_state", rebake):
                with self.assertRaisesRegex(PrecompileError, "different check") as ctx:
                    with _CaptureToFiles(
                        _precompile_accum_forward,
                        backend="eager",
                        dynamic=False,
                        tracer="dynamo",
                    ) as cap:
                        cap(model, x, "a")
                    cap.result()
        self.assertTrue(
            any("L['mode'] == 'z'" in payload for _, payload in drift), drift
        )
        self.assertIn("L['mode'] == 'z'", str(ctx.exception))

    def test_guard_drift_into_another_variants_check_is_reported(self):
        # Union-of-all-variants would hide this: the drifted check IS a live
        # leaf, only from a DIFFERENT variant. Two variants are captured, and
        # the rebuild of variant "a" is edited to rebake its EQUALS_MATCH into
        # "b", the value the OTHER variant legitimately guards. Keyed per
        # variant that is still drift; unioned it would vanish.
        import torch._dynamo.package as package

        real_load = package.load_guards_state

        def rebake(guards_state):
            loaded = real_load(guards_state)
            scope = loaded.output_graph.local_scope
            if scope.get("mode") == "a":
                scope["mode"] = "b"
            return loaded

        model = _PrecompileAccumModel()
        x = torch.randn(3, 8)
        drift = []
        with contextlib.ExitStack() as stack, torch.no_grad():
            self._spy_on_guard_drift(stack, drift)
            with _CaptureToFiles(
                _precompile_accum_forward,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(model, x, "a")
                cap(model, x, "b")
            self.assertEqual(drift, [])
            with mock.patch.object(package, "load_guards_state", rebake):
                with self.assertRaisesRegex(PrecompileError, "different check") as ctx:
                    with _CaptureToFiles(
                        _precompile_accum_forward,
                        backend="eager",
                        dynamic=False,
                        tracer="dynamo",
                    ) as cap:
                        cap(model, x, "a")
                        cap(model, x, "b")
                    cap.result()
        self.assertTrue(
            any("L['mode'] == 'b'" in payload for _, payload in drift), drift
        )
        self.assertIn("L['mode'] == 'b'", str(ctx.exception))

    def test_precompile_records_what_each_dropped_slot_checked(self):
        # A slot is named by its type and SOURCE, and for some types that is
        # not enough to judge the drop. A dropped HASATTR on a source may be
        # the benign companion of a kept TENSOR_MATCH on the same source, or
        # the only thing pinning an optional attribute, and those want very
        # different reactions from whoever is auditing the artifact. The
        # rendered check names the attribute and tells them apart.
        from torch._precompile import _parse_artifact_metadata

        cfg = _PrecompileClassAttrCfg()
        x = torch.ones(3)
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_class_attr_branch,
                tracer="dynamo",
                backend="eager",
                dynamic=False,
                require_no_risky_drops=False,
            ) as cap:
                cap(cfg, x)
            code, cache = cap.result()
        meta = _parse_artifact_metadata(code)
        rendered = meta["DROPPED_GUARD_CODE"]
        self.assertTrue(rendered, "no dropped slot carried its check")
        for gtype, name, check in rendered:
            self.assertIsInstance(gtype, str)
            self.assertIsInstance(name, str)
            # The point of the field: something to read, not an empty slot.
            self.assertTrue(check)
        # And every slot named here is one of the slots reported as dropped.
        every_drop = {
            (t, n)
            for key in (
                "DROPPED_GUARDS",
                "RISKY_DROPPED_GUARDS",
                "POLICY_DROPPED_GUARDS",
            )
            for t, n in meta[key]
        }
        for gtype, name, _ in rendered:
            self.assertIn((gtype, name), every_drop)

    @parametrize("subject", ["object_attr", "tensor_attr"])
    def test_hasattr_guard_survives_serialization(self, subject):
        # hasattr is a branch, and the HASATTR guard is all that pins it. It
        # was lost two ways. A single-variant capture makes every slot look
        # invariant -- varying_guard_slots over one variant is empty by
        # construction -- so the policy dropped an object's HASATTR on fully
        # default gates, with RISKY_DROPPED_GUARDS = [] and no error. And a
        # tensor attribute whose ONLY guard is HASATTR (a getattr dynamo traces
        # without a value guard) was never registered with the guard tree, so
        # serialization pruned it and the rebuilt HASATTR recomputed val=False.
        # Either way the artifact answered a caller on the OTHER branch with
        # the captured one instead of refusing.
        if subject == "object_attr":
            entry, gates = _precompile_hasattr_branch, {}
            with_attr = _PrecompileHasattrCfg()
            with_attr.fast = True
            x = torch.ones(3)
            served, refused = (with_attr, x), (_PrecompileHasattrCfg(), x)
            self.assertNotEqual(entry(*served)[0].item(), entry(*refused)[0].item())
        else:
            entry, gates = _precompile_attr_probe, {"require_no_risky_drops": False}
            noted = torch.randn(4)
            noted.side_note = torch.ones(1)
            served, refused = (noted,), (torch.randn(4),)
        with torch.no_grad():
            # Deliberately every other gate at its default.
            with _CaptureToFiles(
                entry, tracer="dynamo", backend="eager", dynamic=False, **gates
            ) as cap:
                cap(*served)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(*served), entry(*served))
            # Refused, rather than served the captured branch's answer.
            with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                loaded(*refused)

    def test_precompile_keeps_a_guard_whose_derived_type_must_survive(self):
        # One Guard can emit several checks: a DICT_KEYS_MATCH emits the
        # SEQUENCE_LENGTH for the same dict, as a DERIVED type. The filter
        # removes whole Guards, so judging only the top-level type took the
        # length check down with its parent and a four-key dict was answered
        # with the two-key graph.
        two = {"a": torch.ones(3), "b": torch.ones(3)}
        four = {k: torch.ones(3) for k in ("a", "b", "c", "e")}
        x = torch.ones(3)
        self.assertNotEqual(
            _precompile_dict_len(two, x)[0].item(),
            _precompile_dict_len(four, x)[0].item(),
        )
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_dict_len, tracer="dynamo", backend="eager", dynamic=False
            ) as cap:
                cap(two, x)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(two, x), _precompile_dict_len(two, x))
            with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                loaded(four, x)

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

    def test_precompile_installed_entry_keeps_its_defaults(self):
        # An installed artifact rebuilds its entry from a code object, which
        # carries neither defaults nor closure values. Without them a defaulted
        # parameter is simply absent at the served call, so the guard written
        # against it has nothing to bind and every variant misses.
        from torch._precompile import _parse_artifact_metadata

        model = _PrecompileBreakingModule().eval()
        x = torch.randn(3, 8)
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_defaulted_entry,
                tracer="dynamo",
                backend="eager",
                dynamic=False,
                require_no_risky_drops=False,
            ) as cap:
                cap(model, x)
            code, cache = cap.result()
        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], "installed")
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            # Called WITHOUT scale, so the served frame only has it if the
            # artifact carried the default.
            self.assertEqual(loaded(model, x), _precompile_defaulted_entry(model, x))

    def test_precompile_user_global_wins_over_the_artifacts_own(self):
        # The rendered backend source is exec'd into the artifact module and
        # brings its own names with it. Binding the frames to that live dict let
        # those shadow a user global of the same name -- which their guards were
        # written against -- so every variant missed.
        x = torch.randn(4)
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_shadowed_global_entry,
                tracer="dynamo",
                backend="eager",
                require_no_risky_drops=False,
            ) as cap:
                cap(x)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(x), _precompile_shadowed_global_entry(x))

    def test_precompile_serving_does_not_mutate_the_artifact(self):
        # CompilePackage appends to the entry's per-code records, so a
        # serve-time recompile used to write its new backend id back into the
        # artifact. The next install then resolved that id against the
        # artifact's backends, which never had it. The first install consumes a
        # prepared copy, so the corruption only surfaced on the third.
        model = _PrecompileBreakingModule().eval()
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_attr_entry,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(model, torch.randn(3, 8))
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        # A new shape each round misses every captured variant, which is what
        # makes the install recompile and mutate.
        for rows in (3, 4, 5, 6):
            with _maybe_scoped(loaded), torch.no_grad():
                x = torch.randn(rows, 8)
                self.assertEqual(loaded(model, x), _precompile_attr_entry(model, x))

    def test_installed_artifact_unload_takes_back_only_its_own_keys(self):
        # Backend keys are per-capture uuids, so only another handle on this
        # artifact can legitimately hold them; a foreign object filed under one
        # stands in for that here. unload() takes back only what this install
        # put there: neither a key already filed nor one re-filed while
        # installed. And two loads of one artifact share every key, so the
        # second install must not take over the first one's entries, with
        # nothing of either left once both have unloaded.
        from torch._dynamo.precompile_context import EagerCacheArtifact

        with _CaptureToFiles(
            _precompile_unreachable_helper_caller,
            backend="eager",
            dynamic=False,
            tracer="dynamo",
        ) as cap:
            cap(torch.randn(4))
        code, cache = cap.result()
        x = torch.randn(4)
        expected = _precompile_unreachable_helper_caller(x)
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        keys = list(loaded._compiled._backend_keys)
        self.assertGreater(len(keys), 1)
        for key in keys:
            PrecompileContext.take_artifact(key)
        theirs = EagerCacheArtifact(key=keys[0], content="theirs")

        def only_theirs_remains():
            # Their object is what remains, not the install's own backend.
            kept = PrecompileContext.serialize_artifact_by_key(keys[0])
            self.assertEqual(kept.content, "theirs")
            for key in keys[1:]:
                self.assertIsNone(PrecompileContext.serialize_artifact_by_key(key))
            PrecompileContext.take_artifact(keys[0])

        PrecompileContext.record_artifact(theirs)
        with torch.no_grad(), loaded:
            self.assertEqual(loaded(x), expected)
            for key in keys:
                self.assertIsNotNone(PrecompileContext.serialize_artifact_by_key(key))
        only_theirs_remains()
        with torch.no_grad(), loaded:
            self.assertEqual(loaded(x), expected)
            PrecompileContext.record_artifact(theirs)
        only_theirs_remains()

        a = _load_pair(code, cache)
        b = _load_pair(code, cache)
        self.addCleanup(a.unload)
        self.addCleanup(b.unload)
        with torch.no_grad():
            a.__enter__()
            filed = [PrecompileContext.serialize_artifact_by_key(k) for k in keys]
            b.__enter__()
            self.assertEqual(b(x), expected)
            for key, artifact in zip(keys, filed):
                self.assertIs(
                    PrecompileContext.serialize_artifact_by_key(key), artifact
                )
            b.__exit__(None, None, None)
            for key, artifact in zip(keys, filed):
                self.assertIs(
                    PrecompileContext.serialize_artifact_by_key(key), artifact
                )
            self.assertEqual(a(x), expected)
            a.__exit__(None, None, None)
        for key in keys:
            self.assertIsNone(PrecompileContext.serialize_artifact_by_key(key))

    @parametrize("mode", ["standalone", "installed"])
    def test_artifact_refuses_a_foreign_torch_build(self, mode):
        # A dynamo artifact carries Dynamo internals in its opaque blobs, so it
        # is locked to the build that made it. The standalone driver emitted
        # TORCH_VERSION and read it with nothing; the installed driver unpickles
        # the same internals and did not check at all. Either way a cross-build
        # artifact failed wherever its blob happened to break first, and the
        # docs promise the version/build locks are catchable as PrecompileError.
        from torch._precompile import _parse_artifact_metadata

        model = _PrecompileBreakingModule().eval()
        entry, args = (
            (_precompile_single_graph, (torch.randn(4),))
            if mode == "standalone"
            else (_precompile_attr_entry, (model, torch.randn(3, 8)))
        )
        with torch.no_grad():
            with _CaptureToFiles(
                entry,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(*args)
            code, cache = cap.result()
        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], mode)
        torch._dynamo.reset()
        with (
            mock.patch.object(torch, "__version__", "0.0.0+notreal"),
            self.assertRaisesRegex(PrecompileError, "produced by torch"),
        ):
            _load_pair(code, cache)

    def test_standalone_artifact_refuses_drifted_inlined_source(self):
        # A standalone artifact builds no CompilePackage, so it never ran the
        # inlined-source check the installed mode gets for free -- and an
        # artifact whose inlined helper has since changed does not fail, it
        # answers with the OLD number.
        import importlib.util

        def import_from_path(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module

        with tempfile.TemporaryDirectory() as tmp_dir:
            original = os.path.join(tmp_dir, "orig.py")
            modified = os.path.join(tmp_dir, "modified.py")
            with open(original, "w") as f:
                f.write("def scaled(x):\n    return x * 3.0\n")
            with open(modified, "w") as f:
                f.write("def scaled(x):\n    return x * 5.0\n")
            global _DRIFT_MODULE
            self.addCleanup(sys.modules.pop, "torch.test_precompile_drift", None)
            self.addCleanup(globals().__setitem__, "_DRIFT_MODULE", None)
            _DRIFT_MODULE = import_from_path("torch.test_precompile_drift", original)
            entry = _precompile_drift_entry
            x = torch.ones(4)
            with torch.no_grad():
                with _CaptureToFiles(
                    entry,
                    backend="eager",
                    dynamic=False,
                    tracer="dynamo",
                    require_no_risky_drops=False,
                ) as cap:
                    cap(x)
                code, cache = cap.result()
            from torch._precompile import _parse_artifact_metadata

            self.assertEqual(
                _parse_artifact_metadata(code)["SERVING_MODE"], "standalone"
            )
            torch._dynamo.reset()
            loaded = _load_pair(code, cache)
            with _maybe_scoped(loaded), torch.no_grad():
                self.assertEqual(loaded(x), entry(x))

            _DRIFT_MODULE = import_from_path("torch.test_precompile_drift", modified)
            torch._dynamo.reset()
            with self.assertRaisesRegex(
                PrecompileError, "source code changes detected"
            ):
                _load_pair(code, cache)

            # A serving host WITHOUT the module (its code is baked into the
            # captured graphs) makes drift unverifiable, not detected: the
            # check must skip rather than surface a raw ModuleNotFoundError
            # on an artifact that serves identically -- the CAPTURED answer,
            # x * 3, not the modified module's.
            del sys.modules["torch.test_precompile_drift"]
            torch._dynamo.reset()
            loaded = _load_pair(code, cache)
            with _maybe_scoped(loaded), torch.no_grad():
                self.assertEqual(loaded(x), (x * 3.0).sum())

    @parametrize("tracer", ["make_fx", "dynamo"])
    def test_capture_accumulates_gradients_like_eager(self, tracer):
        # The caller makes the calls inside the block, so precompile has no
        # example backward of its own: a grad already present when capture starts
        # is neither cleared nor snapshotted. Both tracers run each cap() for
        # real, so a .backward() ACCUMULATES onto the model's .grad exactly as
        # eager does, and the same grad OBJECT stays in place (the make_fx driver
        # accumulates in place onto a pre-existing grad, so optimizer state keyed
        # on its identity survives).
        torch.manual_seed(0)
        model = _PrecompileTrainMod()
        xs = [torch.randn(n, 8) for n in (3, 5)]
        # make_fx captures a single call; dynamo takes as many as we make. Each
        # captured call runs for real either way, so its backward lands on .grad.
        capture_calls = xs[:1] if tracer == "make_fx" else xs
        extra = {} if tracer == "make_fx" else {"dynamic": False}
        with torch.enable_grad():
            _precompile_backward_step(model, xs[0])  # warmup populates .grad
            before = [(p.grad, p.grad.detach().clone()) for p in model.parameters()]
            # Deepcopy drops .grad, so replay the warmup on the reference and
            # then every captured call cap() executes for real, so the reference
            # is warmup plus the captured calls' contribution.
            reference = copy.deepcopy(model)
            _precompile_backward_step(reference, xs[0])
            for x in capture_calls:
                _precompile_backward_step(reference, x)
            with _CaptureToFiles(
                _precompile_backward_step,
                backend="eager",
                tracer=tracer,
                training=True,
                **extra,
            ) as cap:
                for x in capture_calls:
                    cap(model, x)
        for p, (grad_object, _warmup), ref in zip(
            model.parameters(), before, reference.parameters()
        ):
            self.assertIs(p.grad, grad_object)
            self.assertEqual(p.grad, ref.grad)

    def test_precompile_records_a_backend_for_a_short_circuited_noop_graph(self):
        # A graph that runs nothing and returns nothing never reaches the
        # backend: output_graph substitutes noop_graph_call rather than pay a
        # metadata pass and a joint trace for it. Ordinary torch.compile then
        # discards the frame, but capture sets allow_empty_graphs, so it stays
        # compiled, its id lands in backend_ids, and nothing was ever filed for
        # it -- which the harvest reported as a compiled graph that lost its
        # code, failing the whole capture. Only the inductor path: the eager
        # one turns every cached backend into an artifact already.
        #
        # is_noop_graph is forced rather than provoked, on a graph whose output
        # IS already empty and whose one call is dead, so short-circuiting it
        # changes nothing observable; what this pins is the packaging.
        import torch._dynamo.output_graph as output_graph

        fired = []

        def force_noop(gm):
            outs = list(gm.graph.find_nodes(op="output"))
            empty = bool(outs) and all(_pytree.tree_leaves(n.args) == [] for n in outs)
            fired.append(empty)
            return empty

        model = _PrecompileDeadResultModel()
        x = torch.randn(2, 4)
        with mock.patch.object(output_graph, "is_noop_graph", force_noop):
            with torch.no_grad():
                with _CaptureToFiles(
                    _precompile_dead_result,
                    tracer="dynamo",
                    backend="inductor",
                    dynamic=False,
                    require_complete=False,
                    require_no_risky_drops=False,
                ) as cap:
                    cap(model, x)
                code, cache = cap.result()
        self.assertTrue(any(fired), "the no-op short-circuit never fired")
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertIsNone(loaded(model, x))

    @parametrize("entry", [_precompile_calls_unkeyable, _precompile_mixed_keyability])
    def test_capture_records_a_graph_the_cache_will_not_key(self, entry):
        # AOTAutogradCache refuses to KEY a graph calling anything outside its
        # allowlist, and a refusal means it never saves -- so the bundled
        # artifact was never recorded and the capture ended with nothing to
        # serialize. Any sharded model hits this: threading a process group or
        # a stream into a graph goes through exactly such a call. Backend
        # recording is all-or-nothing, so a capture that keys most of its graphs
        # and not the rest is the worst case; capture pins
        # bypass_autograd_cache_key, so every graph is recorded either way.
        model, x = torch.nn.Linear(8, 8).eval(), torch.randn(4, 8)
        with torch.no_grad():
            want = entry(model, x)
            with _CaptureToFiles(
                entry,
                backend="inductor",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
                require_complete=False,
            ) as cap:
                cap(model, x)
            code, cache = cap.result()
        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(model, x), want)

    def test_installed_artifact_reports_what_it_compiles_at_serve(self):
        # An installed artifact answers a guard miss by COMPILING, not by
        # refusing -- a frame reachable only through the frame evaluator has no
        # other way to run. That is deliberate, but it was invisible: the
        # generated banner claimed the opposite, and isolate_recompiles gives
        # the artifact a private cache identity so TORCH_LOGS=recompiles prints
        # nothing. An artifact quietly serving less of itself looked exactly
        # like one that was serving. And the module class is the same for every
        # graph a model produces, so a capture that recompiled nine graphs said
        # "GraphModule" nine times: each warning has to be traceable to one
        # graph, including the continuations in a resume chain.
        from torch._precompile import _parse_artifact_metadata

        model = _PrecompileBreakingModule().eval()
        captured, uncovered = torch.randn(3, 8), torch.randn(5, 8)
        with torch.no_grad():
            with _CaptureToFiles(
                _precompile_attr_entry,
                backend="eager",
                dynamic=False,
                tracer="dynamo",
                require_no_risky_drops=False,
            ) as cap:
                cap(model, captured)
            code, cache = cap.result()
        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], "installed")
        # The banner has to describe the mode it was emitted for.
        self.assertNotIn("Nothing is installed onto your code objects", code)
        self.assertIn("compiled fresh", code)

        torch._dynamo.reset()
        loaded = _load_pair(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            loaded(model, captured)
            self.assertEqual(loaded.serve_time_compiles(), 0)
            with self.assertLogs("torch._dynamo.precompile_package", "WARNING") as cm:
                loaded(model, uncovered)
            self.assertGreater(loaded.serve_time_compiles(), 0)

        # The count is what a job reads after the scope, when the live inner is
        # already gone: it survives the exit rather than resetting to zero, and
        # a re-install only adds to the running total.
        served = loaded.serve_time_compiles()
        self.assertGreater(served, 0)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertGreaterEqual(loaded.serve_time_compiles(), served)

        warnings = [m for m in cm.output if "serving compiled a NEW graph" in m]
        self.assertTrue(warnings)
        for message in warnings:
            self.assertIn("compile id ", message)
            self.assertIn("backend id ", message)
            self.assertIn("first traced at ", message)
        # Distinct per graph, which is the whole point.
        self.assertEqual(len(set(warnings)), len(warnings))

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

    def test_precompile_artifact_takes_no_positional_examples(self):
        # artifact() takes fn as its only positional argument -- capture is
        # caller-driven, so the example is a cap(...) call inside the block, not
        # a positional to the constructor.
        x = torch.randn(3)
        with self.assertRaises(TypeError):
            _CaptureToFiles(lambda y: y + 1, x, backend="eager")


def _graph_devices_literal(code: str) -> str:
    """The GRAPH_DEVICES line the artifact records, for tests that assert on it."""
    for line in code.splitlines():
        if line.startswith("GRAPH_DEVICES"):
            return line
    raise AssertionError("artifact has no GRAPH_DEVICES line")


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
class TestPrecompileNumerics(TestCase):
    # Numeric-correctness tests run device-generically so the same coverage
    # exercises the CUDA lowering, not just CPU.

    def test_plain_function(self, device):
        def f(x, y):
            return (x @ y).sin(), x + y

        a = make_tensor((4, 4), device=device, dtype=torch.float32)
        b = make_tensor((4, 4), device=device, dtype=torch.float32)
        with _CaptureToFiles(f) as cap:
            cap(a, b)
        code, cache = cap.result()
        self.assertIsInstance(code, str)
        self.assertIsInstance(cache, bytes)

        f_c = _load_pair(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_module_params_and_buffers_are_lifted(self, device):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_multiple_module_args(self, device):
        # More than one nn.Module arg: each module's params are lifted with
        # m{i}.-prefixed names. Both modules are passed again at runtime.
        a = torch.nn.Linear(4, 4).to(device).eval()
        b = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((2, 4), device=device, dtype=torch.float32)
        ref = b(torch.relu(a(x)))

        with _CaptureToFiles(lambda ma, mb, x: mb(torch.relu(ma(x)))) as cap:
            cap(a, b, x)
        code, cache = cap.result()
        self.assertIn(
            "PARAM_NAMES = ['m0.weight', 'm0.bias', 'm1.weight', 'm1.bias']", code
        )

        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(a, b, x), ref)

    def test_inplace_on_intermediate_is_allowed(self, device):
        # In-place ops on intermediates (e.g. nn.ReLU(inplace=True)) are fine -- they
        # do not touch any input -- and must NOT be rejected as input mutation.
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(inplace=True))
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_training_backward_harvest_matches_eager(self, device):
        # A training step that calls loss.backward(): precompile scatters the
        # parameter grads onto the runtime model's .grad fields (mirroring eager
        # .backward()) and returns fn's own result (None here).
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        # Keep magnitudes small (make_tensor defaults to a wide range) so the SGD
        # loop below converges rather than diverges.
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        with _CaptureToFiles(train_step, training=True) as cap:
            cap(model, x, target)
        code, cache = cap.result()
        # Capture runs the step once for real (caller-driven contract), so the example
        # model already carries one backward; reset so what we measure below is the
        # loaded artifact's own scatter, matching one eager step.
        model.zero_grad(set_to_none=True)
        f_c = _load_pair(code, cache)

        # The model is passed at runtime (no weights baked); the artifact mutates
        # model.parameters().grad in place, returning fn's result (None).
        out = f_c(model, x, target)
        self.assertIsNone(out)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)

        # Grads accumulate like eager: a second call without zeroing doubles them.
        f_c(model, x, target)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg * 2)

        # A standard zero_grad / step loop reduces loss.
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            f_c(model, x, target)
            losses.append(loss_fn(model(x), target).item())
            opt.step()
        self.assertLess(losses[-1], losses[0])

    def test_frozen_params_grad_matches_eager(self, device):
        # Params that do not receive a gradient -- a frozen (requires_grad=False)
        # backbone, or a param that does not contribute to the loss -- must keep
        # .grad = None after the step, exactly like eager .backward(). precompile must
        # NOT zero-fill them (regression test for the old all-params zero-fill).
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        for p in model[0].parameters():
            p.requires_grad_(False)  # freeze the first linear
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        with _CaptureToFiles(train_step, training=True) as cap:
            cap(model, x, target)
        code, cache = cap.result()
        # Capture runs the step once for real (caller-driven contract); reset so the
        # check below sees only the loaded artifact's scatter, matching one eager step.
        model.zero_grad(set_to_none=True)
        f_c = _load_pair(code, cache)
        f_c(model, x, target)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            if rp.grad is None:
                self.assertIsNone(p.grad, f"{n}: expected no grad, matching eager")
            else:
                self.assertEqual(p.grad, rp.grad)

    def test_multiple_modules_backward_grad_scatter(self, device):
        # Two distinct module args + a backward: grads must scatter onto the correct
        # module's params via the cross-module GRAD_PARAM_INDICES mapping. One module
        # is partly frozen so the test also pins the index shift across modules.
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4).to(device)
        b = torch.nn.Linear(4, 3).to(device)
        a.bias.requires_grad_(False)  # a frozen param shifts later indices
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref_a, ref_b = copy.deepcopy(a), copy.deepcopy(b)
        loss_fn(ref_b(torch.relu(ref_a(x))), target).backward()

        def train_step(ma, mb, x, target):
            loss_fn(mb(torch.relu(ma(x))), target).backward()

        with _CaptureToFiles(train_step, training=True) as cap:
            cap(a, b, x, target)
        code, cache = cap.result()
        # Capture runs the step once for real (caller-driven contract); reset both
        # models so the check sees only the loaded artifact's scatter.
        a.zero_grad(set_to_none=True)
        b.zero_grad(set_to_none=True)
        f_c = _load_pair(code, cache)
        f_c(a, b, x, target)
        for (n, p), (_, rp) in zip(a.named_parameters(), ref_a.named_parameters()):
            if rp.grad is None:
                self.assertIsNone(p.grad, f"a.{n}: expected no grad")
            else:
                self.assertEqual(p.grad, rp.grad, f"a.{n}")
        for (n, p), (_, rp) in zip(b.named_parameters(), ref_b.named_parameters()):
            self.assertEqual(p.grad, rp.grad, f"b.{n}")

    def test_tied_weights_lifted_once(self, device):
        # A tied weight (same tensor under multiple names) must become a single
        # lifted input: otherwise it is double-counted (double optimizer step) and
        # gradients are split rather than accumulated.
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 4, bias=False)
                self.b = torch.nn.Linear(4, 4, bias=False)
                self.b.weight = self.a.weight  # tie

            def forward(self, x):
                return self.b(torch.relu(self.a(x)))

        torch.manual_seed(0)
        m = Tied().to(device)
        x = make_tensor((3, 4), device=device, dtype=torch.float32)

        with _CaptureToFiles(lambda model, x: model(x)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        # The tied weight is lifted once (single name), so it is one graph input.
        self.assertIn("PARAM_NAMES = ['a.weight']", code)

        # Training scatters a single grad onto the shared weight, matching eager's
        # accumulation into the tied parameter.
        ref = copy.deepcopy(m)
        ref(x).sum().backward()
        ref_grad = ref.a.weight.grad

        with _CaptureToFiles(
            lambda model, x: model(x).sum().backward(), training=True
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        # Capture runs the step once for real (caller-driven contract); reset so the
        # check below sees only the loaded artifact's single scatter onto the tie.
        m.zero_grad(set_to_none=True)
        f_c = _load_pair(code, cache)
        f_c(m, x)
        self.assertEqual(m.a.weight.grad, ref_grad)
        # The tie means a.weight and b.weight are the same object, so b sees it too.
        self.assertIs(m.a.weight.grad, m.b.weight.grad)

    def test_backend_eager_plain_function(self, device):
        # backend="eager" runs the captured graph as-is and matches eager.
        def f(x, y):
            return (x @ y).sin(), x + y

        a = make_tensor((4, 4), device=device, dtype=torch.float32)
        b = make_tensor((4, 4), device=device, dtype=torch.float32)
        with _CaptureToFiles(f, backend="eager") as cap:
            cap(a, b)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_backend_eager_module(self, device):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda model, x: model(x), backend="eager") as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_backend_eager_training_harvest(self, device):
        # The backward-harvest contract holds for the eager backend too.
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        with _CaptureToFiles(train_step, backend="eager", training=True) as cap:
            cap(model, x, target)
        code, cache = cap.result()
        # Capture runs the step once for real (caller-driven contract); reset so the
        # check below sees only the loaded artifact's own scatter.
        model.zero_grad(set_to_none=True)
        f_c = _load_pair(code, cache)
        out = f_c(model, x, target)
        self.assertIsNone(out)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)

    def test_backend_eager_batchnorm(self, device):
        # The captured graph bakes a ``device`` constant (BatchNorm's
        # num_batches_tracked path), one of fx's custom builtins. The eager
        # standalone source must inject the full custom-builtin set, else this
        # raises NameError: name 'device' is not defined.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            m.train()
            return m.to(device)

        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()

        with _CaptureToFiles(lambda m, xx: m(xx), backend="eager") as cap:
            cap(fresh(), x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        run = fresh()
        self.assertEqual(f_c(run, x), ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_backend_eager_inf_constant(self, device):
        # masked_fill to -inf bakes a bare ``inf`` token into gm.code (another fx
        # custom builtin); the eager standalone source must provide it.
        def f(x):
            return torch.relu(x).masked_fill(x < 0, float("-inf"))

        x = make_tensor((8,), device=device, dtype=torch.float32)
        with _CaptureToFiles(f, backend="eager") as cap:
            cap(x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(x), f(x))

    def test_batchnorm_train_with_backward(self, device):
        # Training a model containing BatchNorm exercises buffer mutation (running
        # stats) and grad harvest together; grads and running stats must match eager.
        # Inductor fuses the BN backward, so rely on assertEqual's tolerance.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 3)
            )
            m.train()
            return m.to(device)

        loss_fn = torch.nn.MSELoss()
        x = make_tensor((16, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor(
            (16, 3), device=device, dtype=torch.float32, low=-1, high=1
        )

        ref = fresh()
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]
        ref_rm = ref[1].running_mean.clone()

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        with _CaptureToFiles(train_step, training=True) as cap:
            cap(fresh(), x, target)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        run = fresh()
        f_c(run, x, target)
        for p, rg in zip(run.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_output_alias_supported(self, device):
        # An output that is a view of an input goes through AOTAutograd's output-
        # alias epilogue; precompile reproduces it.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda a: a.t()) as cap:
            cap(x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(x), x.t())

    def test_input_mutation_supported(self, device):
        # In-place input mutation is reflected on the passed tensor (and matches
        # eager), via AOTAutograd's mutation handling composed into the artifact.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda a: a.add_(1.0)) as cap:
            cap(scratch)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    @unittest.skipUnless(TEST_CUDA, "functionalize_rng_ops seeds via CUDA rng state")
    def test_functionalized_rng_supported(self, device):
        # Functionalized RNG (dropout) threads seed/offset; the AOT backend composes
        # the RNG wrapper in. The artifact runs and produces a valid dropout mask. Even
        # for a CPU tensor the wrapper seeds from CUDARngStateHelper.get_torch_state_as_tuple,
        # which raises unless CUDA is available, so the whole test is gated on TEST_CUDA
        # rather than on the tensor's device.
        import torch._functorch.config as functorch_config

        x = make_tensor((64,), device=device, dtype=torch.float32)
        with functorch_config.patch(functionalize_rng_ops=True):
            with _CaptureToFiles(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True)
            ) as cap:
                cap(x)
            code, cache = cap.result()
            f_c = _load_pair(code, cache)
            out = f_c(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue((out == 0).any())

    def test_batchnorm_train_buffer_mutation(self, device):
        # A stateful module (BatchNorm in training mode) mutates its running stats.
        # precompile reflects that onto the runtime model's buffers and matches eager
        # -- the mutation handling comes from AOTAutograd's codegen.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            m.train()
            return m.to(device)

        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda model, xx: model(xx)) as cap:
            cap(fresh(), x)
        code, cache = cap.result()

        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()
        ref_rv = ref[1].running_var.clone()
        ref_nbt = ref[1].num_batches_tracked.clone()

        f_c = _load_pair(code, cache)
        run = fresh()
        out = f_c(run, x)
        self.assertEqual(out, ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)
        self.assertEqual(run[1].running_var, ref_rv)
        self.assertEqual(run[1].num_batches_tracked, ref_nbt)

    def test_mutated_duplicate_input(self, device):
        # The same tensor passed twice with a mutation: make_fx resolves the aliasing
        # at trace time (the graph mutates one input and reuses the result), so the
        # artifact reproduces eager when run with the same aliasing. Storage-aliased
        # mutated inputs go through AOTAutograd's now-codegen'd synthetic-base wrapper.
        fn = lambda a, b: (a.mul_(2.0), a + b)[1]  # noqa: E731
        t = make_tensor((4,), device=device, dtype=torch.float32)
        # Clone references BEFORE precompile: capture runs fn once, mutating t.
        ref = t.clone()
        ref_out = fn(ref, ref)
        run = t.clone()

        with _CaptureToFiles(fn) as cap:
            cap(t, t)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        out = f_c(run, run)
        self.assertEqual(out, ref_out)

    def test_dynamic_shapes_runs_across_sizes(self, device):
        # An UNBACKED-dynamic batch dim (opted in via mark_unbacked on the input): one
        # artifact runs on many runtime batch sizes (cached AND inlined paths), matching
        # eager. Device-generic so the CUDA unbacked-symint lowering is exercised.
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )
        m.to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 dynamic
        f_c = _load_pair(code, cache)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = None
        buf = io.BytesIO()
        torch.save(blob, buf)
        f_i = _load_pair(code, buf.getvalue())
        for bs in (8, 16, 1):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt), m(xt))  # cached path
            self.assertEqual(f_i(m, xt), m(xt))  # inlined path

    def test_dynamic_shapes_training_across_sizes(self, device):
        # Training (backward) with a dynamic batch; harvested grads match eager across
        # sizes (loss is output.sum() so no cross-input dim-equality guard is needed).
        # Device-generic so the CUDA unbacked-symint backward lowering is exercised.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3).to(device)
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        with _CaptureToFiles(
            lambda model, t: model(t).sum().backward(), training=True
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        for bs in (8, 16, 5):
            run = torch.nn.Linear(4, 3).to(device)
            run.load_state_dict(m.state_dict())
            ref = torch.nn.Linear(4, 3).to(device)
            ref.load_state_dict(m.state_dict())
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            f_c(run, xt)
            ref(xt).sum().backward()
            self.assertEqual(run.weight.grad, ref.weight.grad)

    def test_dynamic_shapes_shared_shape_id(self, device):
        # Two inputs whose batch dims share a shape_id reuse ONE unbacked symbol, so a
        # cross-input matched-batch op (here an add) traces with no dim-equality guard and
        # runs across sizes. Device-generic so the CUDA lowering is exercised.
        m = torch.nn.Linear(4, 4).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        y = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0, shape_id="b")
        mark_unbacked(y, 0, shape_id="b")
        with _CaptureToFiles(lambda mm, a, b: mm(a) + b) as cap:
            cap(m, x, y)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        for bs in (8, 16, 3):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            yt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt, yt), m(xt) + yt)

    def test_mark_unbacked_strict_honored(self, device):
        # mark_unbacked(x, 0, strict=True) is HONORED: the dim is captured as an unbacked
        # symint, so USER_INPUT_SHAPES records None for it and the single artifact runs
        # across runtime sizes, matching eager (device-generic for CUDA coverage).
        m = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0, strict=True)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)
        f_c = _load_pair(code, cache)
        for bs in (8, 16, 2):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt), m(xt))

    def test_unbacked_zero_batch_runs(self, device):
        # bs=0 on an unbacked dynamic dim is a valid runtime size (the symbol is >= 0);
        # the artifact runs on an empty batch and matches eager.
        m = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        with _CaptureToFiles(lambda mm, t: mm(t)) as cap:
            cap(m, x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        xt = make_tensor((0, 4), device=device, dtype=torch.float32)
        self.assertEqual(f_c(m, xt), m(xt))

    def test_channels_last_marked_input_roundtrips(self, device):
        # A channels_last-marked dynamic input round-trips at the SAME layout for a
        # LAYOUT-PRESERVING (pointwise) op: _detect_memory_format records channels_last so
        # the refaked leaf preserves it, and the artifact accepts a channels_last runtime
        # input (matching eager). (conv output has a separate inductor layout limitation,
        # so this uses a pointwise op.)
        x = make_tensor((2, 3, 4, 4), device=device, dtype=torch.float32)
        x = x.to(memory_format=torch.channels_last)
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
        mark_unbacked(x, 0)
        with _CaptureToFiles(lambda t: torch.relu(t) * 2.0) as cap:
            cap(x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        xt = make_tensor((5, 3, 4, 4), device=device, dtype=torch.float32)
        xt = xt.to(memory_format=torch.channels_last)
        out = f_c(xt)
        self.assertEqual(out, torch.relu(xt) * 2.0)

    def test_marked_exotic_layout_rejected(self, device):
        # _detect_memory_format cannot preserve a layout that is neither contiguous nor
        # channels_last(_3d) through the refake, so a mark_unbacked input in such a layout
        # (here a transposed, non-contiguous 2D tensor) is rejected LOUDLY at capture rather
        # than silently forced contiguous (which would bake a wrong assert_size_stride).
        # Transpose makes a non-contiguous (8, 4) tensor in neither channels_last format.
        x = make_tensor((4, 8), device=device, dtype=torch.float32).t()
        self.assertFalse(x.is_contiguous())
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            with _CaptureToFiles(lambda t: t.contiguous() * 2.0) as cap:
                cap(x)

    def test_eager_backend_input_mutation(self, device):
        # The eager backend replays the raw ATen graph, so input mutation is reflected on
        # the passed tensor and matches eager, like the inductor backend.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda a: a.add_(1.0), backend="eager") as cap:
            cap(scratch)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    def test_eager_backend_output_alias(self, device):
        # The eager backend reproduces an output that aliases an input (a view), matching
        # eager, via the raw ATen replay.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        with _CaptureToFiles(lambda a: a.t(), backend="eager") as cap:
            cap(x)
        code, cache = cap.result()
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(x), x.t())

    def test_tracer_dynamo_roundtrip_device(self, device):
        # The dynamo tracer's inductor lowering emits device kernels (it lowers the
        # Dynamo-produced subgraph through the same AOTAutograd + Inductor codegen as
        # make_fx), and its eager backend inlines the subgraph as device-agnostic source.
        # Run a numeric roundtrip device-generically -- like the make_fx numeric tests -- so
        # the dynamo capture + subgraph lowering is exercised on CUDA, not only CPU. Both
        # backends must round-trip to eager.
        m = (
            torch.nn.Sequential(
                torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3)
            )
            .to(device)
            .eval()
        )
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        for backend in ("inductor", "eager"):
            with _CaptureToFiles(
                lambda model, xx: model(xx),
                training=True,
                tracer="dynamo",
                backend=backend,
            ) as cap:
                cap(m, x)
            code, cache = cap.result()
            f_c = _load_pair(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_tracer_dynamo_dynamic_shapes_device(self, device):
        # The dynamo tracer's mark_unbacked capture, run device-generically: the unbacked
        # symint reaches the inductor lowering (CUDA kernels included) and the eager
        # backend inlines the symbolic subgraph, so one artifact matches eager across
        # runtime batch sizes on either backend.
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )
        m.to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        for backend in ("inductor", "eager"):
            with _CaptureToFiles(
                lambda model, xx: model(xx),
                training=True,
                tracer="dynamo",
                backend=backend,
            ) as cap:
                cap(m, x)
            code, cache = cap.result()
            f_c = _load_pair(code, cache)
            for bs in (8, 16, 1):
                xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
                self.assertEqual(f_c(m, xt), m(xt))

    @parametrize("backend", ("eager", "inductor"))
    def test_artifact_reproduces_capture_time_autocast(self, device, backend):
        # make_fx only. That artifact checks NO guards, so ambient autocast in the
        # serving process must not reach it -- the driver pins the state the capture
        # recorded, keyed off the GRAPH's devices (GRAPH_DEVICES), not the runtime
        # tensors'. The dynamo tracer mirrors torch.compile instead: autocast is part
        # of the guarded global state, so a serving process whose autocast differs
        # from capture misses rather than being silently pinned.
        tracer = "make_fx"

        def fn(model, xx):
            return model(xx)

        device_type = torch.device(device).type
        model = torch.nn.Linear(8, 8).to(device).eval()
        x = make_tensor((4, 8), device=device, dtype=torch.float32)

        with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
            with _CaptureToFiles(fn, backend=backend, tracer=tracer) as cap:
                cap(model, x)
            hot_code, hot_cache = cap.result()
        with torch.no_grad():
            with _CaptureToFiles(fn, backend=backend, tracer=tracer) as cap:
                cap(model, x)
            cold_code, cold_cache = cap.result()

        for code, cache, captured_under_autocast in (
            (hot_code, hot_cache, True),
            (cold_code, cold_cache, False),
        ):
            loaded = _load_pair(code, cache)
            with torch.no_grad():
                plain = loaded(model, x)
            with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
                under = loaded(model, x)
            expected = torch.bfloat16 if captured_under_autocast else torch.float32
            self.assertEqual(plain.dtype, expected)
            # Serving under an autocast the capture did not see must change
            # nothing at all, not merely keep the dtype.
            self.assertEqual(under.dtype, expected)
            self.assertEqual(plain, under)

    def test_artifact_autocast_covers_a_device_no_input_lives_on(self, device):
        # GRAPH_DEVICES comes from the captured graph: a fn that moves to
        # another device mid-way dispatches somewhere no param or input lives,
        # which a scan of the runtime tensors cannot see.
        device_type = torch.device(device).type
        if device_type == "cpu":
            raise unittest.SkipTest("needs a second device")

        def fn(model, xx):
            y = model(xx)
            moved = y.to(device)
            return torch.mm(moved, moved.t())

        model = torch.nn.Linear(8, 8).eval()  # stays on cpu
        x = make_tensor((4, 8), device="cpu", dtype=torch.float32)
        with torch.no_grad():
            with _CaptureToFiles(fn, backend="eager") as cap:
                cap(model, x)
            code, cache = cap.result()
        devices = _graph_devices_literal(code)
        self.assertIn(f"'{device_type}'", devices)
        self.assertIn("'cpu'", devices)

        loaded = _load_pair(code, cache)
        with torch.no_grad():
            plain = loaded(model, x)
        with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
            under = loaded(model, x)
        self.assertEqual(plain.dtype, torch.float32)
        self.assertEqual(under.dtype, torch.float32)
        self.assertEqual(plain, under)

    def test_tracer_dynamo_eager_custom_builtins(self, device):
        # The dynamo eager emitter (_emit_dynamo_eager_subgraph) injects fx's full
        # _custom_builtins set (inf / nan / device / ...) into the standalone subgraph
        # source, its own copy of the loop the make_fx eager path uses. Every other dynamo
        # eager test bakes only tensors needing plain ``torch``, so this guards that loop:
        # dropping it would raise NameError at load. (a) masked_fill to -inf bakes a bare
        # ``inf`` token; (b) a train-mode BatchNorm bakes a ``device`` constant. Both must
        # round-trip to eager (mirrors test_backend_eager_inf_constant / _batchnorm).
        def inf_fn(model, xx):
            y = model(xx)
            return torch.relu(y).masked_fill(y < 0, float("-inf"))

        m = torch.nn.Linear(4, 4).to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        with _CaptureToFiles(
            inf_fn, training=True, tracer="dynamo", backend="eager"
        ) as cap:
            cap(m, x)
        code, cache = cap.result()
        self.assertEqual(_load_pair(code, cache)(m, x), inf_fn(m, x))

        def fresh_bn():
            torch.manual_seed(0)
            bn = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            bn.train()
            return bn.to(device)

        xb = make_tensor((8, 4), device=device, dtype=torch.float32)
        ref_out = fresh_bn()(xb)
        with _CaptureToFiles(
            lambda model, xx: model(xx), training=True, tracer="dynamo", backend="eager"
        ) as cap:
            cap(fresh_bn(), xb)
        code, cache = cap.result()
        self.assertEqual(_load_pair(code, cache)(fresh_bn(), xb), ref_out)


# The accelerator lowering needs triton; without it only the CPU variants run.
instantiate_device_type_tests(
    TestPrecompileNumerics, globals(), only_for=None if HAS_GPU else "cpu"
)


if __name__ == "__main__":
    run_tests()
