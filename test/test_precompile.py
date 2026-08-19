# Owner(s): ["oncall: pt2"]
import copy
import enum
import functools
import gc
import inspect
import io
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
import weakref
from unittest import mock

import torch
import torch.utils._pytree as _pytree
from torch._dynamo.decorators import mark_dynamic, mark_unbacked
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._precompile import PrecompileError
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


# A module-level (global) model + a function referencing it, to exercise the
# constant-tensor guard against a baked global.
_GLOBAL_TENSOR = torch.randn(3)

# Globals holding a tensor NESTED inside a container or an nn.Module, plus a plain scalar
# global, to exercise the dynamo tracer's baked-constant guard (which must look through
# containers/modules) and its uncovered-external-ref handling (a plain global folded into
# the output must be baked, not left dangling).
_GLOBAL_TENSOR_LIST = [torch.randn(3)]
_GLOBAL_TENSOR_DICT = {"w": torch.randn(3)}
_GLOBAL_SUBMODULE = torch.nn.Linear(4, 3).eval()
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


def _precompile_unreachable_helper_caller(x):
    return _precompile_unreachable_helper(x * 2)


def _precompile_closure_entry_factory():
    scale = 3.0

    def entry(x):
        return _precompile_unreachable_helper_caller(x) * scale

    return entry


def _precompile_capture(fn, **kwargs):
    """What the removed public ``precompile.capture`` did, for tests that need
    to drive a capture directly rather than through ``precompile()``."""
    from torch._precompile import _capture_session, PrecompileSession

    return PrecompileSession(_capture_session(fn, **kwargs))


# ---------------------------------------------------------------------------
# Fixtures for the dynamo-tracer break/recompile matrix. Module scope, because
# the capture serializes real Dynamo guards and cannot name a local class.
# ---------------------------------------------------------------------------


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

    return loaded if hasattr(loaded, "__enter__") else contextlib.nullcontext()


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


class _PrecompileTrainMod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Linear(8, 8)
        self.b = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.b(torch.relu(self.a(x))))


def _precompile_scaled(x, k):
    return x * k


def _precompile_branchy(x, flag):
    if flag:
        return (x * 2).sum()
    return (x + 1).sum()


def _precompile_call_model(model, x):
    return model(x)


def _precompile_multi_graph(x):
    x = x * 2
    torch._dynamo.graph_break()
    x = x + 3
    torch._dynamo.graph_break()
    return x.sum()


def _precompile_multi_graph_callable(x, op):
    x = x + 1
    torch._dynamo.graph_break()
    return op(x)


def _precompile_empty_resume(x, flag):
    y = x + 1
    torch._dynamo.graph_break()
    if flag:
        return y
    return y.cos() * 100


def _precompile_single_graph(x):
    return x.sin()


def _precompile_identity(x):
    return x


def _precompile_module_arg(module, x):
    return module(x)


def _precompile_raises_on_flag(x, fail):
    if fail:
        raise KeyError("automatic example failed")
    return x + 1


# A tensor held as a plain attribute of an arbitrary (non-pytree / non-Module) global
# object. The dynamo tracer's baked-constant guard must look THROUGH such an object's
# __dict__: a tensor reached via _GLOBAL_HOLDER.weight is embedded by value into the
# artifact, so it must be rejected (invariant 1), matching the make_fx tracer's get_attr
# scan which bakes-and-rejects the same access.
class _TensorHolder:
    def __init__(self, t):
        self.weight = t


_GLOBAL_HOLDER = _TensorHolder(torch.randn(3))


# A tensor held in a __slots__ slot (no __dict__), and one held as an UNREGISTERED plain
# attribute on an nn.Module (not a param/buffer). Both are baked by value when the owning
# object is a captured global, so the dynamo tracer's invariant-1 scan must reach them --
# through __slots__ and through the module's own __dict__ -- matching what make_fx rejects.
class _SlotHolder:
    __slots__ = ("weight",)

    def __init__(self, t):
        self.weight = t


class _UnregisteredAttrModule(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.foo = t


_GLOBAL_SLOT_HOLDER = _SlotHolder(torch.randn(3))
_GLOBAL_UNREGISTERED_MODULE = _UnregisteredAttrModule(torch.randn(3))


_GRAD_RAIL_MODULE = types.ModuleType("_precompile_grad_rail_mod")
_GRAD_RAIL_MODULE.__file__ = "_precompile_grad_rail_mod.py"
exec(
    compile(
        "import torch\nmodel = torch.nn.Linear(4, 3)\n",
        _GRAD_RAIL_MODULE.__file__,
        "exec",
    ),
    _GRAD_RAIL_MODULE.__dict__,
)
sys.modules["_precompile_grad_rail_mod"] = _GRAD_RAIL_MODULE


def _grad_rail_module_global_step(xx, tt):
    torch.nn.functional.mse_loss(_GRAD_RAIL_MODULE.model(xx), tt).backward()


class _StarvedInner:
    def __init__(self, model):
        self.model = model


class _Phase(enum.Enum):
    GEN = 1


# A functools.partial carrying a tensor: pickle bakes that tensor BY VALUE via the partial's
# __reduce__, even though it lives outside __dict__/__slots__, so the invariant-1 scan (which
# keys off what pickle serializes by value) must reject it. Contrast _global_helper_with_attr:
# a module-level function is pickled BY REFERENCE, so a tensor merely attached to it is NOT
# baked and must NOT be (falsely) rejected.
_GLOBAL_PARTIAL = functools.partial(torch.mul, torch.randn(3))


def _global_helper_with_attr():
    return None


_global_helper_with_attr.cache = torch.randn(3)


# A bound method carries its __self__ (and any tensor __self__ holds) into the pickle BY
# VALUE, so a bound method captured by fn (e.g. as a default callback) bakes that tensor and
# must be rejected (invariant 1) -- the scan replays pickle's traversal, which reaches the
# tensor through __self__.
class _MethodHolder:
    def __init__(self, t):
        self.t = t

    def add(self, z):
        return z + self.t


_GLOBAL_METHOD_HOLDER = _MethodHolder(torch.randn(3))


# A global container holding a tensor FOLLOWED by an unpicklable object (a module-level
# lambda pickle cannot serialize). _baked_tensors pickles the container, records the tensor
# via its persistent_id hook mid-traversal, then swallows the lambda's PicklingError and
# returns the tensor found BEFORE the failure -- so invariant 1's actionable "hard-coded"
# error fires at capture rather than the generic "not picklable" error at metadata build.
_GLOBAL_TENSOR_THEN_UNPICKLABLE = [torch.randn(3), lambda z: z]


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
    yield "default", torch.compiler.precompile.load(code, cache)
    if backend == "inductor":
        yield "inlined", torch.compiler.precompile.load(code, _strip_artifact(cache))


# precompile drives make_fx internally, which cannot symbolically trace a
# dynamo-optimized function; the whole suite is therefore incompatible with
# PYTORCH_TEST_WITH_DYNAMO (dynamo_wrapped CI), so skip it there.
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
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), m, x, decompositions=decomps
        )
        self.assertTrue(called)  # the table was used during capture

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_constant_tensor_is_rejected(self):
        captured = torch.randn(3)
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.compiler.precompile(lambda x: x + captured, torch.randn(3))

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
            torch.compiler.precompile(f, torch.randn(3))

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
            torch.compiler.precompile(lambda model, x: model(x), m, torch.randn(2, 4))

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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)

        self.assertIn("Inductor output code", code)
        self.assertIn("def forward(", code)
        self.assertIn("PARAM_NAMES = ['lin.weight', 'lin.bias']", code)

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_self_contained_exec_needs_no_cache(self):
        # python_code runs standalone with NO cache: exec it and call forward().
        # The default eager backend has no kernels; the captured graph is
        # interpreted directly from the inlined source and the cache is always
        # empty (artifact=None), so python_code is fully self-contained.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(lambda model, x: model(x), m, x)

        ns = {"__name__": "_artifact"}
        exec(compile(code, "<artifact>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    @unittest.skipUnless(
        torch.cuda.is_available(), "needs CUDA + Triton for the kernel cache"
    )
    @torch._inductor.config.patch({"compile_threads": 1})
    def test_cache_primes_inductor_on_reload(self):
        # The cache is a pure acceleration. load() feeds it to load_cache_artifacts to
        # PRIME the inductor kernel caches, then execs the self-contained python_code --
        # which loads the precompiled Triton kernels instead of recompiling. The composed
        # python_code runs its inlined kernels directly (no compile_fx re-entry, so no
        # FxGraphCache lookup); the observable acceleration is the Triton bundler
        # rehydrating the static autotuner on the cold reload. Mirrors
        # test/inductor/test_compile_to_python.py test_warm_load_rehydrates_static_launcher.
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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        self.assertIsInstance(cache, bytes)

        with fresh_cache():
            counters.clear()
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))
            self.assertGreater(
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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        with fresh_cache():
            f_c = torch.compiler.precompile.load(code, cache)
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

            code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
            # Subclass handling is via our own protocol-based driver, not embedded
            # AOTAutograd wrapper source.
            self.assertIn("__tensor_unflatten__", code)
            self.assertNotIn("subclass_wrapper", code)

            # load() takes the bundled-artifact path (real AOTAutograd runtime).
            f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)

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
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_inlined_fallback_when_artifact_absent(self):
        # When the cache holds no serialized artifact, load() falls back to
        # executing the inlined python (recompiling kernels). Force that branch by
        # stripping the artifact and check it still matches eager; this also
        # exercises the self-contained inlined path (JIT from inlined source).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        self.assertIsNotNone(blob["artifact"])

        f_c = torch.compiler.precompile.load(code, _strip_artifact(cache))
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
        _code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(code, cache)

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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(code, _strip_artifact(cache))

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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(
            lambda model, p: model(p.x + p.y), m, inp
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.x + inp.y))

    def test_unserializable_context_in_spec_still_compiles(self):
        # A registered pytree node whose context is not JSON-dumpable makes
        # treespec_dumps raise TypeError (not NotImplementedError); IN_SPEC must still
        # degrade to None rather than crashing precompile.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, h: model(h.a + h.b), m, inp
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
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
            torch.compiler.precompile(lambda x: Out(x + 1, x + 2), torch.randn(4))

    def test_input_leaf_count_mismatch_rejected_when_spec_unserializable(self):
        # When IN_SPEC degrades to None the structural in_spec check is skipped; a runtime
        # input flattening to a DIFFERENT leaf count must still raise a clean
        # PrecompileError (not a raw zip/unpack error) on the live and eager-inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        for backend in ("inductor", "eager"):
            code, cache = torch.compiler.precompile(
                lambda model, h: model(h.a + h.b), m, inp, backend=backend
            )
            self.assertIn("IN_SPEC = None", code)
            f = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
        f_i = torch.compiler.precompile.load(code, _strip_artifact(cache))
        code_e, cache_e = torch.compiler.precompile(
            lambda mm, t: mm(t), m, x, backend="eager"
        )
        f_e = torch.compiler.precompile.load(code_e, cache_e)
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
                torch.compiler.precompile(
                    lambda model, xx: NT(model(xx), model(xx) + 1),
                    m,
                    x,
                    backend=backend,
                )
        # A registered namedtuple output serializes and round-trips on both backends.
        # Registration mutates the process-global pytree registry, so deregister it on
        # cleanup rather than leaking the node into later tests.
        RNT = collections.namedtuple("RNT", ["p", "q"])
        _pytree._register_namedtuple(RNT, serialized_type_name="test_precompile.RNT")
        self.addCleanup(_pytree._deregister_pytree_node, RNT)
        ref = (m(x), m(x) + 1)
        for backend in ("inductor", "eager"):
            code, cache = torch.compiler.precompile(
                lambda model, xx: RNT(model(xx), model(xx) + 1), m, x, backend=backend
            )
            out = torch.compiler.precompile.load(code, cache)(m, x)
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

        code, cache = torch.compiler.precompile(step, a, b, c, x, target)

        def grads(ms):
            return [p.grad for m in ms for p in m.parameters()]

        # deepcopy the three together so the a/b weight tie is preserved.
        ca, cb, cc = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(code, cache)(
            ca, cb, cc, x, target
        )  # cached path

        ia, ib, ic = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(code, _strip_artifact(cache))(
            ia, ib, ic, x, target
        )  # inlined

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
        icode, icache = torch.compiler.precompile(step, a, b, c, x, target)
        ia, ib, ic = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(icode, icache)(
            ia, ib, ic, x, target
        )  # inductor cached path

        ecode, ecache = torch.compiler.precompile(
            step, a, b, c, x, target, backend="eager"
        )
        ea, eb, ec = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(ecode, ecache)(
            ea, eb, ec, x, target
        )  # eager path

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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda xx, model: model(xx), x, m)
        inlined_cache = _strip_artifact(cache)  # force the inlined path
        ecode, ecache = torch.compiler.precompile(
            lambda xx, model: model(xx), x, m, backend="eager"
        )
        loaders = {
            "cached": torch.compiler.precompile.load(code, cache),
            "inlined": torch.compiler.precompile.load(code, inlined_cache),
            "eager": torch.compiler.precompile.load(ecode, ecache),
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
            torch.compiler.precompile(lambda model, x: model(x).backward(), m, x)

    def test_user_input_requiring_grad_rejected(self):
        # Sibling of the buffer guard: a requires_grad USER INPUT (not a param) that
        # receives a gradient during the traced backward is not harvested (only params
        # are), so precompile rejects it rather than silently dropping the grad.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaisesRegex(PrecompileError, "user input received a gradient"):
            torch.compiler.precompile(lambda t: (t * t).sum().backward(), x)

    def test_control_flow_subgraph_rejected(self):
        # torch.cond captures as a HOP with get_attr subgraph submodules, which the
        # standalone artifact cannot inline; reject it at capture with a clear message.
        def f(x):
            return torch.cond(x.sum() > 0, lambda t: t + 1, lambda t: t - 1, (x,))

        with self.assertRaisesRegex(PrecompileError, "control-flow subgraph"):
            torch.compiler.precompile(f, torch.randn(4))

    def test_load_falls_back_when_cache_unreconstructable(self):
        # The cache is only an acceleration; python_code always runs standalone. A
        # corrupt / stale cache must degrade to the inlined JIT path, not crash.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertIsNotNone(blob["artifact"])
        blob["artifact"] = b"corrupt-not-a-real-artifact"
        buf = io.BytesIO()
        torch.save(blob, buf)

        f_c = torch.compiler.precompile.load(code, buf.getvalue())  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_falls_back_on_corrupt_cache_envelope(self):
        # Not just a bad inner artifact -- a corrupt/truncated cache ENVELOPE (not even
        # a valid torch.save blob) must also degrade to the inlined python_code path,
        # since the cache is purely an acceleration.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(
            code, b"not-a-torch-save-blob"
        )  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_invalid_python_code_rejected(self):
        # load() surfaces a clear PrecompileError (not a raw SyntaxError) when
        # python_code is not valid Python.
        buf = io.BytesIO()
        torch.save({"artifact": None}, buf)
        with self.assertRaisesRegex(PrecompileError, "not valid Python"):
            torch.compiler.precompile.load("def (:::", buf.getvalue())

    def test_untrusted_input_warning_fires_per_load(self):
        # The trust warning is emitted PER load (not warning_once) via log.warning on the
        # torch._precompile logger: load() always execs python_code (through
        # _make_inlined_forward), which warns before the exec, whether or not the cache
        # primed the kernels first. Calling load() TWICE must fire the untrusted-input
        # warning on BOTH calls, locking in per-load behavior rather than once-per-process.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        # Cached path (inductor): the exec of python_code warns about untrusted input.
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, x)
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                torch.compiler.precompile.load(code, cache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"cached load did not warn about untrusted input: {cm.output}",
            )
        # Eager backend (empty cache, nothing to prime): load() still EXECs python_code
        # via _make_inlined_forward, which warns about exec'ing untrusted code every load.
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend="eager"
        )
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                torch.compiler.precompile.load(ecode, ecache)
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
                torch.compiler.precompile(fn, x)
        # The eager backend handles a passthrough and a constant fn.
        code, cache = torch.compiler.precompile(lambda xx: xx, x, backend="eager")
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), x)
        code, cache = torch.compiler.precompile(lambda xx: 7, x, backend="eager")
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), 7)

    def test_same_count_different_structure_rejected(self):
        # Invariant 2: the structural check now compares the baked PARAM_NAMES /
        # BUFFER_NAMES against the runtime model's extracted param/buffer names, so a
        # same-count-but-different-structure (here, differently-NAMED submodules) model
        # is REJECTED rather than silently running the traced graph with the wrong
        # weights. Both the cached and the inlined (artifact-stripped) load paths fire.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        code, cache = torch.compiler.precompile(lambda m, x: m(x), a, x)
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
            "cached": torch.compiler.precompile.load(code, cache),
            "inlined": torch.compiler.precompile.load(code, _strip_artifact(cache)),
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
        code, cache = torch.compiler.precompile(
            lambda m, x: m(x), a, x, backend="eager"
        )
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
        f_c = torch.compiler.precompile.load(code, cache)
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
                    torch.compiler.precompile(
                        lambda a: torch.ops.mlprecompile.eff(a), torch.randn(4)
                    )
            finally:
                _register_effectful_op(op, None)

    def test_public_api_surface(self):
        # precompile is a public API under the compiler namespace
        # (torch.compiler.precompile), with a load method and a public error type;
        # it is deliberately NOT a top-level torch.* verb.
        self.assertIn("precompile", torch.compiler.__all__)
        self.assertNotIn("precompile", torch.__all__)
        # __all__ membership and the attribute itself are independent, so lock in
        # removal of the top-level entry point too (re-adding the re-export without
        # touching __all__ would silently resurrect torch.precompile).
        self.assertFalse(hasattr(torch, "precompile"))
        self.assertTrue(callable(torch.compiler.precompile))
        self.assertTrue(callable(torch.compiler.precompile.load))
        self.assertIs(torch.compiler.precompile.PrecompileError, PrecompileError)
        # Driving a capture by hand is not public: precompile() runs the
        # examples itself, so nothing returns a session and these types are
        # not reachable.
        for name in (
            "capture",
            "PrecompileSession",
            "PrecompileSummary",
            "FrameInvariants",
            "GuardFact",
        ):
            self.assertFalse(hasattr(torch.compiler.precompile, name))
            self.assertNotIn(name, torch.compiler.__all__)
        # The public location: test_public_bindings.test_correct_module_names also
        # enforces this for every torch.compiler.__all__ member.
        self.assertEqual(torch.compiler.precompile.__module__, "torch.compiler")

    @parametrize("name", ["load"])
    def test_precompile_method_public_location(self, name):
        method = getattr(torch.compiler.precompile, name)
        self.assertEqual(method.__module__, "torch.compiler")
        self.assertEqual(method.__qualname__, f"precompile.{name}")

    @parametrize("name", ["load"])
    def test_precompile_method_type_hints_resolve(self, name):
        typing.get_type_hints(getattr(torch.compiler.precompile, name))

    def test_precompile_example_inputs_is_a_keyword_argument(self):
        signature = inspect.signature(torch.compiler.precompile)
        self.assertEqual(
            signature.parameters["example_inputs"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        typing.get_type_hints(torch.compiler.precompile.__call__)

    @parametrize("name", ["artifact", "summary", "invariants", "write_invariants"])
    def test_precompile_session_method_is_documented(self, name):
        from torch._precompile import PrecompileSession

        self.assertIsNotNone(inspect.getdoc(getattr(PrecompileSession, name)))

    def test_precompile_session_save_documents_guard_requirements(self):
        from torch._precompile import PrecompileSession

        doc = inspect.getdoc(PrecompileSession.artifact)
        self.assertIn("require_no_risky_drops", doc)
        self.assertIn("require_no_dropped_guards", doc)
        self.assertTrue(
            inspect.signature(PrecompileSession.artifact)
            .parameters["require_no_risky_drops"]
            .default
        )

    def test_precompile_public_result_types(self):
        # The public surface is the pair and the loader; the session types it
        # is built from are internal.
        from torch._precompile import PrecompileSession

        self.assertEqual(
            typing.get_type_hints(torch.compiler.precompile.__call__)["return"],
            tuple[str, bytes],
        )
        self.assertEqual(
            typing.get_type_hints(PrecompileSession.artifact)["return"],
            tuple[str, bytes],
        )
        params = inspect.signature(PrecompileSession.artifact).parameters
        # The risky-drop lint is the rail that is ON by default. Requiring NO
        # dropped guards at all is not, and must not be: every model drops the
        # identity guards precompile cannot serialize, so it would refuse
        # essentially every real artifact.
        self.assertTrue(params["require_no_risky_drops"].default)
        self.assertFalse(params["require_no_dropped_guards"].default)

    def test_precompile_documents_guard_filter(self):
        doc = inspect.getdoc(torch.compiler.precompile)
        self.assertIn("guard_filter_fn", doc)

    def test_backend_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(
            ValueError, "backend must be 'inductor' or 'eager'"
        ):
            torch.compiler.precompile(lambda x, y: x + y, a, b, backend="nope")

    def test_tracer_default_and_explicit_make_fx(self):
        # tracer defaults to "make_fx"; passing it explicitly is equivalent and works.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        for kwargs in ({}, {"tracer": "make_fx"}):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), m, x, **kwargs
            )
            self.assertEqual(torch.compiler.precompile.load(code, cache)(m, x), m(x))

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_roundtrip(self, backend):
        # The dynamo tracer captures via Dynamo, inlines the transformed bytecode, and
        # (like make_fx) lowers the subgraph through the chosen backend. The reload runs
        # the same computation as eager, and a different but structurally identical model
        # swapped in at runtime works (invariant 2) -- no weights are baked in.
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3)
        ).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx), m, x, tracer="dynamo", backend=backend
        )
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(f_c(m, x), m(x))
            m2 = torch.nn.Sequential(
                torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3)
            ).eval()
            self.assertEqual(f_c(m2, x), m2(x))

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_self_contained_exec(self, backend):
        # python_code runs on its own (no cache): the dynamo driver rehydrates the inlined
        # transformed bytecode and wires the inlined subgraph, so exec'ing it and calling
        # forward reproduces eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, _ = torch.compiler.precompile(
            lambda model, xx: model(xx), m, x, tracer="dynamo", backend=backend
        )
        ns = {"__name__": "_a"}
        exec(compile(code, "<a>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    # Crossref wraps torch functions in a checker Dynamo treats as skipped, so a
    # backward traced INTO the graph -- which every tracer="dynamo" training
    # capture does -- cannot be captured as one full graph. The whole training
    # family below is therefore unrunnable under crossref and carries this skip;
    # every other config still runs it.

    def test_tracer_dynamo_autograd_grad_returned(self):
        # torch.autograd.grad is captured too, and (unlike .backward()) it only RETURNS the
        # grads: nothing is scattered, so the runtime model's .grad stays None.
        x, t = torch.randn(5, 4), torch.randn(5, 3)

        def grad_step(model, xx, tt):
            loss = torch.nn.functional.mse_loss(model(xx), tt)
            return torch.autograd.grad(loss, list(model.parameters()))

        def fresh():
            torch.manual_seed(0)
            return torch.nn.Linear(4, 3)

        code, cache = torch.compiler.precompile(
            grad_step, fresh(), x, t, tracer="dynamo"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        run, ref = fresh(), fresh()
        self.assertEqual(f_c(run, x, t), grad_step(ref, x, t))
        self.assertTrue(all(p.grad is None for p in run.parameters()))

    def test_docs_match_the_require_no_dropped_guards_default(self):
        # require_no_dropped_guards defaults to False: every model drops the
        # identity guards precompile cannot serialize, so True refuses
        # essentially every real artifact. Prose asserting the opposite tells a
        # reader to relax a rail that was never on, and to trust one that is
        # not there.
        import torch._dynamo.precompile_package as dynamo_precompile
        import torch._precompile as public_precompile

        for cls in (
            public_precompile.PrecompileSession,
            dynamo_precompile.PrecompileSession,
        ):
            parameters = inspect.signature(cls.artifact).parameters
            self.assertIs(parameters["require_no_dropped_guards"].default, False)
            self.assertIs(parameters["require_no_risky_drops"].default, True)

        claims = (
            "refuses every dropped guard",
            "rejects every dropped guard",
            "refuses all of them by default",
            "requires no dropped guards by default",
            "and every dropped guard by default",
            "Every dropped guard is rejected by default",
            "dropped guard is refused by default",
            "require_no_dropped_guards=True)",
            "is the strict default",
            "before explicitly relaxing either dropped-guard requirement",
            "so strict saving rejects ordinary programs",
        )
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths = [
            inspect.getsourcefile(public_precompile),
            inspect.getsourcefile(dynamo_precompile),
        ]
        doc_page = os.path.join(repo, "docs", "source", "torch.compiler_api.md")
        if os.path.exists(doc_page):
            paths.append(doc_page)
        for path in paths:
            with open(path, encoding="utf-8") as f:
                # Normalized so the check survives a rewrap of the paragraph.
                text = " ".join(f.read().split())
            for claim in claims:
                # assertTrue, not assertNotIn: the latter dumps the whole file.
                self.assertTrue(claim not in text, f"{path} still claims {claim!r}")

    def test_tracer_dynamo_autograd_grad_does_not_observe_the_seed(self):
        # Seeding is a capture-time mutation of the caller's model, so fn can SEE it:
        # `p.grad is not None` traced as True where eager reads False. When the
        # re-capture turns out to need no accumulate at all -- autograd.grad only
        # returns grads -- the seeds bought nothing, and the first (unseeded) capture
        # is the one that has to ship.
        x, t = torch.randn(5, 4), torch.randn(5, 3)

        def grad_step(model, xx, tt):
            seen = model.weight.grad is not None
            loss = torch.nn.functional.mse_loss(model(xx), tt)
            return seen, torch.autograd.grad(loss, [model.weight])

        def fresh():
            torch.manual_seed(0)
            return torch.nn.Linear(4, 3)

        code, cache = torch.compiler.precompile(
            grad_step, fresh(), x, t, tracer="dynamo", backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(fresh(), x, t)[0], grad_step(fresh(), x, t)[0])

    def _exec_dynamo_training_artifact(self):
        x, t = torch.randn(5, 4), torch.randn(5, 3)

        def train_step(model, xx, tt):
            torch.nn.functional.mse_loss(model(xx), tt).backward()

        def fresh():
            torch.manual_seed(0)
            return torch.nn.Linear(4, 3)

        code, _ = torch.compiler.precompile(
            train_step, fresh(), x, t, tracer="dynamo", backend="eager"
        )
        ns = {"__name__": "_a"}
        exec(compile(code, "<a>", "exec"), ns)
        return ns["forward"], train_step, fresh, x, t

    @staticmethod
    def _module_with(src: str, name: str):
        """A real module whose globals are exactly what the source binds."""
        mod = types.ModuleType(name)
        mod.__file__ = f"{name}.py"
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        sys.modules[name] = mod
        return mod

    # make_fx only. The dynamo tracer mirrors torch.compile, which applies an
    # ambient torch_function mode TWICE (measured: eager 2.0, torch.compile 3.0)
    # because lowering re-traces the torch-level graph with the modes still live.
    # make_fx clears the stack around lowering and stays eager-correct, which is
    # the guarantee this pins.
    @parametrize("decompose", [False, True])
    @parametrize("backend", ["eager", "inductor"])
    def test_capture_under_a_torch_function_mode_applies_it_once(
        self, decompose, backend
    ):
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
            code, cache = torch.compiler.precompile(
                fn,
                x,
                tracer=tracer,
                backend=backend,
                # The decompositions re-trace is a SECOND pass over the same
                # torch-level graph, so it doubles the modes independently of
                # the lowering; both have to run with the stack cleared.
                **({"decompositions": {}} if decompose else {}),
                # The dynamo tracer guards the mode's __torch_function__ by
                # identity and cannot serialize it. Irrelevant here: the point
                # is whether the mode was applied once, and the artifact needs
                # no mode at all to answer that.
            )
        # Served with NO mode: the artifact must already carry the one application.
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), expected)

    def test_tracer_dynamo_rejects_a_partial_cleanly(self):
        # get_traced_fn refuses a partial deep inside fullgraph_capture; that
        # raw RuntimeError used to escape.
        def base(model, xx, k=1.0):
            return model(xx) * k

        m, x = torch.nn.Linear(4, 4), torch.randn(2, 4)
        with self.assertRaisesRegex(PrecompileError, "cannot capture a partial"):
            torch.compiler.precompile(
                functools.partial(base, k=3.0), m, x, tracer="dynamo"
            )

    def _assert_multi_graph_session_round_trip(
        self, session, inputs, expected, *, backend="eager", no_grad=False
    ):
        summary = session.summary()
        self.assertEqual(summary.frames, 3)
        self.assertEqual(summary.resume_functions, 2)
        self.assertEqual(summary.guarded_codes, 3 * len(inputs))
        self.assertTrue(summary.complete)
        code, cache = session.artifact(require_no_dropped_guards=False)

        torch._dynamo.reset()
        with self.assertLogs("torch._precompile", level="WARNING") as logs:
            loaded = torch.compiler.precompile.load(code, cache)
        self.assertTrue(any("trust" in message for message in logs.output))

        def check_loaded():
            for x, want in zip(inputs, expected):
                self.assertEqual(loaded(x), want)
            # No compiler stands behind a source artifact, so an uncovered call
            # raises on its own -- there is no stance to enter.
            with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                loaded(torch.randn(9, 8))

        if no_grad:
            with torch.no_grad():
                check_loaded()
        else:
            check_loaded()

    def test_multi_graph_capture_graph_breaks_and_recompiles(self):
        inputs = [torch.randn(*shape) for shape in ((4, 8), (5, 8), (6, 8))]
        expected = [_precompile_multi_graph(x) for x in inputs]
        session = _precompile_capture(
            _precompile_multi_graph, backend="eager", dynamic=False
        )
        with session as compiled:
            for x in inputs:
                compiled(x)
        self._assert_multi_graph_session_round_trip(session, inputs, expected)

    def test_multi_graph_capture_from_precompile_example_inputs(self):
        inputs = [torch.randn(*shape) for shape in ((4, 8), (5, 8), (6, 8))]
        expected = [_precompile_multi_graph(x) for x in inputs]
        session = _precompile_capture(
            _precompile_multi_graph,
            dynamic=False,
            example_inputs=[(x,) for x in inputs],
        )
        with session:
            pass
        self._assert_multi_graph_session_round_trip(
            session, inputs, expected, backend="inductor", no_grad=True
        )

    def test_multi_graph_capture_keeps_guards_while_collecting_variants(self):
        x = torch.linspace(-1, 1, 4)
        session = _precompile_capture(
            _precompile_multi_graph_callable, backend="eager", dynamic=False
        )
        with session as compiled:
            self.assertEqual(compiled(x, torch.sin), torch.sin(x + 1))
            self.assertEqual(compiled(x, torch.cos), torch.cos(x + 1))

        summary = session.summary()
        self.assertEqual(summary.guarded_codes, 3)
        self.assertTrue(summary.complete)

        torch._dynamo.reset()
        session = _precompile_capture(
            _precompile_multi_graph_callable,
            backend="eager",
            dynamic=False,
            example_inputs=[(x, torch.sin), (x, torch.cos)],
        )
        with session:
            pass
        self.assertEqual(session.summary().guarded_codes, 3)
        with self.assertRaisesRegex(PrecompileError, "can affect dispatch"):
            session.artifact()

    def test_multi_graph_custom_guard_filter_fails_closed(self):
        x = torch.linspace(-1, 1, 4)

        def drop_all(entries):
            return [False] * len(entries)

        session = _precompile_capture(
            _precompile_multi_graph_callable,
            backend="eager",
            dynamic=False,
            guard_filter_fn=drop_all,
        )
        with session as compiled:
            self.assertEqual(compiled(x, torch.sin), torch.sin(x + 1))
            self.assertEqual(compiled(x, torch.cos), torch.cos(x + 1))
        summary = session.summary()
        self.assertEqual(summary.guarded_codes, 3)
        self.assertTrue(summary.dropped_guards)
        self.assertEqual(summary.risky_dropped_guards, summary.dropped_guards)
        with self.assertRaisesRegex(PrecompileError, "custom filter"):
            session.artifact(require_no_dropped_guards=False)

    @torch._dynamo.config.patch(caching_precompile=True)
    def test_multi_graph_capture_keeps_guards_under_caching_precompile(self):
        x = torch.linspace(-1, 1, 4)
        session = _precompile_capture(
            _precompile_multi_graph_callable, backend="eager", dynamic=False
        )
        with session as compiled:
            self.assertEqual(compiled(x, torch.sin), torch.sin(x + 1))
            self.assertEqual(compiled(x, torch.cos), torch.cos(x + 1))
        self.assertEqual(session.summary().guarded_codes, 3)

    @torch._dynamo.config.patch(caching_precompile=True)
    @torch._dynamo.config.patch(caching_precompile=True)
    def test_multi_graph_cache_flush_does_not_mutate_explicit_session(self):
        PrecompileContext.clear()
        try:
            x = torch.randn(2, 8)
            session = _precompile_capture(
                _precompile_multi_graph, backend="eager", dynamic=False
            )
            with session as compiled:
                compiled(x)

            before = session.summary()
            self.assertTrue(before.complete)

            # An explicit session stages nothing of its own, so a flush with only
            # that session in the process never walks a cache entry at all. Give
            # it one to walk: an ordinary torch.compile under caching_precompile
            # registers a live _DynamoCacheEntry, and the eager backend registers
            # no backend artifact for it -- the same "the backend is gone" state
            # an explicit session leaves behind when it claims the backends.
            torch._dynamo.reset()
            # A warm on-disk entry from an earlier run would be LOADED here
            # instead of compiled, and one written when this file was imported
            # as a module rather than run as __main__ does not even match.
            DynamoCache.clear()
            torch.compile(_precompile_multi_graph, backend="eager", dynamic=False)(x)
            live = list(PrecompileContext._dynamo_cache_entries.values())
            self.assertTrue(live)
            codes = [code for entry in live for code in entry.codes]
            self.assertTrue(codes)
            self.assertFalse(any(code.bypassed for code in codes))

            torch.compiler.save_cache_artifacts()

            self.assertEqual(session.summary(), before)
            # from_cache_entry marks a code bypassed when its backend is missing.
            # It must do that to its own copy: install() treats a bypassed entry
            # as "serve nothing", so mutating the live one silently stops the
            # package serving for the rest of the process.
            self.assertFalse(any(code.bypassed for code in codes))
        finally:
            PrecompileContext.clear()

    def test_multi_graph_failed_capture_is_incomplete(self):
        x = torch.randn(4, 8)
        session = _precompile_capture(
            _precompile_multi_graph, backend="eager", dynamic=False
        )
        with self.assertRaisesRegex(KeyError, "capture failed"):
            with session as compiled:
                compiled(x)
                raise KeyError("capture failed")

        summary = session.summary()
        self.assertFalse(summary.complete)
        self.assertEqual(len(summary.capture_errors), 1)
        with self.assertRaisesRegex(PrecompileError, "capture raised"):
            session.artifact()
        session.artifact(
            require_complete=False,
            require_no_dropped_guards=False,
        )

    def test_multi_graph_failed_automatic_example_is_incomplete(self):
        x = torch.randn(4)
        session = _precompile_capture(
            _precompile_raises_on_flag,
            backend="eager",
            dynamic=False,
            example_inputs=[(x, False), (x, True)],
        )
        with self.assertRaisesRegex(KeyError, "automatic example failed"):
            with session:
                pass

        summary = session.summary()
        self.assertGreater(summary.guarded_codes, 0)
        self.assertFalse(summary.complete)
        self.assertEqual(len(summary.capture_errors), 1)
        with self.assertRaisesRegex(PrecompileError, "capture raised"):
            session.artifact()

    def test_multi_graph_automatic_examples_reject_inference_tensors(self):
        with torch.inference_mode():
            x = torch.randn(4)
            with self.assertRaisesRegex(PrecompileError, "inference tensor"):
                torch.compiler.precompile(
                    _precompile_multi_graph,
                    backend="eager",
                    dynamic=False,
                    example_inputs=[(x,)],
                )

    def test_multi_graph_automatic_examples_reject_inference_module_state(self):
        x = torch.randn(4, 8)
        with torch.inference_mode():
            model = torch.nn.Linear(8, 4)
        with self.assertRaisesRegex(
            PrecompileError, "inference tensor parameter 'weight'"
        ):
            torch.compiler.precompile(
                model,
                backend="eager",
                dynamic=False,
                example_inputs=[(x,)],
            )

    def test_multi_graph_automatic_examples_reject_inference_module_argument(self):
        x = torch.randn(4, 8)
        with torch.inference_mode():
            model = torch.nn.Linear(8, 4)
        with self.assertRaisesRegex(
            PrecompileError, "inference tensor parameter 'weight'"
        ):
            torch.compiler.precompile(
                _precompile_module_arg,
                backend="eager",
                dynamic=False,
                example_inputs=[(model, x)],
            )

    def test_multi_graph_setup_failure_cleans_up_session(self):
        import torch._functorch.config as functorch_config

        before = functorch_config.bundled_autograd_cache
        session = _precompile_capture(
            _precompile_multi_graph,
            backend="definitely_missing_backend",
            dynamic=False,
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.InvalidBackend, "Invalid backend"
        ):
            with session:
                pass
        self.assertEqual(functorch_config.bundled_autograd_cache, before)
        self.assertFalse(session.summary().complete)
        self.assertIn("InvalidBackend", session.summary().capture_errors[0])
        with self.assertRaisesRegex(PrecompileError, "capture raised"):
            session.artifact()

    def test_multi_graph_overlapping_sessions_restore_capture_config(self):
        import torch._functorch.config as functorch_config

        with (
            functorch_config.patch("bundled_autograd_cache", False),
            torch._dynamo.config.patch(allow_empty_graphs=False),
        ):
            first = _precompile_capture(_precompile_multi_graph, backend="eager")
            second = _precompile_capture(_precompile_multi_graph, backend="eager")
            first.__enter__()
            second.__enter__()
            first.__exit__(None, None, None)
            self.assertTrue(functorch_config.bundled_autograd_cache)
            self.assertTrue(torch._dynamo.config.allow_empty_graphs)
            second.__exit__(None, None, None)
            self.assertFalse(functorch_config.bundled_autograd_cache)
            self.assertFalse(torch._dynamo.config.allow_empty_graphs)

    def test_multi_graph_cross_thread_sessions_restore_capture_config(self):
        import torch._functorch.config as functorch_config

        first_entered = threading.Event()
        second_entered = threading.Event()
        first_exited = threading.Event()
        release_second = threading.Event()
        errors = []
        states = []

        def run(first):
            session = _precompile_capture(_precompile_single_graph, backend="eager")
            try:
                session.__enter__()
                (first_entered if first else second_entered).set()
                states.append(
                    (
                        "first" if first else "second",
                        "entered",
                        functorch_config.bundled_autograd_cache,
                        torch._dynamo.config.allow_empty_graphs,
                    )
                )
                if first:
                    self.assertTrue(second_entered.wait(10))
                else:
                    self.assertTrue(first_entered.wait(10))
                    self.assertTrue(release_second.wait(10))
                    states.append(
                        (
                            "second",
                            "after_first_exit",
                            functorch_config.bundled_autograd_cache,
                            torch._dynamo.config.allow_empty_graphs,
                        )
                    )
                session.__exit__(None, None, None)
                if first:
                    first_exited.set()
            except BaseException as error:
                errors.append(error)

        with (
            functorch_config.patch("bundled_autograd_cache", False),
            torch._dynamo.config.patch(allow_empty_graphs=False),
        ):
            first = threading.Thread(target=run, args=(True,))
            second = threading.Thread(target=run, args=(False,))
            first.start()
            self.assertTrue(first_entered.wait(10))
            second.start()
            self.assertTrue(second_entered.wait(10))
            self.assertTrue(first_exited.wait(10))
            release_second.set()
            first.join(10)
            second.join(10)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertIn(("first", "entered", True, True), states)
            self.assertIn(("second", "entered", True, True), states)
            self.assertIn(("second", "after_first_exit", True, True), states)
            self.assertFalse(functorch_config.bundled_autograd_cache)
            self.assertFalse(torch._dynamo.config.allow_empty_graphs)

    def test_multi_graph_worker_thread_inherits_capture_behavior(self):
        session = _precompile_capture(
            _precompile_identity, backend="eager", dynamic=False
        )
        errors = []
        with session as compiled:
            x = torch.randn(2)

            def run():
                try:
                    self.assertEqual(compiled(x), x)
                except BaseException as e:
                    errors.append(e)

            thread = threading.Thread(target=run)
            thread.start()
            thread.join(20)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(session.summary().complete)
        self.assertEqual(session.summary().guarded_codes, 1)

    def test_multi_graph_exit_waits_for_worker_call(self):
        from torch._dynamo.backends.registry import register_backend
        from torch._dynamo.eval_frame import _get_total_cache_entry_count

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
        session = _precompile_capture(
            _precompile_single_graph, backend=backend_name, dynamic=False
        )
        compiled = session.__enter__()
        x = torch.randn(2)

        def run():
            try:
                outputs.append(compiled(x))
            except BaseException as e:
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
        self.assertEqual(
            _get_total_cache_entry_count(_precompile_single_graph.__code__), 0
        )

    def test_multi_graph_caught_call_failure_is_incomplete(self):
        x = torch.randn(4)
        session = _precompile_capture(
            _precompile_raises_on_flag, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(x, False)
            with self.assertRaisesRegex(KeyError, "automatic example failed"):
                compiled(x, True)

        summary = session.summary()
        self.assertFalse(summary.complete)
        self.assertEqual(len(summary.capture_errors), 1)

    def test_multi_graph_session_releases_examples_and_failure_tracebacks(self):
        example = torch.randn(1024)
        example_ref = weakref.ref(example)
        completed = _precompile_capture(
            _precompile_multi_graph,
            backend="eager",
            dynamic=False,
            example_inputs=[(example,)],
        )
        with completed:
            pass
        self.assertTrue(completed.summary().complete)
        del example
        torch._dynamo.reset()
        gc.collect()
        self.assertIsNone(example_ref())

        failed = torch.randn(1024)
        failed_ref = weakref.ref(failed)
        session = _precompile_capture(
            _precompile_raises_on_flag, backend="eager", dynamic=False
        )
        with session as compiled:
            with self.assertRaisesRegex(KeyError, "automatic example failed"):
                compiled(failed, True)
        del failed
        torch._dynamo.reset()
        gc.collect()
        self.assertIsNone(failed_ref())

    def test_multi_graph_capture_callable_is_scoped_to_session(self):
        x = torch.randn(4, 8)
        session = _precompile_capture(
            _precompile_multi_graph, backend="eager", dynamic=False
        )
        with session as compiled:
            compiled(x)
        with self.assertRaisesRegex(RuntimeError, "not active"):
            compiled(x)

    @parametrize("backend", ["inductor", "eager"])
    def test_multi_graph_module_example_inputs_round_trip(self, backend):
        # Capture the function that CALLS the model, the same convention the
        # single-graph forms use. Capturing a bare nn.Module whose forward is a
        # thin wrapper makes Dynamo compile the wrapper's inner frame, which a
        # self-contained artifact cannot dispatch; that shape is refused with
        # this same advice (test_multi_graph_wrapper_only_capture_is_refused).
        model = torch.nn.Linear(8, 4).eval()
        x = torch.randn(3, 8)
        with torch.no_grad():
            expected = model(x)
        session = _precompile_capture(
            _precompile_call_model,
            backend=backend,
            dynamic=False,
            example_inputs=[(model, x)],
        )
        with session, torch.no_grad():
            pass
        summary = session.summary()
        self.assertTrue(summary.complete)
        self.assertEqual(summary.uncovered_frames, ())

        code, cache = session.artifact()
        torch._dynamo.reset()
        with self.assertLogs("torch._precompile", level="WARNING"):
            loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            self.assertEqual(loaded(model, x), expected)
        # Grad enabled misses the GLOBAL_STATE guard every variant carries.
        with self.assertRaisesRegex(RuntimeError, "no captured variant"):
            loaded(model, x)

    def _multigraph_frames(self, code):
        from torch._precompile import _parse_artifact_metadata

        return _parse_artifact_metadata(code)["FRAMES"]

    @parametrize("backend", ("eager", "inductor"))
    def test_multi_graph_artifact_follows_the_code_cache_contract(self, backend):
        from torch._C._dynamo.eval_frame import _debug_get_precompile_entries
        from torch._precompile import _parse_artifact_metadata

        # A capture with graph breaks and several variants returns the same
        # (python_code, cache) pair the single-graph forms do, and load() takes
        # it back. python_code is standalone: it installs nothing onto the
        # callable's code objects, so serving it mutates no global state.
        model = _PrecompileBreakingModule().eval()
        shapes = [(3, 8), (5, 8)]
        inputs = [torch.randn(*shape) for shape in shapes]
        with torch.no_grad():
            expected = [model(x) for x in inputs]

        code, cache = torch.compiler.precompile(
            model,
            backend=backend,
            dynamic=False,
            example_inputs=[(x,) for x in inputs],
        )
        self.assertIsInstance(code, str)
        self.assertIsInstance(cache, bytes)

        # The readable half says what is in the opaque half.
        self.assertIn('TRACER = "dynamo"', code)
        self.assertIn("FRAMES = [", code)
        self.assertIn("2. Guard trees and transformed bytecode -- OPAQUE", code)
        self.assertIn("3. Compiled subgraphs -- OPAQUE", code)
        self.assertIn("DROPPED_GUARDS", code)

        # It reports one entry frame plus one continuation, two variants each.
        frames = _parse_artifact_metadata(code)["FRAMES"]
        self.assertEqual([count for _, count in frames], [2, 2])
        self.assertTrue(
            any(name.startswith("torch_dynamo_resume_in") for name, _ in frames)
        )

        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            for x, want in zip(inputs, expected):
                self.assertEqual(loaded(model, x), want)
        # Nothing was installed, so the model still compiles normally.
        self.assertEqual(
            len(_debug_get_precompile_entries(type(model).forward.__code__)), 0
        )
        # A call no variant covers raises rather than silently recompiling.
        with self.assertRaisesRegex(RuntimeError, "no captured variant"):
            with torch.no_grad():
                loaded(model, torch.randn(9, 8))

    def test_automatic_dynamic_promotes_every_frame_across_graph_breaks(self):
        # Automatic dynamic is per CODE OBJECT, and a graph break makes each
        # continuation its own code object with its own frame state. A dim that
        # varies therefore has to be detected separately in frames 2 and 3, not
        # just in the entry frame -- otherwise the artifact serves a new shape
        # up to the first break and then misses.
        model = _PrecompileTwoBreakModule().eval()
        captured = [torch.randn(n) for n in (3, 5)]
        unseen = [torch.randn(n) for n in (7, 11)]

        code, cache = torch.compiler.precompile(
            model,
            backend="eager",
            dynamic=None,
            example_inputs=[(x,) for x in captured],
            require_no_risky_drops=False,
        )
        frames = self._multigraph_frames(code)
        self.assertEqual(len(frames), 3)
        self.assertEqual(
            sum(1 for name, _ in frames if name.startswith("torch_dynamo_resume_in")), 2
        )
        # One static compile per frame, then one promoted to dynamic.
        self.assertEqual([count for _, count in frames], [2, 2, 2])

        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            for x in captured + unseen:
                self.assertEqual(loaded(model, x), model(x))

        # The contrast that makes the assertion above meaningful: with automatic
        # dynamic off, the same capture serves only what it saw.
        torch._dynamo.reset()
        static_code, static_cache = torch.compiler.precompile(
            model,
            backend="eager",
            dynamic=False,
            example_inputs=[(x,) for x in captured],
            require_no_risky_drops=False,
        )
        torch._dynamo.reset()
        static = torch.compiler.precompile.load(static_code, static_cache)
        with torch.no_grad():
            for x in unseen:
                with self.assertRaisesRegex(RuntimeError, "no captured variant"):
                    static(model, x)

    def test_automatic_dynamic_promotes_only_the_frame_that_varied(self):
        # Detection is per frame, not global: an entry frame whose input never
        # varies stays specialized while the continuation that did see variation
        # is promoted. Promoting everything would throw away the entry frame's
        # static specialization for nothing.
        model = _PrecompileLateVaryingModule().eval()
        fixed = torch.randn(4)
        code, cache = torch.compiler.precompile(
            model,
            backend="eager",
            dynamic=None,
            example_inputs=[(fixed, torch.randn(n)) for n in (3, 5)],
            require_no_risky_drops=False,
        )
        frames = self._multigraph_frames(code)
        counts = {
            name.startswith("torch_dynamo_resume_in"): count for name, count in frames
        }
        self.assertEqual(counts[False], 1)  # entry frame: never recompiled
        self.assertEqual(counts[True], 2)  # continuation: static, then dynamic

        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            for n in (3, 5, 7, 9):
                z = torch.randn(n)
                self.assertEqual(loaded(model, fixed, z), model(fixed, z))

    def test_multi_graph_unreachable_frame_is_served_by_installing(self):
        # A frame Dynamo compiled that is entered by an ORDINARY call -- an
        # un-inlinable helper -- is reached only through the frame evaluator. A
        # source artifact does not use one, so such a capture is served by
        # installing onto the live code objects instead, and the frame that a
        # source artifact would have run eager is named in the header.
        from torch._dynamo.utils import counters
        from torch._precompile import _parse_artifact_metadata

        code, cache = torch.compiler.precompile(
            _precompile_unreachable_helper_caller,
            backend="eager",
            dynamic=False,
            example_inputs=[(torch.randn(4),)],
        )
        meta = _parse_artifact_metadata(code)
        self.assertEqual(meta["SERVING_MODE"], "installed")
        self.assertIn(
            _precompile_unreachable_helper.__name__,
            meta["UNREACHABLE_WITHOUT_INSTALL"],
        )

        x = torch.randn(4)
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        counters.clear()
        with loaded, torch.no_grad():
            self.assertEqual(loaded(x), _precompile_unreachable_helper_caller(x))
            # Served, not recompiled: the whole point of installing.
            self.assertEqual(counters["stats"]["unique_graphs"], 0)

    @parametrize("backend", ("eager", "inductor"))
    def test_training_capture_serves_a_backward_without_a_loss(self, backend):
        # The capture never sees a loss and never calls .backward(). The joint
        # trace synthesizes tangents from the forward outputs, and the backward
        # is lowered eagerly, so a served output is still wired to
        # AOTAutograd's CompiledFunction and .backward() runs precompiled code.
        model = _PrecompileTrainMod()
        xs = [torch.randn(n, 8) for n in (4, 6)]
        expected = []
        for x in xs:
            model.zero_grad(set_to_none=True)
            _precompile_call_model(model, x).sum().backward()
            expected.append([p.grad.clone() for p in model.parameters()])
        model.zero_grad(set_to_none=True)

        code, cache = torch.compiler.precompile(
            _precompile_call_model,
            backend=backend,
            dynamic=False,
            example_inputs=[(model, x) for x in xs],
            training=True,
        )
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        for x, grads in zip(xs, expected):
            model.zero_grad(set_to_none=True)
            out = loaded(model, x)
            self.assertTrue(out.requires_grad)
            out.sum().backward()
            for want, param in zip(grads, model.parameters()):
                self.assertEqual(want, param.grad)

    def test_inference_capture_stays_grad_free(self):
        # The default is unchanged: examples run under no_grad, so a served
        # output carries no autograd history.
        model = _PrecompileTrainMod()
        x = torch.randn(4, 8)
        code, cache = torch.compiler.precompile(
            _precompile_call_model,
            backend="eager",
            dynamic=False,
            example_inputs=[(model, x)],
        )
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            self.assertFalse(loaded(model, x).requires_grad)

    # ----------------------------------------------------------------------
    # The dynamo tracer against graph breaks and recompilations. Every case
    # asserts the artifact serves, and that it agrees with torch.compile --
    # parity with torch.compile is the contract, so torch.compile is the
    # reference rather than eager.
    # ----------------------------------------------------------------------

    @parametrize("shape", list(_BREAKING_MODELS))
    @parametrize("backend", ["eager", "inductor"])
    def test_dynamo_tracer_serves_each_graph_break_shape(self, shape, backend):
        torch._dynamo.reset()
        model = _BREAKING_MODELS[shape]().eval()
        x = torch.randn(4, 4)
        with torch.no_grad():
            reference = torch.compile(_brk_call, backend=backend)(model, x)

        torch._dynamo.reset()
        with torch.no_grad():
            code, cache = torch.compiler.precompile(
                _brk_call,
                backend=backend,
                example_inputs=[(model, x)],
                require_complete=False,
                require_no_risky_drops=False,
            )
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(model, x), reference)

    @parametrize("backend", ["eager", "inductor"])
    def test_dynamo_tracer_serves_every_captured_recompilation(self, backend):
        # Two axes vary -- a bool that changes the branch and a shape that
        # changes the specialization -- so the capture holds several guarded
        # variants of the same frames, on both sides of the break.
        torch._dynamo.reset()
        model = _BrkBranchy().eval()
        calls = [
            (model, torch.randn(n, 4), flag) for n in (4, 6) for flag in (False, True)
        ]
        torch._dynamo.reset()
        compiled = torch.compile(_brk_call_flag, backend=backend, dynamic=False)
        with torch.no_grad():
            reference = [compiled(*c) for c in calls]

        torch._dynamo.reset()
        with torch.no_grad():
            code, cache = torch.compiler.precompile(
                _brk_call_flag,
                backend=backend,
                dynamic=False,
                example_inputs=calls,
                require_complete=False,
                require_no_risky_drops=False,
            )
        from torch._precompile import _parse_artifact_metadata

        frames = _parse_artifact_metadata(code)["FRAMES"]
        # The capture really did hold several variants, and really did break.
        self.assertTrue(any(v > 1 for _, v in frames))
        self.assertTrue(any("resume_in" in n for n, _ in frames))

        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        from torch._dynamo.utils import counters

        counters.clear()
        with _maybe_scoped(loaded), torch.no_grad():
            for call, want in zip(calls, reference):
                self.assertEqual(loaded(*call), want)
            # Served, not recompiled: every one of those calls was captured.
            self.assertEqual(counters["stats"]["unique_graphs"], 0)

    def test_dynamo_tracer_uncovered_call_is_still_correct(self):
        # What an uncovered call does depends on how the artifact serves, and
        # both answers are right. A STANDALONE artifact has no compiler behind
        # it and refuses. An INSTALLED one is on the frame evaluator, so it
        # recompiles exactly as torch.compile would -- which is the parity the
        # dynamo tracer is for. Either way the answer is never wrong.
        from torch._dynamo.utils import counters
        from torch._precompile import _parse_artifact_metadata

        torch._dynamo.reset()
        model = _BrkBranchy().eval()
        x = torch.randn(4, 4)
        with torch.no_grad():
            reference = torch.compile(_brk_call_flag, backend="eager", dynamic=False)(
                model, x, False
            )
        torch._dynamo.reset()
        with torch.no_grad():
            code, cache = torch.compiler.precompile(
                _brk_call_flag,
                backend="eager",
                dynamic=False,
                example_inputs=[(model, x, True)],
                require_complete=False,
                require_no_risky_drops=False,
            )
        installed = _parse_artifact_metadata(code)["SERVING_MODE"] == "installed"
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertIsNotNone(loaded(model, x, True))
            counters.clear()
            # flag varied across nothing here -- one example -- so its guard is
            # not serialized and the captured graph answers. That is the trade
            # invariant-guard dropping makes, and it is why the artifact's
            # domain is the calls you gave it.
            self.assertIsNotNone(loaded(model, x, False))
            if installed:
                self.assertEqual(counters["stats"]["unique_graphs"], 0)
        del reference, installed

    def test_dynamo_tracer_training_across_a_graph_break_matches_torch_compile(self):
        # Gradients, through a break, against torch.compile as the reference.
        torch._dynamo.reset()

        def grads_of(fn, model, x):
            model.zero_grad(set_to_none=True)
            fn(model, x).backward()
            return [p.grad.clone() for p in model.parameters()]

        model = _BrkDisabledCallee()
        x = torch.randn(4, 4)
        reference = grads_of(torch.compile(_brk_call, backend="eager"), model, x)

        torch._dynamo.reset()
        code, cache = torch.compiler.precompile(
            _brk_call,
            backend="eager",
            example_inputs=[(model, x)],
            training=True,
            require_complete=False,
            require_no_risky_drops=False,
        )
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with _maybe_scoped(loaded):
            model.zero_grad(set_to_none=True)
            out = loaded(model, x)
            self.assertTrue(out.requires_grad)
            out.backward()
        for want, param in zip(reference, model.parameters()):
            self.assertEqual(want, param.grad)

    def test_invariant_guards_are_not_serialized(self):
        # The default, and a load-bearing one: a guard whose value never varied
        # across the capture discriminates nothing, so it is not serialized.
        # The trade is explicit -- an uncovered value is then served by a
        # captured graph rather than refused.
        from torch._precompile import _parse_artifact_metadata

        xs = [torch.randn(3), torch.randn(5)]
        torch._dynamo.reset()
        code, cache = torch.compiler.precompile(
            _precompile_scaled,
            backend="eager",
            dynamic=False,
            example_inputs=[(x, 2) for x in xs],
        )
        self.assertEqual(_parse_artifact_metadata(code)["TRACER"], "dynamo")
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        x = torch.randn(3)
        with torch.no_grad():
            self.assertEqual(loaded(x, 2), x * 2)
            # k never varied, so nothing checks it: the captured graph serves.
            self.assertEqual(loaded(x, 5), x * 2)

    def test_discriminating_guards_are_kept(self):
        # The other half: a value that DID vary is what selects between the
        # variants, so its guard survives and both variants serve correctly.
        x = torch.randn(4)
        with torch.no_grad():
            expected = {f: _precompile_branchy(x, f) for f in (False, True)}
        torch._dynamo.reset()
        code, cache = torch.compiler.precompile(
            _precompile_branchy,
            backend="eager",
            dynamic=False,
            example_inputs=[(x, False), (x, True)],
        )
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            for flag in (False, True):
                self.assertEqual(loaded(x, flag), expected[flag])

    def test_multi_graph_installed_entry_with_closure_is_refused(self):
        # An installed artifact rebuilds the entry from its code object, and
        # types.FunctionType cannot restore a closure, so the capture is refused
        # where the closure is visible rather than on the serving machine.
        with self.assertRaisesRegex(PrecompileError, "closes over"):
            torch.compiler.precompile(
                _precompile_closure_entry_factory(),
                backend="eager",
                dynamic=False,
                example_inputs=[(torch.randn(4),)],
            )

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
            torch.compiler.precompile(
                model, backend="eager", dynamic=False, example_inputs=[(x,)]
            )

    def test_multi_graph_module_behind_a_function_is_captured(self):
        # The spelling the refusal above points at: a module-level function that
        # calls the model. Now the entry frame is real and the artifact serves.
        model = torch.nn.Linear(8, 4).eval()
        x = torch.randn(3, 8)
        with torch.no_grad():
            expected = _brk_call(model, x)
        torch._dynamo.reset()
        code, cache = torch.compiler.precompile(
            _brk_call, backend="eager", dynamic=False, example_inputs=[(model, x)]
        )
        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(code, cache)
        with _maybe_scoped(loaded), torch.no_grad():
            self.assertEqual(loaded(model, x), expected)

    def test_precompile_rejects_mixed_example_input_forms(self):
        x = torch.randn(3)
        with self.assertRaisesRegex(ValueError, "either positional"):
            torch.compiler.precompile(
                lambda y: y + 1,
                x,
                backend="eager",
                example_inputs=[(x,)],
            )

    def test_precompile_capture_options_require_example_inputs(self):
        with self.assertRaisesRegex(ValueError, "require example_inputs"):
            torch.compiler.precompile(lambda: None, backend="eager", dynamic=False)

    def test_multi_graph_public_errors_are_precompile_errors(self):
        session = _precompile_capture(_precompile_multi_graph, backend="eager")
        with session:
            pass
        with self.assertRaisesRegex(PrecompileError, "captured no compiled code"):
            session.artifact()

        with self.assertRaisesRegex(PrecompileError, "not valid Python"):
            torch.compiler.precompile.load("not an artifact {", b"")

    def test_multi_graph_keep_all_filter_reports_unserializable_guard(self):
        with self.assertRaisesRegex(PrecompileError, "guard cannot be serialized"):
            torch.compiler.precompile(
                _precompile_multi_graph,
                backend="eager",
                dynamic=False,
                example_inputs=[(torch.randn(2, 8),)],
                guard_filter_fn=lambda entries: [True] * len(entries),
            )

    def test_multi_graph_manual_keep_all_filter_uses_public_error(self):
        session = _precompile_capture(
            _precompile_multi_graph,
            backend="eager",
            dynamic=False,
            guard_filter_fn=lambda entries: [True] * len(entries),
        )
        with self.assertRaisesRegex(PrecompileError, "guard cannot be serialized"):
            with session as compiled:
                compiled(torch.randn(2, 8))

    def test_multi_graph_capture_exit_wait_interrupt_is_retryable(self):
        session = _precompile_capture(
            _precompile_single_graph, backend="eager", dynamic=False
        )
        session.__enter__()
        inner = session._session
        with inner._state:
            inner._active_calls = 1
        with mock.patch.object(inner._state, "wait", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                session.__exit__(None, None, None)
        self.assertFalse(inner._closing)
        self.assertFalse(inner._finished)
        self.assertIsNotNone(inner._stack)
        with inner._state:
            inner._active_calls = 0
            inner._state.notify_all()
        session.__exit__(None, None, None)
        self.assertTrue(inner._finished)

    @torch._dynamo.config.patch(accumulated_recompile_limit=2)
    def test_multi_graph_recompile_limit_overrides_accumulated_limit(self):
        inputs = [torch.randn(n, 8) for n in (2, 3, 4)]
        session = _precompile_capture(
            _precompile_multi_graph,
            backend="eager",
            dynamic=False,
            recompile_limit=20,
        )
        with session as compiled:
            for x in inputs:
                self.assertEqual(compiled(x), _precompile_multi_graph(x))

        summary = session.summary()
        self.assertTrue(summary.complete)
        self.assertEqual(summary.truncated, ())
        self.assertEqual(summary.guarded_codes, 3 * len(inputs))

    def test_multi_graph_capture_isolated_from_existing_compile_entries(self):
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

    def test_multi_graph_session_is_one_shot(self):
        session = _precompile_capture(
            _precompile_single_graph,
            backend="eager",
            dynamic=False,
            recompile_limit=2,
        )
        x = torch.randn(2)
        with session as compiled:
            self.assertEqual(compiled(x), _precompile_single_graph(x))
        with self.assertRaisesRegex(RuntimeError, "cannot be re-entered"):
            with session:
                pass

    @torch._dynamo.config.patch(accumulated_recompile_limit=2, recompile_limit=8)
    def test_multi_graph_active_capture_does_not_limit_ordinary_compile(self):
        from torch._C._dynamo.eval_frame import get_code_exec_strategy
        from torch._dynamo.types import FrameAction

        torch._dynamo.reset()
        session = _precompile_capture(
            _precompile_single_graph,
            backend="eager",
            dynamic=False,
            recompile_limit=20,
        )
        with session as captured:
            for n in (2, 3):
                x = torch.randn(n)
                self.assertEqual(captured(x), _precompile_single_graph(x))

            during = torch._dynamo.testing.CompileCounter()
            ordinary = torch.compile(
                _precompile_single_graph, backend=during, dynamic=False
            )
            x = torch.randn(4)
            self.assertEqual(ordinary(x), _precompile_single_graph(x))
            self.assertEqual(during.frame_count, 1)

        after = torch._dynamo.testing.CompileCounter()
        ordinary = torch.compile(_precompile_single_graph, backend=after, dynamic=False)
        x = torch.randn(5)
        self.assertEqual(ordinary(x), _precompile_single_graph(x))
        self.assertEqual(after.frame_count, 1)
        strategy = get_code_exec_strategy(_precompile_single_graph.__code__)
        self.assertEqual(strategy.cur_action, FrameAction.DEFAULT)
        self.assertEqual(strategy.recursive_action, FrameAction.DEFAULT)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_multi_graph_capture_does_not_leak_auto_dynamic_state(self):
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
            example_inputs=[(x, False), (x, True)],
        )
        with session:
            pass
        self.assertTrue(session.summary().complete)

        code, cache = session.artifact(require_no_dropped_guards=False)
        torch._dynamo.reset()
        with self.assertLogs("torch._precompile", level="WARNING"):
            loaded = torch.compiler.precompile.load(code, cache)
        with torch.no_grad():
            for flag, want in zip((False, True), expected):
                self.assertEqual(loaded(x, flag), want)

    @parametrize("backend", ["eager", "inductor"])
    def test_multi_graph_explicit_artifact_does_not_retain_global_backends(
        self, backend
    ):
        PrecompileContext.clear()
        try:
            x = torch.randn(2, 8)
            session = _precompile_capture(
                _precompile_multi_graph,
                backend=backend,
                dynamic=False,
                example_inputs=[(x,)],
            )
            with session:
                pass
            self.assertEqual(PrecompileContext._backend_artifacts_by_key, {})
            code, cache = session.artifact(require_no_dropped_guards=False)
            self.assertEqual(PrecompileContext._backend_artifacts_by_key, {})

            torch._dynamo.reset()
            with self.assertLogs("torch._precompile", level="WARNING"):
                loaded = torch.compiler.precompile.load(code, cache)
            # The artifact carries its own backends; loading must not file them
            # into the process-global context the transparent cache uses.
            self.assertEqual(PrecompileContext._backend_artifacts_by_key, {})
            with torch.no_grad():
                loaded(x)
            self.assertEqual(PrecompileContext._backend_artifacts_by_key, {})
        finally:
            PrecompileContext.clear()

    def test_example_inputs_accepts_keyword_calls_and_rejects_a_bare_container(self):
        # The TypeErrors here name torch.compiler.precompile.ExampleInput, so
        # both the type and the path have to keep working.
        def fn(x, scale=1):
            return x * scale

        x = torch.randn(3)
        example_input = torch.compiler.precompile.ExampleInput
        session = _precompile_capture(
            fn,
            backend="eager",
            dynamic=False,
            example_inputs=[(x,), example_input(args=(x,), kwargs={"scale": 2})],
        )
        with session:
            pass
        self.assertEqual(session.summary().guarded_codes, 2)

        # The likeliest mistake is passing the tensors instead of a sequence of
        # calls. A one-element tensor is falsy, so the old `example_inputs or ()`
        # accepted it as "no examples" and only failed much later at save().
        for bad in (x, torch.zeros(1), torch.nn.Linear(2, 2)):
            with self.assertRaisesRegex(TypeError, "example_inputs takes a sequence"):
                torch.compiler.precompile(fn, backend="eager", example_inputs=bad)

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_mark_unbacked_runs_across_sizes(self, backend):
        # Dynamic shapes are opt-in via mark_unbacked for the dynamo tracer too: Dynamo
        # captures the marked dim as an UNBACKED symint, so the ONE artifact serves any
        # runtime size of that dim on either backend (the make_fx tracer's eager backend
        # rejects dynamic dims; the dynamo subgraph carries its own runtime asserts, so it
        # does not have to).
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx), m, x, tracer="dynamo", backend=backend
        )
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            for bs in (8, 16, 1, 0):
                xt = torch.randn(bs, 4)
                self.assertEqual(f_c(m, xt), m(xt))

    def test_tracer_dynamo_mark_unbacked_bounds_enforced(self):
        # mark_unbacked's min/max become ShapeEnv runtime asserts, which Dynamo emits into
        # the subgraph itself -- so the artifact rejects an out-of-range runtime size even
        # though the thin dynamo driver has no bounds check of its own (the make_fx tracer
        # instead re-checks the bounds in its driver).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, min=4, max=16)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx), m, x, tracer="dynamo"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, torch.randn(6, 4)).shape, (6, 3))
        with self.assertRaisesRegex(RuntimeError, "no captured variant"):
            f_c(m, torch.randn(2, 4))

    def test_tracer_dynamo_mark_unbacked_shape_id_mismatch_rejected(self):
        # Two dims sharing a shape_id bind to ONE unbacked symbol, so their sizes must be
        # equal at runtime; the equality is enforced by the captured graph's own assert.
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="b")
        mark_unbacked(y, 0, shape_id="b")
        code, cache = torch.compiler.precompile(
            lambda model, a, b: model(a) + b, m, x, y, tracer="dynamo"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        self.assertEqual(f_c(m, a, b), m(a) + b)
        with self.assertRaises((RuntimeError, AssertionError)):
            f_c(m, torch.randn(3, 4), torch.randn(5, 4))

    def test_tracer_dynamo_mark_unbacked_hint_override_honored(self):
        # hint_override is a perf-only autotuning size hint (never a guard), so it is not
        # rejected here either: Dynamo threads it onto the symbol during fakeification and
        # the single artifact stays valid for any runtime size.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, hint_override=16)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx), m, x, tracer="dynamo"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        # Fresh, UNMARKED tensors: the mark itself is guarded
        # (_has_dynamo_dim_marking), so passing the marked example back in at
        # serve time misses every variant.
        for xt in (torch.randn(8, 4), torch.randn(32, 4)):
            self.assertEqual(f_c(m, xt), m(xt))

    def test_tracer_dynamo_cross_tracer_cache_rejected(self):
        # A cache from the make_fx tracer paired with dynamo python_code is a mismatched
        # pairing and must be rejected (the code_hash and the tracer tag both catch it),
        # not run under foreign metadata.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        dyn_code, _ = torch.compiler.precompile(
            lambda model, xx: model(xx), m, x, tracer="dynamo"
        )
        _, mf_cache = torch.compiler.precompile(lambda model, xx: model(xx), m, x)
        with self.assertRaisesRegex(PrecompileError, "tracer"):
            torch.compiler.precompile.load(dyn_code, mf_cache)

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_returned_global_constant_baked(self, backend):
        # A plain module-level constant folded straight into fn's output is referenced by
        # the transformed bytecode but is NOT a graph input, so it must be baked into the
        # artifact (else it would NameError at load). Baking a non-tensor constant is
        # consistent with the specialization contract; check both backends.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)

        def with_scale(model, xx):
            return model(xx), _GLOBAL_SCALE

        code, cache = torch.compiler.precompile(
            with_scale, m, x, tracer="dynamo", backend=backend
        )
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            out = f_c(m, x)
            self.assertEqual(out[0], m(x))
            self.assertEqual(out[1], _GLOBAL_SCALE)

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_nested_multi_tensor_output(self, backend):
        # Under dynamo the transformed bytecode (NOT the driver's OUT_SPEC path make_fx uses)
        # reassembles fn's output, so a multi-tensor / nested output exercises a distinct
        # mechanism: the subgraph returns several tensors in a fixed order the bytecode must
        # re-thread into the tuple/dict. Assert each leaf on both backends (a mis-ordered
        # flatten or a dropped dict leaf would silently corrupt the output structure).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)

        def f(model, xx):
            y = model(xx)
            return y, y * 2, {"k": y + 1}

        code, cache = torch.compiler.precompile(
            f, m, x, tracer="dynamo", backend=backend
        )
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            out = f_c(m, x)
            ref = f(m, x)
            self.assertEqual(out[0], ref[0])
            self.assertEqual(out[1], ref[1])
            self.assertEqual(out[2]["k"], ref[2]["k"])

    def test_tracer_dynamo_nontensor_output_inductor_ok(self):
        # DIVERGENCE from make_fx: Dynamo puts a non-tensor Python output (float / complex /
        # str) in the transformed bytecode, not the subgraph inductor lowers, so it round-
        # trips on tracer='dynamo' + backend='inductor' where make_fx REJECTS the same output
        # (test_nontensor_output_inductor_clean_error). Pin that so a regression to make_fx's
        # rejection under dynamo is caught. The value is folded as a default arg (baked).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        for val in (3.14, 2 + 3j, "hi"):
            code, cache = torch.compiler.precompile(
                lambda model, xx, b=val: (model(xx), b),
                m,
                x,
                tracer="dynamo",
                backend="inductor",
            )
            for _label, f_c in _default_and_inlined_loaders(code, cache, "inductor"):
                out = f_c(m, x)
                self.assertEqual(out[0], m(x))
                self.assertEqual(out[1], val)

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
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), m, x, tracer="dynamo", backend=backend
            )
            # Exercise BOTH reload paths on the DTensor artifact: load() takes the
            # bundled-artifact path (primes the cache, then execs python_code), while the
            # direct exec below runs the self-contained python_code with no cache.
            f_c = torch.compiler.precompile.load(code, cache)
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

    def test_tracer_dynamo_module_fn_folded_global(self):
        # When fn IS an nn.Module, Dynamo traces fn.forward, whose globals live in the
        # traced code's f_globals -- not fn.__globals__ (an nn.Module has none). A
        # module-level constant folded into the output must still be carried into the
        # artifact from that traced-code globals dict; otherwise the reload NameErrors.
        # Regression for resolving uncovered external_refs via gco.f_globals.
        m = _PrecompileFoldsAGlobal().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(m, x, tracer="dynamo")
        for _label, f_c in _default_and_inlined_loaders(code, cache, "inductor"):
            out = f_c(m, x)
            self.assertEqual(out[0], m(x)[0])
            self.assertEqual(out[1], _GLOBAL_SCALE)

    def test_tracer_dynamo_by_reference_callable_not_rejected(self):
        # A tensor merely ATTACHED to a by-reference object (a module-level function,
        # pickled by reference and re-imported at load) is NOT baked by value, so it must
        # NOT trigger the invariant-1 rejection. The scan keys off what pickle serializes
        # by value, not on attribute presence -- avoiding a false positive.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, xx: (model(xx), _global_helper_with_attr),
            m,
            x,
            tracer="dynamo",
            backend="eager",
        )
        out = torch.compiler.precompile.load(code, cache)(m, x)
        self.assertEqual(out[0], m(x))
        self.assertIs(out[1], _global_helper_with_attr)

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_closure_entry_is_refused(self, backend):
        # Defaults the artifact carries beside the code object; closure cells it
        # cannot. Dynamo guards a cell by IDENTITY, so a cell rebuilt at load
        # holding the same value is a different object and misses every variant.
        # Refuse at capture, where the closure is visible, rather than ship an
        # artifact that loads and then never matches.
        def make(sc, cf):
            return lambda model, xx: model(xx) * sc + cf["bias"]

        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with self.assertRaisesRegex(PrecompileError, "closes over"):
            torch.compiler.precompile(
                make(3.0, {"bias": 1.0}), m, x, tracer="dynamo", backend=backend
            )

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_defaults_roundtrip(self, backend):
        # fn with a positional default and a keyword-only default drives the driver's
        # argdefs / kwdefaults restoration; the defaults must be honored at the runtime
        # call (omitting them would TypeError / use a wrong value if they were dropped).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)

        def fn(model, xx, scale=2.0, *, bias=1.0):
            return model(xx) * scale + bias

        code, cache = torch.compiler.precompile(
            fn, m, x, tracer="dynamo", backend=backend
        )
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(f_c(m, x), fn(m, x))

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_multiple_module_args(self, backend):
        # Two separate nn.Module args exercise the len(mods)>1 interning path and the
        # unboxed->boxed subgraph bridge with graph inputs from two distinct modules; a
        # structurally identical swap confirms no weights are baked (invariant 2).
        a = torch.nn.Linear(4, 3).eval()
        b = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda ma, mb, xx: ma(xx) + mb(xx),
            a,
            b,
            x,
            tracer="dynamo",
            backend=backend,
        )
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(f_c(a, b, x), a(x) + b(x))
            a2 = torch.nn.Linear(4, 3).eval()
            b2 = torch.nn.Linear(4, 3).eval()
            self.assertEqual(f_c(a2, b2, x), a2(x) + b2(x))

    @parametrize("backend", ["inductor", "eager"])
    def test_tracer_dynamo_tied_weights_roundtrip(self, backend):
        # A module with a tied weight (two Linears sharing one weight tensor) round-trips
        # under the dynamo tracer; the shared weight must be read consistently at runtime,
        # so a mis-interned or duplicated read would surface as a wrong result here.
        m = _PrecompileTiedWeights().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx), m, x, tracer="dynamo", backend=backend
        )
        for _label, f_c in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(f_c(m, x), m(x))

    def test_tracer_dynamo_static_under_dynamic_config(self):
        # The dynamo tracer must capture STATIC shapes like the make_fx tracer (invariant 3)
        # regardless of the ambient torch._dynamo config OR per-code-object shape history:
        # precompile pins both assume_static_by_default and automatic_dynamic_shapes, so
        # neither a globally flipped default (a) nor a prior precompile of the SAME fn at
        # another shape (b, reachable with DEFAULT config) yields an out-of-contract dynamic
        # artifact. Without the pins the eager subgraph would carry a SymInt dim.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with torch._dynamo.config.patch(assume_static_by_default=False):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), m, x, tracer="dynamo", backend="eager"
            )
        self.assertNotIn("SymInt", code)
        self.assertEqual(torch.compiler.precompile.load(code, cache)(m, x), m(x))

        # (b) automatic_dynamic (on by DEFAULT): precompiling the SAME fn (one code object)
        # first at one shape then another would otherwise promote the batch dim to dynamic on
        # the second capture. Reuse one fn across two shapes; both must stay static.
        def f(model, xx):
            return model(xx)

        c1, _ = torch.compiler.precompile(
            f, m, torch.randn(5, 4), tracer="dynamo", backend="eager"
        )
        c2, _ = torch.compiler.precompile(
            f, m, torch.randn(7, 4), tracer="dynamo", backend="eager"
        )
        self.assertNotIn("SymInt", c1)
        self.assertNotIn("SymInt", c2)

    def test_load_cache_without_tracer_key(self):
        # BC: a cache produced before the dynamo tracer existed carries no "tracer" key in
        # its envelope. load() must still pair it with its make_fx python_code via the
        # blob.get("tracer", "make_fx") default rather than KeyError. Simulate a legacy
        # envelope by deleting the key. Assert the cache envelope was actually CONSUMED (no
        # "could not read the cache envelope" warning): a KeyError from a reverted fix would
        # be swallowed by load()'s except and fall back to JIT, which still returns the
        # right answer -- so an output-only assertion would not guard the .get default.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(lambda model, xx: model(xx), m, x)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        del blob["tracer"]
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            f_c = torch.compiler.precompile.load(code, buf.getvalue())
        self.assertFalse(
            any("could not read the cache envelope" in msg for msg in cm.output),
            f"legacy cache envelope was not consumed (fell back to JIT): {cm.output}",
        )
        self.assertEqual(f_c(m, x), m(x))

    def test_tracer_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(ValueError, "tracer must be 'make_fx' or 'dynamo'"):
            torch.compiler.precompile(lambda x, y: x + y, a, b, tracer="nope")

    def test_backend_default_is_inductor(self):
        # The default lowers through Inductor: the generated code inlines the Inductor
        # output module. Use a graph_partition-agnostic marker (the ``call = runner.call``
        # form is only emitted when config.graph_partition is on, which is off in fbcode).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _ = torch.compiler.precompile(lambda model, x: model(x), m, x)
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
            code, cache = torch.compiler.precompile(lambda model, xx: model(xx), m, x)
            self.assertNotIn("call = runner.call", code)  # non-partition form
            f_c = torch.compiler.precompile.load(code, cache)
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
                code, cache = torch.compiler.precompile(
                    lambda model, xx: model(xx), m, x
                )
                # No saveable artifact when caches are off; the cache is empty.
                blob = torch.load(io.BytesIO(cache), weights_only=True)
                self.assertIsNone(blob["artifact"], patch)
                # python_code still runs standalone (JITs from inlined source).
                ns = {"__name__": "_a"}
                exec(compile(code, "<a>", "exec"), ns)
                self.assertEqual(ns["forward"](m, x), m(x), patch)
                # ...and load() falls back to the inlined path.
                self.assertEqual(
                    torch.compiler.precompile.load(code, cache)(m, x), m(x), patch
                )

    def test_inductor_cpp_wrapper_pinned_off(self):
        # cpp_wrapper would make Inductor emit a C++ ``call`` (no python module); a
        # python artifact cannot come from it, so compile_to_python pins it off. With
        # cpp_wrapper=True ambient, precompile must still produce a working python artifact.
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(cpp_wrapper=True):
            code, cache = torch.compiler.precompile(lambda model, xx: model(xx), m, x)
            f_c = torch.compiler.precompile.load(code, cache)
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
            torch.compiler.precompile(boom, m, x)
        for n, p in m.named_parameters():
            self.assertIsNone(p.grad, f"{n}: example .grad must be restored on failure")

    def test_unbacked_capture_with_preexisting_grad(self):
        # Regression: in the mark_unbacked path the example params are fakeified BEFORE
        # the grad clear. A model with a pre-existing .grad (the warmup-step-then-
        # precompile flow) plus a backward in fn must still capture -- the clear must
        # precede fakeify so the fakes inherit no grad -- and the real .grad is restored.
        from torch._dynamo.decorators import mark_unbacked

        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(8, 4)
        m(x).sum().backward()  # warmup: populate .grad before precompile
        saved = {n: p.grad.clone() for n, p in m.named_parameters()}
        mark_unbacked(x, 0)
        code, _ = torch.compiler.precompile(lambda mm, t: mm(t).sum().backward(), m, x)
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 is dynamic
        for n, p in m.named_parameters():
            self.assertEqual(p.grad, saved[n])  # warmup grad restored, not clobbered

    def test_backend_eager_no_inductor_lowering(self):
        # backend="eager" skips Inductor: the generated code has no inductor ``call``
        # entry point, and instead embeds the readable captured ATen graph and the
        # eager driver. The eager backend has no kernels to accelerate, so the cache
        # is empty -- python_code is the whole artifact.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), m, x, backend="eager"
        )
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
        code, _cache = torch.compiler.precompile(
            lambda model, x: model(x), m, x, backend="eager"
        )

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

        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx).sum().backward(), m, x
        )
        # Capture must not mutate the example model's pre-existing grad (restored).
        self.assertEqual(m.weight.grad, grad_before)

        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        torch.compiler.precompile.load(code, cache)(run, x)  # run.grad starts None
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
                torch.compiler.precompile(lambda model, t, b=bad: (model(t), b), m, x)
        for extra in (7, None):
            code, cache = torch.compiler.precompile(
                lambda model, t, e=extra: (model(t), e), m, x
            )
            self.assertEqual(
                torch.compiler.precompile.load(code, cache)(m, x)[1], extra
            )
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: (model(t), 3.14), m, x, backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(m, x)[1], 3.14)

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
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, xex)
        self.assertIn("assert_size_stride", code)  # the layout guard we convert
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile.load(code, _strip_artifact(cache))(
                m, xrt
            )  # inlined path
        # A matching (same-stride) input still works on inductor.
        xmatch = torch.randn(8, 6).t()
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(m, xmatch), m(xmatch)
        )
        # The eager backend accepts the differently-strided input.
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), m, xex, backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(m, xrt), m(xrt))

    def test_input_layout_mismatch_enforced_without_size_asserts(self):
        # The layout guard must be a PROACTIVE driver check, not a reliance on inductor's
        # assert_size_stride: with size_asserts=False the assert is elided, so a naive
        # try/except would silently read wrong strides. Both load paths must still raise.
        import torch._inductor.config as ind_config

        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(8, 6).t()  # non-contiguous example, shape (6, 8)
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with ind_config.patch(size_asserts=False):
            code, cache = torch.compiler.precompile(lambda model, t: model(t), m, xex)
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                torch.compiler.precompile.load(code, _strip_artifact(cache))(
                    m, xrt
                )  # inlined

    def test_input_shape_mismatch_clean_error(self):
        # A same-structure but wrong-SHAPE input is an invariant-3 (shape) mismatch, NOT
        # an invariant-6 layout one: the driver must say "shape" / invariant 3 and not
        # misadvise a no-op .contiguous() (both inputs here are already contiguous).
        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(6, 8)  # contiguous example
        xrt = torch.randn(7, 8)  # contiguous, different shape (same pytree structure)
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, xex)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "shape"):
            torch.compiler.precompile.load(code, _strip_artifact(cache))(
                m, xrt
            )  # inlined path
        # The error must NOT mislabel a pure shape mismatch as a memory-format one.
        try:
            torch.compiler.precompile.load(code, cache)(m, xrt)
        except PrecompileError as e:
            self.assertNotIn("memory format", str(e))

    def test_size1_dim_stride_exempt_like_inductor(self):
        # A size-1 dim's stride is irrelevant (one element); inductor's assert_size_stride
        # ignores it (guards.cpp), so the proactive layout check must too -- a kept-dim
        # slice x[i:i+1] (size-1 dim with a wider stride) must RUN, not raise.
        m = torch.nn.Linear(4, 3).eval()
        xex = torch.randn(1, 4)  # contiguous, stride (4, 1)
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, xex)
        row = torch.randn(2, 8)[
            0:1, :4
        ]  # shape (1, 4), stride (8, 1): size-1 dim differs
        self.assertEqual(tuple(row.shape), (1, 4))
        self.assertNotEqual(row.stride(), xex.stride())
        self.assertEqual(torch.compiler.precompile.load(code, cache)(m, row), m(row))
        self.assertEqual(
            torch.compiler.precompile.load(code, _strip_artifact(cache))(m, row),
            m(row),
        )

    def test_empty_input_shape_is_still_checked(self):
        # The numel==0 exemption must relax ONLY the (meaningless) stride check, not the
        # shape check: an empty runtime input whose shape differs from the example must
        # still raise invariant 3, not silently return the traced-shape output.
        code, cache = torch.compiler.precompile(lambda t: t.sum(0), torch.randn(0, 4))
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, a, b: mm(a, b), m, x, y)
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
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
            torch.compiler.precompile(needs_guard, m, x)

    def test_dynamic_shapes_eager_rejected(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(
            NotImplementedError, "only supported with backend='inductor'"
        ):
            torch.compiler.precompile(lambda mm, t: mm(t), m, x, backend="eager")

    @parametrize("path", ("cached", "inlined"))
    def test_dtype_mismatch_rejected(self, path):
        # Each dense input's dtype is baked at capture (invariant 6); a runtime input of
        # a different dtype is rejected up front on BOTH the cached and inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # float32 example
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, x)
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    @parametrize("path", ("cached", "inlined"))
    def test_device_mismatch_rejected(self, path):
        # Each dense input's device is baked at capture (invariant 6); a cpu-traced
        # artifact rejects a cuda input up front on BOTH load paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, x)
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
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
            torch.compiler.precompile(lambda mm, t: mm(t), m, x)

    def test_mark_unbacked_hint_override_honored(self):
        # A mark_unbacked hint_override is a perf-only autotuning size hint (never a
        # guard), so precompile does NOT reject it; the single artifact is valid for any
        # runtime size and the hint is threaded onto the capture ShapeEnv's symbol.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, hint_override=16)
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
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
            torch.compiler.precompile(lambda mm, t: mm(t), m, x)

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
                torch.compiler.precompile(lambda mm, t: mm(t), m, x)
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
        code, cache = torch.compiler.precompile(lambda mm, a, b: mm(a) + b, m, x, y)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, a, b: mm(a) + b, m, x, y)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        self.assertIn("USER_INPUT_BOUNDS = [{0: (4, None)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "size 2.*min=4"):
            f_c(m, torch.randn(2, 4))
        xt = torch.randn(8, 4)
        self.assertEqual(f_c(m, xt), m(xt))

    def test_eager_backend_wrong_static_shape_rejected(self):
        # The eager driver now checks USER_INPUT_SHAPES too: a wrong static shape is
        # rejected (invariant 3).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(7, 4))

    def test_eager_backend_dtype_mismatch_rejected(self):
        # The eager driver checks USER_INPUT_DTYPES too: a dtype mismatch is rejected
        # (invariant 6).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    def test_cache_integrity_tampered_backend_rejected(self):
        # The cache envelope's backend tag is an integrity check: a tampered backend
        # (here flipped to a value that does not match python_code's BACKEND) makes
        # load() raise a clear PrecompileError rather than reconstruct a foreign cache.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, x)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["backend"] = "eager"  # python_code says inductor
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "backend"):
            torch.compiler.precompile.load(code, buf.getvalue())

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
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, x)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        # Tamper either the format string or bump the version to a foreign value.
        blob[tag] = "not-a-precompile-cache" if tag == "format" else 999
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            f_c = torch.compiler.precompile.load(code, buf.getvalue())  # must not raise
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
            torch.compiler.precompile.load("x = 1\n", buf.getvalue())

    def test_nonliteral_calling_convention_metadata_rejected(self):
        code, cache = torch.compiler.precompile(
            lambda x: x.sin(), torch.randn(2), backend="eager"
        )
        bad_code = code.replace("BACKEND = 'eager'", "BACKEND = object()", 1)
        with self.assertRaisesRegex(
            PrecompileError, "BACKEND.*calling-convention metadata"
        ):
            torch.compiler.precompile.load(bad_code, cache)

    def test_singleton_pickle_deepcopy_roundtrip(self):
        # torch.compiler.precompile is a process-wide singleton; pickle and deepcopy
        # must round-trip to the SAME object (it carries no per-call state), and its
        # repr is the stable public name.
        p = torch.compiler.precompile
        self.assertIs(pickle.loads(pickle.dumps(p)), p)
        self.assertIs(copy.deepcopy(p), p)
        self.assertEqual(repr(p), "torch.compiler.precompile")

    def test_standalone_runtime_artifact_execs_in_fresh_process(self):
        # A generated artifact that imports a standalone_runtime helper (here output-
        # aliasing, which emits ``from ...standalone_runtime import gen_alias_from_base``)
        # must EXEC in a FRESH process whose only prior import is ``torch`` -- a
        # regression for the runtime_wrappers <-> _dynamo circular import that a cold
        # exec used to hit. We write python_code to a temp file and exec it in a
        # subprocess that imports only torch, then runs forward().
        x = torch.randn(3, 4)
        code, _cache = torch.compiler.precompile(lambda a: a.t(), x)
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
        codeA, cacheA = torch.compiler.precompile(lambda mm, t: mm(t) * 2, m, x)
        codeB, cacheB = torch.compiler.precompile(lambda mm, t: mm(t) + 100, m, x)
        self.assertNotEqual(codeA, codeB)
        with self.assertRaisesRegex(PrecompileError, "code_hash|does not match"):
            torch.compiler.precompile.load(codeA, cacheB)
        f_a = torch.compiler.precompile.load(codeA, cacheA)
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
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
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
        f = torch.compiler.precompile.load(new_code, buf.getvalue())
        with self.assertRaisesRegex(AssertionError, "my custom user assertion"):
            f(m, x)
        # The original assertion must NOT be relabeled as a layout error.
        try:
            f(m, x)
        except AssertionError as e:
            self.assertNotIn("shape or memory format", str(e))

    def test_public_identity_module_and_qualname(self):
        # PrecompileError and load are public under torch.compiler.precompile, so their
        # __module__ / __qualname__ must report that public location (so Sphinx and
        # introspection anchor them under torch.compiler, not the private module).
        err = torch.compiler.precompile.PrecompileError
        self.assertEqual(err.__module__, "torch.compiler")
        self.assertEqual(err.__qualname__, "precompile.PrecompileError")
        self.assertEqual(torch.compiler.precompile.load.__module__, "torch.compiler")
        self.assertEqual(torch.compiler.precompile.load.__qualname__, "precompile.load")

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
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), m, x, backend=backend
        )
        self.assertIn("BUFFER_NAMES = ['buf']", code)
        renamed = WithBuf("buf2").eval()  # same params, buffer renamed (same shape)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(renamed, x)

    def test_example_input_inplace_mutation_not_restored(self):
        # Capture EXECUTES fn once on the example inputs (invariant 3), so an in-place
        # mutation fn performs on its example user input happens at capture time and is
        # NOT restored -- only .grad is snapshotted/restored. Pin this surprising contract
        # so it stays covered: the example tensor reflects the mutation afterward.
        scratch = torch.zeros(4)
        torch.compiler.precompile(lambda a: a.add_(1.0), scratch)
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
            code, cache = torch.compiler.precompile(
                lambda mm, t: mm(t), m, x, backend="eager"
            )
        else:
            code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
            if path == "inlined":
                cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    def test_eager_device_mismatch_rejected(self):
        # The eager driver bakes each input's device (invariant 6): a cpu-traced eager
        # artifact rejects a cuda input up front, like the inductor backend.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), m, x, backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(
            lambda model, h: model(h.a + h.b), m, inp
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        self.assertIn("USER_INPUT_BOUNDS = [{0: (None, 16)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
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
            code, cache = torch.compiler.precompile(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True), x
            )
            f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend=backend
        )
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
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend=backend
        )
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
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend=backend
        )
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
        code, cache = torch.compiler.precompile(lambda model, t: model(t), m, x)

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
            yield "cached", torch.compiler.precompile.load(code, cache)
            yield (
                "inlined",
                torch.compiler.precompile.load(code, _strip_artifact(cache)),
            )

        for label, f_c in loaders():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, r"memory format.*PARAMETER/BUFFER.*layout"
                ):
                    f_c(with_noncontig_weight(), x)
        # The eager backend accepts the same non-contiguous weight (layout-flexible).
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend="eager"
        )
        run = with_noncontig_weight()
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(run, x), run(x))

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
        code_s, cache_s = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, m, xs, ys
        )
        f_s = torch.compiler.precompile.load(code_s, cache_s)
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
        code_i, cache_i = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, m, xi, yi
        )
        f_i = torch.compiler.precompile.load(code_i, cache_i)
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
        torch.compiler.precompile(lambda mm, t: mm(t).sum().backward(), m, x)
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
            torch.compiler.precompile(lambda x: x + captured, torch.randn(3))

    def test_single_trust_warning_on_inlined_load(self):
        # On the inlined load path (an eager artifact has an empty cache, so there is
        # nothing to prime and load() just EXECs python_code) the untrusted-input / EXEC
        # warning must fire EXACTLY ONCE -- only _make_inlined_forward warns. Asserting
        # "exactly once" guards against the EXEC warning being duplicated on this load.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), m, x, backend="eager"
        )
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), m, t
        )
        self.assertIn("PARAM_NAMES = ['l1.weight']", code)  # tie collapsed to one

        ref = copy.deepcopy(m)  # deepcopy preserves the tie within the object graph
        ref(t).sum().backward()

        torch.compiler.precompile.load(code, cache)(m, t)  # one call: tied grad
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
        code, cache = torch.compiler.precompile(lambda a, b, t: b(a(t)), m1, m2, t)
        self.assertIn("MODULE_POSITIONS = [0, 1]", code)
        self.assertIn("m0.weight", code)  # first module's params prefixed m0.*
        self.assertIn("m1.weight", code)  # second module's params prefixed m1.*
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), m, t
        )

        ref = copy.deepcopy(m)
        ref(t).sum().backward()

        torch.compiler.precompile.load(code, cache)(m, t)
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
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t).sum().backward(), m, x
        )
        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        for p in run.parameters():
            p.requires_grad_(False)  # flip OFF at runtime -- must be a no-op
        torch.compiler.precompile.load(code, cache)(run, x)
        self.assertIsNotNone(run.weight.grad)  # still scattered despite the flip
        ref = torch.nn.Linear(4, 3)
        ref.load_state_dict(m.state_dict())
        ref(x).sum().backward()
        self.assertEqual(run.weight.grad, ref.weight.grad)


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
        code, cache = torch.compiler.precompile(f, a, b)
        self.assertIsInstance(code, str)
        self.assertIsInstance(cache, bytes)

        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_multiple_module_args(self, device):
        # More than one nn.Module arg: each module's params are lifted with
        # m{i}.-prefixed names. Both modules are passed again at runtime.
        a = torch.nn.Linear(4, 4).to(device).eval()
        b = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((2, 4), device=device, dtype=torch.float32)
        ref = b(torch.relu(a(x)))

        code, cache = torch.compiler.precompile(
            lambda ma, mb, x: mb(torch.relu(ma(x))), a, b, x
        )
        self.assertIn(
            "PARAM_NAMES = ['m0.weight', 'm0.bias', 'm1.weight', 'm1.bias']", code
        )

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(a, b, x), ref)

    def test_inplace_on_intermediate_is_allowed(self, device):
        # In-place ops on intermediates (e.g. nn.ReLU(inplace=True)) are fine -- they
        # do not touch any input -- and must NOT be rejected as input mutation.
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(inplace=True))
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(train_step, model, x, target)
        f_c = torch.compiler.precompile.load(code, cache)

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

        code, cache = torch.compiler.precompile(train_step, model, x, target)
        f_c = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(train_step, a, b, x, target)
        f_c = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        # The tied weight is lifted once (single name), so it is one graph input.
        self.assertIn("PARAM_NAMES = ['a.weight']", code)

        # Training scatters a single grad onto the shared weight, matching eager's
        # accumulation into the tied parameter.
        ref = copy.deepcopy(m)
        ref(x).sum().backward()
        ref_grad = ref.a.weight.grad

        code, cache = torch.compiler.precompile(
            lambda model, x: model(x).sum().backward(), m, x
        )
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(f, a, b, backend="eager")
        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_backend_eager_module(self, device):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), m, x, backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(
            train_step, model, x, target, backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(
            lambda m, xx: m(xx), fresh(), x, backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        self.assertEqual(f_c(run, x), ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_backend_eager_inf_constant(self, device):
        # masked_fill to -inf bakes a bare ``inf`` token into gm.code (another fx
        # custom builtin); the eager standalone source must provide it.
        def f(x):
            return torch.relu(x).masked_fill(x < 0, float("-inf"))

        x = make_tensor((8,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(f, x, backend="eager")
        f_c = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(train_step, fresh(), x, target)
        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        f_c(run, x, target)
        for p, rg in zip(run.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_output_alias_supported(self, device):
        # An output that is a view of an input goes through AOTAutograd's output-
        # alias epilogue; precompile reproduces it.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(lambda a: a.t(), x)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(x), x.t())

    def test_input_mutation_supported(self, device):
        # In-place input mutation is reflected on the passed tensor (and matches
        # eager), via AOTAutograd's mutation handling composed into the artifact.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(lambda a: a.add_(1.0), scratch)
        f_c = torch.compiler.precompile.load(code, cache)
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
            code, cache = torch.compiler.precompile(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True), x
            )
            f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda model, xx: model(xx), fresh(), x)

        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()
        ref_rv = ref[1].running_var.clone()
        ref_nbt = ref[1].num_batches_tracked.clone()

        f_c = torch.compiler.precompile.load(code, cache)
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

        code, cache = torch.compiler.precompile(fn, t, t)
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 dynamic
        f_c = torch.compiler.precompile.load(code, cache)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = None
        buf = io.BytesIO()
        torch.save(blob, buf)
        f_i = torch.compiler.precompile.load(code, buf.getvalue())
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
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), m, x
        )
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, a, b: mm(a) + b, m, x, y)
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (8, 16, 2):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt), m(xt))

    def test_unbacked_zero_batch_runs(self, device):
        # bs=0 on an unbacked dynamic dim is a valid runtime size (the symbol is >= 0);
        # the artifact runs on an empty batch and matches eager.
        m = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(lambda mm, t: mm(t), m, x)
        f_c = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(lambda t: torch.relu(t) * 2.0, x)
        f_c = torch.compiler.precompile.load(code, cache)
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
            torch.compiler.precompile(lambda t: t.contiguous() * 2.0, x)

    def test_eager_backend_input_mutation(self, device):
        # The eager backend replays the raw ATen graph, so input mutation is reflected on
        # the passed tensor and matches eager, like the inductor backend.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda a: a.add_(1.0), scratch, backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    def test_eager_backend_output_alias(self, device):
        # The eager backend reproduces an output that aliases an input (a view), matching
        # eager, via the raw ATen replay.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(lambda a: a.t(), x, backend="eager")
        f_c = torch.compiler.precompile.load(code, cache)
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
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), m, x, tracer="dynamo", backend=backend
            )
            f_c = torch.compiler.precompile.load(code, cache)
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
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), m, x, tracer="dynamo", backend=backend
            )
            f_c = torch.compiler.precompile.load(code, cache)
            for bs in (8, 16, 1):
                xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
                self.assertEqual(f_c(m, xt), m(xt))

    # make_fx only. That artifact checks NO guards, so ambient autocast in the
    # serving process must not reach it -- the driver pins the state the capture
    # recorded, keyed off the GRAPH's devices (GRAPH_DEVICES), not the runtime
    # tensors'. The dynamo tracer mirrors torch.compile instead: autocast is part
    # of the guarded global state, so a serving process whose autocast differs
    # from capture misses rather than being silently pinned.
    @parametrize("backend", ("eager", "inductor"))
    def test_artifact_reproduces_capture_time_autocast(self, device, backend):
        tracer = "make_fx"

        def fn(model, xx):
            return model(xx)

        device_type = torch.device(device).type
        model = torch.nn.Linear(8, 8).to(device).eval()
        x = make_tensor((4, 8), device=device, dtype=torch.float32)

        with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16):
            hot_code, hot_cache = torch.compiler.precompile(
                fn, model, x, backend=backend, tracer=tracer
            )
        with torch.no_grad():
            cold_code, cold_cache = torch.compiler.precompile(
                fn, model, x, backend=backend, tracer=tracer
            )

        for code, cache, captured_under_autocast in (
            (hot_code, hot_cache, True),
            (cold_code, cold_cache, False),
        ):
            loaded = torch.compiler.precompile.load(code, cache)
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
            code, cache = torch.compiler.precompile(fn, model, x, backend="eager")
        devices = _graph_devices_literal(code)
        self.assertIn(f"'{device_type}'", devices)
        self.assertIn("'cpu'", devices)

        loaded = torch.compiler.precompile.load(code, cache)
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
        code, cache = torch.compiler.precompile(
            inf_fn, m, x, tracer="dynamo", backend="eager"
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(m, x), inf_fn(m, x)
        )

        def fresh_bn():
            torch.manual_seed(0)
            bn = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            bn.train()
            return bn.to(device)

        xb = make_tensor((8, 4), device=device, dtype=torch.float32)
        ref_out = fresh_bn()(xb)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx),
            fresh_bn(),
            xb,
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(fresh_bn(), xb), ref_out
        )


instantiate_device_type_tests(TestPrecompileNumerics, globals())


if __name__ == "__main__":
    run_tests()
