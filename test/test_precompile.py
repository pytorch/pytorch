# Owner(s): ["oncall: pt2"]
import copy
import errno
import functools
import hashlib
import io
import os
import pickle
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
import warnings
from unittest import mock

import torch
import torch.utils._pytree as _pytree
from torch._dynamo.decorators import mark_dynamic, mark_unbacked
from torch._precompile import (
    _write_artifact,
    capture,
    load,
    MakeFxTracer,
    PrecompiledRunnable,
    PrecompileError,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
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


def _precompile_pair(fn, *args, **kwargs):
    """Public-API entry point for the tests added with the fake-tensor capture, behind one
    indirection so the tracer/module switch above this commit re-points it in one place."""
    return torch.compiler.precompile(fn, *args, **kwargs)


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
    def test_guard_fact_render(self):
        from torch.compiler._precompile_types import GuardFact

        kept = GuardFact("TYPE_MATCH", "L['x']", ("check_type_id(L['x'])",), "", True)
        self.assertEqual(kept.render(), "[enforced] check_type_id(L['x']) on L['x']")
        # No rendered code falls back to <guard_type>, a value is appended, and
        # the dropped label pads to the width of "enforced" so lines align.
        dropped = GuardFact("ID_MATCH", "G['fn']", (), "is @m.py:3#abc fn", False)
        self.assertEqual(
            dropped.render(), "[dropped ] <ID_MATCH> is @m.py:3#abc fn on G['fn']"
        )
        # Several code parts are joined; no source drops the " on ..." suffix.
        joined = GuardFact("GRAD_MODE", "", ("a", "b"), "", True)
        self.assertEqual(joined.render(), "[enforced] a ; b")

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
        code, cache = torch.compiler.precompile(lambda model, x: model(x), m, x)
        self.assertIsInstance(cache, bytes)

        with fresh_cache():
            counters.clear()
            f_c = torch.compiler.precompile.load(code, cache)
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
        # format/version/backend tag plus a code_hash binding the cache to its python_code).
        self.assertEqual(
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
        )
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "inductor")
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
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
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

        # Branches returning differing INT values merge to a symint instead, which asserts
        # a ShapeEnv the static capture does not have and so never reaches the get_attr
        # check above. Same refusal, not a raw AssertionError.
        def g(x):
            return torch.cond(x.sum() > 0, lambda t: 1, lambda t: 2, (x,))

        with self.assertRaisesRegex(PrecompileError, "control-flow subgraph") as cm:
            _precompile_pair(g, torch.randn(4))
        self.assertIsInstance(cm.exception.__cause__, AssertionError)

    @parametrize("carry", ("tensor", "int"))
    def test_while_loop_rejected(self, carry):
        # torch.while_loop is the other HOP that refusal names, and neither spelling reaches
        # the post-trace get_attr check: both want the ShapeEnv a static capture lacks. A
        # TENSOR carry dies in the fake kernel's ignore_fresh_unbacked_symbols()
        # (AttributeError); an INT carry (while_loop's own docstring spelling) dies earlier,
        # in the proxy path that unspecializes it (AssertionError). Both must come back as
        # the control-flow PrecompileError, not leak the internal error.
        def f(x):
            def cond_fn(i, v):
                return i < 3

            def body_fn(i, v):
                return i + 1, v + 1

            init = torch.tensor(0) if carry == "tensor" else 0
            return torch.while_loop(cond_fn, body_fn, (init, x))

        with self.assertRaisesRegex(PrecompileError, "control-flow subgraph") as cm:
            _precompile_pair(f, torch.randn(4))
        # The message is byte-identical to the post-trace get_attr refusal's, so pin the
        # cause too: only these clauses chain one of these types, and without them a
        # while_loop that stopped needing a ShapeEnv would leave the test green, relabel dead.
        expected = AttributeError if carry == "tensor" else AssertionError
        self.assertIsInstance(cm.exception.__cause__, expected)

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
        # The public location: test_public_bindings.test_correct_module_names also
        # enforces this for every torch.compiler.__all__ member.
        self.assertEqual(torch.compiler.precompile.__module__, "torch.compiler")

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

    def test_tracer_dynamo_not_implemented(self):
        # "dynamo" is a valid (planned) tracer value but is not implemented yet; it must
        # raise NotImplementedError, not silently fall back to make_fx.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with self.assertRaisesRegex(NotImplementedError, "tracer='dynamo'"):
            torch.compiler.precompile(
                lambda model, xx: model(xx), m, x, tracer="dynamo"
            )

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
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
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

        with self.assertRaisesRegex(PrecompileError, "guard on a value this capture"):
            _precompile_pair(needs_guard, m, x)

    def test_dynamic_shapes_unbacked_item_captured(self):
        # An unbacked capture is the only path with a ShapeEnv, so where a static capture
        # refuses .item() outright it holds the value as an unbacked symbol: a use that
        # never guards on it captures, and the loaded artifact matches eager. Replay on a
        # DIFFERENT input (fresh values, a different marked size) so a baked .item() value
        # or a specialized batch dim fails here rather than passing on the capture input.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)

        def scale_by_item(mm, t):
            return mm(t) * t.sum().item()

        code, cache = _precompile_pair(scale_by_item, m, x)
        f_c = torch.compiler.precompile.load(code, cache)
        other = torch.randn(16, 4)
        self.assertEqual(f_c(m, other), scale_by_item(m, other))

    def test_dynamic_shapes_unbacked_item_guard_rejected(self):
        # The other half of the same contract: a branch on the .item() value must guard
        # on that unbacked symbol, so capture fails LOUDLY instead of baking the value
        # the example run happened to produce.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)

        def branches_on_item(mm, t):
            return mm(t) if t.sum().item() > 0 else mm(t) + 1

        with self.assertRaisesRegex(
            PrecompileError, "guard on a value this capture"
        ) as cm:
            _precompile_pair(branches_on_item, m, x)
        # The refusal must name the VALUE symbol (an unbacked float, zuf0 > 0.0), not the
        # marked dim (u0 > 4) -- the marked-dim case produces the same top line verbatim.
        self.assertRegex(str(cm.exception), r"Underlying:.*zuf\d+ > 0\.0")

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

    def test_example_input_is_not_mutated_by_capture(self):
        # Capture traces fn on FAKE tensors (invariant 3), so an in-place mutation fn
        # performs on its example user input never reaches the caller's tensor; the
        # served artifact is what mutates a real input, exactly once per call.
        scratch = torch.zeros(4)
        python_code, cache = _precompile_pair(lambda a: a.add_(1.0), scratch)
        self.assertEqual(scratch, torch.zeros(4))
        torch.compiler.precompile.load(python_code, cache)(scratch)
        self.assertEqual(scratch, torch.ones(4))

        # The carve-out: only the tensors capture FAKEIFIES are protected. A real tensor
        # fn merely CLOSES OVER stays real, so an in-place op on it does execute during
        # capture -- and the capture then fails anyway, because a closed-over tensor is a
        # baked constant (invariant 1). A failed capture leaving the caller's tensor
        # mutated is surprising enough to pin.
        closed_over = torch.zeros(4)
        with self.assertRaisesRegex(PrecompileError, "neither a graph input"):
            _precompile_pair(lambda a: a + closed_over.add_(1.0), torch.zeros(4))
        self.assertEqual(closed_over, torch.ones(4))

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

    def test_static_capture_rejects_data_dependent_ops(self):
        # A static make_fx capture traces on fake tensors, so a value the trace
        # cannot know is refused rather than baked from the example. The distinct
        # fake-tensor failure paths, each through the public entry point: .item()
        # (DataDependentOutputException), .nonzero() (DynamicOutputShapeException),
        # and an op with no meta/fake kernel (UnsupportedOperatorException for a
        # library op, a RuntimeError naming the missing fake impl for a custom_op).
        from torch._subclasses.fake_tensor import (
            DataDependentOutputException,
            DynamicOutputShapeException,
        )
        from torch.library import _scoped_library

        model = torch.nn.Linear(4, 4)

        def items(m, x):
            return m(x) * x.sum().item()

        def nonzero(m, x):
            return m(x)[x[:, 0].nonzero().flatten()]

        # The two fake-tensor classes are asserted per case: both raises produce a
        # byte-identical message, so the regex alone cannot tell them apart and a swap
        # (or a future change that degenerates .nonzero() to the value class) would pass.
        for fn, cause in (
            (items, DataDependentOutputException),
            (nonzero, DynamicOutputShapeException),
        ):
            with self.subTest(fn=fn.__name__):
                # Both asserts stay inside the subTest: chained after it, a regressed
                # refusal would be swallowed by subTest and then re-reported as an
                # AttributeError on cm.exception, aborting the loop before the next case.
                with self.assertRaisesRegex(PrecompileError, "data-dependent op") as cm:
                    _precompile_pair(fn, model, torch.randn(3, 4), backend="eager")
                self.assertIsInstance(cm.exception.__cause__, cause)

        # Both op registrations are global, so undo them: the scoped library takes
        # its own op with it, and the custom_op's library is destroyed in finally.
        with _scoped_library("mlprecompile", "FRAGMENT") as lib:
            lib.define("no_meta(Tensor x) -> Tensor")
            lib.impl("no_meta", lambda x: x * 2, "CPU")

            @torch.library.custom_op("mlprecompile::no_fake_impl", mutates_args=())
            def no_fake_impl(x: torch.Tensor) -> torch.Tensor:
                return x * 2

            try:
                for op in (torch.ops.mlprecompile.no_meta, no_fake_impl):
                    with (
                        self.subTest(op=str(op)),
                        self.assertRaisesRegex(PrecompileError, "no meta/fake kernel"),
                    ):
                        _precompile_pair(
                            lambda m, x, op=op: op(m(x)),
                            model,
                            torch.randn(3, 4),
                            backend="eager",
                        )
            finally:
                no_fake_impl._lib._destroy()

        # The ops above live in a namespace FakeTensorMode's unsafe fallback does not
        # allow, so they are refused even with the fallback on. In an ALLOWLISTED
        # namespace (aten, prims, quantized, ...) the fallback would instead run the
        # real kernel on ZERO-FILLED substitutes and bake whatever shape that produced;
        # the capture mode passes allow_fallback_kernels=False so this is refused too.
        with _scoped_library("quantized", "FRAGMENT") as qlib:
            qlib.define("mlprecompile_no_meta(Tensor x) -> Tensor")
            qlib.impl("mlprecompile_no_meta", lambda x: x * 2, "CPU")
            with self.assertRaisesRegex(PrecompileError, "no meta/fake kernel"):
                _precompile_pair(
                    lambda m, x: torch.ops.quantized.mlprecompile_no_meta(m(x)),
                    model,
                    torch.randn(3, 4),
                    backend="eager",
                )

    def test_capture_refuses_a_data_ptr_read(self):
        # Capture traces on fake tensors, which have no real memory behind them, so a
        # .data_ptr() read in fn (or in a kernel that dereferences one) could only
        # return a meaningless value: the capture fake mode is built with
        # fake_tensor_allow_unsafe_data_ptr_access off, which turns the read into a
        # refusal at capture time.
        model = torch.nn.Linear(4, 4)

        def reads_pointer(m, x):
            x.data_ptr()
            return m(x)

        with self.assertRaisesRegex(PrecompileError, "data pointer"):
            _precompile_pair(reads_pointer, model, torch.randn(3, 4), backend="eager")

        # A kernel that dereferences a fake tensor hits the TYPED data-pointer check
        # instead, whose message is about uninitialized storage rather than FakeTensor;
        # both are refused (tensor_split reads its index tensor's values).
        def splits_on_tensor_indices(m, x):
            a, b = torch.tensor_split(x, torch.tensor([1]))
            return m(x) + a.sum() + b.sum()

        with self.assertRaisesRegex(PrecompileError, "data pointer"):
            _precompile_pair(
                splits_on_tensor_indices, model, torch.randn(3, 4), backend="eager"
            )

        # Conversely, a REAL tensor with no storage raises "Cannot access data pointer of
        # Tensor that doesn't have storage" from the same c10 code. It has data (it is
        # sparse, not fake), so the relabel must not claim otherwise: the refusal matches
        # the fake-specific texts only, and this failure reaches the caller as it is.
        sparse = torch.randn(3, 3).to_sparse()

        def reads_a_real_sparse_pointer(m, x):
            sparse.data_ptr()
            return m(x)

        with self.assertRaises(RuntimeError) as cm:
            _precompile_pair(
                reads_a_real_sparse_pointer, model, torch.randn(3, 4), backend="eager"
            )
        self.assertNotIsInstance(cm.exception, PrecompileError)
        self.assertIn("doesn't have storage", str(cm.exception))

    def test_capture_refuses_a_numpy_conversion(self):
        # A NumPy conversion reads the traced tensor's data exactly as .data_ptr() does,
        # but tensor_numpy.cpp rejects it earlier and blames "tensor subclasses", which
        # under capture is usually capture's own FakeTensor and not one the caller wrote --
        # so that text is matched too and relabeled. This is everyday logging/metric code
        # (loss.detach().cpu().numpy()) inside a forward, and unrelabeled it sent the user
        # after a subclass that does not exist. np.asarray(t) funnels through __array__, so
        # calling that directly covers it and keeps the test independent of numpy being
        # importable. All three cases hit the same production substring, so they pin the
        # relabel for the three spellings a user writes, not three distinct branches.
        model = torch.nn.Linear(4, 4)
        for name, read in (
            ("numpy", lambda t: t.numpy()),
            ("numpy_force", lambda t: t.numpy(force=True)),
            ("dunder_array", lambda t: t.__array__()),
        ):

            def logs_through_numpy(m, x, read=read):
                read(x.detach())
                return m(x)

            with (
                self.subTest(read=name),
                self.assertRaisesRegex(PrecompileError, r"reads a tensor's data"),
            ):
                _precompile_pair(
                    logs_through_numpy, model, torch.randn(3, 4), backend="eager"
                )

    def test_unfakeifiable_input_refused_without_clobbering_grad(self):
        # Fakeification runs INSIDE the .grad save/restore window, so an example input
        # the meta converter cannot represent (a quantized tensor) is refused with a
        # PrecompileError naming it, and the caller's example .grad is put back -- the
        # same object, not a copy.
        model = torch.nn.Linear(3, 3)
        grad = torch.ones_like(model.weight)
        model.weight.grad = grad
        q = torch.quantize_per_tensor(torch.randn(3, 3), 0.1, 0, torch.qint8)
        with self.assertRaisesRegex(
            PrecompileError, "user input 0 cannot be represented as a fake tensor"
        ):
            _precompile_pair(lambda m, t: m(t.dequantize()), model, q, backend="eager")
        self.assertIs(model.weight.grad, grad)

    def test_nested_input_refused(self):
        # A nested example input is refused up front with a named PrecompileError rather
        # than the raw internal assertion fakeifying one raises (a static capture has no
        # ShapeEnv to mint the jagged ragged dim's symbolic nested int). The refusal runs
        # ahead of the recorded-shape reads, so the strided layout (whose .shape read
        # raises inside NestedTensorImpl) takes the same path. It is capture-WIDE, so the
        # unbacked path -- whose ShapeEnv could fakeify a jagged input, but which has no
        # nested representation downstream either -- gets the same refusal, and its
        # message claims a restriction rather than that the tensor is unfakeifiable.
        model = torch.nn.Linear(3, 3)
        parts = [torch.randn(2, 3), torch.randn(4, 3)]
        for layout in (torch.jagged, torch.strided):
            nt = torch.nested.nested_tensor(parts, layout=layout)
            with (
                self.subTest(layout=layout),
                self.assertRaisesRegex(
                    PrecompileError, "user input 0 is a nested tensor"
                ),
            ):
                _precompile_pair(lambda m, t: m(t), model, nt, backend="eager")
        # Unbacked capture is inductor-only, so this case takes the default backend; the
        # marked input is the dense one, the nested one is refused before any tracing.
        x = torch.randn(4, 3)
        mark_unbacked(x, 0)
        nt = torch.nested.nested_tensor(parts, layout=torch.jagged)
        with self.assertRaisesRegex(PrecompileError, "user input 1 is a nested tensor"):
            _precompile_pair(lambda m, t, u: m(t), model, x, nt)

        # A nested BUFFER is refused by name too, which is what pins the refusal ahead of
        # the recorded param/buffer shape reads: those read t.shape, so a STRIDED nested
        # buffer would otherwise escape as the raw NestedTensorImpl error.
        class HasNestedBuffer(torch.nn.Module):
            def __init__(self, nt):
                super().__init__()
                self.lin = torch.nn.Linear(3, 3)
                self.register_buffer("nt", nt)

            def forward(self, t):
                return self.lin(t)

        for layout in (torch.jagged, torch.strided):
            mod = HasNestedBuffer(torch.nested.nested_tensor(parts, layout=layout))
            with (
                self.subTest(buffer_layout=layout),
                self.assertRaisesRegex(PrecompileError, "buffer nt is a nested tensor"),
            ):
                _precompile_pair(
                    lambda m, t: m(t), mod, torch.randn(2, 3), backend="eager"
                )

        # ... and a nested PARAMETER of either layout, the other half of input_labels'
        # model side: param_shapes reads t.shape just as buffer_shapes does, so the
        # strided one is the case that escaped before the refusal moved ahead of it.
        class HasNestedParam(torch.nn.Module):
            def __init__(self, nt):
                super().__init__()
                self.p = torch.nn.Parameter(nt)

            def forward(self, t):
                return t

        for layout in (torch.jagged, torch.strided):
            mod = HasNestedParam(torch.nested.nested_tensor(parts, layout=layout))
            with (
                self.subTest(param_layout=layout),
                self.assertRaisesRegex(
                    PrecompileError, "parameter p is a nested tensor"
                ),
            ):
                _precompile_pair(
                    lambda m, t: m(t), mod, torch.randn(2, 3), backend="eager"
                )

    def test_inputs_whose_metadata_a_fake_drops_are_refused(self):
        # from_tensor ACCEPTS these three and silently drops metadata the trace then reads
        # at Python level and bakes -- an mkldnn tensor comes back strided (so
        # to_dense()/to_mkldnn() record no node), a sparse one comes back with 0 nnz (so
        # .values() is annotated empty and a Python nnz read bakes 0), a pinned one comes
        # back unpinned (so is_pinned() reads False and a branch on it bakes the unpinned
        # side) -- each of which a real-tensor trace baked correctly. So they are refused
        # by name in the same capture-wide loop as a nested input, which also puts the
        # refusal ahead of every recorded shape read. The pinned case is a separate test
        # below, since constructing a pinned tensor needs an accelerator.
        model = torch.nn.Linear(4, 4)
        x = torch.randn(3, 4)
        cases = [
            # Each regex carries the layout AND its diagnosis, so one branch cannot
            # stand in for another (every one of them opens "has <layout> layout").
            (x.to_sparse(), r"user input 0 has torch.sparse_coo layout.*reports 0 nnz"),
            (x.to_sparse_csr(), r"user input 0 has torch.sparse_csr layout.*0 nnz"),
        ]
        # mkldnn tensors cannot be built in a build without MKL-DNN (macOS CI).
        mkldnn = torch.backends.mkldnn.is_available()
        if mkldnn:
            cases.append(
                (
                    x.to_mkldnn(),
                    r"user input 0 has torch._mkldnn layout.*comes back STRIDED",
                )
            )
        for t, message in cases:
            with (
                self.subTest(case=message),
                self.assertRaisesRegex(PrecompileError, message),
            ):
                _precompile_pair(lambda m, u: m(u), model, t, backend="eager")

        # The model half is named the same way: torch.utils.mkldnn.to_mkldnn registers the
        # converted weight as a BUFFER, and before this refusal that capture died inside
        # the TorchScript interpreter ("itensor_view_from_dense expects CPU tensor input").
        if mkldnn:
            from torch.utils.mkldnn import to_mkldnn

            with self.assertRaisesRegex(
                PrecompileError,
                r"buffer weight has torch._mkldnn layout.*comes back STRIDED",
            ):
                _precompile_pair(
                    lambda m, t: m(t.to_mkldnn()).to_dense(),
                    to_mkldnn(torch.nn.Linear(4, 4)),
                    x,
                    backend="eager",
                )

        # The pinned member of that table needs an accelerator to CONSTRUCT, so its refusal
        # -- the one whose deletion ships a wrong artifact rather than an error -- goes
        # unexercised on CPU. The probe DISPATCHES, so answering True reaches it anyway.
        with (
            mock.patch.object(torch.Tensor, "is_pinned", return_value=True),
            self.assertRaisesRegex(PrecompileError, "user input 0 is in pinned memory"),
        ):
            _precompile_pair(lambda t: t.sum(), x, backend="eager")

        # The sibling idiom -- fn PINNING a tensor instead of being handed a pinned one --
        # is refused as a missing fake kernel: FakeTensorMode declines aten._pin_memory
        # with a bare AssertionError("NYI: <op>"), which is relabeled rather than escaping
        # raw. Needs no accelerator: the fake dispatch declines before any allocator call.
        with self.assertRaisesRegex(PrecompileError, "no meta/fake kernel"):
            _precompile_pair(lambda m, t: m(t.pin_memory()), model, x, backend="eager")

        # Capture-WIDE: the unbacked path refuses the same input, before it fakeifies
        # anything. Unbacked capture is inductor-only, so no backend override.
        marked = torch.randn(4, 4)
        mark_unbacked(marked, 0)
        with self.assertRaisesRegex(
            PrecompileError, r"user input 1 has torch.sparse_coo layout.*reports 0 nnz"
        ):
            _precompile_pair(lambda m, t, u: m(t), model, marked, x.to_sparse())

    def test_wrapper_subclass_over_sparse_data_refused(self):
        # Those three metadata reads see the OUTER tensor, and a traceable wrapper subclass
        # reports strided, non-nested, non-mkldnn whatever it wraps -- so before the loop
        # unwrapped one, a wrapper over SPARSE data passed every clause and then had the
        # inner nnz dropped by the fake conversion: capture SUCCEEDED and baked "+ 0.0"
        # where eager adds the real nnz (the parent's real-tensor trace baked it correctly),
        # the one wrong-artifact hole the loop exists to close. It recurses through
        # __tensor_flatten__ instead and names the inner tensor. The second half pins that
        # unwrapping refuses only what a fake gets wrong: a wrapper over DENSE data still
        # captures, as MaskedTensor and DTensor do.
        class Wrapper(torch.Tensor):
            @staticmethod
            def __new__(cls, inner):
                return torch.Tensor._make_wrapper_subclass(
                    cls, inner.shape, dtype=inner.dtype, device=inner.device
                )

            def __init__(self, inner):
                self.inner = inner

            def __tensor_flatten__(self):
                return ["inner"], None

            @staticmethod
            def __tensor_unflatten__(inner, ctx, outer_size, outer_stride):
                return Wrapper(inner["inner"])

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                def unwrap(t):
                    return t.inner if isinstance(t, Wrapper) else t

                return func(
                    *_pytree.tree_map(unwrap, args),
                    **_pytree.tree_map(unwrap, kwargs or {}),
                )

        wrapped_sparse = Wrapper(torch.tensor([[1.0, 0.0], [0.0, 2.0]]).to_sparse_coo())
        # The outer read that let it through, asserted so the test cannot pass because the
        # wrapper started reporting its inner layout (TwoTensor does, and is refused above).
        self.assertIs(wrapped_sparse.layout, torch.strided)
        with self.assertRaisesRegex(
            PrecompileError,
            r"user input 0 \(inner tensor 'inner' of a Wrapper subclass\) has "
            r"torch.sparse_coo layout.*reports 0 nnz",
        ):
            _precompile_pair(
                lambda t: t.values().sum() + float(t._nnz()),
                wrapped_sparse,
                backend="eager",
            )

        code, _ = _precompile_pair(
            lambda t: t.sum(), Wrapper(torch.randn(3, 4)), backend="eager"
        )
        self.assertIn("aten.sum", code)

    @unittest.skipUnless(TEST_CUDA, "pin_memory needs an accelerator allocator")
    def test_pinned_input_refused(self):
        # The fourth member of that table, in its own test because constructing the input
        # needs an accelerator: a skip here is reported, where an inline `if TEST_CUDA`
        # would let a CPU-only run claim it covered this branch. It is the branch that
        # matters most -- deleting it makes capture SUCCEED and ship an artifact that
        # baked the unpinned side of a branch on is_pinned().
        model = torch.nn.Linear(4, 4)
        with self.assertRaisesRegex(
            PrecompileError, "user input 0 is in pinned memory"
        ):
            _precompile_pair(
                lambda m, u: m(u),
                model,
                torch.randn(3, 4).pin_memory(),
                backend="eager",
            )

    def test_dispatched_pinned_probe_does_not_escape_under_vmap(self):
        # The is_pinned() probe in that loop DISPATCHES, so a tensor whose dispatch has no
        # rule for it (a vmap-batched one: "Batching rule not implemented for
        # aten::is_pinned") would leak that RuntimeError out of capture. It is guarded, so
        # such an input reaches its own refusal instead: invariant 1, because the batched
        # input is not the tensor make_fx lifts as the placeholder (its unbatched level
        # traces through as a constant). The regex pins WHICH refusal, so an unguarded
        # probe cannot pass by raising something else.
        model = torch.nn.Linear(4, 4)

        def capture_inside_vmap(row):
            with self.assertRaisesRegex(PrecompileError, "neither a graph input"):
                _precompile_pair(
                    lambda m, t: m(t), model, row.unsqueeze(0), backend="eager"
                )
            return row.sum()

        torch.vmap(capture_inside_vmap)(torch.randn(2, 4))

    def test_dispatch_declining_subclass_input_still_captures(self):
        # The same probe raises a TypeError, not a RuntimeError, for the decline protocol
        # PyTorch documents (every __torch_dispatch__ handler returning NotImplemented ->
        # "Multiple dispatch failed"), which torch.masked.MaskedTensor does for is_pinned.
        # Every raise the probe can make is swallowed, so it neither escapes the public API
        # nor costs a supported capture; the filter covers the MaskedTensor build only.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mt = torch.masked.as_masked_tensor(torch.randn(3, 4), torch.randn(3, 4) > 0)
        code, _ = _precompile_pair(lambda t: t.sum(), mt, backend="eager")
        self.assertIn("aten.sum", code)

    def test_capture_inside_another_trace_refused(self):
        # An ambient TracingContext.fake_mode outranks both mode sources capture hands
        # make_fx, and no foreign mode passes allow_fallback_kernels=False, so a meta-less
        # op would be run for real again; one built under DEFAULT config (as here, and as
        # an AOTAutograd / inductor trace builds its own) also lacks the
        # unsafe-data-ptr-access snapshot, so a .data_ptr() read bakes 0 instead of raising
        # (test_capture_refuses_a_data_ptr_read pins the refusal outside a trace). So
        # capture refuses up front, on a fn that captures cleanly on its own, rather than
        # tracing under a foreign contract.
        model = torch.nn.Linear(4, 4)
        x = torch.randn(3, 4)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with torch._guards.tracing(torch._guards.TracingContext(fake_mode)):
            with self.assertRaisesRegex(
                PrecompileError, "cannot run inside another trace"
            ):
                _precompile_pair(lambda m, t: m(t), model, x, backend="eager")
        _precompile_pair(lambda m, t: m(t), model, x, backend="eager")

    def test_unbacked_capture_refuses_an_unfakeifiable_input(self):
        # Both fakeify paths refuse an input the meta converter cannot represent through
        # the same helper, so a quantized example input gets the same named
        # PrecompileError (not the raw converter exception) whether or not some other dim
        # happens to be marked. Unbacked capture is inductor-only, so no backend override.
        model = torch.nn.Linear(3, 3)
        x = torch.randn(4, 3)
        mark_unbacked(x, 0)
        q = torch.quantize_per_tensor(torch.randn(3, 3), 0.1, 0, torch.qint8)
        with self.assertRaisesRegex(
            PrecompileError, "user input 1 cannot be represented as a fake tensor"
        ):
            _precompile_pair(
                lambda m, t, u: m(t) + m(u.dequantize()).sum(), model, x, q
            )

        # Including when the unfakeifiable input is the MARKED one: its unbacked rebuild
        # never consults the meta converter, so the marked branch validates the leaf
        # through the same helper -- without that it escapes as a raw meta-kernel error
        # ("SymIntArrayRef expected to contain only concrete integers").
        marked_q = torch.quantize_per_tensor(torch.randn(3, 3), 0.1, 0, torch.qint8)
        mark_unbacked(marked_q, 0)
        with self.assertRaisesRegex(
            PrecompileError, "user input 0 cannot be represented as a fake tensor"
        ):
            _precompile_pair(lambda m, u: m(u.dequantize()), model, marked_q)

        # The MODEL half is named too, on both paths: the unbacked path routes its
        # params/buffers through the same helper (a bare from_tensor there lets a
        # quantized buffer escape as the raw UnsupportedFakeTensorException), and this is
        # the only assertion on the "buffer {name}" half of input_labels for this refusal.
        class HasQuantizedBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(3, 3)
                self.register_buffer(
                    "q",
                    torch.quantize_per_tensor(torch.randn(3, 3), 0.1, 0, torch.qint8),
                )

            def forward(self, t):
                return self.lin(t)

        for marked_input in (False, True):
            t = torch.randn(4, 3)
            if marked_input:
                mark_unbacked(t, 0)
            with (
                self.subTest(unbacked=marked_input),
                self.assertRaisesRegex(
                    PrecompileError, "buffer q cannot be represented as a fake tensor"
                ),
            ):
                _precompile_pair(
                    lambda m, u: m(u),
                    HasQuantizedBuffer(),
                    t,
                    **({} if marked_input else {"backend": "eager"}),
                )

    def test_unbacked_capture_refuses_a_data_ptr_read(self):
        # The unbacked fake mode carries the same two hardenings as the static one, so the
        # Note's "no op is ever run for real on zero-filled substitutes" holds on both
        # paths. Here: its mode is also built inside the
        # fake_tensor_allow_unsafe_data_ptr_access patch, so a .data_ptr() read in fn is
        # refused instead of returning a meaningless value.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)

        def reads_pointer(mm, t):
            t.data_ptr()
            return mm(t)

        with self.assertRaisesRegex(PrecompileError, "data pointer"):
            _precompile_pair(reads_pointer, m, x)

    def test_unbacked_capture_refuses_a_meta_less_op_in_an_allowlisted_namespace(self):
        # The other unbacked-mode hardening: allow_fallback_kernels=False. An op with no
        # meta/fake kernel in an ALLOWLISTED namespace (aten, prims, quantized, ...) would
        # otherwise have FakeTensorMode's unsafe fallback run its real kernel on
        # zero-filled substitutes and bake whatever shape that produced. The op is called
        # on the UNMARKED input on purpose: the fallback declines symbolic-sized arguments
        # by itself, so only a static one exercises the flag.
        from torch.library import _scoped_library

        m = torch.nn.Linear(4, 3).eval()
        x, y = torch.randn(8, 4), torch.randn(2, 3)
        mark_unbacked(x, 0)
        with _scoped_library("quantized", "FRAGMENT") as qlib:
            qlib.define("mlprecompile_unbacked_no_meta(Tensor x) -> Tensor")
            qlib.impl("mlprecompile_unbacked_no_meta", lambda t: t * 2, "CPU")
            op = torch.ops.quantized.mlprecompile_unbacked_no_meta
            with self.assertRaisesRegex(PrecompileError, "no meta/fake kernel"):
                _precompile_pair(lambda mm, t, u: mm(t) + op(u).sum(), m, x, y)

    def test_data_dependent_refusal_omits_the_mark_unbacked_hint_when_unbacked(self):
        # The refusal's closing advice ("mark a user-input dim with mark_unbacked") is only
        # actionable for a STATIC capture. On the unbacked path the caller has already
        # marked a dim and the ShapeEnv exists, so an op no ShapeEnv can help with
        # (aten.equal) must not be answered with "go mark a dim". Unbacked capture is
        # inductor-only, so that half takes the default backend.
        def equal_branch(m, t):
            return m(t) if torch.equal(t, t) else m(t) * 2

        model = torch.nn.Linear(4, 4)
        x = torch.randn(4, 4)
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(PrecompileError, "data-dependent op") as cm:
            _precompile_pair(equal_branch, model, x)
        self.assertNotIn("mark_unbacked", str(cm.exception))
        with self.assertRaisesRegex(PrecompileError, "data-dependent op") as cm:
            _precompile_pair(equal_branch, model, torch.randn(4, 4), backend="eager")
        self.assertIn("mark_unbacked", str(cm.exception))

    def test_user_runtime_error_from_fn_propagates_unchanged(self):
        # Capture catches EVERY RuntimeError out of the trace to relabel the two it
        # owns (a missing fake impl, a data-pointer read), so a RuntimeError raised by
        # fn itself must fall through the substring checks and reach the caller with
        # its ORIGINAL message, not be relabeled as a missing meta/fake kernel.
        model = torch.nn.Linear(4, 4)

        def raises(m, x):
            raise RuntimeError("my own capture-time failure")

        try:
            _precompile_pair(raises, model, torch.randn(3, 4), backend="eager")
        except RuntimeError as e:
            self.assertIn("my own capture-time failure", str(e))
            # PrecompileError subclasses RuntimeError, so pin that it was not wrapped
            # (the only producer of the relabeled text raises one, so this covers it).
            self.assertNotIsInstance(e, PrecompileError)
        else:
            self.fail("expected fn's RuntimeError to propagate out of capture")

        # The three defensive guards in that same except chain, each otherwise unpinned:
        # an AttributeError from fn must be re-raised (only while_loop's
        # "ignore_fresh_unbacked_symbols" one becomes the control-flow refusal, else fn's
        # own message comes back dressed as a control-flow refusal), an AssertionError
        # from fn likewise (only the "NYI: " prefix FakeTensorMode raises becomes the
        # missing-kernel refusal), and a RuntimeError with an EMPTY message must not turn
        # the propagation into an IndexError off splitlines()[0]. assertIs pins that the
        # SAME exception object came through.
        for raised in (
            AttributeError("my own attribute error"),
            AssertionError("my own assertion"),
            RuntimeError(""),
        ):

            def raises_it(m, x, raised=raised):
                raise raised

            with self.subTest(raised=type(raised).__name__):
                with self.assertRaises(type(raised)) as cm:
                    _precompile_pair(
                        raises_it, model, torch.randn(3, 4), backend="eager"
                    )
                self.assertIs(cm.exception, raised)

    def test_precompile_error_from_fn_is_not_relabeled(self):
        # PrecompileError subclasses RuntimeError, so the trace's RuntimeError clause would
        # relabel a refusal of precompile's OWN whose text happens to carry one of the two
        # matched substrings. An explicit "except PrecompileError: raise" sits ahead of that
        # clause; without it this message comes back as "... no meta/fake kernel ...".
        model = torch.nn.Linear(4, 4)
        message = "precompile: my own refusal, no fake impl registered for t::op"

        def raises(m, x):
            raise PrecompileError(message)

        with self.assertRaises(PrecompileError) as cm:
            _precompile_pair(raises, model, torch.randn(3, 4), backend="eager")
        # Equality is the whole assertion: every relabel site builds a NEW PrecompileError
        # with different text, so a lost "except PrecompileError: raise" reds it here.
        self.assertEqual(str(cm.exception), message)

    def test_mutating_custom_op_captures_without_a_registered_fake(self):
        # The one carve-out in "fake tracing needs a meta/fake kernel for every op": a
        # torch.library.custom_op that only mutates its arguments and returns nothing gets
        # a trivial fake impl synthesized, so it captures with no register_fake and the
        # call is recorded in the artifact (and still mutates when served).
        model = torch.nn.Linear(4, 4)

        @torch.library.custom_op("mlprecompile::add_one_", mutates_args={"x"})
        def add_one_(x: torch.Tensor) -> None:
            x.add_(1.0)

        def fn(m, x):
            torch.ops.mlprecompile.add_one_(x)
            return m(x)

        try:
            x = torch.zeros(3, 4)
            code, cache = _precompile_pair(fn, model, x, backend="eager")
            self.assertIn("mlprecompile.add_one_", code)
            self.assertEqual(x, torch.zeros(3, 4))  # capture ran on fakes
            torch.compiler.precompile.load(code, cache)(model, x)
            self.assertEqual(x, torch.ones(3, 4))
        finally:
            add_one_._lib._destroy()

    def test_callable_api_traces_a_backward_under_ambient_no_grad(self):
        # The callable API keeps grad enabled around the trace whatever the caller's
        # ambient mode, so a training step captured inside no_grad still carries
        # its backward and the artifact does not depend on the call site: the served
        # gradients match the eager ones, not merely being present.
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 2)
        x, t = torch.randn(3, 4), torch.randn(3, 2)

        def step(m, x, t):
            torch.nn.functional.mse_loss(m(x), t).backward()

        with torch.no_grad():
            python_code, cache = _precompile_pair(step, model, x, t, backend="eager")
        torch.compiler.precompile.load(python_code, cache)(model, x, t)
        self.assertIsNotNone(model.weight.grad)
        ref = torch.nn.Linear(4, 2)
        ref.load_state_dict(model.state_dict())
        step(ref, x, t)
        self.assertEqual(model.weight.grad, ref.weight.grad)
        self.assertEqual(model.bias.grad, ref.bias.grad)


class _FilesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))


def _files_fn(model, x):
    return model(x)


def _files_train_step(model, x):
    out = model(x)
    out.sum().backward()
    return out


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompileCaptureFiles(TestCase):
    """capture()/load() through the on-disk artifact pair (MakeFxTracer)."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.artifact = os.path.join(self.dir, "m.py")
        self.cache = os.path.join(self.dir, "m.cache")
        self.model = _FilesModel()
        self.x = torch.randn(2, 4)

    def _capture(self, fn=_files_fn, **kwargs):
        kwargs.setdefault("artifact_path", self.artifact)
        kwargs.setdefault("cache_path", self.cache)
        kwargs.setdefault("tracer", MakeFxTracer())
        kwargs.setdefault("backend", "eager")
        return capture(fn, **kwargs)

    def _leftovers(self):
        return sorted(n for n in os.listdir(self.dir) if n not in ("m.py", "m.cache"))

    def _read(self, path):
        with open(path, "rb") as f:
            return f.read()

    def _rewrite_raises(self, exc_type, regex, backend="inductor"):
        # Recapture into the same paths, expecting the exit write to fail with the
        # patched-in exception.
        with self.assertRaisesRegex(exc_type, regex):
            with self._capture(backend=backend) as cap:
                cap(self.model, self.x)

    def _write_pair(self):
        # The "previous artifact" the recovery tests want on disk, with both halves' bytes.
        with self._capture() as cap:
            cap(self.model, self.x)
        return self._read(self.artifact), self._read(self.cache)

    def _assert_serves(self):
        # Both halves of what a writer test wants: no scratch file or backup left behind,
        # and the named pair loading and running.
        self.assertEqual(self._leftovers(), [])
        f = load(self.artifact, self.cache)
        self.assertEqual(f(self.model, self.x), self.model(self.x))

    def _assert_kept_backup(self, before, logs):
        # The previous source survived as the only leftover, a .bak the warning names.
        leftovers = self._leftovers()
        self.assertEqual(len(leftovers), 1, leftovers)
        self.assertTrue(leftovers[0].endswith(".bak"), leftovers)
        self.assertEqual(self._read(os.path.join(self.dir, leftovers[0])), before)
        self.assertTrue(any(leftovers[0] in m for m in logs.output), logs.output)

    def _replacing(self, *names, exc, after=False, once=True, by_src=False):
        """Patch os.replace so a rename INTO one of ``names`` (or OUT of one, with
        ``by_src``) raises ``exc``: before performing it, or (``after``) once it has,
        which is where a KeyboardInterrupt lands. ``once`` fails only the first one."""
        real, fired = os.replace, []

        def replace(src, dst):
            if (src if by_src else dst) not in names or (once and fired):
                return real(src, dst)
            fired.append(dst)
            if after:
                real(src, dst)
            raise exc

        return mock.patch("os.replace", replace)

    def test_exit_writes_the_pair_and_load_serves_it(self):
        with self._capture() as cap:
            y = cap(self.model, self.x)
        self.assertEqual(y, self.model(self.x))
        self.assertTrue(os.path.exists(self.artifact) and os.path.exists(self.cache))
        # load() EXECs source it did not produce, so it warns first (the capture's own
        # self-load is _trusted, which the writer tests' assertNoLogs pin).
        with self.assertLogs("torch._precompile", level="WARNING") as logs:
            f = load(self.artifact, self.cache)
        self.assertIn("precompile.load is about to EXEC", "\n".join(logs.output))
        self.assertEqual(f(self.model, self.x), self.model(self.x))
        self.assertFalse(f.installed)
        self.assertEqual(self._leftovers(), [])

    def test_inductor_pair_round_trips(self):
        # Called bare, so also the only coverage of capture()'s documented defaults:
        # MakeFxTracer() and backend="inductor".
        a, c = self.artifact, self.cache
        with capture(_files_fn, artifact_path=a, cache_path=c) as cap:
            cap(self.model, self.x)
        self.assertIn("BACKEND = 'inductor'", self._read(self.artifact).decode())
        f = load(self.artifact, self.cache)
        self.assertEqual(f(self.model, self.x), self.model(self.x))

    def test_second_call_is_refused(self):
        with self._capture() as cap:
            cap(self.model, self.x)
            with self.assertRaisesRegex(PrecompileError, "single call"):
                cap(self.model, self.x)

    def test_a_failed_serve_writes_nothing(self):
        # The call renders and then SERVES, where the driver's runtime checks run: a serve
        # that raised is a call that did not work, so a caller who catches it gets no write.
        def raising_serve(*args, **kwargs):
            def f(*call_args):
                raise RuntimeError("serve failed")

            return f

        with mock.patch("torch._precompile._runnable_from_pair", raising_serve):
            with self.assertRaisesRegex(PrecompileError, "while serving"):
                with self._capture() as cap:
                    with self.assertRaisesRegex(RuntimeError, "serve failed"):
                        cap(self.model, self.x)
                    with self.assertRaisesRegex(PrecompileError, "already ran and"):
                        cap(self.model, self.x)
        self.assertEqual(os.listdir(self.dir), [])

    def test_keyword_arguments_are_refused(self):
        with self._capture() as cap:
            with self.assertRaisesRegex(TypeError, "positional arguments only"):
                cap(self.model, x=self.x)
            cap(self.model, self.x)

    def test_exception_in_block_leaves_files_untouched(self):
        before = self._write_pair()
        # A different backend, so a write from the raising block would change both halves'
        # bytes; the same backend renders byte-identical output and could not fail.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with self._capture(backend="inductor") as cap:
                cap(self.model, self.x)
                raise RuntimeError("boom")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self.assertEqual(self._leftovers(), [])

    def test_save_writes_before_exit_and_exit_rewrites_the_same_pair(self):
        with self._capture() as cap:
            with self.assertRaisesRegex(PrecompileError, "before calling save"):
                cap.save()
            cap(self.model, self.x)
            cap.save()
            saved = (self._read(self.artifact), self._read(self.cache))
            # The rewrite repeats the same bytes, so pin the inodes instead: each write
            # renames fresh temps in, and skipping the exit write leaves save()'s.
            inodes = (os.stat(self.artifact).st_ino, os.stat(self.cache).st_ino)
            self._assert_serves()
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), saved)
        rewritten = (os.stat(self.artifact).st_ino, os.stat(self.cache).st_ino)
        self.assertNotEqual(rewritten, inodes)
        self.assertEqual(self._leftovers(), [])

    def test_same_file_for_both_halves_is_refused(self):
        # Also when the two halves are two SPELLINGS of one path: the cache write would
        # clobber the source half, so the comparison is on the RESOLVED absolute paths.
        spellings = [self.artifact, os.path.join(self.dir, ".", "m.py")]
        if sys.platform != "win32":
            # A symlinked path COMPONENT names one file too, which normalizing alone misses.
            os.symlink(self.dir, os.path.join(self.dir, "link"))
            spellings.append(os.path.join(self.dir, "link", "m.py"))
        for cache_path in spellings:
            with self.assertRaisesRegex(ValueError, "same file"):
                self._capture(cache_path=cache_path)
            with self.assertRaisesRegex(ValueError, "same file"):
                load(self.artifact, cache_path)

    @parametrize("half", ("artifact", "cache"))
    def test_a_directory_for_either_half_is_refused(self, half):
        # Refused up front for either half: an earlier, clearer error than the writer's.
        path = getattr(self, half)
        os.mkdir(path)
        with open(os.path.join(path, "keep.txt"), "w", encoding="utf-8") as f:
            f.write("mine")
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            self._capture()
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            load(self.artifact, self.cache)
        # And the writer itself refuses a directory that appears after the up-front check:
        # os.link re-raises EPERM for the source half, while the cache half fails its
        # rename with the source half installed, so the undo removes that orphan.
        with self.assertRaises(OSError):
            torch._precompile._write_artifact(self.artifact, self.cache, "x", b"y")
        self.assertEqual(os.listdir(path), ["keep.txt"])
        self.assertFalse(os.path.isfile(self.artifact))
        self.assertEqual(self._leftovers(), [])

    def test_non_tracer_is_refused(self):
        with self.assertRaisesRegex(TypeError, "must be a MakeFxTracer"):
            self._capture(tracer="make_fx")

    def test_unknown_backend_is_refused(self):
        with self.assertRaisesRegex(ValueError, "backend must be"):
            self._capture(backend="nope")

    @parametrize("backend", ("eager", "inductor"))
    def test_batchnorm_running_stats_update_once(self, backend):
        bn = torch.nn.BatchNorm1d(4)
        ref = copy.deepcopy(bn)
        x = torch.randn(8, 4)
        with self._capture(backend=backend) as cap:
            y = cap(bn, x)
        self.assertEqual(y, ref(x))
        self.assertEqual(bn.num_batches_tracked, ref.num_batches_tracked)
        self.assertEqual(bn.running_mean, ref.running_mean)
        self.assertEqual(bn.running_var, ref.running_var)

    @parametrize("backend", ("eager", "inductor"))
    def test_training_captures_the_backward(self, backend):
        # Captured under an ambient no_grad, so training=True and not the ambient mode is
        # what enables grad: ignoring the flag would raise here.
        ref = copy.deepcopy(self.model)
        _files_train_step(ref, self.x)
        with torch.no_grad():
            cap = self._capture(_files_train_step, backend=backend, training=True)
            with cap:
                cap(self.model, self.x)
        self.assertEqual(self.model.lin.weight.grad, ref.lin.weight.grad)
        self.assertEqual(self.model.lin.bias.grad, ref.lin.bias.grad)

    def test_a_backward_without_training_points_at_training_true(self):
        # training=False traces under no_grad, so nothing fn's .backward() sees has a
        # grad_fn; the capture re-reports autograd's raise naming the switch that fixes it.
        with self.assertRaisesRegex(PrecompileError, "Pass training=True") as cm:
            with self._capture(_files_train_step) as cap:
                cap(self.model, self.x)
        self.assertIn("does not require grad", str(cm.exception.__cause__))
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_backward_on_a_buffer_points_at_training_true(self):
        # The relabel scan reads module BUFFERS too: a requires_grad buffer differentiated
        # with torch.autograd.grad (no .grad write, so invariant 5's buffer refusal does not
        # apply) is a capture no PARAMETER makes grad-requiring.
        model = _FilesModel().requires_grad_(False)
        model.register_buffer("b", torch.randn(4, requires_grad=True))

        def fn(m, x):
            return torch.autograd.grad((m(x) * m.b).sum(), m.b)[0]

        with self.assertRaisesRegex(PrecompileError, "Pass training=True"):
            with self._capture(fn) as cap:
                cap(model, self.x)
        with self._capture(fn, training=True) as cap:
            self.assertEqual(cap(model, self.x).shape, model.b.shape)

    def test_a_frozen_parameter_backward_is_not_blamed_on_training(self):
        # Autograd raises the same "does not require grad" when nothing fn differentiates
        # requires grad (here every parameter is frozen), which training=True does not fix.
        model = _FilesModel().requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "does not require grad") as cm:
            with self._capture(_files_train_step) as cap:
                cap(model, self.x)
        self.assertNotIsInstance(cm.exception, PrecompileError)
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_detached_backward_under_training_is_not_blamed_on_training(self):
        # The same autograd error under training=True is not the grad mode's fault: fn itself
        # blocked the gradient, so relabelling it would advise what the caller already did.
        def fn(m, x):
            m(x).detach().sum().backward()

        with self.assertRaisesRegex(RuntimeError, "does not require grad") as cm:
            with self._capture(fn, training=True) as cap:
                cap(self.model, self.x)
        self.assertNotIsInstance(cm.exception, PrecompileError)
        self.assertEqual(os.listdir(self.dir), [])

    @parametrize("backend", ("eager", "inductor"))
    def test_served_output_requires_grad_contract(self, backend):
        # The documented contract: an output that IS an input comes back as that same tensor,
        # a computed output requires no grad, and an ALIASING output is a view rebuilt at
        # serve time -- off the runtime input by eager, with the capture's value by inductor.
        def fn(t):
            return t, t.t(), t * 2

        with self._capture(fn, backend=backend) as cap:
            cap(torch.randn(2, 3, requires_grad=True))
        f = load(self.artifact, self.cache)
        x = torch.randn(2, 3)
        same, view, _ = f(x)
        self.assertIs(same, x)
        self.assertIs(view._base, x)
        self.assertEqual(view.requires_grad, backend == "inductor")
        # A grad-requiring input is the only source of requires_grad for t * 2, so it is what
        # exercises the served value stripping it; the same/view asserts need the non-grad x.
        x2 = torch.randn(2, 3, requires_grad=True)
        same2, _, computed = f(x2)
        self.assertIs(same2, x2)
        self.assertFalse(computed.requires_grad)

    def test_a_failed_cache_rename_restores_the_previous_pair(self):
        # The first rename landed, so the undo renames the backup back over the new source,
        # which consumes the .bak and leaves the trailing unlink a no-op. The other shape, a
        # failing FIRST rename with the artifact still the hard-linked previous source, is
        # the read-only-first-rename test below.
        before = self._write_pair()
        with self._replacing(self.cache, exc=OSError("disk full")):
            self._rewrite_raises(OSError, "disk full")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self._assert_serves()

    def test_a_failed_exit_write_leaves_the_capture_savable(self):
        # The block is over, but a write that RAISED (here ENOSPC on the first rename) left
        # the files untouched and the render in memory, so save() retries it: a transient
        # failure does not cost a re-trace and re-lower.
        cap = self._capture()
        no_space = OSError(errno.ENOSPC, "no space left")
        with self._replacing(self.artifact, exc=no_space):
            with self.assertRaisesRegex(OSError, "no space left"):
                with cap:
                    cap(self.model, self.x)
        self.assertEqual(os.listdir(self.dir), [])
        # Spent for tracing, but both doors send the caller to save(), which closes on success.
        for door in (cap.__enter__, functools.partial(cap, self.model, self.x)):
            with self.assertRaisesRegex(PrecompileError, "WRITE is what failed"):
                door()
        cap.save()
        self._assert_serves()
        with self.assertRaisesRegex(PrecompileError, r"Call capture\(\) again"):
            cap.save()

    def test_a_failed_in_block_save_leaves_the_capture_savable(self):
        # An in-block save() whose WRITE raised is the same recoverable state as a failed exit
        # write, and its exception leaves the block, so the exit writes nothing: the post-block
        # save() is the retry of that write, not a spent capture.
        cap = self._capture()
        nospc = OSError(errno.ENOSPC, "no space left")
        with self.assertRaisesRegex(OSError, "no space left"):
            with cap:
                cap(self.model, self.x)
                with self._replacing(self.artifact, exc=nospc):
                    cap.save()
        self.assertEqual(os.listdir(self.dir), [])
        cap.save()
        self._assert_serves()
        # The same failure CAUGHT: the clean exit writes the pair after all and clears the
        # flag again, so this capture is spent like any other.
        cap = self._capture()
        with cap:
            cap(self.model, self.x)
            with self._replacing(self.artifact, exc=nospc), self.assertRaises(OSError):
                cap.save()
        self._assert_serves()
        with self.assertRaisesRegex(PrecompileError, r"Call capture\(\) again"):
            cap.save()

    @parametrize("half", ("artifact", "cache"))
    def test_a_failed_rename_on_a_first_write_leaves_nothing_named(self, half):
        # No previous pair to restore, so the undo takes the new artifact back out from under
        # its name: a first write cannot leave a named source with no cache beside it.
        no_space = OSError(errno.ENOSPC, "no space left")
        with self._replacing(getattr(self, half), exc=no_space):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(OSError, "no space left", backend="eager")
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_second_writers_pair_is_not_undone(self):
        # Two writers: this one's first rename landed and then ANOTHER writer installed its
        # whole pair before this one's second rename failed. The artifact name is no longer
        # this call's, so the undo leaves that pair alone -- restoring would put a THIRD,
        # older source beside the other writer's cache -- and reports nothing.
        self._write_pair()
        other = torch.nn.Linear(4, 3)
        w2 = (os.path.join(self.dir, "w.py"), os.path.join(self.dir, "w.cache"))
        with self._capture(artifact_path=w2[0], cache_path=w2[1]) as cap:
            cap(other, self.x)
        w2_bytes = (self._read(w2[0]), self._read(w2[1]))
        real_replace = os.replace

        def replace(src, dst):
            if dst != self.cache:
                return real_replace(src, dst)
            # The other writer lands both halves in the window between this writer's
            # two renames.
            real_replace(w2[0], self.artifact)
            real_replace(w2[1], self.cache)
            raise OSError(errno.ENOSPC, "no space left")

        with mock.patch("os.replace", replace):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(OSError, "no space left")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), w2_bytes)
        self.assertEqual(self._leftovers(), [])
        served = load(self.artifact, self.cache)
        self.assertEqual(served(other, self.x), other(self.x))

    def test_rewrite_without_hard_links(self):
        first = self._write_pair()[1]
        link_fails = OSError(errno.EOPNOTSUPP, "no hard links")
        with mock.patch("os.link", side_effect=link_fails):
            with self._capture(backend="inductor") as cap:
                cap(self.model, self.x)
        self.assertNotEqual(self._read(self.cache), first)
        self._assert_serves()
        # An os.link failure that does NOT mean "no hard links on this filesystem" says
        # something about the path, so it is raised rather than papered over by moving
        # whatever is there aside; it fails before either rename, so nothing is reported.
        with mock.patch("os.link", side_effect=OSError(errno.EIO, "io error")):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(OSError, "io error", backend="eager")
        self._assert_serves()

    def test_double_failure_without_hard_links_keeps_the_previous_source(self):
        # No hard links, so the previous source was MOVED to the backup; the rename into place
        # then fails AND so does the undo, so the backup holds its only copy.
        before = self._write_pair()[0]
        with mock.patch("os.link", side_effect=OSError(errno.EXDEV, "no hard links")):
            with self._replacing(self.artifact, exc=OSError("disk full"), once=False):
                with self.assertLogs("torch._precompile", level="WARNING") as logs:
                    self._rewrite_raises(OSError, "disk full")
        self.assertFalse(os.path.exists(self.artifact))
        self._assert_kept_backup(before, logs)

    def test_double_failure_on_the_second_rename_keeps_the_previous_source(self):
        # Hard link taken, the artifact rename lands, the cache rename fails and the undo
        # fails too: the named pair is the new source beside the previous cache, which load
        # refuses, so the previous source stays in the backup.
        before = self._write_pair()[0]
        before_cache = self._read(self.cache)
        real_replace = os.replace
        calls = []

        def replace(src, dst):
            calls.append(dst)
            if dst == self.cache:
                raise OSError("disk full")
            if len(calls) > 1:
                raise OSError("undo failed")
            return real_replace(src, dst)

        with mock.patch("os.replace", replace):
            with self.assertLogs("torch._precompile", level="WARNING") as logs:
                self._rewrite_raises(OSError, "disk full")
        self.assertNotEqual(self._read(self.artifact), before)
        self.assertEqual(self._read(self.cache), before_cache)
        with self.assertRaisesRegex(PrecompileError, "different precompile captures"):
            load(self.artifact, self.cache)
        self._assert_kept_backup(before, logs)

    def test_a_read_only_first_rename_leaves_the_previous_pair_intact(self):
        # A read-only remount: the artifact rename fails, and so would the undo. The backup is
        # a hard LINK, so the name still IS the previous source: nothing to undo, no report.
        before = self._write_pair()
        read_only = OSError(errno.EROFS, "read-only file system")
        with self._replacing(self.artifact, self.cache, exc=read_only, once=False):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(OSError, "read-only file system")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self._assert_serves()

    def test_a_first_write_that_cannot_be_undone_warns_about_the_orphan(self):
        # A FIRST write whose cache rename failed and whose undo then failed leaves a named
        # artifact with no matching cache beside it, the state the report exists to announce.
        real_unlink = os.unlink
        no_space = OSError(errno.ENOSPC, "no space left")

        def unlink(path):
            if path == self.artifact:
                raise OSError(errno.EPERM, "cannot unlink")
            return real_unlink(path)

        with self._replacing(self.cache, exc=no_space), mock.patch("os.unlink", unlink):
            with self.assertLogs("torch._precompile", level="WARNING") as logs:
                self._rewrite_raises(OSError, "no space left", backend="eager")
        self.assertTrue(os.path.exists(self.artifact))
        self.assertFalse(os.path.exists(self.cache))
        self.assertTrue(
            any("no cache beside it matches" in m for m in logs.output), logs.output
        )
        self.assertEqual(self._leftovers(), [])

    def test_an_interrupted_undo_keeps_the_previous_source(self):
        # The undo can be cut short by a KeyboardInterrupt rather than an OSError, and the
        # rename is then just as undone; without hard links the backup is the only source.
        before = self._write_pair()[0]
        real_replace = os.replace
        installs = []

        def replace(src, dst):
            if dst != self.artifact:
                return real_replace(src, dst)
            installs.append(src)
            if len(installs) == 1:
                raise OSError("disk full")
            raise KeyboardInterrupt("interrupted during the undo")

        with mock.patch("os.link", side_effect=OSError(errno.EXDEV, "no hard links")):
            with mock.patch("os.replace", replace):
                with self.assertLogs("torch._precompile", level="WARNING") as logs:
                    self._rewrite_raises(KeyboardInterrupt, "interrupted")
        self.assertFalse(os.path.exists(self.artifact))
        self._assert_kept_backup(before, logs)

    def test_an_interrupt_after_the_move_aside_restores_the_name(self):
        # Taking the backup has that window too: without hard links the previous source is
        # MOVED aside, so an interrupt after os.replace returned leaves the artifact NAME
        # gone -- a flag set after that call left the caller's path empty, silently.
        before = self._write_pair()
        cut = KeyboardInterrupt("interrupted after the move aside")
        with mock.patch("os.link", side_effect=OSError(errno.EXDEV, "no hard links")):
            with self._replacing(self.artifact, exc=cut, after=True, by_src=True):
                with self.assertNoLogs("torch._precompile", level="WARNING"):
                    self._rewrite_raises(KeyboardInterrupt, "interrupted")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self.assertEqual(self._leftovers(), [])

    def test_an_interrupt_after_the_hard_link_drops_the_backup(self):
        # The same window on the hard-link path: the .bak is already a link to the previous
        # source, and a flag that read "no backup" skipped the cleanup and left it pinning
        # that inode forever. The named pair is untouched, so it just goes.
        before = self._write_pair()
        real_link = os.link

        def link(src, dst):
            real_link(src, dst)
            raise KeyboardInterrupt("interrupted after the hard link")

        with mock.patch("os.link", link):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(KeyboardInterrupt, "interrupted")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self.assertEqual(self._leftovers(), [])

    def test_an_interrupt_after_the_first_rename_restores_the_previous_source(self):
        # A KeyboardInterrupt is raised at the bytecode after os.replace RETURNED, so the
        # first rename can have landed with a flag that records it still unset. Reading that
        # flag called this a write that never started and unlinked the backup -- with hard
        # links, the previous source's last link.
        before = self._write_pair()
        before_ino = os.stat(self.artifact).st_ino
        cut = KeyboardInterrupt("interrupted after the first rename")
        with self._replacing(self.artifact, exc=cut, after=True):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(KeyboardInterrupt, "interrupted")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self.assertEqual(os.stat(self.artifact).st_ino, before_ino)
        self._assert_serves()

    def test_an_interrupt_after_the_second_rename_keeps_the_new_pair(self):
        # The same window one rename later: both halves are already under their names, so the
        # write SUCCEEDED and there is nothing to undo -- undoing anyway puts the previous
        # source back beside the new cache, an unloadable pair.
        before = self._write_pair()
        cut = KeyboardInterrupt("interrupted after the second rename")
        with self._replacing(self.cache, exc=cut, after=True):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(KeyboardInterrupt, "interrupted")
        self.assertNotEqual((self._read(self.artifact), self._read(self.cache)), before)
        self._assert_serves()

    def test_an_interrupt_among_the_probes_keeps_the_previous_source(self):
        # The undo's own reads have that window: an interrupt between the backup read (the
        # True below) and the flag saying the reads RAN dropped the previous source's .bak.
        before = self._write_pair()[0]
        cut = mock.patch("os.path.lexists", side_effect=[True, KeyboardInterrupt()])
        with self._replacing(self.cache, exc=OSError(errno.ENOSPC, "no space")), cut:
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                with self.assertRaises(KeyboardInterrupt):
                    _write_artifact(self.artifact, self.cache, "new", b"new-cache")
        kept = [self._read(os.path.join(self.dir, n)) for n in self._leftovers()]
        self.assertEqual(kept, [before])

    def test_an_interrupt_after_the_undos_rename_reports_nothing(self):
        # The undo's OWN os.replace has that window too: the previous source is back under
        # its name and the .bak it came from is consumed, while the flag that records the
        # undo still reads False. Reporting off that flag named a gone .bak.
        before = self._write_pair()
        before_ino = os.stat(self.artifact).st_ino
        real_replace = os.replace
        installs = []

        def replace(src, dst):
            if dst == self.cache:
                raise OSError(errno.ENOSPC, "no space left")
            real_replace(src, dst)
            if installs:
                raise KeyboardInterrupt("interrupted after the undo's rename")
            installs.append(dst)

        with mock.patch("os.replace", replace):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(KeyboardInterrupt, "interrupted")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self.assertEqual(os.stat(self.artifact).st_ino, before_ino)
        self._assert_serves()

    def test_an_interrupt_after_the_undos_unlink_reports_nothing(self):
        # The same window on a FIRST write, whose undo unlinks the orphan artifact instead:
        # the file the report would name is already gone, so there is nothing to announce.
        real_unlink = os.unlink
        no_space = OSError(errno.ENOSPC, "no space left")

        def unlink(path):
            real_unlink(path)
            if path == self.artifact:
                raise KeyboardInterrupt("interrupted after the undo's unlink")

        with self._replacing(self.cache, exc=no_space), mock.patch("os.unlink", unlink):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(KeyboardInterrupt, "interrupted", backend="eager")
        self.assertFalse(os.path.exists(self.artifact))
        self.assertEqual(os.listdir(self.dir), [])

    @unittest.skipIf(sys.platform == "win32", "chmod does not set a POSIX mode there")
    @parametrize("mask", (0, 0o077))
    def test_a_rewrite_preserves_the_previous_modes(self, mask):
        # os.replace repoints the name at the temp's inode, so a rewrite would carry the temp's
        # umask-derived mode onto the pair and silently widen a mode the caller set. Each half
        # keeps its OWN mode -- two different ones, so a writer reading the wrong half's is
        # caught -- and the temp is CREATED with it: os.fstat at the payload fsync is the window
        # a chmod after the write would leave it wide in. The umask is pinned rather than
        # inherited, and both halves of the mode logic covered: at 0 the mid-write mode is
        # O_CREAT's argument exactly, while a mask that eats the bits leaves the follow-up
        # chmod as the only thing that can restore them. The source's setgid bit must NOT come
        # back: it is masked off rather than recreated on an inode owned by the writing user.
        self.addCleanup(os.umask, os.umask(mask))
        self._write_pair()
        modes = ((self.artifact, stat.S_ISGID | 0o600), (self.cache, 0o640))
        for path, mode in modes:
            os.chmod(path, mode)
        mid_write, real_fsync = [], os.fsync

        def fsync(fd):
            st = os.fstat(fd)
            if not stat.S_ISDIR(st.st_mode):
                mid_write.append(stat.S_IMODE(st.st_mode))
            return real_fsync(fd)

        with mock.patch("os.fsync", fsync):
            with self._capture() as cap:
                cap(self.model, self.x)
        for path, mode in modes:
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), mode & 0o777, path)
        self.assertEqual(mid_write, [0o600 & ~mask, 0o640 & ~mask])
        self._assert_serves()

    def test_a_failed_directory_fsync_does_not_fail_the_write(self):
        # The directory fsync is best effort: a directory fd cannot be fsync'd everywhere,
        # and by then both renames have returned, so there is nothing to undo or retry.
        real_fsync = os.fsync

        def fsync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EINVAL, "no fsync on a directory")
            return real_fsync(fd)

        with mock.patch("os.fsync", fsync):
            with self._capture() as cap:
                cap(self.model, self.x)
        self._assert_serves()

    def test_a_failed_temp_fsync_leaves_no_scratch_files(self):
        # The fsync of a scratch file is NOT best effort: unflushed bytes would be renamed
        # into place, so it propagates -- with every scratch file removed and both named
        # halves untouched, which is why this one rewrites a previous pair.
        before = self._write_pair()
        real_fsync = os.fsync

        def fsync(fd):
            if not stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError(errno.EIO, "fsync failed")
            return real_fsync(fd)

        with mock.patch("os.fsync", fsync):
            self._rewrite_raises(OSError, "fsync failed", backend="eager")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self.assertEqual(self._leftovers(), [])

    def test_a_callable_holding_a_module_or_tensor_is_refused(self):
        # A model or tensor reached other than as a call argument never sees the scan, so its
        # params would bake in as constants: refused for fn itself, for a partial's bind, and
        # for the (nested) callable a partial wraps, where nothing else would look.
        for fn in (
            self.model,
            self.model.forward,
            functools.partial(_files_fn, self.model),
            functools.partial(_files_fn, x=self.x),
            functools.partial(self.model),
            functools.partial(functools.partial(self.model.forward)),
        ):
            with self.assertRaisesRegex(PrecompileError, "HOLDS a tensor or an"):
                self._capture(fn)
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_partial_binding_a_scalar_is_captured(self):
        # Only a bound tensor / module is the hazard, so a partial that binds a
        # plain scalar captures and serves like any other callable.
        def fn(model, x, alpha):
            return model(x) * alpha

        with self._capture(functools.partial(fn, alpha=0.5)) as cap:
            y = cap(self.model, self.x)
        expected = self.model(self.x) * 0.5
        self.assertEqual(y, expected)
        self.assertEqual(load(self.artifact, self.cache)(self.model, self.x), expected)

    def test_mismatched_pair_is_refused(self):
        self._write_pair()
        with open(self.artifact, "a", encoding="utf-8") as f:
            f.write("\n# edited\n")
        with self.assertRaisesRegex(PrecompileError, "code_hash"):
            load(self.artifact, self.cache)

    def test_unreadable_cache_falls_back_to_python_code(self):
        self._write_pair()
        with open(self.cache, "wb") as f:
            f.write(b"not a torch.save envelope")
        with self.assertLogs("torch._precompile", level="WARNING") as logs:
            f = load(self.artifact, self.cache)
        self.assertTrue(
            any(":torch._precompile.load could not read" in m for m in logs.output)
        )
        self.assertEqual(f(self.model, self.x), self.model(self.x))

    def test_load_checks_the_cache_envelope_and_primes_from_it(self):
        # Three things load() documents that the round trips do not pin, since a cache is
        # acceleration only and nothing FAILS without one: the bundle is really handed to the
        # inductor caches (an eager pair's is None and hands nothing over), the backend tag is
        # an integrity check like code_hash, and a foreign format only DEGRADES to JIT.
        real = torch.compiler.load_cache_artifacts
        for backend, calls in (("eager", 0), ("inductor", 1)):
            with self._capture(backend=backend) as cap:
                cap(self.model, self.x)
            with mock.patch("torch.compiler.load_cache_artifacts", wraps=real) as spy:
                load(self.artifact, self.cache)
            blob = torch.load(io.BytesIO(self._read(self.cache)), weights_only=True)
            handed = [c.args for c in spy.call_args_list]
            self.assertEqual(handed, [(blob["artifact"],)] * calls)

        def rewrite(**edits):
            buf = io.BytesIO()
            torch.save({**blob, **edits}, buf)
            with open(self.cache, "wb") as f:
                f.write(buf.getvalue())

        # The pair on disk is the inductor one and code_hash still matches, so each edit below
        # is the only mismatch the load sees.
        rewrite(backend="eager")
        with self.assertRaisesRegex(PrecompileError, "does not match the python_code"):
            load(self.artifact, self.cache)
        rewrite(format="from-another-build")
        with self.assertLogs("torch._precompile", level="WARNING") as logs:
            f = load(self.artifact, self.cache)
        self.assertIn("different torch build", "\n".join(logs.output))
        self.assertEqual(f(self.model, self.x), self.model(self.x))

    def test_non_literal_metadata_is_refused(self):
        self._write_pair()
        source = self._read(self.artifact).decode()
        self.assertIn("BACKEND = 'eager'", source)
        with open(self.artifact, "wb") as f:
            f.write(source.replace("BACKEND = 'eager'", "BACKEND = str(1)").encode())
        # The metadata parse runs ahead of the cache pairing check, so this is the
        # error the caller sees even though the edit also broke the code_hash.
        with self.assertRaisesRegex(PrecompileError, "must be a Python literal"):
            load(self.artifact, self.cache)

    def test_unreadable_paths_are_reported_as_precompile_errors(self):
        self._write_pair()
        missing = os.path.join(self.dir, "gone.py")
        with self.assertRaises(PrecompileError) as cm:
            load(missing, self.cache)
        self.assertIn("could not read the artifact pair", str(cm.exception))
        # The message renders both paths with !r, and a Windows path's repr doubles
        # its backslashes, so compare against the repr rather than the raw string.
        self.assertIn(repr(missing), str(cm.exception))
        self.assertIn(repr(self.cache), str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, FileNotFoundError)
        # A path the filesystem cannot open at all reaches the same diagnostic.
        with self.assertRaises(PrecompileError) as cm:
            load("x" * 5000, self.cache)
        self.assertEqual(cm.exception.__cause__.errno, errno.ENAMETOOLONG)

    def test_transposed_paths_are_reported_as_precompile_errors(self):
        # load takes two same-typed positional paths, so the likeliest caller mistake is
        # swapping them: the cache's envelope then fails to decode as source, named too.
        self._write_pair()
        with self.assertRaises(PrecompileError) as cm:
            load(self.cache, self.artifact)
        self.assertIn("could not read the artifact pair", str(cm.exception))
        self.assertIn(repr(self.artifact), str(cm.exception))
        self.assertIn(repr(self.cache), str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, UnicodeDecodeError)

    def test_paths_come_in_pairs(self):
        # Half a pair never loads, so both entry points refuse one up front, before fn
        # runs (the same file for both halves is refused too, above).
        both = "neither artifact_path nor cache_path"
        cases = [
            ((self.artifact, None), "artifact_path without cache_path"),
            ((None, self.cache), "cache_path without artifact_path"),
            ((None, None), both),
        ]
        for (artifact, cache), regex in cases:
            with self.assertRaisesRegex(ValueError, regex):
                self._capture(artifact_path=artifact, cache_path=cache)
            with self.assertRaisesRegex(ValueError, regex):
                load(artifact, cache)
        self.assertEqual(os.listdir(self.dir), [])

    def test_call_outside_the_block_is_refused(self):
        # Only the block exit writes the files, so a call made without entering
        # would trace and serve and then write nothing.
        cap = self._capture()
        with self.assertRaisesRegex(PrecompileError, "capture is not active"):
            cap(self.model, self.x)
        self.assertFalse(os.path.exists(self.artifact))
        self.assertEqual(os.listdir(self.dir), [])

    def test_call_after_the_block_is_refused(self):
        # The block already wrote the pair, so a later call would trace and serve and write
        # nothing; it is not the single-call refusal.
        with self._capture() as cap:
            cap(self.model, self.x)
        with self.assertRaisesRegex(PrecompileError, "capture is spent"):
            cap(self.model, self.x)
        self.assertEqual(self._leftovers(), [])

    def test_call_after_a_caught_empty_exit_is_refused(self):
        # A caller that catches the nothing-was-captured raise and calls anyway is refused
        # too: the exit is over, so nothing would be written.
        with self.assertRaisesRegex(PrecompileError, "nothing was captured"):
            with self._capture() as cap:
                pass
        with self.assertRaisesRegex(PrecompileError, "capture is spent"):
            cap(self.model, self.x)
        self.assertEqual(os.listdir(self.dir), [])

    def test_save_outside_the_block_is_refused(self):
        # The render survives a block that raised, so save() is gated on the block being
        # live exactly like __call__ is, except as a retry of a WRITE that failed.
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with self._capture() as cap:
                cap(self.model, self.x)
                raise RuntimeError("boom")
        with self.assertRaisesRegex(PrecompileError, "capture is spent"):
            cap.save()
        self.assertEqual(os.listdir(self.dir), [])

    def test_the_block_is_entered_once(self):
        # Single-shot in both directions: a nested block would deactivate the capture on its
        # own exit, mid-use, and a second would rewrite both files from the first's render.
        with self._capture() as cap:
            with self.assertRaisesRegex(PrecompileError, "not re-entrant"):
                with cap:
                    pass
            cap(self.model, self.x)
        # A rewrite repeats the same bytes, so pin inodes: each write renames fresh temps in.
        inodes = (os.stat(self.artifact).st_ino, os.stat(self.cache).st_ino)
        with self.assertRaisesRegex(PrecompileError, "capture is spent"):
            with cap:
                pass
        rewritten = (os.stat(self.artifact).st_ino, os.stat(self.cache).st_ino)
        self.assertEqual(rewritten, inodes)
        self.assertEqual(self._leftovers(), [])

    def test_a_failed_trace_is_reported_as_a_failed_trace(self):
        # The single-call flag counts a RENDER, not an attempt: after a trace that raised
        # nothing was captured, so the retry must describe the failed trace rather than claim
        # a call was captured, and save() / the exit must not ask for a call already made.
        def data_dependent(model, x):
            return model(x) * x.sum().item()

        cap = self._capture(data_dependent)
        with self.assertRaisesRegex(PrecompileError, "raised before it rendered"):
            with cap:
                with self.assertRaisesRegex(PrecompileError, "data-dependent"):
                    cap(self.model, self.x)
                with self.assertRaisesRegex(PrecompileError, "already ran and raised"):
                    cap(self.model, self.x)
                with self.assertRaisesRegex(PrecompileError, "raised before it"):
                    cap.save()
        self.assertEqual(os.listdir(self.dir), [])

    def test_tracer_decompositions_are_used(self):
        # MakeFxTracer.decompositions reaches make_fx through capture(): the custom
        # decomposition runs during the capture, and the pair still serves.
        called = []

        def my_relu_decomp(x):
            called.append(True)
            return (x > 0) * x

        decomps = {torch.ops.aten.relu.default: my_relu_decomp}
        with self._capture(tracer=MakeFxTracer(decompositions=decomps)) as cap:
            y = cap(self.model, self.x)
        self.assertTrue(called)  # the table was used during capture
        self.assertEqual(y, self.model(self.x))
        self._assert_serves()

    def test_loaded_artifact_honors_the_standalone_contract(self):
        # The documented contract for installed=False: __enter__ hands back the same handle,
        # unload() takes nothing out (idempotent), and the handle serves the same result
        # before, inside and after the block.
        self._write_pair()
        f = load(self.artifact, self.cache)
        self.assertIsInstance(f, PrecompiledRunnable)
        self.assertFalse(f.installed)
        expected = self.model(self.x)
        with f as entered:
            self.assertIs(entered, f)
            self.assertEqual(f(self.model, self.x), expected)
        self.assertFalse(f.installed)
        self.assertEqual(f(self.model, self.x), expected)
        f.unload()
        f.unload()
        self.assertEqual(f(self.model, self.x), expected)

    def test_missing_parent_directories_are_created(self):
        nested = os.path.join(self.dir, "a", "b")
        artifact = os.path.join(nested, "m.py")
        cache = os.path.join(nested, "m.cache")
        with self._capture(artifact_path=artifact, cache_path=cache) as cap:
            cap(self.model, self.x)
        self.assertEqual(load(artifact, cache)(self.model, self.x), self.model(self.x))

    def test_carriage_return_in_the_artifact_round_trips(self):
        # The pair goes out and comes back as bytes, so a \r in python_code is not translated
        # on the way in and its code_hash still matches. Written through _write_artifact.
        self._write_pair()
        code = self._read(self.artifact).decode() + "# a trailing comment\r\n"
        blob = torch.load(self.cache, weights_only=True)
        blob["code_hash"] = hashlib.sha256(code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        torch._precompile._write_artifact(
            self.artifact, self.cache, code, buf.getvalue()
        )
        self.assertIn(b"\r\n", self._read(self.artifact))
        self._assert_serves()

    def test_callable_api_load_still_reads_an_in_memory_pair(self):
        python_code, cache = torch.compiler.precompile(
            _files_fn, self.model, self.x, backend="eager"
        )
        # The callable API traces with grad enabled, so training=True renders the
        # same python_code; the file bytes are exactly that string on every platform.
        with self._capture(training=True) as cap:
            cap(self.model, self.x)
        self.assertEqual(self._read(self.artifact), python_code.encode())
        self.assertNotIn(b"\r", self._read(self.artifact))
        read_back, _ = torch._precompile._read_artifact(self.artifact, self.cache)
        self.assertEqual(read_back, python_code)
        f = torch.compiler.precompile.load(python_code, cache)
        self.assertEqual(f(self.model, self.x), self.model(self.x))


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
        # -- the mutation handling comes from AOTAutograd's codegen -- while CAPTURE
        # leaves the example model's buffers alone.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            m.train()
            return m.to(device)

        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        example = fresh()
        bn = example[1]
        pre_rm = bn.running_mean.clone()
        pre_rv = bn.running_var.clone()
        pre_nbt = bn.num_batches_tracked.clone()
        code, cache = _precompile_pair(lambda model, xx: model(xx), example, x)
        # Capture reparametrizes the module with FAKE params/buffers (invariants 1 and 2),
        # so the example module's running stats must not move; a real trace advanced them.
        self.assertEqual(bn.running_mean, pre_rm)
        self.assertEqual(bn.running_var, pre_rv)
        self.assertEqual(bn.num_batches_tracked, pre_nbt)

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
        # Capture traces on fakes and leaves t alone; only the served artifact mutates
        # its input, so give each run its own clone of t.
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


instantiate_device_type_tests(TestPrecompileNumerics, globals())


if __name__ == "__main__":
    run_tests()
