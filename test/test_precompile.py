# Owner(s): ["oncall: pt2"]
import copy
import errno
import functools
import importlib
import inspect
import io
import os
import pickle
import stat
import subprocess
import sys
import tempfile
import textwrap
import typing
import unittest
from unittest import mock

import torch
import torch.utils._pytree as _pytree
from torch._dynamo.decorators import mark_dynamic, mark_unbacked
from torch._precompile import _write_artifact, PrecompileError
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.compiler.precompile import capture, load, MakeFxTracer, PrecompiledRunnable
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

_PRECOMPILE_PUBLIC_MEMBERS = [
    name
    for name in dir(torch.compiler.precompile)
    if not name.startswith("_") and callable(getattr(torch.compiler.precompile, name))
]


def _precompile_pair(fn, *args, **kwargs):
    """Render an in-memory (python_code, cache) pair for ``fn`` traced on ``args``."""
    from torch._precompile import PrecompiledModule

    compiled = PrecompiledModule(fn, **kwargs)
    compiled._compile(args)
    python_code = compiled.to_python_code()
    return python_code, compiled.to_cache_bytes(python_code)


def _load_pair(python_code, cache):
    """Reconstruct a runnable from an in-memory (python_code, cache) pair."""
    return torch._precompile._runnable_from_pair(python_code, cache)


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


# A module-level global the multi-graph driver test's captured function reads,
# so its EQUALS_MATCH guard is rooted at this module's live dict.
_MULTIGRAPH_SCALE = 2


# precompile drives make_fx internally, which cannot symbolically trace a
# dynamo-optimized function; the whole suite is therefore incompatible with
# PYTORCH_TEST_WITH_DYNAMO (dynamo_wrapped CI), so skip it there.
@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompile(TestCase):
    def test_guard_fact_pickle_and_hash(self):
        from torch.compiler._precompile_types import GuardFact

        # A fact is a value: pickle round-trips it and equal facts hash equal.
        fact = GuardFact(
            guard_type="ID_MATCH",
            source="G['fn']",
            code=("___check_obj_id(G['fn'], <id>), type=<class 'function'>",),
            value="is @m.py:3#abc mod.fn",
            enforced=False,
        )
        clone = pickle.loads(pickle.dumps(fact))
        self.assertEqual(clone, fact)
        self.assertEqual(hash(clone), hash(fact))
        # Keyword-only: three str fields in a row would otherwise transpose silently.
        with self.assertRaisesRegex(TypeError, "takes 1 positional argument"):
            GuardFact("ID_MATCH", "G['fn']", (), "is @m.py:3#abc mod.fn", False)

    def test_summary_pickle_and_hash(self):
        from torch.compiler._precompile_types import PrecompileSummary

        # A fully populated summary round-trips through pickle (which resolves
        # the class through its __module__) and hashes equal to its copy.
        risky = (("ID_MATCH", "self.act"),)
        policy = ("BUILTIN_MATCH", "G['__builtins_dict___<n>']['len']")
        summary = PrecompileSummary(
            frames=3,
            resume_functions=1,
            guarded_codes=4,
            backend_graphs=3,
            bypassed=("gen",),
            truncated=("loop (m.py:12)",),
            uncovered_frames=("helper",),
            wont_generalize=("n",),
            dropped_guards=(("HASATTR", "m"),) + risky,
            kept_guards=(("EQUALS_MATCH", "n"), ("TENSOR_MATCH", "x")),
            risky_dropped_guards=risky,
            policy_dropped_guards=(policy,),
            dropped_guard_code=(("HASATTR", "m", "hasattr(L['m'], 'act')"),),
            capture_errors=("RuntimeError: boom",),
        )
        clone = pickle.loads(pickle.dumps(summary))
        self.assertEqual(clone, summary)
        self.assertEqual(hash(clone), hash(summary))
        # Every clause at once: the notes come first and the shouted frame
        # failures last, so a failure never sits between two notes.
        self.assertExpectedInline(
            str(summary),
            """3 frames (1 from graph breaks), 4 guarded codes, 3 backend graphs, dropped guards {'HASATTR': 1, 'ID_MATCH': 1} (2 kept), RISKY drops ['ID_MATCH self.act'], 1 policy-dropped guard, 1 value-pinned source, 1 UNCOVERED: ['helper'], >=1 TRUNCATED: ['loop (m.py:12)'], 1 BYPASSED: ['gen'], 1 CAPTURE ERROR: 'RuntimeError: boom'""",
        )
        # Keyword-only: four leading ints would otherwise transpose silently.
        with self.assertRaisesRegex(TypeError, "takes 1 positional argument"):
            PrecompileSummary(3, 1, 4, 3)

    def test_summary_guard_lists_aggregate_over_frames(self):
        from torch.compiler._precompile_types import PrecompileSummary

        # Two frames' guards on the builtin len are one slot (the producer
        # normalizes the per-compile counter out of the builtins-dict key). One
        # frame's caller filter rejected it and the drop told that frame's
        # variants apart, so it is risky (the risky-drop lint waives a builtin
        # read the ordinary way); the other frame's invariance policy dropped
        # it, which the policy may do to a BUILTIN_MATCH: the slot sits in all
        # three lists.
        # The relations hold per frame and the type checks nothing, so the
        # report still constructs and counts the slot once.
        act = ("BUILTIN_MATCH", "G['__builtins_dict___<n>']['len']")
        check = "___check_obj_id(G['__builtins_dict___<n>']['len'], <id>), type=<class 'builtin_function_or_method'>"
        summary = PrecompileSummary(
            frames=2,
            resume_functions=0,
            guarded_codes=2,
            backend_graphs=2,
            dropped_guards=(act,),
            risky_dropped_guards=(act,),
            policy_dropped_guards=(act,),
            dropped_guard_code=(act + (check,),),
        )
        self.assertEqual(summary.dropped_guard_types, {"BUILTIN_MATCH": 1})
        self.assertExpectedInline(
            str(summary),
            """2 frames (0 from graph breaks), 2 guarded codes, 2 backend graphs, dropped guards {'BUILTIN_MATCH': 1} (0 kept), RISKY drops ["BUILTIN_MATCH G['__builtins_dict___<n>']['len']"], 1 policy-dropped guard""",
        )

    def test_summary_complete_requires_every_term(self):
        from torch.compiler._precompile_types import PrecompileSummary

        def summary(**kw):
            base = dict(frames=1, resume_functions=0, guarded_codes=1, backend_graphs=1)
            base.update(kw)
            return PrecompileSummary(**base)

        self.assertTrue(summary().complete)
        self.assertFalse(summary(backend_graphs=0).complete)
        self.assertFalse(summary(guarded_codes=0).complete)
        self.assertFalse(summary(capture_errors=("boom",)).complete)
        self.assertFalse(summary(bypassed=("f",)).complete)
        self.assertFalse(summary(truncated=("f",)).complete)
        self.assertFalse(summary(uncovered_frames=("f",)).complete)
        # Coverage only: the guard fields never make a capture incomplete.
        risky = (("ID_MATCH", "self.act"),)
        flagged = summary(dropped_guards=risky, risky_dropped_guards=risky)
        self.assertTrue(flagged.complete)
        pinned = summary(wont_generalize=("n",), kept_guards=(("EQUALS_MATCH", "n"),))
        self.assertTrue(pinned.complete)

    def test_summary_digest_and_guard_type_counts(self):
        from torch.compiler._precompile_types import PrecompileSummary

        # The fixtures list slots sorted; the tallies render in first-appearance
        # order. A value-pinned source is one a kept value-equality guard
        # on a bare name pins, so each such fixture keeps that guard too.
        policy = ("BUILTIN_MATCH", "G['__builtins_dict___<n>']['len']")
        plain = PrecompileSummary(
            frames=2,
            resume_functions=1,
            guarded_codes=3,
            backend_graphs=2,
            dropped_guards=(
                ("HASATTR", "m"),
                ("ID_MATCH", "G['fn']"),
                ("ID_MATCH", "G['g']"),
            ),
            kept_guards=(
                ("EQUALS_MATCH", "scale"),
                ("TENSOR_MATCH", "x"),
                ("TYPE_MATCH", "x"),
            ),
            policy_dropped_guards=(policy,),
            # For programmatic consumers: the digest below does not mention it.
            dropped_guard_code=(("HASATTR", "m", "hasattr(L['m'], 'act')"),),
            wont_generalize=("scale",),
        )
        self.assertEqual(plain.dropped_guard_types, {"HASATTR": 1, "ID_MATCH": 2})
        kept = {"EQUALS_MATCH": 1, "TENSOR_MATCH": 1, "TYPE_MATCH": 1}
        self.assertEqual(plain.kept_guard_types, kept)
        self.assertExpectedInline(
            str(plain),
            """2 frames (1 from graph breaks), 3 guarded codes, 2 backend graphs, dropped guards {'HASATTR': 1, 'ID_MATCH': 2} (3 kept), 1 policy-dropped guard, 1 value-pinned source""",
        )
        # No optional clause: kept guards show up only beside the drops.
        clean = PrecompileSummary(
            frames=1,
            resume_functions=0,
            guarded_codes=1,
            backend_graphs=1,
            kept_guards=(("TENSOR_MATCH", "x"),),
        )
        self.assertExpectedInline(
            str(clean),
            """1 frame (0 from graph breaks), 1 guarded code, 1 backend graph""",
        )
        # The risky slots are dropped slots too; the digest names them whole,
        # since a dropped ID_MATCH and its HASATTR companion share a source.
        # Only the first non-empty line of the first capture error is shown.
        risky = (("HASATTR", "self.act"), ("ID_MATCH", "self.act"))
        bad = PrecompileSummary(
            frames=3,
            resume_functions=0,
            guarded_codes=1,
            backend_graphs=1,
            bypassed=("gen",),
            truncated=("loop (m.py:12)",),
            uncovered_frames=("helper",),
            dropped_guards=risky,
            risky_dropped_guards=risky,
            capture_errors=("\nRuntimeError: boom\nHint: do not.",),
        )
        self.assertExpectedInline(
            str(bad),
            """3 frames (0 from graph breaks), 1 guarded code, 1 backend graph, dropped guards {'HASATTR': 1, 'ID_MATCH': 1} (0 kept), RISKY drops ['HASATTR self.act', 'ID_MATCH self.act'], 1 UNCOVERED: ['helper'], >=1 TRUNCATED: ['loop (m.py:12)'], 1 BYPASSED: ['gen'], 1 CAPTURE ERROR: 'RuntimeError: boom'""",
        )
        # The list clauses stop at five entries and count the rest, so the
        # digest stays one line however many frames a model has.
        wide = PrecompileSummary(
            frames=8,
            resume_functions=0,
            guarded_codes=1,
            backend_graphs=1,
            uncovered_frames=tuple(f"f{i}" for i in range(7)),
            dropped_guards=risky,
            risky_dropped_guards=risky,
            capture_errors=("RuntimeError: boom", "TypeError: bad", "ValueError: no"),
        )
        self.assertExpectedInline(
            str(wide),
            """8 frames (0 from graph breaks), 1 guarded code, 1 backend graph, dropped guards {'HASATTR': 1, 'ID_MATCH': 1} (0 kept), RISKY drops ['HASATTR self.act', 'ID_MATCH self.act'], 7 UNCOVERED: ['f0', 'f1', 'f2', 'f3', 'f4'] +2 more, 3 CAPTURE ERRORS: 'RuntimeError: boom' +2 more""",
        )

    @parametrize("mode", ["make_fx", "other", "dynamo", "installed"])
    def test_parse_artifact_metadata_required_set_follows_tracer(self, mode):
        # TRACER picks which calling-convention constants an artifact must carry
        # (absent or anything but "dynamo" means make_fx), and an installed dynamo
        # artifact swaps the per-frame blobs for the package blob.
        from torch._precompile import _parse_artifact_metadata

        make_fx = (
            ["BUFFER_NAMES", "OUT_SPEC", "USER_INPUT_BOUNDS"],
            ["TRACER", "_FRAMES", "_ENTRY_BINDING"],
        )
        src, required, not_required = {
            "make_fx": ("BACKEND = 'inductor'\n", *make_fx),
            "other": ("TRACER = 'other'\n", *make_fx),
            "dynamo": (
                "TRACER = 'dynamo'\n",
                ["FN_NAME", "_FRAMES", "_BACKENDS", "_ENTRY_BINDING", "TORCH_VERSION"],
                ["OUT_SPEC", "TRACER", "SERVING_MODE", "_PACKAGE"],
            ),
            "installed": (
                "TRACER = 'dynamo'\nSERVING_MODE = 'installed'\n",
                ["_PACKAGE", "UNREACHABLE_WITHOUT_INSTALL", "_ENTRY_BINDING"],
                ["OUT_SPEC", "_FRAMES", "_BACKENDS"],
            ),
        }[mode]
        with self.assertRaises(PrecompileError) as cm:
            _parse_artifact_metadata(src)
        msg = str(cm.exception)
        self.assertIn("missing calling-convention metadata", msg)
        for name in required:
            self.assertIn(repr(name), msg)
        for name in not_required:
            self.assertNotIn(repr(name), msg)

    def test_parse_artifact_metadata_literals(self):
        from torch._precompile import _parse_artifact_metadata

        fields = [
            ("BACKEND", "eager"),
            ("TRACER", "dynamo"),
            ("FN_NAME", "step"),
            ("FRAMES", [{"is_entry": True, "variants": []}]),
            ("DROPPED_GUARDS", []),
            ("RISKY_DROPPED_GUARDS", []),
            ("WONT_GENERALIZE", ()),
            ("_FRAMES", "blob"),
            ("_BACKENDS", "blob"),
            ("_DYNAMO_PYTHON_VERSION", "3.12"),
            ("_ENTRY_BINDING", "step"),
            ("TORCH_VERSION", "2.0"),
        ]
        src = "".join(f"{name} = {value!r}\n" for name, value in fields)
        meta = _parse_artifact_metadata(src)
        self.assertEqual(meta["FRAMES"], [{"is_entry": True, "variants": []}])
        # Reported but never required: the serving mode defaults for artifacts
        # predating it, and the guard-audit sections come back as data.
        self.assertEqual(meta["SERVING_MODE"], "standalone")
        self.assertNotIn("POLICY_DROPPED_GUARDS", meta)
        audit = "POLICY_DROPPED_GUARDS = ['g']\nDROPPED_GUARD_CODE = {'g': 'code'}\n"
        meta = _parse_artifact_metadata(src + audit)
        self.assertEqual(meta["POLICY_DROPPED_GUARDS"], ["g"])
        self.assertEqual(meta["DROPPED_GUARD_CODE"], {"g": "code"})
        # An installed artifact parses without the per-frame blobs.
        blobs = "_FRAMES = 'blob'\n_BACKENDS = 'blob'\n"
        package = "SERVING_MODE = 'installed'\n_PACKAGE = 'pkg'\n"
        installed = src.replace(blobs, package) + "UNREACHABLE_WITHOUT_INSTALL = []\n"
        meta = _parse_artifact_metadata(installed)
        self.assertEqual((meta["SERVING_MODE"], meta["_PACKAGE"]), ("installed", "pkg"))
        # The last top-level assignment wins for the set selection and the
        # reported value alike, as it would under exec.
        shadowed = src.replace("TRACER = 'dynamo'", "TRACER = 'other'")
        meta = _parse_artifact_metadata(shadowed + "TRACER = 'dynamo'\n")
        self.assertEqual(meta["TRACER"], "dynamo")
        # A consumed name whose value is not a literal (a call, or a set with an
        # unhashable member) is named, including the ones that select the required
        # set; an unconsumed one is skipped.
        for bad, name in (
            ("TRACER = object()\n", "TRACER"),
            ("TRACER = 'dynamo'\nSERVING_MODE = {[]}\n", "SERVING_MODE"),
        ):
            with self.assertRaisesRegex(PrecompileError, f"{name!r} .* is malformed"):
                _parse_artifact_metadata(bad)
        meta = _parse_artifact_metadata(src + "_x = f()\n")
        self.assertEqual(meta["TRACER"], "dynamo")

    @parametrize("backend", ["eager", "inductor"])
    def test_artifact_neutralizes_ambient_autocast(self, backend):
        # The casts a capture ran under are baked into the artifact, but the graph
        # still re-dispatches at serve time, so the driver runs it with autocast
        # excluded and leaves the caller's autocast state as it found it. addbmm is
        # AutocastCPU-registered and an inductor fallback with no decomposition, so
        # both artifacts re-dispatch through aten.addbmm.default and each backend's
        # guard is load-bearing (without it inductor's assert_tensor_metadata trips).
        from torch._precompile import PrecompiledModule

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self, x):
                return torch.addbmm(self.bias, x, x.transpose(1, 2))

        model = Model()
        x = torch.randn(2, 3, 4)
        expected = model(x)
        compiled = PrecompiledModule(lambda m, x: m(x), backend=backend)
        compiled._compile((model, x))
        ns: dict[str, object] = {"__name__": "precompile_test_artifact"}
        exec(compile(compiled.to_python_code(), "<artifact>", "exec"), ns)
        forward = ns["forward"]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = forward(model, x)
            self.assertEqual(model(x).dtype, torch.bfloat16)
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out, expected)

    def test_precompiled_module_is_a_standalone_runnable(self):
        # A loaded make_fx artifact is the standalone PrecompiledRunnable: it
        # installs nothing, so entering and unloading it are no-ops, and it hands
        # positional and keyword arguments alike to the loaded forward.
        from torch._precompile import PrecompiledModule, PrecompiledRunnable

        f = PrecompiledModule._from_loaded(lambda *a, **k: (a, k), backend="eager")
        self.assertIsInstance(f, PrecompiledRunnable)
        self.assertFalse(f.installed)
        with f as entered:
            self.assertIs(entered, f)
            self.assertEqual(f(1, k=2), ((1,), {"k": 2}))
        f.unload()
        self.assertEqual(f(3), ((3,), {}))
        with self.assertRaisesRegex(PrecompileError, "not runnable"):
            PrecompiledModule(lambda x: x)(1)

    def test_inlined_forward_warns_unless_told_not_to(self):
        # exec of an artifact is untrusted input on the load path and warns on
        # every load; only the capture-time self-load of source this process just
        # rendered turns the warning off.
        from torch._precompile import _make_inlined_forward

        code = "def forward(x):\n    return x + 1\n"
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            self.assertEqual(_make_inlined_forward(code)(1), 2)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("about to EXEC python_code", cm.output[0])
        with self.assertNoLogs("torch._precompile", level="WARNING"):
            self.assertEqual(_make_inlined_forward(code, warn=False)(1), 2)

    def test_multigraph_frames_record_every_dynamo_frame(self):
        # The entry is codes[0], where CompilePackage records the captured
        # callable and reads it back from, so neither a bypassed entry nor a
        # same-named helper frame moves it. A bypassed code keeps its record
        # without variants: the frame ahead of a bypassed continuation still
        # LOAD_GLOBALs its resume name, and its guarded codes are dead.
        from torch._dynamo.package import (
            _DynamoCacheEntry,
            _DynamoCodeCacheEntry,
            _GuardedCodeCacheEntry,
            SerializedCode,
            SourceInfo,
        )
        from torch._precompile import _multigraph_frames

        def forward(x):
            return x

        def helper(x):
            return x

        def code_entry(fn, resume_name=None, variants=(), bypassed=False):
            return _DynamoCodeCacheEntry(
                python_code=SerializedCode.from_code_object(fn.__code__),
                python_module=__name__,
                function_names=[resume_name] if resume_name else [],
                guarded_codes=list(variants),
                import_sources={"__import_torch": "torch"},
                backend_ids=[],
                code_source=None,
                install_to_global=resume_name is not None,
                bypassed=bypassed,
            )

        variant = _GuardedCodeCacheEntry(
            guards_state=b"",
            dynamo_code=SerializedCode.from_code_object(helper.__code__),
        )
        entry = _DynamoCacheEntry(
            codes=[
                code_entry(forward, variants=[variant], bypassed=True),
                code_entry(forward),  # a submodule's forward: same co_name
                code_entry(helper, "__resume_at_12_3", [variant]),
                code_entry(helper, "__resume_at_40_7", [variant], bypassed=True),
            ],
            source_info=SourceInfo(inlined_sources=set()),
            device_type="cpu",
            fn_name="Model.forward",
        )
        frames = _multigraph_frames(entry)
        self.assertEqual([f["is_entry"] for f in frames], [True, False, False, False])
        self.assertEqual([f["bypassed"] for f in frames], [True, False, False, True])
        self.assertEqual(
            [f["resume_names"] for f in frames],
            [[], [], ["__resume_at_12_3"], ["__resume_at_40_7"]],
        )
        self.assertEqual([len(f["variants"]) for f in frames], [0, 0, 1, 0])
        self.assertEqual(frames[2]["variants"][0]["dynamo_code"], variant.dynamo_code)
        self.assertEqual(frames[0]["code"].co_name, "forward")
        self.assertEqual(frames[0]["import_sources"], {"__import_torch": "torch"})

    def test_reachable_frames_follow_resume_names(self):
        # A continuation is reachable only through a reachable parent's bytecode
        # (nested code objects included), so carrying a resume name is not
        # enough: one named only by an unreachable helper is just as dead.
        from torch._dynamo.package import SerializedCode
        from torch._precompile import _reachable_frames, _serving_mode

        ns = {}
        exec(
            "def entry(x): return __resume_at_12_3(x)\n"
            "def cont_a(x): return (lambda: __resume_at_40_7(x))()\n"
            "def cont_b(x): return x\n"
            "def helper(x): return __resume_at_99_1(x)\n",
            ns,
        )

        def frame(name, resume_names=(), is_entry=False, nvariants=1):
            code = SerializedCode.from_code_object(ns[name].__code__)
            return {
                "is_entry": is_entry,
                "bypassed": nvariants == 0,
                "code": code,
                "python_module": "m",
                "import_sources": {},
                "resume_names": list(resume_names),
                "variants": [{"guards_state": b"", "dynamo_code": code}] * nvariants,
            }

        entry = frame("entry", is_entry=True)
        cont_a = frame("cont_a", ["__resume_at_12_3"])
        cont_b = frame("cont_b", ["__resume_at_40_7"])
        helper = frame("helper")
        # orphan is named only by the unreachable helper; nothing names stray.
        orphan = frame("cont_b", ["__resume_at_99_1"])
        stray = frame("cont_b", ["__resume_at_7_7"])
        frames = [entry, cont_a, cont_b, helper, orphan, stray]
        self.assertEqual(_reachable_frames(frames), {0, 1, 2})
        self.assertEqual(_serving_mode(frames), "installed")
        self.assertEqual(_serving_mode([entry, cont_a, cont_b]), "standalone")
        # A reachable continuation with no variant (bypassed) would raise on the
        # captured path in a standalone artifact, so that capture installs.
        bypassed = frame("cont_a", ["__resume_at_12_3"], nvariants=0)
        self.assertEqual(_reachable_frames([entry, bypassed, cont_b]), {0, 1})
        self.assertEqual(_serving_mode([entry, bypassed, cont_b]), "installed")

    def test_no_dispatchable_graph_names_the_cause(self):
        # An entry frame with no variants has two very different causes. If
        # Dynamo BYPASSED the frame, saying so beats the thin-wrapper advice,
        # which in that case is simply wrong; a bypassed helper or continuation
        # says nothing about the entry.
        from torch._dynamo.package import (
            _DynamoCacheEntry,
            _DynamoCodeCacheEntry,
            _GuardedCodeCacheEntry,
            SerializedCode,
            SourceInfo,
        )
        from torch._precompile import _multigraph_frames, _reject_uninstallable_entry

        def fwd_loss_bwd(x):
            return x

        def helper(x):
            return x

        scale = 2

        def step(x):
            return x * scale

        def code_entry(fn, resume_name=None, variants=(), bypassed=False):
            return _DynamoCodeCacheEntry(
                python_code=SerializedCode.from_code_object(fn.__code__),
                python_module=__name__,
                function_names=[resume_name] if resume_name else [],
                guarded_codes=list(variants),
                import_sources={},
                backend_ids=[],
                code_source=None,
                install_to_global=resume_name is not None,
                bypassed=bypassed,
            )

        def reject(fn_name, codes):
            entry = _DynamoCacheEntry(
                codes=codes,
                source_info=SourceInfo(inlined_sources=set()),
                device_type="cpu",
                fn_name=fn_name,
            )
            _reject_uninstallable_entry(_multigraph_frames(entry), entry)

        variant = _GuardedCodeCacheEntry(
            guards_state=b"",
            dynamo_code=SerializedCode.from_code_object(helper.__code__),
        )
        healthy = code_entry(fwd_loss_bwd, variants=[variant])
        bypassed = code_entry(fwd_loss_bwd, variants=[variant], bypassed=True)
        thin = code_entry(fwd_loss_bwd)
        dead_helper = code_entry(helper, bypassed=True)
        dead_cont = code_entry(helper, "__resume_at_12_3", bypassed=True)
        msg = "entry frame was BYPASSED during capture.*precompile_cache_bypass"
        with self.assertRaisesRegex(PrecompileError, msg):
            reject("fwd_loss_bwd", [bypassed])
        # A bypassed HELPER beside a variant-less entry is the thin-wrapper case.
        with self.assertRaisesRegex(PrecompileError, "thin wrapper"):
            reject("fwd_loss_bwd", [thin, dead_helper])
        # A healthy entry beside a bypassed continuation has nothing to refuse,
        # and neither has an empty capture.
        reject("fwd_loss_bwd", [healthy, dead_cont])
        reject("fwd_loss_bwd", [])
        with self.assertRaisesRegex(PrecompileError, r"closes over \['scale'\]"):
            reject("step", [code_entry(step, variants=[variant])])

    def test_multigraph_driver_dispatches_entry_frame(self):
        # The driver rebuilds the entry frame's f_locals (a keyword-only default
        # the call omits, *args) and guards them against this module's LIVE dict.
        import inspect
        from unittest import mock

        from torch import _precompile_driver as driver
        from torch._dynamo.output_graph import get_builtins_dict
        from torch._dynamo.package import (
            CompilePackage,
            load_guards_state,
            SerializedCode,
        )
        from torch._dynamo.precompile_context import EagerCacheArtifact
        from torch._dynamo.precompile_package import default_guard_filter_fn
        from torch._precompile import _b64, _multigraph_frames, _serving_mode

        def step(model, x, *rest, scale=2.0):
            return model(x) * scale * _MULTIGRAPH_SCALE + len(rest)

        model = torch.nn.Linear(4, 4)
        x = torch.randn(3, 4)
        package = CompilePackage(step)
        compiled = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=default_guard_filter_fn
        )(step)
        # len(rest) is a constant in the graph, so the second call recompiles the
        # entry: two variants whose outputs differ by one, and dispatch has to pick.
        expected = compiled(model, x)
        expected_rest = compiled(model, x, torch.ones(1))
        self.assertNotEqual(expected, expected_rest)
        frames = _multigraph_frames(package.cache_entry())
        self.assertEqual(_serving_mode(frames), "standalone")
        self.assertEqual([len(f["variants"]) for f in frames], [2])
        backends = {
            backend_id: EagerCacheArtifact(key=backend_id, content=backend)
            for backend_id, backend in package.cached_backends.items()
        }
        torch._dynamo.reset()
        # A serving process never traced, so the names Dynamo minted into this
        # module during capture must not be what makes the guards pass.
        scope = step.__globals__
        minted = ("__compiled_fn", "__builtins_dict__", "__import_")

        def scrub():
            return {n: scope.pop(n) for n in list(scope) if n.startswith(minted)}

        removed = scrub()
        self.addCleanup(lambda: (scrub(), scope.update(removed)))
        # The records name this module, which is __main__ under a script run and
        # the driver refuses that; serve them from an importable alias of it.
        module = "precompile_test_captured_module"
        alias = mock.patch.dict(sys.modules, {module: sys.modules[__name__]})
        self.addCleanup(alias.stop)
        alias.start()
        for frame in frames:
            frame["python_module"] = module
        binding = {"defaults": step.__defaults__, "kwdefaults": step.__kwdefaults__}
        ns = {
            "__name__": "precompile_test_artifact",
            "_FRAMES": _b64(frames),
            "_BACKENDS": _b64(backends),
            "_ENTRY_BINDING": _b64(binding),
            "_DYNAMO_PYTHON_VERSION": tuple(sys.version_info[:2]),
            "TORCH_VERSION": torch.__version__,
        }
        exec(inspect.getsource(driver._build_multigraph_forward), ns)
        build = ns["_build_multigraph_forward"]
        forward = build()
        self.assertEqual(forward(model, x), expected)
        self.assertEqual(forward(model, x, torch.ones(1)), expected_rest)
        # Rebinding a global the graph baked in has to fail its guard and refuse,
        # not serve the stale graph (restored, it serves again); a call neither
        # variant covers refuses the same way: no compiler backs the artifact.
        entry_miss = "no captured variant of 'step'"
        with mock.patch.dict(scope, {"_MULTIGRAPH_SCALE": 3}):
            with self.assertRaisesRegex(PrecompileError, entry_miss):
                forward(model, x)
        self.assertEqual(forward(model, x), expected)
        with self.assertRaisesRegex(PrecompileError, entry_miss):
            forward(model, x, scale=3.0)
        # The builtins dict Dynamo guards through is re-minted into the captured
        # module, not the artifact's namespace, and a key bound to anything
        # else is refused.
        guards_state = load_guards_state(frames[0]["variants"][0]["guards_state"])
        key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
        self.assertNotIn(key, ns)
        self.assertIs(scope[key], get_builtins_dict(scope))
        with mock.patch.dict(scope, {key: {}}):
            with self.assertRaisesRegex(PrecompileError, "other than the module's"):
                build()
        # Every other refusal is diagnosed by its cause at build: the two version
        # locks, a __main__ or unimportable module, an entry with no variant
        # (BYPASSED, or trivial and run eager), a closure entry, no entry at all.

        def record(**edits):
            return _b64([{**frames[0], **edits}])

        def closes_over_model():
            return model

        closure = SerializedCode.from_code_object(closes_over_model.__code__)
        refused = (
            ("_DYNAMO_PYTHON_VERSION", (3, 9), "produced on Python 3.9"),
            ("TORCH_VERSION", "0.0", "produced by torch 0.0"),
            ("_FRAMES", record(python_module="__main__"), "__main__ module"),
            ("_FRAMES", record(python_module="test_missing_module"), "not importable"),
            ("_FRAMES", record(bypassed=True, variants=[]), "'step' was BYPASSED"),
            ("_FRAMES", record(variants=[]), "produced no guarded code"),
            ("_FRAMES", record(code=closure), "closes over"),
            ("_FRAMES", record(is_entry=False), "no entry frame"),
        )
        for name, value, message in refused:
            with mock.patch.dict(ns, {name: value}):
                with self.assertRaisesRegex(PrecompileError, message):
                    build()

    def test_multigraph_driver_dispatches_captured_frames(self):
        # The standalone driver rebuilds each frame's f_locals for the guard
        # check, so the shapes it has to bind are all here: a keyword-only
        # default the call omits, *args, a continuation closing over a cell of
        # the entry frame (y, which rows() captures), and a module global the
        # entry reads (_MULTIGRAPH_SCALE, guarded by EQUALS_MATCH).
        import inspect
        from unittest import mock

        from torch import _precompile_driver as driver
        from torch._dynamo.package import CompilePackage
        from torch._dynamo.precompile_context import EagerCacheArtifact
        from torch._dynamo.precompile_package import default_guard_filter_fn
        from torch._precompile import _b64, _multigraph_frames, _serving_mode

        def step(model, x, *rest, scale=2.0):
            y = model(x) * scale * _MULTIGRAPH_SCALE
            torch._dynamo.graph_break()

            def rows():
                # Not x: a captured argument keeps a fast-local slot beside the
                # continuation's free var, and on 3.13+ the LOAD_FAST closure
                # load reads that slot, which Dynamo dropped, skipping the frame.
                return y.shape[0]

            return y + rows() + len(rest)

        model = torch.nn.Linear(4, 4)
        x = torch.randn(3, 4)
        package = CompilePackage(step)
        compiled = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=default_guard_filter_fn
        )(step)
        # The second call recompiles both frames (rest is an entry local guarded
        # at length 0; len(rest) is a constant in the continuation's graph), so
        # the continuation carries two variants whose outputs differ by one:
        # dispatch has to pick, not just run.
        expected = compiled(model, x)
        expected_rest = compiled(model, x, torch.ones(1))
        self.assertNotEqual(expected, expected_rest)
        frames = _multigraph_frames(package.cache_entry())
        shape = [(f["bypassed"], len(f["variants"])) for f in frames]
        self.assertEqual(_serving_mode(frames), "standalone", shape)
        self.assertGreater(len(frames[1]["variants"]), 1)
        backends = {
            backend_id: EagerCacheArtifact(key=backend_id, content=backend)
            for backend_id, backend in package.cached_backends.items()
        }
        torch._dynamo.reset()
        # The records name this module, which is __main__ under a script run and
        # the driver refuses that; serve them from an importable alias of it.
        module = "precompile_test_captured_module"
        alias = mock.patch.dict(sys.modules, {module: sys.modules[__name__]})
        self.addCleanup(alias.stop)
        alias.start()
        for frame in frames:
            frame["python_module"] = module

        ns = {
            "__name__": "precompile_test_artifact",
            # An artifact-namespace name the captured module also binds: the
            # module's binding is the one the rebuilt bytecode must LOAD_GLOBAL.
            "torch": None,
            "_FRAMES": _b64(frames),
            "_BACKENDS": _b64(backends),
            "_ENTRY_BINDING": _b64(
                {"defaults": step.__defaults__, "kwdefaults": step.__kwdefaults__}
            ),
            "_DYNAMO_PYTHON_VERSION": tuple(sys.version_info[:2]),
            "TORCH_VERSION": torch.__version__,
        }
        exec(inspect.getsource(driver._build_multigraph_forward), ns)
        build = ns["_build_multigraph_forward"]
        # In the process that captured, the live compile of step still holds the
        # continuation's resume name: the load refuses, before seeding anything,
        # and sends the user to a fresh process (which the scrub below stands for).
        with self.assertRaisesRegex(PrecompileError, "fresh process"):
            build()
        # A serving process never traced, so the names Dynamo minted into this
        # module during capture must not be what makes the guards pass; the
        # driver binds the same names, so the cleanup drops those too.
        scope = step.__globals__
        minted = ("__compiled_fn", "__resume_at", "__builtins_dict__", "__import_")

        def scrub():
            return {n: scope.pop(n) for n in list(scope) if n.startswith(minted)}

        removed = scrub()
        self.addCleanup(lambda: (scrub(), scope.update(removed)))
        forward = build()
        self.assertEqual(forward(model, x), expected)
        self.assertEqual(forward(model, x, scale=2.0), expected)
        self.assertEqual(forward(model, x, torch.ones(1)), expected_rest)
        # The guards check the captured module's LIVE globals: the graph baked
        # _MULTIGRAPH_SCALE in as a constant, so rebinding it after load has to
        # fail the guard and refuse, not serve the stale graph; the original
        # binding serves again once restored.
        entry_miss = "no captured variant of 'step'"
        with mock.patch.dict(scope, {"_MULTIGRAPH_SCALE": 3}):
            with self.assertRaisesRegex(PrecompileError, entry_miss):
                forward(model, x)
        self.assertEqual(forward(model, x), expected)
        # Neither call was captured: the entry frame refuses the first, the
        # continuation (guarding len(rest)) the second. There is no compiler
        # behind the artifact, so both are coverage gaps rather than recompiles.
        with self.assertRaisesRegex(PrecompileError, entry_miss):
            forward(model, x, scale=3.0)
        resume_miss = "no captured variant of 'torch_dynamo_resume_in_step"
        with self.assertRaisesRegex(PrecompileError, resume_miss):
            forward(model, x, torch.ones(1), torch.ones(1))
        # Rebuilding the same artifact rebinds its names. Another capture of this
        # module mints the same resume names (backend ids carry a uuid, resume
        # names only the per-process counter), so loading it beside a live one is
        # refused on the resume name and the live one keeps serving; the rows
        # below scrub the live artifact first.
        self.assertEqual(build()(model, x), expected)
        # An untagged holder (a user binding, a live compile) refuses: rebinding
        # would repoint its LOAD_GLOBAL.
        resume_name = frames[1]["resume_names"][0]
        with mock.patch.dict(scope, {resume_name: lambda *args: None}):
            with self.assertRaisesRegex(PrecompileError, resume_name):
                build()
        # Two artifacts of one capture keep the backend ids and differ only in
        # _BACKENDS (here: the subgraphs rotated across the ids): the tag covers
        # both, and the refusal precedes any seeding, so the live artifact's
        # subgraph bindings are untouched and it still serves its own answer.
        subgraphs = list(backends.values())
        twin = _b64(dict(zip(backends, subgraphs[1:] + subgraphs[:1])))
        with mock.patch.dict(ns, {"_BACKENDS": twin}):
            with self.assertRaisesRegex(PrecompileError, "only one standalone"):
                build()
        self.assertEqual(forward(model, x), expected)
        other = _b64({f"{k}_other": v for k, v in backends.items()})
        trivial = [frames[0], {**frames[1], "variants": []}]
        with mock.patch.dict(ns, {"_FRAMES": _b64(trivial)}):
            with mock.patch.dict(ns, {"_BACKENDS": other}):
                with self.assertRaisesRegex(PrecompileError, "'__resume_at_"):
                    build()
            self.assertEqual(forward(model, x), expected)
            # A zero-variant continuation is diagnosed by cause (trivial or
            # BYPASSED), not as a coverage gap, at the call that reaches it; the
            # build succeeds where the entry would refuse.
            scrub()
            with self.assertRaisesRegex(PrecompileError, "produced no guarded code"):
                build()(model, x)
        bypassed = [frames[0], {**frames[1], "bypassed": True, "variants": []}]
        with mock.patch.dict(ns, {"_FRAMES": _b64(bypassed)}):
            scrub()
            forward_bypassed = build()
        with self.assertRaisesRegex(PrecompileError, "was BYPASSED"):
            forward_bypassed(model, x)
        # A dead record from a module this process cannot import is not what the
        # artifact dispatches, so it neither refuses the load nor binds anything.
        dead = {**frames[1], "variants": [], "bypassed": True}
        dead["python_module"] = "precompile_test_no_such_module"
        dead["resume_names"] = ["__resume_at_dead"]
        with mock.patch.dict(ns, {"_FRAMES": _b64(frames + [dead])}):
            scrub()
            self.assertEqual(build()(model, x), expected)
        self.assertNotIn("__resume_at_dead", scope)

    def test_entry_binding_records_defaults(self):
        from torch._precompile import _entry_binding

        def step(model, x, scale=2.0, *, mode="sum"):
            return model(x) * scale

        def plain(model, x):
            return model(x)

        self.assertEqual(
            _entry_binding(step), {"defaults": (2.0,), "kwdefaults": {"mode": "sum"}}
        )
        self.assertEqual(_entry_binding(plain), {"defaults": None, "kwdefaults": None})

    def test_emit_multigraph_driver_source_is_a_complete_section(self):
        import ast

        from torch._precompile import _DRIVER_MAIN, _emit_multigraph_driver_source

        source = _emit_multigraph_driver_source()
        self.assertTrue(source.endswith(_DRIVER_MAIN))
        body = ast.parse(source).body
        kinds = [type(node).__name__ for node in body]
        self.assertEqual(kinds, ["FunctionDef", "Assign", "If"])
        self.assertEqual(body[0].name, "_build_multigraph_forward")
        self.assertEqual(ast.unparse(body[1]), "forward = _build_multigraph_forward()")

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
        code, cache = _precompile_pair(
            lambda model, x: model(x), m, x, decompositions=decomps
        )
        self.assertTrue(called)  # the table was used during capture

        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_constant_tensor_is_rejected(self):
        captured = torch.randn(3)
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            _precompile_pair(lambda x: x + captured, torch.randn(3))

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
            _precompile_pair(f, torch.randn(3))

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
            _precompile_pair(lambda model, x: model(x), m, torch.randn(2, 4))

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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)

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
        code, _cache = _precompile_pair(lambda model, x: model(x), m, x)

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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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

            code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)

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
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_inlined_fallback_when_artifact_absent(self):
        # When the cache holds no serialized artifact, load() falls back to
        # executing the inlined python (recompiling kernels). Force that branch by
        # stripping the artifact and check it still matches eager; this also
        # exercises the self-contained inlined path (JIT from inlined source).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)

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
        _code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda model, p: model(p.x + p.y), m, inp)
        self.assertIn("IN_SPEC = None", code)
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.x + inp.y))

    def test_unserializable_context_in_spec_still_compiles(self):
        # A registered pytree node whose context is not JSON-dumpable makes
        # treespec_dumps raise TypeError (not NotImplementedError); IN_SPEC must still
        # degrade to None rather than crashing precompile.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = _precompile_pair(lambda model, h: model(h.a + h.b), m, inp)
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
            _precompile_pair(lambda x: Out(x + 1, x + 2), torch.randn(4))

    def test_input_leaf_count_mismatch_rejected_when_spec_unserializable(self):
        # When IN_SPEC degrades to None the structural in_spec check is skipped; a runtime
        # input flattening to a DIFFERENT leaf count must still raise a clean
        # PrecompileError (not a raw zip/unpack error) on the live and eager-inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        for backend in ("inductor", "eager"):
            code, cache = _precompile_pair(
                lambda model, h: model(h.a + h.b), m, inp, backend=backend
            )
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

        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
        f_c = _load_pair(code, cache)
        f_i = _load_pair(code, _strip_artifact(cache))
        code_e, cache_e = _precompile_pair(lambda mm, t: mm(t), m, x, backend="eager")
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
                _precompile_pair(
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
            code, cache = _precompile_pair(
                lambda model, xx: RNT(model(xx), model(xx) + 1), m, x, backend=backend
            )
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

        code, cache = _precompile_pair(step, a, b, c, x, target)

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
        icode, icache = _precompile_pair(step, a, b, c, x, target)
        ia, ib, ic = copy.deepcopy((a, b, c))
        _load_pair(icode, icache)(ia, ib, ic, x, target)  # inductor cached path

        ecode, ecache = _precompile_pair(step, a, b, c, x, target, backend="eager")
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda xx, model: model(xx), x, m)
        inlined_cache = _strip_artifact(cache)  # force the inlined path
        ecode, ecache = _precompile_pair(
            lambda xx, model: model(xx), x, m, backend="eager"
        )
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
            _precompile_pair(lambda model, x: model(x).backward(), m, x)

    def test_user_input_requiring_grad_rejected(self):
        # Sibling of the buffer guard: a requires_grad USER INPUT (not a param) that
        # receives a gradient during the traced backward is not harvested (only params
        # are), so precompile rejects it rather than silently dropping the grad.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaisesRegex(PrecompileError, "user input received a gradient"):
            _precompile_pair(lambda t: (t * t).sum().backward(), x)

    def test_control_flow_subgraph_rejected(self):
        # torch.cond captures as a HOP with get_attr subgraph submodules, which the
        # standalone artifact cannot inline; reject it at capture with a clear message.
        def f(x):
            return torch.cond(x.sum() > 0, lambda t: t + 1, lambda t: t - 1, (x,))

        with self.assertRaisesRegex(PrecompileError, "control-flow subgraph"):
            _precompile_pair(f, torch.randn(4))

    def test_load_falls_back_when_cache_unreconstructable(self):
        # The cache is only an acceleration; python_code always runs standalone. A
        # corrupt / stale cache must degrade to the inlined JIT path, not crash.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, _cache = _precompile_pair(lambda model, x: model(x), m, x)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x)
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                _load_pair(code, cache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"cached load did not warn about untrusted input: {cm.output}",
            )
        # Eager backend (empty cache, nothing to prime): load() still EXECs python_code
        # via _make_inlined_forward, which warns about exec'ing untrusted code every load.
        ecode, ecache = _precompile_pair(
            lambda model, t: model(t), m, x, backend="eager"
        )
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
                _precompile_pair(fn, x)
        # The eager backend handles a passthrough and a constant fn.
        code, cache = _precompile_pair(lambda xx: xx, x, backend="eager")
        self.assertEqual(_load_pair(code, cache)(x), x)
        code, cache = _precompile_pair(lambda xx: 7, x, backend="eager")
        self.assertEqual(_load_pair(code, cache)(x), 7)

    def test_same_count_different_structure_rejected(self):
        # Invariant 2: the structural check now compares the baked PARAM_NAMES /
        # BUFFER_NAMES against the runtime model's extracted param/buffer names, so a
        # same-count-but-different-structure (here, differently-NAMED submodules) model
        # is REJECTED rather than silently running the traced graph with the wrong
        # weights. Both the cached and the inlined (artifact-stripped) load paths fire.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        code, cache = _precompile_pair(lambda m, x: m(x), a, x)
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
        code, cache = _precompile_pair(lambda m, x: m(x), a, x, backend="eager")
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
                    _precompile_pair(
                        lambda a: torch.ops.mlprecompile.eff(a), torch.randn(4)
                    )
            finally:
                _register_effectful_op(op, None)

    def test_tracer_default_and_explicit_make_fx(self):
        # tracer defaults to "make_fx"; passing it explicitly is equivalent and works.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        for kwargs in ({}, {"tracer": "make_fx"}):
            code, cache = _precompile_pair(lambda model, xx: model(xx), m, x, **kwargs)
            self.assertEqual(_load_pair(code, cache)(m, x), m(x))

    def test_tracer_dynamo_not_implemented(self):
        # "dynamo" is a valid (planned) tracer value but is not implemented yet; it must
        # raise NotImplementedError, not silently fall back to make_fx.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with self.assertRaisesRegex(NotImplementedError, "tracer='dynamo'"):
            _precompile_pair(lambda model, xx: model(xx), m, x, tracer="dynamo")

    def test_backend_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(
            ValueError, "backend must be 'inductor' or 'eager'"
        ):
            _precompile_pair(lambda x, y: x + y, a, b, backend="nope")

    def test_tracer_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(ValueError, "tracer must be 'make_fx' or 'dynamo'"):
            _precompile_pair(lambda x, y: x + y, a, b, tracer="nope")

    def test_artifact_backend_value_is_validated_before_exec(self):
        # A source whose BACKEND tag names an unknown backend is refused as a
        # malformed artifact, with the loader's own error type and before any
        # exec: the source is parsed before the cache is read.
        from torch._precompile import _parse_artifact_metadata

        m, x = torch.nn.Linear(4, 3).eval(), torch.randn(2, 4)
        code, cache = _precompile_pair(
            lambda model, xx: model(xx), m, x, backend="eager"
        )
        bad_code = code.replace("BACKEND = 'eager'", "BACKEND = 'nope'")
        self.assertNotEqual(bad_code, code)
        with self.assertRaisesRegex(PrecompileError, "unknown backend 'nope'"):
            _parse_artifact_metadata(bad_code)
        with self.assertRaisesRegex(PrecompileError, "unknown backend 'nope'"):
            _load_pair(bad_code, cache)

    def test_backend_default_is_inductor(self):
        # The default lowers through Inductor: the generated code inlines the Inductor
        # output module. Use a graph_partition-agnostic marker (the ``call = runner.call``
        # form is only emitted when config.graph_partition is on, which is off in fbcode).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _ = _precompile_pair(lambda model, x: model(x), m, x)
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
            code, cache = _precompile_pair(lambda model, xx: model(xx), m, x)
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
                code, cache = _precompile_pair(lambda model, xx: model(xx), m, x)
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
            code, cache = _precompile_pair(lambda model, xx: model(xx), m, x)
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
            _precompile_pair(boom, m, x)
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
        code, _ = _precompile_pair(lambda mm, t: mm(t).sum().backward(), m, x)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x, backend="eager")
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
        code, _cache = _precompile_pair(
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

        code, cache = _precompile_pair(
            lambda model, xx: model(xx).sum().backward(), m, x
        )
        # Capture must not mutate the example model's pre-existing grad (restored).
        self.assertEqual(m.weight.grad, grad_before)

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
                _precompile_pair(lambda model, t, b=bad: (model(t), b), m, x)
        for extra in (7, None):
            code, cache = _precompile_pair(
                lambda model, t, e=extra: (model(t), e), m, x
            )
            self.assertEqual(_load_pair(code, cache)(m, x)[1], extra)
        ecode, ecache = _precompile_pair(
            lambda model, t: (model(t), 3.14), m, x, backend="eager"
        )
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, xex)
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
        ecode, ecache = _precompile_pair(
            lambda model, t: model(t), m, xex, backend="eager"
        )
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
            code, cache = _precompile_pair(lambda model, t: model(t), m, xex)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, xex)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, xex)
        row = torch.randn(2, 8)[
            0:1, :4
        ]  # shape (1, 4), stride (8, 1): size-1 dim differs
        self.assertEqual(tuple(row.shape), (1, 4))
        self.assertNotEqual(row.stride(), xex.stride())
        self.assertEqual(_load_pair(code, cache)(m, row), m(row))
        self.assertEqual(_load_pair(code, _strip_artifact(cache))(m, row), m(row))

    def test_empty_input_shape_is_still_checked(self):
        # The numel==0 exemption must relax ONLY the (meaningless) stride check, not the
        # shape check: an empty runtime input whose shape differs from the example must
        # still raise invariant 3, not silently return the traced-shape output.
        code, cache = _precompile_pair(lambda t: t.sum(0), torch.randn(0, 4))
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
        code, cache = _precompile_pair(lambda mm, a, b: mm(a, b), m, x, y)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
            _precompile_pair(needs_guard, m, x)

    def test_dynamic_shapes_eager_rejected(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(
            NotImplementedError, "only supported with backend='inductor'"
        ):
            _precompile_pair(lambda mm, t: mm(t), m, x, backend="eager")

    @parametrize("path", ("cached", "inlined"))
    def test_dtype_mismatch_rejected(self, path):
        # Each dense input's dtype is baked at capture (invariant 6); a runtime input of
        # a different dtype is rejected up front on BOTH the cached and inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # float32 example
        code, cache = _precompile_pair(lambda model, t: model(t), m, x)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x)
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
            _precompile_pair(lambda mm, t: mm(t), m, x)

    def test_mark_unbacked_hint_override_honored(self):
        # A mark_unbacked hint_override is a perf-only autotuning size hint (never a
        # guard), so precompile does NOT reject it; the single artifact is valid for any
        # runtime size and the hint is threaded onto the capture ShapeEnv's symbol.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, hint_override=16)
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
            _precompile_pair(lambda mm, t: mm(t), m, x)

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
                _precompile_pair(lambda mm, t: mm(t), m, x)
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
        code, cache = _precompile_pair(lambda mm, a, b: mm(a) + b, m, x, y)
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
        code, cache = _precompile_pair(lambda mm, a, b: mm(a) + b, m, x, y)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x, backend="eager")
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(7, 4))

    def test_eager_backend_dtype_mismatch_rejected(self):
        # The eager driver checks USER_INPUT_DTYPES too: a dtype mismatch is rejected
        # (invariant 6).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = _precompile_pair(lambda model, t: model(t), m, x, backend="eager")
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    def test_cache_integrity_tampered_backend_rejected(self):
        # The cache envelope's backend tag is an integrity check: a tampered backend
        # (here flipped to a value that does not match python_code's BACKEND) makes
        # load() raise a clear PrecompileError rather than reconstruct a foreign cache.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = _precompile_pair(lambda model, t: model(t), m, x)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x)
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

    def test_standalone_runtime_artifact_execs_in_fresh_process(self):
        # A generated artifact that imports a standalone_runtime helper (here output-
        # aliasing, which emits ``from ...standalone_runtime import gen_alias_from_base``)
        # must EXEC in a FRESH process whose only prior import is ``torch`` -- a
        # regression for the runtime_wrappers <-> _dynamo circular import that a cold
        # exec used to hit. We write python_code to a temp file and exec it in a
        # subprocess that imports only torch, then runs forward().
        x = torch.randn(3, 4)
        code, _cache = _precompile_pair(lambda a: a.t(), x)
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
        codeA, cacheA = _precompile_pair(lambda mm, t: mm(t) * 2, m, x)
        codeB, cacheB = _precompile_pair(lambda mm, t: mm(t) + 100, m, x)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x, backend=backend)
        self.assertIn("BUFFER_NAMES = ['buf']", code)
        renamed = WithBuf("buf2").eval()  # same params, buffer renamed (same shape)
        f_c = _load_pair(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(renamed, x)

    def test_example_input_inplace_mutation_not_restored(self):
        # Capture EXECUTES fn once on the example inputs (invariant 3), so an in-place
        # mutation fn performs on its example user input happens at capture time and is
        # NOT restored -- only .grad is snapshotted/restored. Pin this surprising contract
        # so it stays covered: the example tensor reflects the mutation afterward.
        scratch = torch.zeros(4)
        _precompile_pair(lambda a: a.add_(1.0), scratch)
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
            code, cache = _precompile_pair(lambda mm, t: mm(t), m, x, backend="eager")
        else:
            code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x, backend="eager")
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
        code, cache = _precompile_pair(lambda model, h: model(h.a + h.b), m, inp)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
            code, cache = _precompile_pair(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True), x
            )
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x, backend=backend)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x, backend=backend)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x, backend=backend)
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
        code, cache = _precompile_pair(lambda model, t: model(t), m, x)

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
            yield "inlined", _load_pair(code, _strip_artifact(cache))

        for label, f_c in loaders():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, r"memory format.*PARAMETER/BUFFER.*layout"
                ):
                    f_c(with_noncontig_weight(), x)
        # The eager backend accepts the same non-contiguous weight (layout-flexible).
        ecode, ecache = _precompile_pair(
            lambda model, t: model(t), m, x, backend="eager"
        )
        run = with_noncontig_weight()
        self.assertEqual(_load_pair(ecode, ecache)(run, x), run(x))

    def test_unbacked_equality_shared_vs_independent_shape_id(self):
        # MAJOR1 (invariant 3, shared shape_id): two mark_unbacked dims that the graph requires
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
        code_s, cache_s = _precompile_pair(lambda mm, a, b: mm(a) + b, m, xs, ys)
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
        code_i, cache_i = _precompile_pair(lambda mm, a, b: mm(a) + b, m, xi, yi)
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
        _precompile_pair(lambda mm, t: mm(t).sum().backward(), m, x)
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
            _precompile_pair(lambda x: x + captured, torch.randn(3))

    def test_single_trust_warning_on_inlined_load(self):
        # On the inlined load path (an eager artifact has an empty cache, so there is
        # nothing to prime and load() just EXECs python_code) the untrusted-input / EXEC
        # warning must fire EXACTLY ONCE -- only _make_inlined_forward warns. Asserting
        # "exactly once" guards against the EXEC warning being duplicated on this load.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = _precompile_pair(lambda model, t: model(t), m, x, backend="eager")
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
        code, cache = _precompile_pair(lambda model, t: model(t).sum().backward(), m, t)
        self.assertIn("PARAM_NAMES = ['l1.weight']", code)  # tie collapsed to one

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
        code, cache = _precompile_pair(lambda a, b, t: b(a(t)), m1, m2, t)
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
        code, cache = _precompile_pair(lambda model, t: model(t).sum().backward(), m, t)

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
        code, cache = _precompile_pair(lambda mm, t: mm(t).sum().backward(), m, x)
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

    def test_precompile_public_members_are_the_exported_types(self):
        # Pins the list the parametrized tests below iterate over: an emptied
        # export list would otherwise generate no cases and pass vacuously.
        exported = {
            "capture",
            "load",
            "Capture",
            "MakeFxTracer",
            "PrecompiledRunnable",
            "PrecompileSummary",
        }
        members = set(_PRECOMPILE_PUBLIC_MEMBERS)
        self.assertEqual(set(torch.compiler.precompile.__all__), exported)
        self.assertEqual(members, exported | {"PrecompileError"})
        # A public API under the compiler namespace, deliberately NOT a top-level
        # torch.* verb. __all__ membership and the attribute are independent, so pin
        # both: re-adding the re-export alone would resurrect torch.precompile.
        self.assertIn("precompile", torch.compiler.__all__)
        self.assertNotIn("precompile", torch.__all__)
        self.assertFalse(hasattr(torch, "precompile"))

    @parametrize("name", _PRECOMPILE_PUBLIC_MEMBERS)
    def test_precompile_public_members_resolve(self, name):
        # The re-homing to torch.compiler.precompile must leave every annotation
        # resolvable (get_type_hints looks names up through __module__) and resolved.
        member = getattr(torch.compiler.precompile, name)
        hints = typing.get_type_hints(member)
        self.assertEqual(set(hints), set(inspect.get_annotations(member)))
        for hint in hints.values():
            self.assertNotIsInstance(hint, str)

    def test_precompile_module_identity(self):
        # torch.compiler.precompile is a submodule: re-importing it resolves to the
        # SAME module object, and its name is the stable public path.
        p = torch.compiler.precompile
        self.assertIs(importlib.import_module("torch.compiler.precompile"), p)
        self.assertIs(sys.modules["torch.compiler.precompile"], p)
        self.assertEqual(p.__name__, "torch.compiler.precompile")

    @parametrize("name", _PRECOMPILE_PUBLIC_MEMBERS)
    def test_precompile_member_module_and_qualname_resolve_to_it(self, name):
        # Each member's __module__/__qualname__ walk back to the object itself,
        # so pickle and test_public_bindings resolve it at the public path. The
        # re-homing costs inspect.getsource, which reads the file of
        # sys.modules[cls.__module__] and finds no class there; torch.onnx makes
        # the same trade for its public types.
        member = getattr(torch.compiler.precompile, name)
        target = sys.modules[member.__module__]
        for part in member.__qualname__.split("."):
            target = getattr(target, part)
        self.assertIs(target, getattr(member, "__func__", member))

    def test_nested_input_refused(self):
        # A nested example input is refused up front with a named PrecompileError rather
        # than the raw internal error one produces further down (a static capture has no
        # ShapeEnv to mint the jagged ragged dim's symbolic nested int, and nothing below
        # the trace has a nested representation on either path). The refusal runs
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

    def test_capture_inside_another_trace_refused(self):
        # An ambient fake mode outranks the one the UNBACKED path builds, and no foreign mode
        # passes allow_fallback_kernels=False, so a meta-less op would be run for real again;
        # one built under DEFAULT config (as here, and as an AOTAutograd / inductor trace
        # builds its own) also lacks the unsafe-data-ptr-access snapshot, so a .data_ptr()
        # read bakes 0 instead of raising. So an unbacked capture refuses up front, on a fn
        # that captures cleanly on its own, rather than tracing under a foreign contract.
        model = torch.nn.Linear(4, 4)
        x = torch.randn(3, 4)
        marked = torch.randn(3, 4)
        mark_unbacked(marked, 0)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with torch._guards.tracing(torch._guards.TracingContext(fake_mode)):
            with self.assertRaisesRegex(
                PrecompileError, "unbacked capture cannot run inside another trace"
            ):
                _precompile_pair(lambda m, t: m(t), model, marked)
            # The STATIC path traces on the real example tensors (make_fx's "real" mode
            # resolves no fake mode at all), so it has no mode of its own to lose to the
            # ambient one and is NOT refused.
            _precompile_pair(lambda m, t: m(t), model, x, backend="eager")
        # detect_fake_mode also ranks the dispatch-mode stack, so an enclosing
        # `with FakeTensorMode()` -- no TracingContext at all -- gets the same named refusal
        # rather than the mode-mismatch AssertionError inside detect_fake_mode.
        with FakeTensorMode():
            with self.assertRaisesRegex(
                PrecompileError, "unbacked capture cannot run inside another trace"
            ):
                _precompile_pair(lambda m, t: m(t), model, marked)
        _precompile_pair(lambda m, t: m(t), model, marked)

    def test_unbacked_capture_refuses_a_data_ptr_read(self):
        # The unbacked mode is built inside the
        # fake_tensor_allow_unsafe_data_ptr_access patch, so a .data_ptr() read in fn is
        # refused instead of returning a meaningless value. The refusal comes out of the
        # trace RAW here; the commit above relabels it as a PrecompileError.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)

        def reads_pointer(mm, t):
            t.data_ptr()
            return mm(t)

        with self.assertRaisesRegex(RuntimeError, "Cannot access data pointer") as cm:
            _precompile_pair(reads_pointer, m, x)
        self.assertNotIsInstance(cm.exception, PrecompileError)

    def test_unbacked_capture_refuses_a_meta_less_op_in_an_allowlisted_namespace(self):
        # The other unbacked-mode hardening: allow_fallback_kernels=False. An op with no
        # meta/fake kernel in an ALLOWLISTED namespace (aten, prims, quantized, ...) would
        # otherwise have FakeTensorMode's unsafe fallback run its real kernel on
        # zero-filled substitutes and bake whatever shape that produced. The op is called
        # on the UNMARKED input on purpose: the fallback declines symbolic-sized arguments
        # by itself, so only a static one exercises the flag. Raw out of the trace here too.
        from torch._subclasses.fake_tensor import UnsupportedOperatorException
        from torch.library import _scoped_library

        m = torch.nn.Linear(4, 3).eval()
        x, y = torch.randn(8, 4), torch.randn(2, 3)
        mark_unbacked(x, 0)
        with _scoped_library("quantized", "FRAGMENT") as qlib:
            qlib.define("mlprecompile_unbacked_no_meta(Tensor x) -> Tensor")
            qlib.impl("mlprecompile_unbacked_no_meta", lambda t: t * 2, "CPU")
            op = torch.ops.quantized.mlprecompile_unbacked_no_meta
            with self.assertRaises(UnsupportedOperatorException):
                _precompile_pair(lambda mm, t, u: mm(t) + op(u).sum(), m, x, y)


class _FilesModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))


def _files_fn(model, x):
    return model(x)


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompileCaptureFiles(TestCase):
    """The on-disk artifact pair writer, through its failure shapes."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.artifact = os.path.join(self.dir, "m.py")
        self.cache = os.path.join(self.dir, "m.cache")
        self.model = _FilesModel()
        self.x = torch.randn(2, 4)

    def _leftovers(self):
        return sorted(n for n in os.listdir(self.dir) if n not in ("m.py", "m.cache"))

    def _read(self, path):
        with open(path, "rb") as f:
            return f.read()

    def _rewrite_raises(self, exc_type, regex, backend="inductor"):
        # Rewrite the same paths from a freshly rendered pair, expecting the write to fail
        # with the patched-in exception.
        python_code, cache = _precompile_pair(
            _files_fn, self.model, self.x, backend=backend
        )
        with self.assertRaisesRegex(exc_type, regex):
            _write_artifact(self.artifact, self.cache, python_code, cache)

    def _write_pair(self):
        # The "previous artifact" the recovery tests want on disk, with both halves' bytes.
        python_code, cache = _precompile_pair(
            _files_fn, self.model, self.x, backend="eager"
        )
        _write_artifact(self.artifact, self.cache, python_code, cache)
        return self._read(self.artifact), self._read(self.cache)

    def _assert_serves(self):
        # Both halves of what a writer test wants: no scratch file or backup left behind,
        # and the named pair reading back and serving.
        self.assertEqual(self._leftovers(), [])
        python_code, cache = torch._precompile._read_artifact(self.artifact, self.cache)
        f = _load_pair(python_code, cache)
        self.assertEqual(f(self.model, self.x), self.model(self.x))

    def _assert_kept_backup(self, before, logs):
        # The previous source survived as the only leftover, a .bak the warning names.
        leftovers = self._leftovers()
        self.assertEqual(len(leftovers), 1, leftovers)
        self.assertTrue(leftovers[0].endswith(".bak"), leftovers)
        self.assertEqual(self._read(os.path.join(self.dir, leftovers[0])), before)
        # The warning names that file and the one rename that recovers from either shape.
        joined = "\n".join(logs.output)
        self.assertIn(leftovers[0], joined)
        self.assertIn("moved back over the first path", joined)

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

    @parametrize("half", ("artifact", "cache"))
    def test_a_failed_rename_on_a_first_write_leaves_nothing_named(self, half):
        # No previous pair to restore, so the undo takes the new artifact back out from under
        # its name: a first write cannot leave a named source with no cache beside it.
        no_space = OSError(errno.ENOSPC, "no space left")
        with self._replacing(getattr(self, half), exc=no_space):
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                self._rewrite_raises(OSError, "no space left", backend="eager")
        self.assertEqual(os.listdir(self.dir), [])

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

    def test_a_zero_inode_double_failure_keeps_the_previous_source(self):
        # st_ino is a file identifier only when NON-zero, and it is 0 on a FAT/exFAT mount, a
        # CIFS mount with noserverino, or Windows without FILE_ID_INFO -- the filesystems the
        # move-aside fallback exists for. Every name here is in one directory, so st_dev alone
        # called the old cache the new one: the undo read that as a completed write, restored
        # nothing, reported nothing, and unlinked the .bak holding the previous source's only
        # copy. The shape: no hard links, so the source is moved aside, the artifact rename
        # lands, the CACHE rename fails, and the restore rename fails too.
        before = self._write_pair()[0]
        real_stat, real_replace, installs = os.stat, os.replace, []

        def no_ino(path, **kwargs):
            st = real_stat(path, **kwargs)
            return os.stat_result((st.st_mode, 0) + tuple(st)[2:])

        def replace(src, dst):
            if dst == self.cache or (dst == self.artifact and installs):
                raise OSError("disk full")
            if dst == self.artifact:
                installs.append(dst)
            return real_replace(src, dst)

        no_links = OSError(errno.EXDEV, "no hard links")
        with mock.patch("os.stat", no_ino), mock.patch("os.link", side_effect=no_links):
            with mock.patch("os.replace", replace):
                with self.assertLogs("torch._precompile", level="WARNING") as logs:
                    with self.assertRaisesRegex(OSError, "disk full"):
                        _write_artifact(self.artifact, self.cache, "new", b"new-cache")
        # The first rename did land, so the new source is under the name and the previous one
        # is recoverable only from the .bak the report names.
        self.assertEqual(self._read(self.artifact), b"new")
        self._assert_kept_backup(before, logs)

    def test_an_eperm_moves_aside_a_source_the_caller_owns(self):
        # EPERM is the errno the fallback sees most in practice (fs.protected_hardlinks), so
        # an owned source on a filesystem without hard links rewrites like any other.
        self._write_pair()
        with mock.patch("os.link", side_effect=OSError(errno.EPERM, "not permitted")):
            self._write_pair()
        self._assert_serves()

    def test_an_eperm_on_a_source_the_caller_does_not_own_propagates(self):
        # The same errno is what fs.protected_hardlinks=1 (a Linux default) raises for a
        # source the caller does not own, and chattr +i for one it cannot write, on a
        # filesystem that DOES have hard links: moving that file aside would take it out from
        # under its name. st_uid is never -1, so this euid owns nothing (create=True because
        # Windows has no geteuid, where the ownership test does not run).
        before = self._write_pair()
        with mock.patch("os.link", side_effect=OSError(errno.EPERM, "not permitted")):
            with mock.patch.object(os, "geteuid", return_value=-1, create=True):
                with self.assertRaisesRegex(OSError, "not permitted"):
                    _write_artifact(self.artifact, self.cache, "new", b"new-cache")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self.assertEqual(self._leftovers(), [])

    def test_a_link_error_outside_the_set_propagates(self):
        # An os.link failure that is not about hard-link support says nothing about a move
        # being safe, so it propagates with the previous pair untouched.
        before = self._write_pair()
        with mock.patch("os.link", side_effect=OSError(errno.EIO, "io error")):
            with self.assertRaisesRegex(OSError, "io error"):
                _write_artifact(self.artifact, self.cache, "new", b"new-cache")
        self.assertEqual((self._read(self.artifact), self._read(self.cache)), before)
        self._assert_serves()

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

    def test_a_zero_inode_cache_half_in_that_window_keeps_the_new_pair(self):
        # Blindness is a property of each NAME's filesystem, and the two are independent
        # arguments: here the artifact has inodes and the cache does not. Reading one flag
        # off the artifact half left the cache half with neither read -- no inode match,
        # and no temp fallback either -- so the same window called a finished write
        # incomplete and restored the previous source beside the new cache, a pair ``load``
        # refuses on the sha256 with the previous cache already overwritten.
        before = self._write_pair()
        real_stat = os.stat

        def no_cache_ino(path, **kwargs):
            st = real_stat(path, **kwargs)
            if not str(path).startswith(self.cache):
                return st
            return os.stat_result((st.st_mode, 0) + tuple(st)[2:])

        cut = KeyboardInterrupt("interrupted after the second rename")
        with mock.patch("os.stat", no_cache_ino):
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

    def test_an_unreadable_or_undecodable_half_is_a_precompile_error(self):
        # Both of the reader's failures: a half that will not open, and a readable half that
        # is not the source (the two paths passed the wrong way round).
        self._write_pair()
        read = torch._precompile._read_artifact
        missing = os.path.join(self.dir, "gone.py")
        with self.assertRaises(PrecompileError) as cm:
            read(missing, self.cache)
        self.assertIn("could not read the artifact pair", str(cm.exception))
        # The message renders both paths with !r, and a Windows path's repr doubles its
        # backslashes, so compare against the repr rather than the raw string.
        self.assertIn(repr(missing), str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, FileNotFoundError)
        with self.assertRaises(PrecompileError) as cm:
            read(self.cache, self.artifact)
        self.assertIsInstance(cm.exception.__cause__, UnicodeDecodeError)


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
class TestPrecompileNumerics(TestCase):
    # Numeric-correctness tests run device-generically so the same coverage
    # exercises the CUDA lowering, not just CPU.

    def test_plain_function(self, device):
        def f(x, y):
            return (x @ y).sin(), x + y

        a = make_tensor((4, 4), device=device, dtype=torch.float32)
        b = make_tensor((4, 4), device=device, dtype=torch.float32)
        code, cache = _precompile_pair(f, a, b)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_multiple_module_args(self, device):
        # More than one nn.Module arg: each module's params are lifted with
        # m{i}.-prefixed names. Both modules are passed again at runtime.
        a = torch.nn.Linear(4, 4).to(device).eval()
        b = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((2, 4), device=device, dtype=torch.float32)
        ref = b(torch.relu(a(x)))

        code, cache = _precompile_pair(lambda ma, mb, x: mb(torch.relu(ma(x))), a, b, x)
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
        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
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

        code, cache = _precompile_pair(train_step, model, x, target)
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

        code, cache = _precompile_pair(train_step, model, x, target)
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

        code, cache = _precompile_pair(train_step, a, b, x, target)
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

        code, cache = _precompile_pair(lambda model, x: model(x), m, x)
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        # The tied weight is lifted once (single name), so it is one graph input.
        self.assertIn("PARAM_NAMES = ['a.weight']", code)

        # Training scatters a single grad onto the shared weight, matching eager's
        # accumulation into the tied parameter.
        ref = copy.deepcopy(m)
        ref(x).sum().backward()
        ref_grad = ref.a.weight.grad

        code, cache = _precompile_pair(lambda model, x: model(x).sum().backward(), m, x)
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
        code, cache = _precompile_pair(f, a, b, backend="eager")
        f_c = _load_pair(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_backend_eager_module(self, device):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = _precompile_pair(lambda model, x: model(x), m, x, backend="eager")
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

        code, cache = _precompile_pair(train_step, model, x, target, backend="eager")
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

        code, cache = _precompile_pair(lambda m, xx: m(xx), fresh(), x, backend="eager")
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
        code, cache = _precompile_pair(f, x, backend="eager")
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

        code, cache = _precompile_pair(train_step, fresh(), x, target)
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
        code, cache = _precompile_pair(lambda a: a.t(), x)
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(x), x.t())

    def test_input_mutation_supported(self, device):
        # In-place input mutation is reflected on the passed tensor (and matches
        # eager), via AOTAutograd's mutation handling composed into the artifact.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = _precompile_pair(lambda a: a.add_(1.0), scratch)
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
            code, cache = _precompile_pair(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True), x
            )
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
        code, cache = _precompile_pair(lambda model, xx: model(xx), fresh(), x)

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

        code, cache = _precompile_pair(fn, t, t)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
        code, cache = _precompile_pair(lambda model, t: model(t).sum().backward(), m, x)
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
        code, cache = _precompile_pair(lambda mm, a, b: mm(a) + b, m, x, y)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
        code, cache = _precompile_pair(lambda mm, t: mm(t), m, x)
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
        code, cache = _precompile_pair(lambda t: torch.relu(t) * 2.0, x)
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
            _precompile_pair(lambda t: t.contiguous() * 2.0, x)

    def test_eager_backend_input_mutation(self, device):
        # The eager backend replays the raw ATen graph, so input mutation is reflected on
        # the passed tensor and matches eager, like the inductor backend.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = _precompile_pair(lambda a: a.add_(1.0), scratch, backend="eager")
        f_c = _load_pair(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    def test_eager_backend_output_alias(self, device):
        # The eager backend reproduces an output that aliases an input (a view), matching
        # eager, via the raw ATen replay.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        code, cache = _precompile_pair(lambda a: a.t(), x, backend="eager")
        f_c = _load_pair(code, cache)
        self.assertEqual(f_c(x), x.t())


instantiate_device_type_tests(TestPrecompileNumerics, globals())


_FRESH_PROCESS_LOADER = """
import sys
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        return torch.relu(self.lin(x))

artifact, cache, state = sys.argv[1:4]
saved = torch.load(state)
model = M()
model.load_state_dict(saved["state_dict"])
f = torch.compiler.precompile.load(artifact, cache)
torch.testing.assert_close(f(model, saved["x"]), saved["expected"])
print("served")
"""


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompileLoad(TestCase):
    """load() over the on-disk pair a capture writes."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.artifact = os.path.join(self.dir, "m.py")
        self.cache = os.path.join(self.dir, "m.cache")
        self.model = _FilesModel()
        self.x = torch.randn(2, 4)

    def _write(self, artifact, cache, backend="eager", x=None):
        pair = _precompile_pair(
            _files_fn, self.model, self.x if x is None else x, backend=backend
        )
        _write_artifact(artifact, cache, *pair)
        return pair

    @parametrize("backend", ["inductor", "eager"])
    def test_load_round_trips_the_pair(self, backend):
        self._write(self.artifact, self.cache, backend=backend)
        f = load(self.artifact, self.cache)
        self.assertIsInstance(f, PrecompiledRunnable)
        self.assertFalse(f.installed)
        self.assertEqual(f(self.model, self.x), self.model(self.x))
        # No weights are baked in: a structurally identical model with other
        # weights serves its own answer.
        other = _FilesModel()
        self.assertEqual(f(other, self.x), other(self.x))
        # A standalone artifact installs nothing, so the handle's context
        # manager and unload() are no-ops.
        with f as entered:
            self.assertIs(entered, f)
        f.unload()
        self.assertEqual(f(self.model, self.x), self.model(self.x))

    @parametrize("backend", ["inductor", "eager"])
    def test_load_in_a_fresh_process(self, backend):
        state = os.path.join(self.dir, "state.pt")
        self._write(self.artifact, self.cache, backend=backend)
        expected = self.model(self.x)
        torch.save(
            {"state_dict": self.model.state_dict(), "x": self.x, "expected": expected},
            state,
        )
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                _FRESH_PROCESS_LOADER,
                self.artifact,
                self.cache,
                state,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("served", out.stdout)
        # load degrades to JIT with a warning rather than failing; the fresh
        # process must have served from the cache.
        self.assertNotIn("Falling back to JIT", out.stderr)

    def test_load_refuses_a_pair_from_two_captures_or_a_missing_cache(self):
        self._write(self.artifact, self.cache)
        other_artifact = os.path.join(self.dir, "other.py")
        other_cache = os.path.join(self.dir, "other.cache")
        self._write(other_artifact, other_cache, x=torch.randn(3, 4))
        with self.assertRaisesRegex(PrecompileError, "its code_hash"):
            load(self.artifact, other_cache)
        with self.assertRaisesRegex(PrecompileError, "could not read"):
            load(self.artifact, os.path.join(self.dir, "missing.cache"))


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompileCapture(TestCase):
    """capture() and load() over the on-disk pair, with the MakeFxTracer."""

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
        kwargs.setdefault("tracer", MakeFxTracer())
        return capture(fn, artifact_path=self.artifact, cache_path=self.cache, **kwargs)

    @parametrize("backend", ["inductor", "eager"])
    def test_capture_and_load_round_trip(self, backend):
        # The call is served through the artifact, without the exec warning a load
        # of untrusted source emits, and returns what fn returns.
        with self._capture(backend=backend) as cap:
            with self.assertNoLogs("torch._precompile", level="WARNING"):
                y = cap(self.model, self.x)
        self.assertEqual(y, self.model(self.x))
        self.assertTrue(os.path.exists(self.artifact))
        self.assertTrue(os.path.exists(self.cache))
        f = load(self.artifact, self.cache)
        self.assertIsInstance(f, PrecompiledRunnable)
        self.assertFalse(f.installed)
        self.assertEqual(f(self.model, self.x), self.model(self.x))
        # No weights are baked in: a structurally identical model with other
        # weights serves its own answer.
        other = _FilesModel()
        self.assertEqual(f(other, self.x), other(self.x))
        # A standalone artifact installs nothing, so the handle's context
        # manager and unload() are no-ops.
        with f as entered:
            self.assertIs(entered, f)
        f.unload()
        self.assertEqual(f(self.model, self.x), self.model(self.x))

    def test_a_make_fx_capture_takes_exactly_one_positional_call(self):
        with self._capture(backend="eager") as cap:
            with self.assertRaisesRegex(ValueError, "positional arguments only"):
                cap(self.model, x=self.x)
            cap(self.model, self.x)
            with self.assertRaisesRegex(PrecompileError, "single call"):
                cap(self.model, self.x)
        self.assertEqual(
            load(self.artifact, self.cache)(self.model, self.x), self.model(self.x)
        )

    def test_a_capture_must_be_entered_before_it_is_called(self):
        # Outside the block nothing would be written at exit, so the call is
        # refused rather than running the whole trace for a missing artifact.
        cap = self._capture(backend="eager")
        with self.assertRaisesRegex(PrecompileError, "before calling it"):
            cap(self.model, self.x)
        with self.assertRaisesRegex(PrecompileError, "before calling save"):
            cap.save()
        self.assertFalse(os.path.exists(self.artifact))

    def test_decompositions_forward_through_the_tracer(self):
        called = []

        def my_relu_decomp(x):
            called.append(True)
            return (x > 0) * x

        tracer = MakeFxTracer(
            decompositions={torch.ops.aten.relu.default: my_relu_decomp}
        )
        with self._capture(backend="eager", tracer=tracer) as cap:
            y = cap(self.model, self.x)
        self.assertTrue(called)
        self.assertEqual(y, self.model(self.x))

    def test_a_capture_is_over_once_its_block_exits(self):
        # After the block, a call would trace and serve with nothing left to
        # write it; it is refused the same way as a call before the block, on
        # both a clean and a raising exit.
        cap = self._capture(backend="eager")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with cap:
                raise RuntimeError("boom")
        with self.assertRaisesRegex(PrecompileError, "already exited"):
            cap(self.model, self.x)
        self.assertFalse(os.path.exists(self.artifact))
        with self._capture(backend="eager") as cap:
            cap(self.model, self.x)
        with self.assertRaisesRegex(PrecompileError, "already exited"):
            cap(self.model, self.x)

    def test_a_serve_that_raises_leaves_the_capture_retryable(self):
        # The pair is recorded only once the serve has returned: a call whose
        # serve raised writes nothing at exit and can be made again.
        with self._capture(backend="eager") as cap:
            with mock.patch(
                "torch._precompile._runnable_from_pair",
                side_effect=RuntimeError("serve failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "serve failed"):
                    cap(self.model, self.x)
            self.assertFalse(os.path.exists(self.artifact))
            self.assertEqual(cap(self.model, self.x), self.model(self.x))
        self.assertTrue(os.path.exists(self.artifact))

    def test_a_swallowed_serve_failure_is_named_at_exit(self):
        serve = RuntimeError("serve failed")
        patch = mock.patch("torch._precompile._runnable_from_pair", side_effect=serve)
        with self.assertRaisesRegex(PrecompileError, "capture's call raised"):
            with self._capture(backend="eager") as cap:
                with patch, self.assertRaisesRegex(RuntimeError, "serve failed"):
                    cap(self.model, self.x)
        self.assertFalse(os.path.exists(self.artifact))

    def test_capture_takes_pathlike_paths_and_the_default_tracer(self):
        import pathlib

        artifact = pathlib.Path(self.dir) / "sub" / "m.py"
        cache = pathlib.Path(self.dir) / "sub" / "m.cache"
        with capture(
            _files_fn, artifact_path=artifact, cache_path=cache, backend="eager"
        ) as cap:
            cap(self.model, self.x)
        self.assertEqual(load(artifact, cache)(self.model, self.x), self.model(self.x))
        backend = inspect.signature(capture).parameters["backend"].default
        self.assertEqual(backend, "inductor")

    def test_a_block_without_a_call_or_that_raises_writes_nothing(self):
        with self.assertRaisesRegex(PrecompileError, "call the capture with"):
            with self._capture(backend="eager"):
                pass
        self.assertFalse(os.path.exists(self.artifact))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with self._capture(backend="eager") as cap:
                cap(self.model, self.x)
                raise RuntimeError("boom")
        self.assertFalse(os.path.exists(self.artifact))
        self.assertFalse(os.path.exists(self.cache))

    def test_save_checkpoints_the_pair_exit_rewrites(self):
        with self._capture(backend="eager") as cap:
            with self.assertRaisesRegex(PrecompileError, "nothing was captured"):
                cap.save()
            cap(self.model, self.x)
            cap.save()
            with open(self.artifact, "rb") as f:
                saved = f.read()
            self.assertEqual(
                load(self.artifact, self.cache)(self.model, self.x), self.model(self.x)
            )
            os.unlink(self.artifact)
        with open(self.artifact, "rb") as f:
            self.assertEqual(f.read(), saved)

    def test_save_after_a_raising_block_is_refused(self):
        cap = self._capture(backend="eager")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with cap:
                cap(self.model, self.x)
                raise RuntimeError("boom")
        with self.assertRaisesRegex(PrecompileError, "already exited"):
            cap.save()
        self.assertFalse(os.path.exists(self.artifact))

    def test_a_capture_is_single_use(self):
        cap = self._capture(backend="eager")
        with cap:
            with self.assertRaisesRegex(PrecompileError, "already been entered"):
                with cap:
                    pass
            cap(self.model, self.x)
        with self.assertRaisesRegex(PrecompileError, "already exited"):
            with cap:
                pass

    def test_an_exit_time_write_failure_raises_precompile_error(self):
        disk_full = OSError("disk full")
        with mock.patch("torch._precompile._write_artifact", side_effect=disk_full):
            with self.assertRaisesRegex(PrecompileError, "could not write"):
                with self._capture(backend="eager") as cap:
                    cap(self.model, self.x)

    def test_capture_validates_backend_and_tracer(self):
        with self.assertRaisesRegex(ValueError, "backend must be"):
            self._capture(backend="nope")
        with self.assertRaisesRegex(TypeError, "MakeFxTracer"):
            self._capture(tracer="make_fx")
        with self.assertRaisesRegex(TypeError, "partial"):
            self._capture(functools.partial(_files_fn, self.model))

    @parametrize("backend", ["inductor", "eager"])
    def test_a_backward_step_scatters_grads_onto_the_runtime_model(self, backend):
        def train_step(model, x):
            model(x).sum().backward()

        expected = _FilesModel()
        expected.load_state_dict(self.model.state_dict())
        expected(self.x).sum().backward()
        with self._capture(train_step, backend=backend) as cap:
            self.assertIsNone(cap(self.model, self.x))
        self.assertEqual(self.model.lin.weight.grad, expected.lin.weight.grad)
        runtime = _FilesModel()
        runtime.load_state_dict(self.model.state_dict())
        load(self.artifact, self.cache)(runtime, self.x)
        self.assertEqual(runtime.lin.weight.grad, expected.lin.weight.grad)
        self.assertEqual(runtime.lin.bias.grad, expected.lin.bias.grad)

    @parametrize("backend", ["inductor", "eager"])
    def test_a_capture_applies_in_place_updates_once(self, backend):
        # The trace and the serve both run fn; only the serve's updates may land.
        def step(model, x):
            y = model(x)
            x.add_(1)
            return y.sum()

        model, expected = torch.nn.BatchNorm1d(4).train(), torch.nn.BatchNorm1d(4)
        x = torch.randn(3, 4)
        expected_x = x.clone()
        expected_out = step(expected.train(), expected_x)
        with self._capture(step, backend=backend) as cap:
            self.assertEqual(cap(model, x), expected_out)
        self.assertEqual(model.num_batches_tracked, 1)
        self.assertEqual(model.running_mean, expected.running_mean)
        self.assertEqual(x, expected_x)

    @parametrize("backend", ["eager", "inductor"])
    def test_a_parameter_passed_as_an_argument_is_updated_once(self, backend):
        def step(model, w, x):
            with torch.no_grad():
                w.add_(1)
            return model(x)

        expected = self.model.lin.weight + 1
        with self._capture(step, backend=backend) as cap:
            cap(self.model, self.model.lin.weight, self.x)
        self.assertEqual(self.model.lin.weight, expected)

    @parametrize("backend", ["eager", "inductor"])
    @parametrize("write", ["direct", "data", "chunk", "unbind", "foreach"])
    def test_a_capture_refuses_an_in_place_parameter_update(self, backend, write):
        # A write through ``.data`` bumps no version counter the parameter shares.
        def step(model, x):
            with torch.no_grad():
                w = model.lin.weight
                if write == "foreach":
                    torch._foreach_add_(list(model.parameters()), 1)
                elif write == "chunk":
                    w.chunk(2)[0].add_(1)
                elif write == "unbind":
                    for row in w.unbind(0):
                        row.mul_(0.9)
                else:
                    (w.data if write == "data" else w).add_(1)
            return model(x)

        with self.assertRaisesRegex(PrecompileError, "updates a parameter in place"):
            with self._capture(step, backend=backend) as cap:
                cap(self.model, self.x)
        self.assertFalse(os.path.exists(self.artifact))

    def test_a_trace_that_raises_leaves_buffers_and_inputs_as_they_were(self):
        def step(model, x):
            model(x)
            x.add_(1)
            raise RuntimeError("fn failed after its in-place updates")

        model, x = torch.nn.BatchNorm1d(4).train(), torch.randn(3, 4)
        running_mean, expected_x = model.running_mean.clone(), x.clone()
        with self.assertRaisesRegex(RuntimeError, "fn failed after"):
            with self._capture(step, backend="eager") as cap:
                cap(model, x)
        self.assertEqual(model.num_batches_tracked, 0)
        self.assertEqual(model.running_mean, running_mean)
        self.assertEqual(x, expected_x)
        self.assertFalse(os.path.exists(self.artifact))


if __name__ == "__main__":
    run_tests()
