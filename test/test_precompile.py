# Owner(s): ["oncall: pt2"]
import copy
import errno
import io
import os
import pickle
import stat
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.utils._pytree as _pytree
from torch._dynamo.decorators import mark_unbacked
from torch._precompile import _write_artifact, PrecompileError
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
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
        f = torch.compiler.precompile.load(python_code, cache)
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


if __name__ == "__main__":
    run_tests()
