# Owner(s): ["oncall: pt2"]
import errno
import functools
import hashlib
import importlib
import inspect
import io
import os
import pickle
import stat
import subprocess
import sys
import tempfile
import typing
from unittest import mock

import torch
from torch._dynamo.decorators import mark_unbacked
from torch._precompile import _make_inlined_forward, _write_artifact, PrecompileError
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.compiler.precompile import capture, load, MakeFxTracer, PrecompiledRunnable
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


_PRECOMPILE_PUBLIC_MEMBERS = [
    name
    for name in dir(torch.compiler.precompile)
    if not name.startswith("_") and callable(getattr(torch.compiler.precompile, name))
]


def _precompile_pair(fn, *args, **kwargs):
    """A rendered (python_code, cache) pair, built the way the retired callable
    ``torch.compiler.precompile(fn, *args, **kwargs)`` built the one it returned."""
    from torch._precompile import PrecompiledModule

    compiled = PrecompiledModule(fn, **kwargs)
    compiled._compile(args)
    python_code = compiled.to_python_code()
    return python_code, compiled.to_cache_bytes(python_code)


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

    @parametrize("name", _PRECOMPILE_PUBLIC_MEMBERS)
    def test_precompile_public_members_resolve(self, name):
        # The re-homing to torch.compiler.precompile must leave every annotation
        # resolvable (get_type_hints looks names up through __module__) and resolved.
        member = getattr(torch.compiler.precompile, name)
        hints = typing.get_type_hints(member)
        self.assertEqual(set(hints), set(inspect.get_annotations(member)))
        for hint in hints.values():
            self.assertNotIsInstance(hint, str)

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

    def test_trivial_continuation_is_served_as_plain_python(self):
        # The continuation after a trailing .backward() reaches no tensor, so
        # Dynamo skips it before tracing and it runs as plain Python during
        # capture. Its record says so, a standalone artifact counts it as
        # covered, and the driver rebuilds it as the plain function it was.
        import inspect
        from unittest import mock

        from torch import _precompile_driver as driver
        from torch._dynamo.package import CompilePackage
        from torch._dynamo.precompile_context import EagerCacheArtifact
        from torch._dynamo.precompile_package import default_guard_filter_fn
        from torch._precompile import _b64, _multigraph_frames, _serving_mode

        def step(model, x):
            (model(x) * _MULTIGRAPH_SCALE).sum().backward()

        model = torch.nn.Linear(4, 4)
        x = torch.randn(3, 4)
        package = CompilePackage(step)
        compiled = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=default_guard_filter_fn
        )(step)
        compiled(model, x)
        expected = torch.nn.Linear(4, 4)
        expected.load_state_dict(model.state_dict())
        step(expected, x)
        frames = _multigraph_frames(package.cache_entry())
        self.assertEqual([f["trivial"] for f in frames], [False, True])
        self.assertEqual([len(f["variants"]) for f in frames], [1, 0])
        self.assertEqual(_serving_mode(frames), "standalone")
        # Only a continuation Dynamo never traced is served that way: one it
        # compiled but kept no variant of still sends the capture to installing.
        frames[1]["trivial"] = False
        self.assertEqual(_serving_mode(frames), "installed")
        frames[1]["trivial"] = True
        backends = {
            backend_id: EagerCacheArtifact(key=backend_id, content=backend)
            for backend_id, backend in package.cached_backends.items()
        }
        torch._dynamo.reset()
        scope = step.__globals__
        minted = ("__compiled_fn", "__resume_at", "__builtins_dict__", "__import_")

        def scrub():
            return {n: scope.pop(n) for n in list(scope) if n.startswith(minted)}

        removed = scrub()
        self.addCleanup(lambda: (scrub(), scope.update(removed)))
        module = "precompile_test_captured_module"
        alias = mock.patch.dict(sys.modules, {module: sys.modules[__name__]})
        self.addCleanup(alias.stop)
        alias.start()
        for frame in frames:
            frame["python_module"] = module
        ns = {
            "__name__": "precompile_test_artifact",
            "_FRAMES": _b64(frames),
            "_BACKENDS": _b64(backends),
            "_ENTRY_BINDING": _b64({"defaults": None, "kwdefaults": None}),
            "_DYNAMO_PYTHON_VERSION": tuple(sys.version_info[:2]),
            "TORCH_VERSION": torch.__version__,
        }
        exec(inspect.getsource(driver._build_multigraph_forward), ns)
        forward = ns["_build_multigraph_forward"]()
        served = torch.nn.Linear(4, 4)
        served.load_state_dict(model.state_dict())
        self.assertIsNone(forward(served, x))
        self.assertEqual(served.weight.grad, expected.weight.grad)
        self.assertEqual(served.bias.grad, expected.bias.grad)

    def test_multigraph_artifact_round_trips_a_hand_built_package(self):
        # The renderer turns a package Dynamo filled into the (python_code, cache)
        # pair load reads: readable metadata beside the opaque blobs, the tracer
        # tag pairing the two halves, and a driver that serves the captured
        # variants and refuses the rest.
        from unittest import mock

        from torch._dynamo.package import CompilePackage
        from torch._dynamo.precompile_context import EagerCacheArtifact
        from torch._dynamo.precompile_package import default_guard_filter_fn
        from torch._precompile import (
            _build_multigraph_artifact,
            _parse_artifact_metadata,
            _runnable_from_pair,
        )
        from torch.compiler.precompile import PrecompileSummary

        def step(model, x, *, scale=2.0):
            y = model(x) * scale * _MULTIGRAPH_SCALE
            torch._dynamo.graph_break()
            return y + y.shape[0]

        model = torch.nn.Linear(4, 4)
        x2, x3 = torch.randn(2, 4), torch.randn(3, 4)
        package = CompilePackage(step)
        compiled = torch._dynamo.optimize(
            backend="eager", package=package, guard_filter_fn=default_guard_filter_fn
        )(step)
        expected2, expected3 = compiled(model, x2), compiled(model, x3)
        entry = package.cache_entry()
        backends = {
            backend_id: EagerCacheArtifact(key=backend_id, content=backend)
            for backend_id, backend in package.cached_backends.items()
        }
        summary = PrecompileSummary(
            frames=len(entry.codes),
            resume_functions=1,
            guarded_codes=sum(len(c.guarded_codes) for c in entry.codes),
            backend_graphs=len(backends),
            dropped_guards=(("MODULE_MATCH", "model"),),
        )
        # The records name this module, which is __main__ under a script run and
        # the driver refuses that; serve them from an importable alias of it.
        module = "precompile_test_captured_module"
        alias = mock.patch.dict(sys.modules, {module: sys.modules[__name__]})
        self.addCleanup(alias.stop)
        alias.start()
        for code in entry.codes:
            code.python_module = module
        python_code, cache = _build_multigraph_artifact(
            entry, backends, summary, "eager", step
        )
        meta = _parse_artifact_metadata(python_code)
        self.assertEqual(meta["TRACER"], "dynamo")
        self.assertEqual(meta["SERVING_MODE"], "standalone")
        self.assertEqual(meta["BACKEND"], "eager")
        self.assertEqual(meta["FN_NAME"], step.__qualname__)
        self.assertEqual(
            [name for name, _ in meta["FRAMES"]],
            [c.python_code.co_name for c in entry.codes],
        )
        self.assertEqual(meta["DROPPED_GUARDS"], [["MODULE_MATCH", "model"]])
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertEqual(blob["tracer"], "dynamo")
        self.assertIsNone(blob["artifact"])
        # A serving process never traced, so the names Dynamo minted into this
        # module during capture must not be what makes the guards pass.
        torch._dynamo.reset()
        scope = step.__globals__
        minted = ("__compiled_fn", "__resume_at", "__builtins_dict__", "__import_")

        def scrub():
            return {n: scope.pop(n) for n in list(scope) if n.startswith(minted)}

        removed = scrub()
        self.addCleanup(lambda: (scrub(), scope.update(removed)))
        f = _runnable_from_pair(python_code, cache, _trusted=True)
        self.assertFalse(f.installed)
        self.assertEqual(f(model, x2), expected2)
        self.assertEqual(f(model, x3), expected3)
        with self.assertRaisesRegex(PrecompileError, "no captured variant"):
            f(model, x2.double())
        # The two halves pair on the tracer tag: a make_fx cache is refused.
        _, fx_cache = _precompile_pair(_files_fn, _FilesModel(), x2, backend="eager")
        with self.assertRaisesRegex(PrecompileError, "tracer"):
            _runnable_from_pair(python_code, fx_cache)

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
        # The retired load() refused a cache whose code_hash did not pair with the source
        # before serving; keep that check so a stale cache half cannot pass as the new pair.
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        code_hash = hashlib.sha256(python_code.encode()).hexdigest()
        self.assertEqual(blob["code_hash"], code_hash)
        f = _make_inlined_forward(python_code, warn=False)
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
        self._write(self.artifact, self.cache, backend=backend)
        state = os.path.join(self.dir, "state.pt")
        expected = _make_inlined_forward(self._read(self.artifact), warn=False)(
            self.model, self.x
        )
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

    def _read(self, path):
        with open(path, "rb") as f:
            return f.read().decode()

    def test_load_refuses_a_pair_from_two_captures_or_a_missing_half(self):
        self._write(self.artifact, self.cache)
        other_artifact = os.path.join(self.dir, "other.py")
        other_cache = os.path.join(self.dir, "other.cache")
        self._write(other_artifact, other_cache, x=torch.randn(3, 4))
        with self.assertRaisesRegex(PrecompileError, "does not match"):
            load(self.artifact, other_cache)
        with self.assertRaisesRegex(PrecompileError, "could not read"):
            load(self.artifact, os.path.join(self.dir, "missing.cache"))

    def test_load_pairs_the_cache_on_its_tracer_tag(self):
        # The envelope names the tracer that produced it; a tag that differs from
        # the python_code's is a wrong pairing, and a pair written before the tag
        # (absent on both sides) still reads as make_fx.
        _, cache = self._write(self.artifact, self.cache)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertEqual(blob["tracer"], "make_fx")
        blob["tracer"] = "dynamo"
        buf = io.BytesIO()
        torch.save(blob, buf)
        _write_artifact(
            self.artifact, self.cache, self._read(self.artifact), buf.getvalue()
        )
        with self.assertRaisesRegex(PrecompileError, "tracer"):
            load(self.artifact, self.cache)
        del blob["tracer"]
        buf = io.BytesIO()
        torch.save(blob, buf)
        _write_artifact(
            self.artifact, self.cache, self._read(self.artifact), buf.getvalue()
        )
        self.assertEqual(
            load(self.artifact, self.cache)(self.model, self.x), self.model(self.x)
        )


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
        with self._capture(backend=backend) as cap:
            y = cap(self.model, self.x)
        # The call is served through the artifact and returns what fn returns.
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

    def test_a_block_without_a_call_or_that_raises_writes_nothing(self):
        with self.assertRaisesRegex(PrecompileError, "nothing was captured"):
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
        with open(self.artifact, "rb") as f:
            self.assertEqual(f.read(), saved)

    def test_capture_validates_backend_and_tracer(self):
        with self.assertRaisesRegex(ValueError, "backend must be"):
            self._capture(backend="nope")
        with self.assertRaisesRegex(TypeError, "MakeFxTracer"):
            self._capture(tracer="make_fx")
        with self.assertRaisesRegex(PrecompileError, "partial"):
            self._capture(functools.partial(_files_fn, self.model))

    def test_training_capture_scatters_grads_onto_the_runtime_model(self):
        def train_step(model, x):
            model(x).sum().backward()

        expected = _FilesModel()
        expected.load_state_dict(self.model.state_dict())
        expected(self.x).sum().backward()
        with self._capture(train_step, backend="eager", training=True) as cap:
            self.assertIsNone(cap(self.model, self.x))
        self.assertEqual(self.model.lin.weight.grad, expected.lin.weight.grad)
        runtime = _FilesModel()
        runtime.load_state_dict(self.model.state_dict())
        load(self.artifact, self.cache)(runtime, self.x)
        self.assertEqual(runtime.lin.weight.grad, expected.lin.weight.grad)
        self.assertEqual(runtime.lin.bias.grad, expected.lin.bias.grad)


if __name__ == "__main__":
    run_tests()
