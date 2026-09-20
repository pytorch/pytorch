# Owner(s): ["oncall: pt2"]
import functools
import importlib
import inspect
import io
import pickle
import sys
import typing

import torch
from torch._precompile import PrecompileError
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
        exported = {"Capture", "MakeFxTracer", "PrecompileSummary"}
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

    def test_static_capture_rejects_data_dependent_ops(self):
        # A static make_fx capture traces on fake tensors, so a value the trace
        # cannot know must be refused cleanly, naming the op, rather than baked
        # from the example or leaked as a raw fake-tensor exception. .item() and
        # a tensor-value branch raise DataDependentOutputException; .nonzero()
        # raises the sibling DynamicOutputShapeException, which neither inherits
        # from the other.
        from torch._precompile import _capture

        model = torch.nn.Linear(4, 4)

        def branches(m, x):
            return m(x) if x.sum() > 0 else m(-x)

        def items(m, x):
            return m(x) * x.sum().item()

        def nonzero(m, x):
            return m(x)[x.nonzero()[:, 0]]

        scalar = "_local_scalar_dense"
        for fn, op in ((branches, scalar), (items, scalar), (nonzero, "aten.nonzero")):
            with self.assertRaisesRegex(PrecompileError, f"data-dependent.*{op}"):
                _capture(fn, (model, torch.randn(3, 4)), None)
        # The other way a fake trace fails where a real one ran: an op with no
        # fake/meta kernel, outside the namespaces fake mode falls back for.
        with torch.library._scoped_library("precompile_test", "DEF") as lib:
            lib.define("nometa(Tensor t) -> Tensor")
            lib.impl("nometa", lambda t: t.clone(), "CPU")

            def nometa(m, x):
                return m(torch.ops.precompile_test.nometa(x))

            with self.assertRaisesRegex(PrecompileError, "nometa.*register_fake"):
                _capture(nometa, (model, torch.randn(3, 4)), None)

    def test_static_capture_runs_no_real_compute(self):
        # The static trace runs on fakes: a backward in fn leaves the example
        # model's grads alone and an in-place op leaves the example input alone.
        # A revert to tracing_mode="real" would fail both.
        from torch._precompile import _capture

        model = torch.nn.Linear(4, 4)
        x = torch.randn(3, 4)
        expected = x.clone()
        _capture(lambda m, x: m(x).sum().backward(), (model, x), None)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        _capture(lambda m, x: m(x.add_(1)), (model, x), None)
        self.assertEqual(x, expected)

    def test_cache_envelope_carries_the_tracer_tag(self):
        # The envelope is what a loader checks before trusting a (python_code,
        # cache) pair; the eager backend has no compiled artifact, so the tag
        # set is all it holds.
        from torch._precompile import PrecompiledModule

        pm = PrecompiledModule(lambda m, x: m(x), backend="eager")
        pm._compile((torch.nn.Linear(4, 4), torch.randn(3, 4)))
        env = torch.load(io.BytesIO(pm.to_cache_bytes()), weights_only=True)
        keys = ["format", "version", "backend", "tracer", "code_hash", "artifact"]
        self.assertEqual(list(env), keys)
        self.assertEqual(env["backend"], "eager")
        self.assertEqual(env["tracer"], "make_fx")
        self.assertIsNone(env["artifact"])

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

    def test_precompile_error_result_defaults_to_sentinel(self):
        # Nothing ran before an ordinary refusal, so the default is the sentinel, not
        # None: a training step that ends in .backward() returns None for real. The
        # public test for it is the comparison against the class default.
        err = PrecompileError("refused")
        self.assertIs(err.result, PrecompileError.result)
        self.assertIsNotNone(err.result)
        err.result = None
        self.assertIsNone(err.result)

    def test_make_fx_capture_constructor(self):
        # A partial hides its bound arguments from the capture, so it is refused
        # up front with the fix, rather than failing later as a baked constant; the
        # tracer name is refused too, since the constructor reads the MakeFxTracer.
        from torch._precompile import _MakeFxCapture, MakeFxTracer

        def step(model, x):
            return model(x)

        bound = functools.partial(step, torch.nn.Linear(2, 2))
        table = {torch.ops.aten.add.Tensor: lambda *a, **k: None}
        tracer = MakeFxTracer(decompositions=table)
        with self.assertRaisesRegex(PrecompileError, "cannot capture a partial"):
            _MakeFxCapture(
                bound, "m.py", "m.cache", backend="eager", tracer=tracer, training=False
            )
        with self.assertRaisesRegex(PrecompileError, "MakeFxTracer instance as tracer"):
            _MakeFxCapture(
                step,
                "m.py",
                "m.cache",
                backend="eager",
                tracer="make_fx",
                training=False,
            )
        cap = _MakeFxCapture(
            step, "m.py", "m.cache", backend="eager", tracer=tracer, training=False
        )
        self.assertIs(cap.__enter__(), cap)
        self.assertIs(cap._module._decompositions, table)
        self.assertEqual(cap._artifact_path, "m.py")
        self.assertEqual(cap._cache_path, "m.cache")
        self.assertFalse(cap._training)
        self.assertFalse(cap._traced)
        self.assertIsNone(cap._rendered)

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
        # Nothing hung off the singleton rewrites __module__/__qualname__: the
        # docs place these under torch.compiler.precompile.<name>, but only a
        # name torch.compiler.__all__ exports may claim torch.compiler, or
        # pickle cannot resolve the class and inspect cannot find its source.
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
        # the entry frame (x, which rows() captures), and a module global the
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
                return x.shape[0]

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


if __name__ == "__main__":
    run_tests()
