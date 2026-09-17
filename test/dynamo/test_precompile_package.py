# Owner(s): ["module: dynamo"]

import copy
import functools
import hashlib
import io
import os
import pickle
import tempfile

import torch._inductor.test_case
from torch.compiler._precompile_types import PrecompileSummary


class TestPrecompilePackage(torch._inductor.test_case.TestCase):
    def test_summary_is_incomplete_without_a_backend_graph(self):
        def summary(**kw):
            base = dict(
                frames=1,
                resume_functions=0,
                guarded_codes=1,
                backend_graphs=1,
                bypassed=(),
            )
            base.update(kw)
            return PrecompileSummary(**base)

        self.assertTrue(summary().complete)
        self.assertFalse(summary(backend_graphs=0).complete)
        self.assertFalse(summary(guarded_codes=0).complete)
        self.assertFalse(summary(capture_errors=("boom",)).complete)

    def test_source_graph_module_copies_are_isolated(self):
        # _src and the exec'd forward are shared between copies; everything else
        # nn.Module keeps on an instance is state, hook dicts included, and a
        # copy sharing it lets an update on one copy silently edit the other.
        # __reduce__ used to pickle the SHARED _src, whose body aliases the
        # original's parameter/buffer containers -- so mutating the original
        # after a deepcopy round-tripped the mutated tensors into the copy's
        # pickle even though live calls were isolated. __reduce__ now
        # snapshots the instance's own state.
        from torch._dynamo.precompile_context import (
            _EagerGraphSource,
            _SourceGraphModule,
        )

        src = _EagerGraphSource(
            code="def forward(self, x):\n    return x + self.b\n",
            import_block="",
            body={"_buffers": {"b": torch.ones(3)}},
        )
        original = _SourceGraphModule(src)
        dup = copy.deepcopy(original)
        dup.register_forward_hook(lambda *args: None)
        dup._non_persistent_buffers_set.add("b")
        self.assertEqual(len(original._forward_hooks), 0)
        self.assertEqual(original._non_persistent_buffers_set, set())
        self.assertIs(dup._src, original._src)

        x = torch.randn(3)
        original._buffers["b"].mul_(100)
        self.assertEqual(original(x), x + 100)
        self.assertEqual(dup(x), x + 1)  # live isolation
        self.assertEqual(pickle.loads(pickle.dumps(dup))(x), x + 1)
        # And an instance pickles its CURRENT parameters/buffers/submodules,
        # not its load-time ones (other nn.Module state still comes from _src).
        self.assertEqual(pickle.loads(pickle.dumps(original))(x), x + 100)

    def test_eager_artifact_round_trips_a_hop_graph_as_source(self):
        # GraphModule.__reduce__ re-traces the generated source at load; cond
        # rejects the Proxy and autocast enter/exit EXECUTE and leave no node.
        # The top level must travel as source, its HOP bodies as real Graphs.
        from torch._dynamo.precompile_context import (
            _SourceGraphModule,
            EagerCacheArtifact,
        )

        def fn(x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))
            return y + 1

        gms = []

        def backend(gm, example_inputs):
            gms.append(gm)
            return gm.forward

        x = torch.randn(3)
        torch.compile(fn, backend=backend, fullgraph=True)(x)
        (gm,) = gms
        artifact = EagerCacheArtifact(key="k", content=gm.forward)
        loaded = pickle.loads(pickle.dumps(artifact)).after_deserialization()
        module = loaded.__self__
        self.assertIsInstance(module, _SourceGraphModule)
        self.assertFalse(hasattr(module, "graph"))
        self.assertTrue(module._modules)
        for sub in module._modules.values():
            self.assertIsInstance(sub, torch.fx.GraphModule)
            self.assertGreater(len(sub.graph.nodes), 0)
        self.assertEqual(loaded(x), gm.forward(x))
        self.assertEqual(loaded(-x.abs()), gm.forward(-x.abs()))
        # Re-serializing a loaded artifact goes through _SourceGraphModule.__reduce__.
        reloaded = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(artifact))))
        self.assertEqual(reloaded.after_deserialization()(x), gm.forward(x))

    def test_take_artifact_removes_the_staged_backend(self):
        from torch._dynamo.precompile_context import (
            EagerCacheArtifact,
            PrecompileContext,
        )

        PrecompileContext.record_artifact(EagerCacheArtifact(key="k", content=None))
        self.assertEqual(PrecompileContext.take_artifact("k").key, "k")
        self.assertIsNone(PrecompileContext.take_artifact("k"))
        self.assertIsNone(PrecompileContext.serialize_artifact_by_key("k"))


class _SessionStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)

    def forward(self, x):
        y = self.lin(x)
        if y.shape[0] > 2:
            y = y * 2
        return torch.relu(y)


def _session_breaks(x):
    y = x * 2
    torch._dynamo.graph_break()
    return y + 3


def _session_raises(x, boom):
    if boom:
        raise ValueError("boom")
    return x + 1


class TestPrecompileSession(torch._inductor.test_case.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _session(self, fn, **kwargs):
        from torch._dynamo.precompile_package import precompile_capture

        kwargs.setdefault("backend", "eager")
        kwargs.setdefault("dynamic", False)
        return precompile_capture(fn, **kwargs)

    def test_each_call_folds_into_the_entry_as_a_guarded_variant(self):
        model = _SessionStep()
        session = self._session(model)
        with session as cap:
            for rows in (2, 3, 4):
                x = torch.randn(rows, 4)
                self.assertEqual(cap(x), model(x))
        entry = session._package.cache_entry()
        self.assertEqual(entry.fn_name, "_SessionStep.forward")
        self.assertEqual(len(entry.codes), 1)
        self.assertEqual(len(entry.codes[0].guarded_codes), 3)
        self.assertFalse(entry.codes[0].bypassed)

    def test_a_graph_break_records_the_resume_frame(self):
        session = self._session(_session_breaks)
        with session as cap:
            self.assertEqual(cap(torch.ones(3)), torch.ones(3) * 2 + 3)
        entry = session._package.cache_entry()
        self.assertEqual(len(entry.codes), 2)
        self.assertTrue(any(c.install_to_global for c in entry.codes))
        self.assertEqual(len(entry.backend_ids), 2)

    def test_recompile_limit_caps_the_variants_but_not_the_calls(self):
        model = _SessionStep()
        session = self._session(model, recompile_limit=2)
        with session as cap:
            for rows in (2, 3, 4, 5):
                x = torch.randn(rows, 4)
                self.assertEqual(cap(x), model(x))
        entry = session._package.cache_entry()
        self.assertLessEqual(len(entry.codes[0].guarded_codes), 2)

    def test_session_is_one_shot_and_refuses_reentry(self):
        from torch._dynamo.exc import PackageError

        session = self._session(_session_breaks)
        with session as cap:
            with self.assertRaisesRegex(PackageError, "already active"):
                session.__enter__()
            cap(torch.ones(3))
        with self.assertRaisesRegex(RuntimeError, "not active"):
            cap(torch.ones(3))
        with self.assertRaisesRegex(RuntimeError, "cannot be re-entered"):
            session.__enter__()

    def test_an_error_inside_the_block_is_recorded_once_and_propagates(self):
        session = self._session(_session_raises)
        with session as cap:
            self.assertEqual(cap(torch.ones(2), False), torch.ones(2) + 1)
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "boom"):
                    cap(torch.ones(2), True)
        self.assertEqual(session._capture_errors, ["ValueError: boom"])

    def test_eager_backends_survive_exit_for_the_render(self):
        session = self._session(_session_breaks)
        with session as cap:
            cap(torch.ones(3))
        entry = session._package.cache_entry()
        self.assertEqual(set(session._package.cached_backends), set(entry.backend_ids))
        for backend in session._package.cached_backends.values():
            self.assertTrue(callable(backend))

    def test_inductor_artifacts_are_taken_from_the_precompile_context(self):
        from torch._dynamo.precompile_context import PrecompileContext

        session = self._session(_session_breaks, backend="inductor")
        with session as cap:
            cap(torch.ones(3))
        entry = session._package.cache_entry()
        self.assertEqual(set(session._backend_artifacts), set(entry.backend_ids))
        for backend_id in entry.backend_ids:
            self.assertIsNone(PrecompileContext.serialize_artifact_by_key(backend_id))
        self.assertEqual(session._package.cached_backends, {})

    def test_entry_fn_of_resolves_modules_and_refuses_the_rest(self):
        from torch._dynamo.precompile_package import _entry_fn_of

        model = _SessionStep()
        self.assertIs(_entry_fn_of(model).__func__, _SessionStep.forward)
        self.assertIs(_entry_fn_of(model).__self__, model)
        self.assertIs(_entry_fn_of(_session_breaks), _session_breaks)
        with self.assertRaisesRegex(TypeError, "has no __code__"):
            _entry_fn_of(functools.partial(_session_breaks))
        with self.assertRaisesRegex(TypeError, "expected a callable"):
            _entry_fn_of(3)


class _SessionReadsAttr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.scale = 2

    def forward(self, x):
        return self.lin(x) * self.scale


def _drop_scale(entries):
    return ["scale" not in e.name for e in entries]


def _session_summary_raises(x):
    raise ValueError("boom")


class TestPrecompileSessionSummary(torch._inductor.test_case.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _session(self, fn, **kwargs):
        from torch._dynamo.precompile_package import precompile_capture

        kwargs.setdefault("backend", "eager")
        kwargs.setdefault("dynamic", False)
        return precompile_capture(fn, **kwargs)

    def _gate(self, session, **flags):
        flags = {
            "require_complete": True,
            "require_no_risky_drops": True,
            "require_no_dropped_guards": False,
            **flags,
        }
        return session._gated_summary(**flags)

    def test_summary_counts_frames_variants_and_guards(self):
        model = _SessionReadsAttr()
        session = self._session(model)
        with session as cap:
            cap(torch.randn(2, 4))
            cap(torch.randn(3, 4))
        summary = session.summary()
        self.assertEqual(summary.frames, 1)
        self.assertEqual(summary.guarded_codes, 2)
        self.assertEqual(summary.backend_graphs, 2)
        self.assertEqual(summary.bypassed, ())
        self.assertEqual(summary.capture_errors, ())
        self.assertTrue(summary.complete)
        self.assertIn("TENSOR_MATCH", summary.kept_guard_types())
        self.assertIn("MODULE_MATCH", summary.dropped_guard_types())
        self.assertEqual(summary.risky_dropped_guards, ())
        self.assertEqual(summary.policy_dropped_guards, ())

    def test_a_custom_filter_composes_with_the_default_and_its_drops_are_risky(self):
        session = self._session(_SessionReadsAttr(), guard_filter_fn=_drop_scale)
        with session as cap:
            cap(torch.randn(2, 4))
        summary = session.summary()
        self.assertTrue(any("scale" in name for _, name in summary.dropped_guards))
        self.assertTrue(
            any("scale" in name for _, name in summary.risky_dropped_guards)
        )
        self.assertIn("MODULE_MATCH", summary.dropped_guard_types())
        self.assertTrue(
            any("scale" in name for _, name, _ in summary.dropped_guard_code)
        )

    def test_invariants_classify_guards_across_variants(self):
        session = self._session(_SessionReadsAttr())
        with session as cap:
            cap(torch.randn(2, 4))
            cap(torch.randn(3, 4))
        (frame,) = session.invariants()
        self.assertEqual(frame.frame, "forward")
        self.assertEqual(frame.variants, 2)
        varying = {(f.guard_type, f.source) for f in frame.varying}
        self.assertIn(("TENSOR_MATCH", "x"), varying)
        self.assertTrue(any("scale" in f.source for f in frame.invariant))
        self.assertIsInstance(frame.invariant[0].render(), str)

    def test_invariants_report_is_written_on_a_clean_exit_only(self):
        model = _SessionReadsAttr()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "inv.txt")
            with self._session(model, invariants=path) as cap:
                cap(torch.randn(2, 4))
            with open(path, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("frame forward", text)
            self.assertIn("invariant", text)
            os.unlink(path)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with self._session(model, invariants=path) as cap:
                    cap(torch.randn(2, 4))
                    raise RuntimeError("boom")
            self.assertFalse(os.path.exists(path))

    def test_gates_refuse_an_empty_or_failed_capture(self):
        from torch._dynamo.exc import PackageError

        session = self._session(_SessionReadsAttr())
        with session:
            pass
        with self.assertRaisesRegex(PackageError, "captured no compiled code"):
            self._gate(session)
        self.assertEqual(self._gate(session, require_complete=False).guarded_codes, 0)

        session = self._session(_session_summary_raises)
        with session as cap:
            with self.assertRaisesRegex(ValueError, "boom"):
                cap(torch.ones(2))
        with self.assertRaisesRegex(PackageError, "incomplete because capture raised"):
            self._gate(session)

    def test_gates_refuse_risky_and_plain_drops_as_asked(self):
        from torch._dynamo.exc import PackageError

        session = self._session(_SessionReadsAttr(), guard_filter_fn=_drop_scale)
        with session as cap:
            cap(torch.randn(2, 4))
        with self.assertRaisesRegex(PackageError, "can affect dispatch"):
            self._gate(session)
        with self.assertRaisesRegex(PackageError, "were not serialized"):
            self._gate(
                session, require_no_risky_drops=False, require_no_dropped_guards=True
            )
        self.assertTrue(self._gate(session, require_no_risky_drops=False).complete)


class _RenderChild(torch.nn.Module):
    def forward(self, x):
        y = x * 2
        torch._dynamo.graph_break()
        return y + 1


class _RenderParent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _RenderChild()

    def forward(self, x):
        return self.child(x) - 1


def _render_with_tensor_default(x, bias=torch.ones(1)):
    return x + bias


class TestPrecompileRender(torch._inductor.test_case.TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _session(self, fn, **kwargs):
        from torch._dynamo.precompile_package import precompile_capture

        kwargs.setdefault("backend", "eager")
        kwargs.setdefault("dynamic", False)
        return precompile_capture(fn, **kwargs)

    def _forward(self, python_code):
        from torch._precompile import _make_inlined_forward

        return _make_inlined_forward(python_code, "precompile.load", warn=False)

    def test_standalone_artifact_serves_every_captured_variant(self):
        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        model = _SessionStep()
        session = self._session(model)
        xs = [torch.randn(2, 4), torch.randn(3, 4)]
        with session as cap:
            for x in xs:
                cap(x)
        python_code, cache = session.snapshot_artifact()
        self.assertIn('SERVING_MODE = "standalone"', python_code)
        self.assertIn('TRACER = "dynamo"', python_code)
        self.assertIn("BACKEND = 'eager'", python_code)
        self.assertIn("FRAMES = [\n    ('forward', 2),\n]", python_code)
        self.assertIn("_FRAMES = ", python_code)
        self.assertIn("_BACKENDS = ", python_code)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["tracer"], "dynamo")
        self.assertEqual(blob["backend"], "eager")
        self.assertEqual(
            blob["code_hash"], hashlib.sha256(python_code.encode()).hexdigest()
        )
        self.assertIsNone(blob["artifact"])
        forward = self._forward(python_code)
        for x in xs:  # the entry is forward(self, x): the model is passed again
            self.assertEqual(forward(model, x), model(x))

    def test_a_graph_break_in_the_entry_frame_stays_standalone(self):
        session = self._session(_session_breaks)
        with session as cap:
            cap(torch.ones(3))
        python_code, _ = session.snapshot_artifact()
        self.assertIn('SERVING_MODE = "standalone"', python_code)
        forward = self._forward(python_code)
        self.assertEqual(forward(torch.ones(3)), torch.ones(3) * 2 + 3)

    def test_inductor_subgraphs_render_as_readable_source(self):
        model = _SessionStep()
        session = self._session(model, backend="inductor", keep_graphs=True)
        x = torch.randn(3, 4)
        # An inference capture: with grad enabled AOTAutograd lowers the
        # backward lazily and records the bundle only once one runs.
        with session as cap, torch.no_grad():
            cap(x)
        python_code, cache = session.snapshot_artifact()
        self.assertIn("_SUBGRAPHS = {}", python_code)
        self.assertIn("def call_s0(", python_code)
        self.assertIn("] = call_s0", python_code)
        self.assertIn("READABLE below", python_code)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertEqual(blob["backend"], "inductor")
        forward = self._forward(python_code)
        with torch.no_grad():  # the variant was captured, and is guarded, under no_grad
            self.assertEqual(forward(model, x), model(x))

    def test_snapshot_mid_block_leaves_the_capture_open(self):
        model = _SessionStep()
        session = self._session(model)
        with session as cap:
            cap(torch.randn(2, 4))
            first, _ = session.snapshot_artifact()
            self.assertIn("('forward', 1)", first)
            cap(torch.randn(3, 4))
            second, _ = session.snapshot_artifact()
            self.assertIn("('forward', 2)", second)
        self.assertIn("('forward', 2)", session.snapshot_artifact()[0])

    def test_a_break_inside_a_child_frame_needs_the_installed_mode(self):
        from torch._precompile import PrecompileError

        # Without nested graph breaks the child's frame is compiled on its own,
        # out of reach of the entry's bytecode: only the installed serving mode
        # can dispatch it, and this build refuses rather than serving it eager.
        with torch._dynamo.config.patch(nested_graph_breaks=False):
            session = self._session(_RenderParent())
            with session as cap:
                cap(torch.ones(3))
        self.assertEqual(len(session._package.cache_entry().codes), 4)
        with self.assertRaisesRegex(PrecompileError, "not available yet"):
            session.snapshot_artifact()

    def test_a_frame_without_variants_is_uncovered_and_refused(self):
        from torch._dynamo.exc import PackageError

        # With nested graph breaks (the default in this harness) the child's
        # continuation is recorded as a frame Dynamo never compiled a variant
        # for, which require_complete reports as an uncovered frame.
        session = self._session(_RenderParent())
        with session as cap:
            cap(torch.ones(3))
        summary = session.summary()
        self.assertTrue(summary.uncovered_frames)
        self.assertFalse(summary.complete)
        with self.assertRaisesRegex(PackageError, "produced NO guarded code"):
            session.snapshot_artifact()
        python_code, _ = session.snapshot_artifact(require_complete=False)
        self.assertIn('SERVING_MODE = "standalone"', python_code)

    def test_a_tensor_default_is_refused_before_capture(self):
        from torch._precompile import (
            _reject_uninstallable_entry_defaults,
            PrecompileError,
        )

        with self.assertRaisesRegex(PrecompileError, "tensor default argument"):
            _reject_uninstallable_entry_defaults(_render_with_tensor_default)
        _reject_uninstallable_entry_defaults(_session_breaks)
        _reject_uninstallable_entry_defaults(functools.partial(_session_breaks))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
