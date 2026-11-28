from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from tools.testing.target_determination import qualification as q
from tools.testing.test_run import TestRun


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_candidate() -> ModuleType:
    path = REPO_ROOT / "tools/testing/target_determination/heuristics/interface.py"
    spec = importlib.util.spec_from_file_location("_td_test_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the actual candidate source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CopyTrap(TestRun):
    def __copy__(self) -> TestRun:
        raise RuntimeError("copy called")


class MutatingUnion(TestRun):
    def __ior__(self, other: TestRun) -> TestRun:
        self._included = frozenset()
        self._excluded = frozenset()
        return self


class ReflectTrap(TestRun):
    def __rand__(self, other: TestRun) -> TestRun:
        if self.test_file != other.test_file:
            raise RuntimeError("cross-file interception")
        return TestRun.__and__(other, self)


class IdentityAware(TestRun):
    first: TestRun

    def __rand__(self, other: TestRun) -> TestRun:
        if other is self.first:
            raise RuntimeError("original key identity")
        return TestRun.__and__(other, self)


class TestGuardCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate = load_candidate()

    def test_exact_reviewed_candidate_hash(self) -> None:
        self.assertEqual(q._file_digest(Path(self.candidate.__file__)), q.CANDIDATE_SHA256)

    def test_copy_fallback(self) -> None:
        tp = self.candidate.TestPrioritizations(["a"], {})
        tp._test_scores = {CopyTrap("a"): 0.0}
        with self.assertRaisesRegex(RuntimeError, "copy called"):
            tp.validate()

    def test_all_keys_guard_preserves_second_member_identity(self) -> None:
        tp = self.candidate.TestPrioritizations(["a"], {})
        first = TestRun("a", included=["C1"])
        second = IdentityAware("a", excluded=["C1"])
        second.first = first
        tp._test_scores = {first: 0.5, second: 0.0}
        tp.validate()
        self.assertEqual(first.included(), frozenset(["C1"]))

    def test_mutating_union_does_not_mutate_input_key(self) -> None:
        tp = self.candidate.TestPrioritizations(["a"], {})
        first = MutatingUnion("a", included=["C1"])
        tp._test_scores = {first: 0.5, TestRun("a", excluded=["C1"]): 0.0}
        tp.validate()
        self.assertEqual(first.included(), frozenset(["C1"]))

    def test_reflected_cross_file_fallback(self) -> None:
        for method in ("set_test_score", "add_test_score"):
            with self.subTest(method=method):
                tp = self.candidate.TestPrioritizations(["a", "b"], {})
                with self.assertRaisesRegex(RuntimeError, "cross-file interception"):
                    getattr(tp, method)(ReflectTrap("a"), 0.5)

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("_td_test_candidate", None)

    def test_instance_overrides_use_copy_fallback(self) -> None:
        with q.load_reference(REPO_ROOT) as original:
            for field in ("is_full_file", "is_empty"):
                outcomes = []
                for module in (original, self.candidate):
                    tp = module.TestPrioritizations(["a"], {})
                    run = TestRun("a")
                    setattr(run, field, lambda: False)
                    tp._test_scores = {run: 0.0}
                    try:
                        tp.validate()
                        outcomes.append(None)
                    except Exception as error:
                        outcomes.append((type(error).__name__, str(error)))
                self.assertEqual(outcomes, [None, None])

    def test_extra_instance_field_disables_cross_file_shortcut(self) -> None:
        run = TestRun("a")
        run.is_empty = lambda: False
        calls = []
        original = TestRun.__and__

        def observe(left: TestRun, right: TestRun) -> TestRun:
            calls.append(True)
            return original(left, right)

        with patch.object(TestRun, "__and__", observe):
            self.assertFalse(self.candidate._test_runs_intersect(run, TestRun("b")))
        self.assertEqual(calls, [True])

    def test_malformed_copy_error_matches_reference(self) -> None:
        with q.load_reference(REPO_ROOT) as original:
            for mode in ("both", "stored-name"):
                outcomes = []
                for module in (original, self.candidate):
                    tp = module.TestPrioritizations(["a"], {})
                    run = TestRun("a")
                    if mode == "both":
                        run._included = frozenset(["C1"])
                        run._excluded = frozenset(["C2"])
                    else:
                        run.test_file = "a::C"
                        tp._original_tests = frozenset(["a::C"])
                    tp._test_scores = {run: 0.0}
                    try:
                        tp.validate()
                    except Exception as error:
                        outcomes.append((type(error).__name__, str(error)))
                    else:
                        outcomes.append(None)
                self.assertEqual(outcomes[0], outcomes[1])

    def test_ordered_state_after_mixed_operations(self) -> None:
        rng = random.Random(20261123)
        with q.load_reference(REPO_ROOT) as original:
            left = original.TestPrioritizations(["a", "b", "c"], {})
            right = self.candidate.TestPrioritizations(["a", "b", "c"], {})
            for _ in range(100):
                method = rng.choice(["set_test_score", "add_test_score"])
                name = rng.choice(["a", "b", "c", "unknown"])
                mode = rng.randrange(3)
                classes = rng.sample(["C1", "C2", "C3"], rng.randrange(1, 4))
                score = rng.choice([-1.0, 0.0, 0.25, 0.5, 1.0])
                for tp in (left, right):
                    getattr(tp, method)(TestRun(name, included=classes if mode == 1 else [], excluded=classes if mode == 2 else []), score)
                self.assertEqual(q._encoded(q._state(left)), q._encoded(q._state(right)))


class TestSameInputQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate = load_candidate()

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("_td_test_candidate", None)

    def inputs(self) -> tuple[q.SnapshotRecorder, object, object]:
        keys = [q.HeuristicKey("normal", False), q.HeuristicKey("trial", True)]
        recorder = q.SnapshotRecorder(["a", "b"], keys)
        normal = self.candidate.TestPrioritizations(["a", "b"], {TestRun("a"): 0.25})
        trial = self.candidate.TestPrioritizations(["a", "b"], {TestRun("a", included=["C1"]): 0.75})
        recorder.capture(keys[0], normal)
        recorder.capture(keys[1], trial)
        return recorder, normal, trial

    def test_capture_contains_trial_class_fragments_and_zero_scores(self) -> None:
        recorder, _, _ = self.inputs()
        data = recorder.inputs()
        self.assertTrue(data["heuristics"][1]["trial_mode"])
        self.assertTrue(any(row[1] == ["C1"] for row in data["heuristics"][1]["scores"]))
        self.assertTrue(any(row[3] == 0.0 for row in data["heuristics"][1]["scores"]))
        recorder.check_live()

    def test_restore_preserves_operation_prefix_order_not_constructor_proxy(self) -> None:
        rng = random.Random(20261123)
        tp = self.candidate.TestPrioritizations(["a", "b", "c"], {})
        found_proxy_difference = False
        for _ in range(40):
            mode = rng.randrange(3)
            classes = rng.sample(["C1", "C2", "C3"], rng.randrange(1, 4))
            getattr(tp, rng.choice(["set_test_score", "add_test_score"]))(TestRun(rng.choice(["a", "b", "c"]), included=classes if mode == 1 else [], excluded=classes if mode == 2 else []), rng.choice([-1.0, 0.0, 0.25, 0.5, 1.0]))
            record = q.CapturedResult("trial", True, tp._original_tests, q._scores(tp))
            restored = q._restore(record, self.candidate)
            self.assertEqual(q._encoded(q._state(restored)), q._encoded(q._state(tp)))
            proxy = self.candidate.TestPrioritizations(tp._original_tests, dict(tp._test_scores))
            found_proxy_difference |= q._state(proxy) != q._state(tp)
        self.assertTrue(found_proxy_difference)

    def test_actual_default_and_trial_match_reference(self) -> None:
        recorder, normal, trial = self.inputs()
        actual = self.candidate.AggregatedHeuristics(list(recorder.tests))
        for result, tp in zip(recorder.results, (normal, trial)):
            actual.add_heuristic_results(q.HeuristicKey(result.name, result.trial_mode), tp)
        expected = {"default": q._output(actual.get_aggregated_priorities()), "trial": q._output(actual.get_aggregated_priorities(include_trial=True))}
        with q.load_reference(REPO_ROOT) as original:
            for module in (original, self.candidate):
                _, result = q.measure_component(recorder, module)
                self.assertEqual(q._encoded(result), q._encoded(expected))
        recorder.check_live()

    def test_trial_correctness_is_after_primary_timer(self) -> None:
        recorder, _, _ = self.inputs()
        events = []
        method = self.candidate.AggregatedHeuristics.get_aggregated_priorities

        def observe(aggregate: object, include_trial: bool = False) -> object:
            events.append("trial" if include_trial else "default")
            return method(aggregate, include_trial=include_trial)

        def wall_clock() -> int:
            events.append("wall")
            return len(events)

        with patch.object(self.candidate.AggregatedHeuristics, "get_aggregated_priorities", observe), patch.object(q.time, "perf_counter_ns", wall_clock):
            q.measure_component(recorder, self.candidate)
        self.assertEqual(events, ["wall", "default", "wall", "trial"])

    def test_live_score_and_universe_changes_rejected(self) -> None:
        recorder, normal, _ = self.inputs()
        normal._test_scores[next(iter(normal._test_scores))] = 99
        with self.assertRaises(q.QualificationInvalid):
            recorder.check_live()
        recorder, _, _ = self.inputs()
        recorder.live_tests.reverse()
        with self.assertRaises(q.QualificationInvalid):
            recorder.check_live()

    def test_missing_trial_input_is_invalid_not_zero_success(self) -> None:
        recorder, _, _ = self.inputs()
        recorder.results.pop()
        with self.assertRaises(q.QualificationInvalid):
            recorder.check_live()
        with self.assertRaises(q.QualificationInvalid):
            q.SnapshotRecorder([], [q.HeuristicKey("trial", True)])

    def test_noncanonical_input_invalidates_capture(self) -> None:
        recorder, normal, _ = self.inputs()
        normal._test_scores = {ReflectTrap("a"): 0.5}
        recorder.capture(q.HeuristicKey("extra", False), normal)
        self.assertTrue(recorder.errors)
        with self.assertRaises(q.QualificationInvalid):
            recorder.check_live()

    def test_offline_reference_missing_has_no_network_fallback(self) -> None:
        with patch.object(q, "_read", side_effect=FileNotFoundError("fixture missing")):
            with self.assertRaises(FileNotFoundError):
                with q.load_reference(REPO_ROOT):
                    self.fail("Missing fixture must not be admitted")

    def test_reference_hash_mismatch_is_invalid(self) -> None:
        with patch.object(q, "_read", return_value=b"wrong"):
            with self.assertRaisesRegex(q.QualificationInvalid, "SHA-256 mismatch"):
                with q.load_reference(REPO_ROOT):
                    self.fail("Mismatched reference must not be imported")

    def test_empty_registry_and_report_bound(self) -> None:
        with self.assertRaises(q.QualificationInvalid):
            q.SnapshotRecorder(["a"], [])
        with patch.object(q, "MAX_BYTES", 10):
            with self.assertRaises(q.QualificationInvalid):
                q._encoded({"large": "x" * 20})

    def test_invalid_context_is_not_confused_with_runtime_or_module_errors(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(q.experiment_context())

    def test_local_runtime_rejection_has_exact_reason(self) -> None:
        context = {"repository": "other", "head_repository": q.EXPERIMENT_FORK,
                   "head_ref": q.EXPERIMENT_BRANCH, "head_sha": "0" * 40}
        with self.assertRaisesRegex(q.QualificationInvalid, "Owned ordinary CI stock CPython 3.12"):
            q._runtime_context(context, REPO_ROOT)

    def test_existing_timer_is_not_overwritten(self) -> None:
        with patch.object(q.signal, "getitimer", return_value=(1.0, 0.0)):
            with self.assertRaises(q.QualificationInvalid):
                with q._deadline():
                    self.fail("Existing timer must not be changed")

    def test_both_cpu_and_wall_timers_are_restored(self) -> None:
        with q._deadline():
            self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
            self.assertGreater(signal.getitimer(signal.ITIMER_PROF)[0], 0)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(signal.getitimer(signal.ITIMER_PROF), (0.0, 0.0))


class TestQualificationWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate = load_candidate()

    @classmethod
    def tearDownClass(cls) -> None:
        sys.modules.pop("_td_test_candidate", None)

    def live(self) -> tuple[q.SnapshotRecorder, object, object]:
        keys = [q.HeuristicKey("normal", False), q.HeuristicKey("trial", True)]
        recorder = q.SnapshotRecorder(["a", "b"], keys)
        aggregate = self.candidate.AggregatedHeuristics(["a", "b"])
        for key in keys:
            tp = self.candidate.TestPrioritizations(
                ["a", "b"], {TestRun("a", included=["C1"]): 0.5}
            )
            recorder.capture(key, tp)
            aggregate.add_heuristic_results(key, tp)
        return recorder, aggregate, aggregate.get_aggregated_priorities()

    def context(self) -> dict[str, object]:
        return {"repository": "pytorch/pytorch", "head_repository": q.EXPERIMENT_FORK,
                "head_ref": q.EXPERIMENT_BRANCH, "head_sha": "0" * 40,
                "pull_request": 123, "run_id": "unit-only", "run_attempt": "unit-only"}

    def test_full_success_path_uses_actual_sources_and_restores_before_emission(self) -> None:
        recorder, aggregate, default = self.live()
        # Only context/runtime admission is isolated. This is a local wiring test, not CI timing.
        with patch.object(q, "_runtime_context", return_value="unit-checkout"):
            report = q.run_ci_qualification(recorder, aggregate, default, self.context())
        self.assertEqual(report["status"], "passed")
        self.assertEqual([pair["order"] for pair in report["pairs"]], [list(p) for p in q.PAIRS])
        self.assertEqual(report["pre_post_input_sha256"][0], report["pre_post_input_sha256"][1])
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertEqual(signal.getitimer(signal.ITIMER_PROF), (0.0, 0.0))
        output = io.StringIO()
        with redirect_stdout(output):
            q.emit_result(report)
        self.assertEqual(output.getvalue().count("TD_QUALIFICATION_RESULT_JSON "), 1)

    def test_mismatch_preserves_partial_times_and_one_invalid_record(self) -> None:
        recorder, aggregate, default = self.live()
        measure = q.measure_component

        def corrupt(rec: object, module: object) -> object:
            timings, outputs = measure(rec, module)
            outputs["default"]["state"]["scores"][0][3] = 999
            return timings, outputs

        with patch.object(q, "_runtime_context", return_value="unit-checkout"), patch.object(q, "measure_component", corrupt):
            with self.assertRaises(q.QualificationInvalid) as caught:
                q.run_ci_qualification(recorder, aggregate, default, self.context())
        self.assertIn("Measured output/order differs", str(caught.exception))
        report = caught.exception.report
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(len(report["pairs"][0]["arms"]), 1)
        output = io.StringIO()
        with redirect_stdout(output):
            q.emit_result(report)
        self.assertEqual(output.getvalue().count("TD_QUALIFICATION_RESULT_JSON "), 1)
        self.assertNotIn('"status":"passed"', output.getvalue())

    def test_expiry_during_cleanup_cannot_return_passed(self) -> None:
        recorder, aggregate, default = self.live()

        @contextmanager
        def expire_on_cleanup() -> object:
            yield
            raise q.QualificationInvalid("injected expiry during cleanup")

        with patch.object(q, "_runtime_context", return_value="unit-checkout"), patch.object(q, "_deadline", expire_on_cleanup):
            with self.assertRaisesRegex(q.QualificationInvalid, "injected expiry") as caught:
                q.run_ci_qualification(recorder, aggregate, default, self.context())
        self.assertEqual(caught.exception.report["status"], "invalid")

    def test_loaded_stock_wrapper_is_not_attested_by_disk_hash(self) -> None:
        source = (REPO_ROOT / "tools/testing/test_run.py").read_bytes()
        old = TestRun.__and__

        def wrapper(left: TestRun, right: TestRun) -> TestRun:
            return old(left, right)

        with q._module(source, "_td_unit_stock") as stock, patch.object(TestRun, "__and__", wrapper):
            with self.assertRaisesRegex(q.QualificationInvalid, "Loaded TestRun code differs"):
                q._verify_test_run(stock)

    def test_instance_overrides_are_not_normalized_away(self) -> None:
        recorder, aggregate, _ = self.live()
        tp = recorder.live[0][1]
        next(iter(tp._test_scores)).is_full_file = lambda: False
        with self.assertRaisesRegex(q.QualificationInvalid, "instance overrides"):
            recorder.check_live()
        tp = self.candidate.TestPrioritizations(["a"], {})
        tp.validate = lambda: None
        with self.assertRaisesRegex(q.QualificationInvalid, "instance overrides"):
            q._scores(tp)

    def test_equal_live_keys_cannot_hide_rejected_work(self) -> None:
        class EqualKey:
            def __init__(self, name: str, trial: bool) -> None:
                self.name, self.trial_mode = name, trial

            def __hash__(self) -> int:
                return 1

            def __eq__(self, other: object) -> bool:
                return True

        keys = [EqualKey("a", False), EqualKey("b", True)]
        recorder = q.SnapshotRecorder(["a"], keys)
        aggregate = self.candidate.AggregatedHeuristics(["a"])
        for key in keys:
            tp = self.candidate.TestPrioritizations(["a"], {})
            recorder.capture(key, tp)
            try:
                aggregate.add_heuristic_results(key, tp)
            except ValueError:
                pass
        with self.assertRaisesRegex(q.QualificationInvalid, "did not accept"):
            q._live_bindings(recorder, aggregate, aggregate.get_aggregated_priorities())

    def test_cumulative_streaming_capture_bound(self) -> None:
        keys = [q.HeuristicKey("normal", False), q.HeuristicKey("trial", True)]
        with patch.object(q, "MAX_BYTES", 1024):
            recorder = q.SnapshotRecorder(["a"], keys)
            tp = self.candidate.TestPrioritizations(
                ["a"], {TestRun("a", included=["X" * 512, "Y" * 512]): 0.5}
            )
            recorder.capture(keys[0], tp)
            self.assertTrue(recorder.errors)
            self.assertEqual(recorder.results, [])

    def test_owned_event_predicate_not_downstream_runtime_error(self) -> None:
        event = {"repository": {"full_name": "pytorch/pytorch"},
                 "pull_request": {"number": 123, "head": {
                     "repo": {"full_name": q.EXPERIMENT_FORK},
                     "ref": q.EXPERIMENT_BRANCH, "sha": "0" * 40}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.json"
            path.write_text(json.dumps(event))
            env = {"CI": "true", "GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "pytorch/pytorch",
                   "GITHUB_HEAD_REF": q.EXPERIMENT_BRANCH, "GITHUB_EVENT_NAME": "pull_request",
                   "GITHUB_EVENT_PATH": str(path)}
            with patch.dict(os.environ, env, clear=True):
                self.assertIsNotNone(q.experiment_context())
                event["pull_request"]["head"]["repo"]["full_name"] = "other/pytorch"
                path.write_text(json.dumps(event))
                self.assertIsNone(q.experiment_context())

    def test_legacy_entry_call_keeps_exact_one_argument(self) -> None:
        # Load unmodified function bodies without optional package dependencies; IO is unit-isolated.
        path = REPO_ROOT / "tools/testing/do_target_determination_for_s3.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = {"Any": object, "AggregatedHeuristics": self.candidate.AggregatedHeuristics,
                     "TestPrioritizations": self.candidate.TestPrioritizations, "TESTS": ["a", "b"]}
        for name in ("get_test_times", "get_test_class_times", "get_test_file_ratings", "get_test_class_ratings",
                     "get_td_heuristic_historial_edited_files_json", "get_td_heuristic_profiling_json",
                     "copy_pytest_cache", "copy_additional_previous_failures"):
            namespace[name] = lambda: None
        calls = []

        def one_arg(tests: object) -> object:
            calls.append(tests)
            return self.candidate.AggregatedHeuristics(["a", "b"])

        namespace["get_test_prioritizations"] = one_arg
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        namespace["_prioritizations"]()
        self.assertEqual(calls, [["a", "b"]])


if __name__ == "__main__":
    unittest.main()
