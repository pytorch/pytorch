# Owner(s): ["module: dynamo"]

import io
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.dynamo import check_graph_breaks


sys.path.remove(str(REPO_ROOT))


def actual_results(**overrides):
    row = {
        "name": "model",
        "accuracy": "pass",
        "unique_graphs": 1,
        "graph_breaks": 0,
        "calls_captured": 100,
        "fallbacks_to_eager": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def expected_results(**overrides):
    row = {
        "name": "model",
        "accuracy": "pass",
        "graph_breaks": 0,
        "calls_captured": 100,
        "fallbacks_to_eager": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestCheckGraphBreaks(TestCase):
    def check(self, actual, expected):
        output = io.StringIO()
        with redirect_stdout(output):
            failures, message = check_graph_breaks.check_graph_breaks(
                actual, expected, "expected.csv"
            )
        return failures, message, output.getvalue()

    def test_valid_results(self):
        failures, message, output = self.check(actual_results(), expected_results())

        self.assertEqual(failures, [])
        self.assertEqual(message, "")
        self.assertIn("PASS", output)

    def test_zero_tracked_metrics_are_valid(self):
        actual = actual_results(calls_captured=0)
        expected = expected_results(calls_captured=0)

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, [])
        self.assertEqual(message, "")
        self.assertIn("PASS", output)

    def test_zero_graph_capture_is_invalid(self):
        failures, message, output = self.check(
            actual_results(unique_graphs=0), expected_results()
        )

        self.assertEqual(failures, ["model"])
        self.assertIn("missing or invalid required Dynamo metrics", message)
        self.assertIn("unique_graphs=0, expected Dynamo graph capture", output)

    def test_zero_capture_matches_zero_coverage_baseline(self):
        metrics = {"graph_breaks": 2, "calls_captured": 0, "fallbacks_to_eager": 2}
        failures, message, output = self.check(
            actual_results(unique_graphs=0, **metrics), expected_results(**metrics)
        )

        self.assertEqual(failures, [])
        self.assertEqual(message, "")
        self.assertIn("PASS", output)

    @parametrize("missing_column", (False, True))
    def test_zero_capture_requires_explicit_baseline(self, missing_column):
        expected = expected_results(calls_captured=None)
        if missing_column:
            expected = expected.drop(columns="calls_captured")

        failures, message, output = self.check(
            actual_results(unique_graphs=0, calls_captured=0), expected
        )

        self.assertEqual(failures, ["model"])
        self.assertIn("invalid", message)
        self.assertIn("expected Dynamo graph capture", output)

    @parametrize(
        "field",
        ("unique_graphs", "graph_breaks", "calls_captured", "fallbacks_to_eager"),
    )
    def test_zero_coverage_baseline_still_requires_valid_metrics(self, field):
        actual = actual_results(unique_graphs=0, calls_captured=0)
        actual[field] = None

        failures, message, output = self.check(
            actual, expected_results(calls_captured=0)
        )

        self.assertEqual(failures, ["model"])
        self.assertIn("invalid", message)
        self.assertIn(field, output)

    @parametrize("field", ("graph_breaks", "fallbacks_to_eager"))
    @parametrize("actual,expected,status", ((3, 2, "FAIL"), (1, 2, "IMPROVED")))
    def test_zero_coverage_baseline_still_checks_metric_changes(
        self, field, actual, expected, status
    ):
        failures, message, output = self.check(
            actual_results(unique_graphs=0, calls_captured=0, **{field: actual}),
            expected_results(calls_captured=0, **{field: expected}),
        )

        self.assertEqual(failures, ["model"])
        self.assertIn(f"{status}:", output)
        self.assertIn(f"{field}={actual}, expected={expected}", output)
        self.assertIn("reflect the new baseline", message)

    @parametrize(
        "field",
        ("unique_graphs", "graph_breaks", "calls_captured", "fallbacks_to_eager"),
    )
    @parametrize("value", (float("nan"), float("inf"), "invalid", -1, 0.5, True))
    def test_invalid_required_actual_metric(self, field, value):
        failures, message, output = self.check(
            actual_results(**{field: value}), expected_results()
        )

        self.assertEqual(failures, ["model"])
        self.assertIn("missing or invalid required Dynamo metrics", message)
        self.assertIn(field, output)

    @parametrize(
        "field",
        ("unique_graphs", "graph_breaks", "calls_captured", "fallbacks_to_eager"),
    )
    def test_missing_required_actual_metric(self, field):
        actual = actual_results().drop(columns=field)
        failures, message, output = self.check(actual, expected_results())

        self.assertEqual(failures, ["model"])
        self.assertIn("missing or invalid required Dynamo metrics", message)
        self.assertIn(field, output)

    def test_older_expected_schema_does_not_require_new_metrics(self):
        actual = actual_results().drop(columns=["calls_captured", "fallbacks_to_eager"])
        expected = expected_results().drop(
            columns=["calls_captured", "fallbacks_to_eager"]
        )

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, [])
        self.assertEqual(message, "")
        self.assertIn("PASS", output)

    def test_unavailable_expected_metrics_do_not_require_actual_values(self):
        actual = actual_results().drop(columns=["calls_captured", "fallbacks_to_eager"])
        expected = expected_results(
            calls_captured=float("nan"), fallbacks_to_eager=None
        )

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, [])
        self.assertEqual(message, "")
        self.assertIn("PASS", output)

    @parametrize(
        "actual_status,expected_status",
        (
            ("eager_1st_run_fail", "pass"),
            ("fail_to_run", "pass"),
            ("model_fail_to_load", "pass"),
            ("pass_due_to_skip", "pass"),
            ("OOM", "pass"),
            ("timeout", "pass"),
            ("fail_accuracy", "fail_accuracy"),
        ),
    )
    @parametrize("unique_graphs", (None, float("nan"), 0))
    def test_unavailable_capture_is_allowed_when_capture_is_not_required(
        self, actual_status, expected_status, unique_graphs
    ):
        actual = pd.DataFrame(
            [
                {
                    "name": "model",
                    "accuracy": actual_status,
                    "unique_graphs": unique_graphs,
                }
            ]
        )
        expected = pd.DataFrame(
            [
                {
                    "name": "model",
                    "accuracy": expected_status,
                    "graph_breaks": 0,
                }
            ]
        )

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, [])
        self.assertEqual(message, "")
        self.assertIn("EAGER_FAILED", output)

    @parametrize("field", ("unique_graphs", "calls_captured"))
    @parametrize("value", ("invalid", float("inf"), -1, 0.5, True))
    def test_optional_capture_rejects_malformed_supplied_metrics(self, field, value):
        overrides = {"accuracy": "fail_to_run", "unique_graphs": 0}
        overrides[field] = value
        actual = actual_results(**overrides)
        expected = expected_results(accuracy="fail_to_run")

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, ["model"])
        self.assertIn("invalid", message)
        self.assertIn(field, output)

    @parametrize("calls_captured", (100, None, float("nan")))
    def test_partial_capture_requires_metrics(self, calls_captured):
        actual = actual_results(accuracy="fail_to_run", calls_captured=calls_captured)
        expected = expected_results(accuracy="fail_to_run")

        failures, message, output = self.check(actual, expected)

        if calls_captured == 100:
            self.assertEqual(failures, [])
            self.assertIn("PASS", output)
        else:
            self.assertEqual(failures, ["model"])
            self.assertIn("invalid", message)
            self.assertIn("calls_captured", output)

    @parametrize("graphs", (0, 1))
    @parametrize(
        "field,value",
        (
            ("graph_breaks", None),
            ("graph_breaks", float("nan")),
            ("graph_breaks", float("inf")),
            ("calls_captured", "invalid"),
            ("fallbacks_to_eager", -1),
        ),
    )
    def test_invalid_baseline_counters(self, graphs, field, value):
        actual = actual_results(accuracy="fail_to_run", unique_graphs=graphs)
        expected = expected_results(accuracy="fail_to_run", **{field: value})

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, ["model"])
        self.assertIn("invalid", message)
        self.assertIn(f"expected {field}", output)
        self.assertNotIn("reflect the new baseline", message)

    def test_unreviewed_compiled_failure_requires_capture(self):
        actual = pd.DataFrame([{"name": "model", "accuracy": "fail_accuracy"}])

        failures, message, output = self.check(actual, expected_results())

        self.assertEqual(failures, ["model"])
        self.assertIn("missing or invalid required Dynamo metrics", message)
        self.assertIn("unique_graphs=None", output)

    def test_zero_capture_is_allowed_for_normalized_eager_result(self):
        model = "mobilenet_v3_large"
        actual = pd.DataFrame([{"name": model, "accuracy": "pass"}])
        expected = pd.DataFrame(
            [{"name": model, "accuracy": "pass", "graph_breaks": 0}]
        )

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, [])
        self.assertEqual(message, "")
        self.assertIn("EAGER_FAILED", output)

    def test_normalized_eager_model_rejects_malformed_capture(self):
        actual = actual_results(name="mobilenet_v3_large", unique_graphs="invalid")
        expected = expected_results(name="mobilenet_v3_large")

        failures, message, output = self.check(actual, expected)

        self.assertEqual(failures, ["mobilenet_v3_large"])
        self.assertIn("invalid", message)
        self.assertIn("unique_graphs", output)

    def test_model_name_does_not_exempt_unreviewed_compiled_failure(self):
        model = "mobilenet_v3_large"
        actual = actual_results(name=model, accuracy="fail_accuracy", unique_graphs=0)

        failures, message, output = self.check(actual, expected_results(name=model))

        self.assertEqual(failures, [model])
        self.assertIn("invalid", message)
        self.assertIn("expected Dynamo graph capture", output)

    @parametrize("field", ("graph_breaks", "fallbacks_to_eager"))
    @parametrize("actual,expected,status", ((1, 0, "FAIL"), (0, 1, "IMPROVED")))
    def test_exact_metric_changes_require_baseline_update(
        self, field, actual, expected, status
    ):
        failures, message, output = self.check(
            actual_results(**{field: actual}), expected_results(**{field: expected})
        )

        self.assertEqual(failures, ["model"])
        self.assertIn(f"{status}:", output)
        self.assertIn(f"{field}={actual}, expected={expected}", output)
        self.assertIn("regressed" if status == "FAIL" else "Improvement", message)
        self.assertIn("reflect the new baseline", message)

    @parametrize("calls_captured,should_fail", ((95, False), (94, True), (101, False)))
    def test_calls_captured_tolerance_is_unchanged(self, calls_captured, should_fail):
        failures, _, _ = self.check(
            actual_results(calls_captured=calls_captured), expected_results()
        )

        self.assertEqual(bool(failures), should_fail)

    def test_new_model_requires_baseline_update(self):
        failures, message, output = self.check(
            actual_results(), expected_results(name="other_model")
        )

        self.assertEqual(failures, ["model"])
        self.assertIn("MISSING", output)
        self.assertIn("Improvement", message)
        self.assertIn("reflect the new baseline", message)

    @parametrize("calls_captured,invalid", ((90, False), (float("nan"), True)))
    def test_flaky_models_only_exempt_valid_metric_changes(
        self, calls_captured, invalid
    ):
        model = "yolov3"
        failures, message, output = self.check(
            actual_results(name=model, calls_captured=calls_captured),
            expected_results(name=model),
        )

        if invalid:
            self.assertEqual(failures, [model])
            self.assertIn("invalid", message)
            self.assertIn("calls_captured", output)
        else:
            self.assertEqual(failures, [])
            self.assertIn("FAIL_BUT_FLAKY", output)

    @parametrize("side", ("actual", "expected"))
    def test_empty_results_are_invalid(self, side):
        actual, expected = actual_results(), expected_results()
        if side == "actual":
            actual = actual.iloc[:0]
        else:
            expected = expected.iloc[:0]

        failures, message, _ = self.check(actual, expected)

        self.assertTrue(failures)
        self.assertIn(side, message)
        self.assertIn("contains no model results", message)
        self.assertNotIn("Improvement", message)
        self.assertNotIn("reflect the new baseline", message)

    @parametrize("side", ("actual", "expected"))
    @parametrize("name", (None, float("nan"), "", "   "))
    def test_missing_model_names_are_invalid(self, side, name):
        actual = actual_results(**({"name": name} if side == "actual" else {}))
        expected = expected_results(**({"name": name} if side == "expected" else {}))

        failures, message, _ = self.check(actual, expected)

        self.assertTrue(failures)
        self.assertIn(side, message)
        self.assertIn("model name", message)
        self.assertNotIn("Improvement", message)

    @parametrize("side", ("actual", "expected"))
    def test_duplicate_results_are_invalid(self, side):
        actual, expected = actual_results(), expected_results()
        if side == "actual":
            actual = pd.concat([actual, actual], ignore_index=True)
        else:
            expected = pd.concat([expected, expected], ignore_index=True)

        failures, message, _ = self.check(actual, expected)

        self.assertEqual(failures, ["model"])
        self.assertIn(side, message)
        self.assertIn("multiple results for: model", message)

    def test_missing_expected_primary_metric_is_invalid(self):
        failures, message, _ = self.check(
            actual_results(), expected_results().drop(columns="graph_breaks")
        )

        self.assertTrue(failures)
        self.assertIn("missing required columns: graph_breaks", message)
        self.assertNotIn("Improvement", message)


class TestCheckGraphBreaksCLI(TestCase):
    @parametrize(
        "case,exit_code,diagnostic",
        (
            ("valid", 0, "PASS"),
            ("numeric_name", 0, "00123"),
            ("zero_capture", 1, "expected Dynamo graph capture"),
            ("missing_name", 1, "model name"),
            ("empty_expected", 1, "expected CSV contains no model results"),
            ("malformed_csv", 2, "error:"),
        ),
    )
    def test_cli_validation(self, case, exit_code, diagnostic):
        with tempfile.TemporaryDirectory() as directory:
            actual_path = Path(directory, "actual.csv")
            expected_path = Path(directory, "expected.csv")
            actual, expected = actual_results(), expected_results()
            if case == "zero_capture":
                actual["unique_graphs"] = 0
            elif case == "numeric_name":
                actual["name"] = "00123"
                expected["name"] = "00123"
            elif case == "missing_name":
                actual["name"] = None
            elif case == "empty_expected":
                expected = expected.iloc[:0]
            actual.to_csv(actual_path, index=False)
            expected.to_csv(expected_path, index=False)
            if case == "malformed_csv":
                actual_path.write_text('name,accuracy\n"unterminated', encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "benchmarks/dynamo/check_graph_breaks.py"),
                    "--actual",
                    str(actual_path),
                    "--expected",
                    str(expected_path),
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )

        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, exit_code, output)
        self.assertIn(diagnostic, output)
        self.assertNotIn("Traceback", output)


instantiate_parametrized_tests(TestCheckGraphBreaks)
instantiate_parametrized_tests(TestCheckGraphBreaksCLI)


if __name__ == "__main__":
    run_tests()
