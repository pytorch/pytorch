# Owner(s): ["module: ci"]

import io
import json
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import compare_td_tracer
import pr_tests

from tools.testing import select_tests_from_td_tracer

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def make_pr(number, commit, *changed_files):
    return compare_td_tracer.LandedPR(
        number=number,
        landing_commit=commit,
        committed_at="2026-07-27T12:00:00+00:00",
        subject=f"PR {number}",
        changed_files=changed_files,
    )


def make_selection(matches, affected, matched, *, usable=True, revisions=("trace",)):
    metadata = select_tests_from_td_tracer.TraceMetadata(
        schema_version=4,
        run_id="run",
        complete=usable,
        successful=usable,
        usable=usable,
        revisions=revisions,
    )
    return select_tests_from_td_tracer.SelectionResult(
        tests=sorted(matches),
        matches_by_test=matches,
        affected=frozenset(affected),
        matched=frozenset(matched),
        metadata=metadata,
    )


def make_success_record(pr, *, fully_covered, metrics):
    return {
        "pr_number": pr.number,
        "url": pr.url,
        "landing_commit": pr.landing_commit,
        "committed_at": pr.committed_at,
        "subject": pr.subject,
        "changed_files": list(pr.changed_files),
        "changed_file_count": len(pr.changed_files),
        "unsupported_files": [],
        "unmatched_files": [],
        "fully_covered_changes": fully_covered,
        "status": "success",
        "error": None,
        "collection": {"workflow_runs": 1, "artifacts": 2, "reports": 3},
        "metrics": metrics,
    }


class TestCompareTDTracer(TestCase):
    def test_enumerate_landed_prs_skips_bot_reverts_and_deduplicates_relands(self):
        trailer = "Pull Request resolved: https://github.com/pytorch/pytorch/pull/{}"
        entries = [
            ("new-10", "date", "Feature (#10)", trailer.format(10)),
            (
                "bot-revert",
                "date",
                'Revert "Feature (#99)"',
                "This reverts commit old-99.",
            ),
            ("old-10", "date", "Earlier land (#10)", trailer.format(10)),
            (
                "wrong-repo",
                "date",
                "Not a PyTorch trailer (#40)",
                "Pull Request resolved: https://github.com/example/repo/pull/40",
            ),
            (
                "landed-revert-20",
                "date",
                "Revert behavior through a PR (#20)",
                trailer.format(20),
            ),
            ("new-30", "date", "Feature (#30)", trailer.format(30)),
        ]

        with (
            mock.patch.object(
                compare_td_tracer, "_resolve_commit", return_value="main-tip"
            ),
            mock.patch.object(compare_td_tracer, "_history", return_value=entries),
            mock.patch.object(
                select_tests_from_td_tracer,
                "changed_files_for_commit",
                side_effect=lambda commit, _root: [f"torch/{commit}.py"],
            ) as changed_files,
        ):
            tip, prs = compare_td_tracer.enumerate_landed_prs(
                "origin/main", 3, Path("repo")
            )

        self.assertEqual(tip, "main-tip")
        self.assertEqual([pr.number for pr in prs], [10, 20, 30])
        self.assertEqual(
            [pr.landing_commit for pr in prs],
            ["new-10", "landed-revert-20", "new-30"],
        )
        self.assertEqual(
            [call.args for call in changed_files.call_args_list],
            [
                ("new-10", Path("repo")),
                ("landed-revert-20", Path("repo")),
                ("new-30", Path("repo")),
            ],
        )

    @parametrize(
        "test,expected",
        [
            (
                "test/test_cpp_extensions_aot_ninja.py::TestAOT::test_build",
                "test/test_cpp_extensions_aot.py::TestAOT::test_build",
            ),
            (
                "test_cpp_extensions_aot_no_ninja.py",
                "test_cpp_extensions_aot.py",
            ),
            ("test/test_tensor.py::TestTensor::test_add", None),
        ],
    )
    def test_canonical_test_id(self, test, expected):
        self.assertEqual(compare_td_tracer.canonical_test_id(test), expected or test)

    def test_prediction_index_groups_paths_and_separates_non_junit_targets(self):
        prs = [
            make_pr(1, "one", "torch/a.py", "setup.py", "torch/missing.py"),
            make_pr(2, "two", "torch/b.py"),
        ]
        alias = "test/test_cpp_extensions_aot.py::TestAOT::test_build"
        other = "test/test_other.py::TestOther::test_case"
        non_junit = "tools/testing/check.py::test_cli"
        matches = {
            "test/test_cpp_extensions_aot_ninja.py::TestAOT::test_build": [
                "torch/a.py"
            ],
            "test/test_cpp_extensions_aot_no_ninja.py::TestAOT::test_build": [
                "torch/b.py"
            ],
            other: ["torch/a.py", "torch/b.py"],
            non_junit: ["torch/a.py"],
            "test/test_unrelated.py::test_case": ["torch/unrelated.py"],
        }
        selection = make_selection(
            matches,
            {"torch/a.py", "torch/b.py", "setup.py", "torch/missing.py"},
            {"torch/a.py", "torch/b.py"},
        )

        predictions = compare_td_tracer.build_prediction_index(prs, selection)

        self.assertEqual(predictions.comparable_masks, {alias: 3, other: 3})
        self.assertEqual(predictions.non_junit_masks, {non_junit: 1})
        self.assertEqual(predictions.comparable_counts, (2, 2))
        self.assertEqual(predictions.non_junit_counts, (1, 0))
        self.assertEqual(predictions.unsupported, (("setup.py",), ()))
        self.assertEqual(predictions.unmatched, (("torch/missing.py",), ()))

        metrics = compare_td_tracer.comparison_metrics(
            {alias, other, "test/test_actual_only.py::test_case"}, 0, predictions
        )
        self.assertEqual(
            metrics,
            {
                "actual_unique": 3,
                "selected_unique": 3,
                "selected_comparable_unique": 2,
                "selected_non_junit_unique": 1,
                "intersection_unique": 2,
                "actual_only_unique": 1,
                "selected_only_unique": 1,
                "selected_only_comparable_unique": 0,
                "tests_reduced": 1,
                "reduction_percent": 100 / 3,
            },
        )
        zero_actual = compare_td_tracer.comparison_metrics([], 1, predictions)
        self.assertEqual(zero_actual["tests_reduced"], -2)
        self.assertIsNone(zero_actual["reduction_percent"])

    def test_aggregate_results_reports_raw_and_fully_covered_cohorts(self):
        full_metrics = {
            "actual_unique": 10,
            "selected_unique": 4,
            "selected_comparable_unique": 3,
            "selected_non_junit_unique": 1,
            "intersection_unique": 3,
            "actual_only_unique": 7,
            "selected_only_unique": 1,
            "selected_only_comparable_unique": 0,
            "tests_reduced": 7,
            "reduction_percent": 70.0,
        }
        partial_metrics = {
            "actual_unique": 5,
            "selected_unique": 6,
            "selected_comparable_unique": 5,
            "selected_non_junit_unique": 1,
            "intersection_unique": 2,
            "actual_only_unique": 3,
            "selected_only_unique": 4,
            "selected_only_comparable_unique": 3,
            "tests_reduced": 0,
            "reduction_percent": 0.0,
        }
        full = make_success_record(
            make_pr(1, "one", "torch/a.py"),
            fully_covered=True,
            metrics=full_metrics,
        )
        partial = make_success_record(
            make_pr(2, "two", "setup.py"),
            fully_covered=False,
            metrics=partial_metrics,
        )
        error = {
            "pr_number": 3,
            "landing_commit": "three",
            "status": "error",
            "error": "expired artifacts",
            "fully_covered_changes": True,
        }

        summary = compare_td_tracer.aggregate_results([full, partial, error])

        self.assertEqual(summary["requested_prs"], 3)
        self.assertEqual(summary["completed_prs"], 2)
        self.assertEqual(summary["error_prs"], 1)
        self.assertEqual(summary["scope_incomplete_prs"], 1)
        self.assertEqual(summary["raw"]["pr_count"], 2)
        self.assertEqual(summary["raw"]["actual_unique"], 15)
        self.assertEqual(summary["raw"]["selected_unique"], 10)
        self.assertEqual(summary["raw"]["tests_reduced"], 7)
        self.assertAlmostEqual(summary["raw"]["reduction_percent"], 140 / 3)
        self.assertEqual(summary["fully_covered"]["pr_count"], 1)
        self.assertEqual(summary["fully_covered"]["actual_unique"], 10)
        self.assertEqual(summary["fully_covered"]["reduction_percent"], 70.0)

    @parametrize(
        "usable,revisions,warning_count",
        [
            (True, ("revision",), 1),
            (False, ("old", "new"), 3),
            (True, (), 2),
        ],
    )
    def test_configuration_marks_historical_comparisons_provisional(
        self, usable, revisions, warning_count
    ):
        metadata = select_tests_from_td_tracer.TraceMetadata(
            schema_version=4,
            run_id="run",
            complete=usable,
            successful=usable,
            usable=usable,
            revisions=revisions,
        )

        configuration = compare_td_tracer._configuration(
            Path("trace.json"), "digest", metadata, "origin/main", "tip", 100
        )

        self.assertTrue(configuration["provisional"])
        self.assertEqual(len(configuration["warnings"]), warning_count)
        self.assertEqual(configuration["trace"]["revisions"], list(revisions))
        self.assertEqual(
            configuration["collector_scope"],
            {"mode": "all-completed-actions-runs-for-pr-head"},
        )

    def test_configuration_records_filtered_collector_scope(self):
        metadata = select_tests_from_td_tracer.TraceMetadata(
            schema_version=4,
            run_id="run",
            complete=True,
            successful=True,
            usable=True,
            revisions=("revision",),
        )
        job_filter = pr_tests.TestJobFilter(
            build_environment="linux-jammy-py3.14t-clang*",
            test_configs=frozenset({"slow", "default"}),
            workflow_names=frozenset({"pull"}),
            events=frozenset({"pull_request"}),
        )

        configuration = compare_td_tracer._configuration(
            Path("trace.json"),
            "digest",
            metadata,
            "origin/main",
            "tip",
            100,
            job_filter,
        )

        self.assertEqual(
            configuration["collector_scope"],
            {
                "mode": "filtered-test-jobs-for-pr-head",
                "build_environment": "linux-jammy-py3.14t-clang*",
                "test_configs": ["default", "slow"],
                "workflow_names": ["pull"],
                "events": ["pull_request"],
            },
        )

    def test_compare_pr_isolates_collection_errors(self):
        pr = make_pr(12, "commit", "setup.py")
        job_filter = pr_tests.TestJobFilter("linux-jammy-py3.14t-clang*")
        predictions = compare_td_tracer.PredictionIndex(
            comparable_masks={},
            non_junit_masks={},
            comparable_counts=(0,),
            non_junit_counts=(0,),
            unsupported=(("setup.py",),),
            unmatched=((),),
        )
        with mock.patch.object(
            pr_tests,
            "collect_pr_tests",
            side_effect=pr_tests.PRTestsError("artifacts expired"),
        ) as collect:
            record = compare_td_tracer.compare_pr(pr, 0, predictions, job_filter)

        collect.assert_called_once_with("12", job_filter=job_filter)
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error"], "artifacts expired")
        self.assertFalse(record["fully_covered_changes"])
        self.assertEqual(record["unsupported_files"], ["setup.py"])
        self.assertNotIn("metrics", record)

    def test_run_comparison_continues_after_collection_error_and_returns_one(self):
        prs = [make_pr(1, "one", "torch/a.py"), make_pr(2, "two", "torch/b.py")]
        selection = make_selection(
            {
                "test/test_a.py::test_a": ["torch/a.py"],
                "test/test_b.py::test_b": ["torch/b.py"],
            },
            {"torch/a.py", "torch/b.py"},
            {"torch/a.py", "torch/b.py"},
        )
        collected = pr_tests.CollectionResult(
            tests=frozenset({"test/test_a.py::test_a"}),
            workflow_runs=1,
            artifacts=2,
            reports=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            output = Path(directory) / "report.json"
            with (
                mock.patch.object(
                    compare_td_tracer,
                    "enumerate_landed_prs",
                    return_value=("tip", prs),
                ),
                mock.patch.object(compare_td_tracer, "_sha256", return_value="hash"),
                mock.patch.object(
                    select_tests_from_td_tracer,
                    "select_tests_with_details",
                    return_value=selection,
                ) as select_tests,
                mock.patch.object(
                    pr_tests,
                    "collect_pr_tests",
                    side_effect=[collected, pr_tests.PRTestsError("missing reports")],
                ),
            ):
                report, return_code = compare_td_tracer.run_comparison(
                    trace,
                    "origin/main",
                    2,
                    output,
                    None,
                    workers=1,
                    resume=False,
                )

            select_tests.assert_called_once_with(trace, ["torch/a.py", "torch/b.py"])
            self.assertEqual(return_code, 1)
            self.assertEqual(
                [record["status"] for record in report["results"]],
                ["success", "error"],
            )
            self.assertEqual(report["summary"]["completed_prs"], 1)
            self.assertEqual(report["summary"]["error_prs"], 1)
            self.assertEqual(report["summary"]["requested_prs"], 2)
            self.assertTrue(report["complete"])
            with output.open(encoding="utf-8") as input_file:
                self.assertEqual(json.load(input_file), report)

    def test_run_comparison_refuses_to_overwrite_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            with self.assertRaisesRegex(
                compare_td_tracer.ComparisonError, "must not overwrite"
            ):
                compare_td_tracer.run_comparison(
                    trace,
                    "origin/main",
                    1,
                    trace,
                    None,
                    workers=1,
                    resume=False,
                )

    def test_atomic_json_output_is_deterministic(self):
        first = {"z": 1, "nested": {"b": 2, "a": 1}, "a": [2, 1]}
        second = {"a": [2, 1], "nested": {"a": 1, "b": 2}, "z": 1}
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            compare_td_tracer._atomic_write_json(first_path, first)
            compare_td_tracer._atomic_write_json(second_path, second)
            first_bytes = first_path.read_bytes()
            second_bytes = second_path.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertTrue(first_bytes.endswith(b"\n"))

    def test_partial_report_retains_requested_count(self):
        report = compare_td_tracer._report({}, [], requested_prs=100, complete=False)

        self.assertFalse(report["complete"])
        self.assertEqual(report["summary"]["requested_prs"], 100)
        self.assertEqual(report["summary"]["completed_prs"], 0)

    def test_checkpoint_reuses_only_compatible_successful_records(self):
        prs = [make_pr(1, "one"), make_pr(2, "two")]
        metrics = {
            "actual_unique": 1,
            "selected_unique": 1,
            "selected_comparable_unique": 1,
            "selected_non_junit_unique": 0,
            "intersection_unique": 1,
            "actual_only_unique": 0,
            "selected_only_unique": 0,
            "selected_only_comparable_unique": 0,
            "tests_reduced": 0,
            "reduction_percent": 0.0,
        }
        success = make_success_record(prs[0], fully_covered=True, metrics=metrics)
        failed = {
            "pr_number": 2,
            "landing_commit": "two",
            "status": "error",
            "error": "retry me",
        }
        unrelated = make_success_record(
            make_pr(3, "three"), fully_covered=True, metrics=metrics
        )
        configuration = {"trace": "configuration"}
        checkpoint = {
            "schema_version": compare_td_tracer.REPORT_SCHEMA_VERSION,
            "configuration": configuration,
            "summary": {},
            "results": [success, failed, unrelated],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            compare_td_tracer._atomic_write_json(path, checkpoint)
            reusable = compare_td_tracer._load_reusable_records(
                path, configuration, prs, resume=True
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                incompatible = compare_td_tracer._load_reusable_records(
                    path, {"trace": "different"}, prs, resume=True
                )

        self.assertEqual(reusable, {(1, "one"): success})
        self.assertEqual(incompatible, {})
        self.assertIn("checkpoint", stderr.getvalue())
        self.assertIn("incompatible", stderr.getvalue())

    def test_job_filter_from_options(self):
        options = compare_td_tracer._parser().parse_args(
            [
                "trace.json",
                "--output",
                "report.json",
                "--build-environment",
                "linux-jammy-py3.14t-clang*",
                "--test-config",
                "default",
                "--test-config",
                "slow",
                "--workflow-name",
                "pull",
                "--event",
                "pull_request",
            ]
        )

        self.assertEqual(
            compare_td_tracer._job_filter_from_options(options),
            pr_tests.TestJobFilter(
                build_environment="linux-jammy-py3.14t-clang*",
                test_configs=frozenset({"default", "slow"}),
                workflow_names=frozenset({"pull"}),
                events=frozenset({"pull_request"}),
            ),
        )


instantiate_parametrized_tests(TestCompareTDTracer)


if __name__ == "__main__":
    run_tests()
