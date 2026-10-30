# Owner(s): ["module: inductor"]

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_pointwise_coverage import (
    coverage_case,
    CoverageRecord,
    main,
    record_class_fixtures,
    record_result,
    summarize,
    write_event,
)


class TestPointwiseCoverage(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "outcome",
        (
            "matched",
            "known_mismatch",
            "skipped",
            "failed",
            "setup_error",
            "class_skip",
            "expected_failure",
        ),
    )
    def test_runner_outcomes(self, outcome):
        record = CoverageRecord(
            {"test": "example"}, expected_mismatch=outcome == "known_mismatch"
        )

        class Example(unittest.TestCase):
            def setUp(self):
                if outcome == "setup_error":
                    raise RuntimeError("setup failed")

            def runTest(self):
                if outcome == "skipped":
                    self.skipTest("no Triton kernel generated")
                if outcome in ("failed", "expected_failure"):
                    self.fail("numerical mismatch")
                record.compared = 1
                record.generated_kernels = 1
                if outcome == "known_mismatch":
                    record.mismatches["nan-payload"] += 1
                else:
                    record.matched = 1

        if outcome == "class_skip":
            Example = unittest.skip("unsupported hardware")(Example)
        if outcome == "expected_failure":
            Example.runTest = unittest.expectedFailure(Example.runTest)
        result = unittest.TestResult()
        original = result.addSuccess
        test = Example()
        with record_result(record, result, test):
            test.run(result)
        self.assertEqual(result.addSuccess, original)
        expected = {"setup_error": "failed", "class_skip": "skipped"}.get(
            outcome, outcome
        )
        self.assertEqual(record.status, expected)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(
            len(result.failures) + len(result.errors), int(expected == "failed")
        )
        self.assertEqual(len(result.skipped), int(expected == "skipped"))

    def test_unverified_success(self):
        record = CoverageRecord({"test": "example"}, compared=1, matched=1)
        test = unittest.FunctionTestCase(lambda: None)
        result = unittest.TestResult()
        with record_result(record, result, test):
            test.run(result)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(record.status, "unverified")

    def test_failed_subtest(self):
        record = CoverageRecord({"test": "example"})

        class Example(unittest.TestCase):
            def runTest(self):
                with self.subTest("sample"):
                    self.fail("value mismatch")

        test = Example()
        result = unittest.TestResult()
        with record_result(record, result, test):
            test.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(record.status, "failed")

    @parametrize("skip", (False, True))
    @parametrize("fixture", ("setUpClass", "tearDownClass"))
    def test_class_fixture_outcomes(self, skip, fixture):
        class Example(unittest.TestCase):
            def run(self, result=None):
                record = CoverageRecord(
                    coverage_case(self._testMethodName),
                    compared=1,
                    matched=1,
                    generated_kernels=1,
                )
                with record_result(record, result, self):
                    super().run(result)
                write_event(directory, record.event())

            def test_pointwise_bitwise_relu_cuda_float32(self):
                pass

            def test_pointwise_bitwise_log_cuda_float32(self):
                self.fail("unselected method must not run")

        @classmethod
        def fail_fixture(cls):
            if skip:
                raise unittest.SkipTest("unsupported hardware")
            raise RuntimeError("CUDA allocation failed")

        setattr(Example, fixture, fail_fixture)
        method = "test_pointwise_bitwise_relu_cuda_float32"
        with tempfile.TemporaryDirectory() as directory:
            write_event(
                directory,
                {
                    "kind": "manifest",
                    "configuration": {},
                    "cases": [
                        coverage_case(name)
                        for name in unittest.defaultTestLoader.getTestCaseNames(Example)
                    ],
                },
            )
            record_class_fixtures(Example, directory)
            result = unittest.TestResult()
            unittest.TestSuite([Example(method)]).run(result)
            summary = summarize(directory)
            with (
                mock.patch("sys.argv", ["coverage", directory]),
                mock.patch("builtins.print"),
            ):
                self.assertEqual(main(), int(not skip))
        expected = "skipped" if skip else "failed"
        self.assertEqual(summary["class_fixture_counts"], {expected: 1})
        self.assertEqual(
            summary["counts"],
            {"not_run": 1, "matched": 1}
            if fixture == "tearDownClass"
            else {"not_run": 2},
        )
        self.assertEqual(result.wasSuccessful(), skip)
        self.assertEqual(len(result.skipped), int(skip))
        self.assertEqual(len(result.errors), int(not skip))
        self.assertEqual(result.testsRun, int(fixture == "tearDownClass"))
        self.assertEqual(summary["class_fixture_attempts"][0]["fixture"], fixture)

    def test_worker_filenames_across_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            for worker in ("first", "second"):
                with (
                    mock.patch(
                        "torch.testing._internal.inductor_pointwise_coverage._WORKER_ID",
                        worker,
                    ),
                    mock.patch("os.getpid", return_value=1),
                ):
                    write_event(directory, {"kind": "manifest"})
            self.assertEqual(len(list(Path(directory).glob("pointwise-*.log"))), 2)

    def test_worker_merge_and_unexecuted_cases(self):
        methods = [
            f"test_pointwise_bitwise_{op}_cuda_float32"
            for op in ("relu", "float_power", "signbit", "log")
        ]
        cases = [coverage_case(method) for method in methods]
        manifest = {
            "kind": "manifest",
            "configuration": {"all_samples": False},
            "cases": cases,
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch("os.getpid", return_value=1):
                write_event(directory, manifest)
                failed = CoverageRecord(
                    cases[0], status="failed", reason="value mismatch"
                )
                write_event(directory, failed.event())
                known = CoverageRecord(
                    cases[1], expected_mismatch=True, status="known_mismatch"
                )
                known.mismatches["value"] = 1
                known.filters["backward_requires_reduction"] = 2
                write_event(directory, known.event())
            with mock.patch("os.getpid", return_value=2):
                write_event(directory, manifest)
                write_event(
                    directory, CoverageRecord(cases[0], status="matched").event()
                )
                write_event(
                    directory,
                    CoverageRecord(
                        cases[2], status="skipped", reason="no kernel"
                    ).event(),
                )
            summary = summarize(directory)
            self.assertEqual(len(list(Path(directory).glob("pointwise-*.log"))), 2)
        self.assertEqual(
            summary["counts"],
            {"failed": 1, "known_mismatch": 1, "skipped": 1, "not_run": 1},
        )
        rows = {row["op"]: row for row in summary["rows"]}
        self.assertEqual(rows["relu"]["attempts"], 2)
        self.assertEqual(rows["relu"]["attempt_statuses"], ["failed", "matched"])
        self.assertEqual(
            summary["filter_events_across_attempts"], {"backward_requires_reduction": 2}
        )

    def test_mixed_configurations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for all_samples in (False, True):
                write_event(
                    directory,
                    {
                        "kind": "manifest",
                        "configuration": {"all_samples": all_samples},
                        "cases": [],
                    },
                )
            with self.assertRaisesRegex(ValueError, "different configurations"):
                summarize(directory)

    def test_missing_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_event(directory, CoverageRecord({"test": "example"}).event())
            with self.assertRaisesRegex(ValueError, "manifest"):
                summarize(directory)

    def test_case_identity(self):
        case = coverage_case(
            "test_pointwise_edges_backward_nn_functional_silu_complex_cuda_complex128"
        )
        self.assertEqual(case["op"], "nn_functional_silu_complex")
        self.assertEqual(case["dtype"], "complex128")
        self.assertEqual(case["family"], "test_pointwise_edges_backward")
        self.assertIsNone(coverage_case("test_keyword_input_grads_cuda_float16"))


instantiate_parametrized_tests(TestPointwiseCoverage)

if __name__ == "__main__":
    run_tests()
