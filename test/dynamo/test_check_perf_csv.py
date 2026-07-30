# Owner(s): ["module: dynamo"]

import contextlib
import csv
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


CSV_COLUMNS = [
    "dev",
    "name",
    "batch_size",
    "speedup",
    "abs_latency",
    "compilation_latency",
    "compression_ratio",
    "eager_peak_mem",
    "dynamo_peak_mem",
]


@contextlib.contextmanager
def _perf_csv(speedup=1.0, abs_latency=10.0, columns=CSV_COLUMNS, rows=None):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            default_row = {
                "dev": "cpu",
                "name": "test_model",
                "batch_size": 1,
                "speedup": speedup,
                "abs_latency": abs_latency,
                "compilation_latency": 0.1,
                "compression_ratio": 1.0,
                "eager_peak_mem": 1.0,
                "dynamo_peak_mem": 1.0,
            }
            for overrides in rows if rows is not None else [{}]:
                row = {**default_row, **overrides}
                writer.writerow({column: row[column] for column in columns})
        yield path
    finally:
        os.remove(path)


@unittest.skipUnless(
    importlib.util.find_spec("pandas") is not None, "pandas is not installed"
)
class CheckPerfCsvTest(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from benchmarks.dynamo.check_perf_csv import check_perf_csv

        cls.check_perf_csv = staticmethod(check_perf_csv)

    def _run_check(
        self,
        *,
        speedup,
        abs_latency=10.0,
        metric="speedup",
        threshold=None,
        threshold_scale=0.99,
        fail_on_improvement=True,
    ):
        if threshold is None:
            threshold = 1.0 if metric == "speedup" else 10.0

        output = io.StringIO()
        with (
            _perf_csv(speedup, abs_latency) as path,
            contextlib.redirect_stdout(output),
        ):
            self.check_perf_csv(
                path,
                threshold,
                threshold_scale,
                metric=metric,
                fail_on_improvement=fail_on_improvement,
            )
        return output.getvalue()

    def _run_check_expecting_failure(self, **kwargs):
        output = io.StringIO()
        with self.assertRaisesRegex(SystemExit, "^1$"):
            with (
                _perf_csv(
                    kwargs.pop("speedup"), kwargs.pop("abs_latency", 10.0)
                ) as path,
                contextlib.redirect_stdout(output),
            ):
                self.check_perf_csv(
                    path,
                    kwargs.pop("threshold", 1.0),
                    kwargs.pop("threshold_scale", 0.99),
                    metric=kwargs.pop("metric", "speedup"),
                    fail_on_improvement=kwargs.pop("fail_on_improvement", True),
                )
        self.assertFalse(kwargs)
        return output.getvalue()

    def test_default_speedup_check_allows_large_improvement(self):
        output = self._run_check(speedup=2.0, fail_on_improvement=False)
        self.assertIn("passed threshold check", output)

    def test_two_sided_speedup_check_fails_regression(self):
        output = self._run_check_expecting_failure(speedup=0.98)
        self.assertIn("performance regressed", output)

    def test_two_sided_speedup_check_passes_in_band(self):
        output = self._run_check(speedup=1.005)
        self.assertIn("passed threshold check", output)

    def test_two_sided_speedup_check_fails_large_improvement(self):
        output = self._run_check_expecting_failure(speedup=1.02)
        self.assertIn("performance improved", output)

    def test_two_sided_latency_check_uses_lower_is_better_direction(self):
        regression = self._run_check_expecting_failure(
            speedup=1.0, abs_latency=10.2, metric="abs_latency", threshold=10.0
        )
        improvement = self._run_check_expecting_failure(
            speedup=1.0, abs_latency=9.8, metric="abs_latency", threshold=10.0
        )
        self.assertIn("performance regressed", regression)
        self.assertIn("performance improved", improvement)
        self.assertIn("abs_latency=9.8 ms/iter", improvement)
        self.assertIn("-1.0% from bound", improvement)

    def test_two_sided_latency_check_passes_in_band(self):
        output = self._run_check(
            speedup=1.0,
            abs_latency=10.05,
            metric="abs_latency",
            threshold=10.0,
        )
        self.assertIn("passed threshold check", output)

    def test_threshold_scale_greater_than_one_normalizes_bounds(self):
        output = self._run_check(speedup=1.0, threshold_scale=1.01)
        self.assertIn("0.990x <= speedup <= 1.010x", output)

    def test_threshold_scale_must_be_positive(self):
        with _perf_csv() as path:
            for threshold_scale in (0.0, -0.01):
                with (
                    self.subTest(threshold_scale=threshold_scale),
                    self.assertRaisesRegex(
                        ValueError, "threshold_scale must be positive"
                    ),
                ):
                    self.check_perf_csv(path, 1.0, threshold_scale)

    def test_multi_row_check_reports_only_out_of_band_count(self):
        output = io.StringIO()
        with self.assertRaisesRegex(SystemExit, "^1$"):
            with (
                _perf_csv(
                    rows=[
                        {"name": "passing_model", "speedup": 1.0},
                        {"name": "failing_model", "speedup": 0.98},
                    ]
                ) as path,
                contextlib.redirect_stdout(output),
            ):
                self.check_perf_csv(path, 1.0, 0.99, fail_on_improvement=True)

        self.assertIn("Error: 1 model(s) performance regressed", output.getvalue())
        self.assertIn("    failing_model", output.getvalue())

    def test_explicit_threshold_scale_widens_latency_band(self):
        target = 10.0
        only_in_wider_band = 10.4
        output = self._run_check(
            speedup=1.0,
            abs_latency=only_in_wider_band,
            metric="abs_latency",
            threshold=target,
            threshold_scale=0.95,
        )
        self.assertIn("passed threshold check", output)

        default_scale_failure = self._run_check_expecting_failure(
            speedup=1.0,
            abs_latency=only_in_wider_band,
            metric="abs_latency",
            threshold=target,
        )
        self.assertIn("performance regressed", default_scale_failure)

    def test_submillisecond_latency_failure_keeps_precision(self):
        output = self._run_check_expecting_failure(
            speedup=1.0,
            abs_latency=0.29,
            metric="abs_latency",
            threshold=0.297512,
            threshold_scale=0.985,
        )
        self.assertIn("abs_latency=0.29 ms/iter", output)
        self.assertIn("< 0.293049 ms/iter", output)
        self.assertIn("-1.0% from bound", output)

    def test_latency_summary_without_speedup_column_has_no_leading_comma(self):
        output = io.StringIO()
        with (
            _perf_csv(
                speedup=1.0,
                abs_latency=10.0,
                columns=[column for column in CSV_COLUMNS if column != "speedup"],
            ) as path,
            contextlib.redirect_stdout(output),
        ):
            self.check_perf_csv(
                path,
                10.0,
                0.99,
                metric="abs_latency",
                fail_on_improvement=True,
            )

        self.assertIn(
            "test_model                         latency=10 ms/iter", output.getvalue()
        )
        self.assertNotIn(
            "test_model                         , latency", output.getvalue()
        )


if __name__ == "__main__":
    run_tests()
