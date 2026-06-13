# Owner(s): ["module: dynamo"]

import contextlib
import csv
import io
import os
import sys
import tempfile
from pathlib import Path

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dynamo.check_perf_csv import check_perf_csv


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
def _perf_csv(speedup, abs_latency=10.0, columns=CSV_COLUMNS):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        with open(path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            row = {
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
            writer.writerow({column: row[column] for column in columns})
        yield path
    finally:
        os.remove(path)


class CheckPerfCsvTest(TestCase):
    def _run_check(
        self,
        *,
        speedup,
        abs_latency=10.0,
        metric="speedup",
        threshold=None,
        fail_on_improvement=True,
    ):
        if threshold is None:
            threshold = 1.0 if metric == "speedup" else 10.0

        output = io.StringIO()
        with (
            _perf_csv(speedup, abs_latency) as path,
            contextlib.redirect_stdout(output),
        ):
            check_perf_csv(
                path,
                threshold,
                0.99,
                metric=metric,
                fail_on_improvement=fail_on_improvement,
            )
        return output.getvalue()

    def _run_check_expecting_failure(self, **kwargs):
        output = io.StringIO()
        with self.assertRaises(SystemExit) as cm:
            with (
                _perf_csv(
                    kwargs.pop("speedup"), kwargs.pop("abs_latency", 10.0)
                ) as path,
                contextlib.redirect_stdout(output),
            ):
                check_perf_csv(
                    path,
                    kwargs.pop("threshold", 1.0),
                    0.99,
                    metric=kwargs.pop("metric", "speedup"),
                    fail_on_improvement=kwargs.pop("fail_on_improvement", True),
                )
        self.assertEqual(cm.exception.code, 1)
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
            check_perf_csv(
                path,
                10.0,
                0.99,
                metric="abs_latency",
                fail_on_improvement=True,
            )

        self.assertIn(
            "test_model                         latency=10.0 ms/iter", output.getvalue()
        )
        self.assertNotIn(
            "test_model                         , latency", output.getvalue()
        )


if __name__ == "__main__":
    run_tests()
