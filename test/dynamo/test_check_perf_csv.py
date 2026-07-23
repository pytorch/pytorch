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


class CheckPerfCsvTest(TestCase):
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
            check_perf_csv(
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
                check_perf_csv(
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
                    check_perf_csv(path, 1.0, threshold_scale)

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
                check_perf_csv(path, 1.0, 0.99, fail_on_improvement=True)

        self.assertIn("Error: 1 model(s) performance regressed", output.getvalue())
        self.assertIn("    failing_model", output.getvalue())

    def test_osdc_baseline_optional_threshold_scales(self):
        target_file = (
            REPO_ROOT
            / "benchmarks/dynamo/expected_ci_abs_latency_inductor_torchbench_cpu_osdc.csv"
        )
        default_target = None
        override_targets = []
        with open(target_file, newline="") as csv_file:
            for row in csv.reader(csv_file):
                if not row or row[0].startswith("#"):
                    continue
                target = float(row[5])
                if len(row) > 6:
                    override_targets.append((target, float(row[6])))
                elif default_target is None:
                    default_target = target

        self.assertIsNotNone(default_target)
        default_in_band = default_target / 0.99 * 0.999
        output = self._run_check(
            speedup=1.0,
            abs_latency=default_in_band,
            metric="abs_latency",
            threshold=default_target,
        )
        self.assertIn("passed threshold check", output)

        override_target, override_scale = min(
            override_targets, key=lambda item: item[1]
        )
        override_in_band = override_target / override_scale * 0.999
        output = self._run_check(
            speedup=1.0,
            abs_latency=override_in_band,
            metric="abs_latency",
            threshold=override_target,
            threshold_scale=override_scale,
        )
        self.assertIn("passed threshold check", output)

        default_scale_failure = self._run_check_expecting_failure(
            speedup=1.0,
            abs_latency=override_in_band,
            metric="abs_latency",
            threshold=override_target,
        )
        self.assertIn("performance regressed", default_scale_failure)

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
