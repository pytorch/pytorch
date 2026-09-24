# Owner(s): ["module: dynamo"]

import io
import os
import subprocess
import sys
import tempfile
import types
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.dynamo import check_memory_compression_ratio, check_perf_csv, common


sys.path.remove(str(REPO_ROOT))


class TestDynamoBenchmarkArguments(TestCase):
    @parametrize("option", ("--repeat", "-n"))
    @parametrize("repeat", ("0", "-1"))
    def test_repeat_must_be_positive(self, option, repeat) -> None:
        runner = mock.Mock()
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "2") as cm:
            common.main(
                runner,
                args=[
                    "--performance",
                    "--inference",
                    "--device",
                    "cpu",
                    option,
                    repeat,
                ],
            )

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("expected a positive integer", stderr.getvalue())
        runner.load_model.assert_not_called()

    @parametrize("option", ("--repeat", "-n"))
    def test_positive_repeat_is_accepted(self, option) -> None:
        args = common.parse_args(
            ["--performance", "--inference", "--device", "cpu", option, "1"]
        )
        self.assertEqual(args.repeat, 1)


class TestDynamoBenchmarkResultValidation(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

    def write_csv(self, filename, rows, columns=None):
        path = os.path.join(self.temp_dir.name, filename)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        return path

    @staticmethod
    def perf_row(**overrides):
        row = {
            "dev": "cpu",
            "name": "model",
            "batch_size": 1,
            "speedup": 1.2,
            "abs_latency": 10.0,
            "compilation_latency": 0.1,
            "compression_ratio": 1.25,
            "eager_peak_mem": 0.5,
            "dynamo_peak_mem": 0.4,
        }
        row.update(overrides)
        return row

    def run_check(self, checker, *args, exit_code=0):
        output = io.StringIO()
        expected = self.assertRaises(SystemExit) if exit_code else nullcontext()
        with redirect_stdout(output), expected as error:
            checker(*args)
        if exit_code:
            self.assertEqual(error.exception.code, exit_code)
        return output.getvalue()

    def run_perf_check(self, rows, columns=None, *, exit_code=0, scale=1.0):
        path = self.write_csv("perf.csv", rows, columns)
        return self.run_check(
            check_perf_csv.check_perf_csv, path, 1.0, scale, exit_code=exit_code
        )

    def run_memory_check(
        self, actual_rows, expected_rows, actual_columns=None, *, exit_code=0
    ):
        actual = self.write_csv("actual.csv", actual_rows, actual_columns)
        expected = self.write_csv("expected.csv", expected_rows)
        args = types.SimpleNamespace(actual=actual, expected=expected)
        return self.run_check(
            check_memory_compression_ratio.main, args, exit_code=exit_code
        )

    @parametrize(
        "speedup", (None, "bad", float("nan"), float("inf"), -float("inf"), True, False)
    )
    def test_perf_rejects_invalid_required_speedup(self, speedup) -> None:
        output = self.run_perf_check([self.perf_row(speedup=speedup)], exit_code=1)

        self.assertIn("invalid required result field", output)
        self.assertIn("model (speedup=", output)

    def test_perf_rejects_missing_speedup_column(self) -> None:
        row = self.perf_row()
        del row["speedup"]
        self.run_perf_check([row], exit_code=1)

    @parametrize("name", (None, "   "))
    def test_perf_rejects_missing_model_name(self, name) -> None:
        output = self.run_perf_check([self.perf_row(name=name)], exit_code=1)

        self.assertIn("row 2 has a missing model name", output)

    def test_perf_rejects_empty_results(self) -> None:
        self.run_perf_check([], self.perf_row().keys(), exit_code=1)

    def test_perf_rejects_zero_byte_file(self) -> None:
        path = Path(self.temp_dir.name, "empty.csv")
        path.touch()
        output = self.run_check(
            check_perf_csv.check_perf_csv, path, 1.0, 1.0, exit_code=1
        )

        self.assertIn("contains no benchmark results", output)

    @parametrize("name", ("model", "00123"))
    def test_perf_preserves_threshold_check(self, name) -> None:
        output = self.run_perf_check(
            [self.perf_row(name=name, speedup=0.9)], exit_code=1
        )

        self.assertIn("performance regressed", output)
        self.assertIn(f"  - {name}: 0.900x", output)

    @parametrize(
        "speedup,scale,passes",
        (
            (1.0, 1.0, True),
            (0.999, 1.0, False),
            (0.99, 0.99, True),
            (0.989, 0.99, False),
        ),
    )
    def test_perf_threshold_boundary(self, speedup, scale, passes) -> None:
        output = self.run_perf_check(
            [self.perf_row(speedup=speedup)], scale=scale, exit_code=0 if passes else 1
        )

        diagnostic = "passed threshold check" if passes else "performance regressed"
        self.assertIn(diagnostic, output)

    @parametrize(
        "speedup,diagnostic",
        ((float("nan"), "invalid required result"), (0.9, "performance regressed")),
    )
    def test_perf_checks_later_rows(self, speedup, diagnostic) -> None:
        output = self.run_perf_check(
            [self.perf_row(name="valid"), self.perf_row(name="later", speedup=speedup)],
            exit_code=1,
        )

        self.assertIn("later", output)
        self.assertIn(diagnostic, output)

    @parametrize("value", (None, "bad", float("inf"), True))
    def test_perf_accepts_unavailable_optional_metrics(self, value) -> None:
        output = self.run_perf_check(
            [
                self.perf_row(
                    abs_latency=value,
                    compilation_latency=value,
                    compression_ratio=value,
                    eager_peak_mem=value,
                    dynamo_peak_mem=value,
                )
            ]
        )

        self.assertNotIn("latency=", output)
        self.assertNotIn("compile=", output)
        self.assertNotIn("mem_ratio", output)
        self.assertIn("All 1 model(s) passed", output)

    def test_perf_accepts_absent_optional_columns(self) -> None:
        output = self.run_perf_check([{"name": "model", "speedup": 1.2}])

        self.assertNotIn("latency=", output)
        self.assertNotIn("compile=", output)
        self.assertNotIn("mem_ratio", output)
        self.assertIn("All 1 model(s) passed", output)

    @parametrize("compilation_latency", (0.0, -0.01))
    def test_perf_accepts_optional_compilation_latency(
        self, compilation_latency
    ) -> None:
        output = self.run_perf_check(
            [self.perf_row(compilation_latency=compilation_latency)]
        )

        self.assertIn("All 1 model(s) passed", output)
        self.assertIn(f"compile={compilation_latency:.3f}s", output)

    def test_perf_treats_zero_memory_ratio_as_unavailable(self) -> None:
        output = self.run_perf_check(
            [
                self.perf_row(
                    compression_ratio=0.0,
                    eager_peak_mem=0.0,
                    dynamo_peak_mem=0.0,
                )
            ]
        )

        self.assertNotIn("mem_ratio", output)
        self.assertIn("All 1 model(s) passed", output)

    @parametrize("unselected_ratio", (2.0, None))
    def test_memory_accepts_selected_subset(self, unselected_ratio) -> None:
        output = self.run_memory_check(
            [{"name": "model", "compression_ratio": 1.2}],
            [
                {"name": "model", "compression_ratio": 1.0},
                {"name": "unselected", "compression_ratio": unselected_ratio},
            ],
        )

        self.assertIn("PASS", output)

    @parametrize(
        "compression_ratio",
        (
            None,
            "bad",
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            True,
            False,
        ),
    )
    def test_memory_rejects_invalid_required_ratio(self, compression_ratio) -> None:
        output = self.run_memory_check(
            [{"name": "model", "compression_ratio": compression_ratio}],
            [{"name": "model", "compression_ratio": 1.0}],
            exit_code=1,
        )

        self.assertIn("model has invalid compression_ratio=", output)
        self.assertIn("actual.csv", output)
        self.assertIn("finite value greater than zero", output)

    @parametrize("name", (None, "   "))
    def test_memory_rejects_missing_model_name(self, name) -> None:
        output = self.run_memory_check(
            [{"name": name, "compression_ratio": 1.0}],
            [{"name": "model", "compression_ratio": 1.0}],
            exit_code=1,
        )

        self.assertIn("row 2", output)
        self.assertIn("missing model name", output)

    def test_memory_rejects_empty_results(self) -> None:
        self.run_memory_check(
            [],
            [{"name": "model", "compression_ratio": 1.0}],
            ["name", "compression_ratio"],
            exit_code=1,
        )

    def test_memory_rejects_zero_byte_file(self) -> None:
        actual = Path(self.temp_dir.name, "actual.csv")
        actual.touch()
        expected = self.write_csv(
            "expected.csv", [{"name": "model", "compression_ratio": 1.0}]
        )
        args = types.SimpleNamespace(actual=actual, expected=expected)
        output = self.run_check(check_memory_compression_ratio.main, args, exit_code=1)

        self.assertIn("contains no benchmark results", output)

    def test_memory_rejects_missing_ratio_column(self) -> None:
        self.run_memory_check(
            [{"name": "model"}],
            [{"name": "model", "compression_ratio": 1.0}],
            exit_code=1,
        )

    def test_memory_rejects_duplicate_expected_models(self) -> None:
        self.run_memory_check(
            [{"name": "model", "compression_ratio": 1.0}],
            [
                {"name": "model", "compression_ratio": 1.0},
                {"name": "model", "compression_ratio": 1.0},
            ],
            exit_code=1,
        )

    @parametrize("compression_ratio", (float("nan"), True))
    def test_memory_rejects_invalid_selected_expected_ratio(
        self, compression_ratio
    ) -> None:
        output = self.run_memory_check(
            [{"name": "model", "compression_ratio": 1.0}],
            [{"name": "model", "compression_ratio": compression_ratio}],
            exit_code=1,
        )

        self.assertIn("model has invalid compression_ratio=", output)
        self.assertIn("expected.csv", output)

    def test_memory_rejects_missing_expected_model(self) -> None:
        output = self.run_memory_check(
            [{"name": "model", "compression_ratio": 1.0}],
            [{"name": "unselected", "compression_ratio": 2.0}],
            exit_code=1,
        )

        self.assertIn("model is missing from", output)
        self.assertIn("expected.csv", output)

    @parametrize("name", ("model", "00123"))
    def test_memory_preserves_threshold_check(self, name) -> None:
        output = self.run_memory_check(
            [{"name": name, "compression_ratio": 0.9}],
            [{"name": name, "compression_ratio": 1.0}],
            exit_code=1,
        )

        self.assertIn(f"{name:34}:", output)
        self.assertIn("below expected memory compression ratio", output)

    @parametrize("ratio,passes", ((0.95, True), (0.949, False)))
    def test_memory_threshold_boundary(self, ratio, passes) -> None:
        output = self.run_memory_check(
            [{"name": "model", "compression_ratio": ratio}],
            [{"name": "model", "compression_ratio": 1.0}],
            exit_code=0 if passes else 1,
        )

        diagnostic = "PASS" if passes else "below expected memory compression ratio"
        self.assertIn(diagnostic, output)

    @parametrize(
        "ratio,diagnostic",
        (
            (float("nan"), "invalid memory compression"),
            (0.9, "below expected memory compression ratio"),
        ),
    )
    def test_memory_checks_later_rows(self, ratio, diagnostic) -> None:
        output = self.run_memory_check(
            [
                self.perf_row(name="valid"),
                self.perf_row(name="later", compression_ratio=ratio),
            ],
            [{"name": name, "compression_ratio": 1.0} for name in ("valid", "later")],
            exit_code=1,
        )

        self.assertIn("later", output)
        self.assertIn(diagnostic, output)

    @parametrize("checker", ("check_perf_csv", "check_memory_compression_ratio"))
    @parametrize("case,exit_code", (("valid", 0), ("invalid", 1), ("empty", 1)))
    def test_checker_cli(self, checker, case, exit_code) -> None:
        row = self.perf_row()
        performance = checker == "check_perf_csv"
        if case == "invalid":
            row["speedup" if performance else "compression_ratio"] = float("nan")
        actual = self.write_csv(
            "actual.csv", [] if case == "empty" else [row], row.keys()
        )
        command = [sys.executable, str(REPO_ROOT / f"benchmarks/dynamo/{checker}.py")]
        if performance:
            command += ["-f", actual, "-t", "1.0", "-s", "0.99"]
        else:
            expected = self.write_csv(
                "expected.csv",
                [
                    {"name": "model", "compression_ratio": 1.0},
                    {"name": "unselected", "compression_ratio": None},
                ],
            )
            command += ["--actual", actual, "--expected", expected]

        result = subprocess.run(
            command,
            cwd=self.temp_dir.name,
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, exit_code, output)
        self.assertNotIn("Traceback", output)
        if case == "empty":
            self.assertIn("contains no benchmark results", output)
        elif case == "invalid":
            self.assertIn("invalid", output)
            self.assertIn("speedup" if performance else "compression_ratio", output)
        else:
            self.assertIn("passed threshold check" if performance else "PASS", output)


instantiate_parametrized_tests(TestDynamoBenchmarkArguments)
instantiate_parametrized_tests(TestDynamoBenchmarkResultValidation)


if __name__ == "__main__":
    run_tests()
