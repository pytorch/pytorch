# Owner(s): ["module: ci"]

import functools
import importlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
import time
import unittest.mock
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_SANDCASTLE,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TemporaryFileName,
    TEST_CUDA,
    TEST_WITH_CROSSREF,
    TEST_WITH_ROCM,
    TestCase,
    TestEnvironment,
)
from torch.testing._internal.torchci import report as torchci_report


_TEST_DIR = Path(__file__).resolve().parent.parent
# For importing run_test.
sys.path.append(str(_TEST_DIR))
# test/test_testing.py's TestJunitXml fixture suite, which covers every outcome the
# pytest path reports.
_PYTEST_SUITE = _TEST_DIR / "junit_xml_testdata/pytest_suite.py"
# The report tests' fixtures run from here, below test/ so pytest loads
# test/conftest.py as in CI; their reports and caches go to temporary directories.
_REPORT_TESTDATA = Path(__file__).resolve().parent / "testdata"
# Run the fixture as the default config: drop the job's CI / PYTORCH_TEST_* /
# PYTEST_ADDOPTS settings and a PYTHONPATH whose repo root would shadow the
# installed torch, and don't write bytecode into test/.
_CHILD_ENV = {
    k: v
    for k, v in os.environ.items()
    if k not in ("PYTHONPATH", "CI", "PYTEST_ADDOPTS")
    and not k.startswith("PYTORCH_TEST_")
} | {"PYTHONDONTWRITEBYTECODE": "1"}

# The fixture's run context, which the report records, without the job's device
# filter (XPU and CUDA jobs set PYTORCH_TESTING_DEVICE_ONLY_FOR).
_REPORT_CHILD_ENV = {
    k: v for k, v in _CHILD_ENV.items() if not k.startswith("PYTORCH_TESTING_DEVICE_")
} | {
    "GITHUB_REPOSITORY": "pytorch/pytorch",
    "JOB_ID": "123456789",
    "BUILD_ENVIRONMENT": "report-build",
    "TEST_CONFIG": "report-config",
    "RUNNER_NAME": "report-runner",
}


# Each run line's keys, in order, and their types.
_RUN_TYPES = {
    "type": str,
    "schema_version": str,
    "file": str,
    "suite": str,
    "case_name": str,
    "language": str,
    "declared_case_name": str,
    "rerun_number": int,
    "outcome": str,
    "outcome_summary": str,
    "started_at": int,
    "ended_at": int,
    "properties": dict,
}
_RUN_OUTCOMES = {
    "passed",
    "failed",
    "error",
    "skipped",
    "xfailed",
    "xpassed",
    "crashed",
    "timed_out",
}


def _assert_run_line(
    test: TestCase, run: dict[str, Any], t0_ms: int, t1_ms: int
) -> None:
    test.assertEqual([(k, type(v)) for k, v in run.items()], list(_RUN_TYPES.items()))
    test.assertEqual(
        (run["type"], run["schema_version"], run["properties"]), ("run", "0.1", {})
    )
    test.assertIn(run["language"], ("python", "cpp"))
    test.assertIn(run["outcome"], _RUN_OUTCOMES)
    test.assertTrue(t0_ms <= run["started_at"] <= run["ended_at"] <= t1_ms, run)


def _report_files(prefix: Path) -> list[Path]:
    return sorted(prefix.parent.glob(f"{prefix.name}-*{torchci_report.REPORT_SUFFIX}"))


def _run_plugin(
    cwd: str, args: list[str], prefix: Path, env: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess, Path]:
    """Runs pytest with the report plugin; returns the process and its one report."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *args,
            "-p",
            "torch.testing._internal.torchci.plugin",
            f"--torchci-report-prefix={prefix}",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=cwd,
        env=_REPORT_CHILD_ENV | (env or {}),
        capture_output=True,
        text=True,
        timeout=300,
    )
    reports = _report_files(prefix)
    if len(reports) != 1:
        raise RuntimeError(
            f"pytest produced {len(reports)} reports\n{proc.stdout}\n{proc.stderr}"
        )
    return proc, reports[0]


def _runs(report: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in report.read_text().splitlines()[1:]]


def _table(header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    """Rows as aligned columns under a header, for assertExpectedInline."""
    lines = [header, *(tuple(str(cell) for cell in row) for row in rows)]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    return "\n".join(
        " | ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip()
        for line in lines
    )


# torchci test run reports, on TestJunitXml's fixture suite.
@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(
    TEST_CUDA or TEST_WITH_ROCM, "report shape doesn't depend on the device"
)
class TestReportJsonl(TestCase):
    raw: str
    report_name: str
    t0_ms: int
    t1_ms: int

    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cls.t0_ms = int(time.time() * 1000)
            # Non-default settings, to show up in flags.
            env = {
                "PYTORCH_TEST_WITH_SLOW": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "OPINFO_RESTRICT_TO_DSL": "triton",
            }
            _, report = _run_plugin(
                str(_TEST_DIR), [str(_PYTEST_SUITE)], Path(tmp) / "suite", env
            )
            cls.t1_ms = int(time.time() * 1000)
            cls.raw = report.read_text()
            cls.report_name = report.name
        super().setUpClass()

    def test_runs(self) -> None:
        _, *runs = (json.loads(line) for line in self.raw.splitlines())
        columns = ("case_name", "rerun_number", "outcome", "outcome_summary")
        self.assertExpectedInline(
            _table(columns, [tuple(run[column] for column in columns) for run in runs]),
            """\
case_name                 | rerun_number | outcome | outcome_summary
test_pass                 | 0            | passed  |
test_assert_failure       | 0            | failed  | AssertionError: values differ
test_raises_non_assertion | 0            | failed  | RuntimeError: runtime error!
test_error_in_setup       | 0            | error   | RuntimeError: setup error
test_error_in_teardown    | 0            | error   | RuntimeError: teardown error
test_skipped              | 0            | skipped | skipped unconditionally
test_skipif               | 0            | skipped | skipped conditionally
test_xfail                | 0            | xfailed | known bad
test_xpass_non_strict     | 0            | xpassed |
test_xpass_strict         | 0            | failed  | [XPASS(strict)] strictly expected to fail
test_rerun_then_pass      | 0            | failed  | AssertionError: attempt 1 fails
test_rerun_then_pass      | 1            | failed  | AssertionError: attempt 2 fails
test_rerun_then_pass      | 2            | passed  |
test_rerun_then_fail      | 0            | failed  | AssertionError: attempt 1 fails
test_rerun_then_fail      | 1            | failed  | AssertionError: attempt 2 fails
test_rerun_then_fail      | 2            | failed  | AssertionError: attempt 3 fails
test_no_rerun_needed      | 0            | passed  |""",
        )

    def test_schema(self) -> None:
        report, *runs = (json.loads(line) for line in self.raw.splitlines())
        self.assertEqual(
            list(report),
            [
                "type",
                "schema_version",
                "repo",
                "github_workflow_job_id",
                "report_uuid",
                "environment",
                "flags",
                "properties",
            ],
        )
        self.assertEqual(report["type"], "report")
        self.assertEqual(report["schema_version"], "0.1")
        self.assertEqual(report["repo"], "pytorch/pytorch")
        self.assertEqual(report["github_workflow_job_id"], 123456789)
        self.assertEqual(str(uuid.UUID(report["report_uuid"])), report["report_uuid"])
        self.assertEqual(
            self.report_name, f"suite-{report['report_uuid']}.report.jsonl"
        )
        flags = report["flags"]
        self.assertEqual(list(flags), sorted(TestEnvironment.env_var_values))
        self.assertTrue(all(isinstance(value, str) for value in flags.values()))
        self.assertEqual(flags["PYTORCH_TEST_WITH_SLOW"], "1")
        self.assertEqual(flags["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(flags["OPINFO_RESTRICT_TO_DSL"], "triton")
        self.assertEqual(flags["PYTORCH_TEST_WITH_INDUCTOR"], "")
        self.assertEqual(report["properties"]["build_environment"], "report-build")
        self.assertEqual(report["properties"]["test_config"], "report-config")
        self.assertEqual(report["properties"]["runner_name"], "report-runner")
        for run in runs:
            _assert_run_line(self, run, self.t0_ms, self.t1_ms)
            self.assertEqual(run["language"], "python")

    def test_identities(self) -> None:
        rendered = {}
        columns = ("suite", "case_name", "declared_case_name")
        with tempfile.TemporaryDirectory() as tmp:
            # In process, identities come from collection; under xdist, from the
            # worker's setup reports.
            for mode, args in (("in_process", []), ("xdist", ["-n", "1"])):
                t0_ms = int(time.time() * 1000)
                proc, report = _run_plugin(
                    str(_REPORT_TESTDATA),
                    ["identity_report.py", *args],
                    Path(tmp) / mode,
                )
                t1_ms = int(time.time() * 1000)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                runs = _runs(report)
                for run in runs:
                    _assert_run_line(self, run, t0_ms, t1_ms)
                    # The launched module's node id, relative to the repo root, also
                    # for TestImported, which identity_helpers.py defines.
                    self.assertEqual(
                        run["file"], "test/torchci/testdata/identity_report.py"
                    )
                    self.assertEqual(run["language"], "python")
                rendered[mode] = _table(
                    columns,
                    sorted(tuple(run[column] for column in columns) for run in runs),
                )
        self.assertEqual(rendered["xdist"], rendered["in_process"])
        # test_added is a lambda and test_made_cpu a factory's inner function named
        # test, so both keep their collected name.
        self.assertExpectedInline(
            rendered["in_process"],
            """\
suite           | case_name                | declared_case_name
                | test_module_function     | test_module_function
                | test_pytest_ids[a::b]    | test_pytest_ids
                | test_pytest_ids[c[d]]    | test_pytest_ids
TestDecorated   | test_decorated           | test_decorated
TestDeviceCPU   | test_device_cpu          | test_device
TestDeviceCPU   | test_dtype_cpu_float32   | test_dtype
TestDeviceCPU   | test_dtype_cpu_float64   | test_dtype
TestFactoryCPU  | test_made_cpu            | test_made_cpu
TestImported    | test_imported            | test_imported
TestInner       | test_nested              | test_nested
TestParametrize | test_parametrize_value_1 | test_parametrize
TestSetattr     | test_added               | test_added
TestUnittest    | test_unittest            | test_unittest""",
        )

    def test_subtests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, report = _run_plugin(
                str(_REPORT_TESTDATA), ["subtests_report.py"], Path(tmp) / "subtests"
            )
            runs = _runs(report)
        # One run per test, however many subtests it has.
        self.assertEqual(
            [(run["case_name"], run["outcome"]) for run in runs],
            [("test_failing_subtest", "failed")],
        )
        self.assertIn("1 != 0", runs[0]["outcome_summary"])

    def test_skip_after_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, report = _run_plugin(
                str(_REPORT_TESTDATA),
                ["skip_after_failure_report.py"],
                Path(tmp) / "skip",
            )
            runs = _runs(report)
        # The skip in teardown doesn't replace the failure's message.
        self.assertEqual(
            [(run["outcome"], run["outcome_summary"]) for run in runs],
            [("failed", "AssertionError: the real failure")],
        )

    @unittest.skipIf(
        not all(
            importlib.util.find_spec(name) for name in ("xdist", "pytest_flakefinder")
        ),
        "needs pytest-xdist and pytest-flakefinder",
    )
    def test_concurrent_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = [
                "concurrent_copies_report.py",
                "-n",
                "2",
                "--flake-finder",
                "--flake-runs=4",
            ]
            _, report = _run_plugin(str(_REPORT_TESTDATA), args, Path(tmp) / "copies")
            runs = _runs(report)
        # flakefinder's copies of a unittest test share its node id, and the two
        # workers run them at the same time; each copy still gets its own run line.
        self.assertEqual(sorted(run["rerun_number"] for run in runs), [0, 1, 2, 3])
        self.assertTrue(
            all(run["ended_at"] - run["started_at"] >= 300 for run in runs), runs
        )


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(TEST_CUDA or TEST_WITH_ROCM, "report failures don't need GPU coverage")
class TestReportFailureIsolation(TestCase):
    """A failing writer must not change outcomes or the exit code, and warns once.
    The fixture directory's conftest.py breaks the writer at the point FAIL_MODE
    names."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # For the junit XML, reports and caches.
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.baseline = cls._run("baseline", ["-p", "no:cacheprovider"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()
        super().tearDownClass()

    @classmethod
    def _run(cls, name: str, args: list[str], env: dict[str, str] | None = None):
        xml = cls.dir / f"{name}.xml"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "failure_isolation.py",
                "-q",
                f"--junitxml={xml}",
                *args,
            ],
            cwd=_REPORT_TESTDATA / "failure_isolation",
            env=_CHILD_ENV | (env or {}),
            capture_output=True,
            text=True,
            timeout=300,
        )
        outcomes = []
        for case in ET.parse(xml).iter("testcase"):
            child = next(iter(case), None)
            outcomes.append(
                (case.attrib["name"], "passed" if child is None else child.tag)
            )
        return proc, outcomes

    @parametrize(
        "mode",
        [
            subtest(mode, name=mode)
            for mode in ("capture", "write", "name", "finish", "worker", "cache")
        ],
    )
    def test_writer_error_does_not_change_tests(self, mode) -> None:
        args = [
            "-p",
            "torch.testing._internal.torchci.plugin",
            f"--torchci-report-prefix={self.dir / mode}",
        ]
        if mode == "cache":
            # The in-flight run is only published through the stepcurrent cache.
            args += ["--sc=report-failure", "-o", f"cache_dir={self.dir / 'cache'}"]
        else:
            args += ["-p", "no:cacheprovider"]
        if mode == "worker":
            # The writer runs on the controller; break the worker's makereport.
            args += ["-n", "1"]
        proc, outcomes = self._run(mode, args, {"FAIL_MODE": mode})
        baseline_proc, baseline_outcomes = self.baseline
        self.assertEqual(
            proc.returncode, baseline_proc.returncode, proc.stdout + proc.stderr
        )
        self.assertEqual(outcomes, baseline_outcomes)
        self.assertEqual(
            proc.stderr.count("torchci: report disabled after error:"), 1, proc.stderr
        )


instantiate_parametrized_tests(TestReportFailureIsolation)


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(
    TEST_CUDA or TEST_WITH_ROCM, "report enablement doesn't need GPU coverage"
)
class TestReportEnablement(TestCase):
    def test_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "reports"
            script = [
                sys.executable,
                "enablement_report.py",
                "--use-pytest",
                "-p",
                "no:cacheprovider",
            ]
            t0_ms = int(time.time() * 1000)
            proc = subprocess.run(
                [*script, f"--save-torchci-reports={report_dir}"],
                cwd=_REPORT_TESTDATA,
                env=_REPORT_CHILD_ENV,
                capture_output=True,
                text=True,
                timeout=300,
            )
            t1_ms = int(time.time() * 1000)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            reports = _report_files(report_dir / "enablement_report")
            self.assertEqual(len(reports), 1)
            self.assertRegex(
                reports[0].name,
                r"^enablement_report-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}\.report\.jsonl$",
            )
            records = [json.loads(line) for line in reports[0].read_text().splitlines()]
        self.assertEqual([record["type"] for record in records], ["report", "run"])
        _assert_run_line(self, records[1], t0_ms, t1_ms)
        self.assertEqual(records[1]["outcome"], "passed")

    def test_run_test_forwarding(self) -> None:
        run_test_module = importlib.import_module("run_test")
        args = run_test_module._torchci_report_args
        with unittest.mock.patch.object(run_test_module, "HAS_TORCHCI_REPORTS", True):
            self.assertEqual(args("/reports"), ["--save-torchci-reports=/reports"])
            self.assertEqual(args(None), [])
        with unittest.mock.patch.object(run_test_module, "HAS_TORCHCI_REPORTS", False):
            self.assertEqual(args("/reports"), [])


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(
    TEST_CUDA or TEST_WITH_ROCM, "crash recording doesn't need GPU coverage"
)
class TestReportCrashes(TestCase):
    def test_xdist_worker_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # -x, as run_test.py passes for C++ tests, stops xdist after the last
            # test's final failure, before its teardown.
            args = ["xdist_crash_report.py", "-n", "1", "--reruns", "1", "-x"]
            proc, report = _run_plugin(str(_REPORT_TESTDATA), args, Path(tmp) / "crash")
            runs = _runs(report)
        # 2: xdist reports a session stopped by -x as interrupted.
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        # pytest-rerunfailures reschedules the crashed test once.
        self.assertEqual(
            [(run["case_name"], run["outcome"], run["rerun_number"]) for run in runs],
            [
                ("test_before", "passed", 0),
                ("test_crash", "crashed", 0),
                ("test_crash", "crashed", 1),
                ("test_after", "passed", 0),
                ("test_fail_last", "failed", 0),
                ("test_fail_last", "failed", 1),
            ],
        )
        self.assertIn("crashed while running", runs[1]["outcome_summary"])

    def test_xdist_worker_crash_in_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, report = _run_plugin(
                str(_REPORT_TESTDATA),
                ["setup_crash_report.py", "-n", "1"],
                Path(tmp) / "setup_crash",
            )
            runs = _runs(report)
        # The worker died before sending the test's identity with its setup report,
        # so the controller takes it from the node id.
        test = tuple(
            runs[0][key]
            for key in ("file", "suite", "case_name", "declared_case_name", "outcome")
        )
        expected = (
            "test/torchci/testdata/setup_crash_report.py",
            "TestSetup",
            "test_setup_crash[one]",
            "test_setup_crash",
            "crashed",
        )
        self.assertEqual(test, expected)

    def test_interrupt_in_setup_stays_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "interrupted_setup_report.py",
                    "--sc=key",
                    "-o",
                    f"cache_dir={tmp}/cache",
                    "-p",
                    "torch.testing._internal.torchci.plugin",
                    f"--torchci-report-prefix={tmp}/interrupted",
                ],
                cwd=_REPORT_TESTDATA,
                env=_REPORT_CHILD_ENV,
                capture_output=True,
                text=True,
                timeout=300,
            )
            path = Path(tmp) / "cache/v/cache/stepcurrent/key/report_inflight"
            self.assertTrue(path.exists(), proc.stdout + proc.stderr)
            inflight = json.loads(path.read_text())
        # A timeout's SIGINT during setup comes before the setup report, and the test
        # stays published for run_test.py to record.
        self.assertIsNotNone(inflight)
        self.assertEqual(inflight["test"]["case_name"], "test_interrupted")

    def test_subprocess_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "reports"
            t0_ms = int(time.time() * 1000)
            proc = subprocess.run(
                [
                    sys.executable,
                    "subprocess_crash_report.py",
                    "--use-pytest",
                    "--subprocess",
                    f"--save-torchci-reports={report_dir}",
                    "-p",
                    "no:cacheprovider",
                ],
                cwd=_REPORT_TESTDATA,
                env=_REPORT_CHILD_ENV,
                capture_output=True,
                text=True,
                timeout=300,
            )
            t1_ms = int(time.time() * 1000)
            reports = _report_files(report_dir / "subprocess_crash_report")
            runs = [run for report in reports for run in _runs(report)]
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # Both child attempts (retry_shell retries once) wrote a report line before
        # dying; the parent wrote the third report, with the run.
        self.assertEqual(len(reports), 3)
        self.assertEqual(len(runs), 1)
        _assert_run_line(self, runs[0], t0_ms, t1_ms)
        # The parent names it from its pytest item, as the child would have.
        test = tuple(
            runs[0][key]
            for key in ("file", "suite", "case_name", "declared_case_name", "outcome")
        )
        expected = (
            "test/torchci/testdata/subprocess_crash_report.py",
            "TestCrash",
            "test_crash_x_1",
            "test_crash",
            "crashed",
        )
        self.assertEqual(test, expected)
        self.assertEqual(
            runs[0]["outcome_summary"], "the test process exited with code -9 (SIGKILL)"
        )

    @staticmethod
    def _inflight(case_name: str, rerun_number: int) -> dict[str, Any]:
        test = torchci_report.TestId(
            "test/test_x.py", "TestX", f"{case_name}[param]", "python", case_name
        )
        return {
            "test": test._asdict(),
            "started_at": time.time(),
            "rerun_number": rerun_number,
        }

    @parametrize(
        "exit_code, outcome",
        [
            subtest((-11, "crashed"), name="crashed"),
            subtest((124, "timed_out"), name="timed_out"),
        ],
    )
    def test_finish(self, exit_code, outcome) -> None:
        recovery = importlib.import_module("torch.testing._internal.torchci.recovery")
        inflight = self._inflight("test_hangs", 1)
        test = torchci_report.TestId(**inflight["test"])
        first = torchci_report.run_record(
            test, 0, "failed", inflight["started_at"], time.time(), "failed"
        )
        torn = '{"type":"run","file":"test/te'
        t0_ms = int(inflight["started_at"] * 1000)
        with TemporaryFileName() as path:
            # A process that died while writing its last line.
            Path(path).write_text(
                '{"type":"report"}\n' + torchci_report.line(first) + torn,
                encoding="utf-8",
            )
            recovery.finish(path, inflight, exit_code)
            # With nothing in flight there is nothing to record.
            recovery.finish(path, None, -11)
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        # The torn line stays a line of its own.
        self.assertEqual(lines[2], torn)
        runs = [json.loads(line) for line in (lines[1], *lines[3:])]
        for run in runs:
            _assert_run_line(self, run, t0_ms, int(time.time() * 1000))
        self.assertEqual(
            [(run["case_name"], run["rerun_number"], run["outcome"]) for run in runs],
            [("test_hangs[param]", 0, "failed"), ("test_hangs[param]", 1, outcome)],
        )
        self.assertEqual(runs[1]["declared_case_name"], "test_hangs")
        self.assertEqual(
            runs[1]["outcome_summary"], torchci_report.exit_summary(exit_code)
        )

    def test_finish_skips_a_written_run(self) -> None:
        recovery = importlib.import_module("torch.testing._internal.torchci.recovery")
        inflight = self._inflight("test_complete", 0)
        test = torchci_report.TestId(**inflight["test"])
        run = torchci_report.run_record(
            test, 0, "passed", inflight["started_at"], time.time()
        )
        with TemporaryFileName() as path:
            Path(path).write_text(
                '{"type":"report"}\n' + torchci_report.line(run), encoding="utf-8"
            )
            recovery.finish(path, inflight, -11)
            lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)

    @parametrize(
        "returned, elapsed, timeout, expected",
        [
            subtest((2, 10.0, 5.0, 124), name="pytest_exits_in_grace"),
            subtest((-2, 10.0, 5.0, 124), name="sigint_kills"),
            subtest((1, 10.0, 5.0, 1), name="failure_after_timeout"),
            subtest((2, 1.0, 5.0, 2), name="before_timeout"),
            subtest((2, 10.0, None, 2), name="no_timeout"),
        ],
    )
    def test_effective_exit_code(self, returned, elapsed, timeout, expected) -> None:
        recovery = importlib.import_module("torch.testing._internal.torchci.recovery")
        self.assertEqual(
            recovery.effective_exit_code(returned, elapsed, timeout), expected
        )

    def _run_test_retries(
        self,
        tmp: str,
        ret_code: int,
        cache: dict[str, str],
        timeout: float | None = None,
    ) -> str:
        run_test_module = importlib.import_module("run_test")
        cache_dir = Path(tmp) / ".pytest_cache/v/cache/stepcurrent/key"
        cache_dir.mkdir(parents=True)
        for name, value in cache.items():
            (cache_dir / name).write_text(value, encoding="utf-8")
        output = io.StringIO()
        with unittest.mock.patch.object(run_test_module, "REPO_ROOT", Path(tmp)):
            with unittest.mock.patch.object(
                run_test_module, "retry_shell", return_value=(ret_code, False)
            ):
                run_test_module.run_test_retries(
                    [], tmp, {}, timeout, "key", output, False, "test_file", object()
                )
        return output.getvalue()

    def test_run_test_finishes_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.jsonl"
            report.write_text('{"type":"report"}\n', encoding="utf-8")
            cache = {
                "report_path": json.dumps(str(report)),
                "report_inflight": json.dumps(self._inflight("test_t", 0)),
            }
            # pytest exited on the SIGINT that followed a timeout.
            self._run_test_retries(tmp, 2, cache, timeout=0)
            run = json.loads(report.read_text(encoding="utf-8").splitlines()[1])
            left = [
                path.name
                for path in (
                    Path(tmp) / ".pytest_cache/v/cache/stepcurrent/key"
                ).iterdir()
            ]
        self.assertEqual(run["outcome"], "timed_out")
        # Removed, so the next retry process doesn't finish this report again.
        self.assertEqual(left, [])

    def test_run_test_survives_a_corrupt_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = self._run_test_retries(tmp, -11, {"report_path": "{"})
        self.assertIn("Could not finish the test run report:", output)


instantiate_parametrized_tests(TestReportCrashes)


class TestReportHelpers(TestCase):
    @parametrize(
        "nodeid, expected",
        [
            subtest(
                (
                    "test/test_torch.py::TestTorch::test_add",
                    (
                        "test/test_torch.py",
                        "TestTorch",
                        "test_add",
                        "python",
                        "test_add",
                    ),
                ),
                name="python_method",
            ),
            subtest(
                (
                    "test/test_x.py::Outer::Inner::test_n",
                    ("test/test_x.py", "Inner", "test_n", "python", "test_n"),
                ),
                name="nested_class",
            ),
            subtest(
                (
                    "test/test_x.py::test_fn",
                    ("test/test_x.py", "", "test_fn", "python", "test_fn"),
                ),
                name="python_function",
            ),
            subtest(
                (
                    "test/test_x.py::TestP::test_p[a::b-1]",
                    ("test/test_x.py", "TestP", "test_p[a::b-1]", "python", "test_p"),
                ),
                name="pytest_parameter",
            ),
        ],
    )
    def test_nodeid_identity(self, nodeid, expected):
        self.assertEqual(torchci_report.nodeid_identity(nodeid), expected)

    def test_identity_survives_unwrap_errors(self):
        import pytest

        plugin = importlib.import_module("torch.testing._internal.torchci.plugin")

        def test_case(self):
            pass

        # inspect.unwrap raises on a wrapper loop.
        test_case.__wrapped__ = test_case
        module = unittest.mock.Mock(nodeid="test/test_x.py")
        item = unittest.mock.Mock(
            spec=pytest.Function, obj=test_case, originalname="test_case_cpu"
        )
        item.name = "test_case_cpu"
        item.getparent.side_effect = (
            lambda cls: module if cls is pytest.Module else None
        )
        # The declared name stays pytest's originalname.
        self.assertEqual(
            plugin.identity(item),
            ("test/test_x.py", "", "test_case_cpu", "python", "test_case_cpu"),
        )

    def test_lone_surrogate_is_replaced(self) -> None:
        now = time.time()
        test = torchci_report.TestId(
            "test/test_x.py", "", "test_surrogate", "python", "test_surrogate"
        )
        line = torchci_report.line(
            torchci_report.run_record(
                test, 0, "failed", now, now, outcome_summary="\ud800"
            )
        )
        line.encode("utf-8")
        run = json.loads(line)
        _assert_run_line(self, run, int(now * 1000), int(now * 1000))
        # json.dumps escapes it as \ud800, which ClickHouse rejects.
        self.assertEqual(run["outcome_summary"], "?")

    @skipIfTorchDynamo("Dynamo calls the patched torch._C function while tracing")
    def test_capture_skips_torch_accelerator(self) -> None:
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        with unittest.mock.patch.object(
            torch._C, "_accelerator_getAccelerator"
        ) as probe:
            environment.capture()
        probe.assert_not_called()

    def test_rocm_version(self):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        with unittest.mock.patch.multiple(
            torch.version, cuda=None, hip="7.16.26385", rocm="10.1.0"
        ):
            # The ROCm release, not HIP's version.
            self.assertEqual(environment._accelerator(), ("rocm", "10.1.0"))

    @parametrize(
        "config, expected",
        [
            # clang defines __GNUC__, so a clang build also prints a GCC line.
            subtest(
                (
                    "  - GCC 4.2\n  - C++ Version: 201703\n  - clang 21.1.0\n",
                    ("clang", "21"),
                ),
                name="clang",
            ),
            subtest(
                ("  - GCC 11.4\n  - C++ Version: 201703\n", ("gcc", "11")), name="gcc"
            ),
            # _MSC_FULL_VER of MSVC 19.41.34120.
            subtest(
                ("  - C++ Version: 201703\n  - MSVC 194134120\n", ("msvc", "19")),
                name="msvc",
            ),
            subtest(("  - C++ Version: 201703\n", ("", "")), name="unknown"),
        ],
    )
    def test_compiler(self, config, expected):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        self.assertEqual(
            environment._compiler(f"PyTorch built with:\n{config}"), expected
        )

    def test_windows_os_version(self):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        with (
            unittest.mock.patch.object(platform, "system", return_value="Windows"),
            unittest.mock.patch.object(platform, "version", return_value="10.0.17763"),
        ):
            # Server 2019's build, not the 10.0 shared by every Windows since 10.
            self.assertEqual(environment._os(), ("windows", "10.0.17763", "10.0.17763"))

    @skipIfTorchDynamo("environment capture does not need Dynamo coverage")
    def test_capture(self) -> None:
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        captured = environment.capture()
        env = captured.identity()
        self.assertEqual(
            list(env),
            [
                "os",
                "os_version",
                "cpu_architecture",
                "cpu_capability",
                "python_version",
                "cc_compiler",
                "cc_compiler_version",
                "accelerator",
                "accelerator_version",
                "device_count",
                "device_name",
            ],
        )
        for name, value in env.items():
            self.assertIsInstance(value, int if name == "device_count" else str)
        self.assertEqual(
            env["os"],
            {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}[
                platform.system()
            ],
        )
        free_threaded = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
        self.assertEqual(
            env["python_version"],
            f"{sys.version_info.major}.{sys.version_info.minor}{free_threaded}",
        )
        self.assertIn(env["cc_compiler"], ("", "gcc", "clang", "msvc"))
        self.assertIn(env["accelerator"], ("cpu", "cuda", "rocm", "xpu", "mps"))
        if env["device_count"] == 0:
            self.assertEqual(env["device_name"], "")
        self.assertEqual(captured.flags, TestEnvironment.env_var_values)
        property_names = {
            "torch_version",
            "os_release",
            "device_memory_mib",
            "driver_version",
            "host_memory_mib",
            "build_environment",
            "test_config",
            "runner_name",
        }
        self.assertTrue(set(captured.properties) <= property_names)
        self.assertTrue(
            all(
                isinstance(value, str) and value
                for value in captured.properties.values()
            )
        )


instantiate_parametrized_tests(TestReportHelpers)


class TestEnvVarValues(TestCase):
    """TestEnvironment.env_var_values, which test run reports record as flags."""

    def setUp(self):
        super().setUp()
        self._defined: list[str] = []

    def tearDown(self):
        for name in self._defined:
            # def_flag and def_setting also bind the name in common_utils.
            if hasattr(common_utils, name):
                delattr(common_utils, name)
            TestEnvironment.env_var_values.pop(name, None)
            TestEnvironment.repro_env_vars.pop(name, None)
        super().tearDown()

    def _def_flag(self, name, **kwargs):
        self._defined.append(name)
        kwargs.setdefault("include_in_repro", False)
        return TestEnvironment.def_flag(name, **kwargs)

    def _def_setting(self, name, **kwargs):
        self._defined.append(name)
        return TestEnvironment.def_setting(name, **kwargs)

    def test_env_var_values(self):
        # What test run reports record: each env var's value as set, "" if unset,
        # an implied flag as "1", and include_in_repro=False ones left out.
        env = {k: v for k, v in os.environ.items() if not k.startswith("FOO_EV_")}
        with unittest.mock.patch.dict(
            os.environ,
            env | {"FOO_EV_SET": "1", "FOO_EV_ZERO": "0", "FOO_EV_STR": "triton"},
            clear=True,
        ):
            self._def_flag("FOO_EV_SET", env_var="FOO_EV_SET", include_in_repro=True)
            self._def_flag("FOO_EV_ZERO", env_var="FOO_EV_ZERO", include_in_repro=True)
            self._def_flag(
                "FOO_EV_IMPLIED",
                env_var="FOO_EV_IMPLIED",
                include_in_repro=True,
                implied_by_fn=lambda: True,
            )
            self._def_flag("FOO_EV_OFF", env_var="FOO_EV_OFF", include_in_repro=True)
            self._def_flag(
                "FOO_EV_EXCLUDED", env_var="FOO_EV_EXCLUDED", implied_by_fn=lambda: True
            )
            self._def_setting("FOO_EV_STR", env_var="FOO_EV_STR")
            self._def_setting("FOO_EV_UNSET", env_var="FOO_EV_UNSET")
        values = {
            k: v
            for k, v in TestEnvironment.env_var_values.items()
            if k.startswith("FOO_EV_")
        }
        self.assertEqual(
            values,
            {
                "FOO_EV_SET": "1",
                "FOO_EV_ZERO": "0",
                "FOO_EV_IMPLIED": "1",
                "FOO_EV_OFF": "",
                "FOO_EV_STR": "triton",
                "FOO_EV_UNSET": "",
            },
        )
        # The repro command only needs what was set explicitly.
        repro = {
            k: v
            for k, v in TestEnvironment.repro_env_vars.items()
            if k.startswith("FOO_EV_")
        }
        self.assertEqual(repro, {"FOO_EV_SET": "1", "FOO_EV_STR": "triton"})

    def test_env_var_values_are_unparsed(self):
        # Like EXPANDABLE_SEGMENTS, which reads PYTORCH_CUDA_ALLOC_CONF, and
        # OPINFO_RESTRICT_TO_DSL: the env var's string, not the flag's bool or the
        # setting's parsed value.
        conf = "garbage_collection_threshold:0.6,expandable_segments:True"
        with unittest.mock.patch.dict(
            os.environ,
            {"FOO_EV_ALLOC_CONF": conf, "FOO_EV_DSL": "triton", "FOO_EV_INT": "08"},
        ):
            enabled_fn = functools.partial(
                common_utils.allocator_option_enabled_fn, option="expandable_segments"
            )
            self.assertTrue(
                self._def_flag(
                    "FOO_EV_ALLOC_CONF",
                    env_var="FOO_EV_ALLOC_CONF",
                    include_in_repro=True,
                    enabled_fn=enabled_fn,
                )
            )
            self.assertEqual(
                self._def_setting(
                    "FOO_EV_DSL",
                    env_var="FOO_EV_DSL",
                    parse_fn=lambda val: None if val is None else str(val),
                ),
                "triton",
            )
            self.assertEqual(
                self._def_setting(
                    "FOO_EV_INT",
                    env_var="FOO_EV_INT",
                    parse_fn=lambda val: None if val is None else int(val),
                ),
                8,
            )
        values = {
            k: v
            for k, v in TestEnvironment.env_var_values.items()
            if k.startswith("FOO_EV_")
        }
        self.assertEqual(
            values,
            {"FOO_EV_ALLOC_CONF": conf, "FOO_EV_DSL": "triton", "FOO_EV_INT": "08"},
        )


if __name__ == "__main__":
    run_tests()
