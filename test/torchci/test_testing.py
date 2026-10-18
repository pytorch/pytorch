# Owner(s): ["module: ci"]

import contextlib
import functools
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
import time
import types
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

    def test_reruns_keep_outcomes(self) -> None:
        # run_test.py always passes --reruns, under which pytest-rerunfailures sends
        # an attempt's reports only once it ends. Each test's last run must still
        # match the run without it.
        def last_outcomes(raw: str) -> dict[str, str]:
            runs = (json.loads(line) for line in raw.splitlines()[1:])
            return {run["case_name"]: run["outcome"] for run in runs}

        with tempfile.TemporaryDirectory() as tmp:
            args = [str(_PYTEST_SUITE), "--reruns", "2"]
            _, report = _run_plugin(str(_TEST_DIR), args, Path(tmp) / "suite")
            raw = report.read_text()
        self.assertEqual(last_outcomes(raw), last_outcomes(self.raw))

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
        self.assertEqual(flags["TEST_CONFIG"], "report-config")
        self.assertEqual(report["properties"]["build_environment"], "report-build")
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
        # With --reruns, as run_test.py passes, pytest-rerunfailures sends the
        # test's setup report after its subtests' reports.
        for args in ([], ["--reruns", "1"]):
            with tempfile.TemporaryDirectory() as tmp:
                _, report = _run_plugin(
                    str(_REPORT_TESTDATA),
                    ["subtests_report.py", *args],
                    Path(tmp) / "subtests",
                )
                runs = _runs(report)
            # One run per test, however many subtests it has.
            self.assertEqual(
                [(run["case_name"], run["outcome"]) for run in runs],
                [("test_failing_subtest", "failed")],
                args,
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

    def test_xdist_maxfail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # -x, as run_test.py passes for C++ tests, stops xdist after the last
            # test's final failure, before its teardown report.
            args = ["xdist_report.py", "-n", "1", "--reruns", "1", "-x"]
            proc, report = _run_plugin(str(_REPORT_TESTDATA), args, Path(tmp) / "xdist")
            runs = _runs(report)
        # 2: xdist reports a session stopped by -x as interrupted.
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        # test_crash killed its worker. pytest-rerunfailures sends a test's reports
        # only after each attempt, so the controller never got its identity, and
        # the crash is left to junit and the log.
        self.assertEqual(
            [(run["case_name"], run["outcome"], run["rerun_number"]) for run in runs],
            [
                ("test_before", "passed", 0),
                ("test_after", "passed", 0),
                ("test_fail_last", "failed", 0),
                ("test_fail_last", "failed", 1),
            ],
        )

    def test_session_without_tests_leaves_no_report(self) -> None:
        # Like run_test.py's serial pass over a file without serial tests. A session
        # interrupted after its test started keeps its report for run_test.py
        # (TestReportCrashes.test_interrupt_leaves_the_run_in_flight).
        with tempfile.TemporaryDirectory() as tmp:
            for args in ([], ["-n", "1"]):
                prefix = Path(tmp) / f"empty{len(args)}"
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "xdist_report.py",
                        "-k",
                        "no_such_test",
                        *args,
                        "-p",
                        "torch.testing._internal.torchci.plugin",
                        f"--torchci-report-prefix={prefix}",
                        "-p",
                        "no:cacheprovider",
                        "-q",
                    ],
                    cwd=_REPORT_TESTDATA,
                    env=_REPORT_CHILD_ENV,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                # 5: no tests ran.
                self.assertEqual(proc.returncode, 5, proc.stdout + proc.stderr)
                self.assertNotIn("torchci:", proc.stderr)
                self.assertEqual(_report_files(prefix), [])


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
            for mode in ("capture", "write", "name", "finish", "worker")
        ],
    )
    def test_writer_error_does_not_change_tests(self, mode) -> None:
        args = [
            "-p",
            "torch.testing._internal.torchci.plugin",
            f"--torchci-report-prefix={self.dir / mode}",
            "-p",
            "no:cacheprovider",
        ]
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
            self.assertEqual(
                args("test_type_info", False, "/reports"),
                ["--save-torchci-reports=/reports"],
            )
            # C++ tests don't go through run_tests, so they get the plugin itself.
            self.assertEqual(
                args("cpp/test_api", True, "/reports"),
                [
                    "-p",
                    "torch.testing._internal.torchci.plugin",
                    "--torchci-report-prefix=/reports/cpp.test_api",
                ],
            )
            self.assertEqual(args("test_type_info", False, None), [])
            self.assertEqual(args("cpp/test_api", True, None), [])
        with unittest.mock.patch.object(run_test_module, "HAS_TORCHCI_REPORTS", False):
            self.assertEqual(args("test_type_info", False, "/reports"), [])

    def test_run_test_default_reports_dir(self) -> None:
        run_test_module = importlib.import_module("run_test")
        default = run_test_module._torchci_reports_default
        ci_dir = str(run_test_module.REPO_ROOT / "test/torchci-reports")
        for is_ci, xpu, expected in (
            (False, None, None),
            (True, None, ci_dir),
            # XPU builds write none in CI.
            (True, "20250101", None),
        ):
            with (
                unittest.mock.patch.object(run_test_module, "IS_CI", is_ci),
                unittest.mock.patch.object(torch.version, "xpu", xpu),
            ):
                self.assertEqual(default(), expected)


# gtest's JSON test list (--gtest_list_tests --gtest_output=json:<file>), with the
# fields gtest_identities reads.
_GTEST_LISTING = {
    "testsuites": [
        {"name": "ModulesTest", "testsuite": [{"name": "Linear"}]},
        {"name": "TypedTest/0", "testsuite": [{"name": "Add", "type_param": "int"}]},
        {
            "name": "Inst/TypedPTest/1",
            "testsuite": [{"name": "Add", "type_param": "float"}],
        },
        # gtest doesn't check a custom type name generator's names.
        {
            "name": "TypedTest/c10.Half",
            "testsuite": [{"name": "Add", "type_param": "Half"}],
        },
        {
            "name": "Inst/PrimeTest",
            "testsuite": [
                {"name": "Returns/0", "value_param": "2"},
                {"name": "Returns/Small", "value_param": "3"},
            ],
        },
    ]
}


@unittest.skipIf(IS_WINDOWS, "Skipping because doesn't work for windows")
@unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
@skipIfTorchDynamo("subprocess test does not need Dynamo coverage")
@unittest.skipIf(TEST_WITH_CROSSREF, "subprocess test does not need crossref coverage")
@unittest.skipIf(
    TEST_CUDA or TEST_WITH_ROCM, "crash recording doesn't depend on the device"
)
class TestReportCrashes(TestCase):
    """run_test.py's retry loop records the run a process was running when it
    died, from what the writer published in the stepcurrent cache."""

    @staticmethod
    def _cache_dir(tmp: str) -> Path:
        stepcurrent_dir = importlib.import_module("conftest").STEPCURRENT_CACHE_DIR
        return Path(tmp) / ".pytest_cache/v" / stepcurrent_dir / "key"

    @staticmethod
    def _pytest(tmp: str, fixture: str, prefix: Path) -> list[str]:
        """A pytest command whose cache is under tmp, as run_test.py's would be."""
        return [
            sys.executable,
            "-m",
            "pytest",
            fixture,
            "-q",
            "-o",
            f"cache_dir={tmp}/.pytest_cache",
            "-p",
            "torch.testing._internal.torchci.plugin",
            f"--torchci-report-prefix={prefix}",
        ]

    def _run_test_retries(
        self, tmp, command, retry_shell=None, retries=0, env=None, timeout=None
    ) -> str:
        """Runs run_test.py's retry loop on command and returns its log. With no
        process retries, a failure is consistent after one process."""
        run_test_module = importlib.import_module("run_test")
        options = types.SimpleNamespace(upload_artifacts_while_running=False)
        log = Path(tmp) / "run_test.log"
        with contextlib.ExitStack() as stack:
            patch = functools.partial(unittest.mock.patch.object, run_test_module)
            stack.enter_context(patch("REPO_ROOT", Path(tmp)))
            stack.enter_context(patch("NUM_PROCESS_RETRIES", retries))
            if retry_shell is not None:
                stack.enter_context(patch("retry_shell", retry_shell))
            output = stack.enter_context(open(log, "w"))
            run_test_module.run_test_retries(
                command,
                str(_REPORT_TESTDATA),
                _REPORT_CHILD_ENV | (env or {}),
                timeout,
                "key",
                output,
                False,
                "crash_report",
                options,
            )
        return log.read_text()

    def _publish(self, tmp: str, report: Path, test, rerun: int, start: float):
        """Publishes a run in flight as the writer does; returns the cache dir."""
        cache_dir = self._cache_dir(tmp)
        cache_dir.mkdir(parents=True)
        inflight = {
            "path": str(report),
            "report_size": report.stat().st_size if report.exists() else 0,
            "test": test._asdict(),
            "rerun_number": rerun,
            "started_at": start,
        }
        (cache_dir / "report_inflight").write_text(json.dumps(inflight))
        return cache_dir

    def test_retry_sequence(self) -> None:
        # Under --sc, test_b fails and then kills the process on its in-process
        # rerun; it passes under --rs. test_c crashes under --scs and again under
        # --rs, so it fails consistently after one process retry.
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "reports/crash_sequence_report"
            state = Path(tmp) / "state"
            state.mkdir()
            fixture = "crash_sequence_report.py"
            command = self._pytest(tmp, fixture, prefix) + ["--reruns=1"]
            env = {"CRASH_SEQUENCE_STATE": str(state)}
            t0_ms = int(time.time() * 1000)
            log = self._run_test_retries(tmp, command, retries=1, env=env)
            t1_ms = int(time.time() * 1000)
            # One report per process, in the order the processes ran.
            reports = sorted(
                (_runs(report) for report in _report_files(prefix)),
                key=lambda runs: runs[0]["started_at"] if runs else 0,
            )
            left = sorted(file.name for file in self._cache_dir(tmp).iterdir())
        self.assertIn("Got exit code -9 (SIGKILL)", log)
        self.assertIn("Test succeeded in new process", log)
        self.assertIn(
            "FAILED CONSISTENTLY: test/torchci/testdata/crash_sequence_report.py::test_c",
            log,
        )
        columns = (
            "suite", "case_name", "declared_case_name", "rerun_number", "outcome", "outcome_summary",
        )  # fmt: skip
        rows = []
        for process, runs in enumerate(reports, 1):
            for run in runs:
                _assert_run_line(self, run, t0_ms, t1_ms)
                rows.append((process, *(run[column] for column in columns)))
        # The crashed rerun continues the in-process count; the passing --rs
        # process leaves nothing in flight for the --scs process after it, and a
        # crash in a new process is rerun 0 again.
        self.assertExpectedInline(
            _table(("process", *columns), rows),
            """\
process | suite | case_name   | declared_case_name | rerun_number | outcome | outcome_summary
1       |       | test_a      | test_a             | 0            | passed  |
1       | TestB | test_b[one] | test_b             | 0            | failed  | AssertionError: first attempt fails
1       | TestB | test_b[one] | test_b             | 1            | crashed | exit code -9 (SIGKILL)
2       | TestB | test_b[one] | test_b             | 0            | passed  |
3       |       | test_c      | test_c             | 0            | crashed | exit code -9 (SIGKILL)
4       |       | test_c      | test_c             | 0            | crashed | exit code -9 (SIGKILL)""",
        )
        # Consumed, so no later process records the run again.
        self.assertNotIn("report_inflight", left)

    def test_shutdown_crash_records_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "reports/shutdown_crash_report"
            command = self._pytest(tmp, "shutdown_crash_report.py", prefix)
            log = self._run_test_retries(tmp, command)
            (report,) = _report_files(prefix)
            runs = _runs(report)
            left = sorted(file.name for file in self._cache_dir(tmp).iterdir())
        # The process died after its last run line was written and the in-flight
        # record cleared, so there is no run to record.
        self.assertIn("Got exit code -9 (SIGKILL)", log)
        self.assertEqual(
            [(run["case_name"], run["outcome"]) for run in runs],
            [("test_only", "passed")],
        )
        self.assertNotIn("report_inflight", left)

    def test_disabled_writer_clears_the_run_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command = self._pytest(tmp, "failure_isolation.py", Path(tmp) / "finish")
            proc = subprocess.run(
                command + ["--sc=key"],
                cwd=_REPORT_TESTDATA / "failure_isolation",
                env=_REPORT_CHILD_ENV | {"FAIL_MODE": "finish"},
                capture_output=True,
                text=True,
                timeout=300,
            )
            inflight = (self._cache_dir(tmp) / "report_inflight").read_text()
        self.assertIn("torchci: report disabled after error:", proc.stderr)
        # The writer published its first test, then broke in _finish. Nothing it
        # published may outlive it for run_test.py to record.
        self.assertEqual(inflight, "null")

    def test_publishes_only_under_the_retry_loop(self) -> None:
        # Without --sc nothing is published, and under xdist the controller, which
        # writes the report, runs no test.
        with tempfile.TemporaryDirectory() as tmp:
            command = self._pytest(tmp, "crash_sequence_report.py", Path(tmp) / "a")
            for args in ([], ["--sc=key", "-n", "1"]):
                proc = subprocess.run(
                    command + ["-k", "test_a", *args],
                    cwd=_REPORT_TESTDATA,
                    env=_REPORT_CHILD_ENV,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            published = (self._cache_dir(tmp) / "report_inflight").exists()
        self.assertFalse(published)

    def test_interrupt_leaves_the_run_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "interrupted"
            command = self._pytest(tmp, "interrupted_setup_report.py", prefix)
            proc = subprocess.run(
                command + ["--sc=key"],
                cwd=_REPORT_TESTDATA,
                env=_REPORT_CHILD_ENV,
                capture_output=True,
                text=True,
                timeout=300,
            )
            cache_dir = self._cache_dir(tmp)
            inflight = json.loads((cache_dir / "report_inflight").read_text())
            (report,) = _report_files(prefix)
            content = report.read_bytes()
        # pytest exits 2 on the KeyboardInterrupt a timeout's SIGINT raises; the
        # test, interrupted before its setup report, wrote no run line and stays
        # published for run_test.py, with the report's size at its start.
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertEqual(content.count(b"\n"), 1)
        self.assertEqual(inflight["path"], str(report))
        self.assertEqual(inflight["report_size"], len(content))
        self.assertEqual(inflight["test"]["case_name"], "test_interrupted")
        self.assertEqual(inflight["rerun_number"], 0)

    def test_timeout_reaches_retry_shell(self) -> None:
        # pytest exits on the SIGINT that follows a timeout, within the grace
        # period; retry_shell must still report the timeout as 124.
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        with tempfile.TemporaryFile("w+") as output:
            code, _ = common_utils.retry_shell(
                command, stdout=output, stderr=output, timeout=1, retries=0
            )
            output.seek(0)
            log = output.read()
        self.assertEqual(code, 124)
        self.assertIn("Command took >0min, returning 124", log)

    @parametrize(
        "exit_code, outcome, summary",
        [
            subtest((-11, "crashed", "exit code -11 (SIGSEGV)"), name="signal"),
            subtest((1, "crashed", "exit code 1"), name="exit"),
            subtest((124, "timed_out", "timed out after 30min"), name="timed_out"),
            subtest((2, None, ""), name="interrupted"),
            subtest((3, None, ""), name="internal_error"),
        ],
    )
    def test_records_the_run_in_flight(self, exit_code, outcome, summary) -> None:
        # retry_shell's exit code: 124 is its timeout; 2 and 3 are pytest's own
        # exits (interrupted, internal error), not deaths, so nothing is recorded.
        test = torchci_report.TestId(
            "test/test_x.py", "TestX", "test_x[p]", "python", "test_x"
        )
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "x.report.jsonl"
            report.write_text('{"type":"report"}\n', encoding="utf-8")
            cache_dir = self._publish(tmp, report, test, 2, t0)
            shell = unittest.mock.Mock(return_value=(exit_code, False))
            self._run_test_retries(tmp, [], retry_shell=shell, timeout=30 * 60)
            # Consumed: a second exit records nothing more.
            self._run_test_retries(tmp, [], retry_shell=shell, timeout=30 * 60)
            lines = report.read_text(encoding="utf-8").split("\n")
            left = sorted(file.name for file in cache_dir.iterdir())
        self.assertEqual(left, [])
        if outcome is None:
            self.assertEqual(lines, ['{"type":"report"}', ""])
            return
        self.assertEqual(len(lines), 3, lines)
        run = json.loads(lines[1])
        _assert_run_line(self, run, int(t0 * 1000), int(time.time() * 1000))
        self.assertEqual({k: run[k] for k in test._fields}, test._asdict())
        self.assertEqual((run["rerun_number"], run["outcome"]), (2, outcome))
        self.assertEqual(run["outcome_summary"], summary)

    def test_skips_a_stale_record(self) -> None:
        # The writer wrote after publishing, so its clear was lost and the run did
        # finish: it must not be recorded again.
        test = torchci_report.TestId("test/x.py", "", "test_x", "python", "test_x")
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "x.report.jsonl"
            report.write_text('{"type":"report"}\n', encoding="utf-8")
            cache_dir = self._publish(tmp, report, test, 0, time.time())
            with open(report, "a", encoding="utf-8") as f:
                f.write('{"type":"run"}\n')
            crash = unittest.mock.Mock(return_value=(-11, False))
            log = self._run_test_retries(tmp, [], retry_shell=crash)
            lines = report.read_text(encoding="utf-8").splitlines()
            consumed = not (cache_dir / "report_inflight").exists()
        self.assertEqual(lines, ['{"type":"report"}', '{"type":"run"}'])
        self.assertTrue(consumed)
        self.assertNotIn("Could not finish", log)

    def test_skips_a_removed_report(self) -> None:
        test = torchci_report.TestId("test/x.py", "", "test_x", "python", "test_x")
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "gone.report.jsonl"
            cache_dir = self._publish(tmp, report, test, 0, time.time())
            crash = unittest.mock.Mock(return_value=(-11, False))
            log = self._run_test_retries(tmp, [], retry_shell=crash)
            created = report.exists()
            consumed = not (cache_dir / "report_inflight").exists()
        self.assertFalse(created)
        self.assertTrue(consumed)
        self.assertNotIn("Could not finish", log)

    def test_run_test_survives_a_corrupt_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = self._cache_dir(tmp)
            cache_dir.mkdir(parents=True)
            (cache_dir / "report_inflight").write_text("{")
            crash = unittest.mock.Mock(return_value=(-11, False))
            log = self._run_test_retries(tmp, [], retry_shell=crash)
            consumed = not (cache_dir / "report_inflight").exists()
        self.assertIn("Could not finish the test run report:", log)
        self.assertIn("No stepcurrent file found", log)
        # Consumed although unreadable, so no later process logs it again.
        self.assertTrue(consumed)


instantiate_parametrized_tests(TestReportCrashes)


class TestReportHelpers(TestCase):
    def test_gtest_identities(self) -> None:
        def cpp(suite: str, case_name: str, declared_case_name: str) -> tuple:
            return ("cpp/test_api", suite, case_name, "cpp", declared_case_name)

        self.assertEqual(
            torchci_report.gtest_identities("build/bin/test_api.exe", _GTEST_LISTING),
            {
                "ModulesTest.Linear": cpp("ModulesTest", "Linear", "Linear"),
                "TypedTest/0.Add": cpp("TypedTest/0", "Add", "Add"),
                "Inst/TypedPTest/1.Add": cpp("Inst/TypedPTest/1", "Add", "Add"),
                "TypedTest/c10.Half.Add": cpp("TypedTest/c10.Half", "Add", "Add"),
                "Inst/PrimeTest.Returns/0": cpp(
                    "Inst/PrimeTest", "Returns/0", "Returns"
                ),
                "Inst/PrimeTest.Returns/Small": cpp(
                    "Inst/PrimeTest", "Returns/Small", "Returns"
                ),
            },
        )

    @unittest.skipIf(
        importlib.util.find_spec("pytest_cpp") is None, "pytest-cpp isn't installed"
    )
    def test_gtest_item_identity(self) -> None:
        from pytest_cpp.boost import BoostTestFacade
        from pytest_cpp.catch2 import Catch2Facade
        from pytest_cpp.google import GoogleTestFacade
        from pytest_cpp.plugin import CppItem

        plugin = importlib.import_module("torch.testing._internal.torchci.plugin")

        def list_tests(args: list[str], **kwargs: Any) -> None:
            listing = Path(args[-1].removeprefix("--gtest_output=json:"))
            listing.write_text(json.dumps(_GTEST_LISTING))

        # identity reads only the item's binary, name and framework.
        item = object.__new__(CppItem)
        item.path, item.facade = Path("build/bin/test_api"), GoogleTestFacade()
        plugin._gtest_identities.cache_clear()
        self.addCleanup(plugin._gtest_identities.cache_clear)
        with unittest.mock.patch("subprocess.run", side_effect=list_tests) as run:
            item.name = "TypedTest/c10.Half.Add"
            self.assertEqual(
                plugin.identity(item),
                ("cpp/test_api", "TypedTest/c10.Half", "Add", "cpp", "Add"),
            )
            item.name = "Unlisted.Test"
            self.assertIsNone(plugin.identity(item))
            # pytest-cpp's other frameworks don't have gtest's test list.
            for facade in (BoostTestFacade(), Catch2Facade()):
                item.facade = facade
                self.assertIsNone(plugin.identity(item))
        # Once per binary.
        run.assert_called_once()

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

    @skipIfTorchDynamo("Dynamo calls the patched torch._C functions while tracing")
    @parametrize(
        "accelerator, library_count",
        [
            subtest(("cuda", "torch.cuda._device_count_nvml"), name="cuda"),
            subtest(("rocm", "torch.cuda._device_count_amdsmi"), name="rocm"),
            subtest(("xpu", "torch.xpu._device_count_zes"), name="xpu"),
        ],
    )
    def test_device_count_never_poisons_fork(self, accelerator, library_count):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        # The device runtimes' counts register the fork handler that poisons fork.
        with (
            unittest.mock.patch.object(
                torch._C, "_cuda_getDeviceCount", create=True
            ) as cuda_runtime,
            unittest.mock.patch.object(
                torch._C, "_xpu_getDeviceCount", create=True
            ) as xpu_runtime,
        ):
            with unittest.mock.patch(library_count, return_value=2):
                self.assertEqual(environment._device_count(accelerator), 2)
            # The library can't count devices: 0, with no runtime fallback.
            with unittest.mock.patch(library_count, return_value=-1):
                self.assertEqual(environment._device_count(accelerator), 0)
        cuda_runtime.assert_not_called()
        xpu_runtime.assert_not_called()

    @skipIfTorchDynamo("patches torch.mps")
    def test_device_count_mps(self) -> None:
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        # Not 0, which would also leave the device name out of MPS reports.
        with unittest.mock.patch("torch.mps.device_count", return_value=1):
            self.assertEqual(environment._device_count("mps"), 1)

    def test_rocm_version(self):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        with unittest.mock.patch.multiple(
            torch.version, cuda=None, hip="7.16.26385", rocm="10.1.0"
        ):
            # The ROCm release, not HIP's version.
            self.assertEqual(environment._accelerator(), ("rocm", "10.1.0"))

    @skipIfTorchDynamo("patches torch.__config__")
    @parametrize(
        "compiler, expected",
        [
            subtest((("clang", "21.1.0"), ("clang", "21.1.0")), name="clang"),
            subtest((("gcc", "11.4"), ("gcc", "11.4")), name="gcc"),
            # _MSC_FULL_VER of MSVC 19.41.34120.
            subtest((("msvc", "194134120"), ("msvc", "194134120")), name="msvc"),
            # A torch that predates torch.__config__._cxx_compiler.
            subtest((None, ("", "")), name="missing"),
        ],
    )
    def test_compiler(self, compiler, expected):
        environment = importlib.import_module(
            "torch.testing._internal.torchci.environment"
        )
        cxx_compiler = None if compiler is None else (lambda: compiler)
        with unittest.mock.patch.object(
            torch.__config__, "_cxx_compiler", cxx_compiler, create=True
        ):
            self.assertEqual(environment._compiler(), expected)

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
