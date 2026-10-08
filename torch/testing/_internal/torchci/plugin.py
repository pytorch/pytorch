"""pytest plugin that writes the test run report (README.md). Enable it with
``-p torch.testing._internal.torchci.plugin --torchci-report-dir=<dir>``. The
first error turns the writer off, so it never changes a test's outcome."""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import pytest

from torch.testing._internal.torchci import environment, report


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from typing import IO

    from _pytest.config import Config
    from _pytest.config.argparsing import Parser
    from _pytest.main import Session
    from _pytest.reports import TestReport

# Set where the test runs; xdist ships it to the controller with the report. Not
# item.user_properties, which pytest also writes into the junit XML CI ingests.
_DECLARED_CASE_NAME = "_torchci_declared_case_name"

_disabled = False
_writer: ReportWriter | None = None


def _disable(error: Exception) -> None:
    global _disabled
    if _disabled:
        return
    _disabled = True
    try:
        print(f"torchci: report disabled after error: {error!r}", file=sys.stderr)
        if _writer is not None and _writer.file is not None:
            report_file, _writer.file = _writer.file, None
            report_file.close()
    except Exception:
        pass


@contextlib.contextmanager
def _guard() -> Iterator[None]:
    try:
        yield
    except Exception as error:
        _disable(error)


def _item_declared_case_name(item: Any) -> str:
    """The test's name before parametrize or device-type suffixes."""
    fallback = report.fallback_declared_case_name(item.nodeid)
    try:
        function = inspect.unwrap(item.obj)
        # A function made by a factory, like create_test_func's inner test, is
        # named for the factory's code, not for the test.
        if "<locals>" in getattr(function, "__qualname__", "<locals>"):
            return fallback
        name = function.__name__
        case_name = report.identity(item.nodeid).case_name
        if case_name == name or case_name.startswith((name + "_", name + "[")):
            return name
        return fallback
    except Exception:
        return fallback


def _failure_summary(test_report: TestReport) -> str:
    crash = getattr(test_report.longrepr, "reprcrash", None)
    return crash.message if crash is not None else str(test_report.longrepr)


@dataclass
class _Run:
    started: float
    declared_case_name: str
    ended: float = 0.0
    failed_phase: str = ""
    skipped: bool = False
    wasxfail: bool = False
    outcome_summary: str = ""

    def outcome(self) -> str:
        if self.failed_phase:
            return "failed" if self.failed_phase == "call" else "error"
        if self.skipped:
            return "xfailed" if self.wasxfail else "skipped"
        return "xpassed" if self.wasxfail else "passed"


class ReportWriter:
    def __init__(self, path: str, report_uuid: str) -> None:
        self.path = path
        self.report_uuid = report_uuid
        self.file: IO[str] | None = None
        self.runs: dict[str, _Run] = {}
        self.rerun_numbers: Counter[str] = Counter()
        self.declared_case_names: dict[str, str] = {}

    def pytest_sessionstart(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            record = report.report_record(self.report_uuid, environment.capture())
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.file = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
            self.file.write(report.line(record))
            self.file.flush()

    def pytest_collection_finish(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            # Empty on an xdist controller; workers send names via makereport.
            self.declared_case_names = {
                item.nodeid: _item_declared_case_name(item) for item in session.items
            }

    def pytest_runtest_logstart(self, nodeid: str, location: Any) -> None:
        if _disabled:
            return
        with _guard():
            names = self.declared_case_names
            declared = names.get(nodeid) or report.fallback_declared_case_name(nodeid)
            self.runs[nodeid] = _Run(time.time(), declared)

    # Before test/conftest.py's LogXMLReruns rewrites skip longreprs.
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if _disabled:
            return
        with _guard():
            self._logreport(report)

    def _logreport(self, test_report: TestReport) -> None:
        run = self.runs.get(test_report.nodeid)
        if run is None or test_report.when not in ("setup", "call", "teardown"):
            return
        if test_report.when == "setup":
            run.started = test_report.start
            declared_case_name = getattr(test_report, _DECLARED_CASE_NAME, None)
            if declared_case_name:
                run.declared_case_name = declared_case_name
        run.ended = test_report.stop
        # A failing pytest-subtests subtest fails the run.
        subtest = getattr(test_report, "context", None) is not None
        if test_report.failed or test_report.outcome == "rerun":
            if not run.failed_phase:
                run.failed_phase = test_report.when
                run.outcome_summary = _failure_summary(test_report)
        elif test_report.skipped and not (subtest or run.skipped or run.failed_phase):
            from _pytest.terminal import _get_raw_skip_reason

            run.skipped = True
            run.wasxfail = hasattr(test_report, "wasxfail")
            run.outcome_summary = _get_raw_skip_reason(test_report)
        elif test_report.when == "call" and not subtest:
            run.wasxfail = hasattr(test_report, "wasxfail")
        # A rerun attempt ends at its "rerun" report; others end at teardown.
        if test_report.outcome == "rerun" or test_report.when == "teardown":
            self._finish(test_report.nodeid, run)

    def _finish(self, nodeid: str, run: _Run) -> None:
        del self.runs[nodeid]
        if self.file is not None:
            record = report.run_record(
                nodeid,
                self.rerun_numbers[nodeid],
                run.outcome(),
                run.started,
                run.ended,
                outcome_summary=run.outcome_summary,
                declared_case_name=run.declared_case_name,
            )
            self.file.write(report.line(record))
            self.file.flush()
        self.rerun_numbers[nodeid] += 1

    def pytest_sessionfinish(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            if self.file is not None:
                self.file.close()
                self.file = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Generator[None, Any, None]:
    # Runs on xdist workers too. Skips results whose hook raised.
    outcome = yield
    if _disabled or call.when != "setup" or outcome.excinfo is not None:
        return
    with _guard():
        name = _item_declared_case_name(item)
        setattr(outcome.get_result(), _DECLARED_CASE_NAME, name)


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--torchci-report-dir",
        action="store",
        default=None,
        metavar="dir",
        help="write this process's test run report into this directory",
    )


def pytest_configure(config: Config) -> None:
    global _writer
    with _guard():
        directory = config.getoption("torchci_report_dir")
        # The xdist controller writes; collect-only sessions run nothing.
        worker = hasattr(config, "workerinput")
        if directory and not worker and not config.getoption("collectonly"):
            report_uuid = str(uuid.uuid4())
            path = report.report_path(directory, report_uuid)
            _writer = ReportWriter(path, report_uuid)
            config.pluginmanager.register(_writer, "torchci_report_writer")
