"""pytest plugin that writes the test run report (README.md). Enable it with
``-p torch.testing._internal.torchci.plugin --torchci-report-prefix=<prefix>``. The
first error turns the writer off, so it never changes a test's outcome."""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
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
_TEST_ID = "_torchci_test_id"

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


def identity(item: pytest.Item) -> report.TestId | None:
    """None for an item that isn't a Python test function."""
    if not isinstance(item, pytest.Function):
        return None
    # originalname drops pytest parameters. Unwrapping the function recovers the
    # source name of PyTorch's generated device, dtype and parametrize variants.
    declared_case_name = item.originalname
    with contextlib.suppress(Exception):
        function = inspect.unwrap(item.obj)
        name = function.__name__
        # A function made by a factory, like create_test_func's inner test, is
        # named for the factory's code, not for the test.
        if "<locals>" not in function.__qualname__ and (
            item.name == name or item.name.startswith((name + "_", name + "["))
        ):
            declared_case_name = name
    cls = item.getparent(pytest.Class)
    return report.TestId(
        file=item.getparent(pytest.Module).nodeid,
        suite=cls.name if cls is not None else "",
        case_name=item.name,
        language="python",
        declared_case_name=declared_case_name,
    )


def _failure_summary(test_report: TestReport) -> str:
    crash = getattr(test_report.longrepr, "reprcrash", None)
    return crash.message if crash is not None else str(test_report.longrepr)


@dataclass
class _Run:
    started: float
    test: report.TestId | None
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
        # By node id and xdist worker (report.node, None without xdist), as junitxml
        # keys its test cases: flakefinder's copies of a test share a node id and can
        # run on two workers at once.
        self.runs: dict[tuple[str, Any], _Run] = {}
        self.rerun_numbers: Counter[str] = Counter()
        self.tests: dict[str, report.TestId | None] = {}

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
            # Empty on an xdist controller; workers send identities via makereport.
            self.tests = {item.nodeid: identity(item) for item in session.items}

    # Before test/conftest.py's LogXMLReruns rewrites skip longreprs.
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if _disabled:
            return
        with _guard():
            self._logreport(report)

    def _logreport(self, test_report: TestReport) -> None:
        nodeid = test_report.nodeid
        key = (nodeid, getattr(test_report, "node", None))
        if test_report.when == "setup":
            test_id = getattr(test_report, _TEST_ID, None)
            test = report.TestId(**test_id) if test_id else self.tests.get(nodeid)
            self.runs[key] = _Run(test_report.start, test)
        run = self.runs.get(key)
        if run is None or test_report.when not in ("setup", "call", "teardown"):
            return
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
            self._finish(key, run)

    def _finish(self, key: tuple[str, Any], run: _Run) -> None:
        del self.runs[key]
        nodeid = key[0]
        # No identity: the item isn't a Python test function, or, under xdist, the
        # worker's writer turned itself off and its setup report came without one.
        if self.file is not None and run.test is not None:
            record = report.run_record(
                run.test,
                self.rerun_numbers[nodeid],
                run.outcome(),
                run.started,
                run.ended,
                outcome_summary=run.outcome_summary,
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
        if test := identity(item):
            # A dict, which xdist can serialize.
            setattr(outcome.get_result(), _TEST_ID, test._asdict())


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--torchci-report-prefix",
        action="store",
        default=None,
        metavar="prefix",
        help="write this process's test run report to <prefix>-<uuid>.report.jsonl",
    )


def pytest_configure(config: Config) -> None:
    global _writer
    with _guard():
        prefix = config.getoption("torchci_report_prefix")
        # The xdist controller writes; collect-only sessions run nothing.
        worker = hasattr(config, "workerinput")
        if prefix and not worker and not config.getoption("collectonly"):
            report_uuid = str(uuid.uuid4())
            path = report.report_path(prefix, report_uuid)
            _writer = ReportWriter(path, report_uuid)
            config.pluginmanager.register(_writer, "torchci_report_writer")
