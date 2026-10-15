"""pytest plugin that writes the test run report (README.md). Enable it with
``-p torch.testing._internal.torchci.plugin --torchci-report-prefix=<prefix>``. The
first error turns the writer off, so it never changes a test's outcome."""

from __future__ import annotations

import contextlib
import inspect
import json
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
_TEST_ID = "_torchci_test_id"
# The run in flight, published for run_test.py in test/conftest.py's stepcurrent
# directory (take_inflight). Fields: path (the report's absolute path),
# report_size (its size when the run started), test (the run line's identity
# fields), rerun_number, started_at (epoch seconds); null when nothing is in
# flight.
_INFLIGHT_FILE = "report_inflight"

_disabled = False
_writer: ReportWriter | None = None


def _disable(error: Exception) -> None:
    global _disabled
    if _disabled:
        return
    _disabled = True
    try:
        print(f"torchci: report disabled after error: {error!r}", file=sys.stderr)
        if _writer is not None:
            # Nothing more gets published, so what's there would go stale.
            _writer._publish(None)
            if _writer.file is not None:
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
    started: float = 0.0
    test: report.TestId | None = None
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
        # run_test.py reads the published path from another working directory.
        self.path = os.path.abspath(path)
        self.report_uuid = report_uuid
        self.file: IO[str] | None = None
        # By node id and xdist worker (report.node, None without xdist), as junitxml
        # keys its test cases: flakefinder's copies of a test share a node id and can
        # run on two workers at once.
        self.runs: dict[tuple[str, Any], _Run] = {}
        self.rerun_numbers: Counter[str] = Counter()
        self.tests: dict[str, report.TestId | None] = {}
        # Set when test/conftest.py's stepcurrent plugin is active (--sc, --scs or
        # --rs) without xdist; only run_test.py's retry loop reads what is
        # published (take_inflight).
        self.cache: pytest.Cache | None = None
        self.inflight_key = ""

    def _publish(self, inflight: dict[str, Any] | None) -> None:
        # Best effort: a failed publication must not turn the report off.
        if self.cache is not None:
            with contextlib.suppress(Exception):
                self.cache.set(self.inflight_key, inflight)

    def pytest_sessionstart(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            config = session.config
            stepcurrent = config.pluginmanager.get_plugin("stepcurrentplugin")
            if stepcurrent and not config.getoption("numprocesses", default=None):
                self.cache = config.cache
                self.inflight_key = f"{stepcurrent.directory}/{_INFLIGHT_FILE}"
            record = report.report_record(self.report_uuid, environment.capture())
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.file = open(self.path, "w", encoding="utf-8", newline="\n")  # noqa: SIM115
            self.file.write(report.line(record))
            self.file.flush()

    def pytest_collection_finish(self, session: Session) -> None:
        if _disabled:
            return
        with _guard():
            # Names the run in flight at logstart. Empty on an xdist controller,
            # whose workers send identities via makereport.
            self.tests = {item.nodeid: identity(item) for item in session.items}

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        if _disabled or self.file is None:
            return
        with _guard():
            # Before setup, so run_test.py can record the run if the process dies.
            # pytest-rerunfailures starts each attempt here.
            if (test := self.tests.get(nodeid)) is not None:
                inflight = {
                    "path": self.path,
                    "report_size": os.fstat(self.file.fileno()).st_size,
                    "test": test._asdict(),
                    "rerun_number": self.rerun_numbers[nodeid],
                    "started_at": time.time(),
                }
                self._publish(inflight)

    # Before test/conftest.py's LogXMLReruns rewrites skip longreprs.
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if _disabled:
            return
        with _guard():
            self._logreport(report)

    def _logreport(self, test_report: TestReport) -> None:
        if test_report.when not in ("setup", "call", "teardown"):
            return
        nodeid = test_report.nodeid
        key = (nodeid, getattr(test_report, "node", None))
        # As in junitxml, a test's first report creates its run. That's usually
        # setup, but pytest-rerunfailures sends an attempt's setup, call and teardown
        # reports only once the attempt ends, while pytest-subtests sends a
        # subtest's report during the call, so it can come first.
        run = self.runs.setdefault(key, _Run())
        if test_report.when == "setup":
            test_id = getattr(test_report, _TEST_ID, None)
            run.started = test_report.start
            run.test = report.TestId(**test_id) if test_id else self.tests.get(nodeid)
        # A failing subtest fails the run. Both pytest-subtests' SubTestReport and
        # pytest 9's built-in pytest.SubtestReport carry a context; only the latter
        # is public.
        subtest = getattr(test_report, "context", None) is not None
        run.ended = test_report.stop
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
        # Cleared before the line is written: a process killed between the two
        # loses the run rather than recording it twice.
        self._publish(None)
        # No identity: the item isn't a Python test function, its setup report never
        # arrived (the attempt was interrupted), or, under xdist, the worker's writer
        # turned itself off and its setup report came without one.
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


def take_inflight(cache_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """For run_test.py: the run the writer left in flight, or None. Removed even
    when unreadable, so no later process records it."""
    file = os.path.join(cache_dir, _INFLIGHT_FILE)
    try:
        with open(file, encoding="utf-8") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(file)
    # Empty when the process died inside the cache write.
    return json.loads(data) if data.strip() else None


def append_run(inflight: dict[str, Any], outcome: str, outcome_summary: str) -> None:
    """For run_test.py: appends the run in flight to its report with ``outcome``."""
    path = inflight["path"]
    try:
        size = os.path.getsize(path)
    except FileNotFoundError:
        return
    # The writer wrote after publishing this run, so it did finish: its clear was
    # lost (pytest's cache swallows write errors), or its last line is torn.
    if size != inflight["report_size"]:
        return
    run = report.run_record(
        report.TestId(**inflight["test"]),
        inflight["rerun_number"],
        outcome,
        inflight["started_at"],
        time.time(),
        outcome_summary=outcome_summary,
    )
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(report.line(run))


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
