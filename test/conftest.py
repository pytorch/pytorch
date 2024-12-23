import copy
import functools
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from types import MethodType
from typing import Any, List, Optional, TYPE_CHECKING, Union

import pytest
from _pytest.config import Config, filename_arg
from _pytest.config.argparsing import Parser
from _pytest.junitxml import _NodeReporter, bin_xml_escape, LogXML
from _pytest.python import Module
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import _get_raw_skip_reason

from pytest_shard_custom import pytest_addoptions as shard_addoptions, PytestShardPlugin


if TYPE_CHECKING:
    from _pytest._code.code import ReprFileLocation

# a lot of this file is copied from _pytest.junitxml and modified to get rerun info

xml_key = StashKey["LogXMLReruns"]()
STEPCURRENT_CACHE_DIR = "cache/stepcurrent"


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("general")

    parser.addoption("--sc", action="store", default=None, dest="stepcurrent")

    parser.addoption("--use-main-module", action="store_true")
    group = parser.getgroup("terminal reporting")
    group.addoption(
        "--junit-xml-reruns",
        action="store",
        dest="xmlpath_reruns",
        metavar="path",
        type=functools.partial(filename_arg, optname="--junit-xml-reruns"),
        default=None,
        help="create junit-xml style report file at given path.",
    )
    group.addoption(
        "--junit-prefix-reruns",
        action="store",
        metavar="str",
        default=None,
        help="prepend prefix to classnames in junit-xml output",
    )
    parser.addini(
        "junit_suite_name_reruns", "Test suite name for JUnit report", default="pytest"
    )
    parser.addini(
        "junit_logging_reruns",
        "Write captured log messages to JUnit report: "
        "one of no|log|system-out|system-err|out-err|all",
        default="no",
    )
    parser.addini(
        "junit_log_passing_tests_reruns",
        "Capture log information for passing tests to JUnit report: ",
        type="bool",
        default=True,
    )
    parser.addini(
        "junit_duration_report_reruns",
        "Duration time to report: one of total|call",
        default="total",
    )
    parser.addini(
        "junit_family_reruns",
        "Emit XML for schema: one of legacy|xunit1|xunit2",
        default="xunit2",
    )
    shard_addoptions(parser)


def pytest_configure(config: Config) -> None:
    xmlpath = config.option.xmlpath_reruns
    # Prevent opening xmllog on worker nodes (xdist).
    if xmlpath and not hasattr(config, "workerinput"):
        junit_family = config.getini("junit_family_reruns")
        config.stash[xml_key] = LogXMLReruns(
            xmlpath,
            config.option.junitprefix,
            config.getini("junit_suite_name_reruns"),
            config.getini("junit_logging_reruns"),
            config.getini("junit_duration_report_reruns"),
            junit_family,
            config.getini("junit_log_passing_tests_reruns"),
        )
        config.pluginmanager.register(config.stash[xml_key])
    if config.getoption("stepcurrent"):
        config.pluginmanager.register(StepcurrentPlugin(config), "stepcurrentplugin")
    if config.getoption("num_shards"):
        config.pluginmanager.register(PytestShardPlugin(config), "pytestshardplugin")


def pytest_unconfigure(config: Config) -> None:
    xml = config.stash.get(xml_key, None)
    if xml:
        del config.stash[xml_key]
        config.pluginmanager.unregister(xml)


class _NodeReporterReruns(_NodeReporter):
    def _prepare_content(self, content: str, header: str) -> str:
        return content

    def _write_content(self, report: TestReport, content: str, jheader: str) -> None:
        if content == "":
            return
        tag = ET.Element(jheader)
        tag.text = bin_xml_escape(content)
        self.append(tag)

    def append_skipped(self, report: TestReport) -> None:
        # Referenced from the below
        # https://github.com/pytest-dev/pytest/blob/2178ee86d7c1ee93748cfb46540a6e40b4761f2d/src/_pytest/junitxml.py#L236C6-L236C6
        # Modified to escape characters not supported by xml in the skip reason.  Everything else should be the same.
        if hasattr(report, "wasxfail"):
            # Super here instead of the actual code so we can reduce possible divergence
            super().append_skipped(report)
        else:
            assert isinstance(report.longrepr, tuple)
            filename, lineno, skipreason = report.longrepr
            if skipreason.startswith("Skipped: "):
                skipreason = skipreason[9:]
            details = f"{filename}:{lineno}: {skipreason}"

            skipped = ET.Element(
                "skipped", type="pytest.skip", message=bin_xml_escape(skipreason)
            )
            skipped.text = bin_xml_escape(details)
            self.append(skipped)
            self.write_captured_output(report)


class LogXMLReruns(LogXML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append_rerun(self, reporter: _NodeReporter, report: TestReport) -> None:
        if hasattr(report, "wasxfail"):
            reporter._add_simple("skipped", "xfail-marked test passes unexpectedly")
        else:
            assert report.longrepr is not None
            reprcrash: Optional[ReprFileLocation] = getattr(
                report.longrepr, "reprcrash", None
            )
            if reprcrash is not None:
                message = reprcrash.message
            else:
                message = str(report.longrepr)
            message = bin_xml_escape(message)
            reporter._add_simple("rerun", message, str(report.longrepr))

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        super().pytest_runtest_logreport(report)
        if report.outcome == "rerun":
            reporter = self._opentestcase(report)
            self.append_rerun(reporter, report)
        if report.outcome == "skipped":
            if isinstance(report.longrepr, tuple):
                fspath, lineno, reason = report.longrepr
                reason = f"{report.nodeid}: {_get_raw_skip_reason(report)}"
                report.longrepr = (fspath, lineno, reason)

    def node_reporter(self, report: Union[TestReport, str]) -> _NodeReporterReruns:
        nodeid: Union[str, TestReport] = getattr(report, "nodeid", report)
        # Local hack to handle xdist report order.
        workernode = getattr(report, "node", None)

        key = nodeid, workernode

        if key in self.node_reporters:
            # TODO: breaks for --dist=each
            return self.node_reporters[key]

        reporter = _NodeReporterReruns(nodeid, self)

        self.node_reporters[key] = reporter
        self.node_reporters_ordered.append(reporter)

        return reporter


# imitating summary_failures in pytest's terminal.py
# both hookwrapper and tryfirst to make sure this runs before pytest's
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # prints stack traces for reruns
    if terminalreporter.config.option.tbstyle != "no":
        reports = terminalreporter.getreports("rerun")
        if reports:
            terminalreporter.write_sep("=", "RERUNS")
            if terminalreporter.config.option.tbstyle == "line":
                for rep in reports:
                    line = terminalreporter._getcrashline(rep)
                    terminalreporter.write_line(line)
            else:
                for rep in reports:
                    msg = terminalreporter._getfailureheadline(rep)
                    terminalreporter.write_sep("_", msg, red=True, bold=True)
                    terminalreporter._outrep_summary(rep)
                    terminalreporter._handle_teardown_sections(rep.nodeid)
    yield


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, path, parent) -> Module:
    if parent.config.getoption("--use-main-module"):
        mod = Module.from_parent(parent, path=module_path)
        mod._getobj = MethodType(lambda x: sys.modules["__main__"], mod)
        return mod


@pytest.hookimpl(hookwrapper=True)
def pytest_report_teststatus(report, config):
    # Add the test time to the verbose output, unforunately I don't think this
    # includes setup or teardown
    pluggy_result = yield
    if not isinstance(report, pytest.TestReport):
        return
    outcome, letter, verbose = pluggy_result.get_result()
    if verbose:
        pluggy_result.force_result(
            (outcome, letter, f"{verbose} [{report.duration:.4f}s]")
        )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: List[Any]) -> None:
    """
    This hook is used when rerunning disabled tests to get rid of all skipped tests
    instead of running and skipping them N times. This avoids flooding the console
    and XML outputs with junk. So we want this to run last when collecting tests.
    """
    rerun_disabled_tests = os.getenv("PYTORCH_TEST_RERUN_DISABLED_TESTS", "0") == "1"
    if not rerun_disabled_tests:
        return

    disabled_regex = re.compile(r"(?P<test_name>.+)\s+\([^\.]+\.(?P<test_class>.+)\)")
    disabled_tests = defaultdict(set)

    # This environment has already been set by run_test before it calls pytest
    disabled_tests_file = os.getenv("DISABLED_TESTS_FILE", "")
    if not disabled_tests_file or not os.path.exists(disabled_tests_file):
        return

    with open(disabled_tests_file) as fp:
        for disabled_test in json.load(fp):
            m = disabled_regex.match(disabled_test)
            if m:
                test_name = m["test_name"]
                test_class = m["test_class"]
                disabled_tests[test_class].add(test_name)

    # When rerunning disabled test, ignore all test cases that are not disabled
    filtered_items = []

    for item in items:
        test_name = item.name
        test_class = item.parent.name

        if (
            test_class not in disabled_tests
            or test_name not in disabled_tests[test_class]
        ):
            continue

        cpy = copy.copy(item)
        cpy._initrequest()

        filtered_items.append(cpy)

    items.clear()
    # NB: Need to edit items directly here to have the list reflected back to pytest
    items.extend(filtered_items)


class StepcurrentPlugin:
    """
    This is meant to work in with test/run_test.py to ensure that every test is
    run, gets retries in new subprocesses, and creates xml.  To do this, it
    keeps track of each test's status in the cache.  Normal tests are run
    together in the same process, but when a test fails, it needs to be run
    singly.  This class and run_test.py have logic to ensure that this happens,
    especially for unusual cases like segfaults where the process exits
    immediately and doesn't produce xml for any of the tests, and tests that
    cause the process to fail at exit which creates xml that believes the test
    is successful.

    If a test fails normally, it gets run singly in a new subprocess.  If a test
    segfaults, all tests prior to the segfault get rerun to generate xml, and
    the segfaulting test gets run singly in a new subprocess.  If a process
    exist with non 0 exit code but no tests seem to fail, then all the test will
    be rerun singly to narrow down which test caused the failure.

    Cache file contents:
    pytest_previous_status: None on the first run, and then is either an exit
    code or "no xml"
    to_run: all tests that have still not created xml
    prev_run: tests that are expected to run during this run.  Poorly named in
    this context, but makes sense from test/run_test.py's POV.  If a test fails,
    then it is possibly that not all tests in this actually ran. This should be
    a prefix of to_run
    already_ran: tests that have created xml

    Test that are still to be run have 4 possible statuses, which are in
    test/run_test.py and are: cont, s, r2, and r1, abbreviated c, s, 2, and 1 in
    the table

    cont: Test can be run in same process as other tests, this is the default status
    s: Test should be run singly
    r2: Test should be run singly, and has 2 retries left
    r1: Test should be run singly, and has 1 retry left

    At the start of the run, all tests get the status c. This plugin will either
    run all tests with status c, or run a single test with the status s, 2, or
    1. If a test is known to fail, then the status of the test is changed to 2
    or 1 depending on how many retires it still has left.

    Examples failure causes with 3 tests:
    +-------------+--------+----------+
    | already_ran | to_run | prev_run |
    +-------------+--------+----------+
    2 fails normally, generates correct xml, so all we need to do is rerun it
    more in a new subprocess to see if it consistently fails (which it does in
    this example)
    |     | ccc | ccc |
    | c   |  2c |  2  |
    | c   |  1c |  1  |
    | cf  |   c |   c |
    +-----+-----+-----+
    2 segfaults, so no xml gets created.  Rerun the previous tests to generate
    xml, and then rerun 2 on its own in a new subprocess
    |     | ccc | ccc | 2 segfaults
    |     | c2c | c   | Rerun previous tests to generate xml
    | c   |  2c |  2  |
    | c   |  1c |  1  |
    | cf  |   c |   c |
    +-----+-----+-----+
    A test (2 in this example) caused the process to fail but doesn't fail
    during the run.  An example of this might be incorrectly freeing memory on
    process exit.  Rerun all the tests singly to figure out which one caused
    this
    |     | ccc | ccc | 2 causes process to fail
    |     | sss | s   | rerun all tests singly
    | s   |  ss |  s  | 2 is the one causing the failure
    | s   |  2s |  2  |
    | s   |  1s |  1  |
    | cf  |   s |   s |
    +-----+-----+-----+
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.report_status = ""
        assert config.cache is not None
        self.cache: pytest.Cache = config.cache
        directory = f"{STEPCURRENT_CACHE_DIR}/{config.getoption('stepcurrent')}"
        self.cache_info_init_path = f"{directory}/init"
        self.cache_info_active_path = f"{directory}/active"

        init_cache = self.cache.get(self.cache_info_init_path, None)
        active_cache = self.cache.get(self.cache_info_active_path, None)
        self.prev_run = init_cache["prev_run"] if init_cache else []
        self.to_run = init_cache["to_run"] if init_cache else []
        self.already_ran = init_cache["already_ran"] if init_cache else []
        self.ended_at = active_cache["ended_at"] if active_cache else None
        self.pytest_previous_status = (
            active_cache["pytest_previous_status"] if active_cache else None
        )

    def get_index(self, nodeid: str, items) -> int:
        for index, item in enumerate(items):
            if item.nodeid == nodeid:
                return index
        raise ValueError(f"Could not find {nodeid} in items")

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config: Config, items: List[Any]) -> None:
        if self.pytest_previous_status is None:
            # This is the first run, so we don't need to do anything
            self.report_status += "First run, starting from the beginning"
            # Special case of the status == "cont" below where everything is cont
            for item in items:
                self.prev_run.append([item.nodeid, "cont"])
                self.to_run.append([item.nodeid, "cont"])
            self.pytest_previous_status = "no xml"
            self.save_cache()
            return
        # validate that the cache is correct
        for item, test in zip(
            items,
            self.already_ran + self.to_run,
        ):
            assert (
                item.nodeid == test[0]
            ), f"Cache and discovered tests do not match, {item.nodeid} != {test[0]}"

        first_test, status = self.to_run[0]
        first_index = self.get_index(first_test, items)
        self.prev_run = []
        deselected = []

        if status == "cont":
            # We're in the middle of a test run
            deselected = items[:first_index]
            items[:] = items[first_index:]
            self.report_status += f"Continuing from {first_test}"
            i, test = next(
                ((i, t) for i, t in enumerate(self.to_run) if t[1] != "cont"),
                (len(self.to_run), None),
            )
            deselected += items[i:]
            items[:] = items[:i]
            if test is not None:
                self.report_status += f", stopping at {test[0]} (exclusive)"
            self.prev_run = self.to_run[:i]
        else:
            # Run single test
            deselected = items[:first_index] + items[first_index + 1 :]
            items[:] = items[first_index : first_index + 1]
            self.prev_run = [[first_test, status]]
            self.report_status += f"Running single test {first_test}"

        self.pytest_previous_status = "no xml"
        self.save_cache()
        config.hook.pytest_deselected(items=deselected)

    def pytest_report_collectionfinish(self) -> Optional[str]:
        if self.config.getoption("verbose") >= 0 and self.report_status:
            return f"stepcurrent: {self.report_status}"
        return None

    def pytest_runtest_protocol(self, item, nextitem) -> None:
        self.ended_at = item.nodeid
        self.save_cache_active()

    def save_cache(self) -> None:
        self.cache.set(
            self.cache_info_init_path,
            {
                "prev_run": self.prev_run,
                "to_run": self.to_run,
                "already_ran": self.already_ran,
            },
        )
        self.save_cache_active()

    def save_cache_active(self) -> None:
        self.cache.set(
            self.cache_info_active_path,
            {
                "ended_at": self.ended_at,
                "pytest_previous_status": self.pytest_previous_status,
            },
        )

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        self.pytest_previous_status = 0 if exitstatus == 5 else exitstatus
        self.save_cache_active()
