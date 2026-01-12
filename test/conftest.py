import copy
import functools
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from types import MethodType
from typing import Any, Optional, TYPE_CHECKING, Union

import pytest
from _pytest.config import Config, filename_arg
from _pytest.config.argparsing import Parser
from _pytest.junitxml import _NodeReporter, bin_xml_escape, LogXML
from _pytest.python import Module
from _pytest.reports import TestReport
from _pytest.stash import StashKey
from _pytest.terminal import _get_raw_skip_reason
from pytest_shard_custom import pytest_addoptions as shard_addoptions, PytestShardPlugin


try:
    from torch.testing._internal.common_utils import parse_cmd_line_args
except ImportError:
    # Temporary workaround needed until parse_cmd_line_args makes it into a nightlye because
    # main / PR's tests are sometimes run against the previous day's nightly which won't
    # have this function.
    def parse_cmd_line_args():
        pass


if TYPE_CHECKING:
    from _pytest._code.code import ReprFileLocation

# a lot of this file is copied from _pytest.junitxml and modified to get rerun info

xml_key = StashKey["LogXMLReruns"]()
STEPCURRENT_CACHE_DIR = "cache/stepcurrent"


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("general")
    group.addoption("--scs", action="store", default=None, dest="stepcurrent_skip")
    group.addoption("--sc", action="store", default=None, dest="stepcurrent")
    group.addoption("--rs", action="store", default=None, dest="run_single")

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
    parse_cmd_line_args()
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
    if config.getoption("stepcurrent_skip"):
        config.option.stepcurrent = config.getoption("stepcurrent_skip")
    if config.getoption("run_single"):
        config.option.stepcurrent = config.getoption("run_single")
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
            skipreason = skipreason.removeprefix("Skipped: ")
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
    # Add the test time to the verbose output, unfortunately I don't think this
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
def pytest_collection_modifyitems(items: list[Any]) -> None:
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
    # Modified fromo _pytest/stepwise.py in order to save the currently running
    # test instead of the last failed test
    def __init__(self, config: Config) -> None:
        self.config = config
        self.report_status = ""
        assert config.cache is not None
        self.cache: pytest.Cache = config.cache
        directory = f"{STEPCURRENT_CACHE_DIR}/{config.getoption('stepcurrent')}"
        self.lastrun_location = f"{directory}/lastrun"
        self.lastrun: Optional[str] = self.cache.get(self.lastrun_location, None)
        self.initial_val = self.lastrun
        self.skip: bool = config.getoption("stepcurrent_skip")
        self.run_single: bool = config.getoption("run_single")

        self.made_failing_xml_location = f"{directory}/made_failing_xml"
        self.cache.set(self.made_failing_xml_location, False)

    def pytest_collection_modifyitems(self, config: Config, items: list[Any]) -> None:
        if not self.lastrun:
            self.report_status = "Cannot find last run test, not skipping"
            return

        # check all item nodes until we find a match on last run
        failed_index = None
        for index, item in enumerate(items):
            if item.nodeid == self.lastrun:
                failed_index = index
                if self.skip:
                    failed_index += 1
                break

        # If the previously failed test was not found among the test items,
        # do not skip any tests.
        if failed_index is None:
            self.report_status = "previously run test not found, not skipping."
        else:
            self.report_status = f"skipping {failed_index} already run items."
            deselected = items[:failed_index]
            del items[:failed_index]
            if self.run_single:
                self.report_status += f" Running only {items[0].nodeid}"
                deselected += items[1:]
                del items[1:]
            config.hook.pytest_deselected(items=deselected)

    def pytest_report_collectionfinish(self) -> Optional[str]:
        if self.config.getoption("verbose") >= 0 and self.report_status:
            return f"stepcurrent: {self.report_status}"
        return None

    def pytest_runtest_protocol(self, item, nextitem) -> None:
        self.lastrun = item.nodeid
        self.cache.set(self.lastrun_location, self.lastrun)

    def pytest_sessionfinish(self, session, exitstatus):
        if exitstatus == 0:
            self.cache.set(self.lastrun_location, self.initial_val)
        if exitstatus != 0:
            self.cache.set(self.made_failing_xml_location, True)
