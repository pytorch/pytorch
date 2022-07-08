from _pytest.junitxml import LogXML, _NodeReporter, bin_xml_escape
from _pytest.stash import StashKey
from _pytest.reports import TestReport
from _pytest.config.argparsing import Parser
from _pytest.config import filename_arg
from _pytest.config import Config
from _pytest._code.code import ReprFileLocation
from typing import Union
from typing import Optional
import xml.etree.ElementTree as ET
import functools


xml_key = StashKey["LogXML2"]()


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("terminal reporting")
    group.addoption(
        "--junit-xml-reruns",
        action="store",
        dest="xmlpath_reruns",
        metavar="path",
        type=functools.partial(filename_arg, optname="--junit-xml-me"),
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
    )  # choices=['total', 'call'])
    parser.addini(
        "junit_family_me",
        "Emit XML for schema: one of legacy|xunit1|xunit2",
        default="xunit2",
    )


def pytest_configure(config: Config) -> None:
    xmlpath = config.option.xmlpath_reruns
    # Prevent opening xmllog on worker nodes (xdist).
    if xmlpath and not hasattr(config, "workerinput"):
        junit_family = config.getini("junit_family")
        config.stash[xml_key] = LogXML2(
            xmlpath,
            config.option.junitprefix,
            config.getini("junit_suite_name_reruns"),
            config.getini("junit_logging_reruns"),
            config.getini("junit_duration_report_reruns"),
            junit_family,
            config.getini("junit_log_passing_tests_reruns"),
        )
        config.pluginmanager.register(config.stash[xml_key])


def pytest_unconfigure(config: Config) -> None:
    xml = config.stash.get(xml_key, None)
    if xml:
        del config.stash[xml_key]
        config.pluginmanager.unregister(xml)


class NodeReporter(_NodeReporter):
    def _prepare_content(self, content: str, header: str) -> str:
        return content

    def _write_content(self, report: TestReport, content: str, jheader: str) -> None:
        if content == "":
            return
        tag = ET.Element(jheader)
        tag.text = bin_xml_escape(content)
        self.append(tag)


class LogXML2(LogXML):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append_rerun(self, reporter: _NodeReporter, report: TestReport) -> None:
        # msg = str(report.longrepr.reprtraceback.extraline)
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

    def node_reporter(self, report: Union[TestReport, str]) -> NodeReporter:
        nodeid: Union[str, TestReport] = getattr(report, "nodeid", report)
        # Local hack to handle xdist report order.
        workernode = getattr(report, "node", None)

        key = nodeid, workernode

        if key in self.node_reporters:
            # TODO: breaks for --dist=each
            return self.node_reporters[key]

        reporter = NodeReporter(nodeid, self)

        self.node_reporters[key] = reporter
        self.node_reporters_ordered.append(reporter)

        return reporter
