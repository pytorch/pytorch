from __future__ import annotations

from _pytest import nodes
from _pytest.cacheprovider import Cache
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.reports import TestReport


STEPWISE_CACHE_DIR = "cache/stepwise"


def pytest_addoption(parser: Parser) -> None:
    group = parser.getgroup("general")
    group.addoption(
        "--sw",
        "--stepwise",
        action="store_true",
        default=False,
        dest="stepwise",
        help="Exit on test failure and continue from last failing test next time",
    )
    group.addoption(
        "--sw-skip",
        "--stepwise-skip",
        action="store_true",
        default=False,
        dest="stepwise_skip",
        help="Ignore the first failing test but stop on the next failing test. "
        "Implicitly enables --stepwise.",
    )


def pytest_configure(config: Config) -> None:
    if config.option.stepwise_skip:
        # allow --stepwise-skip to work on its own merits.
        config.option.stepwise = True
    if config.getoption("stepwise"):
        config.pluginmanager.register(StepwisePlugin(config), "stepwiseplugin")


def pytest_sessionfinish(session: Session) -> None:
    if not session.config.getoption("stepwise"):
        assert session.config.cache is not None
        if hasattr(session.config, "workerinput"):
            # Do not update cache if this process is a xdist worker to prevent
            # race conditions (#10641).
            return
        # Clear the list of failing tests if the plugin is not active.
        session.config.cache.set(STEPWISE_CACHE_DIR, [])


class StepwisePlugin:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.session: Session | None = None
        self.report_status = ""
        assert config.cache is not None
        self.cache: Cache = config.cache
        self.lastfailed: str | None = self.cache.get(STEPWISE_CACHE_DIR, None)
        self.skip: bool = config.getoption("stepwise_skip")

    def pytest_sessionstart(self, session: Session) -> None:
        self.session = session

    def pytest_collection_modifyitems(
        self, config: Config, items: list[nodes.Item]
    ) -> None:
        if not self.lastfailed:
            self.report_status = "no previously failed tests, not skipping."
            return

        # check all item nodes until we find a match on last failed
        failed_index = None
        for index, item in enumerate(items):
            if item.nodeid == self.lastfailed:
                failed_index = index
                break

        # If the previously failed test was not found among the test items,
        # do not skip any tests.
        if failed_index is None:
            self.report_status = "previously failed test not found, not skipping."
        else:
            self.report_status = f"skipping {failed_index} already passed items."
            deselected = items[:failed_index]
            del items[:failed_index]
            config.hook.pytest_deselected(items=deselected)

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if report.failed:
            if self.skip:
                # Remove test from the failed ones (if it exists) and unset the skip option
                # to make sure the following tests will not be skipped.
                if report.nodeid == self.lastfailed:
                    self.lastfailed = None

                self.skip = False
            else:
                # Mark test as the last failing and interrupt the test session.
                self.lastfailed = report.nodeid
                assert self.session is not None
                self.session.shouldstop = (
                    "Test failed, continuing from this test next run."
                )

        else:
            # If the test was actually run and did pass.
            if report.when == "call":
                # Remove test from the failed ones, if exists.
                if report.nodeid == self.lastfailed:
                    self.lastfailed = None

    def pytest_report_collectionfinish(self) -> str | None:
        if self.config.get_verbosity() >= 0 and self.report_status:
            return f"stepwise: {self.report_status}"
        return None

    def pytest_sessionfinish(self) -> None:
        if hasattr(self.config, "workerinput"):
            # Do not update cache if this process is a xdist worker to prevent
            # race conditions (#10641).
            return
        self.cache.set(STEPWISE_CACHE_DIR, self.lastfailed)
