"""A test that kills its xdist worker, between passing tests and a failing last
test (TestReportCrashes.test_xdist_worker_crash)."""

import os


def test_before():
    pass


def test_crash():
    os._exit(1)


def test_after():
    pass


def test_fail_last():
    raise AssertionError("expected failure")
