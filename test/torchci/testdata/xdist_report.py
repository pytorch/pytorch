"""Tests run under xdist with -x (TestReportJsonl.test_xdist_maxfail): one kills
its worker, and the last one fails."""

import os


def test_before():
    pass


def test_crash():
    os._exit(1)


def test_after():
    pass


def test_fail_last():
    raise AssertionError("expected failure")
