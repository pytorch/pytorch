"""A test that fails once and then kills its process, after a passing test
(TestReportCrashes.test_crash_under_run_test)."""

import itertools
import os
import signal

import pytest


_attempts = itertools.count(1)


def test_before():
    pass


class TestCrash:
    @pytest.mark.parametrize("value", [1], ids=["one"])
    def test_crash(self, value):
        if next(_attempts) == 1:
            raise AssertionError("first attempt fails")
        os.kill(os.getpid(), signal.SIGKILL)
