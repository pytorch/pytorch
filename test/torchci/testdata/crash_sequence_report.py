"""Tests scripted across processes through a state directory, for run_test.py's
retry loop (TestReportCrashes.test_retry_sequence): test_b fails its first
attempt, kills the process on its in-process rerun and passes in the next
process; test_c always kills the process."""

import os
import signal
from pathlib import Path

import pytest


def _attempt(name):
    """How many times name has started so far, across processes."""
    state = Path(os.environ["CRASH_SEQUENCE_STATE"]) / name
    count = int(state.read_text()) + 1 if state.exists() else 1
    state.write_text(str(count))
    return count


def test_a():
    pass


class TestB:
    @pytest.mark.parametrize("value", [1], ids=["one"])
    def test_b(self, value):
        attempt = _attempt("test_b")
        if attempt == 1:
            raise AssertionError("first attempt fails")
        if attempt == 2:
            os.kill(os.getpid(), signal.SIGKILL)


def test_c():
    os.kill(os.getpid(), signal.SIGKILL)
