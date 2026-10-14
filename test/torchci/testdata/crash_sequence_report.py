"""Tests scripted across processes through a state directory: test_b crashes the
first process that runs it and passes in the next, test_c always crashes
(TestReportCrashes.test_retry_sequence)."""

import os
import signal
from pathlib import Path


def _attempt(name):
    """How many processes have run name so far, this one included."""
    state = Path(os.environ["CRASH_SEQUENCE_STATE"]) / name
    count = int(state.read_text()) + 1 if state.exists() else 1
    state.write_text(str(count))
    return count


def test_a():
    pass


def test_b():
    if _attempt("test_b") == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def test_c():
    os.kill(os.getpid(), signal.SIGKILL)
