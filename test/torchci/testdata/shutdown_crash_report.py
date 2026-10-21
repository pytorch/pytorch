"""A process that dies at interpreter shutdown, after its tests
(TestReportCrashes.test_shutdown_crash_records_nothing)."""

import atexit
import os
import signal


atexit.register(os.kill, os.getpid(), signal.SIGKILL)


def test_only():
    pass
