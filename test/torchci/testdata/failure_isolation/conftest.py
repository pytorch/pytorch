"""Breaks the report writer at the point FAIL_MODE names (TestReportFailureIsolation)."""

import os

import pytest

from torch.testing._internal.torchci import environment, plugin


def boom(*args, **kwargs):
    raise RuntimeError(os.environ["FAIL_MODE"] + " failed")


class FailingFile:
    def write(self, value):
        raise OSError("write failed")

    def flush(self):
        pass

    def close(self):
        pass


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    mode = os.environ.get("FAIL_MODE")
    if mode == "capture":
        environment.capture = boom
    elif mode == "write":
        plugin.open = lambda *args, **kwargs: FailingFile()
    elif mode == "name":
        plugin.identity = boom
    elif mode == "finish":
        plugin.ReportWriter._finish = boom
    elif mode == "worker" and hasattr(config, "workerinput"):
        plugin.identity = boom
