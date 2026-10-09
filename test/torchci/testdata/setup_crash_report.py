"""A fixture that kills its xdist worker before the setup report
(TestReportCrashes.test_xdist_worker_crash_in_setup)."""

import os

import pytest


@pytest.fixture
def dies():
    os._exit(1)


class TestSetup:
    @pytest.mark.parametrize("value", [1], ids=["one"])
    def test_setup_crash(self, dies, value):
        pass
