"""A fixture interrupted the way a timeout's SIGINT would interrupt it
(TestReportCrashes.test_interrupt_in_setup_stays_in_flight)."""

import pytest


@pytest.fixture
def interrupted():
    raise KeyboardInterrupt


def test_interrupted(interrupted):
    pass
