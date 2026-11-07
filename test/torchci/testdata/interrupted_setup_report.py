"""A fixture interrupted as a timeout's SIGINT would interrupt it
(TestReportCrashes.test_interrupt_leaves_the_run_in_flight)."""

import pytest


@pytest.fixture
def interrupted():
    raise KeyboardInterrupt


def test_interrupted(interrupted):
    pass
