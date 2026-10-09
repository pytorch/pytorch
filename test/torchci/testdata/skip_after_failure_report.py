"""A test that fails, then skips in teardown (TestReportJsonl.test_skip_after_failure)."""

import pytest


@pytest.fixture
def skips_in_teardown():
    yield
    pytest.skip("skip in teardown")


def test_fails_then_skips(skips_in_teardown):
    raise AssertionError("the real failure")
