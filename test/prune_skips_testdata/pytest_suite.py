# Fixture suite for TestPruneSkipsAtCollection in test/test_testing.py.
import unittest

import pytest

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import lazy_skip_if, run_tests, TestCase


class TestPruneFixture(TestCase):
    def test_runs(self):
        pass

    @unittest.skip("static mark")
    def test_static_mark(self):
        pass

    @lazy_skip_if(lambda: True, "runtime condition")
    def test_lazy_skip(self):
        pass

    def test_body_skip(self):
        self.skipTest("decided in the body")


class TestSetupDependentSkip(TestCase):
    ready = False

    def setUp(self):
        super().setUp()
        type(self).ready = True

    @lazy_skip_if(lambda: not TestSetupDependentSkip.ready, "initialized by setUp")
    def test_runs_after_setup(self):
        self.assertTrue(self.ready)


class TestOverriddenSetup(TestCase):
    def setUp(self):
        # This class does not use TestCase.setUp's slow/fast policy.
        self.ready = True

    def test_runs(self):
        self.assertTrue(self.ready)


class TestPruneFixtureDevice(TestCase):
    @onlyCUDA
    def test_cuda_only(self, device):
        pass

    def test_any_device(self, device):
        pass


instantiate_device_type_tests(TestPruneFixtureDevice, globals(), only_for="cpu")


@pytest.mark.skip(reason="pytest mark")
def test_pytest_mark():
    pass


skip_dynamic = True


class DeferredSkip:
    def __bool__(self):
        return skip_dynamic


def test_enable_dynamic_conditions():
    global skip_dynamic
    skip_dynamic = False


@pytest.mark.skipif("skip_dynamic", reason="runtime string condition")
def test_pytest_dynamic_string():
    pass


@pytest.mark.skipif(DeferredSkip(), reason="runtime object condition")
def test_pytest_dynamic_object():
    pass


@pytest.mark.skipif(True, reason="static boolean condition")
def test_pytest_static_condition():
    pass


class XfailCondition:
    def __bool__(self):
        pytest.xfail("runtime condition takes precedence over the unittest skip")


@pytest.mark.skipif(XfailCondition(), reason="runtime condition")
@unittest.skip("inner unittest skip")
def test_pytest_condition_before_unittest_skip():
    pass


@pytest.mark.xfail(run=False, reason="xfail takes precedence over the unittest skip")
@unittest.skip("inner unittest skip")
def test_pytest_xfail_before_unittest_skip():
    pass


if __name__ == "__main__":
    run_tests()
