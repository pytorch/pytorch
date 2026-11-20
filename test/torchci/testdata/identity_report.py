"""Each kind of test name the report writer reads from a pytest item
(TestReportJsonl.test_identities). TestImported is defined in identity_helpers.py,
like the jit/ classes test_jit.py imports."""

import functools
import unittest

import pytest
from identity_helpers import TestImported  # noqa: F401

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)


def test_module_function():
    pass


@pytest.mark.parametrize("value", [1, 2], ids=["a::b", "c[d]"])
def test_pytest_ids(value):
    pass


class TestOuter:
    class TestInner:
        def test_nested(self):
            pass


class TestUnittest(unittest.TestCase):
    def test_unittest(self):
        pass


class TestDevice(TestCase):
    def test_device(self, device):
        pass

    @dtypes(torch.float32, torch.float64)
    def test_dtype(self, device, dtype):
        pass


instantiate_device_type_tests(TestDevice, globals(), only_for="cpu")


@instantiate_parametrized_tests
class TestParametrize(TestCase):
    @parametrize("value", [1])
    def test_parametrize(self, value):
        pass


def wrapped(test):
    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        return test(*args, **kwargs)

    return wrapper


class TestDecorated(TestCase):
    @wrapped
    def test_decorated(self):
        pass


class TestSetattr:
    pass


TestSetattr.test_added = lambda self: None


def create_test_func():
    def test(self, device):
        pass

    return test


class TestFactory(TestCase):
    test_made = create_test_func()


instantiate_device_type_tests(TestFactory, globals(), only_for="cpu")
