# Owner(s): ["module: dynamo"]

""" Test printing of scalar types.

"""
import functools

from unittest import skipIf as skipif

import pytest

import torch._numpy as np
from torch._numpy.testing import assert_
from torch.testing._internal.common_utils import run_tests, TestCase

skip = functools.partial(skipif, True)


class A:
    pass


class B(A, np.float64):
    pass


class C(B):
    pass


class D(C, B):
    pass


class B0(np.float64, A):
    pass


class C0(B0):
    pass


class HasNew:
    def __new__(cls, *args, **kwargs):
        return cls, args, kwargs


class B1(np.float64, HasNew):
    pass


@skip(reason="scalar repr: numpy plans to make it more explicit")
class TestInherit(TestCase):
    def test_init(self):
        x = B(1.0)
        assert_(str(x) == "1.0")
        y = C(2.0)
        assert_(str(y) == "2.0")
        z = D(3.0)
        assert_(str(z) == "3.0")

    def test_init2(self):
        x = B0(1.0)
        assert_(str(x) == "1.0")
        y = C0(2.0)
        assert_(str(y) == "2.0")

    def test_gh_15395(self):
        # HasNew is the second base, so `np.float64` should have priority
        x = B1(1.0)
        assert_(str(x) == "1.0")

        # previously caused RecursionError!?
        with pytest.raises(TypeError):
            B1(1.0, 2.0)


if __name__ == "__main__":
    run_tests()
