# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_numeric_tower.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

# test interactions between int, float, Decimal and Fraction

import unittest
import random
import math
import sys
import operator

from decimal import Decimal as D
from fractions import Fraction as F

# Constants related to the hash implementation;  hash(x) is based
# on the reduction of x modulo the prime _PyHASH_MODULUS.
_PyHASH_MODULUS = sys.hash_info.modulus
_PyHASH_INF = sys.hash_info.inf


class DummyIntegral(int):
    """Dummy Integral class to test conversion of the Rational to float."""

    def __mul__(self, other):
        return DummyIntegral(super().__mul__(other))
    __rmul__ = __mul__

    def __truediv__(self, other):
        return NotImplemented
    __rtruediv__ = __truediv__

    @property
    def numerator(self):
        return DummyIntegral(self)

    @property
    def denominator(self):
        return DummyIntegral(1)


class HashTest(__TestCase):
    def check_equal_hash(self, x, y):
        # check both that x and y are equal and that their hashes are equal
        self.assertEqual(hash(x), hash(y),
                         "got different hashes for {!r} and {!r}".format(x, y))
        self.assertEqual(x, y)

    def test_bools(self):
        self.check_equal_hash(False, 0)
        self.check_equal_hash(True, 1)

    def test_integers(self):
        # check that equal values hash equal

        # exact integers
        for i in range(-1000, 1000):
            self.check_equal_hash(i, float(i))
            self.check_equal_hash(i, D(i))
            self.check_equal_hash(i, F(i))

        # the current hash is based on reduction modulo 2**n-1 for some
        # n, so pay special attention to numbers of the form 2**n and 2**n-1.
        for i in range(100):
            n = 2**i - 1
            if n == int(float(n)):
                self.check_equal_hash(n, float(n))
                self.check_equal_hash(-n, -float(n))
            self.check_equal_hash(n, D(n))
            self.check_equal_hash(n, F(n))
            self.check_equal_hash(-n, D(-n))
            self.check_equal_hash(-n, F(-n))

            n = 2**i
            self.check_equal_hash(n, float(n))
            self.check_equal_hash(-n, -float(n))
            self.check_equal_hash(n, D(n))
            self.check_equal_hash(n, F(n))
            self.check_equal_hash(-n, D(-n))
            self.check_equal_hash(-n, F(-n))

        # random values of various sizes
        for _ in range(1000):
            e = random.randrange(300)
            n = random.randrange(-10**e, 10**e)
            self.check_equal_hash(n, D(n))
            self.check_equal_hash(n, F(n))
            if n == int(float(n)):
                self.check_equal_hash(n, float(n))

    def test_binary_floats(self):
        # check that floats hash equal to corresponding Fractions and Decimals

        # floats that are distinct but numerically equal should hash the same
        self.check_equal_hash(0.0, -0.0)

        # zeros
        self.check_equal_hash(0.0, D(0))
        self.check_equal_hash(-0.0, D(0))
        self.check_equal_hash(-0.0, D('-0.0'))
        self.check_equal_hash(0.0, F(0))

        # infinities and nans
        self.check_equal_hash(float('inf'), D('inf'))
        self.check_equal_hash(float('-inf'), D('-inf'))

        for _ in range(1000):
            x = random.random() * math.exp(random.random()*200.0 - 100.0)
            self.check_equal_hash(x, D.from_float(x))
            self.check_equal_hash(x, F.from_float(x))

    def test_complex(self):
        # complex numbers with zero imaginary part should hash equal to
        # the corresponding float

        test_values = [0.0, -0.0, 1.0, -1.0, 0.40625, -5136.5,
                       float('inf'), float('-inf')]

        for zero in -0.0, 0.0:
            for value in test_values:
                self.check_equal_hash(value, complex(value, zero))

    def test_decimals(self):
        # check that Decimal instances that have different representations
        # but equal values give the same hash
        zeros = ['0', '-0', '0.0', '-0.0e10', '000e-10']
        for zero in zeros:
            self.check_equal_hash(D(zero), D(0))

        self.check_equal_hash(D('1.00'), D(1))
        self.check_equal_hash(D('1.00000'), D(1))
        self.check_equal_hash(D('-1.00'), D(-1))
        self.check_equal_hash(D('-1.00000'), D(-1))
        self.check_equal_hash(D('123e2'), D(12300))
        self.check_equal_hash(D('1230e1'), D(12300))
        self.check_equal_hash(D('12300'), D(12300))
        self.check_equal_hash(D('12300.0'), D(12300))
        self.check_equal_hash(D('12300.00'), D(12300))
        self.check_equal_hash(D('12300.000'), D(12300))

    def test_fractions(self):
        # check special case for fractions where either the numerator
        # or the denominator is a multiple of _PyHASH_MODULUS
        self.assertEqual(hash(F(1, _PyHASH_MODULUS)), _PyHASH_INF)
        self.assertEqual(hash(F(-1, 3*_PyHASH_MODULUS)), -_PyHASH_INF)
        self.assertEqual(hash(F(7*_PyHASH_MODULUS, 1)), 0)
        self.assertEqual(hash(F(-_PyHASH_MODULUS, 1)), 0)

        # The numbers ABC doesn't enforce that the "true" division
        # of integers produces a float.  This tests that the
        # Rational.__float__() method has required type conversions.
        x = F._from_coprime_ints(DummyIntegral(1), DummyIntegral(2))
        self.assertRaises(TypeError, lambda: x.numerator/x.denominator)
        self.assertEqual(float(x), 0.5)

    def test_hash_normalization(self):
        # Test for a bug encountered while changing long_hash.
        #
        # Given objects x and y, it should be possible for y's
        # __hash__ method to return hash(x) in order to ensure that
        # hash(x) == hash(y).  But hash(x) is not exactly equal to the
        # result of x.__hash__(): there's some internal normalization
        # to make sure that the result fits in a C long, and is not
        # equal to the invalid hash value -1.  This internal
        # normalization must therefore not change the result of
        # hash(x) for any x.

        class HalibutProxy:
            def __hash__(self):
                return hash('halibut')
            def __eq__(self, other):
                return other == 'halibut'

        x = {'halibut', HalibutProxy()}
        self.assertEqual(len(x), 1)

class ComparisonTest(__TestCase):
    def test_mixed_comparisons(self):

        # ordered list of distinct test values of various types:
        # int, float, Fraction, Decimal
        test_values = [
            float('-inf'),
            D('-1e425000000'),
            -1e308,
            F(-22, 7),
            -3.14,
            -2,
            0.0,
            1e-320,
            True,
            F('1.2'),
            D('1.3'),
            float('1.4'),
            F(275807, 195025),
            D('1.414213562373095048801688724'),
            F(114243, 80782),
            F(473596569, 84615),
            7e200,
            D('infinity'),
            ]
        for i, first in enumerate(test_values):
            for second in test_values[i+1:]:
                self.assertLess(first, second)
                self.assertLessEqual(first, second)
                self.assertGreater(second, first)
                self.assertGreaterEqual(second, first)

    def test_complex(self):
        # comparisons with complex are special:  equality and inequality
        # comparisons should always succeed, but order comparisons should
        # raise TypeError.
        z = 1.0 + 0j
        w = -3.14 + 2.7j

        for v in 1, 1.0, F(1), D(1), complex(1):
            self.assertEqual(z, v)
            self.assertEqual(v, z)

        for v in 2, 2.0, F(2), D(2), complex(2):
            self.assertNotEqual(z, v)
            self.assertNotEqual(v, z)
            self.assertNotEqual(w, v)
            self.assertNotEqual(v, w)

        for v in (1, 1.0, F(1), D(1), complex(1),
                  2, 2.0, F(2), D(2), complex(2), w):
            for op in operator.le, operator.lt, operator.ge, operator.gt:
                self.assertRaises(TypeError, op, z, v)
                self.assertRaises(TypeError, op, v, z)


if __name__ == '__main__':
    run_tests()
