# -*- coding: utf-8 -*-
# Owner(s): ["oncall: pt2"]

import itertools
import math

import sympy
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.value_ranges import ValueRangeAnalysis, ValueRanges


UNARY_OPS = [
    "reciprocal",
    "square",
    "abs",
    "neg",
    "exp",
    "log",
    "sqrt",
    "floor",
    "ceil",
]
BINARY_OPS = ["truediv", "div", "add", "mul", "sub", "pow", "minimum", "maximum"]
# a mix of constants, powers of two, primes
CONSTANTS = [
    -1,
    0,
    1,
    2,
    3,
    4,
    5,
    8,
    16,
    32,
    64,
    100,
    101,
    2**24,
    2**32,
    2**37 - 1,
]


# The normal Python interpretation of the operators
# TODO: maybe make this work with sympy?
class ReferenceAnalysis:
    @staticmethod
    def reciprocal(x):
        return 1 / x

    @staticmethod
    def square(x):
        return x * x

    @staticmethod
    def abs(x):
        return abs(x)

    @staticmethod
    def neg(x):
        return -x

    @staticmethod
    def truediv(a, b):
        return a / b

    @staticmethod
    def div(a, b):
        return a // b

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def exp(x):
        return sympy.exp(x)

    @staticmethod
    def log(x):
        return sympy.log(x)

    @staticmethod
    def sqrt(x):
        return sympy.sqrt(x)

    @staticmethod
    def pow(a, b):
        return a**b

    @staticmethod
    def minimum(a, b):
        return min(a, b)

    @staticmethod
    def maximum(a, b):
        return max(a, b)

    @staticmethod
    def floor(x):
        return math.floor(x)

    @staticmethod
    def ceil(x):
        return math.ceil(x)


class TestValueRanges(TestCase):
    @parametrize("fn", UNARY_OPS)
    def test_unary_ref(self, fn):
        for v in CONSTANTS:
            if fn == "log" and v <= 0:
                continue
            if fn == "reciprocal" and v == 0:
                continue
            with self.subTest(v=v):
                ref_r = getattr(ReferenceAnalysis, fn)(sympy.Integer(v))
                r = getattr(ValueRangeAnalysis, fn)(
                    ValueRanges(sympy.Integer(v), sympy.Integer(v))
                )
                self.assertEqual(r.lower, r.upper)
                self.assertEqual(ref_r, r.lower)

    @parametrize("fn", BINARY_OPS)
    def test_binary_ref(self, fn):
        for a, b in itertools.product(CONSTANTS, repeat=2):
            if fn == "pow" and (b > 4 or b == -1 or (a == b == 0)):
                continue
            if (fn == "div" or fn == "truediv") and b == 0:
                continue
            with self.subTest(a=a, b=b):
                ref_r = getattr(ReferenceAnalysis, fn)(
                    sympy.Integer(a), sympy.Integer(b)
                )
                r = getattr(ValueRangeAnalysis, fn)(
                    ValueRanges(sympy.Integer(a), sympy.Integer(a)),
                    ValueRanges(sympy.Integer(b), sympy.Integer(b)),
                )
                self.assertEqual(r.lower, r.upper)
                self.assertEqual(ref_r, r.lower)


instantiate_parametrized_tests(TestValueRanges)


if __name__ == "__main__":
    run_tests()
