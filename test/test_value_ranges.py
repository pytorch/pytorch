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

UNARY_BOOL_OPS = ["not_"]
BINARY_BOOL_OPS = ["or_", "and_"]
COMPARE_OPS = ["eq", "ne", "lt", "gt", "le", "ge"]

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
# less constants for N^2 situations
LESS_CONSTANTS = [-1, 0, 1, 2, 100]


# The normal Python interpretation of the operators
# NB: For magic methods this needs to use normal magic methods
# so that test_magic_methods works
class ReferenceAnalysis:
    @staticmethod
    def or_(a, b):
        assert not isinstance(a, bool) and not isinstance(b, bool)
        return a | b

    @staticmethod
    def and_(a, b):
        assert not isinstance(a, bool) and not isinstance(b, bool)
        return a & b

    @staticmethod
    def eq(a, b):
        if isinstance(a, sympy.Expr) or isinstance(b, sympy.Expr):
            return sympy.Eq(a, b)
        return a == b

    @classmethod
    def ne(cls, a, b):
        return cls.not_(cls.eq(a, b))

    @staticmethod
    def lt(a, b):
        return a < b

    @staticmethod
    def gt(a, b):
        return a > b

    @staticmethod
    def le(a, b):
        return a <= b

    @staticmethod
    def ge(a, b):
        return a >= b

    @staticmethod
    def not_(a):
        assert not isinstance(a, bool)
        return ~a

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


def valid_unary(fn, v):
    if fn == "log" and v <= 0:
        return False
    if fn == "reciprocal" and v == 0:
        return False
    if fn == "sqrt" and v < 0:
        return False
    return True


def valid_binary(fn, a, b):
    if fn == "pow" and (
        b > 4
        or (  # sympy will expand to x*x*... for integral b; don't do it if it's big
            a <= 0 and b == -1
        )
        or (a == b == 0)  # no imaginary numbers  # 0**0 is undefined
    ):
        return False
    if (fn == "div" or fn == "truediv") and b == 0:
        return False
    return True


def generate_range(vals):
    for a1, a2 in itertools.product(vals, repeat=2):
        if a1 in [sympy.true, sympy.false]:
            if a1 == sympy.true and a2 == sympy.false:
                continue
        else:
            if a1 > a2:
                continue
        # ranges that only admit infinite values are not interesting
        if a1 == sympy.oo or a2 == -sympy.oo:
            continue
        yield ValueRanges(a1, a2)


class TestValueRanges(TestCase):
    @parametrize("fn", UNARY_OPS)
    def test_unary_ref(self, fn):
        for v in CONSTANTS:
            if not valid_unary(fn, v):
                continue
            with self.subTest(v=v):
                ref_r = getattr(ReferenceAnalysis, fn)(sympy.Integer(v))
                r = getattr(ValueRangeAnalysis, fn)(ValueRanges.wrap(v))
                self.assertEqual(r.lower, r.upper)
                self.assertEqual(ref_r, r.lower)

    @parametrize("fn", BINARY_OPS)
    def test_binary_ref(self, fn):
        for a, b in itertools.product(CONSTANTS, repeat=2):
            if not valid_binary(fn, a, b):
                continue
            with self.subTest(a=a, b=b):
                ref_r = getattr(ReferenceAnalysis, fn)(
                    sympy.Integer(a), sympy.Integer(b)
                )
                r = getattr(ValueRangeAnalysis, fn)(
                    ValueRanges.wrap(a),
                    ValueRanges.wrap(b),
                )
                self.assertEqual(r.lower, r.upper)
                self.assertEqual(ref_r, r.lower)

    def test_mul_zero_unknown(self):
        self.assertEqual(
            ValueRangeAnalysis.mul(ValueRanges.wrap(0), ValueRanges.unknown()),
            ValueRanges.wrap(0),
        )

    @parametrize("fn", UNARY_BOOL_OPS)
    def test_unary_bool_ref_range(self, fn):
        vals = [sympy.false, sympy.true]
        for a in generate_range(vals):
            with self.subTest(a=a):
                ref_r = getattr(ValueRangeAnalysis, fn)(a)
                unique = set()
                for a0 in vals:
                    if a0 not in a:
                        continue
                    with self.subTest(a0=a0):
                        r = getattr(ReferenceAnalysis, fn)(a0)
                        self.assertIn(r, ref_r)
                        unique.add(r)
                if ref_r.lower == ref_r.upper:
                    self.assertEqual(len(unique), 1)
                else:
                    self.assertEqual(len(unique), 2)

    @parametrize("fn", BINARY_BOOL_OPS)
    def test_binary_bool_ref_range(self, fn):
        vals = [sympy.false, sympy.true]
        for a, b in itertools.product(generate_range(vals), repeat=2):
            with self.subTest(a=a, b=b):
                ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                unique = set()
                for a0, b0 in itertools.product(vals, repeat=2):
                    if a0 not in a or b0 not in b:
                        continue
                    with self.subTest(a0=a0, b0=b0):
                        r = getattr(ReferenceAnalysis, fn)(a0, b0)
                        self.assertIn(r, ref_r)
                        unique.add(r)
                if ref_r.lower == ref_r.upper:
                    self.assertEqual(len(unique), 1)
                else:
                    self.assertEqual(len(unique), 2)

    @parametrize("fn", UNARY_OPS)
    def test_unary_ref_range(self, fn):
        vals = [-sympy.oo, *CONSTANTS, sympy.oo]
        for a in generate_range(vals):
            with self.subTest(a=a):
                ref_r = getattr(ValueRangeAnalysis, fn)(a)
                for a0 in CONSTANTS:
                    if a0 not in a:
                        continue
                    if not valid_unary(fn, a0):
                        continue
                    with self.subTest(a0=a0):
                        r = getattr(ReferenceAnalysis, fn)(sympy.Integer(a0))
                        self.assertIn(r, ref_r)

    # This takes about 4s for all the variants
    @parametrize("fn", BINARY_OPS + COMPARE_OPS)
    def test_binary_ref_range(self, fn):
        vals = [-sympy.oo, *LESS_CONSTANTS, sympy.oo]
        for a, b in itertools.product(generate_range(vals), repeat=2):
            # don't attempt pow on exponents that are too large (but oo is OK)
            if fn == "pow" and b.upper > 4 and b.upper != sympy.oo:
                continue
            with self.subTest(a=a, b=b):
                ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                for a0, b0 in itertools.product(LESS_CONSTANTS, repeat=2):
                    if a0 not in a or b0 not in b:
                        continue
                    if not valid_binary(fn, a0, b0):
                        continue
                    with self.subTest(a0=a0, b0=b0):
                        r = getattr(ReferenceAnalysis, fn)(
                            sympy.Integer(a0), sympy.Integer(b0)
                        )
                        self.assertIn(r, ref_r)


instantiate_parametrized_tests(TestValueRanges)


if __name__ == "__main__":
    run_tests()
