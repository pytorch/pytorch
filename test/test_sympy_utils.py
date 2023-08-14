# -*- coding: utf-8 -*-
# Owner(s): ["oncall: pt2"]

import itertools
import sys

import sympy
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.value_ranges import ValueRangeAnalysis, ValueRanges
from torch.utils._sympy.reference import ReferenceAnalysis
from torch.utils._sympy.interp import sympy_interp


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
BINARY_OPS = ["truediv", "div", "floordiv", "truncdiv", "add", "mul", "sub", "pow", "minimum", "maximum", "mod"]

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
    sys.maxsize - 1,
    sys.maxsize,
]
# less constants for N^2 situations
LESS_CONSTANTS = [-1, 0, 1, 2, 100]


def valid_unary(fn, v):
    if fn == "log" and v <= 0:
        return False
    elif fn == "reciprocal" and v == 0:
        return False
    elif fn == "sqrt" and v < 0:
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
    elif fn == "mod" and b == 0:
        return False
    elif (fn == "div" or fn == "truediv") and b == 0:
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
    @parametrize("dtype", ("int", "float"))
    def test_unary_ref(self, fn, dtype):
        dtype = {"int": sympy.Integer, "float": sympy.Float}[dtype]
        for v in CONSTANTS:
            if not valid_unary(fn, v):
                continue
            with self.subTest(v=v):
                v = dtype(v)
                ref_r = getattr(ReferenceAnalysis, fn)(v)
                r = getattr(ValueRangeAnalysis, fn)(v)
                self.assertEqual(r.lower.is_integer, r.upper.is_integer)
                self.assertEqual(r.lower, r.upper)
                self.assertEqual(ref_r.is_integer, r.upper.is_integer)
                self.assertEqual(ref_r, r.lower)

    def test_pow_half(self):
        ValueRangeAnalysis.pow(ValueRanges.unknown(), ValueRanges.wrap(0.5))

    @parametrize("fn", BINARY_OPS)
    @parametrize("dtype_a", ("int", "float"))
    @parametrize("dtype_b", ("int", "float"))
    def test_binary_ref(self, fn, dtype_a, dtype_b):
        to_dtype = {"int": sympy.Integer, "float": sympy.Float}
        dtype_a = to_dtype[dtype_a]
        dtype_b = to_dtype[dtype_b]
        for a, b in itertools.product(CONSTANTS, repeat=2):
            if not valid_binary(fn, a, b):
                continue
            a = dtype_a(a)
            b = dtype_b(b)
            with self.subTest(a=a, b=b):
                r = getattr(ValueRangeAnalysis, fn)(a, b)
                if r == ValueRanges.unknown():
                    continue
                ref_r = getattr(ReferenceAnalysis, fn)(a, b)

                # sympy.floordiv does 1.0 // 1.0 == 1 rather than 1.0. wtf
                if fn != "floordiv":
                    self.assertEqual(r.lower.is_integer, r.upper.is_integer)
                    self.assertEqual(ref_r.is_integer, r.upper.is_integer)
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
                        if r.is_finite:
                            self.assertIn(r, ref_r)

    def test_rational_bounds(self):
        # Repro from https://github.com/pytorch/pytorch/issues/105097
        from sympy import floor, Eq
        shape_0 = sympy.Symbol('shape_0', positive=True, integer=True)
        new_expr = (
            Eq(30 * floor(4 * (((shape_0 + 1) // 96)) *
                          (((shape_0 + 62017) // (((shape_0 + 1) // 96) + 646))) / 647 +
                          2584 * (((shape_0 + 62017) // (((shape_0 + 1) // 96) + 646))) / 647),
               2880 * floor((((shape_0 + 1) // 96)) *
                            (((shape_0 + 62017) // (((shape_0 + 1) // 96) + 646))) / 15528 +
                            323 * (((shape_0 + 62017) // (((shape_0 + 1) // 96) + 646))) / 7764)))
        new_range_env = {shape_0: ValueRanges(lower=1, upper=190)}
        self.assertTrue(new_expr.subs({shape_0: 95}))
        self.assertIn(True, sympy_interp(ValueRangeAnalysis, new_range_env, new_expr))


class TestSympyInterp(TestCase):
    @parametrize("fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS)
    def test_interp(self, fn):
        # SymPy does not implement truncation for Expressions
        if fn in ("div", "truncdiv", "minimum", "maximum"):
            return

        from sympy.abc import x, y
        vals = CONSTANTS
        if fn in {*UNARY_BOOL_OPS, *BINARY_BOOL_OPS}:
            vals = [True, False]
        arity = 1
        if fn in {*BINARY_OPS, *BINARY_BOOL_OPS, *COMPARE_OPS}:
            arity = 2
        symbols = [x]
        if arity == 2:
            symbols = [x, y]
        for args in itertools.product(vals, repeat=arity):
            if arity == 1 and not valid_unary(fn, *args):
                continue
            elif arity == 2 and not valid_binary(fn, *args):
                continue
            with self.subTest(args=args):
                sargs = [sympy.sympify(a) for a in args]
                sympy_expr = getattr(ReferenceAnalysis, fn)(*symbols)
                ref_r = getattr(ReferenceAnalysis, fn)(*sargs)
                # Yes, I know this is a longwinded way of saying xreplace; the
                # point is to test sympy_interp
                r = sympy_interp(ReferenceAnalysis, dict(zip(symbols, sargs)), sympy_expr)
                self.assertEqual(ref_r, r)


instantiate_parametrized_tests(TestValueRanges)
instantiate_parametrized_tests(TestSympyInterp)


if __name__ == "__main__":
    run_tests()
