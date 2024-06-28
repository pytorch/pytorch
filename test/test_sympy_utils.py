# Owner(s): ["oncall: pt2"]

import itertools
import math
import sys

import sympy
from typing import Callable, List, Tuple, Type
from torch.testing._internal.common_device_type import skipIf
from torch.testing._internal.common_utils import (
    TEST_Z3,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.solve import INEQUALITY_TYPES, mirror_rel_op, try_solve
from torch.utils._sympy.value_ranges import ValueRangeAnalysis, ValueRanges
from torch.utils._sympy.reference import ReferenceAnalysis, PythonReferenceAnalysis
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._sympy.numbers import int_oo, IntInfinity, NegativeIntInfinity
from sympy.core.relational import is_ge, is_le, is_gt, is_lt
import functools
import torch.fx as fx



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
BINARY_OPS = [
    "truediv", "floordiv",
    # "truncdiv",  # TODO
    # NB: pow is float_pow
    "add", "mul", "sub", "pow", "pow_by_natural", "minimum", "maximum", "mod"
]

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
# SymPy relational types.
RELATIONAL_TYPES = [sympy.Eq, sympy.Ne, sympy.Gt, sympy.Ge, sympy.Lt, sympy.Le]


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
        # sympy will expand to x*x*... for integral b; don't do it if it's big
        b > 4
        # no imaginary numbers
        or a <= 0
        # 0**0 is undefined
        or (a == b == 0)
    ):
        return False
    elif fn == "pow_by_natural" and (
        # sympy will expand to x*x*... for integral b; don't do it if it's big
        b > 4
        or b < 0
        or (a == b == 0)
    ):
        return False
    elif fn == "mod" and (a < 0 or b <= 0):
        return False
    elif (fn in ["div", "truediv", "floordiv"]) and b == 0:
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


class TestNumbers(TestCase):
    def test_int_infinity(self):
        self.assertIsInstance(int_oo, IntInfinity)
        self.assertIsInstance(-int_oo, NegativeIntInfinity)
        self.assertTrue(int_oo.is_integer)
        # is tests here are for singleton-ness, don't use it for comparisons
        # against numbers
        self.assertIs(int_oo + int_oo, int_oo)
        self.assertIs(int_oo + 1, int_oo)
        self.assertIs(int_oo - 1, int_oo)
        self.assertIs(-int_oo - 1, -int_oo)
        self.assertIs(-int_oo + 1, -int_oo)
        self.assertIs(-int_oo + (-int_oo), -int_oo)
        self.assertIs(-int_oo - int_oo, -int_oo)
        self.assertIs(1 + int_oo, int_oo)
        self.assertIs(1 - int_oo, -int_oo)
        self.assertIs(int_oo * int_oo, int_oo)
        self.assertIs(2 * int_oo, int_oo)
        self.assertIs(int_oo * 2, int_oo)
        self.assertIs(-1 * int_oo, -int_oo)
        self.assertIs(-int_oo * int_oo, -int_oo)
        self.assertIs(2 * -int_oo, -int_oo)
        self.assertIs(-int_oo * 2, -int_oo)
        self.assertIs(-1 * -int_oo, int_oo)
        self.assertIs(int_oo / 2, sympy.oo)
        self.assertIs(-(-int_oo), int_oo)  # noqa: B002
        self.assertIs(abs(int_oo), int_oo)
        self.assertIs(abs(-int_oo), int_oo)
        self.assertIs(int_oo ** 2, int_oo)
        self.assertIs((-int_oo) ** 2, int_oo)
        self.assertIs((-int_oo) ** 3, -int_oo)
        self.assertEqual(int_oo ** -1, 0)
        self.assertEqual((-int_oo) ** -1, 0)
        self.assertIs(int_oo ** int_oo, int_oo)
        self.assertTrue(int_oo == int_oo)
        self.assertFalse(int_oo != int_oo)
        self.assertTrue(-int_oo == -int_oo)
        self.assertFalse(int_oo == 2)
        self.assertTrue(int_oo != 2)
        self.assertFalse(int_oo == sys.maxsize)
        self.assertTrue(int_oo >= sys.maxsize)
        self.assertTrue(int_oo >= 2)
        self.assertTrue(int_oo >= -int_oo)

    def test_relation(self):
        self.assertIs(sympy.Add(2, int_oo), int_oo)
        self.assertFalse(-int_oo > 2)

    def test_lt_self(self):
        self.assertFalse(int_oo < int_oo)
        self.assertIs(min(-int_oo, -4), -int_oo)
        self.assertIs(min(-int_oo, -int_oo), -int_oo)

    def test_float_cast(self):
        self.assertEqual(float(int_oo), math.inf)
        self.assertEqual(float(-int_oo), -math.inf)

    def test_mixed_oo_int_oo(self):
        # Arbitrary choice
        self.assertTrue(int_oo < sympy.oo)
        self.assertFalse(int_oo > sympy.oo)
        self.assertTrue(sympy.oo > int_oo)
        self.assertFalse(sympy.oo < int_oo)
        self.assertIs(max(int_oo, sympy.oo), sympy.oo)
        self.assertTrue(-int_oo > -sympy.oo)
        self.assertIs(min(-int_oo, -sympy.oo), -sympy.oo)


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
    @parametrize("dtype", ("int", "float"))
    def test_binary_ref(self, fn, dtype):
        to_dtype = {"int": sympy.Integer, "float": sympy.Float}
        # Don't test float on int only methods
        if dtype == "float" and fn in ["pow_by_natural", "mod"]:
            return
        dtype = to_dtype[dtype]
        for a, b in itertools.product(CONSTANTS, repeat=2):
            if not valid_binary(fn, a, b):
                continue
            a = dtype(a)
            b = dtype(b)
            with self.subTest(a=a, b=b):
                r = getattr(ValueRangeAnalysis, fn)(a, b)
                if r == ValueRanges.unknown():
                    continue
                ref_r = getattr(ReferenceAnalysis, fn)(a, b)

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
        # TODO: bring back sympy.oo testing for float unary fns
        vals = CONSTANTS
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
        # TODO: bring back sympy.oo testing for float unary fns
        vals = LESS_CONSTANTS
        for a, b in itertools.product(generate_range(vals), repeat=2):
            # don't attempt pow on exponents that are too large (but oo is OK)
            if fn == "pow" and b.upper > 4 and b.upper != sympy.oo:
                continue
            with self.subTest(a=a, b=b):
                for a0, b0 in itertools.product(LESS_CONSTANTS, repeat=2):
                    if a0 not in a or b0 not in b:
                        continue
                    if not valid_binary(fn, a0, b0):
                        continue
                    with self.subTest(a0=a0, b0=b0):
                        ref_r = getattr(ValueRangeAnalysis, fn)(a, b)
                        r = getattr(ReferenceAnalysis, fn)(
                            sympy.Integer(a0), sympy.Integer(b0)
                        )
                        if r.is_finite:
                            self.assertIn(r, ref_r)


class TestSympyInterp(TestCase):
    @parametrize("fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS)
    def test_interp(self, fn):
        # SymPy does not implement truncation for Expressions
        if fn in ("div", "truncdiv", "minimum", "maximum", "mod"):
            return

        is_integer = None
        if fn == "pow_by_natural":
            is_integer = True

        x = sympy.Dummy('x', integer=is_integer)
        y = sympy.Dummy('y', integer=is_integer)

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

    @parametrize("fn", UNARY_OPS + BINARY_OPS + UNARY_BOOL_OPS + BINARY_BOOL_OPS + COMPARE_OPS)
    def test_python_interp_fx(self, fn):
        # These never show up from symbolic_shapes
        if fn in ("log", "exp"):
            return

        # Sympy does not support truncation on symbolic shapes
        if fn in ("truncdiv", "mod"):
            return

        vals = CONSTANTS
        if fn in {*UNARY_BOOL_OPS, *BINARY_BOOL_OPS}:
            vals = [True, False]

        arity = 1
        if fn in {*BINARY_OPS, *BINARY_BOOL_OPS, *COMPARE_OPS}:
            arity = 2

        is_integer = None
        if fn == "pow_by_natural":
            is_integer = True

        x = sympy.Dummy('x', integer=is_integer)
        y = sympy.Dummy('y', integer=is_integer)

        symbols = [x]
        if arity == 2:
            symbols = [x, y]

        for args in itertools.product(vals, repeat=arity):
            if arity == 1 and not valid_unary(fn, *args):
                continue
            elif arity == 2 and not valid_binary(fn, *args):
                continue
            if fn == "truncdiv" and args[1] == 0:
                continue
            elif fn in ("pow", "pow_by_natural") and (args[0] == 0 and args[1] <= 0):
                continue
            elif fn == "floordiv" and args[1] == 0:
                continue
            with self.subTest(args=args):
                # Workaround mpf from symbol error
                if fn == "minimum":
                    sympy_expr = sympy.Min(x, y)
                elif fn == "maximum":
                    sympy_expr = sympy.Max(x, y)
                else:
                    sympy_expr = getattr(ReferenceAnalysis, fn)(*symbols)

                if arity == 1:
                    def trace_f(px):
                        return sympy_interp(PythonReferenceAnalysis, {x: px}, sympy_expr)
                else:
                    def trace_f(px, py):
                        return sympy_interp(PythonReferenceAnalysis, {x: px, y: py}, sympy_expr)

                gm = fx.symbolic_trace(trace_f)

                self.assertEqual(
                    sympy_interp(PythonReferenceAnalysis, dict(zip(symbols, args)), sympy_expr),
                    gm(*args)
                )


def type_name_fn(type: Type) -> str:
    return type.__name__

def parametrize_relational_types(*types):
    def wrapper(f: Callable):
        return parametrize("op", types or RELATIONAL_TYPES, name_fn=type_name_fn)(f)
    return wrapper


class TestSympySolve(TestCase):
    def _create_integer_symbols(self) -> List[sympy.Symbol]:
        return sympy.symbols("a b c", integer=True)

    def test_give_up(self):
        from sympy import Eq, Ne

        a, b, c = self._create_integer_symbols()

        cases = [
            # Not a relational operation.
            a + b,
            # 'a' appears on both sides.
            Eq(a, a + 1),
            # 'a' doesn't appear on neither side.
            Eq(b, c + 1),
            # Result is a 'sympy.And'.
            Eq(FloorDiv(a, b), c),
            # Result is a 'sympy.Or'.
            Ne(FloorDiv(a, b), c),
        ]

        for case in cases:
            e = try_solve(case, a)
            self.assertEqual(e, None)

    @parametrize_relational_types()
    def test_noop(self, op):
        a, b, _ = self._create_integer_symbols()

        lhs, rhs = a, 42 * b
        expr = op(lhs, rhs)

        r = try_solve(expr, a)
        self.assertNotEqual(r, None)

        r_expr, r_rhs = r
        self.assertEqual(r_expr, expr)
        self.assertEqual(r_rhs, rhs)

    @parametrize_relational_types()
    def test_noop_rhs(self, op):
        a, b, _ = self._create_integer_symbols()

        lhs, rhs = 42 * b, a

        mirror = mirror_rel_op(op)
        self.assertNotEqual(mirror, None)

        expr = op(lhs, rhs)

        r = try_solve(expr, a)
        self.assertNotEqual(r, None)

        r_expr, r_rhs = r
        self.assertEqual(r_expr, mirror(rhs, lhs))
        self.assertEqual(r_rhs, lhs)

    def _test_cases(self, cases: List[Tuple[sympy.Basic, sympy.Basic]], thing: sympy.Basic, op: Type[sympy.Rel], **kwargs):
        for source, expected in cases:
            r = try_solve(source, thing, **kwargs)

            self.assertTrue(
                (r is None and expected is None)
                or (r is not None and expected is not None)
            )

            if r is not None:
                r_expr, r_rhs = r
                self.assertEqual(r_rhs, expected)
                self.assertEqual(r_expr, op(thing, expected))

    def test_addition(self):
        from sympy import Eq

        a, b, c = self._create_integer_symbols()

        cases = [
            (Eq(a + b, 0), -b),
            (Eq(a + 5, b - 5), b - 10),
            (Eq(a + c * b, 1), 1 - c * b),
        ]

        self._test_cases(cases, a, Eq)

    @parametrize_relational_types(sympy.Eq, sympy.Ne)
    def test_multiplication_division(self, op):
        a, b, c = self._create_integer_symbols()

        cases = [
            (op(a * b, 1), 1 / b),
            (op(a * 5, b - 5), (b - 5) / 5),
            (op(a * b, c), c / b),
        ]

        self._test_cases(cases, a, op)

    @parametrize_relational_types(*INEQUALITY_TYPES)
    def test_multiplication_division_inequality(self, op):
        a, b, _ = self._create_integer_symbols()
        intneg = sympy.Symbol("neg", integer=True, negative=True)
        intpos = sympy.Symbol("pos", integer=True, positive=True)

        cases = [
            # Divide/multiply both sides by positive number.
            (op(a * intpos, 1), 1 / intpos),
            (op(a / (5 * intpos), 1), 5 * intpos),
            (op(a * 5, b - 5), (b - 5) / 5),
            # 'b' is not strictly positive nor negative, so we can't
            # divide/multiply both sides by 'b'.
            (op(a * b, 1), None),
            (op(a / b, 1), None),
            (op(a * b * intpos, 1), None),
        ]

        mirror_cases = [
            # Divide/multiply both sides by negative number.
            (op(a * intneg, 1), 1 / intneg),
            (op(a / (5 * intneg), 1), 5 * intneg),
            (op(a * -5, b - 5), -(b - 5) / 5),
        ]
        mirror_op = mirror_rel_op(op)
        assert mirror_op is not None

        self._test_cases(cases, a, op)
        self._test_cases(mirror_cases, a, mirror_op)

    @parametrize_relational_types()
    def test_floordiv(self, op):
        from sympy import Eq, Ne, Gt, Ge, Lt, Le

        a, b, c = sympy.symbols("a b c")
        pos = sympy.Symbol("pos", positive=True)
        integer = sympy.Symbol("integer", integer=True)

        # (Eq(FloorDiv(a, pos), integer), And(Ge(a, integer * pos), Lt(a, (integer + 1) * pos))),
        # (Eq(FloorDiv(a + 5, pos), integer), And(Ge(a, integer * pos), Lt(a, (integer + 1) * pos))),
        # (Ne(FloorDiv(a, pos), integer), Or(Lt(a, integer * pos), Ge(a, (integer + 1) * pos))),

        special_case = {
            # 'FloorDiv' turns into 'And', which can't be simplified any further.
            Eq: (Eq(FloorDiv(a, pos), integer), None),
            # 'FloorDiv' turns into 'Or', which can't be simplified any further.
            Ne: (Ne(FloorDiv(a, pos), integer), None),
            Gt: (Gt(FloorDiv(a, pos), integer), (integer + 1) * pos),
            Ge: (Ge(FloorDiv(a, pos), integer), integer * pos),
            Lt: (Lt(FloorDiv(a, pos), integer), integer * pos),
            Le: (Le(FloorDiv(a, pos), integer), (integer + 1) * pos),
        }[op]

        cases: List[Tuple[sympy.Basic, sympy.Basic]] = [
            # 'b' is not strictly positive
            (op(FloorDiv(a, b), integer), None),
            # 'c' is not strictly positive
            (op(FloorDiv(a, pos), c), None),
        ]

        # The result might change after 'FloorDiv' transformation.
        # Specifically:
        #   - [Ge, Gt] => Ge
        #   - [Le, Lt] => Lt
        if op in (sympy.Gt, sympy.Ge):
            r_op = sympy.Ge
        elif op in (sympy.Lt, sympy.Le):
            r_op = sympy.Lt
        else:
            r_op = op

        self._test_cases([special_case, *cases], a, r_op)
        self._test_cases([(special_case[0], None), *cases], a, r_op, floordiv_inequality=False)

    def test_floordiv_eq_simplify(self):
        from sympy import Eq, Lt, Le

        a = sympy.Symbol("a", positive=True, integer=True)

        def check(expr, expected):
            r = try_solve(expr, a)
            self.assertNotEqual(r, None)
            r_expr, _ = r
            self.assertEqual(r_expr, expected)

        # (a + 10) // 3 == 3
        # =====================================
        # 3 * 3 <= a + 10         (always true)
        #          a + 10 < 4 * 3 (not sure)
        check(Eq(FloorDiv(a + 10, 3), 3), Lt(a, (3 + 1) * 3 - 10))

        # (a + 10) // 2 == 4
        # =====================================
        # 4 * 2 <= 10 - a         (not sure)
        #          10 - a < 5 * 2 (always true)
        check(Eq(FloorDiv(10 - a, 2), 4), Le(a, -(4 * 2 - 10)))

    @skipIf(not TEST_Z3, "Z3 not installed")
    def test_z3_proof_floordiv_eq_simplify(self):
        import z3
        from sympy import Eq, Lt

        a = sympy.Symbol("a", positive=True, integer=True)
        a_ = z3.Int("a")

        # (a + 10) // 3 == 3
        # =====================================
        # 3 * 3 <= a + 10         (always true)
        #          a + 10 < 4 * 3 (not sure)
        solver = z3.SolverFor("QF_NRA")

        # Add assertions for 'a_'.
        solver.add(a_ > 0)

        expr = Eq(FloorDiv(a + 10, 3), 3)
        r_expr, _ = try_solve(expr, a)

        # Check 'try_solve' really returns the 'expected' below.
        expected = Lt(a, (3 + 1) * 3 - 10)
        self.assertEqual(r_expr, expected)

        # Check whether there is an integer 'a_' such that the
        # equation below is satisfied.
        solver.add(
            # expr
            (z3.ToInt((a_ + 10) / 3.0) == 3)
            !=
            # expected
            (a_ < (3 + 1) * 3 - 10)
        )

        # Assert that there's no such an integer.
        # i.e. the transformation is sound.
        r = solver.check()
        self.assertEqual(r, z3.unsat)

class TestSingletonInt(TestCase):
    def test_basic(self):
        j1 = SingletonInt(1, coeff=1)
        j1_copy = SingletonInt(1, coeff=1)
        j2 = SingletonInt(2, coeff=1)
        j1x2 = SingletonInt(1, coeff=2)

        def test_eq(a, b, expected):
            self.assertEqual(sympy.Eq(a, b), expected)
            self.assertEqual(sympy.Ne(b, a), not expected)

        # eq, ne
        test_eq(j1, j1, True)
        test_eq(j1, j1_copy, True)
        test_eq(j1, j2, False)
        test_eq(j1, j1x2, False)
        test_eq(j1, sympy.Integer(1), False)
        test_eq(j1, sympy.Integer(3), False)

        def test_ineq(a, b, expected, *, strict=True):
            greater = (sympy.Gt, is_gt) if strict else (sympy.Ge, is_ge)
            less = (sympy.Lt, is_lt) if strict else (sympy.Le, is_le)

            if isinstance(expected, bool):
                # expected is always True
                for fn in greater:
                    self.assertEqual(fn(a, b), expected)
                    self.assertEqual(fn(b, a), not expected)
                for fn in less:
                    self.assertEqual(fn(b, a), expected)
                    self.assertEqual(fn(a, b), not expected)
            else:
                for fn in greater:
                    with self.assertRaisesRegex(ValueError, expected):
                        fn(a, b)
                for fn in less:
                    with self.assertRaisesRegex(ValueError, expected):
                        fn(b, a)

        # ge, le, gt, lt
        for strict in (True, False):
            _test_ineq = functools.partial(test_ineq, strict=strict)
            _test_ineq(j1, sympy.Integer(0), True)
            _test_ineq(j1, sympy.Integer(3), "indeterminate")
            _test_ineq(j1, j2, "indeterminate")
            _test_ineq(j1x2, j1, True)

        # Special cases for ge, le, gt, lt:
        for ge in (sympy.Ge, is_ge):
            self.assertTrue(ge(j1, j1))
            self.assertTrue(ge(j1, sympy.Integer(2)))
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                ge(sympy.Integer(2), j1)
        for le in (sympy.Le, is_le):
            self.assertTrue(le(j1, j1))
            self.assertTrue(le(sympy.Integer(2), j1))
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                le(j1, sympy.Integer(2))

        for gt in (sympy.Gt, is_gt):
            self.assertFalse(gt(j1, j1))
            self.assertFalse(gt(sympy.Integer(2), j1))
            # it is only known to be that j1 >= 2, j1 > 2 is indeterminate
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                gt(j1, sympy.Integer(2))

        for lt in (sympy.Lt, is_lt):
            self.assertFalse(lt(j1, j1))
            self.assertFalse(lt(j1, sympy.Integer(2)))
            with self.assertRaisesRegex(ValueError, "indeterminate"):
                lt(sympy.Integer(2), j1)

        # mul
        self.assertEqual(j1 * 2, j1x2)
        # Unfortunately, this doesn't not automatically simplify to 2*j1
        # since sympy.Mul doesn't trigger __mul__ unlike the above.
        self.assertIsInstance(sympy.Mul(j1, 2), sympy.core.mul.Mul)

        with self.assertRaisesRegex(ValueError, "cannot be multiplied"):
            j1 * j2

        self.assertEqual(j1.free_symbols, set())


instantiate_parametrized_tests(TestValueRanges)
instantiate_parametrized_tests(TestSympyInterp)
instantiate_parametrized_tests(TestSympySolve)


if __name__ == "__main__":
    run_tests()
