# Owner(s): ["module: dynamic shapes"]
import contextlib
import copy
import enum
import inspect
import operator
import os
import pickle
import random
import re
import subprocess
import sys
import types
from unittest import mock

import sympy
from sympy.core.assumptions import _assume_defined, _assume_rules

import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental import proxy_tensor, sym_node, symbolic_shapes
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.sym_node import _NO_HINT, SymNode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import (
    BitwiseFn_bitwise_and,
    BitwiseFn_bitwise_or,
    BitwiseFn_bitwise_xor,
    CeilDiv,
    CeilToInt,
    CleanDiv,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    Identity,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    LShift,
    Max,
    Min,
    Mod,
    ModularIndexing,
    PowByNatural,
    PythonMod,
    RoundDecimal,
    RoundToInt,
    RShift,
    ToFloat,
    TruncToFloat,
    TruncToInt,
)
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import (
    _default_symbol_range,
    _rewrite_for_value_range_analysis,
    bound_sympy,
    SymPyValueRangeAnalysis,
    ValueRangeError,
    ValueRanges,
)


NativeUnsupported = torch._C._symbolic.NativeUnsupported
_NativeSymNode = torch._C._symbolic._NativeSymNode

s0 = sympy.Symbol("s0", integer=True, positive=True)
s1 = sympy.Symbol("s1", integer=True, positive=True)
u0 = sympy.Symbol("u0", integer=True)
zf = sympy.Symbol("zf", real=True, positive=True)
LEAVES = [s0, s1, u0, zf, *map(sympy.Integer, range(-3, 4))]
LEAVES += [sympy.Rational(1, 2), sympy.Rational(-2, 3)]
FACTS = sorted(_assume_defined)
# Order of the C++ Fact enum.
NATIVE_FACT_ORDER = [
    "commutative",
    "integer",
    "noninteger",
    "rational",
    "irrational",
    "real",
    "extended_real",
    "finite",
    "infinite",
    "zero",
    "nonzero",
    "positive",
    "negative",
    "nonnegative",
    "nonpositive",
    "extended_positive",
    "extended_negative",
    "extended_nonnegative",
    "extended_nonpositive",
    "extended_nonzero",
    "even",
    "odd",
    "prime",
    "composite",
    "algebraic",
    "transcendental",
    "complex",
    "imaginary",
    "hermitian",
    "antihermitian",
    "polar",
]


def sort_facts_natively(facts):
    facts.sort(key=NATIVE_FACT_ORDER.index)


W = sympy.Symbol("w", complex=True, real=False)
FACT_LEAVES = [
    s0,
    s1,
    u0,
    zf,
    W,
    sympy.Symbol("e", even=True),
    sympy.Symbol("o", odd=True),
    sympy.Symbol("n", integer=True, nonnegative=True),
    sympy.Symbol("m", integer=True, negative=True),
    sympy.Symbol("r", real=True),
    sympy.Symbol("i", imaginary=True),
    sympy.Symbol("c", complex=True),
    sympy.Symbol("q", rational=True, nonzero=True),
    sympy.Symbol("t", irrational=True),
    sympy.Symbol("p", prime=True),
    *map(sympy.Integer, [-2, -1, 2, 3, 4, 6]),
    sympy.Rational(1, 2),
    sympy.Rational(-3, 4),
]


# Expression trees are nested tuples (op, *children); leaves are sympy atoms.
def sympy_eval(t):
    if not isinstance(t, tuple):
        return t
    op, *args = t
    args = [sympy_eval(a) for a in args]
    if op == "add":
        return sympy.Add(*args)
    if op == "mul":
        return sympy.Mul(*args)
    if op == "sub":
        return args[0] - args[1]
    if op == "neg":
        return -args[0]
    return args[0] ** args[1]


def native_eval(arena, t):
    if not isinstance(t, tuple):
        return arena.from_sympy(t)
    op, *args = t
    if op == "pow":
        return arena.pow(native_eval(arena, args[0]), arena.integer(args[1]))
    args = [native_eval(arena, a) for a in args]
    if op in ("add", "mul"):
        return getattr(arena, op)(args)
    return getattr(arena, op)(*args)


def subtrees(t):
    yield t
    if isinstance(t, tuple):
        for a in t[1:]:
            yield from subtrees(a)


def random_tree(rng, depth, leaves=LEAVES):
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(leaves)
    op = rng.choice(["add", "mul", "sub", "neg", "pow"])
    if op == "pow":
        return (op, random_tree(rng, depth - 1, leaves), rng.randint(-2, 3))
    if op == "neg":
        return (op, random_tree(rng, depth - 1, leaves))
    n = rng.randint(2, 3) if op in ("add", "mul") else 2
    return (op, *(random_tree(rng, depth - 1, leaves) for _ in range(n)))


class TestNativeSymNodeFlag(TestCase):
    def _read_flag(self, env_value):
        env = dict(os.environ)
        env.pop("CPP_SYMNODE", None)
        if env_value is not None:
            env["CPP_SYMNODE"] = env_value
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "import torch; print(torch._dynamo.config.use_cpp_symnode)",
            ],
            env=env,
            text=True,
        )
        return out.strip().splitlines()[-1]

    def test_flag_env(self):
        self.assertEqual(self._read_flag(None), "False")
        self.assertEqual(self._read_flag("0"), "False")
        self.assertEqual(self._read_flag("1"), "True")


class TestNativeExpr(TestCase):
    def check(self, t):
        arena = torch._C._symbolic._Arena()
        expected = sympy_eval(t)
        n = native_eval(arena, t)
        got = arena.to_sympy(n)
        self.assertTrue(got == expected, f"{t}: native {got} != sympy {expected}")
        self.assertEqual(str(got), str(expected))
        self.assertEqual(arena.from_sympy(expected), n)
        native_args = tuple(arena.to_sympy(a) for a in arena.args(n))
        self.assertEqual(native_args, sympy.sympify(expected).args)

    def test_edge_cases(self):
        x1 = ("add", s0, 1)
        cases = [
            ("pow", ("mul", 2, ("pow", x1, -1)), -1),
            ("mul", 2, x1),
            ("mul", sympy.Rational(1, 2), ("add", ("mul", 2, s0), 2)),
            ("mul", s0, ("pow", s0, -1)),
            ("mul", 0, s0),
            ("pow", ("mul", s0, s1), 2),
            ("pow", ("mul", s0, s1), -2),
            ("pow", ("mul", sympy.Rational(2, 3), s0), -2),
            ("pow", ("neg", s0), 3),
            ("pow", ("mul", -2, s0), 2),
            ("pow", ("pow", s0, 2), -3),
            ("pow", ("sub", 1, s0), 2),
            ("pow", ("neg", x1), 3),
            ("sub", x1, x1),
            ("sub", ("mul", 3, s0, s1), ("mul", s1, s0)),
            (
                "add",
                ("mul", sympy.Rational(1, 2), s0),
                ("mul", sympy.Rational(1, 2), s0),
            ),
            ("mul", ("pow", s0, 2), ("pow", s0, -2), 3),
            ("mul", x1, ("pow", x1, -1)),
            ("mul", x1, x1),
            ("mul", -1, ("add", s0, s1, u0)),
            ("add", ("mul", 2, s0, zf), ("mul", zf, s0), u0, -4),
            ("pow", 0, 0),
            ("pow", sympy.Rational(-2, 3), -3),
            ("add", 1, -1),
            ("mul", 1),
            ("add", 0),
            ("add",),
            ("mul",),
        ]
        for t in cases:
            self.check(t)

    @parametrize("seed", range(10))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        checked = 0
        for _ in range(300):
            t = random_tree(rng, 4)
            try:
                native_eval(torch._C._symbolic._Arena(), t)
            except NativeUnsupported:
                # Only division by zero (zoo) is outside the domain here.
                values = [sympy.sympify(sympy_eval(st)) for st in subtrees(t)]
                self.assertTrue(any(v.has(sympy.zoo) for v in values))
                continue
            self.check(t)
            checked += 1
        self.assertGreater(checked, 250)

    def test_args_order(self):
        # Same-name symbols order by their assumptions, like Basic.compare.
        leaves = [
            s0,
            u0,
            zf,
            sympy.Symbol("s0", integer=True),
            sympy.Symbol("a", positive=True),
            sympy.Symbol("a", real=True),
            sympy.Symbol("B", integer=True),
            sympy.Symbol("b", integer=True),
            sympy.Symbol("s", integer=True),
            sympy.Symbol("p", positive=True),
            sympy.Symbol("p", negative=True),
            sympy.Integer(2),
            sympy.Integer(-1),
            sympy.Rational(1, 2),
        ]
        rng = random.Random(0)
        for _ in range(300):
            t = random_tree(rng, 3, leaves)
            try:
                native_eval(torch._C._symbolic._Arena(), t)
            except NativeUnsupported:
                continue
            self.check(t)

    def test_atoms(self):
        arena = torch._C._symbolic._Arena()
        for v in [
            int_oo,
            -int_oo,
            sympy.Rational(-7, 4),
            sympy.Integer(-(2**63)),
            s0,
            zf,
        ]:
            n = arena.from_sympy(v)
            self.assertIs(arena.to_sympy(n), arena.to_sympy(arena.from_sympy(v)))
            self.assertTrue(arena.to_sympy(n) == v)
        self.assertEqual(arena.from_sympy(int_oo).kind, "IntInfinity")
        self.assertEqual(arena.from_sympy(5), arena.integer(5))
        self.assertEqual(arena.rational(4, -6), arena.from_sympy(sympy.Rational(-2, 3)))
        self.assertEqual(arena.rational(4, 2), arena.integer(2))
        # Same name, different assumptions: distinct symbols, like sympy.
        s0_int = sympy.Symbol("s0", integer=True)
        self.assertNotEqual(arena.from_sympy(s0_int), arena.from_sympy(s0))
        self.assertIs(arena.to_sympy(arena.from_sympy(s0_int)), s0_int)
        x_finite = sympy.Symbol("x", finite=True)
        x_complex = sympy.Symbol("x", complex=True)
        self.assertNotEqual(arena.from_sympy(x_finite), arena.from_sympy(x_complex))

    def test_unsupported(self):
        arena = torch._C._symbolic._Arena()
        big = arena.integer(2**62)
        cases = [
            lambda: arena.integer(2**63),
            lambda: arena.from_sympy(sympy.Integer(2**70)),
            lambda: arena.add([big, big]),
            lambda: arena.mul([big, arena.integer(4)]),
            lambda: arena.pow(arena.integer(2), arena.integer(64)),
            lambda: arena.pow(arena.integer(0), arena.integer(-1)),
            lambda: arena.pow(arena.from_sympy(s0), arena.rational(1, 2)),
            lambda: arena.add([arena.from_sympy(int_oo), big]),
            lambda: arena.rational(1, 0),
            lambda: arena.from_sympy(sympy.Dummy("d", integer=True)),
            lambda: arena.from_sympy(sympy.Symbol("x")),
            lambda: arena.from_sympy(sympy.Float(1.5, 30)),
            lambda: arena.from_sympy(sympy.floor(zf)),
            lambda: arena.from_sympy(sympy.Symbol("z", zero=True)),
            lambda: arena.from_sympy(sympy.Mul(2, s0 + 1, evaluate=False)),
            lambda: arena.rational(2**64, 3),
            lambda: arena.as_numer_denom(arena.from_sympy(-(2**63) * s0 + s1)),
            lambda: arena.ask(arena.from_sympy(W**2), "extended_real"),
        ]
        for f in cases:
            with self.assertRaises(NativeUnsupported):
                f()
        other = torch._C._symbolic._Arena()
        with self.assertRaisesRegex(RuntimeError, "different arena"):
            arena.add([big, other.integer(1)])


class TestNativeAssumptions(TestCase):
    def test_assume_rules(self):
        imp, beta, triggers, prereq = torch._C._symbolic._assume_rules()

        def nonempty(d):
            return {k: set(v) for k, v in d.items() if v}

        self.assertEqual(nonempty(imp), nonempty(_assume_rules.full_implications))
        self.assertEqual(beta, [(set(c), i) for c, i in _assume_rules.beta_rules])
        self.assertEqual(nonempty(triggers), nonempty(_assume_rules.beta_triggers))
        self.assertEqual(nonempty(prereq), nonempty(_assume_rules.prereq))
        self.assertTrue(all(b < len(beta) for t in triggers.values() for b in t))

    def check_facts(self, v, rng):
        arena = torch._C._symbolic._Arena()
        n = arena.from_sympy(v)
        facts = list(FACTS)
        rng.shuffle(facts)
        for f in facts:
            self.assertIs(arena.ask(n, f), getattr(v, "is_" + f), f"{v}.is_{f}")

    def test_atom_facts(self):
        values = [sympy.Integer(i) for i in range(-10, 40)]
        values += map(
            sympy.Integer,
            [
                2**61 - 1,
                2**63 - 25,
                2**63 - 1,
                -(2**63),
                3**39,
                (10**9 + 7) * 998244353,
                3215031751,
                3825123056546413051,
            ],
        )
        values += [sympy.Rational(p, q) for p, q in [(1, 2), (-1, 2), (7, 3), (-2, 3)]]
        values += [int_oo, -int_oo, s0, u0, zf]
        values += [
            sympy.Symbol("n", integer=True, nonnegative=True),
            sympy.Symbol("e", even=True),
            sympy.Symbol("p", prime=True),
            sympy.Symbol("c", composite=True),
            sympy.Symbol("r", real=True),
            sympy.Symbol("x", finite=True),
            sympy.Symbol("q", rational=True, nonzero=True),
        ]
        rng = random.Random(0)
        for v in values:
            for _ in range(3):
                self.check_facts(v, rng)


class TestNativeCompoundAssumptions(TestCase):
    def compare_facts(self, v, rng):
        """Asks every fact of v natively in random order; returns how many were
        answered and how many raised NativeUnsupported."""
        arena = torch._C._symbolic._Arena()
        n = arena.from_sympy(v)
        facts = list(FACTS)
        rng.shuffle(facts)
        answered = unsupported = 0
        for f in facts:
            try:
                got = arena.ask(n, f)
            except NativeUnsupported:
                unsupported += 1
                continue
            self.assertIs(got, getattr(v, "is_" + f), f"({v}).is_{f}")
            answered += 1
        return answered, unsupported

    def test_known_answers(self):
        e, o, i = (
            sympy.Symbol(x, **{k: True})
            for x, k in [("e", "even"), ("o", "odd"), ("i", "imaginary")]
        )
        cases = [
            (2 * s0, "even", True),
            (s0 * s1, "integer", True),
            (s0**2, "positive", True),
            (s0**-1, "positive", True),
            (-s0, "negative", True),
            (e / 2, "integer", True),
            (e / 3, "integer", False),
            (o / 2, "integer", False),
            (o * o, "odd", True),
            (i**2, "extended_real", True),
            (i**3, "imaginary", True),
            (i * zf, "imaginary", True),
            (s0 * zf, "positive", True),
            (s0 + 1, "integer", True),
            (s0 + zf, "real", True),
            (e + o, "odd", True),
            (e + 2, "odd", False),
            (sympy.Symbol("t", irrational=True) + 1, "irrational", True),
            (zf**-2, "finite", True),
            (u0**-1, "zero", False),
            (s0 + 1, "positive", True),
            (s0 - 1, "nonnegative", True),
            (1 - s0, "nonpositive", True),
            (s0 - s1, "positive", None),
            (s0**2 - s0, "nonnegative", None),
            (s0**2 - s0, "extended_nonnegative", None),
            (zf - 1, "extended_negative", None),
            (s0 * s1 - 1, "nonnegative", True),
            (1 / (s0 + 1) + 2, "positive", True),
        ]
        for v, f, expected in cases:
            arena = torch._C._symbolic._Arena()
            self.assertIs(arena.ask(arena.from_sympy(v), f), expected, f"({v}).is_{f}")
            self.assertIs(getattr(v, "is_" + f), expected)

    @parametrize("seed", range(8))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        answered = unsupported = 0
        for _ in range(150):
            t = random_tree(rng, 3, FACT_LEAVES)
            try:
                v = sympy.sympify(sympy_eval(t))
            except ZeroDivisionError:
                continue
            if v.has(sympy.zoo, sympy.nan):
                continue
            a, u = self.compare_facts(v, rng)
            answered += a
            unsupported += u
        self.assertGreater(answered, 10 * unsupported)


def undummy(e):
    """Replaces each Dummy by a Symbol with its name and assumptions so results
    built from fresh Dummies can be compared with ==."""
    from sympy.core.exprtools import _eps

    reps = {d: sympy.Symbol(d.name, **d.assumptions0) for d in e.atoms(sympy.Dummy)}
    reps[_eps] = sympy.Symbol("_eps", positive=True)
    return e.xreplace(reps)


class TestNativeExprTools(TestCase):
    SIGN_LEAVES = [
        s0,
        s1,
        zf,
        sympy.Symbol("n", integer=True, nonnegative=True),
        sympy.Symbol("m", integer=True, negative=True),
        sympy.Symbol("p", prime=True),
        sympy.Symbol("e", even=True, positive=True),
        *map(sympy.Integer, [-2, -1, 1, 2, 3]),
        sympy.Rational(1, 2),
    ]

    def test_monotonic_sign_known(self):
        from sympy.core.exprtools import _monotonic_sign

        n = sympy.Symbol("n", integer=True, nonnegative=True)
        cases = [n + 1, s0 - 1, n * s0 + 1, s0 * s1 + 1, n - 1, s0**2 - s0]
        cases += [s0**2 + s0, 1 / (s0 + 1), -s0, zf, zf - 1, 2 * s0 + 3 * s1 - 4]
        for v in cases:
            arena = torch._C._symbolic._Arena()
            got = arena.monotonic_sign(arena.from_sympy(v))
            expected = _monotonic_sign(v)
            got = None if got is None else undummy(arena.to_sympy(got))
            self.assertEqual(
                got, None if expected is None else undummy(expected), f"{v}"
            )

    @parametrize("seed", range(4))
    def test_fuzz(self, seed):
        from sympy.core.exprtools import _monotonic_sign

        rng = random.Random(seed)
        answered = unsupported = 0
        for _ in range(200):
            v = sympy.sympify(sympy_eval(random_tree(rng, 3, self.SIGN_LEAVES)))
            if v.has(sympy.zoo, sympy.nan):
                continue
            arena = torch._C._symbolic._Arena()
            n = arena.from_sympy(v)
            self.assertEqual(arena.is_polynomial(n), v.is_polynomial(), f"{v}")
            try:
                nd = [arena.to_sympy(x) for x in arena.as_numer_denom(n)]
                self.assertEqual(tuple(nd), v.as_numer_denom(), f"{v}")
            except NativeUnsupported:
                pass
            for x in v.free_symbols:
                d = arena.diff(n, arena.from_sympy(x))
                self.assertEqual(arena.to_sympy(d), v.diff(x), f"d({v})/d{x}")
            try:
                got = arena.monotonic_sign(n)
            except NativeUnsupported:
                unsupported += 1
                continue
            answered += 1
            expected = _monotonic_sign(v)
            got = None if got is None else undummy(arena.to_sympy(got))
            self.assertEqual(
                got, None if expected is None else undummy(expected), f"{v}"
            )
        self.assertGreater(answered, 10 * unsupported)


RELATIONS = ["Eq", "Ne", "Lt", "Le", "Gt", "Ge"]


class TestNativeRelational(TestCase):
    def check_rel(self, op, a, b):
        """Returns False if native raised NativeUnsupported, else checks the
        result against sympy."""
        arena = torch._C._symbolic._Arena()
        try:
            expected = getattr(sympy, op)(a, b)
        except TypeError:
            with self.assertRaises(NativeUnsupported, msg=f"{op}({a}, {b})"):
                arena.rel(op, arena.from_sympy(a), arena.from_sympy(b))
            return True
        try:
            r = arena.rel(op, arena.from_sympy(a), arena.from_sympy(b))
        except NativeUnsupported:
            return False
        got = arena.to_sympy(r)
        self.assertEqual(got, expected, f"{op}({a}, {b})")
        self.assertEqual(type(got), type(expected), f"{op}({a}, {b})")
        self.assertIs(arena.from_sympy(expected).id, r.id)
        return True

    def test_numbers(self):
        nums = [int_oo, -int_oo, *map(sympy.Integer, [-3, -1, 0, 1, 5])]
        nums += [sympy.Rational(1, 2), sympy.Rational(-7, 3)]
        for op in RELATIONS:
            for a in nums:
                for b in nums:
                    self.assertTrue(self.check_rel(op, a, b), f"{op}({a}, {b})")

    def test_float_numbers(self):
        nums = [int_oo, -int_oo, *map(sympy.Integer, [-3, 0, 1, 2**60 + 1])]
        nums += [sympy.Rational(1, 2), sympy.Rational(1, 10), sympy.Rational(-7, 3)]
        nums += [sympy.Float(v) for v in [0.5, 0.1, -2.0, 0.0, 1.0, 2.0**60, -1e-300]]
        for op in RELATIONS:
            for a in nums:
                for b in nums:
                    answered = self.check_rel(op, a, b)
                    if not (a.is_infinite or b.is_infinite):
                        self.assertTrue(answered, f"{op}({a}, {b})")

    def test_float_known(self):
        f = sympy.Float
        cases = [(s0, f(0.5)), (s0, f(1.0)), (u0, f(0.5)), (u0, f(2.0)), (zf, f(-1.5))]
        cases += [(s0 + f(0.5), 1), (f(0.5) * s0, 1), (f(2.0) * u0, 3)]
        cases += [(u0 + f(0.5), u0), (zf + f(0.5), zf), (f(0.5) * s0, s0 / 2)]
        cases += [(s0 + f(0.0), s0), (s0, f(0.0))]
        cases += [(u0 + f(2.0), u0 + 2), (f(0.5) * zf, zf / 2), (s0 * f(-1.0), -s0)]
        answered = 0
        for op in RELATIONS:
            for a, b in cases:
                for l, r in ((a, b), (b, a)):
                    answered += self.check_rel(op, sympy.sympify(l), sympy.sympify(r))
        self.assertGreater(answered, 150)

    def test_known(self):
        u = sympy.Symbol("u", integer=True)
        x = sympy.Symbol("x")
        eq = sympy.Eq(s0, 1, evaluate=False)
        cases = [(s0, 0), (s0, 1), (s0 + 1, 1), (u, 0), (s0, s1), (s0, s0)]
        cases += [(zf, -1), (u * u, -1), (s0 - 1, -1), (1 / s0, 0), (x, 1)]
        cases += [(s0, sympy.I), (W, 1), (2 * s0, 1), (u, sympy.Rational(1, 2))]
        cases += [(int_oo, s0), (-int_oo, u), (eq, 3), (eq, s0), (eq, s0 + 1)]
        cases += [(sympy.true, s0), (sympy.true, 3), (sympy.true, sympy.false)]
        cases += [(sympy.true, sympy.true), (eq, eq), (eq, sympy.true)]
        for op in RELATIONS:
            for a, b in cases:
                for l, r in ((a, b), (b, a)):
                    self.check_rel(op, sympy.sympify(l), sympy.sympify(r))

    def test_unevaluated(self):
        arena = torch._C._symbolic._Arena()
        one, n = arena.integer(1), arena.from_sympy(s0)
        for op in RELATIONS:
            r = arena.rel(op, one, n, evaluate=False)
            self.assertEqual(r.kind, op)
            self.assertEqual(
                arena.to_sympy(r), getattr(sympy, op)(1, s0, evaluate=False)
            )
            self.assertIs(arena.from_sympy(arena.to_sympy(r)).id, r.id)

    def test_derived(self):
        u = sympy.Symbol("u", integer=True)
        exprs = [(s0, 1), (u, s0 + 1), (2, u), (u, sympy.true), (int_oo, u)]
        for op in RELATIONS:
            for a, b in exprs:
                r = getattr(sympy, op)(a, b, evaluate=False)
                arena = torch._C._symbolic._Arena()
                n = arena.from_sympy(r)
                for name in ["reversed", "reversedsign", "negated", "weak", "strict"]:
                    try:
                        expected = getattr(r, name)
                    except TypeError:
                        with self.assertRaises(NativeUnsupported):
                            getattr(arena, name)(n)
                        continue
                    try:
                        got = arena.to_sympy(getattr(arena, name)(n))
                    except NativeUnsupported:
                        continue
                    self.assertEqual(got, expected, f"({r}).{name}")

    def test_not(self):
        u = sympy.Symbol("u", integer=True)
        vals = [sympy.true, sympy.false, sympy.Integer(0), sympy.Integer(2)]
        vals += [sympy.Rational(1, 2), int_oo, -int_oo, s0 + 1, u]
        vals += [sympy.Eq(s0, 1, evaluate=False), sympy.Lt(u, 2, evaluate=False)]
        vals += [sympy.Not(s0 + 1)]
        for v in vals:
            arena = torch._C._symbolic._Arena()
            got = arena.to_sympy(arena.logical_not(arena.from_sympy(v)))
            self.assertEqual(got, sympy.Not(v), f"Not({v})")

    def test_booleans_have_no_facts(self):
        arena = torch._C._symbolic._Arena()
        eq = arena.from_sympy(sympy.Eq(s0, 1, evaluate=False))
        for e in [arena.boolean(True), eq, arena.logical_not(arena.from_sympy(s0))]:
            for f in FACTS:
                self.assertIsNone(arena.ask(e, f), f)
            with self.assertRaises(NativeUnsupported):
                arena.add([e, arena.integer(1)])

    @parametrize("seed", range(6))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        answered = unsupported = 0
        for _ in range(100):
            a, b = (sympy_eval(random_tree(rng, 2, FACT_LEAVES)) for _ in range(2))
            a, b = sympy.sympify(a), sympy.sympify(b)
            if a.has(sympy.zoo, sympy.nan) or b.has(sympy.zoo, sympy.nan):
                continue
            for op in RELATIONS:
                if self.check_rel(op, a, b):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 10 * unsupported)

    @parametrize("seed", range(4))
    def test_float_fuzz(self, seed):
        rng = random.Random(seed)
        answered = unsupported = 0
        for _ in range(100):
            leaves = FACT_LEAVES + FLOAT_LEAVES
            a, b = (sympy_eval(random_tree(rng, 2, leaves)) for _ in range(2))
            a, b = sympy.sympify(a), sympy.sympify(b)
            if a.has(sympy.zoo, sympy.nan) or b.has(sympy.zoo, sympy.nan):
                continue
            for op in RELATIONS:
                if self.check_rel(op, a, b):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 3 * unsupported)


class TestNativeSorting(TestCase):
    def check_expr(self, v):
        arena = torch._C._symbolic._Arena()
        n = arena.from_sympy(v)
        self.assertTrue(arena.sort_key(n) == v.sort_key(), f"sort_key({v})")
        for name in ["as_ordered_terms", "as_ordered_factors"]:
            got = [arena.to_sympy(x) for x in getattr(arena, name)(n)]
            self.assertEqual(got, getattr(v, name)(), f"({v}).{name}")
        got = arena.could_extract_minus_sign(n)
        self.assertEqual(got, v.could_extract_minus_sign(), f"{v}")

    def check_ordered(self, xs):
        arena = torch._C._symbolic._Arena()
        got = arena.ordered([arena.from_sympy(x) for x in xs])
        self.assertEqual([arena.to_sympy(x) for x in got], list(sympy.ordered(xs)))

    def check_canonical(self, r):
        arena = torch._C._symbolic._Arena()
        n = arena.from_sympy(r)
        got = arena.to_sympy(arena.canonical(n))
        self.assertEqual(got, r.canonical, f"({r}).canonical")
        self.assertEqual(type(got), type(r.canonical), f"({r}).canonical")

    def test_known(self):
        u = sympy.Symbol("u", integer=True)
        cases = [1 - s0, s0 - 1, 2 - 3 * s0, s0**2 * s1 - s0 * s1**3 + 2 + s0]
        cases += [1 / s0 + 1 + s0, -2 * s0 * s1, s0 / 2, -s0 * s1, (s0 + 1) ** 2]
        cases += [(s0 + 1) ** -2 * s1, -((s0 + 1) ** 3), u - s0, s0 - u, -u - s0]
        cases += [2 * s0 * (s1 + 1), s0**2 + s0 * s1 + s1**2, 1 / s0 - s1]
        cases += [int_oo, -int_oo, sympy.Integer(-3), sympy.Rational(-1, 2)]
        for v in cases:
            self.check_expr(v)

    def test_shared_subexpressions(self):
        v = s0
        for _ in range(11):
            v = (v + 1) ** 2 * s1 + v
        self.check_expr(v)

    def test_boolean_keys(self):
        eq = sympy.Eq(s0, 1, evaluate=False)
        lt = sympy.Lt(u0, s1, evaluate=False)
        items = [sympy.true, sympy.false, eq, lt, sympy.Not(s0), s0, u0, int_oo]
        items += [-int_oo, sympy.Integer(0), sympy.Integer(-1), sympy.Integer(7)]
        items += [sympy.Rational(1, 2), sympy.Rational(3, 2), s0 + 1, 2 * s0]
        items += [sympy.Eq(eq, sympy.true, evaluate=False), sympy.Ge(s1, s0)]
        items += [sympy.Ne(s0, s1), sympy.Gt(s0, 2), sympy.Le(u0, 3), s0**2]
        arena = torch._C._symbolic._Arena()
        natives = [arena.from_sympy(x) for x in items]
        for a, na in zip(items, natives):
            self.assertTrue(arena.sort_key(na) == a.sort_key(), f"sort_key({a})")
            for b, nb in zip(items, natives):
                self.assertEqual(arena.compare(na, nb), a.compare(b), f"{a}, {b}")
        self.check_ordered(items)
        self.check_ordered(items[::-1])

    def test_canonical_known(self):
        u = sympy.Symbol("u", integer=True)
        pairs = [(s0, 1), (1, s0), (-s0, 3), (s0, -s1), (-s0, -s1), (s1, s0)]
        pairs += [(s0 - s1, 0), (1 - s0, s1), (-2 * u, s0), (3, 2), (int_oo, s0)]
        pairs += [(s0, int_oo), (-int_oo, int_oo), (int_oo, int_oo), (u, -u)]
        pairs += [(s0, -s1 + 1), (s1 - s0, s0 - s1), (-s0 - s1, 2)]
        eq = sympy.Eq(s0, 1, evaluate=False)
        for op in RELATIONS:
            for a, b in pairs:
                self.check_canonical(getattr(sympy, op)(a, b, evaluate=False))
        for b in [sympy.true, sympy.false, eq, sympy.Eq(-s0, 1, evaluate=False)]:
            for op in ["Eq", "Ne"]:
                self.check_canonical(getattr(sympy, op)(eq, b, evaluate=False))
                self.check_canonical(getattr(sympy, op)(b, eq, evaluate=False))
        inner = sympy.Eq(1, s0, evaluate=False)
        self.check_canonical(sympy.Eq(inner, eq, evaluate=False))
        self.check_canonical(sympy.Ne(inner, sympy.true, evaluate=False))

    def test_float_canonical(self):
        f, h = sympy.Float, sympy.Rational(1, 2)
        pairs = [(s0, f(0.5)), (f(0.5), s0), (-s0, f(0.5)), (f(0.5), h), (h, f(0.5))]
        pairs += [(f(0.1), sympy.Rational(1, 10)), (sympy.Rational(1, 10), f(0.1))]
        pairs += [(f(2.0), 1), (f(0.5) * s0, s1), (s0 + f(0.5), -s1)]
        pairs += [(f(-0.5) * s0, s1)]
        pairs += [(s1, f(2.0) * s0), (-s0 - f(1.5), s1), (f(0.5) * s0, -f(0.5) * s1)]
        for op in RELATIONS:
            for a, b in pairs:
                self.check_canonical(getattr(sympy, op)(a, b, evaluate=False))

    @parametrize("seed", range(6))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        pool = []
        for _ in range(120):
            v = sympy.sympify(sympy_eval(random_tree(rng, 3, FACT_LEAVES)))
            if v.has(sympy.zoo, sympy.nan):
                continue
            self.check_expr(v)
            pool.append(v)
        for _ in range(40):
            self.check_ordered(rng.sample(pool, rng.randint(2, 8)))
        for _ in range(120):
            a, b = rng.sample(pool, 2)
            self.check_canonical(
                getattr(sympy, rng.choice(RELATIONS))(a, b, evaluate=False)
            )


class TestNativeLattice(TestCase):
    # as_Boolean turns nonzero symbols (s0, zf, a, b) into true; u0 stays.
    a = sympy.Symbol("a", integer=True, nonzero=True)
    b = sympy.Symbol("b", integer=True, nonzero=True)
    v = sympy.Symbol("v", integer=True, nonzero=True)
    vp = sympy.Symbol("v", integer=True, positive=True)
    rels = [
        sympy.Lt(a, b, evaluate=False),
        sympy.Le(b, a, evaluate=False),
        sympy.Gt(b, a, evaluate=False),
        sympy.Ge(a, b, evaluate=False),
        sympy.Eq(a, 1, evaluate=False),
        sympy.Ne(1, a, evaluate=False),
        sympy.Lt(-a, 2, evaluate=False),
        sympy.Ge(a, -2, evaluate=False),
        sympy.Eq(s0, b + 1, evaluate=False),
        sympy.Ne(b, s0 - 1, evaluate=False),
    ]
    atoms = [
        sympy.Not(s0 + 1),
        sympy.Not(a * b),
        sympy.Lt(v, 2, evaluate=False),
        sympy.Lt(vp, 2, evaluate=False),
        s0,
        zf,
        u0,
        sympy.true,
        sympy.false,
        *map(sympy.Integer, [0, 1, 2]),
        sympy.Rational(1, 2),
        int_oo,
        s0 + 1,
    ]

    def check(self, op, args):
        """Returns False if native raised NativeUnsupported, else checks the
        result against sympy."""
        arena = torch._C._symbolic._Arena()
        fn = getattr(arena, f"logical_{op.lower()}")
        try:
            expected = getattr(sympy, op)(*args)
        except (TypeError, AttributeError):
            with self.assertRaises(NativeUnsupported, msg=f"{op}{args}"):
                fn([arena.from_sympy(a) for a in args])
            return True
        try:
            r = fn([arena.from_sympy(a) for a in args])
        except NativeUnsupported:
            return False
        got = arena.to_sympy(r)
        self.assertEqual(got, expected, f"{op}{args}")
        self.assertEqual(type(got), type(expected), f"{op}{args}")
        self.assertEqual(got.args, expected.args, f"{op}{args}")
        self.assertIs(arena.from_sympy(expected).id, r.id)
        return True

    def test_known(self):
        lt, le, gt, ge, eq, ne = self.rels[:6]
        y, x, lv, lvp, p = self.atoms[:5]
        cases = [(), (y,), (y, y), (y, x), (y, p), (y, zf), (p,), (zf,), (lv, x)]
        cases += [(lt, le), (lt, gt), (lt, ge), (eq, ne), (lt, gt, le), (y, lt)]
        cases += [(sympy.true, y), (sympy.false, y), (y, sympy.Not(y)), (y, lv)]
        cases += [(u0, lt), (u0, sympy.Not(u0)), (u0, u0), (u0, s0), (u0, sympy.false)]
        cases += [(sympy.Integer(1), y), (sympy.Integer(0), y), (s0 + 1, y)]
        cases += [(sympy.Integer(2), sympy.false), (int_oo, sympy.true)]
        cases += [(sympy.And(lt, y), le), (sympy.Or(lt, y), le)]
        cases += [(sympy.And(y, x), sympy.Or(y, x)), (sympy.And(lt, y), x)]
        cases += [(sympy.Or(lt, eq), sympy.Or(y, ne)), (sympy.Eq(eq, sympy.true),)]
        cases += [(sympy.Eq(sympy.Eq(1, u0, evaluate=False), eq, evaluate=False), lt)]
        for args in cases:
            for op in ["And", "Or"]:
                self.assertTrue(self.check(op, args), f"{op}{args}")
                self.assertTrue(self.check(op, args[::-1]), f"{op}{args[::-1]}")
        # The args tie on sort_key, so sympy's order depends on hashes.
        for op in ["And", "Or"]:
            self.assertFalse(self.check(op, (lv, lvp)))

    def test_or_checks_relationals_before_flattening(self):
        lt, le = self.rels[:2]
        y = self.atoms[0]
        self.assertTrue(self.check("Or", (sympy.Or(lt, y), le)))
        self.assertEqual(set(sympy.Or(sympy.Or(lt, y), le).args), {y, le, lt})
        self.assertTrue(self.check("And", (sympy.And(lt, y), le)))
        self.assertIs(sympy.And(sympy.And(lt, y), le), sympy.false)

    def test_keys(self):
        lt, le, gt, ge, eq, ne = self.rels[:6]
        y, x = self.atoms[:2]
        items = [sympy.And(lt, y), sympy.Or(lt, y), sympy.And(y, x), sympy.Or(y, x)]
        items += [sympy.Or(sympy.Or(lt, y), le), sympy.Not(sympy.And(y, x)), lt]
        items += [sympy.And(eq, sympy.Or(ne, y)), y, sympy.true, sympy.Not(y)]
        arena = torch._C._symbolic._Arena()
        natives = [arena.from_sympy(v) for v in items]
        for a, na in zip(items, natives):
            self.assertEqual(arena.to_sympy(na), a)
            self.assertTrue(arena.sort_key(na) == a.sort_key(), f"sort_key({a})")
            for f in FACTS:
                self.assertEqual(arena.ask(na, f), getattr(a, f"is_{f}"), f"{a}: {f}")
            for b, nb in zip(items, natives):
                self.assertEqual(arena.compare(na, nb), a.compare(b), f"{a}, {b}")
        got = arena.ordered(natives)
        self.assertEqual([arena.to_sympy(v) for v in got], list(sympy.ordered(items)))
        n = arena.logical_not(natives[0])
        self.assertEqual(arena.to_sympy(n), sympy.Not(items[0]))

    @parametrize("seed", range(6))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        pool = self.rels + self.atoms
        answered = unsupported = 0
        for _ in range(300):
            op = rng.choice(["And", "Or"])
            args = rng.choices(pool, k=rng.randint(0, 5))
            if not self.check(op, args):
                unsupported += 1
                continue
            answered += 1
            try:
                v = getattr(sympy, op)(*args)
            except (TypeError, AttributeError):
                continue
            if isinstance(v, (sympy.And, sympy.Or)) and len(pool) < 60:
                pool.append(v)
        self.assertGreater(answered, 10 * unsupported)


class TestNativePrinter(TestCase):
    def check(self, v):
        """Returns False if native raised NativeUnsupported, else checks
        arena.sstr against str."""
        arena = torch._C._symbolic._Arena()
        try:
            got = arena.sstr(arena.from_sympy(v))
        except NativeUnsupported:
            return False
        self.assertEqual(got, str(v))
        return True

    def test_known(self):
        u = sympy.Symbol("u", integer=True)
        cases = [s0 / s1, -s0 / s1, s0 * u0 / s1, s0 / (2 * s1), 3 * s0 / 2, -s0 / 2]
        cases += [1 / (s0 + 1), (s0 + 1) / (s1 + 2), s1 * (s0 + 1) ** 2, s0 * (s1 + 1)]
        cases += [-s0 * (s1 + 1), 1 - s0, 2 - s0 / 2, 2 - 3 * s0, 3 - s0 * u0, u - 1]
        cases += [s0**2 - 2 * s0 * s1 + s1**2, s0 / s1 + s1, s0**-2, -(s0**-2)]
        cases += [(s0 + 1) ** -2 * s1, -3 * s0**-2 / 2, s0**-1 * s1**-1, -s0 - s1]
        cases += [-((s0 + 1) ** 3), (s0 + 1) ** -1, 1 / (s0 * s1), s0 * s1 / 6 - 1]
        cases += [sympy.Symbol("s10", integer=True) + sympy.Symbol("s2", integer=True)]
        cases += [int_oo, -int_oo, sympy.Integer(-3), sympy.Integer(0)]
        cases += [sympy.Rational(-1, 2), sympy.Rational(7, 2), s0, sympy.true]
        cases += [sympy.false, zf / 3 + 1, -zf * u0 / 5, u0 * (s0 - 1) * (s1 + 1)]
        for v in cases:
            self.assertTrue(self.check(v), f"{v}")

    def test_booleans(self):
        eq = sympy.Eq(s0, 1, evaluate=False)
        lt = sympy.Lt(u0, s1, evaluate=False)
        cases = [eq, lt, sympy.Ne(s0 + 1, -s1), sympy.Ge(u0, s1), sympy.Gt(2, u0)]
        cases += [sympy.Lt(s0 + 1, 2 * s1), sympy.Le(-s0, 3), sympy.Lt(-3, u0)]
        cases += [sympy.Lt(u0 / 2, sympy.Rational(-1, 2)), sympy.Lt(u0, -int_oo)]
        cases += [sympy.Eq(eq, sympy.true, evaluate=False), sympy.Not(s0 + 1)]
        cases += [sympy.Not(u0), sympy.Not(lt), sympy.Not(eq), sympy.And(eq, lt)]
        cases += [sympy.Or(eq, lt), sympy.And(sympy.Or(eq, lt), u0), sympy.Or(u0, zf)]
        cases += [sympy.Or(sympy.And(eq, lt), sympy.Not(u0)), sympy.Ne(u0, 2) | ~lt]
        cases += [sympy.Lt(eq, lt, evaluate=False), sympy.Eq(lt, eq, evaluate=False)]
        for v in cases:
            self.assertTrue(self.check(v), f"{v}")

    @parametrize("seed", range(6))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        pool = []
        answered = unsupported = 0
        for _ in range(80):
            t = random_tree(rng, 4, FACT_LEAVES if seed % 2 else LEAVES)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                pool.append(v)
                if self.check(v):
                    answered += 1
                else:
                    unsupported += 1
        bools = []
        for _ in range(150):
            a, b = rng.sample(pool, 2)
            r = getattr(sympy, rng.choice(RELATIONS))(a, b, evaluate=False)
            bools.append(rng.choice([r, sympy.Not(r)]))
        bools += TestNativeLattice.rels + TestNativeLattice.atoms
        for _ in range(150):
            args = rng.sample(bools, rng.randint(1, 4))
            try:
                v = getattr(sympy, rng.choice(["And", "Or"]))(*args)
            except (TypeError, AttributeError):
                continue
            bools.append(v)
        for v in bools:
            if self.check(v):
                answered += 1
            else:
                unsupported += 1
        self.assertGreater(answered, 20 * unsupported)


FLOATS = [0.5, -2.0, 0.1, 1.0, 0.0, 1e-3, 3.25, -0.3, 1e20, 1e-5]
FLOAT_LEAVES = [sympy.Float(v) for v in FLOATS]


class TestNativeFloat(TestCase):
    def check_expr(self, v):
        arena = torch._C._symbolic._Arena()
        n = arena.from_sympy(v)
        got = arena.to_sympy(n)
        self.assertTrue(got == v and type(got) is type(v), f"{got} != {v}")
        self.assertEqual(arena.sstr(n), str(v))
        self.assertTrue(arena.sort_key(n) == v.sort_key(), f"sort_key({v})")
        got = [arena.to_sympy(x) for x in arena.as_ordered_terms(n)]
        self.assertEqual(got, v.as_ordered_terms(), f"({v}).as_ordered_terms")
        got = arena.could_extract_minus_sign(n)
        self.assertEqual(got, v.could_extract_minus_sign(), f"{v}")
        for f in FACTS:
            self.assertIs(arena.ask(n, f), getattr(v, "is_" + f), f"{v}.is_{f}")

    def test_numbers(self):
        values = FLOATS + [5e-324, 1e300, -1e-300, 123456789012345678.0, 1e15, 1e16]
        values += [0.3333333333333333, 1.5e-5, 99999999999999.99, 2.0**53, -7.5]
        values += [1e-4, 12345.678, 2.5e-7, 1.7976931348623157e308, 4.0, -0.0]
        for v in values:
            self.check_expr(sympy.Float(v))
        # Doubles of a precision-53 Float round-trip, others fall back.
        arena = torch._C._symbolic._Arena()
        for v in [sympy.Float("0.1", 30), sympy.Float("1e400"), sympy.Float("1e-400")]:
            with self.assertRaises(NativeUnsupported):
                arena.from_sympy(v)

    def test_known(self):
        f = sympy.Float
        cases = [f(0.5) * s0, f(1.0) * s0, 2 * (f(0.5) * s0 + s1), s0 + f(0.0)]
        cases += [f(0.5) * s0 + f(1.5) + s1, -f(2.0) * s0 - 1, (f(2.0) * s0) ** -1]
        cases += [f(-0.5) * s0 * s1, s0 - f(0.5), s0 * (s1 + f(0.5)), f(0.1) + s0]
        cases += [(f(0.5) * s0 + 1) ** 2, f(1e20) * s0 + f(1e-5) * zf, f(0.5) * u0]
        cases += [sympy.Rational(1, 3) * (f(0.5) * s0), f(0.1) * s0 + f(0.2) * s0]
        cases += [f(3.0) * zf / 7, f(2.0) ** 2 * s0, s0 / f(-4.0), f(0.0) * s0 + 1]
        cases += [sympy.Mul(f(1.0), sympy.Rational(1, 3)) + s0, f(-1.0) * s0]
        for v in cases:
            self.check_expr(v)

    @parametrize("seed", range(6))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        leaves = LEAVES + FLOAT_LEAVES
        checked = 0
        for _ in range(300):
            t = random_tree(rng, 4, leaves)
            arena = torch._C._symbolic._Arena()
            try:
                n = native_eval(arena, t)
            except NativeUnsupported:
                continue
            expected = sympy.sympify(sympy_eval(t))
            got = arena.to_sympy(n)
            self.assertTrue(got == expected, f"{t}: native {got} != sympy {expected}")
            self.assertEqual(arena.from_sympy(expected), n)
            native_args = tuple(arena.to_sympy(a) for a in arena.args(n))
            self.assertEqual(native_args, expected.args)
            with contextlib.suppress(NativeUnsupported):
                self.assertEqual(arena.sstr(n), str(expected))
                self.assertTrue(arena.sort_key(n) == expected.sort_key(), f"{t}")
            for f in FACTS:
                try:
                    got = arena.ask(n, f)
                except NativeUnsupported:
                    continue
                self.assertIs(got, getattr(expected, "is_" + f), f"{t}.is_{f}")
            checked += 1
        self.assertGreater(checked, 150)

    def test_compare(self):
        values = FLOAT_LEAVES + [sympy.Float(v) for v in [0.25, -0.25, 2.0, 1e-300]]
        values += [sympy.Integer(i) for i in (-2, 0, 1, 2)]
        values += [sympy.Rational(1, 2), sympy.Rational(-1, 3), sympy.Rational(1, 10)]
        values += [s0, sympy.Float(0.5) * s0, s0 / 2, sympy.Float(2.0) * s0]
        arena = torch._C._symbolic._Arena()
        for a in values:
            for b in values:
                na, nb = arena.from_sympy(a), arena.from_sympy(b)
                self.assertEqual(arena.compare(na, nb), a.compare(b), f"{a}, {b}")
        terms = [s0 + v for v in values if not v.free_symbols]
        for v in terms:
            self.check_expr(v)
        big = sympy.Integer(2**62 + 1)
        tiny = sympy.Rational(1, 2**62 + 1)
        for v in [big, -big, tiny, -tiny]:
            for w in [sympy.Float(float(v)), sympy.Float(-float(v))]:
                self.check_expr(s0 * v + s1 * w)

    def test_unsupported(self):
        arena = torch._C._symbolic._Arena()
        f = arena.from_sympy(sympy.Float(0.5))
        x = arena.from_sympy(s0)
        big = arena.from_sympy(sympy.Float(1e300))
        tiny = arena.from_sympy(sympy.Float(1e-300))
        cases = [
            lambda: arena.function("Max", [f, x]),
            lambda: arena.function("FloorDiv", [arena.mul([f, x]), x]),
            lambda: arena.safe_expand(arena.pow(arena.add([x, f]), arena.integer(2))),
            lambda: arena.mul([big, big]),
            lambda: arena.mul([tiny, tiny]),
            lambda: arena.pow(f, arena.integer(3)),
            lambda: arena.pow(arena.from_sympy(sympy.Float(0.0)), arena.integer(-1)),
            lambda: arena.add([f, arena.from_sympy(sympy.Rational(1, 2**60 + 1))]),
        ]
        # Sort keys that tie on Float(0.5) and Rational(1, 2).
        h, fh = sympy.Rational(1, 2), sympy.Float(0.5)
        ties = [(s0 + fh) ** 2 * (s0 + h) ** 3, (fh + 1 / s0) * (h + 1 / s1)]
        for v in ties:
            cases.append(lambda v=v: arena.sstr(arena.from_sympy(v)))
        for i, c in enumerate(cases):
            with self.assertRaises(NativeUnsupported, msg=f"case {i}"):
                c()


class TestNativeInfinity(TestCase):
    NUMS = [sympy.oo, -sympy.oo, int_oo, -int_oo, *map(sympy.Integer, [0, 3, -2])]
    NUMS += [sympy.Rational(1, 2), *map(sympy.Float, [0.5, -2.0, 0.0])]

    def check_result(self, arena, what, expected, fn):
        """Checks fn() against sympy's result. Returns False if native raised
        NativeUnsupported."""
        if expected is sympy.nan:
            with self.assertRaises(NativeUnsupported, msg=what):
                fn()
            return True
        try:
            n = fn()
        except NativeUnsupported:
            return False
        got = arena.to_sympy(n)
        self.assertTrue(got == expected and type(got) is type(expected), what)
        self.assertIs(arena.from_sympy(expected).id, n.id)
        return True

    def test_leaves(self):
        for v in [sympy.oo, -sympy.oo]:
            TestNativeFloat.check_expr(self, v)
            arena = torch._C._symbolic._Arena()
            self.assertEqual(arena.from_sympy(v).kind, type(v).__name__)

    def test_arithmetic(self):
        arena = torch._C._symbolic._Arena()
        for a in self.NUMS:
            na = arena.from_sympy(a)
            # int_oo is not supported in core arithmetic.
            finite_a = a not in (int_oo, -int_oo)
            ok = self.check_result(arena, f"-{a}", -a, lambda: arena.neg(na))
            self.assertTrue(ok or not finite_a, f"-{a}")
            for e in range(-2, 4):
                with contextlib.suppress(ZeroDivisionError):
                    expected = a**e
                    self.check_result(
                        arena,
                        f"{a}**{e}",
                        expected,
                        lambda: arena.pow(na, arena.integer(e)),
                    )
            for b in self.NUMS:
                nb = arena.from_sympy(b)
                supported = finite_a and b not in (int_oo, -int_oo)
                for name, expected, fn in [
                    ("add", sympy.Add(a, b), lambda: arena.add([na, nb])),
                    ("mul", sympy.Mul(a, b), lambda: arena.mul([na, nb])),
                    ("sub", a - b, lambda: arena.sub(na, nb)),
                ]:
                    ok = self.check_result(arena, f"{name}({a}, {b})", expected, fn)
                    self.assertTrue(ok or not supported, f"{name}({a}, {b})")

    def test_compare(self):
        nums = self.NUMS + [sympy.Rational(-1, 3), s0, zf]
        arena = torch._C._symbolic._Arena()
        for a in nums:
            for b in nums:
                na, nb = arena.from_sympy(a), arena.from_sympy(b)
                self.assertEqual(arena.compare(na, nb), a.compare(b), f"{a}, {b}")
        for v in [s0 * sympy.Float(0.5) + zf, s0 + u0]:
            terms = [arena.from_sympy(v), arena.from_sympy(sympy.oo)]
            got = [arena.to_sympy(x) for x in arena.ordered(terms)]
            self.assertEqual(got, list(sympy.ordered([v, sympy.oo])))

    def test_relational(self):
        rel = TestNativeRelational.check_rel
        for op in RELATIONS:
            for a in self.NUMS:
                for b in self.NUMS:
                    self.assertTrue(rel(self, op, a, b), f"{op}({a}, {b})")
        oo = sympy.oo
        cases = [("Lt", s0, oo), ("Ge", u0, -oo), ("Eq", s0, oo), ("Le", oo, zf)]
        cases += [("Lt", -oo, u0), ("Ne", zf, -oo), ("Gt", u0, oo), ("Eq", oo, oo)]
        cases += [("Lt", u0, -oo), ("Gt", -oo, u0), ("Le", s0 + 1, oo)]
        for op, a, b in cases:
            self.assertTrue(rel(self, op, a, b), f"{op}({a}, {b})")
        arena = torch._C._symbolic._Arena()
        for v in [sympy.Lt(u0, -oo, evaluate=False), sympy.Gt(-oo, u0, evaluate=False)]:
            self.assertEqual(arena.sstr(arena.from_sympy(v)), str(v))
        n = arena.from_sympy(sympy.oo)
        self.assertIs(arena.to_sympy(arena.logical_not(n)), sympy.false)

    def test_symbolic(self):
        oo = sympy.oo
        cases = [[s0, oo], [u0, -oo], [zf, oo], [2 * s0, 1, oo], [-zf, -oo]]
        cases += [[s0, u0, -oo], [oo, u0, zf], [sympy.Float(0.5) * zf, -oo]]
        for terms in cases:
            arena = torch._C._symbolic._Arena()
            n = arena.add([arena.from_sympy(a) for a in terms])
            self.assertIs(arena.to_sympy(n), sympy.Add(*terms), f"{terms}")

    def test_unsupported(self):
        arena = torch._C._symbolic._Arena()
        oo = arena.from_sympy(sympy.oo)
        neg_oo = arena.from_sympy(-sympy.oo)
        x = arena.from_sympy(s0)
        cases = [
            lambda: arena.add([oo, neg_oo]),
            lambda: arena.mul([arena.integer(0), oo]),
            lambda: arena.mul([oo, x]),
            lambda: arena.function("Max", [oo, x]),
            lambda: arena.safe_expand(oo),
            lambda: arena.add([oo, arena.from_sympy(int_oo)]),
        ]
        for i, fn in enumerate(cases):
            with self.assertRaises(NativeUnsupported, msg=f"case {i}"):
                fn()

    @parametrize("seed", range(4))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        leaves = LEAVES + FLOAT_LEAVES + [sympy.oo, -sympy.oo]
        checked = 0
        for _ in range(300):
            t = random_tree(rng, 3, leaves)
            try:
                expected = sympy.sympify(sympy_eval(t))
            except ZeroDivisionError:
                continue
            arena = torch._C._symbolic._Arena()
            if expected.has(sympy.nan, sympy.zoo):
                with contextlib.suppress(NativeUnsupported):
                    n = native_eval(arena, t)
                    self.fail(f"{t}: native {arena.to_sympy(n)}, sympy {expected}")
                continue
            try:
                n = native_eval(arena, t)
            except NativeUnsupported:
                continue
            got = arena.to_sympy(n)
            self.assertTrue(got == expected, f"{t}: native {got} != sympy {expected}")
            self.assertEqual(arena.from_sympy(expected), n)
            with contextlib.suppress(NativeUnsupported):
                self.assertEqual(arena.sstr(n), str(expected))
            for f in FACTS:
                with contextlib.suppress(NativeUnsupported):
                    got = arena.ask(n, f)
                    self.assertIs(got, getattr(expected, "is_" + f), f"{t}.is_{f}")
            checked += 1
        self.assertGreater(checked, 100)


class TestNativeFunctions(TestCase):
    FUNCTIONS = {
        "Mod": Mod,
        "PythonMod": PythonMod,
        "FloorDiv": FloorDiv,
        "CleanDiv": CleanDiv,
        "Max": Max,
        "Min": Min,
        "PowByNatural": PowByNatural,
        "FloatPow": FloatPow,
        "FloatTrueDiv": FloatTrueDiv,
        "IntTrueDiv": IntTrueDiv,
    }
    UNARY_FUNCTIONS = {
        "CeilToInt": CeilToInt,
        "FloorToInt": FloorToInt,
        "TruncToInt": TruncToInt,
        "RoundToInt": RoundToInt,
        "ToFloat": ToFloat,
        "TruncToFloat": TruncToFloat,
    }
    # Always evaluate, so they are arena methods rather than function kinds.
    CONSTRUCTORS = {"CeilDiv": CeilDiv, "LShift": LShift, "RShift": RShift}
    ALL_FUNCTIONS = {**FUNCTIONS, **UNARY_FUNCTIONS, **CONSTRUCTORS}
    ALL_FUNCTIONS["RoundDecimal"] = RoundDecimal
    ALL_FUNCTIONS["IsNonOverlappingAndDenseIndicator"] = (
        IsNonOverlappingAndDenseIndicator
    )
    ALL_FUNCTIONS["ModularIndexing"] = ModularIndexing
    BITWISE = (BitwiseFn_bitwise_and, BitwiseFn_bitwise_or, BitwiseFn_bitwise_xor)
    ALL_FUNCTIONS.update((f.__name__, f) for f in BITWISE)
    ALL_FUNCTIONS["Identity"] = Identity
    NODE_TYPES = (Mod, PythonMod, FloorDiv, Max, Min, PowByNatural, FloatPow)
    NODE_TYPES += (FloatTrueDiv, IntTrueDiv, CeilToInt, FloorToInt, TruncToInt)
    NODE_TYPES += (RoundToInt, ToFloat, TruncToFloat, RoundDecimal)
    NODE_TYPES += (IsNonOverlappingAndDenseIndicator, ModularIndexing, *BITWISE)
    NODE_TYPES += (Identity,)
    PYTHON_ERRORS = (ZeroDivisionError, AssertionError, TypeError, ValueError)
    PYTHON_ERRORS += (OverflowError,)

    def check_call(self, name, *xs):
        """Returns the sympy result, or None if native raised
        NativeUnsupported."""
        arena = torch._C._symbolic._Arena()
        call = f"{name}{xs}"
        try:
            args = [arena.from_sympy(x) for x in xs]
        except NativeUnsupported:
            return None

        def native():
            if name in self.CONSTRUCTORS:
                return getattr(arena, name.lower())(*args)
            return arena.function(name, args)

        try:
            expected = self.ALL_FUNCTIONS[name](*xs)
        except self.PYTHON_ERRORS:
            with self.assertRaises(NativeUnsupported, msg=call):
                native()
            return None
        try:
            r = native()
        except NativeUnsupported:
            return None
        got = arena.to_sympy(r)
        self.assertEqual(got, expected, call)
        self.assertEqual(type(got), type(expected), call)
        self.assertEqual(got.args, expected.args, call)
        self.assertIs(arena.from_sympy(expected).id, r.id)
        return expected

    def check_expr(self, v, rng):
        """Returns False if native raised NativeUnsupported, else checks facts,
        str and sort_key against sympy."""
        arena = torch._C._symbolic._Arena()
        try:
            n = arena.from_sympy(v)
            got = arena.sstr(n)
        except NativeUnsupported:
            return False
        self.assertEqual(got, str(v))
        self.assertTrue(arena.sort_key(n) == v.sort_key(), f"sort_key({v})")
        self.assertEqual(arena.to_sympy(n), v)
        # Max/Min handlers are weaker than the rules, so sympy's answers depend
        # on the query order and the order _ask visits prerequisites in. Make
        # sympy visit them in the native Fact order, starting from empty KBs.
        for x in sympy.preorder_traversal(v):
            if not x.is_Atom:
                x._assumptions = type(x).default_assumptions
        facts = list(FACTS)
        rng.shuffle(facts)
        assumptions = sys.modules["sympy.core.assumptions"]
        with mock.patch.object(assumptions, "shuffle", sort_facts_natively):
            for f in facts:
                try:
                    got = arena.ask(n, f)
                except NativeUnsupported:
                    continue
                self.assertIs(got, getattr(v, "is_" + f), f"({v}).is_{f}")
        return True

    def test_known(self):
        cases = [
            ("Mod", 2 * s0, 2, 0),
            ("Mod", 2 * s0 + 1, 2, 1),
            ("Mod", s0 * s1, s1, 0),
            ("Mod", s0, s0 + 1, s0),
            ("Mod", s0, 2 * s0, s0),
            ("Mod", s0 + 1, s0, Mod(s0 + 1, s0)),
            ("Mod", 4 * s0 + 2, 4, Mod(4 * s0 + 2, 4)),
            ("Mod", u0, -u0, 0),
            ("Mod", 7, 3, 1),
            ("Mod", sympy.Rational(7, 2), 2, sympy.Rational(3, 2)),
            ("PythonMod", 7, -2, -1),
            ("PythonMod", -7, 2, 1),
            ("PythonMod", 2 * u0 + 1, 2, 1),
            ("PythonMod", s0, s0 + s1, s0),
            ("FloorDiv", 7, -2, -4),
            ("FloorDiv", -7, 2, -4),
            ("FloorDiv", int_oo, 2, int_oo),
            ("FloorDiv", int_oo, -2, -int_oo),
            ("FloorDiv", -int_oo, sympy.Rational(1, 2), -int_oo),
            ("FloorDiv", 5, int_oo, 0),
            ("FloorDiv", -5, int_oo, 0),
            ("FloorDiv", int_oo, 1, int_oo),
            ("FloorDiv", 0, s0, 0),
            ("FloorDiv", s0, 1, s0),
            ("FloorDiv", u0, -1, -u0),
            ("FloorDiv", s0, s0, 1),
            ("FloorDiv", 2 * s0 + 3, 2, s0 + 1),
            ("FloorDiv", 2 * u0 + 1, 2, u0),
            ("FloorDiv", s0 + 2, 2, FloorDiv(s0, 2) + 1),
            ("FloorDiv", s0 - 2, 2, FloorDiv(s0, 2) - 1),
            ("FloorDiv", s0 + 2 * s1, 2, s1 + FloorDiv(s0, 2)),
            ("FloorDiv", 2 * s0 + 3 * s1, 2, s0 + FloorDiv(3 * s1, 2)),
            ("FloorDiv", 2 * s0, 4, FloorDiv(s0, 2)),
            ("FloorDiv", 6 * s0 * s1, 4 * s1, FloorDiv(3 * s0, 2)),
            ("FloorDiv", 3 * s0, 3 * s1, FloorDiv(s0, s1)),
            ("FloorDiv", s0 * s1, s0, s1),
            ("FloorDiv", 1024 * s0, 128 * s0, 8),
            ("FloorDiv", 4096 * s0, s0, 4096),
            ("FloorDiv", s0, -s0, -1),
            ("FloorDiv", -s0, -s1, FloorDiv(s0, s1)),
            ("FloorDiv", s0**2, s0, FloorDiv(s0**2, s0)),
            ("FloorDiv", 1, s0, FloorDiv(1, s0)),
            ("FloorDiv", FloorDiv(s0, 2), 3, FloorDiv(s0, 6)),
            ("FloorDiv", FloorDiv(s0, s1), s1, FloorDiv(s0, s1**2)),
            ("FloorDiv", CleanDiv(s0, s1), 2, FloorDiv(s0, 2 * s1)),
            ("FloorDiv", s0, s1 * (s0 + 1), FloorDiv(s0, s1 * (s0 + 1))),
            ("FloorDiv", s0, s1 + 1, FloorDiv(s0, s1 + 1)),
            (
                "FloorDiv",
                3 * s0**2 * s1,
                s0 * s1 + 2,
                FloorDiv(3 * s0**2 * s1, s0 * s1 + 2),
            ),
            ("FloorDiv", 1, 2 * s0 + 3, FloorDiv(1, 2 * s0 + 3)),
            ("CleanDiv", s0 * s1, s1, s0),
            ("CleanDiv", s0, s1, CleanDiv(s0, s1)),
            ("CleanDiv", 2 * s0 + 2, 2, s0 + 1),
            ("PowByNatural", 2, 62, 2**62),
            ("PowByNatural", 2, 63, int_oo),
            ("PowByNatural", -2, 63, -int_oo),
            ("PowByNatural", -2, 64, int_oo),
            ("PowByNatural", -(2**63), 1, -int_oo),
            ("PowByNatural", 3037000499, 2, 3037000499**2),
            ("PowByNatural", 3037000500, 2, int_oo),
            ("PowByNatural", -3, 3, -27),
            ("PowByNatural", 0, 0, 1),
            ("PowByNatural", 0, 5, 0),
            ("PowByNatural", 1, 2**62, 1),
            ("PowByNatural", -1, 2**62 + 1, -1),
            ("PowByNatural", s0, 2, s0**2),
            ("PowByNatural", s0, 0, 1),
            ("PowByNatural", s0, -1, 1 / s0),
            ("PowByNatural", sympy.Rational(1, 2), 3, sympy.Rational(1, 8)),
            ("PowByNatural", s0, int_oo, int_oo),
            ("PowByNatural", sympy.Rational(1, 2), int_oo, int_oo),
            ("PowByNatural", 0, int_oo, int_oo),
            ("PowByNatural", u0, int_oo, PowByNatural(u0, int_oo)),
            ("PowByNatural", s0, -int_oo, PowByNatural(s0, -int_oo)),
            ("PowByNatural", 2, s0, PowByNatural(2, s0)),
            ("PowByNatural", s0, s1, PowByNatural(s0, s1)),
            ("PowByNatural", int_oo, s0, PowByNatural(int_oo, s0)),
            ("FloatPow", s0, 2, FloatPow(s0, 2)),
            ("FloatPow", zf, s0, FloatPow(zf, s0)),
            ("FloatTrueDiv", s0, s1, FloatTrueDiv(s0, s1)),
            ("FloatTrueDiv", s0, int_oo, FloatTrueDiv(s0, int_oo)),
            ("FloatTrueDiv", 1, zf, FloatTrueDiv(1, zf)),
            ("IntTrueDiv", s0, 2, IntTrueDiv(s0, 2)),
            ("IntTrueDiv", s0 + 1, s1, IntTrueDiv(s0 + 1, s1)),
            ("CeilDiv", 4 * s0, 2, 2 * s0),
            ("CeilDiv", 4 * s0 + 2, 4, s0 + 1),
            ("CeilDiv", 2 * s0, 4, FloorDiv(2 * s0 + 3, 4)),
            ("CeilDiv", 0, -2, 1),
            ("CeilDiv", 0, 2 * s0, 0),
            ("CeilDiv", 0, -2 * s0, FloorDiv(-2 * s0 - 1, -2 * s0)),
            ("CeilDiv", 7, 2, 4),
            ("CeilDiv", -7, 2, -3),
            ("CeilDiv", 6, 3, 2),
            ("CeilDiv", 6, -3, -1),
            ("CeilDiv", s0, 1, s0),
            ("CeilDiv", s0, -1, 2 - s0),
            ("CeilDiv", s0 * s1, s0, s1),
            ("CeilDiv", 4 * s0**2, 2 * s0, FloorDiv(2 * s0**2, s0)),
            ("CeilDiv", s0**2 * s1 + s0, s0, CleanDiv(s0**2 * s1 + s0, s0)),
            ("CeilDiv", s0, s1, FloorDiv(s0 + s1 - 1, s1)),
            ("CeilDiv", s0, s0 + 1, FloorDiv(2 * s0, s0 + 1)),
        ]
        for name, a, b, expected in cases:
            got = self.check_call(name, sympy.sympify(a), sympy.sympify(b))
            self.assertEqual(got, expected, f"{name}({a}, {b})")

    def test_unsupported(self):
        arena = torch._C._symbolic._Arena()
        cases = [
            ("Mod", -7, 3),
            ("Mod", 7, 0),
            ("Mod", s0, 0),
            ("Mod", int_oo, 2),
            ("Mod", s0, int_oo),
            ("PythonMod", u0, 2),
            ("PythonMod", s0 + 1, s1),
            ("FloorDiv", s0, 0),
            ("FloorDiv", int_oo, int_oo),
            ("FloorDiv", s0, int_oo),
            ("FloorDiv", s0 * s1 + s0, s1 + 1),
            ("FloorDiv", 4 * s0 + 2, 4 * s1),
            ("FloorDiv", s0, s0 * s1 + s0),
            ("FloorDiv", s0 + 1, s1 + 1),
            ("FloorDiv", sympy.Rational(7, 2), 2),
            ("PowByNatural", 2, -1),
            ("PowByNatural", -1, int_oo),
            ("PowByNatural", -int_oo, int_oo),
            ("PowByNatural", 1, -int_oo),
            ("FloatTrueDiv", s0, 0),
            ("IntTrueDiv", sympy.Rational(1, 2), sympy.Rational(1, 3)),
            ("IntTrueDiv", s0, 0),
            ("CeilDiv", 0, 0),
            ("CeilDiv", s0, 0),
            ("CeilDiv", s0 + 1, s1 + 1),
            ("CeilDiv", 0, s0 + 1),
            ("CeilDiv", s0 / 2, 2),
            ("CeilDiv", FloorDiv(s0, 2), 2),
            ("CeilDiv", s0, int_oo),
        ]
        for name, a, b in cases:
            args = [arena.from_sympy(sympy.sympify(x)) for x in (a, b)]
            with self.assertRaises(NativeUnsupported, msg=f"{name}({a}, {b})"):
                if name == "CeilDiv":
                    arena.ceildiv(*args)
                else:
                    arena.function(name, args)
        with self.assertRaises(NativeUnsupported):
            arena.function("Mod", [arena.from_sympy(s0)])
        with self.assertRaises(NativeUnsupported):
            arena.function("Mod", [arena.boolean(True), arena.from_sympy(s0)])
        with self.assertRaises(NativeUnsupported):
            arena.from_sympy(Mod(4, 2, evaluate=False))
        v = Mod(s0, 3, evaluate=False)
        self.assertEqual(arena.to_sympy(arena.from_sympy(v)), v)
        x = arena.from_sympy(s0)
        with self.assertRaises(NativeUnsupported):
            arena.diff(arena.from_sympy(Mod(s0, 3)), x)

    def test_printing(self):
        m = Mod(s0, 2)
        cases = [s0 - m, -2 * m, m**2, 1 / m, s0 / m, m * s1, -m * s1, m + 1]
        cases += [sympy.Lt(m, s0), sympy.Eq(m, 1), Mod(s0 + 1, s1 + 2), -m]
        cases += [Mod(m + s1, 3), Mod(u0, 2) * u0, s0 * m / (s1 + 1)]
        f = FloorDiv(s0, 2)
        cases += [f, f + 1, 2 * f, f**2, sympy.Lt(FloorDiv(u0, 2), s0 + 1)]
        cases += [Mod(f, 3), sympy.Eq(f, 1), FloorDiv(-u0, 2), FloorDiv(u0, -3)]
        cases += [FloorDiv(s0 + 1, 2), FloorDiv(s0, s1 + 1), FloorDiv(2 * s0, s1)]
        cases += [FloorDiv(s0**2, s0), FloorDiv(s0 * f, 2), CleanDiv(s0, s1)]
        cases += [-f, s0 - f, f * Mod(s0, 3), 1 / f, CleanDiv(s0 + 1, s1) * s0]
        p, fp = PowByNatural(s0, s1), FloatPow(zf, s0)
        cases += [p, 2 * p, p**2, -p, s0 * p, 1 / p, p + 1, sympy.Lt(p, s0)]
        cases += [fp, 2 * fp, fp**2, -fp, 1 / fp, fp + zf, PowByNatural(s0 + 1, 2 * s1)]
        d, i = FloatTrueDiv(zf, s0), IntTrueDiv(s0, s1)
        cases += [d, 2 * d, d**2, -d, 1 / d, d + i, s0 * i, sympy.Eq(i, zf)]
        cases += [FloatTrueDiv(s0 + 1, zf), PowByNatural(u0, int_oo), p * fp * d]
        rng = random.Random(0)
        for v in cases:
            self.assertTrue(self.check_expr(v, rng), f"{v}")

    def test_sorting(self):
        items = [Mod(s0, 2), Mod(s0, 3), Mod(s1, 2), Mod(u0, 2), s0, u0]
        items += [Mod(s0 + 1, s1), s0 + Mod(s0, 2), sympy.Integer(2), s0**2]
        items += [sympy.Eq(s0, 1, evaluate=False), sympy.Not(u0), sympy.true]
        items += [FloorDiv(s0, 2), CleanDiv(s0, 2), CleanDiv(s0, s1), FloorDiv(u0, 2)]
        items += [Max(2, s0), Min(2, u0), Max(s0, u0), Min(s0, u0), Max(s1, s0 + 1)]
        items += [PowByNatural(s0, s1), PowByNatural(2, s0), FloatPow(zf, s0)]
        items += [FloatTrueDiv(zf, s0), IntTrueDiv(s0, s1), IntTrueDiv(s0, 2)]
        arena = torch._C._symbolic._Arena()
        natives = [arena.from_sympy(x) for x in items]
        for a, na in zip(items, natives):
            for b, nb in zip(items, natives):
                self.assertEqual(arena.compare(na, nb), a.compare(b), f"{a}, {b}")
        got = arena.ordered(natives)
        self.assertEqual([arena.to_sympy(x) for x in got], list(sympy.ordered(items)))

    @parametrize("seed", range(4))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        calls = supported = 0
        nodes = []
        for _ in range(150):
            a = sympy.sympify(sympy_eval(random_tree(rng, 2, FACT_LEAVES)))
            b = sympy.sympify(sympy_eval(random_tree(rng, 2, FACT_LEAVES)))
            if a.has(sympy.zoo, sympy.nan) or b.has(sympy.zoo, sympy.nan):
                continue
            for name in self.FUNCTIONS:
                calls += 1
                r = self.check_call(name, a, b)
                if r is not None:
                    supported += 1
                    if isinstance(r, self.NODE_TYPES):
                        nodes.append(r)
        self.assertGreater(supported, calls // 3)
        self.assertGreater(len(nodes), 10)
        answered = unsupported = 0
        for _ in range(60):
            t = random_tree(rng, 3, LEAVES + nodes)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                if self.check_expr(v, rng):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 5 * unsupported)

    @parametrize("seed", range(4))
    def test_floordiv_fuzz(self, seed):
        # Shapes that FloorDiv sees on the SymNode path: sums of monomials over
        # integer symbols divided by integers or monomials.
        rng = random.Random(seed)
        syms = [s0, s1, u0, sympy.Symbol("n", integer=True, nonnegative=True)]

        def monomial():
            c = rng.choice([-4, -2, -1, 1, 1, 2, 3, 4, 6, 8, 128])
            return sympy.Mul(c, *rng.sample(syms, rng.randint(0, 2)))

        def poly():
            return sympy.Add(*(monomial() for _ in range(rng.randint(1, 3))))

        calls = supported = 0
        nodes = []
        for _ in range(300):
            a = rng.choice([poly, monomial])()
            b = rng.choice(
                [monomial, poly, lambda: sympy.Integer(rng.randint(-4, 8))]
            )()
            if rng.random() < 0.2:
                a = FloorDiv(a, b) if b != 0 else a
            for name in ("FloorDiv", "CleanDiv"):
                calls += 1
                r = self.check_call(name, a, b)
                if r is not None:
                    supported += 1
                    nodes += [
                        x
                        for x in sympy.preorder_traversal(r)
                        if isinstance(x, FloorDiv)
                    ]
        self.assertGreater(supported, calls // 2)
        answered = unsupported = 0
        for _ in range(40):
            t = random_tree(rng, 3, LEAVES + nodes)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                if self.check_expr(v, rng):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 5 * unsupported)

    @parametrize("seed", range(4))
    def test_ceildiv_fuzz(self, seed):
        rng = random.Random(seed)
        syms = [s0, s1, u0, zf, sympy.Symbol("n", integer=True, nonnegative=True)]

        def monomial():
            c = rng.choice([-4, -2, -1, 0, 1, 1, 2, 3, 4, 6, 8, 128])
            return sympy.Mul(c, *(rng.choice(syms) for _ in range(rng.randint(0, 3))))

        def poly():
            return sympy.Add(*(monomial() for _ in range(rng.randint(1, 3))))

        calls = supported = 0
        for _ in range(300):
            a = rng.choice([poly, monomial])()
            b = rng.choice([monomial, monomial, poly])()
            calls += 1
            if self.check_call("CeilDiv", a, b) is not None:
                supported += 1
        self.assertGreater(supported, calls // 2)

    def test_modular_indexing_known(self):
        n = sympy.Symbol("n", integer=True, nonnegative=True)
        cases = [
            ((7, 2, 3), 0),
            ((-7, 2, 3), 2),
            ((7, -2, 3), 2),
            ((7, 2, -3), 0),
            ((0, s0, s1), 0),
            ((s0, s1, 1), 0),
            ((s0, 2, 8), ModularIndexing(s0, 2, 8)),
            ((4 * s0 + 8, 2, 8), ModularIndexing(2 * s0 + 4, 1, 8)),
            ((2 * s0 * s1, 2 * s1, 8), ModularIndexing(s0, 1, 8)),
            ((6 * s0**2 * s1, 4 * s0, 8), ModularIndexing(3 * s0 * s1, 2, 8)),
            ((s0 + 8 * n, 1, 8), ModularIndexing(s0, 1, 8)),
            ((u0 + 8 * n, 1, 8), ModularIndexing(u0 + 8 * n, 1, 8)),
            ((16 * n + 8 * s0, 2, 4), 0),
            ((FloorDiv(s0, 4), 2, 8), ModularIndexing(s0, 8, 8)),
            ((s0 * s1 + s0, s0, 8), ModularIndexing(s1 + 1, 1, 8)),
            ((s0**2 * s1 + s0**2, s0, 8), None),
            ((s0 + 1, s0 + 2, 8), None),
            ((sympy.Rational(1, 2), 2, 3), None),
            ((s0 * FloorDiv(s0 * (s1 + 1), 3), FloorDiv(s0 * s1 + s0, 3), 5), None),
            ((8 * s1 * Max(3, s0 * (s1 + 1)) + s0, Max(3, s0 * (s1 + 1)), 8), None),
            ((2 * s0 + 2 * FloorDiv(-u0 - 3, -s1), 2, 8), None),
        ]
        for args, expected in cases:
            got = self.check_call("ModularIndexing", *map(sympy.sympify, args))
            self.assertEqual(got, expected, f"ModularIndexing{args}")

    @parametrize("seed", range(4))
    def test_modular_indexing_fuzz(self, seed):
        rng = random.Random(seed)
        n = sympy.Symbol("n", integer=True, nonnegative=True)
        atoms = [s0, s1, u0, n, FloorDiv(s0, 2), Mod(s1, 3), FloorDiv(u0, 4)]

        def monomial():
            c = rng.choice([-4, -2, -1, 1, 1, 2, 3, 4, 6, 8, 16, 128])
            return sympy.Mul(c, *(rng.choice(atoms) for _ in range(rng.randint(0, 2))))

        def poly():
            return sympy.Add(*(monomial() for _ in range(rng.randint(1, 4))))

        def integer():
            return sympy.Integer(rng.choice([-4, -2, 1, 1, 2, 3, 4, 8, 64]))

        calls = supported = 0
        nodes = []
        for _ in range(300):
            a = rng.choice([poly, poly, monomial])()
            if rng.random() < 0.2:
                a = FloorDiv(a, rng.choice([2, 4, s1]))
            b = rng.choice([monomial, integer, integer, poly])()
            c = rng.choice([monomial, integer, integer])()
            calls += 1
            r = self.check_call("ModularIndexing", a, b, c)
            if r is not None:
                supported += 1
                nodes += [
                    x
                    for x in sympy.preorder_traversal(r)
                    if isinstance(x, ModularIndexing)
                ]
        self.assertGreater(supported, calls // 2)
        self.assertGreater(len(nodes), 20)
        answered = unsupported = 0
        for _ in range(40):
            t = random_tree(rng, 3, LEAVES + nodes)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                if self.check_expr(v, rng):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 5 * unsupported)

    def test_bitwise_known(self):
        And, Or, Xor = self.BITWISE
        t, f, eq = sympy.true, sympy.false, sympy.Eq(u0, 1)
        both = sympy.And(u0 > 1, s0 > 2)
        either = sympy.Or(u0 > 1, s0 > 2)
        cases = [
            (And, (5, 3), 1),
            (Or, (5, 3), 7),
            (Xor, (5, 3), 6),
            (And, (-5, 3), 3),
            (Or, (-5, 3), -5),
            (Xor, (-5, 3), -8),
            (And, (2**63 - 1, -(2**63)), 0),
            (Xor, (2**63 - 1, -(2**63)), -1),
            (And, (t, f), f),
            (Or, (t, f), t),
            (Xor, (t, t), f),
            (Xor, (f, t), t),
            (And, (t, 3), 1),
            (Or, (f, 3), 3),
            (Xor, (3, t), 2),
            (And, (both, 3), 1),
            (Xor, (3, either), 2),
            (Xor, (sympy.Not(both), 3), 2),
            (And, (both, t), both),
            (And, (either, f), f),
            (Or, (both, t), t),
            (Or, (either, f), either),
            (Xor, (both, t), sympy.Not(both)),
            (Xor, (f, either), either),
            (Xor, (both, both), f),
            (And, (both, either), sympy.And(both, either)),
            (Or, (both, sympy.Not(both)), sympy.Or(both, sympy.Not(both))),
            (Xor, (both, either), None),
            (And, (u0, s0), And(u0, s0)),
            (Or, (u0 + 1, 3), Or(u0 + 1, 3)),
            (Xor, (2 * s0, -1), Xor(2 * s0, -1)),
            (And, (eq, t), None),
            (And, (eq, 3), None),
            (Or, (t, u0), None),
            (Xor, (both, u0), None),
            (And, (sympy.Rational(1, 2), 3), None),
            (Or, (sympy.Float(2.0), 3), None),
            (Xor, (int_oo, 3), None),
            (And, (sympy.oo, t), None),
        ]
        for fn, args, expected in cases:
            got = self.check_call(fn.__name__, *map(sympy.sympify, args))
            self.assertEqual(got, expected, f"{fn.__name__}{args}")

    @parametrize("seed", range(2))
    def test_bitwise_fuzz(self, seed):
        rng = random.Random(seed)
        n = sympy.Symbol("n", integer=True, nonnegative=True)
        both = sympy.And(u0 > 1, s0 > 2)
        atoms = [s0, u0, n, zf, 2 * s0 + 1, u0 - n, FloorDiv(u0, 2)]
        atoms += [*map(sympy.Integer, [-(2**63), -9, -5, -1, 0, 1, 2, 3, 12])]
        atoms += [sympy.Integer(2**63 - 1), sympy.Rational(1, 2), sympy.Float(3.0)]
        atoms += [int_oo, -sympy.oo, sympy.true, sympy.false, sympy.Eq(u0, 1)]
        atoms += [both, sympy.Or(u0 > 1, s0 > 2), sympy.Not(both)]
        nodes = []
        for fn in self.BITWISE:
            for a in atoms:
                for b in atoms:
                    r = self.check_call(fn.__name__, a, b)
                    if isinstance(r, self.BITWISE):
                        nodes.append(r)
        self.assertGreater(len(nodes), 100)
        answered = unsupported = 0
        for _ in range(60):
            t = random_tree(rng, 3, LEAVES + nodes)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                if self.check_expr(v, rng):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 5 * unsupported)

    def test_identity_known(self):
        for x in [s0, u0, zf, s0 + 1, 2 * u0, -s0, Identity(s0), FloorDiv(u0, 2)]:
            self.assertEqual(self.check_call("Identity", x), Identity(x))
        for x in [2, sympy.Rational(1, 2), sympy.Float(2.0), int_oo, sympy.true]:
            self.assertIsNone(self.check_call("Identity", sympy.sympify(x)))
        self.assertIsNone(self.check_call("Identity", s0, s1))
        i = Identity(u0)
        exprs = [i + 1, 2 * i, -i, i**2, i / s0, 1 / i, s0 - i, i * s0]
        exprs += [i < 3, sympy.Eq(i, s0), FloorDiv(i, 2), Mod(i, 3), Max(i, s0)]
        exprs += [Identity(zf) + s0, Identity(s0 + u0) * 3]
        exprs += [ModularIndexing(i, 1, 4), BitwiseFn_bitwise_and(i, 3)]
        rng = random.Random(0)
        for e in exprs:
            self.assertTrue(self.check_expr(e, rng), str(e))
        a = torch._C._symbolic._Arena()
        ids = [a.function("Identity", [a.from_sympy(x)]) for x in [s0, u0, zf]]
        want = sympy.ordered([Identity(s0), Identity(u0), Identity(zf), s0, FloorDiv(u0, 2)])
        extra = [a.from_sympy(s0), a.from_sympy(FloorDiv(u0, 2))]
        got = [a.to_sympy(n) for n in a.ordered(ids + extra)]
        self.assertEqual(got, list(want))

    @parametrize("seed", range(2))
    def test_identity_fuzz(self, seed):
        rng = random.Random(seed)
        nodes = [Identity(x) for x in [s0, u0, zf, 2 * s0 + 1, u0 - s1, FloorDiv(u0, 2)]]
        answered = unsupported = 0
        for _ in range(60):
            t = random_tree(rng, 3, LEAVES + nodes)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                if self.check_expr(v, rng):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 5 * unsupported)

    def test_minmax_known(self):
        u1 = sympy.Symbol("u1", integer=True)
        n = sympy.Symbol("n", integer=True, nonnegative=True)
        oo = int_oo
        # sympy leaves the redundant u0 in the nested Min.
        redundant = Min(s0, u0, u1, evaluate=False)
        cases = [
            ("Max", (s0,), s0),
            ("Max", (2, s0), Max(2, s0)),
            ("Max", (0, s0), s0),
            ("Min", (0, s0), 0),
            ("Max", (1, s0), s0),
            ("Min", (1, s0), 1),
            ("Max", (0, n), n),
            ("Max", (1, n), Max(1, n)),
            ("Max", (1, u0), Max(1, u0)),
            ("Max", (2, 3, s0), Max(3, s0)),
            ("Min", (2, 3, s0), Min(2, s0)),
            ("Max", (2, 3), 3),
            ("Min", (sympy.Rational(1, 2), 2), sympy.Rational(1, 2)),
            ("Max", (s0, s0 + 1), Max(s0, s0 + 1)),
            ("Max", (s0, 2 * s0), 2 * s0),
            ("Min", (s0, 2 * s0), s0),
            ("Max", (s0, -s0), s0),
            ("Min", (-s0, -2 * s0), -2 * s0),
            ("Max", (u0, 2 * u0), Max(u0, 2 * u0)),
            ("Max", (s0, Max(s1, 2)), Max(2, s0, s1)),
            ("Max", (s0 + s1, s1), Max(s1, s0 + s1)),
            ("Max", (s0, Min(s0, s1)), s0),
            ("Min", (s0, Max(s0, s1)), s0),
            ("Min", (2, Max(3, u0)), 2),
            ("Max", (3, Min(2, u0)), 3),
            ("Max", (2, Min(3, u0)), Max(2, Min(3, u0))),
            ("Max", (Min(s0, u0), Min(s0, s1)), Min(s0, Max(s1, u0))),
            ("Min", (Max(s0, u0), Max(s0, s1)), Max(s0, Min(s1, u0))),
            ("Min", (Max(s0, u0), Max(s0, u0, s1)), Max(s0, u0)),
            ("Min", (u0, Max(s1, Min(u0, s0, u1))), Min(u0, Max(s1, Min(s0, u1)))),
            (
                "Min",
                (u0, Max(s1, Min(s0, u1, Max(u0, n)))),
                Min(u0, Max(s1, redundant, evaluate=False), evaluate=False),
            ),
            ("Max", (oo, 3), oo),
            ("Min", (oo, 3), 3),
            ("Max", (-oo, s0), Max(-oo, s0)),
            ("Min", (oo, s0), Min(oo, s0)),
            ("Max", (oo, s0), Max(oo, s0)),
            ("Max", (oo, Min(3, u0)), oo),
            ("Min", (3, Max(oo, u0)), 3),
            ("Min", (-oo, Max(-oo, u0)), -oo),
            ("Max", (u0, Min(u0, s0)), u0),
            ("Max", (s0 + s1, u0 + u1), Max(s0 + s1, u0 + u1)),
            ("Max", (Max(s0 + s1, u0 + u1), n + zf), Max(n + zf, s0 + s1, u0 + u1)),
            (
                "Max",
                (Min(s0 + s1, u0 + u1), n + zf),
                Max(n + zf, Min(s0 + s1, u0 + u1)),
            ),
        ]
        for name, xs, expected in cases:
            got = self.check_call(name, *map(sympy.sympify, xs))
            self.assertEqual(got, expected, f"{name}{xs}")

    def test_minmax_unsupported(self):
        arena = torch._C._symbolic._Arena()
        u1 = sympy.Symbol("u1", integer=True)
        x = arena.from_sympy(s0)
        # Python leaves Max(2, 3) unevaluated inside the result.
        cases = [("Max", (Min(2, u0), Min(3, u0))), ("Max", ())]
        # Tied sort keys: sympy orders a frozenset in hash order.
        cases += [("Max", (u1, sympy.Symbol("u1", integer=True, positive=True)))]
        for name, xs in cases:
            args = [arena.from_sympy(sympy.sympify(v)) for v in xs]
            with self.assertRaises(NativeUnsupported, msg=f"{name}{xs}"):
                arena.function(name, args)
        with self.assertRaises(NativeUnsupported):
            arena.function("Max", [arena.boolean(True), x])
        with self.assertRaises(NativeUnsupported):
            arena.function("Max", [arena.from_sympy(W), x])
        with self.assertRaises(NativeUnsupported):
            arena.from_sympy(Max(2, 3, evaluate=False))
        with self.assertRaises(NativeUnsupported):
            arena.diff(arena.from_sympy(Max(s0, u0)), x)
        v = Max(s0, 2 * s0, evaluate=False)
        self.assertEqual(arena.to_sympy(arena.from_sympy(v)).args, v.args)

    def test_minmax_printing(self):
        m = Max(s0 + s1, u0)
        cases = [Max(2, s0), Max(s1, s0 + 1), Min(0, -s0), m, m + 1, 2 * m, -m]
        cases += [m**2, 1 / m, m * s1, (m + 1) * s1, sympy.Lt(m, s0), Mod(m, 2)]
        cases += [Max(2 * s1, s0 + s1, u0 + zf), Min(u0, Max(s1, FloorDiv(s0, 2)))]
        cases += [Max(u0, int_oo), Min(-int_oo, u0), sympy.Eq(Min(s0, u0), 1)]
        cases += [Max(s0, u0) + Min(s0, u0), Max(s0, u0) * Min(s0, u0)]
        rng = random.Random(0)
        for v in cases:
            self.assertTrue(self.check_expr(v, rng), f"{v}")

    def test_to_int_known(self):
        x = sympy.Symbol("x", real=True)
        big = sympy.Integer(2**62 + 1)
        half = sympy.Rational(5, 2)
        cases = [
            ("CeilToInt", int_oo, int_oo),
            ("CeilToInt", -int_oo, -int_oo),
            ("CeilToInt", big, 2**62),
            ("CeilToInt", sympy.Integer(-(2**63)), -(2**63)),
            ("CeilToInt", sympy.Rational(7, 2), 4),
            ("CeilToInt", sympy.Rational(-5, 2), -2),
            ("CeilToInt", s0, CeilToInt(s0)),
            ("CeilToInt", x / 2, CeilToInt(x / 2)),
            ("FloorToInt", big, big),
            ("FloorToInt", sympy.Integer(2**63 - 1), 2**63 - 1),
            ("FloorToInt", sympy.Rational(-5, 2), -3),
            ("FloorToInt", -int_oo, -int_oo),
            ("FloorToInt", IntTrueDiv(s0, 1), FloorToInt(IntTrueDiv(s0, 1))),
            ("TruncToInt", big, big),
            ("TruncToInt", sympy.Rational(-5, 2), -2),
            ("TruncToInt", sympy.Rational(7, 2), 3),
            ("TruncToInt", int_oo, int_oo),
            ("TruncToInt", s0, s0),
            ("TruncToInt", FloorDiv(s0, 2), FloorDiv(s0, 2)),
            ("TruncToInt", IntTrueDiv(s0, 1), s0),
            ("TruncToInt", IntTrueDiv(x, -1), -x),
            ("TruncToInt", IntTrueDiv(s0, 2), TruncToInt(IntTrueDiv(s0, 2))),
            ("TruncToInt", x, TruncToInt(x)),
            ("RoundToInt", half, 2),
            ("RoundToInt", sympy.Rational(7, 2), 4),
            ("RoundToInt", sympy.Rational(-5, 2), -2),
            ("RoundToInt", sympy.Rational(-7, 2), -4),
            ("RoundToInt", sympy.Rational(-1, 4), 0),
            ("RoundToInt", big, 2**62),
            ("RoundToInt", s0, RoundToInt(s0)),
            ("ToFloat", s0, ToFloat(s0)),
            ("ToFloat", x + 1, ToFloat(x + 1)),
            ("TruncToFloat", x, TruncToFloat(x)),
        ]
        for name, a, expected in cases:
            got = self.check_call(name, sympy.sympify(a))
            self.assertEqual(got, expected, f"{name}({a})")
        cases = [
            ("RoundDecimal", (x, 2), RoundDecimal(x, 2)),
            ("RoundDecimal", (sympy.Integer(3), s0), RoundDecimal(3, s0)),
            ("LShift", (s0, 3), 8 * s0),
            ("LShift", (s0, u0), s0 * PowByNatural(2, u0)),
            ("LShift", (3, 2), 12),
            ("RShift", (s0, 3), FloorDiv(s0, 8)),
            ("RShift", (s0, s1), FloorDiv(s0, PowByNatural(2, s1))),
            ("RShift", (-7, 1), -4),
        ]
        for name, xs, expected in cases:
            got = self.check_call(name, *map(sympy.sympify, xs))
            self.assertEqual(got, expected, f"{name}{xs}")

    def test_float_folds(self):
        F = sympy.Float
        nums = [0, 1, -3, 7, 2**53, 2**53 + 1, 10**15 - 1, 10**15]
        nums += [sympy.Rational(1, 3), sympy.Rational(-7, 2), F(0.5), F(-2.25)]
        nums += [F(0.0), F(1e308), F(1e-300), int_oo, -int_oo, sympy.oo, -sympy.oo]
        nums = list(map(sympy.sympify, nums))
        answered = 0
        for a in nums:
            answered += self.check_call("ToFloat", a) is not None
            for name in ("TruncToFloat", *self.UNARY_FUNCTIONS):
                answered += self.check_call(name, a) is not None
            for b in nums:
                for name in ("FloatTrueDiv", "IntTrueDiv", "FloatPow"):
                    answered += self.check_call(name, a, b) is not None
        self.assertGreater(answered, 700)
        cases = [
            ("ToFloat", (2,), F(2.0)),
            ("ToFloat", (int_oo,), sympy.oo),
            ("IntTrueDiv", (6, 3), F(2.0)),
            ("IntTrueDiv", (1, 3), F(1 / 3)),
            ("IntTrueDiv", (-2, int_oo), F(0.0)),
            ("IntTrueDiv", (int_oo, 2), sympy.oo),
            ("FloatTrueDiv", (sympy.Rational(1, 3), F(0.5)), F(2 / 3)),
            ("FloatTrueDiv", (F(1e308), F(1e-10)), sympy.oo),
            ("FloatPow", (F(2.0), -1074), F(5e-324)),
            ("FloatPow", (-sympy.oo, 3), -sympy.oo),
            ("FloatPow", (0, -sympy.oo), sympy.oo),
            ("FloatPow", (F(-8.0), sympy.Rational(1, 3)), None),
            ("FloatPow", (F(10.0), 400), None),
            ("TruncToFloat", (F(-2.5),), F(-2.0)),
            ("TruncToFloat", (sympy.Rational(7, 2),), F(3.0)),
            ("TruncToInt", (F(0.0),), F(0.0)),
            ("TruncToInt", (F(-2.5),), -2),
            ("RoundToInt", (F(2.5),), 2),
            ("RoundToInt", (-sympy.oo,), -int_oo),
            ("CeilToInt", (F(0.5),), 1),
            ("FloorToInt", (sympy.oo,), int_oo),
            ("RoundDecimal", (F(2.675), 2), F(2.67)),
            ("RoundDecimal", (F(149.5), -2), F(100.0)),
            ("RoundDecimal", (F(250.0), -2), F(200.0)),
            ("RoundDecimal", (int_oo, 2), sympy.oo),
            ("RoundDecimal", (F(1.7976931348623157e308), -308), None),
        ]
        for name, xs, expected in cases:
            got = self.check_call(name, *map(sympy.sympify, xs))
            self.assertEqual(got, expected, f"{name}{xs}")

    def test_round_decimal(self):
        rng = random.Random(0)
        xs = [0.125, 2.675, 149.5, 150.0, 250.0, 5e-324, 1e-310, -3.0, 0.5, 5.0]
        xs += [1.5e308, 1.7976931348623157e308, 123456789.987654321, -0.0]
        xs += [rng.uniform(-1, 1) * 10.0 ** rng.randint(-30, 30) for _ in range(150)]
        xs += [rng.randint(-(10**6), 10**6) / 8 for _ in range(50)]
        ndigits = [-400, -309, -308, -307, -30, -3, -2, -1, 0, 1, 2, 3, 17, 323, 324]
        answered = 0
        for x in xs:
            for n in ndigits + [rng.randint(-20, 20)]:
                r = self.check_call("RoundDecimal", sympy.Float(x), n)
                answered += r is not None
        self.assertGreater(answered, 3000)

    def test_float_int_folds(self):
        F = sympy.Float
        nums = [0, 1, -1, 2, -3, 7, 2**53 + 1, 2**62, sympy.Rational(1, 3)]
        nums += [sympy.Rational(-7, 2), F(0.5), F(-2.25), F(2.0), F(-2.0), F(1.0)]
        nums += [F(-1.0), F(0.0), F(3.5), F(0.1), F(1e308), F(1e-300), F(5e-324)]
        nums += [F(2.0**62), int_oo, -int_oo, sympy.oo, -sympy.oo]
        nums = list(map(sympy.sympify, nums))
        names = ("FloorDiv", "CleanDiv", "Mod", "PythonMod", "PowByNatural")
        answered = 0
        for a in nums:
            for b in nums:
                for name in names:
                    if name == "PowByNatural" and b.is_Integer and abs(b) > 64:
                        continue  # sympy computes the huge Rational power
                    answered += self.check_call(name, a, b) is not None
        self.assertGreater(answered, 1300)
        rng = random.Random(0)
        xs = [F(rng.uniform(-1, 1) * 10.0 ** rng.randint(-20, 20)) for _ in range(40)]
        xs += [F(rng.randint(-(10**6), 10**6) / 64) for _ in range(20)]
        xs += [sympy.Integer(rng.randint(-(2**63), 2**63 - 1)) for _ in range(10)]
        xs += [
            sympy.Rational(rng.randint(-99, 99), rng.randint(2, 99)) for _ in range(10)
        ]
        answered = 0
        for a in xs:
            for b in xs:
                answered += self.check_call("PythonMod", a, b) is not None
        self.assertGreater(answered, 4000)
        cases = [
            ("FloorDiv", (7, F(-1.0)), -7),
            ("FloorDiv", (sympy.oo, F(-0.5)), -int_oo),
            ("FloorDiv", (F(2.5), sympy.oo), 0),
            ("FloorDiv", (F(5.0), 2), None),
            ("PythonMod", (F(1.0), F(0.1)), F(0.0)),
            ("Mod", (F(1.0), F(0.1)), None),
            ("PythonMod", (F(1e308), F(1e-300)), F(0.0)),
            ("Mod", (F(0.0), 7), 0),
            ("Mod", (7, F(2.0)), F(1.0)),
            ("Mod", (F(0.5), 1), 0),
            ("PythonMod", (F(-1e-300), 2), F(2.0)),
            ("PythonMod", (F(-2.25), sympy.Rational(1, 3)), F(1 / 12)),
            ("PythonMod", (sympy.oo, sympy.oo), 0),
            ("PythonMod", (F(0.5), sympy.oo), None),
            ("PowByNatural", (F(0.5), sympy.oo), int_oo),
            ("PowByNatural", (F(-0.5), int_oo), None),
            ("PowByNatural", (F(1.5), 2), F(2.25)),
            ("PythonMod", (F(0.0), sympy.Rational(1, 3)), F(0.0)),
        ]
        for name, args, expected in cases:
            got = self.check_call(name, *map(sympy.sympify, args))
            self.assertEqual(got, expected, f"{name}{args}")

    def test_to_int_unsupported(self):
        arena = torch._C._symbolic._Arena()
        cases = [
            ("RoundToInt", (int_oo,)),
            ("RoundToInt", (-int_oo,)),
            ("CeilToInt", (2**63 - 1,)),
            ("CeilToInt", (sympy.Rational(2**60 + 1, 3),)),
            ("ToFloat", (sympy.Rational(1, 2),)),
            ("TruncToFloat", (int_oo,)),
            ("RoundDecimal", (sympy.Rational(1, 2), sympy.Rational(1, 2))),
            ("IsNonOverlappingAndDenseIndicator", (s0,)),
            ("IsNonOverlappingAndDenseIndicator", (sympy.Rational(1, 2), 3, 1, 1)),
            ("CeilToInt", (s0, s1)),
            ("RoundDecimal", (s0,)),
        ]
        for name, xs in cases:
            args = [arena.from_sympy(sympy.sympify(x)) for x in xs]
            with self.assertRaises(NativeUnsupported, msg=f"{name}{xs}"):
                arena.function(name, args)
        m = sympy.Symbol("m", integer=True, negative=True)
        for f in (arena.lshift, arena.rshift):
            with self.assertRaises(NativeUnsupported):
                f(arena.from_sympy(s0), arena.from_sympy(m))
            with self.assertRaises(NativeUnsupported):
                f(arena.from_sympy(s0), arena.integer(-1))

    def test_is_non_overlapping_and_dense(self):
        F = "IsNonOverlappingAndDenseIndicator"
        cases = [
            ((), 1),
            ((s0, 1), 1),
            ((s0, 2), 0),
            ((1, s0), 1),
            ((sympy.Rational(1, 2), s0), 1),
            ((-int_oo, s0), 1),
            ((int_oo, s0), IsNonOverlappingAndDenseIndicator(int_oo, s0)),
            ((s0, int_oo), IsNonOverlappingAndDenseIndicator(s0, int_oo)),
            ((3, 2), 0),
            ((0, 5), 1),
            ((2, 3, 3, 1), 1),
            ((2, 3, 1, 2), 1),
            ((2, 3, 1, 3), 0),
            ((2, s0, 3, 1, 2, 1), 0),
            ((s0, 2, 2, 1), 1),
            ((2, s0, 1, 2), 1),
            ((s0, 3, 1, s0), IsNonOverlappingAndDenseIndicator(s0, 3, 1, s0)),
            ((s0, 2, 2 * s0, 1), IsNonOverlappingAndDenseIndicator(s0, 2, 2 * s0, 1)),
            ((s0, s1, 2, 1), IsNonOverlappingAndDenseIndicator(s0, s1, 2, 1)),
            ((2**62, 2**62, 2**62, 1, 2**62, 2**62), 0),
            ((1, 2**62, 1, 5, 1, 1), 1),
            ((1, 1, 1, 7, 1, 1), 1),
        ]
        for xs, expected in cases:
            got = self.check_call(F, *map(sympy.sympify, xs))
            self.assertEqual(got, expected, f"{F}{xs}")

    def test_to_int_printing(self):
        x = sympy.Symbol("x", real=True)
        c, t = CeilToInt(x), ToFloat(s0)
        i = IsNonOverlappingAndDenseIndicator(s0, 2, 2 * s0, 1)
        cases = [c, FloorDiv(c, 2), c**2, 2 * c, -c, c + 1, t, t**2, -t, 1 / t]
        cases += [FloorDiv(i, 2), i + 1, 2 * i, RoundDecimal(x, s0) * s0]
        cases += [TruncToInt(x) * FloorToInt(x), Mod(RoundToInt(x), 3)]
        cases += [sympy.Lt(TruncToFloat(x), 2), FloorDiv(s0, TruncToInt(x))]
        rng = random.Random(0)
        for v in cases:
            self.assertTrue(self.check_expr(v, rng), f"{v}")
        items = [CeilToInt(x), FloorToInt(x), ToFloat(s0), ToFloat(s1), i]
        items += [IsNonOverlappingAndDenseIndicator(s0, 3, 1, s0), s0, Mod(s0, 2)]
        items += [RoundDecimal(x, 2), RoundDecimal(x, s0), TruncToFloat(x)]
        items += [RoundToInt(x), TruncToInt(x), FloorDiv(s0, 2), Max(s0, u0)]
        arena = torch._C._symbolic._Arena()
        natives = [arena.from_sympy(v) for v in items]
        for a, na in zip(items, natives):
            self.assertTrue(arena.sort_key(na) == a.sort_key(), f"sort_key({a})")
            for b, nb in zip(items, natives):
                self.assertEqual(arena.compare(na, nb), a.compare(b), f"{a}, {b}")
        got = arena.ordered(natives)
        self.assertEqual([arena.to_sympy(x) for x in got], list(sympy.ordered(items)))

    @parametrize("seed", range(4))
    def test_to_int_fuzz(self, seed):
        rng = random.Random(seed)

        def number():
            k = rng.choice([1, 2, 3, 10, 30, 53, 54, 62, 63])
            p = rng.randint(-(2**k), 2**k)
            q = rng.choice([1, 2, 3, 4, 7, rng.randint(1, 2**k)])
            return sympy.Rational(p, q)

        leaves = FACT_LEAVES + [int_oo, -int_oo]
        leaves += [IntTrueDiv(s0, 1), IntTrueDiv(u0, -1), IntTrueDiv(zf, 2)]
        calls = supported = 0
        nodes = []
        for _ in range(200):
            a = rng.choice(
                [number, lambda: sympy.sympify(sympy_eval(random_tree(rng, 2, leaves)))]
            )()
            b = rng.choice([number, lambda: rng.choice(leaves)])()
            if a.has(sympy.zoo, sympy.nan) or b.has(sympy.zoo, sympy.nan):
                continue
            for name in self.UNARY_FUNCTIONS:
                calls += 1
                r = self.check_call(name, a)
                if r is not None:
                    supported += 1
                    if isinstance(r, self.NODE_TYPES):
                        nodes.append(r)
            for name in ("RoundDecimal", "LShift", "RShift"):
                calls += 1
                r = self.check_call(name, a, b)
                if r is not None:
                    supported += 1
                    if isinstance(r, self.NODE_TYPES):
                        nodes.append(r)
        self.assertGreater(supported, calls // 2)
        pool = [s0, s1, u0, 2 * s0, s0 * s1, s0 + 1, int_oo, sympy.Rational(1, 2)]
        pool += [sympy.Integer(i) for i in (0, 1, 1, 2, 2, 3, 4, 6, 12)]
        for _ in range(200):
            dim = rng.randint(0, 3)
            xs = [rng.choice(pool) for _ in range(2 * dim)]
            calls += 1
            r = self.check_call("IsNonOverlappingAndDenseIndicator", *xs)
            if r is not None:
                supported += 1
                if isinstance(r, self.NODE_TYPES):
                    nodes.append(r)
        self.assertGreater(len(nodes), 20)
        answered = unsupported = 0
        for _ in range(40):
            t = random_tree(rng, 3, LEAVES + nodes)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                if self.check_expr(v, rng):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 5 * unsupported)

    @parametrize("seed", range(4))
    def test_minmax_fuzz(self, seed):
        rng = random.Random(seed)
        u1 = sympy.Symbol("u1", integer=True)
        n = sympy.Symbol("n", integer=True, nonnegative=True)
        m = sympy.Symbol("m", integer=True, negative=True)
        pool = [s0, s1, u0, u1, n, m, zf, *map(sympy.Integer, range(-2, 4))]
        pool += [sympy.Rational(1, 2), int_oo, -int_oo, s0 + s1, u0 + u1, n + m]
        pool += [2 * s0, -s0, s0 - 1, 3 * u0, -2 * n, s0 * s1, FloorDiv(s0, 2)]
        calls = supported = 0
        for _ in range(400):
            name = rng.choice(["Max", "Min"])
            xs = rng.sample(pool, rng.randint(1, 4))
            calls += 1
            r = self.check_call(name, *xs)
            if r is None:
                continue
            supported += 1
            if isinstance(r, (Max, Min)) and len(pool) < 200:
                pool.append(r)
        self.assertGreater(supported, calls * 3 // 4)
        answered = unsupported = 0
        nodes = [x for x in pool if isinstance(x, (Max, Min))]
        for _ in range(40):
            t = random_tree(rng, 3, LEAVES + nodes)
            for sub in subtrees(t):
                v = sympy.sympify(sympy_eval(sub))
                if v.has(sympy.zoo, sympy.nan):
                    continue
                if self.check_expr(v, rng):
                    answered += 1
                else:
                    unsupported += 1
        self.assertGreater(answered, 5 * unsupported)


class TestNativeValueRanges(TestCase):
    PYTHON_ERRORS = (AssertionError, TypeError, ValueError, ValueRangeError)
    PYTHON_ERRORS += (RecursionError, ZeroDivisionError, OverflowError)
    u1 = sympy.Symbol("u1", integer=True)
    n = sympy.Symbol("n", integer=True, nonnegative=True)
    INT_LEAVES = [s0, s1, u0, u1, n, *map(sympy.Integer, range(-3, 5))]
    RELATIONALS = [sympy.Eq, sympy.Ne, sympy.Lt, sympy.Le, sympy.Gt, sympy.Ge]
    BOUNDS = [-int_oo, int_oo, *map(sympy.Integer, [-7, -3, -2, -1, 0, 1, 2, 3, 5, 9])]

    def check(self, e, ranges):
        """Returns the Python range, or None if either side raised."""
        arena = torch._C._symbolic._Arena()
        msg = f"{e} {ranges}"
        try:
            n = arena.from_sympy(e)
            native_ranges = [
                (arena.from_sympy(s), arena.from_sympy(lo), arena.from_sympy(hi))
                for s, (lo, hi) in ranges.items()
            ]
        except NativeUnsupported:
            return None
        env = {s: ValueRanges(lo, hi) for s, (lo, hi) in ranges.items()}
        try:
            expected = sympy_interp(
                SymPyValueRangeAnalysis, env, e, missing_handler=_default_symbol_range
            )
        except self.PYTHON_ERRORS:
            expected = None
        if expected is None:
            with self.assertRaises(NativeUnsupported, msg=msg):
                arena.value_range(n, native_ranges)
            return None
        try:
            lo, hi = arena.value_range(n, native_ranges)
        except NativeUnsupported:
            return None
        got = (arena.to_sympy(lo), arena.to_sympy(hi))
        self.assertEqual(got, (expected.lower, expected.upper), msg)
        want_types = (type(expected.lower), type(expected.upper))
        self.assertEqual(tuple(map(type, got)), want_types, msg)
        return expected

    def test_known(self):
        u1, n = self.u1, self.n
        oo = int_oo
        cases = [
            (s0, {}, (1, oo)),
            (n, {}, (0, oo)),
            (u0, {}, (-oo, oo)),
            (zf, {}, (-sympy.oo, sympy.oo)),
            (u0, {u0: (2, 5)}, (2, 5)),
            (s0 + u0, {u0: (-3, 5)}, (-2, oo)),
            (s0 * u0, {u0: (-3, 5)}, (-oo, oo)),
            (2 * u0 - 1, {u0: (-3, 5)}, (-7, 9)),
            (u0 + u1, {u0: (oo, oo), u1: (-oo, 0)}, None),
            (u0 * u1, {u0: (0, oo), u1: (-oo, 0)}, (-oo, 0)),
            (Mod(u0, 3), {u0: (0, oo)}, (0, 2)),
            (Mod(u0, 3), {u0: (oo, oo)}, None),
            (Mod(u0, -3), {u0: (-oo, 5)}, (-2, 2)),
            (Mod(u0, 3), {u0: (-7, -7)}, (-1, -1)),
            (Mod(u0, 3), {u0: (-2, 2)}, (-2, 2)),
            (Mod(u0, 3), {u0: (-2, 7)}, (-2, 2)),
            (Mod(u0, u1), {u1: (2, 5)}, (-4, 4)),
            (FloorDiv(u0, u1), {u1: (1, oo)}, (-oo, oo)),
            (FloorDiv(u0, u1), {u0: (0, 9), u1: (0, 3)}, (0, oo)),
            (FloorDiv(u0, 2), {u0: (-7, 9)}, (-4, 4)),
            (FloorDiv(u0, u1), {u0: (oo, oo), u1: (oo, oo)}, (oo, oo)),
            (PowByNatural(u0, n), {u0: (-3, 5)}, None),
            (PowByNatural(u0, u1), {u0: (1, 5), u1: (-2, 3)}, (1, 125)),
            (PowByNatural(u0, n), {u0: (1, oo)}, (1, oo)),
            (PowByNatural(u0, u1), {u0: (2, 2), u1: (oo, oo)}, None),
            (u0**2, {u0: (-oo, 2)}, (0, oo)),
            (u0**2, {u0: (-3, -2)}, (4, 9)),
            (u0**3, {u0: (-oo, -oo)}, (-oo, -oo)),
            (u0**3, {u0: (-3, 2)}, (-27, 8)),
            (u0**-1, {u0: (1, 2)}, (-sympy.oo, sympy.oo)),
            (Max(u0, u1, 3), {u0: (-oo, 0), u1: (1, 5)}, (3, 5)),
            (Min(u0, 3), {u0: (-oo, oo)}, (-oo, 3)),
            (TruncToInt(u0), {u0: (-3, oo)}, (-3, oo)),
            (RoundToInt(u0), {u0: (-3, 4)}, (-3, 4)),
            (RoundToInt(u0), {u0: (-3, oo)}, None),
            (FloorToInt(u0), {u0: (-3, oo)}, (-3, oo)),
            (ToFloat(u0), {}, (-sympy.oo, sympy.oo)),
            (sympy.Eq(u0, 3), {u0: (4, 6)}, (False, False)),
            (sympy.Eq(u0, u1), {u0: (4, 4), u1: (4, 4)}, (True, True)),
            (sympy.Ne(u0, 3), {u0: (0, 6)}, (False, True)),
            (sympy.Lt(u0, u1), {u0: (0, 3), u1: (4, 6)}, (True, True)),
            (sympy.Le(u0, u1), {u0: (4, 6), u1: (0, 4)}, (False, True)),
            (sympy.Gt(u0, u1), {u0: (4, 6), u1: (0, 4)}, (False, True)),
            (sympy.Ge(u0, u1), {u0: (4, 6), u1: (0, 4)}, (True, True)),
            (sympy.And(u0 < 3, u0 >= 0), {u0: (0, 2)}, (True, True)),
            (sympy.Or(u0 < 3, u1 > 0), {u0: (5, 6)}, (False, True)),
            (sympy.Not(sympy.Eq(u0, 0)), {u0: (1, 6)}, (True, True)),
        ]
        for e, ranges, want in cases:
            ranges = {s: tuple(map(sympy.sympify, r)) for s, r in ranges.items()}
            got = self.check(e, ranges)
            msg = f"{e} {ranges}"
            if want is None:
                self.assertIsNone(got, msg)
            else:
                self.assertIsNotNone(got, msg)
                want = tuple(map(sympy.sympify, want))
                self.assertEqual((got.lower, got.upper), want, msg)

    def test_invalid_ranges(self):
        arena = torch._C._symbolic._Arena()
        x = arena.from_sympy(u0)
        for lo, hi in [(3, 2), (int_oo, 0), (sympy.Rational(1, 2), 0), (0, sympy.true)]:
            bounds = [arena.from_sympy(sympy.sympify(b)) for b in (lo, hi)]
            with self.assertRaises(NativeUnsupported):
                arena.value_range(x, [(x, *bounds)])

    def int_expr(self, rng, depth):
        if depth == 0 or rng.random() < 0.25:
            return rng.choice(self.INT_LEAVES)
        a = self.int_expr(rng, depth - 1)
        b = self.int_expr(rng, depth - 1)
        op = rng.choice(
            ["add", "mul", "sub", "pow", "FloorDiv", "Mod", "PythonMod", "Max"]
            + ["Min", "PowByNatural", "CeilToInt", "TruncToInt", "RoundToInt"]
        )
        if op == "add":
            return a + b
        if op == "mul":
            return a * b
        if op == "sub":
            return a - b
        if op == "pow":
            return a ** rng.randint(0, 3)
        if op in ("CeilToInt", "TruncToInt", "RoundToInt"):
            return TestNativeFunctions.UNARY_FUNCTIONS[op](a)
        return TestNativeFunctions.FUNCTIONS[op](a, b)

    def bool_expr(self, rng, depth):
        if depth == 0 or rng.random() < 0.3:
            op = rng.choice(self.RELATIONALS)
            return op(self.int_expr(rng, 2), self.int_expr(rng, 2))
        op = rng.choice([sympy.And, sympy.Or, sympy.Not])
        if op is sympy.Not:
            return op(self.bool_expr(rng, depth - 1))
        return op(*(self.bool_expr(rng, depth - 1) for _ in range(rng.randint(2, 3))))

    @parametrize("seed", range(4))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        symbols = [s0, s1, u0, self.u1, self.n]
        calls = answered = 0
        for _ in range(600):
            try:
                if rng.random() < 0.7:
                    e = self.int_expr(rng, 3)
                else:
                    e = self.bool_expr(rng, 2)
            except (ZeroDivisionError, ValueError, TypeError, AssertionError):
                continue
            if e.has(sympy.zoo, sympy.nan):
                continue
            ranges = {}
            for s in rng.sample(symbols, rng.randint(0, len(symbols))):
                ranges[s] = tuple(sorted(rng.sample(self.BOUNDS, 2), key=float))
            calls += 1
            if self.check(e, ranges) is not None:
                answered += 1
        self.assertGreater(answered, calls // 3)

    def test_float_known(self):
        u1, oo, half = self.u1, sympy.oo, sympy.Rational(1, 2)
        zg = sympy.Symbol("zg", real=True)
        cases = [
            (half, {}, (half, half)),
            (sympy.Float(0.5), {}, (0.5, 0.5)),
            (oo, {}, (oo, oo)),
            (-zf, {}, (-oo, oo)),
            (s0 / 2, {}, (half, int_oo)),
            (0.5 * s0, {}, (0.5, oo)),
            (s0 + 1.5, {}, (2.5, oo)),
            (1 / s0, {}, (-oo, oo)),
            (u0**-2 + 1, {u0: (1, 3)}, (-oo, oo)),
            (Max(u0, half), {u0: (0, 3)}, (half, 3)),
            (Max(u0, zf), {u0: (0, 3), zf: (-0.5, 2.0)}, (0, 3)),
            (Max(zg, 1), {}, (1, int_oo)),
            (Min(zg, 1), {}, (-int_oo, 1)),
            (Max(u0, zf), {u0: (2, 2), zf: (2.0, 2.0)}, None),
            (u0 * zf, {u0: (0, 0)}, (0, 0)),
            (zf + u0, {zf: (0.5, 2.5), u0: (-1, 1)}, (-0.5, 3.5)),
            (u0 * zf, {zf: (-0.5, 2.0), u0: (-1, 1)}, (-2.0, 2.0)),
            (u0 * zf, {zf: (0.0, 2.0), u0: (1, int_oo)}, (0.0, oo)),
            (u0 * u1, {u0: (-half, oo), u1: (0, 2)}, (-1, int_oo)),
            (zf, {zf: (0, oo)}, (0, int_oo)),
            (sympy.Lt(u0, zf), {u0: (0, 1), zf: (2.5, oo)}, (True, True)),
            (sympy.Eq(zf, 1), {zf: (2.5, 3.0)}, (False, False)),
            (FloorDiv(u0, 2) + zf, {}, (-oo, oo)),
            (FloorDiv(zf, 2), {}, (-int_oo, int_oo)),
        ]
        for e, ranges, want in cases:
            ranges = {s: tuple(map(sympy.sympify, r)) for s, r in ranges.items()}
            got = self.check(e, ranges)
            msg = f"{e} {ranges}"
            if want is None:
                self.assertIsNone(got, msg)
            else:
                self.assertIsNotNone(got, msg)
                self.assertEqual(
                    (got.lower, got.upper), tuple(map(sympy.sympify, want)), msg
                )

    def test_float_bound_ops(self):
        zg = sympy.Symbol("zg", real=True)
        nums = [0, 2, -3, sympy.Rational(1, 2), sympy.Rational(-1, 3), 0.5, -2.0, 0.0]
        nums = [*map(sympy.sympify, nums), int_oo, -int_oo, sympy.oo, -sympy.oo]
        answered = 0
        for x in nums:
            for y in nums:
                ranges = {zf: (x, x), zg: (y, y)}
                for e in [zf + zg, zf * zg, Max(zf, zg), Min(zf, zg)]:
                    answered += self.check(e, ranges) is not None
        self.assertGreater(answered, 4 * len(nums) ** 2 * 3 // 4)

    def test_float_handlers_known(self):
        u1, oo, zg = self.u1, sympy.oo, sympy.Symbol("zg", real=True)
        cases = [
            (FloatTrueDiv(u0, 2), {u0: (1, 7)}, (0.5, 3.5)),
            (IntTrueDiv(u0, 2), {u0: (1, 7)}, (0.5, 3.5)),
            (IntTrueDiv(u0, u1), {u0: (1, 7), u1: (-1, 2)}, (-oo, oo)),
            (IntTrueDiv(u0, u1), {u0: (1, int_oo), u1: (2, int_oo)}, (-oo, oo)),
            (IntTrueDiv(u0, u1), {u0: (1, int_oo), u1: (2, 3)}, (1 / 3, oo)),
            (FloatTrueDiv(s0, 2), {}, (0.5, oo)),
            (FloatTrueDiv(u0, zf), {u0: (1, 4), zf: (2.0, oo)}, (0.0, 2.0)),
            (FloatTrueDiv(u0, zf), {zf: (2.0, oo)}, None),
            (FloatTrueDiv(u0, zf), {u0: (-int_oo, int_oo), zf: (2.0, oo)}, None),
            (FloatTrueDiv(zg, zf), {zf: (2.0, oo)}, (-oo, oo)),
            (ToFloat(u0), {u0: (0, int_oo)}, (0.0, oo)),
            (ToFloat(u0), {}, (-oo, oo)),
            (FloorToInt(zf), {zf: (0.5, 2.5)}, (0, 2)),
            (CeilToInt(zf), {zf: (0.5, oo)}, (1, int_oo)),
            (CeilToInt(zg), {}, (-oo, oo)),
            (FloorToInt(u0 / 2), {u0: (-7, 7)}, (-4, 3)),
            (CeilToInt(u0 / 2), {u0: (-7, 7)}, (-3, 4)),
            (CeilToInt(zf), {zf: (1e20, oo)}, None),
            (TruncToInt(zf), {zf: (-2.5, 2.5)}, (-2, 2)),
            (RoundToInt(zf), {zf: (0.5, 2.5)}, (0, 2)),
            (TruncToFloat(zf), {zf: (-2.5, oo)}, (-2.0, oo)),
            (RoundDecimal(zf, 1), {zf: (0.25, 1.5)}, (0.2, 1.5)),
            (RoundDecimal(zf, u0), {zf: (0.5, 1.5), u0: (0, 2)}, (-oo, oo)),
            (FloatPow(zf, 2), {}, (-oo, oo)),
        ]
        for e, ranges, want in cases:
            ranges = {s: tuple(map(sympy.sympify, r)) for s, r in ranges.items()}
            got = self.check(e, ranges)
            msg = f"{e} {ranges}"
            if want is None:
                self.assertIsNone(got, msg)
            else:
                self.assertIsNotNone(got, msg)
                want = tuple(map(sympy.sympify, want))
                self.assertEqual((got.lower, got.upper), want, msg)

    def test_float_handler_ops(self):
        zg = sympy.Symbol("zg", real=True)
        half, m7_2 = sympy.Rational(1, 2), sympy.Rational(-7, 2)
        nums = [-3, 0, 2, half, m7_2, -0.5, 2.5, 0.0, 1e20]
        nums = [*map(sympy.sympify, nums), int_oo, -int_oo, sympy.oo, -sympy.oo]
        pairs = [(lo, hi) for lo in nums for hi in nums if lo <= hi]
        unary = [*TestNativeFunctions.UNARY_FUNCTIONS.values()]
        answered = 0
        for x in pairs:
            for f in unary:
                answered += self.check(f(zf), {zf: x}) is not None
            for n in [-1, 0, 1, 2]:
                answered += self.check(RoundDecimal(zf, n), {zf: x}) is not None
            for y in pairs:
                ranges = {zf: x, zg: y}
                for f in [FloatTrueDiv, IntTrueDiv]:
                    answered += self.check(f(zf, zg), ranges) is not None
        self.assertGreater(answered, len(pairs) ** 2)

    def test_modular_indexing(self):
        u1, n = self.u1, self.n
        exprs = [
            ModularIndexing(u0, 2, 8),
            ModularIndexing(u0, -2, 5),
            ModularIndexing(u0 + u1, n, 3),
            ModularIndexing(n, 4, u1),
            ModularIndexing(u0, u1, n),
        ]
        rng = random.Random(0)
        answered = 0
        for e in exprs:
            for _ in range(200):
                ranges = {}
                for s in sorted(e.free_symbols, key=str):
                    lo, hi = sorted(rng.sample(self.BOUNDS, 2))
                    ranges[s] = (lo, hi)
                answered += self.check(e, ranges) is not None
        self.assertGreater(answered, 500)

    def test_identity(self):
        u1, n = self.u1, self.n
        exprs = [Identity(u0), Identity(u0) + 1, Identity(u0 + u1) * n, Identity(zf)]
        rng = random.Random(0)
        answered = 0
        for e in exprs:
            for _ in range(100):
                ranges = {}
                for s in sorted(e.free_symbols, key=str):
                    lo, hi = sorted(rng.sample(self.BOUNDS, 2))
                    ranges[s] = (lo, hi)
                answered += self.check(e, ranges) is not None
        self.assertGreater(answered, 300)

    def test_bitwise(self):
        zg = sympy.Symbol("zg", real=True)
        t, f = sympy.true, sympy.false
        nums = [-(2**63), -9, -3, -1, 0, 1, 2, 5, 2**63 - 1, sympy.Rational(-7, 2)]
        nums += [sympy.Rational(-(2**63), 3)]
        nums += [sympy.Rational(1, 2), -0.5, -1e-300, 0.0, 2.5, 1e20]
        nums = [*map(sympy.sympify, nums), int_oo, -int_oo, sympy.oo, -sympy.oo]
        pairs = [(lo, hi) for lo in nums for hi in nums if lo <= hi]
        bools = [(f, f), (f, t), (t, t)]
        pairs += bools
        rng = random.Random(0)
        answered = 0
        for fn in TestNativeFunctions.BITWISE:
            for x in pairs:
                for y in bools + rng.sample(pairs, 30):
                    answered += self.check(fn(zf, zg), {zf: x, zg: y}) is not None
        self.assertGreater(answered, 80 * len(pairs))
        r = sympy.Rational(-(2**63), 3)
        e = BitwiseFn_bitwise_and(zf, zg)
        self.assertEqual(self.check(e, {zf: (r, 0), zg: (r, 0)}), ValueRanges(-(2**62), 0))

    def test_float_int_handlers_known(self):
        half, oo, zg = sympy.Rational(1, 2), sympy.oo, sympy.Symbol("zg", real=True)
        cases = [
            (zf**2, {zf: (half, 3 * half)}, (half**2, 9 * half**2)),
            (zf**2, {zf: (-half, 3 * half)}, (0, 9 * half**2)),
            (zf**2, {zf: (-0.5, 1.5)}, (0.0, 2.25)),
            (zf**3, {zf: (-0.5, 1.5)}, (-0.125, 3.375)),
            (zf**2, {zf: (-oo, oo)}, (0, int_oo)),
            (zf**3, {}, (-int_oo, int_oo)),
            (zf**2, {zf: (half, half)}, (half**2, half**2)),
            (Mod(u0, zg), {u0: (1, 7), zg: (half, 2)}, (-1, 1)),
            (Mod(u0, zg), {zg: (2.5, oo)}, (-oo, oo)),
            (Mod(u0, zg), {zg: (-0.5, 2.0)}, (-int_oo, int_oo)),
            (Mod(u0, zg), {zg: (0.25, 0.5)}, None),
            (FloorDiv(u0, zg), {u0: (1, 7), zg: (0.0, 2.0)}, (0, int_oo)),
            (FloorDiv(zg, u0), {zg: (-oo, -0.5), u0: (-3, 0)}, (0, int_oo)),
            (FloorDiv(zg, u0), {zg: (2.5, oo), u0: (1, 3)}, None),
            (FloorDiv(zg, u0), {zg: (-oo, oo), u0: (1, 3)}, (-int_oo, int_oo)),
            (PowByNatural(zg, u0), {zg: (1.5, 2.5), u0: (0, int_oo)}, (1, int_oo)),
            (PowByNatural(zg, u0), {zg: (1.5, oo), u0: (0, int_oo)}, None),
        ]
        for e, ranges, want in cases:
            ranges = {s: tuple(map(sympy.sympify, r)) for s, r in ranges.items()}
            got = self.check(e, ranges)
            msg = f"{e} {ranges}"
            if want is None:
                self.assertIsNone(got, msg)
            else:
                self.assertIsNotNone(got, msg)
                want = tuple(map(sympy.sympify, want))
                self.assertEqual((got.lower, got.upper), want, msg)

    def test_float_int_handler_ops(self):
        zg = sympy.Symbol("zg", real=True)
        half, m7_2 = sympy.Rational(1, 2), sympy.Rational(-7, 2)
        nums = [-3, 0, 1, 2, half, m7_2, -0.5, 1.5, 0.0]
        nums = [*map(sympy.sympify, nums), int_oo, -int_oo, sympy.oo, -sympy.oo]
        pairs = [(lo, hi) for lo in nums for hi in nums if lo <= hi]
        int_pairs = [(lo, hi) for lo, hi in pairs if lo.is_integer and hi.is_integer]
        answered = 0
        for x in pairs:
            for n in range(4):
                answered += self.check(zf**n, {zf: x}) is not None
            for y in int_pairs:
                answered += self.check(PowByNatural(zf, u0), {zf: x, u0: y}) is not None
                answered += self.check(FloorDiv(u0, zf), {zf: x, u0: y}) is not None
                answered += self.check(Mod(u0, zf), {zf: x, u0: y}) is not None
            for y in pairs:
                ranges = {zf: x, zg: y}
                for f in [FloorDiv, Mod]:
                    answered += self.check(f(zf, zg), ranges) is not None
        self.assertGreater(answered, len(pairs) ** 2)

    @parametrize("seed", range(4))
    def test_float_fuzz(self, seed):
        rng = random.Random(seed)
        leaves = [s0, u0, self.n, zf, *map(sympy.Integer, range(-2, 3))]
        leaves += [sympy.Rational(1, 2), sympy.Rational(-1, 3), sympy.Float(0.5)]
        leaves += [sympy.Float(-2.0), sympy.Float(0.0)]
        bounds = [-int_oo, -sympy.oo, int_oo, sympy.oo, *map(sympy.Integer, [-3, 0, 2])]
        bounds += [sympy.Rational(1, 2), sympy.Float(-0.5), sympy.Float(2.0)]

        def expr(depth):
            if depth == 0 or rng.random() < 0.25:
                return rng.choice(leaves)
            a, b = expr(depth - 1), expr(depth - 1)
            op = rng.choice(["add", "mul", "sub", "pow", "Max", "Min", "Lt", "unary"])
            if rng.random() < 0.2:
                op = rng.choice(["FloatTrueDiv", "IntTrueDiv"])
            if op == "unary":
                return rng.choice([*TestNativeFunctions.UNARY_FUNCTIONS.values()])(a)
            if op == "add":
                return a + b
            if op == "mul":
                return a * b
            if op == "sub":
                return a - b
            if op == "pow":
                return a ** rng.choice([-1, 2, -2])
            if op == "Lt":
                return sympy.Lt(a, b)
            return TestNativeFunctions.FUNCTIONS[op](a, b)

        calls = answered = 0
        for _ in range(500):
            try:
                e = expr(3)
            except (ZeroDivisionError, ValueError, TypeError, AssertionError):
                continue
            if e.has(sympy.zoo, sympy.nan) or isinstance(
                e, sympy.logic.boolalg.BooleanAtom
            ):
                continue
            ranges = {}
            for s in rng.sample([u0, zf], rng.randint(0, 2)):
                lo, hi = sorted(rng.sample(bounds, 2), key=float)
                if lo <= hi:
                    ranges[s] = (lo, hi)
            calls += 1
            answered += self.check(e, ranges) is not None
        self.assertGreater(answered, calls // 3)

    def check_bound(self, e, ranges, context_ranges=None):
        """Differential bound_sympy; returns the Python range, or None if either side raised."""
        context_ranges = context_ranges or {}
        arena = torch._C._symbolic._Arena()
        msg = f"{e} {ranges} {context_ranges}"

        def to_native(rs):
            return [
                (arena.from_sympy(s), arena.from_sympy(lo), arena.from_sympy(hi))
                for s, (lo, hi) in rs.items()
            ]

        try:
            n = arena.from_sympy(e)
            native_ranges = to_native(ranges)
            native_context = to_native(context_ranges)
        except NativeUnsupported:
            return None
        merged = {**context_ranges, **ranges}
        env = {s: ValueRanges(lo, hi) for s, (lo, hi) in merged.items()}
        try:
            expected = bound_sympy(e, env)
        except self.PYTHON_ERRORS:
            expected = None
        if expected is None:
            with self.assertRaises(NativeUnsupported, msg=msg):
                arena.bound_sympy(n, native_ranges, native_context)
            return None
        try:
            lo, hi = arena.bound_sympy(n, native_ranges, native_context)
        except NativeUnsupported:
            return None
        got = (arena.to_sympy(lo), arena.to_sympy(hi))
        self.assertEqual(got, (expected.lower, expected.upper), msg)
        want_types = (type(expected.lower), type(expected.upper))
        self.assertEqual(tuple(map(type, got)), want_types, msg)
        return expected

    def test_bound_sympy_known(self):
        u1, n = self.u1, self.n
        oo = int_oo
        r = {u0: (2, oo), u1: (1, oo)}
        cases = [
            (sympy.Integer(3), {}, {}, (3, 3)),
            (int_oo, {}, {}, (oo, oo)),
            (sympy.Rational(1, 2), {}, {}, (sympy.Rational(1, 2),) * 2),
            (u0, {}, {u0: (2, 5)}, (2, 5)),
            (u0, {u0: (3, 4)}, {u0: (2, 5)}, (3, 4)),
            (u0 + u1, {u0: (3, 4)}, {u1: (2, 5)}, (5, 9)),
            (u0 - Mod(u0, 8), r, {}, (0, oo)),
            (u0 - Mod(u0, 8), {}, r, (0, oo)),
            (1 + u0 - Mod(u0, 8), r, {}, (1, oo)),
            (2 * u0 - 2 * Mod(u0, 8), r, {}, (0, oo)),
            (u0 - Mod(u0, u1), r, {}, (0, oo)),
            ((2 * u0 + 1) - Mod(2 * u0 + 1, 8), r, {}, (0, oo)),
            (Mod(u0, 8) - u0, r, {}, (-oo, 0)),
            (u0 - Mod(u0, 8), {u0: (0, 20)}, {}, (0, 16)),
            (u0 - Mod(u0, 8), {u0: (-2, oo)}, {}, (-9, oo)),
            (u0 - Mod(u0, u1), {u1: (0, 5)}, {u0: (0, 9)}, (-oo, oo)),
            (u0 - Mod(u0, n), {u0: (0, 9)}, {}, (-oo, oo)),
            (s0 - Mod(s0, 8), {}, {}, (0, oo)),
            (3 * u0 - Mod(u0, 8), r, {}, (4, oo)),
            (u0 - 2 * Mod(u0, 8), r, {}, (-12, oo)),
            (u0 + u1 - Mod(u0 + u1, 4), r, {}, (0, oo)),
            (u0 - Mod(u0 + u1, 4), r, {}, (-1, oo)),
            (Max(u0 - Mod(u0, 8), 1), r, {}, (1, oo)),
            (sympy.Ge(u0 - Mod(u0, 8), 0), r, {}, (True, True)),
            (u0 - Mod(u0, 8) - Mod(u1, 3) + u1, r, {}, (0, oo)),
            (FloorDiv(u0 - Mod(u0, 8), 8), r, {}, (0, oo)),
        ]
        for e, ranges, context, want in cases:
            ranges = {s: tuple(map(sympy.sympify, b)) for s, b in ranges.items()}
            context = {s: tuple(map(sympy.sympify, b)) for s, b in context.items()}
            got = self.check_bound(e, ranges, context)
            msg = f"{e} {ranges} {context}"
            if want is None:
                self.assertIsNone(got, msg)
            else:
                self.assertIsNotNone(got, msg)
                want = tuple(map(sympy.sympify, want))
                self.assertEqual((got.lower, got.upper), want, msg)

    @parametrize("seed", range(4))
    def test_bound_sympy_fuzz(self, seed):
        rng = random.Random(seed)
        u1, n = self.u1, self.n
        symbols = [s0, s1, u0, u1, n]
        terms = [*symbols, s0 * u0, n**2, sympy.Integer(1)]
        divisors = [*map(sympy.Integer, [-3, 1, 2, 3, 8]), s0, s1, u1, n, s1 + 1]

        def linear(k):
            return sympy.Add(*(rng.randint(-2, 3) * t for t in rng.sample(terms, k)))

        calls = answered = rewritten = 0
        for _ in range(500):
            e = linear(rng.randint(0, 2))
            for _ in range(rng.randint(1, 2)):
                base = linear(rng.randint(1, 3))
                j = rng.choice([-2, -1, 1, 1, 2, 3])
                k = j if rng.random() < 0.7 else rng.choice([-2, -1, 1, 2])
                try:
                    e = e + j * base - k * Mod(base, rng.choice(divisors))
                except (ZeroDivisionError, TypeError, AssertionError):
                    continue
            if rng.random() < 0.2:
                e = rng.choice([Max(e, 0), sympy.Ge(e, 1), FloorDiv(e, 2)])
            ranges, context = {}, {}
            for s in symbols:
                roll = rng.random()
                target = ranges if roll < 0.4 else context if roll < 0.6 else None
                if target is not None:
                    target[s] = tuple(sorted(rng.sample(self.BOUNDS, 2), key=float))
            env = {s: ValueRanges(*b) for s, b in {**context, **ranges}.items()}
            try:
                rewritten += _rewrite_for_value_range_analysis(e, env) != e
            except self.PYTHON_ERRORS:
                pass
            calls += 1
            if self.check_bound(e, ranges, context) is not None:
                answered += 1
        self.assertGreater(answered, calls // 3)
        self.assertGreater(rewritten, calls // 10)


class TestNativeStaticPasses(TestCase):
    u1 = sympy.Symbol("u1", integer=True)
    INT_LEAVES = [s0, s1, u0, u1, *map(sympy.Integer, range(-3, 5))]
    RELATIONALS = [sympy.Eq, sympy.Ne, sympy.Lt, sympy.Le, sympy.Gt, sympy.Ge]

    def assertSameTree(self, got, want, msg):
        self.assertIs(type(got), type(want), msg)
        self.assertEqual(got, want, msg)
        self.assertEqual(len(got.args), len(want.args), msg)
        for g, w in zip(got.args, want.args):
            self.assertSameTree(g, w, msg)

    def check(self, name, e):
        """Differential test of arena.<name> against symbolic_shapes.<name>; returns whether native answered."""
        arena = torch._C._symbolic._Arena()
        try:
            n = arena.from_sympy(e)
        except NativeUnsupported:
            return False
        try:
            want = getattr(symbolic_shapes, name)(e)
        except (AssertionError, TypeError, ValueError, RecursionError):
            with self.assertRaises(NativeUnsupported, msg=str(e)):
                getattr(arena, name)(n)
            return False
        try:
            got = arena.to_sympy(getattr(arena, name)(n))
        except NativeUnsupported:
            return False
        self.assertSameTree(got, want, f"{name}({e})")
        return True

    def test_safe_expand_known(self):
        u1 = self.u1
        cases = [
            (s0 + 1) ** 2,
            (s0 + s1 + u0) ** 3,
            (2 * s0 - 3 * s1) ** 4,
            (s0 + s1 + u0 + 1) ** 6,
            (s0 + 1) ** 20,
            (s0 + 1) ** -1,
            (s0 + u1) ** -3,
            (s0 + 1) * (s1 + 2),
            (s0 + 1) * (s1 + 2) * u0,
            (s0 + 1) * (s1 + 2) / (u0 + 3),
            (s0 + 1) / ((s1 + 2) * (u0 - 1)),
            (s0 + 1) / (s1 + 2),
            s0 / (u1 * (s1 + 1)),
            (s0 + 1) ** 2 * (s1 - 1),
            ((s0 + 1) ** 2 + u0) ** 2,
            FloorDiv((s0 + 1) ** 2, 2),
            Max((s0 + 1) * (s1 + 1), 3),
            Mod((s0 + u1) * (s1 + 1), s0 + 2),
            sympy.Eq((s0 + 1) ** 2, u1),
            sympy.Lt((s0 + 1) * (s1 - 1), u1, evaluate=False),
            sympy.And(sympy.Eq((s0 + 1) ** 2, u1), sympy.Lt(u0, 3)),
            sympy.Or(sympy.Eq((s0 + 1) ** 2, u1), sympy.Lt(u0, 3)),
            sympy.true,
            s0,
            sympy.Integer(3),
        ]
        for e in cases:
            self.assertTrue(self.check("safe_expand", e), str(e))

    def test_canonicalize_bool_expr_known(self):
        u1 = self.u1
        a, b, c = sympy.Gt(u0, 1), sympy.Le(u1, 2), sympy.Eq(u0, s0 + 1)
        cases = [
            sympy.Gt(s0, s1),
            sympy.Ge(2 * u0, 4 * u1 + 6),
            sympy.Eq(6 * u0 + 4 * u1, 2),
            sympy.Eq(-6 * u0, 4 * u1 - 2),
            sympy.Ne(u0, -3),
            sympy.Lt(u0, 0),
            sympy.Le(-3 * u0, 0),
            sympy.Lt(3, u0 + u1),
            sympy.Lt(u0, u0, evaluate=False),
            sympy.Ge(s0 * u0 - 2 * s1, 4 * u1),
            sympy.And(a, b),
            sympy.And(a, sympy.Or(b, c)),
            sympy.Or(sympy.And(a, b), c),
            sympy.Or(sympy.And(a, b), sympy.And(c, sympy.Ne(u1, 0))),
            sympy.Not(sympy.And(a, b)),
            sympy.Not(sympy.Or(a, sympy.And(b, c))),
            sympy.Eq(sympy.Lt(u0, 3), sympy.true, evaluate=False),
            sympy.Ne(sympy.Gt(u0, 3), sympy.Lt(u1, u0), evaluate=False),
            sympy.true,
            s0 + 1,
        ]
        for e in cases:
            self.assertTrue(self.check("canonicalize_bool_expr", e), str(e))
        f = sympy.Float
        cases = [sympy.Gt(s0, f(0.5)), sympy.Ge(2 * u0, f(0.5) * u1 + 3)]
        cases += [sympy.Eq(f(0.5) * u0, 2), sympy.Lt(u0, f(-1.5))]
        cases += [sympy.Le(u0 - f(2.5), s0)]
        cases += [sympy.And(sympy.Gt(u0, f(0.5)), b), sympy.Lt(f(-2.0) * u0, 4 * u1)]
        for e in cases:
            self.assertTrue(self.check("canonicalize_bool_expr", e), str(e))
        # Or(a, b) stays an Or but Or(b, a) is true, so to_nnf's Or(*set) depends
        # on hash order.
        a = sympy.Eq(-2 * u0 - 4, -4 * u0 - 4)
        b = sympy.Ne(2 * u0 + 4, 4 * u0 + 4)
        c = sympy.And(sympy.Ge(-3 * u1, -6 * u1), sympy.Gt(3 * u1, 6 * u1))
        arena = torch._C._symbolic._Arena()
        for e in [sympy.Or(a, b), sympy.Not(c)]:
            with self.assertRaises(NativeUnsupported):
                arena.canonicalize_bool_expr(arena.from_sympy(e))

    def int_expr(self, rng, depth):
        if depth == 0 or rng.random() < 0.25:
            return rng.choice(self.INT_LEAVES)
        a = self.int_expr(rng, depth - 1)
        b = self.int_expr(rng, depth - 1)
        op = rng.choice(["add", "add", "mul", "mul", "sub", "pow", "FloorDiv", "Max"])
        if op == "add":
            return a + b
        if op == "mul":
            return a * b
        if op == "sub":
            return a - rng.randint(1, 3) * b
        if op == "pow":
            return a ** rng.choice([-2, -1, 2, 3])
        return TestNativeFunctions.FUNCTIONS[op](a, b)

    def bool_expr(self, rng, depth):
        if depth == 0 or rng.random() < 0.3:
            op = rng.choice(self.RELATIONALS)
            evaluate = rng.random() < 0.8
            return op(self.int_expr(rng, 2), self.int_expr(rng, 2), evaluate=evaluate)
        op = rng.choice([sympy.And, sympy.Or, sympy.Not])
        if op is sympy.Not:
            return op(self.bool_expr(rng, depth - 1))
        return op(*(self.bool_expr(rng, depth - 1) for _ in range(rng.randint(2, 3))))

    @parametrize("seed", range(4))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        calls = answered = 0
        for _ in range(400):
            name = rng.choice(["safe_expand", "canonicalize_bool_expr"])
            try:
                if name == "safe_expand" and rng.random() < 0.8:
                    e = self.int_expr(rng, 4)
                else:
                    e = self.bool_expr(rng, 3)
            except (ZeroDivisionError, ValueError, TypeError, AssertionError):
                continue
            if e.has(sympy.zoo, sympy.nan):
                continue
            calls += 1
            answered += self.check(name, e)
        self.assertGreater(answered, calls // 2)


class TestNativeShapeEnv(TestCase):
    # (hint, positive, range override): Qwen-like [2, int_oo] sizes, tighter
    # and shifted ranges, and symbols that are only integer.
    SPECS = [
        (3, True, None),
        (5, True, None),
        (7, True, (4, 64)),
        (9, True, (1, int_oo)),
        (11, None, None),
        (13, None, (-3, 8)),
        (15, None, (0, int_oo)),
    ]
    assertSameTree = TestNativeStaticPasses.assertSameTree

    def make_envs(self):
        env = ShapeEnv()
        syms = []
        for i, (hint, positive, vr) in enumerate(self.SPECS):
            source = ConstantSource(f"x{i}")
            s = env.create_symbol(hint, source, DimDynamic.DYNAMIC, positive=positive)
            if vr is not None:
                env.var_to_range[s] = ValueRanges(*vr)
            syms.append(s)
        native = torch._C._symbolic.NativeShapeEnv(torch._C._symbolic._Arena())
        arena = native.arena
        for s in syms:
            vr = env.var_to_range[s]
            lower, upper = arena.from_sympy(vr.lower), arena.from_sympy(vr.upper)
            hint = int(env.backed_var_to_val[s])
            native.add_symbol(
                arena.from_sympy(s), hint, lower, upper, s in env.size_like
            )
        return env, native, syms

    def check(self, name, e, hint=None):
        """Differential test of a NativeShapeEnv entry point against a fresh Python ShapeEnv; returns whether native answered."""
        env, native, _ = self.make_envs()
        arena = native.arena
        try:
            n = arena.from_sympy(e)
        except NativeUnsupported:
            return False
        guards = len(env.guards)
        try:
            if name == "simplify":
                want = env.simplify(e)
            elif name == "static_eval":
                want = env._maybe_evaluate_static(e)
            else:
                want = env.evaluate_expr(e, hint)
        except Exception:
            want = NativeUnsupported
        if name == "simplify":
            try:
                got = native.simplify(n)
            except NativeUnsupported:
                return False
            self.assertIsNot(want, NativeUnsupported, str(e))
            self.assertSameTree(arena.to_sympy(got), want, f"simplify({e})")
            return True
        if name == "static_eval":
            answered, got = native.static_eval(n)
            if not answered:
                return False
            self.assertIsNot(want, NativeUnsupported, str(e))
            if want is None:
                self.assertIsNone(got, str(e))
            else:
                self.assertSameTree(arena.to_sympy(got), want, f"static_eval({e})")
            return True
        got = native.evaluate_expr(n, hint)
        if got is None:
            return False
        self.assertIsNot(want, NativeUnsupported, str(e))
        self.assertEqual(len(env.guards), guards, f"evaluate_expr({e}) guarded")
        self.assertSameTree(arena.to_sympy(got), want, f"evaluate_expr({e})")
        return True

    def test_known(self):
        a, b, c, d, i, j, k = self.make_envs()[2]
        static = [
            sympy.Eq(a, 1),
            sympy.Ne(a, 1),
            a > 1,
            a >= 2,
            a < 2,
            a**2 >= a,
            a <= 9223372036854775807,
            sympy.Eq(a * b, 1),
            a * b > a,
            c <= 64,
            c > 64,
            sympy.Eq(c, 3),
            d >= 1,
            d > 1,
            i >= 0,
            j <= 8,
            j + 3 >= 0,
            k >= 0,
            2 * a >= a + 2,
            FloorDiv(a, 2) >= 1,
            Max(a, b) >= 2,
            Min(a, 1) <= 1,
            sympy.Eq(Mod(a, 2), 0),
            sympy.And(a > 1, b > 1),
            sympy.Or(sympy.Eq(a, 1), sympy.Eq(b, 1)),
            a + b,
            sympy.Integer(3),
        ]
        simplify = [
            Max(1, a),
            Max(0, a - 2),
            Max(a, a**2),
            Max(a, b),
            Max(1, a - b),
            Max(1, i),
            Max(0, j),
            Max(0, k),
            Min(a, 1),
            Min(a, b, 2 * a),
            Max(a, b, a + b),
            Min(c, 64),
            Max(1, Min(a, b)),
            Max(1, a) + Min(b, 2),
            TruncToInt(IntTrueDiv(2 * a, 2)),
            TruncToInt(IntTrueDiv(a, 2)),
            sympy.Eq(Max(1, a), b),
            (a + 1) ** 2,
        ]
        for e in static:
            self.check("static_eval", e)
            self.check("evaluate_expr", e)
        for e in simplify:
            self.assertTrue(self.check("simplify", e), str(e))
            self.check("static_eval", e)
        for e in [sympy.Eq(a, 1), a > 1, a**2 >= a, a >= 0, sympy.Eq(c, 3), Max(1, a)]:
            self.assertTrue(self.check("static_eval", e), str(e))
        for e in [a >= 0, a + b >= 0, sympy.Le(0, k + 1), a > 1, sympy.Integer(3)]:
            self.assertTrue(self.check("evaluate_expr", e), str(e))
        self.assertTrue(self.check("evaluate_expr", sympy.Integer(3), 3))
        self.assertFalse(self.check("evaluate_expr", sympy.Integer(3), 4))
        self.assertFalse(self.check("evaluate_expr", sympy.Eq(a, b)))

    def test_pristine_gate(self):
        env, native, syms = self.make_envs()
        a, k = syms[0], syms[6]
        arena = native.arena
        ge = arena.from_sympy(k >= 0)
        self.assertEqual(native.static_eval(ge), (True, arena.boolean(True)))
        native.update_range(
            arena.from_sympy(k), arena.integer(3), arena.from_sympy(int_oo)
        )
        self.assertFalse(native.pristine)
        self.assertEqual(native.static_eval(ge), (False, None))
        # Delegates even where the fast comparison would answer.
        self.assertIsNone(native.evaluate_expr(ge))
        native.mark_replacements()
        self.assertIsNone(native.evaluate_expr(ge))
        self.assertEqual(native.evaluate_expr(arena.boolean(True)), arena.boolean(True))
        self.assertIsNone(native.evaluate_expr(arena.from_sympy(a >= 2), True))
        other = sympy.Symbol("other", integer=True, positive=True)
        _, native, syms = self.make_envs()
        arena = native.arena
        self.assertEqual(native.static_eval(arena.from_sympy(other > 1)), (False, None))
        three = arena.integer(3)
        self.assertEqual(native.evaluate_expr(three, 3), three)
        self.assertIsNone(native.evaluate_expr(three, 3.0))
        self.assertIsNone(native.evaluate_expr(three, 2**70))
        with self.assertRaisesRegex(RuntimeError, "already mirrored"):
            native.add_symbol(arena.from_sympy(syms[0]), 3, three, three, False)

    def test_xreplace(self):
        s2 = sympy.Symbol("s2", integer=True, positive=True)
        cases = [
            (sympy.Eq(s0, s1), {s0: s1}),
            (sympy.And(sympy.Eq(s0, 1), s1 > 2), {s0: sympy.Integer(1)}),
            (sympy.Or(sympy.Eq(s0, 1), s1 > 2), {s0: sympy.Integer(1)}),
            (Max(s0, s1), {s1: s0}),
            (sympy.Lt(s0, s0, evaluate=False), {s0: s0}),
            (sympy.Lt(s0, s1, evaluate=False), {s1: s0 + 1}),
            (sympy.Not(sympy.Eq(s0, s2)), {s2: s0}),
            (sympy.Ge(s0**2, s0), {s0: s2 + 1}),
            (sympy.Eq(s0 + 1, 3), {s1: s0}),
            (sympy.Eq(s0 + 1, 3), {s0 + 1: s1}),
            (s0 * s1 + 1, {s0: s1, s1: s0}),
        ]
        arena = torch._C._symbolic._Arena()
        for e, rule in cases:
            reps = [(arena.from_sympy(k), arena.from_sympy(v)) for k, v in rule.items()]
            got = arena.to_sympy(arena.xreplace(arena.from_sympy(e), reps))
            self.assertSameTree(got, e.xreplace(rule), f"{e}.xreplace({rule})")

    def int_expr(self, rng, syms, depth):
        if depth == 0 or rng.random() < 0.3:
            return rng.choice([*syms, *map(sympy.Integer, range(-2, 4))])
        a = self.int_expr(rng, syms, depth - 1)
        b = self.int_expr(rng, syms, depth - 1)
        op = rng.choice(
            ["add", "add", "mul", "sub", "pow", "Max", "Min", "FloorDiv", "trunc"]
        )
        if op == "add":
            return a + b
        if op == "mul":
            return a * b
        if op == "sub":
            return a - rng.randint(1, 3) * b
        if op == "pow":
            return a ** rng.choice([2, 3])
        if op == "trunc":
            return TruncToInt(IntTrueDiv(a, b))
        return TestNativeFunctions.FUNCTIONS[op](a, b)

    @parametrize("seed", range(4))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        syms = self.make_envs()[2]
        rels = TestNativeStaticPasses.RELATIONALS
        calls = answered = 0
        for _ in range(150):
            name = rng.choice(["simplify", "static_eval", "evaluate_expr"])
            try:
                e = self.int_expr(rng, syms, 3)
                if name != "simplify" or rng.random() < 0.3:
                    e = rng.choice(rels)(e, self.int_expr(rng, syms, 2))
            except (ZeroDivisionError, ValueError, TypeError, AssertionError):
                continue
            if e.has(sympy.zoo, sympy.nan):
                continue
            calls += 1
            answered += self.check(name, e)
        self.assertGreater(answered, calls // 3)


class TestNativeShapeEnvSync(TestCase):
    # (name, mutation of a ShapeEnv with symbols s, t, clears replacements_empty)
    MUTATIONS = [
        ("guard", lambda env, s, t: env.evaluate_expr(sympy.Eq(s, 5)), True),
        (
            "guard_lt",
            lambda env, s, t: env.evaluate_expr(sympy.Lt(s, t), hint=True),
            False,
        ),
        ("guards_append", lambda env, s, t: env.guards.append(None), False),
        ("axioms", lambda env, s, t: env.axioms.update({s > 1: sympy.true}), False),
        ("replacement", lambda env, s, t: env._set_replacement(s, t, "test"), True),
        ("replacements_pop", lambda env, s, t: env.replacements.pop(s, None), True),
        (
            "range",
            lambda env, s, t: env._update_var_to_range(s, ValueRanges(3, 8)),
            False,
        ),
        (
            "range_setitem",
            lambda env, s, t: env.var_to_range.__setitem__(s, ValueRanges(2, 9)),
            False,
        ),
        ("range_del", lambda env, s, t: env.var_to_range.pop(t), False),
        ("divisible", lambda env, s, t: env._add_divisible(Mod(s, 2)), False),
        ("size_like", lambda env, s, t: env._constrain_range_for_size(s), False),
        (
            "runtime_assert_setdefault",
            lambda env, s, t: env.deferred_runtime_asserts.setdefault(None, []),
            False,
        ),
        (
            "backed_var_to_val",
            lambda env, s, t: env.add_backed_var_to_val(
                sympy.Symbol("x", integer=True), 4
            ),
            False,
        ),
    ]
    DISABLING_CONFIGS = [
        ("backed_size_oblivious", True),
        ("aggressive_guard_free_semantics", 1),
        ("symbol_guard_limit_before_specialize", 3),
        ("translation_validation", True),
    ]

    # Queries over the symbols s = 5, t = 7 of make_env.
    QUERY_POOL = [
        lambda s, t: sympy.Gt(s, 1),
        lambda s, t: sympy.Lt(s, t),
        lambda s, t: sympy.Eq(s, 5),
        lambda s, t: sympy.Ge(s + t, 4),
        lambda s, t: sympy.Ne(s * t, 0),
        lambda s, t: sympy.Le(s, 2 * t),
        lambda s, t: sympy.Gt(t, 3),
        lambda s, t: sympy.Eq(Mod(s, 2), 1),
        lambda s, t: sympy.Lt(FloorDiv(t, 2), s),
        lambda s, t: sympy.Eq(t - s, 2),
        lambda s, t: sympy.Le(Max(s, t), s + t),
    ]

    def make_env(self, allow_native=True, **kwargs):
        env = ShapeEnv(_allow_native=allow_native, **kwargs)
        s = env.create_symbol(5, ConstantSource("a"), DimDynamic.DYNAMIC)
        t = env.create_symbol(7, ConstantSource("b"), DimDynamic.DYNAMIC)
        return env, s, t

    @staticmethod
    def answer(env, evaluate, e, hint=None, fallback_value=None):
        """A query as a native SymNode would issue it: native first, then Python."""
        native = env._native_env
        n = None
        if native is not None:
            try:
                n = native.arena.from_sympy(e)
            except NativeUnsupported:
                pass
        if evaluate:
            if n is not None:
                r = native.evaluate_expr(n, hint, fallback_value)
                if r is not None:
                    return native.arena.to_sympy(r)
            # fx_node=False, as SymNode passes it.
            return env.evaluate_expr(e, hint, False, False, fallback_value)
        if n is not None:
            answered, r = native.static_eval(n)
            if answered:
                return None if r is None else native.arena.to_sympy(r)
        try:
            return env._maybe_evaluate_static(e)
        except Exception:
            return None

    def run_steps(self, allow_native, steps):
        env, s, t = self.make_env(allow_native)
        answers = []
        for evaluate, i, use_hint, suppress in steps:
            e = self.QUERY_POOL[i](s, t)
            hint = bool(e.xreplace({s: 5, t: 7})) if use_hint else None
            with env.suppress_guards() if suppress else contextlib.nullcontext():
                answers.append(self.answer(env, evaluate, e, hint))
        return env, answers

    def test_flag(self):
        self.assertIsNone(ShapeEnv(_allow_native=False)._native_env)
        for on in (False, True):
            with torch._dynamo.config.patch(use_cpp_symnode=on):
                env = ShapeEnv()
            self.assertEqual(env._native_env is not None, on)

    def test_flag_off_state(self):
        env = ShapeEnv(_allow_native=False)
        for name in (
            "guards",
            "axioms",
            "replacements",
            "var_to_range",
            "divisible",
            "size_like",
            "deferred_runtime_asserts",
        ):
            self.assertIn(type(getattr(env, name)), (list, dict, set), name)

    @parametrize("name,value", DISABLING_CONFIGS)
    def test_disabling_config(self, name, value):
        with symbolic_shapes.config.patch(**{name: value}):
            self.assertIsNone(ShapeEnv(_allow_native=True)._native_env)

    def test_disabling_settings(self):
        for kwargs in (
            {"prefer_deferred_runtime_asserts_over_guards": True},
            {"trace_asserts": True},
        ):
            self.assertIsNone(ShapeEnv(_allow_native=True, **kwargs)._native_env)
        self.assertIsNone(
            ShapeEnv(_allow_native=True, should_record_events=True)._native_env
        )

    def test_mirror(self):
        env, s, t = self.make_env()
        native = env._native_env
        # Duck sizing reuses d; positive=None gives an unbounded range.
        d = env.create_symbol(11, ConstantSource("c"))
        self.assertEqual(env.create_symbol(11, ConstantSource("c2")), d)
        u = env.create_symbol(9, ConstantSource("d"), DimDynamic.DYNAMIC, positive=None)
        f = env.create_symbol(2.5, ConstantSource("e"), DimDynamic.DYNAMIC)
        big = env.create_symbol(2**70, ConstantSource("g"), DimDynamic.DYNAMIC)
        self.assertEqual(native.mirrored(s), (5, 2, int_oo, False))
        self.assertEqual(native.mirrored(t), (7, 2, int_oo, False))
        self.assertEqual(native.mirrored(u), (9, -int_oo, int_oo, False))
        self.assertIsNone(native.mirrored(f))
        self.assertIsNone(native.mirrored(big))
        self.assertEqual(native.mirrored(d), (11, 2, int_oo, False))
        for sym in (s, t, u, d):
            vr = env.var_to_range[sym]
            self.assertEqual(
                native.mirrored(sym),
                (
                    int(env.backed_var_to_val[sym]),
                    vr.lower,
                    vr.upper,
                    sym in env.size_like,
                ),
            )
        # Unbacked symbols are never mirrored and creating them keeps the env pristine.
        i0 = env.create_unbacked_symint().node.expr
        self.assertIsNone(native.mirrored(i0))
        self.assertTrue(native.pristine)
        # Static answers do not guard.
        self.assertEqual(env.evaluate_expr(sympy.Gt(s, 1)), sympy.true)
        self.assertTrue(native.pristine)

    def test_constraint_range(self):
        env = ShapeEnv(_allow_native=True)
        vr = ValueRanges(3, 10)
        constraint = symbolic_shapes.StrictMinMaxConstraint(vr=vr, warn_only=False)
        s = env.create_symbol(
            5, ConstantSource("a"), DimDynamic.DYNAMIC, constraint_dim=constraint
        )
        self.assertEqual(env._native_env.mirrored(s), (5, 3, 10, False))
        self.assertFalse(env._native_env.pristine)

    @parametrize(
        "name,mutate,clears_replacements", MUTATIONS, name_fn=lambda name, *_: name
    )
    def test_mutation(self, name, mutate, clears_replacements):
        env, s, t = self.make_env()
        native = env._native_env
        self.assertTrue(native.pristine)
        mutate(env, s, t)
        self.assertFalse(native.pristine)
        self.assertEqual(native.replacements_empty, not clears_replacements)

    def test_notify_before(self):
        env, s, t = self.make_env()
        seen = []
        env.guards._on_mutate = lambda: seen.append(list(env.guards))
        env.guards.append(1)
        env.guards += [2]
        self.assertEqual(seen, [[], [1]])
        self.assertEqual(env.guards, [1, 2])

    def test_update_divisible(self):
        env, s, t = self.make_env()
        env._update_divisible()
        self.assertTrue(env._native_env.pristine)
        env.divisible.add(Mod(s, 2))
        self.assertFalse(env._native_env.pristine)

    def test_copy_and_pickle(self):
        import copy
        import pickle

        env, s, t = self.make_env()
        env.evaluate_expr(sympy.Eq(s, 5))
        env2 = copy.deepcopy(env)
        self.assertIsNone(env2._native_env)
        self.assertIs(type(env2.guards), list)
        self.assertIs(type(env2.replacements), dict)
        self.assertEqual(env2.replacements, env.replacements)
        self.assertIsNone(pickle.loads(pickle.dumps(env._native_env)))
        for name in (
            "axioms",
            "replacements",
            "var_to_range",
            "size_like",
            "_symop_cache",
        ):
            value = getattr(env, name)
            self.assertIs(
                type(pickle.loads(pickle.dumps(value))), type(value).__mro__[-2]
            )
        from torch.fx._graph_pickler import _ShapeEnvPickleData

        self.assertNotIn("_native_env", _ShapeEnvPickleData(env).data)

    def test_check_equal(self):
        def run(allow_native):
            env = ShapeEnv(_allow_native=allow_native)
            s = env.create_symbol(5, ConstantSource("a"), DimDynamic.DYNAMIC)
            t = env.create_symbol(7, ConstantSource("b"), DimDynamic.DYNAMIC)
            env.evaluate_expr(sympy.Lt(s, t))
            env.evaluate_expr(sympy.Eq(s * 2, 10))
            env._constrain_range_for_size(t)
            return env

        on, off = run(True), run(False)
        self.assertIsNotNone(on._native_env)
        on.check_equal(off)

    def test_take_queries(self):
        env, s, t = self.make_env()
        native = env._native_env
        a = native.arena
        gt, ge = a.from_sympy(sympy.Gt(s, 1)), a.from_sympy(sympy.Ge(s + t, 4))
        lt = a.from_sympy(sympy.Lt(s, t))
        self.assertEqual(native.static_eval(gt)[0], True)
        self.assertEqual(native.static_eval(lt), (True, None))
        self.assertIsNotNone(native.evaluate_expr(ge, True, False))
        with env.suppress_guards():
            self.assertIsNotNone(native.evaluate_expr(ge, True, False))
        self.assertIsNotNone(native.evaluate_expr(ge, 1))
        self.assertEqual(native.static_eval(gt)[0], True)
        # Unanswered queries are not logged.
        self.assertIsNone(native.evaluate_expr(lt))
        self.assertTrue(torch._C._symbolic._native_queries_pending())
        queries = native.take_queries()
        self.assertEqual([q[0] for q in queries], sorted(q[0] for q in queries))
        self.assertEqual(
            [q[1:] for q in queries],
            [
                (sympy.Lt(s, t), False, None, None, False, None),
                (sympy.Ge(s + t, 4), True, True, False, False, sympy.true),
                (sympy.Ge(s + t, 4), True, True, False, True, sympy.true),
                (sympy.Ge(s + t, 4), True, 1, None, False, sympy.true),
                (sympy.Gt(s, 1), False, None, None, False, sympy.true),
            ],
        )
        self.assertEqual(native.take_queries(), [])

    def test_static_memo(self):
        env, s, t = self.make_env()
        native = env._native_env
        a = native.arena
        e = sympy.Ge(2 * s * t + s, t)
        n = a.from_sympy(e)
        # Memoized answers are still logged for replay.
        for _ in range(2):
            answered, got = native.static_eval(n)
            self.assertTrue(answered)
            self.assertEqual(a.to_sympy(got), sympy.true)
            self.assertEqual([q[1] for q in native.take_queries()], [e])
        env.guards.append(None)
        self.assertFalse(native.pristine)
        self.assertEqual(native.static_eval(n), (False, None))
        with self.assertRaises(NativeUnsupported):
            native.maybe_evaluate_static(n)

    def test_flush_populates_python_caches(self):
        env, s, t = self.make_env()
        native = env._native_env
        a = native.arena
        static_cache = ShapeEnv._maybe_evaluate_static
        evaluate_cache = ShapeEnv._inner_evaluate_expr
        lt, ge = sympy.Lt(s, t), sympy.Ge(s + t, 4)
        self.assertEqual(native.static_eval(a.from_sympy(lt)), (True, None))
        with env.suppress_guards():
            self.assertIsNotNone(native.evaluate_expr(a.from_sympy(ge), None, True))
        self.assertTrue(torch._C._symbolic._native_queries_pending())
        # Any Python evaluation flushes first.
        env.evaluate_expr(sympy.Gt(s, 1))
        self.assertEqual(native.take_queries(), [])
        self.assertFalse(ShapeEnv._suppress_guards_tls())
        self.assertFalse(torch._C._symbolic._suppress_guards())
        hits = static_cache.cache_info().hits
        self.assertIsNone(env._maybe_evaluate_static(lt))
        self.assertEqual(static_cache.cache_info().hits, hits + 1)
        hits = evaluate_cache.cache_info().hits
        with env.suppress_guards():
            env.evaluate_expr(ge, None, False, fallback_value=True)
        self.assertEqual(evaluate_cache.cache_info().hits, hits + 1)
        self.assertTrue(native.pristine)

    def test_flush_on_mutation(self):
        env, s, t = self.make_env()
        native = env._native_env
        order = []
        native.static_eval(native.arena.from_sympy(sympy.Gt(s, 1)))
        with mock.patch.object(
            env,
            "_maybe_evaluate_static",
            side_effect=lambda e: order.append(("replay", len(env.guards))),
        ):
            env.guards.append(None)
        self.assertEqual(order, [("replay", 0)])
        self.assertFalse(native.pristine)

    def test_flush_on_cache_hit(self):
        env, s, t = self.make_env()
        native = env._native_env
        env._maybe_evaluate_static(sympy.Gt(s, 1))
        native.static_eval(native.arena.from_sympy(sympy.Lt(s, t)))
        hits = ShapeEnv._maybe_evaluate_static.cache_info().hits
        env._maybe_evaluate_static(sympy.Gt(s, 1))
        self.assertEqual(native.take_queries(), [])
        # The replayed Lt clears nothing; Gt stays a hit.
        self.assertEqual(ShapeEnv._maybe_evaluate_static.cache_info().hits, hits + 1)

    def test_flush_order_across_envs(self):
        env1, s1_, _ = self.make_env()
        env2, s2_, _ = self.make_env()
        order = []

        def replay(env, name):
            return lambda e: order.append((name, e))

        for env, s, name in ((env1, s1_, "a"), (env2, s2_, "b"), (env1, s1_, "c")):
            e = sympy.Gt(s, 1) if name != "c" else sympy.Gt(s, 0)
            env._native_env.static_eval(env._native_env.arena.from_sympy(e))
        with (
            mock.patch.object(
                env1, "_maybe_evaluate_static", side_effect=replay(env1, 1)
            ),
            mock.patch.object(
                env2, "_maybe_evaluate_static", side_effect=replay(env2, 2)
            ),
        ):
            symbolic_shapes._flush_native_queries()
        self.assertEqual(
            order, [(1, sympy.Gt(s1_, 1)), (2, sympy.Gt(s2_, 1)), (1, sympy.Gt(s1_, 0))]
        )

    def test_suppress_guards_mirror(self):
        env, _, _ = self.make_env(allow_native=False)
        self.assertFalse(torch._C._symbolic._suppress_guards())
        with env.suppress_guards():
            self.assertTrue(torch._C._symbolic._suppress_guards())
            with env.suppress_guards():
                self.assertTrue(torch._C._symbolic._suppress_guards())
            self.assertTrue(torch._C._symbolic._suppress_guards())
        self.assertFalse(torch._C._symbolic._suppress_guards())

    @parametrize(
        "name,value",
        [("backed_size_oblivious", True), ("aggressive_guard_free_semantics", 1)],
    )
    def test_live_config(self, name, value):
        env, s, t = self.make_env()
        native = env._native_env
        n = native.arena.from_sympy(sympy.Gt(s, 1))
        self.assertTrue(torch._C._symbolic._native_config_is_default())
        with symbolic_shapes.config.patch(**{name: value}):
            self.assertFalse(torch._C._symbolic._native_config_is_default())
            self.assertIsNone(native.evaluate_expr(n))
            self.assertEqual(native.static_eval(n)[0], True)
        self.assertTrue(torch._C._symbolic._native_config_is_default())
        self.assertIsNotNone(native.evaluate_expr(n))

    @mock.patch.object(symbolic_shapes, "_NATIVE_SYMNODE_CHECK", True)
    def test_check_mode(self):
        env, s, t = self.make_env()
        native = env._native_env
        n = native.arena.from_sympy(sympy.Gt(s, 1))
        native.static_eval(n)
        symbolic_shapes._flush_native_queries()
        # A write that bypasses the notifying container.
        dict.__setitem__(env.var_to_range, s, ValueRanges(3, 9))
        native.static_eval(n)
        with self.assertRaisesRegex(AssertionError, "native mirror"):
            symbolic_shapes._flush_native_queries()
        list.append(env.guards, None)
        native.static_eval(n)
        with self.assertRaisesRegex(AssertionError, "pristine native env"):
            symbolic_shapes._flush_native_queries()

    @parametrize("seed", range(8))
    def test_replay_matches_python(self, seed):
        rng = random.Random(seed)
        steps = [
            (
                rng.random() < 0.5,
                rng.randrange(len(self.QUERY_POOL)),
                rng.random() < 0.3,
                rng.random() < 0.2,
            )
            for _ in range(30)
        ]
        off, want = self.run_steps(False, steps)
        with mock.patch.object(symbolic_shapes, "_NATIVE_SYMNODE_CHECK", True):
            on, got = self.run_steps(True, steps)
            symbolic_shapes._flush_native_queries()
        self.assertEqual(got, want)
        self.assertEqual(
            [str(g.expr) for g in on.guards], [str(g.expr) for g in off.guards]
        )
        on.check_equal(off)


class TestNativeSymNode(TestCase):
    INT_OPS = [
        "add",
        "sub",
        "mul",
        "floordiv",
        "int_floordiv",
        "mod",
        "sym_min",
        "sym_max",
        "eq",
        "ne",
        "gt",
        "lt",
        "le",
        "ge",
    ]
    GUARDS = [
        "guard_bool",
        "guard_int",
        "bool_",
        "int_",
        "guard_or_false",
        "guard_or_true",
        "statically_known_true",
        "expect_true",
    ]

    def make_env(self, native=True):
        env = ShapeEnv(_allow_native=native)
        syms = [
            env.create_symbol(5, ConstantSource("a"), DimDynamic.DYNAMIC),
            env.create_symbol(7, ConstantSource("b"), DimDynamic.DYNAMIC),
            env.create_symbol(3, ConstantSource("c"), DimDynamic.DYNAMIC),
            env.create_symbol(
                -4, ConstantSource("d"), DimDynamic.DYNAMIC, positive=None
            ),
        ]
        return env, syms

    @staticmethod
    def pair(env, expr, pytype, hint):
        # A native None hint is no hint; a Python one is computed.
        py_hint = _NO_HINT if hint is None else hint
        native = env._native_env.make_node(expr, pytype, hint)
        return native, SymNode(expr, env, pytype, py_hint)

    @staticmethod
    def node(env, expr, pytype, hint):
        """A native node on a native env, else a Python one."""
        if env._native_env is not None:
            return env._native_env.make_node(expr, pytype, hint)
        return SymNode(expr, env, pytype, _NO_HINT if hint is None else hint)

    def check(self, n, p):
        self.assertEqual(sympy.srepr(n._expr), sympy.srepr(p._expr))
        self.assertIs(n.pytype, p.pytype)
        self.assertEqual(n.hint, p.hint)
        self.assertIs(type(n.hint), type(p.hint))
        self.assertEqual(n.constant, p.constant)
        self.assertEqual(n._optimized_summation, p._optimized_summation)
        self.assertEqual(n.str(), p.str())
        try:
            want = p.maybe_as_int()
        except Exception as e:
            # int(-int_oo) raises.
            with self.assertRaises(type(e)):
                n.maybe_as_int()
        else:
            self.assertEqual(n.maybe_as_int(), want)

    def call(self, n, p, method, *args):
        """Runs method on both; returns the native result or None if it raised."""
        try:
            want = getattr(p, method)(*(a[1] for a in args))
        except Exception as e:
            with self.assertRaises(type(e)):
                getattr(n, method)(*(a[0] for a in args))
            return None
        got = getattr(n, method)(*(a[0] for a in args))
        self.check(got, want)
        return got, want

    @parametrize("seed", range(8))
    def test_differential(self, seed):
        rng = random.Random(seed)
        env, syms = self.make_env()
        ints = [self.pair(env, s, int, int(env.backed_var_to_val[s])) for s in syms]
        ints.append(self.pair(env, syms[0], int, None))
        ints += [
            (n.wrap_int(c), p.wrap_int(c)) for c in (0, 1, 2, -3) for n, p in ints[:1]
        ]
        bools = []
        native = 0
        for _ in range(120):
            k = rng.randrange(10)
            if k < 7 or not bools:
                method = rng.choice(self.INT_OPS)
                r = self.call(*rng.choice(ints), method, rng.choice(ints))
            elif k < 8:
                r = self.call(*rng.choice(ints), "neg")
            elif k < 9:
                r = self.call(
                    *rng.choice(bools),
                    rng.choice(["sym_and", "sym_or"]),
                    rng.choice(bools),
                )
            else:
                r = self.call(*rng.choice(bools), "sym_not")
            if r is None or type(r[0]) is SymNode:
                continue
            native += 1
            pool = ints if r[0].pytype is int else bools
            if len(pool) < 16:
                pool.append(r)
            else:
                pool[rng.randrange(len(pool))] = r
        self.assertGreater(native, 80)
        self.assertEqual(len(env.guards), 0)
        self.assertTrue(env._native_env.pristine)

    def test_optimized_summation(self):
        env, syms = self.make_env()
        a, b, c, d = (self.pair(env, s, int, 1) for s in syms)
        ab = self.call(*a, "add", b)
        self.assertTrue(ab[0]._optimized_summation)
        abc = self.call(*ab, "add", c)
        self.assertTrue(abc[0]._optimized_summation)
        self.assertFalse(self.call(*abc, "add", a)[0]._optimized_summation)
        cd = self.call(*c, "add", d)
        self.call(*ab, "add", cd)
        self.call(*abc, "add", cd)
        self.call(*cd, "add", abc)
        self.call(*a, "add", abc)
        self.call(*ab, "add", ab)

    def test_pow_by_natural(self):
        env, syms = self.make_env()
        a = self.pair(env, syms[0], int, 5)
        for c in (0, 1, 3, -1):
            r = self.call(*a, "pow_by_natural", (a[0].wrap_int(c), a[1].wrap_int(c)))
            self.assertIs(type(r[0]) is SymNode, c < 0)
        big = self.pair(env, syms[1], int, 7)
        r = self.call(
            *big, "pow_by_natural", (big[0].wrap_int(40), big[1].wrap_int(40))
        )
        self.assertIs(type(r[0]), SymNode)

    def test_mod(self):
        env, syms = self.make_env()
        e = env.create_symbol(4, ConstantSource("e"), DimDynamic.DYNAMIC, positive=None)
        a, c, d, e = (self.pair(env, s, int, 4) for s in (syms[0], syms[2], syms[3], e))
        self.assertIs(type(self.call(*a, "mod", c)[0]._expr), Mod)
        self.assertIs(type(self.call(*d, "mod", c)[0]._expr), PythonMod)
        self.assertIs(type(self.call(*c, "mod", d)[0]._expr), PythonMod)
        # a - 2 is nonnegative by range only.
        two = (a[0].wrap_int(2), a[1].wrap_int(2))
        r = self.call(*self.call(*a, "sub", two), "mod", c)
        self.assertIsNot(type(r[0]), SymNode)
        self.assertIs(type(r[0]._expr), Mod)
        # A range answer needs a pristine env.
        env._update_var_to_range(e[1]._expr, ValueRanges(0, 10))
        r = self.call(*e, "mod", c)
        self.assertIs(type(r[0]), SymNode)
        self.assertIs(type(r[0]._expr), Mod)

    @parametrize("mutation", ["range", "replacement", "deepcopy", "pickle"])
    def test_mod_memo(self, mutation):
        # _symop_cache keeps the first Mod/PythonMod choice, which reads ranges.
        import copy
        import pickle

        def mods(cache):
            return {k: v for k, v in cache.items() if k[0] == "mod"}

        caches = []
        for native in (False, True):
            env, syms = self.make_env(native)
            a, c = self.node(env, syms[0], int, 5), self.node(env, syms[2], int, 3)
            # a - 2 is nonnegative by range only.
            x = a.sub(a.wrap_int(2))
            self.assertIs(type(x.mod(c)._expr), Mod)
            if mutation == "pickle":
                caches.append(mods(pickle.loads(pickle.dumps(env._symop_cache))))
                continue
            if mutation == "replacement":
                env._set_replacement(syms[3], sympy.Integer(-4), "test")
                # Native results are Python from here on, and a Python node
                # cannot take a native operand yet.
                c = SymNode(syms[2], env, int, 3)
            else:
                if mutation == "deepcopy":
                    env = copy.deepcopy(env)
                    x = SymNode(syms[0] - 2, env, int, 3)
                    c = SymNode(syms[2], env, int, 3)
                env.var_to_range[syms[0]] = ValueRanges(-10, 10)
            r = x.mod(c)
            self.assertIs(type(r._expr), Mod)
            self.assertIs(type(r) is SymNode, not native or mutation != "range")
            # A new choice reads the current range.
            r = x.sub(x.wrap_int(1)).mod(c)
            self.assertIs(type(r._expr), PythonMod)
            self.assertIs(type(r), SymNode)
            caches.append(mods(env._symop_cache))
        self.assertEqual(caches[0], caches[1])
        want = {"range": 2, "replacement": 3, "deepcopy": 2, "pickle": 1}[mutation]
        self.assertEqual(len(caches[0]), want)

    def test_hint_fallback(self):
        env, syms = self.make_env()
        big = self.pair(env, syms[0], int, 2**62)
        r = self.call(*big, "add", big)
        self.assertIs(type(r[0]), SymNode)
        self.assertEqual(r[0].hint, 2**63)
        zero = (big[0].wrap_int(0), big[1].wrap_int(0))
        self.assertIsNone(self.call(*big, "floordiv", zero))
        self.assertIsNone(self.call(*big, "mod", zero))
        low = self.pair(env, syms[3], int, -(2**63))
        self.assertIs(type(self.call(*low, "neg")[0]), SymNode)
        minus_one = (low[0].wrap_int(-1), low[1].wrap_int(-1))
        self.assertIs(type(self.call(*low, "floordiv", minus_one)[0]), SymNode)
        self.call(*low, "mod", minus_one)

    def test_int_oo(self):
        env, syms = self.make_env()
        a = self.pair(env, syms[0], int, 5)
        oo = self.pair(env, int_oo, int, None)
        for method in ("add", "sub", "mul", "sym_max", "lt"):
            self.call(*a, method, oo)
            self.call(*oo, method, a)
        self.call(*oo, "neg")
        self.assertEqual(oo[0].str(), oo[1].str())

    def test_unbacked(self):
        env, _ = self.make_env()
        u = env.create_unbacked_symint().node.expr
        n, p = self.pair(env, u, int, None)
        three = (n.wrap_int(3), p.wrap_int(3))
        for method in ("eq", "ne", "lt", "ge"):
            self.call(n, p, method, three)
            self.call(*three, method, (n, p))
        self.assertIsNone(self.call(n, p, "add", three)[0].hint)

    def test_python_fallbacks(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, 5) for s in syms[:2])
        for method in ("truediv", "float_truediv", "int_truediv", "pow", "float_pow"):
            r = self.call(*a, method, b)
            self.assertIs(type(r[0]), SymNode)
        for method in ("ceil", "floor", "sym_float"):
            self.assertIs(type(self.call(*a, method)[0]), SymNode)
        f = a[0].wrap_float(2.5)
        self.assertIs(type(f), SymNode)
        self.assertEqual(f.constant, 2.5)
        # A Python operand makes the op Python.
        r = self.call(a[0], a[1], "add", (b[1], b[1]))
        self.assertIs(type(r[0]), SymNode)
        # So does an int operand of a logical op.
        t = self.call(*a, "lt", b)
        self.assertIs(type(self.call(*t, "sym_and", a)[0]), SymNode)

    SIZES_STRIDES = [
        "is_contiguous",
        "is_channels_last_contiguous_2d",
        "is_channels_last_contiguous_3d",
        "is_channels_last_strides_2d",
        "is_channels_last_strides_3d",
        "is_non_overlapping_and_dense_indicator",
        "is_non_overlapping_and_dense",
    ]

    def int_pairs(self, env, syms):
        a, b, c, d = syms
        exprs = [a, b, c, d, 2 * a, a + 1, a * b, b * c, a * b * c]
        vals = env.backed_var_to_val
        pairs = [self.pair(env, e, int, int(e.xreplace(vals))) for e in exprs]
        n, p = pairs[0]
        return pairs + [(n.wrap_int(k), p.wrap_int(k)) for k in (0, 1, 2, 3)]

    @parametrize("seed", range(8))
    def test_sizes_strides(self, seed):
        rng = random.Random(seed)
        env, syms = self.make_env()
        pool = self.int_pairs(env, syms)
        native = 0
        for _ in range(60):
            dim = rng.choice([1, 2, 3, 4, 4, 5, 5])
            if rng.random() < 0.3:
                # Contiguous strides make the relationals fold.
                sizes = [rng.choice(pool) for _ in range(dim)]
                strides = [pool[-3]]
                for s in reversed(sizes[1:]):
                    strides.insert(0, self.call(*strides[0], "mul", s))
            else:
                sizes = [rng.choice(pool) for _ in range(dim)]
                strides = [rng.choice(pool) for _ in range(dim)]
            base = rng.choice(sizes + strides)
            method = rng.choice(self.SIZES_STRIDES)
            r = self.call(
                *base,
                method,
                ([x[0] for x in sizes], [x[1] for x in sizes]),
                ([x[0] for x in strides], [x[1] for x in strides]),
            )
            if r is not None and type(r[0]) is _NativeSymNode:
                native += 1
        for method in self.SIZES_STRIDES:
            self.call(*pool[0], method, ([], []), ([], []))
        self.assertGreater(native, 50)
        self.assertEqual(len(env.guards), 0)
        self.assertTrue(env._native_env.pristine)

    def test_sizes_strides_fallback(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, int(env.backed_var_to_val[s])) for s in syms[:2])
        one = (a[0].wrap_int(1), a[1].wrap_int(1))
        no_hint = self.pair(env, syms[0], int, None)
        py_b = (b[1], b[1])
        for sizes, strides in [
            ([a, py_b], [b, one]),  # Python operand
            ([a, no_hint], [b, one]),  # Python computes a hint
            ([a, b], [b]),  # unequal lengths
        ]:
            sizes = tuple(list(x) for x in zip(*sizes))
            strides = tuple(list(x) for x in zip(*strides))
            for method in self.SIZES_STRIDES:
                r = self.call(*a, method, sizes, strides)
                if r is not None:
                    self.assertIs(type(r[0]), SymNode)

    def test_sizes_strides_tensor(self):
        # SymbolicShapeMeta calls the virtuals on the first symbolic node.
        def run(native):
            env, syms = self.make_env(native)
            a, b = (
                torch.SymInt(self.node(env, s, int, int(env.backed_var_to_val[s])))
                for s in syms[:2]
            )
            t = torch.empty_strided((a, b, 3), (3 * b, 3, 1), device="meta")
            u = torch.empty_strided(
                (2, a, b, 3), (3 * a * b, 1, 3 * a, a), device="meta"
            )
            out = []
            for x in (t, t.transpose(0, 2), u, u.transpose(1, 3)):
                out += [
                    x.is_contiguous(),
                    x.is_contiguous(memory_format=torch.channels_last),
                    # preserve_format asks is_non_overlapping_and_dense.
                    str(torch.empty_like(x).stride()),
                ]
            return out, [str(g.expr) for g in env.guards]

        self.assertEqual(run(True), run(False))

    def test_sym_sum(self):
        env, syms = self.make_env()
        pool = self.int_pairs(env, syms)
        no_hint = self.pair(env, syms[1], int, None)
        for args in [
            pool[:3],
            pool[4:9],
            [pool[0], pool[-1], pool[0]],
            [no_hint, pool[0]],
        ]:
            r = self.call(
                *args[0], "sym_sum", ([x[0] for x in args], [x[1] for x in args])
            )
            self.assertIs(type(r[0]), _NativeSymNode)
        # A Python operand makes it Python.
        r = self.call(
            *pool[0], "sym_sum", ([pool[0][0], pool[1][1]], [pool[0][1], pool[1][1]])
        )
        self.assertIs(type(r[0]), SymNode)
        # int64 overflow of the hint falls back to Python's bigint.
        big = (pool[0][0].wrap_int(2**62), pool[0][1].wrap_int(2**62))
        args = [big, big, pool[0]]
        r = self.call(*pool[0], "sym_sum", ([x[0] for x in args], [x[1] for x in args]))
        self.assertIs(type(r[0]), SymNode)
        x, y = (torch.SymInt(n) for n, _ in pool[:2])
        self.assertIs(type(torch.sym_sum([x, y, 3]).node), _NativeSymNode)
        self.assertEqual(len(env.guards), 0)

    def test_wrap(self):
        env, syms = self.make_env()
        n, p = self.pair(env, syms[0], int, 5)
        self.check(n.wrap_int(3), p.wrap_int(3))
        self.check(n.wrap_bool(True), p.wrap_bool(True))
        self.assertIsNone(n.maybe_as_int())
        self.assertEqual(n.wrap_int(-2).maybe_as_int(), -2)
        self.assertTrue(n.has_hint())
        self.assertTrue(n.is_int())
        self.assertFalse(n.is_bool())
        self.assertEqual(str(n), str(syms[0]))

    def test_replacements(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, 5) for s in syms[:2])
        env._set_replacement(syms[1], syms[0] + 2, "test")
        r = self.call(*a, "add", b)
        self.assertIs(type(r[0]), SymNode)
        self.assertEqual(b[0].str(), b[1].str())
        self.assertIs(type(self.call(*b, "neg")[0]), SymNode)

    def test_guards(self):
        env, syms = self.make_env()
        native = env._native_env
        a, b = (native.make_node(s, int, h) for s, h in zip(syms, (5, 7)))
        two = a.wrap_int(2)
        # s0 >= 2 by range.
        ge = a.ge(two)
        self.assertTrue(ge.guard_bool("", 0))
        self.assertTrue(ge.bool_())
        self.assertTrue(ge.guard_or_false("", 0))
        self.assertTrue(ge.guard_or_true("", 0))
        self.assertTrue(ge.statically_known_true("", 0))
        self.assertTrue(ge.expect_true("", 0))
        # s0 + s1 >= 4 by static evaluation.
        self.assertTrue(a.add(b).ge(a.wrap_int(4)).guard_bool("", 0))
        self.assertEqual(a.wrap_int(3).int_(), 3)
        self.assertFalse(a.wrap_int(0).bool_())
        self.assertTrue(a.wrap_bool(True).guard_bool("", 0))
        # The hint disproves it without a query.
        self.assertFalse(a.lt(two).statically_known_true("", 0))
        eq = a.eq(a.wrap_int(5))
        self.assertFalse(eq.statically_known_true("", 0))
        self.assertEqual(len(env.guards), 0)
        self.assertTrue(native.pristine)
        s0 = syms[0]
        ge_e, eq_e = sympy.Ge(s0, 2), sympy.Eq(s0, 5)
        self.assertEqual(
            [q[1:] for q in native.take_queries()],
            [
                (ge_e, True, True, False, False, sympy.true),
                (ge_e, True, True, True, False, sympy.true),
                (ge_e, False, None, None, False, sympy.true),
                (ge_e, True, True, None, False, sympy.true),
                (sympy.Ge(s0 + syms[1], 4), True, True, None, False, sympy.true),
                (sympy.Integer(3), True, 3, None, False, sympy.Integer(3)),
                (sympy.Integer(0), True, 0, None, False, sympy.Integer(0)),
                (sympy.true, True, True, None, False, sympy.true),
                (eq_e, False, None, None, False, None),
            ],
        )

        # Unknown statically: Python guards, and native answers stop.
        self.assertTrue(eq.guard_or_false("", 0))
        self.assertEqual([g.expr for g in env.guards], [eq_e])
        self.assertTrue(ge.guard_bool("", 0))
        self.assertEqual(native.take_queries(), [])
        with self.assertRaisesRegex(AssertionError, "bool"):
            a.guard_or_false("", 0)
        with self.assertRaisesRegex(AssertionError, "bool"):
            a.statically_known_true("", 0)
        self.assertEqual(a.int_(), 5)

    def test_guard_no_hint(self):
        env, syms = self.make_env()
        n, p = self.pair(env, sympy.Ge(syms[0], 2), bool, None)
        # expect_true without a hint defers to Python.
        self.assertTrue(n.expect_true("", 0))
        self.assertEqual(env._native_env.take_queries(), [])
        self.assertTrue(n.guard_bool("", 0))
        self.assertEqual(len(env._native_env.take_queries()), 1)

    @parametrize(
        "name,value",
        [
            ("backed_size_oblivious", True),
            ("aggressive_guard_free_semantics", 1),
            ("aggressive_guard_free_semantics", 2),
        ],
    )
    def test_guard_live_config(self, name, value):
        results = []
        for native in (True, False):
            env, syms = self.make_env(native)
            ge, eq = (sympy.Ge(syms[0], 2), sympy.Eq(syms[0], 5))
            nodes = [
                self.node(env, ge, bool, True),
                self.node(env, eq, bool, True),
                # Level 2 returns the fallback value without range analysis.
                self.node(env, ge, bool, None),
            ]
            with symbolic_shapes.config.patch(**{name: value}):
                results.append(
                    (
                        [n.guard_or_false("", 0) for n in nodes],
                        [n.guard_or_true("", 0) for n in nodes],
                        [str(g.expr) for g in env.guards],
                    )
                )
            if native:
                self.assertEqual(env._native_env.take_queries(), [])
        self.assertEqual(results[0], results[1])

    def guard_program(self, native, seed):
        """Short random programs of ops then guards, each on a fresh env."""
        rng = random.Random(seed)
        answers, envs = [], []
        for _ in range(12):
            env, syms = self.make_env(native)
            envs.append(env)
            ints = [self.node(env, s, int, int(env.backed_var_to_val[s])) for s in syms]
            ints += [ints[0].wrap_int(c) for c in (0, 1, 2, 5, -3)]
            bools = [ints[0].wrap_bool(True)]
            for _ in range(rng.randrange(1, 8)):
                a, b = rng.choice(ints), rng.choice(ints)
                if type(a) is SymNode and type(b) is not SymNode:
                    # A Python SymNode op does not take native operands yet.
                    b = SymNode(b._expr, env, int, b.hint, constant=b.constant)
                try:
                    r = getattr(a, rng.choice(self.INT_OPS))(b)
                except Exception as e:
                    answers.append(type(e))
                    continue
                (ints if r.pytype is int else bools).append(r)
            for _ in range(rng.randrange(1, 6)):
                method = rng.choice(self.GUARDS)
                n = rng.choice(bools if rng.random() < 0.8 else ints)
                args = () if method in ("bool_", "int_") else ("", 0)
                suppress = rng.random() < 0.2
                with env.suppress_guards() if suppress else contextlib.nullcontext():
                    try:
                        answers.append((method, getattr(n, method)(*args)))
                    except Exception as e:
                        answers.append((method, type(e)))
        return answers, envs

    @parametrize("seed", range(8))
    def test_guard_differential(self, seed):
        python_calls = []

        def counted(fn):
            def wrapper(*args, **kwargs):
                python_calls.append(fn.__name__)
                return fn(*args, **kwargs)

            return wrapper

        with (
            mock.patch.object(
                ShapeEnv, "evaluate_sym_node", counted(ShapeEnv.evaluate_sym_node)
            ),
            mock.patch.object(
                symbolic_shapes,
                "_static_eval_sym_bool",
                counted(symbolic_shapes._static_eval_sym_bool),
            ),
        ):
            want, off = self.guard_program(False, seed)
            off_calls = len(python_calls)
            python_calls.clear()
            with mock.patch.object(symbolic_shapes, "_NATIVE_SYMNODE_CHECK", True):
                got, on = self.guard_program(True, seed)
                symbolic_shapes._flush_native_queries()
        self.assertEqual(got, want)
        for on_env, off_env in zip(on, off):
            self.assertEqual(
                [str(g.expr) for g in on_env.guards],
                [str(g.expr) for g in off_env.guards],
            )
            on_env.check_equal(off_env)
        self.assertGreater(off_calls - len(python_calls), 10)

    def trace(self, native, fn, pre_dispatch):
        env, syms = self.make_env(native)
        a, b = (
            torch.SymInt(self.node(env, s, int, int(env.backed_var_to_val[s])))
            for s in syms[:2]
        )
        gm = make_fx(fn, tracing_mode="real", pre_dispatch=pre_dispatch)(a, b)
        return env, gm

    @parametrize("pre_dispatch", [False, True])
    def test_proxy_make_fx(self, pre_dispatch):
        def f(a, b):
            c = a + b
            d = c * 2 - a // b
            e = (d % 3 + 1) * 1
            m = torch.sym_max(a, b) + torch.sym_min(a, 3)
            one = a.node.wrap_int(1)
            sizes, strides = [a.node, b.node], [b.node, one]
            contig = torch.SymBool(a.node.is_contiguous(sizes, strides))
            dense = torch.SymBool(a.node.is_non_overlapping_and_dense(sizes, strides))
            ite = torch.sym_ite(a < b, a, b)
            total = torch.sym_sum([a, b, 3])
            return (
                e,
                m,
                a**2,
                a / b,
                torch.sym_float(a),
                -a,
                ite,
                contig,
                dense,
                total,
            )

        def g(a, b):
            lt = a < b
            return (
                lt & (b > 2),
                lt | (a == b),
                torch.sym_not(lt),
                a != b,
                a <= b,
                a >= b,
            )

        for fn in (f, g):
            on_env, on = self.trace(True, fn, pre_dispatch)
            off_env, off = self.trace(False, fn, pre_dispatch)
            self.assertEqual(on.code, off.code)
            self.assertEqual(
                [str(x.expr) for x in on_env.guards],
                [str(x.expr) for x in off_env.guards],
            )
            on_vals = [n.meta.get("val") for n in on.graph.nodes]
            off_vals = [n.meta.get("val") for n in off.graph.nodes]
            for x, y in zip(on_vals, off_vals, strict=True):
                if isinstance(y, torch.SymInt | torch.SymBool | torch.SymFloat):
                    self.assertEqual(str(x), str(y))
            ops = [n for n in on.graph.nodes if n.op == "call_function"]
            self.assertIsInstance(ops[0].meta["val"].node, _NativeSymNode)

        # The inner op of the dispatch runs natively.
        _, gm = self.trace(True, lambda a, b: a * b + 1, pre_dispatch)
        vals = [n.meta["val"] for n in gm.graph.nodes if n.op == "call_function"]
        self.assertTrue(all(isinstance(v.node, _NativeSymNode) for v in vals))

    def test_proxy_pre_dispatch_excluded(self):
        # get_proxy_mode() reads the global pre-dispatch slot even with
        # PreDispatch removed from the include set (torch._export.wrappers).
        def f(a, b):
            PreDispatch = torch._C.DispatchKey.PreDispatch
            include = torch._C._dispatch_tls_local_include_set().remove(PreDispatch)
            exclude = (
                torch._C._dispatch_tls_local_exclude_set()
                | torch._C.DispatchKeySet(PreDispatch)
            )
            with torch._C._ForceDispatchKeyGuard(include, exclude):
                return a * b + 1

        _, on = self.trace(True, f, True)
        _, off = self.trace(False, f, True)
        self.assertIn("mul", off.code)
        self.assertEqual(on.code, off.code)

    def test_expr(self):
        env, syms = self.make_env()
        n = env._native_env.make_node(syms[0], int, 5)
        self.assertEqual(n.add(n).expr, 2 * syms[0])

    def test_proxy_mul_by_one(self):
        # __sym_dispatch__ returns the operand itself.
        _, gm = self.trace(True, lambda a, b: (a * 1, 1 * b), False)
        self.assertNotIn("mul", gm.code)

    def test_proxy_sym_dispatch(self):
        # ProxyTorchDispatchMode.__sym_dispatch__ runs natively, including a
        # constant lhs, which Python evaluates through the reflected or
        # mirrored method of the rhs.
        def f(a, b):
            lt = a < b
            w = a.node.wrap_int
            return (
                3 + a,
                3 - a,
                2 * a,
                7 // a,
                7 % a,
                a * b - b,
                3 < a,
                3 > a,
                3 <= a,
                3 >= a,
                3 == a,
                3 != a,
                a == 3,
                torch.sym_max(3, a),
                torch.sym_min(3, a),
                torch.sym_max(a, 3),
                torch.sym_max(a, a),
                True & lt,
                False | lt,
                lt & True,
                lt | (a == b),
                -a,
                torch.sym_not(lt),
                w(3).add(w(4)).constant,
                w(3).lt(w(4)).constant,
            )

        sym_dispatch = mock.patch.object(
            proxy_tensor,
            "handle_sym_dispatch",
            wraps=proxy_tensor.handle_sym_dispatch,
        )
        sym_register = mock.patch.object(
            proxy_tensor, "_sym_register", wraps=proxy_tensor._sym_register
        )
        with sym_dispatch as m, sym_register as r:
            on_env, on = self.trace(True, f, False)
        # Only the two-constant ops, which are Python arithmetic.
        self.assertEqual(m.call_count, 2)
        # Those two, and sym_max(a, a), whose Symbol result set_proxy_slot tracks.
        self.assertEqual(r.call_count, 3)
        off_env, off = self.trace(False, f, False)
        self.assertEqual(on.code, off.code)
        self.assertEqual(
            [str(x.expr) for x in on_env.guards],
            [str(x.expr) for x in off_env.guards],
        )
        for x, y in zip(on.graph.nodes, off.graph.nodes, strict=True):
            self.assertEqual(list(x.meta), list(y.meta))
            self.assertEqual(str(x.meta.get("val")), str(y.meta.get("val")))

        # MAGIC logging keeps the Python dispatch.
        def logged(native):
            with self.assertLogs(sym_node.sym_node_log, level="DEBUG") as logs:
                _, gm = self.trace(native, lambda a, b: a * b + 1, False)
            return gm.code, [x for x in logs.output if "MAGIC" in x]

        with sym_dispatch as m:
            on = logged(True)
        self.assertEqual(m.call_count, 2)
        self.assertEqual(on, logged(False))

    def test_proxy_torch_function(self):
        # sym_min/sym_max/sym_not dispatch through torch functions, which a
        # TorchFunctionMode sees.
        class Spy(torch.overrides.TorchFunctionMode):
            def __init__(self):
                super().__init__()
                self.funcs = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.funcs.append(func.__name__)
                return func(*args, **(kwargs or {}))

        def traced(native):
            spy = Spy()

            def f(a, b):
                with spy:
                    return a.__sym_max__(b), a.__sym_min__(b), (a < b).__sym_not__()

            _, gm = self.trace(native, f, False)
            return gm.code, spy.funcs

        on_code, on_funcs = traced(True)
        self.assertEqual(on_code, traced(False)[0])
        # Not compared to off: the Python SymNode also calls torch.sym_max and
        # torch.sym_min on the hints.
        self.assertEqual(on_funcs, ["sym_max", "sym_min", "sym_not"])

    def test_proxy_hint_raises(self):
        # binary_magic_impl computes the hint before dispatching.
        env, syms = self.make_env()
        a = torch.SymInt(env._native_env.make_node(syms[0], int, 5))
        zero = torch.SymInt(env._native_env.make_node(syms[0] - 5, int, 0))
        with self.assertRaises(ZeroDivisionError):
            make_fx(lambda x, y: x // y, tracing_mode="real")(a, zero)

    def test_proxy_direct(self):
        # Native operands skip the Python impl unless its hint computation may
        # raise, or capture_provenance logs.
        sym_node = torch.fx.experimental.sym_node
        spy = mock.patch.object(
            sym_node, "method_to_operator", wraps=sym_node.method_to_operator
        )
        fn = lambda a, b: (a * b + 1, -a, a < b, torch.sym_not(a < b), a % b, a**b)  # noqa: E731
        with spy as m:
            _, gm = self.trace(True, fn, False)
        self.assertEqual([c.args for c in m.call_args_list], [("pow_by_natural",)])
        self.assertEqual(gm.code, self.trace(False, fn, False)[1].code)

        env, syms = self.make_env()
        a = torch.SymInt(env._native_env.make_node(syms[0], int, 5))
        zero = torch.SymInt(env._native_env.make_node(syms[0] - 5, int, 0))
        with (
            spy as m,
            mock.patch.object(torch._logging._internal, "GET_DTRACE_STRUCTURED", True),
            mock.patch.object(sym_node, "dtrace_structured"),
        ):
            make_fx(lambda x: -(x + x), tracing_mode="real")(a)
        self.assertEqual([c.args for c in m.call_args_list], [("add",), ("neg",)])
        with self.assertRaises(ZeroDivisionError):
            make_fx(lambda x, y: x % y, tracing_mode="real")(a, zero)

    def test_expr_facts(self):
        # has_free_symbols and fetch_sym_proxy read expr.is_number / is_Boolean.
        env, syms = self.make_env()
        a, b, c = (self.node(env, s, int, h) for s, h in zip(syms, (5, 7, 3)))
        lt = a.lt(b)
        nodes = [
            a,
            a.add(b),
            a.sub(a),
            a.mul(a.wrap_int(3)),
            a.int_floordiv(b),
            a.wrap_int(4).int_floordiv(a.wrap_int(3)),
            a.sym_max(b),
            lt,
            a.eq(a),
            a.eq(a.add(a.wrap_int(1))),
            lt.sym_and(b.lt(c)),
            lt.sym_or(b.lt(c)),
            lt.sym_and(b.lt(c)).sym_not(),
            self.node(env, sympy.Integer(3), int, 3),
            self.node(env, sympy.true, bool, True),
            b.mul(b),
        ]
        for n in nodes:
            self.assertIsInstance(n, _NativeSymNode)
            self.assertIs(n._expr_is_number, n.expr.is_number, n.expr)
            self.assertIs(n._expr_is_Boolean, n.expr.is_Boolean, n.expr)
        self.assertEqual(
            [symbolic_shapes.has_free_symbols(torch.SymInt(n)) for n in nodes[:7]],
            [True, True, False, True, True, False, True],
        )
        env._set_replacement(syms[1], sympy.Integer(7), "test")
        for n in nodes:
            self.assertIs(n._expr_is_number, n.expr.is_number, n.expr)
            self.assertIs(n._expr_is_Boolean, n.expr.is_Boolean, n.expr)
        self.assertFalse(symbolic_shapes.has_free_symbols(torch.SymInt(nodes[-1])))

        # A numeric SymInt enters the graph as its value.
        fn = lambda a, b: torch.ones(2) + (a - a)  # noqa: E731
        gm = self.trace(True, fn, False)[1]
        self.assertEqual(gm.code, self.trace(False, fn, False)[1].code)
        self.assertIn("add = torch.ops.aten.add.Tensor(ones, 0)", gm.code)

    def test_make_node(self):
        env, syms = self.make_env()
        native = env._native_env
        with self.assertRaisesRegex(RuntimeError, "int or bool"):
            native.make_node(syms[0], float, 5.0)
        with self.assertRaisesRegex(RuntimeError, "pytype"):
            native.make_node(syms[0], int, True)
        with self.assertRaisesRegex(RuntimeError, "pytype"):
            native.make_node(sympy.Eq(syms[0], 1), bool, 1)
        with self.assertRaises(NativeUnsupported):
            native.make_node(sympy.Float(1.5), int, None)

    def test_lifetime(self):
        # A native node holds its ShapeEnv, as a Python SymNode does.
        import gc
        import weakref

        env, syms = self.make_env()
        ref = weakref.ref(env)
        n = env._native_env.make_node(syms[0], int, 5)
        del n
        m = env._native_env.make_node(syms[1], int, 7)
        n = m.add(m)
        del env
        gc.collect()
        self.assertIsNotNone(ref())
        self.assertIs(n.shape_env, ref())
        self.assertEqual(n.truediv(m).hint, 2.0)
        del m
        gc.collect()
        self.assertIsNotNone(ref())
        del n
        gc.collect()
        self.assertIsNone(ref())
        env, syms = self.make_env()
        native = env._native_env
        del env
        gc.collect()
        n = native.make_node(syms[0], int, 5)
        self.assertIsNone(n.shape_env)
        with self.assertRaisesRegex(
            RuntimeError, "ShapeEnv of a native SymNode is gone"
        ):
            n.truediv(n)

    def test_create_symintnode(self):
        env, syms = self.make_env()
        s = syms[0]
        x = env.create_symintnode(s, hint=5)
        n = x.node
        self.assertIsInstance(n, _NativeSymNode)
        self.assertEqual((n._expr, n.hint), (s, 5))
        want = SymNode(s, env, int, 5)
        self.assertEqual((str(n), repr(n)), (str(want), repr(want)))
        # A backed node without a hint gets the computed one.
        n = env.create_symintnode(s * syms[1], hint=None).node
        self.assertIsInstance(n, _NativeSymNode)
        self.assertEqual(n.hint, 35)
        n = env.create_symboolnode(sympy.Eq(s, 5)).node
        self.assertIsInstance(n, _NativeSymNode)
        self.assertIs(n.hint, True)
        self.assertEqual(env.create_symintnode(sympy.Integer(3), hint=3), 3)
        big = env.create_symbol(2**70, ConstantSource("big"), DimDynamic.DYNAMIC)
        u = env.create_unbacked_symint().node.expr
        m2, mx = sympy.Mul(s, 2, evaluate=False), sympy.Max(s, syms[1])
        for sym, hint in ((big, 2**70), (u, None), (s, x), (5, 5), (m2, 10), (mx, 7)):
            self.assertIs(type(env.create_symintnode(sym, hint=hint).node), SymNode)
        self.assertIs(type(env.create_symintnode(5, hint=5).node.expr), int)
        flag_off, syms = self.make_env(native=False)
        n = flag_off.create_symintnode(syms[0], hint=5).node
        self.assertIs(type(n), SymNode)

    def test_attributes(self):
        env, syms = self.make_env()
        n, p = self.pair(env, syms[0], int, 5)
        self.assertIs(n.shape_env, env)
        self.assertIs(n.fx_node, p.fx_node)
        self.assertEqual(n._hint, p._hint)
        self.assertEqual(repr(n), repr(p))
        self.assertIsNone(self.pair(env, syms[3], int, None)[0]._hint)
        self.assertTrue(n._value_eq(p))
        self.assertEqual(n._value_hash(), p._value_hash())
        w = n.with_shape_env(env)
        self.assertIs(type(w), SymNode)
        self.check(w, p)
        with self.assertRaises(AttributeError):
            n.nonexistent
        with self.assertRaises(AttributeError):
            n.__add__

    def test_python_methods(self):
        # Methods without a native binding run the Python impl on the native node.
        env, syms = self.make_env()
        a, d = self.pair(env, syms[0], int, 5), self.pair(env, syms[3], int, -4)
        t = self.call(*a, "lt", d)
        binary = ("lshift", "rshift", "bitwise_and", "bitwise_or", "bitwise_xor")
        for x in (a, d):
            for method in ("abs", "pos", "trunc", "round", "is_integer", "sym_sqrt"):
                self.call(*x, method)
            for method in binary:
                self.call(*x, method, a)
            self.assertIsNone(x[0].maybe_as_float())
        self.assertIsNone(t[0].maybe_as_bool())
        self.assertIs(t[0].wrap_bool(True).maybe_as_bool(), True)
        self.call(*t, "xor", t)
        self.assertEqual(len(env.guards), 0)
        self.assertEqual(a[0].evaluate(), a[1].evaluate())
        self.assertEqual(len(env.guards), 1)

    def test_python_lhs(self):
        env, syms = self.make_env()
        a, b = (self.pair(env, s, int, int(env.backed_var_to_val[s])) for s in syms[:2])
        for method in self.INT_OPS + ["truediv", "pow"]:
            self.check(getattr(a[1], method)(b[0]), getattr(a[1], method)(b[1]))
        t = self.call(*a, "lt", b)
        self.check(t[1].sym_ite(a[0], b[0]), t[1].sym_ite(a[1], b[1]))
        ab = torch.SymInt(a[1]) * torch.SymInt(b[0])
        self.assertEqual(str(ab), str(syms[0] * syms[1]))

        # C++ PythonSymNodeImpl ops with a native operand.
        def run(native):
            env, syms = self.make_env(native)
            x = torch.SymInt(SymNode(syms[0], env, int, 5))
            y = torch.SymInt(self.node(env, syms[1], int, 7))
            t = torch.empty((x, y), device="meta")
            return str(t.numel()), str(t.stride()), str(torch.sym_ite(x < y, x, y))

        self.assertEqual(run(True), run(False))

    def test_casters(self):
        env, syms = self.make_env()
        n = env._native_env.make_node(syms[0], int, 5)
        t = n.lt(n.wrap_int(9))
        self.assertIs(type(t), _NativeSymNode)
        b = torch._C._symbolic._roundtrip_symbool(torch.SymBool(t))
        self.assertIs(b.node, t)
        f = torch._C._symbolic._roundtrip_symfloat(torch.SymFloat(n.sym_float()))
        self.assertEqual(str(f), f"ToFloat({syms[0]})")
        x = torch.SymInt(n)
        self.assertIs(x.node, x.node)
        t = torch.empty((x, 2), device="meta")
        self.assertIs(t.size(0).node, n)
        self.assertIs(t.size(0).node, t.size(0).node)

    def test_copy(self):
        env, syms = self.make_env()
        n, p = self.pair(env, syms[0], int, 5)
        memo = {id(env): env}
        c = copy.deepcopy(n, memo)
        self.assertIs(type(c), SymNode)
        self.assertIs(c.shape_env, env)
        self.check(c, copy.deepcopy(p, memo))
        self.assertIs(type(copy.deepcopy(n).shape_env), ShapeEnv)
        self.assertIs(copy.copy(n).shape_env, env)
        self.check(pickle.loads(pickle.dumps(n)), pickle.loads(pickle.dumps(p)))

    def test_module_helpers(self):
        from torch.fx.experimental.symbolic_shapes import (
            _iterate_nodes,
            free_symbols,
            guard_or_false,
            guard_or_true,
            guarding_hint_or_throw,
            has_free_symbols,
            statically_known_false,
            statically_known_true,
        )

        def run(native):
            env, syms = self.make_env(native)
            a, b = (self.node(env, s, int, h) for s, h in zip(syms, (5, 7)))
            no_hint = self.node(env, syms[0] + syms[1], int, None)
            out = [
                guarding_hint_or_throw(a),
                guarding_hint_or_throw(no_hint),
                list(_iterate_nodes([a, torch.SymInt(b)])) == [a, b],
                str(free_symbols([torch.SymInt(a), b])),
                has_free_symbols(a),
            ]
            # Native nodes do not cache a computed hint.
            self.assertEqual(no_hint._hint, None if native else 12)
            bools = [
                a.ge(a.wrap_int(2)),
                a.lt(a.wrap_int(2)),
                a.add(b).ge(a.wrap_int(4)),
                a.eq(a.wrap_int(5)),
                a.lt(b),
                a.gt(b),
            ]
            out += [statically_known_true(torch.SymBool(t)) for t in bools]
            # Answered natively: queries wait in the replay log.
            pending = torch._C._symbolic._native_queries_pending()
            self.assertEqual(pending, native)
            out += [statically_known_false(torch.SymBool(t)) for t in bools]
            out.append([g.expr for g in env.guards])
            out.append(guard_or_false(torch.SymBool(bools[0])))
            pending = torch._C._symbolic._native_queries_pending()
            self.assertEqual(pending, native)
            for fn in (guard_or_false, guard_or_true, statically_known_true):
                out += [fn(torch.SymBool(t)) for t in bools]
            out.append([g.expr for g in env.guards])
            with torch.fx.experimental._config.patch(backed_size_oblivious=True):
                out += [guard_or_true(torch.SymBool(t)) for t in bools]
            return out

        self.assertEqual(run(True), run(False))

    def test_no_plain_symnode_isinstance(self):
        # A native node is not a Python SymNode: type checks must use SymNodeTypes.
        import ast

        # Native nodes are never unbacked, so these checks are False for them.
        allowed = {
            ("torch/fx/experimental/symbolic_shapes.py", "_advise_is_size"),
            ("torch/fx/experimental/symbolic_shapes.py", "_advise_is_bounded"),
        }
        root = os.path.dirname(os.path.dirname(torch.__file__))
        found = set()

        class Visitor(ast.NodeVisitor):
            def __init__(self, path):
                self.path = path
                self.func = None

            def visit_FunctionDef(self, node):
                outer, self.func = self.func, node.name
                self.generic_visit(node)
                self.func = outer

            def visit_Call(self, node):
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in ("isinstance", "issubclass")
                    and len(node.args) == 2
                ):
                    for n in ast.walk(node.args[1]):
                        name = getattr(n, "id", None) or getattr(n, "attr", None)
                        if name == "SymNode":
                            found.add((self.path, self.func))
                self.generic_visit(node)

        for dirpath, _, files in os.walk(os.path.join(root, "torch")):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf-8") as fh:
                    src = fh.read()
                if "SymNode" in src:
                    Visitor(os.path.relpath(path, root)).visit(ast.parse(src))
        self.assertEqual(found, allowed)


class TestNativeSymNodeCompile(TestCase):
    """torch.compile with dynamic shapes, flag on vs off."""

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(8, 16)
            self.fc2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            h = torch.nn.functional.gelu(self.fc1(x))
            return torch.softmax(self.fc2(h), dim=-1).view(x.shape[0], -1)

    @staticmethod
    def attention(q, k):
        b, t, d = q.shape
        s = (q @ k.transpose(1, 2)) / d**0.5
        mask = torch.ones(t, t, dtype=torch.bool).tril()
        s = s.masked_fill(~mask, float("-inf")).softmax(-1)
        return (s @ k).reshape(b, t * d)[:, : t * d // 2]

    @staticmethod
    def size_branch(x, y):
        n = x.shape[0]
        z = torch.cat([x, y[: n // 2]])
        if n % 2 == 0 and z.shape[0] > 6:
            return z.flatten()[n:] * n
        return z.sum(0) - n

    CASES = {
        "mlp": (Mlp, lambda n: (torch.randn(n, 3, 8),), (4, 6, 9)),
        "attention": (
            lambda: TestNativeSymNodeCompile.attention,
            lambda n: (torch.randn(2, n, 4), torch.randn(2, n, 4)),
            (5, 8),
        ),
        "size_branch": (
            lambda: TestNativeSymNodeCompile.size_branch,
            lambda n: (torch.randn(n, 3), torch.randn(n + 2, 3)),
            (4, 8, 7, 10),
        ),
    }

    def run_compiled(self, case, native):
        from torch._dynamo.backends.common import aot_autograd
        from torch._guards import detect_fake_mode

        make_fn, make_args, sizes = self.CASES[case]
        torch.manual_seed(0)
        fn = make_fn()
        graphs, envs, vals = [], [], []

        def fw_compiler(gm, example_inputs):
            graphs.append(gm.code)
            return gm.forward

        def backend(gm, example_inputs):
            graphs.append(gm.code)
            envs.append(detect_fake_mode(example_inputs).shape_env)
            vals.extend(
                n.meta["example_value"]
                for n in gm.graph.nodes
                if isinstance(n.meta.get("example_value"), torch.SymInt)
            )
            return aot_autograd(fw_compiler=fw_compiler)(gm, example_inputs)

        torch._dynamo.reset()
        with torch._dynamo.config.patch(use_cpp_symnode=native):
            compiled = torch.compile(fn, backend=backend, dynamic=True)
            outs = []
            for n in sizes:
                args = make_args(n)
                outs.append((compiled(*args), fn(*args)))
        state = [
            (
                [str(g.expr) for g in env.guards],
                sorted(map(str, env.replacements.items())),
                sorted(map(str, env.var_to_range.items())),
            )
            for env in envs
        ]
        return graphs, state, outs, vals, envs

    @parametrize("case", list(CASES))
    def test_compile(self, case):
        graphs, state, outs, vals, envs = self.run_compiled(case, native=True)
        want_graphs, want_state, _, want_vals, _ = self.run_compiled(case, native=False)
        self.assertEqual(graphs, want_graphs)
        self.assertEqual(state, want_state)
        for got, want in outs:
            self.assertEqual(got, want)
        self.assertTrue(all(env._native_env is not None for env in envs))
        self.assertTrue(vals)
        self.assertTrue(all(isinstance(v.node, _NativeSymNode) for v in vals))
        self.assertEqual(
            [(str(v), v.node.hint) for v in vals],
            [(str(v), v.node.hint) for v in want_vals],
        )
        self.assertTrue(not any(isinstance(v.node, _NativeSymNode) for v in want_vals))


@contextlib.contextmanager
def python_glue():
    """Puts the Python magic methods back for the duration."""
    for (cls, attr), fn in sym_node._native_glue_originals.items():
        setattr(cls, attr, fn)
    try:
        yield
    finally:
        sym_node._install_native_glue()


class TestNativeSymIntGlue(TestCase):
    ENTRIES = [
        (cls, attr, kind)
        for cls, attr, _, kind in sym_node._user_magic_entries
        if cls is not torch.SymFloat and kind in ("unary", "binary", "rbinary")
    ]
    OPERATORS = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "mod": operator.mod,
        "and": operator.and_,
        "or": operator.or_,
        "xor": operator.xor,
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "le": operator.le,
        "gt": operator.gt,
        "ge": operator.ge,
        "lshift": operator.lshift,
        "rshift": operator.rshift,
    }
    PLAIN = [0, 1, 2, -3, 6, True, False]
    SYMINT_BODY = ["__bool__", "__int__", "__index__", "__truediv__", "__rtruediv__"]
    SYMINT_BODY += ["__floordiv__", "__rfloordiv__", "__pow__", "__rpow__"]
    SYMINT_BODY += ["__repr__", "__hash__", "has_hint"]
    SYMBOOL_BODY = ["__bool__", "__int__", "__hash__", "__sym_float__", "__repr__"]
    SYMBOOL_BODY += ["__sym_ite__"]
    CORE = {"add", "sub", "mul", "mod", "int_floordiv", "and", "or", "eq", "ne"}
    CORE |= {"lt", "le", "gt", "ge", "neg", "sym_not", "sym_min", "sym_max"}

    def setUp(self):
        super().setUp()
        ShapeEnv(_allow_native=True)
        self.assertTrue(sym_node._native_glue_originals)

    @staticmethod
    def make_env(native=True):
        env = ShapeEnv(_allow_native=native)
        hints = [5, 7, 3, -4, 6]
        syms = [
            env.create_symbol(
                h, ConstantSource(f"x{i}"), DimDynamic.DYNAMIC, positive=h > 0 or None
            )
            for i, h in enumerate(hints)
        ]
        ints = [env.create_symintnode(s, hint=h) for s, h in zip(syms, hints)]
        bools = [
            env.create_symboolnode(sympy.Ne(syms[0], 3)),
            env.create_symboolnode(sympy.Lt(syms[1], syms[2])),
        ]
        return env, ints, bools

    @staticmethod
    def describe(r):
        if isinstance(r, torch.SymInt | torch.SymBool | torch.SymFloat):
            n = r.node
            return (type(r), type(n), str(n), n.hint, n.constant, n.pytype)
        return (type(r), r)

    def program(self, seed):
        rng = random.Random(seed)
        env, ints, bools = self.make_env()
        out = []
        core = [
            e for e in self.ENTRIES if e[1].strip("_").removeprefix("r") in self.CORE
        ]
        for _ in range(150):
            cls, attr, kind = rng.choice(core if rng.random() < 0.8 else self.ENTRIES)
            pool = ints if cls is torch.SymInt else bools
            x = rng.choice(pool)
            method = attr[3:-2] if kind == "rbinary" else attr[2:-2]
            try:
                if kind == "unary":
                    r = getattr(x, attr)()
                else:
                    if "pow" in method or "shift" in method:
                        y = rng.choice([0, 1, 2])
                    else:
                        y = rng.choice(ints + bools + self.PLAIN)
                    fn = self.OPERATORS.get(method)
                    style = rng.randrange(3)
                    if style == 0 or fn is None:
                        r = getattr(x, attr)(y)
                    elif style == 1:
                        r = getattr(cls, attr)(x, y)
                    elif kind == "binary":
                        r = fn(x, y)
                    else:
                        r = fn(rng.choice(self.PLAIN), x)
            except Exception as e:
                out.append(("raise", type(e)))
                continue
            out.append(self.describe(r))
            if isinstance(r, torch.SymInt):
                target = ints
            elif isinstance(r, torch.SymBool):
                target = bools
            else:
                continue
            if type(r.node) is not _NativeSymNode and rng.random() < 0.8:
                continue
            if len(target) < 12:
                target.append(r)
            else:
                target[rng.randrange(len(target))] = r
        out.append([q[1:] for q in env._native_env.take_queries()])
        out.append([str(g.expr) for g in env.guards])
        return out

    @parametrize("seed", range(8))
    def test_differential(self, seed):
        got = self.program(seed)
        with python_glue():
            self.assertIsInstance(torch.SymInt.__add__, types.FunctionType)
            want = self.program(seed)
        self.assertEqual(got, want)
        native = [r for r in got[:-2] if len(r) == 6 and r[1] is _NativeSymNode]
        self.assertGreater(len(native), 40)

    def body_program(self, seed):
        rng = random.Random(seed)
        env, ints, bools = self.make_env()
        ints += [ints[0] + ints[1], ints[2] * 2, ints[3] - 1]
        exponents = [0, 1, 2, -1, True, False]
        ops = [
            lambda x, y, p: bool(x),
            lambda x, y, p: bool(p),
            lambda x, y, p: int(p),
            lambda x, y, p: x / y,
            lambda x, y, p: y / x,
            lambda x, y, p: x // y,
            lambda x, y, p: y // x,
            lambda x, y, p: x ** rng.choice(exponents),
            lambda x, y, p: rng.choice([2, -2, 0]) ** x,
            lambda x, y, p: x ** rng.choice(ints[:5]),
            lambda x, y, p: x.__rpow__(y),
            lambda x, y, p: x / p,
            lambda x, y, p: x // p,
            lambda x, y, p: x**p,
            lambda x, y, p: x / 2.0,
            lambda x, y, p: repr(x),
            lambda x, y, p: repr(p),
            lambda x, y, p: hash(x),
            lambda x, y, p: hash(p),
            lambda x, y, p: x.has_hint(),
            lambda x, y, p: torch.sym_float(p),
            lambda x, y, p: torch.sym_ite(p, x, y),
            lambda x, y, p: torch.sym_ite(p, 3, 3),
            lambda x, y, p: torch.sym_ite(p, rng.choice(bools), rng.choice(bools)),
            lambda x, y, p: p.__sym_ite__(rng.choice(self.PLAIN), y),
            lambda x, y, p: p.__sym_ite__(x, 1.5),
        ]
        out = []
        for _ in range(120):
            x = rng.choice(ints)
            y = rng.choice(ints + self.PLAIN)
            p = rng.choice(bools)
            try:
                r = rng.choice(ops)(x, y, p)
            except Exception as e:
                out.append(("raise", type(e), str(e)))
                continue
            out.append(self.describe(r))
            if isinstance(r, torch.SymInt) and type(r.node) is _NativeSymNode:
                ints[rng.randrange(5, len(ints))] = r
        # int() specializes, so it goes last.
        for x in rng.sample(ints, 3):
            try:
                out.append((int(x), operator.index(x)))
            except Exception as e:
                out.append(("raise", type(e)))
        out.append([q[1:] for q in env._native_env.take_queries()])
        out.append([str(g.expr) for g in env.guards])
        return out

    @parametrize("seed", range(6))
    def test_body_differential(self, seed):
        got = self.body_program(seed)
        with python_glue():
            self.assertIsInstance(torch.SymInt.__bool__, types.FunctionType)
            want = self.body_program(seed)
        self.assertEqual(got, want)
        native = [r for r in got[:-2] if len(r) == 6 and r[1] is _NativeSymNode]
        self.assertGreater(len(native), 10)

    def test_body_native_path(self):
        originals = sym_node._native_glue_originals.values()
        codes = {fn.__code__ for fn in originals}

        def run():
            _, (a, b, _, d, e), (p, q) = self.make_env()
            calls = []

            def profile(frame, event, arg):
                if event != "call" or frame.f_code not in codes:
                    return
                # SymFloat's magic methods share their code with SymInt's.
                first = frame.f_locals[frame.f_code.co_varnames[0]]
                if not isinstance(first, torch.SymFloat):
                    calls.append(frame.f_code.co_qualname)

            prev = sys.getprofile()
            sys.setprofile(profile)
            try:
                out = [bool(a), bool(p), int(q), a / b, 3 / a, a // 2, 7 // a]
                out += [a**2, a**True, a**b, 2**a, repr(a), repr(q), hash(p)]
                out += [a.has_hint(), torch.sym_ite(p, a, b), torch.sym_ite(q, 1, 2)]
                out += [a**d, 2**d]
                # int() specializes, so it goes last.
                out += [p.__sym_ite__(True, q), int(e), operator.index(e)]
            finally:
                sys.setprofile(prev)
            return [self.describe(r) for r in out], calls

        got, calls = run()
        self.assertEqual(calls, [])
        with python_glue():
            want, calls = run()
        self.assertGreater(len(calls), 20)
        self.assertEqual(got, want)

    def sum_program(self, seed):
        rng = random.Random(seed)
        env, ints, bools = self.make_env()
        ints += [ints[0] + ints[1], ints[2] * 2, ints[3] - 1]
        py_int = torch.SymInt(SymNode(ints[0].node.expr + 1, env, int, 6))

        class Small(enum.IntEnum):
            ONE = 1

        class Items(list):
            pass

        class Sub(torch.SymInt):
            pass

        odd = [py_int, Small.ONE, 2**70, 1.5, bools[0], Sub(ints[1].node)]
        out = []
        for _ in range(80):
            items = [rng.choice(ints + self.PLAIN) for _ in range(rng.randrange(4))]
            if rng.random() < 0.15:
                items.insert(rng.randrange(len(items) + 1), rng.choice(odd))
            style = rng.randrange(3)
            try:
                if style == 0:
                    r = torch.sym_sum(items if rng.random() < 0.8 else Items(items))
                elif style == 1:
                    r = torch.sym_sum(tuple(items))
                else:
                    r = torch.sym_sum(*items)
                # A bool item reaches SymNode.sym_sum as sympy.true, which sympy.Add rejects.
                out.append(self.describe(r))
            except Exception as e:
                out.append(("raise", type(e), re.sub("0x[0-9a-f]+", "", str(e))))
                continue
            if isinstance(r, torch.SymInt) and type(r.node) is _NativeSymNode:
                ints[rng.randrange(5, len(ints))] = r
        out.append([q[1:] for q in env._native_env.take_queries()])
        out.append([str(g.expr) for g in env.guards])
        return out

    @parametrize("seed", range(4))
    def test_sym_sum_differential(self, seed):
        got = self.sum_program(seed)
        with python_glue():
            want = self.sum_program(seed)
        self.assertEqual(got, want)
        native = [r for r in got[:-2] if len(r) == 6 and r[1] is _NativeSymNode]
        self.assertGreater(len(native), 20)

    def test_sym_sum_native_path(self):
        codes = {sym_node.to_node.__code__, sym_node.wrap_node.__code__}

        def run():
            _, (a, b, c, *_), _ = self.make_env()
            calls = []

            def profile(frame, event, arg):
                if event == "call" and frame.f_code in codes:
                    calls.append(frame.f_code.co_name)

            prev = sys.getprofile()
            sys.setprofile(profile)
            try:
                out = [torch.sym_sum([a, b, 3]), torch.sym_sum(a, 2, c)]
                out += [torch.sym_sum((4, a)), torch.sym_sum([a, -a])]
            finally:
                sys.setprofile(prev)
            return [self.describe(r) for r in out], calls

        got, calls = run()
        self.assertEqual(calls, [])
        with python_glue():
            want, calls = run()
        self.assertGreater(len(calls), 10)
        self.assertEqual(got, want)

    def test_body_fallbacks(self):
        reasons = []
        sym_float = torch.sym_float

        def recording(x):
            reasons.append(torch._C._symbolic._glue_fallback_reason())
            return sym_float(x)

        _, (a, b, *_), (p, _) = self.make_env()
        with mock.patch.object(torch, "sym_float", recording):
            self.assertIsInstance(a**-1, torch.SymFloat)
            self.assertIsInstance(a / 2.0, torch.SymFloat)
        self.assertEqual(reasons[0], "negative exponent")
        self.assertEqual(reasons[-1], "operand type")
        with self.assertRaisesRegex(TypeError, "positional argument"):
            torch.SymInt.__pow__(a, 2, 3)
        with self.assertRaisesRegex(TypeError, "non-nested SymInt"):
            hash(a)
        with self.assertRaisesRegex(TypeError, "unsupported operand"):
            a / p
        with self.assertRaisesRegex(AssertionError, "same pytype"):
            p.__sym_ite__(a, p)
        unbound = [
            lambda: torch.SymInt.__bool__(p),
            lambda: torch.SymBool.__bool__(a),
            lambda: torch.SymInt.__int__(p),
            lambda: torch.SymBool.__hash__(a),
            lambda: torch.SymInt.__truediv__(p, 2),
            lambda: torch.SymInt.__pow__(p, 2),
            lambda: torch.SymBool.__sym_ite__(a, 1, 2),
        ]
        for fn in unbound:
            results = []
            for ctx in (contextlib.nullcontext(), python_glue()):
                with ctx:
                    try:
                        results.append(self.describe(fn()))
                    except Exception as e:
                        results.append(("raise", type(e), str(e)))
            self.assertEqual(results[0], results[1])
        self.assertIsNone(torch._C._symbolic._glue_fallback_reason())

    def test_installed(self):
        originals = sym_node._native_glue_originals
        body = {(torch.SymInt, a) for a in self.SYMINT_BODY}
        body |= {(torch.SymBool, a) for a in self.SYMBOOL_BODY}
        entries = {(cls, attr) for cls, attr, _ in self.ENTRIES}
        self.assertEqual(set(originals), entries | body)
        for cls, attr in originals:
            glue = cls.__dict__[attr]
            self.assertIsInstance(glue, torch._C._symbolic._SymGlueMethod)
            fn = originals[(cls, attr)]
            self.assertIs(glue.__wrapped__, fn)
            self.assertEqual(glue.__name__, fn.__name__)
            self.assertEqual(glue.__qualname__, fn.__qualname__)
            self.assertEqual(inspect.signature(glue), inspect.signature(fn))
            self.assertIs(getattr(cls, attr), glue)
        self.assertIsInstance(torch.SymFloat.__add__, types.FunctionType)
        self.assertIsInstance(torch.SymInt.__round__, types.FunctionType)
        _, ints, _ = self.make_env()
        a = ints[0]
        self.assertEqual(str(a.__add__(1)), f"{a} + 1")
        self.assertIs(a.__add__.__self__, a)
        self.assertEqual(str(torch.SymInt.__add__(a, 1)), f"{a} + 1")

    def test_native_path(self):
        calls = []

        def counted(fn):
            def wrapper(*args):
                calls.append(fn.__name__)
                return fn(*args)

            return wrapper

        def run(native):
            env, (a, b, c, d, _), (p, q) = self.make_env(native)
            out = [a + b, a * 2, 3 - a, a // b, a % 3, a == 5, a < b, -a, p & q]
            out += [p | True, a > c, 2 * a, torch.sym_min(a, 3), a != 4, a - 5]
            return [self.describe(r) for r in out]

        with (
            mock.patch.object(sym_node, "to_node", counted(sym_node.to_node)),
            mock.patch.object(sym_node, "wrap_node", counted(sym_node.wrap_node)),
        ):
            got = run(True)
            self.assertEqual(calls, [])
            want = run(False)
            self.assertGreater(len(calls), 20)
        self.assertEqual([r[2:] for r in got], [r[2:] for r in want])
        self.assertTrue(all(r[1] is _NativeSymNode for r in got))

    def test_fallbacks(self):
        reasons = []
        to_node = sym_node.to_node

        def recording(self_node, num):
            reasons.append(torch._C._symbolic._glue_fallback_reason())
            return to_node(self_node, num)

        class Small(enum.IntEnum):
            ONE = 1

        cases = [
            (lambda a, p: a + 1.5, "operand type"),
            (lambda a, p: a * Small.ONE, "operand type"),
            (lambda a, p: a + 2**70, "int beyond int64"),
            (lambda a, p: torch.SymInt.__add__(a, other=1), "keywords"),
            (lambda a, p: torch.SymInt.__sub__(a, 1, 2), "arity"),
            (lambda a, p: torch.SymInt.__neg__(a, 1), "arity"),
            (
                lambda a, p: a
                + torch.SymInt(SymNode(a.node._expr, a.node.shape_env, int, 5)),
                "python node",
            ),
            (lambda a, p: a + "x", "operand type"),
            (lambda a, p: 1.5 - a, "operand type"),
            (lambda a, p: p & 1.5, "operand type"),
            (lambda a, p: torch.SymInt.__add__(3, 4), "operand type"),
        ]
        for fn, reason in cases:
            results = []
            for glue in (True, False):
                _, ints, bools = self.make_env()
                reasons.clear()
                ctx = contextlib.nullcontext() if glue else python_glue()
                with ctx, mock.patch.object(sym_node, "to_node", recording):
                    try:
                        results.append(self.describe(fn(ints[0], bools[0])))
                    except Exception as e:
                        results.append(("raise", type(e)))
                if glue and reasons:
                    self.assertEqual(reasons[0], reason)
            self.assertEqual(results[0], results[1])
        self.assertIsNone(torch._C._symbolic._glue_fallback_reason())

    def test_magic_logging(self):
        _, (a, *_), _ = self.make_env()
        with self.assertLogs(sym_node.sym_node_log, level="DEBUG") as logs:
            r = a + 1
        self.assertEqual(
            logs.output, [f"DEBUG:{sym_node.sym_node_log.name}:MAGIC add {a} 1"]
        )
        self.assertIsInstance(r.node, _NativeSymNode)

    def test_modified_class(self):
        _, (a, *_), _ = self.make_env()
        reasons = []
        to_node = sym_node.to_node

        def recording(self_node, num):
            reasons.append(torch._C._symbolic._glue_fallback_reason())
            return to_node(self_node, num)

        with mock.patch.object(sym_node, "to_node", recording):
            try:
                torch.SymInt._glue_test_attr = 1
                self.assertEqual(str(a + 1), f"{a} + 1")
                del torch.SymInt._glue_test_attr
                self.assertEqual(str(a + 1), f"{a} + 1")
                self.assertEqual(reasons, ["glue modified"] * 2)
            finally:
                sym_node._install_native_glue()
            reasons.clear()
            self.assertEqual(str(a + 1), f"{a} + 1")
            self.assertEqual(reasons, [])

    @parametrize("pre_dispatch", [False, True])
    def test_make_fx(self, pre_dispatch):
        def f(a, b, p, d):
            c = a + b
            d = c * 2 - a // b
            e = (d % 3 + 1) * 1
            m = torch.sym_max(a, b) + 3 * torch.sym_min(a, 3)
            r = e, m, -a, 2 - a, a < b, (p & (a == b)) | (b > 2), p + 1, True * p
            r += a**2, a**b, 2**a, a // 2, 7 // b, a / b, torch.sym_ite(p, a, b)
            return *r, a**d, 2**d, torch.sym_sum([a, b, 3]), torch.sym_sum(4, a)

        def trace():
            env, (a, b, _, d, _), (p, _) = self.make_env()
            fx = make_fx(f, tracing_mode="real", pre_dispatch=pre_dispatch)
            with self.count_inits() as inits:
                gm = fx(a, b, p, d)
            return gm.code, [str(g.expr) for g in env.guards], len(inits)

        *got, got_inits = trace()
        with python_glue():
            *want, want_inits = trace()
        self.assertEqual(got, want)
        self.assertLess(got_inits, want_inits)

    @staticmethod
    @contextlib.contextmanager
    def count_inits():
        codes = {torch.SymInt.__init__.__code__, torch.SymBool.__init__.__code__}
        inits = []

        def profile(frame, event, arg):
            if event == "call" and frame.f_code in codes:
                inits.append(frame.f_code.co_qualname)

        prev = sys.getprofile()
        sys.setprofile(profile)
        try:
            yield inits
        finally:
            sys.setprofile(prev)

    def test_native_construction(self):
        _, (a, b, *_), (p, _) = self.make_env()
        t = torch.empty((a, b), device="meta")
        with self.count_inits() as inits:
            size, stride = t.size(0), t.stride(0)
            numel = t.numel()
            q = torch._C._symbolic._roundtrip_symbool(p)
        self.assertEqual(inits, [])
        self.assertIs(type(size), torch.SymInt)
        self.assertEqual(size.__dict__, {"node": a.node})
        self.assertEqual(str(stride), str(b))
        self.assertEqual(str(numel), str(a * b))
        self.assertIs(type(q), torch.SymBool)
        self.assertEqual(q.__dict__, {"node": p.node})

        calls = []
        init = torch.SymInt.__init__

        def counted(self, node):
            calls.append(node)
            init(self, node)

        try:
            with mock.patch.object(torch.SymInt, "__init__", counted):
                self.assertIs(t.size(0).node, a.node)
            self.assertEqual(calls, [a.node])
        finally:
            sym_node._install_native_glue()
        with self.count_inits() as inits:
            t.size(0)
        self.assertEqual(inits, [])

        init = torch.SymBool.__init__
        try:
            with mock.patch.object(torch.SymBool, "__init__", counted):
                sym_node._install_native_glue()
                self.assertIs(
                    type(torch._C._symbolic._roundtrip_symbool(p)), torch.SymBool
                )
            self.assertEqual(calls, [a.node, p.node])
        finally:
            sym_node._install_native_glue()
        with self.count_inits() as inits:
            torch._C._symbolic._roundtrip_symbool(p)
        self.assertEqual(inits, [])

    def test_flag_off_unchanged(self):
        script = (
            "import types, torch\n"
            "from torch.fx.experimental import sym_node\n"
            "from torch.fx.experimental.symbolic_shapes import ShapeEnv\n"
            "ShapeEnv()\n"
            "print(isinstance(torch.SymInt.__add__, types.FunctionType),"
            " not sym_node._native_glue_originals)\n"
        )
        env = dict(os.environ)
        env.pop("CPP_SYMNODE", None)
        out = subprocess.check_output(
            [sys.executable, "-c", script], env=env, text=True
        )
        self.assertEqual(out.strip().splitlines()[-1], "True True")


instantiate_parametrized_tests(TestNativeExpr)
instantiate_parametrized_tests(TestNativeFloat)
instantiate_parametrized_tests(TestNativeInfinity)
instantiate_parametrized_tests(TestNativeCompoundAssumptions)
instantiate_parametrized_tests(TestNativeExprTools)
instantiate_parametrized_tests(TestNativeRelational)
instantiate_parametrized_tests(TestNativeSorting)
instantiate_parametrized_tests(TestNativeLattice)
instantiate_parametrized_tests(TestNativePrinter)
instantiate_parametrized_tests(TestNativeFunctions)
instantiate_parametrized_tests(TestNativeValueRanges)
instantiate_parametrized_tests(TestNativeStaticPasses)
instantiate_parametrized_tests(TestNativeShapeEnv)
instantiate_parametrized_tests(TestNativeShapeEnvSync)
instantiate_parametrized_tests(TestNativeSymNode)
instantiate_parametrized_tests(TestNativeSymNodeCompile)
instantiate_parametrized_tests(TestNativeSymIntGlue)


if __name__ == "__main__":
    run_tests()
