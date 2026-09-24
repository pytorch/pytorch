# Owner(s): ["module: dynamic shapes"]
import contextlib
import os
import random
import subprocess
import sys
from unittest import mock

import sympy
from sympy.core.assumptions import _assume_defined, _assume_rules

import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental import symbolic_shapes
from torch.fx.experimental.sym_node import _NO_HINT, SymNode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import (
    CeilDiv,
    CeilToInt,
    CleanDiv,
    FloatPow,
    FloatTrueDiv,
    FloorDiv,
    FloorToInt,
    IntTrueDiv,
    IsNonOverlappingAndDenseIndicator,
    LShift,
    Max,
    Min,
    Mod,
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
            lambda: arena.from_sympy(sympy.Float(1.5)),
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
    NODE_TYPES = (Mod, PythonMod, FloorDiv, Max, Min, PowByNatural, FloatPow)
    NODE_TYPES += (FloatTrueDiv, IntTrueDiv, CeilToInt, FloorToInt, TruncToInt)
    NODE_TYPES += (RoundToInt, ToFloat, TruncToFloat, RoundDecimal)
    NODE_TYPES += (IsNonOverlappingAndDenseIndicator,)
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
            ("FloatPow", 2, 3),
            ("FloatPow", int_oo, 2),
            ("FloatTrueDiv", 1, 3),
            ("FloatTrueDiv", s0, 0),
            ("IntTrueDiv", 6, 3),
            ("IntTrueDiv", int_oo, 2),
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

    def test_to_int_unsupported(self):
        arena = torch._C._symbolic._Arena()
        cases = [
            ("RoundToInt", (int_oo,)),
            ("RoundToInt", (-int_oo,)),
            ("CeilToInt", (2**63 - 1,)),
            ("CeilToInt", (sympy.Rational(2**60 + 1, 3),)),
            ("ToFloat", (2,)),
            ("ToFloat", (int_oo,)),
            ("ToFloat", (sympy.Rational(1, 2),)),
            ("TruncToFloat", (int_oo,)),
            ("TruncToFloat", (sympy.Rational(1, 2),)),
            ("RoundDecimal", (3, 2)),
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
        if expected is None or not (expected.is_int or expected.is_bool):
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
            (zf, {}, None),
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
            (u0**-1, {u0: (1, 2)}, None),
            (Max(u0, u1, 3), {u0: (-oo, 0), u1: (1, 5)}, (3, 5)),
            (Min(u0, 3), {u0: (-oo, oo)}, (-oo, 3)),
            (TruncToInt(u0), {u0: (-3, oo)}, (-3, oo)),
            (RoundToInt(u0), {u0: (-3, 4)}, (-3, 4)),
            (RoundToInt(u0), {u0: (-3, oo)}, None),
            (FloorToInt(u0), {u0: (-3, oo)}, (-3, oo)),
            (ToFloat(u0), {}, None),
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
        for lo, hi in [(3, 2), (int_oo, 0), (0, sympy.Rational(1, 2)), (0, sympy.true)]:
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
        if expected is None or not (expected.is_int or expected.is_bool):
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
            (sympy.Rational(1, 2), {}, {}, None),
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
        for name in ("axioms", "replacements", "var_to_range", "size_like"):
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

    def make_env(self):
        env = ShapeEnv(_allow_native=True)
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
        a = self.pair(env, syms[0], int, 5)
        b = self.pair(env, syms[1], int, 7)
        n, p = self.call(*a, "lt", b)
        self.assertTrue(n.guard_bool("", 0))
        self.assertEqual(len(env.guards), 1)
        self.assertEqual(a[0].guard_int("", 0), 5)
        self.assertEqual(n.bool_(), True)

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

    def test_env_gone(self):
        import gc

        env, syms = self.make_env()
        n = env._native_env.make_node(syms[0], int, 5)
        self.assertEqual(n.add(n).hint, 10)
        del env
        gc.collect()
        with self.assertRaisesRegex(
            RuntimeError, "ShapeEnv of a native SymNode is gone"
        ):
            n.truediv(n)


instantiate_parametrized_tests(TestNativeExpr)
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


if __name__ == "__main__":
    run_tests()
