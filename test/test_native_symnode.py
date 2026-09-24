# Owner(s): ["module: dynamic shapes"]
import os
import random
import subprocess
import sys

import sympy
from sympy.core.assumptions import _assume_defined, _assume_rules

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._sympy.functions import Mod, PythonMod
from torch.utils._sympy.numbers import int_oo


NativeUnsupported = torch._C._symbolic.NativeUnsupported

s0 = sympy.Symbol("s0", integer=True, positive=True)
s1 = sympy.Symbol("s1", integer=True, positive=True)
u0 = sympy.Symbol("u0", integer=True)
zf = sympy.Symbol("zf", real=True, positive=True)
LEAVES = [s0, s1, u0, zf, *map(sympy.Integer, range(-3, 4))]
LEAVES += [sympy.Rational(1, 2), sympy.Rational(-2, 3)]
FACTS = sorted(_assume_defined)
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
    FUNCTIONS = {"Mod": Mod, "PythonMod": PythonMod}

    def check_call(self, name, a, b):
        """Returns the sympy result, or None if native raised
        NativeUnsupported."""
        arena = torch._C._symbolic._Arena()
        args = [arena.from_sympy(a), arena.from_sympy(b)]
        try:
            expected = self.FUNCTIONS[name](a, b)
        except (ZeroDivisionError, AssertionError, TypeError):
            with self.assertRaises(NativeUnsupported, msg=f"{name}({a}, {b})"):
                arena.function(name, args)
            return None
        try:
            r = arena.function(name, args)
        except NativeUnsupported:
            return None
        got = arena.to_sympy(r)
        self.assertEqual(got, expected, f"{name}({a}, {b})")
        self.assertEqual(type(got), type(expected), f"{name}({a}, {b})")
        self.assertEqual(got.args, expected.args, f"{name}({a}, {b})")
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
        facts = list(FACTS)
        rng.shuffle(facts)
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
        ]
        for name, a, b in cases:
            args = [arena.from_sympy(sympy.sympify(x)) for x in (a, b)]
            with self.assertRaises(NativeUnsupported, msg=f"{name}({a}, {b})"):
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
        rng = random.Random(0)
        for v in cases:
            self.assertTrue(self.check_expr(v, rng), f"{v}")

    def test_sorting(self):
        items = [Mod(s0, 2), Mod(s0, 3), Mod(s1, 2), Mod(u0, 2), s0, u0]
        items += [Mod(s0 + 1, s1), s0 + Mod(s0, 2), sympy.Integer(2), s0**2]
        items += [sympy.Eq(s0, 1, evaluate=False), sympy.Not(u0), sympy.true]
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
                    if isinstance(r, (Mod, PythonMod)):
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


instantiate_parametrized_tests(TestNativeExpr)
instantiate_parametrized_tests(TestNativeCompoundAssumptions)
instantiate_parametrized_tests(TestNativeExprTools)
instantiate_parametrized_tests(TestNativeRelational)
instantiate_parametrized_tests(TestNativeSorting)
instantiate_parametrized_tests(TestNativeLattice)
instantiate_parametrized_tests(TestNativePrinter)
instantiate_parametrized_tests(TestNativeFunctions)


if __name__ == "__main__":
    run_tests()
