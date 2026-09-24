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
            lambda: arena.ask(arena.from_sympy(s0 + 1), "positive"),
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
        ]
        for v, f, expected in cases:
            arena = torch._C._symbolic._Arena()
            self.assertIs(arena.ask(arena.from_sympy(v), f), expected, f"({v}).is_{f}")
            self.assertIs(getattr(v, "is_" + f), expected)

    @parametrize("seed", range(8))
    def test_fuzz(self, seed):
        rng = random.Random(seed)
        # Signs of Adds are not ported yet, so only Add-free trees are expected
        # to be mostly answered natively.
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
            if not v.atoms(sympy.Add):
                answered += a
                unsupported += u
        self.assertGreater(answered, 10 * unsupported)


instantiate_parametrized_tests(TestNativeExpr)
instantiate_parametrized_tests(TestNativeCompoundAssumptions)


if __name__ == "__main__":
    run_tests()
