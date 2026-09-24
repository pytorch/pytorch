# Owner(s): ["module: dynamic shapes"]
import os
import random
import subprocess
import sys

import sympy

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


def random_tree(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        return rng.choice(LEAVES)
    op = rng.choice(["add", "mul", "sub", "neg", "pow"])
    if op == "pow":
        return (op, random_tree(rng, depth - 1), rng.randint(-2, 3))
    if op == "neg":
        return (op, random_tree(rng, depth - 1))
    n = rng.randint(2, 3) if op in ("add", "mul") else 2
    return (op, *(random_tree(rng, depth - 1) for _ in range(n)))


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
        ]
        for f in cases:
            with self.assertRaises(NativeUnsupported):
                f()
        other = torch._C._symbolic._Arena()
        with self.assertRaisesRegex(RuntimeError, "different arena"):
            arena.add([big, other.integer(1)])


instantiate_parametrized_tests(TestNativeExpr)


if __name__ == "__main__":
    run_tests()
