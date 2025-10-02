# mypy: ignore-errors
# Owner(s): ["module: unknown"]
import token
from itertools import product
from pathlib import Path

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


MM = "@"

BINARY_RETURNS_BOOL = "!=", "<", "<=", "==", ">", ">="
BINARY_ACCEPTS_FLOAT_OR_INT = "%", "*", "**", "+", "-", "/", "//"
BINARY_ACCEPTS_INT_ONLY = "&", "<<", ">>", "^", "|"
BINARY_OPS = (
    *BINARY_RETURNS_BOOL,
    *BINARY_ACCEPTS_FLOAT_OR_INT,
    *BINARY_ACCEPTS_INT_ONLY,
    MM,
)

BINARY_RETURNS_FLOAT = ("/",)

UNARY_ACCEPTS_FLOAT_OR_INT = "+", "-"
UNARY_ACCEPTS_INT_ONLY = ("~",)
UNARY_OPS = *UNARY_ACCEPTS_FLOAT_OR_INT, *UNARY_ACCEPTS_INT_ONLY

PUNCTUATION = ",", ";"

OPERATORS = *UNARY_OPS, *BINARY_OPS, *PUNCTUATION

FLOATS = 1.5, torch.tensor((2.5, 3.5))
INTS = 3, torch.tensor((1, 2))
ALL = *FLOATS, *INTS

TYPE_TEST_FILE = Path(__file__).parent / "pass/arithmetic_ops.py"


class TestPythonOperators(TestCase):
    # Prove that UNARY_OPS, BINARY_OPS, and OPERATORS are correct and complete
    def test_operators_are_correct_and_complete(self):
        self.assertFalse(set(OPERATORS).difference(token.EXACT_TOKEN_TYPES))

        unary, binary, punctuation = {}, {}, {}

        for op in token.EXACT_TOKEN_TYPES:
            if op in PUNCTUATION:
                punctuation[op] = True
            else:
                try:
                    unary[op] = compile(f"{op}1 ; {op}a", op, "single")
                except SyntaxError:
                    pass
                try:
                    binary[op] = compile(f"2 {op} 3 ; a {op} b", op, "single")
                except SyntaxError:
                    pass

        self.assertEqual(sorted(unary), sorted(UNARY_OPS))
        self.assertEqual(sorted(binary), sorted(BINARY_OPS))
        self.assertEqual(sorted(punctuation), sorted(PUNCTUATION))

    def test_type_tests_are_complete(self):
        binary, unary = {}, []

        with TYPE_TEST_FILE.open() as fp:
            # Looking for lines like:  assert_type(TENSOR ^ BOOL, Tensor)
            # But not:                 assert_type(BOOL ^ BINARY, Binary)
            lines = (i for i in fp if "TENSOR" in i)
            for line in lines:
                if expr := line.partition("assert_type(")[2].partition(",")[0]:
                    if expr[0].isalpha():
                        # ** formats differently from all other operators
                        a, op, b = expr.replace("**", " ** ").split()
                        binary.setdefault(op, []).append((a, b))
                    else:
                        unary.append(expr[0])

        self.assertEqual(sorted(unary), sorted(UNARY_OPS))
        self.assertEqual(sorted(binary), sorted(BINARY_OPS))
        value, *values = binary.values()
        self.assertEqual(values, [value] * len(values))

    @parametrize("a, op, b", product(ALL, BINARY_OPS, ALL))
    def test_binary(self, a, op, b):
        try:
            r = eval(f"a {op} b")
        except Exception as e:
            r = e

        any_tensor = isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
        any_float = _any_float(a, b)
        returns_float = any_float or op in BINARY_RETURNS_FLOAT

        if op == MM:
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                self.assertIsInstance(r, TypeError)
            elif a is b:
                self.assertIsInstance(r, torch.Tensor)
            else:
                self.assertIsInstance(r, RuntimeError)

        elif any_tensor:
            if op in BINARY_ACCEPTS_INT_ONLY and any_float:
                # See https://github.com/pytorch/pytorch/issues/15754
                self.assertIsInstance(r, NotImplementedError)
            else:
                self.assertIsInstance(r, torch.Tensor)

                if op in BINARY_RETURNS_BOOL:
                    self.assertEqual(r.dtype, torch.bool)
                elif op in BINARY_ACCEPTS_INT_ONLY:
                    self.assertFalse(r.dtype.is_floating_point)
                elif op in BINARY_ACCEPTS_FLOAT_OR_INT:
                    self.assertEqual(r.dtype.is_floating_point, returns_float)
                else:
                    self.assertFalse("Logic error")

        elif op in BINARY_RETURNS_BOOL:
            self.assertIsInstance(r, bool)

        elif op in BINARY_ACCEPTS_INT_ONLY:
            if any_float:
                self.assertIsInstance(r, TypeError)
            else:
                self.assertIsInstance(r, int)

        elif returns_float:
            self.assertIsInstance(r, float)

        else:
            self.assertIsInstance(r, int)

    @parametrize("op, a", product(UNARY_OPS, ALL))
    def test_unary(self, op, a):
        try:
            r = eval(f"{op} a")
        except Exception as e:
            r = e

        if op in UNARY_ACCEPTS_INT_ONLY and _any_float(a):
            self.assertIsInstance(r, TypeError)
        elif isinstance(a, torch.Tensor):
            self.assertIsInstance(r, torch.Tensor)
        elif op in UNARY_ACCEPTS_INT_ONLY:
            self.assertIsInstance(r, int)
        elif isinstance(a, float):
            self.assertIsInstance(r, float)
        else:
            self.assertIsInstance(r, int)


def _any_float(*x):
    for i in x:
        if isinstance(i, float) or (
            isinstance(i, torch.Tensor) and i.dtype.is_floating_point
        ):
            return True
    return False


instantiate_parametrized_tests(TestPythonOperators)


if __name__ == "__main__":
    run_tests()
