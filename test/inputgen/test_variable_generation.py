# Owner(s): ["module: tests"]

import math

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.variable.gen import VariableGenerator
from torch.testing._internal.inputgen.variable.solve import SolvableVariable
from torch.testing._internal.inputgen.variable.type import (
    ScalarDtype,
    SUPPORTED_TENSOR_DTYPES,
)
from torch.testing._internal.inputgen.variable.utils import nextdown, nextup


class TestUtils(TestCase):
    def test_nextup(self):
        self.assertEqual(nextup(-math.inf), -1.7976931348623157e308)
        self.assertEqual(nextup(-5e-324), 0.0)
        self.assertEqual(nextup(0.0), 5e-324)
        self.assertEqual(nextup(0.9999999999999999), 1.0),
        self.assertEqual(nextup(1.0), 1.0000000000000002)
        self.assertEqual(nextup(1.7976931348623157e308), math.inf)
        self.assertEqual(nextup(math.inf), math.inf)

    def test_nextdown(self):
        self.assertEqual(nextdown(math.inf), 1.7976931348623157e308)
        self.assertEqual(nextdown(1.0000000000000002), 1.0)
        self.assertEqual(nextdown(1.0), 0.9999999999999999),
        self.assertEqual(nextdown(5e-324), 0.0)
        self.assertEqual(nextdown(0.0), -5e-324)
        self.assertEqual(nextdown(-1.7976931348623157e308), -math.inf)
        self.assertEqual(nextdown(-math.inf), -math.inf)


class TestVariableGenerator(TestCase):
    def test_bool_generator(self):
        s = SolvableVariable(bool)
        s.Ne(1)
        v = VariableGenerator(s.space)
        self.assertEqual(v.gen(1), [False])

    def test_int_generator(self):
        s = SolvableVariable(int)
        s.Gt(2)
        s.Lt(5)
        s.Ne(3)
        v = VariableGenerator(s.space)
        self.assertEqual(v.gen(1), [4])

    def test_float_generator(self):
        s = SolvableVariable(float)
        s.Eq(2.4)
        v = VariableGenerator(s.space)
        self.assertEqual(v.gen(1), [2.4])

    def test_str_generator(self):
        s = SolvableVariable(str)
        s.In(["a"])
        v = VariableGenerator(s.space)
        self.assertEqual(v.gen(1), ["a"])

    def test_tensor_dtype_generator(self):
        s = SolvableVariable(torch.dtype)
        s.Ne(torch.int8)
        v = VariableGenerator(s.space)
        self.assertFalse(v.gen(1)[0] == torch.int8)
        self.assertTrue(v.gen(1)[0] in SUPPORTED_TENSOR_DTYPES)

    def test_scalar_dtype_solver(self):
        s = SolvableVariable(ScalarDtype)
        s.In([ScalarDtype.int, ScalarDtype.float])
        s.Ne(ScalarDtype.float)
        v = VariableGenerator(s.space)
        self.assertTrue(v.gen(1), [ScalarDtype.int])


if __name__ == "__main__":
    run_tests()
