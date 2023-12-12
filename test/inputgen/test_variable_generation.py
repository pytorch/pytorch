# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.variable.gen import VariableGenerator
from torch.testing._internal.inputgen.variable.solve import SolvableVariable
from torch.testing._internal.inputgen.variable.type import (
    ScalarDtype,
    SUPPORTED_TENSOR_DTYPES,
)


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
