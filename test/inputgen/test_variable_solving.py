# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.variable.constants import INT64_MAX, INT64_MIN
from torch.testing._internal.inputgen.variable.solve import SolvableVariable
from torch.testing._internal.inputgen.variable.type import (
    ScalarDtype,
    SUPPORTED_TENSOR_DTYPES,
)


class TestSolvableVariable(TestCase):
    def test_bool_solver(self):
        s = SolvableVariable(bool)
        self.assertTrue(s.space.discrete.initialized)
        self.assertEqual(s.space.discrete.values, {False, True})

        s.Ne(0.0)
        self.assertEqual(str(s.space), "{True}")

        s = SolvableVariable(bool)
        s.Ne(1)
        self.assertEqual(str(s.space), "{False}")

        s = SolvableVariable(bool)
        s.Eq(1)
        self.assertEqual(str(s.space), "{True}")

        s = SolvableVariable(bool)
        s.NotIn([0.0, 1.0])
        self.assertEqual(str(s.space), "{}")

        s = SolvableVariable(bool)
        s.Gt(2)
        self.assertEqual(str(s.space), "{}")

    def test_int_solver(self):
        s = SolvableVariable(int)
        self.assertFalse(s.space.empty())

        s.Ne(0.0)
        self.assertEqual(str(s.space), "(-inf, 0) (0, inf)")
        self.assertFalse(s.space.contains(0))

        s = SolvableVariable(int)
        s.Eq(1.5)
        self.assertEqual(str(s.space), "{}")

        s = SolvableVariable(int)
        s.Ge(3.5)
        self.assertEqual(str(s.space), "[4, inf)")

        s.Gt(5.0)
        self.assertEqual(str(s.space), "(5, inf)")

        s = SolvableVariable(int)
        s.Gt(INT64_MAX)
        self.assertTrue(s.space.empty())

        s = SolvableVariable(int)
        s.Lt(INT64_MIN)
        self.assertTrue(s.space.empty())

        s = SolvableVariable(int)
        s.Gt(3)
        s.Lt(4)
        self.assertTrue(s.space.empty())

        s = SolvableVariable(int)
        s.Lt(float("inf"))
        self.assertEqual(str(s.space), "(-inf, inf)")

        s = SolvableVariable(int)
        s.Lt(float("-inf"))
        self.assertEqual(str(s.space), "{}")

    def test_float_solver(self):
        s = SolvableVariable(float)
        s.Ge(1.7976931348623157e308)
        self.assertEqual(str(s.space), "[1.7976931348623157e+308, inf]")

        s = SolvableVariable(float)
        s.Gt(float("inf"))
        self.assertEqual(str(s.space), "{}")
        self.assertTrue(s.space.empty())

        s = SolvableVariable(float)
        s.Eq(2)
        self.assertEqual(str(s.space), "{2.0}")
        s.Eq(1)
        self.assertEqual(str(s.space), "{}")
        self.assertTrue(s.space.empty())

        s = SolvableVariable(float)
        s.Eq(2)
        s.Lt(2)
        self.assertTrue(s.space.empty())

        s = SolvableVariable(float)
        s.Le(2)
        self.assertEqual(str(s.space), "[-inf, 2.0]")
        s.Eq(2)
        self.assertEqual(str(s.space), "{2.0}")

        s = SolvableVariable(float)
        s.Ne(3)
        self.assertEqual(str(s.space), "[-inf, 3.0) (3.0, inf]")
        s.Le(3.5)
        self.assertEqual(str(s.space), "[-inf, 3.0) (3.0, 3.5]")
        s.Ge(3)
        self.assertEqual(str(s.space), "(3.0, 3.5]")
        s.In([3, 3.5])
        self.assertEqual(str(s.space), "{3.5}")

        s = SolvableVariable(float)
        s.Lt(float("inf"))
        self.assertEqual(str(s.space), "[-inf, inf)")

        s = SolvableVariable(float)
        s.Lt(float("-inf"))
        self.assertEqual(str(s.space), "{}")

        s = SolvableVariable(float)
        s.Le(float("-inf"))
        self.assertEqual(str(s.space), "[-inf, -inf]")

        s = SolvableVariable(float)
        s.Ge(float("inf"))
        self.assertEqual(str(s.space), "[inf, inf]")

    def test_str_solver(self):
        s = SolvableVariable(str)

        s.In(["a", "b", "c", "d"])
        self.assertEqual(s.space.discrete.values, {"a", "b", "c", "d"})

        s.Ne("a")
        self.assertEqual(s.space.discrete.values, {"b", "c", "d"})

        s.Eq("b")
        self.assertEqual(s.space.discrete.values, {"b"})

        s.NotIn(["b", "c"])
        self.assertTrue(s.space.empty())

        with self.assertRaises(Exception):
            s.Le(3)

    def test_tensor_dtype_solver(self):
        s = SolvableVariable(torch.dtype)
        self.assertTrue(s.space.discrete.initialized)
        self.assertEqual(s.space.discrete.values, set(SUPPORTED_TENSOR_DTYPES))

        s.In([torch.bool, torch.uint8, torch.int8, torch.int32, torch.float32])
        self.assertEqual(
            s.space.discrete.values,
            {torch.bool, torch.uint8, torch.int8, torch.int32, torch.float32},
        )

        s.Ne(torch.float32)
        self.assertEqual(
            s.space.discrete.values, {torch.bool, torch.uint8, torch.int8, torch.int32}
        )

        s.NotIn([torch.bool, torch.uint8])
        self.assertEqual(s.space.discrete.values, {torch.int8, torch.int32})

        s.Eq(torch.int32)
        self.assertEqual(s.space.discrete.values, {torch.int32})

        with self.assertRaises(Exception):
            s.Ge(3)

    def test_scalar_dtype_solver(self):
        s = SolvableVariable(ScalarDtype)
        self.assertTrue(s.space.discrete.initialized)
        self.assertEqual(s.space.discrete.values, set(ScalarDtype))

        s.In([ScalarDtype.int, ScalarDtype.float])
        self.assertEqual(s.space.discrete.values, {ScalarDtype.int, ScalarDtype.float})

        s.Ne(ScalarDtype.float)
        self.assertEqual(s.space.discrete.values, {ScalarDtype.int})

        with self.assertRaises(Exception):
            s.Gt(3)


if __name__ == "__main__":
    run_tests()
