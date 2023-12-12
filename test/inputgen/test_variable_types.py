# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inputgen.variable.type import (
    convert_to_vtype,
    invalid_vtype,
    is_integer,
    ScalarDtype,
    VariableType,
)


class TestVariableType(TestCase):
    def test_variable_type(self):
        self.assertEqual(VariableType.Bool.value, bool)
        self.assertEqual(VariableType.Int.value, int)
        self.assertEqual(VariableType.Float.value, float)
        self.assertEqual(VariableType.String.value, str)
        self.assertEqual(VariableType.Tuple.value, tuple)
        self.assertEqual(VariableType.ScalarDtype.value, ScalarDtype)
        self.assertEqual(VariableType.TensorDtype.value, torch.dtype)

    def test_is_integer(self):
        self.assertTrue(is_integer(1))
        self.assertTrue(is_integer(5))
        self.assertTrue(is_integer(-5))
        self.assertTrue(is_integer(1.0))
        self.assertTrue(is_integer(0.0))
        self.assertTrue(is_integer(-3.0))
        self.assertTrue(is_integer(True))
        self.assertTrue(is_integer(False))
        self.assertFalse(is_integer(3.5))
        self.assertFalse(is_integer(float("inf")))
        self.assertFalse(is_integer(float("-inf")))
        self.assertFalse(is_integer(float("nan")))

    def test_convert(self):
        self.assertEqual(convert_to_vtype(VariableType.Bool.value, 1.0), True)
        self.assertEqual(convert_to_vtype(VariableType.Int.value, False), 0)
        self.assertEqual(convert_to_vtype(VariableType.Float.value, 3), 3.0)
        self.assertEqual(
            convert_to_vtype(VariableType.Int.value, float("inf")), float("inf")
        )

    def test_invalid_vtype(self):
        self.assertFalse(invalid_vtype(VariableType.Bool.value, 1.0))
        self.assertFalse(invalid_vtype(VariableType.Int.value, 1.0))
        self.assertFalse(invalid_vtype(VariableType.Float.value, 1.0))
        self.assertFalse(invalid_vtype(VariableType.String.value, "hello"))
        self.assertFalse(invalid_vtype(VariableType.Tuple.value, (1, 2)))
        self.assertFalse(
            invalid_vtype(VariableType.ScalarDtype.value, ScalarDtype.bool)
        )
        self.assertFalse(invalid_vtype(VariableType.ScalarDtype.value, ScalarDtype.int))
        self.assertFalse(
            invalid_vtype(VariableType.ScalarDtype.value, ScalarDtype.float)
        )
        self.assertFalse(invalid_vtype(VariableType.TensorDtype.value, torch.bool))
        self.assertTrue(invalid_vtype(VariableType.Float.value, "1.0"))
        self.assertTrue(invalid_vtype(VariableType.String.value, 1))
        self.assertTrue(invalid_vtype(VariableType.ScalarDtype.value, torch.int8))


if __name__ == "__main__":
    run_tests()
