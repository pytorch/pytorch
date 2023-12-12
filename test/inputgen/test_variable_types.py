import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.inputgen.variable.type import (
    ScalarDtype,
    VariableType,
    check_vtype,
    convert_to_vtype,
    is_integer,
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

    def test_check(self):
        check_vtype(VariableType.Bool.value, 1.0)
        check_vtype(VariableType.Int.value, 1.0)
        check_vtype(VariableType.Float.value, 1.0)
        check_vtype(VariableType.String.value, "hello")
        check_vtype(VariableType.Tuple.value, (1, 2))
        check_vtype(VariableType.ScalarDtype.value, ScalarDtype.bool)
        check_vtype(VariableType.ScalarDtype.value, ScalarDtype.int)
        check_vtype(VariableType.ScalarDtype.value, ScalarDtype.float)
        check_vtype(VariableType.TensorDtype.value, torch.bool)

        with self.assertRaises(Exception):
            check_vtype(VariableType.Float.value, "1.0")
        with self.assertRaises(Exception):
            check_vtype(VariableType.String.value, 1)
        with self.assertRaises(Exception):
            check_vtype(VariableType.ScalarDtype.value, torch.int8)


if __name__ == "__main__":
    run_tests()
