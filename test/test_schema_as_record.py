# Owner(s): ["module: tests"]

# pyre-unsafe
import unittest

import numpy as np

from caffe2.python.schema import as_record, Scalar, Struct
from torch.testing._internal.common_utils import run_tests


class TestAsRecord(unittest.TestCase):
    def test_named_tuple_list_returns_struct(self):
        result = as_record([("field1", np.float32), ("field2", np.int32)])
        self.assertIsInstance(result, Struct)

    def test_named_tuple_returns_correct_field_names(self):
        result = as_record([("a", np.float32), ("b", np.int32)])
        self.assertEqual(
            sorted(result.field_names()),
            sorted(["a", "b"]),
        )

    def test_single_named_tuple_returns_struct(self):
        result = as_record([("only_field", np.float64)])
        self.assertIsInstance(result, Struct)

    def test_dict_returns_struct(self):
        result = as_record({"x": np.float32, "y": np.int32})
        self.assertIsInstance(result, Struct)

    def test_scalar_passthrough(self):
        s = Scalar(dtype=np.float32)
        result = as_record(s)
        self.assertIs(result, s)

    def test_nested_named_tuples(self):
        result = as_record([("outer", [("inner", np.float32)])])
        self.assertIsInstance(result, Struct)
        inner = result["outer"]
        self.assertIsInstance(inner, Struct)

    def test_named_fields_with_scalar_values(self):
        s = Scalar(dtype=np.float32)
        result = as_record([("my_scalar", s)])
        self.assertIsInstance(result, Struct)


if __name__ == "__main__":
    run_tests()
