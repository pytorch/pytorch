# Owner(s): ["module: dynamo"]
"""Tests for nb_int_impl: unified __int__ / int() protocol in Dynamo."""

import torch
import torch._dynamo.testing
from torch.testing._internal.common_utils import make_dynamo_test, run_tests, TestCase


class NbIntTests(TestCase):
    # --- int / bool (ConstantVariable) ---

    @make_dynamo_test
    def test_int_int(self):
        self.assertEqual(5, 5)

    @make_dynamo_test
    def test_bool_int(self):
        self.assertEqual(int(True), 1)
        self.assertEqual(int(False), 0)

    @make_dynamo_test
    def test_int_dunder_int(self):
        self.assertEqual((5).__int__(), 5)

    @make_dynamo_test
    def test_bool_dunder_int(self):
        self.assertEqual(True.__int__(), 1)

    # --- float (ConstantVariable) ---

    @make_dynamo_test
    def test_float_int(self):
        self.assertEqual(int(3.14), 3)

    @make_dynamo_test
    def test_float_negative_int(self):
        self.assertEqual(int(-2.9), -2)

    @make_dynamo_test
    def test_float_dunder_int(self):
        self.assertEqual((3.14).__int__(), 3)

    # --- TypeError for non-int types ---

    def test_complex_int_raises(self):
        def fn(x):
            try:
                return int(1 + 2j)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_none_int_raises(self):
        def fn(x):
            try:
                return int(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_list_int_raises(self):
        def fn(x):
            try:
                return int([1, 2])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_dict_int_raises(self):
        def fn(x):
            try:
                return int({})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_set_int_raises(self):
        def fn(x):
            try:
                return int(set())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_tuple_int_raises(self):
        def fn(x):
            try:
                return int(())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- str parsing (constructor, not nb_int) ---

    @make_dynamo_test
    def test_str_int_parsing(self):
        self.assertEqual(int("123"), 123)

    @make_dynamo_test
    def test_str_int_base(self):
        self.assertEqual(int("ff", 16), 255)

    # --- UserDefinedObjectVariable with __int__ ---

    def test_user_defined_int(self):
        class MyInt:
            def __init__(self, v):
                self.v = v

            def __int__(self):
                return self.v

        obj = MyInt(42)

        def fn(x):
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42
        )

    def test_user_defined_dunder_int(self):
        class MyInt:
            def __init__(self, v):
                self.v = v

            def __int__(self):
                return self.v

        obj = MyInt(7)

        def fn(x):
            return obj.__int__()

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 7
        )

    def test_user_defined_no_int_raises(self):
        class NoInt:
            pass

        obj = NoInt()

        def fn(x):
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("int() argument must be a string", result)

    def test_int_returning_non_int_raises(self):
        class Bad:
            def __int__(self):
                return "not an int"  # noqa: PLE0305

        obj = Bad()

        def fn(x):
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("__int__ returned non-int", result)

    def test_int_raising_exception_propagates(self):
        class RaisingInt:
            def __int__(self):
                raise ValueError("custom error")

        obj = RaisingInt()

        def fn(x):
            try:
                return int(obj)
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    def test_user_defined_staticmethod_int(self):
        class StaticInt:
            @staticmethod
            def __int__():
                return 3

        obj = StaticInt()

        def fn(x):
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 3
        )

    def test_user_defined_classmethod_int(self):
        class ClassInt:
            @classmethod
            def __int__(cls):
                return 4

        obj = ClassInt()

        def fn(x):
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 4
        )

    # --- nb_index fallback (PyNumber_Long step 3) ---

    def test_index_fallback_for_int(self):
        class HasIndex:
            def __index__(self):
                return 42

        obj = HasIndex()

        def fn(x):
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42
        )

    def test_index_fallback_no_int_no_index_raises(self):
        class NoSlots:
            pass

        obj = NoSlots()

        def fn(x):
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertIn("int() argument must be a string", result)
        self.assertEqual(result, eager_result)

    # --- Tensor ---

    def test_tensor_int_dtype(self):
        def fn(x):
            return int(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5)

    def test_tensor_float_dtype(self):
        def fn(x):
            return int(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(3.7))
        self.assertEqual(result, 3)

    def test_tensor_complex_raises(self):
        def fn(x):
            try:
                return int(x)
            except RuntimeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(1 + 2j)
        )
        eager_result = fn(torch.tensor(1 + 2j))
        self.assertIn(
            "value cannot be converted to type int64_t without overflow", result
        )
        self.assertEqual(result, eager_result)

    def test_tensor_dunder_int(self):
        def fn(x):
            return x.__int__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5)

    # --- SymNodeVariable ---

    def test_symnode_int(self):
        def fn(x):
            return int(x.size(0))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(10, 20))
        self.assertEqual(result, 10)


if __name__ == "__main__":
    run_tests()
