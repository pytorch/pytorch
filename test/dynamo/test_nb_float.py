# Owner(s): ["module: dynamo"]
"""Tests for nb_float_impl: unified __float__ / float() protocol in Dynamo."""

import torch
import torch._dynamo.testing
from torch.testing._internal.common_utils import (
    make_dynamo_test,
    run_tests,
    skipIfCrossRef,
    TestCase,
)


class NbFloatTests(TestCase):
    # --- float / int / bool (ConstantVariable) ---

    @make_dynamo_test
    def test_float_float(self):
        self.assertEqual(float(3.14), 3.14)  # noqa: UP018

    @make_dynamo_test
    def test_int_float(self):
        self.assertEqual(float(5), 5.0)

    @make_dynamo_test
    def test_bool_float(self):
        self.assertEqual(float(True), 1.0)
        self.assertEqual(float(False), 0.0)

    @make_dynamo_test
    def test_float_dunder_float(self):
        self.assertEqual((3.14).__float__(), 3.14)

    @make_dynamo_test
    def test_int_dunder_float(self):
        self.assertEqual((5).__float__(), 5.0)

    @make_dynamo_test
    def test_bool_dunder_float(self):
        self.assertEqual(True.__float__(), 1.0)

    # --- TypeError for non-float types ---

    def test_complex_float_raises(self):
        def fn(x):
            try:
                return float(1 + 2j)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_none_float_raises(self):
        def fn(x):
            try:
                return float(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_list_float_raises(self):
        def fn(x):
            try:
                return float([1, 2])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_dict_float_raises(self):
        def fn(x):
            try:
                return float({})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_set_float_raises(self):
        def fn(x):
            try:
                return float(set())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_tuple_float_raises(self):
        def fn(x):
            try:
                return float(())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- str parsing (constructor, not nb_float) ---

    @make_dynamo_test
    def test_str_float_parsing(self):
        self.assertEqual(float("3.14"), 3.14)

    @make_dynamo_test
    def test_str_float_int(self):
        self.assertEqual(float("123"), 123.0)

    # --- UserDefinedObjectVariable with __float__ ---

    def test_user_defined_float(self):
        class MyFloat:
            def __init__(self, v):
                self.v = v

            def __float__(self):
                return self.v

        obj = MyFloat(3.14)

        def fn(x):
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 3.14
        )

    def test_user_defined_dunder_float(self):
        class MyFloat:
            def __init__(self, v):
                self.v = v

            def __float__(self):
                return self.v

        obj = MyFloat(2.71)

        def fn(x):
            return obj.__float__()

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 2.71
        )

    def test_user_defined_no_float_raises(self):
        class NoFloat:
            pass

        obj = NoFloat()

        def fn(x):
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("float() argument must be a string", result)

    def test_float_returning_non_float_raises(self):
        class Bad:
            def __float__(self):
                return "not a float"

        obj = Bad()

        def fn(x):
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("__float__ returned non-float", result)

    def test_float_raising_exception_propagates(self):
        class RaisingFloat:
            def __float__(self):
                raise ValueError("custom error")

        obj = RaisingFloat()

        def fn(x):
            try:
                return float(obj)
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    def test_user_defined_staticmethod_float(self):
        class StaticFloat:
            @staticmethod
            def __float__():
                return 3.0

        obj = StaticFloat()

        def fn(x):
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 3.0
        )

    def test_user_defined_classmethod_float(self):
        class ClassFloat:
            @classmethod
            def __float__(cls):
                return 4.0

        obj = ClassFloat()

        def fn(x):
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 4.0
        )

    # --- nb_index fallback (PyNumber_Float step 3) ---

    def test_index_fallback_for_float(self):
        class HasIndex:
            def __index__(self):
                return 42

        obj = HasIndex()

        def fn(x):
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42.0
        )

    def test_index_fallback_no_float_no_index_raises(self):
        class NoSlots:
            pass

        obj = NoSlots()

        def fn(x):
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertIn("float() argument must be a string", result)
        self.assertEqual(result, eager_result)

    # --- Tensor ---

    @skipIfCrossRef
    def test_tensor_int_dtype(self):
        def fn(x):
            return float(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5.0)

    @skipIfCrossRef
    def test_tensor_float_dtype(self):
        def fn(x):
            return float(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(3.14))
        self.assertAlmostEqual(result, 3.14, places=2)

    @skipIfCrossRef
    def test_tensor_complex_raises(self):
        def fn(x):
            try:
                return float(x)
            except RuntimeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(1 + 2j)
        )
        eager_result = fn(torch.tensor(1 + 2j))
        self.assertIn(
            "value cannot be converted to type double without overflow", result
        )
        self.assertEqual(result, eager_result)

    @skipIfCrossRef
    def test_tensor_dunder_float(self):
        def fn(x):
            return x.__float__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5.0)

    # --- SymNodeVariable ---

    @skipIfCrossRef
    def test_symnode_float(self):
        def fn(x):
            return float(x.size(0))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(10, 20))
        self.assertEqual(result, 10.0)


if __name__ == "__main__":
    run_tests()
