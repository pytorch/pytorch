# Owner(s): ["module: dynamo"]
"""Tests for nb_negative / generic_neg: unary - via PyNumber_Negative in Dynamo."""

import operator

import torch
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class NbNegativeTests(TestCase):
    # --- int (ConstantVariable, long_neg) ---

    @make_dynamo_test
    def test_int_positive(self):
        self.assertEqual(-42, -42)

    @make_dynamo_test
    def test_int_negative(self):
        self.assertEqual(-(-7), 7)  # noqa: B002

    @make_dynamo_test
    def test_int_zero(self):
        self.assertEqual(-0, 0)

    @make_dynamo_test
    def test_int_large(self):
        self.assertEqual(-(2**100), -(2**100))

    # --- float (ConstantVariable, float_neg) ---

    @make_dynamo_test
    def test_float_positive(self):
        self.assertEqual(-3.14, -3.14)

    @make_dynamo_test
    def test_float_negative(self):
        self.assertEqual(-(-2.5), 2.5)  # noqa: B002

    @make_dynamo_test
    def test_float_zero(self):
        import math

        result = -0.0
        self.assertEqual(math.copysign(1.0, result), -1.0)

    @make_dynamo_test
    def test_float_neg_zero(self):
        import math

        result = -(-0.0)  # noqa: B002
        self.assertEqual(math.copysign(1.0, result), 1.0)

    @make_dynamo_test
    def test_float_inf(self):
        import math

        self.assertEqual(-math.inf, -math.inf)

    @make_dynamo_test
    def test_float_neg_inf(self):
        import math

        self.assertEqual(-(-math.inf), math.inf)  # noqa: B002

    def test_float_nan(self):
        import math

        def fn(x):
            return -float("nan")

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(math.isnan(result))

    # --- complex (ConstantVariable, complex_neg) ---

    @make_dynamo_test
    def test_complex(self):
        self.assertEqual(-(3 + 4j), (-3 - 4j))

    @make_dynamo_test
    def test_complex_zero(self):
        self.assertEqual(-(0 + 0j), 0j)

    @make_dynamo_test
    def test_complex_real_only(self):
        self.assertEqual(-(5 + 0j), (-5 + 0j))

    @make_dynamo_test
    def test_complex_imag_only(self):
        self.assertEqual(-(0 + 3j), -3j)

    # --- bool (ConstantVariable, inherits long_neg from int) ---

    @make_dynamo_test
    def test_bool_true(self):
        self.assertEqual(-True, -1)

    @make_dynamo_test
    def test_bool_false(self):
        self.assertEqual(-False, 0)

    def test_bool_result_type(self):
        # CPython: -True returns int, not bool (long_neg returns PyLong)
        def fn(x):
            return -True

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIsInstance(result, int)
        self.assertNotIsInstance(result, bool)

    # --- operator.neg on constants ---

    @make_dynamo_test
    def test_operator_neg_int(self):
        self.assertEqual(operator.neg(10), -10)

    @make_dynamo_test
    def test_operator_neg_float(self):
        self.assertEqual(operator.neg(2.5), -2.5)

    @make_dynamo_test
    def test_operator_neg_bool(self):
        self.assertEqual(operator.neg(True), -1)

    @make_dynamo_test
    def test_operator_neg_complex(self):
        self.assertEqual(operator.neg(1 + 2j), (-1 - 2j))

    # --- __neg__ dunder method on constants ---

    @make_dynamo_test
    def test_dunder_neg_int(self):
        self.assertEqual((42).__neg__(), -42)

    @make_dynamo_test
    def test_dunder_neg_float(self):
        self.assertEqual((3.14).__neg__(), -3.14)

    @make_dynamo_test
    def test_dunder_neg_bool(self):
        self.assertEqual(True.__neg__(), -1)

    @make_dynamo_test
    def test_dunder_neg_complex(self):
        self.assertEqual((1 + 2j).__neg__(), (-1 - 2j))

    # --- TypeError for types without nb_negative ---
    # CPython: "bad operand type for unary -: '<type>'"

    def test_str_neg_raises(self):
        def fn(x):
            try:
                return -"hello"
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_list_neg_raises(self):
        def fn(x):
            try:
                return -[1, 2, 3]
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_dict_neg_raises(self):
        def fn(x):
            try:
                return -{"a": 1}
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_set_neg_raises(self):
        def fn(x):
            try:
                return -{1, 2}
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_tuple_neg_raises(self):
        def fn(x):
            try:
                return -()
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_none_neg_raises(self):
        def fn(x):
            try:
                return -None
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_bytes_neg_raises(self):
        def fn(x):
            try:
                return -b"hello"
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_operator_neg_none_raises(self):
        def fn(x):
            try:
                return operator.neg(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- Tensor (TensorVariable, proxy-based) ---

    def test_tensor_neg(self):
        def fn(x):
            return -x

        x = torch.randn(3, 4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    def test_tensor_neg_integer_dtype(self):
        def fn(x):
            return -x

        x = torch.tensor([1, -2, 3])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    def test_tensor_neg_complex_dtype(self):
        def fn(x):
            return -x

        x = torch.tensor([1 + 2j, -3 + 4j])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    def test_tensor_neg_creates_graph_node(self):
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            return -x

        fn(torch.randn(3))
        self.assertEqual(len(backend.graphs), 1)
        graph_str = backend.graphs[0].print_readable(False)
        self.assertIn("neg", graph_str.lower())

    def test_tensor_operator_neg(self):
        def fn(x):
            return operator.neg(x)

        x = torch.randn(5)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    def test_tensor_dunder_neg(self):
        def fn(x):
            return x.__neg__()

        x = torch.randn(4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- SymNodeVariable ---

    def test_symnode_neg(self):
        def fn(x):
            s = x.size(0)
            return x[:(-s)]

        x = torch.randn(10, 5)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    def test_symnode_operator_neg(self):
        def fn(x):
            s = x.size(0)
            return x[: operator.neg(s)]

        x = torch.randn(8, 4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    def test_symnode_dunder_neg(self):
        def fn(x):
            s = x.size(0)
            return x[: s.__neg__()]

        x = torch.randn(6, 3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- UserDefinedObjectVariable with __neg__ ---

    def test_user_defined_neg(self):
        class MyNum:
            def __init__(self, val):
                self.val = val

            def __neg__(self):
                return MyNum(-self.val)

        obj = MyNum(10)

        def fn(x):
            return (-obj).val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -10)

    def test_user_defined_operator_neg(self):
        class MyNum:
            def __init__(self, val):
                self.val = val

            def __neg__(self):
                return MyNum(-self.val)

        obj = MyNum(7)

        def fn(x):
            return operator.neg(obj).val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -7)

    def test_user_defined_dunder_neg(self):
        class MyNum:
            def __init__(self, val):
                self.val = val

            def __neg__(self):
                return MyNum(-self.val)

        obj = MyNum(3)

        def fn(x):
            return obj.__neg__().val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -3)

    def test_user_defined_no_neg_raises(self):
        class NoNeg:
            pass

        obj = NoNeg()

        def fn(x):
            try:
                return -obj
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_user_defined_neg_returns_arbitrary_type(self):
        # CPython does NOT validate __neg__ return type
        class WeirdNeg:
            def __neg__(self):
                return "negated"

        obj = WeirdNeg()

        def fn(x):
            return -obj

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "negated")

    def test_user_defined_neg_raising_exception_propagates(self):
        class RaisingNeg:
            def __neg__(self):
                raise ValueError("custom error")

        obj = RaisingNeg()

        def fn(x):
            try:
                return -obj
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    def test_user_defined_subclass_neg(self):
        class Base:
            def __init__(self, v):
                self.v = v

            def __neg__(self):
                return type(self)(-self.v)

        class Sub(Base):
            pass

        obj = Sub(5)

        def fn(x):
            result = -obj
            return result.v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -5)

    # --- nn.Module with __neg__ ---

    def test_nn_module_with_neg(self):
        class NegModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def __neg__(self):
                return torch.tensor(-1.0)

            def forward(self, x):
                return x

        mod = NegModule()

        def fn(x):
            return (-mod) + x

        x = torch.tensor(5.0)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- Metaclass with __neg__ (UserDefinedClassVariable path) ---
    # In CPython, -SomeClass calls type(SomeClass)->nb_negative,
    # i.e. the metaclass's __neg__, not the class's.

    def test_metaclass_neg(self):
        class Meta(type):
            def __neg__(cls):
                return f"negated {cls.__name__}"

        class A(metaclass=Meta):
            pass

        def fn(x):
            return -A

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "negated A")

    def test_metaclass_no_neg_raises(self):
        class Bar(type):
            pass

        class B(metaclass=Bar):
            pass

        def fn(x):
            try:
                return -B
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_metaclass_inherited_neg(self):
        class Meta(type):
            def __neg__(cls):
                return f"negated {cls.__name__}"

        class SubMeta(Meta):
            pass

        class D(metaclass=SubMeta):
            pass

        def fn(x):
            return -D

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "negated D")

    def test_default_metaclass_neg_raises(self):
        # type itself has no __neg__, so -SomeClass raises TypeError
        class Plain:
            pass

        def fn(x):
            try:
                return -Plain
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- Unbound method calls: Type.__neg__(instance) ---

    def test_unbound_user_defined_neg(self):
        # MyNum.__neg__(obj) — unbound method call on user-defined class
        class MyNum:
            def __init__(self, v):
                self.v = v

            def __neg__(self):
                return MyNum(-self.v)

        obj = MyNum(5)

        def fn(x):
            return MyNum.__neg__(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -5)

    def test_unbound_parent_class_neg(self):
        # Base.__neg__(sub_instance) — calling parent's __neg__ on a subclass instance
        class Base:
            def __init__(self, v):
                self.v = v

            def __neg__(self):
                return type(self)(-self.v)

        class Sub(Base):
            pass

        obj = Sub(10)

        def fn(x):
            return Base.__neg__(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -10)

    @make_dynamo_test
    def test_unbound_builtin_int_neg(self):
        self.assertEqual(int.__neg__(42), -42)

    @make_dynamo_test
    def test_unbound_builtin_float_neg(self):
        self.assertEqual(float.__neg__(3.14), -3.14)

    @make_dynamo_test
    def test_unbound_builtin_complex_neg(self):
        self.assertEqual(complex.__neg__(1 + 2j), (-1 - 2j))

    @make_dynamo_test
    def test_unbound_builtin_bool_neg(self):
        self.assertEqual(bool.__neg__(True), -1)

    # --- Double negation ---

    @make_dynamo_test
    def test_double_neg_int(self):
        self.assertEqual(-(-42), 42)  # noqa: B002

    @make_dynamo_test
    def test_double_neg_float(self):
        self.assertEqual(-(-3.14), 3.14)  # noqa: B002

    def test_double_neg_tensor(self):
        def fn(x):
            return -(-x)  # noqa: B002

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- Negation used in expressions ---

    def test_neg_in_arithmetic(self):
        def fn(x):
            return x + (-x)

        x = torch.randn(4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, torch.zeros(4))

    def test_neg_constant_in_tensor_op(self):
        def fn(x):
            val = -5
            return x + val

        x = torch.ones(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))


if __name__ == "__main__":
    run_tests()
