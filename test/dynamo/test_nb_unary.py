# Owner(s): ["module: dynamo"]
"""Tests for nb_negative / nb_positive: unary ops via PyNumber_Negative/Positive."""

import operator

import torch
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
    subtest,
)


UNARY_OPS = [
    subtest((operator.neg, "-", "__neg__"), name="neg"),
    subtest((operator.pos, "+", "__pos__"), name="pos"),
]


class NbUnaryTests(TestCase):
    # --- int (ConstantVariable) ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_int(self, op, symbol, dunder):
        self.assertEqual(op(42), op(42))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_int_negative_value(self, op, symbol, dunder):
        self.assertEqual(op(-7), op(-7))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_int_zero(self, op, symbol, dunder):
        self.assertEqual(op(0), op(0))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_int_large(self, op, symbol, dunder):
        self.assertEqual(op(2**100), op(2**100))

    # --- float (ConstantVariable) ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float(self, op, symbol, dunder):
        self.assertEqual(op(3.14), op(3.14))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float_negative_value(self, op, symbol, dunder):
        self.assertEqual(op(-2.5), op(-2.5))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_float_zero_copysign(self, op, symbol, dunder):
        import math

        def fn(x):
            return op(0.0)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        expected = op(0.0)
        self.assertEqual(math.copysign(1.0, result), math.copysign(1.0, expected))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_float_double_apply_zero(self, op, symbol, dunder):
        import math

        def fn(x):
            return op(op(0.0))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        expected = op(op(0.0))
        self.assertEqual(math.copysign(1.0, result), math.copysign(1.0, expected))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float_inf(self, op, symbol, dunder):
        import math

        self.assertEqual(op(math.inf), op(math.inf))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_float_neg_inf(self, op, symbol, dunder):
        import math

        self.assertEqual(op(-math.inf), op(-math.inf))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_float_nan(self, op, symbol, dunder):
        import math

        def fn(x):
            return op(float("nan"))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(math.isnan(result))

    # --- complex (ConstantVariable) ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex(self, op, symbol, dunder):
        self.assertEqual(op(3 + 4j), op(3 + 4j))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex_zero(self, op, symbol, dunder):
        self.assertEqual(op(0 + 0j), op(0 + 0j))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex_real_only(self, op, symbol, dunder):
        self.assertEqual(op(5 + 0j), op(5 + 0j))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_complex_imag_only(self, op, symbol, dunder):
        self.assertEqual(op(0 + 3j), op(0 + 3j))

    # --- bool (ConstantVariable, inherits slot from int) ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_bool_true(self, op, symbol, dunder):
        self.assertEqual(op(True), op(True))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_bool_false(self, op, symbol, dunder):
        self.assertEqual(op(False), op(False))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_bool_result_type(self, op, symbol, dunder):
        def fn(x):
            return op(True)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIsInstance(result, int)
        self.assertNotIsInstance(result, bool)

    # --- operator.X on constants ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_operator_int(self, op, symbol, dunder):
        self.assertEqual(op(10), op(10))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_operator_float(self, op, symbol, dunder):
        self.assertEqual(op(2.5), op(2.5))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_operator_bool(self, op, symbol, dunder):
        self.assertEqual(op(True), op(True))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_operator_complex(self, op, symbol, dunder):
        self.assertEqual(op(1 + 2j), op(1 + 2j))

    # --- dunder method on constants ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_dunder_int(self, op, symbol, dunder):
        self.assertEqual(getattr(42, dunder)(), op(42))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_dunder_float(self, op, symbol, dunder):
        self.assertEqual(getattr(3.14, dunder)(), op(3.14))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_dunder_bool(self, op, symbol, dunder):
        self.assertEqual(getattr(True, dunder)(), op(True))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_dunder_complex(self, op, symbol, dunder):
        self.assertEqual(getattr(1 + 2j, dunder)(), op(1 + 2j))

    # --- TypeError for types without the slot ---
    # CPython: "bad operand type for unary <symbol>: '<type>'"

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_str_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op("hello")
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_list_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op([1, 2, 3])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_dict_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op({"a": 1})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_set_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op({1, 2})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_tuple_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op(())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_none_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_bytes_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op(b"hello")
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_operator_none_raises(self, op, symbol, dunder):
        def fn(x):
            try:
                return op(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- Tensor (TensorVariable, proxy-based) ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_tensor(self, op, symbol, dunder):
        def fn(x):
            return op(x)

        x = torch.randn(3, 4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_tensor_integer_dtype(self, op, symbol, dunder):
        def fn(x):
            return op(x)

        x = torch.tensor([1, -2, 3])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_tensor_complex_dtype(self, op, symbol, dunder):
        def fn(x):
            return op(x)

        x = torch.tensor([1 + 2j, -3 + 4j])
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_tensor_creates_graph_node(self, op, symbol, dunder):
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            return op(x)

        fn(torch.randn(3))
        self.assertEqual(len(backend.graphs), 1)
        graph_str = backend.graphs[0].print_readable(False)
        self.assertIn(op.__name__, graph_str.lower())

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_tensor_operator(self, op, symbol, dunder):
        def fn(x):
            return op(x)

        x = torch.randn(5)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_tensor_dunder(self, op, symbol, dunder):
        def fn(x):
            return getattr(x, dunder)()

        x = torch.randn(4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- SymNodeVariable ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_symnode(self, op, symbol, dunder):
        def fn(x):
            s = x.size(0)
            return x[: op(s)]

        x = torch.randn(10, 5)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_symnode_operator(self, op, symbol, dunder):
        def fn(x):
            s = x.size(0)
            return x[: op(s)]

        x = torch.randn(8, 4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_symnode_dunder(self, op, symbol, dunder):
        def fn(x):
            s = x.size(0)
            return x[: getattr(s, dunder)()]

        x = torch.randn(6, 3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- UserDefinedObjectVariable ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_user_defined(self, op, symbol, dunder):
        class MyNum:
            def __init__(self, val):
                self.val = val

        setattr(MyNum, dunder, lambda self: MyNum(-self.val))
        obj = MyNum(10)

        def fn(x):
            return op(obj).val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -10)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_user_defined_operator(self, op, symbol, dunder):
        class MyNum:
            def __init__(self, val):
                self.val = val

        setattr(MyNum, dunder, lambda self: MyNum(-self.val))
        obj = MyNum(7)

        def fn(x):
            return op(obj).val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -7)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_user_defined_dunder(self, op, symbol, dunder):
        class MyNum:
            def __init__(self, val):
                self.val = val

        setattr(MyNum, dunder, lambda self: MyNum(-self.val))
        obj = MyNum(3)

        def fn(x):
            return getattr(obj, dunder)().val

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -3)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_user_defined_no_slot_raises(self, op, symbol, dunder):
        class NoSlot:
            pass

        obj = NoSlot()

        def fn(x):
            try:
                return op(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_user_defined_returns_arbitrary_type(self, op, symbol, dunder):
        class Weird:
            pass

        setattr(Weird, dunder, lambda self: "applied")
        obj = Weird()

        def fn(x):
            return op(obj)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "applied")

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_user_defined_raising_exception_propagates(self, op, symbol, dunder):
        class Raising:
            pass

        def raise_error(self):
            raise ValueError("custom error")

        setattr(Raising, dunder, raise_error)
        obj = Raising()

        def fn(x):
            try:
                return op(obj)
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_user_defined_subclass(self, op, symbol, dunder):
        class Base:
            def __init__(self, v):
                self.v = v

        setattr(Base, dunder, lambda self: type(self)(-self.v))

        class Sub(Base):
            pass

        obj = Sub(5)

        def fn(x):
            return op(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -5)

    # --- nn.Module ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_nn_module(self, op, symbol, dunder):
        class OpModule(torch.nn.Module):
            def forward(self, x):
                return x

        setattr(OpModule, dunder, lambda self: torch.tensor(-1.0))
        mod = OpModule()

        def fn(x):
            return op(mod) + x

        x = torch.tensor(5.0)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- Metaclass (UserDefinedClassVariable path) ---
    # In CPython, op(SomeClass) calls type(SomeClass)->nb_slot,
    # i.e. the metaclass's dunder, not the class's.

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_metaclass(self, op, symbol, dunder):
        class Meta(type):
            pass

        setattr(Meta, dunder, lambda cls: f"applied {cls.__name__}")

        class A(metaclass=Meta):
            pass

        def fn(x):
            return op(A)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "applied A")

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_metaclass_no_slot_raises(self, op, symbol, dunder):
        class Bar(type):
            pass

        class B(metaclass=Bar):
            pass

        def fn(x):
            try:
                return op(B)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_metaclass_inherited(self, op, symbol, dunder):
        class Meta(type):
            pass

        setattr(Meta, dunder, lambda cls: f"applied {cls.__name__}")

        class SubMeta(Meta):
            pass

        class D(metaclass=SubMeta):
            pass

        def fn(x):
            return op(D)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "applied D")

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_default_metaclass_raises(self, op, symbol, dunder):
        class Plain:
            pass

        def fn(x):
            try:
                return op(Plain)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- Unbound method calls: Type.dunder(instance) ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_unbound_user_defined(self, op, symbol, dunder):
        class MyNum:
            def __init__(self, v):
                self.v = v

        setattr(MyNum, dunder, lambda self: MyNum(-self.v))
        obj = MyNum(5)

        def fn(x):
            return getattr(MyNum, dunder)(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -5)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_unbound_parent_class(self, op, symbol, dunder):
        class Base:
            def __init__(self, v):
                self.v = v

        setattr(Base, dunder, lambda self: type(self)(-self.v))

        class Sub(Base):
            pass

        obj = Sub(10)

        def fn(x):
            return getattr(Base, dunder)(obj).v

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -10)

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_int(self, op, symbol, dunder):
        self.assertEqual(getattr(int, dunder)(42), op(42))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_float(self, op, symbol, dunder):
        self.assertEqual(getattr(float, dunder)(3.14), op(3.14))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_complex(self, op, symbol, dunder):
        self.assertEqual(getattr(complex, dunder)(1 + 2j), op(1 + 2j))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_unbound_builtin_bool(self, op, symbol, dunder):
        self.assertEqual(getattr(bool, dunder)(True), op(True))

    # --- Double application ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_double_apply_int(self, op, symbol, dunder):
        self.assertEqual(op(op(42)), op(op(42)))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    @make_dynamo_test
    def test_double_apply_float(self, op, symbol, dunder):
        self.assertEqual(op(op(3.14)), op(op(3.14)))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_double_apply_tensor(self, op, symbol, dunder):
        def fn(x):
            return op(op(x))

        x = torch.randn(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    # --- Used in expressions ---

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_in_arithmetic(self, op, symbol, dunder):
        def fn(x):
            return x + op(x)

        x = torch.randn(4)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))

    @parametrize("op,symbol,dunder", UNARY_OPS)
    def test_constant_in_tensor_op(self, op, symbol, dunder):
        def fn(x):
            val = op(5)
            return x + val

        x = torch.ones(3)
        result = torch.compile(fn, backend="eager", fullgraph=True)(x)
        self.assertEqual(result, fn(x))


instantiate_parametrized_tests(NbUnaryTests)


if __name__ == "__main__":
    run_tests()
