# Owner(s): ["module: dynamo"]

"""Tests for nb_and (&) operator in PyTorch Dynamo."""

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
)
from torch.utils._ordered_set import OrderedSet


class TestNbAnd(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

    # --- Integer and ---

    @make_dynamo_test
    def test_and_integers(self):
        self.assertEqual(0b1100 & 0b1010, 0b1000)
        self.assertEqual(0xFF & 0x0F, 0x0F)
        self.assertEqual(7 & 3, 3)

    @make_dynamo_test
    def test_and_zero(self):
        self.assertEqual(0 & 0xFFFF, 0)
        self.assertEqual(0xFFFF & 0, 0)

    @make_dynamo_test
    def test_and_negative_integers(self):
        self.assertEqual(-1 & 0xFF, 0xFF)
        self.assertEqual(-2 & 3, 2)

    @make_dynamo_test
    def test_and_chained(self):
        self.assertEqual(0b1110 & 0b1100 & 0b1000, 0b1000)

    # --- Booleans ---

    @make_dynamo_test
    def test_and_bools(self):
        self.assertEqual(True & True, True)
        self.assertEqual(True & False, False)
        self.assertEqual(False & True, False)
        self.assertEqual(False & False, False)

    @make_dynamo_test
    def test_and_int_and_bool(self):
        self.assertEqual(5 & True, 1)
        self.assertEqual(True & 3, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_and_float_raises(self):
        with self.assertRaises(TypeError):
            1 & 1.0
        with self.assertRaises(TypeError):
            1.0 & 1

    # --- Inplace &= ---

    @make_dynamo_test
    def test_inplace_and_integers(self):
        x = 0b1110
        x &= 0b1100
        self.assertEqual(x, 0b1100)

    @make_dynamo_test
    def test_inplace_and_chained(self):
        x = 0b1111
        x &= 0b1110
        x &= 0b1100
        self.assertEqual(x, 0b1100)

    # --- Sets ---

    @make_dynamo_test
    def test_set_and_set(self):
        self.assertEqual({1, 2, 3} & {2, 3, 4}, {2, 3})

    @make_dynamo_test
    def test_set_iand_set(self):
        s = {1, 2, 3}
        s &= {2, 3, 4}
        self.assertEqual(s, {2, 3})

    @make_dynamo_test
    def test_frozenset_and(self):
        self.assertEqual(frozenset({1, 2, 3}) & frozenset({2, 3, 4}), frozenset({2, 3}))

    # --- Dict views ---

    @make_dynamo_test
    def test_dict_keys_and_set(self):
        self.assertEqual({1: 0, 2: 0, 3: 0}.keys() & {2, 3, 4}, {2, 3})

    @make_dynamo_test
    def test_set_and_dict_keys(self):
        self.assertEqual({2, 3, 4} & {1: 0, 2: 0, 3: 0}.keys(), {2, 3})

    @make_dynamo_test
    def test_dict_items_and_set(self):
        d = {1: "a", 2: "b", 3: "c"}
        self.assertEqual(d.items() & {(2, "b"), (3, "x")}, {(2, "b")})

    # --- OrderedSet ---

    @make_dynamo_test
    def test_ordered_set_and(self):
        self.assertEqual(
            OrderedSet([1, 2, 3]) & OrderedSet([2, 3, 4]), OrderedSet([2, 3])
        )

    # --- User-defined __and__ ---

    @make_dynamo_test
    def test_user_defined_and_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __and__(self, other):
                return type(self)(self.value & other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) & C(0b1100), C(0b1100))

    @make_dynamo_test
    def test_user_defined_and_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __and__(self, other):
                return type(self)(self.value & other)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) & 0b1100, C(0b1100))

    @make_dynamo_test
    def test_reversed_and_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rand__(self, other):
                return type(self)(other & self.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(0b1110 & C(0b1100), C(0b1100))

    @make_dynamo_test
    def test_user_defined_and_chained(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __and__(self, other):
                return type(self)(self.value & other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(0b1110) & C(0b1100) & C(0b1000), C(0b1000))

    @make_dynamo_test
    def test_user_defined_iand(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __iand__(self, other):
                self.value &= other.value
                return self

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        a = C(0b1110)
        a &= C(0b1100)
        self.assertEqual(a, C(0b1100))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __and__(self, other):
                return "IntSub.__and__"

            def __rand__(self, other):
                return "IntSub.__rand__"

        self.assertEqual(IntSub(1) & 1, "IntSub.__and__")
        self.assertEqual(1 & IntSub(1), "IntSub.__rand__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __and__(self, other):
                return "Base.__and__"

            def __rand__(self, other):
                return "Base.__rand__"

        self.assertEqual(Base() & 1, "Base.__and__")
        self.assertEqual(1 & Base(), "Base.__rand__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_and_not_implemented_returns_type_error(self):
        class C:
            def __and__(self, other):
                return NotImplemented

            def __rand__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a & a

    @make_dynamo_test
    def test_and_mixed_not_implemented_fallback(self):
        class A:
            def __and__(self, other):
                return NotImplemented

        class B:
            def __rand__(self, other):
                return "B.__rand__ called"

        result = A() & B()
        self.assertEqual(result, "B.__rand__ called")

    # --- Tensor and ---

    def test_and_tensor_and_tensor(self):
        def fn(x, y):
            return x & y

        x = torch.tensor([True, False, True, False])
        y = torch.tensor([True, True, False, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_inplace_and_tensor(self):
        def fn(x, y):
            x &= y
            return x

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        y = torch.tensor([0b1100, 0b1010], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x.clone(), y), fn(x.clone(), y))

    def test_and_tensor_and_tensor_int(self):
        def fn(x, y):
            return x & y

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        y = torch.tensor([0b1100, 0b1010], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_and_tensor_and_int(self):
        def fn(x):
            return x & 7

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_int_and_tensor(self):
        def fn(x):
            return 7 & x

        x = torch.tensor([0b1110, 0b1100], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_tensor_and_bool(self):
        def fn(x):
            return x & True

        x = torch.tensor([True, False, True, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_bool_and_tensor(self):
        def fn(x):
            return True & x

        x = torch.tensor([True, False, True, False])
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    # --- SymNode and ---

    def test_and_symnode_and_int(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(s & 7)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_and_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(7 & s)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


instantiate_parametrized_tests(TestNbAnd)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
