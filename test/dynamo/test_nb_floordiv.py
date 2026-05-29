# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbFloorDivide(torch._dynamo.test_case.TestCase):
    # --- Integer floordiv ---
    @make_dynamo_test
    def test_floordiv_integers(self):
        self.assertEqual(7 // 2, 3)
        self.assertEqual(8 // 2, 4)
        self.assertEqual(1 // 1, 1)
        self.assertEqual(0 // 5, 0)

    @make_dynamo_test
    def test_floordiv_negative(self):
        self.assertEqual(-7 // 2, -4)
        self.assertEqual(7 // -2, -4)
        self.assertEqual(-7 // -2, 3)

    @make_dynamo_test
    def test_floordiv_chained(self):
        self.assertEqual(100 // 5 // 2, 10)

    # --- Floats ---

    @make_dynamo_test
    def test_floordiv_floats(self):
        self.assertEqual(7.0 // 2.0, 3.0)
        self.assertEqual(7.5 // 2.0, 3.0)

    @make_dynamo_test
    def test_floordiv_mixed_int_float(self):
        self.assertEqual(7 // 2.0, 3.0)
        self.assertEqual(7.0 // 2, 3.0)

    # --- Booleans ---

    @make_dynamo_test
    def test_floordiv_bools(self):
        self.assertEqual(True // True, 1)
        self.assertEqual(False // True, 0)

    @make_dynamo_test
    def test_floordiv_int_and_bool(self):
        self.assertEqual(5 // True, 5)
        self.assertEqual(True // 1, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_floordiv_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            1 // 0
        with self.assertRaises(ZeroDivisionError):
            1.0 // 0.0

    @make_dynamo_test
    def test_floordiv_str_raises(self):
        with self.assertRaises(TypeError):
            1 // "a"
        with self.assertRaises(TypeError):
            "a" // 1

    # --- Inplace //= ---

    @make_dynamo_test
    def test_inplace_floordiv_integers(self):
        x = 17
        x //= 4
        self.assertEqual(x, 4)

    @make_dynamo_test
    def test_inplace_floordiv_chained(self):
        x = 100
        x //= 5
        x //= 2
        self.assertEqual(x, 10)

    # --- User-defined __floordiv__ ---

    @make_dynamo_test
    def test_user_defined_floordiv_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __floordiv__(self, other):
                return type(self)(self.value // other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) // C(2), C(3))

    @make_dynamo_test
    def test_user_defined_floordiv_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __floordiv__(self, other):
                return type(self)(self.value // other)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) // 2, C(3))

    @make_dynamo_test
    def test_reversed_floordiv_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rfloordiv__(self, other):
                return type(self)(other // self.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(7 // C(2), C(3))

    @make_dynamo_test
    def test_inplace_user_defined_floordiv(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __ifloordiv__(self, other):
                self.value //= other.value
                return self

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        x = C(17)
        x //= C(4)
        self.assertEqual(x, C(4))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __floordiv__(self, other):
                return "IntSub.__floordiv__"

            def __rfloordiv__(self, other):
                return "IntSub.__rfloordiv__"

        self.assertEqual(IntSub(7) // 2, "IntSub.__floordiv__")
        self.assertEqual(7 // IntSub(2), "IntSub.__rfloordiv__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __floordiv__(self, other):
                return "Base.__floordiv__"

            def __rfloordiv__(self, other):
                return "Base.__rfloordiv__"

        self.assertEqual(Base() // 1, "Base.__floordiv__")
        self.assertEqual(1 // Base(), "Base.__rfloordiv__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_floordiv_not_implemented_returns_type_error(self):
        class C:
            def __floordiv__(self, other):
                return NotImplemented

            def __rfloordiv__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a // a

    @make_dynamo_test
    def test_floordiv_mixed_not_implemented_fallback(self):
        class A:
            def __floordiv__(self, other):
                return NotImplemented

        class B:
            def __rfloordiv__(self, other):
                return "B.__rfloordiv__ called"

        result = A() // B()
        self.assertEqual(result, "B.__rfloordiv__ called")

    # --- SymNode floordiv ---

    def test_floordiv_symnode_and_int(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(s // 2)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_floordiv_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(16 // s)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_floordiv_tensor(self):
        def fn(x, y):
            return x // y

        x = torch.randn(4)
        y = torch.randn(4).abs() + 1
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
