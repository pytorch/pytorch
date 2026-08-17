# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbTrueDivide(torch._dynamo.test_case.TestCase):
    # --- Integer truediv (always promotes to float) ---
    @make_dynamo_test
    def test_truediv_integers(self):
        self.assertEqual(7 / 2, 3.5)
        self.assertEqual(8 / 2, 4.0)
        self.assertEqual(0 / 5, 0.0)

    @make_dynamo_test
    def test_truediv_negative(self):
        self.assertEqual(-7 / 2, -3.5)
        self.assertEqual(7 / -2, -3.5)

    @make_dynamo_test
    def test_truediv_chained(self):
        self.assertEqual(100 / 5 / 2, 10.0)

    # --- Floats ---

    @make_dynamo_test
    def test_truediv_floats(self):
        self.assertEqual(7.0 / 2.0, 3.5)
        self.assertEqual(1.0 / 4.0, 0.25)

    @make_dynamo_test
    def test_truediv_mixed_int_float(self):
        self.assertEqual(7 / 2.0, 3.5)
        self.assertEqual(7.0 / 2, 3.5)

    # --- Complex ---

    @make_dynamo_test
    def test_truediv_complex(self):
        self.assertEqual((4 + 2j) / 2, (2 + 1j))

    # --- Booleans ---

    @make_dynamo_test
    def test_truediv_bools(self):
        self.assertEqual(True / True, 1.0)
        self.assertEqual(False / True, 0.0)

    @make_dynamo_test
    def test_truediv_int_and_bool(self):
        self.assertEqual(5 / True, 5.0)
        self.assertEqual(True / 2, 0.5)

    # --- Errors ---

    @make_dynamo_test
    def test_truediv_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            1 / 0
        with self.assertRaises(ZeroDivisionError):
            1.0 / 0.0

    @make_dynamo_test
    def test_truediv_str_raises(self):
        with self.assertRaises(TypeError):
            1 / "a"
        with self.assertRaises(TypeError):
            "a" / 1

    # --- Inplace /= ---

    @make_dynamo_test
    def test_inplace_truediv_integers(self):
        x = 10
        x /= 4
        self.assertEqual(x, 2.5)

    @make_dynamo_test
    def test_inplace_truediv_chained(self):
        x = 100
        x /= 5
        x /= 2
        self.assertEqual(x, 10.0)

    # --- User-defined __truediv__ ---

    @make_dynamo_test
    def test_user_defined_truediv_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __truediv__(self, other):
                return type(self)(self.value / other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) / C(2), C(3.5))

    @make_dynamo_test
    def test_user_defined_truediv_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __truediv__(self, other):
                return type(self)(self.value / other)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) / 2, C(3.5))

    @make_dynamo_test
    def test_reversed_truediv_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rtruediv__(self, other):
                return type(self)(other / self.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(7 / C(2), C(3.5))

    @make_dynamo_test
    def test_inplace_user_defined_truediv(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __itruediv__(self, other):
                self.value /= other.value
                return self

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        x = C(10)
        x /= C(4)
        self.assertEqual(x, C(2.5))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __truediv__(self, other):
                return "IntSub.__truediv__"

            def __rtruediv__(self, other):
                return "IntSub.__rtruediv__"

        self.assertEqual(IntSub(7) / 2, "IntSub.__truediv__")
        self.assertEqual(7 / IntSub(2), "IntSub.__rtruediv__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __truediv__(self, other):
                return "Base.__truediv__"

            def __rtruediv__(self, other):
                return "Base.__rtruediv__"

        self.assertEqual(Base() / 1, "Base.__truediv__")
        self.assertEqual(1 / Base(), "Base.__rtruediv__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_truediv_not_implemented_returns_type_error(self):
        class C:
            def __truediv__(self, other):
                return NotImplemented

            def __rtruediv__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a / a

    @make_dynamo_test
    def test_truediv_mixed_not_implemented_fallback(self):
        class A:
            def __truediv__(self, other):
                return NotImplemented

        class B:
            def __rtruediv__(self, other):
                return "B.__rtruediv__ called"

        result = A() / B()
        self.assertEqual(result, "B.__rtruediv__ called")

    # --- SymNode truediv ---

    def test_truediv_symnode_and_int(self):
        def fn(x):
            s = x.shape[0]
            return x + (s / 2)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_truediv_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            return x + (16 / s)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_truediv_tensor(self):
        def fn(x, y):
            return x / y

        x = torch.randn(4)
        y = torch.randn(4).abs() + 1
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
