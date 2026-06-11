# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbRemainder(torch._dynamo.test_case.TestCase):
    # --- Integer remainder ---
    @make_dynamo_test
    def test_mod_integers(self):
        self.assertEqual(7 % 2, 1)
        self.assertEqual(8 % 4, 0)
        self.assertEqual(5 % 3, 2)

    @make_dynamo_test
    def test_mod_negative(self):
        # Python remainder follows the sign of the divisor.
        self.assertEqual(-7 % 3, 2)
        self.assertEqual(7 % -3, -2)

    @make_dynamo_test
    def test_mod_chained(self):
        self.assertEqual(100 % 7 % 3, 2)

    # --- Floats ---

    @make_dynamo_test
    def test_mod_floats(self):
        self.assertEqual(7.5 % 2.0, 1.5)
        self.assertEqual(7.0 % 2.0, 1.0)

    @make_dynamo_test
    def test_mod_mixed_int_float(self):
        self.assertEqual(7 % 2.0, 1.0)
        self.assertEqual(7.5 % 2, 1.5)

    # --- Booleans ---

    @make_dynamo_test
    def test_mod_bools(self):
        self.assertEqual(True % True, 0)
        self.assertEqual(False % True, 0)

    @make_dynamo_test
    def test_mod_int_and_bool(self):
        self.assertEqual(5 % True, 0)
        self.assertEqual(True % 2, 1)

    # --- String %-formatting (str/bytes also own nb_remainder) ---

    @make_dynamo_test
    def test_mod_str_formatting(self):
        self.assertEqual("%d" % 5, "5")  # noqa: UP031
        self.assertEqual("%s world" % "hello", "hello world")  # noqa: UP031
        self.assertEqual("%d-%d" % (1, 2), "1-2")  # noqa: UP031

    @make_dynamo_test
    def test_mod_bytes_formatting(self):
        self.assertEqual(b"%d" % 5, b"5")

    # --- Errors ---

    @make_dynamo_test
    def test_mod_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            1 % 0
        with self.assertRaises(ZeroDivisionError):
            1.0 % 0.0

    @make_dynamo_test
    def test_mod_int_and_str_raises(self):
        with self.assertRaises(TypeError):
            1 % "a"

    # --- Inplace %= ---

    @make_dynamo_test
    def test_inplace_mod_integers(self):
        x = 17
        x %= 5
        self.assertEqual(x, 2)

    @make_dynamo_test
    def test_inplace_mod_str(self):
        x = "%d"
        x %= 7
        self.assertEqual(x, "7")

    # --- User-defined __mod__ ---

    @make_dynamo_test
    def test_user_defined_mod_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __mod__(self, other):
                return type(self)(self.value % other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) % C(3), C(1))

    @make_dynamo_test
    def test_user_defined_mod_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __mod__(self, other):
                return type(self)(self.value % other)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(7) % 3, C(1))

    @make_dynamo_test
    def test_reversed_mod_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rmod__(self, other):
                return type(self)(other % self.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(7 % C(3), C(1))

    @make_dynamo_test
    def test_inplace_user_defined_mod(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __imod__(self, other):
                self.value %= other.value
                return self

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        x = C(17)
        x %= C(5)
        self.assertEqual(x, C(2))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __mod__(self, other):
                return "IntSub.__mod__"

            def __rmod__(self, other):
                return "IntSub.__rmod__"

        self.assertEqual(IntSub(7) % 3, "IntSub.__mod__")
        self.assertEqual(7 % IntSub(3), "IntSub.__rmod__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __mod__(self, other):
                return "Base.__mod__"

            def __rmod__(self, other):
                return "Base.__rmod__"

        self.assertEqual(Base() % 1, "Base.__mod__")
        self.assertEqual(1 % Base(), "Base.__rmod__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_mod_not_implemented_returns_type_error(self):
        class C:
            def __mod__(self, other):
                return NotImplemented

            def __rmod__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a % a

    @make_dynamo_test
    def test_mod_mixed_not_implemented_fallback(self):
        class A:
            def __mod__(self, other):
                return NotImplemented

        class B:
            def __rmod__(self, other):
                return "B.__rmod__ called"

        result = A() % B()
        self.assertEqual(result, "B.__rmod__ called")

    # --- SymNode remainder ---

    def test_mod_symnode_and_int(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(s % 3)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_mod_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(17 % s)

        x = torch.randn(5)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_mod_tensor(self):
        def fn(x, y):
            return x % y

        x = torch.randn(4).abs() + 1
        y = torch.randn(4).abs() + 1
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
