# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbPower(torch._dynamo.test_case.TestCase):
    # --- Integer power ---

    @make_dynamo_test
    def test_pow_integers(self):
        self.assertEqual(2**10, 1024)
        self.assertEqual(3**3, 27)
        self.assertEqual(1**100, 1)
        self.assertEqual(0**0, 1)

    @make_dynamo_test
    def test_pow_negative_exponent(self):
        self.assertEqual(2**-1, 0.5)
        self.assertEqual(4**-1, 0.25)

    @make_dynamo_test
    def test_pow_chained(self):
        self.assertEqual(2**2**3, 256)

    # --- Floats ---

    @make_dynamo_test
    def test_pow_floats(self):
        self.assertEqual(2.0**3.0, 8.0)
        self.assertEqual(9.0**0.5, 3.0)

    @make_dynamo_test
    def test_pow_mixed_int_float(self):
        self.assertEqual(2**3.0, 8.0)
        self.assertEqual(2.0**3, 8.0)

    # --- Complex ---

    @make_dynamo_test
    def test_pow_complex(self):
        self.assertEqual((1 + 1j) ** 2, 2j)
        self.assertEqual((2 + 0j) ** 3, (8 + 0j))

    # --- Booleans ---

    @make_dynamo_test
    def test_pow_bools(self):
        self.assertEqual(True**True, 1)
        self.assertEqual(False**True, 0)
        self.assertEqual(True**False, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_pow_zero_base_negative_exp_raises(self):
        with self.assertRaises(ZeroDivisionError):
            0**-1

    @make_dynamo_test
    def test_pow_str_raises(self):
        with self.assertRaises(TypeError):
            2 ** "a"
        with self.assertRaises(TypeError):
            "a" ** 2

    # --- Inplace **= ---

    @make_dynamo_test
    def test_inplace_pow_integers(self):
        x = 2
        x **= 8
        self.assertEqual(x, 256)

    @make_dynamo_test
    def test_inplace_pow_chained(self):
        x = 2
        x **= 3
        x **= 2
        self.assertEqual(x, 64)

    # --- User-defined __pow__ ---

    @make_dynamo_test
    def test_user_defined_pow_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __pow__(self, other):
                return type(self)(self.value**other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(2) ** C(10), C(1024))

    @make_dynamo_test
    def test_user_defined_pow_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __pow__(self, other):
                return type(self)(self.value**other)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(2) ** 10, C(1024))

    @make_dynamo_test
    def test_reversed_pow_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rpow__(self, other):
                return type(self)(other**self.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(2 ** C(10), C(1024))

    @make_dynamo_test
    def test_inplace_user_defined_pow(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __ipow__(self, other):
                self.value **= other.value
                return self

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        x = C(2)
        x **= C(10)
        self.assertEqual(x, C(1024))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __pow__(self, other, modulo=None):
                return "IntSub.__pow__"

            def __rpow__(self, other, modulo=None):
                return "IntSub.__rpow__"

        self.assertEqual(IntSub(2) ** 10, "IntSub.__pow__")
        self.assertEqual(2 ** IntSub(10), "IntSub.__rpow__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __pow__(self, other):
                return "Base.__pow__"

            def __rpow__(self, other):
                return "Base.__rpow__"

        self.assertEqual(Base() ** 1, "Base.__pow__")
        self.assertEqual(1 ** Base(), "Base.__rpow__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_pow_not_implemented_returns_type_error(self):
        class C:
            def __pow__(self, other):
                return NotImplemented

            def __rpow__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a**a

    @make_dynamo_test
    def test_pow_mixed_not_implemented_fallback(self):
        class A:
            def __pow__(self, other):
                return NotImplemented

        class B:
            def __rpow__(self, other):
                return "B.__rpow__ called"

        result = A() ** B()
        self.assertEqual(result, "B.__rpow__ called")

    # --- 3-arg pow ---

    @make_dynamo_test
    def test_three_arg_pow_constants(self):
        self.assertEqual(pow(2, 10, 1000), 24)
        self.assertEqual(pow(3, 4, 7), 4)

    # --- SymNode power ---

    def test_pow_symnode_and_int(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(s**2)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_pow_symnode_negative_exponent(self):
        # int ** negative_int produces float; shape expressions hit this
        def fn(x):
            s = x.shape[0]
            return s**-1

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(opt_fn(x), fn(x))

    def test_pow_tensor(self):
        def fn(x, y):
            return x**y

        x = torch.randn(4).abs() + 1
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_pow_inplace_tensor(self):
        def fn(x, y):
            x **= y
            return x

        x = torch.randn(4).abs() + 1
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x_clone = x.clone()
        self.assertEqual(opt_fn(x, y), fn(x_clone, y))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
