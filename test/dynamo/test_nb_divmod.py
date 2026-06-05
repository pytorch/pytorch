# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbDivmod(torch._dynamo.test_case.TestCase):
    # --- Integer divmod ---
    @make_dynamo_test
    def test_divmod_integers(self):
        self.assertEqual(divmod(7, 2), (3, 1))
        self.assertEqual(divmod(8, 4), (2, 0))
        self.assertEqual(divmod(5, 3), (1, 2))

    @make_dynamo_test
    def test_divmod_negative(self):
        self.assertEqual(divmod(-7, 2), (-4, 1))
        self.assertEqual(divmod(7, -2), (-4, -1))

    # --- Floats ---

    @make_dynamo_test
    def test_divmod_floats(self):
        self.assertEqual(divmod(7.5, 2.0), (3.0, 1.5))
        self.assertEqual(divmod(7.0, 2.0), (3.0, 1.0))

    @make_dynamo_test
    def test_divmod_mixed_int_float(self):
        self.assertEqual(divmod(7, 2.0), (3.0, 1.0))
        self.assertEqual(divmod(7.5, 2), (3.0, 1.5))

    # --- Booleans ---

    @make_dynamo_test
    def test_divmod_bools(self):
        self.assertEqual(divmod(True, True), (1, 0))
        self.assertEqual(divmod(5, True), (5, 0))

    # --- Errors ---

    @make_dynamo_test
    def test_divmod_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            divmod(1, 0)
        with self.assertRaises(ZeroDivisionError):
            divmod(1.0, 0.0)

    @make_dynamo_test
    def test_divmod_str_raises(self):
        with self.assertRaises(TypeError):
            divmod(1, "a")
        with self.assertRaises(TypeError):
            divmod("a", 1)

    # --- User-defined __divmod__ ---

    @make_dynamo_test
    def test_user_defined_divmod_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __divmod__(self, other):
                return divmod(self.value, other.value)

        self.assertEqual(divmod(C(7), C(2)), (3, 1))

    @make_dynamo_test
    def test_user_defined_divmod_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __divmod__(self, other):
                return divmod(self.value, other)

        self.assertEqual(divmod(C(7), 2), (3, 1))

    @make_dynamo_test
    def test_reversed_divmod_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rdivmod__(self, other):
                return divmod(other, self.value)

        self.assertEqual(divmod(7, C(2)), (3, 1))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __divmod__(self, other):
                return "IntSub.__divmod__"

            def __rdivmod__(self, other):
                return "IntSub.__rdivmod__"

        self.assertEqual(divmod(IntSub(7), 2), "IntSub.__divmod__")
        self.assertEqual(divmod(7, IntSub(2)), "IntSub.__rdivmod__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __divmod__(self, other):
                return "Base.__divmod__"

            def __rdivmod__(self, other):
                return "Base.__rdivmod__"

        self.assertEqual(divmod(Base(), 1), "Base.__divmod__")
        self.assertEqual(divmod(1, Base()), "Base.__rdivmod__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_divmod_not_implemented_returns_type_error(self):
        class C:
            def __divmod__(self, other):
                return NotImplemented

            def __rdivmod__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            divmod(a, a)

    @make_dynamo_test
    def test_divmod_mixed_not_implemented_fallback(self):
        class A:
            def __divmod__(self, other):
                return NotImplemented

        class B:
            def __rdivmod__(self, other):
                return "B.__rdivmod__ called"

        result = divmod(A(), B())
        self.assertEqual(result, "B.__rdivmod__ called")

    # --- Tensor ---

    def test_divmod_tensor_raises_type_error(self):
        def fn(x, y):
            return divmod(x, y)

        x = torch.randn(4)
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaises(TypeError):
            opt_fn(x, y)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
