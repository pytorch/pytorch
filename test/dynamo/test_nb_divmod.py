# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    HardwareClassification,
    make_dynamo_test,
)


class UserDefinedTensor(torch.Tensor):
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data)

    intercepted = False

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if func in (torch.Tensor.__divmod__, torch.divmod):
            cls.intercepted = True

        return super().__torch_function__(func, types, args, kwargs)


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbDivmod(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

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
    def test_divmod_tensor(self):
        def fn(x, y):
            return divmod(x, y)

        x = torch.randn(4)
        y = torch.randn(4)
        ret = divmod(x, y)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(ret, opt_fn(x, y))

    def test_divmod_tensor_scalar(self):
        def fn(x):
            return divmod(x, 2), divmod(2, x)

        x = torch.randn(4)
        ret = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(ret, opt_fn(x))

    def test_divmod_scalar_tensor_inductor(self):
        """divmod(Scalar, Tensor), examine aten.divmod.Scalar_Tensor decomp path."""

        def fn(x):
            return divmod(8, x)

        x = 2
        qt, rt = torch.compile(
            fn,
            backend="inductor",
            fullgraph=True,
        )(torch.as_tensor(x))
        q, r = fn(x)
        self.assertEqual(torch.as_tensor(q), qt)
        self.assertEqual(torch.as_tensor(r), rt)

    def test_divmod_compile_subclass_preserve_type(self):
        x, y = (
            torch.tensor([10.0, 20.0]).as_subclass(UserDefinedTensor),
            torch.tensor([3.0, 4.0]).as_subclass(UserDefinedTensor),
        )

        def fn(_x, _y):
            return divmod(_x, _y)

        def _assert(q, r):
            self.assertIsInstance(q, UserDefinedTensor)
            self.assertIsInstance(r, UserDefinedTensor)
            self.assertTrue(UserDefinedTensor.intercepted)

        eager_q, eager_r = fn(x, y)
        _assert(eager_q, eager_r)

        torch._dynamo.reset()
        UserDefinedTensor.intercepted = False

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled_q, compiled_r = compiled_fn(x, y)

        _assert(compiled_q, compiled_r)
        self.assertEqual(eager_q, compiled_q)
        self.assertEqual(eager_r, compiled_r)
        UserDefinedTensor.intercepted = False

    def test_divmod_torch_function_mode(self):
        class Mode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.call_count = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func is torch.divmod:
                    self.call_count += 1
                return func(*args)

        mode = Mode()
        x, y = torch.tensor([10.0, 20.0]), torch.tensor([3.0, 4.0])

        def fn(_x, _y):
            return divmod(_x, _y)

        torch._dynamo.reset()
        with mode:
            compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
            compiled_q, compiled_r = compiled_fn(x, y)

        self.assertEqual(mode.call_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
