# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbLshift(torch._dynamo.test_case.TestCase):
    # --- Integer lshift ---
    @make_dynamo_test
    def test_lshift_integers(self):
        self.assertEqual(1 << 0, 1)
        self.assertEqual(1 << 1, 2)
        self.assertEqual(1 << 4, 16)
        self.assertEqual(3 << 2, 12)

    @make_dynamo_test
    def test_lshift_zero(self):
        self.assertEqual(0 << 5, 0)
        self.assertEqual(5 << 0, 5)

    @make_dynamo_test
    def test_lshift_large(self):
        self.assertEqual(1 << 31, 2**31)
        self.assertEqual(1 << 62, 2**62)

    @make_dynamo_test
    def test_lshift_negative_value(self):
        self.assertEqual(-1 << 1, -2)
        self.assertEqual(-3 << 2, -12)

    @make_dynamo_test
    def test_lshift_chained(self):
        self.assertEqual(1 << 2 << 3, 32)

    # --- Booleans ---

    @make_dynamo_test
    def test_lshift_bools(self):
        self.assertEqual(True << True, 2)
        self.assertEqual(True << False, 1)
        self.assertEqual(False << True, 0)

    @make_dynamo_test
    def test_lshift_int_and_bool(self):
        self.assertEqual(5 << True, 10)
        self.assertEqual(True << 3, 8)

    # --- Errors ---

    @make_dynamo_test
    def test_lshift_negative_count_raises(self):
        with self.assertRaises(ValueError):
            1 << -1

    @make_dynamo_test
    def test_lshift_float_raises(self):
        with self.assertRaises(TypeError):
            1 << 1.0
        with self.assertRaises(TypeError):
            1.0 << 1

    # --- Inplace <<= ---

    @make_dynamo_test
    def test_inplace_lshift_integers(self):
        x = 1
        x <<= 4
        self.assertEqual(x, 16)

    @make_dynamo_test
    def test_inplace_lshift_chained(self):
        x = 1
        x <<= 2
        x <<= 3
        self.assertEqual(x, 32)

    # --- User-defined __lshift__ ---

    @make_dynamo_test
    def test_user_defined_lshift_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __lshift__(self, other):
                return type(self)(self.value << other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(1) << C(4), C(16))

    @make_dynamo_test
    def test_user_defined_lshift_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __lshift__(self, other):
                return type(self)(self.value << other)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(1) << 4, C(16))

    @make_dynamo_test
    def test_reversed_lshift_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rlshift__(self, other):
                return type(self)(other << self.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(1 << C(2), C(4))

    @make_dynamo_test
    def test_user_defined_lshift_chained(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __lshift__(self, other):
                return type(self)(self.value << other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(1) << C(2) << C(1), C(8))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __lshift__(self, other):
                return "IntSub.__lshift__"

            def __rlshift__(self, other):
                return "IntSub.__rlshift__"

        self.assertEqual(IntSub(1) << 1, "IntSub.__lshift__")
        self.assertEqual(1 << IntSub(1), "IntSub.__rlshift__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __lshift__(self, other):
                return "Base.__lshift__"

            def __rlshift__(self, other):
                return "Base.__rlshift__"

        self.assertEqual(Base() << 1, "Base.__lshift__")
        self.assertEqual(1 << Base(), "Base.__rlshift__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_lshift_not_implemented_returns_type_error(self):
        class C:
            def __lshift__(self, other):
                return NotImplemented

            def __rlshift__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a << a

    @make_dynamo_test
    def test_lshift_mixed_not_implemented_fallback(self):
        class A:
            def __lshift__(self, other):
                return NotImplemented

        class B:
            def __rlshift__(self, other):
                return "B.__rlshift__ called"

        result = A() << B()
        self.assertEqual(result, "B.__rlshift__ called")

    # --- SymNode lshift ---

    def test_lshift_tensor_and_int(self):
        def fn(x):
            return x << 2

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_lshift_int_and_tensor(self):
        def fn(x):
            return 2 << x

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_lshift_tensor_and_tensor(self):
        def fn(x, y):
            return x << y

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        y = torch.tensor([1, 0, 2], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_lshift_symnode_and_int(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(s << 1)

        x = torch.randn(4)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_lshift_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(1 << s)

        x = torch.randn(3)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


@torch._dynamo.config.patch(enable_trace_unittest=True)
@torch._dynamo.config.patch(enable_trace_load_build_class=True)
class TestNbRshift(torch._dynamo.test_case.TestCase):
    # --- Integer rshift ---
    @make_dynamo_test
    def test_rshift_integers(self):
        self.assertEqual(16 >> 0, 16)
        self.assertEqual(16 >> 1, 8)
        self.assertEqual(16 >> 4, 1)
        self.assertEqual(12 >> 2, 3)

    @make_dynamo_test
    def test_rshift_zero(self):
        self.assertEqual(0 >> 5, 0)
        self.assertEqual(5 >> 0, 5)

    @make_dynamo_test
    def test_rshift_large(self):
        self.assertEqual((2**31) >> 1, 2**30)

    @make_dynamo_test
    def test_rshift_to_zero(self):
        self.assertEqual(1 >> 10, 0)

    @make_dynamo_test
    def test_rshift_negative_value(self):
        # arithmetic shift on negatives
        self.assertEqual(-1 >> 1, -1)
        self.assertEqual(-8 >> 1, -4)

    @make_dynamo_test
    def test_rshift_chained(self):
        self.assertEqual(32 >> 1 >> 2, 4)

    # --- Booleans ---

    @make_dynamo_test
    def test_rshift_bools(self):
        self.assertEqual(True >> True, 0)
        self.assertEqual(True >> False, 1)
        self.assertEqual(False >> True, 0)

    @make_dynamo_test
    def test_rshift_int_and_bool(self):
        self.assertEqual(10 >> True, 5)
        self.assertEqual(True >> 0, 1)

    # --- Errors ---

    @make_dynamo_test
    def test_rshift_negative_count_raises(self):
        with self.assertRaises(ValueError):
            1 >> -1

    @make_dynamo_test
    def test_rshift_float_raises(self):
        with self.assertRaises(TypeError):
            1 >> 1.0
        with self.assertRaises(TypeError):
            1.0 >> 1

    # --- Inplace >>= ---

    @make_dynamo_test
    def test_inplace_rshift_integers(self):
        x = 16
        x >>= 2
        self.assertEqual(x, 4)

    @make_dynamo_test
    def test_inplace_rshift_chained(self):
        x = 32
        x >>= 1
        x >>= 2
        self.assertEqual(x, 4)

    # --- User-defined __rshift__ ---

    @make_dynamo_test
    def test_user_defined_rshift_basic(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rshift__(self, other):
                return type(self)(self.value >> other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(16) >> C(2), C(4))

    @make_dynamo_test
    def test_user_defined_rshift_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rshift__(self, other):
                return type(self)(self.value >> other)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(16) >> 2, C(4))

    @make_dynamo_test
    def test_reversed_rshift_with_integer(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rrshift__(self, other):
                return type(self)(other >> self.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(16 >> C(1), C(8))

    @make_dynamo_test
    def test_user_defined_rshift_chained(self):
        class C:
            def __init__(self, v):
                self.value = v

            def __rshift__(self, other):
                return type(self)(self.value >> other.value)

            def __eq__(self, other):
                return type(other) is type(self) and self.value == other.value

        self.assertEqual(C(32) >> C(1) >> C(2), C(4))

    # --- Subclass dispatch ---

    @make_dynamo_test
    def test_subclass_of_int_gets_priority(self):
        class IntSub(int):
            def __rshift__(self, other):
                return "IntSub.__rshift__"

            def __rrshift__(self, other):
                return "IntSub.__rrshift__"

        self.assertEqual(IntSub(1) >> 1, "IntSub.__rshift__")
        self.assertEqual(1 >> IntSub(1), "IntSub.__rrshift__")

    @make_dynamo_test
    def test_subclass_of_object_baseline(self):
        class Base:
            def __rshift__(self, other):
                return "Base.__rshift__"

            def __rrshift__(self, other):
                return "Base.__rrshift__"

        self.assertEqual(Base() >> 1, "Base.__rshift__")
        self.assertEqual(1 >> Base(), "Base.__rrshift__")

    # --- NotImplemented handling ---

    @make_dynamo_test
    def test_rshift_not_implemented_returns_type_error(self):
        class C:
            def __rshift__(self, other):
                return NotImplemented

            def __rrshift__(self, other):
                return NotImplemented

        a = C()
        with self.assertRaises(TypeError):
            a >> a

    @make_dynamo_test
    def test_rshift_mixed_not_implemented_fallback(self):
        class A:
            def __rshift__(self, other):
                return NotImplemented

        class B:
            def __rrshift__(self, other):
                return "B.__rrshift__ called"

        result = A() >> B()
        self.assertEqual(result, "B.__rrshift__ called")

    # --- SymNode rshift ---

    def test_rshift_tensor_and_int(self):
        def fn(x):
            return x >> 1

        x = torch.tensor([16, 8, 4], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rshift_int_and_tensor(self):
        def fn(x):
            return 32 >> x

        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rshift_tensor_and_tensor(self):
        def fn(x, y):
            return x >> y

        x = torch.tensor([16, 8, 4], dtype=torch.int64)
        y = torch.tensor([1, 2, 0], dtype=torch.int64)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_rshift_symnode_and_int(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(s >> 1)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rshift_int_and_symnode(self):
        def fn(x):
            s = x.shape[0]
            return x.new_zeros(32 >> s)

        x = torch.randn(3)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
