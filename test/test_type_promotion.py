from functools import wraps
import itertools
import unittest

import torch

from torch.testing._internal.common_utils import (TestCase, run_tests, load_tests,
                                                  TEST_NUMPY, torch_to_numpy_dtype_dict)
from torch.testing._internal.common_device_type import (instantiate_device_type_tests, onlyOnCPUAndCUDA,
                                                        dtypes, dtypesIfCUDA, onlyCPU)

if TEST_NUMPY:
    import numpy as np

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

# Not thread-safe decorator that runs the decorated test once with
# the default dtype being torch.float and again with the default dtype
# being torch.double.
def float_double_default_dtype(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        cur_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float)
            fn(*args, **kwargs)
            torch.set_default_dtype(torch.double)
            fn(*args, **kwargs)
        finally:
            torch.set_default_dtype(cur_dtype)

    return wrapped_fn

class TestTypePromotion(TestCase):

    # In-place operations don't promote.
    # `int+float -> float` but `int.add_(float)` is rejected as an error.
    # Promoting inplace would require re-allocating and copying the memory of the
    # tensor data, since element size could change.
    @float_double_default_dtype
    def test_inplace(self, device):
        int_tensor = torch.ones([4, 4, 4], dtype=torch.int32, device=device)

        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: int_tensor.add_(1.5))

        expected = torch.ones([4, 4, 4], dtype=torch.int32, device=device)

        long_tensor = torch.ones([4, 4, 4], dtype=torch.int64, device=device)
        int_tensor.add_(long_tensor)
        int_tensor.add_(1)
        three = expected + 2
        self.assertEqual(int_tensor, three)
        self.assertEqual(int_tensor.dtype, torch.int32)

        bool_tensor = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
        uint8_tensor = torch.tensor([1, 1, 1], dtype=torch.uint8, device=device)
        # We treat bool as a separate category, which means uint8 cannot cast to bool.
        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: bool_tensor.add_(uint8_tensor))

        # We allow demotion from signed to unsigned, unlike numpy, because:
        # * We don't want the performance penalty of inspecting scalar values.
        # * We don't want 'signed' to be considered a distinct 'category'
        # in promotion rules.
        # We don't want signed to be a separate category because if it was,
        # uint16_tensor + 5 would result in a long_tensor, which is not what we want.
        int16_tensor = torch.tensor([1, 1, 1], dtype=torch.int16, device=device)
        uint8_tensor *= int16_tensor

    @float_double_default_dtype
    def test_unsigned(self, device):
        dont_promote = torch.ones(3, dtype=torch.uint8, device=device) + 5
        self.assertEqual(dont_promote.dtype, torch.uint8)

    # some basic examples

    @float_double_default_dtype
    def test_int_promotion(self, device):
        a = torch.ones([4, 4, 4], dtype=torch.int32, device=device)
        b = torch.ones([4, 4, 4], dtype=torch.int64, device=device)
        c = a + b
        self.assertEqual(c, b + b)
        self.assertEqual(c.dtype, torch.int64)

    @float_double_default_dtype
    def test_float_promotion(self, device):
        def test_promotion(dtype_float, dtype_double):
            a = torch.ones([4, 4, 4], dtype=dtype_float, device=device)
            b = torch.ones([4, 4, 4], dtype=dtype_double, device=device)
            c = a + b
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)
            c = b + a
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)
        test_promotion(torch.float, torch.double)

    @float_double_default_dtype
    def test_complex_promotion(self, device):
        def test_promotion(dtype_float, dtype_double):
            a = torch.ones([4, 4, 4], dtype=dtype_float, device=device)
            b = torch.ones([4, 4, 4], dtype=dtype_double, device=device)
            c = a + b
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)
            c = b + a
            self.assertEqual(c, b + b)
            self.assertEqual(c.dtype, dtype_double)

        test_promotion(torch.complex64, torch.complex128)

        a = torch.randn(3, dtype=torch.complex64, device=device)
        self.assertEqual((a * 5).dtype, torch.complex64)
        # not a "wrapped number"
        other = torch.tensor(5.5, dtype=torch.double, device=device)
        self.assertEqual((a + other).dtype, torch.complex64)

    @float_double_default_dtype
    def test_complex_scalar_mult_tensor_promotion(self, device):
        a = 1j * torch.ones(2, device=device)
        a = a + 1j
        b = torch.tensor([2j, 2j], device=device)
        self.assertEqual(a, b)
        self.assertEqual(a.dtype, b.dtype)

    @float_double_default_dtype
    def test_add_wrapped(self, device):
        a = torch.ones([4, 4, 4], dtype=torch.int, device=device)
        b = 1
        c = a + b
        self.assertEqual(c, a + a)
        self.assertEqual(c.dtype, torch.int)

    @float_double_default_dtype
    def test_int_to_float(self, device):
        a = torch.ones([4, 4, 4], dtype=torch.int32, device=device)
        b = torch.ones([4, 4, 4], dtype=torch.float, device=device)
        c = a + b
        self.assertEqual(c.dtype, torch.float32)

    # some examples from:
    # https://github.com/pytorch/pytorch/issues/9515

    @float_double_default_dtype
    def test_from_issue(self, device):
        a = torch.rand(3, dtype=torch.float32, device=device)
        u = torch.tensor([0, 0, 1], dtype=torch.uint8, device=device)
        self.assertEqual((a * 5).dtype, torch.float32)
        self.assertEqual((u + 1).dtype, torch.uint8)
        self.assertEqual((u + 1000).dtype, torch.uint8)  # integer overflow

        # not a "wrapped number"
        other = torch.tensor(5.5, dtype=torch.double, device=device)

        self.assertEqual((u + 5.5).dtype, torch.get_default_dtype())
        self.assertEqual((u + other).dtype, torch.double)
        # adding a 0-dim tensor to a float doesn't promote to double unless first
        # type was integral.
        self.assertEqual((a + other).dtype, torch.float32)

    @float_double_default_dtype
    def test_half(self, device):
        half = torch.tensor(5.5, dtype=torch.float16, device=device)
        self.assertEqual((half + 2.2).dtype, torch.float16)
        self.assertEqual((half + 100000).dtype, torch.float16)  # inf
        default_tensor = torch.tensor(100000.0, device=device)
        self.assertEqual((half + default_tensor).dtype, torch.get_default_dtype())

    def test_bfloat16(self, device):
        # with scalar
        bf = torch.tensor(5.5, dtype=torch.bfloat16, device=device)
        for scalar in (2.2, 5, 100000):   # bf + 100000 is inf
            self.assertEqual((bf + scalar).dtype, torch.bfloat16)
            self.assertEqual(scalar + bf, bf + scalar)

        for scalar in (complex(1, 1), complex(-2, 0), complex(0, -3)):
            self.assertEqual((bf + scalar).dtype, torch.cfloat)
            self.assertEqual(bf + scalar, scalar + bf)

        # with tensor
        for dtype in torch.testing.get_all_dtypes():
            t = torch.tensor(1, dtype=dtype, device=device)
            self.assertEqual(bf + t, t + bf)
            if dtype in (torch.float16, torch.float32, torch.float64, torch.cfloat, torch.cdouble):
                # Handles bfloat16 x float16 -> float32 promotion
                expected_dtype = dtype if dtype != torch.half else torch.float32
            elif dtype in (torch.bool, torch.uint8,
                           torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16):
                expected_dtype = torch.bfloat16
            else:
                raise AssertionError(f'Missing dtype {dtype} not tested.')

            self.assertEqual(torch.promote_types(dtype, torch.bfloat16), expected_dtype)
            self.assertEqual(torch.promote_types(torch.bfloat16, dtype), expected_dtype)
            self.assertEqual((bf + t).dtype, expected_dtype)

    @float_double_default_dtype
    def test_alternate_result(self, device):
        f = torch.tensor([1, 1, 1, 1], dtype=torch.float, device=device)
        o = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)
        self.assertRaisesRegex(RuntimeError,
                               "can't be cast to",
                               lambda: torch.add(f, f, out=o))
        d = torch.tensor([1, 1, 1, 1], dtype=torch.double, device=device)
        torch.add(f, f, out=d)
        self.assertEqual(d.dtype, torch.double)
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(f + f, d)

    @float_double_default_dtype
    def test_mixed_type_backward(self, device):
        f = torch.ones([3, 3], dtype=torch.float, requires_grad=True, device=device)
        ten = torch.tensor([10.], dtype=torch.double, device=device)
        tens = f * ten
        s = (tens + 2).sum()
        s.backward()
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(f.grad, tens)

        # If we don't convert the returned grad_input to the actual input type
        # we get an error like:
        # RuntimeError: Function SubBackward0 returned an invalid gradient at index 0 - expected type \
        # torch.FloatTensor but got torch.DoubleTensor
        f_dtypes = [torch.float, torch.double]
        if self.device_type == 'cuda':
            f_dtypes = f_dtypes + [torch.half]
        i_dtypes = [torch.int, torch.long]
        for func in [torch.add, torch.sub, torch.rsub, torch.mul, torch.div]:
            for dtype1, dtype2 in itertools.product(f_dtypes, f_dtypes + i_dtypes):
                x = torch.ones(10, requires_grad=True, dtype=dtype1, device=device)
                y = torch.ones(10, dtype=dtype2, device=device)
                func(x, y).sum().backward()

    def _get_test_tensor(self, device, dtype, remove_zeros=False):
        shape = [5, 5, 5]
        if dtype == torch.bool:
            tensor = torch.randint(int(remove_zeros), 2, shape, device=device, dtype=dtype)
        elif dtype.is_floating_point or dtype.is_complex:
            # "_th_normal_ not supported on CPUType for Half" so simpler create and convert
            tensor = torch.randn(shape, device=device)
            tensor = tensor.to(dtype)
            if remove_zeros:
                tensor[torch.abs(tensor) < 0.05] = 5
        else:
            tensor = torch.randint(-5 if dtype.is_signed else 0, 10, shape, device=device, dtype=dtype)
            if remove_zeros:
                tensor[tensor == 0] = 5
        return tensor

    # verifies that torch.<op>(first, second) is the same as
    # torch.<op>(first.to(common_dtype), second.to(common_dtype)) in cases where that should hold.
    @float_double_default_dtype
    def test_many_promotions(self, device):
        # Can also include half on CPU in cases where it will be promoted to a
        # supported dtype
        dtypes1 = torch.testing.get_all_math_dtypes('cuda')
        dtypes2 = torch.testing.get_all_math_dtypes(device)
        ops = [torch.add, torch.sub, torch.mul, torch.div, torch.rsub]
        for dt1, dt2 in itertools.product(dtypes1, dtypes2):
            for op, non_contiguous in itertools.product(ops, [True, False]):
                common_dtype = torch.promote_types(dt1, dt2)
                if common_dtype == torch.half and self.device_type == 'cpu':
                    continue
                if op == torch.sub and common_dtype != torch.bool:
                    # Subtraction, the `-` operator, with a bool tensor is not supported.
                    continue
                first = self._get_test_tensor(device, dt1)
                second = self._get_test_tensor(device, dt2, op == torch.div)
                # test ops with non-contiguous tensors
                if non_contiguous:
                    first = first.transpose(0, 2)
                    second = second.transpose(2, 1)
                    self.assertNotEqual(first.stride(), second.stride(),
                                        msg="some non-contiguous issues could be missed if tensors have same strides")

                self.assertEqual(not first.is_contiguous(), non_contiguous)
                self.assertEqual(not second.is_contiguous(), non_contiguous)
                result = op(first, second)
                expected = op(first.to(common_dtype), second.to(common_dtype))
                self.assertEqual(result.dtype, expected.dtype, msg='{} with {}, {}'.format(op.__name__, dt1, dt2))
                self.assertEqual(result, expected, msg='{} with {}, {}'.format(op.__name__, dt1, dt2))

    @float_double_default_dtype
    def test_non_promoting_ops(self, device):
        x = torch.ones(4, dtype=torch.double, device=device)
        with self.assertRaises(RuntimeError):
            torch.lerp(x, torch.ones(4, dtype=torch.float, device=device), 1)

    @float_double_default_dtype
    def test_alpha_mismatch(self, device):
        x = torch.ones(4, dtype=torch.int, device=device)
        err = 'alpha must not be'
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.add(x, x, alpha=1.1))
        x = x.to(torch.bool)
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.add(x, x, alpha=1.1))
        self.assertEqual(x + x, torch.add(x, x, alpha=True))

    @float_double_default_dtype
    def test_booleans(self, device):
        onedim = torch.tensor([True], device=device)

        self.assertEqual(onedim + onedim, onedim)
        self.assertEqual(onedim + True, onedim)
        self.assertEqual(torch.add(True, True), True)
        self.assertEqual(torch.add(False, False), False)
        self.assertEqual(torch.add(False, True), True)

        self.assertRaisesRegex(RuntimeError, "Boolean alpha only supported",
                               lambda: torch.add(1, 1, alpha=True))
        self.assertEqual(torch.add(torch.tensor(True, device=device),
                         torch.tensor(True, device=device), True),
                         torch.tensor(True, device=device))

    @float_double_default_dtype
    def test_create_bool_tensors(self, device):
        expected = torch.tensor([0], dtype=torch.int64, device=device)
        self.assertEqual(torch.arange(False, True, device=device), expected)
        self.assertEqual(torch.arange(True, device=device), expected)
        expected = torch.tensor([0, 0.5], dtype=torch.get_default_dtype(), device=device)
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(torch.arange(False, True, 0.5, device=device), expected)
        expected = torch.ones(0, dtype=torch.int64, device=device)
        self.assertEqual(torch.arange(False, False, device=device), expected)

        self.assertEqual(torch.linspace(False, True, device=device), torch.linspace(0, 1, device=device))
        self.assertEqual(torch.logspace(False, True, device=device), torch.logspace(0, 1, device=device))

        # this seems like odd behavior but ints also create float tensors, numpy doesn't have this function.
        self.assertEqual(torch.scalar_tensor(False, device=device), torch.tensor(0., device=device))

    @dtypes(*itertools.product(torch.testing.get_all_dtypes(), torch.testing.get_all_dtypes()))
    def test_result_type(self, device, dtypes):
        "Test result_type for tensor vs tensor and scalar vs scalar."

        def _get_dtype(x):
            "Get the dtype of x if x is a tensor. If x is a scalar, get its corresponding dtype if it were a tensor."
            if torch.is_tensor(x):
                return x.dtype
            elif isinstance(x, bool):
                return torch.bool
            elif isinstance(x, int):
                return torch.int64
            elif isinstance(x, float):
                return torch.float32
            elif isinstance(x, complex):
                return torch.complex64
            else:
                raise AssertionError(f"Unkonwn type {x}")

        # tensor against tensor
        a_tensor = torch.tensor((0, 1), device=device, dtype=dtypes[0])
        a_single_tensor = torch.tensor(1, device=device, dtype=dtypes[0])
        a_scalar = a_single_tensor.item()
        b_tensor = torch.tensor((1, 0), device=device, dtype=dtypes[1])
        b_single_tensor = torch.tensor(1, device=device, dtype=dtypes[1])
        b_scalar = b_single_tensor.item()
        combo = ((a_tensor, a_single_tensor, a_scalar), (b_tensor, b_single_tensor, b_scalar))
        for a, b in itertools.product(*combo):
            dtype_a = _get_dtype(a)
            dtype_b = _get_dtype(b)
            try:
                result = a + b
            except RuntimeError:
                with self.assertRaises(RuntimeError):
                    torch.promote_types(dtype_a, dtype_b)
                with self.assertRaises(RuntimeError):
                    torch.result_type(a, b)
            else:
                dtype_res = _get_dtype(result)
                if a is a_scalar and b is b_scalar and dtype_a == torch.bool and dtype_b == torch.bool:
                    # special case: in Python, True + True is an integer
                    self.assertEqual(dtype_res, torch.int64, f"a == {a}, b == {b}")
                else:
                    self.assertEqual(dtype_res, torch.result_type(a, b), f"a == {a}, b == {b}")
                if a is a_scalar and b is b_scalar:  # Python internal type determination is good enough in this case
                    continue
                if any(a is a0 and b is b0 for a0, b0 in zip(*combo)):  # a and b belong to the same class
                    self.assertEqual(dtype_res, torch.promote_types(dtype_a, dtype_b), f"a == {a}, b == {b}")

    # Spot check some result type for tensor against scalar (including single-element tensor).
    @float_double_default_dtype
    def test_result_type_tensor_vs_scalar(self, device):
        def _test_spot(a, b, res_dtype):
            self.assertEqual(torch.result_type(a, b), res_dtype)
            self.assertEqual(torch.result_type(b, a), res_dtype)

        _test_spot(torch.tensor([1, 2], dtype=torch.half, device=device),
                   torch.tensor(1, dtype=torch.long, device=device), torch.half)
        _test_spot(torch.tensor(1, dtype=torch.float, device=device),
                   torch.tensor([1, 2], dtype=torch.double, device=device), torch.double)
        _test_spot(torch.tensor(1, dtype=torch.int, device=device), 1, torch.int)
        _test_spot(torch.tensor(1, device=device), 1., torch.get_default_dtype())
        _test_spot(torch.tensor(1, dtype=torch.long, device=device),
                   torch.tensor([1, 1], dtype=torch.int, device=device), torch.int)
        _test_spot(torch.tensor([1., 1.], dtype=torch.float, device=device), 1., torch.float)
        _test_spot(torch.tensor([1., 1.], dtype=torch.complex64, device=device),
                   torch.tensor(1., dtype=torch.complex128, device=device), torch.complex64)
        _test_spot(torch.tensor([1., 1.], dtype=torch.complex128, device=device),
                   torch.tensor(1., dtype=torch.complex64, device=device), torch.complex128)
        _test_spot(torch.tensor([1, 1], dtype=torch.bool, device=device), 1., torch.get_default_dtype())

    @float_double_default_dtype
    def test_can_cast(self, device):
        self.assertTrue(torch.can_cast(torch.double, torch.float))
        self.assertFalse(torch.can_cast(torch.float, torch.int))

    @float_double_default_dtype
    def test_comparison_ops_with_type_promotion(self, device):
        value_for_type = {
            torch.uint8: (1 << 5),
            torch.int8: (1 << 5),
            torch.int16: (1 << 10),
            torch.int32: (1 << 20),
            torch.int64: (1 << 35),
            torch.float16: (1 << 10),
            torch.float32: (1 << 20),
            torch.float64: (1 << 35),
            torch.complex64: (1 << 20),
            torch.complex128: (1 << 35)
        }
        comparison_ops = [
            dict(
                name="lt",
                out_op=lambda x, y, d: torch.lt(x, y, out=torch.empty(0, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.lt(x, y),
                compare_op=lambda x, y: x < y,
            ),
            dict(
                name="le",
                out_op=lambda x, y, d: torch.le(x, y, out=torch.empty(0, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.le(x, y),
                compare_op=lambda x, y: x <= y,
            ),
            dict(
                name="gt",
                out_op=lambda x, y, d: torch.gt(x, y, out=torch.empty(0, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.gt(x, y),
                compare_op=lambda x, y: x > y,
            ),
            dict(
                name="ge",
                out_op=lambda x, y, d: torch.ge(x, y, out=torch.empty(0, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.ge(x, y),
                compare_op=lambda x, y: x >= y,
            ),
            dict(
                name="eq",
                out_op=lambda x, y, d: torch.eq(x, y, out=torch.empty(0, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.eq(x, y),
                compare_op=lambda x, y: x == y,
            ),
            dict(
                name="ne",
                out_op=lambda x, y, d: torch.ne(x, y, out=torch.empty(0, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.ne(x, y),
                compare_op=lambda x, y: x != y,
            ),
        ]
        for op in comparison_ops:
            for dt1 in torch.testing.get_all_math_dtypes(device):
                for dt2 in torch.testing.get_all_math_dtypes(device):
                    if (dt1.is_complex or dt2.is_complex) and not (op["name"] == "eq" or op["name"] == "ne"):
                        continue
                    val1 = value_for_type[dt1]
                    val2 = value_for_type[dt2]
                    t1 = torch.tensor([val1], dtype=dt1, device=device)
                    t2 = torch.tensor([val2], dtype=dt2, device=device)
                    expected = torch.tensor([op["compare_op"](val1, val2)], dtype=torch.bool)

                    out_res = op["out_op"](t1, t2, device)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == torch.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

                    out_res = op["ret_op"](t1, t2)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == torch.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

                    # test that comparing a zero dim tensor with another zero dim tensor has type promotion behavior
                    t1 = torch.tensor(val1, dtype=dt1, device=device)
                    t2 = torch.tensor(val2, dtype=dt2, device=device)
                    expected = torch.tensor(op["compare_op"](val1, val2), dtype=torch.bool)

                    out_res = op["out_op"](t1, t2, device)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == torch.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

                    out_res = op["ret_op"](t1, t2)
                    self.assertEqual(out_res, expected)
                    self.assertTrue(out_res.dtype == torch.bool)
                    self.assertTrue(t1.dtype == dt1)
                    self.assertTrue(t2.dtype == dt2)

    # XLA tests fail for self.assertRaises for complex dtypes
    @onlyOnCPUAndCUDA
    def test_complex_assertraises(self, device):
        comparison_ops = [
            dict(name="lt", compare_op=lambda x, y: x < y, ),
            dict(name="le", compare_op=lambda x, y: x <= y, ),
            dict(name="gt", compare_op=lambda x, y: x > y, ),
            dict(name="ge", compare_op=lambda x, y: x >= y, ),
            dict(name="eq", compare_op=lambda x, y: x == y, ),
            dict(name="ne", compare_op=lambda x, y: x != y, ),
        ]
        for op in comparison_ops:
            for dt1 in torch.testing.get_all_math_dtypes(device):
                for dt2 in torch.testing.get_all_math_dtypes(device):
                    if (dt1.is_complex or dt2.is_complex) and not (op["name"] == "eq" or op["name"] == "ne"):
                        u = torch.tensor([1], dtype=dt1, device=device)
                        v = torch.tensor([2], dtype=dt2, device=device)
                        self.assertRaises(RuntimeError, lambda: torch.tensor([op["compare_op"](u, v)], dtype=torch.bool))

        for dtype in [torch.complex64, torch.complex128]:
            t = self._get_test_tensor(device, dtype, False)
            self.assertRaises(RuntimeError, lambda: t.to_sparse())

    @float_double_default_dtype
    def test_lt_with_type_promotion(self, device):
        for dt in torch.testing.get_all_math_dtypes(device):
            x = torch.tensor([0], dtype=dt, device=device)
            expected = torch.tensor([True], dtype=torch.bool, device=device)

            if dt.is_complex:
                continue

            actual = x < 0.5
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

            actual = x < torch.tensor(0.5, device=device)
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

            x = torch.tensor(0, dtype=dt, device=device)
            expected = torch.tensor(True, dtype=torch.bool, device=device)
            actual = x < 0.5
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

            actual = x < torch.tensor(0.5, device=device)
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

    @float_double_default_dtype
    def test_promote_types(self, device):
        self.assertEqual(torch.promote_types(torch.float, torch.int), torch.float)
        self.assertEqual(torch.promote_types(torch.float, torch.double), torch.double)
        self.assertEqual(torch.promote_types(torch.int, torch.uint8), torch.int)

    @float_double_default_dtype
    def test_promote_self(self, device):
        for dtype in torch.testing.get_all_dtypes():
            self.assertEqual(torch.promote_types(dtype, dtype), dtype)

    @float_double_default_dtype
    def test_indexing(self, device):
        # https://github.com/pytorch/pytorch/issues/28010
        a = torch.ones(5, 2, dtype=torch.double, device=device)
        b = torch.zeros(5, dtype=torch.int, device=device)
        with self.assertRaises(RuntimeError):
            a[:, [1]] = b.unsqueeze(-1)

        x = torch.ones(5, 2, dtype=torch.double, device=device)
        y = torch.zeros(5, dtype=torch.double, device=device)
        x[:, [1]] = y.unsqueeze(-1)
        expected = torch.tensor([(1, 0), (1, 0), (1, 0), (1, 0), (1, 0)], dtype=torch.double, device=device)
        self.assertEqual(x, expected)


        # https://github.com/pytorch/pytorch/issues/27824
        tmp = torch.ones(9, 9, dtype=torch.float, device=device)
        mask = torch.ones(10, 10, dtype=torch.uint8, device=device)
        result = tmp + mask[1:, 1:]
        expected = torch.full([9, 9], 2., dtype=torch.float, device=device).fill_(2.)
        self.assertEqual(result, expected)

    @float_double_default_dtype
    def test_transpose(self, device):
        # https://github.com/pytorch/pytorch/issues/28502
        a = torch.tensor([[True, True], [False, True]], device=device)
        self.assertEqual(a.t() == 0, a.t() == False)  # noqa: E712

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    @float_double_default_dtype
    def test_div_promotion(self, device, dtype):
        for op in (torch.div, torch.true_divide):
            dividend = (torch.randn(5, device=device) * 100).to(dtype)
            divisor = torch.arange(1, 6, device=device).to(dtype)

            # Tests tensor/tensor division
            casting_result = dividend.to(torch.get_default_dtype()) / divisor.to(torch.get_default_dtype())
            self.assertEqual(casting_result, op(dividend, divisor))

            # Tests tensor/scalar division
            casting_result = dividend.to(torch.get_default_dtype()) / 2
            self.assertEqual(casting_result, op(dividend, 2.))

    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double,
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_div_promotion_out(self, device, dtype):
        for op in (torch.div, torch.true_divide):
            dividend = (torch.randn(5, device=device) * 100).to(dtype)
            divisor = torch.arange(1, 6, device=device).to(dtype)

            # Tests that requests for an integer quotient fail
            if not dtype.is_floating_point:
                integral_quotient = torch.empty(5, device=device, dtype=dtype)
                with self.assertRaises(RuntimeError):
                    op(dividend, divisor, out=integral_quotient)
                with self.assertRaises(RuntimeError):
                    op(dividend, 2, out=integral_quotient)
            else:
                # Tests that requests for a floating quotient succeed
                floating_quotient = torch.empty(5, device=device, dtype=dtype)
                div_result = dividend / divisor
                self.assertEqual(div_result,
                                 op(dividend, divisor, out=floating_quotient))
                self.assertEqual(dividend / 2,
                                 op(dividend, 2, out=floating_quotient))

    @dtypes(torch.float, torch.double,
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_div_promotion_inplace(self, device, dtype):
        for op in (torch.Tensor.div_, torch.Tensor.true_divide_):
            dividend = (torch.randn(5, device=device) * 100).to(dtype)
            divisor = torch.arange(1, 6, device=device).to(dtype)

            # Tests that requests for an integer quotient fail
            if not dtype.is_floating_point:
                with self.assertRaises(RuntimeError):
                    op(dividend, divisor)
                with self.assertRaises(RuntimeError):
                    op(dividend, 2)
            else:
                # Tests that requests for a floating quotient succeed
                div_result = dividend.clone().div_(divisor)
                self.assertEqual(div_result, op(dividend.clone(), divisor))
                self.assertEqual(dividend.clone().div_(2), op(dividend.clone(), 2))

    def _test_sparse_op_input_tensors(self, device, dtype, coalesced, zeros=True):
        t = self._get_test_tensor(device, dtype, not zeros)
        if zeros and dtype != torch.bool:
            # ensure sparsity. Bool should already have sufficient sparsity.
            mask = self._get_test_tensor(device, torch.bool)
            t = t * mask

        if coalesced:
            s = t.to_sparse()
        else:
            s = t.to_sparse()
            indices = torch.cat((s.indices(), s.indices()), 1)
            values = torch.cat((s.values(), s.values()), 0)
            s = torch.sparse_coo_tensor(indices=indices, values=values, size=s.size(), dtype=dtype, device=device)
            t = s.to_dense()
        self.assertEqual(s.is_coalesced(), coalesced)
        self.assertEqual(s.dtype, dtype)
        self.assertEqual(t.dtype, s.dtype)
        return t, s

    def _get_precision(self, dtype, coalesced):
        if dtype == torch.half and not coalesced:
            # very low precision for uncoalesced float16 sparse tensors since
            # ops like (s1 + s2).to_dense() will add four low-precision
            # floating point values.
            return 5e-2
        if dtype == torch.half:
            return 1e-3
        # uses default
        return None

    def _test_sparse_op(self, op_name, inplace, dtype1, dtype2, device, coalesced):
        if dtype1.is_complex or dtype2.is_complex:
            return

        suffix = '_' if inplace else ''
        err = "{} {}({}, {})".format("  coalesced" if coalesced else "uncoalesced", op_name + suffix, dtype1, dtype2)

        def op(t1, t2):
            return getattr(t1, op_name + suffix)(t2)

        add_sub = op_name == 'add' or op_name == 'sub'

        (dense1, sparse1) = self._test_sparse_op_input_tensors(device, dtype1, coalesced)
        (dense2, sparse2) = self._test_sparse_op_input_tensors(device, dtype2, coalesced, op_name != 'div')

        common_dtype = torch.result_type(dense1, dense2)
        if self.device_type == 'cpu' and common_dtype == torch.half:
            self.assertRaises(RuntimeError, lambda: op(s1, d2))

        # Skip inplace tests that would fail due to inability to cast to the output type.
        # Some of these would also raise errors due to not being a supported op.
        if inplace and not torch.can_cast(common_dtype, dtype1):
            self.assertRaises(RuntimeError, lambda: op(dense1, sparse2))
            self.assertRaises(RuntimeError, lambda: op(sparse1, sparse2))
            self.assertRaises(RuntimeError, lambda: op(sparse1, dense2))
            return

        expected = op(dense1.clone(), dense2)
        precision = self._get_precision(expected.dtype, coalesced)
        rtol = None if precision is None else 0
        test_tensors = [expected, dense1, sparse1, dense2, sparse2]
        e, d1, s1, d2, s2 = [x.clone() for x in test_tensors] if inplace else test_tensors

        # Test op(sparse, sparse)
        if op_name != 'div':
            sparse = op(s1, s2)
            self.assertEqual(sparse.dtype, e.dtype)
            self.assertEqual(e, sparse.to_dense(), atol=precision, rtol=rtol, msg=err)
        else:
            # sparse division only supports division by a scalar
            self.assertRaises(RuntimeError, lambda: op(s1, s2).to_dense())

        # Test op(dense, sparse)
        if add_sub:
            if inplace:
                e, d1, s1, d2, s2 = [x.clone() for x in test_tensors]
            dense_sparse = op(d1, s2)
            self.assertEqual(e, dense_sparse, atol=precision, rtol=rtol, msg=err)
        else:
            # sparse division only supports division by a scalar
            # mul: Didn't find kernel to dispatch to for operator 'aten::_nnz'
            self.assertRaises(RuntimeError, lambda: op(d1, s2))

        # Test op(sparse, dense) not supported for any ops:
        # add(sparse, dense) is not supported. Use add(dense, sparse) instead.
        # sparse division only supports division by a scalar
        # mul: Didn't find kernel to dispatch to for operator 'aten::_nnz'.
        self.assertRaises(RuntimeError, lambda: op(s1, d2))

        # Test op(sparse, scalar)
        if not add_sub and not (self.device_type == 'cpu' and dtype1 == torch.half):
            if inplace:
                e, d1, s1, d2, s2 = [x.clone() for x in test_tensors]
            scalar = d2.view(d2.numel())[0].item()

            sparse = op(s1, scalar)
            dense_scalar = op(d1, scalar)
            self.assertEqual(sparse.dtype, dense_scalar.dtype)
            self.assertEqual(dense_scalar, sparse.to_dense(), atol=precision, rtol=rtol, msg=err)
        else:
            # add(sparse, dense) is not supported. Use add(dense, sparse) instead.
            # "mul_cpu" / "div_cpu" not implemented for 'Half'
            self.assertRaises(RuntimeError, lambda: op(s1, d2.view(d2.numel())[0].item()))

    def _run_all_tests_for_sparse_op(self, op_name, device, dtypes):
        for dtype1, dtype2 in itertools.product(dtypes, dtypes):
            for inplace, coalesced in itertools.product([True, False], [True, False]):
                self._test_sparse_op(op_name, inplace, dtype1, dtype2, device, coalesced)

    @onlyOnCPUAndCUDA
    def test_sparse_add(self, device):
        self._run_all_tests_for_sparse_op('add', device,
                                          dtypes=torch.testing.get_all_math_dtypes(device))

    @onlyOnCPUAndCUDA
    def test_sparse_mul(self, device):
        self._run_all_tests_for_sparse_op('mul', device,
                                          dtypes=torch.testing.get_all_math_dtypes(device))

    @onlyOnCPUAndCUDA
    def test_sparse_div(self, device):
        self._run_all_tests_for_sparse_op('div', device,
                                          dtypes=(torch.float32, torch.float64,
                                                  torch.complex64, torch.complex128))

    @onlyOnCPUAndCUDA
    def test_sparse_sub(self, device):
        self._run_all_tests_for_sparse_op('sub', device,
                                          dtypes=torch.testing.get_all_math_dtypes(device))

    @onlyOnCPUAndCUDA
    @dtypes(torch.bool, torch.short, torch.uint8, torch.int, torch.long)
    @float_double_default_dtype
    def test_sparse_div_promotion(self, device, dtype):
        for op in (torch.div, torch.true_divide):
            dividend = torch.randn(5, device=device).to(dtype)
            divisor = 2
            dividend_sparse = dividend.to_sparse()
            casting_result = dividend.to(torch.get_default_dtype()) / 2
            self.assertEqual(casting_result, op(dividend_sparse, 2).to_dense())

    @onlyOnCPUAndCUDA
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    def test_integer_addcdiv_deprecated(self, device, dtype):
        t = torch.tensor(1, device=device, dtype=dtype)

        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported.+'):
            torch.addcdiv(t, t, t)
        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported.+'):
            torch.addcdiv(t, t, t, out=t)
        with self.assertRaisesRegex(RuntimeError, '^Integer division.+is no longer supported+'):
            t.addcdiv_(t, t)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @float_double_default_dtype
    @onlyCPU
    @dtypes(*list(itertools.product(torch_to_numpy_dtype_dict.keys(),
                                    torch_to_numpy_dtype_dict.keys())))
    def test_numpy_array_binary_ufunc_promotion(self, device, dtypes):
        import operator
        np_type = torch_to_numpy_dtype_dict[dtypes[0]]
        torch_type = dtypes[1]

        t = torch.tensor((1,), device=device, dtype=torch_type)
        a = np.array((1,), dtype=np_type)
        a_as_t = torch.from_numpy(a).to(device=device)

        for np_first in (True, False):
            for op in (operator.add, torch.add):

                # Acquires results of binary ufunc type promotion.
                try:
                    actual = op(a, t) if np_first else op(t, a)
                except Exception as e:
                    actual = e

                try:
                    expected = op(a_as_t, t) if np_first else op(t, a_as_t)
                except Exception as e:
                    expected = e

                same_result = (type(expected) == type(actual)) and expected == actual

                # Note: An "undesired failure," as opposed to an "expected failure"
                # is both expected (we know the test will fail) and
                # undesirable (if PyTorch was working properly the test would
                # not fail). This test is affected by three issues (see below)
                # that will cause undesired failures. It detects when these
                # issues will occur and updates this bool accordingly.
                undesired_failure = False

                # A NumPy array as the first argument to the plus operator
                # or as any argument to torch.add is not working as
                # intended.
                # See https://github.com/pytorch/pytorch/issues/36363.
                if np_first and op is operator.add:
                    undesired_failure = True
                if op is torch.add:
                    undesired_failure = True

                # Expects the same result if undesired_failure is false
                # and a different result otherwise.
                # Note: These cases prettyprint the failing inputs to make
                # debugging test failures easier.
                if undesired_failure and same_result:
                    msg = ("Failure: {0} == {1}. "
                           "torch type was {2}. NumPy type was {3}. np_first is {4} "
                           "default type is {5}.").format(actual, expected,
                                                          torch_type, np_type,
                                                          np_first,
                                                          torch.get_default_dtype())
                    self.fail(msg)

                if not undesired_failure and not same_result:
                    msg = ("Failure: {0} != {1}. "
                           "torch type was {2}. NumPy type was {3}. np_first is {4} "
                           "default type is {5}.").format(actual, expected,
                                                          torch_type, np_type,
                                                          np_first,
                                                          torch.get_default_dtype())
                    self.fail(msg)


    @onlyOnCPUAndCUDA
    def test_cat_different_dtypes(self, device):
        dtypes = torch.testing.get_all_dtypes(include_bfloat16=False)
        for x_dtype, y_dtype in itertools.product(dtypes, dtypes):
            x_vals, y_vals = [1, 2, 3], [4, 5, 6]

            x = torch.tensor(x_vals, device=device, dtype=x_dtype)
            y = torch.tensor(y_vals, device=device, dtype=y_dtype)

            if x_dtype is torch.bool:
                x_vals = [1, 1, 1]
            if y_dtype is torch.bool:
                y_vals = [1, 1, 1]

            res_dtype = torch.result_type(x, y)
            expected_res = torch.tensor(x_vals + y_vals, device=device, dtype=res_dtype)
            res = torch.cat([x, y])
            self.assertEqual(res, expected_res, exact_dtype=True)

    @onlyOnCPUAndCUDA
    def test_cat_out_different_dtypes(self, device):
        dtypes = torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False)
        for x_dtype, y_dtype, out_dtype in itertools.product(dtypes, dtypes, dtypes):
            out = torch.zeros(6, device=device, dtype=out_dtype)
            x = torch.tensor([1, 2, 3], device=device, dtype=x_dtype)
            y = torch.tensor([4, 5, 6], device=device, dtype=y_dtype)
            expected_out = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=out_dtype)
            if (((x_dtype.is_floating_point or y_dtype.is_floating_point)
                    and not (out_dtype.is_floating_point or out_dtype.is_complex))
                    or ((x_dtype.is_complex or y_dtype.is_complex) and not out_dtype.is_complex)):
                # This combinations do not support type conversion to a different class out type
                with self.assertRaises(RuntimeError):
                    torch.cat([x, y], out=out)
            else:
                torch.cat([x, y], out=out)
                self.assertEqual(out, expected_out, exact_dtype=True)

    # Verfies that unary ops require matching out types
    @onlyOnCPUAndCUDA
    @dtypes(*itertools.product((torch.int64,
                                torch.float32, torch.float64,
                                torch.complex64, torch.complex128),
                               (torch.int64,
                                torch.float32, torch.float64,
                                torch.complex64, torch.complex128)))
    def test_unary_op_out_casting(self, device, dtypes):
        t = torch.tensor((1), dtype=dtypes[0], device=device)
        out = torch.empty(0, dtype=dtypes[1], device=device)

        ops = (torch.neg, torch.floor, torch.ceil)
        float_only_ops = {torch.floor, torch.ceil}
        real_only_ops = {torch.floor, torch.ceil}
        for op in ops:
            if dtypes[0] is not dtypes[1]:
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            elif op in real_only_ops and dtypes[0].is_complex:
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            elif op in float_only_ops and (not dtypes[0].is_floating_point and not dtypes[0].is_complex):
                with self.assertRaises(RuntimeError):
                    op(t, out=out)
            else:
                self.assertEqual(op(t, out=out), op(t))
                self.assertEqual(op(t, out=out), out)

    # Verifies that the out= argument doesn't affect the computation, that
    # is, out = op(...) and op(..., out=out) produce the same result.
    @onlyOnCPUAndCUDA
    def test_computation_ignores_out(self, device):
        t = torch.tensor(33000, dtype=torch.float16, device=device)
        out = torch.empty(0, dtype=torch.float64, device=device)
        result = torch.add(t, t, out=out)
        self.assertEqual(result, t + t, exact_dtype=False)
        self.assertNotEqual(result, t.double() + t, exact_dtype=False)

        a = torch.tensor(1.5, dtype=torch.float16, device=device)
        b = torch.tensor(.666, dtype=torch.float16, device=device)
        result = torch.true_divide(a, b, out=out)
        self.assertEqual(result, a / b, exact_dtype=False)
        self.assertNotEqual(result, a.double() / a, exact_dtype=False)

        a = torch.tensor(5, dtype=torch.uint8, device=device)
        b = torch.tensor(8, dtype=torch.uint8, device=device)
        result = torch.sub(a, b, out=out)
        self.assertEqual(result, a - b, exact_dtype=False)
        self.assertNotEqual(result, a.double() - b, exact_dtype=False)

    @dtypesIfCUDA(*itertools.product(torch.testing.get_all_dtypes(include_bfloat16=False, include_complex=False),
                                     torch.testing.get_all_dtypes(include_bfloat16=False, include_complex=False)))
    @dtypes(*itertools.product(torch.testing.get_all_dtypes(include_half=False, include_bfloat16=False,
                                                            include_complex=False),
                               torch.testing.get_all_dtypes(include_half=False, include_bfloat16=False,
                                                            include_complex=False)))
    def test_atan2_type_promotion(self, device, dtypes):
        dtype1, dtype2 = dtypes
        default_float = torch.get_default_dtype()

        def is_int(dtype):
            return dtype in torch.testing.get_all_int_dtypes() + [torch.bool]

        def is_float(dtype):
            return dtype in torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False)

        def get_binary_float_result_type(x, y):
            dtype1 = x.dtype
            dtype2 = y.dtype
            if is_float(dtype1) and is_float(dtype2):
                return torch.result_type(x, y)
            elif is_float(dtype1) and is_int(dtype2):
                return dtype1
            elif is_int(dtype1) and is_float(dtype2):
                return dtype2
            elif is_int(dtype1) and is_int(dtype2):
                return default_float

        x = torch.tensor(1, dtype=dtype1, device=device)
        y = torch.tensor(2, dtype=dtype2, device=device)
        self.assertEqual(get_binary_float_result_type(x, y), torch.atan2(x, y).dtype)

instantiate_device_type_tests(TestTypePromotion, globals())

if __name__ == '__main__':
    run_tests()
