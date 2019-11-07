import torch
import unittest
import itertools

from common_utils import TestCase, run_tests, load_tests
import itertools

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestTypePromotion(TestCase):

    def setUp(self):
        super(TestTypePromotion, self).setUp()
        torch.set_default_dtype(torch.float32)
        self.device = 'cpu'

    # In-place operations don't promote.
    # `int+float -> float` but `int.add_(float)` is rejected as an error.
    # Promoting inplace would require re-allocating and copying the memory of the
    # tensor data, since element size could change.
    def test_inplace(self):
        int_tensor = torch.ones([4, 4, 4], dtype=torch.int32, device=self.device)

        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: int_tensor.add_(1.5))

        expected = torch.ones([4, 4, 4], dtype=torch.int32, device=self.device)

        long_tensor = torch.ones([4, 4, 4], dtype=torch.int64, device=self.device)
        int_tensor.add_(long_tensor)
        int_tensor.add_(1)
        three = expected + 2
        self.assertEqual(int_tensor, three)
        self.assertEqual(int_tensor.dtype, torch.int32)

        bool_tensor = torch.tensor([1, 1, 1], dtype=torch.bool)
        uint8_tensor = torch.tensor([1, 1, 1], dtype=torch.uint8)
        # We treat bool as a separate category, which means uint8 cannot cast to bool.
        self.assertRaisesRegex(RuntimeError, "can't be cast to", lambda: bool_tensor.add_(uint8_tensor))

        # We allow demotion from signed to unsigned, unlike numpy, because:
        # * We don't want the performance penalty of inspecting scalar values.
        # * We don't want 'signed' to be considered a distinct 'category'
        # in promotion rules.
        # We don't want signed to be a separate category because if it was,
        # uint16_tensor + 5 would result in a long_tensor, which is not what we want.
        int16_tensor = torch.tensor([1, 1, 1], dtype=torch.int16)
        uint8_tensor *= int16_tensor

    def test_unsinged(self):
        dont_promote = torch.ones(3, dtype=torch.uint8) + 5
        self.assertEqual(dont_promote.dtype, torch.uint8)

    # some basic examples

    def test_int_promotion(self):
        a = torch.ones([4, 4, 4], dtype=torch.int32, device=self.device)
        b = torch.ones([4, 4, 4], dtype=torch.int64, device=self.device)
        c = a + b
        self.assertEqual(c, b + b)
        self.assertEqual(c.dtype, torch.int64)

    def test_float_promotion(self):
        a = torch.ones([4, 4, 4], dtype=torch.float, device=self.device)
        b = torch.ones([4, 4, 4], dtype=torch.double, device=self.device)
        c = a + b
        self.assertEqual(c, b + b)
        self.assertEqual(c.dtype, torch.double)
        c = b + a
        self.assertEqual(c, b + b)
        self.assertEqual(c.dtype, torch.double)

    def test_add_wrapped(self):
        a = torch.ones([4, 4, 4], dtype=torch.int, device=self.device)
        b = 1
        c = a + b
        self.assertEqual(c, a + a)
        self.assertEqual(c.dtype, torch.int)

    def test_int_to_float(self):
        a = torch.ones([4, 4, 4], dtype=torch.int32, device=self.device)
        b = torch.ones([4, 4, 4], dtype=torch.float, device=self.device)
        c = a + b
        self.assertEqual(c.dtype, torch.float32)

    # some examples from:
    # https://github.com/pytorch/pytorch/issues/9515

    def test_from_issue(self):
        a = torch.rand(3, dtype=torch.float32, device=self.device)
        u = torch.tensor([0, 0, 1], dtype=torch.uint8, device=self.device)
        self.assertEqual((a * 5).dtype, torch.float32)
        self.assertEqual((u + 1).dtype, torch.uint8)
        self.assertEqual((u + 1000).dtype, torch.uint8)  # integer overflow

        # not a "wrapped number"
        other = torch.tensor(5.5, dtype=torch.double, device=self.device)

        self.assertEqual((u + 5.5).dtype, torch.get_default_dtype())
        self.assertEqual((u + other).dtype, torch.double)
        # adding a 0-dim tensor to a float doesn't promote to double unless first
        # type was integral.
        self.assertEqual((a + other).dtype, torch.float32)

    def test_half(self):
        half = torch.tensor(5.5, dtype=torch.float16, device=self.device)
        if(self.device == 'cpu'):
            self.assertRaisesRegex(RuntimeError, "not implemented for 'Half'",
                                   lambda: half + 2.2)
        else:
            self.assertEqual((half + 2.2).dtype, torch.float16)
            self.assertEqual((half + 100000).dtype, torch.float16)  # inf
            default_tensor = torch.tensor(100000.0, device=self.device)
            self.assertEqual((half + default_tensor).dtype, torch.get_default_dtype())

    def test_alternate_result(self):
        f = torch.tensor([1, 1, 1, 1], dtype=torch.float, device=self.device)
        o = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=self.device)
        self.assertRaisesRegex(RuntimeError,
                               "can't be cast to",
                               lambda: torch.add(f, f, out=o))
        d = torch.tensor([1, 1, 1, 1], dtype=torch.double, device=self.device)
        torch.add(f, f, out=d)
        self.assertEqual(d.dtype, torch.double)
        self.assertEqual(f + f, d)

    def test_mixed_type_backward(self):
        f = torch.ones([3, 3], dtype=torch.float, requires_grad=True, device=self.device)
        ten = torch.tensor([10.], dtype=torch.double, device=self.device)
        tens = f * ten
        s = (tens + 2).sum()
        s.backward()
        self.assertEqual(f.grad, tens)

        # If we don't convert the returned grad_input to the actual input type
        # we get an error like:
        # RuntimeError: Function SubBackward0 returned an invalid gradient at index 0 - expected type \
        # torch.FloatTensor but got torch.DoubleTensor
        f_dtypes = [torch.float, torch.double]
        f_dtypes = f_dtypes if self.device == 'cpu' else f_dtypes + [torch.half]
        i_dtypes = [torch.int, torch.long]
        for func in [torch.add, torch.sub, torch.rsub, torch.mul, torch.div]:
            for dtype1, dtype2 in itertools.product(f_dtypes, f_dtypes + i_dtypes):
                x = torch.ones(10, requires_grad=True, dtype=dtype1, device=self.device)
                y = torch.ones(10, dtype=dtype2, device=self.device)
                func(x, y).sum().backward()

    def _get_test_tensor(self, dtype, remove_zeros=False):
        shape = [20, 20, 20]
        if dtype == torch.bool:
            tensor = torch.randint(0, 2, shape, device=self.device, dtype=dtype)
        elif dtype.is_floating_point:
            # "_th_normal_ not supported on CPUType for Half" so simpler create and convert
            tensor = torch.randn(shape, device=self.device)
            tensor = tensor.to(dtype)
        else:
            tensor = torch.randint(0, 15, shape, device=self.device, dtype=dtype)
        if remove_zeros:
            # ensures no div-by-zero (with care for low precision uint8/half)
            tensor[torch.abs(tensor) < 0.05] = 5
        return tensor

    # verifies that torch.<op>(first, second) is the same as 
    # torch.<op>(first.to(common_dtype), second.to(common_dtype)) in cases where that should hold.
    def test_many_promotions(self):
        # Can also include half on CPU in cases where it will be promoted to a
        # supported dtype
        dtypes1 = torch.testing.get_all_math_dtypes('cuda')
        dtypes2 = torch.testing.get_all_math_dtypes(self.device)
        ops = [torch.add, torch.sub, torch.mul, torch.div, torch.rsub]
        for dt1, dt2 in itertools.product(dtypes1, dtypes2):
            for op, non_contiguous in itertools.product(ops, [True, False]):
                common_dtype = torch.promote_types(dt1, dt2)
                if common_dtype == torch.half and self.device == 'cpu':
                    continue
                if op == torch.sub and common_dtype != torch.bool:
                    # Subtraction, the `-` operator, with a bool tensor is not supported.
                    continue
                first = self._get_test_tensor(dt1)
                second = self._get_test_tensor(dt2, op == torch.div)
                # test ops with non-contiguous tensors
                if non_contiguous:
                    first = first.transpose(0, 2)
                    second = second.transpose(2, 1)
                    self.assertNotEqual(first.stride(), second.stride(), "some non-contiguous issues could be missed if tensors have same strides")

                self.assertEqual(not first.is_contiguous(), non_contiguous)
                self.assertEqual(not second.is_contiguous(), non_contiguous)
                result = op(first, second)
                expected = op(first.to(common_dtype), second.to(common_dtype))
                self.assertEqual(result.dtype, expected.dtype, message='{} with {}, {}'.format(op.__name__, dt1, dt2))
                self.assertEqual(result, expected, message='{} with {}, {}'.format(op.__name__, dt1, dt2))

    def test_non_promoting_ops(self):
        x = torch.ones(4, dtype=torch.double)
        err = 'expected dtype .ouble .*but got dtype .loat'
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.neg(torch.ones(4, dtype=torch.float), out=x))
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.lerp(x, torch.ones(4, dtype=torch.float), 1))

    def test_alpha_mismatch(self):
        x = torch.ones(4, dtype=torch.int)
        err = 'alpha must not be'
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.add(x, x, alpha=1.1))
        x = x.to(torch.bool)
        self.assertRaisesRegex(RuntimeError, err,
                               lambda: torch.add(x, x, alpha=1.1))
        self.assertEqual(x + x, torch.add(x, x, alpha=True))

    def test_booleans(self):
        onedim = torch.tensor([True])

        self.assertEqual(onedim + onedim, onedim)
        self.assertEqual(onedim + True, onedim)
        self.assertEqual(torch.add(True, True), True)
        self.assertEqual(torch.add(False, False), False)
        self.assertEqual(torch.add(False, True), True)

        self.assertRaisesRegex(RuntimeError, "Boolean alpha only supported",
                               lambda: torch.add(1, 1, alpha=True))
        self.assertEqual(torch.add(torch.tensor(True), torch.tensor(True), True), torch.tensor(True))

    def test_create_bool_tensors(self):
        expected = torch.tensor([0], dtype=torch.int64, device=self.device)
        self.assertEqual(torch.arange(False, True, device=self.device), expected)
        self.assertEqual(torch.arange(True, device=self.device), expected)
        expected = torch.tensor([0, 0.5], dtype=torch.get_default_dtype(), device=self.device)
        self.assertEqual(torch.arange(False, True, 0.5, device=self.device), expected)
        expected = torch.ones(0, dtype=torch.int64, device=self.device)
        self.assertEqual(torch.arange(False, False, device=self.device), expected)

        self.assertEqual(torch.linspace(False, True, device=self.device), torch.linspace(0, 1, device=self.device))
        self.assertEqual(torch.logspace(False, True, device=self.device), torch.logspace(0, 1, device=self.device))

        # this seems like odd behavior but ints also create float tensors, numpy doesn't have this function.
        self.assertEqual(torch.scalar_tensor(False, device=self.device), torch.tensor(0., device=self.device))

    def test_result_type(self):
        self.assertEqual(torch.result_type(torch.tensor(1, dtype=torch.int), 1), torch.int)
        self.assertEqual(torch.result_type(1, torch.tensor(1, dtype=torch.int)), torch.int)
        self.assertEqual(torch.result_type(1, 1.), torch.get_default_dtype())
        self.assertEqual(torch.result_type(torch.tensor(1), 1.), torch.get_default_dtype())
        self.assertEqual(torch.result_type(torch.tensor(1, dtype=torch.long), torch.tensor([1, 1], dtype=torch.int)), torch.int)
        self.assertEqual(torch.result_type(torch.tensor([1., 1.], dtype=torch.float), 1.), torch.float)
        self.assertEqual(torch.result_type(torch.tensor(1., dtype=torch.float), torch.tensor(1, dtype=torch.double)), torch.double)

    def test_can_cast(self):
        self.assertTrue(torch.can_cast(torch.double, torch.float))
        self.assertFalse(torch.can_cast(torch.float, torch.int))

    def test_comparison_ops_with_type_promotion(self):
        value_for_type = {
            torch.uint8: (1 << 5),
            torch.int8: (1 << 5),
            torch.int16: (1 << 10),
            torch.int32: (1 << 20),
            torch.int64: (1 << 35),
            torch.float16: (1 << 10),
            torch.float32: (1 << 20),
            torch.float64: (1 << 35)
        }
        comparison_ops = [
            dict(
                name="lt",
                out_op=lambda x, y, d: torch.lt(x, y, out=torch.empty(1, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.lt(x, y),
                compare_op=lambda x, y: x < y,
            ),
            dict(
                name="le",
                out_op=lambda x, y, d: torch.le(x, y, out=torch.empty(1, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.le(x, y),
                compare_op=lambda x, y: x <= y,
            ),
            dict(
                name="gt",
                out_op=lambda x, y, d: torch.gt(x, y, out=torch.empty(1, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.gt(x, y),
                compare_op=lambda x, y: x > y,
            ),
            dict(
                name="ge",
                out_op=lambda x, y, d: torch.ge(x, y, out=torch.empty(1, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.ge(x, y),
                compare_op=lambda x, y: x >= y,
            ),
            dict(
                name="eq",
                out_op=lambda x, y, d: torch.eq(x, y, out=torch.empty(1, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.eq(x, y),
                compare_op=lambda x, y: x == y,
            ),
            dict(
                name="ne",
                out_op=lambda x, y, d: torch.ne(x, y, out=torch.empty(1, dtype=torch.bool, device=d)),
                ret_op=lambda x, y: torch.ne(x, y),
                compare_op=lambda x, y: x != y,
            ),
        ]
        device = self.device
        for op in comparison_ops:
            for dt1 in torch.testing.get_all_math_dtypes(device):
                for dt2 in torch.testing.get_all_math_dtypes(device):
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

    def test_lt_with_type_promotion(self):
        for dt in torch.testing.get_all_math_dtypes(self.device):
            x = torch.tensor([0], dtype=dt, device=self.device)
            expected = torch.tensor([True], dtype=torch.bool, device=self.device)

            actual = x < 0.5
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

            actual = x < torch.tensor(0.5)
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

            x = torch.tensor(0, dtype=dt, device=self.device)
            expected = torch.tensor(True, dtype=torch.bool, device=self.device)
            actual = x < 0.5
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

            actual = x < torch.tensor(0.5)
            self.assertTrue(actual, expected)
            self.assertTrue(actual.dtype == torch.bool)

    def test_promote_types(self):
        self.assertEqual(torch.promote_types(torch.float, torch.int), torch.float)
        self.assertEqual(torch.promote_types(torch.float, torch.double), torch.double)
        self.assertEqual(torch.promote_types(torch.int, torch.uint8), torch.int)

    def test_promote_self(self):
        for dtype in torch.testing.get_all_dtypes():
            self.assertEqual(torch.promote_types(dtype, dtype), dtype)

    def test_indexing(self):
        a = torch.ones(5, 2, dtype=torch.double)
        b = torch.zeros(5, dtype=torch.int)

        # lambda cannot contain assignment
        def f():
            a[:, [1]] = b.unsqueeze(-1)
        # https://github.com/pytorch/pytorch/issues/28010
        self.assertRaisesRegex(RuntimeError, 'expected dtype',
                               lambda: f())

        # https://github.com/pytorch/pytorch/issues/27824
        tmp = torch.ones(9, 9, dtype=torch.float, device=self.device)
        mask = torch.ones(10, 10, dtype=torch.uint8, device=self.device)
        result = tmp + mask[1:, 1:]
        expected = torch.full([9, 9], 2., dtype=torch.float, device=self.device).fill_(2.)
        self.assertEqual(result, expected)

    def test_transpose(self):
        # https://github.com/pytorch/pytorch/issues/28502
        a = torch.tensor([[True, True], [False, True]])
        self.assertEqual(a.t() == 0, a.t() == False)  # noqa: E712

    def get_sparse_tensors(self, dtype1=torch.int, value=5, zeros=True):
        t = torch.full([5, 5], value, dtype=dtype1, device=self.device)
        if zeros:
            t[0, 0] = 0
            t[1, 1] = 0
            t[2, 2] = 0
        s = t.to_sparse()
        return (t, s)

    def _test_sparse_op(self, op_name, dtype1, dtype2):
        # print("testing ", op_name, " with ", dtype1, " and ", dtype2)

        def op(t1, t2):
            return getattr(t1, op_name)(t2)

        inplace = op_name[-1] == '_'
        common_dtype = torch.promote_types(dtype1, dtype2)
        div = op_name.find('div') == 0
        sub = op_name.find('sub') == 0
        mul = op_name.find('mul') == 0
        add = op_name.find('add') == 0
        # Subtraction, the `-` operator, with a bool tensor is not supported.
        sub_bool = sub and (dtype1 == torch.bool or dtype2 == torch.bool)
        # "add_cpu/sub_cpu" not implemented for 'Half'
        cpu = self.device == 'cpu'
        half = common_dtype == torch.half

        # skip non-promoting case
        if dtype1 == dtype2:
            return

        (d1, s1) = self.get_sparse_tensors(dtype1, 5)
        (d2, s2) = self.get_sparse_tensors(dtype2, 6, not div)

        if inplace and not torch.can_cast(common_dtype, dtype1):
            self.assertRaises(RuntimeError, lambda: op(d1, s2))
            self.assertRaises(RuntimeError, lambda: op(s1, s2))
            self.assertRaises(RuntimeError, lambda: op(s1, d2))
            return

        # expected value (using dense op)
        if not sub_bool and not (cpu and half):
            expected = d1
            expected = op(expected, d2)

        # Test op(sparse, sparse)
        if not div and (not mul or common_dtype != torch.bool) and not sub_bool and not (cpu and half):
            if inplace:
                (d1, s1) = self.get_sparse_tensors(dtype1, 5)
                (d2, s2) = self.get_sparse_tensors(dtype2, 6, not div)
            sparse = op(s1, s2)
            if not inplace:
                self.assertEqual(sparse.dtype, expected.dtype)
            if sparse.dtype == torch.half and cpu:
                # only dtype1 is half, can't to_dense on cpu half.
                sparse = sparse.to(torch.float)
            std = sparse.to_dense()
            self.assertEqual(expected, sparse.to_dense())
        else:
            # sparse division only supports division by a scalar
            # "mul_out_sparse" not implemented for 'Bool'
            self.assertRaises(RuntimeError, lambda: op(s1, s2).to_dense())

        # Test op(dense, sparse)
        if (add or sub) and not sub_bool and not (cpu and half):
            if inplace:
                (d1, s1) = self.get_sparse_tensors(dtype1, 5)
                (d2, s2) = self.get_sparse_tensors(dtype2, 6, not div)
            dense_sparse = op(d1, s2)
            self.assertEqual(expected, dense_sparse, "{}\n{}\n{}\n{}".format(d1, s2, expected, dense_sparse))
        else:
            # sparse division only supports division by a scalar
            # mul: Didn't find kernel to dispatch to for operator 'aten::_nnz'
            self.assertRaises(RuntimeError, lambda: op(d1, s2))

        # Test op(sparse, dense)
        # op( sparse, dense) not supported for any ops:
        # add(sparse, dense) is not supported. Use add(dense, sparse) instead.
        # sparse division only supports division by a scalar
        # mul: Didn't find kernel to dispatch to for operator 'aten::_nnz'.
        self.assertRaises(RuntimeError, lambda: op(s1, d2))

        # Test op(sparse, scalar)
        scalar = torch.tensor(2, dtype=dtype2).item()
        if not add and not sub and not (cpu and dtype1 == torch.half):
            if inplace:
                (d1, s1) = self.get_sparse_tensors(dtype1, 5)
                (d2, s2) = self.get_sparse_tensors(dtype2, 6, not div)

            sparse = op(s1, scalar)
            dense_scalar = op(d1, scalar)
            if not inplace:
                self.assertEqual(sparse.dtype, dense_scalar.dtype)
            # not sure why this combination gives:
            # "add_dense_sparse" not implemented for 'Bool'
            if op != torch.mul or common_dtype != torch.bool:
                self.assertEqual(dense_scalar, sparse.to_dense())
            else:
                self.assertRaises(RuntimeError, lambda: sparse.to_dense())
        else:
            # add(sparse, dense) is not supported. Use add(dense, sparse) instead.
            # "div_cpu" not implemented for 'Bool'
            # "mul_cpu" / "div_cpu" not implemented for 'Half'
            self.assertRaises(RuntimeError, lambda: op(s1, scalar))

    def test_sparse_ops(self):
        dtypes = torch.testing.get_all_dtypes()
        ops = ['add', 'sub', 'mul', 'div']
        inplace = [x + '_' for x in ops]
        for dtype1, dtype2 in itertools.product(dtypes, dtypes):
            for op_name in ops + inplace:
                # can't to_sparse in bfloat16
                if dtype1 != torch.bfloat16 and dtype2 != torch.bfloat16:
                    self._test_sparse_op(op_name, dtype1, dtype2)


@unittest.skipIf(not torch.cuda.is_available(), "no cuda")
class TestTypePromotionCuda(TestTypePromotion):
    def setUp(self):
        super(TestTypePromotionCuda, self).setUp()
        self.device = 'cuda'

# ensure type promotion logic properly handles an alternate default dtype.
class TestTypePromotionDefaultDouble(TestTypePromotion):
    def setUp(self):
        super(TestTypePromotionDefaultDouble, self).setUp()
        torch.set_default_dtype(torch.double)


if __name__ == '__main__':
    run_tests()
