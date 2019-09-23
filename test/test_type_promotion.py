import torch
import unittest

from common_utils import TestCase, run_tests, load_tests

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
        for f in ['add', 'sub', 'rsub', 'mul', 'div']:
            for dtype1 in f_dtypes:
                for dtype2 in (f_dtypes + i_dtypes):
                    x = torch.ones(10, requires_grad=True, dtype=dtype1, device=self.device)
                    y = torch.ones(10, dtype=dtype2, device=self.device)

                    func = getattr(torch, f)
                    func(x, y).sum().backward()

    # verifies that a.add(b) is the same as a.to(b.dtype).add(b) in cases
    # where that should hold.
    def test_many_promotions(self):
        from_to = {
            torch.float16: torch.float32,
            torch.half: torch.float16,
            torch.int: torch.long,
            torch.uint8: torch.long,
            torch.uint8: torch.float,
            torch.int: torch.float,
            torch.int: torch.double,
            torch.int16: torch.long,
            torch.float16: torch.double,
            torch.bool: torch.long,
            torch.bool: torch.float
        }

        for k, v in from_to.items():
            a = torch.rand([3, 3], device=self.device).to(k)  # no _th_uniform for half on cpu.
            b = torch.rand([3, 3], device=self.device).to(v)
            c = a.add(b)
            d = a.to(v).add(b)
            self.assertEqual(c.dtype, d.dtype, message='from {} to {}'.format(k, v))
            self.assertEqual(c, d, message='from {} to {}'.format(k, v))

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

    def test_promote_types(self):
        self.assertEqual(torch.promote_types(torch.float, torch.int), torch.float)
        self.assertEqual(torch.promote_types(torch.float, torch.double), torch.double)
        self.assertEqual(torch.promote_types(torch.int, torch.uint8), torch.int)

    def test_sparse(self):
        t = torch.full([5, 5], 5, dtype=torch.int, device=self.device)
        t[0, 0] = 0
        t[1, 1] = 0
        t[2, 2] = 0
        s = t.to_sparse()
        self.assertEqual((s + s).to_dense(), t + t)
        self.assertEqual(t + s, (s + s).to_dense())
        self.assertEqual(t + s.to(torch.double), (t + t))
        self.assertEqual((t + s.to(torch.double)).dtype, torch.double)
        self.assertEqual((s + s.to(torch.double)).to_dense(), t + t)
        self.assertEqual((s + s.to(torch.double)).dtype, torch.double)

        inplace = s.to(torch.float, copy=True)
        torch.add(inplace, s.to(torch.double), out=inplace)
        self.assertEqual(inplace.to_dense(), t + t)
        self.assertEqual(inplace.dtype, torch.float)

        zeroDim = torch.tensor(5, device=self.device).to_sparse()
        self.assertEqual((zeroDim - zeroDim).to_dense(), torch.tensor(0, device=self.device))
        self.assertEqual((s - s).to_dense(), t - t)
        self.assertEqual((s - s).dtype, s.dtype)
        self.assertEqual((s - s.to(torch.float)).to_dense(), t - t.to(torch.float))
        self.assertEqual((s - s.to(torch.float)).dtype, torch.float)

        self.assertEqual((s * s).to_dense(), t * t)
        self.assertEqual(s * s.to(torch.double), (t * t).to_sparse())
        self.assertEqual((s * s.to(torch.double)).dtype, torch.double)

        inplace = s.clone()
        uint = s.to(torch.uint8)
        self.assertRaisesRegex(RuntimeError, "Can't convert result type", 
                               lambda: inplace.mul_(s.to(torch.double)))
        inplace *= uint
        self.assertEqual(inplace.to_dense(), t * t)
        uint *= s
        self.assertEqual(uint.to_dense(), t * t)
        self.assertEqual(uint.dtype, torch.uint8)

        o = t.to_sparse().to(torch.float)
        x = torch.mul(s, s.to(torch.double), out=o)
        self.assertEqual(o.to_dense(), t * t)
        self.assertEqual(o.dtype, torch.float)

        self.assertEqual((s * 2).to_dense(), t * 2)
        self.assertEqual((s * 2).to_dense().dtype, (t * 2).dtype)
        self.assertEqual((s * 2.0).to_dense().dtype, (t * 2.0).dtype)

        # sparse division only supports division by a scalar
        self.assertEqual((s / 2).to_dense(), t / 2)
        self.assertEqual((s / 2).to_dense().dtype, (t / 2).dtype)
        self.assertEqual((s / 2.0).to_dense().dtype, (t / 2.0).dtype)

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
