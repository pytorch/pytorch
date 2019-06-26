import torch
import unittest

from common_utils import TestCase, run_tests, load_tests

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

class TestTypePromotion(TestCase):

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float32)
        self.device = 'cpu'

    # In-place operations don't promote.
    # int+float==float but int.add_(float)==int.
    # Promoting inplace would require re-allocating and copying the memory of the
    # tensor data, since element size could change.
    def test_inplace(self):
        int_tensor = torch.ones([4, 4, 4], dtype=torch.int32, device=self.device)

        self.assertRaisesRegex(RuntimeError, "doesn't match the", lambda: int_tensor.add_(1.5))

        expected = torch.ones([4, 4, 4], device=self.device)
        expected = expected.to(torch.int32)
        self.assertEqual(int_tensor, expected)

        long_tensor = torch.ones([4, 4, 4], dtype=torch.int64, device=self.device)
        self.assertRaisesRegex(RuntimeError, "doesn't match the", lambda: int_tensor.add_(long_tensor))
        # int_tensor.add_(long_tensor)
        # self.assertEqual(int_tensor, expected + 1)
        int_tensor.add_(int_tensor)
        int_tensor.add_(1)
        self.assertEqual(int_tensor, expected + 2)

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
        if(self.device == 'cpu'):
            raise unittest.SkipTest('add_cpu not implemented for Half')
        half = torch.tensor(5.5, dtype=torch.float16, device=self.device)
        self.assertEqual((half + 2.2).dtype, torch.float16)
        self.assertEqual((half + 100000).dtype, torch.float16)  # inf
        default_tensor = torch.tensor(100000.0, device=self.device)
        self.assertEqual((half + default_tensor).dtype, torch.get_default_dtype())

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
            torch.float16: torch.double
        }

        for k, v in from_to.items():
            a = torch.rand([3, 3], device=self.device).to(k)  # not _th_uniform for half on cpu.
            b = torch.rand([3, 3], device=self.device).to(v)
            c = a.add(b)
            d = a.to(v).add(b)
            self.assertEqual(c.dtype, d.dtype, message='from {} to {}'.format(k, v))
            self.assertEqual(c, d, message='from {} to {}'.format(k, v))

@unittest.skipIf(not torch.cuda.is_available(), "no cuda")
class TestTypePromotionCuda(TestTypePromotion):
    def setUp(self):
        super().setUp()
        self.device = 'cuda'

# ensure type promotion logic properly handles an alternate default dtype.
class TestTypePromotionDefaultDouble(TestTypePromotion):
    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.double)


if __name__ == '__main__':
    run_tests()
