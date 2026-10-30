# Owner(s): ["module: signal"]

import unittest
import torch
import torch.nn.functional as F
from torch.signal import sliding_window_max, sliding_window_min

try:
    from torch.testing._internal.common_utils import TestCase, run_tests
except ImportError:
    TestCase = unittest.TestCase
    def run_tests():
        unittest.main()


class TestSlidingWindow(TestCase):
    def test_sliding_window_max_1d(self):
        x = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
        # window size 3, stride 1:
        # [1, 5, 2] -> 5
        # [5, 2, 8] -> 8
        # [2, 8, 3] -> 8
        # [8, 3, 7] -> 8
        out = sliding_window_max(x, kernel_size=3)
        expected = torch.tensor([5.0, 8.0, 8.0, 8.0])
        torch.testing.assert_close(out, expected)

    def test_sliding_window_min_1d(self):
        x = torch.tensor([1.0, 5.0, 2.0, 8.0, 3.0, 7.0])
        # window size 3, stride 1:
        # [1, 5, 2] -> 1
        # [5, 2, 8] -> 2
        # [2, 8, 3] -> 2
        # [8, 3, 7] -> 3
        out = sliding_window_min(x, kernel_size=3)
        expected = torch.tensor([1.0, 2.0, 2.0, 3.0])
        torch.testing.assert_close(out, expected)

    def test_sliding_window_strides(self):
        x = torch.arange(10, dtype=torch.float32)
        out = sliding_window_max(x, kernel_size=3, stride=2)
        # [0, 1, 2] -> 2
        # [2, 3, 4] -> 4
        # [4, 5, 6] -> 6
        # [6, 7, 8] -> 8
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0])
        torch.testing.assert_close(out, expected)

    def test_sliding_window_large_padding(self):
        # max_pool1d normally rejects padding > kernel_size // 2;
        # sliding_window_max should handle arbitrary padding seamlessly.
        x = torch.tensor([2.0, 4.0, 1.0])
        out = sliding_window_max(x, kernel_size=3, padding=2)
        # Padded with -inf: [-inf, -inf, 2, 4, 1, -inf, -inf]
        # k=3:
        # [-inf, -inf, 2] -> 2
        # [-inf, 2, 4] -> 4
        # [2, 4, 1] -> 4
        # [4, 1, -inf] -> 4
        # [1, -inf, -inf] -> 1
        expected = torch.tensor([2.0, 4.0, 4.0, 4.0, 1.0])
        torch.testing.assert_close(out, expected)

    def test_sliding_window_nd(self):
        # Check that arbitrary leading batch/channel dimensions are preserved
        x = torch.randn(2, 3, 4, 20)
        out_max = sliding_window_max(x, kernel_size=5, stride=2)
        self.assertEqual(out_max.shape, (2, 3, 4, 8))

        out_min = sliding_window_min(x, kernel_size=5, stride=2)
        self.assertEqual(out_min.shape, (2, 3, 4, 8))

    def test_sliding_window_return_indices(self):
        x = torch.tensor([10.0, 2.0, 50.0, 4.0, 1.0])
        vals, indices = sliding_window_max(x, kernel_size=3, return_indices=True)
        # [10, 2, 50] -> val 50, idx 2
        # [2, 50, 4]  -> val 50, idx 2
        # [50, 4, 1]  -> val 50, idx 2
        torch.testing.assert_close(vals, torch.tensor([50.0, 50.0, 50.0]))
        torch.testing.assert_close(indices, torch.tensor([2, 2, 2]))

        min_vals, min_indices = sliding_window_min(x, kernel_size=3, return_indices=True)
        # [10, 2, 50] -> val 2, idx 1
        # [2, 50, 4]  -> val 2, idx 1
        # [50, 4, 1]  -> val 1, idx 4
        torch.testing.assert_close(min_vals, torch.tensor([2.0, 2.0, 1.0]))
        torch.testing.assert_close(min_indices, torch.tensor([1, 1, 4]))

    def test_sliding_window_autograd(self):
        x = torch.randn(2, 10, requires_grad=True)
        out = sliding_window_max(x, kernel_size=3)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        # Total scattered gradient should equal total elements in out
        self.assertAlmostEqual(x.grad.sum().item(), float(out.numel()), places=5)

    def test_invalid_parameters(self):
        x = torch.randn(10)
        with self.assertRaises(ValueError):
            sliding_window_max(x, kernel_size=0)
        with self.assertRaises(ValueError):
            sliding_window_max(x, kernel_size=3, stride=0)
        with self.assertRaises(ValueError):
            sliding_window_max(x, kernel_size=3, padding=-1)
        with self.assertRaises(ValueError):
            sliding_window_max(torch.tensor(1.0), kernel_size=1)


if __name__ == "__main__":
    run_tests()
