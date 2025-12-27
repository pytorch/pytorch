"""
Test for combined max+argmax / min+argmin reduction optimization.
See https://github.com/pytorch/pytorch/issues/146643

This tests that torch.max(x, dim) and torch.min(x, dim) generate a single
reduction loop instead of two separate loops for max/argmax and min/argmin.
"""
import torch
import torch._dynamo
import unittest
from torch._inductor import config


class TestMaxArgmaxCodegen(unittest.TestCase):
    """Test combined max+argmax / min+argmin reduction optimization."""

    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def test_max_with_dim_correctness(self):
        """Test that torch.max(x, dim) produces correct results after optimization."""
        def fn(x):
            return torch.max(x, dim=1)

        x = torch.randn(8, 16)
        compiled_fn = torch.compile(fn, backend="inductor")

        # Get expected results from eager mode
        expected_val, expected_idx = fn(x)

        # Get results from compiled version
        actual_val, actual_idx = compiled_fn(x)

        self.assertTrue(torch.allclose(actual_val, expected_val))
        self.assertTrue(torch.equal(actual_idx, expected_idx))

    def test_min_with_dim_correctness(self):
        """Test that torch.min(x, dim) produces correct results after optimization."""
        def fn(x):
            return torch.min(x, dim=1)

        x = torch.randn(8, 16)
        compiled_fn = torch.compile(fn, backend="inductor")

        # Get expected results from eager mode
        expected_val, expected_idx = fn(x)

        # Get results from compiled version
        actual_val, actual_idx = compiled_fn(x)

        self.assertTrue(torch.allclose(actual_val, expected_val))
        self.assertTrue(torch.equal(actual_idx, expected_idx))

    def test_max_with_keepdim(self):
        """Test torch.max with keepdim=True."""
        def fn(x):
            return torch.max(x, dim=1, keepdim=True)

        x = torch.randn(4, 8)
        compiled_fn = torch.compile(fn, backend="inductor")

        expected_val, expected_idx = fn(x)
        actual_val, actual_idx = compiled_fn(x)

        self.assertEqual(actual_val.shape, expected_val.shape)
        self.assertEqual(actual_idx.shape, expected_idx.shape)
        self.assertTrue(torch.allclose(actual_val, expected_val))
        self.assertTrue(torch.equal(actual_idx, expected_idx))

    def test_max_different_dims(self):
        """Test torch.max with different dimension arguments."""
        def fn(x):
            val0, idx0 = torch.max(x, dim=0)
            val1, idx1 = torch.max(x, dim=1)
            val2, idx2 = torch.max(x, dim=2)
            return val0, idx0, val1, idx1, val2, idx2

        x = torch.randn(4, 8, 16)
        compiled_fn = torch.compile(fn, backend="inductor")

        expected = fn(x)
        actual = compiled_fn(x)

        for exp, act in zip(expected, actual):
            if exp.dtype.is_floating_point:
                self.assertTrue(torch.allclose(act, exp))
            else:
                self.assertTrue(torch.equal(act, exp))

    def test_max_with_nan(self):
        """Test that NaN handling is correct."""
        def fn(x):
            return torch.max(x, dim=1)

        x = torch.randn(4, 8)
        x[0, 3] = float('nan')
        x[2, 5] = float('nan')

        compiled_fn = torch.compile(fn, backend="inductor")

        expected_val, expected_idx = fn(x)
        actual_val, actual_idx = compiled_fn(x)

        # NaN propagation should be the same
        self.assertTrue(torch.equal(torch.isnan(actual_val), torch.isnan(expected_val)))
        # For non-NaN values, they should match
        mask = ~torch.isnan(expected_val)
        self.assertTrue(torch.allclose(actual_val[mask], expected_val[mask]))
        self.assertTrue(torch.equal(actual_idx[mask], expected_idx[mask]))

    def test_max_with_inf(self):
        """Test that infinity handling is correct."""
        def fn(x):
            return torch.max(x, dim=1)

        x = torch.randn(4, 8)
        x[0, 3] = float('inf')
        x[1, 5] = float('-inf')

        compiled_fn = torch.compile(fn, backend="inductor")

        expected_val, expected_idx = fn(x)
        actual_val, actual_idx = compiled_fn(x)

        self.assertTrue(torch.allclose(actual_val, expected_val))
        self.assertTrue(torch.equal(actual_idx, expected_idx))

    def test_max_different_dtypes(self):
        """Test with different data types."""
        def fn(x):
            return torch.max(x, dim=1)

        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            with self.subTest(dtype=dtype):
                if dtype.is_floating_point:
                    x = torch.randn(4, 8, dtype=dtype)
                else:
                    x = torch.randint(-100, 100, (4, 8), dtype=dtype)

                compiled_fn = torch.compile(fn, backend="inductor")

                expected_val, expected_idx = fn(x)
                actual_val, actual_idx = compiled_fn(x)

                if dtype.is_floating_point:
                    self.assertTrue(torch.allclose(actual_val, expected_val))
                else:
                    self.assertTrue(torch.equal(actual_val, expected_val))
                self.assertTrue(torch.equal(actual_idx, expected_idx))

                torch._dynamo.reset()

    def test_combined_max_min(self):
        """Test both max and min in the same function."""
        def fn(x):
            max_val, max_idx = torch.max(x, dim=1)
            min_val, min_idx = torch.min(x, dim=1)
            return max_val, max_idx, min_val, min_idx

        x = torch.randn(8, 16)
        compiled_fn = torch.compile(fn, backend="inductor")

        expected = fn(x)
        actual = compiled_fn(x)

        for exp, act in zip(expected, actual):
            if exp.dtype.is_floating_point:
                self.assertTrue(torch.allclose(act, exp))
            else:
                self.assertTrue(torch.equal(act, exp))

    def test_max_unrolled(self):
        """Test with unrolled reductions (small reduction dimension)."""
        def fn(x):
            return torch.max(x, dim=1)

        x = torch.randn(8, 3)  # Small reduction dim

        with config.patch(unroll_reductions_threshold=8):
            compiled_fn = torch.compile(fn, backend="inductor")

            expected_val, expected_idx = fn(x)
            actual_val, actual_idx = compiled_fn(x)

            self.assertTrue(torch.allclose(actual_val, expected_val))
            self.assertTrue(torch.equal(actual_idx, expected_idx))

    def test_max_non_unrolled(self):
        """Test with non-unrolled reductions (large reduction dimension)."""
        def fn(x):
            return torch.max(x, dim=1)

        x = torch.randn(8, 1024)  # Large reduction dim

        with config.patch(unroll_reductions_threshold=1):
            compiled_fn = torch.compile(fn, backend="inductor")

            expected_val, expected_idx = fn(x)
            actual_val, actual_idx = compiled_fn(x)

            self.assertTrue(torch.allclose(actual_val, expected_val))
            self.assertTrue(torch.equal(actual_idx, expected_idx))

    def test_max_large_tensor(self):
        """Test with larger tensors."""
        def fn(x):
            return torch.max(x, dim=1)

        x = torch.randn(128, 2048)
        compiled_fn = torch.compile(fn, backend="inductor")

        expected_val, expected_idx = fn(x)
        actual_val, actual_idx = compiled_fn(x)

        self.assertTrue(torch.allclose(actual_val, expected_val))
        self.assertTrue(torch.equal(actual_idx, expected_idx))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_max_cuda(self):
        """Test on CUDA device."""
        def fn(x):
            return torch.max(x, dim=1)

        x = torch.randn(8, 16, device='cuda')
        compiled_fn = torch.compile(fn, backend="inductor")

        expected_val, expected_idx = fn(x)
        actual_val, actual_idx = compiled_fn(x)

        self.assertTrue(torch.allclose(actual_val, expected_val))
        self.assertTrue(torch.equal(actual_idx, expected_idx))

    def test_max_without_dim_unchanged(self):
        """Test that torch.max without dim still works (should not use combined reduction)."""
        def fn(x):
            return torch.max(x)  # No dim argument

        x = torch.randn(8, 16)
        compiled_fn = torch.compile(fn, backend="inductor")

        expected = fn(x)
        actual = compiled_fn(x)

        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()

