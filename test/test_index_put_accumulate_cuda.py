"""
Test reproducing issue #41162: Advanced indexing gradient is extremely slow
when there are many duplicate indices.

The backward pass for advanced indexing with accumulate=true always takes
the slow sort-based path, even when deterministic algorithms are not required.
"""
import unittest
import torch
import time
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_cuda import TEST_CUDA


class TestIndexPutAccumulateCUDA(TestCase):
    """Test cases for CUDA index_put with accumulate=True optimization.

    See https://github.com/pytorch/pytorch/issues/41162
    """

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_index_put_accumulate_correctness(self):
        """Verify correctness of index_put with accumulate=True on CUDA."""
        torch.use_deterministic_algorithms(False)

        for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                # Test with duplicate indices
                torch.manual_seed(42)
                target_cpu = torch.zeros(100, dtype=dtype)
                target_cuda = torch.zeros(100, dtype=dtype, device='cuda')

                indices = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
                values = torch.randn(9, dtype=dtype)

                target_cpu.index_put_((indices,), values, accumulate=True)
                target_cuda.index_put_((indices.cuda(),), values.cuda(), accumulate=True)

                # Check results match CPU
                rtol, atol = (1e-2, 1e-2) if dtype in [torch.float16, torch.bfloat16] else (1e-5, 1e-5)
                self.assertTrue(
                    torch.allclose(target_cpu, target_cuda.cpu(), rtol=rtol, atol=atol),
                    f"Mismatch for dtype {dtype}"
                )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_index_put_accumulate_multidim(self):
        """Test index_put with accumulate on multidimensional tensors."""
        torch.use_deterministic_algorithms(False)

        target = torch.zeros(10, 20, device='cuda')
        indices = (torch.tensor([0, 1, 0, 1], device='cuda'),
                   torch.tensor([0, 0, 0, 0], device='cuda'))  # duplicates at (0,0)
        values = torch.ones(4, device='cuda')

        target.index_put_(indices, values, accumulate=True)

        # Position (0,0) should have value 2 (two ones accumulated)
        # Position (1,0) should have value 2 (two ones accumulated)
        self.assertEqual(target[0, 0].item(), 2.0)
        self.assertEqual(target[1, 0].item(), 2.0)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_index_put_deterministic_mode(self):
        """Verify deterministic mode still uses the sort-based path."""
        # In deterministic mode, index_put with accumulate should not
        # raise an error and should produce correct results
        torch.use_deterministic_algorithms(True)

        try:
            target = torch.zeros(100, device='cuda')
            indices = torch.tensor([0, 1, 0, 1], device='cuda')
            values = torch.ones(4, device='cuda')

            target.index_put_((indices,), values, accumulate=True)

            self.assertEqual(target[0].item(), 2.0)
            self.assertEqual(target[1].item(), 2.0)
        finally:
            torch.use_deterministic_algorithms(False)

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_index_put_accumulate_large_duplicates(self):
        """Test with many duplicate indices (embedding gradient scenario)."""
        torch.use_deterministic_algorithms(False)

        VOCAB_SIZE = 10000
        NUM_INDICES = 100000

        # Many duplicates - only 100 unique indices
        indices = torch.randint(0, 100, (NUM_INDICES,), device='cuda')
        values = torch.randn(NUM_INDICES, device='cuda')
        target = torch.zeros(VOCAB_SIZE, device='cuda')

        # This should use the fast atomic path
        target.index_put_((indices,), values, accumulate=True)

        # Verify by computing expected result on CPU
        target_cpu = torch.zeros(VOCAB_SIZE)
        indices_cpu = indices.cpu()
        values_cpu = values.cpu()
        target_cpu.index_put_((indices_cpu,), values_cpu, accumulate=True)

        self.assertTrue(torch.allclose(target.cpu(), target_cpu, atol=1e-4))


def benchmark():
    """Run benchmarks to verify performance improvement."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    print("\n" + "=" * 60)
    print("Benchmarking issue #41162 fix")
    print("=" * 60)

    torch.use_deterministic_algorithms(False)

    # Parameters mimicking embedding gradient accumulation
    BATCH_SIZE = 1024
    EMBEDDING_DIM = 768
    VOCAB_SIZE = 50000

    # Indices with many duplicates
    indices = torch.randint(0, 100, (BATCH_SIZE, EMBEDDING_DIM), device='cuda')
    grad = torch.randn(BATCH_SIZE, EMBEDDING_DIM, device='cuda')
    target = torch.zeros(VOCAB_SIZE, device='cuda')

    # Warmup
    for _ in range(5):
        target.zero_()
        target.index_put_((indices.flatten(),), grad.flatten(), accumulate=True)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    num_iters = 100
    for _ in range(num_iters):
        target.zero_()
        target.index_put_((indices.flatten(),), grad.flatten(), accumulate=True)
    torch.cuda.synchronize()
    accum_time = (time.time() - start) / num_iters * 1000

    print(f"index_put (accumulate=True): {accum_time:.3f} ms")

    # Compare with scatter_add as reference
    indices_flat = indices.flatten()
    grad_flat = grad.flatten()
    target_scatter = torch.zeros(VOCAB_SIZE, device='cuda')

    for _ in range(5):
        target_scatter.zero_()
        target_scatter.scatter_add_(0, indices_flat, grad_flat)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        target_scatter.zero_()
        target_scatter.scatter_add_(0, indices_flat, grad_flat)
    torch.cuda.synchronize()
    scatter_time = (time.time() - start) / num_iters * 1000

    print(f"scatter_add (reference):      {scatter_time:.3f} ms")
    print(f"Ratio (should be ~1x):        {accum_time / scatter_time:.1f}x")


if __name__ == "__main__":
    run_tests()
    benchmark()

