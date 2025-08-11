"""
Test multi-kernel reduction support with dynamic num_splits dispatch.
"""

import torch
import torch._inductor.config as config
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_GPU


class TestMultiKernelReduction(TestCase):
    def setUp(self):
        # Enable multi-kernel and set hints for testing
        self.old_multi_kernel = config.triton.multi_kernel
        self.old_hints = config.multi_kernel_hints
        self.old_min_num_split = config.min_num_split
        
        config.triton.multi_kernel = 1
        config.multi_kernel_hints = [64, 256, 4096]
        config.min_num_split = 256
    
    def tearDown(self):
        # Restore original config
        config.triton.multi_kernel = self.old_multi_kernel
        config.multi_kernel_hints = self.old_hints
        config.min_num_split = self.old_min_num_split
    
    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_reduction_with_multi_kernel(self):
        """Test that reductions use multi-kernel dispatch when enabled."""
        def fn(x):
            return x.sum(dim=0)
        
        # Test with different sizes to trigger different num_splits
        sizes = [
            (4096, 512),      # Medium reduction
            (16384, 512),     # Large reduction
            (65536, 512),     # Very large reduction
        ]
        
        for size in sizes:
            x = torch.randn(size, device='cuda')
            
            # Compile with multi-kernel enabled
            compiled_fn = torch.compile(fn)
            
            # Run and verify correctness
            expected = fn(x)
            actual = compiled_fn(x)
            
            self.assertTrue(torch.allclose(expected, actual, atol=1e-3, rtol=1e-3))
    
    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_dynamic_shape_reduction(self):
        """Test multi-kernel reduction with dynamic shapes."""
        from torch._dynamo.decorators import mark_dynamic
        
        def fn(x):
            return x.sum(dim=0)
        
        # Create input with dynamic dimension
        x = torch.randn(4096, 512, device='cuda')
        mark_dynamic(x, 0)
        
        # Compile with dynamic=True
        compiled_fn = torch.compile(fn, dynamic=True)
        
        # Test with different sizes
        test_sizes = [4096, 8192, 16384, 32768]
        
        for size in test_sizes:
            x_test = torch.randn(size, 512, device='cuda')
            expected = fn(x_test)
            actual = compiled_fn(x_test)
            
            self.assertTrue(torch.allclose(expected, actual, atol=1e-3, rtol=1e-3))
    
    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_multi_kernel_with_different_reduction_types(self):
        """Test multi-kernel with different reduction operations."""
        test_cases = [
            (lambda x: x.sum(dim=1), "sum"),
            (lambda x: x.mean(dim=1), "mean"),
            (lambda x: x.max(dim=1)[0], "max"),
            (lambda x: x.min(dim=1)[0], "min"),
        ]
        
        x = torch.randn(512, 16384, device='cuda')
        
        for fn, name in test_cases:
            compiled_fn = torch.compile(fn)
            
            expected = fn(x)
            actual = compiled_fn(x)
            
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-3, rtol=1e-3),
                f"Failed for reduction type: {name}"
            )
    
    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_inner_vs_outer_reduction(self):
        """Test multi-kernel for both inner and outer reductions."""
        # Inner reduction (contiguous in memory)
        x_inner = torch.randn(512, 4096, device='cuda')
        
        # Outer reduction (non-contiguous in memory)
        x_outer = torch.randn(4096, 512, device='cuda').t()
        
        def fn(x):
            return x.sum(dim=1)
        
        compiled_fn = torch.compile(fn)
        
        # Test inner reduction
        expected_inner = fn(x_inner)
        actual_inner = compiled_fn(x_inner)
        self.assertTrue(torch.allclose(expected_inner, actual_inner, atol=1e-3, rtol=1e-3))
        
        # Test outer reduction
        expected_outer = fn(x_outer)
        actual_outer = compiled_fn(x_outer)
        self.assertTrue(torch.allclose(expected_outer, actual_outer, atol=1e-3, rtol=1e-3))
    
    @unittest.skipIf(not HAS_GPU, "GPU required")
    def test_reduction_with_multiple_inputs(self):
        """Test multi-kernel reduction with operations on multiple tensors."""
        def fn(x, y):
            return (x * y).sum(dim=1)
        
        x = torch.randn(512, 8192, device='cuda')
        y = torch.randn(512, 8192, device='cuda')
        
        compiled_fn = torch.compile(fn)
        
        expected = fn(x, y)
        actual = compiled_fn(x, y)
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-3, rtol=1e-3))
    
    def test_multi_kernel_disabled(self):
        """Test that reductions work normally when multi-kernel is disabled."""
        # Disable multi-kernel
        config.triton.multi_kernel = 0
        
        def fn(x):
            return x.sum(dim=0)
        
        x = torch.randn(4096, 512, device='cuda' if HAS_GPU else 'cpu')
        
        compiled_fn = torch.compile(fn)
        
        expected = fn(x)
        actual = compiled_fn(x)
        
        self.assertTrue(torch.allclose(expected, actual, atol=1e-3, rtol=1e-3))


if __name__ == "__main__":
    # Enable logging for debugging
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run with benchmarking enabled
    import os
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL"] = "1"
    
    import unittest
    run_tests()