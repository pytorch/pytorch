"""Tests for fast normalization implementations."""

import unittest
import torch
import torch.nn as nn
from torch.nn.modules.fast_normalization import (
    FastRMSNorm, enable_fast_rmsnorm, disable_fast_rmsnorm,
    is_fast_rmsnorm_enabled, TRITON_AVAILABLE
)


class TestFastRMSNorm(unittest.TestCase):
    """Test FastRMSNorm implementation."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not TRITON_AVAILABLE, "Triton not available")
    def test_correctness(self):
        """Test that FastRMSNorm produces correct results."""
        torch.manual_seed(42)
        
        for shape in [(768,), (1024,), (2048,)]:
            for batch_shape in [(32, 512), (16, 1024), (8, 2048)]:
                with self.subTest(shape=shape, batch_shape=batch_shape):
                    # Create input
                    x = torch.randn(*batch_shape, *shape).cuda()
                    
                    # Standard RMSNorm
                    standard = nn.RMSNorm(shape, eps=1e-5).cuda()
                    
                    # Fast RMSNorm
                    fast = FastRMSNorm(shape, eps=1e-5).cuda()
                    fast.weight.data = standard.weight.data
                    
                    # Compare outputs
                    y_standard = standard(x)
                    y_fast = fast(x)
                    
                    self.assertTrue(
                        torch.allclose(y_standard, y_fast, rtol=1e-4, atol=1e-6),
                        f"Outputs don't match for shape {shape}, batch {batch_shape}"
                    )
    
    def test_cpu_fallback(self):
        """Test that CPU fallback works correctly."""
        x = torch.randn(32, 512, 768)
        
        fast = FastRMSNorm(768)
        standard = nn.RMSNorm(768)
        fast.weight.data = standard.weight.data
        
        y_fast = fast(x)
        y_standard = standard(x)
        
        self.assertTrue(torch.allclose(y_fast, y_standard, rtol=1e-4))
    
    def test_global_enable_disable(self):
        """Test global enable/disable functionality."""
        # Check initial state
        original_state = is_fast_rmsnorm_enabled()
        
        try:
            # Enable fast RMSNorm
            enable_fast_rmsnorm()
            self.assertTrue(is_fast_rmsnorm_enabled())
            
            # Create a new RMSNorm - should be FastRMSNorm
            if TRITON_AVAILABLE:
                norm = nn.RMSNorm(768)
                self.assertIsInstance(norm, FastRMSNorm)
            
            # Disable
            disable_fast_rmsnorm()
            self.assertFalse(is_fast_rmsnorm_enabled())
            
            # Create a new RMSNorm - should be standard
            norm = nn.RMSNorm(768)
            self.assertNotIsInstance(norm, FastRMSNorm)
            
        finally:
            # Restore original state
            if original_state:
                enable_fast_rmsnorm()
            else:
                disable_fast_rmsnorm()
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not TRITON_AVAILABLE, "Triton not available")
    def test_performance(self):
        """Basic performance test to ensure fast version is actually faster."""
        import time
        
        x = torch.randn(32, 512, 768).cuda()
        
        # Standard
        standard = nn.RMSNorm(768).cuda()
        
        # Fast
        fast = FastRMSNorm(768).cuda()
        fast.weight.data = standard.weight.data
        
        # Warmup
        for _ in range(10):
            _ = standard(x)
            _ = fast(x)
        
        torch.cuda.synchronize()
        
        # Time standard
        start = time.time()
        for _ in range(100):
            _ = standard(x)
        torch.cuda.synchronize()
        standard_time = time.time() - start
        
        # Time fast
        start = time.time()
        for _ in range(100):
            _ = fast(x)
        torch.cuda.synchronize()
        fast_time = time.time() - start
        
        # Fast should be at least 1.5x faster
        speedup = standard_time / fast_time
        self.assertGreater(
            speedup, 1.5,
            f"Fast RMSNorm not fast enough: {speedup:.2f}x (expected >1.5x)"
        )
        
        print(f"\nPerformance: FastRMSNorm is {speedup:.2f}x faster")


if __name__ == '__main__':
    unittest.main()
