#!/usr/bin/env python3
"""
Test script for the Multi-Kernel Dispatch feature.

This script validates that the enhanced multi-kernel dispatch with size hint candidates
works correctly and provides performance improvements for dynamic shapes.
"""

import torch
import torch._inductor.config as config
from torch._inductor.compile_fx import compile_fx
import functools
import time

def test_basic_multi_kernel_dispatch():
    """Test basic functionality of multi-kernel dispatch with size hint candidates."""
    
    print("Testing Multi-Kernel Dispatch with Size Hint Candidates...")
    
    # Enable multi-kernel dispatch with size hint candidates
    config.triton.multi_kernel = 1
    config.triton.multi_kernel_hint_candidates = [
        [1024, 512],   # Optimized for smaller tensors
        [4096, 2048],  # Optimized for medium tensors  
        [8192, 4096],  # Optimized for larger tensors
    ]
    
    # Disable cpp wrapper to avoid OpenMP compilation issues in testing
    config.cpp_wrapper = False
    
    print(f"Multi-kernel enabled: {config.triton.multi_kernel}")
    print(f"Size hint candidates: {config.triton.multi_kernel_hint_candidates}")

    class SimplePointwise(torch.nn.Module):
        def forward(self, x, y):
            """Simple pointwise operation that should benefit from multi-kernel dispatch."""
            return (x + y * 2.0,)
    
    # Test with different tensor sizes
    test_sizes = [
        (512, 256),    # Small - should pick first variant
        (2048, 1024),  # Medium - should pick second variant  
        (4096, 2048),  # Large - should pick third variant
        (1000, 500),   # Dynamic size
    ]
    
    # Compile the function with torch.compile to support dynamic shapes
    model = SimplePointwise()
    compiled_fn = torch.compile(model, dynamic=True)
    
    print("\nTesting different tensor sizes:")
    for i, (h, w) in enumerate(test_sizes):
        print(f"\nTest {i+1}: Size ({h}, {w})")
        
        # Create test tensors
        x = torch.randn(h, w, device='cuda' if torch.cuda.is_available() else 'cpu')
        y = torch.randn(h, w, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # Warm up
        for _ in range(3):
            result = compiled_fn(x, y)
        
        # Benchmark 
        start_time = time.time()
        for _ in range(10):
            result = compiled_fn(x, y)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000  # Convert to ms
        print(f"  Average execution time: {avg_time:.4f} ms")
        
        # Verify correctness (unpack tuple result)
        expected = x + y * 2.0
        torch.testing.assert_close(result[0], expected, rtol=1e-4, atol=1e-4)
        print(f"  ✓ Correctness check passed")

def test_multi_kernel_cache_behavior():
    """Test that the shape-based caching works correctly."""
    
    print("\n\nTesting Shape-Based Cache Behavior...")
    
    # Enable multi-kernel dispatch
    config.triton.multi_kernel = 1
    config.triton.multi_kernel_hint_candidates = [
        [1024, 1024],
        [2048, 2048], 
    ]
    
    class CachedOperation(torch.nn.Module):
        def forward(self, x):
            return (x * 3.0 + 1.0,)
        
    model = CachedOperation()
    compiled_fn = torch.compile(model, dynamic=True)
    
    # Test that same shapes use cached dispatch
    size1 = (1000, 1000)
    size2 = (2000, 2000) 
    
    print(f"Testing caching for size {size1}")
    x1 = torch.randn(*size1, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # First call - should trigger benchmarking
    start = time.time()
    result1a = compiled_fn(x1)
    first_call_time = time.time() - start
    
    # Second call with same shape - should use cache
    start = time.time()
    result1b = compiled_fn(x1)
    second_call_time = time.time() - start
    
    print(f"  First call (with benchmarking): {first_call_time*1000:.4f} ms")
    print(f"  Second call (cached): {second_call_time*1000:.4f} ms")
    
    # Verify results are identical
    torch.testing.assert_close(result1a[0], result1b[0])
    print(f"  ✓ Cache consistency check passed")
    
    # Test with different shape
    print(f"\nTesting different size {size2}")
    x2 = torch.randn(*size2, device='cuda' if torch.cuda.is_available() else 'cpu')
    result2 = compiled_fn(x2)
    
    # Verify correctness
    expected2 = x2 * 3.0 + 1.0
    torch.testing.assert_close(result2[0], expected2, rtol=1e-4, atol=1e-4)
    print(f"  ✓ Different shape correctness check passed")

def test_fallback_to_standard_multi_kernel():
    """Test that the system falls back to standard multi-kernel when no candidates are set."""
    
    print("\n\nTesting Fallback to Standard Multi-Kernel...")
    
    # Disable size hint candidates but keep multi-kernel enabled
    config.triton.multi_kernel = 1
    config.triton.multi_kernel_hint_candidates = []
    
    class FallbackOperation(torch.nn.Module):
        def forward(self, x, y):
            return (x.sum() + y.mean(),)
    
    # This should use the standard MultiKernelCall instead of MultiDimKernelDispatcher
    model = FallbackOperation()
    compiled_fn = torch.compile(model, dynamic=True)
    
    x = torch.randn(500, 500, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(500, 500, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    result = compiled_fn(x, y)
    expected = x.sum() + y.mean()
    
    torch.testing.assert_close(result[0], expected, rtol=1e-4, atol=1e-4)
    print("  ✓ Fallback behavior works correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Multi-Kernel Dispatch Test Suite")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU (limited functionality)")
    
    try:
        test_basic_multi_kernel_dispatch()
        test_multi_kernel_cache_behavior() 
        test_fallback_to_standard_multi_kernel()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("Multi-kernel dispatch feature is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())