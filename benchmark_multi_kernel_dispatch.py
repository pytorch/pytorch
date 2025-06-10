#!/usr/bin/env python3
"""
Benchmark script for Multi-Kernel Dispatch performance comparison.

This script compares the performance of standard compilation vs multi-kernel dispatch
with size hint candidates for varying dynamic shapes.
"""

import torch
import torch._inductor.config as config
import time
import statistics

def benchmark_kernel_dispatch():
    """Benchmark multi-kernel dispatch vs standard compilation."""
    
    print("=" * 70)
    print("Multi-Kernel Dispatch Performance Benchmark")
    print("=" * 70)
    
    # Test different tensor sizes that would benefit from different optimizations
    test_shapes = [
        (512, 256),     # Small tensors
        (1024, 512),    # Medium tensors 
        (2048, 1024),   # Large tensors
        (4096, 2048),   # Very large tensors
        (256, 512),     # Different aspect ratio
        (1500, 750),    # Non-power-of-2 sizes
    ]
    
    class BenchmarkOperation(torch.nn.Module):
        def forward(self, x, y):
            # Complex pointwise operation that benefits from optimization
            return (x * 2.0 + y.sin() - x.exp().clamp(max=10.0),)
    
    # Disable cpp wrapper to avoid compilation issues in benchmarking
    config.cpp_wrapper = False
    
    print("\n1. Testing Standard Multi-Kernel (no size hint candidates)")
    print("-" * 50)
    
    # Test without size hint candidates (standard multi-kernel)
    config.triton.multi_kernel = 1  
    config.triton.multi_kernel_hint_candidates = []
    
    model_standard = BenchmarkOperation()
    compiled_standard = torch.compile(model_standard, dynamic=True)
    
    standard_times = {}
    for shape in test_shapes:
        h, w = shape
        x = torch.randn(h, w)
        y = torch.randn(h, w)
        
        # Warmup
        for _ in range(3):
            _ = compiled_standard(x, y)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            result = compiled_standard(x, y)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        standard_times[shape] = avg_time
        
        print(f"  Shape {shape}: {avg_time:.3f} ± {std_time:.3f} ms")
    
    print("\n2. Testing Enhanced Multi-Kernel Dispatch (with size hint candidates)")
    print("-" * 50)
    
    # Test with size hint candidates (enhanced multi-kernel dispatch)
    config.triton.multi_kernel = 1
    config.triton.multi_kernel_hint_candidates = [
        [512, 256],     # Optimized for small tensors
        [1024, 512],    # Optimized for medium tensors
        [2048, 1024],   # Optimized for large tensors  
        [4096, 2048],   # Optimized for very large tensors
    ]
    
    model_enhanced = BenchmarkOperation()
    compiled_enhanced = torch.compile(model_enhanced, dynamic=True)
    
    enhanced_times = {}
    for shape in test_shapes:
        h, w = shape
        x = torch.randn(h, w)
        y = torch.randn(h, w)
        
        # Warmup
        for _ in range(3):
            _ = compiled_enhanced(x, y)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            result = compiled_enhanced(x, y)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        enhanced_times[shape] = avg_time
        
        print(f"  Shape {shape}: {avg_time:.3f} ± {std_time:.3f} ms")
    
    print("\n3. Performance Comparison")
    print("-" * 50)
    
    total_standard = 0
    total_enhanced = 0
    improvements = []
    
    for shape in test_shapes:
        standard_time = standard_times[shape]
        enhanced_time = enhanced_times[shape]
        
        total_standard += standard_time
        total_enhanced += enhanced_time
        
        if standard_time > 0:
            speedup = standard_time / enhanced_time
            improvement = ((standard_time - enhanced_time) / standard_time) * 100
        else:
            speedup = 1.0
            improvement = 0.0
            
        improvements.append(improvement)
        
        print(f"  Shape {shape}:")
        print(f"    Standard: {standard_time:.3f} ms")
        print(f"    Enhanced: {enhanced_time:.3f} ms") 
        print(f"    Speedup: {speedup:.2f}x ({improvement:+.1f}%)")
        print()
    
    avg_improvement = statistics.mean(improvements)
    total_speedup = total_standard / total_enhanced if total_enhanced > 0 else 1.0
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Average performance improvement: {avg_improvement:+.1f}%")
    print(f"Overall speedup: {total_speedup:.2f}x")
    print(f"Total time standard: {total_standard:.1f} ms")
    print(f"Total time enhanced: {total_enhanced:.1f} ms")
    
    if avg_improvement > 0:
        print("✅ Enhanced multi-kernel dispatch shows performance improvement!")
    else:
        print("ℹ️  Performance similar - may need more complex workloads to see benefits")
    
    print("\nNote: Performance improvements are most visible with:")
    print("- GPU execution (CUDA available)")
    print("- Complex kernels with multiple optimization opportunities") 
    print("- Workloads with highly varying input shapes")

def main():
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU with limited performance visibility.")
        print("For best results, run this benchmark on a system with CUDA support.\n")
    
    try:
        benchmark_kernel_dispatch()
        return 0
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())