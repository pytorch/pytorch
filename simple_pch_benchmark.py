#!/usr/bin/env python3
"""
Simple PCH benchmark: Compile the same kernel many times with/without PCH.

Usage:
    python benchmarks/simple_pch_benchmark.py --pch         # Test with PCH
    python benchmarks/simple_pch_benchmark.py --no-pch     # Test without PCH
    python benchmarks/simple_pch_benchmark.py              # Test both
"""

import argparse
import os
import sys
import time
from statistics import mean, stdev

import torch
from torch.cuda._utils import _nvrtc_compile


def check_cuda_pch_support():
    """Check if CUDA version supports PCH (12.8+)"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    cuda_version = torch.version.cuda
    if cuda_version:
        major, minor = cuda_version.split('.')[:2]
        version = float(f"{major}.{minor}")
        if version >= 12.8:
            return True, f"CUDA {cuda_version}"
        else:
            return False, f"CUDA {cuda_version} (PCH requires 12.8+)"
    return False, "Unknown CUDA version"


def benchmark_compilation(use_pch, iterations=100):
    """Compile the same kernel many times with or without PCH."""
    
    # CUB kernel that benefits from PCH
    kernel_source = """
    #include <cub/block/block_reduce.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/warp/warp_reduce.cuh>
    
    extern "C"
    __global__ void test_kernel(const float* input, float* output, int n) {
        using BlockReduce = cub::BlockReduce<float, 256>;
        using BlockScan = cub::BlockScan<float, 256>;
        using WarpReduce = cub::WarpReduce<float>;
        
        __shared__ union {
            typename BlockReduce::TempStorage reduce;
            typename BlockScan::TempStorage scan;
            typename WarpReduce::TempStorage warp[8];
        } temp_storage;
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float val = (idx < n) ? input[idx] : 0.0f;
        
        float sum = BlockReduce(temp_storage.reduce).Sum(val);
        __syncthreads();
        
        float scan_result;
        BlockScan(temp_storage.scan).ExclusiveSum(val, scan_result);
        __syncthreads();
        
        int warp_id = threadIdx.x / 32;
        float warp_sum = WarpReduce(temp_storage.warp[warp_id]).Sum(val);
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = sum + scan_result + warp_sum;
        }
    }
    """
    
    # Setup
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    compute_capability = f"{major}{minor}"
    
    CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_include_dirs = []
    if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, "include")):
        cuda_include_dirs.append(os.path.join(CUDA_HOME, "include"))
    
    compile_times = []
    
    print(f"Compiling kernel {iterations} times {'WITH' if use_pch else 'WITHOUT'} PCH...")
    
    for i in range(iterations):
        # Use unique kernel name to avoid caching between iterations
        kernel_name = f"test_kernel_{i}"
        unique_source = kernel_source.replace("test_kernel", kernel_name)
        
        start = time.perf_counter()
        
        try:
            ptx, mangled_name = _nvrtc_compile(
                unique_source,
                kernel_name,
                compute_capability,
                header_code="",
                cuda_include_dirs=cuda_include_dirs,
                nvcc_options=["-std=c++17"],
                enable_automatic_pch=use_pch
            )
            
            elapsed = time.perf_counter() - start
            compile_times.append(elapsed * 1000)  # Convert to ms
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{iterations} compilations")
                
        except RuntimeError as e:
            print(f"Compilation {i} failed: {e}")
            return None
    
    return compile_times


def main():
    parser = argparse.ArgumentParser(description="Simple PCH Compilation Benchmark")
    parser.add_argument("--pch", action="store_true", help="Test with PCH only")
    parser.add_argument("--no-pch", action="store_true", help="Test without PCH only")
    parser.add_argument("--iterations", type=int, default=100, help="Number of compilations")
    args = parser.parse_args()
    
    # Check environment
    pch_supported, cuda_info = check_cuda_pch_support()
    
    print("="*60)
    print("Simple PCH Compilation Benchmark")
    print("="*60)
    print(f"CUDA Version: {cuda_info}")
    print(f"PCH Support: {'Yes' if pch_supported else 'No'}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Iterations: {args.iterations}")
    print()
    
    # Check for CUB headers
    CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.path.exists(os.path.join(CUDA_HOME, "include", "cub")):
        print("ERROR: CUB headers not found in CUDA installation")
        print(f"Looked in: {CUDA_HOME}/include/cub")
        sys.exit(1)
    
    # Determine what to test
    test_both = not args.pch and not args.no_pch
    
    results = {}
    
    # Test without PCH
    if args.no_pch or test_both:
        print("Testing WITHOUT PCH:")
        print("-" * 30)
        times_no_pch = benchmark_compilation(use_pch=False, iterations=args.iterations)
        
        if times_no_pch:
            avg_no_pch = mean(times_no_pch)
            std_no_pch = stdev(times_no_pch) if len(times_no_pch) > 1 else 0
            print(f"Average: {avg_no_pch:.2f} ms (±{std_no_pch:.2f} ms)")
            print(f"Min: {min(times_no_pch):.2f} ms")
            print(f"Max: {max(times_no_pch):.2f} ms")
            results['no_pch'] = avg_no_pch
        print()
    
    # Test with PCH
    if args.pch or test_both:
        if not pch_supported:
            print("PCH not supported on this CUDA version (requires 12.8+)")
        else:
            print("Testing WITH PCH:")
            print("-" * 30)
            times_with_pch = benchmark_compilation(use_pch=True, iterations=args.iterations)
            
            if times_with_pch:
                avg_with_pch = mean(times_with_pch)
                std_with_pch = stdev(times_with_pch) if len(times_with_pch) > 1 else 0
                print(f"Average: {avg_with_pch:.2f} ms (±{std_with_pch:.2f} ms)")
                print(f"Min: {min(times_with_pch):.2f} ms")
                print(f"Max: {max(times_with_pch):.2f} ms")
                results['pch'] = avg_with_pch
            print()
    
    # Summary
    if len(results) == 2:
        speedup = results['no_pch'] / results['pch']
        improvement = ((results['no_pch'] - results['pch']) / results['no_pch']) * 100
        
        print("="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Without PCH: {results['no_pch']:.2f} ms average")
        print(f"With PCH:    {results['pch']:.2f} ms average")
        print(f"Speedup:     {speedup:.2f}x")
        print(f"Improvement: {improvement:.1f}% faster with PCH")
        print("="*60)


if __name__ == "__main__":
    main()