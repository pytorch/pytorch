#!/usr/bin/env python3
"""
Benchmark demonstrating NVRTC compilation speedup with Automatic Precompiled Headers (PCH).

This benchmark compares compilation times with and without automatic PCH for kernels using CUB headers.
Automatic PCH support requires CUDA 12.8+ and automatically caches commonly included headers.

Usage:
    python benchmarks/nvrtc_pch_benchmark.py
"""

import argparse
import os
import sys
import time
from statistics import mean, stdev

import torch
from torch.cuda._utils import _nvrtc_compile

# Check CUDA version for PCH support
def check_cuda_version():
    """Check if CUDA version supports PCH (12.8+)"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    # Get CUDA runtime version
    cuda_version = torch.version.cuda
    if cuda_version:
        major, minor = cuda_version.split('.')[:2]
        version = float(f"{major}.{minor}")
        if version >= 12.8:
            return True, f"CUDA {cuda_version}"
        else:
            return False, f"CUDA {cuda_version} (PCH requires 12.8+)"
    return False, "Unknown CUDA version"


def benchmark_compilation(kernel_source, kernel_name, use_pch=False, iterations=10):
    """Benchmark NVRTC compilation time with proper isolation."""
    
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    compute_capability = f"{major}{minor}"
    
    # Setup include paths
    CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_include_dirs = []
    if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, "include")):
        cuda_include_dirs.append(os.path.join(CUDA_HOME, "include"))
    
    compile_times = []
    
    for i in range(iterations):
        try:
            # Create unique kernel name to avoid caching between iterations
            import random
            unique_suffix = f"{i}_{random.randint(10000, 99999)}"
            unique_kernel_name = f"{kernel_name}_{unique_suffix}"
            unique_source = kernel_source.replace(kernel_name, unique_kernel_name)
            
            start = time.perf_counter()
            
            ptx, mangled_name = _nvrtc_compile(
                unique_source,
                unique_kernel_name,
                compute_capability,
                header_code="",
                cuda_include_dirs=cuda_include_dirs,
                nvcc_options=["-std=c++17"],
                enable_automatic_pch=use_pch
            )
            
            elapsed = time.perf_counter() - start
            compile_times.append(elapsed)
            
        except RuntimeError as e:
            print(f"Compilation failed: {e}")
            return None
    
    return compile_times


def run_alternating_benchmark(kernel_source, kernel_name, iterations=10):
    """Run benchmark alternating between PCH and no-PCH to eliminate order bias."""
    import random
    
    # Create list of configurations, alternating PCH/no-PCH
    configs = []
    for i in range(iterations):
        configs.append(('no_pch', False))
        configs.append(('pch', True))
    
    # Shuffle to randomize order
    random.shuffle(configs)
    
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    compute_capability = f"{major}{minor}"
    
    CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_include_dirs = []
    if CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, "include")):
        cuda_include_dirs.append(os.path.join(CUDA_HOME, "include"))
    
    no_pch_times = []
    pch_times = []
    
    for i, (config_name, use_pch) in enumerate(configs):
        try:
            # Create unique kernel name
            unique_suffix = f"{i}_{random.randint(10000, 99999)}"
            unique_kernel_name = f"{kernel_name}_{unique_suffix}"
            unique_source = kernel_source.replace(kernel_name, unique_kernel_name)
            
            start = time.perf_counter()
            
            ptx, mangled_name = _nvrtc_compile(
                unique_source,
                unique_kernel_name,
                compute_capability,
                header_code="",
                cuda_include_dirs=cuda_include_dirs,
                nvcc_options=["-std=c++17"],
                enable_automatic_pch=use_pch
            )
            
            elapsed = time.perf_counter() - start
            
            if use_pch:
                pch_times.append(elapsed)
            else:
                no_pch_times.append(elapsed)
                
        except RuntimeError as e:
            print(f"Compilation failed: {e}")
            return None, None
    
    return no_pch_times, pch_times


def main():
    parser = argparse.ArgumentParser(description="NVRTC PCH Compilation Benchmark")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of compilation iterations per configuration")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Number of warmup iterations")
    args = parser.parse_args()
    
    # Check CUDA version
    pch_supported, cuda_info = check_cuda_version()
    
    print("="*80)
    print("NVRTC Automatic Precompiled Headers (PCH) Compilation Benchmark")
    print("="*80)
    print(f"CUDA Version: {cuda_info}")
    print(f"PCH Support: {'Yes' if pch_supported else 'No'}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Iterations: {args.iterations} (+ {args.warmup} warmup)")
    print()
    
    # Check for CUB headers
    CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    if not os.path.exists(os.path.join(CUDA_HOME, "include", "cub")):
        print("ERROR: CUB headers not found in CUDA installation")
        print(f"Looked in: {CUDA_HOME}/include/cub")
        print("Please ensure CUDA toolkit is properly installed with CUB headers.")
        sys.exit(1)
    
    # Test kernel 1: Simple CUB block reduction
    kernel1_source = """
    #include <cub/block/block_reduce.cuh>
    
    extern "C"
    __global__ void block_reduce_kernel(const float* input, float* output, int n) {
        using BlockReduce = cub::BlockReduce<float, 256>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float val = (idx < n) ? input[idx] : 0.0f;
        float sum = BlockReduce(temp_storage).Sum(val);
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = sum;
        }
    }
    """
    
    # Test kernel 2: Very complex kernel with many CUB includes (should show better PCH speedup)
    kernel2_source = """
    #include <cub/block/block_reduce.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_radix_sort.cuh>
    #include <cub/warp/warp_reduce.cuh>
    #include <cub/warp/warp_scan.cuh>
    #include <cub/device/device_reduce.cuh>
    #include <cub/iterator/counting_input_iterator.cuh>
    #include <cub/iterator/transform_input_iterator.cuh>
    
    extern "C"
    __global__ void complex_cub_kernel(const float* input, float* output, int n) {
        using BlockReduce = cub::BlockReduce<float, 256>;
        using BlockScan = cub::BlockScan<float, 256>;
        using BlockLoad = cub::BlockLoad<float, 256, 4>;
        using BlockStore = cub::BlockStore<float, 256, 4>;
        using BlockRadixSort = cub::BlockRadixSort<float, 256>;
        using WarpReduce = cub::WarpReduce<float>;
        using WarpScan = cub::WarpScan<float>;
        
        __shared__ union {
            typename BlockReduce::TempStorage reduce;
            typename BlockScan::TempStorage scan;
            typename BlockLoad::TempStorage load;
            typename BlockStore::TempStorage store;
            typename BlockRadixSort::TempStorage sort;
            typename WarpReduce::TempStorage warp_reduce[8];
            typename WarpScan::TempStorage warp_scan[8];
        } temp_storage;
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        float val = (idx < n) ? input[idx] : 0.0f;
        
        // Use many different CUB operations to stress header compilation
        float sum = BlockReduce(temp_storage.reduce).Sum(val);
        __syncthreads();
        
        float scan_result;
        BlockScan(temp_storage.scan).ExclusiveSum(val, scan_result);
        __syncthreads();
        
        int warp_id = threadIdx.x / 32;
        float warp_sum = WarpReduce(temp_storage.warp_reduce[warp_id]).Sum(val);
        float warp_scan_result;
        WarpScan(temp_storage.warp_scan[warp_id]).ExclusiveSum(val, warp_scan_result);
        
        if (threadIdx.x == 0) {
            output[blockIdx.x] = sum + scan_result + warp_sum + warp_scan_result;
        }
    }
    """
    
    print("Benchmark 1: Simple CUB Block Reduction")
    print("-"*40)
    
    # Warmup
    print("Warming up...", end="", flush=True)
    for _ in range(args.warmup):
        benchmark_compilation(kernel1_source, "block_reduce_kernel", use_pch=False, iterations=1)
        if pch_supported:
            benchmark_compilation(kernel1_source, "block_reduce_kernel", use_pch=True, iterations=1)
    print(" done")
    
    # Run alternating benchmark to eliminate order bias
    if pch_supported:
        print("Running alternating PCH/no-PCH benchmark...", end="", flush=True)
        times_no_pch, times_with_pch = run_alternating_benchmark(
            kernel1_source, "block_reduce_kernel", iterations=args.iterations
        )
        print(" done")
        
        if times_no_pch and times_with_pch:
            avg_no_pch = mean(times_no_pch) * 1000
            std_no_pch = stdev(times_no_pch) * 1000 if len(times_no_pch) > 1 else 0
            avg_with_pch = mean(times_with_pch) * 1000
            std_with_pch = stdev(times_with_pch) * 1000 if len(times_with_pch) > 1 else 0
            
            print(f"  Without PCH: {avg_no_pch:.2f} ms (±{std_no_pch:.2f} ms)")
            print(f"  With PCH:    {avg_with_pch:.2f} ms (±{std_with_pch:.2f} ms)")
            
            speedup = avg_no_pch / avg_with_pch
            print(f"  Speedup: {speedup:.2f}x")
    else:
        print("Running without PCH only...", end="", flush=True)
        times_no_pch = benchmark_compilation(kernel1_source, "block_reduce_kernel", 
                                            use_pch=False, iterations=args.iterations)
        print(" done")
        
        if times_no_pch:
            avg_no_pch = mean(times_no_pch) * 1000
            std_no_pch = stdev(times_no_pch) * 1000 if len(times_no_pch) > 1 else 0
            print(f"  Average: {avg_no_pch:.2f} ms (±{std_no_pch:.2f} ms)")
    
    print()
    print("Benchmark 2: Complex Kernel with Many CUB Headers (should show better speedup)")
    print("-"*40)
    
    # Warmup
    print("Warming up...", end="", flush=True)
    for _ in range(args.warmup):
        benchmark_compilation(kernel2_source, "complex_cub_kernel", use_pch=False, iterations=1)
        if pch_supported:
            benchmark_compilation(kernel2_source, "complex_cub_kernel", use_pch=True, iterations=1)
    print(" done")
    
    # Run alternating benchmark to eliminate order bias
    if pch_supported:
        print("Running alternating PCH/no-PCH benchmark...", end="", flush=True)
        times_no_pch, times_with_pch = run_alternating_benchmark(
            kernel2_source, "complex_cub_kernel", iterations=args.iterations
        )
        print(" done")
        
        if times_no_pch and times_with_pch:
            avg_no_pch = mean(times_no_pch) * 1000
            std_no_pch = stdev(times_no_pch) * 1000 if len(times_no_pch) > 1 else 0
            avg_with_pch = mean(times_with_pch) * 1000
            std_with_pch = stdev(times_with_pch) * 1000 if len(times_with_pch) > 1 else 0
            
            print(f"  Without PCH: {avg_no_pch:.2f} ms (±{std_no_pch:.2f} ms)")
            print(f"  With PCH:    {avg_with_pch:.2f} ms (±{std_with_pch:.2f} ms)")
            
            speedup = avg_no_pch / avg_with_pch
            print(f"  Speedup: {speedup:.2f}x")
    else:
        print("Running without PCH only...", end="", flush=True)
        times_no_pch = benchmark_compilation(kernel2_source, "complex_cub_kernel", 
                                            use_pch=False, iterations=args.iterations)
        print(" done")
        
        if times_no_pch:
            avg_no_pch = mean(times_no_pch) * 1000
            std_no_pch = stdev(times_no_pch) * 1000 if len(times_no_pch) > 1 else 0
            print(f"  Average: {avg_no_pch:.2f} ms (±{std_no_pch:.2f} ms)")
    
    print()
    print("="*80)
    
    if not pch_supported:
        print("NOTE: Automatic PCH compilation requires CUDA 12.8 or later.")
        print("      Upgrade CUDA to benefit from faster compilation with automatic PCH.")
    else:
        print("Automatic PCH compilation can significantly speed up kernel compilation")
        print("when using complex headers like CUB, Thrust, or custom libraries.")
        print("NVRTC automatically detects and caches commonly included headers.")
    
    print("="*80)


if __name__ == "__main__":
    main()