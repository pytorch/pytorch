# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# pyre-strict
"""
Local benchmark for vec8 vectorization on 1-byte data types.

Measures performance impact of vec8 vectorization for uint8/int8 types.
Run with: buck run //caffe2/test:test_vec8_local_bench
Or as test: buck2 test //caffe2/test:test_vec8_bench_b200
"""

import time
from typing import Callable

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase


def benchmark_op(
    dtype: torch.dtype,
    op: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    warmup: int = 50,
    iters: int = 500,
) -> float:
    """Benchmark a unary operation, return time in us."""
    if dtype in (torch.uint8, torch.int8):
        x = torch.randint(0, 128, (size,), dtype=dtype, device="cuda")
    elif dtype == torch.bool:
        x = torch.randint(0, 2, (size,), dtype=dtype, device="cuda")
    else:
        x = torch.randn(size, device="cuda").to(dtype)

    # Warmup
    for _ in range(warmup):
        _ = op(x)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = op(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters * 1e6  # us


class TestVec8Benchmark(TestCase):
    """Benchmark tests for vec8 vectorization."""

    def test_vec8_performance(self) -> None:
        """Benchmark vec8 performance on 1-byte types."""
        if not TEST_CUDA:
            self.skipTest("CUDA not available")

        print("=" * 70)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {cap[0]}.{cap[1]} (sm{cap[0] * 10 + cap[1]})")
        print("=" * 70)

        dtypes = [
            (torch.uint8, "uint8", 1),
            (torch.int8, "int8", 1),
            (torch.float16, "fp16", 2),
            (torch.float32, "fp32", 4),
        ]

        ops: list[tuple[Callable[[torch.Tensor], torch.Tensor], str]] = [
            (lambda x: x.clone(), "clone"),
            (lambda x: x + 1, "add_scalar"),
            (lambda x: x * 2, "mul_scalar"),
        ]

        size = 134217728  # 128MB

        print("\nPerformance Analysis for 128MB (vec8 impact on 1-byte types)")
        print("=" * 70)
        print("\nExpected behavior with vec8 enabled on sm90+:")
        print("- uint8/int8 should achieve ~2x the bandwidth of fp32")
        print("- This is because vec8 loads 8 bytes per thread vs vec4's 4 bytes")
        print()

        for op_fn, op_name in ops:
            print(f"\n{op_name}:")
            for dtype, dtype_name, elem_size in dtypes:
                time_us = benchmark_op(dtype, op_fn, size)
                bytes_transferred = size * elem_size * 2
                bandwidth_gb = bytes_transferred / (time_us / 1e6) / 1e9
                print(f"  {dtype_name}: {time_us:.2f} us, {bandwidth_gb:.1f} GB/s")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]} (sm{cap[0] * 10 + cap[1]})")
    print("=" * 70)

    # Test sizes
    sizes = [
        (1048576, "1M"),
        (16777216, "16M"),
        (134217728, "128M"),
    ]

    dtypes = [
        (torch.uint8, "uint8", 1),
        (torch.int8, "int8", 1),
        (torch.float16, "fp16", 2),
        (torch.float32, "fp32", 4),
    ]

    ops: list[tuple[Callable[[torch.Tensor], torch.Tensor], str]] = [
        (lambda x: x.clone(), "clone"),
        (lambda x: x + 1, "add_scalar"),
        (lambda x: x * 2, "mul_scalar"),
    ]

    for op_fn, op_name in ops:
        print(f"\nOperation: {op_name}")
        print("-" * 70)
        header = f"{'Size':<10}"
        for _, dtype_name, _ in dtypes:
            header += f" {dtype_name + ' (us)':<14} {'BW (GB/s)':<10}"
        print(header)
        print("-" * 70)

        for size, size_name in sizes:
            row = f"{size_name:<10}"
            for dtype, _, elem_size in dtypes:
                time_us = benchmark_op(dtype, op_fn, size)
                bytes_transferred = size * elem_size * 2  # read + write
                bandwidth_gb = bytes_transferred / (time_us / 1e6) / 1e9
                row += f" {time_us:<14.2f} {bandwidth_gb:<10.1f}"
            print(row)

    # Detailed analysis for 128MB
    print("\n" + "=" * 70)
    print("Performance Analysis for 128MB (vec8 impact on 1-byte types)")
    print("=" * 70)
    size = 134217728  # 128MB

    print("\nExpected behavior with vec8 enabled on sm90+:")
    print("- uint8/int8 should achieve ~2x the bandwidth of fp32")
    print("- This is because vec8 loads 8 bytes per thread vs vec4's 4 bytes")
    print()

    for op_fn, op_name in ops:
        print(f"\n{op_name}:")
        for dtype, dtype_name, elem_size in dtypes:
            time_us = benchmark_op(dtype, op_fn, size)
            bytes_transferred = size * elem_size * 2
            bandwidth_gb = bytes_transferred / (time_us / 1e6) / 1e9
            print(f"  {dtype_name}: {time_us:.2f} us, {bandwidth_gb:.1f} GB/s")


if __name__ == "__main__":
    run_tests()
