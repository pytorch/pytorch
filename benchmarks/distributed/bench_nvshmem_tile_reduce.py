#!/usr/bin/env python3
"""
Benchmark for NVSHMEM tile reduce operations.

Usage:
python benchmarks/distributed/bench_nvshmem_tile_reduce.py

This benchmark measures the performance of tile reduce operations across different
matrix sizes and tile configurations.
"""

import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    requires_cuda_p2p_access,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)


# Decorator
def requires_nvshmem():
    return skip_but_pass_in_sandcastle_if(
        not symm_mem.is_nvshmem_available(),
        "bench_nvshmem_tile_reduce requires NVSHMEM, skipping benchmark",
    )


# So that benchmarks are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_nvshmem()
@requires_cuda_p2p_access()
class NVSHMEMTileReduceBenchmark(MultiProcContinuousTest):
    def _init_device(self) -> None:
        # TODO: relieve this (seems to hang if without)
        device_module.set_device(self.device)
        # Set NVSHMEM as SymmMem backend
        symm_mem.set_backend("NVSHMEM")

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def _benchmark_tile_reduce_single(
        self,
        full_size: int,
        tile_size: int,
        warmup_iters: int = 5,
        bench_iters: int = 10,
    ) -> dict:
        """
        Benchmark a single configuration of tile reduce.

        Args:
            full_size: Size of the full matrix (full_size x full_size)
            warmup_iters: Number of warmup iterations
            bench_iters: Number of benchmark iterations

        Returns:
            Dictionary with benchmark results
        """
        self._init_device()
        group_name = dist.group.WORLD.group_name

        dtype = torch.float

        # Allocate full matrices
        full_inp = symm_mem.empty(
            full_size, full_size, dtype=dtype, device=self.device
        ).fill_(self.rank)
        full_out = symm_mem.empty(
            full_size, full_size, dtype=dtype, device=self.device
        ).fill_(0)

        slice_ut = slice(0, tile_size)
        inp_tile = full_inp[slice_ut, slice_ut]
        out_tile = full_out[slice_ut, slice_ut]

        root = 0

        # Warmup iterations
        for _ in range(warmup_iters):
            torch.ops.symm_mem.tile_reduce(inp_tile, out_tile, root, group_name)
            torch.cuda.synchronize(self.device)

        # Benchmark iterations
        times = []

        dist.barrier()
        torch.cuda.synchronize(self.device)
        start_time = time.perf_counter()

        for _ in range(bench_iters):
            torch.ops.symm_mem.tile_reduce(inp_tile, out_tile, root, group_name)

        torch.cuda.synchronize(self.device)
        end_time = time.perf_counter()
        times.append((end_time - start_time) / bench_iters)

        # Calculate statistics
        times = torch.tensor(times, dtype=torch.float64)
        tile_elements = tile_size * tile_size
        tile_bytes = (
            tile_elements * dtype.itemsize
            if hasattr(dtype, "itemsize")
            else tile_elements * 4
        )

        results = {
            "full_size": full_size,
            "tile_size": tile_size,
            "tile_elements": tile_elements,
            "tile_bytes": tile_bytes,
            "world_size": self.world_size,
            "mean_time_ms": times.mean().item() * 1000,
            "std_time_ms": times.std().item() * 1000,
            "min_time_ms": times.min().item() * 1000,
            "max_time_ms": times.max().item() * 1000,
            "throughput_gb_s": tile_bytes / (times.mean().item() * 1e9),
            "elements_per_sec": tile_elements / times.mean().item(),
        }

        return results

    @skipIfRocm
    def test_benchmark_tile_reduce_various_sizes(self) -> None:
        """
        Benchmark tile reduce across various matrix sizes.
        """
        # Test various matrix sizes
        tile_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        full_size = tile_sizes[-1]
        warmup_iters = 5
        bench_iters = 20

        results = []

        for tile_size in tile_sizes:
            try:
                result = self._benchmark_tile_reduce_single(
                    full_size, tile_size, warmup_iters, bench_iters
                )
                results.append(result)

                if self.rank == 0:
                    print(
                        f"Matrix Size: {full_size}x{full_size}, Tile Size: {tile_size}x{tile_size}"
                    )
                    print(
                        f"  Mean Time: {result['mean_time_ms']:.3f} ± {result['std_time_ms']:.3f} ms"
                    )
                    print(f"  Throughput: {result['throughput_gb_s']:.2f} GB/s")
                    print(f"  Bytes: {result['tile_bytes']:.0f}")
                    print()

            except Exception as e:
                if self.rank == 0:
                    print(f"Failed to benchmark matrix size {full_size}: {e}")

        # Print summary
        if self.rank == 0 and results:
            print("=== BENCHMARK SUMMARY ===")
            print(
                f"{'Matrix Size':<12} {'Tile Size':<10} {'Time (ms)':<12} {'Throughput (GB/s)':<18} {'Bytes':<15}"
            )
            print("-" * 70)

            for result in results:
                print(
                    f"{result['full_size']}x{result['full_size']:<7} "
                    f"{result['tile_size']}x{result['tile_size']:<5} "
                    f"{result['mean_time_ms']:<12.3f} "
                    f"{result['throughput_gb_s']:<18.2f} "
                    f"{result['tile_bytes']:<15.0f}"
                )


if __name__ == "__main__":
    # For standalone usage, you'd need to set up distributed environment
    # For now, this is meant to be run via the PyTorch test framework
    from torch.testing._internal.common_utils import run_tests

    run_tests()
