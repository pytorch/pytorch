import torch
import torch.func
from torch.func import jacfwd, jacrev, grad
import time
import gc
from memory_profiler import memory_usage
import numpy as np


# Original implementation
def hessian_base(func, argnums=0):
    return jacfwd(jacrev(func, argnums), argnums)


# Optimized implementation for scalar functions
def hessian_optimized(func, argnums=0, is_scalar=True):
    if is_scalar:
        return jacfwd(grad(func, argnums), argnums)
    else:
        return jacfwd(jacrev(func, argnums), argnums)


# Test functions
def quadratic_scalar(x):
    """Scalar function: R^n -> R"""
    return torch.sum(x @ x.t())


def rosenbrock_scalar(x):
    """Scalar function: R^n -> R"""
    return torch.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def vector_output(x):
    """Vector function: R^n -> R^m"""
    return x @ x.t()


def benchmark_hessian(func, x, implementations, num_runs=10, warmup=3):
    """Benchmark hessian implementations"""
    results = {}

    for name, hessian_func in implementations.items():
        # Warmup
        for _ in range(warmup):
            result = hessian_func(func)(x)
            del result
            torch.cuda.empty_cache() if x.is_cuda else gc.collect()

        # Time benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            result = hessian_func(func)(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
            del result
            torch.cuda.empty_cache() if x.is_cuda else gc.collect()

        # Memory benchmark (CPU only for accurate measurement)
        if not x.is_cuda:
            mem_usage = memory_usage(
                (lambda: hessian_func(func)(x).sum().item(),), max_usage=True
            )
        else:
            # For GPU, we'll use torch.cuda memory stats
            torch.cuda.reset_peak_memory_stats()
            result = hessian_func(func)(x)
            mem_usage = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            del result
            torch.cuda.empty_cache()

        results[name] = {
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "memory_mb": mem_usage,
        }

    return results


def run_comprehensive_benchmark():
    """Run benchmark with various configurations"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    dimensions = [10, 50, 100, 500]
    implementations = {
        "base": hessian_base,
        "optimized": lambda func: lambda x: hessian_optimized(func, is_scalar=True)(x),
    }

    results = {}

    for device in devices:
        for dim in dimensions:
            print(f"Testing on {device} with dimension {dim}")

            # Create test tensor
            x = torch.randn(dim, requires_grad=True)
            if device == "cuda":
                x = x.cuda()

            # Test scalar functions
            scalar_results = {}
            for func_name, func in [
                ("quadratic", quadratic_scalar),
                ("rosenbrock", rosenbrock_scalar),
            ]:
                print(f"  Testing {func_name}...")
                scalar_results[func_name] = benchmark_hessian(
                    func, x, implementations, num_runs=5, warmup=2
                )

            # Test vector function (should use base implementation)
            print("  Testing vector function...")
            vector_results = benchmark_hessian(
                vector_output, x, {"base": hessian_base}, num_runs=5, warmup=2
            )

            results[f"{device}_dim{dim}"] = {
                "scalar": scalar_results,
                "vector": vector_results,
            }

    return results


def print_results(results):
    """Print benchmark results in a readable format"""
    for config, data in results.items():
        print(f"\n=== {config} ===")

        print("Scalar functions:")
        for func_name, func_data in data["scalar"].items():
            print(f"  {func_name}:")
            for impl_name, impl_data in func_data.items():
                time_mean = impl_data["time_mean"]
                time_std = impl_data["time_std"]
                memory = impl_data["memory_mb"]
                print(
                    f"    {impl_name}: {time_mean:.2f} ± {time_std:.2f} ms, {memory:.2f} MB"
                )

        print("Vector function:")
        for impl_name, impl_data in data["vector"].items():
            time_mean = impl_data["time_mean"]
            time_std = impl_data["time_std"]
            memory = impl_data["memory_mb"]
            print(
                f"    {impl_name}: {time_mean:.2f} ± {time_std:.2f} ms, {memory:.2f} MB"
            )


def verify_correctness():
    """Verify that both implementations give the same results"""
    x = torch.randn(10, requires_grad=True)

    # Test scalar function
    hess_base = hessian_base(quadratic_scalar)(x)
    hess_opt = hessian_optimized(quadratic_scalar, is_scalar=True)(x)

    print("Scalar function correctness check:")
    print(f"  Max difference: {torch.max(torch.abs(hess_base - hess_opt)).item()}")
    print(f"  All close: {torch.allclose(hess_base, hess_opt, atol=1e-6)}")

    # Test that vector function raises appropriate error with is_scalar=True
    try:
        hessian_optimized(vector_output, is_scalar=True)(x)
        print("ERROR: Vector function should not work with is_scalar=True")
    except Exception as e:
        print(f"  Vector function correctly raises error: {type(e).__name__}")


if __name__ == "__main__":
    print("Verifying correctness first...")
    verify_correctness()

    print("\nRunning comprehensive benchmark...")
    results = run_comprehensive_benchmark()

    print("\nBenchmark results:")
    print_results(results)
