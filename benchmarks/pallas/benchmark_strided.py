#!/usr/bin/env python3
"""
Benchmark: Strided (Non-Contiguous) vs Contiguous Tensor Performance in Pallas

This script measures the performance overhead of converting non-contiguous tensors
to contiguous tensors in the Inductor-Pallas path. It auto-detects GPU vs TPU
and runs appropriate benchmarks.

Usage:
    python benchmark_strided_pallas.py [--warmup N] [--iters N] [--sizes S1,S2,...]
"""

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch._inductor.config as inductor_config


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    shape: tuple
    stride_pattern: str
    contiguous_time_ms: float
    non_contiguous_time_ms: float
    slowdown: float  # non_contiguous / contiguous
    device: str
    dtype: torch.dtype

    def __str__(self):
        return (
            f"{self.name:<30} | {str(self.shape):<20} | {self.stride_pattern:<15} | "
            f"Contig: {self.contiguous_time_ms:>8.3f}ms | "
            f"Strided: {self.non_contiguous_time_ms:>8.3f}ms | "
            f"Slowdown: {self.slowdown:>5.2f}x"
        )


class DeviceDetector:
    """Detect available accelerator device."""

    def __init__(self):
        self.device_type = self._detect_device()
        self.device_name = self._get_device_name()
        self._verify_backend_dependencies()

    def _verify_backend_dependencies(self):
        """Verify required backend dependencies are installed."""
        if self.device_type == 'tpu':
            # TPU requires JAX for Pallas backend
            try:
                import jax
                print(f"JAX version: {jax.__version__}")
            except ImportError:
                raise RuntimeError(
                    "JAX is required for TPU benchmark but not installed. "
                    "Install with: pip install jax[tpu]"
                )
        elif self.device_type == 'cuda':
            # GPU requires Triton for native strided support
            try:
                import triton
                print(f"Triton version: {triton.__version__}")
            except ImportError:
                raise RuntimeError(
                    "Triton is required for GPU benchmark but not installed. "
                    "Install with: pip install triton"
                )

    def _detect_device(self) -> str:
        """Detect if we're on GPU or TPU."""
        # Check for TPU first (via JAX)
        try:
            import jax
            tpu_devices = jax.devices('tpu')
            if len(tpu_devices) > 0:
                return 'tpu'
        except Exception:
            pass

        # Check for CUDA GPU
        if torch.cuda.is_available():
            return 'cuda'

        return 'cpu'

    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.device_type == 'tpu':
            try:
                import jax
                device = jax.devices('tpu')[0]
                return f"TPU {device.platform} ({device.device_kind})"
            except Exception:
                return "TPU (unknown)"
        elif self.device_type == 'cuda':
            return torch.cuda.get_device_name(0)
        else:
            return "CPU"

    def get_torch_device(self) -> str:
        """Get PyTorch device string for tensor creation."""
        # For TPU path, we use CPU tensors that get transferred via jax.device_put
        if self.device_type == 'tpu':
            return 'cpu'
        return self.device_type

    def get_backend_config(self) -> dict:
        """Get inductor backend config.

        - TPU: Uses Pallas backend (forces contiguous tensors)
        - GPU: Uses default Triton backend (supports strided access)
        - CPU: Uses Pallas backend for comparison
        """
        if self.device_type == 'tpu':
            return {'cpu_backend': 'pallas'}
        elif self.device_type == 'cuda':
            # Use default Triton backend - supports native strided access
            return {}
        else:
            return {'cpu_backend': 'pallas'}


class StridedTensorFactory:
    """Factory for creating various non-contiguous tensor patterns."""

    @staticmethod
    def create_strided_tensor(
        base_shape: tuple,
        pattern: str,
        dtype: torch.dtype,
        device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create a pair of tensors: (contiguous, non_contiguous) with same logical data.

        Returns:
            Tuple of (contiguous_tensor, non_contiguous_tensor)
        """
        if pattern == 'transpose':
            # Create base tensor and transpose it
            base = torch.randn(*base_shape, dtype=dtype, device=device)
            non_contig = base.T
            contig = non_contig.contiguous()
            return contig, non_contig

        elif pattern == 'stride_2':
            # Every other element (stride=2)
            expanded_shape = tuple(s * 2 for s in base_shape)
            base = torch.randn(*expanded_shape, dtype=dtype, device=device)
            if len(base_shape) == 1:
                non_contig = base[::2]
            elif len(base_shape) == 2:
                non_contig = base[::2, ::2]
            else:
                # Generic N-dim stride
                slices = tuple(slice(None, None, 2) for _ in base_shape)
                non_contig = base[slices]
            contig = non_contig.contiguous()
            return contig, non_contig

        elif pattern == 'stride_4':
            # Every 4th element
            expanded_shape = tuple(s * 4 for s in base_shape)
            base = torch.randn(*expanded_shape, dtype=dtype, device=device)
            if len(base_shape) == 1:
                non_contig = base[::4]
            elif len(base_shape) == 2:
                non_contig = base[::4, ::4]
            else:
                slices = tuple(slice(None, None, 4) for _ in base_shape)
                non_contig = base[slices]
            contig = non_contig.contiguous()
            return contig, non_contig

        elif pattern == 'channels_last':
            # NCHW -> NHWC memory format (4D only)
            if len(base_shape) != 4:
                raise ValueError("channels_last requires 4D tensor (NCHW)")
            base = torch.randn(*base_shape, dtype=dtype, device=device)
            non_contig = base.to(memory_format=torch.channels_last)
            contig = non_contig.contiguous()
            return contig, non_contig

        elif pattern == 'slice_middle':
            # Slice out middle portion (non-zero offset)
            expanded_shape = tuple(s * 3 for s in base_shape)
            base = torch.randn(*expanded_shape, dtype=dtype, device=device)
            if len(base_shape) == 1:
                non_contig = base[base_shape[0]:base_shape[0]*2]
            elif len(base_shape) == 2:
                non_contig = base[base_shape[0]:base_shape[0]*2, base_shape[1]:base_shape[1]*2]
            else:
                slices = tuple(slice(s, s*2) for s in base_shape)
                non_contig = base[slices]
            contig = non_contig.contiguous()
            return contig, non_contig

        elif pattern == 'expand':
            # Expanded tensor (stride=0 in expanded dims)
            if len(base_shape) < 2:
                raise ValueError("expand requires at least 2D shape")
            # Create smaller tensor and expand
            small_shape = (1,) + base_shape[1:]
            base = torch.randn(*small_shape, dtype=dtype, device=device)
            non_contig = base.expand(*base_shape)
            contig = non_contig.contiguous()
            return contig, non_contig

        elif pattern == 'diagonal':
            # Diagonal of a matrix (large stride)
            if len(base_shape) != 1:
                raise ValueError("diagonal pattern expects 1D output shape")
            n = base_shape[0]
            base = torch.randn(n, n, dtype=dtype, device=device)
            non_contig = base.diagonal()
            contig = non_contig.contiguous()
            return contig, non_contig

        elif pattern == 'permute':
            # Arbitrary permutation (3D+)
            if len(base_shape) < 3:
                raise ValueError("permute requires at least 3D tensor")
            base = torch.randn(*base_shape, dtype=dtype, device=device)
            # Reverse dimension order
            dims = list(range(len(base_shape)))[::-1]
            non_contig = base.permute(*dims)
            contig = non_contig.contiguous()
            return contig, non_contig

        else:
            raise ValueError(f"Unknown stride pattern: {pattern}")


def benchmark_single_op(
    op_fn: Callable,
    tensor_contig: torch.Tensor,
    tensor_strided: torch.Tensor,
    n_warmup: int = 10,
    n_iters: int = 100,
    device_type: str = 'cpu'
) -> tuple[float, float]:
    """
    Benchmark a single operation on contiguous vs strided tensors.

    Returns:
        Tuple of (contiguous_time_ms, strided_time_ms)
    """
    # Compile the function once for each input type
    compiled_fn = op_fn

    # Warmup with contiguous
    for _ in range(n_warmup):
        _ = compiled_fn(tensor_contig)

    # Sync before timing
    if device_type == 'cuda':
        torch.cuda.synchronize()
    elif device_type == 'tpu':
        # For TPU, JAX handles sync internally in device_put/device_get
        pass

    # Time contiguous
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = compiled_fn(tensor_contig)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    contig_time = (time.perf_counter() - start) / n_iters * 1000  # ms

    # Warmup with strided (may trigger recompilation)
    for _ in range(n_warmup):
        _ = compiled_fn(tensor_strided)

    if device_type == 'cuda':
        torch.cuda.synchronize()

    # Time strided
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = compiled_fn(tensor_strided)
    if device_type == 'cuda':
        torch.cuda.synchronize()
    strided_time = (time.perf_counter() - start) / n_iters * 1000  # ms

    return contig_time, strided_time


def create_benchmark_ops(backend_config: dict) -> dict[str, Callable]:
    """Create compiled operations for benchmarking."""

    # Simple elementwise - memory bound
    @torch.compile(backend='inductor', options=backend_config)
    def elementwise_add(x):
        return x + x

    @torch.compile(backend='inductor', options=backend_config)
    def elementwise_mul(x):
        return x * 2.0

    @torch.compile(backend='inductor', options=backend_config)
    def elementwise_sin(x):
        return torch.sin(x)

    # Fused ops - still memory bound but more compute
    @torch.compile(backend='inductor', options=backend_config)
    def fused_gelu(x):
        return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

    @torch.compile(backend='inductor', options=backend_config)
    def fused_silu(x):
        # SiLU/Swish activation: x * sigmoid(x)
        return x * torch.sigmoid(x)

    # Reduction ops
    @torch.compile(backend='inductor', options=backend_config)
    def reduce_sum(x):
        return x.sum()

    @torch.compile(backend='inductor', options=backend_config)
    def reduce_mean(x):
        return x.mean()

    return {
        'elementwise_add': elementwise_add,
        'elementwise_mul': elementwise_mul,
        'elementwise_sin': elementwise_sin,
        'fused_gelu': fused_gelu,
        'fused_silu': fused_silu,
        'reduce_sum': reduce_sum,
        'reduce_mean': reduce_mean,
    }


def run_benchmark_suite(
    detector: DeviceDetector,
    n_warmup: int = 10,
    n_iters: int = 100,
    sizes: Optional[list[int]] = None
) -> list[BenchmarkResult]:
    """Run the full benchmark suite."""

    results = []
    torch_device = detector.get_torch_device()
    backend_config = detector.get_backend_config()

    # Enable TPU targeting if on TPU
    if detector.device_type == 'tpu':
        inductor_config._debug_cpu_to_tpu_pallas = True

    # Create ops
    ops = create_benchmark_ops(backend_config)

    # Default sizes if not specified
    if sizes is None:
        sizes = [1024, 4096, 16384]

    # Define test configurations
    # (shape_template, stride_patterns, compatible_ops)
    test_configs = [
        # 1D tensors - elementwise ops
        ('1d', lambda s: (s * s,), ['stride_2', 'stride_4', 'slice_middle'],
         ['elementwise_add', 'elementwise_mul', 'elementwise_sin', 'reduce_sum']),

        # 2D tensors - matrix-like ops
        ('2d', lambda s: (s, s), ['transpose', 'stride_2', 'slice_middle'],
         ['elementwise_add', 'elementwise_mul', 'fused_gelu', 'fused_silu', 'reduce_mean']),

        # 3D tensors - sequence/batch ops
        ('3d', lambda s: (4, s, s), ['permute', 'stride_2'],
         ['elementwise_add', 'elementwise_sin', 'reduce_sum']),

        # 4D tensors - vision ops (channels_last)
        ('4d_vision', lambda s: (2, 64, s//4, s//4), ['channels_last', 'stride_2'],
         ['elementwise_add', 'elementwise_mul', 'fused_gelu']),

        # Diagonal access pattern (very large stride)
        ('diagonal', lambda s: (s,), ['diagonal'],
         ['elementwise_add', 'elementwise_mul', 'reduce_sum']),

        # Expand/broadcast pattern (stride=0)
        ('expand', lambda s: (s, s), ['expand'],
         ['elementwise_add', 'elementwise_mul']),
    ]

    # Determine backend name for display
    if detector.device_type == 'tpu':
        backend_name = "Pallas (forces contiguous)"
    elif detector.device_type == 'cuda':
        backend_name = "Triton (native strided support)"
    else:
        backend_name = "Pallas/CPU"

    print(f"\n{'='*100}")
    print(f"Strided vs Contiguous Benchmark - {detector.device_name}")
    print(f"Backend: {backend_name}")
    print(f"Warmup: {n_warmup}, Iterations: {n_iters}")
    print(f"{'='*100}\n")

    for config_name, shape_fn, stride_patterns, op_names in test_configs:
        print(f"\n--- {config_name.upper()} Tensors ---")
        print("-" * 100)

        for size in sizes:
            try:
                shape = shape_fn(size)
            except Exception:
                continue

            for pattern in stride_patterns:
                # Skip incompatible shape/pattern combinations
                try:
                    contig, strided = StridedTensorFactory.create_strided_tensor(
                        shape, pattern, torch.float32, torch_device
                    )
                except ValueError as e:
                    continue

                for op_name in op_names:
                    if op_name not in ops:
                        continue

                    op_fn = ops[op_name]

                    try:
                        # Clear any cached compilations
                        torch._dynamo.reset()

                        contig_time, strided_time = benchmark_single_op(
                            op_fn, contig, strided,
                            n_warmup=n_warmup,
                            n_iters=n_iters,
                            device_type=detector.device_type
                        )

                        slowdown = strided_time / contig_time if contig_time > 0 else float('inf')

                        result = BenchmarkResult(
                            name=op_name,
                            shape=shape,
                            stride_pattern=pattern,
                            contiguous_time_ms=contig_time,
                            non_contiguous_time_ms=strided_time,
                            slowdown=slowdown,
                            device=detector.device_type,
                            dtype=torch.float32
                        )
                        results.append(result)
                        print(result)

                    except Exception as e:
                        print(f"  SKIP {op_name} on {shape} ({pattern}): {e}")

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print summary statistics."""
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    # Group by stride pattern
    by_pattern = {}
    for r in results:
        if r.stride_pattern not in by_pattern:
            by_pattern[r.stride_pattern] = []
        by_pattern[r.stride_pattern].append(r)

    print("\nSlowdown by Stride Pattern:")
    print("-" * 60)
    for pattern, pattern_results in sorted(by_pattern.items()):
        slowdowns = [r.slowdown for r in pattern_results]
        avg_slowdown = sum(slowdowns) / len(slowdowns)
        max_slowdown = max(slowdowns)
        min_slowdown = min(slowdowns)
        print(f"  {pattern:<20}: avg={avg_slowdown:.2f}x, min={min_slowdown:.2f}x, max={max_slowdown:.2f}x")

    # Group by op
    by_op = {}
    for r in results:
        if r.name not in by_op:
            by_op[r.name] = []
        by_op[r.name].append(r)

    print("\nSlowdown by Operation:")
    print("-" * 60)
    for op_name, op_results in sorted(by_op.items()):
        slowdowns = [r.slowdown for r in op_results]
        avg_slowdown = sum(slowdowns) / len(slowdowns)
        print(f"  {op_name:<25}: avg={avg_slowdown:.2f}x (n={len(op_results)})")

    # Overall
    all_slowdowns = [r.slowdown for r in results]
    print(f"\nOverall Average Slowdown: {sum(all_slowdowns)/len(all_slowdowns):.2f}x")
    print(f"Worst Case Slowdown: {max(all_slowdowns):.2f}x")

    # Highlight worst cases
    print("\nWorst 5 Cases:")
    print("-" * 100)
    worst = sorted(results, key=lambda r: r.slowdown, reverse=True)[:5]
    for r in worst:
        print(f"  {r}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark strided vs contiguous tensor performance in Pallas'
    )
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=100, help='Number of benchmark iterations')
    parser.add_argument('--sizes', type=str, default='512,1024,2048',
                        help='Comma-separated list of tensor sizes to test')
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',')]

    # Detect device
    detector = DeviceDetector()
    print(f"Detected device: {detector.device_name} ({detector.device_type})")

    # Run benchmarks
    results = run_benchmark_suite(
        detector,
        n_warmup=args.warmup,
        n_iters=args.iters,
        sizes=sizes
    )

    # Print summary
    print_summary(results)


if __name__ == '__main__':
    main()
