import importlib.util
import os
from types import ModuleType
from typing import Optional

import torch
from torch.torch_version import TorchVersion


"""
Shared utilities for scaled_mm and scaled_grouped_mm benchmarks.

Both benchmarks use FP8, MX, and NVFP4 quantization with helpers from
test/test_scaled_matmul_cuda.py. This module provides common imports
and utilities to reduce code duplication.
"""

_TEST_SCALED_MATMUL_CUDA_MOD: Optional[ModuleType] = None


# Shared benchmark shapes for scaled matmul operations
# These shapes are used by both scaled_mm and scaled_grouped_mm benchmarks
SCALED_MM_BASE_SHAPES = [
    # Small shapes for faster benchmarking
    (1024, 1024, 1024),
    (2048, 4096, 2048),
    (4096, 2048, 4096),
    # Original larger shapes
    (16384, 8192, 5120),
    (128000, 8192, 5120),
    (16384, 1536, 5120),
    (128000, 1536, 5120),
    (16384, 2048, 7168),
    (128000, 2048, 7168),
]


def get_test_scaled_matmul_cuda() -> ModuleType:
    """
    Reuse scale/quantization helpers from `test/test_scaled_matmul_cuda.py`.

    `test/` isn't a package, so we import by path and cache the module.
    """
    global _TEST_SCALED_MATMUL_CUDA_MOD
    if _TEST_SCALED_MATMUL_CUDA_MOD is not None:
        return _TEST_SCALED_MATMUL_CUDA_MOD

    pytorch_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    test_file = os.path.join(pytorch_root, "test", "test_scaled_matmul_cuda.py")
    if not os.path.exists(test_file):
        raise RuntimeError(
            f"Expected to find {test_file} to reuse scaled matmul test helpers, but it does not exist."
        )

    spec = importlib.util.spec_from_file_location(
        "_test_scaled_matmul_cuda_bench_import", test_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {test_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TEST_SCALED_MATMUL_CUDA_MOD = mod
    return mod


def get_float8_dtype(float8_dtype):
    """Normalize the FP8 dtype arg (handles ROCm fnuz variants via test aliases)."""
    from torch.testing._internal.common_device_type import e4m3_type, e5m2_type

    if float8_dtype in ("e4m3fn", e4m3_type, torch.float8_e4m3fn):
        return e4m3_type
    if float8_dtype in ("e5m2", e5m2_type, torch.float8_e5m2):
        return e5m2_type
    return e4m3_type  # default


def build_equal_k_group_offs(total_k: int, groups: int, device: str) -> torch.Tensor:
    """
    Build equal-sized group offsets for grouped-K operations.

    Used by scaled_grouped_mm to partition the K dimension into equal groups.
    Returns a tensor of group end offsets.

    Args:
        total_k: Total K dimension to partition
        groups: Number of groups
        device: Device to create tensor on

    Returns:
        Tensor of shape (groups,) with group end offsets
    """
    if groups <= 0:
        raise ValueError(f"groups must be > 0, got {groups}")
    if total_k % groups != 0:
        raise ValueError(f"total_k ({total_k}) must be divisible by groups ({groups})")
    k_per_group = total_k // groups
    if k_per_group % 32 != 0:
        raise ValueError(
            f"K per group must be divisible by 32 for these kernels, got {k_per_group}"
        )
    return torch.arange(
        k_per_group, total_k + 1, k_per_group, device=device, dtype=torch.int32
    )


def supports_fp8_deepseek_blockwise_scaling() -> bool:
    """
    Check if the platform supports DeepSeek-style FP8 blockwise scaling.

    PyTorch currently gates "DeepSeek style" FP8 blockwise scaling (1x128/128x128)
    to SM90 (H100) only. On SM100 (B200) this errors with NotImplementedError.
    """
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    # These scaling modes require CUDA 12.9+ (see `aten/src/ATen/cuda/CUDABlas.cpp:get_scale_mode`).
    if torch.version.hip is None and TorchVersion(torch.version.cuda) < "12.9":
        return False
    if (
        torch.version.hip is not None
        and "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):  # Blockwise scaling is not supported on ROCm
        return False
    return torch.cuda.get_device_capability(0) == (9, 0)
