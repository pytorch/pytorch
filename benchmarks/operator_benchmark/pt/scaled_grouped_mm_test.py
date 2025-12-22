import importlib.util
import os
from types import ModuleType
from typing import Optional

from pt import configs  # noqa: F401

import operator_benchmark as op_bench

import torch
from torch.nn.functional import ScalingType
from torch.testing._internal.common_cuda import (
    IS_SM100,
    IS_SM90,
    PLATFORM_SUPPORTS_FP8_GROUPED_GEMM,
)
from torch.torch_version import TorchVersion


"""
Operator microbenchmarks for `scaled_grouped_mm`.

This benchmark supports:
- FP8 (e4m3/e5m2) with TensorWise and RowWise scaling:
  * CUDA SM90 (H100) only - not supported on SM100 (B200)
  * ROCm MI300+ (gfx94x) with grouped GEMM support
- MXFP8/MXFP4/NVFP4 grouped-K path with blocked scaling:
  * CUDA-only (non-HIP), SM90+ and SM100+
  * Requires swizzled scales

All modes reuse the same conversion helpers as `test/test_scaled_matmul_cuda.py`.
"""

_TEST_SCALED_MATMUL_CUDA_MOD: Optional[ModuleType] = None


def _should_generate_scaled_grouped_mm_configs() -> bool:
    # Minimum requirements:
    # - PyTorch 2.9+ (scaled_grouped_mm introduced)
    # - CUDA: compute capability exactly 9.0 (SM90) or 10.0 (SM100) and CUDA 12.8+
    # - ROCm: MI300+ (gfx94x) grouped GEMM support
    if TorchVersion(torch.__version__) < "2.9" or not hasattr(
        torch.nn.functional, "scaled_grouped_mm"
    ):
        return False
    if not torch.cuda.is_available():
        return False

    if torch.version.hip is not None:
        return bool(PLATFORM_SUPPORTS_FP8_GROUPED_GEMM)

    # CUDA build: some scale modes require CUDA 12.8+ (see `aten/src/ATen/cuda/CUDABlas.cpp:get_scale_mode`).
    if TorchVersion(torch.version.cuda or "0.0") < "12.8":
        return False

    return bool(IS_SM90) or bool(IS_SM100)


def _get_test_scaled_matmul_cuda() -> ModuleType:
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
            f"Expected to find {test_file} to reuse scaled_grouped_mm test helpers, but it does not exist."
        )

    spec = importlib.util.spec_from_file_location(
        "_test_scaled_matmul_cuda_grouped_bench_import", test_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {test_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TEST_SCALED_MATMUL_CUDA_MOD = mod
    return mod


def _build_equal_k_group_offs(total_k: int, groups: int, device: str) -> torch.Tensor:
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


def _get_float8_dtype(float8_dtype):
    """Normalize the FP8 dtype arg (handles ROCm fnuz variants via test aliases)."""
    from torch.testing._internal.common_device_type import e4m3_type, e5m2_type

    if float8_dtype in ("e4m3fn", e4m3_type, torch.float8_e4m3fn):
        return e4m3_type
    if float8_dtype in ("e5m2", e5m2_type, torch.float8_e5m2):
        return e5m2_type
    return e4m3_type  # default


class ScaledGroupedMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(
        self,
        M,
        N,
        K,
        G,
        device,
        scaling="mxfp8",
        output_dtype="bfloat16",
        float8_dtype="e4m3fn",
    ):
        if output_dtype != "bfloat16":
            raise ValueError(
                "scaled_grouped_mm benchmark currently supports bfloat16 output only"
            )
        self.output_dtype = torch.bfloat16

        if device != "cuda":
            raise ValueError("scaled_grouped_mm benchmark is CUDA-only")
        if torch.version.hip is not None and not PLATFORM_SUPPORTS_FP8_GROUPED_GEMM:
            raise ValueError(
                "scaled_grouped_mm benchmark requires ROCm MI300+ (gfx94x) grouped GEMM support"
            )

        self.scaling = scaling
        self.base_dtype = torch.bfloat16

        if scaling in ("fp8_tensorwise", "fp8_rowwise"):
            self._init_fp8(M, N, K, G, device, float8_dtype, scaling)
        elif scaling in ("mxfp8", "mxfp4", "nvfp4"):
            self._init_mx_nvfp4(M, N, K, G, device, scaling)
        else:
            raise ValueError(f"Unsupported scaling format: {scaling}")

        self.set_module_name("scaled_grouped_mm")

    def _init_fp8(self, M, N, K, G, device, float8_dtype, scaling):
        """Initialize FP8 tensorwise or rowwise scaling."""
        self.float8_dtype = _get_float8_dtype(float8_dtype)

        # We interpret offs as group end offsets along K (grouped-K).
        # Use deterministic equal-sized groups.
        offs = _build_equal_k_group_offs(K, G, device)

        # Create FP8 inputs directly (similar to test_scaled_grouped_gemm_2d_2d)
        # Input shapes: (M, K) and (N, K) where K is the total K dimension
        x_lp = torch.randn(M, K, device=device, dtype=self.base_dtype).to(
            self.float8_dtype
        )
        w_lp = torch.randn(N, K, device=device, dtype=self.base_dtype).to(
            self.float8_dtype
        )

        if scaling == "fp8_tensorwise":
            # Tensorwise scaling: one scale per group
            # For tensorwise, we still use RowWise recipe but with repeated scales
            # scale_a: (M*G,), scale_b: (N*G,)
            # Each group gets the same scale value repeated M or N times
            scale_a = torch.rand(
                G, device=device, dtype=torch.float32
            ).repeat_interleave(M)
            scale_b = torch.rand(
                G, device=device, dtype=torch.float32
            ).repeat_interleave(N)

            self._scale_recipe_a = ScalingType.RowWise
            self._scale_recipe_b = ScalingType.RowWise

        elif scaling == "fp8_rowwise":
            # Rowwise scaling: M scales per group, N scales per group
            # scale_a: (M*G,), scale_b: (N*G,)
            # Organized as [group0_M_scales, group1_M_scales, ..., group_{G-1}_M_scales]
            scale_a = torch.rand(M * G, device=device, dtype=torch.float32)
            scale_b = torch.rand(N * G, device=device, dtype=torch.float32)

            self._scale_recipe_a = ScalingType.RowWise
            self._scale_recipe_b = ScalingType.RowWise

        self._swizzle_a = None
        self._swizzle_b = None

        # For grouped-K, mat_b is expected as (K, N).
        self.inputs = {
            "x": x_lp,
            "w_t": w_lp.t(),
            "offs": offs,
            "scale_a": scale_a,
            "scale_b": scale_b,
        }

    def _init_mx_nvfp4(self, M, N, K, G, device, scaling):
        """Initialize MX or NVFP4 blocked scaling (CUDA-only, non-HIP)."""
        helpers = _get_test_scaled_matmul_cuda()

        if torch.version.hip is not None:
            raise ValueError(
                f"{scaling} benchmarks are only wired for CUDA swizzled scales (non-HIP)."
            )

        # We interpret offs as group end offsets along K (grouped-K).
        # Use deterministic equal-sized groups.
        offs = _build_equal_k_group_offs(K, G, device)

        # Create high-precision inputs and quantize per-group along K into the requested format.
        # Use modest magnitudes to avoid degenerate saturation.
        x_hp = torch.randn((M, K), device=device, dtype=torch.bfloat16) * 0.1
        w_hp = torch.randn((N, K), device=device, dtype=torch.bfloat16) * 0.1

        _, xq, x_scales, x_global = helpers._2d_grouped_tensor_to_blocked_scaled(
            x_hp, M, G, offs, format=scaling
        )
        _, wq, w_scales, w_global = helpers._2d_grouped_tensor_to_blocked_scaled(
            w_hp, N, G, offs, format=scaling
        )

        if scaling == "nvfp4":
            kwargs = helpers._build_scaled_grouped_mm_kwargs(
                [x_scales, x_global],
                [w_scales, w_global],
                offs,
                format=scaling,
            )
        else:
            kwargs = helpers._build_scaled_grouped_mm_kwargs(
                x_scales, w_scales, offs, format=scaling
            )

        self._scale_recipe_a = kwargs["scale_recipe_a"]
        self._scale_recipe_b = kwargs["scale_recipe_b"]
        self._swizzle_a = kwargs.get("swizzle_a", None)
        self._swizzle_b = kwargs.get("swizzle_b", None)

        # For grouped-K, mat_b is expected as (K, N).
        self.inputs = {
            "x": xq,
            "w_t": wq.t(),
            "offs": offs,
            "scale_a": kwargs["scale_a"],
            "scale_b": kwargs["scale_b"],
        }

    def forward(self, x, w_t, offs, scale_a, scale_b):
        call_kwargs = {
            "scale_a": scale_a,
            "scale_recipe_a": self._scale_recipe_a,
            "scale_b": scale_b,
            "scale_recipe_b": self._scale_recipe_b,
            "offs": offs,
            "output_dtype": self.output_dtype,
        }
        if self._swizzle_a is not None:
            call_kwargs["swizzle_a"] = self._swizzle_a
        if self._swizzle_b is not None:
            call_kwargs["swizzle_b"] = self._swizzle_b

        return torch.nn.functional.scaled_grouped_mm(x, w_t, **call_kwargs)


# Reduced shapes for faster runs.
# Note: K must be divisible by G and (K/G) must be divisible by 32.
MNKG_list = [
    (16384, 8192, 5120, 1),
    (16384, 8192, 5120, 2),
    (16384, 8192, 5120, 4),
    (16384, 8192, 5120, 8),
    (128000, 8192, 5120, 1),
    (128000, 8192, 5120, 2),
    (128000, 8192, 5120, 4),
    (128000, 8192, 5120, 8),
    (16384, 1536, 5120, 1),
    (16384, 1536, 5120, 8),
    (128000, 1536, 5120, 1),
    (128000, 1536, 5120, 8),
    (16384, 2048, 7168, 1),
    (16384, 2048, 7168, 8),
    (128000, 2048, 7168, 1),
    (128000, 2048, 7168, 8),
]

scaled_grouped_mm_configs_long = []

if _should_generate_scaled_grouped_mm_configs():
    # FP8 tensorwise and rowwise: works on both CUDA and ROCm
    # Requires PLATFORM_SUPPORTS_FP8_GROUPED_GEMM (SM90/H100, not SM100/B200)
    if PLATFORM_SUPPORTS_FP8_GROUPED_GEMM:
        scaled_grouped_mm_configs_long += op_bench.config_list(
            attr_names=["M", "N", "K", "G"],
            attrs=[[m, n, k, g] for (m, n, k, g) in MNKG_list],
            cross_product_configs={
                "device": ["cuda"],
                "float8_dtype": ["e4m3fn"],
                "output_dtype": ["bfloat16"],
                "scaling": ["fp8_tensorwise", "fp8_rowwise"],
            },
            tags=["long"],
        )

    # MX + NVFP4 are CUDA-only (non-HIP) due to swizzled scale requirements.
    if torch.version.hip is None:
        scaled_grouped_mm_configs_long += op_bench.config_list(
            attr_names=["M", "N", "K", "G"],
            attrs=[[m, n, k, g] for (m, n, k, g) in MNKG_list],
            cross_product_configs={
                "device": ["cuda"],
                "scaling": ["mxfp4", "mxfp8", "nvfp4"],
                "output_dtype": ["bfloat16"],
            },
            tags=["long"],
        )

op_bench.generate_pt_test(
    scaled_grouped_mm_configs_long,
    ScaledGroupedMMBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
