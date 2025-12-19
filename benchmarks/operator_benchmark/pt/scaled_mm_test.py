from pt import configs

import operator_benchmark as op_bench

import importlib.util
import os
from types import ModuleType
from typing import Optional

import torch
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.torch_version import TorchVersion


"""
Operator microbenchmarks for `scaled_mm`.

Uses the same dtype + scale/quantize helpers as `test/test_scaled_matmul_cuda.py`
(bf16/fp16/fp32, fp8 e4m3/e5m2, MX e8m0 scales, NVFP4 packed fp4).
"""

_TEST_SCALED_MATMUL_CUDA_MOD: Optional[ModuleType] = None


def _should_generate_scaled_mm_configs() -> bool:
    # `torch._scaled_mm` / `torch.nn.functional.scaled_mm` are only available in torch 2.9+.
    if TorchVersion(torch.__version__) < "2.9":
        return False
    if not hasattr(torch.nn.functional, "scaled_mm"):
        return False

    # If CUDA isn't available on the current machine, keep configs around and let the
    # operator_benchmark harness skip CUDA tests (see benchmark_core._build_test).
    if not torch.cuda.is_available():
        return True

    # `PLATFORM_SUPPORTS_FP8` matches the kernel support checks:
    # CUDA: SM90+ or SM89; ROCm: MI300+.
    return bool(PLATFORM_SUPPORTS_FP8)


def _get_test_scaled_matmul_cuda() -> ModuleType:
    """
    Reuse scale/quantization helpers from `test/test_scaled_matmul_cuda.py`.

    `test/` isn't a package, so we import by path and cache the module.
    """
    global _TEST_SCALED_MATMUL_CUDA_MOD
    if _TEST_SCALED_MATMUL_CUDA_MOD is not None:
        return _TEST_SCALED_MATMUL_CUDA_MOD

    pytorch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    test_file = os.path.join(pytorch_root, "test", "test_scaled_matmul_cuda.py")
    if not os.path.exists(test_file):
        raise RuntimeError(
            f"Expected to find {test_file} to reuse scaled_mm test helpers, but it does not exist."
        )

    spec = importlib.util.spec_from_file_location("_test_scaled_matmul_cuda_bench_import", test_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {test_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _TEST_SCALED_MATMUL_CUDA_MOD = mod
    return mod


def _get_float8_dtype(float8_dtype):
    """Normalize the FP8 dtype arg (handles ROCm fnuz variants via test aliases)."""
    from torch.testing._internal.common_device_type import e4m3_type, e5m2_type

    if float8_dtype in ("e4m3fn", e4m3_type, torch.float8_e4m3fn):
        return e4m3_type
    if float8_dtype in ("e5m2", e5m2_type, torch.float8_e5m2):
        return e5m2_type
    return e4m3_type  # default


def _supports_fp8_deepseek_blockwise_scaling() -> bool:
    # PyTorch currently gates "DeepSeek style" FP8 blockwise scaling (1x128/128x128)
    # to SM90 (H100) only. On SM100 (B200) this errors with NotImplementedError.
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    return torch.cuda.get_device_capability(0) == (9, 0)


def _cuda_version_tuple() -> tuple[int, int]:
    v = torch.version.cuda
    if v is None:
        return (0, 0)
    # torch.version.cuda is typically like "12.9"
    parts = v.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return (0, 0)


def _supports_fp8_rowwise_fp32_output() -> bool:
    # Mirrors test_scaled_mm_vs_emulated_row_wise gating:
    # fp32 rowwise kernels are cuBLAS-only, CUDA 12.9+, and SM90-only.
    if torch.version.hip is not None:
        return False
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    if _cuda_version_tuple() < (12, 9):
        return False
    return torch.cuda.get_device_capability(0) == (9, 0)


class ScaledMMBenchmark(op_bench.TorchBenchmarkBase):
    _MX_BLOCK_SIZE: int = 32
    _NVFP4_BLOCK_SIZE: int = 16
    _FP8_BLOCK_K: int = 128

    def _set_output_dtype(self, output_dtype: str) -> None:
        if output_dtype == "bfloat16":
            self.output_dtype = torch.bfloat16
        elif output_dtype == "float32":
            self.output_dtype = torch.float32
        else:
            self.output_dtype = torch.bfloat16  # default

    def _set_scaled_mm_call_config(
        self,
        *,
        scale_recipe_a: ScalingType | list[ScalingType],
        scale_recipe_b: ScalingType | list[ScalingType],
        swizzle_a: SwizzleType | list[SwizzleType] | None = None,
        swizzle_b: SwizzleType | list[SwizzleType] | None = None,
    ) -> None:
        # Store call-time config so forward() is a single straight-line call.
        self._scale_recipe_a = scale_recipe_a
        self._scale_recipe_b = scale_recipe_b
        self._swizzle_a = swizzle_a
        self._swizzle_b = swizzle_b

    def _init_fp8_tensorwise(self, M: int, N: int, K: int, device: str, helpers: ModuleType) -> None:
        self.float8_dtype = _get_float8_dtype(self._float8_dtype_arg)

        # Base tensors carry grad in backward benches; fp8 tensors are created as leaves.
        x_base = torch.randn(
            M,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        )
        y_base = torch.randn(
            N,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        ).t()

        # Tensorwise scales; detach so the backward bench doesn't include scale computation.
        x_scale = helpers.tensor_to_scale(x_base, self.float8_dtype).float().detach()
        y_scale = helpers.tensor_to_scale(y_base, self.float8_dtype).float().detach()

        # Quantize with the same saturation logic as the reference tests.
        with torch.no_grad():
            x_lp = (
                helpers.to_fp8_saturated(x_base * x_scale, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )
            y_lp = (
                helpers.to_fp8_saturated(y_base * y_scale, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )

        self.inputs = {
            "x": x_lp,
            "y": y_lp,
            "scale_a": x_scale.reciprocal(),
            "scale_b": y_scale.reciprocal(),
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.TensorWise,
            scale_recipe_b=ScalingType.TensorWise,
        )

    def _init_fp8_rowwise(self, M: int, N: int, K: int, device: str, helpers: ModuleType) -> None:
        # Row-wise scaling (per-row A scales and per-column B scales).
        # Mirrors `test_scaled_mm_vs_emulated_row_wise` in test_scaled_matmul_cuda.py.
        self.float8_dtype = _get_float8_dtype(self._float8_dtype_arg)

        x_base = torch.randn(
            M,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        )
        # Start from (N, K) and transpose so mat_b is (K, N), matching scaled_mm signature.
        y_base = torch.randn(
            N,
            K,
            device=device,
            dtype=self.base_dtype,
            requires_grad=self.auto_set(),
        ).t()

        x_scales = helpers.tensor_to_scale(x_base, self.float8_dtype, dim=1).float().detach()  # (M, 1)
        y_scales = helpers.tensor_to_scale(y_base, self.float8_dtype, dim=0).float().detach()  # (1, N)

        with torch.no_grad():
            x_lp = (
                helpers.to_fp8_saturated(x_base * x_scales, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )
            y_lp = (
                helpers.to_fp8_saturated(y_base * y_scales, self.float8_dtype)
                .detach()
                .requires_grad_(self.auto_set())
            )

        self.inputs = {
            "x": x_lp,
            "y": y_lp,
            "scale_a": x_scales.reciprocal(),
            "scale_b": y_scales.reciprocal(),
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.RowWise,
            scale_recipe_b=ScalingType.RowWise,
        )

    def _init_fp8_blockwise_1x128(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # FP8 blockwise scaling with 1x128 blocks.
        # Scale layout mirrors `test_scaled_mm_block_wise_numerics` for the 1x128 case.
        self.float8_dtype = _get_float8_dtype(self._float8_dtype_arg)
        if device == "cuda" and torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError(
                "FP8 BlockWise1x128 (DeepSeek style) scaling is only supported on CUDA SM90 (H100)."
            )
        if K % self._FP8_BLOCK_K != 0:
            raise RuntimeError(f"FP8 BlockWise1x128 requires K divisible by {self._FP8_BLOCK_K}, got K={K}")

        x_hp = torch.randn(M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())
        y_hp = torch.randn(N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())

        with torch.no_grad():
            x_lp, x_scales = helpers.tensor_to_scale_block(x_hp, self.float8_dtype, 1, self._FP8_BLOCK_K)
            y_lp, y_scales = helpers.tensor_to_scale_block(y_hp, self.float8_dtype, 1, self._FP8_BLOCK_K)

            x_lp = x_lp.detach().requires_grad_(self.auto_set())
            y_lp = y_lp.detach().requires_grad_(self.auto_set())

        # 1x128 requires "outer-dim-major" scales: (M, K//128) with stride ~ (1, M).
        x_scales = x_scales.t().contiguous().t().detach()
        y_scales = y_scales.t().contiguous().t().detach()

        self.inputs = {
            "x": x_lp,
            "y": y_lp.t(),  # mat_b is (K, N)
            "scale_a": x_scales.reciprocal(),
            "scale_b": y_scales.reciprocal(),
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.BlockWise1x128,
            scale_recipe_b=ScalingType.BlockWise1x128,
        )

    def _init_fp8_blockwise_128x128(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # FP8 blockwise scaling with 128x128 blocks.
        # Scale layout mirrors `test_scaled_mm_block_wise_numerics` for the 128x128 case.
        self.float8_dtype = _get_float8_dtype(self._float8_dtype_arg)
        if device == "cuda" and torch.cuda.get_device_capability(0) != (9, 0):
            raise RuntimeError(
                "FP8 BlockWise128x128 (DeepSeek style) scaling is only supported on CUDA SM90 (H100)."
            )
        if (M % self._FP8_BLOCK_K) != 0 or (N % self._FP8_BLOCK_K) != 0 or (K % self._FP8_BLOCK_K) != 0:
            raise RuntimeError(
                f"FP8 BlockWise128x128 requires M,N,K divisible by {self._FP8_BLOCK_K}, got M={M}, N={N}, K={K}"
            )

        x_hp = torch.randn(M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())
        y_hp = torch.randn(N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())

        with torch.no_grad():
            x_lp, x_scales = helpers.tensor_to_scale_block(
                x_hp, self.float8_dtype, self._FP8_BLOCK_K, self._FP8_BLOCK_K
            )
            y_lp, y_scales = helpers.tensor_to_scale_block(
                y_hp, self.float8_dtype, self._FP8_BLOCK_K, self._FP8_BLOCK_K
            )
            x_lp = x_lp.detach().requires_grad_(self.auto_set())
            y_lp = y_lp.detach().requires_grad_(self.auto_set())

        # For 128x128, scales need L padded to L4 (multiple of 4), then transposed:
        #   scales: [M//128, L] -> pad -> [M//128, L4] -> transpose -> [L4, M//128]
        x_scales, _ = helpers._pad_128x128_scales(x_scales.detach())
        y_scales, _ = helpers._pad_128x128_scales(y_scales.detach())
        x_scales = x_scales.t()
        y_scales = y_scales.t()

        self.inputs = {
            "x": x_lp,
            "y": y_lp.t(),  # mat_b is (K, N)
            "scale_a": x_scales.reciprocal(),
            "scale_b": y_scales.reciprocal(),
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.BlockWise128x128,
            scale_recipe_b=ScalingType.BlockWise128x128,
        )

    def _init_mx_blockwise(self, M: int, N: int, K: int, device: str, *, mx_format: str) -> None:
        # MX uses BlockWise1x32 with swizzled scales on CUDA.
        if device != "cuda":
            raise RuntimeError(f"MX scaling requires CUDA device, got: {device}")
        if torch.version.hip is not None:
            raise RuntimeError("MXFP benchmarks are only wired for CUDA swizzled scales (non-HIP).")

        # Important cuBLASLt requirement: mat_b must be column-major.
        # We satisfy this by passing a transpose view (non-contiguous) for `mat_b`.
        #
        # NOTE: we intentionally import from torch.testing._internal to reuse the exact
        # reference implementation used by test/test_scaled_matmul_cuda.py.
        from torch.testing._internal.common_quantized import to_blocked, to_mxfp

        x_hp = torch.randn(M, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())
        y_hp = torch.randn(N, K, device=device, dtype=self.base_dtype, requires_grad=self.auto_set())

        scale_a, x_lp = to_mxfp(x_hp.contiguous(), block_size=self._MX_BLOCK_SIZE, format=mx_format)
        scale_b, y_lp = to_mxfp(y_hp.contiguous(), block_size=self._MX_BLOCK_SIZE, format=mx_format)

        scale_a = to_blocked(scale_a)
        scale_b = to_blocked(scale_b)

        self.inputs = {
            "x": x_lp,
            "y": y_lp.t(),  # column-major mat_b
            "scale_a": scale_a,
            "scale_b": scale_b,
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
        )

    def _init_nvfp4_blockwise_and_tensorwise(
        self, M: int, N: int, K: int, device: str, helpers: ModuleType
    ) -> None:
        # NVFP4 uses packed fp4 inputs and two-level scaling:
        # - blockwise (1x16) decode scales (swizzled for CUDA)
        # - tensorwise (global) decode scale (NO_SWIZZLE)
        #
        # scaled_mm expects these as LISTS for both `scale_*` and `scale_recipe_*`.
        if device != "cuda":
            raise RuntimeError(f"NVFP4 scaling requires CUDA device, got: {device}")
        if torch.version.hip is not None:
            raise RuntimeError("NVFP4 benchmarks are only wired for CUDA (non-HIP).")
        if K % 32 != 0:
            raise RuntimeError(f"NVFP4 requires K divisible by 32, got K={K}")

        # NOTE: We reuse the same reference implementation as test/test_scaled_matmul_cuda.py.
        from torch.testing._internal.common_quantized import to_blocked

        # Use nontrivial distribution so scaling isn't degenerate.
        a_ref = torch.randn((M, K), device=device, dtype=self.base_dtype) * 1000
        b_ref = torch.randn((N, K), device=device, dtype=self.base_dtype) * 1000

        a_lp, a_scale, a_global_scale = helpers.data_to_nvfp4_with_global_scale(
            a_ref, self._NVFP4_BLOCK_SIZE
        )
        b_lp, b_scale, b_global_scale = helpers.data_to_nvfp4_with_global_scale(
            b_ref, self._NVFP4_BLOCK_SIZE
        )

        a_scale = to_blocked(a_scale)
        b_scale = to_blocked(b_scale)

        self.inputs = {
            "x": a_lp,
            "y": b_lp.t(),
            "scale_a": [a_scale, a_global_scale],
            "scale_b": [b_scale, b_global_scale],
        }
        self._set_scaled_mm_call_config(
            scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
            swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
            swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        )

    def init(
        self,
        M,
        N,
        K,
        device,
        float8_dtype="e4m3fn",
        output_dtype="bfloat16",
        scaling="fp8_tensorwise",
    ):
        helpers = _get_test_scaled_matmul_cuda()
        self._float8_dtype_arg = float8_dtype
        self.base_dtype = torch.bfloat16
        self.scaling = scaling
        self.float8_dtype = None

        self._set_output_dtype(output_dtype)

        if scaling == "fp8_tensorwise":
            self._init_fp8_tensorwise(M, N, K, device, helpers)
        elif scaling == "fp8_rowwise":
            self._init_fp8_rowwise(M, N, K, device, helpers)
        elif scaling == "fp8_blockwise_1x128":
            self._init_fp8_blockwise_1x128(M, N, K, device, helpers)
        elif scaling == "fp8_blockwise_128x128":
            self._init_fp8_blockwise_128x128(M, N, K, device, helpers)
        elif scaling == "mxfp8":
            self._init_mx_blockwise(M, N, K, device, mx_format="mxfp8")
        elif scaling == "mxfp4":
            self._init_mx_blockwise(M, N, K, device, mx_format="mxfp4")
        elif scaling == "nvfp4":
            self._init_nvfp4_blockwise_and_tensorwise(M, N, K, device, helpers)
        else:
            raise ValueError(f"Unsupported scaling mode: {scaling}")

        self.set_module_name("scaled_mm")

    def forward(self, x, y, scale_a, scale_b):
        kwargs = {
            "scale_a": scale_a,
            "scale_recipe_a": self._scale_recipe_a,
            "scale_b": scale_b,
            "scale_recipe_b": self._scale_recipe_b,
            "output_dtype": self.output_dtype,
        }
        if self._swizzle_a is not None:
            kwargs["swizzle_a"] = self._swizzle_a
        if self._swizzle_b is not None:
            kwargs["swizzle_b"] = self._swizzle_b

        return torch.nn.functional.scaled_mm(x, y, **kwargs)

# FP8 matmul only supports E4M3.
if _should_generate_scaled_mm_configs():
    scaled_mm_configs_short = op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=[
            [16384, 2048, 7168],   # DSv3 671B
            [16384, 8192, 5120],   # Llama4 16e
        ],
        cross_product_configs={
            "device": ["cuda"],
            "float8_dtype": ["e4m3fn"],
            "output_dtype": ["bfloat16"],
        },
        tags=["short"],
    )
else:
    scaled_mm_configs_short = []

# Reduced shapes for faster runs.
_scaled_mm_long_shapes = [
    [2048, 2048, 2048],
    [4096, 2048, 2048],
    [16384, 2048, 7168],
    [16384, 8192, 5120],
]

# Build long configs in groups so we can gate unsupported (scaling, output_dtype)
# combinations based on the running platform.
scaled_mm_configs_long = []

if _should_generate_scaled_mm_configs():
    # FP8 tensorwise supports bf16 and fp32 output.
    scaled_mm_configs_long += op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=_scaled_mm_long_shapes,
        cross_product_configs={
            "device": ["cuda"],
            "float8_dtype": ["e4m3fn"],
            "output_dtype": ["bfloat16"],
            "scaling": ["fp8_tensorwise"],
        },
        tags=["long"],
    )

    # FP8 rowwise: keep bf16-only for now.
    rowwise_output_dtypes = ["bfloat16"]
    scaled_mm_configs_long += op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=_scaled_mm_long_shapes,
        cross_product_configs={
            "device": ["cuda"],
            "float8_dtype": ["e4m3fn"],
            "output_dtype": rowwise_output_dtypes,
            "scaling": ["fp8_rowwise"],
        },
        tags=["long"],
    )

    # MX + NVFP4: keep bf16-only for now.
    scaled_mm_configs_long += op_bench.config_list(
        attr_names=["M", "N", "K"],
        attrs=_scaled_mm_long_shapes,
        cross_product_configs={
            "device": ["cuda"],
            "float8_dtype": ["e4m3fn"],
            "output_dtype": ["bfloat16"],
            "scaling": ["mxfp8", "mxfp4", "nvfp4"],
        },
        tags=["long"],
    )

    # DeepSeek FP8 blockwise (1x128 / 128x128) is SM90-only.
    if _supports_fp8_deepseek_blockwise_scaling():
        scaled_mm_configs_long += op_bench.config_list(
            attr_names=["M", "N", "K"],
            attrs=_scaled_mm_long_shapes,
            cross_product_configs={
                "device": ["cuda"],
                "float8_dtype": ["e4m3fn"],
                "output_dtype": ["bfloat16"],
                "scaling": ["fp8_blockwise_1x128", "fp8_blockwise_128x128"],
            },
            tags=["long"],
        )

# Generate tests for scaled_mm
op_bench.generate_pt_test(
    scaled_mm_configs_short + scaled_mm_configs_long, ScaledMMBenchmark
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
