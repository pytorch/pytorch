from pt import configs  # noqa: F401

import operator_benchmark as op_bench

import importlib.util
import os
from types import ModuleType
from typing import Optional

import torch
from torch.nn.functional import ScalingType
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.torch_version import TorchVersion

"""
Operator microbenchmarks for `scaled_grouped_mm`.

This benchmark focuses on the MXFP8/MXFP4/NVFP4 grouped-K path (offs along K),
reusing the same conversion helpers as `test/test_scaled_matmul_cuda.py`.
"""

_TEST_SCALED_MATMUL_CUDA_MOD: Optional[ModuleType] = None


def _should_generate_scaled_grouped_mm_configs() -> bool:
    # `torch._scaled_grouped_mm` / `torch.nn.functional.scaled_grouped_mm` are only available in torch 2.9+.
    if TorchVersion(torch.__version__) < "2.9":
        return False
    if not hasattr(torch.nn.functional, "scaled_grouped_mm"):
        return False

    # If CUDA isn't available on the current machine, keep configs around and let the
    # operator_benchmark harness skip CUDA tests (see benchmark_core._build_test).
    if not torch.cuda.is_available():
        return True

    if torch.version.hip is not None:
        # ROCm: MI300+ (rely on shared platform check used by tests)
        return bool(PLATFORM_SUPPORTS_FP8)

    # CUDA: compute capability must be exactly one of [9.0, 10.0]
    try:
        cc = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return cc in {(9, 0), (10, 0)}


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


class ScaledGroupedMMBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, G, device, scaling="mxfp8", output_dtype="bfloat16"):
        helpers = _get_test_scaled_matmul_cuda()

        if output_dtype != "bfloat16":
            raise ValueError("scaled_grouped_mm benchmark currently supports bfloat16 output only")
        self.output_dtype = torch.bfloat16

        if device != "cuda":
            raise ValueError("scaled_grouped_mm benchmark is CUDA-only")

        if scaling not in ("mxfp8", "mxfp4", "nvfp4"):
            raise ValueError(f"Unsupported scaling format: {scaling}")
        self.scaling = scaling

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
        self.set_module_name("scaled_grouped_mm")

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
scaled_grouped_mm_configs_short = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [256, 256, 1024, 4],
        [512, 256, 2048, 4],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "scaling": ["mxfp8", "mxfp4", "nvfp4"],
        "output_dtype": ["bfloat16"],
    },
    tags=["short"],
)

scaled_grouped_mm_configs_long = op_bench.config_list(
    attr_names=["M", "N", "K", "G"],
    attrs=[
        [2048, 2048, 2048, 4],
        [4096, 2048, 2048, 4],
    ],
    cross_product_configs={
        "device": ["cuda"],
        "scaling": ["mxfp8", "mxfp4", "nvfp4"],
        "output_dtype": ["bfloat16"],
    },
    tags=["long"],
)

if _should_generate_scaled_grouped_mm_configs():
    _scaled_grouped_mm_configs = scaled_grouped_mm_configs_short + scaled_grouped_mm_configs_long
else:
    _scaled_grouped_mm_configs = []

op_bench.generate_pt_test(_scaled_grouped_mm_configs, ScaledGroupedMMBenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()


