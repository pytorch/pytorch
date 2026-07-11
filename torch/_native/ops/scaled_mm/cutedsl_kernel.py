"""CuTeDSL MXFP8 tensor-core kernel for logical M=1..8."""

import hashlib
from functools import cache
from pathlib import Path

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU

import torch
from torch._inductor.kernel.vendored_templates.cutedsl import (
    dense_blockscaled_gemm_persistent,
)
from torch._native.instrumentation import instrumented_cutedsl_cache


_MMA_N = 8
_SUPPORTED_CAPABILITIES = {(10, 0), (10, 3)}
_SOURCE_FINGERPRINT = hashlib.sha256(
    Path(__file__).read_bytes()
    + Path(dense_blockscaled_gemm_persistent.__file__).read_bytes()
).hexdigest()


def _blocked_scale_numel(rows: int, k: int) -> int:
    return ((rows + 127) // 128) * 128 * (((k // 32) + 3) // 4) * 4


@cache
def _device_compile_properties(device: int) -> tuple[tuple[int, int], int]:
    with torch.cuda.device(device):
        return (
            torch.cuda.get_device_capability(device),
            cutlass.utils.HardwareInfo().get_max_active_clusters(1),
        )


@instrumented_cutedsl_cache(
    "aten::_scaled_mm_v2",
    key_fn=lambda device, capability, max_clusters, n, k, source: (
        f"mxfp8_small_m device={device} capability={capability} "
        f"clusters={max_clusters} N={n} K={k} source={source[:12]}"
    ),
)
def _compile_mxfp8_small_m(
    device: int,
    capability: tuple[int, int],
    max_active_clusters: int,
    n: int,
    k: int,
    source_fingerprint: str,
):
    """Compile one hardware- and N/K-specialized kernel with runtime M."""
    with torch.cuda.device(device):
        kernel = (
            dense_blockscaled_gemm_persistent.Sm100BlockScaledPersistentDenseGemmKernel(
                sf_vec_size=32,
                mma_tiler_mn=(128, _MMA_N),
                cluster_shape_mn=(1, 1),
            )
        )
        weight = cute.runtime.make_fake_tensor(
            Float8E4M3FN, (n, k), stride=(k, 1), assumed_align=16
        )
        logical_m = cute.sym_int()
        q_input_t = cute.runtime.make_fake_tensor(
            Float8E4M3FN, (k, logical_m), stride=(1, k), assumed_align=16
        )
        weight_scale = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (_blocked_scale_numel(n, k),),
            stride=(1,),
            assumed_align=32,
        )
        input_scale = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (_blocked_scale_numel(_MMA_N, k),),
            stride=(1,),
            assumed_align=32,
        )
        output_t = cute.runtime.make_fake_tensor(
            BFloat16, (n, logical_m), stride=(1, n), assumed_align=16
        )
        return cute.compile(
            kernel,
            weight,
            q_input_t,
            weight_scale,
            input_scale,
            output_t,
            max_active_clusters,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 2",
        )


def mxfp8_small_m_scaled_mm(
    q_input: torch.Tensor,
    weight_t: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write ``q_input @ weight_t`` directly into a contiguous BF16 output."""
    k = q_input.shape[1]
    n = weight_t.shape[1]
    device = q_input.device.index
    if device is None:
        device = torch.cuda.current_device()

    def launch() -> None:
        capability, max_active_clusters = _device_compile_properties(device)
        if capability not in _SUPPORTED_CAPABILITIES:
            raise RuntimeError(
                f"small-M MXFP8 scaled_mm requires SM100 or SM103, got {capability}"
            )
        tensors = (
            (q_input, 16, "q_input"),
            (weight_t, 16, "weight"),
            (input_scale, 32, "input scale"),
            (weight_scale, 32, "weight scale"),
            (output, 16, "output"),
        )
        for tensor, alignment, name in tensors:
            if tensor.data_ptr() % alignment:
                raise RuntimeError(f"{name} must be {alignment}-byte aligned")
        _compile_mxfp8_small_m(
            device,
            capability,
            max_active_clusters,
            n,
            k,
            _SOURCE_FINGERPRINT,
        )(
            weight_t.T,
            q_input.T,
            weight_scale,
            input_scale,
            output.T,
        )

    if device == torch.cuda.current_device():
        launch()
    else:
        with torch.cuda.device(device):
            launch()
    return output
