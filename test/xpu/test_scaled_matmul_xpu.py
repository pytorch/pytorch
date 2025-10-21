# Owner(s): ["module: intel"]

from typing import Optional

import torch
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_device_type import (
    E4M3_MAX_POS,
    e4m3_type,
    E5M2_MAX_POS,
    e5m2_type,
)
from torch.testing._internal.common_utils import TestCase


f8_msg = "FP8 is not supported on this device"
f8_grouped_msg = "FP8 grouped is not supported on this device"
# avoid division by zero when calculating scale
EPS = 1e-12


def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
):
    """Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: the float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == e4m3_type:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == e5m2_type:
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)

    scale.copy_(res)
    return scale


def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype, dim=None):
    if dim is None:
        amax = torch.max(torch.abs(x))
    else:
        amax = torch.max(torch.abs(x), dim=dim, keepdim=True).values

    return amax_to_scale(amax, float8_dtype, x.dtype)


def tensor_to_scale_block(
    x: torch.Tensor,
    float8_dtype: torch.dtype,
    block_outer: int,
    block_inner: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.unflatten(1, (-1, block_inner)).unflatten(0, (-1, block_outer))
    amax = x.abs().amax(dim=[1, 3], keepdim=True).float()
    scale = torch.finfo(float8_dtype).max / amax
    x = x.mul(scale).to(float8_dtype)
    x = x.flatten(2, 3).flatten(0, 1)
    scale = scale.flatten(2, 3).flatten(0, 1)
    return x, scale


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def infer_scale_swizzle(mat, scale):
    # Currently, XPU does not support SWIZZLE, so there will always be NO_SWIZZLE
    # Tensor-wise
    if scale.numel() == 1:
        return ScalingType.TensorWise, SwizzleType.NO_SWIZZLE

    # Row-wise
    if (scale.shape[0] == mat.shape[0] and scale.shape[1] == 1) or (
        scale.shape[0] == 1 and scale.shape[1] == mat.shape[1]
    ):
        return ScalingType.RowWise, SwizzleType.NO_SWIZZLE

    # TODO: Other scaling type, like BlockWise / MX / NV dtype are not supported in this PR.
    # So we simply return None for them.

    return None, None


def scaled_mm_wrap(
    a,
    b,
    scale_a,
    scale_b,
    scale_result=None,
    out_dtype=torch.bfloat16,
    use_fast_accum=False,
    bias=None,
):
    # Basically this don't need to be wrapped. Since there is a new API for scaled_mm_v2
    # We keep it here for future extension.
    # See https://github.com/pytorch/pytorch/pull/164142
    return torch._scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        scale_result=scale_result,
        out_dtype=out_dtype,
        bias=bias,
        use_fast_accum=use_fast_accum,
    )


def mm_float8_emulated(x, x_scale, y, y_scale, out_dtype) -> torch.Tensor:
    # naive implementation: dq -> op -> q
    x_fp32 = x.to(torch.float) / x_scale
    y_fp32 = y.to(torch.float) / y_scale
    out_fp32 = torch.mm(x_fp32, y_fp32)

    return out_fp32.to(out_dtype)


def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a_inverse_scale = a_scale.reciprocal()
    b_inverse_scale = b_scale.reciprocal()
    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = scaled_mm_wrap(
            a_data,
            b_data,
            scale_a=a_inverse_scale,
            scale_b=b_inverse_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
        )
        output += bias
        return output
    output = scaled_mm_wrap(
        a_data,
        b_data,
        bias=bias,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        scale_result=output_scale,
        out_dtype=output_dtype,
    )
    return output


class TestFP8Matmul(TestCase):
    def _test_tautological_mm(
        self,
        device: str = "xpu",
        x_dtype: torch.dtype = e4m3_type,
        y_dtype: torch.dtype = e4m3_type,
        out_dtype: Optional[torch.dtype] = None,
        size: int = 16,
    ) -> None:
        x_fp8 = torch.rand(size, size, device=device).to(x_dtype)
        y_fp8 = torch.eye(size, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = scaled_mm_wrap(x_fp8, y_fp8, scale_a, scale_b, out_dtype=out_dtype)
        if out_dtype is not None:
            self.assertEqual(out_dtype, out_fp8.dtype)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    def test_float8_basics(self, device="xpu") -> None:
        self._test_tautological_mm(device, e4m3_type, e4m3_type, size=16)

        self._test_tautological_mm(device, e5m2_type, e5m2_type)

        self._test_tautological_mm(device, e4m3_type, e5m2_type, size=32)
        self._test_tautological_mm(device, e5m2_type, e4m3_type, size=48)

        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)

        with self.assertRaises(AssertionError if device == "xpu" else RuntimeError):
            self._test_tautological_mm(device, out_dtype=e5m2_type)

    def test_float8_scale(self, device="xpu") -> None:
        size = (16, 16)
        x = torch.full(size, 0.5, device=device, dtype=e4m3_type)
        # hipblaslt does not yet support mixed e4m3_type input
        y_type = e4m3_type if torch.version.hip else e5m2_type
        y = torch.full(size, 0.5, device=device, dtype=y_type).t()
        scale_one = torch.tensor(1.0, device=device)
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8 = scaled_mm_wrap(x, y, scale_a=scale_one, scale_b=scale_one)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4.0, device=device))
        out_fp8_s = scaled_mm_wrap(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)
