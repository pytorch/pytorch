# Owner(s): ["module: intel"]

from typing import Optional

import torch
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_device_type import (
    E4M3_MAX_POS,
    e4m3_type,
    E5M2_MAX_POS,
    e5m2_type,
    instantiate_device_type_tests,
    onlyXPU,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


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


def to_fp8_saturated(x: torch.Tensor, fp8_dtype: torch.dtype):
    if fp8_dtype == e4m3_type:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif fp8_dtype == e5m2_type:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    else:
        raise ValueError(f"to_fp8_saturated(): Unsupported fp8_dtype: {fp8_dtype}")

    return x.to(fp8_dtype)


def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the error between two tensors in dB.

    For more details see:
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        x: The original tensor.
        y: The tensor to compare to the original tensor.
    """
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


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
        y_type = e5m2_type
        y = torch.full(size, 0.5, device=device, dtype=y_type).t()
        scale_one = torch.tensor(1.0, device=device)
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8 = scaled_mm_wrap(x, y, scale_a=scale_one, scale_b=scale_one)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4.0, device=device))
        out_fp8_s = scaled_mm_wrap(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)

    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_vs_emulated(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        x = torch.randn(16, 16, device="xpu", dtype=base_dtype)
        y = torch.randn(32, 16, device="xpu", dtype=base_dtype).t()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # Calculate actual F8 mm
        out_scaled_mm = scaled_mm_wrap(
            x_fp8,
            y_fp8,
            scale_a=x_scale.reciprocal(),
            scale_b=y_scale.reciprocal(),
            out_dtype=output_dtype,
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(x_fp8, x_scale, y_fp8, y_scale, output_dtype)

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_change_stride(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        x = torch.empty_strided((16, 16), (16, 1), device="xpu", dtype=base_dtype)
        y = torch.empty_strided((16, 32), (1, 64), device="xpu", dtype=base_dtype)

        x.normal_()
        y.normal_()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # Calculate actual F8 mm
        out_scaled_mm = scaled_mm_wrap(
            x_fp8,
            y_fp8,
            scale_a=x_scale.reciprocal(),
            scale_b=y_scale.reciprocal(),
            out_dtype=output_dtype,
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(x_fp8, x_scale, y_fp8, y_scale, output_dtype)

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @onlyXPU
    def test_float8_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.ones((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), 0.25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = scaled_mm_wrap(x, y, scale_a=scale_a, scale_b=scale_b)
        outb_fp8 = scaled_mm_wrap(x, y, scale_a=scale_a, scale_b=scale_b, bias=bias)
        # this fails on ROCm currently because hipblaslt doesn't have amax op
        out_fp32 = out_fp8.to(torch.float32)
        outb_fp32 = outb_fp8.to(torch.float32)
        difference = torch.abs(out_fp32 - outb_fp32)
        self.assertEqual(
            difference, torch.tensor(4.0, device=device).expand_as(out_fp32)
        )

    @onlyXPU
    @parametrize("bias", [True, False])
    def test_non_divisible_leading_dim(self, device, bias: bool) -> None:
        x = torch.rand((17, 16), device=device).to(e4m3_type)
        y = torch.rand((16, 16), device=device).to(e4m3_type).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        input_bias = None
        if bias:
            input_bias = torch.rand((16,), device=device).to(torch.bfloat16)
        _ = scaled_mm_wrap(x, y, scale_a, scale_b, bias=input_bias)

    @onlyXPU
    def test_float8_bias_relu_edgecase(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.full((k, l), 0.0, device=device).to(e4m3_type)
        y = torch.full((m, l), 1.0, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), -3.0, device=device, dtype=torch.bfloat16)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        outb_fp8 = scaled_mm_wrap(x, y, scale_a, scale_b, bias=bias)
        outb_fp32 = outb_fp8.to(torch.float32)
        self.assertEqual(
            outb_fp32, torch.tensor(-3.0, device=device).expand_as(outb_fp32)
        )

    @onlyXPU
    def test_float32_output_errors_with_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), 0.25, device=device, dtype=e4m3_type).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        self.assertRaisesRegex(
            RuntimeError,
            "Bias is not supported when out_dtype is set to Float32",
            lambda: scaled_mm_wrap(
                x, y, scale_a, scale_b, bias=bias, out_dtype=torch.float32
            ),
        )


instantiate_device_type_tests(TestFP8Matmul, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
