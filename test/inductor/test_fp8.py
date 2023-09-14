# Owner(s): ["module: inductor"]

import unittest

import torch
from torch import Tensor
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CUDA

torch.set_float32_matmul_precision("high")


# Utility functions are copied from
# https://github.com/pytorch-labs/float8_playground/blob/main/float8_playground/float8_utils.py.
# define the e4m3/e5m2 constants
E4M3_MAX_POS = 448.0
E5M2_MAX_POS = 57344.0

def _to_fp8_saturated(x: Tensor, float8_dtype: torch.dtype) -> Tensor:
    # The default behavior in PyTorch for casting to `float8_e4m3fn`
    # and `e5m2` is to not saturate. In this context, we should saturate.
    # A common case where we want to saturate is when the history of a
    # tensor has a maximum value of `amax1`, and the current amax value
    # is `amax2`, where `amax1 < amax2`. This is common when using delayed
    # scaling.
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1*E4M3_MAX_POS, max=E4M3_MAX_POS)
    else:
        x = x.clamp(min=-1*E5M2_MAX_POS, max=E5M2_MAX_POS)
    return x.to(float8_dtype)


@instantiate_parametrized_tests
class TestFP8Types(TestCase):
    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_eager_fallback(self, dtype: torch.dtype):
        weight_shape = (32, 16)

        def fp8_matmul_unwrapped(x):
            a_scale = torch.Tensor([1.0]).to(device="cuda")
            b_scale = torch.Tensor([1.0]).to(device="cuda")
            output_scale = None
            input_bias = torch.rand(32, device="cuda", dtype=dtype)
            weight = torch.rand(*weight_shape, device="cuda", dtype=dtype).T.to(
                torch.float8_e4m3fn
            )
            a_inverse_scale = 1 / a_scale
            b_inverse_scale = 1 / b_scale
            output, updated_amax = torch._scaled_mm(
                x,
                weight,
                bias=input_bias,
                out_dtype=dtype,
                scale_a=a_inverse_scale,
                scale_b=b_inverse_scale,
                scale_result=output_scale,
            )
            return output

        compiled_fp8_matmul = torch.compile(
            fp8_matmul_unwrapped, backend="inductor", dynamic=True
        )

        x_shape = (16, 16)
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(torch.float8_e4m3fn)
        y_fp8 = compiled_fp8_matmul(x)

        x_shape = (15, 16)
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(torch.float8_e4m3fn)
        y_fp8 = compiled_fp8_matmul(x)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    def test_valid_cast(self, dtype: torch.dtype):
        def fp8_cast(x):
            y0 = x.to(dtype=torch.float8_e4m3fn).to(dtype)
            y1 = x.to(dtype=torch.float8_e5m2).to(dtype)
            return y0, y1

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        x_shape = (16, 16, 16)
        x = torch.rand(*x_shape, device="cuda", dtype=dtype)
        y0_fp8, y1_fp8 = compiled_fp8_cast(x)

        torch.testing.assert_close(y0_fp8, x, rtol=5e-1, atol=5e-1)
        torch.testing.assert_close(y1_fp8, x, rtol=5e-1, atol=5e-1)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    def test_bad_cast(self):
        def fp8_cast(x, dtype):
            return x.to(dtype=dtype)

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        x_shape = (16, 16, 16)

        with self.assertRaises(Exception):
            x = torch.rand(*x_shape, device="cuda").to(dtype=torch.float8_e4m3fn)
            y = compiled_fp8_cast(x, torch.float8_e5m2)

        with self.assertRaises(Exception):
            x = torch.rand(*x_shape, device="cuda").to(dtype=torch.float8_e5m2)
            y = compiled_fp8_cast(x, torch.float8_e4m3fn)


    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    def test_layernorm_fp8_quant(self, float8_dtype: torch.dtype):
        hidden_size = 4096
        sequence_length = 2048
        batch_size = 4

        def ln_fp8(x: Tensor, scale: float, amax_buffer: Tensor):
            x = torch.nn.functional.layer_norm(x, [hidden_size], weight=None, bias=None, eps=1e-05)
            amax_buffer.fill_(torch.max(torch.abs(x)))
            x_scaled = x * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_ln_fp8_quant = torch.compile(ln_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        scale = 0.2

        amax_buffer_compiled = torch.zeros((1), device="cuda", dtype=torch.half)
        y_compiled = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
        amax_buffer = torch.zeros((1), device="cuda", dtype=torch.half)
        y = ln_fp8(x, scale, amax_buffer)

        torch.testing.assert_close(y_compiled, y, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
