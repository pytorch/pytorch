# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA

isSM90orLaterDevice = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability()
    >= (
        9,
        0,
    )
)

torch.set_float32_matmul_precision("high")


@instantiate_parametrized_tests
class TestFP8Types(TestCase):
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @unittest.skipIf(not isSM90orLaterDevice, "Requires SM90 device")
    def test_eager_fallback(self, dtype: torch.dtype):
        x_shape = (16, 16)
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

        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(torch.float8_e4m3fn)
        compiled_fp8_matmul = torch.compile(fp8_matmul_unwrapped, backend="inductor")
        y_fp8 = compiled_fp8_matmul(x)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
