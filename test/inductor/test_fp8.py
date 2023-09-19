# Owner(s): ["module: inductor"]

import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CUDA

torch.set_float32_matmul_precision("high")


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

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
        ):
            x = torch.rand(*x_shape, device="cuda").to(dtype=torch.float8_e4m3fn)
            y = compiled_fp8_cast(x, torch.float8_e5m2)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
        ):
            x = torch.rand(*x_shape, device="cuda").to(dtype=torch.float8_e5m2)
            y = compiled_fp8_cast(x, torch.float8_e4m3fn)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
