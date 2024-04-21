# Owner(s): ["module: inductor"]

import functools
import unittest

import torch
from torch import Tensor
from torch._inductor import utils
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CUDA

torch.set_float32_matmul_precision("high")


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
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    else:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
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
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    def test_valid_cast(self, dtype: torch.dtype, shape: str):
        def fp8_cast(x):
            y0 = x.to(dtype=torch.float8_e4m3fn).to(dtype)
            y1 = x.to(dtype=torch.float8_e5m2).to(dtype)
            return y0, y1

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device="cuda", dtype=dtype)
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

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("src_dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("dst_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("16,16,16", "4,2048,4096"))
    def test_to_fp8_saturated(
        self, src_dtype: torch.dtype, dst_dtype: torch.dtype, shape: str
    ):
        def fp8_saturated(x, dtype):
            return _to_fp8_saturated(x, dtype)

        compiled_fp8_cast = torch.compile(
            fp8_saturated, backend="inductor", dynamic=True
        )
        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device="cuda", dtype=src_dtype)
        y_compiled = compiled_fp8_cast(x, dst_dtype)
        y = fp8_saturated(x, dst_dtype)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=5e-1, atol=5e-1)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_amax_fp8_quant(self, float8_dtype: torch.dtype, shape: str):
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def amax_fp8(x: Tensor, scale: Tensor):
            y = torch.amax(torch.abs(x))
            y_scaled = y.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(y_scaled, float8_dtype)
            return bits_fp8

        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        scale = torch.tensor(0.2, device="cuda", dtype=torch.float)

        y_compiled = compiled_amax_fp8_quant(x, scale)
        y = amax_fp8(x, scale)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-2, atol=1e-2)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_amax_along_with_fp8_quant(self, float8_dtype: torch.dtype, shape: str):
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def amax_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            amax_buffer.fill_(torch.amax(torch.abs(x)))
            x_scaled = x.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        scale = torch.tensor(1.0, device="cuda", dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device="cuda", dtype=torch.half)
        y_compiled = compiled_amax_fp8_quant(x, scale, amax_buffer_compiled)
        amax_buffer = torch.zeros((1), device="cuda", dtype=torch.half)
        y = amax_fp8(x, scale, amax_buffer)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("amax_keep_dim", (True, False))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    def test_layernorm_fp8_quant(
        self, float8_dtype: torch.dtype, amax_keep_dim: bool, shape: str
    ):
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def ln_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            x = torch.nn.functional.layer_norm(
                x.to(dtype=torch.float),
                [hidden_size],
                weight=None,
                bias=None,
                eps=1e-05,
            )
            amax_buffer.fill_(
                torch.amax(torch.abs(x), keepdim=amax_keep_dim).reshape(-1)[0]
            )
            x_scaled = x * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_ln_fp8_quant = torch.compile(ln_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        scale = torch.tensor(0.2, device="cuda", dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device="cuda", dtype=torch.half)
        y_compiled = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
        amax_buffer = torch.zeros((1), device="cuda", dtype=torch.half)
        y = ln_fp8(x, scale, amax_buffer)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("4,2048,4096",))
    def test_layernorm_fp8_quant_benchmark(
        self,
        float8_dtype: torch.dtype,
        shape: str,
    ):
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def ln(x: Tensor):
            x = torch.nn.functional.layer_norm(
                x.to(dtype=torch.float),
                [hidden_size],
                weight=None,
                bias=None,
                eps=1e-05,
            )
            return x

        def ln_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            x = torch.nn.functional.layer_norm(
                x.to(dtype=torch.float),
                [hidden_size],
                weight=None,
                bias=None,
                eps=1e-05,
            )
            amax_buffer.fill_(torch.amax(torch.abs(x)))
            x_scaled = x * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_ln_fp8_quant = torch.compile(ln_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device="cuda", dtype=torch.half)
        scale = torch.tensor(0.2, device="cuda", dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device="cuda", dtype=torch.half)
        amax_buffer = torch.zeros((1), device="cuda", dtype=torch.half)
        _ = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
        compiled_latency = utils.do_bench_using_profiling(
            functools.partial(compiled_ln_fp8_quant, x, scale, amax_buffer_compiled)
        )
        eager_latency = utils.do_bench_using_profiling(
            functools.partial(ln_fp8, x, scale, amax_buffer)
        )

        compiled_ln = torch.compile(ln, backend="inductor")
        _ = compiled_ln(x)
        ln_latency = utils.do_bench_using_profiling(functools.partial(compiled_ln, x))

        print(
            f"Config: {float8_dtype=}, {shape=}. "
            f"Benchmark results: Inductor: {compiled_latency}ms, Eager: {eager_latency}ms, "
            f"LN only Inductor: {ln_latency}ms."
        )


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
