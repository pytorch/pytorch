# Owner(s): ["module: inductor"]

import functools
import unittest

import torch
from torch import Tensor
from torch._inductor import config, utils
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8, SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils._triton import has_triton_tma_device


torch.set_float32_matmul_precision("high")


f8_msg = "FP8 is only supported on H100+ and sm_89 and MI300+ devices"

# define the e4m3/e5m2 constants
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max
E4M3FNUZ_MAX_POS = torch.finfo(torch.float8_e4m3fnuz).max
E5M2FNUZ_MAX_POS = torch.finfo(torch.float8_e5m2fnuz).max

FP16_MAX_POS: float = torch.finfo(torch.float16).max
EPS: float = 1e-12


def _to_fp8_saturated(x: Tensor, float8_dtype: torch.dtype) -> Tensor:
    # The default behavior in PyTorch for casting to `float8_e4m3fn`
    # and `e5m2` is to not saturate. In this context, we should saturate.
    # A common case where we want to saturate is when the history of a
    # tensor has a maximum value of `amax1`, and the current amax value
    # is `amax2`, where `amax1 < amax2`. This is common when using delayed
    # scaling.
    if float8_dtype == torch.float8_e4m3fn:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif float8_dtype == torch.float8_e5m2:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    elif float8_dtype == torch.float8_e4m3fnuz:
        x = x.clamp(min=-1 * E4M3FNUZ_MAX_POS, max=E4M3FNUZ_MAX_POS)
    elif float8_dtype == torch.float8_e5m2fnuz:
        x = x.clamp(min=-1 * E5M2FNUZ_MAX_POS, max=E5M2FNUZ_MAX_POS)
    else:
        raise TypeError(f"Unsupported float8_dtype: {float8_dtype}")
    return x.to(float8_dtype)


@torch.no_grad()
def _amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
) -> torch.Tensor:
    # To make scale dtype to be fp32 for accuracy
    amax = amax.float()
    if float8_dtype == torch.float8_e4m3fn:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:  # e5m2
        res = E5M2_MAX_POS / torch.clamp(amax, min=EPS)

    # Ensure that the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16.
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=FP16_MAX_POS)
    return res


def _quantize_tensorwise(x: Tensor, float8_dtype: torch.dtype):
    amax = torch.max(torch.abs(x))
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x_fp8 = _to_fp8_saturated(x * scale, float8_dtype)
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


def _quantize_rowwise(x: Tensor, float8_dtype: torch.dtype):
    amax = torch.max(torch.abs(x), dim=1, keepdim=True).values
    scale = _amax_to_scale(amax, float8_dtype, x.dtype)
    x_fp8 = _to_fp8_saturated(x * scale, float8_dtype)
    inverse_scale = scale.reciprocal()
    return x_fp8, inverse_scale


@instantiate_parametrized_tests
class TestFP8Types(TestCase):
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported yet")
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    def test_xblock_for_small_numel(self, float8_dtype: torch.dtype):
        """
        TritonOverrides.to_dtype will set min_elem_per_thread to 2 or 4
        depends on the variant of fp8 type.
        This cause triton_heuristics.triton_config pick a XBLOCK larger
        than numel and fail the config sanity check.

        We should not pick a XBLOCK larger than xnumel
        """

        def f(x):
            return x.to(dtype=float8_dtype)

        x = torch.randn(1, device="cuda")
        expected = f(x)
        actual = torch.compile(f)(x)
        torch.testing.assert_close(expected.half(), actual.half(), rtol=1e-2, atol=1e-2)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(TEST_WITH_ROCM, "Not supported yet")
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_eager_fallback(self, dtype: torch.dtype):
        weight_shape = (32, 16)

        e4m3_type = (
            torch.float8_e4m3fn if torch.version.hip is None else torch.float8_e4m3fnuz
        )

        def fp8_matmul_unwrapped(x):
            a_scale = torch.Tensor([1.0]).to(device="cuda")
            b_scale = torch.Tensor([1.0]).to(device="cuda")
            output_scale = None
            input_bias = torch.rand(32, device="cuda", dtype=dtype)
            weight = torch.rand(*weight_shape, device="cuda", dtype=dtype).T.to(
                e4m3_type
            )
            a_inverse_scale = 1 / a_scale
            b_inverse_scale = 1 / b_scale
            output = torch._scaled_mm(
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
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(e4m3_type)
        y_fp8 = compiled_fp8_matmul(x)

        x_shape = (15, 16)
        x = torch.rand(*x_shape, device="cuda", dtype=dtype).to(e4m3_type)
        y_fp8 = compiled_fp8_matmul(x)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    @parametrize(
        "dst_types",
        [(torch.float8_e4m3fn, torch.float8_e5m2)]
        if torch.version.hip is None
        else [(torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)],
    )
    def test_valid_cast(self, dtype: torch.dtype, shape: str, dst_types: tuple):
        e4m3, e5m2 = dst_types

        def fp8_cast(x):
            y0 = x.to(dtype=e4m3).to(dtype)
            y1 = x.to(dtype=e5m2).to(dtype)
            return y0, y1

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device="cuda", dtype=dtype)
        y0_fp8, y1_fp8 = compiled_fp8_cast(x)

        torch.testing.assert_close(y0_fp8, x, rtol=5e-1, atol=5e-1)
        torch.testing.assert_close(y1_fp8, x, rtol=5e-1, atol=5e-1)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
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

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("src_dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize(
        "dst_dtype",
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
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

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm fails with accuracy issue")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize(
        "float8_dtype",
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
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

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize(
        "float8_dtype",
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
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

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm fails with accuracy issue")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize(
        "float8_dtype",
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
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

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize(
        "float8_dtype",
        (torch.float8_e4m3fn, torch.float8_e5m2)
        if torch.version.hip is None
        else (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz),
    )
    @parametrize("shape", ("4,2048,4096",))
    @parametrize("keepdim", (False, True))
    def test_layernorm_fp8_quant_benchmark(
        self,
        float8_dtype: torch.dtype,
        shape: str,
        keepdim: bool,
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
            amax = torch.amax(torch.abs(x), keepdim=keepdim)
            amax_buffer.view_as(amax).copy_(amax)
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
            f"Config: {float8_dtype=}, {shape=}, {keepdim=}. "
            f"Benchmark results: Inductor: {compiled_latency}ms, Eager: {eager_latency}ms, "
            f"LN only Inductor: {ln_latency}ms."
        )


@instantiate_parametrized_tests
class TestFP8Lowering(TestCase):
    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    @parametrize("shape", ("16,16,32", "1024,1024,512"))
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_tensorwise_scaling(
        self,
        dtype: torch.dtype,
        shape: str,
        has_bias: bool,
        use_fast_accum: bool,
        persistent_matmul: bool,
    ):
        if dtype is torch.float32 and has_bias:
            self.skipTest("bias is not supported when output dtype is float32")

        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        # input and output dtypes of _scaled_mm do not need to be the same, but
        # typically in a model they are
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_tensorwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)
            # depending on the kernel config (BLOCK_M size, etc) selected during Inductor
            # autotuning for the compiled case, the results can be different because of
            # the way blocks of results are accumulated (float addition not associative), so
            # setting a small absolute tolerance in these tests
            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("shape", ("16,16,32", "1024,1024,512"))
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_rowwise_scaling(
        self, shape: str, has_bias: bool, use_fast_accum: bool, persistent_matmul: bool
    ):
        # Only bf16 output type is supported for row-wise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
        y_compiled = linear_compiled(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 1024))
    @parametrize("N", (16, 2048))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_tensorwise_scaling_acceptable_input_dims(
        self, M: int, K: int, N: int, persistent_matmul: bool
    ):
        # alignment requirements: K and N divisible by 16
        dtype: torch.dtype = torch.bfloat16
        use_fast_accum = True
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn

        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        x_fp8, x_inverse_scale = _quantize_tensorwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.07)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 1024))
    @parametrize("N", (16, 2048))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_rowwise_scaling_acceptable_input_dims(
        self, M: int, K: int, N: int, persistent_matmul: bool
    ):
        dtype: torch.dtype = torch.bfloat16
        use_fast_accum = True
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn

        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)
        x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        with config.patch({"triton.enable_persistent_tma_matmul": persistent_matmul}):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.07)

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    def test_unacceptable_input_dims(self):
        # for compiled ops, type checking is in torch/_meta_registrations.py
        dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn
        M, K, N = 64, 15, 2048  # K needs to be a multiple of 16
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()

        def linear(x, w_t_fp8, w_inverse_scale, bias):
            x_fp8, x_inverse_scale = _quantize_tensorwise(x, dtype_float8)
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=True,
            )
            return y

        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        with self.assertRaises(torch._dynamo.exc.TorchRuntimeError) as cm:
            y_compiled = linear_compiled(
                x,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertTrue(
            f"Expected self.size(1) to be divisible by 16, but got self.size(1)={K}"
            in str(cm.exception)
        )

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(not SM90OrLater, "FP8 is only supported on H100+")
    def test_unacceptable_scale_dims_rowwise_scaling(self):
        dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn
        M, K, N = 233, 32, 128
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()

        def linear(x, w_t_fp8, w_inverse_scale, bias):
            x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                w_inverse_scale.t(),  # testing with w and x scales switched
                x_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=True,
            )
            return y

        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        with self.assertRaises(torch._dynamo.exc.TorchRuntimeError) as cm:
            y_compiled = linear_compiled(
                x,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertTrue("Invalid scaling configuration." in str(cm.exception))


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
