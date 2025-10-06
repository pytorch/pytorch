# Owner(s): ["module: inductor"]

import functools
import unittest
from typing import Union

import torch
from torch import Tensor
from torch._inductor import config, utils
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
)
from torch.testing._internal.common_quantized import ceil_div, to_blocked
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    _quantize_rowwise,
    _quantize_tensorwise,
    _to_fp8_saturated,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
)
from torch.testing._internal.jit_utils import FileCheck
from torch.utils._triton import has_triton_tma_device


torch.set_float32_matmul_precision("high")


f8_msg = "FP8 is only supported on H100+, SM 8.9 and MI300+ devices"


def _fix_fp8_dtype_for_rocm(
    dtype: Union[torch.dtype, list[torch.dtype], tuple[torch.dtype]], device
) -> Union[torch.dtype, list[torch.dtype], tuple[torch.dtype]]:
    # This function is used to change FP8 data types
    # with MI300 supported FP8 types if device is GPU:
    #    e4m3fn -> e4m3fnuz
    #    e5m2   -> e5m2fnuz
    # Supports single, tuple and list of dtypes
    # Keeps the same test name for CUDA and ROCm
    # Also it allows to enable FP8 inductor tests for CPU
    if (
        torch.version.hip
        and ("cuda" in device)
        and ("gfx94" in torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])
    ):
        # MI300 uses different float8 dtypes
        if isinstance(dtype, tuple):
            return tuple(_fix_fp8_dtype_for_rocm(x, device) for x in dtype)
        if isinstance(dtype, list):
            return [_fix_fp8_dtype_for_rocm(x, device) for x in dtype]
        if dtype == torch.float8_e4m3fn:
            return torch.float8_e4m3fnuz
        elif dtype == torch.float8_e5m2:
            return torch.float8_e5m2fnuz
    return dtype


@instantiate_parametrized_tests
class TestFP8Types(TestCase):
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("device", ("cuda", "cpu"))
    def test_xblock_for_small_numel(self, float8_dtype: torch.dtype, device: str):
        """
        TritonOverrides.to_dtype will set min_elem_per_thread to 2 or 4
        depends on the variant of fp8 type.
        This cause triton_heuristics.triton_config pick a XBLOCK larger
        than numel and fail the config sanity check.

        We should not pick a XBLOCK larger than xnumel
        """
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        if device == "cuda" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)

        def f(x):
            return x.to(dtype=float8_dtype)

        x = torch.randn(1, device=device)
        expected = f(x)
        actual = torch.compile(f)(x)
        torch.testing.assert_close(expected.half(), actual.half(), rtol=1e-2, atol=1e-2)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("device", ("cuda", "cpu"))
    def test_eager_fallback(self, dtype: torch.dtype, device: torch.device):
        if device == "cuda" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        weight_shape = (32, 16)

        e4m3_type = torch.float8_e4m3fn
        e4m3_type = _fix_fp8_dtype_for_rocm(e4m3_type, device=device)

        def fp8_matmul_unwrapped(x):
            a_scale = torch.Tensor([1.0]).to(device=device)
            b_scale = torch.Tensor([1.0]).to(device=device)
            output_scale = None
            input_bias = torch.rand(32, device=device, dtype=dtype)
            weight = torch.rand(*weight_shape, device=device, dtype=dtype).T.to(
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
        x = torch.rand(*x_shape, device=device, dtype=dtype).to(e4m3_type)
        y_fp8 = compiled_fp8_matmul(x)  # noqa: F841

        x_shape = (15, 16)
        x = torch.rand(*x_shape, device=device, dtype=dtype).to(e4m3_type)
        y_fp8 = compiled_fp8_matmul(x)  # noqa: F841

    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("shape", ("15,3,13", "4,2048,4096"))
    @parametrize("dst_types", [(torch.float8_e4m3fn, torch.float8_e5m2)])
    @parametrize("device", ("cuda", "cpu"))
    def test_valid_cast(
        self, dtype: torch.dtype, shape: str, dst_types: tuple, device: torch.device
    ):
        if device == "cuda" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        dst_types = _fix_fp8_dtype_for_rocm(dst_types, device=device)
        e4m3, e5m2 = dst_types

        def fp8_cast(x):
            y0 = x.to(dtype=e4m3).to(dtype)
            y1 = x.to(dtype=e5m2).to(dtype)
            return y0, y1

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device=device, dtype=dtype)
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
            compiled_fp8_cast(x, torch.float8_e5m2)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
        ):
            x = torch.rand(*x_shape, device="cuda").to(dtype=torch.float8_e5m2)
            compiled_fp8_cast(x, torch.float8_e4m3fn)

    @parametrize("src_dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("dst_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("16,16,16", "4,2048,4096"))
    @parametrize("device", ("cuda", "cpu"))
    def test_to_fp8_saturated(
        self,
        src_dtype: torch.dtype,
        dst_dtype: torch.dtype,
        shape: str,
        device: torch.device,
    ):
        if device == "cuda" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        dst_dtype = _fix_fp8_dtype_for_rocm(dst_dtype, device=device)

        def fp8_saturated(x, dtype):
            return _to_fp8_saturated(x, dtype)

        compiled_fp8_cast = torch.compile(
            fp8_saturated, backend="inductor", dynamic=True
        )
        shape = [int(dim) for dim in shape.split(",")]
        x = torch.rand(*shape, device=device, dtype=src_dtype)
        y_compiled = compiled_fp8_cast(x, dst_dtype)
        y = fp8_saturated(x, dst_dtype)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=5e-1, atol=5e-1)

    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    @parametrize("device", ("cuda", "cpu"))
    def test_amax_fp8_quant(
        self, float8_dtype: torch.dtype, shape: str, device: torch.device
    ):
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        if device == "cuda" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(
                "FP8 is only supported on H100+ and sm_89 and MI300+ devices"
            )
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def amax_fp8(x: Tensor, scale: Tensor):
            y = torch.amax(torch.abs(x))
            y_scaled = y.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(y_scaled, float8_dtype)
            return bits_fp8

        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device=device, dtype=torch.half)
        scale = torch.tensor(0.2, device=device, dtype=torch.float)

        y_compiled = compiled_amax_fp8_quant(x, scale)
        y = amax_fp8(x, scale)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-2, atol=1e-2)

    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    @parametrize("device", ("cuda", "cpu"))
    def test_amax_along_with_fp8_quant(
        self, float8_dtype: torch.dtype, shape: str, device: torch.device
    ):
        if device == "cuda" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        shape = [int(dim) for dim in shape.split(",")]
        batch_size, sequence_length, hidden_size = shape

        def amax_fp8(x: Tensor, scale: Tensor, amax_buffer: Tensor):
            amax_buffer.fill_(torch.amax(torch.abs(x)))
            x_scaled = x.to(dtype=torch.float) * scale
            bits_fp8 = _to_fp8_saturated(x_scaled, float8_dtype)
            return bits_fp8

        compiled_amax_fp8_quant = torch.compile(amax_fp8, backend="inductor")

        x_shape = (batch_size, sequence_length, hidden_size)
        x = torch.rand(*x_shape, device=device, dtype=torch.half)
        scale = torch.tensor(1.0, device=device, dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device=device, dtype=torch.half)
        y_compiled = compiled_amax_fp8_quant(x, scale, amax_buffer_compiled)
        amax_buffer = torch.zeros((1), device=device, dtype=torch.half)
        y = amax_fp8(x, scale, amax_buffer)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )

    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("amax_keep_dim", (True, False))
    @parametrize("shape", ("1,1,15", "1,10,15", "1,10,512", "1,10,4096", "4,2048,4096"))
    @parametrize("device", ("cuda", "cpu"))
    def test_layernorm_fp8_quant(
        self,
        float8_dtype: torch.dtype,
        amax_keep_dim: bool,
        shape: str,
        device: torch.device,
    ):
        if device == "cuda" and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(
                "FP8 is only supported on H100+ and sm_89 and MI300+ devices"
            )
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
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
        x = torch.rand(*x_shape, device=device, dtype=torch.half)
        scale = torch.tensor(0.2, device=device, dtype=torch.float)

        amax_buffer_compiled = torch.zeros((1), device=device, dtype=torch.half)
        y_compiled = compiled_ln_fp8_quant(x, scale, amax_buffer_compiled)
        amax_buffer = torch.zeros((1), device=device, dtype=torch.half)
        y = ln_fp8(x, scale, amax_buffer)

        torch.testing.assert_close(y_compiled.half(), y.half(), rtol=1e-1, atol=1e-1)
        torch.testing.assert_close(
            amax_buffer_compiled, amax_buffer, rtol=1e-2, atol=1e-2
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("4,2048,4096",))
    @parametrize("keepdim", (False, True))
    def test_layernorm_fp8_quant_benchmark(
        self,
        float8_dtype: torch.dtype,
        shape: str,
        keepdim: bool,
    ):
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device="cuda")
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
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    @parametrize("shape", ("16,16,32", "16,32,32", "1024,1024,512"))
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
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

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
            if dtype == torch.bfloat16:
                self.assertEqual(y_eager, y_compiled, rtol=5e-2, atol=0.07)
            else:
                self.assertEqual(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_scaled_mm_preserves_strides(self):
        """Test that scaled_mm preserves stride ordering through a custom pass."""

        GPU_TYPE = "cuda"

        def f(a, b, scale_a, scale_b):
            # Convert to fp8 with correct strides for scaled_mm
            dtype_float8 = torch.float8_e4m3fn
            dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, GPU_TYPE)
            a_fp8 = a.to(dtype_float8).contiguous()  # row-major
            b_fp8 = b.t().contiguous().t().to(dtype_float8)  # column-major
            return torch._scaled_mm(
                a_fp8, b_fp8, scale_a, scale_b, out_dtype=torch.bfloat16
            )

        class ScaledMMStridePass(PatternMatcherPass):
            def __init__(self) -> None:
                super().__init__()
                self.called = False

            def __call__(self, g: torch.fx.Graph):
                # Directly manipulate the graph without using pattern matching
                for node in g.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == torch.ops.aten._scaled_mm.default
                    ):
                        # Insert clone operations before scaled_mm
                        with g.inserting_before(node):
                            a_fp8, b_fp8 = node.args[0], node.args[1]

                            # Clone the inputs to potentially change stride ordering
                            a_cloned = g.call_function(
                                torch.ops.aten.clone,
                                (a_fp8,),
                                {"memory_format": torch.contiguous_format},
                            )
                            b_cloned = g.call_function(
                                torch.ops.aten.clone,
                                (b_fp8,),
                                {"memory_format": torch.contiguous_format},
                            )

                            # Replace the arguments in the scaled_mm call
                            node.args = (a_cloned, b_cloned) + node.args[2:]
                            self.called = True

                g.lint()
                return g

        stride_pass = ScaledMMStridePass()

        # Create inputs with correct strides for scaled_mm
        a = torch.randn((64, 128), dtype=torch.bfloat16, device=GPU_TYPE)
        b = torch.randn((128, 64), dtype=torch.bfloat16, device=GPU_TYPE)
        scale_a = torch.tensor(1.0, device=GPU_TYPE)
        scale_b = torch.tensor(1.0, device=GPU_TYPE)

        # First, verify that f works without the pass (baseline)
        expected = f(a, b, scale_a, scale_b)

        from torch._inductor import config

        with config.patch(post_grad_custom_post_pass=stride_pass):
            f_compiled = torch.compile(f, dynamic=False)
            result = f_compiled(a, b, scale_a, scale_b)

            # Verify the pattern was called
            self.assertTrue(stride_pass.called, "Stride ordering pass was not called")

            # Verify correctness - the pass should preserve correctness
            # even though it modified strides
            self.assertEqual(expected, result, atol=1e-2, rtol=1e-2)

            # Verify the generated code contains the clones inserted by our pass
            _, (wrapper,) = run_and_get_code(f_compiled, a, b, scale_a, scale_b)
            self.assertIn("scaled_mm", wrapper.lower())
            # The clones should be visible in the generated code
            self.assertIn("clone", wrapper.lower())

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    @parametrize("shape", ("16,32,32", "1024,1024,512"))
    @parametrize("use_fast_accum", (False, True))
    def test_tensorwise_scaling_tma_template(
        self,
        dtype: torch.dtype,
        shape: str,
        use_fast_accum: bool,
    ):
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        # input and output dtypes of _scaled_mm do not need to be the same, but
        # typically in a model they are
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None

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
        with config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "triton_scaled_mm_device_tma",
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune": True,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled, code = run_and_get_code(
                linear_compiled,
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )

            FileCheck().check("SCALING_ROWWISE : tl.constexpr = False").run(code[0])
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)
            # depending on the kernel config (BLOCK_M size, etc) selected during Inductor
            # autotuning for the compiled case, the results can be different because of
            # the way blocks of results are accumulated (float addition not associative), so
            # setting a small absolute tolerance in these tests
            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("shape", ("16,16,32", "16,32,32", "1024,1024,512"))
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
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

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
        torch.testing.assert_close(y_eager, y_compiled, rtol=5e-2, atol=0.07)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("shape", ("16,32,32", "1024,1024,512"))
    @parametrize("use_fast_accum", (False, True))
    def test_rowwise_scaling_tma_template(
        self,
        shape: str,
        use_fast_accum: bool,
    ):
        # Only bf16 output type is supported for row-wise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        shape = [int(dim) for dim in shape.split(",")]
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None

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
        with config.patch(
            {
                "triton.enable_persistent_tma_matmul": True,
                "test_configs.autotune_choice_name_regex": "triton_scaled_mm_device_tma",
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune": True,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled, code = run_and_get_code(
                linear_compiled,
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )

        FileCheck().check("SCALING_ROWWISE : tl.constexpr = True").run(code[0])
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 32, 1024))
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
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

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
        torch.testing.assert_close(y_eager, y_compiled, rtol=5e-2, atol=0.07)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 32, 1024))
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
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

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

    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, "Not supported on non B200")
    def test_mx_fp8_max_autotune(self):
        M, K, N = 128, 32, 128
        BLOCK_SIZE = 32
        device = "cuda"
        dtype = torch.bfloat16
        A_ref = torch.eye(M, device=device, dtype=torch.bfloat16)
        B_ref = torch.eye(N, device=device, dtype=torch.bfloat16)
        A = A_ref.to(torch.float8_e4m3fn)
        B = B_ref.to(torch.float8_e4m3fn)
        A_scale = torch.full(
            (M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu
        )
        B_scale = torch.full(
            (N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu
        )
        A_scale = to_blocked(A_scale)
        B_scale = to_blocked(B_scale)

        def linear(A, B, A_scale, B_scale):
            y = torch._scaled_mm(
                A,
                B.t(),
                A_scale,
                B_scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=False,
            )
            return y

        y_eager = linear(A, B, A_scale, B_scale)

        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        y_compiled = linear_compiled(A, B, A_scale, B_scale)
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.07)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_unacceptable_input_dims(self):
        # for compiled ops, type checking is in torch/_meta_registrations.py
        dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

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
            linear_compiled(
                x,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertTrue(
            f"Expected self.size(1) to be divisible by 16, but got self.size(1)={K}"
            in str(cm.exception)
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_unacceptable_scale_dims_rowwise_scaling(self):
        dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

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
            linear_compiled(
                x,
                w_t_fp8,
                w_inverse_scale,
                bias,
            )
        self.assertTrue("Invalid scaling configuration." in str(cm.exception))


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON or HAS_CPU:
        run_tests()
