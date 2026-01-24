# Owner(s): ["module: inductor"]

import functools
import unittest
from typing import Union

import torch
from torch import Tensor
from torch._C import FileCheck
from torch._inductor import config, inductor_prims, utils
from torch._inductor.fx_passes.misc_patterns import _misc_patterns_init
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.nn.functional import ScalingType  # type: ignore[attr-defined]
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    IS_SM90,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
    SM100OrLater,
    SM90OrLater,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
    onlyOn,
)
from torch.testing._internal.common_quantized import ceil_div, to_blocked
from torch.testing._internal.common_utils import parametrize, xfailIf
from torch.testing._internal.inductor_utils import (
    _quantize_blockwise,
    _quantize_rowwise,
    _quantize_tensorwise,
    _to_fp8_saturated,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    is_big_gpu,
)
from torch.utils._triton import has_triton_tma_device


torch.set_float32_matmul_precision("high")


f8_msg = "FP8 is only supported on H100+, SM 8.9 and MI300+ and XPU devices"


def _is_cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    if isinstance(device, str):
        return "cuda" in device
    return False


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
        and (_is_cuda_device(device))
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


class TestFP8Types(TestCase):
    @parametrize("float8_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    def test_xblock_for_small_numel(self, float8_dtype: torch.dtype, device: str):
        """
        TritonOverrides.to_dtype will set min_elem_per_thread to 2 or 4
        depends on the variant of fp8 type.
        This cause triton_heuristics.triton_config pick a XBLOCK larger
        than numel and fail the config sanity check.

        We should not pick a XBLOCK larger than xnumel
        """
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        if _is_cuda_device(device) and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)

        def f(x):
            return x.to(dtype=float8_dtype)

        x = torch.randn(1, device=device)
        expected = f(x)
        actual = torch.compile(f)(x)
        torch.testing.assert_close(expected.half(), actual.half(), rtol=1e-2, atol=1e-2)

    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_eager_fallback(self, dtype: torch.dtype, device: torch.device):
        if _is_cuda_device(device) and not PLATFORM_SUPPORTS_FP8:
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
    def test_valid_cast(
        self, dtype: torch.dtype, shape: str, dst_types: tuple, device: torch.device
    ):
        if _is_cuda_device(device) and not PLATFORM_SUPPORTS_FP8:
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
    def test_bad_cast(self, device):
        def fp8_cast(x, dtype):
            return x.to(dtype=dtype)

        compiled_fp8_cast = torch.compile(fp8_cast, backend="inductor", dynamic=True)

        x_shape = (16, 16, 16)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
        ):
            x = torch.rand(*x_shape, device=device).to(dtype=torch.float8_e4m3fn)
            compiled_fp8_cast(x, torch.float8_e5m2)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Conversions between float8_e5m2 and float8_e4m3fn is not supported!",
        ):
            x = torch.rand(*x_shape, device=device).to(dtype=torch.float8_e5m2)
            compiled_fp8_cast(x, torch.float8_e4m3fn)

    @parametrize("src_dtype", (torch.float16, torch.bfloat16, torch.float))
    @parametrize("dst_dtype", (torch.float8_e4m3fn, torch.float8_e5m2))
    @parametrize("shape", ("16,16,16", "4,2048,4096"))
    def test_to_fp8_saturated(
        self,
        src_dtype: torch.dtype,
        dst_dtype: torch.dtype,
        shape: str,
        device: torch.device,
    ):
        if _is_cuda_device(device) and not PLATFORM_SUPPORTS_FP8:
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
    def test_amax_fp8_quant(
        self, float8_dtype: torch.dtype, shape: str, device: torch.device
    ):
        float8_dtype = _fix_fp8_dtype_for_rocm(float8_dtype, device=device)
        if _is_cuda_device(device) and not PLATFORM_SUPPORTS_FP8:
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
    def test_amax_along_with_fp8_quant(
        self, float8_dtype: torch.dtype, shape: str, device: torch.device
    ):
        if _is_cuda_device(device) and not PLATFORM_SUPPORTS_FP8:
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
    def test_layernorm_fp8_quant(
        self,
        float8_dtype: torch.dtype,
        amax_keep_dim: bool,
        shape: str,
        device: torch.device,
    ):
        if _is_cuda_device(device) and not PLATFORM_SUPPORTS_FP8:
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

    @onlyCUDA
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

    @unittest.skipIf(
        not SM90OrLater or torch.version.hip, "PDL requires NVIDIA SM 9.0+"
    )
    @onlyOn(["cuda", "xpu"])
    def test_scaled_mm_pdl_handles_none_bias(self, device):
        dtype_float8 = _fix_fp8_dtype_for_rocm(torch.float8_e4m3fn, device)
        M, K, N = 32, 64, 32

        # A row-major, B column-major view (transpose of contiguous)
        a = torch.randn(M, K, device=device, dtype=torch.float16).to(dtype_float8)
        b = torch.randn(N, K, device=device, dtype=torch.float16).to(dtype_float8).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        scale_r = torch.tensor(1.0, device=device)

        def linear(a, b, sa, sb, sr):
            return torch._scaled_mm(a, b, sa, sb, None, sr, out_dtype=torch.bfloat16)

        expected = linear(a, b, scale_a, scale_b, scale_r)

        patch_cfg = {
            "triton.enable_pdl": True,
            "triton.use_tensor_descriptor": False,
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "max_autotune_gemm_search_space": "EXHAUSTIVE",
        }

        with config.patch(patch_cfg):
            compiled = torch.compile(linear, mode="max-autotune")
            actual = compiled(a, b, scale_a, scale_b, scale_r)

        self.assertEqual(expected, actual, rtol=5e-2, atol=0.07)


class TestFP8Lowering(TestCase):
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    @parametrize("shape", ("16,16,32", "16,32,32", "1024,1024,512"))
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    @onlyOn(["cuda", "xpu"])
    def test_tensorwise_scaling(
        self,
        dtype: torch.dtype,
        shape: str,
        has_bias: bool,
        use_fast_accum: bool,
        persistent_matmul: bool,
        device,
    ):
        if dtype is torch.float32 and has_bias:
            self.skipTest("bias is not supported when output dtype is float32")
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

        # if "xpu" in device and use_fast_accum:
        self.skipTest("XPU does not support use_fast_accum=True for now")

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
    @onlyOn(["cuda", "xpu"])
    def test_scaled_mm_preserves_strides(self, device):
        """Test that scaled_mm preserves stride ordering through a custom pass."""

        GPU_TYPE = device
        use_fast_accum = True
        if "xpu" in device:
            use_fast_accum = False

        def f(a, b, scale_a, scale_b):
            # Convert to fp8 with correct strides for scaled_mm
            dtype_float8 = torch.float8_e4m3fn
            dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, GPU_TYPE)
            a_fp8 = a.to(dtype_float8).contiguous()  # row-major
            b_fp8 = b.t().contiguous().t().to(dtype_float8)  # column-major
            return torch._scaled_mm(
                a_fp8,
                b_fp8,
                scale_a,
                scale_b,
                out_dtype=torch.bfloat16,
                use_fast_accum=use_fast_accum,
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

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(
        not has_triton_tma_device() or not is_big_gpu(),
        "Need device-side TMA support in Triton and max-autotune",
    )
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    @parametrize("shape", ("16,32,32", "1024,1024,512"))
    @parametrize("use_fast_accum", (False, True))
    def test_tensorwise_scaling_tma_template(
        self,
        dtype: torch.dtype,
        shape: str,
        use_fast_accum: bool,
        device,
    ):
        if "xpu" in device and use_fast_accum:
            self.skipTest("XPU does not support use_fast_accum=True for now")
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

            FileCheck().check(
                f"SCALE_RECIPE_A : tl.constexpr = {ScalingType.TensorWise.value}"
            ).run(code[0])
            FileCheck().check(
                f"SCALE_RECIPE_B : tl.constexpr = {ScalingType.TensorWise.value}"
            ).run(code[0])
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)
            # depending on the kernel config (BLOCK_M size, etc) selected during Inductor
            # autotuning for the compiled case, the results can be different because of
            # the way blocks of results are accumulated (float addition not associative), so
            # setting a small absolute tolerance in these tests
            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu"])
    @parametrize("shape", ("16,16,32", "16,32,32", "1024,1024,512"))
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_rowwise_scaling(
        self,
        shape: str,
        has_bias: bool,
        use_fast_accum: bool,
        persistent_matmul: bool,
        device,
    ):
        if "xpu" in device and use_fast_accum:
            self.skipTest("XPU does not support use_fast_accum=True for now")
        # Only bf16 output type is supported for row-wise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
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
        not has_triton_tma_device() or not is_big_gpu(),
        "Need device-side TMA support in Triton and max-autotune",
    )
    @onlyCUDA
    @parametrize("shape", ("16,32,32", "1024,1024,512"))
    @parametrize("use_fast_accum", (False, True))
    def test_rowwise_scaling_tma_template(
        self,
        shape: str,
        use_fast_accum: bool,
        device,
    ):
        # Only bf16 output type is supported for row-wise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
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

        FileCheck().check(
            f"SCALE_RECIPE_A : tl.constexpr = {ScalingType.RowWise.value}"
        ).run(code[0])
        FileCheck().check(
            f"SCALE_RECIPE_B : tl.constexpr = {ScalingType.RowWise.value}"
        ).run(code[0])
        self.assertEqual(y_eager.dtype, dtype)
        self.assertEqual(y_compiled.dtype, dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        _get_torch_cuda_version() < (12, 9),
        "cuBLAS blockwise scaling added in CUDA 12.9",
    )
    @onlyCUDA
    @xfailIf(
        torch.cuda.is_available() and torch.cuda.get_device_capability() != (9, 0)
    )  # cuBLAS 128-element blockwise scaling is only supported for CC 9.0
    @parametrize("shape", ((16, 256, 256), (1024, 512, 1024), (32768, 4096, 4096)))
    @parametrize("use_fast_accum", (False, True))
    @parametrize(
        "scaling_block_sizes",
        ((1, 128, 128, 128), (1, 128, 1, 128), (128, 128, 1, 128)),
    )  # (BlockWise1x128, BlockWise128x128), (BlockWise1x128, BlockWise1x128), (BlockWise128x128, BlockWise1x128)
    def test_main_loop_scaling(
        self,
        shape: tuple[int, int, int],
        use_fast_accum: bool,
        scaling_block_sizes: tuple[int, int, int, int],
        device,
    ):
        if "xpu" in device and use_fast_accum:
            self.skipTest("XPU does not support use_fast_accum=True for now")
        # Only bf16 output type is supported for non-tensorwise scaling, not fp32
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        M, N, K = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = None

        am, ak, bn, bk = scaling_block_sizes

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_blockwise(
            w, dtype_float8, block_outer=bn, block_inner=bk
        )
        w_t_fp8 = w_fp8.t()
        if (bn, bk) == (1, 128):
            w_inverse_scale = (
                w_inverse_scale.t().contiguous().t().t()
            )  # 1x128 blocks need scales to be outer-dim-major
        else:
            w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_blockwise(
            x, dtype_float8, block_outer=am, block_inner=ak
        )
        if (am, ak) == (1, 128):
            x_inverse_scale = (
                x_inverse_scale.t().contiguous().t()
            )  # 1x128 blocks need scales to be outer-dim-major

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

        # BlockWise1x128 and BlockWise128x128 scaling modes are not compatible with fast_accum
        # Only take this branch on SM90 because other versions xfail everything
        if use_fast_accum and IS_SM90:
            with self.assertRaisesRegex(
                RuntimeError, "scaled_gemm doesn't support fast accum"
            ):
                y_eager = linear(
                    x_fp8,
                    x_inverse_scale,
                    w_t_fp8,
                    w_inverse_scale,
                    bias,
                )
        else:
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

        # Verify that Inductor chooses the correct scaling recipes
        FileCheck().check(
            f"SCALE_RECIPE_A : tl.constexpr = {ScalingType.BlockWise1x128.value}"
        ).run(code[0])

        if (bn, bk) == (1, 128):
            check_scale_recipe_b = ScalingType.BlockWise1x128.value
        else:
            check_scale_recipe_b = ScalingType.BlockWise128x128.value
        FileCheck().check(
            f"SCALE_RECIPE_B : tl.constexpr = {check_scale_recipe_b}"
        ).run(code[0])

        self.assertEqual(y_compiled.dtype, dtype)
        if not use_fast_accum:
            self.assertEqual(y_eager.dtype, dtype)
            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu"])
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 32, 1024))
    @parametrize("N", (16, 2048))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_tensorwise_scaling_acceptable_input_dims(
        self, M: int, K: int, N: int, persistent_matmul: bool, device
    ):
        # alignment requirements: K and N divisible by 16
        dtype: torch.dtype = torch.bfloat16
        use_fast_accum = True
        # xpu does not support fast_accum now
        if "xpu" in device:
            use_fast_accum = False
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

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @torch._inductor.config.patch("emulate_precision_casts", True)
    def test_mx_fusion(self, device):
        # use a device key for library registration
        device_type = torch.device(device).type
        device_dispatch_key = "CUDA" if device_type == "cuda" else "XPU"
        # Register fake_scaled_mm custom op scoped to this test
        with torch.library._scoped_library("test_fp8", "FRAGMENT") as lib:
            # Define the op schema
            lib.define(
                "fake_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b, "
                "Tensor? bias=None, Tensor? scale_result=None, ScalarType? out_dtype=None, "
                "bool use_fast_accum=False) -> Tensor"
            )
            input_values = []

            # Register CUDA/XPU implementation
            @torch.library.impl(lib, "fake_scaled_mm", device_dispatch_key)
            def fake_scaled_mm_impl(
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                bias=None,
                scale_result=None,
                out_dtype=None,
                use_fast_accum=False,
            ):
                """Software-emulated scaled_mm for testing without CUDA 12.8"""
                out_dtype = out_dtype or torch.bfloat16
                # just using add, because without real dtypes,
                # was seeing overflow/instability
                nonlocal input_values
                input_values.append((mat_a, mat_b, scale_a, scale_b))
                result = mat_a.to(torch.float32) + mat_b.to(torch.float32)
                if bias is not None:
                    result = result + bias.to(torch.float32)
                return result.to(out_dtype)

            # Register fake implementation
            @torch.library.impl(lib, "fake_scaled_mm", "Meta")
            def fake_scaled_mm_meta(
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                bias=None,
                scale_result=None,
                out_dtype=None,
                use_fast_accum=False,
            ):
                """FakeTensor implementation"""
                out_dtype = out_dtype or torch.bfloat16
                M, K = mat_a.shape
                K2, N = mat_b.shape
                torch._check(
                    K == K2,
                    lambda: f"Incompatible shapes: {mat_a.shape} @ {mat_b.shape}",
                )
                return torch.empty((M, N), dtype=out_dtype, device=mat_a.device)

            def forward(
                arg0_1,
                arg1_1,
            ):
                view = torch.ops.aten.reshape.default(arg0_1, [8192, 256, 32])
                abs_1 = torch.ops.aten.abs.default(view)
                amax = torch.ops.aten.amax.default(abs_1, [-1])
                unsqueeze = torch.ops.aten.unsqueeze.default(amax, -1)
                view_1 = torch.ops.aten.view.dtype(unsqueeze, torch.int32)
                bitwise_right_shift = torch.ops.aten.bitwise_right_shift.Tensor_Scalar(
                    view_1, 23
                )
                bitwise_and = torch.ops.aten.bitwise_and.Scalar(
                    bitwise_right_shift, 255
                )
                sub = torch.ops.aten.sub.Tensor(bitwise_and, 127)
                sub_1 = torch.ops.aten.sub.Tensor(sub, 8)
                clamp_min = torch.ops.aten.clamp_min.default(sub_1, -127)
                clamp_max = torch.ops.aten.clamp_max.default(clamp_min, 128)
                add = torch.ops.aten.add.Tensor(clamp_max, 127)
                convert_element_type = torch.ops.prims.convert_element_type.default(
                    add, torch.uint8
                )
                isnan = torch.ops.aten.isnan.default(unsqueeze)
                scalar_tensor = torch.ops.aten.scalar_tensor.default(
                    255, dtype=torch.uint8, layout=torch.strided, device=device
                )
                where = torch.ops.aten.where.self(
                    isnan, scalar_tensor, convert_element_type
                )
                convert_element_type_1 = torch.ops.prims.convert_element_type.default(
                    where, torch.int32
                )
                bitwise_left_shift = torch.ops.aten.bitwise_left_shift.Tensor_Scalar(
                    convert_element_type_1, 23
                )
                view_2 = torch.ops.aten.view.dtype(bitwise_left_shift, torch.float32)
                clamp_min_1 = torch.ops.aten.clamp_min.default(
                    view_2, 1.1754943508222875e-38
                )
                div = torch.ops.aten.div.Tensor(view, clamp_min_1)
                clamp_min_2 = torch.ops.aten.clamp_min.default(div, -448.0)
                clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_2, 448.0)
                convert_element_type_2 = torch.ops.prims.convert_element_type.default(
                    clamp_max_1, torch.float8_e4m3fn
                )
                view_3 = torch.ops.aten.reshape.default(
                    convert_element_type_2, [8192, 8192]
                )
                convert_element_type_2 = None
                view_4 = torch.ops.aten.view.dtype(where, torch.float8_e8m0fnu)
                squeeze = torch.ops.aten.squeeze.dim(view_4, -1)

                view_5 = torch.ops.aten.reshape.default(arg1_1, [8192, 256, 32])
                abs_2 = torch.ops.aten.abs.default(view_5)
                amax_1 = torch.ops.aten.amax.default(abs_2, [-1])
                unsqueeze_1 = torch.ops.aten.unsqueeze.default(amax_1, -1)
                view_6 = torch.ops.aten.view.dtype(unsqueeze_1, torch.int32)
                bitwise_right_shift_1 = (
                    torch.ops.aten.bitwise_right_shift.Tensor_Scalar(view_6, 23)
                )
                bitwise_and_1 = torch.ops.aten.bitwise_and.Scalar(
                    bitwise_right_shift_1, 255
                )
                sub_2 = torch.ops.aten.sub.Tensor(bitwise_and_1, 127)
                sub_3 = torch.ops.aten.sub.Tensor(sub_2, 8)
                clamp_min_3 = torch.ops.aten.clamp_min.default(sub_3, -127)
                clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_3, 128)
                add_1 = torch.ops.aten.add.Tensor(clamp_max_2, 127)
                convert_element_type_3 = torch.ops.prims.convert_element_type.default(
                    add_1, torch.uint8
                )
                isnan_1 = torch.ops.aten.isnan.default(unsqueeze_1)
                unsqueeze_1 = None
                scalar_tensor_1 = torch.ops.aten.scalar_tensor.default(
                    255, dtype=torch.uint8, layout=torch.strided, device=device
                )
                where_1 = torch.ops.aten.where.self(
                    isnan_1, scalar_tensor_1, convert_element_type_3
                )
                convert_element_type_4 = torch.ops.prims.convert_element_type.default(
                    where_1, torch.int32
                )
                bitwise_left_shift_1 = torch.ops.aten.bitwise_left_shift.Tensor_Scalar(
                    convert_element_type_4, 23
                )
                convert_element_type_4 = None
                view_7 = torch.ops.aten.view.dtype(bitwise_left_shift_1, torch.float32)
                bitwise_left_shift_1 = None
                clamp_min_4 = torch.ops.aten.clamp_min.default(
                    view_7, 1.1754943508222875e-38
                )
                div_1 = torch.ops.aten.div.Tensor(view_5, clamp_min_4)
                clamp_min_5 = torch.ops.aten.clamp_min.default(div_1, -448.0)
                clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_5, 448.0)
                convert_element_type_5 = torch.ops.prims.convert_element_type.default(
                    clamp_max_3, torch.float8_e4m3fn
                )
                view_8 = torch.ops.aten.reshape.default(
                    convert_element_type_5, [8192, 8192]
                )
                view_9 = torch.ops.aten.view.dtype(where_1, torch.float8_e8m0fnu)
                squeeze_1 = torch.ops.aten.squeeze.dim(view_9, -1)

                permute = torch.ops.aten.permute.default(view_8, [1, 0])

                view_13 = torch.ops.aten.reshape.default(squeeze, [64, 128, 64, 4])
                permute_2 = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3])
                clone = torch.ops.aten.clone.default(
                    permute_2, memory_format=torch.contiguous_format
                )
                view_14 = torch.ops.aten.reshape.default(clone, [4096, 4, 32, 4])
                permute_3 = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3])
                clone_1 = torch.ops.aten.clone.default(
                    permute_3, memory_format=torch.contiguous_format
                )
                view_15 = torch.ops.aten.reshape.default(clone_1, [4096, 32, 16])

                view_16 = torch.ops.aten.reshape.default(view_15, [2097152])

                view_18 = torch.ops.aten.reshape.default(squeeze_1, [64, 128, 64, 4])
                permute_5 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3])
                clone_2 = torch.ops.aten.clone.default(
                    permute_5, memory_format=torch.contiguous_format
                )
                view_19 = torch.ops.aten.reshape.default(clone_2, [4096, 4, 32, 4])
                permute_6 = torch.ops.aten.permute.default(view_19, [0, 2, 1, 3])
                clone_3 = torch.ops.aten.clone.default(
                    permute_6, memory_format=torch.contiguous_format
                )
                view_20 = torch.ops.aten.reshape.default(clone_3, [4096, 32, 16])

                view_21 = torch.ops.aten.reshape.default(view_20, [2097152])

                _scaled_mm = torch.ops.test_fp8.fake_scaled_mm.default(
                    view_3, permute, view_16, view_21, None, None, torch.float32
                )
                return (_scaled_mm,)

            # Run with largest shape
            M, K, N = 8192, 8192, 8192

            A = torch.randn(M, K, dtype=torch.float32, device=device)
            B = torch.randn(K, N, dtype=torch.float32, device=device)
            f_c = torch.compile(fullgraph=True)(forward)

            _, code = run_and_get_code(f_c, A, B)

            FileCheck().check(".run(").check(".run(").check("fake_scaled_mm").run(
                code[0]
            )

            for seed in range(5):
                input_values.clear()
                torch.manual_seed(seed)
                # without dividing, outputs get way too large
                A = torch.randn(M, K, dtype=torch.float32, device=device)
                B = torch.randn(K, N, dtype=torch.float32, device=device)

                # Uses fake_scaled_mm custom op (no CUDA 12.8 needed!)
                torch._dynamo.reset()
                torch.compile(forward)(A, B)

                torch._dynamo.reset()
                with config.patch({"loop_index_inversion_in_fusion": False}):
                    torch.compile(forward)(A, B)

                assert len(input_values) == 2
                for i in range(4):
                    self.assertEqual(
                        input_values[0][i],
                        input_values[1][i],
                        msg=f"idx {i} seed {seed}",
                    )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @onlyOn(["cuda", "xpu"])
    @parametrize("M", (1, 3, 33, 257, 1024))
    @parametrize("K", (16, 32, 1024))
    @parametrize("N", (16, 2048))
    @parametrize(
        "persistent_matmul", [False, True] if has_triton_tma_device() else [False]
    )
    def test_rowwise_scaling_acceptable_input_dims(
        self, M: int, K: int, N: int, persistent_matmul: bool, device
    ):
        dtype: torch.dtype = torch.bfloat16
        use_fast_accum = True
        # xpu does not support fast_accum now
        if "xpu" in device:
            use_fast_accum = False
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

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, "Not supported on non B200")
    def test_mx_fp8_max_autotune(self, device):
        M, K, N = 128, 32, 128
        BLOCK_SIZE = 32
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

    @onlyOn(["cuda", "xpu"])
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_unacceptable_input_dims(self, device):
        # for compiled ops, type checking is in torch/_meta_registrations.py
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        # xpu does not support fast_accum now
        use_fast_accum = True
        if "xpu" in device:
            use_fast_accum = False
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
                use_fast_accum=use_fast_accum,
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
    @onlyOn(["cuda", "xpu"])
    def test_unacceptable_scale_dims_rowwise_scaling(self, device):
        dtype: torch.dtype = torch.bfloat16
        dtype_float8 = torch.float8_e4m3fn
        dtype_float8 = _fix_fp8_dtype_for_rocm(dtype_float8, device)

        M, K, N = 233, 32, 128
        x = torch.randn(M, K, dtype=dtype, device=device)
        w = torch.randn(N, K, dtype=dtype, device=device)
        bias = torch.randn(N, device=device, dtype=torch.bfloat16)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        # xpu does not support fast_accum now
        use_fast_accum = True
        if "xpu" in device:
            use_fast_accum = False

        def linear(x, w_t_fp8, w_inverse_scale, bias):
            x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype_float8)
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                w_inverse_scale.t(),  # testing with w and x scales switched
                x_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
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


@unittest.skipIf(not SM100OrLater, "Requires SM100+ (Blackwell) for PTX instruction")
class TestCvtE8M0Rceil(TestCase):
    """Tests for cvt_e8m0_rceil prim with PTX lowering on Blackwell."""

    def test_correctness(self):
        """Test correctness for various dtypes."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            inp = torch.cat(
                [
                    torch.arange(-1024, 0, device="cuda", dtype=dtype),
                    torch.arange(1, 1025, device="cuda", dtype=dtype),
                ]
            )
            eager_result = fn(inp)
            compiled_result = torch.compile(fn)(inp)
            self.assertEqual(compiled_result, eager_result)

    def test_pattern_match(self):
        """Test that the log2+ceil pattern gets matched and replaced."""
        _misc_patterns_init()

        E8M0_BIAS = 127

        def fn_with_log2_pattern(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        inp = torch.tensor(
            [1.0, 2.0, 4.0, 3.0, 1.5], device="cuda", dtype=torch.float32
        )

        eager_result = fn_with_log2_pattern(inp)
        compiled_result = torch.compile(fn_with_log2_pattern)(inp)
        self.assertEqual(compiled_result, eager_result)

    def test_ptx_code_generation(self):
        """Test that PTX instruction appears in generated code."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        inp = torch.rand(32, device="cuda", dtype=torch.float32)
        compiled_fn = torch.compile(fn)
        _, code = run_and_get_code(compiled_fn, inp)

        code_str = "\n".join(code)
        self.assertIn("cvt.rp.satfinite.ue8m0x2.f32", code_str)


instantiate_device_type_tests(TestFP8Types, globals(), allow_xpu=True)
instantiate_device_type_tests(TestFP8Lowering, globals(), allow_xpu=True)


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON or HAS_CPU:
        run_tests()
