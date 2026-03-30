# Owner(s): ["module: inductor"]
import logging
import os
import unittest


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import try_import_ck_lib
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MI350_ARCH,
    parametrize,
    skipIfRocmArch,
)
from torch.testing._internal.inductor_utils import (
    _quantize_rowwise,
    _quantize_tensorwise,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
)


if HAS_CUDA_AND_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

log = logging.getLogger(__name__)


# patch env for tests if needed
_test_env = {}


@instantiate_parametrized_tests
class TestCKBackend(TestCase):
    def setUp(self):
        # The new inductor cache refresh mechanism
        # introduced with https://github.com/pytorch/pytorch/pull/122661
        # interacts badly with persistent subprocesses during
        # autotuning. So we need to disable automatic cache refresh
        # before calling setUp() on the parent class.
        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )

        torch.random.manual_seed(1234)

        self.ck_dir, _, _, _ = try_import_ck_lib()
        if not self.ck_dir:
            raise unittest.SkipTest("Composable Kernel library is not installed")

        try:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            super().setUp()
        finally:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = (
                old_disable_fresh_cache_envvar
            )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("CK", "CKTILE", "ATen,Triton,CK"))
    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("use_aoti", (True, False))
    def test_max_autotune_precompile_matmul(
        self, max_autotune_gemm_backends, autotune_in_subproc, use_aoti
    ):
        """
        Make sure autotuning mm doesn't crash.
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(256, 2048, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": autotune_in_subproc,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):
            if use_aoti:
                Y_compiled = AOTIRunnerUtil.run(
                    model=mm,
                    example_inputs=(a, b),
                )
            else:

                @torch.compile(dynamic=False)
                def compiled_mm(x, w):
                    return mm(x, w)

                Y_compiled = compiled_mm(a, b)

            Y = mm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("CK",))
    @parametrize("autotune_in_subproc", (True,))
    def test_max_autotune_precompile_matmul_dynamic(
        self, max_autotune_gemm_backends, autotune_in_subproc
    ):
        """
        Test matmul with dynamic shapes
        """

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(256, 2048, **tensor_options)

        torch._dynamo.mark_dynamic(a, 0)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": autotune_in_subproc,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=True)
            def compiled_mm(a, b):
                return a @ b

            Y_compiled = compiled_mm(a, b)
            Y = a @ b
            torch.testing.assert_close(Y_compiled, Y)

            a1 = torch.randn(1024, 256, **tensor_options)
            Y1_compiled = compiled_mm(a1, b)
            Y1 = a1 @ b
            torch.testing.assert_close(Y1_compiled, Y1)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("CK", "ATen,Triton,CK"))
    def test_max_autotune_precompile_preselected(self, max_autotune_gemm_backends):
        """
        End to end test for picking preselected ck instances
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.float16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(2048, 256, **tensor_options).transpose(0, 1)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 12,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.use_preselected_instances": True,
                }
            ),
            tf32_off(),
        ):
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("Aten,CK",))
    def test_max_autotune_precompile_non_contiguous(self, max_autotune_gemm_backends):
        """
        Make sure the matmul with non-contiguous inputs can fallback
        """

        tensor_options = {"device": "cuda", "dtype": torch.float16}

        a = torch.empty_strided((50257, 32768), (1, 50304), **tensor_options)
        b = torch.empty_strided((32768, 768), (768, 1), **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def mm(a, b):
                return a @ b

            Y_compiled = mm(a, b)
            Y_eager = a @ b
            torch.testing.assert_close(Y_compiled, Y_eager, equal_nan=True)

    # regression in ROCm 7.2, Mismatched elements, significantly
    @skipIfRocmArch(MI350_ARCH)
    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("CK", "ATen,Triton,CK"))
    @parametrize("x_shape", ([4096, 2048], [2048], [4096, 1]))
    def test_max_autotune_addmm(self, max_autotune_gemm_backends, x_shape):
        m, k, n = 4096, 224, 2048
        alpha, beta = 1.0, 1.0

        tensor_options = {"device": "cuda", "dtype": torch.float16}
        x = torch.ones(x_shape, **tensor_options)
        a = torch.randn(m, k, **tensor_options)
        b = torch.randn(k, n, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 2,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 2,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def addmm(x, a, b, alpha, beta):
                return torch.addmm(x, a, b, alpha=alpha, beta=beta)

            Y_compiled = addmm(x, a, b, alpha, beta)
            Y_eager = torch.addmm(x, a, b, alpha=alpha, beta=beta)

            torch.testing.assert_close(Y_compiled, Y_eager)

    @unittest.skip(
        "FIXME(tenpercent): kernel compilation errors on gfx942 as of 09/01/25"
    )
    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("CK", "ATen,Triton,CK"))
    @parametrize("quantize_type", ("tensorwise", "rowwise"))
    @parametrize("has_bias", (True, False))
    def test_max_autotune_scaled_mm(
        self, max_autotune_gemm_backends, quantize_type, has_bias
    ):
        use_fast_accum = False
        runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx94" not in runtime_arch and "gfx95" not in runtime_arch:
            self.skipTest(f"Unsupported arch {runtime_arch}")
        # output dtype
        dtype = torch.bfloat16
        tensor_options = {"device": "cuda", "dtype": dtype}

        M = 2240
        N = 2048
        K = 256

        x = torch.randn(M, K, **tensor_options)
        w = torch.randn(N, K, **tensor_options)

        bias = None
        if has_bias:
            bias = torch.randn(N, **tensor_options)

        dtype_float8 = (
            torch.float8_e4m3fnuz if "gfx94" in runtime_arch else torch.float8_e4m3fn
        )

        f_quantize = (
            _quantize_tensorwise if quantize_type == "tensorwise" else _quantize_rowwise
        )

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = f_quantize(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale_t = w_inverse_scale.t()

        # quantize input x
        x_fp8, x_inverse_scale = f_quantize(x, dtype_float8)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

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
            w_inverse_scale_t,
            bias,
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "compile_threads": 24,
                "rocm.ck_max_profiling_configs": 24,
                "rocm.ck_dir": self.ck_dir,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale_t,
                bias,
            )
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)

            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(
        os.environ,
        {**_test_env, "PYTORCH_MIOPEN_SUGGEST_NHWC": "1"},
    )
    @parametrize("max_autotune_conv_backends", ("CK", "ATEN,CK,TRITON"))
    def test_max_autotune_conv2d(self, max_autotune_conv_backends):
        tensor_options = {"device": "cuda", "dtype": torch.float32}

        x = torch.randn(1, 8, 224, 224, **tensor_options)
        w = torch.randn(64, 8, 7, 7, **tensor_options)
        x_cl = x.to(memory_format=torch.channels_last)
        w_cl = w.to(memory_format=torch.channels_last)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": False,
                    "max_autotune_conv_backends": max_autotune_conv_backends,
                    "compile_threads": 4,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 4,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def conv2d(x, w):
                return torch.conv2d(x, w)

            Y_eager = torch.conv2d(x_cl, w_cl)
            Y_compiled = conv2d(x_cl, w_cl)

            torch.testing.assert_close(Y_compiled, Y_eager, atol=2e-4, rtol=2e-4)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("CK", "ATen,Triton,CK"))
    def test_max_autotune_precompile_bmm(
        self,
        max_autotune_gemm_backends,
    ):
        """
        Test gemm-max-autotune torch.bmm with CK backend
        """

        def bmm(a, b):
            return torch.bmm(a, b)

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(16, 2240, 256, **tensor_options)
        b = torch.randn(16, 2048, 256, **tensor_options).transpose(1, 2)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 2,
                    "rocm.ck_max_profiling_configs": 2,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def compiled_bmm(x, w):
                return bmm(x, w)

            Y_compiled = compiled_bmm(a, b)

            Y_eager = bmm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y_eager)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA_AND_TRITON and HAS_CPU and is_big_gpu():
        run_tests()
