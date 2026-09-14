# Owner(s): ["module: inductor"]
import contextlib
import logging
import os
import re
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil

import torch
from torch._inductor import config
from torch._inductor.ir import Buffer, FixedLayout
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import try_import_ck_lib
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    skipIfRocmVersionAtLeast,
    subtest,
)
from torch.testing._internal.inductor_utils import (
    _quantize_rowwise,
    _quantize_tensorwise,
    HAS_CPU,
    HAS_TRITON,
)


if HAS_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

log = logging.getLogger(__name__)


# patch env for tests if needed
_test_env = {}


def _dtype_test_name(dtype):
    return {torch.float16: "float16", torch.bfloat16: "bfloat16"}[dtype]


_parametrize_dtype = parametrize(
    "dtype",
    (torch.float16, torch.bfloat16),
    name_fn=_dtype_test_name,
)


@unittest.skipIf(not torch.version.hip, "ROCM only")
class TestCKBackend(TestCase):
    hw_classification = HardwareClassification.CUDA

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

    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends,dtype",
        (
            # CK float16 is covered by test_max_autotune_precompile_preselected
            # and test_max_autotune_addmm. Only CKTILE needs a float16 cell here.
            subtest(
                ("CK", torch.bfloat16),
                name="standalone_ck",
                decorators=[skipIfRocmVersionAtLeast([7, 14])],
            ),
            subtest(("CKTILE", torch.float16), name="standalone_cktile_float16"),
            subtest(("CKTILE", torch.bfloat16), name="standalone_cktile_bfloat16"),
            subtest(("ATen,CK", torch.bfloat16), name="fallback"),
        ),
    )
    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("use_aoti", (True, False))
    def test_max_autotune_precompile_matmul(
        self, device, max_autotune_gemm_backends, dtype, autotune_in_subproc, use_aoti
    ):
        """
        Make sure autotuning mm doesn't crash.
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": device, "dtype": dtype}

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

    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    @parametrize("autotune_in_subproc", (True,))
    def test_max_autotune_precompile_matmul_dynamic(
        self, device, max_autotune_gemm_backends, autotune_in_subproc
    ):
        """
        Test matmul with dynamic shapes
        """

        tensor_options = {"device": device, "dtype": torch.bfloat16}

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

    @skipIfRocmVersionAtLeast([7, 14])
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("num_gemms", (1, 2))
    def test_max_autotune_ck_backend_cpp_wrapper(self, device, num_gemms):
        """
        Verify that CK GEMM templates work under JIT cpp_wrapper mode.
        ``num_gemms=2`` chains a second GEMM of a different shape so the
        wrapper has to link against multiple distinct .so files.
        """
        M, N, K = 2240, 2048, 256
        tensor_options = {"device": device, "dtype": torch.bfloat16}

        class MyModel(torch.nn.Module):
            def forward(self, a, b, c):
                out = a @ b
                if num_gemms > 1:
                    out = out @ c
                return out

        model = MyModel().cuda()
        # Non-negative inputs: the num_gemms=2 case chains a second GEMM that
        # contracts over K=2048 in bf16. Signed inputs cause cancellation and
        # near-zero outputs whose relative error against eager's separately
        # rounded bf16 result explodes; rand() keeps outputs well-conditioned so
        # the default assert_close tolerance stays meaningful.
        a = torch.rand(M, K, **tensor_options)
        b = torch.rand(K, N, **tensor_options)
        c = torch.rand(N, N // 2, **tensor_options)
        expected = model(a, b, c)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CK",
                    "compile_threads": 2,
                    "rocm.ck_max_profiling_configs": 2,
                    "rocm.ck_dir": self.ck_dir,
                    "cpp_wrapper": True,
                }
            ),
            tf32_off(),
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, fullgraph=True)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            # JIT path must call the bare extern "C" symbol, not via the
            # AOT-only `kernels.` member.
            FileCheck().check("rocm_").check_not("kernels.rocm_").run(codes[0])

    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    def test_max_autotune_precompile_preselected(self, device, max_autotune_gemm_backends):
        """
        End to end test for picking preselected ck instances
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": device, "dtype": torch.float16}

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

    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("Aten,CK",))
    def test_max_autotune_precompile_non_contiguous(self, device, max_autotune_gemm_backends):
        """
        Make sure the matmul with non-contiguous inputs can fallback
        """

        tensor_options = {"device": device, "dtype": torch.float16}

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

    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    @parametrize(
        "x_shape",
        ([4096, 2048], [2048], [4096, 1]),
        name_fn=lambda x_shape: f"x_shape_{'x'.join(map(str, x_shape))}",
    )
    @_parametrize_dtype
    def test_max_autotune_addmm(self, device, max_autotune_gemm_backends, x_shape, dtype):
        m, k, n = 4096, 224, 2048
        alpha, beta = 1.0, 1.0

        tensor_options = {"device": device, "dtype": dtype}
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
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK", "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    @parametrize("quantize_type", ("tensorwise", "rowwise"))
    @parametrize("has_bias", (True, False))
    def test_max_autotune_scaled_mm(
        self, device, max_autotune_gemm_backends, quantize_type, has_bias
    ):
        use_fast_accum = False
        runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx94" not in runtime_arch and "gfx95" not in runtime_arch:
            self.skipTest(f"Unsupported arch {runtime_arch}")
        # output dtype
        dtype = torch.bfloat16
        tensor_options = {"device": device, "dtype": dtype}

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

    @unittest.mock.patch.dict(
        os.environ,
        {**_test_env, "PYTORCH_MIOPEN_SUGGEST_NHWC": "1"},
    )
    @parametrize(
        "max_autotune_conv_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATEN,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    def test_max_autotune_conv2d(self, device, max_autotune_conv_backends):
        tensor_options = {"device": device, "dtype": torch.float32}

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

    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        (subtest("CK", decorators=[skipIfRocmVersionAtLeast([7, 14])]), "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    def test_max_autotune_precompile_bmm(
        self,
        device,
        max_autotune_gemm_backends,
    ):
        """
        Test gemm-max-autotune torch.bmm with CK backend
        """

        def bmm(a, b):
            return torch.bmm(a, b)

        tensor_options = {"device": device, "dtype": torch.bfloat16}

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


_LEGACY_PIPELINE_HEADER = """
template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full,
          typename ComputeDataType_        = ADataType_,
          bool FixedVectorSize_            = false,
          index_t VectorSizeA_             = 1,
          index_t VectorSizeB_             = 1>
struct UniversalGemmPipelineProblem
{
};
"""

_V2_PIPELINE_HEADER = """
template <typename AsDataType_,
          typename BsDataType_,
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          typename AElementWise_           = ck_tile::element_wise::PassThrough,
          typename BElementWise_           = ck_tile::element_wise::PassThrough,
          typename AComputeDataType_       = AsDataType_,
          typename BComputeDataType_       = BsDataType_,
          bool FixedVectorSize_            = false,
          index_t VectorSizeA_             = 1,
          index_t VectorSizeB_             = 1>
struct UniversalGemmPipelineProblem
{
};
"""


@unittest.skipIf(not torch.version.hip, "ROCM only")
class TestCKTileUniversalGemmTemplate(TestCase):
    hw_classification = HardwareClassification.CUDA

    _MODULE = "torch._inductor.codegen.rocm.ck_tile_universal_gemm_template"

    def setUp(self):
        super().setUp()
        from torch._inductor.codegen.rocm import ck_tile_universal_gemm_template

        self._ck_tile = ck_tile_universal_gemm_template
        # _ck_tile_universal_gemm_v2_api is functools.cache'd on the search path.
        self._ck_tile._ck_tile_universal_gemm_v2_api.cache_clear()

    def _make_template(self, m=2048, n=2048, k=2048, *, dtype):
        device = torch.device("cuda")
        X = Buffer(name="X", layout=FixedLayout(device, dtype, [m, k]))
        W = Buffer(name="W", layout=FixedLayout(device, dtype, [k, n]))
        return self._ck_tile.CKTileGemmTemplate(
            [X, W], FixedLayout(device, dtype, [m, n])
        )

    def _find_ck_tile_op(self, pipeline="CompV3", epilogue="Default", *, dtype):
        ck_dtype = self._ck_tile.CKTileGemmTemplate._TORCH_DTYPE_TO_CK[dtype]
        for op in self._ck_tile.ops():
            if (
                (op.pipeline, op.epilogue) == (pipeline, epilogue)
                and (op.layout_a, op.layout_b, op.layout_c) == ("Row", "Row", "Row")
                and op.datatype_a == ck_dtype
            ):
                return op
        raise AssertionError(f"no CK-Tile gemm op for {pipeline=} {epilogue=}")

    def _compile_ck_tile_source(self, source: str) -> None:
        from torch._inductor.codegen.rocm.compile_command import rocm_compile_command

        if not torch.cuda.is_available():
            raise unittest.SkipTest("ROCm device required to select --offload-arch")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]

        with tempfile.TemporaryDirectory() as tmp_dir:
            src_path = os.path.join(tmp_dir, "instance.hip")
            with open(src_path, "w") as f:
                f.write(source)
            with config.patch({"rocm.arch": [arch]}):
                cmd = rocm_compile_command(
                    [src_path],
                    os.path.join(tmp_dir, "instance.o"),
                    "o",
                    ["-fsyntax-only"],
                )
            result = subprocess.run(
                shlex.split(cmd), capture_output=True, text=True, check=False
            )
        if result.returncode != 0:
            self.fail(f"{cmd}\n{result.stdout}\n{result.stderr}")

    def _write_pipeline_header(self, include_root: str, body: str) -> None:
        header_dir = os.path.join(
            include_root,
            "include",
            os.path.dirname(self._ck_tile._CK_TILE_PIPELINE_PROBLEM_HEADER),
        )
        os.makedirs(header_dir, exist_ok=True)
        with open(
            os.path.join(
                header_dir,
                os.path.basename(self._ck_tile._CK_TILE_PIPELINE_PROBLEM_HEADER),
            ),
            "w",
        ) as f:
            f.write(body)

    @contextlib.contextmanager
    def _probe_env(self, rocm_home_header=None, ck_dir_header=None):
        """
        A fake ROCm install plus an empty CK directory, so that the header the
        probe picks is determined by the test rather than by the environment.
        """
        with (
            tempfile.TemporaryDirectory() as rocm_home,
            tempfile.TemporaryDirectory() as ck_dir,
        ):
            if rocm_home_header is not None:
                self._write_pipeline_header(rocm_home, rocm_home_header)
            if ck_dir_header is not None:
                self._write_pipeline_header(ck_dir, ck_dir_header)
            with config.patch({"rocm.ck_dir": ck_dir}):
                yield rocm_home

    def _probe(self, rocm_home):
        return self._ck_tile._ck_tile_universal_gemm_v2_api(
            rocm_home, config.rocm.ck_dir
        )

    def test_header_probe_legacy_api(self):
        with self._probe_env(rocm_home_header=_LEGACY_PIPELINE_HEADER) as rocm_home:
            self.assertFalse(self._probe(rocm_home))

    def test_header_probe_v2_api(self):
        with self._probe_env(rocm_home_header=_V2_PIPELINE_HEADER) as rocm_home:
            self.assertTrue(self._probe(rocm_home))

    def test_header_probe_prefers_ck_dir_over_rocm_home(self):
        """
        The compiler searches the CK directory before $ROCM_HOME, so the probe
        has to resolve the header the same way.
        """
        with self._probe_env(
            rocm_home_header=_LEGACY_PIPELINE_HEADER,
            ck_dir_header=_V2_PIPELINE_HEADER,
        ) as rocm_home:
            self.assertTrue(self._probe(rocm_home))

    def test_header_probe_on_installed_header(self):
        """
        The synthetic headers above cannot catch the probe silently failing to
        classify the real thing, which is how this shipped broken once already.
        """
        header_path = self._ck_tile._find_ck_tile_header(
            config.rocm.rocm_home, config.rocm.ck_dir
        )
        if header_path is None:
            raise unittest.SkipTest("ck_tile headers are not installed")
        with open(header_path) as f:
            header_text = f.read()
        self.assertIsNotNone(
            self._ck_tile._header_has_v2_universal_gemm_pipeline(header_text),
            f"could not classify the universal GEMM API in {header_path}",
        )

    def test_header_probe_handles_long_template_clause(self):
        padding = "".join(f"          typename Unused{i}_ = void,\n" for i in range(64))
        header_text = _V2_PIPELINE_HEADER.replace(
            "          typename AElementWise_",
            padding + "          typename AElementWise_",
        )
        self.assertTrue(
            self._ck_tile._header_has_v2_universal_gemm_pipeline(header_text)
        )

    def test_header_probe_does_not_use_later_flatmm_signature(self):
        header_text = """
template <typename AsDataType_,
          typename BsDataType_,
          typename EDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          typename AElementWise_           = ck_tile::element_wise::PassThrough,
          typename BElementWise_           = ck_tile::element_wise::PassThrough>
struct UniversalGemmPipelineProblem
{
};

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_ = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                 = true,
          TailNumber TailNum_              = TailNumber::Full>
struct FlatmmPipelineProblem
{
};
"""
        self.assertTrue(
            self._ck_tile._header_has_v2_universal_gemm_pipeline(header_text)
        )

    @patch(f"{_MODULE}.torch.version.hip", "7.14.0")
    def test_header_probe_fallback_v2_when_header_missing(self):
        with self._probe_env() as rocm_home:
            self.assertTrue(self._probe(rocm_home))

    @patch(f"{_MODULE}.torch.version.hip", "7.2.0")
    def test_header_probe_fallback_v1_when_header_missing(self):
        with self._probe_env() as rocm_home:
            self.assertFalse(self._probe(rocm_home))

    @patch(f"{_MODULE}.torch.version.hip", "7.14.0")
    def test_header_probe_fallback_v2_when_header_unrecognized(self):
        header = "struct UniversalGemmPipelineProblem\n{\n};\n"
        with self._probe_env(rocm_home_header=header) as rocm_home:
            self.assertTrue(self._probe(rocm_home))

    @patch(f"{_MODULE}.torch.version.hip", "7.rc1")
    def test_header_probe_fallback_tolerates_malformed_hip_version(self):
        with self._probe_env() as rocm_home:
            self.assertFalse(self._probe(rocm_home))

    def test_ops_dtype_labels_agree_with_maps(self):
        template_cls = self._ck_tile.CKTileGemmTemplate
        globals_src = self._make_template(dtype=torch.float16).globals().getvalue()
        cpp_aliases = set(re.findall(r"using (\w+) =", globals_src))
        labels = {
            dt
            for op in self._ck_tile.ops()
            for dt in (op.datatype_a, op.datatype_b, op.datatype_c)
        }
        self.assertTrue(labels)
        self.assertEqual(labels, set(self._ck_tile.GEMM_DTYPES))
        # emit_ck_instance splices these labels into `using ADataType = ...;`, so a
        # label without a rendered alias produces HIP that does not compile.
        self.assertLessEqual(labels, cpp_aliases)
        self.assertLessEqual(labels, set(template_cls.ck_dtype_to_size))

    @_parametrize_dtype
    @parametrize("epilogue", ("Default", "CShuffle"))
    def test_emit_v1_legacy_instance(self, dtype, epilogue):
        code = self._make_template(dtype=dtype).emit_ck_instance(
            self._find_ck_tile_op(epilogue=epilogue, dtype=dtype), use_v2_api=False
        )
        self.assertIn("has_hot_loop_v", code)
        # "GemmPipelineProblem" alone would also match "UniversalGemmPipelineProblem",
        # which both API variants emit.
        self.assertIn("ck_tile::GemmPipelineProblem<", code)
        self.assertIn("BaseGemmPipeline", code)

    @_parametrize_dtype
    @parametrize("epilogue", ("Default", "CShuffle"))
    def test_emit_v2_simplified_instance(self, dtype, epilogue):
        code = self._make_template(dtype=dtype).emit_ck_instance(
            self._find_ck_tile_op(epilogue=epilogue, dtype=dtype), use_v2_api=True
        )
        self.assertNotIn("has_hot_loop_v", code)
        self.assertNotIn("BaseGemmPipeline", code)
        self.assertIn(
            "using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;",
            code,
        )

    def test_kernel_launch_v1_has_tail_handler(self):
        tmpl = self._ck_tile.CKTileGemmTemplate
        code = tmpl._template_from_string(tmpl.gemm_kernel_launch).render(
            instance_namespace="test_ns", use_v2_api=False
        )
        self.assertIn("BaseGemmPipeline::TailHandler", code)
        self.assertIn("has_hot_loop_v", code)

    def test_kernel_launch_v2_no_tail_handler(self):
        tmpl = self._ck_tile.CKTileGemmTemplate
        code = tmpl._template_from_string(tmpl.gemm_kernel_launch).render(
            instance_namespace="test_ns", use_v2_api=True
        )
        self.assertNotIn("TailHandler", code)
        self.assertNotIn("BaseGemmPipeline", code)
        self.assertIn("Kernel::GridSize", code)

    @_parametrize_dtype
    @parametrize("use_v2_api", (True, False))
    def test_cshuffle_epilogue_offered_only_with_v2_api(self, dtype, use_v2_api):
        """
        Pre-change ck_tile cannot compile the CShuffle epilogue we emit, so those
        instances must not reach autotuning and burn a compile each.
        """
        template = self._make_template(dtype=dtype)
        with (
            config.patch({"rocm.ck_tile_max_profiling_configs": None}),
            patch.object(
                self._ck_tile,
                "_ck_tile_universal_gemm_v2_api",
                lambda *_: use_v2_api,
            ),
        ):
            epilogues = {op.epilogue for op in template.gen_ops()}
        self.assertIn("Default", epilogues)
        self.assertEqual("CShuffle" in epilogues, use_v2_api)

    @_parametrize_dtype
    @parametrize("pipeline", ("CompV3", "CompV4", "Mem"))
    @parametrize("epilogue", ("Default", "CShuffle"))
    def test_offered_instance_compiles(self, dtype, epilogue, pipeline):
        """
        Every instance filter_op accepts must compile against the ck_tile headers
        installed on this host, whichever universal GEMM API they expose.
        """
        rocm = config.rocm
        if self._ck_tile._find_ck_tile_header(rocm.rocm_home, rocm.ck_dir) is None:
            raise unittest.SkipTest("ck_tile headers are not installed")
        template = self._make_template(dtype=dtype)
        op = self._find_ck_tile_op(pipeline=pipeline, epilogue=epilogue, dtype=dtype)
        if template.filter_op(op) is None:
            raise unittest.SkipTest(f"{pipeline}/{epilogue} not offered on this ROCm")
        use_v2_api = self._probe(config.rocm.rocm_home)
        self._compile_ck_tile_source(
            "\n".join(
                [
                    template.header().getvalue(),
                    template.globals().getvalue(),
                    template.emit_ck_instance(op, use_v2_api=use_v2_api),
                ]
            )
        )


instantiate_device_type_tests(TestCKBackend, globals(), only_for="cuda")
instantiate_device_type_tests(
    TestCKTileUniversalGemmTemplate, globals(), only_for="cuda"
)

if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_TRITON and HAS_CPU and is_big_gpu():
        run_tests()
