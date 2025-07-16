# Owner(s): ["module: inductor"]
import itertools
import logging
import math
import os
import re
import sysconfig
import time
import unittest
import unittest.mock as mock
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from torch._dynamo.exc import BackendCompilerFailed
from torch._inductor.codegen.cuda.serialization import get_cutlass_operation_serializer
from torch._inductor.utils import clear_caches
from torch.export import Dim
from torch.testing._internal.logging_utils import log_settings
from torch.utils import _pytree as pytree


try:
    from test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from .test_aot_inductor_utils import AOTIRunnerUtil

import torch
import torch._inductor.codecache
import torch.version
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.codegen.cuda.cutlass_utils import (
    _gen_ops_cached,
    get_max_alignment,
)
from torch._inductor.exc import InductorError
from torch._inductor.ir import FixedLayout
from torch._inductor.select_algorithm import NoValidChoicesError
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FP8,
    SM80OrLater,
    SM90OrLater,
)
from torch.testing._internal.common_utils import (
    IN_RE_WORKER,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    _quantize_rowwise,
    _quantize_tensorwise,
    HAS_CPU,
    HAS_CUDA,
)


torch.set_float32_matmul_precision("high")
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


log = logging.getLogger(__name__)

DEFAULT_INST_LEVEL_MM_CONFIG: int = 78


def _get_path_without_sccache() -> str:
    """
    Get the PATH environment variable without sccache.
    """
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


def _check_if_instances_equal(op1, op2) -> bool:
    """
    Utility function to check if two instances of a class are equal.
    """
    # cutlass uses list and tuple inconsistently
    if isinstance(op1, (list, tuple)):
        return tuple(op1) == tuple(op2)

    if type(op1) != type(op2):
        return False

    # some classes have __eq__ defined but they may be insufficient
    if op1.__class__.__dict__.get("__eq__") and op1 != op2:
        return False

    if isinstance(op1, Enum):
        return op1.value == op2.value

    if hasattr(op1, "__dict__"):
        for key, value in op1.__dict__.items():
            if key not in op2.__dict__:
                return False
            if not _check_if_instances_equal(value, op2.__dict__[key]):
                return False

    return True


un_ops_under_test = [torch.relu]
bin_ops_under_test = [torch.add, torch.mul, torch.sub, torch.div]

evt_all_ops = parametrize(
    "op", un_ops_under_test + bin_ops_under_test, name_fn=lambda f: f.__name__
)

evt_bin_ops = parametrize("op", bin_ops_under_test, name_fn=lambda f: f.__name__)

evt_all_shapes = parametrize("shape", itertools.product([512, 1024], repeat=2))


def gen_args(op, shape, dtype=torch.float16):
    if op in bin_ops_under_test:
        return (torch.rand(*shape, device="cuda:0", dtype=dtype),)
    else:
        return ()


use_evt_config = config.patch(
    {
        "max_autotune": True,
        "max_autotune_gemm_backends": "CUTLASS",
        "cuda.cutlass_max_profiling_configs": 1,
        "benchmark_epilogue_fusion": False,  # EVT doesn't support benchmark fusion yet
        "cuda.cutlass_tma_only": True,
        "cuda.cutlass_epilogue_fusion_enabled": True,
    }
)

fp8_config = config.patch(
    {
        "max_autotune": True,
        "max_autotune_gemm_backends": "CUTLASS",
        "cuda.cutlass_max_profiling_configs": 1,
        "benchmark_epilogue_fusion": False,  # EVT doesn't support benchmark fusion yet
        "cuda.cutlass_tma_only": True,
    }
)


def select_no_algorithm(*args, **kwargs):
    """
    Utility function to skip precompilation and autotuning.
    """
    raise NoValidChoicesError


@instantiate_parametrized_tests
class TestCutlassBackend(TestCase):
    def setUp(self):
        if not HAS_CUDA:
            self.skipTest("CUDA is not available")
        if torch.version.hip:
            self.skipTest("CUTLASS backend is not supported on HIP")

        # The new inductor cache refresh mechanism
        # introduced with https://github.com/pytorch/pytorch/pull/122661
        # interacts badly with persistent subprocesses during
        # autotuning. So we need to disable automatic cache refresh
        # before calling setUp() on the parent class.
        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )
        try:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            super().setUp()
        finally:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = (
                old_disable_fresh_cache_envvar
            )
        torch.random.manual_seed(1234)

    def tearDown(self):
        super().tearDown()
        clear_caches()

    def run_evt_test(self, model, op, shape, num_fusions=1):
        M, N = shape
        a = torch.ones(M, N).cuda().half()
        b = torch.ones(N, N).cuda().half().t()
        extra_args = gen_args(op, (M, N))
        model = model.cuda()

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cuda_epilogue_fusion_counter"],
            num_fusions,
        )
        torch.testing.assert_close(result, ref_result)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_threshold(self):
        """
        Make sure Cutlass GEMM threshold works as intended.
        """

        def mm(a, b):
            return a @ b

        a = torch.randn(100, 10).cuda().half()
        b = torch.randn(100, 10).cuda().half().t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_backend_min_gemm_size": 100000,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            with mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                wraps=select_no_algorithm,
            ) as sa:
                with self.assertRaisesRegex(InductorError, r".*NoValidChoicesError.*"):
                    _ = torch.compile(mm, dynamic=False)(a, b)
                args, _ = sa.call_args
                _, choices, _, __ = args

                self.assertEqual(choices, [])

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_import_cutlass(self):
        from torch._inductor.codegen.cuda.cutlass_utils import try_import_cutlass

        self.assertTrue(try_import_cutlass())

        if config.is_fbcode():
            import python_cutlass
        else:
            import cutlass as python_cutlass  # noqa: F401
        import cutlass_library  # noqa: F401

    def test_cutlass_key(self):
        from torch._inductor.codegen.cuda.cutlass_utils import try_import_cutlass

        self.assertTrue(try_import_cutlass())
        from torch._inductor.codecache import cutlass_key

        self.assertIsNotNone(cutlass_key())

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_subproc_mm(self):
        """
        Test autotune_in_subproc works for mm.

        NOTE: Shape like M, N, K = 100, 100, 10 would get filtered out due to
        alignment mismatch.
        """

        M, N, K = 4096, 2048, 25728

        a = torch.randn(M, K).cuda().half()
        b = torch.randn(N, K).cuda().half().t()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            Y_compiled = torch.compile(torch.mm)(a, b)
            Y = torch.mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(
        True, "FIXME: Disabled temporarily since IMA or crashing in subprocess"
    )
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_subproc_addmm(self, shape_combo):
        """
        Test autotune_in_subproc works for addmm.
        """

        M, N, K = 4096, 2048, 25728

        a = torch.randn(M, K).cuda().half()
        b = torch.randn(N, K).cuda().half().t()

        x_shapes = [
            (M, N),
            (M, 1),
            (1, N),
            (N,),
        ]

        alpha = 2.0
        beta = 0.4

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            for x_shape in x_shapes:
                x = torch.randn(x_shape).cuda().half()
                Y_compiled = torch.compile(torch.addmm)(x, a, b, alpha=alpha, beta=beta)
                Y = torch.addmm(x, a, b, alpha=alpha, beta=beta)
                torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_subproc_bmm(self):
        """
        Test autotune_in_subproc works for bmm.
        """

        B, M, N, K = 10, 4096, 2048, 25728

        a = torch.randn(B, M, K).cuda().half()
        b = torch.randn(B, N, K).cuda().half().permute(0, 2, 1)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_max_profiling_configs": 4,
            }
        ):
            Y_compiled = torch.compile(torch.bmm)(a, b)
            Y = torch.bmm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False, True))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_diff_matmul_share_same_kernel(self, dynamic):
        max_autotune_gemm_backends = "CUTLASS"

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                ab = a @ b
                ac = a @ c
                return ab, ac

        model = MyModel()
        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(128, 16).cuda().half().t()
        c = torch.randn(512, 16).cuda().half().t()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 1,
            }
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, dynamic=dynamic)
            expected = model(a, b, c)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            pattern = r"cutlass_[\w]+\.cutlass_[\w]+"
            match = re.search(pattern, codes[0])
            self.assertTrue(match is not None)
            cutlass_kernel = match.group()
            FileCheck().check_count(
                cutlass_kernel,
                2,
            ).run(codes[0])

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_number_mm_precompiles(self):
        torch._dynamo.utils.counters.clear()
        max_autotune_gemm_backends = "CUTLASS"

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b, c):
                ab = a @ b
                return ab

        model = MyModel()
        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(128, 16).cuda().half().t()
        c = torch.randn(512, 16).cuda().half().t()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 1,
                "cuda.cutlass_max_profiling_swizzle_options": [
                    1,
                    2,
                    4,
                ],  # guarantees > 1 choices
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
                "autotune_local_cache": False,
            }
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, dynamic=True)
            expected = model(a, b, c)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(re.search(r"cutlass_.*.cutlass_.*", codes[0]))
            # Verifies expected number of precompilations
            self.assertEqual(
                torch._dynamo.utils.counters["inductor"][
                    "select_algorithm_num_precompiles"
                ],
                1,
            )

    # NOTE: right now tuned_mm doesn't support cutlass 2x, which is used by A100
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_regular_mm(
        self,
        dynamic: bool,
        max_autotune_gemm_backends: str = "CUTLASS",
        use_aoti: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Main test for mm.
        """

        # M, N, K
        shapes = [
            (128, 128, 16),
            (1024, 1024, 256),
        ]

        # M, N, K
        shapes = shapes if dynamic else shapes[0:1]

        class MyModel(torch.nn.Module):
            def forward(self, a, b):
                return a @ b

        model = MyModel().cuda()

        inputs = [
            (torch.randn(M, K).cuda().to(dtype), torch.randn(K, N).cuda().to(dtype))
            for (M, N, K) in shapes
        ]

        dynamic_shapes = (
            {
                "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            }
            if dynamic
            else None
        )

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "cuda.cutlass_max_profiling_configs": 2,
                }
            ),
            dynamo_config.patch({"error_on_recompile": dynamic}),
        ):
            expected = [model(*input) for input in inputs]
            if use_aoti:
                actual = AOTIRunnerUtil.run_multiple(
                    model, inputs, dynamic_shapes=dynamic_shapes
                )
            else:
                compiled_model = torch.compile(model, dynamic=True)
                actual = [compiled_model(*input) for input in inputs]

            torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float8_e4m3fn,))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_fp8_scaled_mm(
        self,
        dynamic: bool,
        max_autotune_gemm_backends: str = "CUTLASS",
        use_aoti: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Main test for mm.
        """

        # M, N, K
        shapes = [
            (128, 128, 16),
            (1024, 1024, 256),
        ]

        # M, N, K
        shapes = shapes if dynamic else shapes[0:1]

        inputs = []
        for shape in shapes:
            M, N, K = shape
            output_dtype = torch.bfloat16
            device = "cuda"

            x = torch.randn(M, K, dtype=output_dtype, device=device)
            w = torch.randn(N, K, dtype=output_dtype, device=device)

            # quantize weight (prior to inference)
            w_fp8, w_inverse_scale = _quantize_rowwise(w, dtype)
            w_t_fp8 = w_fp8.t()
            w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

            # quantize input x
            x_fp8, x_inverse_scale = _quantize_rowwise(x, dtype)

            inputs.append((x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale))

        class MyModel(torch.nn.Module):
            def forward(self, x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale):
                y = torch._scaled_mm(
                    x_fp8,
                    w_t_fp8,
                    x_inverse_scale,
                    w_inverse_scale,
                    None,
                    out_dtype=torch.bfloat16,
                    use_fast_accum=False,
                )
                return y

        dynamic_shapes = (
            {
                "x_fp8": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                "x_inverse_scale": {0: Dim.DYNAMIC, 1: 1},
                "w_t_fp8": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                "w_inverse_scale": {0: 1, 1: Dim.DYNAMIC},
            }
            if dynamic
            else None
        )
        model = MyModel().cuda()

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "cuda.cutlass_max_profiling_configs": 2,
                    "benchmark_epilogue_fusion": False,  # EVT doesn't support benchmark fusion yet
                    "cuda.cutlass_tma_only": True,
                }
            ),
            dynamo_config.patch({"error_on_recompile": dynamic}),
        ):
            expected = [model(*input) for input in inputs]
            if use_aoti:
                actual = AOTIRunnerUtil.run_multiple(
                    model, inputs, dynamic_shapes=dynamic_shapes
                )
            else:
                compiled_model = torch.compile(model, dynamic=True)
                actual = [compiled_model(*input) for input in inputs]

            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_addmm(
        self,
        dynamic: bool,
        max_autotune_gemm_backends: str = "CUTLASS",
        use_aoti: bool = False,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Main test for addmm.
        """

        class MyModel(torch.nn.Module):
            def forward(self, x, a, b):
                return torch.addmm(x, a, b)

        model = MyModel().cuda()
        # M, N, K
        shapes = [
            (128, 128, 16),
            (512, 512, 128),
        ]
        shapes = shapes[0:1] if not dynamic else shapes

        x_shapes = [
            lambda M, N: (M, N),
            lambda M, N: (M, 1),
            lambda M, N: (1, N),
            lambda M, N: (N,),
        ]
        for x_shape in x_shapes:
            torch._dynamo.reset()
            clear_caches()

            inputs = [
                (
                    torch.randn(x_shape(M, N)).cuda().to(dtype),
                    torch.randn(M, K).cuda().to(dtype),
                    torch.randn(N, K).cuda().to(dtype).t(),
                )
                for (M, N, K) in shapes
            ]
            dynamic_shapes = (
                {
                    "x": {
                        i: v
                        for i, v in enumerate(x_shape(Dim.DYNAMIC, Dim.DYNAMIC))
                        if v != 1
                    },
                    "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                    "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                }
                if dynamic
                else None
            )
            with (
                config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": max_autotune_gemm_backends,
                        "cuda.cutlass_max_profiling_configs": 2,
                    }
                ),
                dynamo_config.patch({"error_on_recompile": dynamic}),
            ):
                expected = [model(*input) for input in inputs]
                if use_aoti:
                    actual = AOTIRunnerUtil.run_multiple(
                        model, inputs, dynamic_shapes=dynamic_shapes
                    )
                else:
                    compiled_model = torch.compile(model, dynamic=dynamic)
                    actual = [compiled_model(*input) for input in inputs]

                torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False, True))
    @parametrize("use_aoti", (False, True))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_bmm(
        self,
        dynamic: bool,
        use_aoti: bool = False,
        max_autotune_gemm_backends: str = "CUTLASS",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Main test for bmm.
        """

        class MyModel(torch.nn.Module):
            def forward(self, a, b):
                return torch.bmm(a, b)

        model = MyModel().cuda()
        # B, M, N, K
        shapes = [
            (10, 4096, 2048, 25728),
            (20, 2048, 1024, 12864),
        ]
        shapes = shapes[0:1] if not dynamic else shapes

        inputs = [
            (
                torch.randn(B, M, K).cuda().to(dtype),
                torch.randn(B, N, K).cuda().to(dtype).permute(0, 2, 1),
            )
            for B, M, N, K in shapes
        ]
        dynamic_shapes = (
            {
                "a": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC, 2: Dim.DYNAMIC},
                "b": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC, 2: Dim.DYNAMIC},
            }
            if dynamic
            else None
        )
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            expected = [model(*input) for input in inputs]
            if use_aoti:
                actual = AOTIRunnerUtil.run_multiple(
                    model, inputs, dynamic_shapes=dynamic_shapes
                )
            else:
                compiled_model = torch.compile(model, dynamic=dynamic)
                actual = [compiled_model(*input) for input in inputs]
            torch.testing.assert_close(actual, expected)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_regular_mm_streamk(
        self, dynamic: bool = False, max_autotune_gemm_backends: str = "CUTLASS"
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            return a @ b

        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(128, 16).cuda().half().t()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
                "cuda.cutlass_op_allowlist_regex": "stream_k",  # only stream-k GEMM Kernels
            }
        ):
            for M, K, N in (
                (128, 16, 128),
                (1024, 256, 1024),
                (
                    16384,
                    1024,
                    16384,
                ),
                (
                    16384,
                    1408,
                    16384,
                ),
            ):
                a = torch.randn(M, K).cuda().half()
                b = torch.randn(N, K).cuda().half().t()
                Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
                Y = mm(a, b)
                # we need relaxed numerical limits due to the sheer size of the
                # matmuls involved. Many small addition differences add up.
                torch.testing.assert_close(Y_compiled, Y, atol=0.01, rtol=0.01)

    def _test_max_autotune_cutlass_backend_epilogue_fusion(
        self,
        dynamic: bool = False,
        max_autotune_gemm_backends: str = "CUTLASS",
        fp16=True,
        expected_fuse_count=0,
        mm: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        batch_size: Optional[int] = None,
    ):
        # Note: The ops that are available
        # also depend on the alignment of the shapes
        # so if these shapes don't all align to at least 8 elements
        # it can happen that no Cutlass 3.x op is available
        # that allows fusions
        if batch_size is None:
            a = torch.randn(256, 32).cuda()
            b = torch.randn(256, 32).cuda().t()
        else:
            a = torch.randn(batch_size, 256, 32).cuda()
            b = torch.randn(batch_size, 256, 32).cuda().permute(0, 2, 1)
        if fp16:
            a = a.half()
            b = b.half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 4,
                "cuda.version": "12.2",  # required to enable the Kernels we need
            }
        ):
            counters["inductor"]["cuda_epilogue_fusion_counter"] = 0
            assert mm is not None
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            actual_count = counters["inductor"]["cuda_epilogue_fusion_counter"]
            assert actual_count == expected_fuse_count, (
                f"Expected fuse count of {expected_fuse_count} but got {actual_count}"
            )
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_max_autotune_cutlass_backend_simple_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.0

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_max_autotune_cutlass_backend_chained_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return (a @ b) * 3.3 - 1.234

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_max_autotune_cutlass_backend_relu_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.nn.functional.relu((a @ b) * 3.3 - 1.234)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_max_autotune_cutlass_backend_relu6_fusion_fp16_fp32acc(self):
        def mm(a, b):
            return torch.clamp(torch.nn.functional.relu(a @ b), max=6.0)

        #  The pointwise ops seem to be pre-fused into a single Pointwise
        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_max_autotune_cutlass_backend_no_fusion_dtype_mismatch(self):
        def mm(a, b):
            # this should not be fused, since the output dtype is different from the matmul dtype
            return (a @ b).to(torch.float32) * 0.00001

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_max_autotune_cutlass_backend_shape_dependent_normalization_fusion(self):
        def mm(a, b):
            return (a @ b) / b.size(1)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False,))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_int_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str = "CUTLASS"
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            return torch._int_mm(a, b)

        # CUTLASS only supports row-major/column-major combination of
        # layouts for this operation, thus the transpose of tensor b
        # (on the other side, Triton at the moment doesn't support
        # this combination, so it's excluded from the test).  Also,
        # for CUTLASS alignment requirements, number of columns in
        # both tensors has to be divisible by 16.
        a = torch.randint(0, 5, (100, 16), dtype=torch.int8).cuda()
        b = torch.randint(0, 5, (32, 16), dtype=torch.int8).cuda().T

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_force_cutlass_backend_aoti_dynamic(self):
        class MyModel(torch.nn.Module):
            def forward(self, x, w):
                return x @ w

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            model = MyModel()
            M, N, K = 16, 32, 64
            dynamic_shapes = {
                "x": {0: M, 1: K},
                "w": {0: K, 1: N},
            }

            x = torch.randn(M, K).cuda().half()
            w = torch.randn(N, K).cuda().half().t()

            actual = AOTIRunnerUtil.run(
                model,
                (x, w),
                dynamic_shapes=dynamic_shapes,
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual)

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_force_cutlass_backend_aoti_cexpr_codegen(self):
        class MyModel(torch.nn.Module):
            def forward(self, x, w):
                x0, x1 = x.shape
                x = x.reshape(x0 // 2, x1, 2)[:, :, 0]
                x = x.contiguous()
                x = x.as_strided(x.size(), x.stride())

                return x @ w

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            model = MyModel()
            M, N, K = 128, 64, 64
            dynamic_shapes = {
                "x": {0: Dim.DYNAMIC},
                "w": None,
            }

            x = torch.randn(M, K).cuda().half()
            w = torch.randn(N, K).cuda().half().t()

            actual = AOTIRunnerUtil.run(
                model,
                (x, w),
                dynamic_shapes=dynamic_shapes,
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual)

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_aoti_workspace_ptr(self):
        class MyModel(torch.nn.Module):
            def forward(self, x, w):
                return x @ w

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_op_allowlist_regex": "128x256x64.*stream_k_warpspecialized_cooperative_epi_nosmem",
                "cuda.cutlass_max_profiling_configs": 1,
            }
        ):
            model = MyModel()
            M, N, K = 200, 5216, 10_432

            x = torch.randn(M, K).cuda().half()
            w = torch.randn(N, K).cuda().half().t()

            actual = AOTIRunnerUtil.run(
                model,
                (x, w),
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual, atol=0.01, rtol=0.01)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM80OrLater or SM90OrLater, "need sm_8x exactly")
    @parametrize("dynamic", (False,))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_sparse_semi_structured_mm(
        self, dynamic: bool
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        SparseSemiStructuredTensor._FORCE_CUTLASS = True

        def mm(a, b):
            return torch.mm(a, b)

        m, n, k = 32, 8, 64
        mask = torch.tensor([0, 0, 1, 1]).tile(m, k // 4).cuda().half()
        a = torch.rand(m, k).cuda().half() * mask
        a_sparse = to_sparse_semi_structured(a)
        b = torch.rand(k, n).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_local_cache": True,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a_sparse, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

        cache = torch._inductor.codecache.LocalCache().lookup(
            "sparse_semi_structured_mm"
        )
        assert cache is not None
        high = cache[
            f"[('cuda', 'torch.float16', {m}, {k // 2}, {k // 2}, 1, 0), "
            f"('cuda', 'torch.int16', {m}, {k // 16}, {k // 16}, 1, 0), "
            f"('cuda', 'torch.float16', {k}, {n}, {n}, 1, 0)]"
        ]["high"]
        cutlass_kernels_count = 0
        for kernel, duration in high.items():
            if kernel.startswith("cutlass_gemm") and not math.isinf(duration):
                cutlass_kernels_count += 1
        assert cutlass_kernels_count > 0

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_op_denylist(
        self,
    ):
        def my_addmm(x, a, b, alpha, beta):
            return torch.addmm(x, a, b, alpha=beta, beta=alpha)

        x = torch.randn((128, 128)).cuda().half()
        a = torch.randn(128, 128).cuda().half()
        b = torch.randn(128, 128).cuda().half().t()

        with fresh_cache():
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cuda.cutlass_max_profiling_configs": 2,
                    "cuda.cutlass_op_allowlist_regex": "",
                    "cuda.cutlass_op_denylist_regex": "pingpong",
                }
            ):
                with mock.patch(
                    "torch._inductor.kernel.mm.autotune_select_algorithm",
                    wraps=select_no_algorithm,
                ) as sa:
                    with self.assertRaisesRegex(
                        InductorError, r".*NoValidChoicesError.*"
                    ):
                        torch.compile(my_addmm, dynamic=False)(x, a, b, 1.0, 2.0)
                    args, _ = sa.call_args
                    op_name, choices, _, __ = args
                    assert op_name == "addmm"
                    cuda_template_count = 0
                    for choice in choices:
                        if isinstance(choice, CUDATemplateCaller):
                            choice_info = choice.info_dict()
                            op_conf_name = choice_info.get("op_conf_name", "")
                            assert isinstance(op_conf_name, str)
                            assert "pingpong" not in op_conf_name, (
                                "All pingpong Kernels should have been filtered"
                            )
                            cuda_template_count += 1
                    assert cuda_template_count > 0, "No CUDATemplateCaller choices"

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_op_allowlist(
        self,
    ):
        def addmm(x, a, b, alpha, beta):
            return torch.addmm(x, a, b, alpha=alpha, beta=beta)

        x = torch.randn((128, 128)).cuda().half()
        a = torch.randn(128, 128).cuda().half()
        b = torch.randn(128, 128).cuda().half().t()

        with fresh_cache():
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cuda.cutlass_max_profiling_configs": 2,
                    "cuda.cutlass_op_allowlist_regex": "pingpong",
                    "cuda.cutlass_op_denylist_regex": None,
                }
            ):
                with mock.patch(
                    "torch._inductor.kernel.mm.autotune_select_algorithm",
                    wraps=select_no_algorithm,
                ) as sa:
                    with self.assertRaisesRegex(
                        InductorError, r".*NoValidChoicesError.*"
                    ):
                        torch.compile(addmm, dynamic=False)(x, a, b, 1.0, 1.0)
                    args, _ = sa.call_args
                    op_name, choices, _, __ = args
                    assert op_name == "addmm"
                    cuda_template_count = 0
                    for choice in choices:
                        if isinstance(choice, CUDATemplateCaller):
                            choice_info = choice.info_dict()
                            op_conf_name = choice_info.get("op_conf_name", "")
                            assert isinstance(op_conf_name, str)
                            assert "pingpong" in op_conf_name, (
                                "Only pingpong Kernels should have been allowed"
                            )
                            cuda_template_count += 1
                    assert cuda_template_count > 0, "No CUDATemplateCaller choices"

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_fp8_scaled_mm_fast_accum_filtering(
        self,
    ):
        float8_dtype = torch.float8_e4m3fn
        # Only bf16 output type is supported for row-wise scaling, not fp32
        output_dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        M, K, N = 128, 128, 128  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=output_dtype, device=device)
        w = torch.randn(N, K, dtype=output_dtype, device=device)
        bias = None
        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, float8_dtype)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_rowwise(x, float8_dtype)

        def linear(
            x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias, use_fast_accum
        ):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=output_dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        linear_compiled = torch.compile(linear, backend="inductor")

        def run_test(use_fast_accum):
            with fresh_cache():
                with config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "CUTLASS",
                        "cuda.cutlass_max_profiling_configs": 2,
                    }
                ):
                    with mock.patch(
                        "torch._inductor.kernel.mm.autotune_select_algorithm",
                        wraps=select_no_algorithm,
                    ) as sa:
                        with self.assertRaisesRegex(
                            InductorError, r".*NoValidChoicesError.*"
                        ):
                            linear_compiled(
                                x_fp8,
                                x_inverse_scale,
                                w_t_fp8,
                                w_inverse_scale,
                                bias,
                                use_fast_accum,
                            )

                        args, _ = sa.call_args
                        _, choices, _, _ = args
                        cuda_template_count = 0
                        for choice in choices:
                            if isinstance(choice, CUDATemplateCaller):
                                choice_info = choice.info_dict()
                                op_conf_name = choice_info.get("op_conf_name", "")
                                assert isinstance(op_conf_name, str)
                                if use_fast_accum:
                                    assert "fastaccum" in op_conf_name, (
                                        "Only fastaccum Kernels should have been allowed"
                                    )
                                else:
                                    assert "fastaccum" not in op_conf_name, (
                                        "fastaccum Kernels should have been filtered"
                                    )
                                cuda_template_count += 1
                        assert cuda_template_count > 0, "No CUDATemplateCaller choices"

        run_test(True)
        run_test(False)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_shape_coverage_mm(
        self,
    ):
        """
        Checks if cutlass backend produces some ops for a variety of shapes.

        This test doesn't compile and check the correctness of the ops.

        NOTE: K has to be even.
        """

        inputs = [
            (torch.randn(128, 500).cuda().half(), torch.randn(500, 576).cuda().half()),
            (
                torch.randn(500, 128).cuda().half(),
                torch.randn(128, 576).cuda().half(),
            ),
            (torch.randn(128, 250).cuda().half(), torch.randn(250, 576).cuda().half()),
            (
                torch.randn(250, 128).cuda().half(),
                torch.randn(128, 576).cuda().half(),
            ),
            (
                torch.randn(125, 128).cuda().half(),
                torch.randn(128, 576).cuda().half(),
            ),
        ]

        with (
            fresh_cache(),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cuda.cutlass_max_profiling_configs": 2,
                }
            ),
            mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                wraps=select_no_algorithm,
            ) as sa,
        ):
            for input in inputs:
                A, B = input
                M, K = A.shape
                _, N = B.shape

                with self.assertRaisesRegex(InductorError, r".*NoValidChoicesError.*"):
                    torch.compile(torch.mm, dynamic=False)(*input)

                self.assertTrue(
                    sa.called,
                    f"autotune_select_algorithm was not called  with shape M={M}, N={N}, K={K}",
                )
                args, _ = sa.call_args
                op_name, choices, _, __ = args
                assert op_name == "mm"
                cuda_template_count = 0
                for choice in choices:
                    if isinstance(choice, CUDATemplateCaller):
                        choice_info = choice.info_dict()
                        op_conf_name = choice_info.get("op_conf_name", "")
                        assert isinstance(op_conf_name, str)
                        cuda_template_count += 1

                self.assertGreater(
                    cuda_template_count,
                    0,
                    "No CUDATemplateCaller choices found for matmul with shape "
                    f"M={M}, N={N}, K={K}",
                )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("presets", ("", "0", "0,999"))
    def test_cutlass_presets(
        self,
        presets: str,
    ):
        """
        Test if some configs can be generated with presets.
        """

        M, N, K = (128, 128, 16)
        A = torch.randn(M, K).cuda().half()
        B = torch.randn(N, K).cuda().half().t()

        with (
            fresh_cache(),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CUTLASS",
                    "cuda.cutlass_max_profiling_configs": 2,
                    "cuda.cutlass_presets": presets,
                }
            ),
            mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                wraps=select_no_algorithm,
            ) as sa,
        ):
            with self.assertRaisesRegex(InductorError, r".*NoValidChoicesError.*"):
                torch.compile(torch.mm)(A, B)

            self.assertTrue(
                sa.called,
                f"autotune_select_algorithm was not called with shape M={M}, N={N}, K={K}",
            )
            args, _ = sa.call_args
            op_name, choices, _, __ = args
            assert op_name == "mm"
            cuda_template_count = 0
            for choice in choices:
                if isinstance(choice, CUDATemplateCaller):
                    choice_info = choice.info_dict()
                    op_conf_name = choice_info.get("op_conf_name", "")
                    assert isinstance(op_conf_name, str)
                    cuda_template_count += 1

            self.assertGreater(
                cuda_template_count,
                0,
                "No CUDATemplateCaller choices found for matmul with shape "
                f"M={M}, N={N}, K={K}",
            )

    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_get_max_alignment(self):
        l4 = FixedLayout(
            torch.device("cpu"), torch.half, size=[1, 2, 4], stride=[0, 4, 1]
        )
        m4 = get_max_alignment(l4)
        self.assertEqual(
            m4, 4, "Wrong max alignment. Should have been 4. (simple, contiguous case)"
        )

        l4_2 = FixedLayout(
            torch.device("cpu"), torch.half, size=[1, 4, 2], stride=[0, 1, 4]
        )
        m4_2 = get_max_alignment(l4_2)
        self.assertEqual(
            m4_2,
            4,
            "Wrong max alignment. Should have been 4. Did not deal with strides correctly",
        )

        l1 = FixedLayout(
            torch.device("cpu"), torch.half, size=[2, 4, 2], stride=[23, 1, 4]
        )
        m1 = get_max_alignment(l1)
        self.assertEqual(
            m1,
            1,
            "Wrong max alignment. Should have been 1. Did not take stride into account correctly",
        )

        l2 = FixedLayout(
            torch.device("cpu"), torch.half, size=[1, 2, 4], stride=[0, 4, 1], offset=6
        )
        m2 = get_max_alignment(l2)
        self.assertEqual(
            m2, 2, "Wrong max alignment. Should have been 2. (due to choice of offset)"
        )

        l8 = FixedLayout(
            torch.device("cpu"),
            torch.half,
            size=[2, 2, 8],
            stride=[32, 8, 1],
            offset=24,
        )
        m8 = get_max_alignment(l8)
        self.assertEqual(m8, 8, "Wrong max alignment. Should have been 8.")

        l4 = FixedLayout(
            torch.device("cpu"),
            torch.float32,
            size=[2, 2, 8],
            stride=[32, 8, 1],
            offset=24,
        )
        m4 = get_max_alignment(l4)
        self.assertEqual(
            m4, 4, "Wrong max alignment. Should have been 4 (due to float32 dtype )."
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_standalone_runner(self):
        max_autotune_gemm_backends = "CUTLASS"

        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(128, 16).cuda().half().t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
                "cuda.generate_test_runner": True,  # put standalone runner in the generated code
            }
        ):
            from tempfile import NamedTemporaryFile

            from torch._inductor.codegen.cuda.cutlass_utils import (
                cuda_standalone_runner_compile_command,
                CUDACompileSourceCapturingContext,
            )

            # Run compilation, check results just in case, and save
            # CUTLASS-based generated code.
            with CUDACompileSourceCapturingContext() as ctx:
                compiled = torch.compile(torch.mm, dynamic=False)

                expected = torch.mm(a, b)
                actual = compiled(a, b)

                torch.testing.assert_close(actual, expected)

                sources = ctx.sources

            assert len(sources) >= 1

            # Get names for temporary source and executable files.
            cu_file = NamedTemporaryFile("w", suffix=".cu", delete=False)
            cu_file.close()
            exe_file = NamedTemporaryFile("w", suffix="", delete=False)
            exe_file.close()

            # Save the generated code into the .cu file.
            with open(cu_file.name, "w") as file:
                file.write(sources[0])

            # Get command to compile .cu file, and run the
            # compilation.
            command = cuda_standalone_runner_compile_command(
                Path(cu_file.name), Path(exe_file.name)
            )

            if IS_FBCODE:
                # hack to bypass the following error:
                # error while loading shared libraries: IX}: invalid mode for dlopen(): Invalid argument
                platform_path = sysconfig.get_config_var("LIBDIR")
                cuda_path = os.path.realpath(os.path.join(platform_path, "libcuda.so"))
                command = command.replace("-lcuda ", f"-L{cuda_path} ")

            repro_message = (
                f"Reproduce with: {command}\n"
                f"exe_file.name: {exe_file.name}\n"
                f"cu_file.name: {cu_file.name}\n"
            )

            retcode = os.system(command)
            self.assertEqual(retcode, 0, repro_message)

            # Run the executable generated.
            if not IS_FBCODE or not IN_RE_WORKER:
                retcode = os.system(exe_file.name)
                self.assertEqual(retcode, 0, repro_message)

            # Remove temporary files.
            os.remove(cu_file.name)
            os.remove(exe_file.name)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_integration(self):
        """
        Test if cutlass backend can be autotune with other backends
        """

        def mm(a, b):
            return a @ b

        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(128, 16).cuda().half().t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
                # needed for log searching
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
            }
        ):
            with (
                log_settings("+inductor"),
                self.assertLogs(
                    logger="torch._inductor.codegen.cuda", level=logging.DEBUG
                ) as test_log,
            ):
                Y_compiled = torch.compile(mm, dynamic=False)(a, b)
                Y = mm(a, b)
                torch.testing.assert_close(Y_compiled, Y)

            output = "\n".join(record.getMessage() for record in test_log.records)

            match = re.search(
                r"Got cutlass configs: total number of ops: (\d+)", output
            )
            assert match, "Expect to find the cutlass configs log"
            num_ops = int(match.group(1))
            self.assertTrue(num_ops > 0, "The number of ops should be greater than 0")

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_cutlass_backend_matmul_same_tensor(self):
        max_autotune_gemm_backends = "CUTLASS"

        M = 128
        A = torch.randn(M, M).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            compiled = torch.compile(torch.mm)

            torch.testing.assert_close(A @ A.t(), compiled(A, A.t()))

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_flexible_layout(self):
        class TestModel(torch.nn.Module):
            def forward(self, B):
                A = torch.zeros_like(B)
                return A @ B.t()

        M = 1024
        B = torch.randn(M, M).cuda().half()
        model = TestModel().cuda()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 1,
            }
        ):
            _ = torch.compile(model)(B)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @use_evt_config
    def test_evt_flexible_layout(self):
        class TestModel(torch.nn.Module):
            def forward(self, B):
                A = torch.zeros_like(B)
                return (A @ B.t()).relu()

        M = 1024
        B = torch.randn(M, M).cuda().half()
        model = TestModel().cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 1,
            }
        ):
            _ = torch.compile(model)(B)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cuda_epilogue_fusion_counter"], 1
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_filtered_ops_cache(self):
        class TestModel(torch.nn.Module):
            def forward(self, B):
                A = torch.zeros_like(B)
                for _ in range(100):
                    A = A @ B.t()
                return A

        M = 1024
        B = torch.randn(M, M).cuda().half()
        model = TestModel().cuda()

        start_time = time.time()
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 1,
            }
        ):
            _ = torch.compile(model)(B)
        self.assertTrue(time.time() - start_time < 60)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("use_aoti", (False, True))
    def test_compilation_time(self, use_aoti):
        M = 1024
        A = torch.randn(M, M).cuda().half()
        B = torch.randn(M, M).cuda().half().t()

        class MyModel(torch.nn.Module):
            def forward(self, a, b):
                return a @ b

        model = MyModel().cuda()
        expected = model(A, B)

        start_time = time.time()
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 1,
            }
        ):
            if use_aoti:
                actual = AOTIRunnerUtil.run(
                    model,
                    (A, B),
                )
            else:
                actual = torch.compile(model, fullgraph=True)(A, B)

            torch.testing.assert_close(actual, expected)
        self.assertTrue(time.time() - start_time < 50)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_all_ops
    @evt_all_shapes
    def test_evt_fusions_basic(self, op, shape):
        class TestModel(torch.nn.Module):
            def forward(self, a, b, extra_args):
                res = (a @ b).relu()  # add extra activation to not hit addmm path
                return op(res, *extra_args)

        self.run_evt_test(TestModel(), op, shape)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_bin_ops
    def test_evt_broadcasting(self, op):
        class TestModel(torch.nn.Module):
            def forward(self, a, b, extra_args):
                acc = a @ b
                return acc, op(acc.relu(), *extra_args)

        M = 1024
        N = 512
        a = torch.ones(M, N).cuda().half()
        b = torch.ones(N, N).cuda().half().t()
        extra_args = gen_args(op, (M, N))
        model = TestModel().cuda()

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cuda_epilogue_fusion_counter"], 1
        )
        torch.testing.assert_close(result, ref_result)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_all_ops
    def test_evt_mixed_dtypes(self, op):
        M = 1024
        N = 256

        fp32_tensor = torch.ones(M, N).cuda().float()

        class TestModel(torch.nn.Module):
            def forward(self, a, b, extra_args):
                acc = a @ b
                out0 = op(acc.relu(), *extra_args)
                out1 = torch.add(out0, fp32_tensor)
                return out1

        model = TestModel().cuda()
        a = torch.ones(M, N).cuda().half()
        b = torch.ones(N, N).cuda().half().t()
        extra_args = gen_args(op, (M, N), dtype=torch.float16)

        # baseline is cutlass kernel + triton
        # matches expected casting behavior
        with config.patch({"cuda.cutlass_epilogue_fusion_enabled": False}):
            ref_result = torch.compile(model)(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cuda_epilogue_fusion_counter"], 0
        )

        torch._dynamo.reset()
        result = torch.compile(model)(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cuda_epilogue_fusion_counter"],
            1,
        )

        torch.testing.assert_close(result, ref_result)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_all_ops
    def test_evt_multi_op(self, op):
        class TestModel(torch.nn.Module):
            def forward(self, a, b, extra_args):
                acc = a @ b
                return torch.add(op(acc.relu(), *extra_args).relu(), acc)

        self.run_evt_test(TestModel(), op, (1024, 512))

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_all_ops
    def test_evt_reuse_matmul_input(self, op):
        class TestModel(torch.nn.Module):
            def forward(self, a, b, extra_args):
                acc = a @ b
                return torch.add(op(acc.relu(), *extra_args).relu(), a)

        self.run_evt_test(TestModel(), op, (1024, 1024))  # shape needs to be square

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    @evt_all_ops
    @parametrize(
        "dynamic", (False, True)
    )  # To not drastically increase test time we only test dynamic on this test
    def test_evt_multi_output(self, op, dynamic):
        class TestModel(torch.nn.Module):
            def forward(self, a, b, extra_args):
                acc = a @ b
                z0 = acc.relu()
                z = op(z0, *extra_args)
                y = z + z0
                return z, y

        M = 1024
        N = 512
        shapes = [(512, 512)] if not dynamic else [(1024, 64), (128, 256)]
        for i, shape in enumerate(shapes):
            M, N = shape
            a = torch.ones(M, N).cuda().half()
            b = torch.ones(N, N).cuda().half().t()
            extra_args = gen_args(op, (M, N))
            model = TestModel().cuda()

            result = torch.compile(model)(a, b, extra_args)
            ref_result = model(a, b, extra_args)

            self.assertEqual(
                torch._dynamo.utils.counters["inductor"][
                    "cuda_epilogue_fusion_counter"
                ],
                2 * (i + 1),
            )
            torch.testing.assert_close(result, ref_result)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @use_evt_config
    def test_evt_return_accumulator(self):
        op = torch.add

        class TestModel(torch.nn.Module):
            def forward(self, a, b, extra_args):
                acc = a @ b
                return acc, op(acc.relu(), *extra_args)

        M = 1024
        N = 512
        a = torch.ones(M, N).cuda().half()
        b = torch.ones(N, N).cuda().half().t()
        extra_args = gen_args(op, (M, N))
        model = TestModel().cuda()

        result = torch.compile(model)(a, b, extra_args)
        ref_result = model(a, b, extra_args)

        self.assertEqual(
            torch._dynamo.utils.counters["inductor"]["cuda_epilogue_fusion_counter"], 1
        )
        torch.testing.assert_close(result, ref_result)

    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    @parametrize("arch", ("90", "100"))
    @parametrize("cuda_version", ("12.4", "12.8"))
    def test_gemm_operation_serialization(self, arch: str, cuda_version: str):
        """
        Testing serialization for GEMM operations generated by CUTLASS.
        This should cover GroupedGemmOperation as well.
        """
        full_ops = _gen_ops_cached(arch, cuda_version)
        ops = pytree.tree_flatten(full_ops)[0]
        self.assertGreater(len(ops), 1000, "Too few ops generated")

        serializer = get_cutlass_operation_serializer()
        self.assertIsNotNone(serializer)

        serialized_ops = [serializer.serialize(op) for op in ops]
        deserialized_ops = [
            serializer.deserialize(serialized_op) for serialized_op in serialized_ops
        ]
        for op, deserialized_op in zip(ops, deserialized_ops):
            self.assertTrue(_check_if_instances_equal(op, deserialized_op))

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 is only supported on H100+")
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @fp8_config
    @parametrize("float8_dtype", (torch.float8_e4m3fn,))
    @parametrize(
        "shape",
        (
            (
                512,
                128,
                64,
            ),
        ),
    )
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False,))
    def test_fp8_rowwise_scaling(
        self,
        float8_dtype: torch.dtype,
        shape: tuple[int, int, int],
        has_bias: bool,
        use_fast_accum: bool,
    ):
        # Only bf16 output type is supported for row-wise scaling, not fp32
        output_dtype: torch.dtype = torch.bfloat16
        device = "cuda"
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        x = torch.randn(M, K, dtype=output_dtype, device=device)
        w = torch.randn(N, K, dtype=output_dtype, device=device)
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_rowwise(w, float8_dtype)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale = w_inverse_scale.t()  # scale_b should be (1, N)

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_rowwise(x, float8_dtype)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=output_dtype,
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
        linear_compiled = torch.compile(linear, backend="inductor")
        y_compiled = linear_compiled(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        self.assertEqual(y_eager.dtype, output_dtype)
        self.assertEqual(y_compiled.dtype, output_dtype)
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 is only supported on H100+")
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @fp8_config
    @parametrize("float8_dtype", (torch.float8_e4m3fn,))
    @parametrize(
        "shape",
        (
            (
                512,
                128,
                64,
            ),
        ),
    )
    @parametrize("has_bias", (False, True))
    @parametrize("use_fast_accum", (False,))
    def test_fp8_tensorwise_scaling(
        self,
        float8_dtype: torch.dtype,
        shape: tuple[int, int, int],
        has_bias: bool,
        use_fast_accum: bool,
    ):
        device = "cuda"
        M, K, N = shape  # Matmul Y = X [M, K] x W [N, K]
        input_dtype = torch.bfloat16
        output_dtype = torch.bfloat16
        # input and output dtypes of _scaled_mm do not need to be the same, but
        # typically in a model they are
        x = torch.randn(M, K, dtype=input_dtype, device=device)
        w = torch.randn(N, K, dtype=input_dtype, device=device)
        bias = None
        if has_bias:
            bias = torch.randn(N, device=device, dtype=torch.bfloat16)

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = _quantize_tensorwise(w, float8_dtype)
        w_t_fp8 = w_fp8.t()

        # quantize input x
        x_fp8, x_inverse_scale = _quantize_tensorwise(x, float8_dtype)

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=output_dtype,
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
        linear_compiled = torch.compile(linear, backend="inductor", mode="max-autotune")
        y_compiled = linear_compiled(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale,
            bias,
        )
        self.assertEqual(y_eager.dtype, output_dtype)
        self.assertEqual(y_compiled.dtype, output_dtype)
        # depending on the kernel config (BLOCK_M size, etc) selected during Inductor
        # autotuning for the compiled case, the results can be different because of
        # the way blocks of results are accumulated (float addition not associative), so
        # setting a small absolute tolerance in these tests
        torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("layout", ("rr", "rc"))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_config_number_post_filtering(
        self, layout: str, dtype: torch.dtype
    ) -> None:
        """
        Test if cutlass backend produces the same number of configs after filtering.
        """
        a = torch.randn(128, 128, dtype=dtype).cuda()
        b = torch.randn(128, 128, dtype=dtype).cuda()
        if layout[0] == "c":
            a = a.t()
        if layout[1] == "c":
            b = b.t()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                # needed for log searching
                "force_disable_caches": True,
                "cuda.cutlass_max_profiling_swizzle_options": [2],
            }
        ):
            with mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                wraps=select_no_algorithm,
            ) as sa:
                with self.assertRaisesRegex(
                    BackendCompilerFailed, r".*NoValidChoicesError.*"
                ):
                    _ = torch.compile(torch.mm, dynamic=False)(a, b)
                args, _ = sa.call_args
                _, choices, _, __ = args

                self.assertEqual(len(choices), DEFAULT_INST_LEVEL_MM_CONFIG)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA and HAS_CPU and is_big_gpu():
        run_tests()
