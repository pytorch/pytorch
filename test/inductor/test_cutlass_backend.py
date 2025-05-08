# Owner(s): ["module: inductor"]
import logging
import math
import os
import re
import sysconfig
import time
import unittest
import unittest.mock as mock
from pathlib import Path
from typing import Callable, Optional

from torch._inductor.utils import clear_inductor_caches
from torch.export import Dim
from torch.testing._internal.logging_utils import log_settings


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
from torch._inductor.codegen.cuda.cutlass_utils import get_max_alignment
from torch._inductor.exc import InductorError
from torch._inductor.ir import FixedLayout
from torch._inductor.select_algorithm import NoValidChoicesError
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, SM90OrLater
from torch.testing._internal.common_utils import (
    IN_RE_WORKER,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


torch.set_float32_matmul_precision("high")
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


log = logging.getLogger(__name__)


def _get_path_without_sccache() -> str:
    """
    Get the PATH environment variable without sccache.
    """
    path_envs = os.environ.get("PATH", "").split(":")
    path_envs = [env for env in path_envs if "/opt/cache/bin" not in env]
    return ":".join(path_envs)


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
            os.environ[
                "INDUCTOR_TEST_DISABLE_FRESH_CACHE"
            ] = old_disable_fresh_cache_envvar
        torch.random.manual_seed(1234)

    def tearDown(self):
        super().tearDown()
        clear_inductor_caches()

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_threshold(self):
        """
        Make sure Cutlass GEMM threshold works as intended.
        """

        def mm(a, b):
            return a @ b

        a = torch.randn(100, 10).cuda().half()
        b = torch.randn(10, 100).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_backend_min_gemm_size": 100000,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):

            def select_no_algorithm(*args, **kwargs):
                raise NoValidChoicesError

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

        import cutlass  # noqa: F401
        import cutlass_library  # noqa: F401

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
        b = torch.randn(K, N).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_max_profiling_configs": 4,
                "autotune_fallback_to_aten": False,
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
        b = torch.randn(K, N).cuda().half()

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
                "autotune_fallback_to_aten": False,
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
        b = torch.randn(B, K, N).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_max_profiling_configs": 4,
                "autotune_fallback_to_aten": False,
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
        b = torch.randn(16, 128).cuda().half()
        c = torch.randn(16, 512).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 1,
                "autotune_fallback_to_aten": False,
            }
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, dynamic=dynamic)
            expected = model(a, b, c)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            FileCheck().check_count(
                "cuda_fused_0.cuda_fused_0",
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
        b = torch.randn(16, 128).cuda().half()
        c = torch.randn(16, 512).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 1,
                "autotune_fallback_to_aten": False,
                "cuda.cutlass_max_profiling_swizzle_options": [
                    1,
                    2,
                    4,
                ],  # guarantees > 1 choices
                "force_disable_caches": True,
            }
        ):
            from torch._inductor.utils import run_and_get_code

            compiled = torch.compile(model, dynamic=True)
            expected = model(a, b, c)
            actual, codes = run_and_get_code(compiled, a, b, c)
            torch.testing.assert_close(actual, expected)
            FileCheck().check_count(
                "cuda_fused_0.cuda_fused_0",
                1,
            ).run(codes[0])
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

        class MyModel(torch.nn.Module):
            def forward(self, a, b):
                return a @ b

        model = MyModel().cuda()
        # M, N, K
        shapes = [
            (128, 128, 16),
            (1024, 1024, 256),
        ]
        shapes = shapes[0:1] if not dynamic else shapes
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

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_fallback_to_aten": False,
            }
        ), dynamo_config.patch({"error_on_recompile": dynamic}):
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
            clear_inductor_caches()

            inputs = [
                (
                    torch.randn(x_shape(M, N)).cuda().to(dtype),
                    torch.randn(M, K).cuda().to(dtype),
                    torch.randn(K, N).cuda().to(dtype),
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
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "cuda.cutlass_max_profiling_configs": 2,
                    "autotune_fallback_to_aten": False,
                }
            ), dynamo_config.patch({"error_on_recompile": dynamic}):
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
                "autotune_fallback_to_aten": False,
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
        b = torch.randn(16, 128).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
                "cuda.cutlass_op_allowlist_regex": "stream_k",  # only stream-k GEMM Kernels
                "autotune_fallback_to_aten": False,
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
                b = torch.randn(K, N).cuda().half()
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
            b = torch.randn(32, 256).cuda()
        else:
            a = torch.randn(batch_size, 256, 32).cuda()
            b = torch.randn(batch_size, 32, 256).cuda()
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
                "autotune_fallback_to_aten": False,
            }
        ):
            counters["inductor"]["cuda_epilogue_fusion_counter"] = 0
            assert mm is not None
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            actual_count = counters["inductor"]["cuda_epilogue_fusion_counter"]
            assert (
                actual_count == expected_fuse_count
            ), f"Expected fuse count of {expected_fuse_count} but got {actual_count}"
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
                "autotune_fallback_to_aten": False,
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
                "autotune_fallback_to_aten": False,
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
            w = torch.randn(K, N).cuda().half()

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
                "autotune_fallback_to_aten": False,
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
            w = torch.randn(K, N).cuda().half()

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
                "autotune_fallback_to_aten": False,
                "cuda.cutlass_op_allowlist_regex": "128x256x64.*stream_k_warpspecialized_cooperative_epi_nosmem",
                "cuda.cutlass_max_profiling_configs": 1,
            }
        ):
            model = MyModel()
            M, N, K = 200, 5216, 10_432

            x = torch.randn(M, K).cuda().half()
            w = torch.randn(K, N).cuda().half()

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
                "autotune_fallback_to_aten": False,
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
        b = torch.randn(128, 128).cuda().half()

        def select_no_algorithm(*args, **kwargs):
            raise NoValidChoicesError

        with fresh_inductor_cache():
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
                            assert (
                                "pingpong" not in op_conf_name
                            ), "All pingpong Kernels should have been filtered"
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
        b = torch.randn(128, 128).cuda().half()

        def select_no_algorithm(*args, **kwargs):
            raise NoValidChoicesError

        with fresh_inductor_cache():
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
                            assert (
                                "pingpong" in op_conf_name
                            ), "Only pingpong Kernels should have been allowed"
                            cuda_template_count += 1
                    assert cuda_template_count > 0, "No CUDATemplateCaller choices"

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

        def select_no_algorithm(*args, **kwargs):
            raise NoValidChoicesError

        with fresh_inductor_cache(), config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_fallback_to_aten": False,
            }
        ), mock.patch(
            "torch._inductor.kernel.mm.autotune_select_algorithm",
            wraps=select_no_algorithm,
        ) as sa:
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
        B = torch.randn(K, N).cuda().half()

        def select_no_algorithm(*args, **kwargs):
            raise NoValidChoicesError

        with fresh_inductor_cache(), config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_fallback_to_aten": False,
                "cuda.cutlass_presets": presets,
            }
        ), mock.patch(
            "torch._inductor.kernel.mm.autotune_select_algorithm",
            wraps=select_no_algorithm,
        ) as sa:
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
        b = torch.randn(16, 128).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_fallback_to_aten": False,
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
        b = torch.randn(16, 128).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
                # needed for log searching
                "force_disable_caches": True,
            }
        ):
            with log_settings("+inductor"), self.assertLogs(
                logger="torch._inductor.codegen.cuda", level=logging.DEBUG
            ) as test_log:
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
                "autotune_fallback_to_aten": False,
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
                return A @ B

        M = 1024
        B = torch.randn(M, M).cuda().half()
        model = TestModel().cuda()

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 1,
                "autotune_fallback_to_aten": False,
            }
        ):
            _ = torch.compile(model)(B)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_filtered_ops_cache(self):
        class TestModel(torch.nn.Module):
            def forward(self, B):
                A = torch.zeros_like(B)
                for _ in range(100):
                    A = A @ B
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
        self.assertTrue(time.time() - start_time < 100)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA and HAS_CPU and is_big_gpu():
        run_tests()
