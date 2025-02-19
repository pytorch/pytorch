# Owner(s): ["module: inductor"]
import logging
import math
import os
import unittest
import unittest.mock as mock
from pathlib import Path
from typing import Callable, Optional

from torch._inductor.utils import clear_inductor_caches
from torch.export import Dim


try:
    from test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from .test_aot_inductor_utils import AOTIRunnerUtil

import torch
import torch._inductor.codecache
import torch.version
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller
from torch._inductor.codegen.cuda.cutlass_utils import get_max_alignment
from torch._inductor.ir import ChoiceCaller, FixedLayout
from torch._inductor.select_algorithm import NoValidChoicesError
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM80OrLater, SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
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
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "compile_threads": 4,
                "cuda.cutlass_backend_min_gemm_size": 100000,
                "cuda.cutlass_max_profiling_configs": 2,
                # allow fallback to aten as intended
                "autotune_fallback_to_aten": True,
            }
        ):
            from torch._inductor.codegen.cuda.cuda_kernel import CUDATemplateCaller

            with mock.patch(
                "torch._inductor.select_algorithm.autotune_select_algorithm"
            ) as mocked_select_algorithm:
                Y_compiled = torch.compile(mm, dynamic=False)(a, b)
                Y = mm(a, b)
                passed_choice_callers: list[ChoiceCaller] = mocked_select_algorithm[0][
                    1
                ]
                assert all(
                    isinstance(cc, ChoiceCaller) for cc in passed_choice_callers
                ), "Argument 1 to autotune_select_algorithm should be a list of ChoiceCaller instances"
                # We expect that no Cutlass Kernels are considered, due to the threshold
                assert all(
                    not isinstance(cc, CUDATemplateCaller)
                    for cc in passed_choice_callers
                ), "Cutlass Kernels should have been filtered, GEMM size is too small"
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_precompile(self):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            return a @ b

        a = torch.randn(100, 10).cuda().half()
        b = torch.randn(10, 100).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS,Triton,ATen",
                "compile_threads": 4,
                "cuda.cutlass_max_profiling_configs": 2,
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_aoti_rerun_with_different_shapes(self):
        """
        Compile with one shape, then re-run with different input shapes
        """
        max_autotune_gemm_backends = "CUTLASS"

        class MyModel(torch.nn.Module):
            def forward(self, a, b):
                return a @ b

        model = MyModel()
        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(16, 512).cuda().half()
        x = torch.randn(256, 32).cuda().half()
        y = torch.randn(32, 256).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 3,
                "autotune_fallback_to_aten": False,
            }
        ):
            from torch.export import Dim

            M = Dim("M", min=1, max=1024)
            N = Dim("N", min=1, max=1024)
            K = Dim("K", min=1, max=1024)
            dynamic_shapes = {
                "a": {0: M, 1: K},
                "b": {0: K, 1: N},
            }

            actual = AOTIRunnerUtil.run_multiple(
                "cuda",
                model,
                [(a, b), (x, y)],
                dynamic_shapes=dynamic_shapes,
            )
            expected = [model(a, b), model(x, y)]
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])

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

    # NOTE: right now tuned_mm doesn't support cutlass 2x, which is used by A100
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False, True))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @parametrize("use_aoti", (False, True))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_regular_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str, use_aoti: bool
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                return a @ b

        model = MyModel()
        a = torch.randn(128, 16).cuda().half()
        b = torch.randn(16, 128).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_fallback_to_aten": False,
            }
        ):
            Y = model(a, b)
            if use_aoti:
                Y_compiled = AOTIRunnerUtil.run(
                    "cuda",
                    model,
                    (a, b),
                )
            else:
                Y_compiled = torch.compile(model, dynamic=dynamic)(a, b)
            torch.testing.assert_close(Y_compiled, Y)

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
    def test_max_autotune_cutlass_backend_simple_bmm(self):
        def bmm(a, b):
            return torch.bmm(a, b)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(  # test bmm
            fp16=True,
            expected_fuse_count=0,
            mm=bmm,
            batch_size=10,
        )

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    def test_max_autotune_cutlass_backend_shape_dependent_normalization_fusion(self):
        def mm(a, b):
            return (a @ b) / b.size(1)

        self._test_max_autotune_cutlass_backend_epilogue_fusion(
            fp16=True, expected_fuse_count=0, mm=mm
        )

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(True, "FIXME: Disabled temporarily since crashing in subprocess")
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,Triton,CUTLASS"))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_mm_bias(
        self, dynamic: bool = False, max_autotune_gemm_backends: str = "CUTLASS"
    ):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b, bias):
            return torch.nn.functional.linear(a, b, bias)

        a = torch.randn(2048, 4096).cuda().half()
        bias = torch.randn(2048).cuda().half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_fallback_to_aten": False,
            }
        ):
            Y = mm(a, a, bias)
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, a, bias)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-1, rtol=1e-1)

    @unittest.skipIf(True, "FIXME: Disabled temporarily since crashing in subprocess")
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "ATen,CUTLASS"))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_addmm(
        self, dynamic, max_autotune_gemm_backends
    ):
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        def addmm(x, a, b, alpha, beta):
            return torch.addmm(x, a, b, alpha=alpha, beta=beta)

        def compare_results(
            m: int, k: int, n: int, alpha: float, beta: float, x_shape: list[int]
        ) -> None:
            x = torch.randn(x_shape).cuda().half()
            a = torch.randn(m, k).cuda().half()
            b = torch.randn(k, n).cuda().half()
            y_expected = addmm(x, a, b, alpha, beta)

            compiled_fn = torch.compile(addmm, dynamic=dynamic)
            y = compiled_fn(x, a, b, alpha, beta)
            torch.testing.assert_close(y, y_expected)

        with config.patch(
            {
                "max_autotune": True,
                # Some Cutlass Kernels fail with IMA on this example, which leads to unrecoverable CUDA errors
                # unless we tune in a subproc here.
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 4,
                "cuda.cutlass_op_allowlist_regex": "",
                "cuda.cutlass_op_denylist_regex": "pingpong",  # Pingpong Kernels can lead to numerical issues
                "autotune_fallback_to_aten": False,
            }
        ):
            # No broadcast
            compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 2048])
            # Broadcast first dim.
            compare_results(4096, 25728, 2048, 2.0, 0.4, [2048])
            # Broadcast last dim.
            compare_results(4096, 25728, 2048, 2.0, 0.4, [4096, 1])

    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_addmm_with_expanded_bias(self):
        class MyModel(torch.nn.Module):
            def forward(self, x, w):
                bias = torch.zeros(
                    size=(w.size(1), x.size(0)), dtype=torch.float16, device="cuda"
                ).t()
                return torch.addmm(bias, x, w)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": False,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 1,
                "autotune_fallback_to_aten": False,
            }
        ):
            model = MyModel()
            M, N, K = 2048, 3072, 6144
            x = torch.randn(M, K).cuda().half()
            w = torch.randn(K, N).cuda().half()

            actual = AOTIRunnerUtil.run(
                "cuda",
                model,
                (x, w),
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM90OrLater, "need sm_90")
    @parametrize("dynamic", (False,))
    @parametrize("max_autotune_gemm_backends", ("CUTLASS", "CUTLASS,ATen"))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_int_mm(
        self, dynamic: bool, max_autotune_gemm_backends: str
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
                "cuda",
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
                "x": {0: Dim.DYNAMIC},  # type: ignore[attr-defined]
                "w": None,
            }

            x = torch.randn(M, K).cuda().half()
            w = torch.randn(K, N).cuda().half()

            actual = AOTIRunnerUtil.run(
                "cuda",
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
                "cuda",
                model,
                (x, w),
            )
            expected = model(x, w)
            torch.testing.assert_close(expected, actual, atol=0.01, rtol=0.01)

    # TODO: Enable dynamic test cases when dynamic support is added.
    @unittest.skipIf(not SM80OrLater or SM90OrLater, "need sm_8x exactly")
    @parametrize("dynamic", (False,))
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_max_autotune_cutlass_backend_mixed_mm(self, dynamic: bool):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            return torch.mm(a, b.to(torch.half))

        # CUTLASS only supports row-major/column-major combination of
        # layouts for this operation, thus the transpose of tensor b.
        # Also, for CUTLASS alignment requirements, number of columns
        # of the first tensor has to be divisible by 16.
        m, n, k = 100, 16, 100
        a = torch.randn(m, k).cuda().half()
        b = torch.randint(0, 5, (n, k), dtype=torch.int8).cuda().T

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "CUTLASS",
                "cuda.cutlass_max_profiling_configs": 2,
                "autotune_local_cache": True,
                "autotune_fallback_to_aten": False,
                "use_mixed_mm": True,
                "mixed_mm_choice": "aten",  # to disable Triton
            }
        ):
            Y_compiled = torch.compile(mm, dynamic=dynamic)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

        cache = torch._inductor.codecache.LocalCache().lookup("mixed_mm")
        assert cache is not None
        high = cache[
            f"[('cuda', 'torch.float16', {m}, {k}, {k}, 1, 0), "
            f"('cuda', 'torch.int8', {k}, {n}, 1, {k}, 0)]"
        ]["high"]
        cutlass_kernels_count = 0
        for kernel, time in high.items():
            if kernel.startswith("cutlass_gemm") and not math.isinf(time):
                cutlass_kernels_count += 1
        assert cutlass_kernels_count > 0

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
        for kernel, time in high.items():
            if kernel.startswith("cutlass_gemm") and not math.isinf(time):
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
                    # Some Cutlass Kernels fail with IMA on this example, which leads to unrecoverable CUDA errors
                    # unless we tune in a subproc here.
                    "autotune_in_subproc": False,
                    "max_autotune_gemm_backends": "CUTLASS,ATen",
                    "cuda.cutlass_max_profiling_configs": 2,
                    "cuda.cutlass_op_allowlist_regex": "",
                    "cuda.cutlass_op_denylist_regex": "pingpong",  # Pingpong Kernels can lead to numerical issues
                }
            ):
                with mock.patch(
                    "torch._inductor.kernel.mm.autotune_select_algorithm",
                    wraps=select_no_algorithm,
                ) as sa:
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
                    # Some Cutlass Kernels fail with IMA on this example, which leads to unrecoverable CUDA errors
                    # unless we tune in a subproc here.
                    "autotune_in_subproc": False,
                    "max_autotune_gemm_backends": "CUTLASS,ATen",
                    "cuda.cutlass_max_profiling_configs": 2,
                    "cuda.cutlass_op_allowlist_regex": "pingpong",
                    "cuda.cutlass_op_denylist_regex": None,  # Pingpong Kernels can lead to numerical issues
                }
            ):
                with mock.patch(
                    "torch._inductor.kernel.mm.autotune_select_algorithm",
                    wraps=select_no_algorithm,
                ) as sa:
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

    @unittest.skipIf(not SM80OrLater, "need sm_80")
    @mock.patch.dict(os.environ, {"PATH": _get_path_without_sccache()})
    def test_standalone_runner(self):
        max_autotune_gemm_backends = "CUTLASS"
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def mm(a, b):
            return torch.mm(a, b.to(torch.half))

        m, n, k = 128, 16, 128
        a = torch.randn(m, k).cuda().half()
        b = torch.randint(0, 5, (n, k), dtype=torch.int8).cuda().T

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "cuda.cutlass_max_profiling_configs": 1,
                "autotune_local_cache": True,
                "autotune_fallback_to_aten": False,
                "cuda.generate_test_runner": True,  # put standalone runner in the generated code
                "use_mixed_mm": True,
                "mixed_mm_choice": "aten",
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
                compiled = torch.compile(mm, dynamic=False)

                expected = mm(a, b)
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
            retcode = os.system(command)
            assert retcode == 0

            # Run the executable generated.
            retcode = os.system(exe_file.name)
            assert retcode == 0

            # Remove temporary files.
            os.remove(cu_file.name)
            os.remove(exe_file.name)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA and HAS_CPU and is_big_gpu():
        run_tests()
