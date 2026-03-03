# Owner(s): ["module: inductor"]
import contextlib
import functools
import inspect
import json
import logging
import math
import os
import random
import re
import tempfile
import time
import unittest
from collections.abc import Callable
from typing import Optional
from unittest import mock
from unittest.mock import patch

import torch
import torch._inductor.async_compile
from torch import multiprocessing as mp, nn
from torch._dynamo import reset
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.testing import rand_strided, reset_rng_state
from torch._dynamo.utils import counters, same
from torch._inductor import config
from torch._inductor.autotune_process import (
    _TestBenchmarkRequest,
    AsyncAutotuner,
    AutotuneProcessPool,
    CUDA_VISIBLE_DEVICES,
    ExternKernelBenchmarkRequest,
    TritonBenchmarkRequest,
    TuningProcess,
    TuningProcessPool,
    use_pipelined_autotuning,
)
from torch._inductor.codegen.common import WorkspaceArg
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Buffer, ChoiceCaller, FixedLayout, FlexibleLayout
from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from torch._inductor.scheduler import Scheduler
from torch._inductor.select_algorithm import (
    add_feedback_saver,
    add_preprocessing_fn,
    AlgorithmSelectorCache,
    autotune_select_algorithm,
    clear_feedback_savers,
    clear_preprocessing_fns,
    ExternKernelCaller,
    NoValidChoicesError,
    TritonTemplate,
    TritonTemplateCaller,
)
from torch._inductor.template_heuristics.registry import override_template_heuristics
from torch._inductor.template_heuristics.triton import (
    CUDAAddmmPersistentTMATemplateConfigHeuristic,
    CUDAAddMMTemplateConfigHeuristic,
    CUDAMMTemplateConfigHeuristic,
    CUDAPersistentTMATemplateConfigHeuristic,
    GemmConfig,
    get_shared_memory_checker_opts,
    XPUMMTemplateConfigHeuristic,
    XPUPersistentTMATemplateConfigHeuristic,
)
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    random_matrix_with_scaled_reduction_dim,
    skipIfRocm,
    TEST_WITH_ROCM,
    TEST_XPU,
)
from torch.testing._internal.logging_utils import multiple_logs_to_string
from torch.utils._triton import (
    has_datacenter_blackwell_tma_device,
    has_triton_stable_tma_api,
    has_triton_tma_device,
)


aten = torch.ops.aten
from torch._inductor.mock_cache import global_stats, PatchCaches, Stats
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    fresh_cache,
    get_k_splits,
    run_and_get_code,
    use_decompose_k_choice,
)
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing import FileCheck
from torch.testing._internal.common_utils import MI300_ARCH, runOnRocmArch, skipIfXpu
from torch.testing._internal.inductor_utils import (
    get_func_call,
    get_kernel_launch,
    GPU_TYPE,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
)


if torch.version.hip:
    # Temporary addition to ensure inductor tests can be
    # enabled. Currently TF32 accuracy issues cause these tests
    # to fail. We will use FP32 as reference to ensure the generated
    # triton kernels are adequately tested.
    #
    # Track in: https://github.com/pytorch/pytorch/issues/169392
    torch.set_float32_matmul_precision("highest")
else:
    torch.set_float32_matmul_precision("high")

if HAS_CUDA_AND_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

# Conditional patch for decompose_k tests - override to 10 on ROCm, no-op elsewhere
_DECOMPOSE_K_PATCH_ROCM = (
    {"triton.num_decompose_k_splits": 10} if torch.version.hip else {}
)


def benchmark_choice(choice, args, out, expected_out, timings):
    result = choice.benchmark(*args, out=out)
    if expected_out is not None:
        torch.testing.assert_close(out, expected_out)

    timings.copy_(torch.tensor(result))


class FailChoiceCaller(ChoiceCaller):
    def benchmark(self, *args, out):
        raise RuntimeError("This choice caller will always throw")


@unittest.mock.patch(
    "torch._inductor.select_algorithm.TritonTemplate.test_cache", new=True
)
@config.patch(enable_caching_generated_triton_templates=True)
@instantiate_parametrized_tests
class TestMaxAutotune(TestCase):
    def _make_matrices(self, M, K, N, *batch_dims, dtype, device, requires_grad):
        make_matrix = functools.partial(
            random_matrix_with_scaled_reduction_dim,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        a = make_matrix(M, K, *batch_dims, reduction_dim=-1)
        b = make_matrix(K, N, *batch_dims, reduction_dim=-2)
        return a, b

    @parametrize("dynamic", (False, True))
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_max_autotune_mm_plus_mm_zero_size_input(self, dynamic, search_space):
        """
        Make sure autotuning mm_plus_mm with zero-size input works without crashes.
        """
        m, n, k = 0, 1536, 64

        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        a = torch.randn(m, k).to(GPU_TYPE)
        b = torch.randn(k, n).to(GPU_TYPE)
        c = torch.randn(m, k).to(GPU_TYPE)
        d = torch.randn(k, n).to(GPU_TYPE)

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_search_space": search_space}
        ):
            torch.compile(mm_plus_mm, dynamic=dynamic)(a, b, c, d)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="XPU TMA requires contiguous last dimension")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_max_autotune_regular_mm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
    ):
        def mm(a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        if has_triton_stable_tma_api():
            make_desc_api = "triton.language.make_tensor_descriptor"
            read_api = "tl.load_tensor_descriptor"
            if tma_store:
                # Note: The tma_descriptor0 is generated by the kernel. If the
                # code generation process changes this could change.
                write_api = "tma_descriptor0.store"
            else:
                write_api = "tl.store"
        else:
            make_desc_api = (
                "triton.language.extra.cuda.experimental_device_tensormap_create2d"
            )
            read_api = "tl._experimental_descriptor_load"
            # TMA store is not supported with the experimental API
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_mm").check(make_desc_api).check(
            read_api
        ).check(write_api).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_persistent_tma_workspace_reuse(self):
        """
        Test that make_kernel_render creates unique workspace names.

        This test patches get_tma_workspace_arg to return the same WorkspaceArg
        instance, simulating the bug condition where templates share workspace_arg.
        The fix in make_kernel_render should create a new WorkspaceArg with a
        unique name for each kernel, preventing self-assignment bugs like
        'workspace_X = workspace_X; del workspace_X'.
        """
        from torch._inductor.codegen.common import WorkspaceZeroMode

        def three_same_shape_matmuls(a, b, c, d, e, f):
            x = torch.mm(a, b)
            y = torch.mm(c, d)
            z = torch.mm(e, f)
            return x, y, z

        M, K, N = 4608, 2048, 7040

        a = torch.randn(M, K, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)

        mm_tma_heuristic = CUDAPersistentTMATemplateConfigHeuristic()
        mm_heuristic = CUDAMMTemplateConfigHeuristic()

        original_tma_configs = mm_tma_heuristic.mm_configs
        original_mm_configs = mm_heuristic.mm_configs

        # Create a single WorkspaceArg to be returned by all calls
        shared_workspace_arg = WorkspaceArg(
            count=1024,
            zero_mode=WorkspaceZeroMode.UNINITIALIZED,
            device=torch.device(GPU_TYPE),
            outer_name="shared_workspace",
        )

        def mock_get_tma_workspace_arg(*args, **kwargs):
            return shared_workspace_arg

        try:
            # Force only TMA template by clearing non-TMA configs
            mm_heuristic.mm_configs = []

            # Use a single TMA config to ensure deterministic behavior
            mm_tma_heuristic.mm_configs = [GemmConfig(128, 128, 64, 4, 8, group_m=8)]

            with (
                config.patch(
                    {
                        "max_autotune_gemm": True,
                        "max_autotune_gemm_backends": "TRITON",
                        "triton.enable_persistent_tma_matmul": True,
                    }
                ),
                fresh_cache(),
                patch(
                    "torch._inductor.template_heuristics.triton.get_tma_workspace_arg",
                    mock_get_tma_workspace_arg,
                ),
            ):
                torch._dynamo.reset()
                compiled_fn = torch.compile(
                    three_same_shape_matmuls, mode="max-autotune-no-cudagraphs"
                )

                _, _ = run_and_get_code(compiled_fn, a, b, a, b, a, b)

        finally:
            mm_tma_heuristic.mm_configs = original_tma_configs
            mm_heuristic.mm_configs = original_mm_configs

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="XPU TMA requires contiguous last dimension")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_strided(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
    ):
        def mm(a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)
            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.mm(a, b)

        def next_multiple_16(a: int) -> int:
            return ((a + 15) // 16) * 16

        M, N, K = 21, 31, 11
        a_shape = (K, M) if a_transposed else (M, K)
        a_stride = (
            (next_multiple_16(M), 1) if a_transposed else (next_multiple_16(K), 1)
        )
        a = torch.empty_strided(a_shape, a_stride, dtype=torch.float16).to(GPU_TYPE)
        a[:] = torch.randn(a_shape, dtype=torch.float16)
        a = a.to(GPU_TYPE)
        b_shape = (N, K) if b_transposed else (K, N)
        b_stride = (
            (next_multiple_16(K), 1) if a_transposed else (next_multiple_16(N), 1)
        )
        b = torch.empty_strided(b_shape, b_stride, dtype=torch.float16)
        b[:] = torch.randn(b_shape, dtype=torch.float16)
        b = b.to(GPU_TYPE)
        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(torch.compile(mm, dynamic=dynamic), a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)
        # Verify that we are using a TMA implementation
        # depending on whether we're using the experimental API, we check for a different string
        check_str = "triton.language.extra.cuda.experimental_device_tensormap_create2d"
        if has_triton_stable_tma_api():
            check_str = "triton.language.make_tensor_descriptor"
        FileCheck().check("triton_tem_fused_mm").check(check_str).run(code[0])

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="Covered by XPU TMA")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_illegal_alignment(self, dynamic):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(mm, dynamic=dynamic)(a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_illegal_output_alignment(
        self, dynamic
    ):
        def mm(a, b, out):
            torch.mm(a, b, out=out)
            return out

        M, N, K = 21, 31, 32
        a = torch.empty_strided((M, K), (K, 1), dtype=torch.float16, device=GPU_TYPE)
        a[:] = torch.randn((M, K), dtype=torch.float16)
        b = torch.empty_strided((K, N), (1, K), dtype=torch.float16, device=GPU_TYPE)
        b[:] = torch.randn((K, N), dtype=torch.float16)
        # allocate an output with a stride not divisible by 16, so it can't satisfy TMA alignment checks.
        out = torch.empty_strided((M, N), (N, 1), dtype=torch.float16, device=GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "triton.enable_template_tma_store": True,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(mm, dynamic=dynamic)(a, b, out)

        # Lowering to the persistent+TMA Triton template should be skipped
        # since the output doesn't have a stride of 1 in any dim
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_regular_mm_tma_dynamic_outer_dim(self):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.mark_dynamic(a, 0)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual = torch.compile(mm)(a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_zero_size_input(self, dynamic: bool):
        """
        Make sure autotuning mm with zero-size input works without crashes.
        """

        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        with config.patch({"max_autotune": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="XPU TMA requires contiguous last dimension")
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    @parametrize("tma_store", (False, True))
    def test_max_autotune_addmm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
        tma_store: bool,
    ):
        def addmm(x, a, b):
            # TMA requires 16-byte alignment: here we repeat the dims
            # by the factor of 8, as float16 is 2-byte. All dims are
            # repeated due to the possible transpositions below.
            x = x.repeat(8)
            a = a.repeat(8, 8)
            b = b.repeat(8, 8)

            if a_transposed:
                a = a.T
            if b_transposed:
                b = b.T

            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = (
            torch.randn(*((K, M) if a_transposed else (M, K)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        b = (
            torch.randn(*((N, K) if b_transposed else (K, N)))
            .to(torch.float16)
            .to(GPU_TYPE)
        )
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "triton.enable_template_tma_store": tma_store,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual, code = run_and_get_code(
                torch.compile(addmm, dynamic=dynamic), x, a, b
            )
            c_expected = addmm(x, a, b)

        if has_triton_stable_tma_api():
            make_desc_api = "triton.language.make_tensor_descriptor"
            read_api = "tl.load_tensor_descriptor"
            if tma_store:
                # Note: The tma_descriptor0 is generated by the kernel. If the
                # code generation process changes this could change.
                write_api = "tma_descriptor0.store"
            else:
                write_api = "tl.store"
        else:
            make_desc_api = (
                "triton.language.extra.cuda.experimental_device_tensormap_create2d"
            )
            read_api = "tl._experimental_descriptor_load"
            # TMA store is not supported with the experimental API
            write_api = "tl.store"

        # Verify that we are using a TMA implementation
        FileCheck().check("triton_tem_fused_addmm").check(make_desc_api).check(
            read_api
        ).check(write_api).run(code[0])

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @skipIfXpu(msg="Covered by XPU TMA")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_persistent_tma_illegal_alignment(self, dynamic):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
                    "triton.native_matmul": False,
                    "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
                }
            ),
        ):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

        # Lowering to the persistent+TMA Triton template should be skipped
        # if any of the input inner dims are not 16-byte aligned. As a result,
        # given the config flags above, we should have no choices left.
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    def test_max_autotune_addmm_tma_dynamic_outer_dim(self):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).to(GPU_TYPE)
        b = torch.randn(K, N).to(torch.float16).to(GPU_TYPE)
        x = torch.randn(N).to(torch.float16).to(GPU_TYPE)

        # TMA requires 16-byte alignment: here we repeat the dims
        # by the factor of 8, as float16 is 2-byte. All dims are
        # repeated due to the possible transpositions below.
        x = x.repeat(8)
        a = a.repeat(8, 8)
        b = b.repeat(8, 8)

        torch._dynamo.mark_dynamic(a, 0)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "triton.native_matmul": False,
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual = torch.compile(addmm)(x, a, b)
            c_expected = addmm(x, a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @fresh_cache()
    @skipIfXpu(msg="XPU doesn't support sm carveout")
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support sm carveout")
    @unittest.skipIf(IS_WINDOWS, "Windows doesn't support persistent TMA")
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @unittest.skipIf(
        has_datacenter_blackwell_tma_device(), "B200 doesn't support sm carveout"
    )
    @parametrize("carveout", (None, 0, 27))
    @parametrize("op", ("mm", "scaled_mm"))
    def test_honor_sm_carveout_with_triton_tma(self, carveout, op: str):
        def mm_func(a, b):
            return torch.mm(a, b)

        def scaled_mm(
            a,
            b,
            scale_a,
            scale_b,
        ):
            return torch._scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)

        # Create large matrices to ensure we use all possible sms
        size = 2560
        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.bfloat16)
        b = (
            torch.randn(size, size, device=GPU_TYPE, dtype=torch.bfloat16)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.tensor(1, dtype=torch.float32, device=GPU_TYPE)
        scale_b = torch.tensor(1, dtype=torch.float32, device=GPU_TYPE)

        args = (
            (a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), scale_a, scale_b)
            if op == "scaled_mm"
            else (a, b)
        )
        func = scaled_mm if op == "scaled_mm" else mm_func

        # Set the specified carveout value
        torch._C._set_sm_carveout_experimental(carveout)
        if carveout is None:
            self.assertIsNone(torch._C._get_sm_carveout_experimental())
        else:
            self.assertEqual(torch._C._get_sm_carveout_experimental(), carveout)

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": True,
                "triton.native_matmul": False,
                "max_autotune_gemm_backends": "TRITON",
                "test_configs.autotune_choice_name_regex": "tma",
            }
        ):
            compiled_mm = torch.compile(func, mode="max-autotune-no-cudagraphs")
            compiled_mm(*args)  # Warm-up compilation

            with tempfile.NamedTemporaryFile() as f:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    # Run with the specified carveout
                    compiled_mm(*args)

                # Export trace and analyze results
                prof.export_chrome_trace(f.name)

                # Extract grid sizes from the trace events for TMA kernels
                kernel_name = "triton_tem_fused"
                with open(f.name) as file:
                    kernel_events = [
                        {
                            "grid": evt.get("args", {}).get("grid", []),
                            "grid_size": math.prod(evt.get("args", {}).get("grid", [])),
                        }
                        for evt in json.load(file)["traceEvents"]
                        if evt.get("cat", "") == "kernel"
                        and kernel_name in evt.get("name", "").lower()
                    ]

                # We should have exactly 1 kernel event for this run
                self.assertEqual(
                    len(kernel_events),
                    1,
                    f"Expected exactly 1 kernel event, but got {len(kernel_events)}",
                )

                # Check that grid size matches expected values based on carveout
                expected_grid_size = None
                max_grid_size = torch.cuda.get_device_properties(
                    "cuda"
                ).multi_processor_count
                careveout = 0 if carveout is None else carveout
                expected_grid_size = max_grid_size - careveout

                self.assertEqual(
                    kernel_events[0]["grid_size"],
                    expected_grid_size,
                    f"Grid size {kernel_events[0]['grid_size']} doesn't match {expected_grid_size} for carveout={carveout}",
                )

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_zero_size_input(self, dynamic):
        """
        Make sure autotuning addmm with zero-size input works without crashes.
        """

        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(0, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        with config.patch({"max_autotune": True}):
            torch.compile(addmm, dynamic=dynamic)(x, a, b)

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_autotune_conv1x1(self, search_space):
        # Assuming input has 3 channels and we want to produce 16 channels as output
        conv1x1 = (
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1)
            .to(memory_format=torch.channels_last)
            .to(GPU_TYPE)
        )

        # Example input tensor: batch size = 4, channels = 3, height = 32, width = 32
        # The memory format is set to `channels_last`
        input_tensor = (
            torch.randn(4, 3, 32, 32)
            .contiguous(memory_format=torch.channels_last)
            .to(GPU_TYPE)
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "max_autotune_gemm_search_space": search_space,
            }
        ):

            @torch.compile()
            def foo(mod, x):
                return mod(x)

            with torch.no_grad():
                out, code = run_and_get_code(foo, conv1x1, input_tensor)

            FileCheck().check_not("extern_kernels.convolution").run(code[0])
            self.assertEqual(conv1x1(input_tensor), out, atol=1e-2, rtol=0)

    @fresh_cache()
    @config.patch(max_autotune=True, max_fusion_size=2)
    def test_jit_fusion_matches_aot_fusion(self):
        # In this example, AOTInductor's JIT-compile will fuse(buf1, buf2) due
        # to proximity, we want to make sure AOT-compile pass does the same.
        # AOT could do fuse(buf2, buf4) instead if buf3 was pushed to the end
        # of the V.graph.buffers list because fuse(buf2, buf4) would have a
        # better proximity score than fuse(buf1, buf2). This scenario is possible
        # since finalizing MultiTemplateBuffers needs to replace buffers.
        def fn(x, number):
            buf0 = x + x
            buf1 = number.item()
            buf2 = x * x
            buf3 = x @ x  # MultiTemplateBuffer
            buf4 = x**2
            return buf0, buf1, buf2, buf3, buf4

        inputs = (
            torch.rand([256, 256], device=GPU_TYPE),
            torch.tensor(3, device=GPU_TYPE),
        )
        torch._export.aot_compile(fn, args=inputs)

    def test_cat_addmm(self):
        def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
            return torch.cat(
                [
                    torch.addmm(a, b, c),
                    torch.addmm(b, c, a),
                ],
                1,
            )

        args = [
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
            torch.randn(4, 4, device=GPU_TYPE),
        ]
        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            expected = fn(*args)
            actual = torch.compile(fn)(*args)
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    @config.patch(
        benchmark_kernel=True,
        fallback_random=True,
        max_autotune_gemm=True,
    )
    @parametrize("device", ("cpu", GPU_TYPE))
    def test_matmul_dropout(self, device):
        def fwd(a, b):
            x = a @ b
            x = torch.nn.functional.dropout(x, 0.1)
            return x

        def fn(a, b):
            x = fwd(a, b).sum()
            x.backward()
            return a.grad

        N = 128
        a = torch.randn(N, N, device=device, requires_grad=True)
        b = torch.randn(N, N, device=device)

        opt_fn = torch.compile(fn)
        reset_rng_state()
        ref = fn(a, b)
        reset_rng_state()
        act = opt_fn(a, b)

        if N <= 8:
            print(f"ref\n{ref}\nact\n{act}")
        torch.testing.assert_close(ref, act, atol=1e-1, rtol=1e-1)

    @config.patch(
        max_autotune_gemm=True,
    )
    @unittest.skipIf(
        getattr(torch, GPU_TYPE).device_count() < 2,
        "Need at least 2 devices for this test",
    )
    def test_autotune_device_guard(self):
        x = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")
        y = torch.randn(1024, 1024, device=f"{GPU_TYPE}:1")

        def f(x, y):
            return x @ y

        with fresh_cache():
            act = torch.compile(f)(x, y)
        ref = f(x, y)
        self.assertTrue(torch.allclose(act, ref, atol=4 * 1e-3, rtol=4 * 1e-3))

    @config.patch(max_autotune=True)
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    @parametrize("kernel_size", (1, 3))
    def test_empty_conv_input(self, search_space, kernel_size):
        x = torch.randn(0, 256, 14, 14, device=GPU_TYPE)
        weight = torch.randn(256, 256, kernel_size, kernel_size, device=GPU_TYPE)

        def f(x, weight):
            return torch.convolution(
                x,
                weight,
                bias=None,
                stride=[1, 1],
                padding=[0, 0],
                dilation=[1, 1],
                transposed=False,
                output_padding=[0, 0],
                groups=1,
            )

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            opt_f = torch.compile(f)
            ref = f(x, weight)
            act = opt_f(x, weight)
            self.assertTrue(torch.allclose(ref, act, atol=4 * 1e-3, rtol=4 * 1e-3))

    @config.patch(max_autotune_gemm_backends="TRITON")
    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_baddmm(self, search_space):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.randn(64, 64, 192, dtype=torch.float16)
                )
                self.bias = torch.nn.Parameter(
                    torch.randn(64, 1, 192, dtype=torch.float16)
                )

            def forward(self, x):
                return torch.ops.aten.baddbmm.default(self.bias, x, self.weight)

        x = torch.randn(
            64, 2048, 64, dtype=torch.float16, requires_grad=False, device=GPU_TYPE
        )
        mod = M().to(GPU_TYPE)

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            m_c = torch.compile(mode="max-autotune")(mod)
            out, code = run_and_get_code(m_c, x)
            self.assertEqual(out, mod(x), atol=2e-3, rtol=2e-3)

            if not config.triton.native_matmul:
                FileCheck().check("triton_tem_fused_baddbmm").run(code[0])

    @config.patch(max_autotune=True)
    def test_conv1x1_with_free_symbols(self):
        """
        Make sure there is no exception due to free symbols.
        """
        conv = nn.Conv2d(
            3, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        ).to(device=GPU_TYPE)

        @torch.compile
        def f(x, y, z):
            h = y.nonzero().size(0)
            w = z.nonzero().size(0)
            x = x[:, :, :h, :w]
            x = conv(x)
            return x

        x = torch.randn(4, 3, 224, 224).to(
            memory_format=torch.channels_last, device=GPU_TYPE
        )
        for _ in range(2):
            y = torch.randint(0, 10, (224,)).to(device=GPU_TYPE)
            z = torch.randint(0, 10, (224,)).to(device=GPU_TYPE)
            f(x, y, z)

    def _test_cat_max_autotune_impl(self, using_triton_mm):
        def f(x, y):
            y = torch.cos(y)
            x = torch.mm(x, x)
            return torch.cat([x, y])

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        inps = [
            torch.randn(32, 32, device=GPU_TYPE),
            torch.randn(32, 32, device=GPU_TYPE),
        ]
        _, code = run_and_get_code(f_c, inps[0], inps[1])
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)

        # mm kernel, and cos kernel
        count = 2 if (using_triton_mm or config.triton.native_matmul) else 1
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), count, exactly=True
        ).run(code[0])

        def f(x, y):
            y = torch.cos(y)
            x = torch.mm(x, x)
            out = torch.cat([x, y])
            return out, x + 1

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        _, code = run_and_get_code(f_c, inps[0], inps[1])
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), 2, exactly=True
        ).run(code[0])

        def f(x, y):
            y = torch.cos(y)
            x = torch.mm(x, x)
            return torch.cat([x, y]), torch.cat([y, x])

        f_c = torch.compile(mode="max-autotune-no-cudagraphs")(f)
        self.assertEqual(f_c(*inps), f(*inps), atol=0.03, rtol=0.25)

    @config.patch("trace.enabled", True)
    @config.patch({"test_configs.force_extern_kernel_in_multi_template": True})
    @config.patch("triton.native_matmul", False)
    def test_mutation_rename(self):
        torch._logging.set_logs(ir_post_fusion=True)

        def f(x, y, z, other):
            mul = x * y
            diag = torch.diagonal(mul)
            diag.copy_(other)
            x = torch.mm(mul, z)
            y = torch.diagonal(x).add_(torch.tensor(1, device=GPU_TYPE))
            return y

        t = functools.partial(torch.randn, device=GPU_TYPE)
        inps = (t(3, 3), t(3, 3), t(3, 3), t(3))
        fn = torch.compile(f, mode="max-autotune-no-cudagraphs")

        (
            (
                pre_fusion_tream,
                post_fusion_stream,
            ),
            ctx,
        ) = multiple_logs_to_string(
            "torch._inductor.debug", "ir_pre_fusion", "ir_post_fusion"
        )

        with config.patch({"trace.debug_dir": tempfile.mkdtemp()}):
            with (
                self.assertLogs(
                    logging.getLogger("torch._inductor.debug"), level=logging.INFO
                ) as cm,
                ctx(),
            ):
                out = fn(*inps)

        self.assertEqual(f(*inps), out)

        pre_fusion_stream = cm.output[0]
        post_fusion_stream = cm.output[1]

        # before and after finalizing multi template buffer, deps should have the same normalization
        # wrt writes
        FileCheck().check("MultiTemplateBuffer").check("unmet").check_same("buf1").run(
            pre_fusion_stream
        )
        FileCheck().check("ExternKernelSchedulerNode").check("unmet").check_same(
            "buf1"
        ).run(post_fusion_stream)

        torch._logging.set_logs()

    @config.patch({"test_configs.force_extern_kernel_in_multi_template": True})
    def test_cat_max_autotune_extern(self):
        self._test_cat_max_autotune_impl(using_triton_mm=False)

    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "benchmark_epilogue_fusion": False,
        }
    )
    def test_cat_max_autotune_triton(self):
        self._test_cat_max_autotune_impl(using_triton_mm=True)

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_conv_cat(self, search_space):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )

            def forward(self, x):
                x = self.conv(x)
                return torch.cat((x, x + 1))

        with config.patch({"max_autotune_gemm_search_space": search_space}):
            with torch.no_grad():
                m = ToyModel().to(device=GPU_TYPE)
                input_tensor = torch.randn(32, 3, 64, 64).to(device=GPU_TYPE)

                # convolution is not currently plannable
                m = torch.compile(m, mode="max-autotune-no-cudagraphs")
                out, code = run_and_get_code(m, input_tensor)
                self.assertEqual(out, m(input_tensor))

                if not TEST_WITH_ROCM:
                    FileCheck().check("def triton_poi_fused_add_cat_").run(code[0])

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    def test_conv3d(self, search_space):
        fn = torch.nn.functional.conv3d
        image = torch.randn([1, 3, 8, 16, 32])
        filt = torch.randn([3, 3, 7, 7, 7])

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_search_space": search_space}
        ):
            expected = fn(image, filt)
            actual = torch.compile(fn)(image, filt)
            torch.testing.assert_close(actual, expected, atol=6e-5, rtol=0.001)

    @config.patch(
        max_autotune=True, max_autotune_conv_backends="", layout_optimization=False
    )
    def test_conv_backend(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 3, 1, 1),
        ).to(GPU_TYPE)
        inp = torch.randn([2, 3, 16, 16]).to(GPU_TYPE)

        with self.assertRaises(BackendCompilerFailed) as context:
            torch.compile(m)(inp)

        self.assertIn("NoValidChoicesError", str(context.exception))

    def test_non_contiguous_input_mm(self):
        """
        Make sure the triton template can work with non-contiguous inputs without crash.
        Check https://github.com/pytorch/pytorch/issues/125437 for more details.
        """
        x = rand_strided(
            (50257, 2048), (1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided((2048, 768), (768, 1), dtype=torch.bfloat16, device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x, y):
            return x @ y

        ref = x @ y
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    def test_non_contiguous_input_addmm(self):
        b = torch.randn((768), dtype=torch.bfloat16, device=GPU_TYPE)
        x = rand_strided(
            (50257, 2048), (1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided((2048, 768), (768, 1), dtype=torch.bfloat16, device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x, y):
            return torch.addmm(b, x, y)

        ref = torch.addmm(b, x, y)
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    def test_non_contiguous_input_bmm(self):
        x = rand_strided(
            (1, 50257, 2048), (0, 1, 50304), dtype=torch.bfloat16, device=GPU_TYPE
        )
        y = rand_strided(
            (1, 2048, 768), (0, 768, 1), dtype=torch.bfloat16, device=GPU_TYPE
        )

        @torch.compile(mode="max-autotune")
        def f(x, y):
            return torch.bmm(x, y)

        ref = torch.bmm(x, y)
        act = f(x, y)
        torch.testing.assert_close(act, ref, atol=2e-2, rtol=1e-2)

    @unittest.skipIf(
        config.triton.native_matmul,
        "native matmul and Triton template both have accuracy fail (2.2%)",
    )
    def test_non_contiguous_input_mm_plus_mm(self):
        x1 = rand_strided((50257, 2048), (1, 50304), device=GPU_TYPE)
        y1 = rand_strided((2048, 768), (768, 1), device=GPU_TYPE)

        x2 = rand_strided((50257, 2048), (1, 50304), device=GPU_TYPE)
        y2 = rand_strided((2048, 768), (768, 1), device=GPU_TYPE)

        @torch.compile(mode="max-autotune")
        def f(x1, y1, x2, y2):
            return x1 @ y1 + x2 @ y2

        ref = x1 @ y1 + x2 @ y2
        act = f(x1, y1, x2, y2)
        torch.testing.assert_close(act, ref, atol=1e-1, rtol=1e-2)

    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="",
    )
    @unittest.skipIf(
        config.triton.native_matmul, "native matmul generates when size >=2"
    )
    def test_no_valid_choices(self):
        a = torch.zeros([2, 2], device=GPU_TYPE)
        b = torch.zeros([2, 2], device=GPU_TYPE)
        with self.assertRaises(BackendCompilerFailed) as context:
            torch.compile(lambda a, b: a.matmul(b))(a, b)
        self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        config.triton.native_matmul, "Only test when template is being called"
    )
    @parametrize("multi_template", (True, False))
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_inf_timing(self, multi_template):
        lookup = AlgorithmSelectorCache.lookup

        def mock_lookup(self, *args, **kwargs):
            timings = lookup(self, *args, **kwargs)
            return {choice: float("inf") for choice in timings}

        a = torch.zeros([16, 16], device=GPU_TYPE)
        b = torch.zeros([16, 16], device=GPU_TYPE)
        with (
            patch.object(AlgorithmSelectorCache, "lookup", mock_lookup),
            config.patch(benchmark_epilogue_fusion=multi_template),
        ):
            with self.assertRaises(BackendCompilerFailed) as context:
                torch.compile(lambda a, b: a.matmul(b))(a, b)
            self.assertIn("NoValidChoicesError", str(context.exception))

    @config.patch(force_shape_pad=True, max_autotune=True)
    def test_linear_and_cel(self):
        """
        Similate a GPU without enough SMs. Make sure max-autotune still
        works even when the MultiTritonTemplate encapsulates just extern
        kernels.
        """

        def mock_is_big_gpu(*args, **kwargs):
            return False

        B, T, C, V = 32, 1024, 768, 50257

        linear = nn.Linear(C, V).bfloat16().to(device=GPU_TYPE)
        ce = torch.nn.CrossEntropyLoss()

        def f(x, y):
            x.grad = None
            linear.weight.grad = None
            linear.bias.grad = None

            loss = ce(linear(x), y)
            loss.backward()
            return loss

        x = torch.randn(B * T, C, requires_grad=True).to(GPU_TYPE).bfloat16()
        x.retain_grad()
        y = torch.randint(0, V, (B * T,)).to(GPU_TYPE)

        import torch._inductor.utils as inductor_utils

        with unittest.mock.patch.object(inductor_utils, "is_big_gpu", mock_is_big_gpu):
            opt_f = torch.compile(f)

            expect = (f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
            actual = (opt_f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
            if not same(expect, actual, tol=1e-2):
                raise AssertionError(f"ref:\n{expect}\nact:\n{actual}")

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @parametrize("dynamic", (True, False))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("sizes", ((32, 32, 32768), (64, 128, 200000), (64, 64, 177147)))
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        comprehensive_padding=False,
        shape_padding=False,
    )
    def test_max_autotune_decompose_k(self, sizes, dtype, dynamic):
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):
            fp16_red_setting = (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            )
            bf16_red_setting = (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            )
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

            M, N, K = sizes

            atol = 1e-4
            rtol = 1e-4
            # K can be huge huge, this is why the data distribution is set to iid N(0, K ** 0.5),
            # which makes the result of reductions distributed as N(0, 1).
            a, b = self._make_matrices(
                M,
                K,
                N,
                dtype=dtype,
                device=GPU_TYPE,
                requires_grad=True,
            )

            possible_splits = range(2, min(K // M, K // N) + 1)

            divisors = {split for split in possible_splits if K % split == 0}

            def check_divisors(code):
                for kernel in code:
                    if "decompose_k" in kernel:
                        divisor_found = False
                        for divisor in divisors:
                            if f"{divisor}_split" in kernel:
                                divisor_found = True
                                break

                        self.assertTrue(
                            divisor_found,
                            f"Could not find a split in {divisors} in {kernel}",
                        )

            compiled_func = torch.compile(lambda a, b: a @ b, dynamic=dynamic)
            # We assume with the large k dim relative to m, n, decompose_k will be most performant
            out, code = run_and_get_code(compiled_func, a, b)

            if dynamic:
                FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                    "decompose_k"
                ).run(code[0])
            else:
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_0.run"
                ).check("decompose_k").run(code[0])
                check_divisors(code)
                torch.testing.assert_close(out, a @ b, atol=atol, rtol=rtol)

            # Test adding epilogue also equivalent to eager
            compiled_func = torch.compile(lambda a, b: (a @ b).relu(), dynamic=dynamic)
            out, code = run_and_get_code(compiled_func, a, b)
            if dynamic:
                FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                    "decompose_k"
                ).run(code[0])
            else:
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_mm_0.run"
                ).check("decompose_k").run(code[0])
                check_divisors(code)
                torch.testing.assert_close(
                    compiled_func(a, b), (a @ b).relu(), atol=atol, rtol=rtol
                )

            # Test adding reinterpret view before subgraph
            a = a.transpose(0, 1)
            compiled_func = torch.compile(
                lambda a, b: (a.transpose(0, 1) @ b).relu(), dynamic=dynamic
            )
            out, code = run_and_get_code(compiled_func, a, b)

            if dynamic:
                FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                    "decompose_k"
                ).run(code[0])
            else:
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_.*_0.run"
                ).check("decompose_k").run(code[0])
                check_divisors(code)
                torch.testing.assert_close(
                    compiled_func(a, b),
                    (a.transpose(0, 1) @ b).relu(),
                    atol=atol,
                    rtol=rtol,
                )

            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                fp16_red_setting
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                bf16_red_setting
            )

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_dynamic_input(self):
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):

            def f(a, b):
                a_in = torch.stack((a, a), dim=0)
                return (a_in @ b).relu()

            a, b = self._make_matrices(
                M=32,
                K=32768,
                N=64,
                dtype=torch.bfloat16,
                device=GPU_TYPE,
                requires_grad=True,
            )

            torch._dynamo.reset()
            torch._dynamo.maybe_mark_dynamic(a, 0)
            compiled_func = torch.compile(f)

            with mock.patch(
                "torch._inductor.kernel.mm.use_decompose_k_choice"
            ) as decomp_mock:
                decomp_mock.side_effect = (
                    lambda *args, **kwargs: kwargs.get("threshold_multiple", 1) == 1
                )

                out, code = run_and_get_code(compiled_func, a, b)
                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_.*.run"
                ).check("decompose_k").check_regex(r"s[0-9]+ = s[0-9]+").check_regex(
                    r"2\*s[0-9]+"
                ).check_regex("s[0-9]+ = 32").run(code[0])
                torch.testing.assert_close(
                    out,
                    f(a, b),
                    atol=1e-4,
                    rtol=1e-4,
                )

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_dynamic_input_bwd(self):
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):

            def f(a, b):
                # 256 * s0
                a_in = torch.cat([a for _ in range(256)], dim=0)
                return (a_in @ b).relu().sum()

            a, b = self._make_matrices(
                M=8,
                K=64,
                N=32768,
                dtype=torch.bfloat16,
                device=GPU_TYPE,
                requires_grad=True,
            )

            torch._dynamo.reset()
            torch._dynamo.maybe_mark_dynamic(a, 0)
            compiled_func = torch.compile(f)
            res = compiled_func(a, b)
            res.backward()

            with mock.patch(
                "torch._inductor.kernel.mm.use_decompose_k_choice"
            ) as decomp_mock:
                decomp_mock.side_effect = (
                    lambda *args, **kwargs: kwargs.get("threshold_multiple", 1) == 1
                )

                out, code = run_and_get_code(compiled_func, a, b)
                out.backward()

                FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                    "triton_.*_fused_0.run"
                ).check("decompose_k").check_regex(r"s[0-9]+ = s[0-9]+").check_regex(
                    r"256\*s[0-9]+"
                ).check_regex("s[0-9]+ = 8").run(
                    # code[1] in this case given backwards
                    code[1]
                )

    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_output_stride(self):
        # UT specific change to force testing decompose K feature on ROCm until
        # enabled by default, same strategy as #169948
        with config.patch(_DECOMPOSE_K_PATCH_ROCM):

            def f(a, b):
                a = a.transpose(0, 1)
                return a @ b

            a = torch.randn((32768, 256), device=GPU_TYPE, dtype=torch.bfloat16)
            b = torch.randn((32768, 1152), device=GPU_TYPE, dtype=torch.bfloat16)

            b = b[:, :1096]

            # Force only decomposeK choice
            with (
                override_template_heuristics(
                    device_type=GPU_TYPE,
                    template_op_pairs=[
                        (torch._inductor.kernel.mm.mm_template.name, "mm")
                    ],
                ),
                mock.patch(
                    "torch._inductor.kernel.mm.use_decompose_k_choice"
                ) as decompose_mock,
            ):
                decompose_mock.return_value = True
                compiled_f = torch.compile(f)
                out, code = run_and_get_code(compiled_f, a, b)

                # Output stride equal to original gm output stride
                # If output stride is not correctly checked, this will be (1152, 1) which can cause nans
                self.assertEqual(out.stride(), (1096, 1))

                FileCheck().check_not("extern_kernels.bmm_dtype").check(
                    "decompose_k"
                ).check(
                    f" empty_strided_{GPU_TYPE}((256, 1096), (1096, 1), torch.bfloat16)"
                ).run(code[0])

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    @parametrize("sizes", ((64, 128, 256), (128, 256, 512), (256, 512, 1024)))
    @config.patch(
        max_autotune=True,
    )
    def test_max_autotune_contiguous_transform_mm(self, sizes, dtype):
        """
        Test the contiguous subgraph transform with A * transpose(B) pattern.
        This transform makes the second matrix contiguous before the matmul.
        """
        M, N, K = sizes

        def mm_transpose(a, b):
            return a @ b.transpose(0, 1)

        a = torch.randn(M, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)

        # Compute fp64 baseline
        a_fp64 = a.to(torch.float64)
        b_fp64 = b.to(torch.float64)
        expected_fp64 = mm_transpose(a_fp64, b_fp64)

        # Force only contiguous choice to test the transform
        with (
            mock.patch(
                "torch._inductor.template_heuristics.contiguous_mm.use_contiguous"
            ) as contiguous_mock,
        ):
            contiguous_mock.return_value = True

            compiled_func = torch.compile(mm_transpose)
            out, code = run_and_get_code(compiled_func, a, b)

            # Verify correctness against fp64 baseline
            torch.testing.assert_close(
                out, expected_fp64.to(dtype), atol=1e-2, rtol=1e-2
            )

            # Check that contiguous transform was used
            FileCheck().check("contiguous_mm").run(code[0])

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    @parametrize("sizes", ((64, 128, 256), (128, 256, 512), (256, 512, 1024)))
    @config.patch(
        max_autotune=True,
    )
    def test_max_autotune_contiguous_transform_addmm(self, sizes, dtype):
        """
        Test the contiguous subgraph transform for addmm with non-contiguous second matrix.
        """
        M, N, K = sizes

        def addmm_transpose(inp, a, b):
            return torch.addmm(inp, a, b.transpose(0, 1))

        inp = torch.randn(M, N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        a = torch.randn(M, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(N, K, dtype=dtype, device=GPU_TYPE, requires_grad=True)

        # Compute fp64 baseline
        inp_fp64 = inp.to(torch.float64)
        a_fp64 = a.to(torch.float64)
        b_fp64 = b.to(torch.float64)
        expected_fp64 = addmm_transpose(inp_fp64, a_fp64, b_fp64)

        # Force contiguous choice to test the transform
        with (
            mock.patch(
                "torch._inductor.template_heuristics.contiguous_mm.use_contiguous"
            ) as contiguous_mock,
        ):
            contiguous_mock.return_value = True

            compiled_func = torch.compile(addmm_transpose)
            out, code = run_and_get_code(compiled_func, inp, a, b)

            # Verify correctness against fp64 baseline
            torch.testing.assert_close(
                out, expected_fp64.to(dtype), atol=1e-2, rtol=1e-2
            )

            # Check that contiguous transform was used
            FileCheck().check("contiguous_addmm").run(code[0])

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @parametrize("dynamic", (False, True))
    def test_max_autotune_contiguous_transform_non_contiguous_second_matrix(
        self, dynamic
    ):
        """
        Test that contiguous transform is only applied when the second matrix is non-contiguous.
        """
        M, N, K = 64, 128, 64

        def mm(a, b):
            return a @ b

        a = torch.randn(M, K, dtype=torch.float32, device=GPU_TYPE)
        b_contiguous = torch.randn(K, N, dtype=torch.float32, device=GPU_TYPE)
        b_non_contiguous = torch.randn(
            N, K, dtype=torch.float32, device=GPU_TYPE
        ).transpose(0, 1)

        # Compute fp64 baselines without max_autotune (since fp64 doesn't work with max_autotune=True)
        a_fp64 = a.to(torch.float64)
        b_contiguous_fp64 = b_contiguous.to(torch.float64)
        b_non_contiguous_fp64 = b_non_contiguous.to(torch.float64)

        expected1_fp64 = mm(a_fp64, b_contiguous_fp64)
        expected2_fp64 = mm(a_fp64, b_non_contiguous_fp64)

        with config.patch(
            max_autotune=True,
        ):
            # Test with contiguous second matrix - should not use contiguous transform
            compiled_func_contiguous = torch.compile(mm, dynamic=dynamic)
            out1, code1 = run_and_get_code(compiled_func_contiguous, a, b_contiguous)

            # Should not contain contiguous transform
            try:
                FileCheck().check("contiguous_mm").run(code1[0])
                self.fail(
                    "Contiguous transform should not be used for contiguous matrices"
                )
            except RuntimeError:
                pass  # Expected - contiguous transform should not be used

            # Test with non-contiguous second matrix - should use contiguous transform
            with (
                mock.patch(
                    "torch._inductor.template_heuristics.contiguous_mm.use_contiguous"
                ) as contiguous_mock,
            ):
                contiguous_mock.return_value = True

                compiled_func_non_contiguous = torch.compile(mm, dynamic=dynamic)
                out2, code2 = run_and_get_code(
                    compiled_func_non_contiguous, a, b_non_contiguous
                )

                # Should contain contiguous transform
                FileCheck().check("contiguous_mm").run(code2[0])

        # Verify correctness against fp64 baselines
        torch.testing.assert_close(
            out1, expected1_fp64.to(torch.float32), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            out2, expected2_fp64.to(torch.float32), atol=1e-2, rtol=1e-2
        )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_contiguous_transform_with_epilogue(self):
        """
        Test contiguous transform with epilogue operations like relu.
        """
        M, N, K = 128, 256, 512

        def mm_transpose_relu(a, b):
            return (a @ b.transpose(0, 1)).relu()

        a = torch.randn(M, K, dtype=torch.float32, device=GPU_TYPE)
        b = torch.randn(N, K, dtype=torch.float32, device=GPU_TYPE)

        # Compute fp64 baseline
        a_fp64 = a.to(torch.float64)
        b_fp64 = b.to(torch.float64)
        expected_fp64 = mm_transpose_relu(a_fp64, b_fp64)

        # Force contiguous transform
        with (
            mock.patch(
                "torch._inductor.template_heuristics.contiguous_mm.use_contiguous"
            ) as contiguous_mock,
        ):
            contiguous_mock.return_value = True

            compiled_func = torch.compile(mm_transpose_relu)
            out, code = run_and_get_code(compiled_func, a, b)

            # Verify correctness against fp64 baseline
            torch.testing.assert_close(
                out, expected_fp64.to(torch.float32), atol=1e-2, rtol=1e-2
            )

            # Check that contiguous transform was used
            FileCheck().check("contiguous_mm").run(code[0])

    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_override_template_heuristics_with_custom_class(self):
        """
        Test that override_template_heuristics works with a custom heuristic class.
        Verifies that get_template_heuristic returns an instance of our custom class
        and that get_template_configs yields the expected configs.
        """
        from torch._inductor.kernel.mm import MMKernelInputs
        from torch._inductor.template_heuristics.registry import (
            get_registered_heuristic_class,
            get_template_heuristic,
        )

        template_uid = torch._inductor.kernel.mm.mm_template.uid

        # Get the base heuristic class that would normally be used for mm_template
        base_heuristic_class = get_registered_heuristic_class(
            template_uid, GPU_TYPE, "mm"
        )
        self.assertIsNotNone(base_heuristic_class)

        # Create a dummy graph for the GraphLowering context
        gm = make_fx(lambda: torch.zeros(2, 3))()
        graph = GraphLowering(gm)

        # The graph handler is needed to create IR nodes and use V.graph.sizevars
        with V.set_graph_handler(graph), config.patch({"max_autotune": True}):
            # Create IR Buffer nodes for testing (not actual tensors)
            mat1 = Buffer(
                name="mat1",
                layout=FixedLayout(
                    torch.device(GPU_TYPE),
                    dtype=torch.float32,
                    size=(64, 64),
                ),
            )
            mat2 = Buffer(
                name="mat2",
                layout=FixedLayout(
                    torch.device(GPU_TYPE),
                    dtype=torch.float32,
                    size=(64, 64),
                ),
            )
            kernel_inputs = MMKernelInputs([mat1, mat2])

            # Get the first config from the original heuristic to use as our expected config
            base_heuristic = base_heuristic_class()
            expected_config = next(
                iter(base_heuristic.get_template_configs(kernel_inputs, "mm"))
            )

            # Create a custom heuristic class that only yields the single expected config
            class CustomMMHeuristic(base_heuristic_class):
                def get_template_configs(self, kernel_inputs, op_name):
                    yield expected_config

            with override_template_heuristics(
                device_type=GPU_TYPE,
                template_op_pairs=[(template_uid, "mm")],
                override_heuristic_class=CustomMMHeuristic,
            ):
                # Get the heuristic and verify it's our custom class
                heuristic = get_template_heuristic(template_uid, GPU_TYPE, "mm")
                self.assertIsInstance(heuristic, CustomMMHeuristic)
                self.assertIsInstance(heuristic, base_heuristic_class)

                # Verify get_template_configs yields only the expected config
                configs = list(heuristic.get_template_configs(kernel_inputs, "mm"))
                self.assertEqual(len(configs), 1)
                self.assertEqual(configs[0], expected_config)

    @unittest.skipIf(config.cpp_wrapper, "out_dtype override not supported for AOTI")
    def test_bmm_out_dtype(self):
        def f(a, b):
            return torch.bmm(a, b, out_dtype=torch.float32)

        a = torch.randn(2, 3, 4, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2, 4, 5, device=GPU_TYPE, dtype=torch.float16)
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
        ):
            compiled_f = torch.compile(f)
            with self.assertRaisesRegex(
                torch._inductor.exc.InductorError,
                r"LoweringException: NoValidChoicesError: No choices to select",
            ):
                out, code = run_and_get_code(compiled_f, a, b)

        compiled_f = torch.compile(f)
        out, code = run_and_get_code(compiled_f, a, b)
        FileCheck().check("extern_kernels.bmm_dtype").run(code[0])

    def test_triton_template_generated_code_cache_key(self):
        generate_and_load_args = len(
            inspect.signature(
                torch._inductor.select_algorithm.TritonTemplate.generate_and_load
            ).parameters
        )
        make_key_args = len(
            inspect.signature(
                torch._inductor.select_algorithm.GeneratedCodeCache.make_key
            ).parameters
        )

        # Make sure all args of generate_and_load_args are passed to make_key_args (Except generate_with_caching)
        # update this function each time new arg added to generate_and_load and make sure arg is added to make_key
        self.assertEqual(generate_and_load_args - 1, make_key_args)
        self.assertEqual(generate_and_load_args, 20)

    @fresh_cache()
    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_cache_strategy(self):
        def func_test1(x, y, z, m):
            a = torch.matmul(x, y)
            b = torch.matmul(z, m)
            return a, b

        a = torch.rand(10, 22, device=GPU_TYPE)
        b = torch.rand(22, 30, device=GPU_TYPE)
        # Test that the testing strategy works by overriding input_dependent_preserved_state and simulate a cache hit.
        with unittest.mock.patch(
            "torch._inductor.select_algorithm.TritonTemplateKernel.input_dependent_preserved_state",
            new=(lambda self: "same always"),
        ):
            with self.assertRaisesRegex(
                torch._inductor.exc.InductorError,
                r".*Generated code cache results in wrong output.*",
            ):
                torch.compile(func_test1, dynamic=False)(a, b, a, b)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_caching(self):
        def reset_counters():
            torch._dynamo.utils.counters.clear()

        def hits():
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_hit"
            ]

        def misses():
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_miss"
            ]

        # remove white space from x.
        def remove_white_space(x: str) -> str:
            return re.sub(r"\s+", "", x)

        def get_cache_key_and_events() -> tuple[str, str]:
            cache = TritonTemplate.all_templates["mm"]._generated_code_cache._cache
            cache_key = next(iter(cache))
            events = str(cache[cache_key].events)
            return cache_key, events

        def func_test1(x, y, z, m):
            a = torch.matmul(x, y)
            b = torch.matmul(z, m)
            return a, b

        a = torch.rand(10, 22, device=GPU_TYPE)
        b = torch.rand(22, 30, device=GPU_TYPE)

        # Valid cache hit.
        with fresh_cache():
            reset_counters()
            compile_results = torch.compile(func_test1, dynamic=False)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)
            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

            cache_key, events = get_cache_key_and_events()

            if not TEST_WITH_ROCM:
                expected = """{
                        'input_nodes':[
                            "[[10,22],[22,1],torch.float32,device(type='cuda',index=0),0]",
                            "[[22,30],[30,1],torch.float32,device(type='cuda',index=0),0]"],
                        'num_stages':1,'num_warps':2,'prefix_args':0,'suffix_args':0,'call_sizes':[10,30],
                        'layout':"[[10,30],[30,1],torch.float32,device(type='cuda',index=0),0]",
                        'num_consumer_groups':0,'num_buffers_warp_spec':0,'epilogue_fn_hash':'identity','tma_store':False,
                        'transpose_discontiguous_tensor_descriptors_override':None,
                        'kwargs':{'EVEN_K':False,'USE_FAST_ACCUM':False,'ACC_TYPE':'tl.float32',
                        'BLOCK_M':16,'BLOCK_N':32,'BLOCK_K':16,'GROUP_M':8,'ALLOW_TF32':True},
                        'hint_override':None,'triton_meta':None}"""

                expected = expected.replace("cuda", GPU_TYPE)
                self.assertExpectedInline(
                    remove_white_space(cache_key),
                    remove_white_space(expected),
                )

                self.assertEqual(
                    remove_white_space(events),
                    remove_white_space("""[('def_kernel', ['A', 'B'], {})]"""),
                )

        # Test symbolic shapes with different symbols. Will cache miss due to different symbols in inputs.
        with fresh_cache():
            a = torch.rand(10, 22, device=GPU_TYPE)
            b = torch.rand(22, 30, device=GPU_TYPE)

            c = torch.rand(9, 21, device=GPU_TYPE)
            d = torch.rand(21, 30, device=GPU_TYPE)
            reset_counters()
            compiled_results = torch.compile(func_test1, dynamic=True)(a, b, c, d)
            eager_results = func_test1(a, b, c, d)

            self.assertEqual(compiled_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 0)
            self.assertEqual(misses(), 8)

            cache_key, events = get_cache_key_and_events()

            if not TEST_WITH_ROCM:
                expected = """{
                    'input_nodes':[
                        "[[s77,s27],[s27,1],torch.float32,device(type='cuda',index=0),0]",
                        "[[s27,s94],[s94,1],torch.float32,device(type='cuda',index=0),0]"],
                    'num_stages':1,'num_warps':2,'prefix_args':0,'suffix_args':0,'call_sizes':[s77,s94],
                    'layout':"[[s77,s94],[s94,1],torch.float32,device(type='cuda',index=0),0]",'num_consumer_groups':0,
                    'num_buffers_warp_spec':0,'epilogue_fn_hash':'identity','tma_store':False,
                    'transpose_discontiguous_tensor_descriptors_override':None,
                    'kwargs':{'EVEN_K':False,'USE_FAST_ACCUM':False,'ACC_TYPE':'tl.float32','BLOCK_M':16,'BLOCK_N':32,
                    'BLOCK_K':16,'GROUP_M':8,'ALLOW_TF32':True},'hint_override':None,'triton_meta':None}"""
                expected = expected.replace("cuda", GPU_TYPE)
                self.assertExpectedInline(
                    remove_white_space(cache_key),
                    remove_white_space(expected),
                )

                self.assertExpectedInline(
                    remove_white_space(events),
                    remove_white_space(
                        """[('def_kernel',['A','B'],{}),('size',['A',0],{}),('size',['B',1],{}),('size',['A',1],{})]"""
                    ),
                )
                self.assertExpectedInline(
                    remove_white_space(events),
                    remove_white_space(
                        """[
                            ('def_kernel', ['A', 'B'], {}),
                            ('size', ['A', 0], {}),
                            ('size', ['B', 1], {}),
                            ('size', ['A', 1], {})]
                        """
                    ),
                )

        # Test duck typing.
        with fresh_cache():
            reset_counters()

            compile_results = torch.compile(func_test1, dynamic=True)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

        # Test loop.
        def test_func2(x):
            for _ in range(10):
                x = torch.matmul(x, x)
            return x

        with fresh_cache():
            reset_counters()
            input = torch.rand(10, 10, device=GPU_TYPE)

            compile_results = torch.compile(test_func2, dynamic=False)(input)
            eager_results = test_func2(input)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 36)
            self.assertEqual(misses(), 4)

        with fresh_cache():
            reset_counters()
            input = torch.rand(10, 10, device=GPU_TYPE)

            compile_results = torch.compile(test_func2, dynamic=True)(input)
            eager_results = test_func2(input)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 36)
            self.assertEqual(misses(), 4)

        # No cache hit due to symbolic expressions passed i.e mm(s0 + s1, 2) vs mm(s3, 2).
        reset_counters()

        def test_func3(x, y, z, m, l):
            a = torch.matmul(x, y)
            b = torch.matmul(torch.cat([x, z], 1), torch.cat([y, m, l], 0))
            return a, b

        with fresh_cache():
            a = torch.rand(10, 22, device=GPU_TYPE)
            b = torch.rand(22, 30, device=GPU_TYPE)
            c = torch.rand(10, 11, device=GPU_TYPE)
            d = torch.rand(8, 30, device=GPU_TYPE)
            e = torch.rand(3, 30, device=GPU_TYPE)

            compile_results = torch.compile(test_func3, dynamic=True)(a, b, c, d, e)
            eager_results = test_func3(a, b, c, d, e)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)

            self.assertEqual(hits(), 0)
            self.assertEqual(misses(), 7)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_caching_bmm(self):
        def func_test1(x, y, z, m):
            a = torch.bmm(x, y)
            b = torch.bmm(z, m)
            return a, b

        a = torch.rand(10, 10, 22, device=GPU_TYPE)
        b = torch.rand(10, 22, 30, device=GPU_TYPE)

        def hits():
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_hit"
            ]

        def misses():
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_miss"
            ]

        # Valid cache hit.
        with fresh_cache():
            torch._dynamo.utils.counters.clear()
            compile_results = torch.compile(func_test1, dynamic=False)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)
            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "ATEN, TRITON",
        }
    )
    @unittest.skipIf(config.triton.native_matmul, "only test on template-based matmul")
    def test_triton_template_generated_code_caching_mm_plus_mm(self):
        def func_test1(x, y, z, m):
            a = torch.mm(x, y)
            b = torch.mm(z, m)
            sum1 = a + b

            c = torch.mm(x, y)
            d = torch.mm(z, m)
            sum2 = c + d
            return sum1, sum2

        a = torch.rand(10, 40, device=GPU_TYPE)
        b = torch.rand(40, 30, device=GPU_TYPE)

        def hits():
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_hit"
            ]

        def misses():
            return torch._dynamo.utils.counters["inductor"][
                "generated_module_cache_miss"
            ]

        # Valid cache hit.
        with fresh_cache():
            torch._dynamo.utils.counters.clear()
            compile_results = torch.compile(func_test1, dynamic=False)(a, b, a, b)
            eager_results = func_test1(a, b, a, b)
            self.assertEqual(compile_results, eager_results, atol=0.05, rtol=0.05)
            self.assertEqual(hits(), 4)
            self.assertEqual(misses(), 4)

    @fresh_cache()
    @skipIfXpu
    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "ignore decompose_k when native matmul codegen",
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
        autotune_fallback_to_aten=False,
    )
    @parametrize("num_decompose_k_splits", (0, 5, 20))
    @parametrize("decompose_k_threshold", (8, 16))
    def test_max_autotune_decompose_k_envvars(
        self, num_decompose_k_splits, decompose_k_threshold
    ):
        shapes = [(32, 32, 32768), (32, 32, 256)]
        for M, N, K in shapes:
            get_k_splits.cache_clear()
            use_decompose_k_choice.cache_clear()
            a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE)
            b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE)

            with config.patch(
                {
                    "triton.num_decompose_k_splits": num_decompose_k_splits,
                    "triton.decompose_k_threshold": decompose_k_threshold,
                }
            ):
                compiled_func = torch.compile(lambda a, b: a @ b)
                _, code = run_and_get_code(compiled_func, a, b)

                decompose_count = 0
                for codegen in code:
                    if "benchmark_decompose_k_mm" in codegen:
                        decompose_count += 1

                if (
                    K // M < decompose_k_threshold
                    or K // N < decompose_k_threshold
                    or num_decompose_k_splits == 0
                ):
                    self.assertEqual(decompose_count, 0)
                else:
                    self.assertTrue(decompose_count > 0)
                    self.assertTrue(decompose_count <= num_decompose_k_splits)

    @unittest.skipIf(
        TEST_WITH_ROCM, "exhaustive currently only thoroughly tested on NVIDIA"
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "native matmul takes different tuning configs",
    )
    @config.patch(max_autotune=True, max_autotune_gemm_search_space="EXHAUSTIVE")
    def test_max_autotune_exhaustive(self):
        def f(a, b):
            return a @ b

        M, N, K = (1024, 1024, 1024)

        a = torch.randn(M, K, dtype=torch.float16, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(K, N, dtype=torch.float16, device=GPU_TYPE, requires_grad=True)

        with mock.patch(
            "torch._inductor.template_heuristics.registry.get_template_heuristic"
        ) as config_mock:
            config_heuristics = (
                XPUMMTemplateConfigHeuristic()
                if GPU_TYPE == "xpu"
                else CUDAMMTemplateConfigHeuristic()
            )

            # Traditionally, this would be set of all possible configs
            # We mock out the code path for the sake of the unit test
            config_heuristics.exhaustive_configs = [
                GemmConfig(32, 32, 32, 1, 8, group_m=8)
            ]
            config_mock.return_value = config_heuristics

            from torch._dynamo.utils import counters

            compiled_func = torch.compile(f)
            compiled_func(a, b)

            # Only benchmarks 2 choices, aten and the exhaustive triton config
            # Counter can be InductorBenchmarker or TritonBenchmarker
            for counter in counters["inductor"]:
                if "benchmark_gpu" in counter:
                    self.assertEqual(counters["inductor"][counter], 2)

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_mm_k_1(self):
        def mm(x, y):
            return x @ y

        for i in range(90, 100):
            torch._dynamo.reset()
            a = torch.randn((i, 1), device=GPU_TYPE, dtype=torch.float32)
            b = torch.randn((1, i), device=GPU_TYPE, dtype=torch.float32)
            compiled_f = torch.compile(mm)

            out, code = run_and_get_code(compiled_f, a, b)
            torch.testing.assert_close(out, mm(a, b), atol=1e-2, rtol=1e-2)

    @parametrize("op", ("mm", "addmm", "bmm", "baddbmm", "mm_plus_mm"))
    @parametrize("max_autotune", (False, True))
    @config.patch(
        {
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "triton.native_matmul": False,
        }
    )
    def test_autotune_gemm_choice_validation(self, op, max_autotune):
        def generate_inputs_and_func(op_name):
            # Base config with just x and w
            base_inputs = [
                torch.randn(128, 256, device=GPU_TYPE),
                torch.randn(256, 128, device=GPU_TYPE),
            ]
            func = torch.mm
            if op_name == "mm":
                # default
                pass
            elif op_name == "addmm":
                # Add bias for addmm
                base_inputs = [torch.randn(128, device=GPU_TYPE)] + base_inputs
                func = torch.addmm
            elif op_name in ["bmm", "baddbmm"]:
                # Override for batch dimensions
                base_inputs[0] = torch.randn(4, 128, 256, device=GPU_TYPE)
                base_inputs[1] = torch.randn(4, 256, 128, device=GPU_TYPE)
                func = torch.bmm
                if op_name == "baddbmm":
                    # Add batch bias
                    base_inputs = [
                        torch.torch.randn(4, 128, 128, device=GPU_TYPE)
                    ] + base_inputs
                    func = torch.baddbmm
            elif op_name == "mm_plus_mm":
                # Add second matrix pair
                base_inputs += [
                    torch.randn(128, 256, device=GPU_TYPE),
                    torch.randn(256, 128, device=GPU_TYPE),
                ]

                def mmpmm(x, w, x2, w2):
                    return torch.mm(x, w) + torch.mm(x2, w2)

                func = mmpmm
            else:
                raise ValueError(f"Unsupported op: {op_name}")
            return base_inputs, func

        choice_types_seen = set()

        def choice_validator(choices):
            for choice in choices:
                choice_types_seen.add(type(choice))
            return choices

        inputs, fn = generate_inputs_and_func(op)

        add_preprocessing_fn(choice_validator)
        try:
            with config.patch({"max_autotune": max_autotune}):
                compiled_fn = torch.compile(fn, dynamic=False)
                compiled_fn(*inputs)

                if max_autotune:
                    self.assertIn(ExternKernelCaller, choice_types_seen)
                    self.assertIn(TritonTemplateCaller, choice_types_seen)
                else:
                    self.assertIn(ExternKernelCaller, choice_types_seen)
                    self.assertNotIn(TritonTemplateCaller, choice_types_seen)
        finally:
            clear_preprocessing_fns()

    @config.patch(
        {"test_configs.max_mm_configs": 4, "max_autotune_gemm_backends": "ATEN,TRITON"}
    )
    @parametrize("max_autotune_enabled", (True, False))
    def test_autotune_layout_optimization(self, max_autotune_enabled):
        """Test that layouts are flexible when every choice is ExternKernelChoice"""

        # we use a proxy here of bias_addmm and max-autotune because this enables us to see
        # multiple choices in both scenarios (bias_addmm, addmm, triton (max-autotune only))
        # and both bias_addmm and addmm are extern kernel choices
        def layout_checker(choices):
            if choices:
                expected_layout = (
                    FixedLayout if max_autotune_enabled else FlexibleLayout
                )
                for choice in choices:
                    self.assertIsInstance(
                        choice.layout,
                        expected_layout,
                        f"Expected {expected_layout.__name__} with max_autotune={max_autotune_enabled}",
                    )
            return choices

        add_preprocessing_fn(layout_checker)

        try:
            bias = torch.randn(64, device=GPU_TYPE)
            x = torch.randn(32, 128, device=GPU_TYPE)
            w = torch.randn(128, 64, device=GPU_TYPE)

            with config.patch({"max_autotune": max_autotune_enabled}):
                compiled_fn = torch.compile(lambda b, x, w: torch.addmm(b, x, w))
                _ = compiled_fn(bias, x, w)
        finally:
            clear_preprocessing_fns(clear_defaults=False)

    @config.patch(
        {"test_configs.max_mm_configs": 4, "max_autotune_gemm_backends": "TRITON"}
    )
    def test_fixed_layout_at_lowering(self):
        """
        Test that max-autotune with addmm/bmm/mm_plus_mm correctly handles
        padding and maintains correct output strides. Specifically, when matrix
        b with shape (4608, 1490) is padded, its stride should become 1536.
        """

        def mm_func(a, b) -> torch.Tensor:
            a_t = torch.permute(a, [1, 0]).to(torch.bfloat16)
            b_dtype = b.to(torch.bfloat16)
            # Add .to() to make sure that mm could be potentially padded
            # Strides for output are not padded
            return (a_t @ b_dtype).to(torch.float32)

        def addmm_func(a, b, bias) -> torch.Tensor:
            a_t = torch.permute(a, [1, 0]).to(torch.bfloat16)
            b_dtype = b.to(torch.bfloat16)
            bias_dtype = bias.to(torch.bfloat16)
            return torch.addmm(bias_dtype, a_t, b_dtype).to(torch.float32)

        def bmm_func(a, b) -> torch.Tensor:
            a_t = torch.permute(a, [2, 0, 1]).to(torch.bfloat16)
            b_dtype = b.to(torch.bfloat16)
            return torch.bmm(a_t, b_dtype).to(torch.float32)

        def mm_plus_mm_func(a1, b1, a2, b2) -> torch.Tensor:
            a1_t = torch.permute(a1, [1, 0]).to(torch.bfloat16)
            b1_dtype = b1.to(torch.bfloat16)
            a2_t = torch.permute(a2, [1, 0]).to(torch.bfloat16)
            b2_dtype = b2.to(torch.bfloat16)
            return (a1_t @ b1_dtype + a2_t @ b2_dtype).to(torch.float32)

        a = torch.randn((4608, 512), device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn((4608, 1490), device=GPU_TYPE)
        bias = torch.randn(1490, device=GPU_TYPE)

        a_bmm = torch.randn((512, 4608, 8), device=GPU_TYPE, dtype=torch.bfloat16)
        b_bmm = torch.randn((8, 4608, 1490), device=GPU_TYPE)

        # Test mm_plus_mm
        a2 = torch.randn((4608, 512), device=GPU_TYPE, dtype=torch.bfloat16)
        b2 = torch.randn((4608, 1490), device=GPU_TYPE)

        # 1490 padded to 1536, check in template code
        output_code_padding_check = "stride_bk = 1536"
        funcs_and_args = [
            (mm_func, (a, b)),
            (addmm_func, (a, b, bias)),
            (bmm_func, (a_bmm, b_bmm)),
            (mm_plus_mm_func, (a, b, a2, b2)),
        ]

        for f, args in funcs_and_args:
            c_f = torch.compile(f, mode="max-autotune-no-cudagraphs")
            _, code_out = run_and_get_code(c_f, *args)
            FileCheck().check(output_code_padding_check).run(code_out[0])

    @parametrize("k", (15, 16))
    @parametrize("dynamic", (False, True))
    def test_even_k(self, k: int, dynamic: bool):
        M, N = 21, 31
        a = torch.randn((M, k), dtype=torch.float16, device=GPU_TYPE)
        b = torch.randn((k, N), dtype=torch.float16, device=GPU_TYPE)

        if dynamic:
            torch._dynamo.mark_dynamic(a, 1)
            torch._dynamo.mark_dynamic(b, 0)

        with config.patch({"max_autotune": True}), fresh_cache():
            _ = torch.compile(torch.mm)(a, b)
            cache = TritonTemplate.all_templates["mm"]._generated_code_cache._cache
            cache_key = next(iter(cache))

        self.assertObjectIn(k, (15, 16))
        self.assertEqual("'EVEN_K': True" in cache_key, k == 16 and not dynamic)

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "force_pointwise_cat": True,
        }
    )
    @parametrize("epilogue", (True, False))
    def test_deferred_layout_constraint_cat_fusion(self, epilogue):
        def mm_with_cat(a, b1, b2, d):
            catted_b = torch.cat([b1, b2], dim=1)
            catted_b_add = catted_b + 1.0

            # Not padded at lowering time
            catted_bf16 = catted_b_add.to(torch.bfloat16)

            # bmm1 as output -> no padding occurs, however after fusion
            # the buffer is padded, layout constraint violated
            bmm1 = torch.bmm(a.to(torch.bfloat16), catted_bf16)
            fused = catted_bf16 - d
            if epilogue:
                bmm1_fp32 = bmm1.to(torch.float32)
                return bmm1, bmm1_fp32, fused

            return bmm1, fused

        batch = 32
        m = 512
        k1, k2 = 256, 256
        n = 1490  # Would normally trigger padding (1490 -> 1536)

        a = torch.randn(batch, m, k1 + k2, device=GPU_TYPE)
        b1 = torch.randn(batch, k1, n, device=GPU_TYPE)
        b2 = torch.randn(batch, k2, n, device=GPU_TYPE)
        d = torch.cat([b1, b2], dim=1).to(torch.bfloat16)

        with (
            fresh_cache(),
            mock.patch.object(
                AlgorithmSelectorCache,
                "benchmark_choice",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
        ):
            compiled = torch.compile(mm_with_cat, mode="max-autotune-no-cudagraphs")
            _, code = run_and_get_code(compiled, a, b1, b2, d)

            # Despite Triton being 10x faster in benchmarks, the layout conflict
            # should force fallback to extern_kernels.bmm
            FileCheck().check_not("triton_tem").run(code[0])

    @parametrize("always_freeze", [True, False])
    def test_mm_layout_freezing_behavior(self, always_freeze):
        """Test that mm layout freezing behavior depends on always_freeze_layout.

        When always_freeze_layout=True, FlexibleLayout should be frozen to FixedLayout.
        When always_freeze_layout=False (default), FlexibleLayout should remain flexible
        and use layout constraints instead.
        """
        from torch._inductor import ir
        from torch._inductor.kernel.mm import mm_template
        from torch._inductor.select_algorithm import TritonTemplateKernel

        flexible_layout_called = False
        orig_stride_call = TritonTemplateKernel.get_stride_and_maybe_freeze_layout

        def tracking_get_stride(self, node):
            nonlocal flexible_layout_called
            flexible_layout = isinstance(node.data.layout, ir.FlexibleLayout)
            result = orig_stride_call(self, node)
            if flexible_layout:
                flexible_layout_called = True
                if always_freeze:
                    if not isinstance(node.data.layout, ir.FixedLayout):
                        raise AssertionError(
                            f"Expected FixedLayout, got {type(node.data.layout)}"
                        )
                else:
                    if not isinstance(node.data.layout, ir.FlexibleLayout):
                        raise AssertionError(
                            f"Expected FlexibleLayout, got {type(node.data.layout)}"
                        )
            return result

        def fn(x, y):
            # Add intermediate computation to create FlexibleLayout buffer
            x = x + 1
            return torch.mm(x, y)

        M, K, N = 256, 128, 256
        x = torch.randn(M, K, device=GPU_TYPE, dtype=torch.float16)
        y = torch.randn(K, N, device=GPU_TYPE, dtype=torch.float16)

        # Save original value to restore later
        original_always_freeze = mm_template.always_freeze_layout

        with (
            fresh_cache(),
            patch.object(
                TritonTemplateKernel,
                "get_stride_and_maybe_freeze_layout",
                tracking_get_stride,
            ),
            config.patch({"test_configs.max_mm_configs": 1}),
        ):
            torch._dynamo.reset()
            mm_template.always_freeze_layout = always_freeze
            try:
                # Temporarily set always_freeze_layout on mm_template
                mm_template.always_freeze_layout = always_freeze
                compiled_f = torch.compile(fn, mode="max-autotune")
                compiled_f(x, y)
            finally:
                # Restore original value
                mm_template.always_freeze_layout = original_always_freeze

        self.assertTrue(flexible_layout_called)

    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 1,
        }
    )
    def test_deffered_layout_constraint_reintepret(self):
        batch, m, k, n = 4608, 40, 112, 1119

        # Shape: batch x m x k (contiguous)
        a = torch.randn(batch, m, k, dtype=torch.bfloat16, device=GPU_TYPE)

        padded_batch_stride = k * n + 48
        b = torch.empty_strided(
            size=(batch, k, n),
            stride=(padded_batch_stride, 1, k),
            dtype=torch.bfloat16,
            device=GPU_TYPE,
        )
        b.copy_(torch.randn_like(b))
        c = torch.randn(batch, k, m, dtype=torch.bfloat16, device=GPU_TYPE)

        def fn(a, b, c):
            # Apply a pointwise op to b to make it FlexibleLayout in Inductor
            # This ensures Inductor doesn't treat it as a fixed/external layout
            # Ends up double padding
            b_flex = b + 0
            return (
                torch.bmm(a, b_flex).to(torch.float32),
                b + 1.0,
                torch.bmm(b_flex.permute(0, 2, 1), c).to(torch.float32),
            )

        with (
            mock.patch(
                "torch._inductor.autotune_process.run_autotune_in_subprocess",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
            mock.patch.object(
                AlgorithmSelectorCache,
                "benchmark_choice",
                mock_benchmark_choice_wrapper(aten_time=1.0, triton_time=0.1),
            ),
        ):
            compiled_fn = torch.compile(fn)

            # Previously would CUDA IMA
            _, code = run_and_get_code(compiled_fn, a, b, c)

            FileCheck().check("triton_tem_fused").run(code[0])


@instantiate_parametrized_tests
class TestTemplateConfigPruning(TestCase):
    """Test class for pruning logic in GEMM autotuning."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Initialize heuristics once for all tests
        cls.addmm_tma_heuristic = CUDAAddmmPersistentTMATemplateConfigHeuristic()
        cls.addmm_heuristic = CUDAAddMMTemplateConfigHeuristic()
        cls.mm_tma_heuristic = CUDAPersistentTMATemplateConfigHeuristic()
        cls.mm_heuristic = CUDAMMTemplateConfigHeuristic()

        block_sizes = [64, 128, 256]
        num_stages = [4, 5]
        from itertools import product

        cls.gemm_configs = [
            GemmConfig(BLOCK_M, BLOCK_N, BLOCK_K, stage, 8)
            for BLOCK_M, BLOCK_N, BLOCK_K, stage in product(
                block_sizes, block_sizes, block_sizes, num_stages
            )
            # Don't test for very large block sizes nor very small ones
            if BLOCK_M + BLOCK_N + BLOCK_K < 512 and BLOCK_M + BLOCK_N + BLOCK_K > 192
        ]

    def setUp(self):
        super().setUp()
        # Save original configs to restore in tearDown
        self.original_tma_mm_configs = self.mm_tma_heuristic.mm_configs
        self.original_mm_mm_configs = self.mm_heuristic.mm_configs
        self.original_addmm_tma_configs = self.addmm_tma_heuristic.mm_configs
        self.original_addmm_configs = self.addmm_heuristic.mm_configs

    def tearDown(self):
        # Restore original configs
        self.addmm_tma_heuristic.mm_configs = self.original_addmm_tma_configs
        self.addmm_heuristic.mm_configs = self.original_addmm_configs
        self.mm_tma_heuristic.mm_configs = self.original_tma_mm_configs
        self.mm_heuristic.mm_configs = self.original_mm_mm_configs
        super().tearDown()

    @contextlib.contextmanager
    def pruning_config_context(self):
        """Context manager for shared memory pruning configuration."""
        with (
            config.patch(
                {
                    "max_autotune_prune_choices_based_on_shared_mem": False,
                    "triton.enable_persistent_tma_matmul": True,
                    "inductor_default_autotune_warmup": 0,
                    "inductor_default_autotune_rep": 1,
                }
            ),
            fresh_cache(),
        ):
            yield

    def create_test_tensors(
        self,
        M,
        N,
        K,
        include_bias=False,
        dtype=torch.bfloat16,
        mat1_transposed=False,
        mat2_transposed=False,
    ):
        if mat1_transposed:
            mat1 = torch.randn(K, M, dtype=dtype, device=GPU_TYPE).t()
        else:
            mat1 = torch.randn(M, K, dtype=dtype, device=GPU_TYPE)

        if mat2_transposed:
            mat2 = torch.randn(N, K, dtype=dtype, device=GPU_TYPE).t()
        else:
            mat2 = torch.randn(K, N, dtype=dtype, device=GPU_TYPE)

        if include_bias:
            bias_1d = torch.randn(N, dtype=dtype, device=GPU_TYPE)
            return bias_1d, mat1, mat2
        return mat1, mat2

    def test_max_autotune_prune_choices(self):
        def mm(x, y):
            return x @ y

        M, K, N = (3, 3, 3)

        x = torch.rand([M, K], device=GPU_TYPE, dtype=torch.float32)
        y = torch.rand([K, N], device=GPU_TYPE, dtype=torch.float32)

        compiled_f = torch.compile(mm, mode="max-autotune")
        compiled_f(x, y)

        self.assertEqual(
            counters["inductor"]["select_algorithm_num_precompilation_exceptions"], 0
        )

    @skipIfXpu(msg="Missing device_properties shared_memory_per_block on xpu.")
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("mat1_transposed", (False, True))
    @parametrize("mat2_transposed", (False, True))
    @parametrize("use_tma", (False, True))
    def test_shared_memory_pruning_addmm(
        self,
        dtype: torch.dtype,
        mat1_transposed: bool,
        mat2_transposed: bool,
        use_tma: bool,
    ):
        """Test shared memory pruning for addmm operation."""

        if use_tma and (dtype == torch.float32 or not has_triton_tma_device()):
            return

        def addmm_op(bias, mat1, mat2):
            return torch.addmm(bias, mat1, mat2)

        M, K, N = 512, 512, 512
        bias_1d, mat1, mat2 = self.create_test_tensors(
            M,
            N,
            K,
            include_bias=True,
            dtype=dtype,
            mat1_transposed=mat1_transposed,
            mat2_transposed=mat2_transposed,
        )
        dtype_size = mat1.dtype.itemsize

        if use_tma:
            self.addmm_heuristic.mm_configs = []
            heuristic = self.addmm_tma_heuristic
        else:
            self.addmm_tma_heuristic.mm_configs = []
            heuristic = self.addmm_heuristic

        shared_memory_checker_opts = get_shared_memory_checker_opts("addmm", dtype_size)

        self.run_op_shared_mem_pruning_check(
            heuristic,
            addmm_op,
            (bias_1d, mat1, mat2),
            dtype_size,
            shared_memory_checker_opts,
        )

    @skipIfXpu(msg="Missing device_properties shared_memory_per_block on xpu.")
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("mat1_transposed", (False, True))
    @parametrize("mat2_transposed", (False, True))
    @parametrize("use_tma", (False, True))
    def test_shared_memory_pruning_mm(
        self,
        dtype: torch.dtype,
        mat1_transposed: bool,
        mat2_transposed: bool,
        use_tma: bool,
    ):
        if use_tma and (dtype == torch.float32 or not has_triton_tma_device()):
            return

        def mm_op(mat1, mat2):
            return mat1 @ mat2

        M, K, N = 512, 512, 512
        mat1, mat2 = self.create_test_tensors(
            M,
            N,
            K,
            include_bias=False,
            dtype=dtype,
            mat1_transposed=mat1_transposed,
            mat2_transposed=mat2_transposed,
        )
        dtype_size = mat1.dtype.itemsize

        if use_tma:
            self.mm_heuristic.mm_configs = []
            heuristic = self.mm_tma_heuristic
        else:
            self.mm_tma_heuristic.mm_configs = []
            heuristic = self.mm_heuristic

        shared_memory_checker_opts = get_shared_memory_checker_opts("mm", dtype_size)

        self.run_op_shared_mem_pruning_check(
            heuristic, mm_op, (mat1, mat2), dtype_size, shared_memory_checker_opts
        )

    def run_op_shared_mem_pruning_check(
        self, heuristic, op, inputs, dtype_size, shared_memory_checker_opts
    ):
        exceeds_checker = heuristic._get_exceeding_shared_memory_checker(
            **shared_memory_checker_opts
        )
        if exceeds_checker is None:
            self.skipTest("Device does not support shared memory size query")
        for c in self.gemm_configs:
            smem_estimation = heuristic.get_shared_memory_estimation(
                c, dtype_size, **shared_memory_checker_opts
            )
            # Configure heuristics to use only this specific config
            heuristic.mm_configs = [c]
            exceeds = exceeds_checker(c, dtype_size)

            original_precompile = CachingAutotuner.precompile
            original_autotune = AlgorithmSelectorCache.autotune

            captured_smem = 0
            triton_compilation_fails = True

            def mock_precompile(self, *args, **kwargs):
                original_precompile(self, *args, **kwargs)
                # Access compile_results after precompilation
                for result in self.compile_results:
                    # Get shared memory from the compiled kernel
                    kernel = result.kernel
                    shared_mem = (
                        kernel.shared
                        if hasattr(kernel, "shared")
                        else kernel.metadata.shared
                    )
                    nonlocal captured_smem
                    captured_smem = shared_mem

            def mock_autotune(self, *args, **kwargs):
                timings = original_autotune(self, *args, **kwargs)
                nonlocal triton_compilation_fails
                for caller, t in timings.items():
                    if isinstance(caller, TritonTemplateCaller) and t != float("inf"):
                        triton_compilation_fails = False
                return timings

            with (
                self.pruning_config_context(),
                mock.patch.object(CachingAutotuner, "precompile", mock_precompile),
                mock.patch.object(AlgorithmSelectorCache, "autotune", mock_autotune),
            ):
                torch._dynamo.reset()
                counters.clear()
                compiled_fn = torch.compile(op, mode="max-autotune")
                run_and_get_code(compiled_fn, *inputs)

            if triton_compilation_fails:
                self.assertTrue(
                    exceeds,
                    f"Config {c} failed to compile due to shared memory, "
                    "but the checker predicted it would NOT exceed shared memory limits.",
                )
            else:
                self.assertTrue(
                    captured_smem <= smem_estimation,
                    f"Estimated maximum smem should exceed actual smem used for config {c}",
                )


class TestMaxAutotunePrecompile(TestCase):
    def test_precompilation_threads(self):
        import threading
        from typing import Any
        from unittest.mock import Mock

        class FakeChoiceCaller(ChoiceCaller):
            def __init__(self) -> None:
                super().__init__("none", [], Mock(), description="")
                self.thread_id = None

            def precompile(self):
                self.thread_id = threading.get_ident()

            def call_name(self) -> str:
                return None

            def to_callable(self):
                return None

            def hash_key(self) -> str:
                return str(hash(self))

            def output_node(self) -> "TensorBox":  # noqa: F821
                return None

        fake_choices = [FakeChoiceCaller() for i in range(10)]
        fake_lookup_result = dict.fromkeys(fake_choices, 0.123)

        def no_lookup(
            choices: list[ChoiceCaller],
            op: str,
            inputs: str,
            benchmark: Callable[[Any], dict[ChoiceCaller, float]],
            hint_override: Optional[int] = None,
        ) -> Optional[dict[ChoiceCaller, float]]:
            if benchmark is not None:
                return benchmark(choices)

        asc = AlgorithmSelectorCache()

        def fake_benchmark_fn(*args, **kwargs):
            return fake_lookup_result

        main_thread_id = threading.get_ident()
        mock_debug_handler = Mock()
        old_debug_handler = V.debug
        try:
            V.set_debug_handler(mock_debug_handler)
            with patch.object(asc, "lookup", new=no_lookup):
                with patch.object(
                    asc, "make_benchmark_fn", return_value=fake_benchmark_fn
                ):
                    with config.patch(
                        {
                            "autotune_in_subproc": False,
                            "compile_threads": len(fake_choices),
                        }
                    ):
                        asc("test_call", fake_choices, [], Mock())
            for fake_choice in fake_choices:
                if fake_choice.thread_id is None:
                    raise AssertionError(
                        "Expected all ChoiceCaller's precompile method to have been called"
                    )
                if fake_choice.thread_id == main_thread_id:
                    raise AssertionError(
                        "Expected all ChoiceCaller's precompile method to have been called on separate thread"
                    )
        finally:
            V.set_debug_handler(old_debug_handler)

    def test_filled_cache_precompile(self):
        def fn(a, b, c):
            a = (a @ b) @ c
            a, b, c = (t.to(torch.float16) for t in [a, b, c])
            return (a @ b) @ c

        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        inputs = [torch.rand([256, 256], device=GPU_TYPE) for _ in range(3)]
        from torch._dynamo.utils import counters

        self.assertEqual(fn(*inputs), fn_c(*inputs), atol=1e-2, rtol=1e-2)

        torch._dynamo.reset()
        counters.clear()

        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        self.assertEqual(counters["inductor"]["select_algorithm_precompile"], 0)

    @config.patch(autotune_local_cache=False, autotune_remote_cache=False)
    @runOnRocmArch(MI300_ARCH)
    @unittest.skipIf(config.triton.native_matmul, "native matmul has counter 0")
    def test_precompilations(self):
        def fn(a, b, c):
            a = (a @ b) @ c
            a, b, c = (t.to(torch.float16) for t in [a, b, c])
            return (a @ b) @ c

        fn_c = torch.compile(mode="max-autotune-no-cudagraphs")(fn)
        inputs = [torch.rand([256, 256], device=GPU_TYPE) for _ in range(3)]

        torch.testing.assert_close(fn_c(*inputs), fn(*inputs), atol=1e-2, rtol=1e-2)

        from torch._dynamo.utils import counters

        self.assertEqual(counters["inductor"]["select_algorithm_precompile"], 2)


@instantiate_parametrized_tests
class TestMaxAutotuneSubproc(TestCase):
    def _create_buffer(self, name, shape):
        return Buffer(
            name=name,
            layout=FixedLayout(
                torch.device(f"{GPU_TYPE}:0"), dtype=torch.float32, size=shape
            ),
        )

    @skipIfXpu(msg="XPU not support multiprocessing tensor reduction")
    def test_benchmark_choice_in_subproc(self):
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is needed to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            layout = FixedLayout(torch.device(f"{GPU_TYPE}:0"), torch.float32, (2, 2))

            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            # expected_out = (mat1 @ mat2) + (mat3 @ mat4)
            expected_out = None

            choice = aten_mm_plus_mm.bind((buf1, buf2, buf3, buf4), layout)
            # use a tensor since the mutation to a python list in a sub process
            # is not synced back to the parent process
            timings = torch.zeros(3, dtype=torch.float32)
            ctx = mp.get_context("spawn")
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            child.start()
            child.join()
            self.assertEqual(0, child.exitcode)
            print(f"timings is {timings}, out {out}, expected_out {expected_out}")

    @skipIfXpu(msg="XPU not support multiprocessing tensor reduction")
    def test_benchmark_choice_fail_in_subproc(self):
        gm = make_fx(
            lambda: torch.zeros(2, 3)
        )()  # a dummy graph to construct the GraphLowering
        graph = GraphLowering(gm)

        # the graph handler is needed to create benchmark example value below
        with V.set_graph_handler(graph):
            buf1 = self._create_buffer("mat1", (2, 3))
            buf2 = self._create_buffer("mat2", (3, 2))
            buf3 = self._create_buffer("mat3", (2, 3))
            buf4 = self._create_buffer("mat4", (3, 2))

            layout = FixedLayout(torch.device(f"{GPU_TYPE}:0"), torch.float32, (2, 2))

            mat1 = AlgorithmSelectorCache.benchmark_example_value(buf1)
            mat2 = AlgorithmSelectorCache.benchmark_example_value(buf2)
            mat3 = AlgorithmSelectorCache.benchmark_example_value(buf3)
            mat4 = AlgorithmSelectorCache.benchmark_example_value(buf4)

            out = AlgorithmSelectorCache.benchmark_example_value(layout)
            expected_out = (mat1 @ mat2) + (mat3 @ mat4)

            choice = FailChoiceCaller("fail_choice_caller", [], None, description="")

            # use a tensor since python list is not synced back
            timings = torch.zeros(3, dtype=torch.float32)
            ctx = mp.get_context("spawn")
            child = ctx.Process(
                target=benchmark_choice,
                args=(choice, (mat1, mat2, mat3, mat4), out, expected_out, timings),
            )
            child.start()
            child.join()
            self.assertNotEqual(0, child.exitcode)

    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("autotune_multi_device", (True, False))
    def test_max_autotune_mm_plus_mm(self, autotune_in_subproc, autotune_multi_device):
        """
        This crash previously due to a triton issue: https://github.com/triton-lang/triton/issues/1298 .
        With autotuning in subprocess, we don't crash anymore.
        """
        m, n, k = 2048, 1536, 64

        def mm_plus_mm(a, b, c, d):
            return a @ b + c @ d

        a = torch.randn(m, k).to(GPU_TYPE)
        b = torch.randn(k, n).to(GPU_TYPE)
        c = torch.randn(m, k).to(GPU_TYPE)
        d = torch.randn(k, n).to(GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": autotune_in_subproc,
                "autotune_multi_device": autotune_multi_device,
            }
        ):
            torch.compile(mm_plus_mm)(a, b, c, d)

    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm(self, dynamic: bool):
        """
        Make sure autotuning mm in sub processes work without crashes.
        """

        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        with config.patch({"max_autotune": True, "autotune_in_subproc": True}):
            torch.compile(mm, dynamic=dynamic)(a, b)

    @parametrize("search_space", ("DEFAULT", "EXHAUSTIVE"))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm(self, search_space, dynamic=False):
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_search_space": search_space,
            }
        ):
            Y_compiled = torch.compile(addmm, dynamic=dynamic)(x, a, b)
            Y = addmm(x, a, b)
            torch.testing.assert_close(Y_compiled, Y, atol=1e-2, rtol=1e-2)

    def test_triton_template_with_epilogues_and_dynamic_shape(self):
        def fn(
            x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, mul: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.nn.functional.relu(
                    torch.matmul(torch.transpose(x, 0, 1), torch.transpose(w, 0, 1))
                    + bias
                )
                * mul
            )

        M0 = 5
        M1 = 8
        K = 4
        N = 3
        w = torch.rand(N, K).to(GPU_TYPE).half()
        b = torch.rand(N).to(GPU_TYPE).half()

        with config.patch(
            {
                "max_autotune": True,
                "autotune_in_subproc": True,
                "max_autotune_gemm_backends": "Triton",
            }
        ):
            compiled_fn = torch.compile(
                fn, fullgraph=True, dynamic=True, mode="max-autotune-no-cudagraphs"
            )

            x0 = torch.rand(K, M0).to(GPU_TYPE).half()
            mul0 = torch.rand(M0, N).to(GPU_TYPE).half()
            y0 = compiled_fn(x0, w, b, mul0)
            y0_expected = fn(x0, w, b, mul0)
            torch.testing.assert_close(y0, y0_expected)

            x1 = torch.rand(K, M1).to(GPU_TYPE).half()
            mul1 = torch.rand(M1, N).to(GPU_TYPE).half()
            y1 = compiled_fn(x1, w, b, mul1)
            y1_expected = fn(x1, w, b, mul1)
            torch.testing.assert_close(y1, y1_expected)


@instantiate_parametrized_tests
class TestMaxAutotuneRemoteCache(TestCase):
    def setUp(self):
        super().setUp()
        PatchCaches.setUp()

    def tearDown(self):
        super().tearDown()
        PatchCaches.tearDown()

    @parametrize("dynamic", (False, True))
    @config.patch(
        {"compile_threads": 1, "prologue_fusion": False}
    )  # Worker processes do not register PatchCaches() properly
    def test_max_autotune_remote_caching(self, dynamic: bool):
        def mm(a, b):
            a = torch.sin(a)
            return a @ b

        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        def f(x, y):
            return Model()(x, y)

        x = torch.randn(100, 100).to(GPU_TYPE)
        y = torch.randn(100, 100).to(GPU_TYPE)

        with (
            config.patch(
                {
                    "autotune_local_cache": False,
                    "autotune_remote_cache": True,
                }
            ),
            patch.dict(os.environ),
            PatchCaches(),
        ):
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            with config.patch({"max_autotune": True}):
                for _ in range(4):
                    with fresh_cache():
                        torch.compile(mm, dynamic=dynamic)(a, b)
                    reset()
                with (
                    torch.compiler.config.patch({"cache_key_tag": "test"}),
                    fresh_cache(),
                ):
                    torch.compile(mm, dynamic=dynamic)(a, b)
                    reset()

                global_stats.report()
                self.assertEqual(global_stats.autotune_remote, Stats(2, 3, 2))

            global_stats.reset()
            for _ in range(4):
                with fresh_cache():
                    torch.compile(f, dynamic=dynamic)(x, y)
                reset()
            with torch.compiler.config.patch({"cache_key_tag": "test"}), fresh_cache():
                torch.compile(mm, dynamic=dynamic)(a, b)
                reset()
            global_stats.report()
            self.assertEqual(global_stats.autotune_remote, Stats(2, 3, 2))


class _TestTritonTemplateCaller(TritonTemplateCaller):
    def __init__(self, bmreq: _TestBenchmarkRequest):
        self.bmreq = bmreq

    def __str__(self) -> str:
        return "test"


class TestTuningProcess(TestCase):
    def check_healthy(self, p: TuningProcess, device: Optional[int] = None):
        result = random.random()
        bmreq = _TestBenchmarkRequest(result, device=device)
        p.put(bmreq.benchmark)
        self.assertEqual(p.get(), result)

    def test_tuning_subproc_timeout(self):
        p = TuningProcess(None)

        bmreq = _TestBenchmarkRequest(0, sleep=120)
        p.put(bmreq.benchmark)
        with self.assertRaises(TimeoutError):
            p.get(timeout=1.0)

        # Make sure the TuningProcess is still usable after a timeout.
        self.check_healthy(p)
        p.shutdown()

    def test_tuning_subproc_exception(self):
        p = TuningProcess(None)

        bmreq = _TestBenchmarkRequest(0, exc=RuntimeError("Fail"))
        p.put(bmreq.benchmark)
        with self.assertRaises(RuntimeError):
            p.get()

        # Make sure the TuningProcess is still usable after an exception.
        self.check_healthy(p)
        p.shutdown()

    def test_tuning_subproc_crash(self):
        p = TuningProcess(None)

        bmreq = _TestBenchmarkRequest(0, crash=True)
        p.put(bmreq.benchmark)
        with self.assertRaises(EOFError):
            p.get()

        # Make sure the TuningProcess is still usable after a crash.
        self.check_healthy(p)
        p.shutdown()

    def test_tuning_subproc_killed(self):
        p = TuningProcess(None)
        p.kill()
        self.check_healthy(p)
        p.shutdown()

    def test_visible_devices(self):
        device_list = TuningProcessPool.get_device_list()
        for device in device_list:
            p = TuningProcess(device)
            self.check_healthy(p, device=device)
            p.shutdown()


class TestTuningProcessPool(TestCase):
    # Use only one device/subprocess so we test the process restarts
    # and is usable after a crash.
    @config.patch({"autotune_multi_device": False})
    def test_tuning_pool_crash(self):
        tuning_pool = TuningProcessPool()

        # First force the tuning process to crash.
        bmreq = _TestBenchmarkRequest(0, crash=True)
        choice = _TestTritonTemplateCaller(bmreq)

        timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], float("inf"))

        # Then send another request and make sure the sub-process
        # has restarted and is operational.
        bmreq = _TestBenchmarkRequest(3.14)
        choice = _TestTritonTemplateCaller(bmreq)

        timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], bmreq.result)

        tuning_pool.shutdown()

    @config.patch({"autotune_multi_device": False})
    def test_tuning_pool_timeout(self):
        tuning_pool = TuningProcessPool()

        # First force the tuning process to timeout.
        bmreq = _TestBenchmarkRequest(0, sleep=120)
        choice = _TestTritonTemplateCaller(bmreq)

        with config.patch({"max_autotune_subproc_result_timeout_seconds": 1.0}):
            timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], float("inf"))

        # Then send another request and make sure the sub-process
        # has restarted and is operational.
        bmreq = _TestBenchmarkRequest(3.14)
        choice = _TestTritonTemplateCaller(bmreq)

        timings = tuning_pool.benchmark([choice])
        self.assertTrue(choice in timings)
        self.assertEqual(timings[choice], bmreq.result)

        tuning_pool.shutdown()

    @skipIfXpu(msg="XPU not support VISIBLE_DEVICES")
    @config.patch({"autotune_multi_device": True})
    def test_tuning_pool_multiple_devices(self):
        # Adapt the test to the available devices (and whether CUDA_VISIBLE_DEVICES
        # is already set in the environment); use a subset of the available devices
        # to ensure only the subset are visible to the sub-processes.
        if CUDA_VISIBLE_DEVICES in os.environ:
            visible_devices = os.environ[CUDA_VISIBLE_DEVICES].split(",")
        else:
            visible_devices = [str(d) for d in range(torch.cuda.device_count())]

        cuda_visible_devices = ",".join(visible_devices[-2:])
        with unittest.mock.patch.dict(
            os.environ, {CUDA_VISIBLE_DEVICES: cuda_visible_devices}
        ):
            tuning_pool = TuningProcessPool()

        choice1 = _TestTritonTemplateCaller(_TestBenchmarkRequest(3.14))
        choice2 = _TestTritonTemplateCaller(_TestBenchmarkRequest(2.718))

        timings = tuning_pool.benchmark([choice1, choice2])
        self.assertEqual(timings[choice1], choice1.bmreq.result)
        self.assertEqual(timings[choice2], choice2.bmreq.result)

        tuning_pool.shutdown()

    def test_add_feedback_saver(self):
        """Test that add_feedback_saver correctly adds feedback functions."""
        from torch._inductor.select_algorithm import get_algorithm_selector_cache

        # Clear any existing feedback savers
        clear_feedback_savers()

        # Create a simple feedback saver function
        feedback_calls = []

        def simple_feedback_saver(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            feedback_calls.append(
                {
                    "name": name,
                    "num_choices": len(choices),
                    "num_timings": len(timings),
                    "has_profiled_time": profiled_time is not None,
                }
            )

        # Add the feedback saver
        add_feedback_saver(simple_feedback_saver)

        # Get the global cache and verify the function was added
        cache = get_algorithm_selector_cache()
        self.assertEqual(len(cache.feedback_saver_fns), 1)
        self.assertEqual(cache.feedback_saver_fns[0], simple_feedback_saver)

        # Test that we can add multiple feedback savers
        def another_feedback_saver(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            pass

        add_feedback_saver(another_feedback_saver)
        self.assertEqual(len(cache.feedback_saver_fns), 2)

        # Clean up
        clear_feedback_savers()

    def test_clear_feedback_savers(self):
        """Test that clear_feedback_savers removes all feedback functions."""
        from torch._inductor.select_algorithm import get_algorithm_selector_cache

        # Add some feedback savers first
        def feedback_saver1(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            pass

        def feedback_saver2(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            pass

        add_feedback_saver(feedback_saver1)
        add_feedback_saver(feedback_saver2)

        # Verify they were added
        cache = get_algorithm_selector_cache()
        self.assertEqual(len(cache.feedback_saver_fns), 2)

        # Clear all feedback savers
        clear_feedback_savers()

        # Verify they were cleared
        self.assertEqual(len(cache.feedback_saver_fns), 0)

    def test_feedback_saver_integration(self):
        """Test that feedback savers are actually called during autotuning."""
        # Clear any existing feedback savers
        clear_feedback_savers()

        feedback_calls = []

        def test_feedback_saver(
            timings, name, input_nodes, choices, profiled_time, precompile_times
        ):
            # Store information about the call for verification
            feedback_calls.append(
                {
                    "name": name,
                    "num_choices": len(choices),
                    "num_timings": len(timings),
                    "input_node_count": len(input_nodes),
                }
            )

        # Add our test feedback saver
        add_feedback_saver(test_feedback_saver)

        # Create a simple matrix multiplication that will trigger autotuning
        def mm(a, b):
            return a @ b

        a = torch.randn(32, 32, device=GPU_TYPE)
        b = torch.randn(32, 32, device=GPU_TYPE)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
                "triton.native_matmul": False,
            }
        ):
            torch.compile(mm)(a, b)

        # Verify that our feedback saver was called
        self.assertGreater(
            len(feedback_calls), 0, "Feedback saver should have been called"
        )

        # Verify the structure of the feedback call
        call = feedback_calls[0]
        self.assertIn("name", call)
        self.assertIn("num_choices", call)
        self.assertIn("num_timings", call)
        self.assertIn("input_node_count", call)
        self.assertGreater(call["num_choices"], 0)
        self.assertEqual(call["input_node_count"], 2)  # Two input matrices

        # Clean up
        clear_feedback_savers()


@instantiate_parametrized_tests
class TestPrologueFusion(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "max_autotune": True,
                    "prologue_fusion": True,
                    "benchmark_epilogue_fusion": False,
                    "shape_padding": False,
                    "max_autotune_gemm_backends": "TRITON",
                    "test_configs.max_mm_configs": 4,  # significantly speeds up tests
                }
            )
        )

    def check_code(self, code_str, num_kernels, num_allocs, num_deallocs):
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(),
            num_kernels,
            exactly=True,
        ).run(code_str)

        if num_allocs is not None:
            FileCheck().check(get_func_call()).check_count(
                "empty_strided", num_allocs, exactly=True
            ).run(code_str)

        # skip the deallocation check when using cpp_wrapper; most deallocations happen
        # outside of our control via RAIIAtenTensorHandle
        if num_deallocs is not None and not config.cpp_wrapper:
            FileCheck().check(get_func_call()).check_count(
                "del", num_deallocs, exactly=True
            ).run(code_str)

    @parametrize("sizes", ((64, 128, 256), (128, 128, 128), (63, 120, 250)))
    def test_upcast(self, sizes):
        M, K, N = sizes

        x = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)

        def foo(x, y):
            return x.to(y.dtype) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)
        if config.triton.native_matmul:
            # native matmul preserves zero mask - need to optimize; see codegen/triton.py
            FileCheck().check("a =").check("tl.where").check("tl.dot").run(code[0])
        else:
            # upcast preserves zero mask
            FileCheck().check("a =").check_not("tl.where").check("tl.dot").run(code[0])

    @unittest.skip("Triton bug in compilation")
    def test_gather_fusion(self):
        M, K, N = (64, 128, 256)
        x = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float16, device=GPU_TYPE)

        index = torch.randperm(M, device=GPU_TYPE)

        def foo(x, y, index):
            return (x[index]) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y, index)
        self.assertEqual(out, foo(x, y, index), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=3)

        # should be done in low precision
        (
            FileCheck()
            .check("for k_idx")
            .check_not("to(tl.float32)")
            .check("dot")
            .run(code[0])
        )

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
    @config.patch({"triton.native_matmul": False})
    def test_low_precision(self):
        M = K = N = 128

        x = torch.rand([M, K], device=GPU_TYPE).to(torch.float8_e4m3fn)
        y = torch.rand([K, N], dtype=torch.bfloat16, device=GPU_TYPE)

        def foo(x, y):
            return x.to(y.dtype) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

        # should be done in low precision, no arithmetic
        (
            FileCheck()
            .check("for k_idx")
            .check_not("to(tl.float32)")
            .check("dot")
            .run(code[0])
        )

        def foo(x, y):
            return (x.to(y.dtype) + 1) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)

        # should not be done in low precision, two kernels
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_downcast(self):
        # per heuristics, dont fuse a downcast into a mm because it would lead to more reads inside kernel
        M, K, N = (64, 128, 256)
        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float16, device=GPU_TYPE)

        def foo(x, y):
            return x.to(y.dtype) @ y

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=3)

    @parametrize("sizes", ((64, 128, 256), (64, 64, 64), (64, 120, 64)))
    @parametrize("use_async_compile", (True, False))
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_multiple_fusions(self, sizes, use_async_compile: bool):
        M, K, N = sizes

        def foo(x, y):
            return ((x - 1.1) @ (y + 1.1)) * 1.1

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

        # check that we do not CSE any variables between prologues, epilogues
        FileCheck().check("def triton").check_count(
            "tl.full([1], 1.1, tl.float32)", 3, exactly=True
        ).check("tl.store").run(code[0])

    @config.patch(
        {
            "max_autotune_gemm_backends": "Triton",
            "benchmark_epilogue_fusion": True,
            "max_epilogue_benchmarked_choices": 3,
        }
    )
    @parametrize("use_async_compile", (True, False))
    def test_pending_fusions_multiple(self, use_async_compile: bool):
        def multi_use(x, y):
            return (x @ x.T) * (y @ y.T)

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([128, 16], device=GPU_TYPE)
        y = torch.rand([128, 32], device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(multi_use), x, y)

        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), 2, exactly=True
        ).run(code[0])
        self.assertEqual(out, multi_use(x, y), atol=0.05, rtol=0.05)

        def resolve_pending(x):
            return (x @ x).relu()

        x = torch.rand([128, 128], device=GPU_TYPE)
        out, code = run_and_get_code(torch.compile(resolve_pending), x)
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), 1, exactly=True
        ).run(code[0])
        self.assertEqual(out, resolve_pending(x), atol=0.05, rtol=0.05)

    @config.patch(
        {
            "max_autotune_gemm_backends": "Triton",
            "benchmark_epilogue_fusion": True,
            "max_epilogue_benchmarked_choices": 3,
        }
    )
    @parametrize("use_async_compile", (True, False))
    def test_pending_fusion_pro_and_epi(self, use_async_compile: bool):
        def test_multiple_fusions(x):
            y = x.to(torch.float)
            return (y @ y).relu()

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([128, 128], dtype=torch.float16, device=GPU_TYPE)
        out, code = run_and_get_code(torch.compile(test_multiple_fusions), x)
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), 1, exactly=True
        ).run(code[0])
        self.assertEqual(out, test_multiple_fusions(x), atol=0.05, rtol=0.05)

    @parametrize("sizes", ((64, 128, 256), (128, 128, 128), (63, 120, 250)))
    @parametrize("use_async_compile", (True, False))
    def test_multiple_inputs(self, sizes, use_async_compile: bool):
        M, K, N = sizes

        def foo(x, y, z):
            return (x + y).to(torch.float) @ z

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        y = torch.rand([M, K], dtype=torch.float16, device=GPU_TYPE)
        z = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)
        out_eager = foo(x, y, z)
        out, code = run_and_get_code(torch.compile(foo), x, y, z)
        self.assertEqual(out, out_eager, atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=3)

    def test_storage_offset_prologue(self):
        def foo(a):
            q = a[:64, :]
            k = a[64:, :]
            return torch.mm(q + 2, k - 2)

        inp = torch.randn(128, 64, device=GPU_TYPE)
        out, code = run_and_get_code(torch.compile(foo), inp)
        self.assertEqual(out, foo(inp), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=1)

    @config.patch(realize_reads_threshold=1, realize_opcount_threshold=1)
    @parametrize("sizes", ((64, 128, 256), (128, 128, 128), (63, 120, 250)))
    @parametrize("use_async_compile", (True, False))
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_prologue_multiple_nodes(self, sizes, use_async_compile: bool):
        M, K, N = sizes

        def foo(x, y):
            return ((((x * 2) - 1) / 2) @ (y * 4)) * 3.0

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    @parametrize("K", (63, 64))
    def test_broadcast_x(self, K):
        def foo(x, y):
            return (x.expand([1, y.shape[0]]) + 1) @ y

        x = torch.rand([1, 1], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, 128], dtype=torch.float, device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo, dynamic=True), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    def test_broadcast_y(self):
        def foo(x, y):
            return x @ y

        M = 20
        N = K = 1
        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 0)

        out, code = run_and_get_code(torch.compile(foo, dynamic=True), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_preserves_zero_analysis(self):
        fns = (
            (lambda x: x.relu(), False),  # preserves zero
            (lambda x: x + 1, True),  # does not
            (
                lambda x: torch.hypot(x, x),
                True,
            ),  # not handled in analysis, conservatively assume does not preserve
        )

        def foo(x, y, fn):
            return fn(x) @ y

        for fn, should_mask in fns:
            x = torch.rand([64, 127], dtype=torch.float, device=GPU_TYPE)
            y = torch.rand([127, 64], dtype=torch.float, device=GPU_TYPE)

            out, code = run_and_get_code(torch.compile(foo), x, y, fn)
            self.assertEqual(out, foo(x, y, fn), atol=0.05, rtol=0.05)
            self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

            if should_mask:
                f = FileCheck().check("k_idx").check("a =").check_same("tl.where")
            else:
                f = FileCheck().check("k_idx").check("a =").check_not("tl.where")
            f.check("tl.dot").run(code[0])

    @config.patch(realize_reads_threshold=1, realize_opcount_threshold=1)
    @parametrize("benchmark_fusion", (True, False))
    @parametrize("use_async_compile", (True, False))
    def test_prologue_read_into_both_inputs(
        self, benchmark_fusion, use_async_compile: bool
    ):
        M = K = 256

        # not supported today. it could be, but typically the pointwise nodes would get
        # inlined into separate nodes.

        def foo(x):
            y = (x + 1) * 2
            return y @ (y - 2)

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        with config.patch(benchmark_epilogue_fusion=benchmark_fusion):
            x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)

            out, code = run_and_get_code(torch.compile(foo), x)
            self.assertEqual(out, foo(x), atol=0.05, rtol=0.05)
            # not guaranteed to fuse, but still checking correctness
            if not benchmark_fusion:
                self.check_code(
                    code[0], num_kernels=2, num_allocs=None, num_deallocs=None
                )

    @config.patch(realize_reads_threshold=1, realize_opcount_threshold=1)
    @config.patch(allow_buffer_reuse=False)
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_mismatched_prologue_group(self):
        def foo(x, y, z):
            a = (x + 2) * 2
            b = a * y
            return b @ z

        x = torch.rand([1, 256], device=GPU_TYPE)
        y = torch.rand([256, 256], device=GPU_TYPE)
        z = torch.rand([256, 128], device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo), x, y, z)
        self.assertEqual(out, foo(x, y, z), atol=0.05, rtol=0.05)
        # there's one more dealloc than there should be because of a buffer reuse. TODO:
        # not sure why disabling buffer reuse doesn't stop
        self.check_code(code[0], num_kernels=2, num_allocs=2, num_deallocs=4)

    @skipIfXpu
    @config.patch(
        {
            "max_autotune": True,
            "benchmark_epilogue_fusion": True,
            "epilogue_fusion": True,
            "prologue_fusion": True,
        }
    )
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    @parametrize("use_async_compile", (True, False))
    def test_lazy_template_fusion_multiple_candidates(self, use_async_compile: bool):
        """
        Test lazy evaluation of template fusions with multiple templates,
        multiple potential prologues, and a shared epilogue.

        This test creates a computation graph with:
        - 2 matmul templates (mm1, mm2)
        - Multiple prologue candidates for each template (type conversions + arithmetic)
        - A shared epilogue candidate that consumes both mm1 and mm2 outputs,
          creating a fusion opportunity with either template

        The lazy evaluation logic should:
        1. Defer fusion decisions until all candidates are identified
        2. Process epilogue fusions before prologue fusions
        3. Cache attempted fusions to avoid redundant compilation
        """

        def foo(a, b, c, d):
            # Prologues for first matmul: transform inputs a and b
            a_transformed = a.to(torch.float) + 1.0
            b_transformed = b.to(torch.float) * 2.0

            # Prologues for second matmul: transform inputs c and d
            c_transformed = c.to(torch.float) - 0.5
            d_transformed = d.to(torch.float) / 2.0

            # Two matmul templates
            mm1 = a_transformed @ b_transformed
            mm2 = c_transformed @ d_transformed

            # Shared epilogue: complex element-wise operations on both mm1 and mm2
            # This creates an epilogue node that could potentially fuse with either template
            # The chain of pointwise ops tests fusion of multiple operations
            combined = mm1 * mm2  # Multiply outputs from both matmuls
            normalized = (combined * 0.5 + 1.0).relu().tanh()

            return normalized

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        M, K, N = 64, 128, 64
        a = torch.rand([M, K], dtype=torch.bfloat16, device=GPU_TYPE)
        b = torch.rand([K, N], dtype=torch.bfloat16, device=GPU_TYPE)
        c = torch.rand([M, K], dtype=torch.bfloat16, device=GPU_TYPE)
        d = torch.rand([K, N], dtype=torch.bfloat16, device=GPU_TYPE)

        _, code = run_and_get_code(torch.compile(foo), a, b, c, d)
        FileCheck().check("tem_fused__to_copy_add_mm_mul").check(
            "to_copy_add_div_mm_mul_relu_sub_tanh_1"
        ).run(code[0])

    @config.patch(shape_padding=True)
    @config.patch(force_shape_pad=True)
    @parametrize("sizes", ((250, 245, 128), (250, 256, 128), (256, 128, 62)))
    @unittest.skipIf(
        config.triton.native_matmul,
        "generated code is different in native matmul",
    )
    def test_prologue_masked_load(self, sizes):
        M, K, N = sizes

        def foo(x, y):
            return x @ y

        x = torch.rand([250, 245], device=GPU_TYPE)
        y = torch.rand([245, 128], device=GPU_TYPE)

        # we should not attempt prologue fusion if it turns an aligned load
        # into an unaligned load
        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=3, num_allocs=3, num_deallocs=4)


def autotune_select_algorithm_wrapper_return_multi():
    def wrapper(*args, **kwargs):
        kwargs["return_multi_template"] = True
        return autotune_select_algorithm(*args, **kwargs)

    return wrapper


def benchmark_choice_override_timings(benchmark_request, *args, aten_time, triton_time):
    if isinstance(
        benchmark_request, (ExternKernelBenchmarkRequest, ExternKernelCaller)
    ):
        return aten_time
    elif isinstance(benchmark_request, (TritonBenchmarkRequest, TritonTemplateCaller)):
        return triton_time
    else:
        return float("inf")


def mock_benchmark_choice_wrapper(aten_time, triton_time):
    return functools.partial(
        benchmark_choice_override_timings, aten_time=aten_time, triton_time=triton_time
    )


@instantiate_parametrized_tests
class TestEpilogueFusionStaticAnalysis(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_fallback_to_aten": False,
                    "benchmark_epilogue_fusion": False,
                    "prologue_fusion": False,
                }
            )
        )

    @contextlib.contextmanager
    def get_common_patches(
        self,
        async_compile: bool,
        persistent_tma: bool,
        *,
        aten_time: float | None = None,
        triton_time: float | None = None,
        mock_n_spills: int | None = None,
        mock_fused_n_regs: int | None = None,
        mock_unfused_n_regs: int | None = None,
        epilogue_runtime: float | None = None,
    ):
        from torch._inductor.autotune_process import TritonBenchmarkRequest
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch._inductor.scheduler import BaseSchedulerNode

        common_patches = [
            config.patch(
                {
                    "triton.enable_persistent_tma_matmul": persistent_tma,
                    "compile_threads": 1
                    if not async_compile
                    else config.compile_threads,
                }
            ),
            mock.patch(
                "torch._inductor.kernel.mm.autotune_select_algorithm",
                autotune_select_algorithm_wrapper_return_multi(),
            ),
            fresh_cache(),
        ]

        if aten_time is not None and triton_time is not None:
            common_patches.extend(
                [
                    mock.patch.object(
                        AlgorithmSelectorCache,
                        "benchmark_choice",
                        mock_benchmark_choice_wrapper(aten_time, triton_time),
                    ),
                    mock.patch(
                        "torch._inductor.autotune_process.run_autotune_in_subprocess",
                        mock_benchmark_choice_wrapper(aten_time, triton_time),
                    ),
                ]
            )

        if mock_n_spills is not None or mock_fused_n_regs is not None:
            original_precompile = CachingAutotuner.precompile

            def mock_precompile(self, *args, **kwargs):
                original_precompile(self, *args, **kwargs)
                for launcher in self.launchers:
                    if mock_n_spills is not None:
                        launcher.n_spills = mock_n_spills
                    if mock_fused_n_regs is not None:
                        launcher.n_regs = mock_fused_n_regs

            common_patches.append(
                mock.patch.object(CachingAutotuner, "precompile", mock_precompile)
            )

        if mock_unfused_n_regs is not None:
            original_bmreq_precompile = TritonBenchmarkRequest.precompile

            def mock_bmreq_precompile(self):
                original_bmreq_precompile(self)
                self.n_regs = mock_unfused_n_regs

            common_patches.append(
                mock.patch.object(
                    TritonBenchmarkRequest, "precompile", mock_bmreq_precompile
                )
            )

        if epilogue_runtime is not None:
            common_patches.append(
                mock.patch.object(
                    BaseSchedulerNode,
                    "_get_estimated_runtime",
                    lambda node: epilogue_runtime,
                )
            )

        with contextlib.ExitStack() as stack:
            for p in common_patches:
                stack.enter_context(p)

            yield

    def _get_mm_inputs(self):
        """Common matmul inputs for epilogue fusion tests."""
        a = torch.randn(512, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(1024, 2048, device=GPU_TYPE, dtype=torch.bfloat16)
        return a, b

    def _get_mm_with_epilogue_fn(self):
        """Common function: matmul with type cast and add epilogue."""

        def f(a, b):
            return (a @ b).to(torch.float32) + 1.0

        return f

    @contextlib.contextmanager
    def _setup_mm_heuristic(self, use_async_compile: bool):
        """Setup MM heuristic with single GemmConfig and handle cleanup."""
        mm_heuristic = CUDAMMTemplateConfigHeuristic()
        original_mm_configs = mm_heuristic.mm_configs
        gemm_config = GemmConfig(64, 64, 32, 2, 4, group_m=8)
        mm_heuristic.mm_configs = [gemm_config]

        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        try:
            yield
        finally:
            mm_heuristic.mm_configs = original_mm_configs

    @unittest.skipIf(not has_triton_tma_device(), "Need TMA support in Triton")
    @skipIfXpu(msg="Bad tma config can be covered by XPU TMA")
    @parametrize("use_async_compile", (True, False))
    def test_template_bad_epilogue_fusion(self, use_async_compile: bool):
        def f(a, b):
            return (a @ b).to(torch.float32)

        a = torch.randn(512, 1152, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(1152, 7680, device=GPU_TYPE, dtype=torch.bfloat16)

        if GPU_TYPE == "xpu":
            tma_heuristic = XPUPersistentTMATemplateConfigHeuristic()
            mm_heuristic = XPUMMTemplateConfigHeuristic()
        else:
            tma_heuristic = CUDAPersistentTMATemplateConfigHeuristic()
            mm_heuristic = CUDAMMTemplateConfigHeuristic()

        # Save original configs to restore later
        original_tma_mm_configs = tma_heuristic.mm_configs
        original_mm_mm_configs = mm_heuristic.mm_configs

        good_tma_config = GemmConfig(128, 64, 64, 4, 8, group_m=8)
        if use_async_compile:
            torch._inductor.async_compile.AsyncCompile.wait_pool_ready()

        original_compile_kernel = Scheduler.compile_kernel

        for simulate_fusion_failure in [True, False]:
            torch._dynamo.reset()
            tma_heuristic.mm_configs = [good_tma_config]
            # Regular mm template gets no configs
            mm_heuristic.mm_configs = []

            def mock_compile_kernel_fail_fusion(self, nodes, hint_override=None):
                fut, mod = original_compile_kernel(self, nodes, hint_override)

                if simulate_fusion_failure and len(nodes) > 1:
                    if fut is not None:

                        def failing_result_fn():
                            raise RuntimeError

                        return torch._inductor.codecache.LambdaFuture(
                            failing_result_fn, fut.future
                        ), mod
                    else:

                        class FailingPrecompile:
                            def precompile(self):
                                raise RuntimeError

                        mod.triton_ = FailingPrecompile()
                        return None, mod

                return fut, mod

            # Different paths:
            # benchmark_epilogue_fusion: True -> always multi_template
            # causes benchmarking always
            # benchmark_epilogue_fusion: False -> TritonTemplateBuffer
            # returns speedup_from_fusion automatically as True
            # What we want: force multi template -> no benchmarking with safety
            try:
                with (
                    self.get_common_patches(use_async_compile, True),
                    mock.patch(
                        "torch._inductor.autotune_process.run_autotune_in_subprocess",
                        mock_benchmark_choice_wrapper(
                            aten_time=float("inf"), triton_time=0.1
                        ),
                    ),
                    mock.patch.object(
                        AlgorithmSelectorCache,
                        "benchmark_choice",
                        mock_benchmark_choice_wrapper(
                            aten_time=float("inf"), triton_time=0.1
                        ),
                    ),
                    mock.patch.object(
                        Scheduler,
                        "compile_kernel",
                        mock_compile_kernel_fail_fusion,
                    ),
                ):
                    compiled_f = torch.compile(f, mode="max-autotune")
                    out, code = run_and_get_code(compiled_f, a, b)

                    if not simulate_fusion_failure:
                        # Fusion should occur
                        FileCheck().check("triton_tem_fused__to_copy_mm").run(code[0])
                    else:
                        # Fusion should fail to occur, unfused kernels
                        FileCheck().check("triton_tem_fused_mm").check(
                            "triton_poi_fused__to_copy"
                        ).run(code[0])

                    if not config.cpp_wrapper:
                        torch.testing.assert_close(out, f(a, b), atol=1e-2, rtol=1e-2)
            finally:
                # Restore original configs
                tma_heuristic.mm_configs = original_tma_mm_configs
                mm_heuristic.mm_configs = original_mm_mm_configs

    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @parametrize(
        "test_case",
        [
            "spills_reject",  # High register spillage should reject fusion
            "timing_reject",  # Triton much slower than aten should reject fusion
            "accept_with_triton_faster",  # Low spills and good timing should accept fusion
            "accept_with_aten_faster",  # Fusion even if aten is slightly faster
        ],
    )
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_static_analysis(
        self, test_case: str, use_async_compile: bool
    ):
        """
        Test static analysis decisions for matmul epilogue fusions.

        Tests the scheduler logic that decides whether to fuse epilogues without
        benchmarking, based on:
        1. Register spillage (n_spills <= 8 required for fusion)
        2. Runtime comparison (epilogue_runtime + ms_min_choice > choice_timings[choice])
        """
        if test_case == "spills_reject":
            mock_n_spills = 100
            triton_time = 0.1
            aten_time = float("inf")
            expect_fusion = False
        elif test_case == "timing_reject":
            mock_n_spills = 0
            triton_time = 100.0
            aten_time = 0.001
            expect_fusion = False
        elif test_case == "accept_with_triton_faster":
            mock_n_spills = 0
            triton_time = 0.1
            aten_time = float("inf")
            expect_fusion = True
        elif test_case == "accept_with_aten_faster":
            mock_n_spills = 0
            triton_time = 0.1
            aten_time = 0.09999
            expect_fusion = True
        else:
            raise RuntimeError("Invalid test case")

        f = self._get_mm_with_epilogue_fn()
        a, b = self._get_mm_inputs()

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=mock_n_spills,
            ):
                compiled_f = torch.compile(f)
                _, code = run_and_get_code(compiled_f, a, b)

                if expect_fusion:
                    FileCheck().check("triton_tem_fused__to_copy_add_mm_0.run").run(
                        code[0]
                    )
                elif triton_time < aten_time:
                    FileCheck().check("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])
                else:
                    FileCheck().check_not("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])

    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @skipIfRocm(msg="Scheduler static analysis needs investigation on ROCm")
    @parametrize("fuse_epilogue", (True, False))
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_extra_reads(
        self, fuse_epilogue: bool, use_async_compile: bool
    ):
        """Test epilogue fusion with extra reads (bias and scale tensors)."""

        def fn(x, w, bias, scale):
            out = torch.mm(x, w)
            return out * scale + bias

        torch._dynamo.reset()

        x = torch.randn(512, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        w = torch.randn(1024, 2048, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(512, 2048, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(512, 2048, device=GPU_TYPE, dtype=torch.bfloat16)

        epilogue_runtime = 0.5
        aten_time = 1.0
        unfused_time = aten_time + epilogue_runtime
        # 2 times extra bytes / 3
        estimated_fused = epilogue_runtime * 2 / 3

        if fuse_epilogue:
            # triton + 1 read / 2 extra memory ratio * epilogue_runtime
            # < aten_time + epilogue_runtime
            triton_time = unfused_time - estimated_fused - 0.01
        else:
            triton_time = unfused_time - estimated_fused + 0.01

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                epilogue_runtime=epilogue_runtime,
            ):
                compiled_fn = torch.compile(fn)
                _, code = run_and_get_code(compiled_fn, x, w, bias, scale)

                if fuse_epilogue:
                    FileCheck().check("triton_tem_fused_add_mm_mul").run(code[0])
                else:
                    FileCheck().check_not("triton_tem").check(
                        "triton_poi_fused_add_mul"
                    ).run(code[0])

    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @skipIfRocm(msg="Scheduler static analysis needs investigation on ROCm")
    @parametrize(
        "test_case",
        [
            "occupancy_ratio_accept",  # ratio > 0.5, accept via Branch B
            "occupancy_ratio_reject",  # ratio <= 0.5, reject
        ],
    )
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_occupancy_ratio(
        self, test_case: str, use_async_compile: bool
    ):
        """
        Test occupancy ratio branch of _fuse_epilogue.

        Occupancy calculation (assuming regs_per_sm = 65536):
        blocks = regs_per_sm // (n_regs * threads_per_block)
        threads_per_block = num_warps * warp_size = 4 * 32 = 128
        """
        triton_time = 0.1
        epilogue_runtime = triton_time

        if test_case == "occupancy_ratio_accept":
            # blocks_unfused=5, blocks_fused=3, ratio=0.6 > 0.5 -> accept
            # aten slightly faster to verify fusion picks triton even when aten wins
            mock_unfused_n_regs, mock_fused_n_regs = 100, 160
            aten_time = 0.09
            expect_fusion = True
        elif test_case == "occupancy_ratio_reject":
            # blocks_unfused=8, blocks_fused=2, ratio=0.25 < 0.5 -> reject
            mock_unfused_n_regs, mock_fused_n_regs = 64, 200
            aten_time = 0.11
            expect_fusion = False
        else:
            raise RuntimeError("Invalid test case")

        f = self._get_mm_with_epilogue_fn()
        a, b = self._get_mm_inputs()

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=0,
                mock_fused_n_regs=mock_fused_n_regs,
                mock_unfused_n_regs=mock_unfused_n_regs,
                epilogue_runtime=epilogue_runtime,
            ):
                compiled_f = torch.compile(f)
                _, code = run_and_get_code(compiled_f, a, b)

                if expect_fusion:
                    FileCheck().check("triton_tem_fused__to_copy_add_mm").run(code[0])
                else:
                    FileCheck().check("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])

    @unittest.skipIf(
        not HAS_CUDA_AND_TRITON, "Scheduler static analysis only tested on cuda"
    )
    @skipIfRocm(msg="Scheduler static analysis needs investigation on ROCm")
    @parametrize(
        "test_case",
        [
            "memory_bound_accept",
            "memory_bound_reject_low_occupancy",
        ],
    )
    @parametrize("use_async_compile", (True, False))
    def test_template_epilogue_fusion_dominating_epilogue(
        self, test_case: str, use_async_compile: bool
    ):
        """
        Test memory-bound epilogue branch of _fuse_epilogue (Branch C).

        When Branches A and B fail (low occupancy), fusion can still be accepted
        if the epilogue is memory-bound (ms2 > 2*ms1) AND blocks_fused > 1.
        """
        triton_time = 0.1

        if test_case == "memory_bound_accept":
            # blocks_fused=2 > 1, ms2=0.3 > 2*ms1=0.2 -> Branch C accepts
            # aten slightly faster to verify fusion picks triton even when aten wins
            mock_unfused_n_regs, mock_fused_n_regs = 64, 256
            aten_time = 0.09
            epilogue_runtime = 0.3
            expect_fusion = True
        elif test_case == "memory_bound_reject_low_occupancy":
            # blocks_fused=1, ms2=0.3 > 2*ms1=0.2 BUT blocks_fused <= 1 -> reject
            mock_unfused_n_regs, mock_fused_n_regs = 64, 512
            aten_time = 0.11
            epilogue_runtime = 0.3
            expect_fusion = False
        else:
            raise RuntimeError("Invalid test case")

        f = self._get_mm_with_epilogue_fn()
        a, b = self._get_mm_inputs()

        with self._setup_mm_heuristic(use_async_compile):
            with self.get_common_patches(
                use_async_compile,
                False,
                aten_time=aten_time,
                triton_time=triton_time,
                mock_n_spills=0,
                mock_fused_n_regs=mock_fused_n_regs,
                mock_unfused_n_regs=mock_unfused_n_regs,
                epilogue_runtime=epilogue_runtime,
            ):
                compiled_f = torch.compile(f)
                _, code = run_and_get_code(compiled_f, a, b)

                if expect_fusion:
                    FileCheck().check("triton_tem_fused__to_copy_add_mm").run(code[0])
                else:
                    FileCheck().check("triton_tem_fused_mm").check(
                        "triton_poi_fused__to_copy"
                    ).run(code[0])


def simple_fn():
    return 42


class TestMaxAutotuneAsyncPipelined(TestMaxAutotune, TestEpilogueFusionStaticAnalysis):
    """Tests for AsyncPipelinedAutotuning path."""

    SKIP_TESTS = {
        "test_max_autotune_decompose_k": "Subgraphs not supported with async pipelining",
        "test_inf_timing": "Logs not consistent with async pipelined autotuning",
        "test_non_contiguous_input_mm_plus_mm": "Flaky on trunk",
        "test_autotune_device_guard": "Flaky on trunk",
        "test_template_bad_epilogue_fusion": "Benchmarking path is different",
        # Contiguous transform tests - SubgraphChoiceCaller not supported with async pipelining
        "test_max_autotune_contiguous_transform_mm": "Subgraphs not supported with async pipelining",
        "test_max_autotune_contiguous_transform_addmm": "Subgraphs not supported with async pipelining",
        "test_max_autotune_contiguous_transform_non_contiguous_second_matrix": "Subgraphs not supported with async pipelining",
        "test_max_autotune_contiguous_transform_with_epilogue": "Subgraphs not supported with async pipelining",
        # XPU specific skips due to lack of multiprocess tensor reduction support (issue #170636)
        "test_max_autotune_addmm_persistent_tma": "No XPU implementation for multiprocess tensor reduction",
        "test_max_autotune_regular_mm_persistent_tma": "No XPU implementation for multiprocess tensor reduction",
        "test_max_autotune_regular_mm_persistent_tma_strided": "No XPU implementation for multiprocess tensor reduction",
        "test_max_autotune_addmm_tma_dynamic_outer_dim": "No XPU implementation for multiprocess tensor reduction",
        "test_max_autotune_regular_mm_tma_dynamic_outer_dim": "No XPU implementation for multiprocess tensor reduction",
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._async_config = config.patch(
            {
                "pipeline_max_autotune_gemm": True,
                "benchmark_epilogue_fusion": False,
                "test_configs.max_mm_configs": 1,
            }
        )
        cls._async_config.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._async_config.__exit__(None, None, None)
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        test_name = self._testMethodName
        for skip_test_name in self.SKIP_TESTS:
            if skip_test_name in test_name or TEST_XPU:
                self.skipTest(self.SKIP_TESTS[skip_test_name])

    def tearDown(self):
        super().tearDown()
        AutotuneProcessPool.shutdown_instance()
        # Clear the AsyncAutotuner cache to prevent test pollution
        AsyncAutotuner.choice_hash_to_future.clear()

    @config.patch(max_autotune_gemm=True)
    def test_async_autotuner_cache_same_inputs(self):
        M, K, N = 128, 64, 256
        M2, K2, N2 = 256, 128, 64

        def three_matmuls(a1, b1, a2, b2, a3, b3):
            return torch.mm(a1, b1), torch.mm(a2, b2), torch.mm(a3, b3)

        # Same shapes for first two matmuls
        a1 = torch.randn(M, K, device=GPU_TYPE, dtype=torch.bfloat16)
        b1 = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)
        a2 = torch.randn(M, K, device=GPU_TYPE, dtype=torch.bfloat16)
        b2 = torch.randn(K, N, device=GPU_TYPE, dtype=torch.bfloat16)

        # Different shapes for third matmul
        a3 = torch.randn(M2, K2, device=GPU_TYPE, dtype=torch.bfloat16)
        b3 = torch.randn(K2, N2, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled_fn = torch.compile(three_matmuls)
        result = compiled_fn(a1, b1, a2, b2, a3, b3)

        # Verify correctness
        expected = three_matmuls(a1, b1, a2, b2, a3, b3)
        for r, e in zip(result, expected):
            torch.testing.assert_close(r, e, atol=1e-2, rtol=1e-2)

        # With max_mm_configs=1, we get 2 configs total (1 per unique shape)
        # First two matmuls share the same shape, third has different shape
        # 1 aten, 1 triton config
        cache_size = len(AsyncAutotuner.choice_hash_to_future)
        self.assertEqual(
            cache_size, 4, "Cache should have 2 entries (one per unique input shape)"
        )

    @patch(
        "torch._inductor.autotune_process.AUTOTUNE_POOL_INACTIVITY_TIMEOUT",
        2,
    )
    def test_autotune_process_pool_inactivity_shutdown(self):
        AutotuneProcessPool.shutdown_instance()
        AutotuneProcessPool._shutdown_for_inactivity = False

        pool_instance = AutotuneProcessPool.get_instance()
        pool_instance.warm_up()

        future = pool_instance.submit(simple_fn)
        result = future.result()
        self.assertEqual(result, 42)
        self.assertIsNotNone(pool_instance._pool)

        time.sleep(5)

        self.assertIsNone(pool_instance._pool)
        self.assertIsNone(pool_instance._timer)
        self.assertTrue(AutotuneProcessPool._shutdown_for_inactivity)

    @patch(
        "torch._inductor.autotune_process.AUTOTUNE_POOL_INACTIVITY_TIMEOUT",
        2,
    )
    @config.patch(max_autotune=True)
    def test_compilation_after_inactivity(self):
        """Test that compilation after pool inactivity shutdown uses synchronous path."""

        # Reset state
        AutotuneProcessPool.shutdown_instance()
        AutotuneProcessPool._shutdown_for_inactivity = False
        AsyncAutotuner.choice_hash_to_future.clear()
        torch._dynamo.reset()

        # First compilation - should use pipelined path
        self.assertTrue(use_pipelined_autotuning())

        def matmul_fn(a, b):
            return torch.mm(a, b)

        a1 = torch.randn(64, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        b1 = torch.randn(32, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        a2 = torch.randn(128, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        b2 = torch.randn(64, 128, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled_fn = torch.compile(matmul_fn)
        compiled_fn(a1, b1)

        # Verify pipelined path was used (cache should have entries)
        cache_entries_after_first = len(AsyncAutotuner.choice_hash_to_future)
        self.assertGreater(cache_entries_after_first, 0)

        # Wait for inactivity shutdown
        time.sleep(5)

        self.assertTrue(AutotuneProcessPool._shutdown_for_inactivity)
        self.assertFalse(use_pipelined_autotuning())

        AsyncAutotuner.choice_hash_to_future.clear()
        torch._dynamo.reset()

        compiled_fn2 = torch.compile(matmul_fn)
        compiled_fn2(a2, b2)

        cache_entries_after_second = len(AsyncAutotuner.choice_hash_to_future)
        self.assertEqual(cache_entries_after_second, 0)

    @config.patch(max_autotune_gemm=True)
    def test_triton_error_precompilation_and_autotuning(self):
        """
        Test error handling when do_autotuning throws NoValidChoicesError
        for Triton choices. The fallback to extern kernels should still work.
        """

        def mock_do_autotuning(*args, **kwargs):
            raise NoValidChoicesError("Simulated: all Triton choices failed")

        a = torch.randn(64, 32, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(32, 64, device=GPU_TYPE, dtype=torch.bfloat16)

        def mm_func(a, b, epilogue):
            if epilogue:
                return torch.mm(a, b) + 1.0
            else:
                return torch.mm(a, b)

        def test_aten_chosen():
            for epilogue in (True, False):
                torch._dynamo.reset()
                compiled_fn = torch.compile(mm_func)
                out, code = run_and_get_code(compiled_fn, a, b, epilogue)
                FileCheck().check_not("triton_tem").run(code[0])

        with mock.patch.object(
            AlgorithmSelectorCache, "do_autotuning", mock_do_autotuning
        ):
            test_aten_chosen()

        original_start = AsyncAutotuner.start
        bmreq = _TestBenchmarkRequest(
            exc=RuntimeError("Simulated benchmark failure in subprocess")
        )
        bmreq.module_cache_key = ""

        def mock_start(choices, inputs_key):
            for choice in choices:
                if isinstance(choice, TritonTemplateCaller):
                    choice.bmreq = bmreq
            return original_start(choices, inputs_key)

        with mock.patch.object(AsyncAutotuner, "start", mock_start):
            test_aten_chosen()


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
