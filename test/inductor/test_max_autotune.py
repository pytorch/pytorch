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
import unittest
from typing import Callable, Optional
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch import multiprocessing as mp, nn
from torch._dynamo import reset
from torch._dynamo.exc import BackendCompilerFailed
from torch._dynamo.testing import rand_strided, reset_rng_state
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.autotune_process import (
    _TestBenchmarkRequest,
    CUDA_VISIBLE_DEVICES,
    TuningProcess,
    TuningProcessPool,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import Buffer, ChoiceCaller, FixedLayout
from torch._inductor.kernel.mm_plus_mm import aten_mm_plus_mm
from torch._inductor.select_algorithm import (
    AlgorithmSelectorCache,
    TritonTemplate,
    TritonTemplateCaller,
)
from torch._inductor.template_heuristics import CUDAConfigHeuristic, GemmConfig
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.logging_utils import multiple_logs_to_string
from torch.utils._triton import has_triton_tma_device


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
    HAS_CUDA,
    HAS_GPU,
)


torch.set_float32_matmul_precision("high")
if HAS_CUDA:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")


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
    @parametrize("dynamic", (False, True))
    def test_max_autotune_mm_plus_mm_zero_size_input(self, dynamic):
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

        with config.patch({"max_autotune": True}):
            torch.compile(mm_plus_mm, dynamic=dynamic)(a, b, c, d)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma(
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

        M, N, K = 21, 31, 11
        a = torch.randn(*((K, M) if a_transposed else (M, K))).to(torch.float16).cuda()
        b = torch.randn(*((N, K) if b_transposed else (K, N))).to(torch.float16).cuda()

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual = torch.compile(mm, dynamic=dynamic)(a, b)
            c_expected = mm(a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("dynamic", (False, True))
    def test_max_autotune_regular_mm_persistent_tma_illegal_alignment(self, dynamic):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).cuda()
        b = torch.randn(K, N).to(torch.float16).cuda()

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
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
    def test_max_autotune_regular_mm_tma_dynamic_outer_dim(self):
        def mm(a, b):
            return torch.mm(a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).cuda()
        b = torch.randn(K, N).to(torch.float16).cuda()

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
    @parametrize("a_transposed", (False, True))
    @parametrize("b_transposed", (False, True))
    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_persistent_tma(
        self,
        a_transposed: bool,
        b_transposed: bool,
        dynamic: bool,
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
        a = torch.randn(*((K, M) if a_transposed else (M, K))).to(torch.float16).cuda()
        b = torch.randn(*((N, K) if b_transposed else (K, N))).to(torch.float16).cuda()
        x = torch.randn(N).to(torch.float16).cuda()

        with config.patch(
            {
                "max_autotune": True,
                "triton.enable_persistent_tma_matmul": "1",
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual = torch.compile(addmm, dynamic=dynamic)(x, a, b)
            c_expected = addmm(x, a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
    )
    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm_persistent_tma_illegal_alignment(self, dynamic):
        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        M, N, K = 21, 31, 11
        a = torch.randn(M, K).to(torch.float16).cuda()
        b = torch.randn(K, N).to(torch.float16).cuda()
        x = torch.randn(N).to(torch.float16).cuda()

        with (
            self.assertRaises(BackendCompilerFailed) as context,
            config.patch(
                {
                    "max_autotune": True,
                    "triton.enable_persistent_tma_matmul": "1",
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
        a = torch.randn(M, K).to(torch.float16).cuda()
        b = torch.randn(K, N).to(torch.float16).cuda()
        x = torch.randn(N).to(torch.float16).cuda()

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
                "test_configs.autotune_choice_name_regex": "mm_persistent_tma",
            }
        ):
            c_actual = torch.compile(addmm)(x, a, b)
            c_expected = addmm(x, a, b)

        torch.testing.assert_close(c_actual, c_expected, atol=1e-2, rtol=1e-2)

    @fresh_cache()
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support sm carveout")
    @unittest.skipIf(IS_WINDOWS, "Windows doesn't support persistent TMA")
    @unittest.skipIf(
        not has_triton_tma_device(), "Need device-side TMA support in Triton"
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
        a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
        b = (
            torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
            .transpose(0, 1)
            .contiguous()
            .transpose(0, 1)
        )
        scale_a = torch.tensor(1, dtype=torch.float32, device="cuda")
        scale_b = torch.tensor(1, dtype=torch.float32, device="cuda")

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
                kernel_events = [
                    {
                        "grid": evt.get("args", {}).get("grid", []),
                        "grid_size": math.prod(evt.get("args", {}).get("grid", [])),
                    }
                    for evt in json.load(open(f.name))["traceEvents"]
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

    def test_autotune_conv1x1(self):
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
            {"max_autotune": True, "max_autotune_gemm_backends": "TRITON"}
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
    def test_empty_conv_input(self, kernel_size=3):
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

        opt_f = torch.compile(f)
        ref = f(x, weight)
        act = opt_f(x, weight)
        self.assertTrue(torch.allclose(ref, act, atol=4 * 1e-3, rtol=4 * 1e-3))

    @config.patch(max_autotune=True)
    def test_empty_conv_input_with_1x1_kernel(self):
        self.test_empty_conv_input(kernel_size=1)

    @config.patch(max_autotune_gemm_backends="TRITON")
    def test_baddmm(self):
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

        m_c = torch.compile(mode="max-autotune")(mod)
        out, code = run_and_get_code(m_c, x)
        self.assertEqual(out, mod(x), atol=2e-3, rtol=1e-3)

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
        count = 2 if using_triton_mm else 1
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

    @skipIfXpu(
        msg="The fusion not happened because it do not speedup on XPU, see issue #146568"
    )
    @config.patch(
        {
            "max_autotune_gemm_backends": "TRITON",
            "benchmark_epilogue_fusion": False,
        }
    )
    def test_cat_max_autotune_triton(self):
        self._test_cat_max_autotune_impl(using_triton_mm=True)

    def test_conv_cat(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )

            def forward(self, x):
                x = self.conv(x)
                return torch.cat((x, x + 1))

        with torch.no_grad():
            m = ToyModel().to(device=GPU_TYPE)
            input_tensor = torch.randn(32, 3, 64, 64).to(device=GPU_TYPE)

            # convolution is not currently plannable
            m = torch.compile(m, mode="max-autotune-no-cudagraphs")
            out, code = run_and_get_code(m, input_tensor)
            self.assertEqual(out, m(input_tensor))

            if not TEST_WITH_ROCM:
                FileCheck().check("def triton_poi_fused_add_cat_").run(code[0])

    def test_conv3d(self):
        fn = torch.nn.functional.conv3d
        image = torch.randn([1, 3, 8, 16, 32])
        filt = torch.randn([3, 3, 7, 7, 7])

        with config.patch({"max_autotune": True}):
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

    # TODO: fix accuracy failure of the triton template on XPU.
    # and enable this test case.
    @skipIfXpu
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
        torch.testing.assert_close(act, ref, atol=1e-2, rtol=1e-2)

    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="",
    )
    def test_no_valid_choices(self):
        a = torch.zeros([2, 2], device=GPU_TYPE)
        b = torch.zeros([2, 2], device=GPU_TYPE)
        with self.assertRaises(BackendCompilerFailed) as context:
            torch.compile(lambda a, b: a.matmul(b))(a, b)
        self.assertIn("NoValidChoicesError", str(context.exception))

    @parametrize("multi_template", (True, False))
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_inf_timing(self, multi_template):
        from unittest.mock import patch

        lookup = AlgorithmSelectorCache.lookup

        def mock_lookup(self, *args, **kwargs):
            timings = lookup(self, *args, **kwargs)
            return {choice: float("inf") for choice in timings.keys()}

        a = torch.zeros([16, 16], device=GPU_TYPE)
        b = torch.zeros([16, 16], device=GPU_TYPE)
        with (
            patch.object(AlgorithmSelectorCache, "lookup", mock_lookup),
            config.patch(benchmark_epilogue_fusion=multi_template),
        ):
            with self.assertRaises(BackendCompilerFailed) as context:
                torch.compile(lambda a, b: a.matmul(b))(a, b)
            self.assertIn("NoValidChoicesError", str(context.exception))

    @unittest.skipIf(
        not torch.cuda.is_available()
        or torch.cuda.get_device_properties().total_memory < 2e10,
        "Only if the GPU has at least 20GB memory to be safe",
    )
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

        x = torch.randn(B * T, C, requires_grad=True).cuda().bfloat16()
        x.retain_grad()
        y = torch.randint(0, V, (B * T,)).cuda()

        import torch._inductor.utils as inductor_utils

        with unittest.mock.patch.object(inductor_utils, "is_big_gpu", mock_is_big_gpu):
            opt_f = torch.compile(f)

            expect = (f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
            actual = (opt_f(x, y), x.grad, linear.weight.grad, linear.bias.grad)
            assert same(expect, actual, tol=1e-2), f"ref:\n{expect}\nact:\n{actual}"

    @skipIfXpu
    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @parametrize("dynamic", (True, False))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("sizes", ((32, 32, 32768), (64, 128, 200000), (64, 64, 177147)))
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k(self, sizes, dtype, dynamic):
        fp16_red_setting = (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        )
        bf16_red_setting = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        M, N, K = sizes

        a = torch.randn(M, K, dtype=dtype, device="cuda", requires_grad=True)
        b = torch.randn(K, N, dtype=dtype, device="cuda", requires_grad=True)

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

        if dynamic or torch.version.hip:
            FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                "decompose_k"
            ).run(code[0])
        else:
            FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                "triton_.*_fused_0.run"
            ).check("decompose_k").run(code[0])
            check_divisors(code)
            torch.testing.assert_close(out, a @ b, atol=1e-2, rtol=1e-2)

        # Test adding epilogue also equivalent to eager
        compiled_func = torch.compile(lambda a, b: (a @ b).relu(), dynamic=dynamic)
        out, code = run_and_get_code(compiled_func, a, b)
        if dynamic or torch.version.hip:
            FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                "decompose_k"
            ).run(code[0])
        else:
            FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                "triton_.*_fused_0.run"
            ).check("decompose_k").run(code[0])
            check_divisors(code)
            torch.testing.assert_close(
                compiled_func(a, b), (a @ b).relu(), atol=1e-2, rtol=1e-2
            )

        # Test adding reinterpret view before subgraph
        a = a.transpose(0, 1)
        compiled_func = torch.compile(
            lambda a, b: (a.transpose(0, 1) @ b).relu(), dynamic=dynamic
        )
        out, code = run_and_get_code(compiled_func, a, b)

        # DecomposeK is not enabled for AMD yet
        if dynamic or torch.version.hip:
            FileCheck().check_not("extern_kernels.bmm_dtype").check_not(
                "decompose_k"
            ).run(code[0])
        else:
            FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                "triton_.*_fused_0.run"
            ).check("decompose_k").run(code[0])
            check_divisors(code)
            torch.testing.assert_close(
                compiled_func(a, b),
                (a.transpose(0, 1) @ b).relu(),
                atol=1e-2,
                rtol=1e-2,
            )

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            fp16_red_setting
        )
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            bf16_red_setting
        )

    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "decompose_k not supported on ROCm")
    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_dynamic_input(self):
        def f(a, b):
            a_in = torch.stack((a, a), dim=0)
            return (a_in @ b).relu()

        a = torch.randn(
            32, 32768, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )
        b = torch.randn(
            32768, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        torch._dynamo.reset()
        torch._dynamo.maybe_mark_dynamic(a, 0)
        compiled_func = torch.compile(f)

        with mock.patch(
            "torch._inductor.kernel.mm.use_decompose_k_choice"
        ) as decomp_mock:
            decomp_mock.return_value = True

            out, code = run_and_get_code(compiled_func, a, b)
            FileCheck().check("extern_kernels.bmm_dtype").check_regex(
                "triton_.*_fused_0.run"
            ).check("decompose_k").check_regex(r"s[0-9]+ = s[0-9]+").check_regex(
                r"2\*s[0-9]+"
            ).check_regex("s[0-9]+ = 32").run(code[0])
            torch.testing.assert_close(
                out,
                f(a, b),
                atol=1e-2,
                rtol=1e-2,
            )

    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "decompose_k not supported on ROCm")
    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_dynamic_input_bwd(self):
        def f(a, b):
            # 256 * s0
            a_in = torch.cat([a for _ in range(256)], dim=0)
            return (a_in @ b).relu().sum()

        a = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        b = torch.randn(
            64, 32768, dtype=torch.bfloat16, device="cuda", requires_grad=True
        )

        torch._dynamo.reset()
        torch._dynamo.maybe_mark_dynamic(a, 0)
        compiled_func = torch.compile(f)
        res = compiled_func(a, b)
        res.backward()

        with mock.patch(
            "torch._inductor.kernel.mm.use_decompose_k_choice"
        ) as decomp_mock:
            decomp_mock.return_value = True

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

    @skipIfXpu
    @unittest.skipIf(TEST_WITH_ROCM, "decompose_k not supported on ROCm")
    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
    )
    @config.patch(
        max_autotune=True,
        max_autotune_gemm_backends="TRITON",
    )
    def test_max_autotune_decompose_k_output_stride(self):
        def f(a, b):
            a = a.transpose(0, 1)
            return a @ b

        a = torch.randn((32768, 256), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((32768, 1152), device="cuda", dtype=torch.bfloat16)

        b = b[:, :1096]

        # Force only decomposeK choice
        with (
            mock.patch(
                "torch._inductor.kernel.mm.V.choices.get_base_mm_configs"
            ) as base_mm_mock,
            mock.patch(
                "torch._inductor.kernel.mm.use_decompose_k_choice"
            ) as decompose_mock,
        ):
            mm_configs_mock = MagicMock()
            mm_configs_mock.return_value = []
            base_mm_mock.return_value = mm_configs_mock
            decompose_mock.return_value = True
            compiled_f = torch.compile(f)
            out, code = run_and_get_code(compiled_f, a, b)

            # Output stride equal to original gm output stride
            # If output stride is not correctly checked, this will be (1152, 1) which can cause nans
            self.assertEqual(out.stride(), (1096, 1))

            FileCheck().check_not("extern_kernels.bmm_dtype").check(
                "decompose_k"
            ).check(" empty_strided_cuda((256, 1096), (1096, 1), torch.bfloat16)").run(
                code[0]
            )

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
        self.assertEqual(generate_and_load_args, 17)

    @fresh_cache()
    @config.patch(
        {
            "max_autotune": True,
            "test_configs.max_mm_configs": 4,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
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
                        'num_consumer_groups':0,'num_buffers_warp_spec':0,'epilogue_fn_hash':'identity',
                        'kwargs':{'EVEN_K':False,'ALLOW_TF32':True,'USE_FAST_ACCUM':False,'ACC_TYPE':'tl.float32',
                        'BLOCK_M':16,'BLOCK_N':32,'BLOCK_K':16,'GROUP_M':8},'hint_override':None}"""

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
                        "[[s77,s17],[s17,1],torch.float32,device(type='cuda',index=0),0]",
                        "[[s17,s94],[s94,1],torch.float32,device(type='cuda',index=0),0]"],
                    'num_stages':1,'num_warps':2,'prefix_args':0,'suffix_args':0,'call_sizes':[s77,s94],
                    'layout':"[[s77,s94],[s94,1],torch.float32,device(type='cuda',index=0),0]",'num_consumer_groups':0,
                    'num_buffers_warp_spec':0,'epilogue_fn_hash':'identity','kwargs':{'EVEN_K':False,'ALLOW_TF32':True,
                    'USE_FAST_ACCUM':False,'ACC_TYPE':'tl.float32','BLOCK_M':16,'BLOCK_N':32,'BLOCK_K':16,'GROUP_M':8},'hint_override':None}"""
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
            for i in range(0, 10):
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
    @unittest.skipIf(TEST_WITH_ROCM, "decompose_k not supported on ROCm")
    @unittest.skipIf(
        config.cpp_wrapper, "decompose_k not supported for cpp_wrapper yet"
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
            a = torch.randn(M, K, dtype=torch.float16, device="cuda")
            b = torch.randn(K, N, dtype=torch.float16, device="cuda")

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

    @skipIfXpu
    @unittest.skipIf(
        TEST_WITH_ROCM, "exhaustive currently only thoroughly tested on NVIDIA"
    )
    @config.patch(max_autotune=True, max_autotune_gemm_search_space="EXHAUSTIVE")
    def test_max_autotune_exhaustive(self):
        def f(a, b):
            return a @ b

        M, N, K = (1024, 1024, 1024)

        a = torch.randn(M, K, dtype=torch.float16, device="cuda", requires_grad=True)
        b = torch.randn(K, N, dtype=torch.float16, device="cuda", requires_grad=True)

        with mock.patch(
            "torch._inductor.kernel.mm.V.choices.get_config_heuristics"
        ) as config_mock:
            config_heuristics = CUDAConfigHeuristic()

            # Traditionally, this would be set of all possible configs
            # We mock out the code path for the sake of the unit test
            config_heuristics.exhaustive_configs = [GemmConfig(32, 32, 32, 1, 8, 8)]
            config_mock.return_value = config_heuristics

            from torch._dynamo.utils import counters

            compiled_func = torch.compile(f)
            compiled_func(a, b)

            # Only benchmarks 2 choices, aten and the exhaustive triton config
            # Counter can be InductorBenchmarker or TritonBenchmarker
            for counter in counters["inductor"]:
                if "benchmark_gpu" in counter:
                    self.assertEqual(counters["inductor"][counter], 2)


class TestMaxAutotunePrecompile(TestCase):
    def test_precompilation_threads(self):
        import threading
        from typing import Any
        from unittest.mock import Mock, patch

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
                assert fake_choice.thread_id is not None, (
                    "Expected all ChoiceCaller's precompile method to have been called"
                )
                assert fake_choice.thread_id != main_thread_id, (
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

    # XPU have not support multiprocessing reduction in torch/multiprocessing/reductions.py
    @skipIfXpu
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

    # XPU have not support multiprocessing reduction in torch/multiprocessing/reductions.py
    @skipIfXpu
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

    @parametrize("dynamic", (False, True))
    def test_max_autotune_addmm(self, dynamic=False):
        """
        Make sure autotuning addmm in sub processes work without crashes.
        """

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        def addmm(x, a, b):
            return torch.addmm(x, a, b)

        x = torch.randn(100).to(GPU_TYPE)
        a = torch.randn(100, 10).to(GPU_TYPE)
        b = torch.randn(10, 100).to(GPU_TYPE)
        with config.patch({"max_autotune": True, "autotune_in_subproc": True}):
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
        from unittest.mock import patch

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

    # XPU have to enable XPU_VISIBLE_DEVICES to control devices visibility.
    @skipIfXpu
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

    @unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FP8,
        "FP8 is only supported on H100+, SM 8.9 and MI300+ devices",
    )
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
    def test_multiple_fusions(self, sizes):
        M, K, N = sizes

        def foo(x, y):
            return ((x - 1.1) @ (y + 1.1)) * 1.1

        x = torch.rand([M, K], dtype=torch.float, device=GPU_TYPE)
        y = torch.rand([K, N], dtype=torch.float, device=GPU_TYPE)

        out, code = run_and_get_code(torch.compile(foo), x, y)
        self.assertEqual(out, foo(x, y), atol=0.05, rtol=0.05)
        self.check_code(code[0], num_kernels=1, num_allocs=1, num_deallocs=2)

        # check that we do not CSE any variables between prologues, epilogues
        FileCheck().check("def triton").check_count("= 1.1", 3, exactly=True).check(
            "tl.store"
        ).run(code[0])

    @config.patch(
        {
            "max_autotune_gemm_backends": "Triton",
            "benchmark_epilogue_fusion": True,
            "max_epilogue_benchmarked_choices": 3,
        }
    )
    @skipIfXpu(
        msg="The fusion not happened because it do not speedup on XPU, see issue #146568"
    )
    def test_pending_fusions_multiple(self):
        def multi_use(x, y):
            return (x @ x.T) * (y @ y.T)

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
    @skipIfXpu(
        msg="The fusion not happened because it do not speedup on XPU, see issue #146568"
    )
    def test_pending_fusion_pro_and_epi(self):
        def test_multiple_fusions(x):
            y = x.to(torch.float)
            return (y @ y).relu()

        x = torch.rand([128, 128], dtype=torch.float16, device=GPU_TYPE)
        out, code = run_and_get_code(torch.compile(test_multiple_fusions), x)
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(), 1, exactly=True
        ).run(code[0])
        self.assertEqual(out, test_multiple_fusions(x), atol=0.05, rtol=0.05)

    @parametrize("sizes", ((64, 128, 256), (128, 128, 128), (63, 120, 250)))
    def test_multiple_inputs(self, sizes):
        M, K, N = sizes

        def foo(x, y, z):
            return (x + y).to(torch.float) @ z

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
    def test_prologue_multiple_nodes(self, sizes):
        M, K, N = sizes

        def foo(x, y):
            return ((((x * 2) - 1) / 2) @ (y * 4)) * 3.0

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
    def test_prologue_read_into_both_inputs(self, benchmark_fusion):
        M = K = 256

        # not supported today. it could be, but typically the pointwise nodes would get
        # inlined into separate nodes.

        def foo(x):
            y = (x + 1) * 2
            return y @ (y - 2)

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

    # XPU have not enabled pad_mm in fx_passes, so there is always one kernel.
    @skipIfXpu
    @config.patch(shape_padding=True)
    @config.patch(force_shape_pad=True)
    @parametrize("sizes", ((250, 245, 128), (250, 256, 128), (256, 128, 62)))
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


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
