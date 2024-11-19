# Owner(s): ["module: inductor"]
import math
import os
import sys

import torch
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.test_operators import realize
from torch._inductor.utils import fresh_inductor_cache, is_big_gpu, run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    skip_if_async_compile,
    slowTest,
    TEST_WITH_ASAN,
)
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

import contextlib
import unittest

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model,
    check_model_cuda,
    copy_tests,
)
from torch._inductor import config
from torch._inductor.scheduler import Scheduler


class TestCase(InductorTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "benchmark_kernel": True,
                    "benchmark_fusion": True,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()


class BenchmarkFusionTestTemplate:
    def test_softmax(self):
        def f(x):
            return torch.nn.functional.softmax(x, dim=-1)

        self.common(f, (torch.rand(2, 8192),))

    @slowTest
    def test_resnet18(self):
        import torchvision

        model = torchvision.models.resnet18()
        model.eval()
        batch_size = 16
        inputs = (torch.randn((batch_size, 3, 224, 224)),)
        self.common(model, inputs, atol=1e-2, rtol=1e-2)

    def test_register_spills(self):
        """
        The test can potentially trigger register spills
        """
        old_benchmark_fn = Scheduler.benchmark_fused_nodes

        def new_benchmark_fn(scheduler, nodes):
            """
            We override Scheduler.benchmark_fused_nodes to return latency 1.0
            if there are no register spills. Without this, we may not able to
            test the code path handling register spilling because before register
            start spilling, the related fusion may have already been skipped
            due to longer lantency.
            """
            ms, path = old_benchmark_fn(scheduler, nodes)
            if not math.isinf(ms):
                ms = 1.0
            return ms, path

        # Disable dynamic_scale_rblock to make it easier to trigger register
        # spilling.
        with unittest.mock.patch.object(
            Scheduler, "benchmark_fused_nodes", new_benchmark_fn
        ), config.patch("dynamic_scale_rblock", False):
            S = 512

            def f(*inputs):
                inputs = list(inputs)
                outputs = []
                out = torch.zeros(S, device=self.device)
                for x in inputs:
                    x = x * 2
                    x = x + 1
                    x = x.sum(dim=-1)
                    outputs.append(x)
                    out = out + x
                return outputs, out

            N = int(os.environ.get("NINP", "30"))
            inputs = [torch.randn(S, 2560, device=self.device) for _ in range(N)]
            opt_f = torch.compile(f)
            opt_f(*inputs)

    def test_foreach_kernel(self):
        """
        Benchmark fusion should skip benchmarking kernels involves foreach kernel
        for now. Without the skipping logic, `codegen_node_schedule` may fail.
        """
        a = torch.randn(1024, 256, device=self.device)
        b = torch.randn(1024, 512, device=self.device)

        def f(a, b):
            a, b = torch._foreach_abs([a, b])
            return a + 1, b + 2

        self.common(f, (a, b))

    @skip_if_async_compile
    @torch._inductor.config.patch(max_autotune_gemm_backends="TRITON")
    def test_avoid_register_spilling(self):
        if self.device != "cuda":
            raise unittest.SkipTest("CUDA only")

        from torch.nn.functional import gelu

        def foo(m, inp):
            curr = m(inp)
            tmps = []
            for _ in range(4):
                curr = gelu(curr)
                for t in tmps:
                    curr = curr + t
                tmps.append(curr)

            return curr

        m = torch.nn.Linear(2048, 2048, bias=True).half().cuda()
        inp = torch.rand([2048, 2048]).half().cuda()

        with torch.no_grad():
            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            _, out_code = run_and_get_code(foo_c, m, inp)

            # occasionally, CI will make this one kernel. just skip in this case
            if not out_code[0].count("def triton_") == 2:
                return

            # should be multiple triton invocations
            FileCheck().check("async_compile.wait").check_count(
                ".run", 2, exactly=True
            ).run(out_code[0])

        with config.patch(
            {"benchmark_fusion": False, "epilogue_fusion": False}
        ), torch.no_grad():
            torch._dynamo.reset()

            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

            _, out_code2 = run_and_get_code(foo_c, m, inp)

        for c in out_code[0], out_code2[0]:
            FileCheck().check("async_compile.wait").check("DeviceGuard").check_count(
                "empty_strided_cuda", 1, exactly=True
            ).check_regex("buf[0-9]* = buf[0-9]*; del buf[0-9]*").check("return").run(c)

    def test_tield_kernel_fusion(self):
        def f(x):
            y = realize(x + x.t())
            return y + 1

        x = torch.randn(1024, 1024, device=self.device)
        self.common(f, (x,))


if HAS_CUDA and not TEST_WITH_ASAN:

    class BenchmarkFusionCudaTest(TestCase):
        common = check_model_cuda
        device = "cuda"

    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCudaTest, "cuda")

    class BenchmarkingTest(TestCase):
        @unittest.skipIf(
            torch.cuda.device_count() < 2, "The test need at least 2 devices"
        )
        def test_benchmark_on_non_zero_device(self):
            hit_count = 0
            with torch.cuda.device("cuda:0"):

                @torch.compile
                def relu(x):
                    return realize(x.relu()) + x

                x = torch.randn(int(16e6), device="cuda:1")

                orig_benchmark_fused_nodes = TritonScheduling.benchmark_fused_nodes

                def mock_benchmark_fused_nodes(*args, **kwargs):
                    nonlocal hit_count
                    hit_count += 1
                    ms, path = orig_benchmark_fused_nodes(*args, **kwargs)
                    self.assertTrue(ms > 0)
                    return ms, path

                with unittest.mock.patch.object(
                    TritonScheduling,
                    "benchmark_fused_nodes",
                    mock_benchmark_fused_nodes,
                ):
                    relu(x)
                self.assertTrue(hit_count > 0)

    class BenchmarkMultiTemplateFusionCudaTest(InductorTestCase):
        @classmethod
        def setUpClass(cls):
            super().setUpClass()
            cls._stack = contextlib.ExitStack()
            cls._stack.enter_context(
                config.patch(
                    {
                        "benchmark_kernel": True,
                        "benchmark_fusion": True,
                        "benchmark_epilogue_fusion": True,
                    }
                )
            )

        @classmethod
        def tearDownClass(cls):
            cls._stack.close()
            super().tearDownClass()

        def setUp(self):
            super().setUp()
            if not is_big_gpu(0):
                return self.skipTest("Need a big GPU to run max_autotune=True")

        def _equivalent_output_code_impl(self, size, first_dim=None, activation=True):
            def foo(m, inp):
                a = m(inp)
                if activation:
                    return torch.nn.functional.relu(a)
                return a

            foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)
            first_dim = first_dim if first_dim is not None else size

            m = torch.nn.Linear(size, size, bias=True).half().cuda()
            inp = torch.rand([first_dim, size]).half().cuda()

            with torch.no_grad():
                res, code = run_and_get_code(foo_c, m, inp)

            torch._dynamo.reset()
            with unittest.mock.patch.object(
                torch._inductor.config, "benchmark_epilogue_fusion", False
            ):
                foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)
                with torch.no_grad():
                    res2, code2 = run_and_get_code(foo_c, m, inp)

            self.assertEqual(res, res2, atol=1e-4, rtol=1.1)
            return code, code2

        @fresh_inductor_cache()
        @torch._inductor.config.patch(max_autotune_gemm_backends="TRITON")
        def test_equivalent_template_code(self):
            code, code2 = self._equivalent_output_code_impl(256)
            for out_code in [code, code2]:
                FileCheck().check("def call").check_count(
                    "empty_strided_cuda", 1, exactly=True
                ).check("triton_tem_fused_addmm_relu_0.run").check_count(
                    "del", 3, exactly=True
                ).check(
                    "return"
                ).run(
                    out_code[0]
                )

        @skip_if_async_compile
        @fresh_inductor_cache()
        @torch._inductor.config.patch(max_autotune_gemm_backends="ATEN")
        def test_equivalent_extern_code(self):
            torch._dynamo.reset()

            code, code2 = self._equivalent_output_code_impl(512, 1, False)

            for out_code in [code, code2]:
                FileCheck().check("def call").check_count(
                    "empty_strided_cuda", 1, exactly=True
                ).check("extern_kernels.").check_count("del", 3, exactly=True).check(
                    "return"
                ).run(
                    out_code[0]
                )

        def test_changed_layout(self):
            # cat addmm planning will change layout - make sure propagated
            def fn(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
                return torch.cat(
                    [
                        torch.addmm(a, b, c),
                        torch.addmm(b, c, a),
                    ],
                    1,
                )

            args = [
                torch.randn(4, 4, device="cuda"),
                torch.randn(4, 4, device="cuda"),
                torch.randn(4, 4, device="cuda"),
            ]

            expected = fn(*args)
            actual = torch.compile(fn, mode="max-autotune")(*args)
            self.assertEqual(expected, actual)

            torch._dynamo.reset()


if HAS_CPU and not torch.backends.mps.is_available():

    class BenchmarkFusionCpuTest(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(BenchmarkFusionTestTemplate, BenchmarkFusionCpuTest, "cpu")

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_CUDA:
        run_tests()
