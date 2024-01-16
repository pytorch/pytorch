# Owner(s): ["module: inductor"]
import contextlib
import subprocess
import sys
from unittest.mock import patch

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._inductor import config
from torch._inductor.codecache import PyCodeCache
from torch._inductor.utils import fresh_inductor_cache
from torch.testing import FileCheck
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestKernelBenchmark(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exit_stack = contextlib.ExitStack()
        cls.exit_stack.enter_context(patch.object(config, "benchmark_kernel", True))

    @classmethod
    def tearDownClass(cls):
        cls.exit_stack.close()

    def setUp(self):
        super().setUp()
        PyCodeCache.cache.clear()

    def get_compiled_module(self):
        compiled_module = None
        for v in PyCodeCache.cache.values():
            if hasattr(v, "benchmark_compiled_module"):
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)
        return compiled_module

    def verify_compiled_kernels(self):
        compiled_module = self.get_compiled_module()

        # now run the compiled module in subprocess and check its output
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__} -kc".split(),
            stderr=subprocess.STDOUT,
        ).decode()

        # make sure we have the bandwidth information in the output
        FileCheck().check_count(
            "GB/s",
            1,
            exactly=1,
        ).run(bench_out)

    def test_pw_kernel_benchmark(self):
        @torch.compile
        def f(x):
            return torch.sin(x) + torch.cos(x)

        inp = torch.rand(2, 3).to(GPU_TYPE)
        out = f(inp)
        self.verify_compiled_kernels()

    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @fresh_inductor_cache()
    def test_matmul_triton_kernel_benchmark(self):
        M = 12544
        N = 256
        K = 64
        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(N, K, dtype=torch.float16, device=GPU_TYPE).t()

        @torch.compile
        def f(a, b):
            return torch.relu(a @ b)

        f(a, b)
        self.verify_compiled_kernels()

    def test_bandwidth_computation(self):
        """
        The test does a matmul and then mul. Without max-autotune, we use
        the matmul in aten. So there is a single triton kernel for mul.
        The kernel we generated is like:

            @triton.jit
            def triton_(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):

        Note the in_out_ptr0 argument. It's for a 1000x1000 tensor, but it's
        inplace udpated, so when computing the bandwidth, we should count
        the total memory access as 2 * 1000 * 1000 * 4 = 8MB. This amount is
        what this test asserts.
        """
        torch.set_float32_matmul_precision("high")  # suggested by a warning

        @torch.compile
        def f(x, y):
            z = x @ y
            w = z * z
            return w

        M, N, K = 1000, 1000, 10
        x = torch.rand(M, K).to(GPU_TYPE)
        y = torch.rand(K, N).to(GPU_TYPE)
        out = f(x, y)

        compiled_module = self.get_compiled_module()

        # now run the compiled module in subprocess and check its output
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__} -k".split(),
            stderr=subprocess.STDOUT,
        ).decode()

        # make sure we have the bandwidth information in the output
        FileCheck().check_count(
            "0.008 GB ",
            1,
            exactly=1,
        ).run(bench_out)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
