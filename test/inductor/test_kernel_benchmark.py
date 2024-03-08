# Owner(s): ["module: inductor"]
import contextlib
import subprocess
import sys
from unittest.mock import patch

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import rand_strided
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

    def verify_compiled_kernels(self, GB_count=1):
        compiled_module = self.get_compiled_module()

        # now run the compiled module in subprocess and check its output
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__} -kc".split(),
            stderr=subprocess.STDOUT,
        ).decode()

        # make sure we have the bandwidth information in the output
        FileCheck().check_count(
            "GB/s",
            GB_count,
            exactly=1,
        ).run(bench_out)

    def check_bandwidth(self, compiled_module, num_gb):
        # now run the compiled module in subprocess and check its output
        bench_out = subprocess.check_output(
            f"{sys.executable} {compiled_module.__file__} -k".split(),
            stderr=subprocess.STDOUT,
        ).decode()

        # make sure we have the bandwidth information in the output
        FileCheck().check_count(
            f"{num_gb} GB ",
            1,
            exactly=1,
        ).run(bench_out)

    def test_pw_kernel_benchmark(self):
        @torch.compile
        def f(x):
            return torch.sin(x) + torch.cos(x)

        inp = torch.rand(2, 3).to(device=GPU_TYPE)
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

    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    @fresh_inductor_cache()
    def test_mm_triton_kernel_benchmark(self):
        M = 2048
        N = 2432
        K = 1949
        K_2 = 3581
        a = rand_strided((M, K_2), (K_2, 1), device="cuda", dtype=torch.float16)
        b = rand_strided((K, N), (1, K), device="cuda", dtype=torch.float16)

        @torch.compile
        def f(a, b):
            a_1 = torch.narrow(a, 1, 0, K)
            c = torch.mm(a_1, b)
            return c

        f(a, b)
        self.verify_compiled_kernels(GB_count=3)

        # make sure we correctly generate the grid info
        compiled_module = self.get_compiled_module()
        with open(compiled_module.__file__) as f:
            source_code = f.read()
        lines = source_code.split("\n")
        meta = [l for l in lines if "meta0 = {" in l]
        scope = {}
        from torch._inductor.kernel.mm_common import mm_grid

        exec(meta[0], scope)
        grid = mm_grid(M, N, scope["meta0"])
        FileCheck().check_count(
            f"grid={grid}",
            2,
            exactly=1,
        ).run(source_code)

    def test_matmul_bandwidth_computation(self):
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
        x = torch.rand(M, K).to(device=GPU_TYPE)
        y = torch.rand(K, N).to(device=GPU_TYPE)
        out = f(x, y)

        compiled_module = self.get_compiled_module()

        self.check_bandwidth(compiled_module, 0.008)

    def test_unused_input_bandwidth_computation(self):
        M, N = 5, 1000000

        @torch.compile
        def f(a, b, c):
            return a + c

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        torch._dynamo.mark_dynamic(c, 0)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_c + size_out
        # num_gb = (5 * 1000000 + 5 * 1000000 + 5 * 1000000) * 2 / 1e9
        #        = 0.030
        self.check_bandwidth(compiled_module, "0.030")

    def test_reduction_bandwidth_computation(self):
        @torch.compile
        def f(a):
            return torch.sum(a, dim=1)

        a = torch.rand(1000, 20, 1000, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a,)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_out
        # num_gb = (1000 * 20 * 1000 + 1000 * 1000) * 2 / 1e9
        #        = 0.042
        self.check_bandwidth(compiled_module, "0.042")

    @config.patch(max_autotune=True)
    def test_fused_layernorm_bandwidth_computation(self):
        M, N = 10, 1000000

        @torch.compile
        def f(a, b, c, d):
            x0 = a + b
            x1 = torch.nn.functional.layer_norm(
                x0, normalized_shape=(N,), weight=c, bias=d, eps=1e-05
            )
            x2 = torch.sigmoid(x1)
            return x0 * x2

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        d = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c, d)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_b + size_c + size_d + size_out
        # num_gb = (10 * 1000000 + 1000000 + 1000000 + 1000000 + 10 * 1000000) * 2 / 1e9
        #        = 0.046
        self.check_bandwidth(compiled_module, "0.046")

    def test_slice_add_cat_bandwidth_computation(self):
        M, N = 5, 1000000

        @torch.compile
        def f(a, b, c):
            x0 = torch.narrow(b, 1, N, N)
            # broadcasting
            x1 = x0 + c
            return torch.cat([a, x1], dim=1)

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N * 5, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # we overestimate the size of "slice_b" due to torch.cat
        # num_gp = size_a + size_slice_b + size_c + size_out
        # num_gb = (5 * 1000000 + 5 * 2000000 + 1000000 + 5 * 2000000) * 2 / 1e9
        #        = 0.052
        self.check_bandwidth(compiled_module, "0.052")

    def test_slice_add_bandwidth_computation(self):
        M, N = 5, 1000000

        @torch.compile
        def f(a, b, c):
            x0 = torch.narrow(b, 1, N, N)
            return a + x0 + c

        a = torch.rand(M, N, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(M, N * 5, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(b, 0)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # num_gb = size_a + size_slice_b + size_c + out_size
        # num_gb = (5 * 1000000 + 5 * 1000000 + 1000000 + 5 * 1000000) * 2 / 1e9
        #        = 0.032
        self.check_bandwidth(compiled_module, "0.032")

    def test_mm_slice_add_bandwidth_computation(self):
        M, N, K = 1000, 1000, 30

        @torch.compile
        def f(a, b, c):
            x0 = torch.mm(a, b)
            x1 = torch.narrow(c, 1, 20 * N, N)
            x2 = torch.narrow(c, 1, 21 * N, N)
            return x0 + x1 + x2

        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, N * 100, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # torch.mm becomes an extern kernel, so we measure the nbytes
        # for the pointwise add kernel:
        # num_gb = x0 + 2 * size_slice_c + size_out
        # num_gb = (1000 * 1000 + 2 * 1000 * 1000 + 1000 * 1000) * 2/ 1e9
        #        = 0.008
        self.check_bandwidth(compiled_module, "0.008")

    def test_mm_slice_add_bandwidth_computation_2(self):
        M, N, K = 1000, 1000, 30

        @torch.compile
        def f(a, b, c):
            x0 = torch.mm(a, b)
            x1 = torch.narrow(c, 1, 20 * N, N)
            x2 = torch.narrow(c, 1, 20 * N, N)
            return x0 + x1 + x2

        a = torch.rand(M, K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        c = torch.rand(N, N * 100, dtype=torch.float16, device=GPU_TYPE)
        inputs = (a, b, c)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()
        # torch.mm becomes an extern kernel, so we measure the nbytes
        # for the pointwise add kernel:
        # num_gb = x0 + size_slice_c + size_out
        # num_gb = (1000 * 1000 + 1000 * 1000 + 1000 * 1000) * 2 / 1e9
        #        = 0.006
        # note that we only count one size_slice_c because two accesses
        # have the same index.
        self.check_bandwidth(compiled_module, "0.006")

    @config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON")
    def test_slice_mm_bandwidth_computation(self):
        M, N, K = 1000, 2000, 3000

        @torch.compile
        def f(a, b):
            x = torch.narrow(a, 1, K, K)
            return torch.mm(x, b)

        a = torch.rand(M, 3 * K, dtype=torch.float16, device=GPU_TYPE)
        b = torch.rand(K, N, dtype=torch.float16, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(a, 0)
        inputs = (a, b)
        out = f(*inputs)

        compiled_module = self.get_compiled_module()

        # c[1000, 2000] = x[1000, 3000] @ b[3000, 2000]
        # num_gb = (1000 * 2000 + 1000 * 3000 + 3000 * 2000) * 2 / 1e9
        #        = 0.022
        self.check_bandwidth(compiled_module, "0.022")

    def test_star_dep(self):
        """
        Test the bandwidth estimation for StarDep
        """

        @torch.compile
        def f(a, b):
            a[b] = 3.0

        a = torch.rand(10000, 5000, device=GPU_TYPE)
        b = torch.randint(
            0, 10000, [20000], device=GPU_TYPE, dtype=torch.int32
        ).unsqueeze(1)
        f(a, b)
        compiled_module = self.get_compiled_module()
        # 20000 * 4 = 80KB for b
        # 20000 * 5000 * 4 = 200MB for a
        self.check_bandwidth(compiled_module, "0.200")


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
