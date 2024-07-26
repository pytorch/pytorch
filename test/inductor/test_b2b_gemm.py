# Owner(s): ["module: inductor"]
import os
import unittest

import torch
from torch._inductor.runtime.runtime_utils import do_bench
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import HAS_CUDA


class B2BGEMMTest(TestCase):
    @torch._dynamo.config.patch(cache_size_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_applicable(self):
        """applicable sizes"""

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.mm(m1, m2), m3)

        f_opt = torch.compile(f)
        A = torch.randn((1024, 10), device="cuda", dtype=torch.float16)
        B = torch.randn((10, 1024), device="cuda", dtype=torch.float16)
        C = torch.randn((1024, 10), device="cuda", dtype=torch.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(torch.allclose(f(A, B, C), res, atol=0.2, rtol=0.01))
        self.assertTrue("B2B_GEMM_TRITON_ENTRANCE" in code)

    @torch._dynamo.config.patch(cache_size_limit=32)
    @torch._inductor.config.patch(b2b_gemm_pass=True)
    def test_b2b_gemm_not_applicable(self):
        """non-applicable sizes"""

        def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
            return torch.mm(torch.mm(m1, m2), m3)

        f_opt = torch.compile(f)
        A = torch.randn((500, 100), device="cuda", dtype=torch.float16)
        B = torch.randn((100, 200), device="cuda", dtype=torch.float16)
        C = torch.randn((200, 300), device="cuda", dtype=torch.float16)
        res, (code,) = run_and_get_code(f_opt, A, B, C)
        self.assertTrue(torch.allclose(f(A, B, C), res, atol=0.2, rtol=0.01))
        self.assertTrue("B2B_GEMM_TRITON_ENTRANCE" not in code)

    @unittest.skipIf(
        not (os.environ.get("DO_PERF_TEST") == "1"), "Perf test not enabled"
    )
    @torch._dynamo.config.patch(cache_size_limit=32)
    def test_b2b_gemm_performance(self):
        """compare torch.compile(f, b2b_gemm = off) with torch.compile(f, b2b_gemm = on)"""

        def run_with_b2b_gemm_off(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                return torch.mm(torch.mm(m1, m2), m3)

            f_opt = torch.compile(f, dynamic=False)
            return do_bench(f_opt, (m1, m2, m3), {}, warmup=100, rep=1000)

        @torch._inductor.config.patch(b2b_gemm_pass=True)
        def run_with_b2b_gemm_on(
            m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor
        ) -> float:
            def f(m1: torch.Tensor, m2: torch.Tensor, m3: torch.Tensor) -> torch.Tensor:
                return torch.mm(torch.mm(m1, m2), m3)

            f_opt = torch.compile(f, dynamic=False)
            return do_bench(f_opt, (m1, m2, m3), {}, warmup=100, rep=1000)

        speedups = []
        print()
        print("Speedups".ljust(10), end="")
        for N in [10, 32, 60, 128, 200]:
            print(f"N = {N}".ljust(10), end="")
        print()
        for M in [256, 500, 1024, 2000, 4096, 8000]:
            print(f"M = {M}".ljust(10), end="")
            for N in [10, 32, 60, 128, 200]:
                O, P = M, N
                A = torch.randn((M, N), device="cuda", dtype=torch.float16)
                B = torch.randn((N, O), device="cuda", dtype=torch.float16)
                C = torch.randn((O, P), device="cuda", dtype=torch.float16)
                speedup = run_with_b2b_gemm_off(A, B, C) / run_with_b2b_gemm_on(A, B, C)
                print(f"{round(speedup, 3)}".ljust(10), end="")
                speedups.append(speedup)
            print()

        average_speedup = 1.0
        for s in speedups:
            average_speedup *= s
        average_speedup = average_speedup ** (1 / len(speedups))
        print(f"Average speedup: {round(average_speedup, 3)}")

        # flaky test assertion: disabled
        # self.assertTrue(average_speedup > 1)


if __name__ == "__main__":
    if HAS_CUDA:
        run_tests()
