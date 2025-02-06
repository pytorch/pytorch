# Owner(s): ["module: inductor"]

import os

from triton.testing import do_bench

import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX, parametrize, instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import HAS_CUDA, GPU_TYPE
from torch._dynamo.utils import same


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


class TestOnlineSoftmax(TestCase):
    def do_test_acc_and_perf(self, op):
        N = 32 * 1024
        V = 50304  # padded version for gpt2

        def f(x):
            return op(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=GPU_TYPE)
        opt_f = torch.compile(f)
        expected = f(x)
        actual = opt_f(x)

        self.assertTrue(torch.allclose(expected, actual, atol=1e-2, rtol=1e-2))

        if DO_PERF_TEST:
            eager_ms = do_bench(lambda: f(x))
            opt_ms = do_bench(lambda: opt_f(x))
            print(f"{eager_ms=}")
            print(f"{opt_ms=}")

    def test_softmax(self):
        self.do_test_acc_and_perf(torch.softmax)

    def test_log_softmax(self):
        self.do_test_acc_and_perf(torch.log_softmax)

    def get_softmax_wrapper(self, V=50304, use_log_softmax=False):
        N = 32 * 1024

        @torch.compile
        def f(x):
            if use_log_softmax:
                return torch.log_softmax(x, dim=-1)
            else:
                return torch.softmax(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16, device=GPU_TYPE)
        out, source_codes = run_and_get_code(f, x)
        return source_codes[0]

    def test_codegen_3pass_softmax_due_to_disable(self):
        with inductor_config.patch(online_softmax=False):
            wrapper_code = self.get_softmax_wrapper()

        self.assertEqual(wrapper_code.count("for r0_offset in"), 3)

    @parametrize("V", [2048, 50304])
    @parametrize("use_log_softmax", [False, True])
    def test_codegen_online_softmax(self, use_log_softmax, V):
        wrapper_code = self.get_softmax_wrapper(use_log_softmax=use_log_softmax, V=V)

        self.assertEqual(wrapper_code.count("for r0_offset in"), 2)


    def test_codegen_softmax_persistent_reduction(self):
        """
        Persistent reduction has no for loops.
        """
        wrapper_code = self.get_softmax_wrapper(1024)
        self.assertEqual(wrapper_code.count("for r0_offset in"), 0)

    # This test only work if we use pattern matcher rather the decompose
    # softmax/log_softmax specially.
    @parametrize("nrow", [2, 2048])
    @parametrize("dim", [-1, 0, 1])
    def no_test_prepare_softmax(self, dim, nrow):
        def f(x, dim):
            xmax = x.amax(dim=dim, keepdim=True)
            xsum = (x - xmax).exp().sum(dim=dim, keepdim=True)
            return xmax, xsum

        x = torch.randn(nrow, 2048, dtype=torch.bfloat16, device=GPU_TYPE)
        act, (code,) = run_and_get_code(torch.compile(f), x, dim)
        ref = f(x, dim)
        self.assertTrue(same(ref, act, tol=1e-2))

        if nrow == 2 and dim == 0:
            # persistent reduction triggered
            expected_num_loop = 0
        else:
            # A single loop due to online softmax
            expected_num_loop = 1
        self.assertEqual(code.count("for r0_offset in"), expected_num_loop)

instantiate_parametrized_tests(TestOnlineSoftmax)

if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
