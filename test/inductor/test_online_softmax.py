# Owner(s): ["module: inductor"]

import os

import torch
import torch._inductor.config as inductor_config
from torch._inductor.runtime.runtime_utils import do_bench_gpu as do_bench
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


@inductor_config.patch(force_disable_caches=True)
class TestOnlineSoftmax(TestCase):
    def do_test_acc_and_perf(self, op):
        N = 32 * 1024
        V = 50304  # padded version for gpt2

        def f(x):
            return op(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16)
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

    def test_logsoftmax(self):
        self.do_test_acc_and_perf(torch.log_softmax)

    def get_softmax_wrapper(self, V=50304):
        N = 32 * 1024

        @torch.compile
        def f(x):
            return torch.softmax(x, dim=-1)

        x = torch.randn(N, V, dtype=torch.bfloat16)
        out, source_codes = run_and_get_code(f, x)
        return source_codes[0]

    def test_codegen_online_softmax(self):
        wrapper_code = self.get_softmax_wrapper()
        self.assertEqual(wrapper_code.count("for roffset in"), 2)

    def test_codegen_3pass_softmax_due_to_disable(self):
        with inductor_config.patch(online_softmax=False):
            wrapper_code = self.get_softmax_wrapper()

        self.assertEqual(wrapper_code.count("for roffset in"), 3)

    def test_codegen_3pass_softmax_due_to_small_rnumel(self):
        """
        Online softmax is only enabled for large rnumel.
        """
        wrapper_code = self.get_softmax_wrapper(2048)
        self.assertEqual(wrapper_code.count("for roffset in"), 3)

    def test_codegen_softmax_persistent_reduction(self):
        """
        Persistent reduction has no for loops.
        """
        wrapper_code = self.get_softmax_wrapper(1024)
        self.assertEqual(wrapper_code.count("for roffset in"), 0)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        torch.set_default_device("cuda")
        run_tests()
