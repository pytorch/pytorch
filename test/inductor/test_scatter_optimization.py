# Owner(s): ["module: inductor"]

import copy
import os

import torch
from torch import nn
from torch._inductor import metrics
from torch._inductor.runtime.runtime_utils import do_bench_gpu as do_bench
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import HAS_GPU

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


class TestScatterOpt(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()

    def check_metric(self, val):
        self.assertEqual(val, metrics.num_matches_for_scatter_upon_const_tensor)

    def test_cross_entropy_loss(self):
        """
        Match full+scatter in CEL and replaces it with a pointwise.

        Perf data on an A100 GPU:
        Without the scatter optimization:
          ms=47.340, peak_mem=10.524 GB
        With the scatter optimization:
          ms=42.768, peak_mem=7.227 GB
        """
        B, T, D, V = 32, 1024, 768, 50257
        ref_model = nn.Linear(D, V).to(torch.bfloat16)
        opt_model = copy.deepcopy(ref_model)
        ce = nn.CrossEntropyLoss()

        def f(m, x, label):
            ce(m(x).view(-1, V), label.view(-1)).backward()

        opt_f = torch.compile(f)

        x = torch.randn(B, T, D).to(torch.bfloat16)
        label = torch.randint(0, V, (B, T)).to(torch.int64)

        f(ref_model, x, label)
        ref_grad = ref_model.weight.grad
        opt_f(opt_model, x, label)
        act_grad = opt_model.weight.grad
        assert torch.allclose(
            ref_grad, act_grad, atol=1e-3, rtol=1e-3
        ), f"{ref_grad=}\n{act_grad=}"

        self.check_metric(1)

        if DO_PERF_TEST:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                opt_f(opt_model, x, label)
            ms = do_bench(lambda: opt_f(opt_model, x, label))
            peak_mem = torch.cuda.max_memory_allocated() / 10**9
            print(f"{ms=:.3f}, {peak_mem=:.3f} GB")


if HAS_GPU:
    torch.set_default_device("cuda")

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()
