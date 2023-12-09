# Owner(s): ["module: inductor"]

import torch
from torch import nn
from torch._dynamo.testing import reset_rng_state

from torch._inductor import config
from torch.nn import functional as F
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA

config.triton.multi_kernel = 1
config.benchmark_kernel = True
config.compile_threads = 1


class TransformerSnippet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(64)

    def forward(self, x1, x2):
        x1 = F.dropout(x1, 0.1)
        x2 = F.dropout(self.ln1(x2), 0.1)

        return self.ln2(x1 + x2)

    def example_inputs(self):
        return (torch.randn(2, 64).cuda(), torch.randn(2, 64).cuda())


class MultiKernelTest(TestCase):
    def test_softmax(self):
        x = torch.rand(2, 1024).cuda()
        ref = torch.softmax(x, -1)
        act = torch.compile(torch.softmax)(x, -1)
        self.assertTrue(torch.allclose(ref, act))

    def test_layernorm(self):
        ln = nn.LayerNorm(1024).cuda()
        x = torch.rand(2, 1024).cuda()
        ref = ln(x)
        act = torch.compile(ln)(x)
        self.assertTrue(
            torch.allclose(ref, act, atol=1e-4, rtol=1e-4), f"ref:\n{ref}\nact:\n{act}"
        )

    def test_inplace_update(self):
        """
        Inductor generate inplace kernel for mul.
        """

        def f(x, y):
            return x.sum(dim=-1, keepdims=True) * (y @ y)

        x = torch.rand(1024, 1024).cuda()
        y = torch.rand(1024, 1024).cuda()
        ref = f(x, y)
        act = torch.compile(f)(x, y)
        self.assertTrue(torch.allclose(ref, act))

    def test_transformer_snippet(self):
        """
        Test a snippet of transformer that will cause different arglist for
        the persistent and non-persistent flavor of reductions.
        """
        model = TransformerSnippet().cuda()
        x = model.example_inputs()

        def f(*x):
            y = model(*x)
            return y

        reset_rng_state()
        ref = f(*x)

        opt_f = torch.compile(f)
        reset_rng_state()
        act = opt_f(*x)

        # don't compare tensor if using inductor random number generator.
        # inductor random number implementation is different to eager.
        # We should fallback to eager if we want to test accuracy.
        if config.fallback_random:
            self.assertTrue(
                torch.allclose(ref, act, atol=1e-4, rtol=1e-4),
                f"ref:\n{ref}\nact:\n{act}",
            )

    def test_transformer_snippet_with_fallback_random(self):
        """
        Same as test_transformer_snippet but fallback the random number
        generator to eager so we can check accuracy.
        """
        with config.patch("fallback_random", True):
            self.test_transformer_snippet()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA:
        run_tests()
