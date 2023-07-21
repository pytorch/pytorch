# Owner(s): ["module: inductor"]
import torch
from torch import nn

from torch._inductor import config
from torch.nn import functional as F
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA

config.triton.multi_kernel = True
config.benchmark_kernel = True


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
        lm = nn.LayerNorm(1024).cuda()
        x = torch.rand(2, 1024).cuda()
        ref = lm(x)
        act = torch.compile(lm)(x)
        self.assertTrue(
            torch.allclose(ref, act, atol=1e-4, rtol=1e-4), f"ref:\n{ref}\nact:\n{act}"
        )

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

        ref = f(*x)

        opt_f = torch.compile(f)
        act = opt_f(*x)

        # don't compare tensor since inductor random number implementation
        # is different to eager. We should fallback to eager if we want to
        # test accuracy.


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if HAS_CUDA:
        run_tests()
