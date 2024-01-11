# Owner(s): ["module: inductor"]

import os
import re
import unittest

import torch
from torch import nn
from torch._dynamo.testing import reset_rng_state

from torch._inductor import config
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.utils import run_and_get_code
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA


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


@config.patch({"triton.multi_kernel": 1, "benchmark_kernel": True})
@instantiate_parametrized_tests
class MultiKernelTest(TestCase):
    def test_softmax(self):
        x = torch.rand(2, 1024).cuda()
        ref = torch.softmax(x, -1)
        compiled_fn = torch.compile(torch.softmax)
        act, (wrapper_code,) = run_and_get_code(compiled_fn, x, -1)
        self.assertTrue(torch.allclose(ref, act))
        self.assertTrue(
            re.search(r"multi_kernel_[^ ]* = MultiKernelCall[(]", wrapper_code)
            is not None
        )

    @parametrize("force_kernel", (0, 1))
    @unittest.mock.patch.dict(
        os.environ, {"TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE": "1"}
    )
    def test_softmax_force_non_persistent_reduction(self, force_kernel):
        """
        Force a specific sub-kernel being picked by mocking the benchmark result.
        """
        x = torch.rand(2, 1024).cuda()
        mock_latency = [0.2, 0.2]
        mock_latency[force_kernel] = 0.1  # this make sure force_kernel will be picked

        def f(x):
            return torch.softmax(x, -1) + force_kernel

        orig_run = MultiKernelCall.run_with_argless_kernels
        picked_kernel = None

        def mock_run(self, kernel_calls):
            out = orig_run(self, kernel_calls)
            nonlocal picked_kernel
            picked_kernel = self.picked_kernel
            return out

        with unittest.mock.patch.object(
            MultiKernelCall, "run_with_argless_kernels", mock_run
        ), unittest.mock.patch.object(
            MultiKernelCall, "benchmark_sub_kernels", lambda *args: mock_latency
        ):
            torch.compile(f)(x)
        self.assertEqual(picked_kernel, force_kernel)

    @config.patch("warn_mix_layout", True)
    def test_softmax_warn_mixed_layout(self):
        self.test_softmax()

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
