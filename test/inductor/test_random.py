# Owner(s): ["module: inductor"]
import functools
import itertools

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA


inp = functools.partial(torch.zeros, 128, 128, device="cuda")


@config.patch({"triton.vectorize_random": True, "fallback_random": False})
class TestInductorRandom(TestCase):
    def _check(self, fn, contains, mean=0.5, std=0.2887, kernels=1):
        torch.cuda.manual_seed(1234567)
        result1, (code,) = run_and_get_code(fn, inp())
        result2 = fn(inp())
        torch.cuda.manual_seed(1234567)
        result3 = fn(inp())
        result4 = fn(inp())

        self.assertEqual(code.count("@triton.jit"), kernels)

        for term in contains:
            self.assertIn(term, code)

        for result in (result1, result2, result3, result4):
            self.assertAlmostEqual(result.mean().item(), mean, delta=0.05)
            self.assertAlmostEqual(result.std().item(), std, delta=0.05)

        self.assertEqual(result1, result3)
        self.assertEqual(result2, result4)
        self.assertNotEqual(result1, result2)
        self.assertNotEqual(result3, result4)

    def test_rand1(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x + torch.rand_like(x)

        self._check(fn, ["rand4x("])

    def test_rand2(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x + torch.rand(x.size(), device=x.device)

        self._check(fn, ["rand4x("])

    def test_randn(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x + torch.randn_like(x)

        self._check(fn, ["randn4x("], 0.0, 1.0)

    def test_tiled(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x + x.T + torch.rand_like(x)

        self._check(fn, ["rand4x(", "XBLOCK", "YBLOCK"])

    def test_inner_reduction(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return (x + torch.rand_like(x)).mean(dim=1)

        self._check(fn, ["rand4x(", "XBLOCK", "RBLOCK"], std=0.026)

    def test_outer_reduction(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return (x + torch.rand_like(x)).mean(dim=0)

        self._check(fn, ["rand4x(", "XBLOCK", "RBLOCK"], std=0.026, kernels=2)

    def test_multiple(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return x + torch.rand_like(x), x + torch.rand_like(x), torch.rand_like(x)

        (r1, r2, r3), (code,) = run_and_get_code(fn, inp())
        self.assertIn("[3]", code)  # combined seed buffer

        for a, b in itertools.product([r1, r2, r3], [r1, r2, r3]):
            if a is not b:
                self.assertNotEqual(a, b)

    def test_broadcast(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return torch.rand_like(x).expand(8, 128, 128) + 1

        r1, (code,) = run_and_get_code(fn, inp())
        self.assertEqual(code.count("@triton.jit"), 2)
        self.assertEqual(code.count("rand4x("), 1)
        self.assertEqual(r1 - r1[0, :, :], torch.zeros_like(r1))

    def test_dropout(self):
        @torch.compile(fullgraph=True)
        def fn(x):
            return torch.dropout(torch.dropout(x, 0.5, True), 0.5, True)

        for grad in [False, True]:
            counters.clear()
            input = torch.ones_like(inp())
            input.requires_grad = grad
            result, (code,) = run_and_get_code(fn, input)
            self.assertAlmostEqual(
                len(result.nonzero()), result.numel() * 0.25, delta=100
            )
            self.assertEqual(code.count("tl.rand4x"), 2)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
