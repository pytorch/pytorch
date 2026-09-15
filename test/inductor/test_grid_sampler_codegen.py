# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


class GridSamplerCodegenTests(InductorTestCase):
    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    def test_grid_sampler_2d_uses_32bit_coordinate_indices(self):
        def fn(a, b):
            return torch.ops.aten.grid_sampler_2d(a, b, 0, 0, False)

        a = torch.randn([2, 3, 16, 16], dtype=torch.float32, device="cuda")
        b = torch.rand([2, 16, 16, 2], dtype=torch.float32, device="cuda") * 2 - 1

        expected = fn(a, b)
        actual, codes = run_and_get_code(torch.compile(fn, backend="inductor"), a, b)

        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        source_code = "\n".join(codes)
        self.assertIn(".to(tl.int32)", source_code)
        self.assertNotIn("tl.int64", source_code)

    @unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA and Triton")
    @parametrize("dims_align", [(2, False), (2, True), (3, False)])
    def test_grid_sampler_backward_deterministic(self, dims_align):
        """Compiled autograd must preserve strict determinism for native and cuDNN routes."""
        dims, align = dims_align

        def fn(a, b):
            return torch.nn.functional.grid_sample(a, b, align_corners=align)

        a = torch.randn((1, 3) + (5,) * dims, device="cuda", requires_grad=True)
        b = (
            torch.rand((1,) + (3,) * dims + (dims,), device="cuda") * 1.5 - 0.75
        ).requires_grad_()
        grad = torch.randn((1, 3) + (3,) * dims, device="cuda")
        with DeterministicGuard(True), torch.backends.cudnn.flags(enabled=align):
            expected = torch.autograd.grad(fn(a, b), (a, b), grad)
            compiled = torch.compile(fn, fullgraph=True)
            first = torch.autograd.grad(compiled(a, b), (a, b), grad)
            for actual, ref in zip(first, expected):
                self.assertEqual(actual, ref, atol=2e-5, rtol=2e-5)
            for _ in range(3):
                actual = torch.autograd.grad(compiled(a, b), (a, b), grad)
                for value, prior in zip(actual, first):
                    self.assertTrue(
                        torch.equal(value.view(torch.int32), prior.view(torch.int32))
                    )


instantiate_parametrized_tests(GridSamplerCodegenTests)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests("sympy")
