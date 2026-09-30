# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests("sympy")
