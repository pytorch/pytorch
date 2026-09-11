# Owner(s): ["module: dsl-native-ops"]
# Smoke tests for few-row, large-N cross-CTA reduction; OpInfo covers numerics.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelXcta(TestCase):
    def test_reduce_row_xcta(self):
        import cutlass

        from torch._native.ops.reductions import kernel_xcta, traits as T

        x = torch.randn(1, 1 << 20, device="cuda")
        out = kernel_xcta.reduce_row_xcta(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32
        )
        self.assertEqual(out, x.double().sum(dim=1).float(), atol=2e-2, rtol=1e-4)

    def test_two_output_split(self):
        # Stage 2 projects both outputs from one accumulator; this path rejects index traits.
        import cutlass

        from torch._native.ops.reductions import kernel_xcta, traits as T

        x = torch.randn(2, 1 << 20, device="cuda")
        res = kernel_xcta.reduce_row_xcta_2out(
            T.AMinMaxOps(acc=cutlass.Float32),
            "smoke_2out",
            x,
            [torch.float32, torch.float32],
        )
        self.assertIsNotNone(res, "the split declined a shape it is meant to serve")
        lo, hi = res
        want = torch.aminmax(x, dim=1)
        self.assertEqual(lo, want.min)
        self.assertEqual(hi, want.max)

    def test_one_kernel_per_vec_class(self):
        # Runtime sub-row geometry lets one kernel serve every M and N in a vector class.
        import cutlass

        from torch._native.ops.reductions import kernel_xcta, traits as T

        key = "vecclass"
        trait = T.SumOps(acc=cutlass.Float32)

        def compiled():
            return len([k for k in kernel_xcta._PLAN if key in repr(k)])

        # Multiples of 4096 guarantee a legal exact split; assert it to keep the count meaningful.
        def run(m, n):
            x = torch.randn(m, n, device="cuda")
            out = kernel_xcta.reduce_row_xcta(trait, key, x, torch.float32)
            self.assertIsNotNone(out, f"declined ({m}, {n})")
            self.assertEqual(out, x.double().sum(dim=1).float(), atol=2e-2, rtol=1e-4)

        kernel_xcta._PLAN.clear()
        for m, n in ((1, 1 << 20), (3, 1 << 20), (2, 1 << 21)):
            run(m, n)
        few = compiled()
        self.assertGreater(few, 0, "nothing compiled -- the count is measuring nothing")
        for n in ((1 << 20) + 4096, (1 << 20) + 8192, (1 << 20) + 12288):
            run(1, n)
        self.assertEqual(
            compiled(), few, "a new N in the same vec class compiled a new kernel"
        )

    def test_declines_are_deliberate(self):
        # Return None for unsplittable rows so the dispatcher can choose a faster path.
        import cutlass

        from torch._native.ops.reductions import kernel_xcta, traits as T

        trait = T.SumOps(acc=cutlass.Float32)
        # Below the sub-row floor, no C > 1 is legal.
        self.assertIsNone(
            kernel_xcta.reduce_row_xcta(
                trait, "decline_short", torch.randn(1, 64, device="cuda"), torch.float32
            )
        )
        # Prime N lacks an in-window divisor; decline instead of folding 65537 scalars.
        self.assertIsNone(
            kernel_xcta.reduce_row_xcta(
                trait,
                "decline_prime",
                torch.randn(1, 65537, device="cuda"),
                torch.float32,
            )
        )


if __name__ == "__main__":
    run_tests()
