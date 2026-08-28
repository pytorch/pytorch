# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the fused two-stage cross-CTA kernel (few-row / huge-N).
# Proves it compiles and runs on a tiny single huge row; real numeric coverage comes
# from the reduction overrides' OpInfo suites in a later commit.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelXcta(TestCase):
    def test_reduce_row_xcta(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_xcta

        x = torch.randn(1, 1 << 20, device="cuda")
        out = kernel_xcta.reduce_row_xcta(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32
        )
        self.assertEqual(out, x.double().sum(dim=1).float(), atol=2e-2, rtol=1e-4)

    def test_two_output_split(self):
        # Stage 2 projects two outputs off ONE combined accumulator, which is what lets aminmax
        # and var_mean take the split for free. It is wired into both _try_fast_row and
        # _reduce_all and was exercised by nothing. A VALUE trait on purpose: this commit still
        # refuses index traits (the chunk index is row % C until the ragged split lands).
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_xcta

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
        # The file's headline property: ONE compiled kernel serves any M and any N in a vec class,
        # because the sub-row length, C and the divisor are runtime args. Assert it the way the row
        # tile does -- distinct shapes in one vec class must not multiply the kernel count.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_xcta

        key = "vecclass"
        trait = T.SumOps(acc=cutlass.Float32)

        def compiled():
            return len([k for k in kernel_xcta._PLAN if key in repr(k)])

        # Every N here is a multiple of 4096, so the exact split has a legal C in its window --
        # a declined shape would make the count assertion below vacuous, hence the not-None check.
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
        # Two refusals keep this path off geometries it serves badly, and the dispatcher relies on
        # getting None back rather than a slow kernel: C == 1 (a "split" into one chunk IS the
        # one-shot that was already declined) and a row too short to split at all.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_xcta

        trait = T.SumOps(acc=cutlass.Float32)
        # Short row: below the sub-row floor there is no legal C > 1.
        self.assertIsNone(
            kernel_xcta.reduce_row_xcta(
                trait, "decline_short", torch.randn(1, 64, device="cuda"), torch.float32
            )
        )
        # A prime N has no divisor in the window, so the exact-split kernel must decline it and
        # leave it to the ragged split / K0 rather than fold a 65537-element fragment at vec=1.
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
