# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the vectorized row kernel on the shared tile datapath. Proves it
# compiles and reduces a contiguous last dim, that its ROLLED fold is what the design claims
# (one compiled kernel serves every N in a vec class, so distinct N must not each add a
# plan-cache entry), and that its two narrow-row options -- one thread per row, and the
# TMA-staged load with its rotated smem read -- are wired and correct. Real numeric coverage
# comes from the reduction overrides' OpInfo suites in a later commit.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelRowTile(TestCase):
    def test_reduce_row_tile(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile

        x = torch.randn(128, 512, device="cuda")
        (out,) = kernel_rowtile.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [torch.float32]
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)

    def test_one_kernel_per_vec_class(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile

        trait = T.SumOps(acc=cutlass.Float32)
        kernel_rowtile._CACHE.clear()
        # same vec class (all multiples of 4 in fp32) and the same config rung, so the
        # rolled fold takes N at runtime and ONE kernel must serve all of them
        for n in (2048, 2052, 2056, 2060, 2064):
            x = torch.randn(64, n, device="cuda")
            (out,) = kernel_rowtile.reduce_row_tile(
                trait, "vecclass", x, [torch.float32]
            )
            self.assertEqual(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)
        # count only THIS test's plans: the reference x.sum() above is itself served by this
        # kernel once the aten overrides land later in the stack, which compiles a second
        # (correct) entry under its own op key.
        mine = [k for k in kernel_rowtile._CACHE if "vecclass" in k]
        self.assertEqual(
            len(mine),
            1,
            f"expected ONE compiled kernel for the vec class, got {sorted(mine)}",
        )

    def test_narrow_row_one_thread_per_row(self):
        # tpr=1: one thread owns a whole row, no lane merge at all.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile

        x = torch.randn(8192, 16, device="cuda")
        (out,) = kernel_rowtile.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32),
            "narrow",
            x,
            [torch.float32],
            tpr=1,
            use_tma=False,
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-3, rtol=1e-3)

    def test_tma_staged_narrow_row_argmax(self):
        # The TMA path rotates each thread's smem read order to de-conflict the banks, so the
        # column a value came from is no longer the loop counter. An index trait is what
        # catches a wrong rotation: the values would still look right.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile

        if not kernel_rowtile.tma_ok(32, 4, 65536, torch.device("cuda")):
            self.skipTest("TMA path not applicable on this device")
        x = torch.randn(65536, 32, device="cuda")
        (idx,) = kernel_rowtile.reduce_row_tile(
            T.ArgMaxOps(acc=cutlass.Float32),
            "narrow_tma",
            x,
            [torch.int32],
            tpr=1,
            use_tma=True,
        )
        self.assertEqual(idx, x.argmax(dim=1).to(torch.int32))


if __name__ == "__main__":
    run_tests()
