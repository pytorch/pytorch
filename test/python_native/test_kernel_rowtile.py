# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the vectorized row kernel on the shared tile datapath. Proves it
# compiles and reduces a contiguous last dim, and that its ROLLED fold is what the design
# claims: one compiled kernel serves every N in a vec class, so distinct N must not each
# add a plan-cache entry. Real numeric coverage comes from the reduction overrides' OpInfo
# suites in a later commit.

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
        self.assertEqual(out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)

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
            self.assertEqual(out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)
        # count only THIS test's plans: the reference x.sum() above is itself served by this
        # kernel once the aten overrides land later in the stack, which compiles a second
        # (correct) entry under its own op key.
        mine = [k for k in kernel_rowtile._CACHE if "vecclass" in k]
        self.assertEqual(
            len(mine),
            1,
            f"expected ONE compiled kernel for the vec class, got {sorted(mine)}",
        )

    def test_two_output_trait(self):
        # nouts=2 is a distinct store path (nslots x nouts) and the docstring calls it out, but
        # nothing crossed it. max.dim returns (values, indices) off one combined accumulator.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        x = torch.randn(64, 512, device="cuda")
        vals, idx = rt.reduce_row_tile(
            T.MaxDimOps(acc=cutlass.Float32),
            "smoke_2out",
            x,
            [torch.float32, torch.int32],
            nouts=2,
        )
        want_v, want_i = x.max(dim=1)
        self.assertEqual(vals, want_v)
        self.assertEqual(idx, want_i.to(torch.int32))

    def test_stage1_partials_are_raw_accumulators(self):
        # final=False is the documented reason this kernel doubles as the cross-CTA stage 1: it
        # stores the RAW per-field accumulator instead of a projection. For Welford that is
        # (mean, M2, count) per row, so the count field must equal the row length exactly --
        # a projected store would put a variance there.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        x = torch.randn(32, 256, device="cuda")
        trait = T.WelfordOps(acc=cutlass.Float32)
        parts = rt.reduce_row_tile(
            trait, "smoke_partials", x, [torch.float32] * trait.nfields, final=False
        )
        self.assertEqual(len(parts), trait.nfields)
        self.assertEqual(parts[0], x.mean(dim=1), atol=1e-5, rtol=1e-5)
        self.assertEqual(parts[2], torch.full((32,), 256.0, device="cuda"))

    def test_single_row_config_rungs_are_valid(self):
        # reduce-all arrives here as ONE row, where the row-packing ladder leaves the device on a
        # fraction of a CTA. single_row_config widens it, but ONLY to a _TPR_RUNGS value: tpr is
        # both the reduce-tree width and the block size, so a computed width got this wrong twice
        # -- 50 threads for a 100-element fp64 row, and tpr=96 (three warps), which silently
        # returned a WRONG variance. Sweep the band and assert the invariants rather than pinning
        # two shapes, since which N get widened is a measured choice that may move.
        from torch._native.ops.reductions import kernel_rowtile as rt

        widened = 0
        for n in (32, 64, 96, 100, 128, 200, 400, 1024, 2048, 4096, 16384, 1 << 20):
            for bits in (16, 32, 64):
                cfg = rt.single_row_config(n, bits, 1)
                if (
                    cfg is None
                ):  # the ladder's own pick stands, or the row cannot feed a warp
                    continue
                widened += 1
                with self.subTest(n=n, bits=bits):
                    self.assertEqual(
                        cfg.tpr & (cfg.tpr - 1), 0, "tpr must be a power of two"
                    )
                    self.assertEqual(
                        cfg.tpr % rt.WARP, 0, "tpr must be a warp multiple"
                    )
                    self.assertIn(cfg.tpr, rt._TPR_RUNGS)
                    self.assertLessEqual(cfg.tpr, cfg.nt)
                    self.assertEqual(cfg.nt % cfg.tpr, 0, "nt must hold whole rows")
                    self.assertGreater(cfg.tpr, rt.row_config(n, bits, 1).tpr)
        self.assertGreater(
            widened, 0, "nothing was widened -- the sweep has gone stale"
        )
        # A row too narrow to feed one warp keeps the ladder's pick.
        self.assertIsNone(rt.single_row_config(32, 32, 1))

    def test_oneshot_gate_bounds_loads_not_just_smem(self):
        # _oneshot_ok is what keeps a row that FITS smem but needs a huge per-thread load count
        # off this path (it belongs on the cross-CTA split). Both bounds must bite.
        import torch
        from torch._native.ops.reductions import kernel_general as kg

        self.assertTrue(kg._oneshot_ok(torch.empty(1, 4096, device="cuda")))
        # A prime N collapses vec to 1, so loads/thread is N/tpr: rejected on the load bound
        # even though the tile is small.
        self.assertFalse(kg._oneshot_ok(torch.empty(1, 65537, device="cuda")))
        # Wide enough to blow the smem budget outright.
        self.assertFalse(kg._oneshot_ok(torch.empty(1, 1 << 22, device="cuda")))

    def test_absmax_absmin_propagate_nan(self):
        # These traits' contract is vector_norm(ord=+-inf), which is amax/amin of |x| and
        # PROPAGATES NaN. A builtin max()/min() that returned the non-NaN operand would diverge
        # silently on exactly one row, so pin it.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        x = torch.randn(8, 512, device="cuda")
        x[0, 17] = float("nan")
        for trait, ord_ in ((T.AbsMaxOps, float("inf")), (T.AbsMinOps, -float("inf"))):
            with self.subTest(trait=trait.__name__):
                (got,) = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32),
                    f"nan_{trait.__name__}",
                    x,
                    [torch.float32],
                )
                want = torch.linalg.vector_norm(x, ord=ord_, dim=-1)
                self.assertEqual(got.isnan(), want.isnan())
                self.assertEqual(got[1:], want[1:])

    def test_welford_divisor_clamps_at_zero(self):
        # correction >= n must divide by ZERO (-> +inf, which is what aten returns), never by a
        # negative number -- unclamped it returned a NEGATIVE variance.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        n = 128
        x = torch.randn(16, n, device="cuda")
        for correction in (n, n + 7):
            with self.subTest(correction=correction):
                (got,) = rt.reduce_row_tile(
                    T.WelfordOps(correction=correction, acc=cutlass.Float32),
                    f"welford_c{correction}",
                    x,
                    [torch.float32],
                )
                self.assertEqual(got, x.var(dim=1, correction=correction))
                self.assertTrue(torch.isinf(got).all())

    def test_integer_accumulator_identities(self):
        # _pos_id / _neg_id have integer arms because Int32/Int64 have no .inf. A wrong sentinel
        # loses to (or beats) every real element, so the result is off by one identity -- visible
        # only on an integer reduction, which this drives directly. int32 is the widest integer
        # this stack wraps; the Int64 arm is covered when that dtype is served.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        x = torch.randint(-(2**20), 2**20, (32, 256), device="cuda", dtype=torch.int32)
        for trait, ref in ((T.AMaxOps, x.amax(dim=1)), (T.AMinOps, x.amin(dim=1))):
            with self.subTest(trait=trait.__name__):
                (got,) = rt.reduce_row_tile(
                    trait(acc=cutlass.Int32), f"int_{trait.__name__}", x, [torch.int32]
                )
                self.assertEqual(got, ref)


if __name__ == "__main__":
    run_tests()
