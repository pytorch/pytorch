# Owner(s): ["module: dsl-native-ops"]
#
# Smoke tests for the shared vectorized row kernel: contiguous reduction, one rolled
# kernel per vector/config class, output/partial paths, and launch/addressing invariants.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelRowTile(TestCase):
    def test_reduce_row_tile(self):
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile, traits as T

        x = torch.randn(128, 512, device="cuda")
        (out,) = kernel_rowtile.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [torch.float32]
        )
        self.assertEqual(out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)

    def test_one_kernel_per_vec_class(self):
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile, traits as T

        trait = T.SumOps(acc=cutlass.Float32)
        kernel_rowtile._CACHE.clear()
        # One runtime-N kernel serves this fp32 vec/config class.
        for n in (2048, 2052, 2056, 2060, 2064):
            x = torch.randn(64, n, device="cuda")
            (out,) = kernel_rowtile.reduce_row_tile(
                trait, "vecclass", x, [torch.float32]
            )
            self.assertEqual(out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)
        # Filter this key because reference x.sum compiles its own plan.
        mine = [k for k in kernel_rowtile._CACHE if "vecclass" in k]
        self.assertEqual(
            len(mine),
            1,
            f"expected ONE compiled kernel for the vec class, got {sorted(mine)}",
        )

    def test_two_output_trait(self):
        # nouts=2 stores values and indices projected from one accumulator.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

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
        # final=False stores raw accumulators; Welford's count must equal the row length.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        x = torch.randn(32, 256, device="cuda")
        trait = T.WelfordOps(acc=cutlass.Float32)
        parts = rt.reduce_row_tile(
            trait, "smoke_partials", x, [torch.float32] * trait.nfields, final=False
        )
        self.assertEqual(len(parts), trait.nfields)
        self.assertEqual(parts[0], x.mean(dim=1), atol=1e-5, rtol=1e-5)
        self.assertEqual(parts[2], torch.full((32,), 256.0, device="cuda"))

    def test_single_row_config_rungs_are_valid(self):
        # Reduce-all widens its lone row only to a legal tree/block rung; a computed width
        # returned wrong variance. Sweep because the widening cutoff is measured.
        from torch._native.ops.reductions import kernel_rowtile as rt

        widened = 0
        for n in (32, 64, 96, 100, 128, 200, 400, 1024, 2048, 4096, 16384, 1 << 20):
            for bits in (16, 32, 64):
                cfg = rt.single_row_config(n, bits)
                if cfg is None:  # ladder stands or row cannot feed a warp
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
                    self.assertGreater(cfg.tpr, rt.row_config(n, bits).tpr)
        self.assertGreater(
            widened, 0, "nothing was widened -- the sweep has gone stale"
        )
        # A row too narrow to feed one warp keeps the ladder's pick.
        self.assertIsNone(rt.single_row_config(32, 32))

    def test_oneshot_gate_bounds_loads_not_just_smem(self):
        # One-shot must reject both oversized rows and excessive per-thread loads.
        import torch
        from torch._native.ops.reductions import kernel_general as kg

        self.assertTrue(kg._oneshot_ok(torch.empty(1, 4096, device="cuda")))
        # Prime N collapses vec to 1 and exceeds the load bound despite fitting smem.
        self.assertFalse(kg._oneshot_ok(torch.empty(1, 65537, device="cuda")))
        # Wide enough to blow the smem budget outright.
        self.assertFalse(kg._oneshot_ok(torch.empty(1, 1 << 22, device="cuda")))

    def test_absmax_absmin_propagate_nan(self):
        # Abs extrema model vector_norm(+/-inf), which propagates NaN. Vary its fold
        # position and leave clean rows to check values too.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        M, N = 64, 512
        half = M // 2
        x = torch.randn(M, N, device="cuda")
        rows = torch.arange(half, device="cuda")
        x[rows, rows * (N // half)] = float("nan")  # a distinct column per row
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
                self.assertEqual(got[half:], want[half:])

    def test_welford_divisor_clamps_at_zero(self):
        # correction >= n divides by zero (+inf like ATen), never a negative denominator.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

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

    def test_welford_agrees_with_aten_on_infinities(self):
        # Without ATen's zero-count guard, padded identity plus an infinite mean yields an
        # intermediate NaN. ATen variance is already nonfinite; avoid two innermost selects
        # and pin agreement instead.
        import math

        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        # Both Ns pad lanes; N=3 uses only 3 of 32.
        for n in (3, 127):
            for where in ("first", "last", "both"):
                with self.subTest(n=n, inf_at=where):
                    x = torch.randn(8, n, device="cuda")
                    if where in ("first", "both"):
                        x[:, 0] = math.inf
                    if where in ("last", "both"):
                        x[:, -1] = -math.inf
                    (got,) = rt.reduce_row_tile(
                        T.WelfordOps(acc=cutlass.Float32),
                        f"welford_inf{n}",
                        x,
                        [torch.float32],
                    )
                    want = x.var(dim=1)
                    # Check NaN masks so equal_nan cannot accept NaN in place of inf.
                    self.assertEqual(got.isnan(), want.isnan())
                    finite = ~got.isnan()
                    self.assertEqual(got[finite], want[finite])

    def test_integer_accumulator_identities(self):
        # Int accumulators lack .inf; a wrong sentinel loses to every element and leaks
        # the identity into the result.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        x = torch.randint(-(2**20), 2**20, (32, 256), device="cuda", dtype=torch.int32)
        for trait, ref in ((T.AMaxOps, x.amax(dim=1)), (T.AMinOps, x.amin(dim=1))):
            with self.subTest(trait=trait.__name__):
                (got,) = rt.reduce_row_tile(
                    trait(acc=cutlass.Int32), f"int_{trait.__name__}", x, [torch.int32]
                )
                self.assertEqual(got, ref)

    def _sum_trait(self):
        # Import locally so this module loads without the DSL.
        import cutlass

        from torch._native.ops.reductions import traits as T

        return T.SumOps(acc=cutlass.Float32)

    def test_narrow_row_one_thread_per_row(self):
        # tpr=1 assigns each row to one thread without lane merging.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile, traits as T

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
        # TMA rotates smem reads to avoid bank conflicts; an index trait verifies that
        # rotated values retain their logical columns.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile, traits as T

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

    def test_narrow_row_and_tma_gates(self):
        # Pin the measured narrow-row tiers and TMA's direct-load stride cliff.
        from torch._native.ops.reductions import kernel_rowtile as rt

        # Width ceiling applies at any M.
        self.assertFalse(rt.narrow_row(rt._MAX_NARROW_N + 1, 4, 1 << 20))
        # Larger M permits larger per-thread chunk budgets.
        self.assertFalse(
            rt.narrow_row(128, 4, 1024)
        )  # below the smallest rung's row count
        self.assertTrue(rt.narrow_row(16, 4, 1 << 20))  # 4 chunks, plenty of rows
        for min_rows, budget in rt._CHUNK_LADDER:
            n = budget * 4  # fp32: vec=4, so chunks == n // 4 == budget exactly
            with self.subTest(min_rows=min_rows, budget=budget):
                self.assertTrue(rt.narrow_row(n, 4, min_rows))
                self.assertFalse(
                    rt.narrow_row(n + 4, 4, min_rows), "budget did not bite"
                )
        # TMA requires fp32, power-of-two N, and a lane stride of at least 128 bytes.
        self.assertFalse(
            rt.tma_ok(16, 4, 1 << 20)
        )  # 64B lane stride: direct load is at SOL
        self.assertTrue(rt.tma_ok(32, 4, 1 << 20))  # 128B: the cliff
        self.assertFalse(rt.tma_ok(48, 4, 1 << 20))  # not a power of two
        self.assertFalse(
            rt.tma_ok(32, 2, 1 << 20)
        )  # bf16 does not map one element per 4-byte bank

    def test_narrow_row_scalar_vec(self):
        # Exercise scalar and short-vector narrow loads.
        from torch._native.ops.reductions import kernel_rowtile as rt

        for n in (1, 2, 3, 5, 7):
            x = torch.randn(1 << 16, n, device="cuda")
            with self.subTest(n=n):
                # fp32 vec=gcd(N, 4), so these loads are one or two elements wide.
                self.assertLess(rt.tile.vec_size(n, 4), 4)
                (out,) = rt.reduce_row_tile(
                    self._sum_trait(), f"narrow_vec{n}", x, [torch.float32], tpr=1
                )
                self.assertEqual(
                    out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
                )

    def test_narrow_row_ragged_m(self):
        # Nonmultiple M exercises the partial tile and TMA zero-fill.
        from torch._native.ops.reductions import kernel_rowtile as rt

        for m in (1, 3, 8191, 65537):
            for n in (16, 32):
                x = torch.randn(m, n, device="cuda")
                with self.subTest(m=m, n=n):
                    (out,) = rt.reduce_row_tile(
                        self._sum_trait(), f"ragged_m{n}", x, [torch.float32], tpr=1
                    )
                    self.assertEqual(
                        out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
                    )

    def test_tma_second_call_rebinds_the_descriptor(self):
        # A cached plan excludes M, but each call must bind TMA to its new pointer and row count.
        from torch._native.ops.reductions import kernel_rowtile as rt

        n = 32
        self.assertTrue(rt.tma_ok(n, 4, 1 << 20), "shape no longer takes the TMA path")
        first = torch.randn(4096, n, device="cuda")
        (a,) = rt.reduce_row_tile(
            self._sum_trait(), "tma_rebind", first, [torch.float32], tpr=1, use_tma=True
        )
        self.assertEqual(a, first.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)
        second = torch.randn(4097, n, device="cuda")  # new pointer AND a new M
        (b,) = rt.reduce_row_tile(
            self._sum_trait(),
            "tma_rebind",
            second,
            [torch.float32],
            tpr=1,
            use_tma=True,
        )
        self.assertEqual(b, second.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)

    def test_one_thread_per_row_is_trait_agnostic(self):
        # No lane merge lets tpr=1 serve three-field and two-output traits.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        x = torch.randn(1 << 16, 16, device="cuda")
        (var,) = rt.reduce_row_tile(
            T.WelfordOps(correction=1, acc=cutlass.Float32),
            "tpr1_welford",
            x,
            [torch.float32],
            tpr=1,
        )
        self.assertEqual(var, x.var(dim=1), atol=1e-4, rtol=1e-4)
        lo, hi = rt.reduce_row_tile(
            T.AMinMaxOps(acc=cutlass.Float32),
            "tpr1_aminmax",
            x,
            [torch.float32, torch.float32],
            nouts=2,
            tpr=1,
        )
        want = torch.aminmax(x, dim=1)
        self.assertEqual(lo, want.min)
        self.assertEqual(hi, want.max)

    def test_dispatcher_takes_the_narrow_arm(self):
        # One-shot is numerically identical; tpr=1 and a consulted TMA gate identify this arm.
        from unittest import mock

        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits as T,
        )

        m, n = 1 << 20, 32
        self.assertTrue(rt.narrow_row(n, 4, m), "the gate no longer admits this shape")
        x = torch.randn(m, n, device="cuda")
        real = rt.reduce_row_tile
        with (
            mock.patch.object(rt, "reduce_row_tile", wraps=real) as served,
            mock.patch.object(rt, "tma_ok", wraps=rt.tma_ok) as gate,
        ):
            got = kg.reduce_dim(
                T.SumOps(acc=cutlass.Float32), "disp_narrow", x, -1, torch.float32
            )
        self.assertEqual(served.call_args.kwargs.get("tpr"), 1, "not the narrow arm")
        self.assertTrue(gate.called, "the use_tma auto-derivation never ran")
        self.assertEqual(got, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)

    def test_use_tma_rejects_a_non_power_of_two_row(self):
        # The rotation mask requires power-of-two N; forced N=24 silently erred by 13.2,
        # so validate it even when caller-set use_tma bypasses tma_ok.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        self.assertFalse(rt.tma_ok(24, 4, 256), "the gate should decline this N")
        x = torch.randn(256, 24, device="cuda")
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            rt.reduce_row_tile(
                T.SumOps(acc=cutlass.Float32),
                "tma_nonpo2",
                x,
                [torch.float32],
                tpr=1,
                use_tma=True,
            )

    def test_tma_gate_does_not_requery_the_device(self):
        # Per-launch tma_ok must use memoized capabilities; raw lookup costs ~1.3us.
        from unittest.mock import patch

        from torch._native.ops.reductions import kernel_rowtile as rt

        dev = torch.device("cuda")
        self.assertTrue(rt.tma_ok(32, 4, 1 << 20, dev))  # warms the caps cache
        with patch("torch.cuda.get_device_properties") as props:
            for _ in range(4):
                rt.tma_ok(32, 4, 1 << 20, dev)
        self.assertEqual(props.call_count, 0)

    def test_gapped_rows_are_addressed_at_runtime(self):
        # Dynamic extents must handle unit inner stride with a gapped row pitch.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        x = torch.randn(4, 512, device="cuda")[::2]
        (got,) = rt.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32), "gapped", x, [torch.float32]
        )
        self.assertEqual(got, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5)

    def test_misaligned_base_is_served_after_an_aligned_call(self):
        # Storage offset can underalign a contiguous row. Run aligned first to ensure the
        # cache key does not reuse its wider claim and fault.
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        trait = T.SumOps(acc=cutlass.Float32)
        aligned = torch.randn(2, 512, device="cuda")
        (first,) = rt.reduce_row_tile(trait, "align_reuse", aligned, [torch.float32])
        ref = aligned.double().sum(dim=1).float()
        self.assertEqual(first, ref, atol=1e-5, rtol=1e-5)
        for skip in (1, 2):
            with self.subTest(skip=skip):
                x = torch.randn(2 * 512 + skip, device="cuda")[skip:].view(2, 512)
                (got,) = rt.reduce_row_tile(trait, "align_reuse", x, [torch.float32])
                self.assertEqual(
                    got, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
                )

    def test_non_power_of_two_warp_count_is_rejected(self):
        # Three warps would drop the third partial (256 instead of 384 at N=384).
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.ones(8, 384, device="cuda")
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            rt.reduce_row_tile(trait, "nw3", x, [torch.float32], tpr=96, nt=96)
        # The neighbouring power-of-two width is served, and correctly.
        (got,) = rt.reduce_row_tile(trait, "nw2", x, [torch.float32], tpr=64, nt=64)
        self.assertEqual(got, torch.full((8,), 384.0, device="cuda"))


if __name__ == "__main__":
    run_tests()
