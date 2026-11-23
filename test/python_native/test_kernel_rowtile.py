# Owner(s): ["module: dsl-native-ops"]
#
# Smoke tests for the shared vectorized row kernel: contiguous reduction, one rolled
# kernel per vector/config class, output/partial paths, and launch/addressing invariants.

import math
import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass

from torch._native.ops.reductions import (
    kernel_general as kg,
    kernel_rowtile as rt,
    tile,
    traits as T,
)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestKernelRowTile(TestCase):
    @parametrize("shape", ((2**31, 1), (1, 2**31)))
    def test_shape_must_fit_int32(self, shape):
        x = mock.Mock(shape=shape, is_cuda=True)
        x.dim.return_value = 2
        x.stride.return_value = 1
        with self.assertRaisesRegex(AssertionError, "M and N < 2\\^31"):
            rt.reduce_row_tile(mock.sentinel.trait, "int32_shape", x, [torch.float32])

    def test_reduce_row_tile(self):
        x = torch.randn(128, 512, device="cuda")
        (out,) = rt.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [torch.float32]
        )
        torch.testing.assert_close(
            out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
        )

    # (65536, 128) takes one thread per row, staged through TMA.
    @parametrize("rows,columns,tpr", ((64, 1000, None), (65536, 128, 1)))
    def test_fp32_row_sum_error_is_no_worse_than_aten(self, rows, columns, tpr):
        """Sum fp32 rows with no more mean absolute error against the fp64 sum than
        ATen, summed over a few seeds."""
        trait, f32 = T.SumOps(acc=cutlass.Float32), [torch.float32]
        kw = {"threads_per_row": tpr, "use_tma": tpr == 1}
        ours = aten = 0.0
        for seed in range(8):
            gen = torch.Generator("cuda").manual_seed(seed)
            x = torch.randn(rows, columns, device="cuda", generator=gen)
            (got,) = rt.reduce_row_tile(trait, "fp32_error", x, f32, **kw)
            with torch.backends.python_native.cutedsl.disabled():
                ref = x.sum(1)
            want = x.double().sum(1)
            ours += (got.double() - want).abs().mean().item()
            aten += (ref.double() - want).abs().mean().item()
        self.assertLessEqual(ours, aten, f"CuTeDSL {ours:.3e} vs ATen {aten:.3e}")

    # 100 columns end in a partial wave of vectors.
    @parametrize("columns", (96, 100))
    @parametrize("dist", ("randn", "rand"))
    def test_fp32_one_thread_rows_match_aten(self, columns, dist):
        """One thread per row folds rows narrower than 128 elements in ATen's
        association, so fp32 sums match it bitwise."""
        gen = torch.Generator("cuda").manual_seed(0)
        x = getattr(torch, dist)(65536, columns, device="cuda", generator=gen)
        (got,) = rt.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32),
            "fp32_one_thread",
            x,
            [torch.float32],
            threads_per_row=1,
            use_tma=False,
        )
        with torch.backends.python_native.cutedsl.disabled():
            ref = x.sum(1)
        self.assertEqual(got, ref, atol=0, rtol=0)

    @parametrize("name", ("sum", "mean", "nansum"))
    def test_tma_rows_of_negative_zeros_sum_to_positive_zero(self, name):
        """TMA rows fold leaves, which keep -0.0; ATen's 0.0 + x gives +0.0."""
        if not rt.tma_ok(128, 4, 65536, torch.device("cuda")):
            self.skipTest("TMA path not applicable on this device")
        ops = {"sum": T.SumOps, "mean": T.MeanOps, "nansum": T.NanSumOps}
        op = ops[name](acc=cutlass.Float32)
        x = -torch.zeros(65536, 128, device="cuda")
        kw = {"threads_per_row": 1, "use_tma": True}
        (got,) = rt.reduce_row_tile(op, f"tma_nz_{name}", x, [x.dtype], **kw)
        self.assertEqual(got.view(torch.int32), torch.zeros(65536, dtype=torch.int32))

    def test_narrow_one_thread_row_chains(self):
        f32 = cutlass.Float32
        sum_trait = T.SumOps(acc=f32)

        def chains(trait, n, **kw):
            kw.setdefault("threads_per_row", 1)
            return tile.TileReduce(trait, f32, "row", n, **kw).row_chains

        self.assertLess(chains(sum_trait, 16), tile.WARP)
        # Multi-field, norm and abs-extremum traits would change bits on fewer chains;
        # they, TMA, ragged and multi-thread rows keep WARP.
        self.assertEqual(chains(T.ArgMaxOps(acc=f32), 8), 32)
        self.assertEqual(chains(T.NormOps(2, acc=f32), 8), 32)
        self.assertEqual(chains(T.AbsMaxOps(acc=f32), 8), 32)
        self.assertEqual(chains(sum_trait, 16, use_tma=True), 32)
        self.assertEqual(chains(sum_trait, 15, ragged_vector=True), 32)
        self.assertEqual(chains(sum_trait, 16, threads_per_row=32), 32)

    @parametrize(
        "dtype, columns",
        (
            (torch.float32, 1),
            (torch.float32, 3),
            (torch.float32, 16),
            (torch.float32, 17),
            (torch.float32, 20),
            (torch.float32, 22),
            (torch.float32, 24),
            (torch.float32, 31),
            (torch.bfloat16, 8),
            (torch.bfloat16, 16),
            (torch.bfloat16, 24),
            (torch.float64, 8),
            (torch.float64, 24),
        ),
    )
    def test_narrow_one_thread_rows_match_warp_chains(self, dtype, columns):
        """Narrow one-thread rows match the row padded with identities to WARP elements
        bitwise, signed zeros and NaNs included, plus a subnormal sum for fp32 and bf16
        (fp64's Float64 accumulator keeps that sum normal). Widths 16 and 24
        (fp32), 8, 16 and 24 (bf16) and 8 and 24 (fp64) reach the leaf branch; 1, 3, 17,
        20 and 22 fold only the masked tail, and 31 keeps WARP chains."""
        M, inf = 4096, float("inf")
        gen = torch.Generator("cuda").manual_seed(0)
        x = torch.randn(M, columns, device="cuda", generator=gen)
        signs = torch.randint(0, 2, (M // 2, columns), device="cuda", generator=gen)
        x[: M // 2] = torch.tensor([0.0, -0.0], device="cuda")[signs]
        # An all -0.0 row, whose sum sign only the final combine with init restores.
        x[64] = -0.0
        x[M // 2 : M // 2 + 64, columns // 2] = float("nan")
        x[-64:, 0], x[-64:, -1] = inf, -inf
        x[:64, 0], x[:64, -1] = -1.5e-38, 1e-38
        x = x.to(dtype)
        kw = {"threads_per_row": 1, "use_tma": False}
        # fp64 accumulates and stores in Float64, as the dispatcher does.
        wide = dtype == torch.float64
        acc = cutlass.Float64 if wide else cutlass.Float32
        out, bits = (
            (torch.float64, torch.int64) if wide else (torch.float32, torch.int32)
        )
        for name, trait, pad in (
            ("sum", T.SumOps, 0.0),
            ("prod", T.ProdOps, 1.0),
            ("amax", T.AMaxOps, -inf),
            ("amin", T.AMinOps, inf),
            ("nansum", T.NanSumOps, 0.0),
            ("absmax", T.AbsMaxOps, 0.0),
            ("all", T.AllOps, 1.0),
            ("any", T.AnyOps, 0.0),
            ("count_nonzero", T.CountNonzeroOps, 0.0),
        ):
            padded = torch.full((M, tile.WARP), pad, device="cuda", dtype=dtype)
            padded[:, :columns] = x
            op = trait(acc=acc)
            (got,) = rt.reduce_row_tile(op, f"narrow_{name}", x, [out], **kw)
            (want,) = rt.reduce_row_tile(op, f"narrow_{name}", padded, [out], **kw)
            with self.subTest(op=name):
                self.assertEqual(got.view(bits), want.view(bits))
            if name == "sum":
                with torch.backends.python_native.cutedsl.disabled():
                    ref = x.double().sum(1).to(out)
                self.assertEqual(want, ref, atol=1e-5, rtol=1e-5)
                # Padding would change a mean's count, so compare it with this sum's
                # quotient; the kernel's division is not correctly rounded.
                mean = T.MeanOps(acc=acc)
                (got,) = rt.reduce_row_tile(mean, "narrow_mean", x, [out], **kw)
                with self.subTest(op="mean"):
                    self.assertEqual(got, want / columns)

    def test_one_kernel_per_vec_class(self):
        trait = T.SumOps(acc=cutlass.Float32)
        rt._CACHE.clear()
        # One runtime-N kernel serves this fp32 vec/config class.
        for n in (2048, 2052, 2056, 2060, 2064):
            x = torch.randn(64, n, device="cuda")
            (out,) = rt.reduce_row_tile(trait, "vecclass", x, [torch.float32])
            torch.testing.assert_close(
                out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
            )
        # Filter this key because reference x.sum compiles its own plan.
        mine = [k for k in rt._CACHE if "vecclass" in k]
        self.assertEqual(
            len(mine),
            1,
            f"expected ONE compiled kernel for the vec class, got {sorted(mine)}",
        )

    @unittest.skipIf(
        torch.cuda.device_count() < 2 or torch.cuda.get_device_capability(1)[0] < 9,
        "requires a second Hopper+ CUDA device",
    )
    def test_plans_are_keyed_by_device(self):
        trait = T.SumOps(acc=cutlass.Float32)
        rt._CACHE.clear()
        for device in ("cuda:0", "cuda:1"):
            x = torch.randn(64, 512, device=device)
            with torch.cuda.device(x.device):
                (out,) = rt.reduce_row_tile(
                    trait, "bydevice", x, [torch.float32], order="linear"
                )
            torch.testing.assert_close(
                out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
            )
        # Row-tile keys end their fixed fields with the device index.
        mine = [k for k in rt._CACHE if "bydevice" in k]
        self.assertEqual(sorted(k[5] for k in mine), [0, 1])

    def test_plans_key_the_device_index(self):
        # The index follows the alignment in row-tile keys and the destination layout in
        # inner-tree keys; without it, those slots would hold the next field.
        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.randn(64, 512, device="cuda")
        out = torch.empty(64, device="cuda")
        (got,) = rt.reduce_row_tile(trait, "bykey", x, [torch.float32], order="linear")
        self.assertTrue(rt.reduce_row_itree(trait, "bykey", x, out))
        for result in (got, out):
            torch.testing.assert_close(
                result, x.double().sum(1).float(), atol=1e-5, rtol=1e-5
            )
        slots = {"rowtile": 5, "rowitree": 7}
        devices = {(k[0], k[slots[k[0]]]) for k in rt._CACHE if k[1] == "bykey"}
        index = x.get_device()
        self.assertEqual(devices, {("rowtile", index), ("rowitree", index)})

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 CUDA devices")
    def test_launchers_reject_a_non_current_device(self):
        # Both entry points check before compiling, so nothing is cached.
        x = torch.randn(64, 512, device="cuda:1")
        trait = T.SumOps(acc=cutlass.Float32)
        out = torch.empty(x.shape[0], device=x.device)
        self.assertIsNotNone(rt.itree_plan(x.shape[1], x.shape[0], x.element_size()))
        before = set(rt._CACHE)
        with torch.cuda.device(0):
            with self.assertRaisesRegex(AssertionError, "is current"):
                rt.reduce_row_tile(trait, "off", x, [torch.float32], order="linear")
            with self.assertRaisesRegex(AssertionError, "is current"):
                rt.reduce_row_itree(trait, "off", x, out)
        self.assertEqual(set(rt._CACHE), before)

    def test_itree_output_shares_the_input_device(self):
        x = torch.randn(64, 512, device="cuda")
        trait = T.SumOps(acc=cutlass.Float32)
        with self.assertRaisesRegex(AssertionError, "out on cpu"):
            rt.reduce_row_itree(trait, "off_out", x, torch.empty(64))

    def test_two_output_trait(self):
        # nouts=2 stores values and indices projected from one accumulator.
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
        x = torch.randn(32, 256, device="cuda")
        trait = T.WelfordOps(acc=cutlass.Float32)
        parts = rt.reduce_row_tile(
            trait, "smoke_partials", x, [torch.float32] * trait.nfields, final=False
        )
        self.assertEqual(len(parts), trait.nfields)
        with torch.backends.python_native.cutedsl.disabled():
            want = x.mean(dim=1)
        torch.testing.assert_close(parts[0], want, atol=1e-5, rtol=1e-5)
        self.assertEqual(parts[2], torch.full((32,), 256.0, device="cuda"))

    def test_single_row_config_rungs_are_valid(self):
        # Reduce-all widens its lone row only to a legal tree/block rung; a computed width
        # returned wrong variance. Sweep because the widening cutoff is measured.
        widened = 0
        for n in (32, 64, 96, 100, 128, 200, 400, 1024, 2048, 4096, 16384, 1 << 20):
            for bits in (16, 32, 64):
                cfg = rt.single_row_config(n, bits)
                if cfg is None:  # ladder stands or row cannot feed a warp
                    continue
                widened += 1
                with self.subTest(n=n, bits=bits):
                    self.assertEqual(
                        cfg.threads_per_row & (cfg.threads_per_row - 1),
                        0,
                        "threads_per_row must be a power of two",
                    )
                    self.assertEqual(
                        cfg.threads_per_row % rt.WARP,
                        0,
                        "threads_per_row must be a warp multiple",
                    )
                    self.assertIn(cfg.threads_per_row, rt._THREADS_PER_ROW_RUNGS)
                    self.assertLessEqual(cfg.threads_per_row, cfg.threads_per_block)
                    self.assertEqual(
                        cfg.threads_per_block % cfg.threads_per_row,
                        0,
                        "threads_per_block must hold whole rows",
                    )
                    self.assertGreater(
                        cfg.threads_per_row, rt.row_config(n, bits).threads_per_row
                    )
        self.assertGreater(
            widened, 0, "nothing was widened -- the sweep has gone stale"
        )
        # A row too narrow to feed one warp keeps the ladder's pick.
        self.assertIsNone(rt.single_row_config(32, 32))

    def test_oneshot_gate_bounds_loads_not_just_smem(self):
        # One-shot must reject both oversized rows and excessive per-thread loads.
        self.assertTrue(kg._oneshot_ok(torch.empty(1, 4096, device="cuda")))
        # Prime N collapses vec to 1 and exceeds the load bound despite fitting smem.
        self.assertFalse(kg._oneshot_ok(torch.empty(1, 65537, device="cuda")))
        # Wide enough to blow the smem budget outright.
        self.assertFalse(kg._oneshot_ok(torch.empty(1, 1 << 22, device="cuda")))

    def test_absmax_absmin_propagate_nan(self):
        # Abs extrema model vector_norm(+/-inf), which propagates NaN. Vary its fold
        # position and leave clean rows to check values too.
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
        x = torch.randint(-(2**20), 2**20, (32, 256), device="cuda", dtype=torch.int32)
        for trait, ref in ((T.AMaxOps, x.amax(dim=1)), (T.AMinOps, x.amin(dim=1))):
            with self.subTest(trait=trait.__name__):
                (got,) = rt.reduce_row_tile(
                    trait(acc=cutlass.Int32), f"int_{trait.__name__}", x, [torch.int32]
                )
                self.assertEqual(got, ref)

    def _sum_trait(self):
        return T.SumOps(acc=cutlass.Float32)

    def test_narrow_row_one_thread_per_row(self):
        # threads_per_row=1 assigns each row to one thread without lane merging.
        x = torch.randn(8192, 16, device="cuda")
        (out,) = rt.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32),
            "narrow",
            x,
            [torch.float32],
            threads_per_row=1,
            use_tma=False,
        )
        with torch.backends.python_native.cutedsl.disabled():
            want = x.sum(dim=1)
        torch.testing.assert_close(out, want, atol=1e-3, rtol=1e-3)

    def test_tma_staged_narrow_row_argmax(self):
        # TMA rotates smem reads to avoid bank conflicts; an index trait verifies that
        # rotated values retain their logical columns.
        if not rt.tma_ok(32, 4, 65536, torch.device("cuda")):
            self.skipTest("TMA path not applicable on this device")
        x = torch.randn(65536, 32, device="cuda")
        (idx,) = rt.reduce_row_tile(
            T.ArgMaxOps(acc=cutlass.Float32),
            "narrow_tma",
            x,
            [torch.int32],
            threads_per_row=1,
            use_tma=True,
        )
        self.assertEqual(idx, x.argmax(dim=1).to(torch.int32))

    def test_narrow_row_and_tma_gates(self):
        # Pin the measured narrow-row tiers and TMA's direct-load stride cliff.
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
        for n in (1, 2, 3, 5, 7):
            x = torch.randn(1 << 16, n, device="cuda")
            with self.subTest(n=n):
                # fp32 vec=gcd(N, 4), so these loads are one or two elements wide.
                self.assertLess(rt.tile.vec_size(n, 4), 4)
                (out,) = rt.reduce_row_tile(
                    self._sum_trait(),
                    f"narrow_vec{n}",
                    x,
                    [torch.float32],
                    threads_per_row=1,
                )
                torch.testing.assert_close(
                    out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
                )

    @parametrize(
        "dtype,columns",
        (
            (torch.float32, 100),
            (torch.float32, 124),
            (torch.float16, 200),
            (torch.float32, 34),
            (torch.float64, 64),
        ),
    )
    def test_one_thread_row_partial_group(self, dtype, columns):
        # 25 whole vectors leave one past the last full group of 32 elements, and 31 a
        # seven-vector tail. fp32 34 and fp64 64 take vec 2: one whole wave and a
        # one-vector tail, and two whole waves (Float64 accumulator). Integer sums are
        # exact, so a dropped or repeated vector shows bitwise.
        x = torch.randint(1, 9, (4096, columns), device="cuda").to(dtype)
        wide = dtype == torch.float64
        (got,) = rt.reduce_row_tile(
            T.SumOps(acc=cutlass.Float64) if wide else self._sum_trait(),
            "tpr1_groups",
            x,
            [dtype if wide else torch.float32],
            threads_per_row=1,
            use_tma=False,
        )
        self.assertEqual(got, x.double().sum(dim=1).to(got.dtype), atol=0, rtol=0)

    def test_narrow_row_ragged_m(self):
        # Nonmultiple M exercises the partial tile and TMA zero-fill.
        for m in (1, 3, 8191, 65537):
            for n in (16, 32):
                x = torch.randn(m, n, device="cuda")
                with self.subTest(m=m, n=n):
                    (out,) = rt.reduce_row_tile(
                        self._sum_trait(),
                        f"ragged_m{n}",
                        x,
                        [torch.float32],
                        threads_per_row=1,
                    )
                    torch.testing.assert_close(
                        out, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
                    )

    def test_tma_second_call_rebinds_the_descriptor(self):
        # A cached plan must bind TMA to each call's pointer, row count, and row stride.
        n = 32
        if not rt.tma_ok(n, 4, 1 << 20, torch.device("cuda")):
            self.skipTest("TMA path not applicable on this device")
        first = torch.randn(4096, n, device="cuda")
        (a,) = rt.reduce_row_tile(
            self._sum_trait(),
            "tma_rebind",
            first,
            [torch.float32],
            threads_per_row=1,
            use_tma=True,
        )
        torch.testing.assert_close(
            a, first.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
        )
        storage = torch.full((4097, n + 4), 1000.0, device="cuda")
        second = storage[:, :n]
        second.normal_()
        (b,) = rt.reduce_row_tile(
            self._sum_trait(),
            "tma_rebind",
            second,
            [torch.float32],
            threads_per_row=1,
            use_tma=True,
        )
        torch.testing.assert_close(
            b, second.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
        )

    def test_forced_tma_rejects_misaligned_input(self):
        n = 32
        x = torch.randn(256 * n + 1, device="cuda")[1:].view(256, n)
        with self.assertRaisesRegex(ValueError, "16-byte aligned input"):
            rt.reduce_row_tile(
                self._sum_trait(),
                "tma_misaligned",
                x,
                [torch.float32],
                threads_per_row=1,
                use_tma=True,
            )

    def test_forced_tma_rejects_misaligned_row_stride(self):
        # Whole 8-byte vectors at a pitch that is not 16-byte aligned.
        x = torch.randn(256, 38, device="cuda")[:, :34]
        with self.assertRaisesRegex(ValueError, "16-byte aligned row stride"):
            rt.reduce_row_tile(
                self._sum_trait(),
                "tma_misaligned_stride",
                x,
                [torch.float32],
                threads_per_row=1,
                use_tma=True,
            )

    def test_one_thread_per_row_is_trait_agnostic(self):
        # No lane merge lets threads_per_row=1 serve three-field and two-output traits.
        x = torch.randn(1 << 16, 16, device="cuda")
        (var,) = rt.reduce_row_tile(
            T.WelfordOps(correction=1, acc=cutlass.Float32),
            "tpr1_welford",
            x,
            [torch.float32],
            threads_per_row=1,
        )
        torch.testing.assert_close(var, x.var(dim=1), atol=1e-4, rtol=1e-4)
        lo, hi = rt.reduce_row_tile(
            T.AMinMaxOps(acc=cutlass.Float32),
            "tpr1_aminmax",
            x,
            [torch.float32, torch.float32],
            nouts=2,
            threads_per_row=1,
        )
        want = torch.aminmax(x, dim=1)
        self.assertEqual(lo, want.min)
        self.assertEqual(hi, want.max)

    def test_dispatcher_takes_the_narrow_arm(self):
        # One-shot is numerically identical; threads_per_row=1 and a consulted TMA gate identify this arm.
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
        self.assertEqual(
            served.call_args.kwargs.get("threads_per_row"), 1, "not the narrow arm"
        )
        self.assertTrue(gate.called, "the use_tma auto-derivation never ran")
        torch.testing.assert_close(
            got, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
        )

    def test_use_tma_rejects_a_non_power_of_two_row(self):
        # The rotation mask requires power-of-two N; forced N=24 silently erred by 13.2,
        # so validate it even when caller-set use_tma bypasses tma_ok.
        self.assertFalse(rt.tma_ok(24, 4, 256), "the gate should decline this N")
        x = torch.randn(256, 24, device="cuda")
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            rt.reduce_row_tile(
                T.SumOps(acc=cutlass.Float32),
                "tma_nonpo2",
                x,
                [torch.float32],
                threads_per_row=1,
                use_tma=True,
            )

    def test_tma_gate_does_not_requery_the_device(self):
        # Per-launch tma_ok must use memoized capabilities; raw lookup costs ~1.3us.
        dev = torch.device("cuda")
        self.assertTrue(rt.tma_ok(32, 4, 1 << 20, dev))  # warms the caps cache
        with mock.patch("torch.cuda.get_device_properties") as props:
            for _ in range(4):
                rt.tma_ok(32, 4, 1 << 20, dev)
        self.assertEqual(props.call_count, 0)

    def test_gapped_rows_are_addressed_at_runtime(self):
        # Dynamic extents must handle unit inner stride with a gapped row pitch.
        x = torch.randn(4, 512, device="cuda")[::2]
        (got,) = rt.reduce_row_tile(
            T.SumOps(acc=cutlass.Float32), "gapped", x, [torch.float32]
        )
        torch.testing.assert_close(
            got, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
        )

    def test_misaligned_base_is_served_after_an_aligned_call(self):
        # Storage offset can underalign a contiguous row. Run aligned first to ensure the
        # cache key does not reuse its wider claim and fault.
        trait = T.SumOps(acc=cutlass.Float32)
        aligned = torch.randn(2, 512, device="cuda")
        (first,) = rt.reduce_row_tile(trait, "align_reuse", aligned, [torch.float32])
        ref = aligned.double().sum(dim=1).float()
        torch.testing.assert_close(first, ref, atol=1e-5, rtol=1e-5)
        for skip in (1, 2):
            with self.subTest(skip=skip):
                x = torch.randn(2 * 512 + skip, device="cuda")[skip:].view(2, 512)
                (got,) = rt.reduce_row_tile(trait, "align_reuse", x, [torch.float32])
                torch.testing.assert_close(
                    got, x.double().sum(dim=1).float(), atol=1e-5, rtol=1e-5
                )

    def test_non_power_of_two_warp_count_is_rejected(self):
        # Three warps would drop the third partial (256 instead of 384 at N=384).
        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.ones(8, 384, device="cuda")
        with self.assertRaisesRegex(ValueError, "power-of-two"):
            rt.reduce_row_tile(
                trait,
                "nw3",
                x,
                [torch.float32],
                threads_per_row=96,
                threads_per_block=96,
            )
        # The neighbouring power-of-two width is served, and correctly.
        (got,) = rt.reduce_row_tile(
            trait, "nw2", x, [torch.float32], threads_per_row=64, threads_per_block=64
        )
        self.assertEqual(got, torch.full((8,), 384.0, device="cuda"))

    def test_rolled_lane_accumulators_are_each_load_bearing(self):
        """Combine whole-vector lane accumulators: each one holds some row's result,
        index ties across them keep the first column, NaN wins from any of them, and
        Welford counts stay exact."""
        # 4104 fp16 columns are 513 whole 8-wide vectors, so every power-of-two
        # threads_per_row leaves the last wave masked. Index traits keep 4 accumulators
        # and fold column c into accumulator c % 4.
        rows, columns = 33, 4096 + 8
        x = torch.randn(rows, columns, device="cuda", dtype=torch.float16)
        r = torch.arange(rows, device="cuda")
        # Row r plants its maximum in accumulator r % 4 and a tie in the next one.
        # Row 32 plants in vector 512, the masked last wave.
        first = r * 128 + r % 4
        x[r, first] = 10.0
        x[r, first + 5] = 10.0
        nan_rows = torch.arange(4, 8, device="cuda")
        x[nan_rows, first[nan_rows] + 2] = float("nan")
        want = first.clone()
        want[nan_rows] += 2
        finite = torch.ones(rows, dtype=torch.bool, device="cuda")
        finite[nan_rows] = False
        (idx,) = rt.reduce_row_tile(
            T.ArgMaxOps(acc=cutlass.Float32), "rolled_argmax", x, [torch.int32]
        )
        self.assertEqual(idx, want.to(torch.int32))
        vals, idx = rt.reduce_row_tile(
            T.MinDimOps(acc=cutlass.Float32),
            "rolled_min_dim",
            -x,
            [torch.float32, torch.int32],
            nouts=2,
        )
        self.assertEqual(idx, want.to(torch.int32))
        self.assertTrue(vals[nan_rows].isnan().all())
        self.assertEqual(vals[finite], torch.full((rows - 4,), -10.0, device="cuda"))
        y = torch.randn(rows, columns, device="cuda", dtype=torch.float16)
        trait = T.WelfordOps(acc=cutlass.Float32)
        mean, _, count = rt.reduce_row_tile(
            trait, "rolled_welford", y, [torch.float32] * trait.nfields, final=False
        )
        want_count = torch.full((rows,), float(columns), device="cuda")
        self.assertEqual(count, want_count, atol=0, rtol=0)
        self.assertEqual(mean, y.double().mean(dim=1).float(), atol=1e-4, rtol=1e-4)

    def _assert_ragged_plan(self, trait_key, *ragged):
        # Row-tile keys end in TileReduce.cache_sig. Reference signatures locate its N
        # field, which keys N for TMA only, and its ragged vector flag: ragged rows take
        # a runtime width.
        trait, f32 = T.SumOps(acc=cutlass.Float32), cutlass.Float32
        ref = tile.TileReduce(trait, f32, "row", 512).cache_sig
        tma = tile.TileReduce(trait, f32, "row", 512, use_tma=True).cache_sig
        rag = tile.TileReduce(trait, f32, "row", 512, ragged_vector=True).cache_sig
        n_at = tma.index(512)
        (ragged_at,) = [i for i in range(len(ref)) if ref[i] != rag[i]]
        sigs = [k[len(k) - len(ref) :] for k in rt._CACHE if k[1] == trait_key]
        want = {(0, r) for r in ragged}
        self.assertEqual({(sig[n_at], sig[ragged_at]) for sig in sigs}, want)

    @parametrize(
        "dtype,columns",
        (
            (torch.float32, 4097),  # vec 4, 8-warp block merge
            (torch.float16, 4095),  # vec 8, 2 warps x 2 rows per block
            (torch.bfloat16, 1001),  # vec 8, single-warp rows, 4 rows per block
            (torch.float64, 2049),  # vec 2
            (torch.float16, 5),  # prefix clamped to n, no full vector group
            (torch.float16, 11),  # rows with and without one full group
            (torch.float64, 1),  # vec 2, no full vector group
            (torch.float32, 4098),  # even N: residues 0 and 2 only
            (torch.float16, 4096),  # whole vectors: the rolled path
            (torch.float16, 255),  # narrowest B200 ragged-row width
            (torch.bfloat16, 16383),  # widest B200 ragged-row width
        ),
    )
    def test_ragged_vector_edges_and_interior(self, dtype, columns):
        """Combine scalar row edges with aligned vector loads across the interior."""
        # Odd N cycles r * N mod vec through every residue. Integer values make the sum
        # exact, so a dropped or repeated element shows bitwise. 33 rows leave dead
        # threads in the last block where blocks hold several rows; fp32 4097 and 4098
        # and fp64 2049 put one row per block.
        x = torch.randint(1, 9, (33, columns), device="cuda").to(dtype)
        out = torch.float64 if dtype == torch.float64 else torch.float32
        acc = cutlass.Float64 if dtype == torch.float64 else cutlass.Float32
        rt._CACHE.clear()
        (got,) = rt.reduce_row_tile(
            T.SumOps(acc=acc),
            "ragged_vector",
            x,
            [out],
            order="linear",
            ragged_vector=True,
        )
        self.assertEqual(got, x.double().sum(dim=1).to(out), atol=0, rtol=0)
        self._assert_ragged_plan("ragged_vector", columns != 4096)

    @parametrize(
        "dtype,columns,threads_per_row",
        ((torch.float16, 4095, None), (torch.bfloat16, 203, 1)),
    )
    def test_ragged_vector_index_traits(self, dtype, columns, threads_per_row):
        # Plant each row's extrema across the scalar prefix, the first and last vector
        # groups, and the scalar tail so the returned columns check every index source.
        rows = 33
        x = torch.randn(rows, columns, device="cuda", dtype=dtype)
        r = torch.arange(rows, device="cuda")
        head = (r // 2) % 16
        tail = columns - 1 - head
        even = r % 2 == 0
        x[r, torch.where(even, head, tail)] = 10.0
        x[r, torch.where(even, tail, head)] = -10.0
        rt._CACHE.clear()
        (idx,) = rt.reduce_row_tile(
            T.ArgMaxOps(acc=cutlass.Float32),
            "ragged_argmax",
            x,
            [torch.int32],
            threads_per_row=threads_per_row,
            ragged_vector=True,
        )
        self.assertEqual(idx, x.argmax(dim=1).to(torch.int32))
        vals, idx = rt.reduce_row_tile(
            T.MinDimOps(acc=cutlass.Float32),
            "ragged_min_dim",
            x,
            [torch.float32, torch.int32],
            nouts=2,
            threads_per_row=threads_per_row,
            ragged_vector=True,
        )
        want = x.min(dim=1)
        self.assertEqual(vals, want.values.float())
        self.assertEqual(idx, want.indices.to(torch.int32))
        for key in ("ragged_argmax", "ragged_min_dim"):
            self._assert_ragged_plan(key, True)

    def test_ragged_vector_welford_partials(self):
        # Raw Welford counts must equal N exactly; any edge element dropped or repeated
        # moves it.
        rows, columns = 33, 4095
        x = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
        trait = T.WelfordOps(acc=cutlass.Float32)
        rt._CACHE.clear()
        mean, m2, count = rt.reduce_row_tile(
            trait,
            "ragged_welford",
            x,
            [torch.float32] * trait.nfields,
            final=False,
            ragged_vector=True,
        )
        want = torch.full((rows,), float(columns), device="cuda")
        self.assertEqual(count, want, atol=0, rtol=0)
        self._assert_ragged_plan("ragged_welford", True)
        xd = x.double()
        self.assertEqual(mean, xd.mean(dim=1).float(), atol=1e-4, rtol=1e-4)
        var = m2 / (columns - 1)
        self.assertEqual(var, xd.var(dim=1).float(), atol=1e-4, rtol=1e-4)

    @parametrize("dtype", (torch.float32, torch.float64))
    def test_one_thread_ragged_sum_and_welford(self, dtype):
        """One-thread ragged rows fold both edges in lane 0, then the interior into vec
        accumulators (SumOps) or Welford's per-lane accumulators."""
        # 203 columns at vec 4 or 2 cycle the prefix through every residue.
        rows, columns = 33, 203
        kw = {"threads_per_row": 1, "ragged_vector": True}
        rt._CACHE.clear()
        x = torch.randint(1, 9, (rows, columns), device="cuda").to(dtype)
        sum_trait, f32 = T.SumOps(acc=cutlass.Float32), [torch.float32]
        (got,) = rt.reduce_row_tile(sum_trait, "tpr1_ragged_sum", x, f32, **kw)
        self.assertEqual(got, x.double().sum(dim=1).float(), atol=0, rtol=0)
        y = torch.randn(rows, columns, device="cuda", dtype=dtype)
        trait = T.WelfordOps(acc=cutlass.Float32)
        _, m2, count = rt.reduce_row_tile(
            trait, "tpr1_ragged_welford", y, f32 * trait.nfields, final=False, **kw
        )
        want = torch.full((rows,), float(columns), device="cuda")
        self.assertEqual(count, want, atol=0, rtol=0)
        var = m2 / (columns - 1)
        self.assertEqual(var, y.double().var(dim=1).float(), atol=1e-4, rtol=1e-4)
        for key in ("tpr1_ragged_sum", "tpr1_ragged_welford"):
            self._assert_ragged_plan(key, True)

    def test_ragged_vector_plans_share_widths_and_pitches(self):
        # Ragged rows take a runtime width and pitch: one plan per launch class.
        trait = T.SumOps(acc=cutlass.Float32)
        rt._CACHE.clear()
        b = torch.randint(1, 9, (64, 2056), device="cuda").float()
        for x in (b[:, :2049], b[:, :2051], b[:7, :2049].contiguous(), b[:1, :2049], b):
            (got,) = rt.reduce_row_tile(
                trait, "ragged_width", x, [torch.float32], ragged_vector=True
            )
            with torch.backends.python_native.cutedsl.disabled():
                want = x.sum(dim=1)
            self.assertEqual(got, want, atol=0, rtol=0)
        mine = [k for k in rt._CACHE if "ragged_width" in k]
        # b itself is whole vectors at a whole pitch: the rolled path's plan.
        self.assertEqual(
            len(mine), 2, f"expected one ragged and one rolled plan, got {mine}"
        )
        self._assert_ragged_plan("ragged_width", True, False)
        # Whole-vector widths fall back to the rolled path's one plan per vector class.
        for columns in (2048, 2052):
            x = torch.randint(1, 9, (64, columns), device="cuda").float()
            (got,) = rt.reduce_row_tile(
                trait, "ragged_whole", x, [torch.float32], ragged_vector=True
            )
            with torch.backends.python_native.cutedsl.disabled():
                want = x.sum(dim=1)
            self.assertEqual(got, want, atol=0, rtol=0)
        mine = [k for k in rt._CACHE if "ragged_whole" in k]
        self.assertEqual(len(mine), 1, f"expected one plan per vec class, got {mine}")
        self._assert_ragged_plan("ragged_whole", False)

    def test_ragged_vector_rejects_inner_tree_and_tma(self):
        trait = T.SumOps(acc=cutlass.Float32)
        # A whole width: the contract holds for every N, not only ragged ones.
        x = torch.randn(64, 32, device="cuda")
        with self.assertRaisesRegex(ValueError, "require the linear order"):
            rt.reduce_row_tile(
                trait,
                "ragged_itree",
                x,
                [torch.float32],
                order="inner_tree",
                ragged_vector=True,
            )
        # 4096 is whole with a TMA-aligned stride; 4097 is ragged, and the ragged check
        # must fire before TMA's own check rejects its 16388-byte row stride.
        f32 = [torch.float32]
        for n in (4096, 4097):
            x = torch.randn(64, n, device="cuda")
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "cannot use TMA"):
                    rt.reduce_row_tile(
                        trait, "ragged_tma", x, f32, use_tma=True, ragged_vector=True
                    )

    @parametrize(
        "axis,n,kwargs,msg",
        (
            ("col", 4095, {"vec": 8}, "require the row axis"),
            ("row", 4095, {"order": "inner_tree"}, "require the linear order"),
            ("row", 4095, {"use_tma": True}, "cannot use TMA"),
            ("row", 4095, {"combine": True}, "cannot combine partials"),
            ("row", 0, {}, "nonempty rows"),
        ),
        name_fn=lambda axis, n, kwargs, msg: "_".join(
            [axis] + [f"{k}_{v}" for k, v in kwargs.items()] + ["empty"] * (n == 0)
        ),
    )
    def test_ragged_vector_tile_rejects(self, axis, n, kwargs, msg):
        trait = T.SumOps(acc=cutlass.Float32)
        with self.assertRaisesRegex(ValueError, msg):
            rt.tile.TileReduce(
                trait, cutlass.Float16, axis, n, ragged_vector=True, **kwargs
            )

    def test_tma_rejects_empty_rows(self):
        """TMA's one-thread rows raise ValueError for an empty row, which has no box."""
        trait, x = T.SumOps(acc=cutlass.Float32), torch.empty(1, 0, device="cuda")
        kwargs = {"threads_per_row": 1, "use_tma": True}
        with self.assertRaisesRegex(ValueError, "nonempty power-of-two row"):
            rt.reduce_row_tile(trait, "empty", x, [torch.float32], **kwargs)

    def test_one_thread_empty_rows_fold_to_identity(self):
        """Non-TMA one-thread rows fold an empty row to the identity, through both the
        single-field and the multi-field branch of fold_linear_rolled."""
        x = torch.empty(4, 4, device="cuda")[:, :0]
        kw, f32 = {"threads_per_row": 1, "use_tma": False}, [torch.float32]
        zeros = torch.zeros(4, device="cuda")
        sum_trait = T.SumOps(acc=cutlass.Float32)
        (got,) = rt.reduce_row_tile(sum_trait, "tpr1_empty_sum", x, f32, **kw)
        self.assertEqual(got, zeros, atol=0, rtol=0)
        # A nonzero identity catches a kernel that leaves the empty output unwritten.
        prod_trait = T.ProdOps(acc=cutlass.Float32)
        (got,) = rt.reduce_row_tile(prod_trait, "tpr1_empty_prod", x, f32, **kw)
        self.assertEqual(got, torch.ones(4, device="cuda"), atol=0, rtol=0)
        trait = T.WelfordOps(acc=cutlass.Float32)
        parts = rt.reduce_row_tile(
            trait, "tpr1_empty_welford", x, f32 * trait.nfields, final=False, **kw
        )
        for part in parts:
            self.assertEqual(part, zeros, atol=0, rtol=0)

    def test_ragged_vector_pitched_rows(self):
        """Take each row's edges from its own start: pitched rows, and whole-vector rows
        at a pitch that is not a multiple of the vector width, sum exactly."""
        trait = T.SumOps(acc=cutlass.Float32)
        cases = ((34, 4100, 4097), (33, 4099, 4096), (17, 1025, 1000))
        for rows, pitch, columns in cases:
            base = torch.randint(1, 9, (rows, pitch), device="cuda").half()
            x = base[:, :columns]
            with self.subTest(pitch=pitch, columns=columns):
                (got,) = rt.reduce_row_tile(
                    trait, "ragged_pitched", x, [torch.float32], ragged_vector=True
                )
                self.assertEqual(got, x.double().sum(dim=1).float(), atol=0, rtol=0)
        self._assert_ragged_plan("ragged_pitched", True)

    def test_ragged_vector_rejects_misaligned_rows(self):
        """Refuse empty rows and a base that is not 16-byte aligned."""
        trait = T.SumOps(acc=cutlass.Float32)
        base = torch.randn(34, 4100, device="cuda")
        cases = (
            (base.flatten()[1:4098].view(1, 4097), "16-byte aligned base"),
            (base[:, :0], "nonempty rows"),
        )
        for x, msg in cases:
            with self.subTest(msg=msg), self.assertRaisesRegex(ValueError, msg):
                rt.reduce_row_tile(
                    trait, "ragged_misaligned", x, [torch.float32], ragged_vector=True
                )

    @parametrize("op", ["argmax", "var_mean"])
    def test_ragged_vector_int64_index_and_var_mean(self, op):
        """Return int64 argmax indices and both var_mean fields from ragged rows."""
        # Small integers give argmax ties to break.
        x = torch.randint(-4, 5, (33, 1025), device="cuda").to(torch.float32)
        trait, outs = {
            "argmax": (T.ArgMaxOps(acc=cutlass.Float32), [torch.int64]),
            "var_mean": (T.VarMeanOps(correction=1), [torch.float32] * 2),
        }[op]
        kw = dict(nouts=len(outs), order="linear", ragged_vector=True)
        got = rt.reduce_row_tile(trait, f"ragged_{op}", x, outs, **kw)
        with torch.backends.python_native.cutedsl.disabled():
            expected = {
                "argmax": lambda: (x.argmax(dim=1),),
                "var_mean": lambda: torch.var_mean(x, dim=1),
            }[op]()
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)


instantiate_parametrized_tests(TestKernelRowTile)


if __name__ == "__main__":
    run_tests()
