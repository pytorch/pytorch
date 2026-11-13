# Owner(s): ["module: dsl-native-ops"]
#
# Inner-tree order promises upstream's exact bits, not closeness. The opt-in gate leaves
# unsupported configurations on the default order; explicit requests raise.

import hashlib
import os
import sys
import unittest
from contextlib import contextmanager

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
    inner_tree_kernel as up,
    kernel_general as kg,
    kernel_rowtile as rt,
    tile,
    traits as T,
)


@contextmanager
def _order_on():
    prev = os.environ.get(rt._INNER_TREE_ENV)
    os.environ[rt._INNER_TREE_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(rt._INNER_TREE_ENV, None)
        else:
            os.environ[rt._INNER_TREE_ENV] = prev


# Pinned bits outlive the temporary reference. This pairwise matrix covers every plan
# shape and crosses each operation and dtype through ragged and split paths.
_GOLDEN = {
    # --- sum ---
    ("sum", "float16", 8, 27): "57a3f67d5fd8f69d",
    ("sum", "float16", 8, 1024): "78a863f321707ebf",
    ("sum", "float16", 8, 4097): "0532ce6becb24fef",
    ("sum", "float16", 8, 100003): "7ca7afe152551d32",
    ("sum", "bfloat16", 8, 27): "6b35fadfb1131edf",
    ("sum", "bfloat16", 8, 1024): "e8368cc6128dc8ea",
    ("sum", "bfloat16", 8, 4097): "d82d5a0eec1a9342",
    ("sum", "bfloat16", 8, 100003): "dd0d994c351ef3d0",
    ("sum", "float32", 8, 4): "73ad01782c9262c4",
    ("sum", "float32", 8, 16): "88f7e0f77961255e",
    ("sum", "float32", 8, 27): "56b42ce6ec0dd5a2",
    ("sum", "float32", 8, 33): "2a38ee077ed8f5eb",
    ("sum", "float32", 8, 128): "1008f735a4f08798",
    ("sum", "float32", 8, 1024): "a648cd0f3c75779a",
    ("sum", "float32", 8, 4096): "c21308629158fe30",
    ("sum", "float32", 8, 4097): "61219bb1c29abdf9",
    ("sum", "float32", 8, 8192): "8928a0edee31a8e6",
    ("sum", "float32", 8, 20000): "dd3c69b9b0dd6d1e",
    ("sum", "float32", 8, 40000): "00f118660d7a6ecf",
    ("sum", "float32", 8, 100003): "7c81ed5e86748261",
    ("sum", "float32", 3, 262144): "43932e982470bb77",
    ("sum", "float64", 8, 27): "684f413c34ebd347",
    ("sum", "float64", 8, 1024): "f6d308aea49a3796",
    ("sum", "float64", 8, 4097): "986dedef9c2f194b",
    ("sum", "float64", 8, 100003): "93bc7250f8575381",
    # --- prod ---
    ("prod", "float16", 8, 4097): "68d56137046f20c2",
    ("prod", "float16", 8, 100003): "2d52e87b8e56a9c5",
    ("prod", "bfloat16", 8, 4097): "e5fc6516cb12b36e",
    ("prod", "bfloat16", 8, 100003): "926c2606d521edb0",
    ("prod", "float32", 8, 16): "6890938d89593965",
    ("prod", "float32", 8, 1024): "77f1b0239bc269fb",
    ("prod", "float32", 8, 4097): "a323b3f23861149f",
    ("prod", "float32", 8, 20000): "d04a833d3ef3875b",
    ("prod", "float32", 8, 100003): "4cef96575c26539a",
    ("prod", "float64", 8, 4097): "35f8e0d3a5f71bdc",
    ("prod", "float64", 8, 100003): "de84b76fd6125a71",
}


def _golden_input(m, n, dtype, prod):
    """Reproduce the table input: rounded, RNG-free, and aperiodic through the largest row."""
    v = torch.arange(m * n, device="cuda", dtype=torch.float64).reshape(m, n)
    vals = ((v % 29) - 14) / 29 + ((v % 7) - 3) / 13 + ((v % 4093) - 2046) / 4093 / 4
    if prod:
        # Bound the product without rounding narrow factors to 1.
        vals = 1.0 + vals / max(8.0, n**0.5)
    return vals.to(dtype).contiguous()


def _sha(t):
    # Hash raw bytes because NumPy lacks bfloat16 and bits are the contract.
    b = t.cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


def _trait_key(trait):
    return f"inner_tree_test_{trait.__name__}"


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestInnerTreeOrder(TestCase):
    def _run(self, trait, x):
        acc = cutlass.Float64 if x.dtype is torch.float64 else cutlass.Float32
        (got,) = rt.reduce_row_tile(
            trait(acc=acc), _trait_key(trait), x, [x.dtype], order="inner_tree"
        )
        return got

    def test_off_by_default(self):
        self.assertFalse(rt.inner_tree_order_enabled())

    def test_plan_covers_every_n(self):
        # Every N needs a plan or silently keeps launch order. Enforce MAX_UNROLL because
        # TileMap raises above it.
        for itemsize in (2, 4, 8):
            for n in (1, 2, 3, 7, 8, 17, 33, 127, 1000, 8191, 8192, 24577, 10**6):
                for m in (1, 1024):
                    with self.subTest(itemsize=itemsize, n=n, m=m):
                        plan = rt.itree_plan(n, m, itemsize)
                        self.assertIsNotNone(plan)
                        for tm in plan.tms:
                            self.assertLessEqual(tm.vec * tm.loads, tile.MAX_UNROLL)

    @parametrize("op", ["sum", "prod"])
    def test_exact_bits_match_golden_and_upstream(self, op):
        # One launch checks both the durable oracle and the temporary reference.
        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        ikind = {2: torch.int16, 4: torch.int32, 8: torch.int64}
        prod = op == "prod"
        trait = T.ProdOps if prod else T.SumOps
        into = up.inner_tree_prod_into if prod else up.inner_tree_sum_into
        checked = 0
        for (kop, dname, m, n), want in _GOLDEN.items():
            if kop != op:
                continue
            dtype = dtypes[dname]
            with self.subTest(dtype=dname, shape=(m, n)):
                x = _golden_input(m, n, dtype, prod)
                got = self._run(trait, x)
                ref = torch.empty(m, device="cuda", dtype=dtype)
                into(ref, x)
                torch.cuda.synchronize()
                kind = rt.itree_plan(n, m, x.element_size()).shape
                self.assertEqual(
                    _sha(got),
                    want,
                    f"{op} {dname} ({m}, {n}) [{kind}]: bit pattern changed",
                )
                self.assertEqual(
                    got.view(ikind[x.element_size()]),
                    ref.view(ikind[x.element_size()]),
                    msg=f"{op} {dname} ({m}, {n}) [{kind}]: bits differ from upstream",
                )
                checked += 1
        self.assertGreater(checked, 0)

    def test_signed_zero_matches_upstream_per_shape(self):
        # A stray identity changes all -0.0 rows because 0.0 + -0.0 is +0.0. Upstream seeds
        # only some shapes, and close comparison hides the resulting disagreement.
        for m, n in ((256, 8), (256, 1024), (64, 100000)):
            with self.subTest(shape=(m, n), shape_kind=rt.itree_plan(n, m, 4).shape):
                x = torch.full((m, n), -0.0, device="cuda")
                got = self._run(T.SumOps, x)
                ref = torch.empty(m, device="cuda")
                up.inner_tree_sum_into(ref, x)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(got.view(torch.int32), ref.view(torch.int32)),
                    f"{(m, n)}: signed zero differs from upstream",
                )

    def test_order_is_reproducible_across_batch(self):
        # An N-only DAG makes each row independent of batch size, unlike the default order.
        n = 4096
        big = torch.randn(64, n, device="cuda")
        trait = T.SumOps(acc=cutlass.Float32)
        (whole,) = rt.reduce_row_tile(
            trait, _trait_key(T.SumOps), big, [torch.float32], order="inner_tree"
        )
        for m in (1, 3, 16):
            with self.subTest(rows=m):
                part = big[:m].contiguous()
                (sub,) = rt.reduce_row_tile(
                    trait,
                    _trait_key(T.SumOps),
                    part,
                    [torch.float32],
                    order="inner_tree",
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(sub.view(torch.int32), whole[:m].view(torch.int32)),
                    f"rows={m}: the order's bits changed with the batch size",
                )

    def test_no_plan_pairs_an_exact_tile_with_a_bound_or_an_offset_base(self):
        # Exact only covers the tile's N from column 0; a bound or offset invalidates its
        # unpredicated load. No plan may pair them because wrong columns can look plausible.
        checked = paired = 0
        for itemsize in (2, 4, 8):
            for n in list(range(1, 512)) + [
                1024,
                2048,
                4097,
                8192,
                40000,
                100003,
                262144,
            ]:
                for m in (1, 8, 256, 4096):
                    plan = rt.itree_plan(n, m, itemsize)
                    if plan is None:
                        continue
                    for b, tm in enumerate(plan.tms):
                        checked += 1
                        if tm.vec * tm.loads * tm.threads_per_row != tm.N:
                            continue  # not exact: takes the predicated path either way
                        base = plan.batches[b][0] if b < len(plan.batches) else 0
                        paired += plan.shape == "split" or bool(base)
        self.assertGreater(checked, 1000, "the sweep stopped covering plans")
        self.assertEqual(
            paired,
            0,
            "a plan now pairs an exact tile with a bound or an offset base; tile.load's "
            "unpredicated wide load does not consult either",
        )

    def test_golden_input_can_detect_a_reorder(self):
        # Verify the input distinguishes orders in fp32/fp64; narrowing can erase differences.
        cases = [
            ("sum", torch.float32, 8, 16),
            ("sum", torch.float64, 8, 1024),
            ("prod", torch.float64, 8, 4097),
            ("prod", torch.float32, 3, 100003),
        ]
        for op, dtype, m, n in cases:
            trait = T.ProdOps if op == "prod" else T.SumOps
            acc = cutlass.Float64 if dtype is torch.float64 else cutlass.Float32
            with self.subTest(op=op, dtype=dtype, shape=(m, n)):
                x = _golden_input(m, n, dtype, op == "prod")
                (tree,) = rt.reduce_row_tile(
                    trait(acc=acc),
                    _trait_key(trait),
                    x,
                    [dtype],
                    order="inner_tree",
                )
                (launch,) = rt.reduce_row_tile(
                    trait(acc=acc), _trait_key(trait), x, [dtype]
                )
                torch.cuda.synchronize()
                self.assertNotEqual(
                    _sha(tree),
                    _sha(launch),
                    "the golden input cannot distinguish the two fold orders, so the "
                    "pinned hashes would pass with the wrong DAG",
                )

    def test_the_gate_routes_the_dispatcher_through_the_order(self):
        # Test dispatcher routing because wrong paths still compute valid reductions.
        for m, n in [(524288, 16), (64, 100000), (8192, 1024)]:
            with self.subTest(shape=(m, n)):
                x = torch.randn(m, n, device="cuda")
                want = self._run(T.SumOps, x)
                with _order_on():
                    self.assertTrue(rt.inner_tree_order_enabled())
                    got = kg.reduce_dim(
                        T.SumOps(acc=cutlass.Float32),
                        _trait_key(T.SumOps),
                        x,
                        [1],
                        torch.float32,
                    )
                torch.cuda.synchronize()
                self.assertEqual(
                    _sha(got.reshape(-1)),
                    _sha(want),
                    f"({m}, {n}) [{rt.itree_plan(n, m, 4).shape}]: the dispatcher served this "
                    "with the launch-shape order while the gate was on",
                )

    def test_multi_field_and_two_output_traits_under_the_order(self):
        # Exercise per-field staging/partials, two outputs from one accumulator, and ragged
        # identity padding. Compare only plumbing with launch order because the DAGs differ.
        x = torch.randn(64, 4097, device="cuda")
        cases = [
            ("welford", T.WelfordOps, {"correction": 1}, 1, [torch.float32]),
            ("var_mean", T.VarMeanOps, {"correction": 1}, 2, [torch.float32] * 2),
            ("max_dim", T.MaxDimOps, {}, 2, [torch.float32, torch.int32]),
        ]
        for label, trait, kw, nouts, out_dtypes in cases:
            with self.subTest(trait=label):
                tree = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32, **kw),
                    f"inner_tree_test_{label}",
                    x,
                    out_dtypes,
                    nouts=nouts,
                    order="inner_tree",
                )
                launch = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32, **kw),
                    f"inner_tree_test_{label}",
                    x,
                    out_dtypes,
                    nouts=nouts,
                )
                torch.cuda.synchronize()
                self.assertEqual(len(tree), nouts)
                for k, (a, b) in enumerate(zip(tree, launch)):
                    # Compare indices exactly so tolerance cannot hide a wrong field buffer.
                    if a.dtype in (torch.int32, torch.int64):
                        self.assertEqual(a, b, msg=f"{label} field {k}")
                    else:
                        torch.testing.assert_close(
                            a, b, atol=1e-4, rtol=1e-4, msg=f"{label} field {k}"
                        )

    def test_tree_fold_matches_the_serial_fold_for_every_value_trait(self):
        # Trait law: combine(leaf(a), leaf(b)) == reduce(reduce(init(), a), b).
        # Otherwise tree order can fold raw values into plausible wrong results.
        x = (
            torch.rand(64, 512, device="cuda") + 0.5
        )  # positive: prod / norm stay finite
        cases = [
            ("sum", T.SumOps, {}),
            ("prod", T.ProdOps, {}),
            ("mean", T.MeanOps, {}),
            ("nansum", T.NanSumOps, {}),
            ("norm2", T.NormOps, {"p": 2.0}),
            ("norm3", T.NormOps, {"p": 3.0}),
            ("all", T.AllOps, {}),
            ("any", T.AnyOps, {}),
            ("count_nonzero", T.CountNonzeroOps, {}),
            ("absmax", T.AbsMaxOps, {}),
            ("absmin", T.AbsMinOps, {}),
            ("amax", T.AMaxOps, {}),
            ("amin", T.AMinOps, {}),
        ]
        for label, trait, kw in cases:
            with self.subTest(trait=label):
                (serial,) = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32, **kw),
                    f"inner_tree_test_{label}",
                    x,
                    [torch.float32],
                )
                (tree,) = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32, **kw),
                    f"inner_tree_test_{label}",
                    x,
                    [torch.float32],
                    order="inner_tree",
                )
                torch.testing.assert_close(tree, serial, atol=1e-4, rtol=1e-4)

    def test_staging_is_actually_used_in_the_mid_band(self):
        # Bit-neutral hashes cannot prove staging ran. Check its gate and both exclusions.
        for n in (1024, 2048, 4096, 8192):
            with self.subTest(n=n, staged=True):
                self.assertGreater(rt.itree_plan(n, 4096, 4).stage_e, 0)
        # A ragged row cannot declare cp.async's statically 16-byte-aligned source.
        self.assertEqual(rt.itree_plan(4097, 4096, 4).stage_e, 0)
        # Multi-batch is outside single-batch tiling; derive N from the plan.
        multi = next(
            n
            for n in range(9216, 65536, 1024)
            if (p := rt.itree_plan(n, 4096, 4)) is not None
            and p.shape == "looped"
            and len(p.batches) > 1
        )
        self.assertEqual(rt.itree_plan(multi, 4096, 4).stage_e, 0)

    def test_staged_fold_pads_a_short_row_bitwise(self):
        # span > N exercises identity padding through cp.async tail redirection and refill.
        # Compare bits because reading live data for padding can still look plausible.
        staged = [
            n
            for n in (1056, 3072, 6144)
            if (p := rt.itree_plan(n, 256, 4)) is not None and p.stage_e > 0
        ]
        self.assertTrue(staged, "no staged shape in the sweep -- the gate has moved")
        for n in staged:
            plan = rt.itree_plan(n, 256, 4)
            with self.subTest(
                n=n, span=plan.tms[0].vec * plan.tms[0].loads * plan.wpr * 32
            ):
                x = torch.randn(256, n, device="cuda")
                (got,) = rt.reduce_row_tile(
                    T.SumOps(acc=cutlass.Float32),
                    _trait_key(T.SumOps),
                    x,
                    [torch.float32],
                    order="inner_tree",
                )
                want = torch.empty(256, device="cuda")
                up.inner_tree_sum_into(want, x)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(got.view(torch.int32), want.view(torch.int32)),
                    f"N={n}: staged fold diverged from the reference bit pattern",
                )


instantiate_parametrized_tests(TestInnerTreeOrder)


if __name__ == "__main__":
    run_tests()
