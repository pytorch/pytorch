# Owner(s): ["module: dsl-native-ops"]
#
# The reproducible-DAG fold order, opt-in via PYTORCH_NATIVE_INNER_TREE.
#
# The claim this order makes is a BIT PATTERN, so that is what gets asserted: equality with
# upstream's inner-tree kernel compared as integers, not allclose. It also must not narrow
# coverage -- a shape the order does not cover keeps the default order rather than falling back
# to aten -- and it must stay off by default.

import os
import unittest
from contextlib import contextmanager

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


@contextmanager
def _order_on():
    from torch._native.ops.reductions import kernel_rowtile as rt

    prev = os.environ.get(rt._INNER_TREE_ENV)
    os.environ[rt._INNER_TREE_ENV] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(rt._INNER_TREE_ENV, None)
        else:
            os.environ[rt._INNER_TREE_ENV] = prev


# GOLDEN BIT PATTERNS, generated from ATen's inner-tree kernel and pinned here so the bitwise
# contract OUTLIVES it: matching ATen exactly is a hard requirement, and a differential test
# cannot state that claim once the kernel it compares against is gone. Upstream pins its own
# hashes for SUM (test_sum_cutedsl.py::test_bitwise) but not for prod, which would otherwise lose
# its reference entirely.
#
# Hardware-independent by construction -- the DAG is fixed by N alone and IEEE add/multiply are
# exact -- so these travel across GPUs and toolkits. A hash that changes means the ORDER changed.
# Regenerate ONLY against ATen's kernel, never against our own output.
_GOLDEN = {
    # --- sum ---
    ("sum", "float16", 8, 4): "61272142b0a95772",  # multirow
    ("sum", "float16", 8, 16): "506a22099895a13f",  # multirow
    ("sum", "float16", 8, 27): "800974ad5e1fb845",  # multirow
    ("sum", "float16", 8, 33): "be7f6e4fafaae06d",  # multirow
    ("sum", "float16", 8, 128): "58d0b9b32b9b8107",  # looped
    ("sum", "float16", 8, 1024): "4805281330714407",  # looped
    ("sum", "float16", 8, 4096): "64c97a8480e18692",  # looped
    ("sum", "float16", 8, 4097): "6758d761cd7a8c75",  # looped
    ("sum", "float16", 8, 6143): "661721bd1a61d55c",  # looped
    ("sum", "float16", 8, 8192): "288c3adef1083a11",  # looped
    ("sum", "float16", 8, 20000): "36a2cb78f4f1e1cf",  # looped
    ("sum", "float16", 8, 40000): "f6e7c62c52a62aa4",  # split
    ("sum", "float16", 8, 100003): "b5c36ffdfc71a834",  # split
    ("sum", "float16", 3, 262144): "2eb317127b795ff7",  # split
    ("sum", "bfloat16", 8, 4): "9b0c04638ff5c68e",  # multirow
    ("sum", "bfloat16", 8, 16): "6f08bc606b7fa777",  # multirow
    ("sum", "bfloat16", 8, 27): "b83818ec1a68d6a6",  # multirow
    ("sum", "bfloat16", 8, 33): "93114919c240553e",  # multirow
    ("sum", "bfloat16", 8, 128): "4d01fbb712e2cf2a",  # looped
    ("sum", "bfloat16", 8, 1024): "766e39a13ce5f3f9",  # looped
    ("sum", "bfloat16", 8, 4096): "f5ba992669996d98",  # looped
    ("sum", "bfloat16", 8, 4097): "8b1e5bafb09145fc",  # looped
    ("sum", "bfloat16", 8, 6143): "20950accd1226f4c",  # looped
    ("sum", "bfloat16", 8, 8192): "504e9089b2e6c669",  # looped
    ("sum", "bfloat16", 8, 20000): "7913ba4c30b9e218",  # looped
    ("sum", "bfloat16", 8, 40000): "8d8b3b337d794523",  # split
    ("sum", "bfloat16", 8, 100003): "d01ff76afddbcf13",  # split
    ("sum", "bfloat16", 3, 262144): "82d3e4b49f91f2e3",  # split
    ("sum", "float32", 8, 4): "04416a3c5ae44e4a",  # multirow
    ("sum", "float32", 8, 16): "f0f60f7a02d80a39",  # multirow
    ("sum", "float32", 8, 27): "e7766868b0f4393c",  # multirow
    ("sum", "float32", 8, 33): "dc89a91f83c21b4b",  # looped
    ("sum", "float32", 8, 128): "5b3d5d83855e8f53",  # looped
    ("sum", "float32", 8, 1024): "90a45668bf74e957",  # looped
    ("sum", "float32", 8, 4096): "b28453dc0bb6510e",  # looped
    ("sum", "float32", 8, 4097): "550222d50e25aa4e",  # looped
    ("sum", "float32", 8, 6143): "9f688fc91f2760cd",  # looped
    ("sum", "float32", 8, 8192): "37dd163910d338c7",  # looped
    ("sum", "float32", 8, 20000): "2fab640262dd3dd0",  # looped
    ("sum", "float32", 8, 40000): "99ce48e16c66f342",  # split
    ("sum", "float32", 8, 100003): "0b2520a54297458c",  # split
    ("sum", "float32", 3, 262144): "6100efb05c63dfc9",  # split
    ("sum", "float64", 8, 4): "485cbddd6dff5593",  # multirow
    ("sum", "float64", 8, 16): "da7fa3654d42dcc0",  # multirow
    ("sum", "float64", 8, 27): "a26a4e702902b904",  # looped
    ("sum", "float64", 8, 33): "cb038012718be8e2",  # looped
    ("sum", "float64", 8, 128): "481731f7c1dd555c",  # looped
    ("sum", "float64", 8, 1024): "26a72b7039e14f8d",  # looped
    ("sum", "float64", 8, 4096): "9c52321e3db22285",  # looped
    ("sum", "float64", 8, 4097): "222cf98540d9a7a3",  # looped
    ("sum", "float64", 8, 6143): "b6f9a349eb87b901",  # looped
    ("sum", "float64", 8, 8192): "da00b110a8cb2b7b",  # looped
    ("sum", "float64", 8, 20000): "6868c0df27a6efd3",  # looped
    ("sum", "float64", 8, 40000): "536c0240a453dae5",  # split
    ("sum", "float64", 8, 100003): "8739942d77be49ff",  # split
    ("sum", "float64", 3, 262144): "05280a6d8286f9da",  # split
    # --- prod ---
    ("prod", "float16", 8, 4): "b2a848d42279cafd",  # multirow
    ("prod", "float16", 8, 16): "c439ed5f4b4de5d2",  # multirow
    ("prod", "float16", 8, 27): "214a4e6b95949847",  # multirow
    ("prod", "float16", 8, 33): "dc1d1e01bff5293d",  # multirow
    ("prod", "float16", 8, 128): "819b2bb8159f8283",  # looped
    ("prod", "float16", 8, 1024): "24c153d62a181453",  # looped
    ("prod", "float16", 8, 4096): "d93c892726ee23a4",  # looped
    ("prod", "float16", 8, 4097): "d93c892726ee23a4",  # looped
    ("prod", "float16", 8, 6143): "850a369516a8ad6f",  # looped
    ("prod", "float16", 8, 8192): "32a0018763d9bbf7",  # looped
    ("prod", "float16", 8, 20000): "32a0018763d9bbf7",  # looped
    ("prod", "float16", 8, 40000): "32a0018763d9bbf7",  # split
    ("prod", "float16", 8, 100003): "32a0018763d9bbf7",  # split
    ("prod", "float16", 3, 262144): "d07c8d92e51dcfc0",  # split
    ("prod", "bfloat16", 8, 4): "a32ce7b5dbc31cae",  # multirow
    ("prod", "bfloat16", 8, 16): "f44be04d72d07c28",  # multirow
    ("prod", "bfloat16", 8, 27): "c571fa4ba567d88c",  # multirow
    ("prod", "bfloat16", 8, 33): "51ccada6c1da3cb4",  # multirow
    ("prod", "bfloat16", 8, 128): "b6da3e9220e9b0e3",  # looped
    ("prod", "bfloat16", 8, 1024): "561dd70e7ee5dfbe",  # looped
    ("prod", "bfloat16", 8, 4096): "561dd70e7ee5dfbe",  # looped
    ("prod", "bfloat16", 8, 4097): "561dd70e7ee5dfbe",  # looped
    ("prod", "bfloat16", 8, 6143): "561dd70e7ee5dfbe",  # looped
    ("prod", "bfloat16", 8, 8192): "561dd70e7ee5dfbe",  # looped
    ("prod", "bfloat16", 8, 20000): "561dd70e7ee5dfbe",  # looped
    ("prod", "bfloat16", 8, 40000): "561dd70e7ee5dfbe",  # split
    ("prod", "bfloat16", 8, 100003): "561dd70e7ee5dfbe",  # split
    ("prod", "bfloat16", 3, 262144): "dc6a48767bd84de8",  # split
    ("prod", "float32", 8, 4): "36369589045511a6",  # multirow
    ("prod", "float32", 8, 16): "b20027bba4596b8a",  # multirow
    ("prod", "float32", 8, 27): "035110cc00a180ee",  # multirow
    ("prod", "float32", 8, 33): "06bb3ec8e3a2f4ef",  # looped
    ("prod", "float32", 8, 128): "943cdde6364fffa9",  # looped
    ("prod", "float32", 8, 1024): "1bc35283a888a0de",  # looped
    ("prod", "float32", 8, 4096): "b9c8dfc476668401",  # looped
    ("prod", "float32", 8, 4097): "565fcfd03899bb99",  # looped
    ("prod", "float32", 8, 6143): "e3c37d13d1a80937",  # looped
    ("prod", "float32", 8, 8192): "bee09afa492f8cf5",  # looped
    ("prod", "float32", 8, 20000): "83862306e03556d2",  # looped
    ("prod", "float32", 8, 40000): "0976e380378e4e31",  # split
    ("prod", "float32", 8, 100003): "48e5f8f0e0dcee10",  # split
    ("prod", "float32", 3, 262144): "f216bab5c5ab01c1",  # split
    ("prod", "float64", 8, 4): "92c7492236305c5a",  # multirow
    ("prod", "float64", 8, 16): "b3c4665a1e0a6323",  # multirow
    ("prod", "float64", 8, 27): "4a47513af7b70c2c",  # looped
    ("prod", "float64", 8, 33): "a04a8131b892c98a",  # looped
    ("prod", "float64", 8, 128): "df5ce7b47af04cac",  # looped
    ("prod", "float64", 8, 1024): "c3c14c286d95d149",  # looped
    ("prod", "float64", 8, 4096): "654431d4944daed9",  # looped
    ("prod", "float64", 8, 4097): "982a41024cb9b4b5",  # looped
    ("prod", "float64", 8, 6143): "b4cad97f1edc3d74",  # looped
    ("prod", "float64", 8, 8192): "7081e995c4b88c13",  # looped
    ("prod", "float64", 8, 20000): "c0e264bd0b0ec4e3",  # looped
    ("prod", "float64", 8, 40000): "d710a2802e7abd0a",  # split
    ("prod", "float64", 8, 100003): "0c42378ac1a6c877",  # split
    ("prod", "float64", 3, 262144): "2fd26ff4d111d1b9",  # split
}


def _golden_input(m, n, dtype, prod):
    """The input the table was generated from, reproduced exactly: alternating +-1 with a per-row
    shift, so any reassociation moves the low bits."""
    compute = torch.float64 if dtype is torch.float64 else torch.float32
    cols = torch.arange(n, device="cuda", dtype=compute)
    vals = ((cols % 2) * 2 - 1).reshape(1, n)
    if m > 1:
        rows = torch.arange(m, device="cuda", dtype=compute).reshape(m, 1)
        vals = vals + ((rows % 5) - 2) / 4
    if prod:
        # Keep a length-n product bounded so it neither overflows nor flushes to zero.
        vals = 1.0 + vals / n
    return vals.to(dtype).contiguous()


def _sha(t):
    # RAW BYTES: numpy has no bfloat16, and the bit pattern is the point anyway.
    import hashlib

    b = t.cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestInnerTreeOrder(TestCase):
    # At least one shape per SHAPE the plan can pick, since each is a different DAG: a row inside
    # one thread's fragment (multirow), warps cooperating over up to three baked batches
    # (looped), and a partials pass plus a per-row combine (split). Plus ragged N in each, where
    # the row does not fill its last vector and the tail folds identities.
    SHAPES = [
        (65536, 4),
        (65536, 16),
        (16384, 31),
        (4096, 33),
        (4096, 128),
        (4096, 257),
        (4096, 1024),
        (512, 4096),
        (512, 4097),
        (512, 6143),
        (1024, 8192),
        (256, 2048),
        (128, 40000),
        (64, 100000),
    ]

    def _run(self, trait, key, x, prod=False):
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt

        acc = cutlass.Float64 if x.dtype is torch.float64 else cutlass.Float32
        (got,) = rt.reduce_row_tile(
            trait(acc=acc), key, x, [x.dtype], order="inner_tree"
        )
        return got

    def test_off_by_default(self):
        from torch._native.ops.reductions import kernel_rowtile as rt

        self.assertFalse(rt.inner_tree_order_enabled())

    def test_plan_covers_every_n(self):
        # Coverage has to be TOTAL: a shape with no plan would keep the launch-shape-derived
        # order silently, which is exactly the property this order exists to remove. Also assert
        # the compile-time tile stays under the unroll ceiling -- over it, TileMap raises, so a
        # missing bound would be a crash rather than a fallback.
        from torch._native.ops.reductions import kernel_rowtile as rt, tile

        for itemsize in (2, 4, 8):
            for n in (1, 2, 3, 7, 8, 17, 33, 127, 1000, 8191, 8192, 24577, 10**6):
                for m in (1, 1024):
                    with self.subTest(itemsize=itemsize, n=n, m=m):
                        plan = rt.itree_plan(n, m, itemsize)
                        self.assertIsNotNone(plan)
                        for tm in plan.tms:
                            self.assertLessEqual(tm.vec * tm.loads, tile.MAX_UNROLL)

    @parametrize("op", ["sum", "prod"])
    def test_bitwise_equal_to_upstream(self, op):
        # The whole point of the order: the same DAG upstream's kernel implements, so the
        # results must agree BIT FOR BIT, not merely closely.
        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import (
            inner_tree_kernel as up,
            kernel_rowtile as rt,
        )

        prod = op == "prod"
        trait = T.ProdOps if prod else T.SumOps
        into = up.inner_tree_prod_into if prod else up.inner_tree_sum_into
        for m, n in self.SHAPES:
            m = max(1, min(m, 2**26 // n))
            with self.subTest(shape=(m, n)):
                x = torch.randn(m, n, device="cuda")
                if prod:
                    # Keep the magnitudes near 1 so a long row neither overflows nor flushes to
                    # zero -- a row of inf or 0 would compare equal whatever the order.
                    x = (x * 0.01 + 1.0).contiguous()
                got = self._run(trait, f"itree_{op}", x, prod)
                ref = torch.empty(m, device="cuda")
                into(ref, x)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(got.view(torch.int32), ref.view(torch.int32)),
                    f"{(m, n)} [{rt.itree_plan(n, m, 4).shape}]: bits differ from upstream",
                )

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
    def test_bitwise_equal_to_upstream_dtypes(self, dtype):
        # vec is 16 bytes' worth of elements, so the dtype sets the tree's WIDTH (8 for the
        # 16-bit types, 2 for fp64) and below width 4 the in-vector fold is linear rather than a
        # tree. Every one of those is a different DAG.
        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import inner_tree_kernel as up

        ikind = {2: torch.int16, 4: torch.int32, 8: torch.int64}
        for m, n in [(65536, 8), (4096, 100), (512, 4096), (512, 4097), (64, 100000)]:
            m = max(1, min(m, 2**26 // n))
            with self.subTest(shape=(m, n)):
                x = torch.randn(m, n, device="cuda", dtype=dtype)
                got = self._run(T.SumOps, "itree_sum_dt", x)
                ref = torch.empty(m, device="cuda", dtype=dtype)
                up.inner_tree_sum_into(ref, x)
                torch.cuda.synchronize()
                ik = ikind[x.element_size()]
                self.assertTrue(
                    torch.equal(got.view(ik), ref.view(ik)),
                    f"{(m, n)} {dtype}: bits differ from upstream",
                )

    def test_signed_zero_matches_upstream_per_shape(self):
        # The case that catches a stray identity in the fold: `0.0 + -0.0` is `+0.0`, so seeding
        # a cross-batch accumulator changes the RESULT BITS for an all -0.0 row. Upstream seeds
        # in its looped kernel and not in the other two, so the shapes genuinely disagree here
        # and matching "closely" would hide it.
        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import (
            inner_tree_kernel as up,
            kernel_rowtile as rt,
        )

        for m, n in ((256, 8), (256, 1024), (64, 100000)):
            with self.subTest(shape=(m, n), shape_kind=rt.itree_plan(n, m, 4).shape):
                x = torch.full((m, n), -0.0, device="cuda")
                got = self._run(T.SumOps, "itree_zero", x)
                ref = torch.empty(m, device="cuda")
                up.inner_tree_sum_into(ref, x)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(got.view(torch.int32), ref.view(torch.int32)),
                    f"{(m, n)}: signed zero differs from upstream",
                )

    def test_order_is_reproducible_across_batch(self):
        # The reason to want this order: its DAG comes from N alone, so a row's result does not
        # depend on how many rows were reduced with it. The default order picks its launch
        # shape per call and carries no such guarantee.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        n = 4096
        big = torch.randn(64, n, device="cuda")
        trait = T.SumOps(acc=cutlass.Float32)
        (whole,) = rt.reduce_row_tile(
            trait, "itree_batch", big, [torch.float32], order="inner_tree"
        )
        for m in (1, 3, 16):
            with self.subTest(rows=m):
                part = big[:m].contiguous()
                (sub,) = rt.reduce_row_tile(
                    trait, "itree_batch", part, [torch.float32], order="inner_tree"
                )
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(sub.view(torch.int32), whole[:m].view(torch.int32)),
                    f"rows={m}: the order's bits changed with the batch size",
                )

    @parametrize("op", ["sum", "prod"])
    def test_golden_bit_pattern(self, op):
        # The claim is a BIT PATTERN, so assert the pattern rather than agreement with a reference
        # that is about to be deleted. Covers all three plan shapes, ragged N in each, 4 dtypes.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        dtypes = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float64": torch.float64,
        }
        prod = op == "prod"
        trait = T.ProdOps if prod else T.SumOps
        checked = 0
        for (kop, dname, m, n), want in _GOLDEN.items():
            if kop != op:
                continue
            dtype = dtypes[dname]
            with self.subTest(dtype=dname, shape=(m, n)):
                x = _golden_input(m, n, dtype, prod)
                acc = cutlass.Float64 if dtype is torch.float64 else cutlass.Float32
                (got,) = rt.reduce_row_tile(
                    trait(acc=acc), f"golden_{op}", x, [dtype], order="inner_tree"
                )
                torch.cuda.synchronize()
                kind = rt.itree_plan(n, m, x.element_size()).shape
                self.assertEqual(
                    _sha(got),
                    want,
                    f"{op} {dname} ({m}, {n}) [{kind}]: bit pattern changed",
                )
                checked += 1
        self.assertEqual(checked, len(_GOLDEN) // 2)

    def test_staging_is_actually_used_in_the_mid_band(self):
        # The 112 hashes prove staging is bit-NEUTRAL, which is exactly why they cannot prove it
        # RAN: if the gate stopped firing, every other test here would still pass and the mid-band
        # would just be slow again. Assert the plan picks it, and that the two cases it cannot
        # serve keep the register fold.
        from torch._native.ops.reductions import kernel_rowtile as rt

        for n in (1024, 2048, 4096, 8192):
            with self.subTest(n=n, staged=True):
                self.assertGreater(rt.itree_plan(n, 4096, 4).stage_e, 0)
        # A ragged row cannot declare cp.async's statically 16-byte-aligned source.
        self.assertEqual(rt.itree_plan(4097, 4096, 4).stage_e, 0)
        # And a MULTI-BATCH shape is outside the single-batch tiling. Pick N from the plan itself
        # rather than by eye: a shape whose batch count exceeds the two-kernel threshold never
        # reaches the staging gate at all, which is not the condition under test.
        multi = next(
            n
            for n in range(9216, 65536, 1024)
            if (p := rt.itree_plan(n, 4096, 4)) is not None
            and p.shape == "looped"
            and len(p.batches) > 1  # `batches` is one tuple per COMPILE-TIME batch
        )
        self.assertEqual(rt.itree_plan(multi, 4096, 4).stage_e, 0)

    def test_staged_fold_pads_a_short_row_bitwise(self):
        # Staging requires a single batch, i.e. N <= span, so span > N (identity-padded columns) is
        # reachable and has two nontrivial pieces: the cp.async source redirect for the out-of-range
        # tail and the refill. Both are silent if wrong -- a padded lane that read live data would
        # still produce a plausible sum -- so compare BITWISE against the reference kernel at the
        # widest span, where the padding is largest.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import (
            inner_tree_kernel as ref,
            kernel_rowtile as rt,
        )

        staged = [
            n
            for n in (1024, 1056, 1088, 1536, 2048, 3072, 4096, 6144, 8192)
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
                    f"stage_pad{n}",
                    x,
                    [torch.float32],
                    order="inner_tree",
                )
                want = torch.empty(256, device="cuda")
                ref.inner_tree_sum_into(want, x)
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(got.view(torch.int32), want.view(torch.int32)),
                    f"N={n}: staged fold diverged from the reference bit pattern",
                )


instantiate_parametrized_tests(TestInnerTreeOrder)


if __name__ == "__main__":
    run_tests()
