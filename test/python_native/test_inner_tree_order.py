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


# GOLDEN BIT PATTERNS, pinned so the bitwise contract outlives its reference. The reference is
# `ops/reductions/inner_tree_kernel.py`. There is no ATen kernel to regenerate against: upstream
# wrote one (#182986, June 2026) and pinned 54 hashes from it, but that iteration never landed --
# what landed two months later is the DSL port, carrying 36 of those hashes unchanged into
# test_sum_cutedsl.py::test_bitwise. Those 36 are the tie back to ATen, for VALUES only: they were
# generated from exactly-representable data, so they cannot detect a reordering, and upstream's
# order-sensitive cross-warp hash was generated from the port rather than from ATen. This table
# exists because the port is a live fallback today but is meant to be deleted once gapped/strided
# inputs are served natively, and a differential test cannot outlive it.
#
# Hardware-independent by construction -- the DAG is fixed by N alone and IEEE add/multiply are
# exact -- so these travel across GPUs and toolkits. A hash that changes means the ORDER changed,
# for every entry test_golden_input_can_detect_a_reorder covers; the narrow dtypes are value
# checks only, for the reason given there. Regenerate against the reference kernel, never against
# an unverified change to our own fold.
_GOLDEN = {
    # --- sum ---
    ("sum", "float16", 8, 4): "7069267eff54930e",  # multirow
    ("sum", "float16", 8, 16): "5c61a5395d0691bb",  # multirow
    ("sum", "float16", 8, 27): "57a3f67d5fd8f69d",  # multirow
    ("sum", "float16", 8, 33): "d171b9e87e8a49d3",  # multirow
    ("sum", "float16", 8, 128): "1f93a9cc8df29f25",  # looped
    ("sum", "float16", 8, 1024): "78a863f321707ebf",  # looped
    ("sum", "float16", 8, 4096): "d49189e15dae81ae",  # looped
    ("sum", "float16", 8, 4097): "0532ce6becb24fef",  # looped
    ("sum", "float16", 8, 6143): "b997bb02a103b199",  # looped
    ("sum", "float16", 8, 8192): "20deb1a11722efb0",  # looped
    ("sum", "float16", 8, 20000): "d4d2445c85161924",  # looped
    ("sum", "float16", 8, 40000): "efb4882aafbd3e01",  # split
    ("sum", "float16", 8, 100003): "7ca7afe152551d32",  # split
    ("sum", "float16", 3, 262144): "bb9d19cc8683c407",  # split
    ("sum", "bfloat16", 8, 4): "e164d1c48f05e7ec",  # multirow
    ("sum", "bfloat16", 8, 16): "83b0ea3d6efe1c17",  # multirow
    ("sum", "bfloat16", 8, 27): "6b35fadfb1131edf",  # multirow
    ("sum", "bfloat16", 8, 33): "91f77421caea42c2",  # multirow
    ("sum", "bfloat16", 8, 128): "b88d7c67be9a0a92",  # looped
    ("sum", "bfloat16", 8, 1024): "e8368cc6128dc8ea",  # looped
    ("sum", "bfloat16", 8, 4096): "38e1201cfc69c4a8",  # looped
    ("sum", "bfloat16", 8, 4097): "d82d5a0eec1a9342",  # looped
    ("sum", "bfloat16", 8, 6143): "9a8d12f94bf3b42e",  # looped
    ("sum", "bfloat16", 8, 8192): "d5c3f2d40762d1ef",  # looped
    ("sum", "bfloat16", 8, 20000): "766b771d69e39c78",  # looped
    ("sum", "bfloat16", 8, 40000): "131d9a1b238a2ec3",  # split
    ("sum", "bfloat16", 8, 100003): "dd0d994c351ef3d0",  # split
    ("sum", "bfloat16", 3, 262144): "9cb55f6fe7d7a1d3",  # split
    ("sum", "float32", 8, 4): "73ad01782c9262c4",  # multirow
    ("sum", "float32", 8, 16): "88f7e0f77961255e",  # multirow
    ("sum", "float32", 8, 27): "56b42ce6ec0dd5a2",  # multirow
    ("sum", "float32", 8, 33): "2a38ee077ed8f5eb",  # looped
    ("sum", "float32", 8, 128): "1008f735a4f08798",  # looped
    ("sum", "float32", 8, 1024): "a648cd0f3c75779a",  # looped
    ("sum", "float32", 8, 4096): "c21308629158fe30",  # looped
    ("sum", "float32", 8, 4097): "61219bb1c29abdf9",  # looped
    ("sum", "float32", 8, 6143): "c0695c71817953b3",  # looped
    ("sum", "float32", 8, 8192): "8928a0edee31a8e6",  # looped
    ("sum", "float32", 8, 20000): "dd3c69b9b0dd6d1e",  # looped
    ("sum", "float32", 8, 40000): "00f118660d7a6ecf",  # split
    ("sum", "float32", 8, 100003): "7c81ed5e86748261",  # split
    ("sum", "float32", 3, 262144): "43932e982470bb77",  # split
    ("sum", "float64", 8, 4): "449faee9ab9f0ec5",  # multirow
    ("sum", "float64", 8, 16): "3cdb33c229796052",  # multirow
    ("sum", "float64", 8, 27): "684f413c34ebd347",  # looped
    ("sum", "float64", 8, 33): "6d2f637976c42052",  # looped
    ("sum", "float64", 8, 128): "4ffced4af7693992",  # looped
    ("sum", "float64", 8, 1024): "f6d308aea49a3796",  # looped
    ("sum", "float64", 8, 4096): "4e3500a9305f2e4c",  # looped
    ("sum", "float64", 8, 4097): "986dedef9c2f194b",  # looped
    ("sum", "float64", 8, 6143): "f377c8777d302c6a",  # looped
    ("sum", "float64", 8, 8192): "f09708f4bcc607ad",  # looped
    ("sum", "float64", 8, 20000): "d3a29d2973b31c7f",  # looped
    ("sum", "float64", 8, 40000): "5ef3b8011926a87d",  # split
    ("sum", "float64", 8, 100003): "93bc7250f8575381",  # split
    ("sum", "float64", 3, 262144): "55d3197971df287a",  # split
    # --- prod ---
    ("prod", "float16", 8, 4): "e9deb19e81484045",  # multirow
    ("prod", "float16", 8, 16): "3adb69ccc709c602",  # multirow
    ("prod", "float16", 8, 27): "8e61a4a27e9b9b06",  # multirow
    ("prod", "float16", 8, 33): "ccbbaccfc0326b14",  # multirow
    ("prod", "float16", 8, 128): "a3e124a6aa5ffd7c",  # looped
    ("prod", "float16", 8, 1024): "55502ae564a8df02",  # looped
    ("prod", "float16", 8, 4096): "7c02b2f7ece60d69",  # looped
    ("prod", "float16", 8, 4097): "68d56137046f20c2",  # looped
    ("prod", "float16", 8, 6143): "14da9cdf82d1eff4",  # looped
    ("prod", "float16", 8, 8192): "b1ae7ed8ec807ba7",  # looped
    ("prod", "float16", 8, 20000): "897cc9f13b2b6ac1",  # looped
    ("prod", "float16", 8, 40000): "c508c206b1f7d16f",  # split
    ("prod", "float16", 8, 100003): "2d52e87b8e56a9c5",  # split
    ("prod", "float16", 3, 262144): "4e5e130954f943dc",  # split
    ("prod", "bfloat16", 8, 4): "a2b1b5a6ad32cd81",  # multirow
    ("prod", "bfloat16", 8, 16): "6e351602c6708f55",  # multirow
    ("prod", "bfloat16", 8, 27): "1fea08827cc8dd42",  # multirow
    ("prod", "bfloat16", 8, 33): "402ba8c627529d7a",  # multirow
    ("prod", "bfloat16", 8, 128): "ed1ac5a5816f7579",  # looped
    ("prod", "bfloat16", 8, 1024): "04f210cba53a126a",  # looped
    ("prod", "bfloat16", 8, 4096): "e97e3807c1fe45dd",  # looped
    ("prod", "bfloat16", 8, 4097): "e5fc6516cb12b36e",  # looped
    ("prod", "bfloat16", 8, 6143): "f450c3ee36a92d89",  # looped
    ("prod", "bfloat16", 8, 8192): "0b151fef6d5bb06c",  # looped
    ("prod", "bfloat16", 8, 20000): "5bd34716574270f5",  # looped
    ("prod", "bfloat16", 8, 40000): "274312fc1141d249",  # split
    ("prod", "bfloat16", 8, 100003): "926c2606d521edb0",  # split
    ("prod", "bfloat16", 3, 262144): "dc6a48767bd84de8",  # split
    ("prod", "float32", 8, 4): "4569bdcd5fb6469c",  # multirow
    ("prod", "float32", 8, 16): "6890938d89593965",  # multirow
    ("prod", "float32", 8, 27): "70745d71c15cadd5",  # multirow
    ("prod", "float32", 8, 33): "819a2332a8ec961c",  # looped
    ("prod", "float32", 8, 128): "3337dccee47d5479",  # looped
    ("prod", "float32", 8, 1024): "77f1b0239bc269fb",  # looped
    ("prod", "float32", 8, 4096): "052d1f11cfe7b81c",  # looped
    ("prod", "float32", 8, 4097): "a323b3f23861149f",  # looped
    ("prod", "float32", 8, 6143): "d2536401b9badc4c",  # looped
    ("prod", "float32", 8, 8192): "2105242844970646",  # looped
    ("prod", "float32", 8, 20000): "d04a833d3ef3875b",  # looped
    ("prod", "float32", 8, 40000): "5e68f115d01cb452",  # split
    ("prod", "float32", 8, 100003): "4cef96575c26539a",  # split
    ("prod", "float32", 3, 262144): "e575349d6caa43b3",  # split
    ("prod", "float64", 8, 4): "1b78d386f0595867",  # multirow
    ("prod", "float64", 8, 16): "7cd73d01fecb15b7",  # multirow
    ("prod", "float64", 8, 27): "3cb3863da517795f",  # looped
    ("prod", "float64", 8, 33): "fb3d42e096254897",  # looped
    ("prod", "float64", 8, 128): "bddb7b94b0a46a5a",  # looped
    ("prod", "float64", 8, 1024): "502ad976ac0dbc22",  # looped
    ("prod", "float64", 8, 4096): "1df5b26dcb0e5e81",  # looped
    ("prod", "float64", 8, 4097): "35f8e0d3a5f71bdc",  # looped
    ("prod", "float64", 8, 6143): "d3dfdc2fb1351676",  # looped
    ("prod", "float64", 8, 8192): "2143eeecee209dbf",  # looped
    ("prod", "float64", 8, 20000): "b7ce5185b397b4d6",  # looped
    ("prod", "float64", 8, 40000): "a32dd963aad3d9ae",  # split
    ("prod", "float64", 8, 100003): "de84b76fd6125a71",  # split
    ("prod", "float64", 3, 262144): "030ca5e848e497ad",  # split
}


def _golden_input(m, n, dtype, prod):
    """The input the table was generated from, reproduced exactly.

    The values have to ROUND, or the table pins nothing about the order: an earlier version used
    alternating +-1 with a per-row shift of (row % 5 - 2)/4 -- every value a multiple of 0.25 and
    every partial sum under 1.5n, hence EXACT in the fp32 accumulator, so reassociation could not
    move a bit and both fold orders hashed identically at every shape.

    They also have to be APERIODIC over the largest row here. Fractions over two small moduli sum
    to zero across each period, so with 29 and 7 alone the result depends only on n mod 203, and
    1024 and 40000 -- one looped, one split -- hash identically. Hence the third modulus: lcm(29,
    7, 4093) = 830879 exceeds the largest m*n (786432), so no row completes a period.

    A low-discrepancy sequence is the wrong fix, incidentally: `(v * phi) % 1` keeps partial sums
    near zero AND leaves only ~33 mantissa bits, so an fp64 sum of it is exact and both orders
    agree bit for bit. Ratios of primes have full-length binary expansions, which is the property
    that matters. All deterministic and RNG-free, so the table reproduces anywhere.
    """
    v = torch.arange(m * n, device="cuda", dtype=torch.float64).reshape(m, n)
    vals = ((v % 29) - 14) / 29 + ((v % 7) - 3) / 13 + ((v % 4093) - 2046) / 4093 / 4
    if prod:
        # Keep a length-n product bounded so it neither overflows nor flushes to zero, while
        # leaving the factors far enough from 1.0 to survive the narrow dtypes' rounding.
        vals = 1.0 + vals / max(8.0, n**0.5)
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

        # PROD as well as sum: the two differ in their identity and in what a ragged tail pads
        # with, and prod was previously checked in fp32 only -- so the 16-bit tree widths and the
        # fp64 pair width went unverified for it.
        ikind = {2: torch.int16, 4: torch.int32, 8: torch.int64}
        for op in ("sum", "prod"):
            trait = T.ProdOps if op == "prod" else T.SumOps
            into = up.inner_tree_prod_into if op == "prod" else up.inner_tree_sum_into
            for m, n in [
                (65536, 8),
                (4096, 100),
                (512, 4096),
                (512, 4097),
                (64, 100000),
            ]:
                m = max(1, min(m, 2**26 // n))
                with self.subTest(op=op, shape=(m, n)):
                    x = torch.randn(m, n, device="cuda", dtype=dtype)
                    if op == "prod":
                        # Near 1 so a long row neither overflows nor flushes to zero; a row of inf
                        # or 0 would compare equal under any order.
                        x = (x * 0.01 + 1.0).contiguous()
                    got = self._run(trait, f"itree_{op}_dt", x)
                    ref = torch.empty(m, device="cuda", dtype=dtype)
                    into(ref, x)
                    torch.cuda.synchronize()
                    ik = ikind[x.element_size()]
                    self.assertEqual(
                        got.view(ik),
                        ref.view(ik),
                        msg=f"{op} {(m, n)} {dtype}: bits differ from upstream",
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

    def test_no_plan_pairs_an_exact_tile_with_a_bound_or_an_offset_base(self):
        # tile.load takes an UNPREDICATED wide load when `tm.exact` -- which only says the tile
        # covers its own N from column 0. A narrower `bound` (the split shape's chunk end, there to
        # stop a short chunk reading the next chunk's elements) or a shifted `base_col` invalidate
        # that, so `load` requires them absent. This is the other half of that guard: it is free
        # only because no plan pairs them, and nothing else says so. A plan change that broke the
        # pairing would otherwise just make the fast path quietly read the wrong columns -- and
        # since the fold is bit-neutral by design, the numbers would still look plausible.
        from torch._native.ops.reductions import kernel_rowtile as rt

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
                        if tm.vec * tm.loads * tm.tpr != tm.N:
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
        # The table is worth nothing if its input cannot tell two orders apart, and the first
        # version of it could not: values were multiples of 0.25 with partial sums under 1.5n,
        # hence EXACT in the accumulator, so all 56 sum entries hashed identically under the
        # launch-shape fold and would have passed with the wrong DAG. Assert the discriminating
        # power rather than trusting the generator, one shape per plan shape.
        #
        # fp32/fp64 only, and N > 4. A narrowing store cannot carry an fp32-accumulator difference
        # (~1e-7 relative) into 8 or 11 mantissa bits, so the fp16/bf16 entries are value checks at
        # any input -- consistent with those dtypes sitting outside the bitwise contract (see
        # cutedsl_impl's dtype note). At N=4 one thread owns the whole row, so there is no
        # reassociation to detect. Both exclusions are structural, not properties of this data.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        for op in ("sum", "prod"):
            trait = T.ProdOps if op == "prod" else T.SumOps
            for dtype in (torch.float32, torch.float64):
                acc = cutlass.Float64 if dtype is torch.float64 else cutlass.Float32
                for m, n in [(8, 16), (8, 4097), (8, 100003)]:
                    with self.subTest(op=op, dtype=dtype, shape=(m, n)):
                        x = _golden_input(m, n, dtype, op == "prod")
                        (tree,) = rt.reduce_row_tile(
                            trait(acc=acc),
                            f"disc_t_{op}",
                            x,
                            [dtype],
                            order="inner_tree",
                        )
                        (launch,) = rt.reduce_row_tile(
                            trait(acc=acc), f"disc_l_{op}", x, [dtype]
                        )
                        torch.cuda.synchronize()
                        self.assertNotEqual(
                            _sha(tree),
                            _sha(launch),
                            "the golden input cannot distinguish the two fold orders, so the "
                            "pinned hashes would pass with the wrong DAG",
                        )

    def test_the_gate_routes_the_dispatcher_through_the_order(self):
        # What the gate is FOR, asserted through the dispatcher rather than the fold. Two arms had
        # to be fixed to make this hold, and neither is visible in the numbers because both wrong
        # paths compute a correct reduction: a narrow row used to arrive with an explicit tpr (which
        # makes reduce_row_tile decline the order), and a split-shape row used to fall through to
        # kernel_xcta, whose TileReduce never consults the gate. MEASURED as differing bits at
        # (524288, 16) and (64, 100000) respectively before those fixes.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
        )

        for m, n in [(524288, 16), (64, 100000), (8192, 1024)]:
            with self.subTest(shape=(m, n)):
                x = torch.randn(m, n, device="cuda")
                want = self._run(T.SumOps, f"gate_ref_{n}", x)
                with _order_on():
                    self.assertTrue(rt.inner_tree_order_enabled())
                    got = kg.reduce_dim(
                        T.SumOps(acc=cutlass.Float32),
                        f"gate_disp_{n}",
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
        # nfields > 1 gives the fold one smem staging buffer and one split partial buffer PER
        # FIELD, and nouts == 2 projects two results from one accumulator -- machinery the
        # single-field value traits never touch. A RAGGED N as well, so the identity padding that
        # the fold's docstring calls part of the DAG is exercised rather than skipped by a tile
        # that happens to cover the row exactly.
        #
        # Compared against the same trait under the launch-shape fold, not against aten: what is
        # under test is that the order's own per-field plumbing carries every field, and the two
        # DAGs associate differently by design, so a tolerance is the right comparison.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

        x = torch.randn(64, 4097, device="cuda")
        cases = [
            ("welford", T.WelfordOps, {"correction": 1}, 1, [torch.float32]),
            ("var_mean", T.VarMeanOps, {"correction": 1}, 2, [torch.float32] * 2),
            ("argmax", T.ArgMaxOps, {}, 1, [torch.int32]),
            ("argmin", T.ArgMinOps, {}, 1, [torch.int32]),
            ("max_dim", T.MaxDimOps, {}, 2, [torch.float32, torch.int32]),
            ("aminmax", T.AMinMaxOps, {}, 2, [torch.float32] * 2),
        ]
        for label, trait, kw, nouts, out_dtypes in cases:
            with self.subTest(trait=label):
                tree = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32, **kw),
                    f"mf_tree_{label}",
                    x,
                    out_dtypes,
                    nouts=nouts,
                    order="inner_tree",
                )
                launch = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32, **kw),
                    f"mf_launch_{label}",
                    x,
                    out_dtypes,
                    nouts=nouts,
                )
                torch.cuda.synchronize()
                self.assertEqual(len(tree), nouts)
                for k, (a, b) in enumerate(zip(tree, launch)):
                    # An index field must agree EXACTLY: a wrong per-field buffer shows up as the
                    # wrong argument, which a tolerance would hide.
                    if a.dtype in (torch.int32, torch.int64):
                        self.assertEqual(a, b, msg=f"{label} field {k}")
                    else:
                        self.assertEqual(
                            a, b, atol=1e-4, rtol=1e-4, msg=f"{label} field {k}"
                        )

    def test_tree_fold_matches_the_serial_fold_for_every_value_trait(self):
        # THE LAW the trait protocol has to satisfy, and the reason `leaf` exists:
        #     combine(leaf(a), leaf(b)) == reduce(reduce(init(), a), b)
        # The default fold walks a row SERIALLY through `reduce`; this order folds it as a TREE
        # through `leaf` + `combine`. A trait carrying its per-element transform only in `reduce`
        # therefore folds RAW values under the tree and returns a plausible WRONG number -- which
        # is exactly what norm/all/any/count_nonzero did before `leaf` existed, and what a
        # protocol test that only checks which methods exist cannot see.
        #
        # Run both shapes over one input and compare. Not bitwise: the two DAGs associate
        # differently on purpose, so a tolerance is the correct comparison here.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_rowtile as rt

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
                    f"law_ser_{label}",
                    x,
                    [torch.float32],
                )
                (tree,) = rt.reduce_row_tile(
                    trait(acc=cutlass.Float32, **kw),
                    f"law_tree_{label}",
                    x,
                    [torch.float32],
                    order="inner_tree",
                )
                self.assertEqual(tree, serial, atol=1e-4, rtol=1e-4)


instantiate_parametrized_tests(TestInnerTreeOrder)


if __name__ == "__main__":
    run_tests()
