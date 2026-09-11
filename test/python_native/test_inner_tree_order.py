# Owner(s): ["module: dsl-native-ops"]
#
# Inner-tree order promises upstream's exact bits, not closeness. The opt-in gate leaves
# unsupported configurations on the default order; explicit requests raise.

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


# Pinned bits outlive the temporary reference kernel; ATen cannot regenerate them. The
# N-only DAG and IEEE addition make them hardware-independent. A changed hash means a
# changed order.
_GOLDEN = {
    # --- sum ---
    ("sum", "float16", 8, 4): "7069267eff54930e",
    ("sum", "float16", 8, 16): "5c61a5395d0691bb",
    ("sum", "float16", 8, 27): "57a3f67d5fd8f69d",
    ("sum", "float16", 8, 33): "d171b9e87e8a49d3",
    ("sum", "float16", 8, 128): "1f93a9cc8df29f25",
    ("sum", "float16", 8, 1024): "78a863f321707ebf",
    ("sum", "float16", 8, 4096): "d49189e15dae81ae",
    ("sum", "float16", 8, 4097): "0532ce6becb24fef",
    ("sum", "float16", 8, 6143): "b997bb02a103b199",
    ("sum", "float16", 8, 8192): "20deb1a11722efb0",
    ("sum", "float16", 8, 20000): "d4d2445c85161924",
    ("sum", "float16", 8, 40000): "efb4882aafbd3e01",
    ("sum", "float16", 8, 100003): "7ca7afe152551d32",
    ("sum", "float16", 3, 262144): "bb9d19cc8683c407",
    ("sum", "bfloat16", 8, 4): "e164d1c48f05e7ec",
    ("sum", "bfloat16", 8, 16): "83b0ea3d6efe1c17",
    ("sum", "bfloat16", 8, 27): "6b35fadfb1131edf",
    ("sum", "bfloat16", 8, 33): "91f77421caea42c2",
    ("sum", "bfloat16", 8, 128): "b88d7c67be9a0a92",
    ("sum", "bfloat16", 8, 1024): "e8368cc6128dc8ea",
    ("sum", "bfloat16", 8, 4096): "38e1201cfc69c4a8",
    ("sum", "bfloat16", 8, 4097): "d82d5a0eec1a9342",
    ("sum", "bfloat16", 8, 6143): "9a8d12f94bf3b42e",
    ("sum", "bfloat16", 8, 8192): "d5c3f2d40762d1ef",
    ("sum", "bfloat16", 8, 20000): "766b771d69e39c78",
    ("sum", "bfloat16", 8, 40000): "131d9a1b238a2ec3",
    ("sum", "bfloat16", 8, 100003): "dd0d994c351ef3d0",
    ("sum", "bfloat16", 3, 262144): "9cb55f6fe7d7a1d3",
    ("sum", "float32", 8, 4): "73ad01782c9262c4",
    ("sum", "float32", 8, 16): "88f7e0f77961255e",
    ("sum", "float32", 8, 27): "56b42ce6ec0dd5a2",
    ("sum", "float32", 8, 33): "2a38ee077ed8f5eb",
    ("sum", "float32", 8, 128): "1008f735a4f08798",
    ("sum", "float32", 8, 1024): "a648cd0f3c75779a",
    ("sum", "float32", 8, 4096): "c21308629158fe30",
    ("sum", "float32", 8, 4097): "61219bb1c29abdf9",
    ("sum", "float32", 8, 6143): "c0695c71817953b3",
    ("sum", "float32", 8, 8192): "8928a0edee31a8e6",
    ("sum", "float32", 8, 20000): "dd3c69b9b0dd6d1e",
    ("sum", "float32", 8, 40000): "00f118660d7a6ecf",
    ("sum", "float32", 8, 100003): "7c81ed5e86748261",
    ("sum", "float32", 3, 262144): "43932e982470bb77",
    ("sum", "float64", 8, 4): "449faee9ab9f0ec5",
    ("sum", "float64", 8, 16): "3cdb33c229796052",
    ("sum", "float64", 8, 27): "684f413c34ebd347",
    ("sum", "float64", 8, 33): "6d2f637976c42052",
    ("sum", "float64", 8, 128): "4ffced4af7693992",
    ("sum", "float64", 8, 1024): "f6d308aea49a3796",
    ("sum", "float64", 8, 4096): "4e3500a9305f2e4c",
    ("sum", "float64", 8, 4097): "986dedef9c2f194b",
    ("sum", "float64", 8, 6143): "f377c8777d302c6a",
    ("sum", "float64", 8, 8192): "f09708f4bcc607ad",
    ("sum", "float64", 8, 20000): "d3a29d2973b31c7f",
    ("sum", "float64", 8, 40000): "5ef3b8011926a87d",
    ("sum", "float64", 8, 100003): "93bc7250f8575381",
    ("sum", "float64", 3, 262144): "55d3197971df287a",
    # --- prod ---
    ("prod", "float16", 8, 4): "e9deb19e81484045",
    ("prod", "float16", 8, 16): "3adb69ccc709c602",
    ("prod", "float16", 8, 27): "8e61a4a27e9b9b06",
    ("prod", "float16", 8, 33): "ccbbaccfc0326b14",
    ("prod", "float16", 8, 128): "a3e124a6aa5ffd7c",
    ("prod", "float16", 8, 1024): "55502ae564a8df02",
    ("prod", "float16", 8, 4096): "7c02b2f7ece60d69",
    ("prod", "float16", 8, 4097): "68d56137046f20c2",
    ("prod", "float16", 8, 6143): "14da9cdf82d1eff4",
    ("prod", "float16", 8, 8192): "b1ae7ed8ec807ba7",
    ("prod", "float16", 8, 20000): "897cc9f13b2b6ac1",
    ("prod", "float16", 8, 40000): "c508c206b1f7d16f",
    ("prod", "float16", 8, 100003): "2d52e87b8e56a9c5",
    ("prod", "float16", 3, 262144): "4e5e130954f943dc",
    ("prod", "bfloat16", 8, 4): "a2b1b5a6ad32cd81",
    ("prod", "bfloat16", 8, 16): "6e351602c6708f55",
    ("prod", "bfloat16", 8, 27): "1fea08827cc8dd42",
    ("prod", "bfloat16", 8, 33): "402ba8c627529d7a",
    ("prod", "bfloat16", 8, 128): "ed1ac5a5816f7579",
    ("prod", "bfloat16", 8, 1024): "04f210cba53a126a",
    ("prod", "bfloat16", 8, 4096): "e97e3807c1fe45dd",
    ("prod", "bfloat16", 8, 4097): "e5fc6516cb12b36e",
    ("prod", "bfloat16", 8, 6143): "f450c3ee36a92d89",
    ("prod", "bfloat16", 8, 8192): "0b151fef6d5bb06c",
    ("prod", "bfloat16", 8, 20000): "5bd34716574270f5",
    ("prod", "bfloat16", 8, 40000): "274312fc1141d249",
    ("prod", "bfloat16", 8, 100003): "926c2606d521edb0",
    ("prod", "bfloat16", 3, 262144): "dc6a48767bd84de8",
    ("prod", "float32", 8, 4): "4569bdcd5fb6469c",
    ("prod", "float32", 8, 16): "6890938d89593965",
    ("prod", "float32", 8, 27): "70745d71c15cadd5",
    ("prod", "float32", 8, 33): "819a2332a8ec961c",
    ("prod", "float32", 8, 128): "3337dccee47d5479",
    ("prod", "float32", 8, 1024): "77f1b0239bc269fb",
    ("prod", "float32", 8, 4096): "052d1f11cfe7b81c",
    ("prod", "float32", 8, 4097): "a323b3f23861149f",
    ("prod", "float32", 8, 6143): "d2536401b9badc4c",
    ("prod", "float32", 8, 8192): "2105242844970646",
    ("prod", "float32", 8, 20000): "d04a833d3ef3875b",
    ("prod", "float32", 8, 40000): "5e68f115d01cb452",
    ("prod", "float32", 8, 100003): "4cef96575c26539a",
    ("prod", "float32", 3, 262144): "e575349d6caa43b3",
    ("prod", "float64", 8, 4): "1b78d386f0595867",
    ("prod", "float64", 8, 16): "7cd73d01fecb15b7",
    ("prod", "float64", 8, 27): "3cb3863da517795f",
    ("prod", "float64", 8, 33): "fb3d42e096254897",
    ("prod", "float64", 8, 128): "bddb7b94b0a46a5a",
    ("prod", "float64", 8, 1024): "502ad976ac0dbc22",
    ("prod", "float64", 8, 4096): "1df5b26dcb0e5e81",
    ("prod", "float64", 8, 4097): "35f8e0d3a5f71bdc",
    ("prod", "float64", 8, 6143): "d3dfdc2fb1351676",
    ("prod", "float64", 8, 8192): "2143eeecee209dbf",
    ("prod", "float64", 8, 20000): "b7ce5185b397b4d6",
    ("prod", "float64", 8, 40000): "a32dd963aad3d9ae",
    ("prod", "float64", 8, 100003): "de84b76fd6125a71",
    ("prod", "float64", 3, 262144): "030ca5e848e497ad",
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
    import hashlib

    b = t.cpu().contiguous().flatten().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()[:16]


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestInnerTreeOrder(TestCase):
    # Cover each plan shape/DAG (multirow, looped, split) and ragged identity tails.
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
        # Every N needs a plan or silently keeps launch order. Enforce MAX_UNROLL because
        # TileMap raises above it.
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
        # The matching DAG must reproduce upstream bit for bit.
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
                    # Avoid order-insensitive overflow or underflow.
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
        # Dtype sets the 16-byte vector width; widths below four fold linearly, changing the DAG.
        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import inner_tree_kernel as up

        # Prod has a distinct identity and ragged padding and previously covered only fp32.
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
                        # Avoid order-insensitive overflow or underflow.
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
        # A stray identity changes all -0.0 rows because 0.0 + -0.0 is +0.0. Upstream seeds
        # only some shapes, and close comparison hides the resulting disagreement.
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
        # An N-only DAG makes each row independent of batch size, unlike the default order.
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
        # Pin bits across all plan shapes, ragged Ns, and four dtypes.
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
        # Exact only covers the tile's N from column 0; a bound or offset invalidates its
        # unpredicated load. No plan may pair them because wrong columns can look plausible.
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
        # Verify the input distinguishes orders in fp32/fp64 with N > 4; narrowing erases the
        # difference, and a single thread owns all of N=4.
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
        # Test dispatcher routing because wrong paths still compute valid reductions.
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
        # Exercise per-field staging/partials, two outputs from one accumulator, and ragged
        # identity padding. Compare only plumbing with launch order because the DAGs differ.
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
                    # Compare indices exactly so tolerance cannot hide a wrong field buffer.
                    if a.dtype in (torch.int32, torch.int64):
                        self.assertEqual(a, b, msg=f"{label} field {k}")
                    else:
                        self.assertEqual(
                            a, b, atol=1e-4, rtol=1e-4, msg=f"{label} field {k}"
                        )

    def test_tree_fold_matches_the_serial_fold_for_every_value_trait(self):
        # Trait law: combine(leaf(a), leaf(b)) == reduce(reduce(init(), a), b).
        # Otherwise tree order can fold raw values into plausible wrong results.
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

    def test_staging_is_actually_used_in_the_mid_band(self):
        # Bit-neutral hashes cannot prove staging ran. Check its gate and both exclusions.
        from torch._native.ops.reductions import kernel_rowtile as rt

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
