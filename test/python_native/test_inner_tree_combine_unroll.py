# Owner(s): ["module: dsl-native-ops"]

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


@unittest.skipUnless(TEST_CUTEDSL, "requires CuTeDSL")
class TestInnerTreeCombineUnroll(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "unroll,rows",
        [(0, 1), (64, 33), (4096, 1)],
    )
    def test_unroll_preserves_staging_geometry(self, unroll, rows):
        """Apply combine unroll without changing the selected staging geometry."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        policy = rt._ITREE_ARCH[(10, 0)]._replace(combine_unroll=0)
        split = rt._ItreePlan(
            "split", 4, 16, 4, 2, (), (), (8192, 8192, 8192, 512, 512)
        )
        caps = SimpleNamespace(smem_per_block_optin=232448)
        unrolled = policy._replace(combine_unroll=unroll)
        kwargs = dict(nfields=3, nrows=rows)
        with mock.patch.object(rt._hw, "caps", return_value=caps):
            baseline = rt.itree_combine_plan(split, 4, arch=policy, **kwargs)
            plan = rt.itree_combine_plan(split, 4, arch=unrolled, **kwargs)
        self.assertGreater(plan.combine_tile, 0)
        self.assertEqual(plan.combine_unroll, min(unroll, plan.combine_tile))
        self.assertEqual(plan._replace(combine_unroll=0), baseline)

    @parametrize(
        "updates",
        [{"combine_unroll": -1}, {"combine_rpb": 64}, {"combine_weights": True}],
        name_fn=lambda u: next(iter(u)),
    )
    def test_invalid_policy(self, updates):
        """Reject a negative unroll, async staging with more rows than copying lanes, and
        weights without uniform counts."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        policy = rt._ITREE_ARCH[(10, 0)]._replace(**updates)
        with self.assertRaisesRegex(ValueError, next(iter(updates))):
            rt.itree_combine_plan(None, 4, arch=policy)

    def test_nfields_must_match_trait(self):
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        trait = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        with self.assertRaisesRegex(ValueError, "disagrees with trait.nfields"):
            rt.itree_combine_plan(None, 4, nfields=2, trait=trait)

    @parametrize(
        "count,last,key,weights,staged",
        [
            (8192, 8192, "welford", False, True),
            (8192, 8192, "welford", True, True),
            (8192, 8191, "welford", True, True),
            (8192, 8192, "sum", True, True),
            (8192, 8192, "fp64", True, True),
            # Without a staged tile the count field stays.
            (8192, 8192, "welford", True, False),
        ],
    )
    def test_uniform_count_eligibility(self, count, last, key, weights, staged):
        """Enable uniform counts only when every partial has the same supported shape
        and the combine stages tiles."""
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        trait = {
            "welford": T.VarMeanOps(correction=1, acc=cutlass.Float32),
            "sum": T.SumOps(acc=cutlass.Float32),
            "fp64": T.VarMeanOps(correction=1, acc=cutlass.Float64),
        }[key]
        itemsize = trait.acc.width // 8
        split = rt._ItreePlan(
            "split", 4, 16, 1, 2, (), (), (1024, count, last, 512, 512)
        )
        arch = rt._ITREE_ARCH["default"]._replace(
            combine_rpb=32,
            combine_group_bytes=16,
            combine_async=8 if staged else 0,
            combine_max=512,
            combine_unroll=4,
        )
        uniform = arch._replace(uniform_count=True, combine_weights=weights)
        kwargs = dict(nrows=1, trait=trait)
        caps = SimpleNamespace(smem_per_block_optin=232448)
        with mock.patch.object(rt._hw, "caps", return_value=caps):
            baseline = rt.itree_combine_plan(split, itemsize, arch=arch, **kwargs)
            candidate = rt.itree_combine_plan(split, itemsize, arch=uniform, **kwargs)
        supported = staged and (count, last, key) == (8192, 8192, "welford")
        self.assertEqual(candidate.combine_tile > 0, staged)
        self.assertEqual(candidate.combine_count, count if supported else 0)
        self.assertEqual(candidate.combine_weights, weights and supported)
        self.assertEqual(
            candidate._replace(combine_count=0, combine_weights=False), baseline
        )

    @parametrize("weights", [False, True])
    def test_uniform_count_smem_budget(self, weights):
        """Budget uniform tiles without the count field, plus one shared weight row."""
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        trait = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        split = rt._ItreePlan(
            "split", 4, 16, 1, 2, (), (), (1024, 8192, 8192, 512, 512)
        )
        arch = rt._ITREE_ARCH[(10, 0)]._replace(combine_max=512)
        uniform = arch._replace(uniform_count=True, combine_weights=weights)
        kwargs = dict(nrows=32, trait=trait)
        # 32 rows of 516 staged floats: three fields exceed the budget and two fit it, and two
        # plus one weight row of 512 fit it exactly but not a byte less.
        fit = 32 * 516 * 4 * 2 + 512 * 4
        for smem, weighted_tile in ((fit - 1, 256), (fit, 512)):
            caps = SimpleNamespace(smem_per_block_optin=smem)
            with mock.patch.object(rt._hw, "caps", return_value=caps):
                baseline = rt.itree_combine_plan(split, 4, arch=arch, **kwargs)
                candidate = rt.itree_combine_plan(split, 4, arch=uniform, **kwargs)
            self.assertEqual(baseline.combine_tile, 256)
            self.assertEqual(candidate.combine_tile, weighted_tile if weights else 512)
            self.assertEqual(candidate.combine_weights, weights)

    @parametrize(
        "split", [(2**24, 8192, 8192, 512, 512), (1024, 2**24, 2**24, 512, 512)]
    )
    def test_uniform_combine_ok_limits(self, split):
        """Accept 2**24 partials and 2**24-value partials, whose counts stay exact in fp32."""
        import cutlass

        from torch._native.ops.reductions import tile, traits as T

        trait = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        self.assertTrue(tile.uniform_combine_ok(trait, split))

    @parametrize(
        "updates,msg",
        [
            ({"combine_count": 0}, "weights require uniform"),
            ({"shape": "split"}, "staged combine plan"),
            ({"combine_tile": 0}, "staged combine plan"),
            ({"split": (1024, 4096, 8192, 512, 512)}, "staged combine plan"),
            ({"trait": "sum"}, "FP32"),
            ({"trait": "fp64"}, "FP32"),
            ({"split": (1024,)}, "staged combine plan"),
            ({"split": (1024, 8192, 8192)}, "staged combine plan"),
            ({"split": (0, 8192, 8192, 512, 512)}, "FP32"),
            ({"split": (2**25, 8192, 8192, 512, 512)}, "FP32"),
            ({"split": (1024, 2**25, 2**25, 512, 512), "combine_count": 2**25}, "FP32"),
            ({"split": (1024, 6144, 6144, 512, 512), "combine_count": 6144}, "FP32"),
            ({"split": (1024, 8192, 8191, 512, 512)}, "FP32"),
            ({"rows_per_block": 64}, "rows_per_block <= 32"),
        ],
        name_fn=lambda updates, msg: "_".join(
            f"{k}_{v}" for k, v in updates.items()
        ).translate(str.maketrans(",", "x", " ()")),
    )
    def test_uniform_count_tile_rejects(self, updates, msg):
        """Reject weights without counts, plans that stage no complete combine tile,
        counts outside complete FP32 Welford partials, and staged blocks of more rows
        than a warp has lanes."""
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, tile, traits as T

        traits = {
            "welford": T.VarMeanOps(correction=1, acc=cutlass.Float32),
            "sum": T.SumOps(acc=cutlass.Float32),
            "fp64": T.VarMeanOps(correction=1, acc=cutlass.Float64),
        }
        valid = rt._ItreePlan(
            "combine",
            1,
            0,
            1,
            2,
            (),
            (),
            (1024, 8192, 8192, 512, 512),
            combine_grp=4,
            combine_tile=512,
            combine_count=8192,
            combine_weights=True,
        )
        kwargs = dict(nouts=2, order="inner_tree")
        tile.TileReduce(traits["welford"], None, "general", 0, itree=valid, **kwargs)
        fields = {"trait": "welford"} | updates
        trait = traits[fields.pop("trait")]
        plan = valid._replace(**fields)
        with self.assertRaisesRegex(ValueError, msg):
            tile.TileReduce(trait, None, "general", 0, itree=plan, **kwargs)


@unittest.skipUnless(
    TEST_CUDA and SM90OrLater and TEST_CUTEDSL, "requires CUDA and CuTeDSL"
)
class TestInnerTreeCombineUnrollDevice(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_unaligned_partials_rejected_with_cached_plan(self, device):
        """Reject unaligned partials even when a compatible plan is already cached."""
        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits,
        )

        plan = rt._ItreePlan(
            "combine",
            1,
            0,
            1,
            2,
            (),
            (),
            (512, 8192, 8192, 512, 512),
            combine_grp=4,
            combine_tile=512,
            combine_unroll=4,
        )
        op = kg.ReduceBlock(
            traits.SumOps(acc=cutlass.Float32),
            count=512,
            num_o=1,
            red_pairs=[],
            kept_pairs=[],
            order="inner_tree",
            itree=plan,
        )
        aligned = torch.ones(512, device=device)
        unaligned = torch.ones(513, device=device)[1:]
        output = torch.empty(1, device=device)
        key = ("test_unroll_alignment", "sum") + op.cache_sig
        kg._launch(op, key, [aligned], [output])
        torch.cuda.synchronize(device)
        self.assertEqual(output, torch.full_like(output, 512))
        with self.assertRaisesRegex(ValueError, "aligned partial buffers"):
            kg._launch(op, key, [unaligned], [output])

    @parametrize("unroll", [0, 4])
    @parametrize(
        "name,uniform",
        [
            ("sum", "off"),
            ("var_mean", "off"),
            ("var_mean", "counts"),
            ("var_mean", "weights"),
        ],
    )
    def test_multi_tile_async_combine_matches_default(
        self, device, name, uniform, unroll
    ):
        """Fold several cp.async tiles, fully and partly unrolled and with uniform Welford
        counts, to the default bits through the row, general-axis and, for var_mean,
        ordered column entry points."""
        import cutlass

        from torch._native.ops.reductions import (
            kernel_coltile as ct,
            kernel_general as kg,
            kernel_rowtile as rt,
            traits as T,
        )

        trait = (
            T.SumOps(acc=cutlass.Float32)
            if name == "sum"
            else T.VarMeanOps(correction=1, acc=cutlass.Float32)
        )
        nouts = 1 if name == "sum" else 2
        dsts = [torch.float32] * nouts
        x = torch.randn(1, 1 << 23, device=device)
        # A strided copy of the row takes the general-axis indexed tree.
        strided = torch.stack([x[0], x[0]], dim=1)[:, 0]
        # 256 split partitions per column, two 128-partial tiles.
        cols = None if name == "sum" else torch.randn(1 << 21, 32, device=device)

        def run(arch, key):
            with mock.patch.object(rt, "_itree_arch", return_value=arch):
                combine = rt.itree_combine_plan(
                    rt.trait_itree_plan(trait, x.shape[1], 1, 4, device=x.device),
                    4,
                    nrows=1,
                    trait=trait,
                )
                rows = rt.reduce_row_tile(
                    trait, key, x, dsts, nouts=nouts, order="inner_tree"
                )
                general = kg._reduce_all(
                    trait, key, strided, dsts, nouts, order="inner_tree"
                )
                if cols is None:
                    return combine, None, rows + general
                plan = rt.trait_itree_plan(trait, *cols.shape, 4, stage=False)
                col_combine = rt.itree_combine_plan(plan, 4, nrows=32, trait=trait)
                config = ct.OrderedColConfig(0)
                col = ct.reduce_ordered_col(
                    trait, f"{key}_col", cols, dsts, nouts, config=config
                )
            return combine, col_combine, rows + general + col

        *_, want = run(rt._ITREE_ARCH["default"], f"multitile_default_{name}")
        # 128-partial tiles still repeat, and keep full unrolling's compile short.
        arch = rt._ITREE_ARCH[(9, 0)]._replace(
            combine_unroll=unroll,
            combine_max=128,
            uniform_count=uniform != "off",
            combine_weights=uniform == "weights",
        )
        key = f"multitile_{unroll}_{uniform}_{name}"
        combine, col_combine, got = run(arch, key)
        self.assertGreater(combine.split[0] // max(combine.combine_tile, 1), 1)
        self.assertEqual(combine.combine_unroll, unroll)
        self.assertEqual(bool(combine.combine_count), uniform != "off")
        self.assertEqual(combine.combine_weights, uniform == "weights")
        # Both entry points ran this plan: the row stage-two key ends with its sig, and the
        # general one has the input alignments after it.
        row_sigs = {k[-1] for k in rt._CACHE if k[:2] == ("rowitree2", key)}
        general = {k[-2] for k in kg._COMPILE_CACHE if k[1:3] == ("genitree2", key)}
        self.assertEqual(row_sigs, {combine.sig})
        self.assertEqual(general, {combine.sig})
        if col_combine is not None:
            # Split columns fold their partials through the general genitree2 combine.
            self.assertGreater(col_combine.split[0] // col_combine.combine_tile, 1)
            self.assertEqual(bool(col_combine.combine_count), uniform != "off")
            self.assertEqual(col_combine.combine_weights, uniform == "weights")
            col_key = ("genitree2", f"{key}_col")
            col_sigs = {k[-2] for k in kg._COMPILE_CACHE if k[1:3] == col_key}
            self.assertEqual(col_sigs, {col_combine.sig})
        for g, w in zip(got, want):
            self.assertEqual(g.view(torch.int32), w.view(torch.int32))

    @parametrize("weights", [False, True])
    @parametrize("rpb", [1, 4])
    @parametrize("unroll", [0, 4])
    def test_uniform_count_bits(self, device, unroll, rpb, weights):
        """Preserve exact bits at uniform-count boundaries where staged tail guards change."""
        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits,
        )

        # 33 rows leave the last 4-row block with one live row.
        rows, partials, count = 33, 1024, 8192
        trait = traits.VarMeanOps(correction=1, acc=cutlass.Float32)
        means = torch.randn(rows, partials, device=device).mul_(8).add_(1024)
        means[0].fill_(-0.0)
        means[1, 17] = float("nan")
        means[2, 33] = float("inf")
        parts = [
            means.flatten(),
            torch.rand_like(means).flatten(),
            torch.full_like(means, count).flatten(),
        ]
        plan = rt._ItreePlan(
            "combine",
            1,
            0,
            rpb,
            2,
            (),
            (),
            (partials, count, count, 512, 512),
            combine_grp=4,
            # 128-partial tiles keep full unrolling's compile short.
            combine_tile=512 if unroll else 128,
            combine_unroll=unroll,
        )
        baseline = [torch.empty(rows, device=device) for _ in range(2)]
        actual = [torch.empty_like(t) for t in baseline]

        def launch(selected, fields, outs):
            block = kg.ReduceBlock(
                trait,
                count=partials,
                num_o=rows,
                red_pairs=[],
                kept_pairs=[],
                project_n=count * partials,
                nouts=2,
                order="inner_tree",
                itree=selected,
            )
            key = ("test_uniform_count",) + block.cache_sig
            kg._launch(block, key, fields, outs)
            return block.cache_sig

        candidate = plan._replace(combine_count=count, combine_weights=weights)
        # The candidate derives counts, so a poisoned count field must not reach it.
        poisoned = parts[:2] + [torch.full_like(parts[2], float("nan"))]
        # Distinct kernels, so the bits below compare two compiled folds.
        sigs = launch(plan, parts, baseline), launch(candidate, poisoned, actual)
        self.assertNotEqual(*sigs)
        for got, expected in zip(actual, baseline):
            self.assertEqual(got.view(torch.uint8), expected.view(torch.uint8))


instantiate_parametrized_tests(TestInnerTreeCombineUnroll)
instantiate_device_type_tests(
    TestInnerTreeCombineUnrollDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
