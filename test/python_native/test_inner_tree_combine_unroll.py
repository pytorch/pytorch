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
        with mock.patch.object(rt._hw, "caps", return_value=caps):
            with mock.patch.object(rt, "_itree_arch", return_value=policy):
                baseline = rt.itree_combine_plan(split, 4, nfields=3, nrows=rows)
            with mock.patch.object(
                rt, "_itree_arch", return_value=policy._replace(combine_unroll=unroll)
            ):
                plan = rt.itree_combine_plan(split, 4, nfields=3, nrows=rows)
        self.assertGreater(plan.combine_tile, 0)
        self.assertEqual(plan.combine_unroll, min(unroll, plan.combine_tile))
        self.assertEqual(plan._replace(combine_unroll=0), baseline)

    @parametrize(
        "updates",
        [{"combine_unroll": -1}, {"combine_rpb": 64}],
        name_fn=lambda u: next(iter(u)),
    )
    def test_invalid_policy(self, updates):
        """Reject a negative unroll, and async staging with more rows than copying lanes."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        policy = rt._ITREE_ARCH[(10, 0)]._replace(**updates)
        with mock.patch.object(rt, "_itree_arch", return_value=policy):
            with self.assertRaisesRegex(ValueError, next(iter(updates))):
                rt.itree_combine_plan(None, 4)


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
    @parametrize("name", ["sum", "var_mean"])
    def test_multi_tile_async_combine_matches_default(self, device, name, unroll):
        """Fold several cp.async tiles, fully and partly unrolled, to the default bits."""
        import cutlass

        from torch._native.ops.reductions import kernel_rowtile as rt, traits as T

        trait = (
            T.SumOps(acc=cutlass.Float32)
            if name == "sum"
            else T.VarMeanOps(correction=1, acc=cutlass.Float32)
        )
        nouts = 1 if name == "sum" else 2
        x = torch.randn(1, 1 << 23, device=device)

        def run(arch, key):
            with mock.patch.object(rt, "_itree_arch", return_value=arch):
                plan = rt.trait_itree_plan(trait, x.shape[1], 1, 4, device=x.device)
                combine = rt.itree_combine_plan(plan, 4, nfields=trait.nfields, nrows=1)
                outs = rt.reduce_row_tile(
                    trait,
                    key,
                    x,
                    [torch.float32] * nouts,
                    nouts=nouts,
                    order="inner_tree",
                )
            return plan.split[0] // max(combine.combine_tile, 1), combine, outs

        _, _, want = run(rt._ITREE_ARCH["default"], f"multitile_default_{name}")
        # 128-partial tiles still repeat, and keep full unrolling's compile short.
        arch = rt._ITREE_ARCH[(9, 0)]._replace(combine_unroll=unroll, combine_max=128)
        ntiles, combine, got = run(arch, f"multitile_{unroll}_{name}")
        self.assertGreater(ntiles, 1)
        self.assertEqual(combine.combine_unroll, unroll)
        for g, w in zip(got, want):
            self.assertEqual(g.view(torch.int32), w.view(torch.int32))


instantiate_parametrized_tests(TestInnerTreeCombineUnroll)
instantiate_device_type_tests(
    TestInnerTreeCombineUnrollDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
