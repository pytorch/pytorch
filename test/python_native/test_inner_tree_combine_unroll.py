# Owner(s): ["module: dsl-native-ops"]

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


@unittest.skipUnless(TEST_CUTEDSL, "requires CuTeDSL")
class TestInnerTreeCombineUnroll(TestCase):
    @parametrize(
        "unroll,rows",
        [(0, 1), (64, 33), (4096, 1)],
    )
    def test_unroll_preserves_staging_geometry(self, unroll, rows):
        """Apply combine unroll without changing the selected staging geometry."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        policy = rt._ITREE_ARCH[(10, 0)]
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

    def test_invalid_unroll(self):
        """Reject invalid combine unroll values before they produce malformed kernels."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        with mock.patch.object(
            rt,
            "_itree_arch",
            return_value=rt._ITREE_ARCH[(10, 0)]._replace(combine_unroll=-1),
        ):
            with self.assertRaisesRegex(ValueError, "combine_unroll"):
                rt.itree_combine_plan(None, 4)

    @parametrize(
        "count,last,itemsize,fields,weights",
        [
            (8192, 8192, 4, 3, False),
            (8192, 8192, 4, 3, True),
            (8192, 8191, 4, 3, True),
            (8192, 8192, 4, 1, True),
        ],
    )
    def test_uniform_count_eligibility(self, count, last, itemsize, fields, weights):
        """Enable uniform counts only when every partial has the same supported shape."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        split = rt._ItreePlan(
            "split", 4, 16, 1, 2, (), (), (1024, count, last, 512, 512)
        )
        arch = rt._ITREE_ARCH["default"]._replace(
            combine_group_bytes=16,
            combine_async=8,
            combine_max=512,
            combine_unroll=4,
        )
        with mock.patch.object(
            rt._hw, "caps", return_value=SimpleNamespace(smem_per_block_optin=232448)
        ):
            baseline = rt.itree_combine_plan(
                split, itemsize, nfields=fields, nrows=1, arch=arch
            )
            candidate = rt.itree_combine_plan(
                split,
                itemsize,
                nfields=fields,
                nrows=1,
                arch=arch,
                uniform_count=True,
                combine_weights=weights,
            )
        supported = (count, last, itemsize, fields) == (8192, 8192, 4, 3)
        self.assertEqual(candidate.combine_count, count if supported else 0)
        self.assertEqual(candidate.combine_weights, weights and supported)
        self.assertEqual(
            candidate._replace(combine_count=0, combine_weights=False), baseline
        )


@unittest.skipUnless(
    TEST_CUDA and SM90OrLater and TEST_CUTEDSL, "requires CUDA and CuTeDSL"
)
class TestInnerTreeCombineUnrollDevice(TestCase):
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

    @parametrize("weights", [False, True])
    def test_uniform_count_bits(self, device, weights):
        """Preserve exact bits at uniform-count boundaries where staged tail guards change."""
        import cutlass

        from torch._native.ops.reductions import (
            kernel_general as kg,
            kernel_rowtile as rt,
            traits,
        )

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
            1,
            2,
            (),
            (),
            (partials, count, count, 512, 512),
            combine_grp=4,
            combine_tile=512,
            combine_unroll=4,
        )
        baseline = [torch.empty(rows, device=device) for _ in range(2)]
        actual = [torch.empty_like(t) for t in baseline]

        def launch(selected, outs):
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
            kg._launch(block, key, parts, outs)

        candidate = plan._replace(combine_count=count, combine_weights=weights)
        launch(plan, baseline)
        launch(candidate, actual)
        for got, expected in zip(actual, baseline):
            self.assertEqual(got.view(torch.uint8), expected.view(torch.uint8))


instantiate_parametrized_tests(TestInnerTreeCombineUnroll)
instantiate_device_type_tests(
    TestInnerTreeCombineUnrollDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
