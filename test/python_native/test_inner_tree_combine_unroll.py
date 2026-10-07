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


instantiate_parametrized_tests(TestInnerTreeCombineUnroll)
instantiate_device_type_tests(
    TestInnerTreeCombineUnrollDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
