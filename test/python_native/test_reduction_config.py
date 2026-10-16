# Owner(s): ["module: dsl-native-ops"]

import os
import sys
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


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

from torch._native.ops.reductions import kernel_general as kg, kernel_rowtile as rt


class TestReductionConfig(TestCase):
    def general(self, **updates):
        args = dict(
            cc=(10, 3),
            dtype=torch.float32,
            trait_key="sum",
            count=1024,
            num_o=65536,
            red_pairs=((1024, 2),),
            kept_pairs=((65536, 2048),),
            order="unordered",
            nfields=1,
            nouts=1,
        )
        args.update(updates)
        return kg.select_general_config(**args)

    @parametrize(
        "sum_gate,native_gate,expected",
        [("0", "0", "unordered"), ("1", "0", "inner_tree"), ("0", "1", "inner_tree")],
    )
    def test_explicit_order_and_live_environment(self, sum_gate, native_gate, expected):
        """Resolve explicit order ahead of live environment gates for predictable routing."""
        with mock.patch.dict(
            os.environ,
            {rt._SUM_INNER_TREE_ENV: sum_gate, rt._INNER_TREE_ENV: native_gate},
        ):
            self.assertEqual(rt.reduction_order(), expected)
            self.assertEqual(rt.reduction_order("unordered"), "unordered")
            self.assertEqual(rt.reduction_order("inner_tree"), "inner_tree")
        with self.assertRaisesRegex(ValueError, "order"):
            rt.reduction_order("linear")

    @parametrize("cc", [(9, 0), (10, 0), (10, 3), (11, 0)])
    def test_architecture_isolation(self, cc):
        """Prevent Rubin tuning tables from changing other architecture defaults."""
        self.assertEqual(self.general(cc=cc), kg._GeneralConfig())
        self.assertEqual(
            rt.select_row_order(
                cc,
                torch.float32,
                "var0",
                1024,
                65536,
                order="unordered",
                nfields=3,
                nouts=1,
            ),
            "linear",
        )
        self.assertEqual(
            self.general(cc=cc, order="inner_tree").kernel_order, "inner_tree"
        )

    def test_inner_tree_cannot_decline_into_unordered_row(self):
        """Forbid required inner-tree calls from silently using unordered row kernels."""
        x = SimpleNamespace(
            dim=lambda: 2,
            is_cuda=True,
            stride=lambda axis: 1,
            shape=(8, 1024),
            dtype=torch.float32,
            element_size=lambda: 4,
            device="cuda",
        )
        with mock.patch.object(rt, "trait_itree_plan", return_value=None):
            with self.assertRaisesRegex(ValueError, "inner_tree.*cannot"):
                rt.reduce_row_tile(
                    object(), "sum", x, [torch.float32], order="inner_tree"
                )


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "CuTeDSL requires Hopper or later")
class TestReductionConfigDevice(TestCase):
    def test_required_inner_tree_cannot_fall_back(self, device):
        """Raise when no ordered route exists instead of changing reduction order."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.empty((8, 2048), device=device)[:, ::2]
        with (
            mock.patch.object(kg, "_try_indexed_itree", return_value=None),
            mock.patch.object(kg, "_launch") as launch,
        ):
            with self.assertRaisesRegex(ValueError, "inner-tree.*cannot"):
                kg.reduce_dim(
                    traits.SumOps(acc=cutlass.Float32),
                    "sum",
                    x,
                    [1],
                    torch.float32,
                    order="inner_tree",
                )
            launch.assert_not_called()

    @parametrize("stride", [1, 2])
    def test_explicit_block_and_order_override_environment(self, device, stride):
        """Let explicit launch and order choices override environment defaults."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn((1024, 1024 * stride), device=device)[:, ::stride]
        with (
            torch._native._unconditional_masked(),
            torch.backends.python_native.cutedsl.disabled(),
        ):
            expected = x.sum(1)
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            mock.patch.object(rt, "select_row_order") as row_select,
            mock.patch.object(kg, "select_general_config") as general_select,
            mock.patch.object(kg, "_try_indexed_itree") as indexed,
            mock.patch.object(rt, "_run_itree") as row_tree,
        ):
            got = kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32),
                "sum",
                x,
                [1],
                torch.float32,
                block=64,
                order="unordered",
            )
            row_select.assert_not_called()
            general_select.assert_not_called()
            indexed.assert_not_called()
            row_tree.assert_not_called()
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)


instantiate_parametrized_tests(TestReductionConfig)
instantiate_device_type_tests(TestReductionConfigDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
