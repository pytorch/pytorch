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

from torch._native.ops.reductions import (
    kernel_coltile as ct,
    kernel_general as kg,
    kernel_rowtile as rt,
)


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

    @parametrize("cc", [(10, 3), (11, 0)])
    def test_architecture_isolation(self, cc):
        """Prevent SM107 tuning tables from changing neighboring architectures."""
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

    @parametrize(
        "dtype,width,mib,key,fields,nouts,expected",
        [
            (torch.float32, 256, 64, "mean", 1, 1, "inner_tree"),
            (torch.bfloat16, 256, 16, "sum", 1, 1, "inner_tree"),
            (torch.bfloat16, 256, 64, "sum", 1, 1, "linear"),
            (torch.bfloat16, 1024, 64, "argmaxi32", 2, 1, "inner_tree"),
            (torch.float32, 4096, 16, "varmean0", 3, 2, "inner_tree"),
        ],
    )
    def test_sm107_row_profiles(self, dtype, width, mib, key, fields, nouts, expected):
        """Pin one representative from every SM107 row-policy branch."""
        rows = (mib << 20) // (width * dtype.itemsize)
        self.assertEqual(
            rt.select_row_order(
                (10, 7),
                dtype,
                key,
                width,
                rows,
                order="unordered",
                nfields=fields,
                nouts=nouts,
            ),
            expected,
        )

    @parametrize(
        "dtype,mib,key,fields,rule",
        [
            (torch.float32, 256, "sum", 1, "rubin_strided_sum"),
            (torch.bfloat16, 16, "var0", 3, "rubin_strided_welford"),
            (torch.float32, 64, "mean", 1, "rubin_strided_inner"),
            (torch.bfloat16, 64, "argmaxi32", 2, "rubin_strided_argmax"),
        ],
    )
    def test_sm107_general_profiles(self, dtype, mib, key, fields, rule):
        """Pin each distinct SM107 strided policy."""
        rows = (mib << 20) // (1024 * dtype.itemsize)
        cfg = kg.select_general_config(
            (10, 7),
            dtype,
            key,
            1024,
            rows,
            ((1024, 2),),
            ((rows, 2048),),
            order="unordered",
            nfields=fields,
            nouts=1,
        )
        self.assertEqual(cfg.rule, rule)
        self.assertEqual(cfg.block, 64 if key == "sum" else 128)
        self.assertEqual(cfg.kernel_order, "linear" if key == "sum" else "inner_tree")

    def test_sm107_config_guards(self):
        """Keep calls outside SM107 policy contracts on generic policies."""
        self.assertEqual(
            self.general(
                cc=(10, 7),
                count=1023,
                red_pairs=((1023, 2),),
                kept_pairs=((65536, 2046),),
            ),
            kg._GeneralConfig(),
        )
        self.assertEqual(
            rt.select_row_order(
                (10, 7),
                torch.float32,
                "var0",
                1024,
                order="unordered",
                M=65536,
                nfields=3,
                nouts=1,
                alignment=8,
            ),
            "linear",
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

    def test_b200_full_reduction_config(self):
        """Pin B200 full-reduction launch choices at measured operation and size anchors."""
        count = (256 << 20) // torch.float16.itemsize
        args = (torch.float16, "argmaxi32", count)
        kwargs = dict(nfields=2, nouts=1)
        self.assertEqual(
            kg.select_all_config((10, 0), *args, **kwargs),
            kg._AllConfig(block=512, grid_mult=8),
        )
        self.assertEqual(kg.select_all_config((9, 0), *args, **kwargs), kg._AllConfig())

    @parametrize(
        "count,trait_key,nfields,expected",
        [
            (
                257,
                "sum",
                1,
                kg._GeneralConfig(kernel_order="inner_tree", rule="b200_strided_c256"),
            ),
            (
                1024,
                "argmaxi32",
                2,
                kg._GeneralConfig(block=32, rule="b200_strided_index"),
            ),
            (
                4095,
                "var1",
                3,
                kg._GeneralConfig(
                    block=64,
                    rule="b200_strided_welford",
                    uniform_tree=True,
                ),
            ),
            (
                4096,
                "mean",
                1,
                kg._GeneralConfig(block=32, rule="b200_strided_one_field"),
            ),
        ],
    )
    def test_b200_strided_config(self, count, trait_key, nfields, expected):
        """Pin B200 general-kernel choices for measured strided geometries."""
        num_o = 4096
        self.assertEqual(
            kg.select_general_config(
                (10, 0),
                torch.float16,
                trait_key,
                count,
                num_o,
                ((count, 2),),
                ((num_o, 2 * count),),
                order="unordered",
                nfields=nfields,
                nouts=1,
            ),
            expected,
        )

    @parametrize(
        "dtype,columns,trait_key,nfields,expected_rule",
        [
            (torch.float16, 257, "sum", 1, "b200_ragged_c257"),
            (torch.float16, 257, "std1", 3, "b200_ragged_welford_c257"),
            (
                torch.float32,
                4095,
                "argmaxi32",
                2,
                "b200_ragged_argmax_c4095_small",
            ),
            (torch.float16, 4095, "mean", 1, "b200_ragged_c4095_ordered"),
        ],
    )
    def test_b200_ragged_column_config(
        self, dtype, columns, trait_key, nfields, expected_rule
    ):
        """Pin B200 ragged column choices where tail handling changes the best mapping."""
        itemsize = dtype.itemsize
        rows = (16 << 20) // (columns * itemsize)
        cfg = ct.select_col_config(
            (10, 0),
            dtype,
            rows,
            columns,
            1,
            nfields,
            trait_key,
            itemsize=itemsize,
            acc_bits=32,
            nouts=1,
            alignment=16,
            contiguous=True,
        )
        self.assertEqual(cfg.rule, expected_rule)


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
