# Owner(s): ["module: dsl-native-ops"]

import os
import sys
import unittest
import warnings
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
        [
            ("0", "0", "unordered"),
            ("1", "0", "unordered"),
            ("0", "1", "inner_tree"),
            ("0", "strict", "inner_tree"),
        ],
    )
    def test_explicit_order_and_live_environment(self, sum_gate, native_gate, expected):
        """Resolve explicit order ahead of the live gate; the sum rollout flag is separate."""
        with mock.patch.dict(
            os.environ,
            {"PYTORCH_SUM_INNER_TREE": sum_gate, rt._INNER_TREE_ENV: native_gate},
        ):
            self.assertEqual(rt.reduction_order(), expected)
            self.assertEqual(rt.reduction_order("unordered"), "unordered")
            self.assertEqual(rt.reduction_order("inner_tree"), "inner_tree")
        with self.assertRaisesRegex(ValueError, "order"):
            rt.reduction_order("linear")

    @parametrize("cc", [(9, 0), (10, 3), (11, 0)])
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

    def test_tuned_inner_tree_without_plan_falls_back(self, device):
        """Use a linear row kernel when tuning picks inner-tree but no plan exists."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(64, 1024, device=device)
        with torch.backends.python_native.cutedsl.disabled():
            expected = x.sum(1)
        with (
            mock.patch.object(rt, "select_row_order", return_value="inner_tree"),
            mock.patch.object(rt, "trait_itree_plan", return_value=None),
        ):
            got = kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32),
                "sum",
                x,
                [1],
                torch.float32,
                order="unordered",
            )
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @parametrize("entry", ["reduce_dim2", "reduce_all", "reduce_all2"])
    def test_order_keyword_reaches_the_row_kernel(self, device, entry):
        """Thread an explicit order through every reduce entry point."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(1, 4096, device=device)
        one = traits.SumOps(acc=cutlass.Float32)
        two = traits.VarMeanOps(correction=1, acc=cutlass.Float32)
        f32 = torch.float32
        calls = {
            "reduce_dim2": lambda order: kg.reduce_dim2(
                two, "var_mean", x, [1], [f32, f32], order=order
            ),
            "reduce_all": lambda order: (
                kg.reduce_all(one, "sum", x, f32, order=order),
            ),
            "reduce_all2": lambda order: kg.reduce_all2(
                two, "var_mean", x, [f32, f32], order=order
            ),
        }
        with mock.patch.object(rt, "reduce_row_tile", wraps=rt.reduce_row_tile) as row:
            ordered = calls[entry]("inner_tree")
        orders = [c.kwargs.get("order") for c in row.call_args_list]
        self.assertIn("inner_tree", orders)
        # An explicit unordered request overrides the env gate.
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            mock.patch.object(rt, "reduce_row_tile", wraps=rt.reduce_row_tile) as row,
        ):
            unordered = calls[entry]("unordered")
        orders = {c.kwargs.get("order") for c in row.call_args_list}
        self.assertEqual(orders, {"linear"})
        for got, ref in zip(ordered, unordered):
            self.assertEqual(got, ref, atol=1e-4, rtol=1e-4)

    def test_inner_tree_rejects_64bit_extents(self, device):
        """Refuse inner-tree order when the reduced extent needs 64-bit indexing."""
        import cutlass

        from torch._native.ops.reductions import traits

        # A stride-0 view reaches the extent check without allocating 2^31 elements.
        x = torch.zeros(1, 1, device=device).expand(2, 2**31)
        with self.assertRaisesRegex(ValueError, "32-bit logical extents"):
            kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32),
                "sum",
                x,
                [1],
                torch.float32,
                order="inner_tree",
            )

    def test_reduce_all_rejects_64bit_extents(self, device):
        """Raise for explicit order beyond 32-bit extents; fall back under the gate."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(4096, device=device)
        trait = traits.SumOps(acc=cutlass.Float32)
        with mock.patch.object(kg, "_INT32_LIMIT", 1024):
            with self.assertRaisesRegex(ValueError, "32-bit logical extents"):
                kg.reduce_all(trait, "sum", x, torch.float32, order="inner_tree")
            with (
                mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
                mock.patch.object(rt, "_warned_inner_tree_fallback", False),
                self.assertWarnsRegex(UserWarning, "32-bit logical extents"),
            ):
                got = kg.reduce_all(trait, "sum", x, torch.float32)
        self.assertEqual(got, x.double().sum().float(), atol=1e-4, rtol=1e-4)

    @parametrize("entry", ["reduce_dim", "reduce_all"])
    def test_empty_reduction_has_no_inner_tree_order(self, device, entry):
        """Treat an empty reduction as geometry the inner-tree order cannot serve."""
        import cutlass

        from torch._native.ops.reductions import traits

        trait = traits.SumOps(acc=cutlass.Float32)
        f32 = torch.float32
        calls = {
            "reduce_dim": lambda **kw: kg.reduce_dim(
                trait, "sum", torch.empty(4, 0, device=device), [1], f32, **kw
            ),
            "reduce_all": lambda **kw: kg.reduce_all(
                trait, "sum", torch.empty(0, device=device), f32, **kw
            ),
        }
        with self.assertRaisesRegex(ValueError, "empty reduction"):
            calls[entry](order="inner_tree")
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "strict"}),
            self.assertRaisesRegex(ValueError, "empty reduction"),
        ):
            calls[entry]()

    @parametrize("block", [None, 64])
    def test_reduce_all_tunes_unless_block_is_given(self, device, block):
        """Tune strided reduce-all with the general selector unless block is given."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(64, 128, device=device)[:, ::2]
        with mock.patch.object(
            kg, "select_general_config", wraps=kg.select_general_config
        ) as select:
            got = kg.reduce_all(
                traits.SumOps(acc=cutlass.Float32), "sum", x, torch.float32, block=block
            )
        self.assertEqual(select.called, block is None)
        self.assertEqual(got, x.double().sum().float(), atol=1e-4, rtol=1e-4)

    def test_env_gate_falls_back_with_one_warning(self, device):
        """Serve unservable geometry unordered under the env gate, warning only once."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn((8, 2048), device=device)[:, ::2]
        with torch.backends.python_native.cutedsl.disabled():
            expected = x.sum(1)
        call = lambda: kg.reduce_dim(  # noqa: E731
            traits.SumOps(acc=cutlass.Float32), "sum", x, [1], torch.float32
        )
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            mock.patch.object(kg, "_try_indexed_itree", return_value=None),
            mock.patch.object(rt, "_warned_inner_tree_fallback", False),
        ):
            with self.assertWarnsRegex(UserWarning, "could not be honoured"):
                got = call()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                again = call()
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(again, expected, atol=1e-4, rtol=1e-4)

    def test_strict_env_gate_raises(self, device):
        """Raise under the strict env gate instead of changing reduction order."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn((8, 2048), device=device)[:, ::2]
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "strict"}),
            mock.patch.object(kg, "_try_indexed_itree", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, f"{rt._INNER_TREE_ENV}=strict"):
                kg.reduce_dim(
                    traits.SumOps(acc=cutlass.Float32), "sum", x, [1], torch.float32
                )


instantiate_parametrized_tests(TestReductionConfig)
instantiate_device_type_tests(TestReductionConfigDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
