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
    HardwareClassification,
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
    hw_classification = HardwareClassification.GENERIC

    def general(self, **updates):
        args = dict(
            cc=(10, 3),
            dtype=torch.float32,
            trait_key="sum",
            count=1024,
            num_o=65536,
            red_pairs=((1024, 2),),
            kept_pairs=((65536, 2048),),
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

    @parametrize("cc", [(9, 0), (10, 0), (10, 3), (11, 0)])
    def test_architecture_isolation(self, cc):
        """Return the generic defaults until an architecture adds its own rules."""
        self.assertEqual(self.general(cc=cc), kg._GeneralConfig())
        self.assertEqual(
            rt.select_row_order(
                cc,
                torch.float32,
                "var0",
                1024,
                65536,
                nfields=3,
                nouts=1,
            ),
            "linear",
        )

    def test_one_output_reductions_keep_wide_blocks(self):
        """Launch one-output strided reductions with the reduce-all block width."""
        self.assertEqual(
            self.general(num_o=1, kept_pairs=()),
            kg._GeneralConfig(block=kg._K0_ALL_BLOCK, rule="general_one_output"),
        )

    def test_inner_tree_cannot_decline_into_unordered_row(self):
        """Raise for explicit inner-tree order on a launch shape the order cannot use."""
        x = SimpleNamespace(
            dim=lambda: 2,
            is_cuda=True,
            stride=lambda axis: 1,
            shape=(8, 1024),
            dtype=torch.float32,
            element_size=lambda: 4,
            device="cuda",
        )
        with self.assertRaisesRegex(ValueError, "inner_tree.*cannot"):
            rt.reduce_row_tile(
                object(),
                "sum",
                x,
                [torch.float32],
                threads_per_row=32,
                order="inner_tree",
            )

    def test_fallback_warns_once_per_reason(self):
        """Report each distinct reason a gated inner-tree request fell back."""
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            mock.patch.object(rt, "_warned_inner_tree_reasons", set()),
        ):
            with self.assertWarnsRegex(UserWarning, "first"):
                rt.inner_tree_fallback("first")
            with self.assertWarnsRegex(UserWarning, "second"):
                rt.inner_tree_fallback("second")
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                rt.inner_tree_fallback("first")


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "CuTeDSL requires Hopper or later")
class TestReductionConfigDevice(TestCase):
    hw_classification = HardwareClassification.CUDA

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
    def test_explicit_order_overrides_environment(self, device, stride):
        """Tune an explicit unordered request as unordered even under the env gate."""
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
            mock.patch.object(
                rt, "select_row_order", wraps=rt.select_row_order
            ) as row_select,
            mock.patch.object(
                kg, "select_general_config", wraps=kg.select_general_config
            ) as general_select,
            mock.patch.object(rt, "inner_tree_fallback") as fallback,
        ):
            got = kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32),
                "sum",
                x,
                [1],
                torch.float32,
                order="unordered",
            )
        # Contiguous rows tune the row kernel; strided rows tune the general kernel.
        self.assertEqual(row_select.called, stride == 1)
        self.assertEqual(general_select.called, stride == 2)
        fallback.assert_not_called()
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @parametrize("stride", [1, 2])
    def test_selectors_can_pick_inner_tree_for_unordered(self, device, stride):
        """Serve a selector's inner-tree pick; a general-route decline stays silent."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn((64, 2048 * stride), device=device)[:, ::stride]
        with (
            mock.patch.object(rt, "select_row_order", return_value="inner_tree"),
            mock.patch.object(
                kg,
                "select_general_config",
                return_value=kg._GeneralConfig(kernel_order="inner_tree"),
            ),
            mock.patch.object(kg, "_try_indexed_itree", return_value=None) as indexed,
            mock.patch.object(rt, "reduce_row_tile", wraps=rt.reduce_row_tile) as row,
            mock.patch.object(rt, "inner_tree_fallback") as fallback,
        ):
            got = kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32), "sum", x, [1], torch.float32
            )
        if stride == 1:
            self.assertEqual(row.call_args.kwargs["order"], "inner_tree")
        else:
            indexed.assert_called_once()
        fallback.assert_not_called()
        self.assertEqual(got, x.double().sum(1).float(), atol=1e-4, rtol=1e-4)

    def test_row_kernel_leaves_the_environment_to_dispatch(self, device):
        """Follow only the order argument in direct row calls, even under strict."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(64, 4096, device=device)
        trait = traits.SumOps(acc=cutlass.Float32)
        f32 = [torch.float32]
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "strict"}),
            mock.patch.object(rt, "_run_itree") as itree,
        ):
            (got,) = rt.reduce_row_tile(trait, "row_env", x, f32)
            with self.assertRaisesRegex(ValueError, "cannot be honoured"):
                rt.reduce_row_tile(
                    trait, "row_env", x, f32, threads_per_row=32, order="inner_tree"
                )
        itree.assert_not_called()
        self.assertEqual(got, x.double().sum(1).float(), atol=1e-4, rtol=1e-4)

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
                mock.patch.object(rt, "_warned_inner_tree_reasons", set()),
                self.assertWarnsRegex(UserWarning, "32-bit logical extents"),
            ):
                got = kg.reduce_all(trait, "sum", x, torch.float32)
        self.assertEqual(got, x.double().sum().float(), atol=1e-4, rtol=1e-4)

    @parametrize("entry", ["reduce_dim", "reduce_all"])
    def test_empty_reduction_satisfies_any_order(self, device, entry):
        """Return the identity for an empty reduction under any order, silently."""
        import cutlass

        from torch._native.ops.reductions import traits

        trait = traits.SumOps(acc=cutlass.Float32)
        f32 = torch.float32
        x = torch.empty(0, device=device)
        calls = {
            "reduce_dim": lambda **kw: kg.reduce_dim(trait, "sum", x, [0], f32, **kw),
            "reduce_all": lambda **kw: kg.reduce_all(trait, "sum", x, f32, **kw),
        }
        expected = torch.zeros((), device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(calls[entry](order="inner_tree"), expected)
            with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "strict"}):
                self.assertEqual(calls[entry](), expected)

    def test_strided_reduce_all_uses_the_general_selector(self, device):
        """Size strided reduce-all launches with the general kernel's selector."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(64, 128, device=device)[:, ::2]
        with mock.patch.object(
            kg, "select_general_config", wraps=kg.select_general_config
        ) as select:
            got = kg.reduce_all(
                traits.SumOps(acc=cutlass.Float32), "sum", x, torch.float32
            )
        select.assert_called_once()
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
            mock.patch.object(rt, "_warned_inner_tree_reasons", set()),
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
