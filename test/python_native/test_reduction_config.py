# Owner(s): ["module: dsl-native-ops"]

import os
import sys
import unittest
import warnings
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

from torch._native.ops.reductions import (
    kernel_coltile as ct,
    kernel_general as kg,
    kernel_rowtile as rt,
    traits as T,
)


def _served(trait_key):
    """Tags of the row and general plans compiled for trait_key."""
    rows = {k[0] for k in rt._CACHE if k[1] == trait_key}
    return rows | {k[0][1] for k in kg._PLAN if k[0][2] == trait_key}


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
        """Return generic defaults on architectures without their own rules."""
        b200 = kg._GeneralConfig(block=32, rule="b200_strided_one_field")
        expected = b200 if cc == (10, 0) else kg._GeneralConfig()
        self.assertEqual(self.general(cc=cc), expected)
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
                M=65536,
                nfields=3,
                nouts=1,
                alignment=8,
            ),
            "linear",
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
        half = (torch.float16, "argmaxi32", count // 2)
        self.assertEqual(
            kg.select_all_config((10, 0), *half, **kwargs), kg._AllConfig()
        )
        plain = dict(nfields=1, nouts=1)
        self.assertEqual(
            kg.select_all_config((10, 0), torch.float16, "sum", count, **plain),
            kg._AllConfig(),
        )

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
                nfields=nfields,
                nouts=1,
            ),
            expected,
        )

    @parametrize(
        "dtype,count,updates,expected",
        [
            (
                torch.float32,
                1024,
                {"trait_key": "argmaxi32", "nfields": 2},
                kg._GeneralConfig(block=64, rule="b200_strided_index"),
            ),
            (torch.float16, 258, {}, kg._GeneralConfig()),
            (
                torch.float16,
                1023,
                {"trait_key": "argmaxi32", "nfields": 2},
                kg._GeneralConfig(),
            ),
            (torch.float16, 4096, {"alignment": 8}, kg._GeneralConfig()),
            (torch.float16, 4096, {"acc_bits": 64}, kg._GeneralConfig()),
        ],
    )
    def test_b200_strided_boundaries(self, dtype, count, updates, expected):
        """Pin the fp32 index block and keep off-anchor strided shapes on defaults."""
        num_o = 4096
        cfg = self.general(
            cc=(10, 0),
            dtype=dtype,
            count=count,
            num_o=num_o,
            red_pairs=((count, 2),),
            kept_pairs=((num_o, 2 * count),),
            **updates,
        )
        self.assertEqual(cfg, expected)

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

    @parametrize(
        "dtype,trait_key,nfields,size,expected",
        [
            (
                torch.bfloat16,
                "sum",
                1,
                256 << 20,
                ct.ColConfig("b200_ragged_c4095", 64, 512, 1, "partition", 8, 16),
            ),
            (
                torch.float32,
                "argmaxi32",
                2,
                16 << 20,
                ct.ColConfig(
                    "b200_ragged_argmax_c4095_small", 32, 8, 1, "column", 0, 64
                ),
            ),
            (
                torch.bfloat16,
                "argmaxi32",
                2,
                2 << 30,
                ct.ColConfig("b200_ragged_argmax_c4095", 64, 1024, 1, "column"),
            ),
            (
                torch.float16,
                "mean",
                1,
                16 << 20,
                ct.ColConfig(
                    "b200_ragged_c4095_ordered",
                    32,
                    33,
                    1,
                    "column",
                    kernel_order="inner_tree",
                ),
            ),
        ],
        name_fn=lambda dtype, trait_key, nfields, size, expected: expected.rule,
    )
    def test_b200_ragged_column_values(self, dtype, trait_key, nfields, size, expected):
        """Pin every field of the B200 ragged column rules, not just their names."""
        rows = size // (4095 * dtype.itemsize)
        kw = dict(itemsize=dtype.itemsize, acc_bits=32, nouts=1, alignment=16)
        args = ((10, 0), dtype, rows, 4095, 1, nfields, trait_key)
        cfg = ct.select_col_config(*args, **kw, contiguous=True)
        self.assertEqual(cfg, expected)

    @parametrize(
        "dtype,N,nfields,routed",
        [
            (torch.float16, 255, 1, True),
            (torch.float16, 253, 1, False),
            (torch.float16, 16383, 1, True),
            (torch.float16, 16385, 1, False),
            (torch.float16, 4096, 1, False),
            (torch.float32, 255, 2, True),
            (torch.bfloat16, 255, 2, False),
            (torch.bfloat16, 1025, 2, False),
            (torch.float32, 1025, 2, True),
            (torch.float32, 1023, 3, False),
            (torch.float32, 1025, 3, True),
        ],
        name_fn=lambda dtype, N, nfields, routed: f"{str(dtype)[6:]}_{N}_{nfields}f",
    )
    def test_b200_ragged_row_gate(self, dtype, N, nfields, routed):
        """Route odd rows to the ragged-vector kernel only inside the measured band."""
        self.assertEqual(kg._ragged_rows_ok((10, 0), dtype, N, 16, nfields), routed)
        self.assertFalse(kg._ragged_rows_ok((9, 0), dtype, N, 16, nfields))
        self.assertFalse(kg._ragged_rows_ok((10, 0), dtype, N, 8, nfields))
        self.assertFalse(kg._ragged_rows_ok((10, 0), dtype, N, 16, nfields, 64))

    def test_b200_ragged_rows_start_late_traits_later(self):
        """Keep absmax, absmin and general-p norms off ragged rows at N = 255."""
        late = (T.AbsMaxOps(), T.AbsMinOps(), T.NormOps(3), T.NormOps(0.5))
        early = (T.AMaxOps(), T.NormOps(1), T.NormOps(2), T.SumOps())
        self.assertEqual({kg._ragged_row_min_n(t) for t in late}, {1023})
        self.assertEqual({kg._ragged_row_min_n(t) for t in early}, {255})
        for N, routed in ((255, False), (1023, True)):
            ok = kg._ragged_rows_ok((10, 0), torch.float16, N, 16, 1, min_n=1023)
            self.assertEqual(ok, routed)


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "CuTeDSL requires Hopper or later")
class TestReductionConfigDevice(TestCase):
    hw_classification = HardwareClassification.CUDA

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
        key = f"explicit_unordered_{stride}"
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            warnings.catch_warnings(),
        ):
            # Compiling may warn on its own; only the env gate's fallback must not.
            warnings.filterwarnings("error", ".*could not be honoured")
            got = kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32),
                key,
                x,
                [1],
                torch.float32,
                order="unordered",
            )
        # Contiguous rows run the row kernel; strided rows run the general kernel.
        tags = _served(key)
        self.assertIn("rowtile" if stride == 1 else "reduce", tags)
        self.assertFalse(any("itree" in t for t in tags), tags)
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @parametrize("stride", [1, 2])
    def test_selectors_can_pick_inner_tree_for_unordered(self, device, stride):
        """Serve a selector's inner-tree pick without the env gate's warning."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn((64, 2048 * stride), device=device)[:, ::stride]
        key = f"selector_inner_tree_{stride}"
        with (
            mock.patch.object(rt, "select_row_order", return_value="inner_tree"),
            mock.patch.object(
                kg,
                "select_general_config",
                return_value=kg._GeneralConfig(kernel_order="inner_tree"),
            ),
            warnings.catch_warnings(),
        ):
            # Compiling may warn on its own; only the env gate's fallback must not.
            warnings.filterwarnings("error", ".*could not be honoured")
            got = kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32), key, x, [1], torch.float32
            )
        want = "rowitree" if stride == 1 else "genitree"
        self.assertTrue(any(t.startswith(want) for t in _served(key)), _served(key))
        self.assertEqual(got, x.double().sum(1).float(), atol=1e-4, rtol=1e-4)

    def test_row_kernel_leaves_the_environment_to_dispatch(self, device):
        """Follow only the order argument in direct row calls, even under strict."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(64, 4096, device=device)
        trait = traits.SumOps(acc=cutlass.Float32)
        f32 = [torch.float32]
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "strict"}):
            (got,) = rt.reduce_row_tile(trait, "row_env", x, f32)
            with self.assertRaisesRegex(ValueError, "cannot be honoured"):
                rt.reduce_row_tile(
                    trait, "row_env", x, f32, threads_per_row=32, order="inner_tree"
                )
        tags = _served("row_env")
        self.assertIn("rowtile", tags)
        self.assertFalse(any("itree" in t for t in tags), tags)
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
            "reduce_dim2": lambda key, order: kg.reduce_dim2(
                two, key, x, [1], [f32, f32], order=order
            ),
            "reduce_all": lambda key, order: (
                kg.reduce_all(one, key, x, f32, order=order),
            ),
            "reduce_all2": lambda key, order: kg.reduce_all2(
                two, key, x, [f32, f32], order=order
            ),
        }
        ordered = calls[entry](f"{entry}_ordered", "inner_tree")
        tags = _served(f"{entry}_ordered")
        self.assertTrue(any(t.startswith("rowitree") for t in tags), tags)
        # An explicit unordered request overrides the env gate.
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}):
            unordered = calls[entry](f"{entry}_unordered", "unordered")
        tags = _served(f"{entry}_unordered")
        self.assertIn("rowtile", tags)
        self.assertFalse(any("itree" in t for t in tags), tags)
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
        key = "strided_reduce_all"
        got = kg.reduce_all(traits.SumOps(acc=cutlass.Float32), key, x, torch.float32)
        self.assertTrue(_served(key) & {"reduce", "gensplit1"}, _served(key))
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
        # The patched limit makes the reduced extent count as 64-bit.
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            mock.patch.object(kg, "_INT32_LIMIT", 1024),
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
            mock.patch.object(kg, "_INT32_LIMIT", 1024),
        ):
            with self.assertRaisesRegex(ValueError, f"{rt._INNER_TREE_ENV}=strict"):
                kg.reduce_dim(
                    traits.SumOps(acc=cutlass.Float32), "sum", x, [1], torch.float32
                )


instantiate_parametrized_tests(TestReductionConfig)
instantiate_device_type_tests(TestReductionConfigDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
