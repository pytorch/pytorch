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
    tile,
    traits as T,
)


_TRAITS = {
    "sum": T.SumOps(),
    "mean": T.MeanOps(),
    "argmax": T.ArgMaxOps(),
    "var": T.WelfordOps(),
    "std": T.WelfordOps(take_sqrt=True),
    "var_mean": T.VarMeanOps(),
}


def _contract(trait):
    """Selector keywords for trait's structure."""
    contract = dict(nfields=trait.nfields, nouts=getattr(trait, "nouts", 1))
    welford = T.welford_outputs(trait)
    return contract | dict(trait_type=type(trait), welford_outs=welford)


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
            trait_type=T.SumOps,
            count=1024,
            num_o=65536,
            red_pairs=((1024, 2),),
            kept_pairs=((65536, 2048),),
            nfields=1,
            nouts=1,
            welford_outs=0,
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
        """Apply B200's strided rule on B200 only; other architectures keep defaults."""
        b200 = kg._GeneralConfig(block=32, rule="b200_strided_one_field")
        want = b200 if cc == (10, 0) else kg._GeneralConfig()
        self.assertEqual(self.general(cc=cc), want)
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
        args = (torch.float16, count)
        kwargs = dict(trait_type=T.ArgMaxOps, nfields=2, nouts=1, acc_bits=32)
        self.assertEqual(
            kg.select_all_config((10, 0), *args, **kwargs),
            kg._AllConfig(block=512, grid_mult=8),
        )
        self.assertEqual(kg.select_all_config((9, 0), *args, **kwargs), kg._AllConfig())
        wide = kwargs | {"acc_bits": 64}
        self.assertEqual(kg.select_all_config((10, 0), *args, **wide), kg._AllConfig())
        half = (torch.float16, count // 2)
        self.assertEqual(
            kg.select_all_config((10, 0), *half, **kwargs), kg._AllConfig()
        )
        plain = dict(trait_type=T.SumOps, nfields=1, nouts=1, acc_bits=32)
        self.assertEqual(
            kg.select_all_config((10, 0), torch.float16, count, **plain),
            kg._AllConfig(),
        )

    @parametrize(
        "count,trait,expected",
        [
            (
                257,
                "sum",
                kg._GeneralConfig(kernel_order="inner_tree", rule="b200_strided_c256"),
            ),
            (
                257,
                "var_mean",
                kg._GeneralConfig(kernel_order="inner_tree", rule="b200_strided_c256"),
            ),
            (
                1024,
                "argmax",
                kg._GeneralConfig(block=32, rule="b200_strided_index"),
            ),
            (
                4095,
                "var",
                kg._GeneralConfig(
                    block=64,
                    rule="b200_strided_welford",
                    uniform_tree=True,
                ),
            ),
            (
                4095,
                "var_mean",
                kg._GeneralConfig(
                    block=64,
                    rule="b200_strided_welford",
                    uniform_tree=True,
                ),
            ),
            (
                4096,
                "mean",
                kg._GeneralConfig(block=32, rule="b200_strided_one_field"),
            ),
        ],
        name_fn=lambda count, trait, expected: f"{trait}_{count}",
    )
    def test_b200_strided_config(self, count, trait, expected):
        """Pin B200 general-kernel choices for measured strided geometries."""
        num_o = 4096
        pairs = (((count, 2),), ((num_o, 2 * count),))
        args = ((10, 0), torch.float16, count, num_o, *pairs)
        cfg = kg.select_general_config(*args, **_contract(_TRAITS[trait]))
        self.assertEqual(cfg, expected)

    def test_b200_rules_match_trait_types(self):
        """Match rules by trait type within each rule's measured scope: argmin shares the
        index rules of full and strided reductions but not of ragged columns, and max_dim
        and nansum match no index or one-field rule."""
        count = (256 << 20) // torch.float16.itemsize
        index = dict(nfields=2, nouts=1, acc_bits=32)
        for trait_type in (T.ArgMaxOps, T.ArgMinOps):
            cfg = kg.select_all_config(
                (10, 0), torch.float16, count, trait_type=trait_type, **index
            )
            self.assertEqual(cfg, kg._AllConfig(block=512, grid_mult=8))
        num_o, count = 4096, 1024
        strided = dict(count=count, red_pairs=((count, 2),), num_o=num_o)
        strided |= dict(kept_pairs=((num_o, 2 * count),), dtype=torch.float16)
        default = kg._GeneralConfig()
        for trait, want in (
            (T.ArgMinOps(), kg._GeneralConfig(block=32, rule="b200_strided_index")),
            (T.MaxDimOps(), default),
            (T.NanSumOps(), default),
        ):
            cfg = self.general(cc=(10, 0), **strided, **_contract(trait))
            self.assertEqual(cfg, want, type(trait).__name__)
        rows = (16 << 20) // (4095 * 4)
        for trait, rule in (
            (T.ArgMaxOps(), "b200_ragged_argmax_c4095_small"),
            (T.ArgMinOps(), "default"),
            (T.NanSumOps(), "default"),
        ):
            contract = _contract(trait)
            nfields = contract.pop("nfields")
            args = ((10, 0), torch.float32, rows, 4095, 1, nfields)
            cfg = ct.select_col_config(*args, itemsize=4, acc_bits=32, **contract)
            self.assertEqual(cfg.rule, rule, type(trait).__name__)

    def test_welford_outputs_follow_trait_structure(self):
        """Count variance outputs from the trait, so selectors need no key spelling."""
        cases = (
            (T.WelfordOps(), 1),
            (T.WelfordOps(take_sqrt=True), 1),
            (T.VarMeanOps(), 2),
            (T.WelfordOps(return_mean=True), 0),
            (T.SumOps(), 0),
            (T.ArgMaxOps(), 0),
        )
        for trait, outputs in cases:
            self.assertEqual(T.welford_outputs(trait), outputs, type(trait).__name__)

    def test_b200_uniform_trees_stay_eligible(self):
        """_reduce drops uniform_tree for ineligible shapes, so B200's must pass the cap."""
        num_o = 4096
        for count in (1024, 4095, 4096):
            cfg = kg.select_general_config(
                (10, 0),
                torch.float32,
                count,
                num_o,
                ((count, 2),),
                ((num_o, 2 * count),),
                trait_type=T.WelfordOps,
                nfields=3,
                nouts=1,
                welford_outs=1,
            )
            self.assertTrue(cfg.uniform_tree)
            self.assertTrue(tile.uniform_tree_ok(T.WelfordOps(), count, cfg.block, 1))

    @parametrize(
        "dtype,count,updates,expected",
        [
            (
                torch.float32,
                1024,
                {"trait_type": T.ArgMaxOps, "nfields": 2},
                kg._GeneralConfig(block=64, rule="b200_strided_index"),
            ),
            (torch.float16, 258, {}, kg._GeneralConfig()),
            (
                torch.float16,
                1023,
                {"trait_type": T.ArgMaxOps, "nfields": 2},
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

    @parametrize("trait", ["sum", "argmax", "var"])
    def test_b200_wide_strided_rules_need_many_outputs(self, trait):
        """Below 2048 outputs only the c256 rule beats the default config."""
        for count, num_o, rule in ((256, 512, "b200_strided_c256"), (1024, 512, None)):
            cfg = self.general(
                cc=(10, 0),
                **_contract(_TRAITS[trait]),
                count=count,
                num_o=num_o,
                red_pairs=((count, 2),),
                kept_pairs=((num_o, 2 * count),),
            )
            self.assertEqual(cfg.rule, rule or kg._GeneralConfig().rule)

    @parametrize(
        "dtype,columns,trait,expected_rule",
        [
            (torch.float16, 257, "sum", "b200_ragged_c257"),
            (torch.float16, 257, "std", "b200_ragged_welford_c257"),
            (torch.float32, 4095, "argmax", "b200_ragged_argmax_c4095_small"),
            (torch.float16, 4095, "mean", "b200_ragged_c4095_ordered"),
            (torch.float32, 4095, "var", "b200_ragged_c4095_ordered"),
            (torch.bfloat16, 4095, "std", "b200_ragged_c4095_ordered"),
            # The Welford rules were measured with one output.
            (torch.float16, 257, "var_mean", "default"),
            (torch.float32, 4095, "var_mean", "default"),
        ],
        name_fn=lambda dtype, columns, trait, rule: f"{trait}_{columns}",
    )
    def test_b200_ragged_column_config(self, dtype, columns, trait, expected_rule):
        """Pin B200 ragged column choices where tail handling changes the best mapping."""
        itemsize = dtype.itemsize
        rows = (16 << 20) // (columns * itemsize)
        contract = _contract(_TRAITS[trait])
        nfields = contract.pop("nfields")
        args = ((10, 0), dtype, rows, columns, 1, nfields)
        cfg = ct.select_col_config(*args, itemsize=itemsize, acc_bits=32, **contract)
        self.assertEqual(cfg.rule, expected_rule)

    @parametrize(
        "dtype,trait,size,expected",
        [
            (
                torch.bfloat16,
                "sum",
                256 << 20,
                ct.ColConfig("b200_ragged_c4095", 64, 512, 1, "partition", 8, 16),
            ),
            (
                torch.float32,
                "argmax",
                16 << 20,
                ct.ColConfig(
                    "b200_ragged_argmax_c4095_small", 32, 8, 1, "column", 0, 64
                ),
            ),
            (
                torch.bfloat16,
                "argmax",
                2 << 30,
                ct.ColConfig("b200_ragged_argmax_c4095", 64, 1024, 1, "column"),
            ),
            (
                torch.float16,
                "mean",
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
            (
                torch.bfloat16,
                "var",
                64 << 20,
                ct.ColConfig(
                    "b200_ragged_welford_c4095", 64, 128, 1, "column", col_group=8
                ),
            ),
            (
                torch.float32,
                "var",
                2 << 30,
                ct.ColConfig(
                    "b200_ragged_welford_c4095", 64, 1928, 1, "column", col_group=8
                ),
            ),
        ],
        name_fn=lambda dtype, trait, size, cfg: f"{cfg.rule}_{size >> 20}mib",
    )
    def test_b200_ragged_column_values(self, dtype, trait, size, expected):
        """Pin every field of the B200 ragged column rules, not just their names."""
        itemsize = dtype.itemsize
        rows = size // (4095 * itemsize)
        contract = _contract(_TRAITS[trait])
        nfields = contract.pop("nfields")
        args = ((10, 0), dtype, rows, 4095, 1, nfields)
        cfg = ct.select_col_config(*args, itemsize=itemsize, acc_bits=32, **contract)
        self.assertEqual(cfg, expected)

    @parametrize(
        "case,updates,expected_rule",
        [
            ("anchor", {}, "b200_ragged_c257"),
            ("explicit_threads", {"threads_per_block": 64}, "explicit"),
            ("explicit_npar", {"npar": 1024}, "explicit"),
            ("explicit_vec", {"vec": 1}, "explicit"),
            ("row_below", {"rows": (16 << 20) // 514 - 1}, "default"),
            ("row_above", {"rows": (16 << 20) // 514 + 1}, "default"),
            ("mib_below", {"rows": (15 << 20) // 514 + 1}, "default"),
            ("sm103", {"cc": (10, 3)}, "default"),
            ("sm90", {"cc": (9, 0)}, "default"),
            ("batched", {"batches": 2}, "default"),
            ("acc64", {"acc_bits": 64}, "default"),
        ],
        name_fn=lambda case, updates, expected_rule: case,
    )
    def test_b200_ragged_column_boundaries(self, case, updates, expected_rule):
        """Keep explicit, off-anchor, non-B200, batched and 64-bit views on defaults."""
        # fp16 C = 257 rows take 514 bytes; the 16 MiB anchor matches only (16 MiB) // 514
        # rows, not a row more or less, nor sizes elsewhere in the MiB below it.
        args = dict(
            cc=(10, 0),
            dtype=torch.float16,
            rows=(16 << 20) // 514,
            columns=257,
            batches=1,
            nfields=1,
            trait_type=T.SumOps,
            itemsize=2,
            acc_bits=32,
            nouts=1,
            welford_outs=0,
        )
        cfg = ct.select_col_config(**(args | updates))
        self.assertEqual(cfg.rule, expected_rule)

    @parametrize(
        "dtype,N,trait,routed",
        [
            (torch.float16, 255, "sum", True),
            (torch.float16, 253, "sum", False),
            (torch.float16, 16383, "sum", True),
            (torch.float16, 16385, "sum", False),
            (torch.float16, 4096, "sum", False),
            (torch.float32, 255, "argmax", True),
            (torch.bfloat16, 255, "argmax", False),
            (torch.bfloat16, 1025, "argmax", False),
            (torch.float32, 1025, "argmax", True),
            (torch.float32, 1023, "var_mean", False),
            (torch.float32, 1025, "var_mean", True),
            (torch.float32, 1023, "var", False),
            (torch.float32, 1025, "var", True),
        ],
        name_fn=lambda dtype, N, trait, routed: f"{str(dtype)[6:]}_{N}_{trait}",
    )
    def test_b200_ragged_row_gate(self, dtype, N, trait, routed):
        """Route odd rows to the ragged-vector kernel only inside the measured band."""
        trait = {
            "sum": T.SumOps(),
            "argmax": T.ArgMaxOps(),
            "var_mean": T.VarMeanOps(),
            "var": T.WelfordOps(),
        }[trait]
        self.assertEqual(kg._ragged_rows_ok((10, 0), dtype, N, 16, trait), routed)
        self.assertFalse(kg._ragged_rows_ok((9, 0), dtype, N, 16, trait))
        self.assertFalse(kg._ragged_rows_ok((10, 0), dtype, N, 8, trait))
        self.assertFalse(kg._ragged_rows_ok((10, 0), dtype, N, 16, trait, 64))

    def test_b200_ragged_rows_start_late_traits_later(self):
        """Keep absmax, absmin and general-p norms off ragged rows at N = 255, and every
        Welford trait until N = 1025."""
        late = (T.AbsMaxOps(), T.AbsMinOps(), T.NormOps(3), T.NormOps(0.5))
        early = (T.AMaxOps(), T.NormOps(1), T.NormOps(2), T.SumOps())
        self.assertEqual({kg._ragged_row_min_n(t) for t in late}, {1023})
        self.assertEqual({kg._ragged_row_min_n(t) for t in early}, {255})
        self.assertEqual(kg._ragged_row_min_n(T.VarMeanOps(take_sqrt=True)), 1025)
        self.assertEqual(kg._ragged_row_min_n(T.WelfordOps(take_sqrt=True)), 1025)
        for N, routed in ((255, False), (1023, True)):
            ok = kg._ragged_rows_ok((10, 0), torch.float16, N, 16, T.AbsMaxOps())
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

    @parametrize(
        "op,dtype,N",
        [
            ("sum", torch.float32, 255),
            ("var_mean", torch.bfloat16, 1025),
            ("var_mean", torch.float16, 16383),
        ],
        name_fn=lambda op, dtype, N: f"{op}_{str(dtype)[6:]}_{N}",
    )
    def test_b200_ragged_rows_route(self, device, op, dtype, N):
        """Serve unordered odd rows inside the B200 band with the ragged-vector plan."""
        import cutlass

        if torch.cuda.get_device_capability(device) != (10, 0):
            self.skipTest("the ragged-row route is measured on B200 only")
        x = torch.randn(64, N, device=device, dtype=dtype)
        key = f"ragged_route_{op}_{str(dtype)[6:]}_{N}"
        if op == "sum":
            trait, outs = T.SumOps(acc=cutlass.Float32), [torch.float32]
            got = (kg.reduce_dim(trait, key, x, [1], outs[0], order="unordered"),)
        else:
            trait, outs = T.VarMeanOps(correction=1), [torch.float32] * 2
            got = kg.reduce_dim2(trait, key, x, [1], outs, order="unordered")
        routed = {k for k in rt._CACHE if k[:2] == ("rowtile", key)}
        self.assertEqual(len(routed), 1, routed)
        # Row keys carry ragged_vector, so an explicit ragged launch reuses the plan.
        kw = dict(nouts=len(outs), order="linear", ragged_vector=True)
        rt.reduce_row_tile(trait, key, x, outs, **kw)
        self.assertEqual({k for k in rt._CACHE if k[:2] == ("rowtile", key)}, routed)
        with torch.backends.python_native.cutedsl.disabled():
            xf = x.float()
            want = (xf.sum(1),) if op == "sum" else torch.var_mean(xf, dim=1)
        self.assertEqual(got, want, atol=1e-4, rtol=1e-4)

    @parametrize("op", ["argmax", "var"])
    def test_b200_strided_c256_route(self, device, op):
        """Serve b200_strided_c256's smallest anchor, 8 outputs, with the indexed tree."""
        import cutlass

        if torch.cuda.get_device_capability(device) != (10, 0):
            self.skipTest("the strided rules are measured on B200 only")
        x = torch.randn(8, 512, device=device)[:, ::2]
        key = f"{op}_strided_c256"
        if op == "argmax":
            trait, out = T.ArgMaxOps(acc=cutlass.Float32), torch.int64
        else:
            trait, out = T.WelfordOps(correction=1, acc=cutlass.Float32), torch.float32
        got = kg.reduce_dim(trait, key, x, [1], out, order="unordered")
        self.assertEqual(_served(key), {"genitree"})
        with torch.backends.python_native.cutedsl.disabled():
            want = x.argmax(1) if op == "argmax" else x.var(1)
        self.assertEqual(got, want, atol=1e-4, rtol=1e-4)

    def test_b200_strided_index_route(self, device):
        """Run b200_strided_index's fp32 64-thread blocks at its smallest anchor."""
        import cutlass

        if torch.cuda.get_device_capability(device) != (10, 0):
            self.skipTest("the strided rules are measured on B200 only")
        count, num_o = 1024, 2048
        x = torch.randn(num_o, 2 * count, device=device)[:, ::2]
        trait, key = T.ArgMaxOps(acc=cutlass.Float32), "argmax_strided_index"
        got = kg.reduce_dim(trait, key, x, [1], torch.int64, order="unordered")
        pairs = dict(red_pairs=((count, 2),), kept_pairs=((num_o, 2 * count),))
        op = kg.ReduceBlock(trait, count=count, num_o=num_o, **pairs, block=64)
        keys = {k for k in kg._COMPILE_CACHE if k[1:3] == ("reduce", key)}
        prefix = (x.get_device(), "reduce", key, x.dtype, (torch.int64,))
        self.assertEqual(keys, {prefix + op.cache_sig})
        with torch.backends.python_native.cutedsl.disabled():
            want = x.argmax(1)
        self.assertEqual(got, want)

    def test_b200_strided_welford_route(self, device):
        """Fold b200_strided_welford's smallest anchor with the uniform Welford tree."""
        import cutlass

        if torch.cuda.get_device_capability(device) != (10, 0):
            self.skipTest("the strided rules are measured on B200 only")
        count, num_o = 1024, 2048
        x = torch.randn(num_o, 2 * count, device=device)[:, ::2]
        trait, key = T.WelfordOps(correction=1, acc=cutlass.Float32), "var_strided_tree"
        got = kg.reduce_dim(trait, key, x, [1], torch.float32, order="unordered")
        pairs = dict(red_pairs=((count, 2),), kept_pairs=((num_o, 2 * count),))
        shape = dict(count=count, num_o=num_o, **pairs, block=64)
        tree = kg.ReduceBlock(trait, **shape, uniform_tree=True)
        self.assertNotEqual(tree.cache_sig, kg.ReduceBlock(trait, **shape).cache_sig)
        keys = {k for k in kg._COMPILE_CACHE if k[1:3] == ("reduce", key)}
        prefix = (x.get_device(), "reduce", key, x.dtype, (torch.float32,))
        self.assertEqual(keys, {prefix + tree.cache_sig})
        with torch.backends.python_native.cutedsl.disabled():
            want = x.var(1)
        self.assertEqual(got, want, atol=1e-4, rtol=1e-4)

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
