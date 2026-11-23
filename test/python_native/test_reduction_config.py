# Owner(s): ["module: dsl-native-ops"]

import os
import sys
import unittest
import warnings
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
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


_TRAITS = {
    "sum": T.SumOps(),
    "mean": T.MeanOps(),
    "amax": T.AMaxOps(),
    "nansum": T.NanSumOps(),
    "argmax": T.ArgMaxOps(),
    "argmin": T.ArgMinOps(),
    "var": T.WelfordOps(),
    "var_mean": T.VarMeanOps(),
}


def _contract(trait):
    """Selector keywords for trait's structure."""
    contract = dict(nfields=trait.nfields, nouts=getattr(trait, "nouts", 1))
    welford = T.welford_outputs(trait)
    return contract | dict(trait_type=type(trait), welford_outs=welford)


# Contracts by _TRAITS key, for parametrized selector rows.
_CONTRACTS = {key: _contract(trait) for key, trait in _TRAITS.items()}


def _contract_name(contract):
    """A stable test name fragment for a selector contract row."""
    return f"{contract['trait_type'].__name__}_{contract['nouts']}out"


def _updates_name(updates):
    """A stable test name fragment for a guard row's updated selector arguments."""

    def value(v):
        if isinstance(v, tuple):
            return "x".join(value(e) for e in v)
        if isinstance(v, torch.dtype):
            return str(v)[6:]
        return v.__name__ if isinstance(v, type) else str(v)

    return "_".join(f"{k}_{value(v)}" for k, v in updates.items())


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
            acc_bits=32,
            alignment=16,
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

    @parametrize("value", ["STRICT", "strict ", "true"])
    def test_unknown_gate_values_raise(self, value):
        """Reject gate values other than unset, 0, 1 and strict."""
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: value}):
            with self.assertRaisesRegex(ValueError, rt._INNER_TREE_ENV):
                rt.reduction_order()
            with self.assertRaisesRegex(ValueError, rt._INNER_TREE_ENV):
                rt.inner_tree_fallback("unused")

    @parametrize("cc", [(9, 0), (10, 3), (11, 0)])
    def test_architecture_isolation(self, cc):
        """Return the generic defaults until an architecture adds its own rules."""
        self.assertEqual(self.general(cc=cc), kg._GeneralConfig())
        self.assertEqual(
            rt.select_row_order(
                cc,
                torch.float32,
                1024,
                65536,
                trait_type=T.WelfordOps,
                nfields=3,
                nouts=1,
                welford_outs=1,
                acc_bits=32,
                alignment=16,
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
        "dtype,width,mib,contract,expected",
        [
            (torch.float32, 256, 64, _CONTRACTS["mean"], "inner_tree"),
            (torch.float32, 256, 16, _CONTRACTS["mean"], "linear"),
            (torch.float32, 256, 64, _CONTRACTS["argmax"], "linear"),
            (torch.float32, 256, 256, _CONTRACTS["argmax"], "inner_tree"),
            (torch.bfloat16, 256, 16, _CONTRACTS["sum"], "inner_tree"),
            (torch.bfloat16, 256, 16, _CONTRACTS["argmax"], "linear"),
            (torch.bfloat16, 256, 64, _CONTRACTS["sum"], "linear"),
            # Unmeasured one-field traits keep the linear order.
            (torch.float32, 256, 64, _CONTRACTS["nansum"], "linear"),
            (torch.float32, 1024, 64, _CONTRACTS["var"], "inner_tree"),
            (torch.float32, 1024, 64, _CONTRACTS["var"] | {"nouts": 2}, "linear"),
            (torch.float32, 1024, 256, _CONTRACTS["argmax"], "linear"),
            (torch.bfloat16, 1024, 64, _CONTRACTS["argmax"], "inner_tree"),
            # Rules match trait types: only argmax was measured.
            (torch.bfloat16, 1024, 64, _CONTRACTS["argmin"], "linear"),
            (torch.float32, 4096, 16, _CONTRACTS["var_mean"], "inner_tree"),
            (torch.bfloat16, 4096, 16, _CONTRACTS["argmax"], "inner_tree"),
            (torch.float32, 4096, 16, _CONTRACTS["mean"], "inner_tree"),
            (torch.float32, 4096, 64, _CONTRACTS["sum"], "linear"),
            (torch.float32, 4096, 256, _CONTRACTS["sum"], "inner_tree"),
            (torch.bfloat16, 4096, 2048, _CONTRACTS["mean"], "linear"),
        ],
        name_fn=lambda dtype, width, mib, contract, expected: (
            f"{_contract_name(contract)}_{str(dtype)[6:]}_{width}_{mib}mib"
        ),
    )
    def test_sm107_row_profiles(self, dtype, width, mib, contract, expected):
        """Pin each arm of the SM107 row policy."""
        rows = (mib << 20) // (width * dtype.itemsize)
        self.assertEqual(
            rt.select_row_order(
                (10, 7),
                dtype,
                width,
                rows,
                **contract,
                acc_bits=32,
                alignment=16,
            ),
            expected,
        )

    @parametrize(
        "dtype,mib,contract,rule",
        [
            (torch.float32, 256, _CONTRACTS["sum"], "rubin_strided_sum"),
            (torch.float32, 256, _CONTRACTS["nansum"], "general_default"),
            (torch.bfloat16, 2048, _CONTRACTS["sum"], "rubin_strided_sum"),
            (torch.bfloat16, 16, _CONTRACTS["var"], "rubin_strided_welford"),
            (torch.float32, 256, _CONTRACTS["var"], "rubin_strided_welford"),
            # Welford output counts must match the trait's.
            (torch.float32, 256, _CONTRACTS["var_mean"], "rubin_strided_welford"),
            (torch.float32, 256, _CONTRACTS["var"] | {"nouts": 2}, "general_default"),
            (torch.float32, 64, _CONTRACTS["mean"], "rubin_strided_inner"),
            (torch.float32, 64, _CONTRACTS["sum"], "rubin_strided_inner"),
            (torch.float32, 64, _CONTRACTS["amax"], "rubin_strided_inner"),
            (torch.float32, 16, _CONTRACTS["sum"], "general_default"),
            (torch.bfloat16, 64, _CONTRACTS["argmax"], "rubin_strided_argmax"),
            (torch.bfloat16, 64, _CONTRACTS["argmin"], "general_default"),
            (torch.float32, 2048, _CONTRACTS["argmax"], "rubin_strided_argmax"),
            (torch.float32, 256, _CONTRACTS["argmax"], "general_default"),
            (torch.float32, 64, _CONTRACTS["argmax"], "general_default"),
        ],
        name_fn=lambda dtype, mib, contract, rule: (
            f"{_contract_name(contract)}_{str(dtype)[6:]}_{mib}mib"
        ),
    )
    def test_sm107_general_profiles(self, dtype, mib, contract, rule):
        """Pin each distinct SM107 strided policy."""
        rows = (mib << 20) // (1024 * dtype.itemsize)
        cfg = kg.select_general_config(
            (10, 7),
            dtype,
            1024,
            rows,
            ((1024, 2),),
            ((rows, 2048),),
            **contract,
            acc_bits=32,
            alignment=16,
        )
        self.assertEqual(cfg.rule, rule)
        self.assertEqual(cfg.block, 64 if rule == "rubin_strided_sum" else kg._K0_BLOCK)
        linear = rule in ("rubin_strided_sum", "general_default")
        self.assertEqual(cfg.kernel_order, "linear" if linear else "inner_tree")

    @parametrize(
        "base,updates",
        [
            (
                {},
                {
                    "count": 2048,
                    "num_o": 32768,
                    "red_pairs": ((2048, 2),),
                    "kept_pairs": ((32768, 4096),),
                },
            ),
            ({}, {"num_o": 65535, "kept_pairs": ((65535, 2048),)}),
            ({}, {"red_pairs": ((1024, 4),)}),
            ({}, {"kept_pairs": ((65536, 4096),)}),
            ({}, {"acc_bits": 64}),
            ({}, {"alignment": 8}),
            (
                {
                    "dtype": torch.bfloat16,
                    "num_o": 131072,
                    "kept_pairs": ((131072, 2048),),
                },
                {"dtype": torch.float16},
            ),
        ],
        name_fn=lambda base, updates: _updates_name(updates),
    )
    def test_sm107_general_guards(self, base, updates):
        """Keep a 256 MiB strided anchor that breaks one guard on the generic policy."""
        selected = dict(base, cc=(10, 7))
        self.assertEqual(self.general(**selected).rule, "rubin_strided_sum")
        self.assertEqual(self.general(**dict(selected, **updates)), kg._GeneralConfig())

    @parametrize(
        "updates",
        [
            {"alignment": 8},
            {"acc_bits": 64},
            {"dtype": torch.float16, "M": 131072},
            {"N": 2048, "M": 32768},
            {"M": 65535},
            # fp32 256 MiB at N = 4096 takes the inner tree but for the trait guard.
            {
                "trait_type": T.ProdOps,
                "nfields": 1,
                "welford_outs": 0,
                "N": 4096,
                "M": 16384,
            },
        ],
        name_fn=_updates_name,
    )
    def test_sm107_row_guards(self, updates):
        """Keep a 256 MiB SM107 row anchor that breaks one guard on the linear order."""
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_type=T.WelfordOps,
            N=1024,
            M=65536,
            nfields=3,
            nouts=1,
            welford_outs=1,
            acc_bits=32,
            alignment=16,
        )
        self.assertEqual(rt.select_row_order(**args), "inner_tree")
        self.assertEqual(rt.select_row_order(**(args | updates)), "linear")

    @parametrize(
        "dtype,mib,trait,unroll",
        [
            (torch.float32, 256, "sum", 128),
            (torch.bfloat16, 2048, "argmax", 128),
            (torch.float32, 256, "var", 64),
            (torch.bfloat16, 2048, "var_mean", 64),
        ],
        name_fn=lambda dtype, mib, trait, unroll: f"{trait}_{mib}mib",
    )
    def test_sm107_full_profiles(self, dtype, mib, trait, unroll):
        """Pin each distinct SM107 full-reduction policy branch."""
        contract = _CONTRACTS[trait]
        trait_type, welford_outs = contract["trait_type"], contract["welford_outs"]
        nfields, nouts = contract["nfields"], contract["nouts"]
        arch = rt.select_full_itree_arch(
            (10, 7),
            dtype,
            (mib << 20) // dtype.itemsize,
            1,
            trait_type=trait_type,
            field_bits=tuple(dt.width for dt in _TRAITS[trait].fdtypes),
            out_dtypes=kg._sm107_outputs(dtype, trait_type, nfields, nouts),
            welford_outs=welford_outs,
            alignment=16,
            contiguous=True,
        )
        welford = welford_outs > 0
        want = rt._ItreeArch(
            split_stage_rows=4,
            combine_rpb=32,
            combine_group_bytes=16,
            stage_rows=False,
            combine_async=32,
            combine_max=2048,
            combine_unroll=unroll,
            lowp_thread_elems=32,
            uniform_combine="weights" if welford else "off",
        )
        self.assertEqual(arch, want)

    @parametrize(
        "updates",
        [
            {"cc": (10, 0)},
            {"N": (256 << 20) // 4 - 1},
            {"trait_type": T.MeanOps},
            {"trait_type": T.NanSumOps},
            {"dtype": torch.float16, "N": (256 << 20) // 2},
            # Only fp32 sums were measured, not bf16 sums into fp32.
            {"dtype": torch.bfloat16, "N": (256 << 20) // 2},
            {"M": 2},
            {"contiguous": False},
            {"alignment": 8},
            {"out_dtypes": (torch.bfloat16,)},
        ],
        name_fn=_updates_name,
    )
    def test_sm107_full_guards(self, updates):
        """Keep calls outside the SM107 policy contract on the generic path."""
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_type=T.SumOps,
            N=(256 << 20) // 4,
            M=1,
            field_bits=(32,),
            out_dtypes=(torch.float32,),
            welford_outs=0,
            alignment=16,
            contiguous=True,
        )
        self.assertIsNotNone(rt.select_full_itree_arch(**args))
        args.update(updates)
        self.assertIsNone(rt.select_full_itree_arch(**args))

    @parametrize(
        "updates",
        [
            {"field_bits": (64,) * 3},
            {"dtype": torch.float16, "N": 128 << 20, "out_dtypes": (torch.float16,)},
            {"welford_outs": 0},
            {"welford_outs": 0, "out_dtypes": ()},
            {"out_dtypes": (torch.bfloat16,)},
            {"out_dtypes": (torch.float32,) * 2},
        ],
        name_fn=_updates_name,
    )
    def test_sm107_full_welford_guards(self, updates):
        """Keep Welford calls outside the SM107 Welford contract on the generic path."""
        args = dict(
            cc=(10, 7),
            dtype=torch.float32,
            trait_type=T.WelfordOps,
            N=(256 << 20) // 4,
            M=1,
            field_bits=(32,) * 3,
            out_dtypes=(torch.float32,),
            welford_outs=1,
            alignment=16,
            contiguous=True,
        )
        self.assertIsNotNone(rt.select_full_itree_arch(**args))
        args.update(updates)
        self.assertIsNone(rt.select_full_itree_arch(**args))

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

    def test_b200_ragged_welford_columns_need_combine_equal(self):
        """Decline the grouped Welford column rule for a trait without combine_equal."""

        class OwnCombine(T.WelfordOps):
            def combine(self, a, b):
                return super().combine(a, b)

        kwargs = dict(itemsize=2, acc_bits=32, nouts=1, welford_outs=1, alignment=16)
        kwargs |= dict(out_dtypes=(torch.float16,))
        # At 16 MiB, 4095 columns take b200_ragged_c4095_ordered first.
        for columns, anchor in ((257, 16 << 20), (4095, 64 << 20)):
            args = ((10, 0), torch.float16, anchor // (columns * 2), columns, 1, 3)
            for kind, rule in (
                (T.WelfordOps, f"b200_ragged_welford_c{columns}"),
                (OwnCombine, "default"),
            ):
                cfg = ct.select_col_config(*args, trait_type=kind, **kwargs)
                self.assertEqual(cfg.rule, rule, msg=f"{kind.__name__} at C={columns}")

    def test_b200_ragged_rows_need_an_aligned_base(self):
        """Decline ragged rows without a 16-byte aligned base: reduce_row_tile refuses
        them."""
        args = ((10, 0), torch.float16, 255)
        kw = dict(trait=T.SumOps(), acc_bits=32, pitch=255)
        self.assertTrue(kg._ragged_rows_ok(*args, 16, **kw))
        self.assertFalse(kg._ragged_rows_ok(*args, 8, **kw))


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
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "0"}),
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
        "op,dtype,N,pitch",
        [
            ("sum", torch.float32, 255, 255),
            ("var_mean", torch.bfloat16, 1025, 1025),
            ("var_mean", torch.float16, 16383, 16383),
            ("sum", torch.float16, 1026, 1026),
            ("sum", torch.bfloat16, 1024, 1025),
            ("sum", torch.float32, 4097, 4104),
        ],
        name_fn=lambda op, dtype, N, pitch: f"{op}_{str(dtype)[6:]}_{N}_{pitch}",
    )
    def test_b200_ragged_rows_route(self, device, op, dtype, N, pitch):
        """Serve unordered rows whose width or pitch is not whole vectors inside the B200
        band with the ragged-vector plan."""
        import cutlass

        if torch.cuda.get_device_capability(device) not in kg._B200_CCS:
            self.skipTest("the ragged-row route is measured on B200 only")
        x = torch.randn(64, pitch, device=device, dtype=dtype)[:, :N]
        key = f"ragged_route_{op}_{str(dtype)[6:]}_{N}_{pitch}"
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
        """Serve b200_strided_c256 at 8 outputs, within its 2-output-and-up range, with
        the indexed tree."""
        import cutlass

        if torch.cuda.get_device_capability(device) not in kg._B200_CCS:
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

        if torch.cuda.get_device_capability(device) not in kg._B200_CCS:
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

    @parametrize("count", [1024, 4095, 4096])
    def test_b200_strided_welford_route(self, device, count):
        """Fold b200_strided_welford's anchors with the uniform Welford tree."""
        import cutlass

        if torch.cuda.get_device_capability(device) not in kg._B200_CCS:
            self.skipTest("the strided rules are measured on B200 only")
        num_o = 2048
        x = torch.randn(num_o, 2 * count, device=device)[:, ::2]
        trait = T.WelfordOps(correction=1, acc=cutlass.Float32)
        key = f"var_strided_tree_{count}"
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
        refused = "order='inner_tree' cannot be honoured here"
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "strict"}):
            (got,) = rt.reduce_row_tile(trait, "row_env", x, f32)
            with self.assertRaisesRegex(ValueError, refused):
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
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "0"}):
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
        with self.assertRaisesRegex(ValueError, rt._INNER_TREE_WIDE):
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
            with self.assertRaisesRegex(ValueError, rt._INNER_TREE_WIDE):
                kg.reduce_all(trait, "sum", x, torch.float32, order="inner_tree")
            with (
                mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
                mock.patch.object(rt, "_warned_inner_tree_reasons", set()),
                self.assertWarnsRegex(UserWarning, rt._INNER_TREE_WIDE),
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
            warnings.filterwarnings("error", ".*could not be honoured")
            self.assertEqual(calls[entry](order="inner_tree"), expected)
            for gate in ("1", "strict"):
                with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: gate}):
                    self.assertEqual(calls[entry](), expected)

    def test_strided_reduce_all_uses_the_general_selector(self, device):
        """Size strided reduce-all launches with the general kernel's selector."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(64, 128, device=device)[:, ::2]
        key = "strided_reduce_all"
        trait = traits.SumOps(acc=cutlass.Float32)
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "0"}):
            got = kg.reduce_all(trait, key, x, torch.float32)
        self.assertTrue(_served(key) & {"reduce", "gensplit1"}, _served(key))
        # threads_per_block follows axis, vec and threads_per_row in the cache_sig.
        blocks = {k[0][7] for k in kg._PLAN if k[0][1:3] == ("gensplit1", key)}
        self.assertEqual(blocks, {kg._K0_ALL_BLOCK})
        self.assertEqual(got, x.double().sum().float(), atol=1e-4, rtol=1e-4)

    def test_env_gate_falls_back_with_one_warning(self, device):
        """Serve 64-bit extents (by a patched limit) unordered under the env gate, warning
        only once."""
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

    @largeTensorTest("5GB")
    def test_strict_env_gate_raises_in_reduce_all(self, device):
        """Raise under the strict env gate when reduce-all has 64-bit extents."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.empty(2**31, device=device, dtype=torch.float16)  # never read
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "strict"}),
            self.assertRaisesRegex(ValueError, f"{rt._INNER_TREE_ENV}=strict"),
        ):
            kg.reduce_all(traits.SumOps(acc=cutlass.Float32), "sum", x, torch.float32)

    def test_env_gate_falls_back_in_strided_reduce_all(self, device):
        """Warn, not raise, when the env gate cannot order a strided reduce-all."""
        import cutlass

        from torch._native.ops.reductions import traits

        # A stride-0 view reaches the extent check without allocating 2^31 elements.
        x = torch.zeros(1, device=device).expand(2**31)
        rt._warned_inner_tree_reasons.discard(rt._INNER_TREE_WIDE)
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            self.assertWarnsRegex(UserWarning, rt._INNER_TREE_WIDE),
        ):
            got = kg.reduce_all(
                traits.SumOps(acc=cutlass.Float32), "sum", x, torch.float32
            )
        self.assertEqual(got, torch.zeros((), device=device))

    def test_explicit_order_raises_with_its_reason(self, device):
        """Raise for explicit order through the two-output entry points, and for a
        geometry the inner-tree kernels cannot serve."""
        import cutlass

        from torch._native.ops.reductions import traits

        two = traits.VarMeanOps(correction=1, acc=cutlass.Float32)
        f32 = torch.float32
        wide = torch.zeros(1, 1, device=device).expand(2, 2**31)
        with self.assertRaisesRegex(ValueError, rt._INNER_TREE_WIDE):
            kg.reduce_dim2(two, "var_mean", wide, [1], [f32, f32], order="inner_tree")
        with self.assertRaisesRegex(ValueError, rt._INNER_TREE_WIDE):
            kg.reduce_all2(two, "var_mean", wide[0], [f32, f32], order="inner_tree")
        # 2^25 outputs of 2^20 elements need 2^32 split partials, past Int32.
        deep = torch.zeros(1, 1, device=device).expand(2**25, 2**20)
        with self.assertRaisesRegex(ValueError, rt._INNER_TREE_GEOMETRY):
            kg.reduce_dim(
                traits.SumOps(acc=cutlass.Float32),
                "sum",
                deep,
                [1],
                f32,
                order="inner_tree",
            )


instantiate_parametrized_tests(TestReductionConfig)
instantiate_device_type_tests(TestReductionConfigDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
