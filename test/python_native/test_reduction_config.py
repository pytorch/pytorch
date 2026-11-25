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


def _served(trait_key):
    """Tags of the row and general plans compiled for trait_key."""
    rows = {k[0] for k in rt._CACHE if k[1] == trait_key}
    return rows | {k[0][1] for k in kg._PLAN if k[0][2] == trait_key}


class TestReductionConfig(TestCase):
    hw_classification = HardwareClassification.CUDA

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

        kwargs = dict(itemsize=2, acc_bits=32, nouts=1, welford_outs=1)
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
        args = ((10, 0), torch.float16, 4095)
        kw = dict(trait=T.SumOps(), acc_bits=32, pitch=4095)
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
        want = {"rowtile"} if stride == 1 else {"reduce", "gensplit1"}
        self.assertTrue(tags & want, tags)
        self.assertFalse(any("itree" in t for t in tags), tags)
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @parametrize("stride", [1, 2])
    def test_selectors_can_pick_inner_tree_for_unordered(self, device, stride):
        """Serve a selector's inner-tree pick with the env gate off."""
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
        ):
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
        wide = x.double()
        with torch.backends.python_native.cutedsl.disabled():
            want = {
                "reduce_dim2": torch.var_mean(wide, 1, correction=1),
                "reduce_all": (wide.sum(),),
                "reduce_all2": torch.var_mean(wide, correction=1),
            }[entry]
        for got, ref, exp in zip(ordered, unordered, want, strict=True):
            self.assertEqual(got, ref, atol=1e-4, rtol=1e-4)
            self.assertEqual(got, exp.float(), atol=1e-4, rtol=1e-4)

    @parametrize("entry", ["reduce_dim", "reduce_dim2", "reduce_all2"])
    def test_inner_tree_rejects_64bit_extents(self, device, entry):
        """Refuse inner-tree order for reduced extents within a batch of 2^31."""
        import cutlass

        from torch._native.ops.reductions import traits

        one = traits.SumOps(acc=cutlass.Float32)
        two = traits.VarMeanOps(correction=1, acc=cutlass.Float32)
        f32 = torch.float32
        pair = [f32, f32]
        # Stride-0 views reach the extent check without allocating 2^31 elements.
        # A count within a batch of 2^31 is wide too: its Int32 columns would wrap.
        wide_re = rt._INNER_TREE_WIDE
        for n in (2**31, 2**31 - 512):
            wide = torch.zeros(1, 1, device=device).expand(2, n)
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, wide_re):
                if entry == "reduce_dim":
                    kg.reduce_dim(one, "sum", wide, [1], f32, order="inner_tree")
                elif entry == "reduce_dim2":
                    kg.reduce_dim2(two, "var_mean", wide, [1], pair, order="inner_tree")
                else:
                    kg.reduce_all2(two, "var_mean", wide[0], pair, order="inner_tree")

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

    def test_strided_reduce_all_takes_the_general_kernel(self, device):
        """Serve strided reduce-all launches with the general kernel."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.randn(64, 128, device=device)[:, ::2]
        key = "strided_reduce_all"
        trait = traits.SumOps(acc=cutlass.Float32)
        with mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "0"}):
            got = kg.reduce_all(trait, key, x, torch.float32)
        self.assertTrue(_served(key) & {"reduce", "gensplit1"}, _served(key))
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
        """Raise under the strict env gate when reduce-all's row is within one batch of
        64-bit extents, where the tree's Int32 columns would wrap."""
        import cutlass

        from torch._native.ops.reductions import traits

        x = torch.empty(2**31 - 512, device=device, dtype=torch.float16)  # never read
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
        # Restore the warned reasons afterwards; cleanups run last in, first out.
        warned = rt._warned_inner_tree_reasons
        self.addCleanup(warned.update, set(warned))
        self.addCleanup(warned.clear)
        warned.discard(rt._INNER_TREE_WIDE)
        with (
            mock.patch.dict(os.environ, {rt._INNER_TREE_ENV: "1"}),
            self.assertWarnsRegex(UserWarning, rt._INNER_TREE_WIDE),
        ):
            got = kg.reduce_all(
                traits.SumOps(acc=cutlass.Float32), "sum", x, torch.float32
            )
        self.assertEqual(got, torch.zeros((), device=device))

    def test_explicit_order_raises_with_its_reason(self, device):
        """Raise for a geometry the inner-tree kernels cannot serve."""
        import cutlass

        from torch._native.ops.reductions import traits

        trait = traits.SumOps(acc=cutlass.Float32)
        # 2^25 outputs of 2^20 elements need 2^32 split partials, past Int32.
        deep = torch.zeros(1, 1, device=device).expand(2**25, 2**20)
        with self.assertRaisesRegex(ValueError, rt._INNER_TREE_GEOMETRY):
            kg.reduce_dim(trait, "sum", deep, [1], torch.float32, order="inner_tree")


instantiate_parametrized_tests(TestReductionConfig)
instantiate_device_type_tests(TestReductionConfigDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
