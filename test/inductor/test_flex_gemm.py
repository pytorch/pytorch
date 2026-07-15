# Owner(s): ["module: inductor"]

import contextlib
import math
import subprocess
import sys
import unittest
from types import SimpleNamespace
from typing import get_args
from unittest import mock

import torch
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import _SUPPORTED_FLEX_GEMM_OP_NAMES
from torch._inductor.ops_handler import ReductionType
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM100OrLater, TEST_CUDA
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


try:
    import cutlass.cute as cute
except ImportError:
    cute = None

if cute is not None:

    @cute.jit
    def relu_epilogue(acc):
        return cute.where(acc > cute.full_like(acc, 0), acc, cute.full_like(acc, 0))

    @cute.jit
    def affine_aux_epilogue(acc, col_bias, row_scale, tile_bias):
        value = (acc + col_bias) * row_scale + tile_bias
        return cute.where(
            value > cute.full_like(value, 0), value, cute.full_like(value, 0)
        )

    @cute.jit
    def row_scale_epilogue(acc, row_scale):
        return acc * row_scale

    @cute.jit
    def tuple_aux_epilogue(acc):
        main = (acc + cute.full_like(acc, 1.0)) * cute.full_like(acc, 0.5)
        aux = acc * acc + cute.full_like(acc, 2.0)
        return main, aux

    @cute.jit
    def captured_tuple_aux_epilogue(acc, col_bias, row_scale, tile_bias):
        biased = (acc + col_bias) * row_scale + tile_bias
        main = cute.where(
            biased > cute.full_like(biased, 0), biased, cute.full_like(biased, 0)
        )
        aux = acc * row_scale + tile_bias
        return main, aux


class TestFlexGemmRuntimeImport(TestCase):
    def test_import_does_not_load_external_quack(self):
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                "import sys; import torch._inductor.kernel.flex_gemm.runtime; assert 'quack' not in sys.modules",
            ]
        )


@instantiate_parametrized_tests
class TestFlexGemmRuntimeHelpers(TestCase):
    @parametrize(
        "reduction_type",
        get_args(ReductionType),
        name_fn=lambda reduction_type: reduction_type,
    )
    def test_tensorssa_reduction_table_covers_inductor_vocabulary(self, reduction_type):
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            tensorssa_reduction,
            TENSORSSA_REDUCTIONS,
        )

        expected = {
            "sum": ("cute.ReductionOp.ADD", "0.0", "lhs + rhs"),
            "prod": ("cute.ReductionOp.MUL", "1.0", "lhs * rhs"),
            "max": (
                "cute.ReductionOp.MAX",
                'float("-inf")',
                "cutlass.max(lhs, rhs)",
            ),
            "min": (
                "cute.ReductionOp.MIN",
                'float("inf")',
                "cutlass.min(lhs, rhs)",
            ),
        }
        self.assertEqual(set(TENSORSSA_REDUCTIONS), set(expected))
        if reduction_type not in expected:
            with self.assertRaisesRegex(
                NotImplementedError,
                f"{reduction_type} does not map to a CuTe TensorSSA reduction",
            ):
                tensorssa_reduction(reduction_type)
            return

        actual = tensorssa_reduction(reduction_type)
        self.assertIs(actual, TENSORSSA_REDUCTIONS[reduction_type])
        self.assertEqual(
            (actual.cute_op, actual.init_val, actual.combine_expr),
            expected[reduction_type],
        )

    def test_dense_config_selection_is_explicit_and_sm110_reuses_sm100(self):
        from torch._inductor.heuristics.template import (
            flex_gemm as flex_gemm_heuristics,
        )
        from torch._vendor.quack.gemm_config import GemmConfig

        def config(tile_m, tile_n, cluster_m, cluster_n, dynamic, **kwargs):
            values = {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "cluster_m": cluster_m,
                "cluster_n": cluster_n,
                "cluster_k": 1,
                "is_dynamic_persistent": dynamic,
                "swap_ab": False,
                "use_tma_gather": False,
                "device_capacity": 10,
                "tile_k": None,
                "num_warps": None,
                "pingpong": False,
                "max_swizzle_size": 8,
            }
            values.update(kwargs)
            return GemmConfig(**values)

        default = config(128, 256, 2, 1, True)
        skinny = config(128, 192, 2, 1, True)
        large_rect = config(256, 256, 2, 1, True)
        large = config(256, 256, 2, 2, True)
        swap_variant = config(128, 128, 1, 1, False, swap_ab=True)
        gather_rejected = config(128, 128, 1, 1, False, use_tma_gather=True)

        fake_graph = SimpleNamespace(
            sizevars=SimpleNamespace(guard_or_false=lambda expr: bool(expr))
        )
        from torch._inductor.virtualized import V

        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(11, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[
                    gather_rejected,
                    swap_variant,
                    large_rect,
                    default,
                    skinny,
                    large,
                ],
            ),
            V.set_graph_handler(fake_graph),
        ):
            self.assertEqual(
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                ),
                [default, skinny, large_rect, large, swap_variant],
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 256, 4096
                ),
                flex_gemm_heuristics.gemm_config_key(skinny),
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 768, 4096
                ),
                flex_gemm_heuristics.gemm_config_key(large),
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 1024, 4096
                ),
                flex_gemm_heuristics.gemm_config_key(large_rect),
            )
            self.assertEqual(
                flex_gemm_heuristics.default_gemm_config_key(
                    torch.device("cuda"), 1024, 1024
                ),
                flex_gemm_heuristics.gemm_config_key(skinny),
            )
            self.assertEqual(
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                ),
                [default, skinny, large_rect, large, swap_variant],
            )
            self.assertEqual(
                GemmConfig(**dict(flex_gemm_heuristics.gemm_config_key(large))), large
            )

        sm120_pingpong = config(
            128,
            128,
            1,
            1,
            True,
            device_capacity=12,
            pingpong=True,
        )
        self.assertNotEqual(
            flex_gemm_heuristics.gemm_config_key(default),
            flex_gemm_heuristics.gemm_config_key(sm120_pingpong),
        )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[default, sm120_pingpong],
            ),
        ):
            self.assertEqual(
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                ),
                [sm120_pingpong],
            )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[default],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "no QuACK configs"):
                flex_gemm_heuristics.candidate_gemm_configs_for_device(
                    torch.device("cuda")
                )

    def test_precompile_metadata_counts_symbolic_skip(self):
        import sympy

        from torch._dynamo.utils import counters
        from torch._inductor.kernel.flex_gemm.template import FlexGemmEpilogueCaller

        counters.clear()
        caller = FlexGemmEpilogueCaller.__new__(FlexGemmEpilogueCaller)
        caller.bmreq = SimpleNamespace(
            input_tensor_meta=[
                SimpleNamespace(
                    sizes=(sympy.Symbol("s0"), 64),
                    strides=(64, 1),
                    dtype=torch.float32,
                    device=torch.device("cuda", 0),
                )
            ],
            output_tensor_meta=SimpleNamespace(
                sizes=(128, 128),
                strides=(128, 1),
                dtype=torch.float32,
                device=torch.device("cuda", 0),
            ),
        )

        self.assertIsNone(caller.precompile_metadata())
        self.assertEqual(
            counters["inductor"]["flex_gemm_precompile_skipped_dynamic"], 1
        )


class FlexGemmTestCase(TestCase):
    def makeTensor(self, *shape, dtype=torch.bfloat16):
        return torch.testing.make_tensor(
            *shape, device="cuda", dtype=dtype, low=-0.1, high=0.1
        )

    def swapAndNonSwapConfigKeys(self, device):
        """Return one swap_ab and one non-swap candidate config key for ``device``."""
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )

        keys = [
            gemm_config_key(config)
            for config in candidate_gemm_configs_for_device(device)
        ]
        swap_keys = [key for key in keys if dict(key)["swap_ab"]]
        non_swap_keys = [key for key in keys if not dict(key)["swap_ab"]]
        self.assertTrue(swap_keys and non_swap_keys)
        return swap_keys[0], non_swap_keys[0]

    def assertMatchesLowPrecisionEager(
        self,
        actual,
        low_precision_expected,
        high_precision_expected,
        reduction_size,
    ):
        actual_error = (actual.double() - high_precision_expected).abs().mean()
        eager_error = (
            (low_precision_expected.double() - high_precision_expected).abs().mean()
        )
        # Model the extra slack as fp32 accumulator rounding across K plus final output rounding.
        fp32_accumulation_eps = (
            math.sqrt(reduction_size) * torch.finfo(torch.float32).eps
        )
        result_rounding_eps = torch.finfo(actual.dtype).eps
        output_scale = high_precision_expected.abs().mean().item()
        rounding_atol = (fp32_accumulation_eps + result_rounding_eps) * output_scale
        self.assertLessEqual(
            actual_error.item(),
            eager_error.item() + rounding_atol,
            msg=(
                lambda msg: f"{msg}\nactual error {actual_error.item()} exceeded low precision eager "
                f"error {eager_error.item()} with fp32_accumulation_eps="
                f"{fp32_accumulation_eps}, result_rounding_eps="
                f"{result_rounding_eps}, output_scale={output_scale}, "
                f"and atol={rounding_atol}"
            ),
        )

    def assertTupleAuxMatchesReference(self, actual, aux, a, b, epilogue_fn):
        """Validate tuple-aux epilogues against low/high precision references."""
        expected, expected_aux = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            (high_precision_acc + 1.0) * 0.5,
            a.shape[-1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            expected_aux,
            high_precision_acc.square() + 2.0,
            a.shape[-1],
        )

    def assertCapturedTupleAuxMatchesReference(
        self, actual, aux, a, b, col_bias, row_scale, tile_bias
    ):
        """Validate composed captured-load and tuple-aux epilogues."""
        acc = a @ b
        acc_float = acc.float()
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            ((acc_float + col_bias) * row_scale + tile_bias).relu(),
            (
                (high_precision_acc + col_bias.double()) * row_scale.double()
                + tile_bias.double()
            ).relu(),
            a.shape[-1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            acc_float * row_scale + tile_bias,
            high_precision_acc * row_scale.double() + tile_bias.double(),
            a.shape[-1],
        )

    def localReduceGeometryPattern(self, group, axis):
        """Return the generated structural local-reduce geometry pattern."""
        return f"FlexGemmLocalReduceGeometry(group={group}, axis={axis})"

    def assertLocalReduceAuxCode(self, code, group, axis=1, callbacks=False):
        """Check generated code passes a structural compressed-aux plan."""
        file_check = (
            FileCheck()
            .check("local_reduce=FlexGemmRuntimeLocalReducePlan")
            .check(self.localReduceGeometryPattern(group, axis))
            .check("out=")
        )
        if callbacks:
            file_check = file_check.check("callbacks=FlexGemmLocalReduceCallbacks")
        file_check.check_not("local_reduce_out=").check_not(
            "local_reduce_group="
        ).check_not("local_reduce_axis=").check_not("local_reduce_op").run(code)

    def runtimeLocalReducePlan(
        self,
        out=None,
        group=8,
        axis=0,
        feeds_main=False,
    ):
        """Build the structural local-reduce runtime plan used by generated code."""
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
        )

        callbacks = None
        if feeds_main or axis == 0 or group > 16:
            callbacks = FlexGemmLocalReduceCallbacks(
                combine_fn=lambda lhs, rhs: lhs,
                finalize_fn=lambda value: value,
            )
        return FlexGemmRuntimeLocalReducePlan(
            FlexGemmLocalReduceGeometry(group, axis),
            out=out,
            callbacks=callbacks,
            feeds_main=feeds_main,
        )

    def assertMatchesEpilogue(
        self, actual, expected, high_precision_expected, reduction_size
    ):
        """Compare one or multiple epilogue outputs against eager references."""
        if isinstance(expected, tuple):
            self.assertEqual(len(actual), len(expected))
            self.assertEqual(len(expected), len(high_precision_expected))
            for actual_item, expected_item, high_precision_item in zip(
                actual, expected, high_precision_expected
            ):
                self.assertMatchesLowPrecisionEager(
                    actual_item,
                    expected_item,
                    high_precision_item,
                    reduction_size,
                )
            return
        self.assertMatchesLowPrecisionEager(
            actual, expected, high_precision_expected, reduction_size
        )

    def assertLocalReduceAuxMatches(self, actual, aux, a, b, epilogue_fn):
        """Validate compressed local-reduce aux output against high precision GEMM."""
        expected, _ = epilogue_fn(a @ b)
        high_precision_expected, high_precision_aux = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        torch.testing.assert_close(
            aux,
            high_precision_aux.float(),
            atol=1e-3,
            rtol=1e-3,
        )


@skipIfNoCuteDSL
@unittest.skipIf(not TEST_CUDA, "CUDA required")
@instantiate_parametrized_tests
class TestFlexGemmRuntime(FlexGemmTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.relu_epilogue = staticmethod(relu_epilogue)
        cls.affine_aux_epilogue = staticmethod(affine_aux_epilogue)
        cls.row_scale_epilogue = staticmethod(row_scale_epilogue)
        cls.captured_tuple_aux_epilogue = staticmethod(captured_tuple_aux_epilogue)
        cls.tuple_aux_epilogue = staticmethod(tuple_aux_epilogue)

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_with_c_alpha_beta_matches_reference(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(0)
        m, n, k = 128, 128, 64
        a = self.makeTensor(k, m).t()
        b = self.makeTensor(k, n)
        c = self.makeTensor(n, m).t()

        out_buffer = torch.empty_strided((m, n), (1, m), device="cuda", dtype=a.dtype)
        out = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_relu_c",
            C=c,
            alpha=0.5,
            beta=1.25,
            out=out_buffer,
        )
        self.assertIs(out, out_buffer)
        low_precision_expected = (
            (0.5 * (a @ b).float() + 1.25 * c.float()).relu().to(out.dtype)
        )
        high_precision_expected = (
            0.5 * (a.double() @ b.double()) + 1.25 * c.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_batched_epilogue_beta_zero_ignores_nan_c(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(7)
        batch, m, n, k = 2, 128, 128, 64
        a = self.makeTensor(batch, m, k)
        b = self.makeTensor(batch, k, n)
        c = torch.full((m, n), float("nan"), device="cuda", dtype=a.dtype)

        out = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_batched_beta_zero",
            C=c,
            alpha=1.5,
            beta=0,
        )

        self.assertFalse(torch.isnan(out).any())
        self.assertMatchesLowPrecisionEager(
            out,
            torch.baddbmm(c, a, b, beta=0, alpha=1.5).relu(),
            torch.baddbmm(c.double(), a.double(), b.double(), beta=0, alpha=1.5).relu(),
            k,
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_explicit_arg_kind_disambiguates_row_arg(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(2)
        m, n, k = 1, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.row_scale_epilogue,
            "test_flex_gemm_row_scale",
            out_dtype=torch.float32,
            epilogue_args=(row_scale,),
            epilogue_arg_kinds=("row",),
        )
        self.assertMatchesLowPrecisionEager(
            out,
            (a @ b).float() * row_scale,
            (a.double() @ b.double()) * row_scale.double(),
            k,
        )

    def test_explicit_arg_kind_disambiguates_col_arg_shape(self):
        from torch._inductor.kernel.flex_gemm.runtime import resolve_epilogue_arg_kinds

        a = torch.empty(128, 64)
        b = torch.empty(64, 1)
        col_bias = torch.empty(128, 1)

        self.assertEqual(
            resolve_epilogue_arg_kinds(a, b, (col_bias,), ("col",)),
            ("col",),
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_infers_captured_aux_arg_kinds(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(4)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.affine_aux_epilogue,
            "test_flex_gemm_infer_aux",
            out_dtype=torch.float32,
            epilogue_args=(col_bias, row_scale, tile_bias),
        )
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_validation_rejects_unsupported_epilogue_arg_combinations(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        a = self.makeTensor(128, 64)
        b = self.makeTensor(64, 128)
        c = self.makeTensor(128, 128)
        row_scale = self.makeTensor(1, 128, dtype=torch.float32)

        with self.assertRaisesRegex(NotImplementedError, "cannot be combined with C"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_c_args",
                C=c,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
            )
        with self.assertRaisesRegex(NotImplementedError, "non-default alpha/beta"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_alpha_args",
                alpha=0.5,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
            )
        with self.assertRaisesRegex(NotImplementedError, "tile/row/col"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_kind",
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("diag",),
            )
        with self.assertRaisesRegex(RuntimeError, "row epilogue arg shape"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_shape",
                epilogue_args=(row_scale.t(),),
                epilogue_arg_kinds=("row",),
            )
        bad_layout = self.makeTensor(256, 128)[::2, ::2]
        with self.assertRaisesRegex(NotImplementedError, "row- or column-major"):
            gemm_epilogue(
                bad_layout,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_layout",
            )
        bad_out_layout = self.makeTensor(256, 128)[::2, ::2]
        with self.assertRaisesRegex(NotImplementedError, "row- or column-major"):
            gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                "test_flex_gemm_reject_bad_out_layout",
                out=bad_out_layout,
            )

    def test_runtime_validation_rejects_local_reduce_rank_and_output_shape(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_local_reduce_out_shape,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            validate_runtime_local_reduce,
        )

        compressed_plan = self.runtimeLocalReducePlan(
            out=torch.empty(128, 8), group=8, axis=1
        )
        with self.assertRaisesRegex(NotImplementedError, "2-D aten.mm"):
            validate_runtime_local_reduce(
                compressed_plan,
                torch.empty(1, 128, 64),
                (1, 128, 64),
                None,
                None,
                1.0,
                1.0,
            )
        with self.assertRaisesRegex(RuntimeError, "local_reduce_out shape"):
            validate_local_reduce_out_shape((128, 7), (128, 8))
        column_major_plan = self.runtimeLocalReducePlan(
            out=torch.empty_strided((128, 8), (1, 128), device="cuda"),
            group=8,
            axis=1,
        )
        with self.assertRaisesRegex(NotImplementedError, "row-major"):
            validate_runtime_local_reduce(
                column_major_plan,
                torch.empty(128, 64, device="cuda"),
                (128, 64),
                None,
                None,
                1.0,
                1.0,
            )

    def test_local_reduce_callbacks_reject_missing_functions(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
        )

        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmLocalReduceCallbacks(None, lambda value: value)
        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmLocalReduceCallbacks(lambda lhs, rhs: lhs, None)

    def test_local_reduce_callbacks_use_generated_cache_keys(self):
        from torch._inductor.kernel.flex_gemm.runtime import local_reduce_callback_key

        def combine(lhs, rhs):
            return lhs

        combine.__cache_key__ = lambda: "combine-cache-key"
        self.assertEqual(
            local_reduce_callback_key(combine, "fallback"), "combine-cache-key"
        )
        self.assertEqual(
            local_reduce_callback_key(lambda value: value, "fallback"), "fallback"
        )

    def test_local_reduce_plan_rejects_invalid_group_axis(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )

        with self.assertRaisesRegex(
            RuntimeError, "local_reduce_group must be positive"
        ):
            FlexGemmLocalReduceGeometry(0, 0)
        with self.assertRaisesRegex(RuntimeError, "local_reduce_axis must be 0 or 1"):
            FlexGemmLocalReduceGeometry(8, 2)

    def test_runtime_local_reduce_plan_rejects_missing_callbacks(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
        )

        with self.assertRaisesRegex(RuntimeError, "local_reduce_out"):
            FlexGemmRuntimeLocalReducePlan(FlexGemmLocalReduceGeometry(8, 0))
        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmRuntimeLocalReducePlan(
                FlexGemmLocalReduceGeometry(8, 0), out=torch.empty(1)
            )
        with self.assertRaisesRegex(RuntimeError, "generated local-reduce callbacks"):
            FlexGemmRuntimeLocalReducePlan(
                FlexGemmLocalReduceGeometry(8, 0), feeds_main=True
            )

    def test_local_reduce_plan_uses_explicit_consumers(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
        )
        from torch._inductor.kernel.flex_gemm.template import (
            FlexGemmEpilogueLocalReduceConfig,
        )

        callbacks = FlexGemmLocalReduceCallbacks(
            lambda lhs, rhs: lhs, lambda value: value
        )
        geometry = FlexGemmLocalReduceGeometry(8, 0)
        self.assertTrue(
            FlexGemmRuntimeLocalReducePlan(
                geometry, callbacks=callbacks, feeds_main=True
            ).feeds_main
        )
        self.assertTrue(
            FlexGemmRuntimeLocalReducePlan(
                geometry,
                out=torch.empty(1),
                callbacks=callbacks,
                feeds_main=True,
            ).feeds_main
        )
        self.assertFalse(
            FlexGemmRuntimeLocalReducePlan(
                geometry, out=torch.empty(1), callbacks=callbacks
            ).feeds_main
        )
        self.assertTrue(
            FlexGemmEpilogueLocalReduceConfig(geometry, feeds_main=True).feeds_main
        )
        self.assertTrue(
            FlexGemmEpilogueLocalReduceConfig(
                geometry, out_index=0, feeds_main=True
            ).feeds_main
        )
        self.assertFalse(
            FlexGemmEpilogueLocalReduceConfig(geometry, out_index=0).feeds_main
        )

    def test_output_plan_rejects_invalid_state(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.epilogue import (
            FlexGemmEpilogueGraph,
            FlexGemmLocalReduceAnalysis,
            FlexGemmLocalReduceMatch,
            FlexGemmLocalReduceStore,
            FlexGemmOutputLocalReducePlan,
            FlexGemmOutputPlan,
            tuple_output_plan,
        )

        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        aux = graph.placeholder("aux")
        geometry = FlexGemmLocalReduceGeometry(8, 0)
        match = FlexGemmLocalReduceMatch(aux, geometry)
        analysis = FlexGemmLocalReduceAnalysis(FlexGemmEpilogueGraph({}))
        with self.assertRaisesRegex(RuntimeError, "output nodes"):
            FlexGemmOutputPlan(object())
        with self.assertRaisesRegex(RuntimeError, "output nodes"):
            FlexGemmOutputPlan(node, (object(),))
        with self.assertRaisesRegex(RuntimeError, "tensor nodes"):
            FlexGemmLocalReduceMatch(object(), geometry)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputLocalReducePlan(object())
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputLocalReducePlan(match)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmLocalReduceStore(object(), 0)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmLocalReduceStore(aux, -1)
        with self.assertRaisesRegex(NotImplementedError, "tensor outputs"):
            tuple_output_plan(object(), (), analysis)
        with self.assertRaisesRegex(NotImplementedError, "tensor outputs"):
            tuple_output_plan(node, (object(),), analysis)
        FlexGemmOutputPlan(
            node,
            (aux,),
            FlexGemmOutputLocalReducePlan(
                match, store=FlexGemmLocalReduceStore(aux, 0)
            ),
        )
        FlexGemmOutputPlan(
            node,
            (aux,),
            FlexGemmOutputLocalReducePlan(match, feeds_main=True),
        )

    def test_ordered_outputs_restore_local_reduce_position(self):
        from torch._inductor.kernel.flex_gemm.lowering import flex_gemm_ordered_outputs

        expected_outputs = (
            ("main", "local", "aux0", "aux1"),
            ("main", "aux0", "local", "aux1"),
            ("main", "aux0", "aux1", "local"),
        )
        for index, expected in enumerate(expected_outputs):
            self.assertEqual(
                flex_gemm_ordered_outputs("main", ("aux0", "aux1"), ("local",), index),
                expected,
            )

    def test_local_reduce_aux_result_requires_grouped_source(self):
        from torch._inductor.kernel.flex_gemm.epilogue import FlexGemmEpilogueEmitter

        graph = torch.fx.Graph()
        aux = graph.placeholder("aux")
        with self.assertRaisesRegex(NotImplementedError, "grouped TensorSSA"):
            FlexGemmEpilogueEmitter.aux_result(aux, {})
        self.assertEqual(FlexGemmEpilogueEmitter.aux_result(aux, {aux: "tmp0"}), "tmp0")
        self.assertIsNone(FlexGemmEpilogueEmitter.aux_result(None, {}))

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_rejects_local_reduce_aux(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
            gemm_epilogue,
        )

        m, n, k = 128, 128, 64
        group = 16
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        local_reduce_out = torch.empty(
            m, n // group, device="cuda", dtype=torch.float32
        )
        swap_key, _ = self.swapAndNonSwapConfigKeys(a.device)

        with self.assertRaisesRegex(NotImplementedError, "do not support swap_ab"):
            gemm_epilogue(
                a,
                b,
                self.relu_epilogue,
                "test_flex_gemm_swap_ab_local_reduce_rejects",
                out_dtype=torch.float32,
                local_reduce=FlexGemmRuntimeLocalReducePlan(
                    FlexGemmLocalReduceGeometry(group, 1),
                    out=local_reduce_out,
                ),
                config_key=swap_key,
            )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_explicit_config_key_matches_reference(self):
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        a = self.makeTensor(128, 64)
        b = self.makeTensor(64, 128)
        row_scale = self.makeTensor(1, 128, dtype=torch.float32)

        config_keys = tuple(
            gemm_config_key(config)
            for config in candidate_gemm_configs_for_device(a.device)
        )
        for index, config_key in enumerate(config_keys[:2]):
            out = gemm_epilogue(
                a,
                b,
                self.row_scale_epilogue,
                f"test_flex_gemm_config_key_{index}",
                out_dtype=torch.float32,
                epilogue_args=(row_scale,),
                epilogue_arg_kinds=("row",),
                config_key=config_key,
            )
            self.assertMatchesLowPrecisionEager(
                out,
                (a @ b).float() * row_scale,
                (a.double() @ b.double()) * row_scale.double(),
                a.shape[1],
            )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("shape", ((128, 512, 256), (512, 128, 256), (256, 256, 256)))
    def test_swap_ab_matches_non_swap_and_eager(self, shape):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k = shape
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        swapped = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_swap_ab_mm",
            out_dtype=torch.float32,
            config_key=swap_key,
        )
        non_swapped = gemm_epilogue(
            a,
            b,
            self.relu_epilogue,
            "test_flex_gemm_non_swap_ab_mm",
            out_dtype=torch.float32,
            config_key=non_swap_key,
        )
        # swap_ab only reorients tile scheduling, so the result is bit-identical.
        self.assertEqual(swapped, non_swapped)
        self.assertMatchesLowPrecisionEager(
            swapped, (a @ b).float().relu(), (a.double() @ b.double()).relu(), k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_with_c_alpha_beta_matches_non_swap(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k = 256, 384, 192
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        c = self.makeTensor(m, n)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            return gemm_epilogue(
                a,
                b,
                self.relu_epilogue,
                name,
                C=c,
                alpha=1.5,
                beta=0.5,
                out_dtype=torch.float32,
                config_key=config_key,
            )

        swapped = run("test_flex_gemm_swap_ab_addmm", swap_key)
        non_swapped = run("test_flex_gemm_non_swap_ab_addmm", non_swap_key)
        # The transposed C view must reproduce the non-swapped addmm result.
        self.assertEqual(swapped, non_swapped)
        self.assertMatchesLowPrecisionEager(
            swapped,
            (0.5 * c.float() + 1.5 * (a @ b).float()).relu(),
            (0.5 * c.double() + 1.5 * (a.double() @ b.double())).relu(),
            k,
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_captured_aux_matches_non_swap(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k = 128, 384, 256
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            return gemm_epilogue(
                a,
                b,
                self.affine_aux_epilogue,
                name,
                out_dtype=torch.float32,
                epilogue_args=(col_bias, row_scale, tile_bias),
                epilogue_arg_kinds=("col", "row", "tile"),
                config_key=config_key,
            )

        swapped = run("test_flex_gemm_swap_ab_aux", swap_key)
        non_swapped = run("test_flex_gemm_non_swap_ab_aux", non_swap_key)
        # Swapped row/col broadcast roles must reproduce the non-swapped result.
        self.assertEqual(swapped, non_swapped)
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        self.assertMatchesLowPrecisionEager(
            swapped, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_captured_args_tuple_aux_matches_non_swap(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(10)
        m, n, k = 128, 384, 256
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            aux = torch.empty(m, n, device="cuda", dtype=torch.float32)
            gemm_epilogue(
                a,
                b,
                self.captured_tuple_aux_epilogue,
                name,
                out=out,
                aux_outs=(aux,),
                epilogue_args=(col_bias, row_scale, tile_bias),
                epilogue_arg_kinds=("col", "row", "tile"),
                config_key=config_key,
            )
            return out, aux

        swapped, swapped_aux = run(
            "test_flex_gemm_swap_ab_captured_tuple_aux", swap_key
        )
        non_swapped, non_swapped_aux = run(
            "test_flex_gemm_non_swap_ab_captured_tuple_aux", non_swap_key
        )

        self.assertEqual(swapped, non_swapped)
        self.assertEqual(swapped_aux, non_swapped_aux)
        self.assertCapturedTupleAuxMatchesReference(
            swapped, swapped_aux, a, b, col_bias, row_scale, tile_bias
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_reads_captured_aux_tensors(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(1)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)

        out = gemm_epilogue(
            a,
            b,
            self.affine_aux_epilogue,
            "test_flex_gemm_affine_aux",
            out_dtype=torch.float32,
            epilogue_args=(col_bias, row_scale, tile_bias),
            epilogue_arg_kinds=("col", "row", "tile"),
        )
        low_precision_expected = (
            ((a @ b).float() + col_bias) * row_scale + tile_bias
        ).relu()
        high_precision_expected = (
            (a.double() @ b.double() + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            out, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_writes_tuple_aux_out(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(8)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        out = torch.empty(m, n, device="cuda", dtype=torch.float32)
        aux = torch.empty(m, n, device="cuda", dtype=torch.float32)

        def epilogue_fn(acc):
            main = (acc.float() + 1.0) * 0.5
            aux = acc.float().square() + 2.0
            return main, aux

        actual = gemm_epilogue(
            a,
            b,
            self.tuple_aux_epilogue,
            "test_flex_gemm_tuple_aux",
            out_dtype=torch.float32,
            out=out,
            aux_outs=(aux,),
        )

        self.assertIs(actual, out)
        self.assertTupleAuxMatchesReference(out, aux, a, b, epilogue_fn)


@instantiate_parametrized_tests
class TestFlexGemmEpilogueHOP(FlexGemmTestCase):
    def assertFlexGemmGeneratedCode(self, code, *checks):
        file_check = (
            FileCheck()
            .check("from torch._inductor.kernel.flex_gemm.runtime import (")
            .check("FlexGemmRuntimeLocalReducePlan")
            .check("gemm_epilogue as flex_gemm_epilogue")
            .check("flex_gemm_epilogue(")
        )
        for check in checks:
            file_check = file_check.check(check)
        file_check = (
            file_check.check("stream=stream").check("config_key=").check_not("tuned=")
        )
        file_check = file_check.check_not("epilogue_source=")
        file_check.check_not("from quack").check_not("import quack").run(code)

    def test_supported_op_names_match_dense_scope(self):
        self.assertEqual(_SUPPORTED_FLEX_GEMM_OP_NAMES, "mm/addmm/bmm/baddbmm")

    @parametrize(
        "case",
        (
            ("mm", torch.mm, lambda a, b: (a, b), lambda a, b: a @ b),
            ("bmm", torch.bmm, lambda a, b: (a, b), lambda a, b: torch.bmm(a, b)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_default_backend_eager_matches_reference(self, case):
        _, op, args_fn, ref_fn = case
        a = torch.randn(2, 8, 16)
        b = torch.randn(2, 16, 12)
        if op is torch.mm:
            a = a[0]
            b = b[0]

        def epilogue_fn(acc):
            return acc.relu()

        actual = flex_gemm(op, args_fn(a, b), epilogue_fn)

        torch.testing.assert_close(actual, epilogue_fn(ref_fn(a, b)))

    def test_default_backend_eager_tuple_aux_matches_reference(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        def epilogue_fn(acc):
            return acc.relu(), acc + 1

        actual, aux = flex_gemm(torch.mm, (a, b), epilogue_fn)
        expected, expected_aux = epilogue_fn(a @ b)

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aux, expected_aux)

    def test_fake_tensor_mode_tuple_aux_returns_fake_tensors(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode() as mode:
            a = mode.from_tensor(torch.randn(8, 16))
            b = mode.from_tensor(torch.randn(16, 12))

            def epilogue_fn(acc):
                return acc.relu(), acc + 1

            actual, aux = flex_gemm(torch.mm, (a, b), epilogue_fn)

        self.assertEqual(actual.shape, torch.Size([8, 12]))
        self.assertEqual(aux.shape, torch.Size([8, 12]))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(aux.dtype, torch.float32)
        self.assertIs(actual.fake_mode, mode)
        self.assertIs(aux.fake_mode, mode)

    def test_autograd_is_not_implemented(self):
        a = torch.randn(8, 16, requires_grad=True)
        b = torch.randn(16, 12, requires_grad=True)

        def epilogue_fn(acc):
            return acc.relu()

        actual = flex_gemm(torch.mm, (a, b), epilogue_fn)

        with self.assertRaisesRegex(RuntimeError, "flex_gemm"):
            actual.sum().backward()

    def test_generated_captured_arg_rejects_unsupported_shape(self):
        def fn(a, b, scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc * scale,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 5)
        scale = torch.randn(1, 1)

        with self.assertRaisesRegex(
            Exception,
            "captured tensor epilogue args currently must match",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, scale)

    def test_generated_captured_arg_rejects_addmm_scope(self):
        def fn(bias, a, b, scale):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                lambda acc: acc * scale,
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 5)
        a = torch.randn(4, 8)
        b = torch.randn(8, 5)
        scale = torch.randn(4, 5)

        with self.assertRaisesRegex(
            Exception,
            "captured tensor reads currently support only aten.mm",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b, scale)

    def test_generated_tuple_aux_rejects_unsupported_scope(self):
        def addmm_fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                lambda acc: (acc.relu(), acc + 1),
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 5)
        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(addmm_fn, backend="inductor", fullgraph=True)(bias, a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_compiled_matches_reference(self):
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return (acc + 1).relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_alpha_clamp_compiled_matches_reference(self):
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return torch.add(acc, 2.0, alpha=0.25).clamp(min=0.0)

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_dynamic_shapes_compiled_matches_reference(self):
        def epilogue_fn(acc):
            return (acc + 1).relu()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        for m in (128, 256):
            a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
            actual = compiled(a, b)
            self.assertMatchesLowPrecisionEager(
                actual,
                epilogue_fn(a @ b),
                epilogue_fn(a.double() @ b.double()),
                a.shape[1],
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_dynamic_shapes_tuned_matches_reference(self):
        def epilogue_fn(acc):
            return (acc + 1).relu()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        device = torch.device("cuda")
        swap_configs = [
            config
            for config in flex_gemm_heuristics.candidate_gemm_configs_for_device(device)
            if config.swap_ab
        ]
        self.assertTrue(swap_configs)
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=swap_configs[:1],
        ):
            compiled = torch.compile(
                fn, backend="inductor", fullgraph=True, dynamic=True
            )
            for m, n in ((128, 128), (256, 192)):
                a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
                actual = compiled(a, b)
                self.assertMatchesLowPrecisionEager(
                    actual,
                    epilogue_fn(a @ b),
                    epilogue_fn(a.double() @ b.double()),
                    a.shape[1],
                )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    @parametrize(
        "tuned",
        (False, True),
        name_fn=lambda tuned: "tuned" if tuned else "untuned",
    )
    def test_mm_dynamic_shapes_reads_captured_tensor_epilogue_arg(self, case, tuned):
        torch._dynamo.reset()
        _, shape_fn = case

        def epilogue_fn(acc, scale):
            return (acc.float() * scale).relu()

        def fn(a, b, scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        config_context = contextlib.nullcontext()
        if tuned:
            from torch._inductor.template_heuristics import (
                flex_gemm as flex_gemm_heuristics,
            )

            configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(
                torch.device("cuda")
            )[:2]
            config_context = mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
                return_value=configs,
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        with config_context:
            for m, k, n in ((128, 64, 128), (256, 64, 192)):
                a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
                scale = torch.randn(*shape_fn(m, n), device="cuda", dtype=torch.float32)
                actual = compiled(a, b, scale)
                self.assertMatchesLowPrecisionEager(
                    actual,
                    epilogue_fn(a @ b, scale),
                    epilogue_fn(a.double() @ b.double(), scale.double()),
                    a.shape[1],
                )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_reads_bool_mask_captured_tensor_epilogue_arg(self, case):
        _, shape_fn = case

        def epilogue_fn(acc, mask):
            acc_float = acc.float()
            return torch.where(mask, acc_float, -acc_float)

        def fn(a, b, mask):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, mask),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        mask = torch.randint(0, 2, shape_fn(m, n), device="cuda", dtype=torch.bool)

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b, mask)

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, mask),
            epilogue_fn(a.double() @ b.double(), mask),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_promotes_low_precision_captured_tensor_epilogue_arg(self, case):
        _, shape_fn = case

        def epilogue_fn(acc, scale):
            return scale * acc.float()

        def fn(a, b, scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(*shape_fn(m, n), device="cuda", dtype=torch.bfloat16)

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b, scale)

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, scale),
            epilogue_fn(a.double() @ b.double(), scale.double()),
            a.shape[1],
        )

    @parametrize(
        "case",
        (
            ("reduce_n_keepdim", lambda acc: acc.sum(dim=1, keepdim=True)),
            ("reduce_m_keepdim", lambda acc: acc.sum(dim=0, keepdim=True)),
            ("reduce_n", lambda acc: acc.sum(dim=1)),
            ("reduce_m", lambda acc: acc.sum(dim=0)),
            ("reduce_all", lambda acc: acc.sum()),
            ("mean_n_keepdim", lambda acc: acc.mean(dim=1, keepdim=True)),
            ("logsumexp_n_keepdim", lambda acc: acc.logsumexp(dim=1, keepdim=True)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_tuple_aux_rejects_partial_reduction_without_contract(self, case):
        _, aux_fn = case

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: (acc.relu(), aux_fn(acc)),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "partial-output contract"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_generated_tuple_aux_rejects_dbias_reduction_without_contract(self):
        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: (acc.relu(), acc.float().sum(dim=0)),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "partial-output contract"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_generated_local_reduce_aux_rejects_addmm_scope(self):
        def fn(bias, a, b):
            def epilogue(acc):
                x = acc.float().view(4, -1, 4)
                return acc.relu(), x.sum(-1)

            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 8)
        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

    def test_generated_local_reduce_rejects_empty_dim_list(self):
        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(4, -1, 4)
                return acc.relu(), torch.ops.aten.sum.dim_IntList(x, [], False)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "innermost grouped dimension"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_generated_local_reduce_rejects_bmm_scope(self):
        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(2, 4, -1, 4)
                return acc.relu(), x.sum(-1)

            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2, 4, 8)
        b = torch.randn(2, 8, 8)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @parametrize(
        "case",
        (
            (
                "non_innermost",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 4).sum(1)),
                8,
                "innermost grouped dimension",
            ),
            (
                "fragment_unsupported",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 7).sum(-1)),
                14,
                "fragment width 32",
            ),
            (
                "fragment_not_dividing",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 24).sum(-1)),
                48,
                "fragment width 32",
            ),
            (
                "large_group",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 48).sum(-1)),
                96,
                "fragment width 32",
            ),
            (
                "degenerate_group",
                lambda acc: (acc.relu(), acc.float().view(4, -1, 1).sum(-1)),
                8,
                "group size greater than 1",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_invalid_group(self, case):
        _, epilogue, n, error = case

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, n)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @parametrize(
        "case",
        (
            ("kwarg", lambda x: x.sum(-1, dtype=torch.float64)),
            (
                "aten_kwarg",
                lambda x: torch.ops.aten.sum.dim_IntList(
                    x, [-1], False, dtype=torch.float64
                ),
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_explicit_reduction_dtype(self, case):
        _, reduce_fn = case

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(4, -1, 4)
                return acc.relu(), reduce_fn(x)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "explicit reduction dtype"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @parametrize(
        "case",
        (
            ("any", lambda x: (x > 0).any(-1)),
            ("all", lambda x: (x > 0).all(-1)),
            ("argmax", lambda x: x.argmax(-1)),
            ("argmin", lambda x: x.argmin(-1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_non_value_reductions(self, case):
        _, reduce_fn = case

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(4, -1, 4)
                return acc.relu(), reduce_fn(x)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(4, 8)
        b = torch.randn(8, 8)

        with self.assertRaisesRegex(Exception, "does not map to a CuTe TensorSSA"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @parametrize(
        "case",
        (
            (
                "m_then_n",
                lambda acc: (
                    acc.relu(),
                    acc.float().view(-1, 4, 8).sum(1).view(1, -1, 4).sum(-1),
                ),
                (4, 8),
                "local-reduce output contract",
            ),
            (
                "n_then_m",
                lambda acc: (
                    acc.relu(),
                    acc.float().view(4, -1, 4).sum(-1).view(-1, 4, 2).sum(1),
                ),
                (4, 8),
                "local-reduce output contract",
            ),
            (
                "direct_block",
                lambda acc: (
                    acc.relu(),
                    acc.float().view(-1, 4, 2, 4).sum((1, 3)),
                ),
                (4, 8),
                "local-reduce output contract",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_block_reductions(self, case):
        _, epilogue_fn, shape, error = case

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        m, n = shape
        a = torch.randn(m, 8)
        b = torch.randn(8, n)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_generated_local_reduce_rejects_mixed_row_column_grouping(self):
        m = 8
        n = 64
        group = 4
        bad_inner_n = 32

        def fn(a, b):
            def epilogue(acc):
                x = acc.float().view(-1, group, bad_inner_n)
                scale = x.sum(1, keepdim=True)
                return (x * scale.reciprocal()).view(m, n)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 8)
        b = torch.randn(8, n)

        with self.assertRaisesRegex(Exception, "grouped reshape must split exactly"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_dynamic_shapes_compiled_matches_reference(self):
        def epilogue_fn(acc):
            main = (acc.float() + 1.0) * 0.5
            aux = acc.float().square() + 2.0
            return main, aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        for m, k, n in ((128, 64, 128), (256, 64, 192)):
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
            actual, aux = compiled(a, b)
            self.assertTupleAuxMatchesReference(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_temporary_grouped_reshape_without_reduction(self):
        m, n, k, group = 128, 96, 64, 3

        def epilogue(acc):
            grouped = acc.float().view(m, n // group, group)
            return (grouped + 1.0).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertMatchesLowPrecisionEager(
            actual,
            (a @ b).float() + 1.0,
            (a.double() @ b.double()) + 1.0,
            k,
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("sum_method", lambda x: x.sum(-1), "cute.ReductionOp.ADD"),
            ("sum_function", lambda x: torch.sum(x, dim=-1), "cute.ReductionOp.ADD"),
            ("mean_method", lambda x: x.mean(-1), "cute.ReductionOp.ADD"),
            ("mean_function", lambda x: torch.mean(x, dim=-1), "cute.ReductionOp.ADD"),
            ("prod_method", lambda x: (x * 0.05).prod(-1), "cute.ReductionOp.MUL"),
            (
                "prod_function",
                lambda x: torch.prod(x * 0.05, dim=-1),
                "cute.ReductionOp.MUL",
            ),
            ("amax_method", lambda x: x.amax(-1), "cute.ReductionOp.MAX"),
            ("amax_function", lambda x: torch.amax(x, dim=-1), "cute.ReductionOp.MAX"),
            ("amin_method", lambda x: x.amin(-1), "cute.ReductionOp.MIN"),
            ("amin_function", lambda x: torch.amin(x, dim=-1), "cute.ReductionOp.MIN"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_n_reduce_compiled_matches_reference(self, case):
        _, reduce_fn, cute_op = case
        m = 128
        group = 16

        for n in (128, 96):

            def epilogue_fn(acc):
                x = acc.float().view(m, -1, group)
                return acc.relu(), reduce_fn(x)

            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK"},
                )

            a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
            (actual, aux), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

            self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
            FileCheck().check(cute_op).run(code)
            self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("axis", (0, 1))
    def test_mm_tuple_aux_local_reduce_supports_explicit_group_count(self, axis):
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            match axis:
                case 1:
                    x = acc.float().view(m, n // group, group)
                    return acc.relu(), x.sum(-1)
                case 0:
                    x = acc.float().view(m // group, group, n)
                    return acc.relu(), x.sum(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertLocalReduceAuxCode(code, group, axis=axis, callbacks=axis == 0)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("axis", (0, 1))
    def test_mm_tuple_aux_local_reduce_dynamic_explicit_group_count(self, axis):
        def epilogue_fn(acc):
            m, n = acc.shape
            match axis:
                case 1:
                    x = acc.float().view(m, n // 16, 16)
                    return acc.relu(), x.sum(-1)
                case 0:
                    x = acc.float().view(m // 16, 16, n)
                    return acc.relu(), x.sum(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        cases = (
            ((128, 64, 128), (128, 64, 192))
            if axis == 0
            else ((128, 64, 128), (256, 64, 128))
        )
        for m, k, n in cases:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
            actual, aux = compiled(a, b)

            self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (2, 16, 32))
    @parametrize(
        "case",
        (
            ("sum", lambda x: x.sum(1), "local_reduce_combine_fn"),
            ("mean", lambda x: x.mean(1), " / {group}.0"),
            ("prod", lambda x: (x * 0.05).prod(1), "lhs * rhs"),
            ("amax", lambda x: x.amax(1), "cutlass.max"),
            ("amin", lambda x: x.amin(1), "cutlass.min"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_m_reduce_compiled_matches_reference(self, case, group):
        _, reduce_fn, code_check = case
        m = 128
        n = 128
        code_check = code_check.format(group=group)

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), reduce_fn(x)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        FileCheck().check(code_check).run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_supports_multiple_same_shape_outputs(self):
        m = 128
        n = 128

        def epilogue_fn(acc):
            acc_f = acc.float()
            return acc.relu(), acc_f + 1, acc_f * 2

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux0, aux1), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux0, aux1),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("aux_outs=(").check(",)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_combines_same_shape_and_compressed_local_reduce(self):
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1), (x * 0.5).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, local_reduce_aux, same_shape_aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected_actual, _, expected_aux = epilogue_fn(a @ b)
        (
            high_precision_actual,
            high_precision_local_reduce_aux,
            high_precision_aux,
        ) = epilogue_fn(a.double() @ b.double())

        self.assertMatchesLowPrecisionEager(
            actual, expected_actual, high_precision_actual, a.shape[1]
        )
        torch.testing.assert_close(
            local_reduce_aux,
            high_precision_local_reduce_aux.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        self.assertMatchesLowPrecisionEager(
            same_shape_aux, expected_aux, high_precision_aux, a.shape[1]
        )
        FileCheck().check("aux_outs=").run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_m_reduce_tuned_matches_reference(self):
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, aux = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_m_reduce_supports_tail_m(self):
        m = 96
        n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), _ = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_m_reduce_rejects_var_reduction(self):
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.var(1, correction=0)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "does not map to a CuTe TensorSSA"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (2, 32))
    def test_mm_tuple_aux_local_reduce_supports_group_extremes(self, group):
        m = 128
        n = 128

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), x.sum(-1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        FileCheck().check("cute.ReductionOp.ADD").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_reduce_supports_tuple_shape_reshape(self):
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().reshape((m, -1, group))
            return acc.relu(), x.sum(-1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            (
                "variance_like",
                lambda x: ((x - x.mean(-1, keepdim=True)).square()).mean(-1) * 0.5
                + 1.0,
                " / 4.0",
                False,
            ),
            (
                "sum_keepdim_squeeze",
                lambda x: x.sum(-1, keepdim=True).squeeze(-1),
                "broadcast_to",
                False,
            ),
            (
                "stable_logsumexp",
                lambda x: (
                    (x - x.amax(-1, keepdim=True)).exp().sum(-1, keepdim=True).log()
                    + x.amax(-1, keepdim=True)
                ).view(x.shape[0], -1),
                "cute.math.log",
                True,
            ),
            (
                "logsumexp_method",
                lambda x: x.logsumexp(-1),
                "cute.math.log",
                True,
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_reduce_supports_chained_grouped_expressions(self, case):
        _, aux_fn, generated_check, checks_max = case
        m = 128
        n = 96
        group = 4

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), aux_fn(x)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        file_check = FileCheck()
        if checks_max:
            file_check = file_check.check("cute.ReductionOp.MAX")
        file_check.check("cute.ReductionOp.ADD").check(generated_check)
        file_check.run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_supports_distinct_output_dtypes(self):
        def epilogue_fn(acc):
            return acc.relu(), acc.float().square() + 2.0

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, aux = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        expected, expected_aux = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(aux.dtype, torch.float32)
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_acc.relu(),
            a.shape[1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            expected_aux,
            high_precision_acc.square() + 2.0,
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_supports_bool_mask_output(self):
        def epilogue_fn(acc):
            return acc.relu(), acc > 0

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        self.assertEqual(aux.dtype, torch.bool)
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            (a.double() @ b.double()).relu(),
            a.shape[1],
        )
        torch.testing.assert_close(aux, expected_aux)
        self.assertFlexGemmGeneratedCode(code, "aux_outs=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_imports_generated_dependencies(self):
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            abs_acc = torch.abs(acc)
            return torch.where(abs_acc > 0.1, acc, -acc)

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.mm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_compiled_matches_reference(self):
        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.addmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_calls_flex_gemm_adapter(self):
        def epilogue_fn(acc):
            return (acc + 1).relu()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": False},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_generated_code_reads_captured_tensor_epilogue_arg(self, case):
        kind, shape_fn = case

        def epilogue_fn(acc, scale):
            return (acc.float() * scale).relu()

        def fn(a, b, scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.randn(*shape_fn(m, n), device="cuda", dtype=torch.float32)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, scale),
            epilogue_fn(a.double() @ b.double(), scale.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(
            code, "epilogue_args=", f"epilogue_arg_kinds=('{kind}',)"
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_reads_multiple_captured_tensor_epilogue_args(self):
        def fn(a, b, col_bias, row_scale, tile_bias):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: ((acc.float() + col_bias) * row_scale + tile_bias).relu(),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        col_bias = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        row_scale = torch.randn(1, n, device="cuda", dtype=torch.float32)
        tile_bias = torch.randn(m, n, device="cuda", dtype=torch.float32)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            col_bias,
            row_scale,
            tile_bias,
        )

        low_precision_expected = fn(a, b, col_bias, row_scale, tile_bias)
        high_precision_expected = (
            ((a.double() @ b.double()) + col_bias.double()) * row_scale.double()
            + tile_bias.double()
        ).relu()
        self.assertMatchesLowPrecisionEager(
            actual, low_precision_expected, high_precision_expected, a.shape[1]
        )
        self.assertFlexGemmGeneratedCode(
            code,
            "epilogue_args=",
            "epilogue_arg_kinds=('col', 'row', 'tile')",
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_reads_captured_args_and_writes_tuple_aux(self):
        def fn(a, b, col_bias, row_scale, tile_bias):
            def epilogue_fn(acc):
                biased = (acc.float() + col_bias) * row_scale + tile_bias
                return biased.relu(), acc.float() * row_scale + tile_bias

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        col_bias = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        row_scale = torch.randn(1, n, device="cuda", dtype=torch.float32)
        tile_bias = torch.randn(m, n, device="cuda", dtype=torch.float32)

        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            col_bias,
            row_scale,
            tile_bias,
        )

        self.assertCapturedTupleAuxMatchesReference(
            actual, aux, a, b, col_bias, row_scale, tile_bias
        )
        self.assertFlexGemmGeneratedCode(
            code,
            "epilogue_args=",
            "epilogue_arg_kinds=('col', 'row', 'tile')",
            "aux_outs=",
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_tuned_matches_reference(self):
        def epilogue_fn(acc):
            return (acc + 1).relu()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_generated_code_tuned_matches_reference(self):
        def epilogue_fn(acc):
            main = (acc.float() + 1.0) * 0.5
            aux = acc.float().square() + 2.0
            return main, aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            (actual, aux), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertTupleAuxMatchesReference(actual, aux, a, b, epilogue_fn)
        self.assertFlexGemmGeneratedCode(code, "aux_outs=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_compiled_matches_reference(self):
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.bmm,
            (a, b),
            epilogue_fn,
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_compiled_matches_reference(self):
        bias = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.baddbmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.baddbmm(
                    bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5
                )
            ),
            a.shape[-1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_broadcast_bias_compiled_matches_reference(self):
        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return acc.relu()

        actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
            torch.baddbmm,
            (bias, a, b),
            epilogue_fn,
            gemm_kwargs={"beta": 0.5, "alpha": 1.5},
            kernel_options={"backend": "QUACK"},
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.baddbmm(
                    bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5
                )
            ),
            a.shape[-1],
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_matrix_dim_broadcast_bias_compiled_matches_reference(self):
        batch, m, n, k = 2, 128, 192, 64
        a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)
        bias_cases = (
            ("row_1d", torch.randn(n, device="cuda", dtype=torch.bfloat16)),
            ("row_2d", torch.randn(1, n, device="cuda", dtype=torch.bfloat16)),
            ("col_2d", torch.randn(m, 1, device="cuda", dtype=torch.bfloat16)),
        )

        def epilogue_fn(acc):
            return acc.relu()

        for name, bias in bias_cases:
            with self.subTest(name=name):
                actual = torch.compile(flex_gemm, backend="inductor", fullgraph=True)(
                    torch.baddbmm,
                    (bias, a, b),
                    epilogue_fn,
                    gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                    kernel_options={"backend": "QUACK"},
                )

                self.assertMatchesLowPrecisionEager(
                    actual,
                    epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
                    epilogue_fn(
                        torch.baddbmm(
                            bias.double(),
                            a.double(),
                            b.double(),
                            beta=0.5,
                            alpha=1.5,
                        )
                    ),
                    k,
                )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_generated_code_calls_flex_gemm_adapter(self):
        def epilogue_fn(acc):
            return acc.relu()

        def fn(a, b):
            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_batch_one_generated_code_calls_flex_gemm_adapter(self):
        def epilogue_fn(acc):
            return acc.relu()

        def fn(a, b):
            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(1, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1, 64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_bmm_generated_code_tuned_matches_reference(self):
        def epilogue_fn(acc):
            return acc.relu()

        def fn(a, b):
            return flex_gemm(
                torch.bmm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_baddbmm_generated_code_calls_flex_gemm_adapter(self):
        def epilogue_fn(acc):
            return acc.relu()

        def fn(bias, a, b):
            return flex_gemm(
                torch.baddbmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(2, 128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.baddbmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.baddbmm(
                    bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5
                )
            ),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code, "C=", "expected_ndim=3")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_generated_code_calls_flex_gemm_adapter(self):
        def epilogue_fn(acc):
            return acc.relu()

        def fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code, "C=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_addmm_generated_code_tuned_matches_reference(self):
        def epilogue_fn(acc):
            return acc.relu()

        def fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        bias = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = flex_gemm_heuristics.candidate_gemm_configs_for_device(a.device)[:2]
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), bias, a, b
            )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            a.shape[1],
        )
        self.assertFlexGemmGeneratedCode(code, "C=")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            (
                "unsupported_epilogue",
                lambda acc: acc.sum(dim=1, keepdim=True),
                {"backend": "QUACK"},
                "unsupported FlexGEMM epilogue",
            ),
            (
                "unknown_kernel_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "split_k": 2},
                "unsupported FlexGEMM kernel options",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_code_rejects_unsupported_cases(self, case):
        _, epilogue_fn, kernel_options, error = case

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options=kernel_options,
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_rejects_unsupported_quack_op(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        with self.assertRaisesRegex(RuntimeError, "unsupported GEMM op"):
            flex_gemm(
                torch.ops.aten.matmul.default,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

    def test_rejects_unknown_backend(self):
        a = torch.randn(8, 16)
        b = torch.randn(16, 12)

        with self.assertRaisesRegex(RuntimeError, "unsupported FlexGEMM backend"):
            flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "CUTLASS"},
            )


if __name__ == "__main__":
    run_tests()
