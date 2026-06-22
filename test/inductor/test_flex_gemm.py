# Owner(s): ["module: inductor"]

import importlib
import math
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import _SUPPORTED_FLEX_GEMM_OP_NAMES
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


class TestFlexGemmRuntimeImport(TestCase):
    def test_import_does_not_load_external_quack(self):
        sys.modules.pop("quack", None)
        importlib.import_module("torch._inductor.kernel.flex_gemm.runtime")
        self.assertNotIn("quack", sys.modules)


class TestFlexGemmRuntimeHelpers(TestCase):
    def test_dense_config_selection_is_explicit_and_sm110_reuses_sm100(self):
        from torch._inductor.template_heuristics import (
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
        from torch._inductor.template_heuristics.flex_gemm import (
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
                f"actual error {actual_error.item()} exceeded low precision eager "
                f"error {eager_error.item()} with fp32_accumulation_eps="
                f"{fp32_accumulation_eps}, result_rounding_eps="
                f"{result_rounding_eps}, output_scale={output_scale}, "
                f"and atol={rounding_atol}"
            ),
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

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_epilogue_explicit_config_key_matches_reference(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue
        from torch._inductor.template_heuristics.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )

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


@instantiate_parametrized_tests
class TestFlexGemmEpilogueHOP(FlexGemmTestCase):
    def assertFlexGemmGeneratedCode(self, code, *checks):
        file_check = (
            FileCheck()
            .check(
                "from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue as flex_gemm_epilogue"
            )
            .check("flex_gemm_epilogue(")
        )
        for check in checks:
            file_check = file_check.check(check)
        file_check = file_check.check("config_key=").check_not("tuned=")
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

    def test_autograd_is_not_implemented(self):
        a = torch.randn(8, 16, requires_grad=True)
        b = torch.randn(16, 12, requires_grad=True)

        def epilogue_fn(acc):
            return acc.relu()

        actual = flex_gemm(torch.mm, (a, b), epilogue_fn)

        with self.assertRaisesRegex(RuntimeError, "flex_gemm"):
            actual.sum().backward()

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
            "torch._inductor.template_heuristics.flex_gemm.candidate_gemm_configs_for_device",
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
            "torch._inductor.template_heuristics.flex_gemm.candidate_gemm_configs_for_device",
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
            "torch._inductor.template_heuristics.flex_gemm.candidate_gemm_configs_for_device",
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
            "torch._inductor.template_heuristics.flex_gemm.candidate_gemm_configs_for_device",
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
