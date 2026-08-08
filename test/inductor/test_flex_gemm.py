# Owner(s): ["module: inductor"]

import contextlib
import dataclasses
import importlib
import logging
import math
import struct
import sys
import unittest
from types import SimpleNamespace
from typing import get_args
from unittest import mock

import torch
from torch._dynamo.testing import CompileCounterWithBackend
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import _SUPPORTED_FLEX_GEMM_OP_NAMES, nvfp4_pack
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor.exc import InductorError
from torch._inductor.ops_handler import ReductionType
from torch._inductor.utils import run_and_get_code
from torch._subclasses.fake_tensor import is_fake
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM100OrLater, SM120OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked, pack_uint4
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


def mx_e8m0_scale(
    amax: torch.Tensor,
    max_value: float = 448.0,
    rounding: str | None = None,
    pack: int = 1,
) -> torch.Tensor:
    """Encode E8M0 scales using only public tensor ops and inline assembly."""
    if (
        not isinstance(max_value, float)
        or not math.isfinite(max_value)
        or max_value <= 0
    ):
        raise ValueError(
            "mx_e8m0_scale max_value must be a finite positive float, "
            f"got {max_value!r}"
        )
    if rounding not in (None, "floor", "rceil"):
        raise ValueError(
            f"mx_e8m0_scale rounding must be 'floor', 'rceil', or None, got {rounding!r}"
        )
    rounding = "rceil" if rounding is None else rounding
    if pack not in (1, 2):
        raise ValueError(f"mx_e8m0_scale pack must be 1 or 2, got {pack}")

    max_abs = amax.float()
    use_inline_asm_path = torch.compiler.is_compiling() or is_fake(max_abs)
    if use_inline_asm_path:
        if rounding == "rceil":
            conversion_input = max_abs / max_value
            conversion_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
            floor_infinity_input_bits = None
        else:  # rounding == "floor"
            max_power = math.floor(math.log2(max_value))
            conversion_input = max_abs * 2.0**-max_power
            conversion_instruction = "cvt.rz.ue8m0x2.f32"
            floor_infinity_input_bits = (
                struct.unpack("<I", struct.pack("<f", 2.0 ** (128 - max_power)))[0]
                if max_power > 0
                else None
            )

        if pack == 1:
            asm_str = f"{conversion_instruction} $0, 0.0, $1;"
            if floor_infinity_input_bits is not None:
                asm_str = (
                    "{ .reg .pred is_inf; .reg .f32 finite; "
                    "testp.infinite.f32 is_inf, $1; "
                    f"selp.f32 finite, 0f{floor_infinity_input_bits:08x}, $1, is_inf; "
                    f"{conversion_instruction} $0, 0.0, finite; }}"
                )
            constraints = "=h,r"
        else:  # pack == 2
            unpack_pair_asm = (
                "cvt.u32.u16 widened, packed; "
                "and.b32 $0, widened, 0xff; "
                "shr.u32 $1, widened, 8; }"
            )
            asm_str = (
                "{ .reg .b16 packed; .reg .u32 widened; "
                f"{conversion_instruction} packed, $3, $2; " + unpack_pair_asm
            )
            if floor_infinity_input_bits is not None:
                asm_str = (
                    "{ .reg .pred is_inf0, is_inf1; "
                    ".reg .f32 finite0, finite1; "
                    ".reg .b16 packed; .reg .u32 widened; "
                    "testp.infinite.f32 is_inf0, $2; "
                    "testp.infinite.f32 is_inf1, $3; "
                    f"selp.f32 finite0, 0f{floor_infinity_input_bits:08x}, $2, is_inf0; "
                    f"selp.f32 finite1, 0f{floor_infinity_input_bits:08x}, $3, is_inf1; "
                    f"{conversion_instruction} packed, finite1, finite0; "
                    + unpack_pair_asm
                )
            constraints = "=r,=r,r,r"
        return inline_asm_elementwise(
            conversion_input,
            asm_str=asm_str,
            constraints=constraints,
            dtype=torch.float8_e8m0fnu,
            pack=pack,
        )

    if rounding == "floor":
        max_power = math.floor(math.log2(max_value))
        max_abs_bits = max_abs.view(torch.int32)
        unbiased_exponent = ((max_abs_bits >> 23) & 0xFF) - 127
        unbiased_scale_exponent = torch.clamp(
            unbiased_exponent - max_power, min=-127, max=128
        )
        encoded_scale = (unbiased_scale_exponent + 127).to(torch.uint8)
    else:  # rounding == "rceil"
        conversion_input_bits = (max_abs / max_value).view(torch.int32)
        biased_exponent = (conversion_input_bits >> 23) & 0xFF
        mantissa_bits = conversion_input_bits & 0x7FFFFF
        encoded_scale = biased_exponent + (mantissa_bits != 0).to(torch.int32)
        encoded_scale = torch.where(
            (biased_exponent == 0) & (mantissa_bits <= 0x400000),
            torch.zeros_like(encoded_scale),
            encoded_scale,
        )
        encoded_scale = torch.clamp(encoded_scale, max=254).to(torch.uint8)

    encoded_scale = torch.where(
        torch.isnan(max_abs),
        torch.full_like(encoded_scale, 255),
        encoded_scale,
    )
    return encoded_scale.view(torch.float8_e8m0fnu)


def nvfp4_e4m3_scale(amax: torch.Tensor, max_value: float = 6.0) -> torch.Tensor:
    """Compute the ordinary E4M3 scale used by NVFP4 test epilogues."""
    return torch.clamp(
        amax.float() / max_value,
        min=torch.finfo(torch.float8_e4m3fn).tiny,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)


try:
    import cutlass
    import cutlass.cute as cute
except ImportError:
    cute = None

if cute is not None:

    @cute.jit
    def relu_epilogue(acc):
        return cute.where(acc > cute.full_like(acc, 0), acc, cute.full_like(acc, 0))

    @cute.jit
    def captured_affine_epilogue(acc, col_bias, row_scale, tile_bias):
        value = (acc + col_bias) * row_scale + tile_bias
        return cute.where(
            value > cute.full_like(value, 0), value, cute.full_like(value, 0)
        )

    @cute.jit
    def row_scale_epilogue(acc, row_scale):
        return acc * row_scale

    @cute.jit
    def scalar_scale_epilogue(acc, scale):
        return acc * scale

    @cute.jit
    def captured_affine_scalar_epilogue(acc, col_bias, row_scale, tile_bias, scale):
        value = ((acc + col_bias) * row_scale + tile_bias) * scale
        return cute.where(
            value > cute.full_like(value, 0), value, cute.full_like(value, 0)
        )

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

    @cute.jit
    def fragment_group32_sum_feed_epilogue(acc):
        """Mirror the generated axis-1 group-32 TensorSSA feed epilogue.

        Uses the same symbolic min()/repeat fragment expressions as generated
        code, so a config whose per-thread fragment extent is smaller than the
        group would silently reduce partial groups and fail the reference check.
        """
        tmp0 = acc.to(cutlass.Float32)
        tmp1 = tmp0.reshape(
            (
                (
                    1,
                    cutlass.const_expr(min(32, cute.size(tmp0.shape, mode=[0]))),
                    cutlass.const_expr(
                        cute.size(tmp0.shape, mode=[0])
                        // min(32, cute.size(tmp0.shape, mode=[0]))
                    ),
                ),
                1,
                1,
            )
        )
        tmp2 = tmp1.reduce(
            cute.ReductionOp.ADD,
            init_val=0.0,
            reduction_profile=((None, 1, None), 1, 1),
        )
        tmp3 = tmp2.reshape(
            (
                (
                    1,
                    1,
                    cutlass.const_expr(
                        cute.size(tmp1.shape, mode=[0])
                        // min(32, cute.size(tmp1.shape, mode=[0]))
                    ),
                ),
                1,
                1,
            )
        )
        tmp4 = tmp3.broadcast_to(tmp1.shape)
        return tmp1 * (cute.full_like(tmp4, 1.0) / tmp4)

    @cute.jit
    def physical_group_sum_feed_epilogue(acc, local_reduce0):
        return acc / local_reduce0

    @cute.jit
    def physical_group_sum_combine(lhs, rhs):
        return lhs + rhs

    @cute.jit
    def physical_group_sum_finalize(value):
        return value


class TestFlexGemmRuntimeImport(TestCase):
    def test_import_does_not_load_external_quack(self):
        for name in list(sys.modules):
            if name == "quack" or name.startswith("quack."):
                del sys.modules[name]
        sys.modules.pop("torch._inductor.kernel.flex_gemm.runtime", None)
        importlib.import_module("torch._inductor.kernel.flex_gemm.runtime")
        self.assertNotIn("quack", sys.modules)


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

    @parametrize(
        "case",
        (
            (
                "sigmoid_fp16",
                torch.ops.aten.sigmoid.default,
                torch.float16,
                (torch.float32, torch.float16),
                torch.float16,
            ),
            (
                "sigmoid_bf16",
                torch.ops.aten.sigmoid.default,
                torch.bfloat16,
                (torch.float32, torch.bfloat16),
                torch.bfloat16,
            ),
            (
                "sigmoid_int",
                torch.ops.aten.sigmoid.default,
                torch.int32,
                (torch.float32,),
                torch.float32,
            ),
            (
                "sigmoid_bool",
                torch.ops.aten.sigmoid.default,
                torch.bool,
                (torch.float32,),
                torch.float32,
            ),
            (
                "silu_fp16",
                torch.ops.aten.silu.default,
                torch.float16,
                (torch.float32, torch.float16),
                torch.float16,
            ),
            (
                "silu_bf16",
                torch.ops.aten.silu.default,
                torch.bfloat16,
                (torch.float32, torch.bfloat16),
                torch.bfloat16,
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_fast_math_decompositions_preserve_type_promotion(self, case):
        from torch._higher_order_ops.flex_gemm import flex_gemm_body_decomposition_table
        from torch._inductor.decomposition import decompositions
        from torch.fx.experimental.proxy_tensor import make_fx

        _, op, input_dtype, conversion_dtypes, result_dtype = case
        decomposition_table = flex_gemm_body_decomposition_table(
            {"backend": "QUACK", "fast_math": True}, decompositions
        )
        graph_module = make_fx(
            lambda x: op(x), decomposition_table=decomposition_table
        )(torch.ones(4, dtype=input_dtype))

        self.assertEqual(
            tuple(
                node.args[1]
                for node in graph_module.graph.nodes
                if node.target is torch.ops.prims.convert_element_type.default
            ),
            conversion_dtypes,
        )
        self.assertEqual(
            graph_module(torch.ones(4, dtype=input_dtype)).dtype, result_dtype
        )
        self.assertTrue(
            any(
                node.target is torch.ops.aten.tanh.default
                for node in graph_module.graph.nodes
            )
        )

    def test_post_grad_addmm_fusion_preserves_flex_gemm_body_mm(self):
        from torch._higher_order_ops.flex_gemm import mark_flex_gemm_body_gemm_node
        from torch._inductor.fx_passes.post_grad import is_valid_addmm_fusion
        from torch.fx.experimental.proxy_tensor import make_fx

        graph_module = make_fx(lambda a, b, bias: torch.mm(a, b) + bias)(
            torch.randn(4, 8), torch.randn(8, 16), torch.randn(16)
        )
        placeholders = [
            node for node in graph_module.graph.nodes if node.op == "placeholder"
        ]
        match = SimpleNamespace(
            args=tuple(placeholders[:2]),
            kwargs={"inp": placeholders[2]},
            nodes=list(graph_module.graph.nodes),
        )
        self.assertTrue(is_valid_addmm_fusion(match))

        mark_flex_gemm_body_gemm_node(graph_module, torch.ops.aten.mm.default)
        self.assertFalse(is_valid_addmm_fusion(match))

    @parametrize(
        "case",
        (
            (
                "addmm",
                torch.ops.aten.addmm.default,
                ((4, 16), (4, 8), (8, 16)),
            ),
            (
                "baddbmm",
                torch.ops.aten.baddbmm.default,
                ((2, 4, 16), (2, 4, 8), (2, 8, 16)),
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_post_grad_unfuse_preserves_flex_gemm_body_gemm(self, case):
        from torch._higher_order_ops.flex_gemm import mark_flex_gemm_body_gemm_node
        from torch._inductor.fx_passes.post_grad import (
            should_prefer_unfused_addmm,
            should_prefer_unfused_baddbmm,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        _, gemm_op, input_shapes = case
        check = {
            torch.ops.aten.addmm.default: should_prefer_unfused_addmm,
            torch.ops.aten.baddbmm.default: should_prefer_unfused_baddbmm,
        }[gemm_op]
        with FakeTensorMode():
            input_values = [torch.empty(shape, device="cuda") for shape in input_shapes]
        graph_module = make_fx(
            lambda inp, mat1, mat2: gemm_op(inp, mat1, mat2).relu(),
            tracing_mode="fake",
        )(*input_values)
        inputs = [node for node in graph_module.graph.nodes if node.op == "placeholder"]
        gemm = next(node for node in graph_module.graph.nodes if node.target is gemm_op)
        match = SimpleNamespace(
            args=tuple(inputs[1:]),
            kwargs={"inp": inputs[0]},
            output_node=lambda: gemm,
        )
        self.assertTrue(check(match))

        mark_flex_gemm_body_gemm_node(graph_module, gemm_op)
        self.assertFalse(check(match))

    def test_epilogue_analysis_matches_prepare_softmax(self):
        from torch._inductor import inductor_prims
        from torch._inductor.kernel.gemm_epilogue_analysis import (
            GemmLocalReduceAnalysis,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(x):
            grouped = x.view(4, 4, 2)
            return inductor_prims.prepare_softmax_online(grouped, -1)[0]

        graph_module = make_fx(body)(torch.randn(4, 8))
        analysis = GemmLocalReduceAnalysis.from_graph_module(graph_module)
        prepare_softmax = next(
            node
            for node in graph_module.graph.nodes
            if node.target is inductor_prims.prepare_softmax_online
        )
        self.assertIn(prepare_softmax, analysis.matches)
        self.assertIsNone(analysis.matches[prepare_softmax].reduction_type)

    def test_epilogue_graph_normalizes_selected_fx_nodes(self):
        import operator

        from torch._inductor.kernel.gemm_epilogue import (
            GemmEpilogueGraph,
            NormalizedDtypeView,
            NormalizedGetItem,
            NormalizedReduction,
            NormalizedSqueeze,
            NormalizedUnsupportedReduction,
            NormalizedView,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(x):
            grouped = x.view(4, 2, 4)
            reduced = grouped.sum(dim=-1, keepdim=True)
            maximum = torch.max(grouped, dim=-1).values
            return (
                reduced.squeeze(-1),
                maximum,
                grouped.var(dim=-1),
                x.view(torch.int32),
            )

        graph_module = make_fx(body)(torch.randn(4, 8))
        normalized_nodes = GemmEpilogueGraph.from_nodes(
            tuple(graph_module.graph.nodes)
        ).normalized_nodes
        nodes = {
            node.target: node
            for node in graph_module.graph.nodes
            if node.target is not operator.getitem
        }
        placeholder = next(
            node for node in graph_module.graph.nodes if node.op == "placeholder"
        )
        view = nodes[torch.ops.aten.view.default]
        reduction = nodes[torch.ops.aten.sum.dim_IntList]
        maximum = nodes[torch.ops.aten.max.dim]
        getitem = next(
            node
            for node in graph_module.graph.nodes
            if node.target is operator.getitem and node.args == (maximum, 0)
        )
        squeeze = nodes[torch.ops.aten.squeeze.dim]
        unsupported = nodes[torch.ops.aten.var.correction]
        dtype_view = nodes[torch.ops.aten.view.dtype]
        self.assertEqual(normalized_nodes[view], NormalizedView(placeholder, (4, 2, 4)))
        self.assertEqual(
            normalized_nodes[reduction],
            NormalizedReduction(view, [-1], True, None, "sum"),
        )
        self.assertEqual(normalized_nodes[squeeze], NormalizedSqueeze(reduction))
        self.assertEqual(normalized_nodes[getitem], NormalizedGetItem(maximum, 0))
        self.assertEqual(
            normalized_nodes[dtype_view], NormalizedDtypeView(placeholder, torch.int32)
        )
        self.assertEqual(
            normalized_nodes[unsupported],
            NormalizedUnsupportedReduction(
                view,
                str(torch.ops.aten.var.correction),
            ),
        )

    def test_local_reduce_propagates_before_grouped_view_matching(self):
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmLocalReduceAnalysis,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(x):
            grouped = x.view(4, 6, 2)
            reduced = grouped.sum(dim=-1, keepdim=True).squeeze(-1)
            return reduced.view(-1, 4, 3)

        graph_module = make_fx(body)(torch.randn(4, 12))
        analysis = FlexGemmLocalReduceAnalysis.from_graph_module(graph_module)
        output = next(
            node for node in graph_module.graph.nodes if node.op == "output"
        ).args[0]
        self.assertIsInstance(output, torch.fx.Node)
        self.assertIn(output, analysis.matches)
        self.assertEqual(
            analysis.matches[output].value_node.target,
            torch.ops.aten.sum.dim_IntList,
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

    def test_output_contraction_device_validation(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )

        group_2 = FlexGemmOutputContraction(2)
        group_4 = FlexGemmOutputContraction(4)
        for capability in (10, 11):
            group_2.validate_quack(capability)
        group_4.validate_quack(10)
        with self.assertRaisesRegex(NotImplementedError, "interleaved group 4"):
            FlexGemmOutputContraction(4, chunked=True).validate_quack(10)
        with self.assertRaisesRegex(NotImplementedError, "group 4 on SM100"):
            group_4.validate_quack(11)
        with self.assertRaisesRegex(NotImplementedError, "not yet supported on SM120"):
            group_2.validate_quack(12)
        with self.assertRaisesRegex(NotImplementedError, "only on SM100 and SM110"):
            group_2.validate_quack(9)
        with self.assertRaisesRegex(NotImplementedError, "only on SM100 and SM110"):
            group_4.validate_quack(9)

    @skipIfNoCuteDSL
    def test_quack_grouped_n_contract_device_validation(self):
        from torch._vendor.quack.gemm_act import validate_grouped_n_contract_device

        validate_grouped_n_contract_device(2, (10, 0))
        validate_grouped_n_contract_device(2, (11, 0))
        validate_grouped_n_contract_device(4, (10, 0))
        with self.assertRaisesRegex(NotImplementedError, "group 2.*group 4"):
            validate_grouped_n_contract_device(4, (11, 0))
        with self.assertRaisesRegex(NotImplementedError, "not yet supported on SM120"):
            validate_grouped_n_contract_device(2, (12, 0))
        with self.assertRaisesRegex(NotImplementedError, "only on SM100 and SM110"):
            validate_grouped_n_contract_device(2, (9, 0))

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    def test_runtime_output_contraction_rejects_unsafe_explicit_config(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        config = SimpleNamespace(
            swap_ab=True,
            cluster_m=1,
            tile_m=128,
            cluster_n=1,
            tile_n=16,
        )
        with (
            mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
                return_value=[config],
            ),
            mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.gemm_config_from_key",
                return_value=config,
            ),
            self.assertRaisesRegex(
                NotImplementedError, "incompatible with output contraction"
            ),
        ):
            gemm_epilogue(
                torch.randn(4, 8, device="cuda"),
                torch.randn(8, 16, device="cuda"),
                lambda acc: acc,
                "unsafe_output_contraction_config",
                output_plan=FlexGemmRuntimeOutputPlan(
                    output_contraction=FlexGemmOutputContraction(2)
                ),
                config_key=(("swap_ab", True),),
                expected_ndim=2,
                device_capacity_override=(10, 0),
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    def test_runtime_output_contraction_selects_safe_default_config(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        unsafe = SimpleNamespace(
            swap_ab=True,
            cluster_m=1,
            tile_m=128,
            cluster_n=1,
            tile_n=16,
        )
        safe = SimpleNamespace(
            swap_ab=False,
            cluster_m=1,
            tile_m=128,
            cluster_n=1,
            tile_n=16,
        )
        with (
            mock.patch(
                "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
                return_value=[unsafe, safe],
            ),
            mock.patch(
                "torch._inductor.kernel.flex_gemm.runtime.dispatch_gemm_act"
            ) as dispatch,
        ):
            gemm_epilogue(
                torch.randn(4, 8, device="cuda"),
                torch.randn(8, 16, device="cuda"),
                lambda acc: acc,
                "safe_output_contraction_config",
                output_plan=FlexGemmRuntimeOutputPlan(
                    output_contraction=FlexGemmOutputContraction(2)
                ),
                device_capacity_override=(10, 0),
            )
        self.assertIs(dispatch.call_args.kwargs["config"], safe)

    def test_explicit_dense_config_supports_partial_constraints(self):
        from torch._inductor.heuristics.template import (
            flex_gemm as flex_gemm_heuristics,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.lowering import flex_gemm_config_keys
        from torch._inductor.virtualized import V
        from torch._vendor.quack.gemm_config import GemmConfig

        supported = GemmConfig(
            tile_m=128,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        )
        alternative = dataclasses.replace(supported, tile_n=256, cluster_m=1)
        config = dict(flex_gemm_heuristics.gemm_config_key(supported))
        device = torch.device("cuda")
        fake_graph = SimpleNamespace(
            sizevars=SimpleNamespace(guard_or_false=lambda expr: bool(expr))
        )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[supported, alternative],
            ),
            V.set_graph_handler(fake_graph),
        ):
            self.assertEqual(
                flex_gemm_heuristics.explicit_gemm_configs_for_device(config, device),
                (supported,),
            )
            self.assertEqual(
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {"tile_m": 128}, device
                ),
                (supported, alternative),
            )
            self.assertEqual(
                flex_gemm_config_keys(
                    device,
                    128,
                    128,
                    (),
                    tuned=False,
                    explicit_config={"tile_m": 128},
                ),
                (flex_gemm_heuristics.gemm_config_key(supported),),
            )
            self.assertEqual(
                flex_gemm_config_keys(
                    device,
                    128,
                    128,
                    (),
                    tuned=True,
                    explicit_config={"tile_m": 128},
                ),
                tuple(
                    flex_gemm_heuristics.gemm_config_key(candidate)
                    for candidate in (supported, alternative)
                ),
            )
            with self.assertRaisesRegex(NotImplementedError, "unexpected.*stages"):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {**config, "stages": 4}, device
                )
            with self.assertRaisesRegex(NotImplementedError, "not supported"):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {"cluster_m": True}, device
                )
            with self.assertRaisesRegex(NotImplementedError, "not supported"):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {**config, "use_tma_gather": True}, device
                )
            with self.assertRaisesRegex(
                NotImplementedError, "targets SM90.*uses SM100"
            ):
                flex_gemm_heuristics.explicit_gemm_configs_for_device(
                    {**config, "device_capacity": 9}, device
                )

        swap_config = dict(
            flex_gemm_heuristics.gemm_config_key(
                dataclasses.replace(supported, swap_ab=True)
            )
        )
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch(
                "torch._vendor.quack.gemm_config.get_all_configs",
                return_value=[dataclasses.replace(supported, swap_ab=True)],
            ),
            V.set_graph_handler(fake_graph),
            self.assertRaisesRegex(
                NotImplementedError, "incompatible with local reduction"
            ),
        ):
            flex_gemm_config_keys(
                device,
                128,
                128,
                (FlexGemmLocalReduceGeometry(group=16, axis=1),),
                tuned=False,
                explicit_config=swap_config,
            )

    def test_multi_cta_local_reduce_full_tile_follows_n_warp_topology(self):
        """Allow full-N groups only when one epilogue warp partition owns N."""
        from torch._inductor.kernel.flex_gemm.constraints import (
            max_flex_gemm_local_reduce_group_for_configs,
            validate_flex_gemm_local_reduce_config,
        )
        from torch._vendor.quack.gemm_config import GemmConfig

        configs = (
            GemmConfig(
                tile_m=128,
                tile_n=256,
                pingpong=False,
                cluster_m=2,
                device_capacity=10,
            ),
            GemmConfig(
                tile_m=256,
                tile_n=512,
                pingpong=False,
                cluster_m=2,
                device_capacity=10,
            ),
        )
        split_n_warp, single_n_warp = configs
        self.assertTrue(validate_flex_gemm_local_reduce_config(split_n_warp, 128, 1))
        self.assertFalse(
            validate_flex_gemm_local_reduce_config(split_n_warp, split_n_warp.tile_n, 1)
        )
        self.assertTrue(validate_flex_gemm_local_reduce_config(single_n_warp, 128, 1))
        self.assertTrue(
            validate_flex_gemm_local_reduce_config(
                single_n_warp, single_n_warp.tile_n, 1
            )
        )
        self.assertFalse(
            validate_flex_gemm_local_reduce_config(
                single_n_warp, single_n_warp.tile_m, 0
            )
        )
        self.assertEqual(max_flex_gemm_local_reduce_group_for_configs(configs, 1), 512)

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
    def makeTensor(self, *shape, device="cuda", dtype=torch.bfloat16):
        return torch.testing.make_tensor(
            *shape, device=device, dtype=dtype, low=-0.1, high=0.1
        )

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
        else:
            file_check = file_check.check_not("callbacks=FlexGemmLocalReduceCallbacks")
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
            LOCAL_REDUCE_FRAGMENT_WIDTH,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
        )

        callbacks = None
        if feeds_main or axis == 0 or group > LOCAL_REDUCE_FRAGMENT_WIDTH:
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

    def assertPhysicalFeedMainCode(self, code, group=None):
        """Check generated code uses the current physical feed-main ABI."""
        file_check = FileCheck().check("local_reduce=FlexGemmRuntimeLocalReducePlan")
        if group is None:
            file_check = file_check.check("FlexGemmLocalReduceGeometry(group=").check(
                "axis=0)"
            )
        else:
            file_check = file_check.check(self.localReduceGeometryPattern(group, 0))
        file_check.check("callbacks=FlexGemmLocalReduceCallbacks").check_not(
            "local_reduce_out="
        ).check_not("local_reduce_feeds_main=True").check_not("local_reduce_op").run(
            code
        )


@skipIfNoCuteDSL
@unittest.skipIf(not TEST_CUDA, "CUDA required")
@instantiate_parametrized_tests
class TestFlexGemmRuntime(FlexGemmTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.relu_epilogue = staticmethod(relu_epilogue)
        cls.captured_affine_epilogue = staticmethod(captured_affine_epilogue)
        cls.row_scale_epilogue = staticmethod(row_scale_epilogue)
        cls.scalar_scale_epilogue = staticmethod(scalar_scale_epilogue)
        cls.captured_affine_scalar_epilogue = staticmethod(
            captured_affine_scalar_epilogue
        )
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
    def test_mm_epilogue_infers_captured_arg_kinds(self):
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
            self.captured_affine_epilogue,
            "test_flex_gemm_infer_captured_args",
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
    def test_mm_epilogue_infers_scalar_captured_arg_kind(self):
        from torch._inductor.kernel.flex_gemm.runtime import (
            gemm_epilogue,
            resolve_epilogue_arg_kinds,
        )

        torch.manual_seed(5)
        m, n, k = 128, 128, 64
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        scale = self.makeTensor(1, 1, dtype=torch.float32)

        # A [1, 1] arg is a scalar even when M == 1 or N == 1 would also
        # make it a valid row/col broadcast.
        self.assertEqual(
            resolve_epilogue_arg_kinds(
                torch.empty(1, k), torch.empty(k, 1), (torch.empty(1, 1),), ()
            ),
            ("scalar",),
        )

        out = gemm_epilogue(
            a,
            b,
            self.scalar_scale_epilogue,
            "test_flex_gemm_infer_scalar_arg",
            out_dtype=torch.float32,
            epilogue_args=(scale,),
        )
        self.assertMatchesLowPrecisionEager(
            out,
            (a @ b).float() * scale,
            (a.double() @ b.double()) * scale.double(),
            k,
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

    @parametrize("group", (33, 64))
    def test_runtime_plan_rejects_cross_warp_feed_main_group(self, group):
        with self.assertRaisesRegex(
            NotImplementedError, "same-warp axis-0 groups <= 32"
        ):
            self.runtimeLocalReducePlan(
                group=group,
                axis=0,
                feeds_main=True,
            )

    def test_runtime_plan_rejects_axis_one_feed_main(self):
        with self.assertRaisesRegex(NotImplementedError, "supports only axis 0"):
            self.runtimeLocalReducePlan(
                group=8,
                axis=1,
                feeds_main=True,
            )

    def test_runtime_validation_rejects_non_divisible_local_reduce_group(self):
        from torch._inductor.kernel.flex_gemm.runtime import (
            validate_runtime_local_reduce,
        )

        plan = self.runtimeLocalReducePlan(
            group=8,
            axis=0,
            feeds_main=True,
        )
        with self.assertRaisesRegex(RuntimeError, "must divide"):
            validate_runtime_local_reduce(
                plan,
                torch.empty(130, 64),
                (130, 64),
                None,
                1.0,
                1.0,
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
                1.0,
                1.0,
            )

    def test_runtime_validation_rejects_local_reduce_with_c_alpha_beta(self):
        from torch._inductor.kernel.flex_gemm.runtime import (
            validate_runtime_local_reduce,
        )

        compressed_plan = self.runtimeLocalReducePlan(
            out=torch.empty(128, 8), group=8, axis=1
        )
        feed_main_plan = self.runtimeLocalReducePlan(
            group=8,
            axis=0,
            feeds_main=True,
        )
        for plan in (compressed_plan, feed_main_plan):
            for effective_C, alpha, beta in (
                (torch.empty(128, 64), 1.0, 1.0),
                (None, 0.5, 1.0),
                (None, 1.0, 0.5),
            ):
                with self.assertRaisesRegex(NotImplementedError, "C/alpha/beta"):
                    validate_runtime_local_reduce(
                        plan,
                        torch.empty(128, 64),
                        (128, 64),
                        effective_C,
                        alpha,
                        beta,
                    )

    @skipIfNoCuteDSL
    def test_quack_feed_main_host_guards_match_runtime_contract(self):
        from torch._vendor.quack.gemm_act import gemm_act

        a = torch.empty(1, 4, 8, dtype=torch.bfloat16)
        b = torch.empty(1, 8, 8, dtype=torch.bfloat16)
        out = torch.empty(1, 4, 8, dtype=torch.bfloat16)
        c = torch.empty(1, 4, 8, dtype=torch.bfloat16)

        def tensor_epilogue(acc, local_reduce0):
            return acc

        def call_gemm(
            C=None,
            mat1=a,
            output=out,
            local_reduce_group=8,
            **kwargs,
        ):
            return gemm_act(
                mat1,
                b,
                None,
                C,
                output,
                None,
                None,
                128,
                128,
                1,
                1,
                tensor_epilogue_fn=tensor_epilogue,
                local_reduce_feeds_main=True,
                local_reduce_group=local_reduce_group,
                local_reduce_axis=0,
                local_reduce_combine_key="combine",
                local_reduce_finalize_key="finalize",
                device_capacity_override=(10, 0),
                **kwargs,
            )

        for kwargs in ({"C": c}, {"alpha": 0.5}, {"beta": 0.5}):
            with self.assertRaisesRegex(NotImplementedError, "C/alpha/beta"):
                call_gemm(**kwargs)

        with self.assertRaisesRegex(RuntimeError, "divide TensorSSA fragment width"):
            call_gemm(local_reduce_group=3)
        with self.assertRaisesRegex(RuntimeError, "divide the selected GEMM output"):
            call_gemm(
                mat1=torch.empty(1, 6, 8, dtype=torch.bfloat16),
                output=torch.empty(1, 6, 8, dtype=torch.bfloat16),
                local_reduce_group=4,
            )

        with self.assertRaisesRegex(
            RuntimeError, "requires tensor_epilogue_fn or tensor_epilogue_key"
        ):
            gemm_act(
                a,
                b,
                None,
                None,
                out,
                None,
                None,
                128,
                128,
                1,
                1,
                local_reduce_feeds_main=True,
                local_reduce_group=8,
                local_reduce_axis=0,
                local_reduce_combine_key="combine",
                local_reduce_finalize_key="finalize",
                device_capacity_override=(10, 0),
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
            local_reduce_gemm_act_kwargs,
        )
        from torch._inductor.kernel.flex_gemm.template import (
            FlexGemmEpilogueLocalReduceConfig,
        )

        callbacks = FlexGemmLocalReduceCallbacks(
            lambda lhs, rhs: lhs, lambda value: value
        )
        geometry = FlexGemmLocalReduceGeometry(8, 0)
        out = torch.empty(1)
        feed_plan = FlexGemmRuntimeLocalReducePlan(
            geometry, callbacks=callbacks, feeds_main=True
        )
        shared_plan = FlexGemmRuntimeLocalReducePlan(
            geometry,
            out=out,
            callbacks=callbacks,
            feeds_main=True,
        )
        store_plan = FlexGemmRuntimeLocalReducePlan(
            geometry, out=out, callbacks=callbacks
        )
        self.assertTrue(feed_plan.feeds_main)
        self.assertTrue(shared_plan.feeds_main)
        self.assertFalse(store_plan.feeds_main)
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

        feed_kwargs = local_reduce_gemm_act_kwargs(
            feed_plan, None, ("combine", "finalize")
        )
        self.assertTrue(feed_kwargs["local_reduce_feeds_main"])
        self.assertFalse(feed_kwargs["tensor_epilogue_returns_local_reduce"])
        shared_kwargs = local_reduce_gemm_act_kwargs(
            shared_plan, out, ("combine", "finalize")
        )
        self.assertTrue(shared_kwargs["local_reduce_feeds_main"])
        self.assertTrue(shared_kwargs["tensor_epilogue_returns_local_reduce"])

    def test_output_plan_rejects_invalid_state(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmLocalReduceAnalysis,
            FlexGemmLocalReduceMatch,
            FlexGemmLocalReduceStore,
            FlexGemmOutputLocalReducePlan,
            FlexGemmOutputPlan,
            tuple_output_plan,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmEpilogueGraph

        graph = torch.fx.Graph()
        node = graph.placeholder("x")
        aux = graph.placeholder("aux")
        geometry = FlexGemmLocalReduceGeometry(8, 0)
        match = FlexGemmLocalReduceMatch(aux, geometry)
        analysis = FlexGemmLocalReduceAnalysis(
            GemmEpilogueGraph(dependencies={}, normalized_nodes={})
        )
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

    def test_output_plan_uses_shared_reduction_plan(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmLocalReduceMatch,
            FlexGemmLocalReduceStore,
            FlexGemmOutputLocalReducePlan,
            FlexGemmOutputPlan,
        )
        from torch._inductor.kernel.gemm_epilogue import GemmReductionPlan

        graph = torch.fx.Graph()
        output = graph.placeholder("output")
        reduced = graph.placeholder("reduced")
        match = FlexGemmLocalReduceMatch(
            reduced,
            FlexGemmLocalReduceGeometry(8, 0),
            reduction_type="max",
        )
        outputs = FlexGemmOutputPlan(
            output,
            local_reduce=FlexGemmOutputLocalReducePlan(
                match,
                store=FlexGemmLocalReduceStore(reduced, 0),
                feeds_main=True,
            ),
        )

        self.assertEqual(
            outputs.reduction_plan,
            GemmReductionPlan(
                "reduced",
                8,
                0,
                "max",
                "identity",
                "output",
                feeds_main=True,
                feed_output="output",
            ),
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
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            FlexGemmEpilogueEmitter,
        )

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
            FlexGemmRuntimeOutputPlan,
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
                output_plan=FlexGemmRuntimeOutputPlan(
                    local_reduce=FlexGemmRuntimeLocalReducePlan(
                        FlexGemmLocalReduceGeometry(group, 1),
                        out=local_reduce_out,
                    )
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
    def test_mm_fragment_group32_forced_config_extremes_match_reference(self):
        """Pin the empirically verified g32 axis-1 fragment contract per config.

        Every config passing the axis-1 group-32 gate was verified to expose a
        full 32-lane fragment to the generated TensorSSA reduce; keep the
        extreme tile_n configs as regression sentinels for that geometry.
        """
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_flex_gemm_local_reduce_config,
        )
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        m, n, k, group = 128, 256, 64, 32
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)
        gated = [
            config
            for config in candidate_gemm_configs_for_device(a.device)
            if validate_flex_gemm_local_reduce_config(config, group, 1)
        ]
        self.assertTrue(gated)
        tile_ns = sorted({config.tile_n for config in gated})
        selected = [
            next(config for config in gated if config.tile_n == tile_n)
            for tile_n in (tile_ns[0], tile_ns[-1])
        ]
        x = (a.double() @ b.double()).view(m, -1, group)
        expected = (x * x.sum(-1, keepdim=True).reciprocal()).view(m, n)
        wrong_group = (a.double() @ b.double()).view(m, -1, group // 2)
        wrong_expected = (
            wrong_group * wrong_group.sum(-1, keepdim=True).reciprocal()
        ).view(m, n)
        for config in selected:
            out = gemm_epilogue(
                a,
                b,
                fragment_group32_sum_feed_epilogue,
                f"test_flex_gemm_fragment_group32_tile_n_{config.tile_n}",
                out_dtype=torch.float32,
                config_key=gemm_config_key(config),
            )
            torch.testing.assert_close(out.double(), expected, atol=1e-4, rtol=1e-4)
            # A partial-fragment reduce would match the half-group reference.
            self.assertGreater((out.double() - wrong_expected).abs().max(), 1e-2)

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_physical_feed_main_forced_config_extremes_match_reference(self):
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceCallbacks,
            FlexGemmLocalReduceGeometry,
            validate_flex_gemm_local_reduce_config,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeLocalReducePlan,
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

        m, n, k, group = 128, 128, 64, 32
        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16) + 0.5
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16) + 0.5
        gated = [
            config
            for config in candidate_gemm_configs_for_device(a.device)
            if validate_flex_gemm_local_reduce_config(config, group, 0)
        ]
        self.assertTrue(gated)
        extremes = (
            min(gated, key=lambda config: (config.tile_m, config.tile_n)),
            max(gated, key=lambda config: (config.tile_m, config.tile_n)),
            min(gated, key=lambda config: (config.tile_n, config.tile_m)),
            max(gated, key=lambda config: (config.tile_n, config.tile_m)),
        )
        selected = {gemm_config_key(config): config for config in extremes}.values()
        accumulator = a.double() @ b.double()
        grouped = accumulator.view(m // group, group, n)
        expected = (grouped / grouped.sum(1, keepdim=True)).view(m, n)
        callbacks = FlexGemmLocalReduceCallbacks(
            physical_group_sum_combine, physical_group_sum_finalize
        )
        local_reduce = FlexGemmRuntimeLocalReducePlan(
            FlexGemmLocalReduceGeometry(group, 0),
            callbacks=callbacks,
            feeds_main=True,
        )
        for config in selected:
            out = gemm_epilogue(
                a,
                b,
                physical_group_sum_feed_epilogue,
                (
                    "test_flex_gemm_physical_feed_main_"
                    f"{config.tile_m}_{config.tile_n}_{config.cluster_m}_{config.cluster_n}"
                ),
                out_dtype=torch.float32,
                output_plan=FlexGemmRuntimeOutputPlan(local_reduce=local_reduce),
                config_key=gemm_config_key(config),
            )
            torch.testing.assert_close(out.double(), expected, atol=1e-4, rtol=1e-4)

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
    def test_swap_ab_captured_args_matches_non_swap(self):
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
                self.captured_affine_epilogue,
                name,
                out_dtype=torch.float32,
                epilogue_args=(col_bias, row_scale, tile_bias),
                epilogue_arg_kinds=("col", "row", "tile"),
                config_key=config_key,
            )

        swapped = run("test_flex_gemm_swap_ab_captured_args", swap_key)
        non_swapped = run("test_flex_gemm_non_swap_ab_captured_args", non_swap_key)
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
    def test_swap_ab_scalar_captured_arg_matches_non_swap(self):
        from torch._inductor.kernel.flex_gemm.runtime import gemm_epilogue

        torch.manual_seed(11)
        m, n, k = 128, 384, 256
        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        col_bias = self.makeTensor(m, 1, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, dtype=torch.float32)
        scale = self.makeTensor(1, 1, dtype=torch.float32)
        swap_key, non_swap_key = self.swapAndNonSwapConfigKeys(a.device)

        def run(name, config_key):
            return gemm_epilogue(
                a,
                b,
                self.captured_affine_scalar_epilogue,
                name,
                out_dtype=torch.float32,
                epilogue_args=(col_bias, row_scale, tile_bias, scale),
                epilogue_arg_kinds=("col", "row", "tile", "scalar"),
                config_key=config_key,
            )

        swapped = run("test_flex_gemm_swap_ab_scalar_arg", swap_key)
        non_swapped = run("test_flex_gemm_non_swap_ab_scalar_arg", non_swap_key)
        # Scalars are orientation-invariant, so swap_ab must be bit-identical.
        self.assertEqual(swapped, non_swapped)
        high_precision_expected = (
            (
                (a.double() @ b.double() + col_bias.double()) * row_scale.double()
                + tile_bias.double()
            )
            * scale.double()
        ).relu()
        low_precision_expected = (
            (((a @ b).float() + col_bias) * row_scale + tile_bias) * scale
        ).relu()
        self.assertMatchesLowPrecisionEager(
            swapped, low_precision_expected, high_precision_expected, k
        )

    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_swap_ab_captured_args_tuple_aux_matches_non_swap(self):
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

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
                output_plan=FlexGemmRuntimeOutputPlan(aux_outs=(aux,)),
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
            self.captured_affine_epilogue,
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
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmRuntimeOutputPlan,
            gemm_epilogue,
        )

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
            output_plan=FlexGemmRuntimeOutputPlan(aux_outs=(aux,)),
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

    def assertOutputContractionRejected(self, a, b, epilogue, error):
        """Assert that an output contraction is rejected during compilation."""
        torch._dynamo.reset()
        with self.assertRaisesRegex(InductorError, error):
            torch.compile(
                lambda a, b: flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue,
                    kernel_options={"backend": "QUACK"},
                ),
                backend="inductor",
                fullgraph=True,
            )(a, b)

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

    @skipIfNoCuteDSL
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
        scale = torch.randn(5)

        with self.assertRaisesRegex(
            Exception,
            "captured tensor epilogue args currently must match",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, scale)

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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
            ("scalar", lambda m, n: (1, 1)),
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
    def test_mm_preserves_integer_scalar_captured_tensor_epilogue_arg(self):
        def epilogue_fn(acc, selector):
            acc_float = acc.float()
            return torch.where(selector.bitwise_and(1).bool(), acc_float, -acc_float)

        def fn(a, b, selector):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, selector),
                kernel_options={"backend": "QUACK"},
            )

        m, k, n = 128, 64, 128
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        selector = torch.tensor([[2**24 + 1]], device="cuda", dtype=torch.int64)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, selector
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, selector),
            epilogue_fn(a.double() @ b.double(), selector),
            a.shape[1],
        )
        FileCheck().check("epilogue_arg_kinds=('scalar',)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile", lambda m, n: (m, n)),
            ("row", lambda m, n: (1, n)),
            ("col", lambda m, n: (m, 1)),
            ("scalar", lambda m, n: (1, 1)),
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

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "tuned",
        (False, True),
        name_fn=lambda tuned: "tuned" if tuned else "untuned",
    )
    def test_mm_scalar_captured_arg_fp8_quant_matches_reference(self, tuned):
        torch._dynamo.reset()
        m, k, n = 128, 64, 128

        def epilogue_fn(acc, scale):
            return acc * scale.abs()

        def fn(a, b, scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc.float(), scale).to(torch.float8_e4m3fn),
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
                "torch._inductor.template_heuristics.flex_gemm.candidate_gemm_configs_for_device",
                return_value=configs,
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([[-0.25]], device="cuda", dtype=torch.float32)
        with config_context:
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
            )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn((a @ b).float(), scale).to(torch.float8_e4m3fn),
            epilogue_fn(a.double() @ b.double(), scale.double()),
            a.shape[1],
        )
        FileCheck().check("epilogue_arg_kinds=('scalar',)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_scalar_and_row_captured_args_matches_reference(self):
        m, k, n = 128, 64, 128

        def epilogue_fn(acc, scale, row_scale):
            return (acc.float() * scale * row_scale).relu()

        def fn(a, b, scale, row_scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale, row_scale),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([[0.5]], device="cuda", dtype=torch.float32)
        row_scale = torch.randn(1, n, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            scale,
            row_scale,
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, scale, row_scale),
            epilogue_fn(a.double() @ b.double(), scale.double(), row_scale.double()),
            a.shape[1],
        )
        FileCheck().check("epilogue_arg_kinds=('scalar', 'row')").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_scalar_captured_arg_composes_with_compressed_local_reduce(self):
        m, k, n = 128, 64, 128
        group = 32

        def epilogue_fn(acc, scale):
            x = acc.float().view(m, -1, group)
            return acc.float() * scale, x.abs().amax(-1)

        def fn(a, b, scale):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, scale),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([[0.25]], device="cuda", dtype=torch.float32)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale
        )

        expected, _ = epilogue_fn(a @ b, scale)
        high_precision_expected, high_precision_aux = epilogue_fn(
            a.double() @ b.double(), scale.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual, expected, high_precision_expected, a.shape[1]
        )
        torch.testing.assert_close(
            aux, high_precision_aux.float(), atol=1e-3, rtol=1e-3
        )
        FileCheck().check("epilogue_arg_kinds=('scalar',)").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
    def test_generated_local_reduce_feed_main_rejects_addmm_scope(self):
        def fn(bias, a, b):
            def epilogue(acc):
                x = acc.float().view(-1, 4, 5)
                scale = x.sum(1, keepdim=True)
                return (x * scale.reciprocal()).view(4, 5)

            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        bias = torch.randn(4, 5)
        a = torch.randn(4, 8)
        b = torch.randn(8, 5)

        with self.assertRaisesRegex(Exception, "currently support only aten.mm"):
            torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

    @skipIfNoCuteDSL
    def test_generated_local_reduce_fragment_feed_main_rejects_addmm_scope(self):
        def fn(bias, a, b):
            def epilogue(acc):
                x = acc.float().view(4, -1, 4)
                scale = x.sum(-1, keepdim=True)
                return (x * scale.reciprocal()).view(4, 8)

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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
    @parametrize(
        "case",
        (
            (
                "m_reduce_feeds_main",
                lambda acc: (
                    acc.float().view(-1, 4, 8)
                    * (acc.float().view(-1, 4, 8).sum(1, keepdim=True) + 1.0)
                ).view(4, 8),
                (4, 8),
                "one generated physical reduction",
            ),
            (
                "m_reduce_feeds_same_shape_aux",
                lambda acc: (
                    acc.relu(),
                    (
                        acc.float().view(-1, 4, 8)
                        * (acc.float().view(-1, 4, 8).mean(1, keepdim=True) + 1.0)
                    ).view(4, 8),
                ),
                (4, 8),
                "one generated physical reduction",
            ),
            (
                "large_n_reduce_feeds_main",
                lambda acc: (
                    acc.float().view(4, -1, 64)
                    * (acc.float().view(4, -1, 64).sum(-1, keepdim=True) + 1.0)
                ).view(4, 128),
                (4, 128),
                "post-reduction pointwise transforms",
            ),
            (
                "large_n_reduce_feeds_same_shape_aux",
                lambda acc: (
                    acc.relu(),
                    (
                        acc.float().view(4, -1, 64)
                        * (acc.float().view(4, -1, 64).mean(-1, keepdim=True) + 1.0)
                    ).view(4, 128),
                ),
                (4, 128),
                "post-reduction pointwise transforms",
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_generated_local_reduce_rejects_physical_result_feeding_pointwise(
        self, case
    ):
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

    @skipIfNoCuteDSL
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

    @skipIfNoCuteDSL
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

    def test_flex_gemm_debug_report(self):
        from torch._inductor.kernel.flex_gemm.debug import (
            flex_gemm_log,
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
            format_flex_gemm_config_key,
            log_flex_gemm_artifact,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b):
            acc = torch.mm(a, b)
            return torch.relu(acc), (acc * acc).view(4, -1, 32).sum(-1)

        graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 64))
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        analysis_report = format_flex_gemm_analysis(analysis)
        analysis_details = format_flex_gemm_analysis_details(analysis)
        config_report = format_flex_gemm_config_key(
            (
                ("tile_m", 256),
                ("tile_n", 128),
                ("tile_k", None),
                ("cluster_m", 2),
                ("cluster_n", 1),
                ("cluster_k", 1),
            )
        )

        self.assertIn("outputs:\n  main: relu: shape=(4, 64)", analysis_report)
        self.assertIn("auxiliary: (none)", analysis_report)
        self.assertIn("output_contraction: none", analysis_report)
        self.assertIn("value: sum_1", analysis_report)
        self.assertIn("dataflow: view -> sum(dim=[-1], keepdim=False)", analysis_report)
        self.assertIn("geometry: axis=N, group=32", analysis_report)
        self.assertIn("consumers: returned", analysis_report)
        self.assertIn("output_layout: dense", analysis_report)
        self.assertIn("config_constraints:\n  axis=N, group=32", analysis_report)
        self.assertNotIn("recognized_dataflow:", analysis_report)
        self.assertNotIn("normalized_nodes:", analysis_report)
        self.assertIn("normalized_nodes:\n  view: NormalizedView", analysis_details)
        self.assertIn("sum_1: NormalizedReduction", analysis_details)
        self.assertIn("grouped_layouts:\n  view:", analysis_details)
        self.assertIn("local_reduce_matches:\n  sum_1:", analysis_details)
        self.assertIn("output_contraction_select_indices:\n  (none)", analysis_details)
        self.assertIn("tile_m: 256", config_report)
        self.assertIn("tile_k: auto", config_report)
        self.assertIn("cluster_m: 2", config_report)
        with self.assertLogs(flex_gemm_log, level="INFO") as records:
            log_flex_gemm_artifact("analysis", lambda: analysis_report)
            log_flex_gemm_artifact(
                "analysis_details", lambda: analysis_details, verbose=True
            )
        self.assertEqual(len(records.output), 1)
        self.assertIn(
            "FLEXGEMM LOWERING\n ===== ANALYSIS =====\noutputs:\n  main: relu",
            records.output[0],
        )

        with self.assertLogs(flex_gemm_log, level="DEBUG") as records:
            log_flex_gemm_artifact(
                "analysis_details", lambda: analysis_details, verbose=True
            )
        self.assertIn(" ===== ANALYSIS DETAILS =====", records.output[0])

    def test_output_contraction_debug_report(self):
        from torch._inductor.kernel.flex_gemm.debug import (
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b):
            grouped = torch.mm(a, b).view(4, 8, 2)
            return grouped.select(-1, 0) - grouped.select(-1, 1)

        graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 16))
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )

        self.assertIn(
            "output_contraction: N-axis, group=2, layout=interleaved",
            format_flex_gemm_analysis(analysis),
        )
        details = format_flex_gemm_analysis_details(analysis)
        self.assertIn("NormalizedSelect", details)
        self.assertIn("output_contraction_select_indices:", details)
        self.assertNotIn("output_contraction_select_indices:\n  (none)", details)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_emits_flex_gemm_debug_report(self):
        from torch._inductor.kernel.flex_gemm.debug import flex_gemm_log

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                torch.relu,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        with self.assertLogs(flex_gemm_log, level="DEBUG") as records:
            actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertEqual(actual, torch.relu(a @ b))
        report = "\n".join(
            record.getMessage()
            for record in records.records
            if record.levelno == logging.INFO
        )
        phases = (
            " ===== PROBLEM =====",
            " ===== ANALYSIS =====",
            " ===== LOWERING PLAN =====",
            " ===== SELECTION =====",
        )
        positions = tuple(report.index(phase) for phase in phases)
        self.assertEqual(positions, tuple(sorted(positions)))
        self.assertIn("gemm_op: aten.mm.default", report)
        self.assertIn("outputs:\n  main: relu", report)
        self.assertNotIn("normalized_nodes:", report)
        self.assertNotIn("GENERATED EPILOGUE", report)
        self.assertNotIn("CONFIG CANDIDATES", report)
        self.assertNotIn("\x1b[", report)
        self.assertIn("FLEXGEMM LOWERING [flex_gemm_body_graph_", report)
        self.assertIn("mode: fixed", report)
        self.assertIn("lowering_approved_candidates: 1", report)
        self.assertIn("template: cutedsl_flex_gemm_epilogue_", report)
        self.assertIn('analysis/codegen: TORCH_LOGS="+flex_gemm"', report)

        verbose_report = "\n".join(
            record.getMessage()
            for record in records.records
            if record.levelno == logging.DEBUG
        )
        verbose_phases = (
            " ===== ANALYSIS DETAILS =====",
            " ===== GENERATED EPILOGUE =====",
            " ===== CONFIG CANDIDATES =====",
        )
        verbose_positions = tuple(
            verbose_report.index(phase) for phase in verbose_phases
        )
        self.assertEqual(verbose_positions, tuple(sorted(verbose_positions)))
        self.assertIn("normalized_nodes:", verbose_report)
        self.assertIn("@cute.jit", verbose_report)
        self.assertIn("candidate 0:", verbose_report)

    def test_grouped_layout_rejects_inexact_inferred_preserved_dimension(self):
        from torch._inductor.kernel.gemm_epilogue_analysis import grouped_tensor_layout

        with self.assertRaisesRegex(
            NotImplementedError, "grouped reshape must split exactly"
        ):
            grouped_tensor_layout((-1, 2, 2), (4, 5))

    @parametrize("unbacked_dim", (1, 2))
    def test_grouped_layout_rejects_unbacked_structural_dimensions(self, unbacked_dim):
        from torch._inductor.kernel.gemm_epilogue_analysis import grouped_tensor_layout
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape = [4, -1, 2]
        shape[unbacked_dim] = ShapeEnv().create_unbacked_symint()
        self.assertIsNone(grouped_tensor_layout(tuple(shape), (4, 8)))

    def test_grouped_layout_guards_backed_group_before_rejection(self):
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.gemm_epilogue_analysis import grouped_tensor_layout
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(3, ConstantSource("group"))
        group = shape_env.create_symintnode(symbol, hint=3)

        self.assertIsNone(grouped_tensor_layout((4, 5, group), (4, 8)))
        self.assertEqual(len(shape_env.guards), 1)

    def test_output_contraction_plan_only_mutates_analysis_on_match(self):
        """Rejected recognitions must not leak grouped layouts."""
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
        )
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def swap_halves_plus_acc(a, b):
            acc = torch.mm(a, b)
            halves = acc.chunk(2, dim=-1)
            return torch.cat((halves[1], halves[0]), dim=-1) + acc

        def silu_mul_halves(a, b):
            acc = torch.mm(a, b)
            halves = acc.chunk(2, dim=-1)
            return torch.nn.functional.silu(halves[0]) * halves[1]

        chunked_contraction = FlexGemmOutputContraction(group=2, chunked=True)
        for body, expected_contraction in (
            (swap_halves_plus_acc, None),
            (silu_mul_halves, chunked_contraction),
        ):
            with self.subTest(body=body.__name__):
                graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 16))
                shape_env = ShapeEnv()
                symbol = shape_env.create_symbol(
                    8, ConstantSource(f"{body.__name__}_split_size")
                )
                split_size = shape_env.create_symintnode(symbol, hint=8)
                for node in graph_module.graph.nodes:
                    if node.target is torch.ops.aten.split.Tensor:
                        node.args = (node.args[0], split_size, node.args[2])
                analysis = analyze_flex_gemm_epilogue(
                    graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
                )
                split_nodes = [
                    node
                    for node in graph_module.graph.nodes
                    if node.target is torch.ops.aten.split.Tensor
                ]
                self.assertTrue(split_nodes)
                self.assertEqual(
                    analysis.outputs.output_contraction, expected_contraction
                )
                registered = [
                    node
                    for node in split_nodes
                    if node in analysis.local_reduce.grouped_tensors
                ]
                self.assertEqual(
                    registered, split_nodes if expected_contraction else []
                )
                self.assertEqual(len(shape_env.guards), 1)

    def test_incomplete_grouped_selects_are_rejected(self):
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def duplicate_group_value(a, b):
            grouped = torch.mm(a, b).view(4, 4, 4)
            return torch.cat(
                (
                    grouped.select(-1, 0),
                    grouped.select(-1, 1),
                    grouped.select(-1, 2),
                    grouped.select(-1, 2),
                ),
                dim=-1,
            )

        graph_module = make_fx(duplicate_group_value)(
            torch.randn(4, 8), torch.randn(8, 16)
        )
        with self.assertRaisesRegex(NotImplementedError, "complete output contraction"):
            analyze_flex_gemm_epilogue(
                graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
            )

    @parametrize("pointwise", (False, True))
    def test_accepted_grouped_reduction_installs_group_guard(self, pointwise):
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def grouped_reduction(a, b):
            grouped = torch.mm(a, b).view(4, 8, 2)
            source = grouped + 1 if pointwise else grouped
            reduced = source.sum(-1, keepdim=True)
            return (grouped / reduced).view(4, 16)

        graph_module = make_fx(grouped_reduction)(torch.randn(4, 8), torch.randn(8, 16))
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(2, ConstantSource("view_group"))
        group = shape_env.create_symintnode(symbol, hint=2)
        self.assertIsInstance(group, torch.SymInt)
        view = next(
            node
            for node in graph_module.graph.nodes
            if node.target is torch.ops.aten.view.default
            and tuple(node.meta["val"].shape) == (4, 8, 2)
        )
        view.args = (view.args[0], (4, 8, group))

        analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        self.assertEqual(len(shape_env.guards), 1)

    def test_output_contraction_does_not_contract_other_axes(self):
        from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def reassemble_row_pairs(a, b):
            acc = torch.mm(a, b)
            rows = acc.view(2, 2, 16)
            return torch.cat((rows.select(1, 0), rows.select(1, 1)), dim=0)

        def split_grouped_view(a, b):
            acc = torch.mm(a, b)
            chunks = acc.view(4, 8, 2).split(1, dim=1)
            pair = (chunks[0] - chunks[1]).reshape(4, 2)
            return pair.repeat(1, 8)

        for body in (reassemble_row_pairs, split_grouped_view):
            with self.subTest(body=body.__name__):
                graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 16))
                analysis = analyze_flex_gemm_epilogue(
                    graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
                )
                self.assertIsNone(analysis.outputs.output_contraction)

    @skipIfNoCuteDSL
    def test_output_contraction_validates_output_layout_and_shape(self):
        from torch._vendor.quack.gemm_act import gemm_act

        def run(b, out, group):
            gemm_act(
                torch.empty(4, 8),
                b,
                None,
                None,
                out,
                None,
                None,
                128,
                128,
                1,
                1,
                tensor_epilogue_fn=lambda acc: acc,
                main_output_transform_group=group,
            )

        with self.assertRaisesRegex(NotImplementedError, "PostAct to be n-major"):
            run(torch.empty(8, 16), torch.empty_strided((4, 8), (1, 4)), 2)
        for group in (2, 4):
            with self.subTest(group=group):
                with self.assertRaisesRegex(RuntimeError, "requires PostAct shape"):
                    run(torch.empty(16, 8), torch.empty(4, 16 // group + 1), group)
        with self.assertRaisesRegex(RuntimeError, "physical N=10 must be divisible"):
            run(torch.empty(10, 8), torch.empty(4, 2), 4)

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
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize(
        "operation,tuned",
        (
            ("silu_mul", False),
            ("sub", False),
            ("uint8_sub", False),
            ("silu_mul", True),
        ),
    )
    def test_mm_output_contraction_compiled_matches_reference(self, operation, tuned):
        m, k, n, group = 64, 64, 64, 2

        def epilogue(acc):
            grouped_values = acc.float().view(m, n, group)
            if operation == "sub":
                gate, up = grouped_values.select(-1, 0), grouped_values.select(-1, 1)
            else:
                gate, up = grouped_values[..., 0], grouped_values[..., 1]
            match operation:
                case "silu_mul":
                    return (torch.nn.functional.silu(gate) * up).to(acc.dtype)
                case "sub":
                    return (gate - up).to(acc.dtype)
                case "uint8_sub":
                    return (gate - up).to(torch.uint8)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        if operation == "uint8_sub":
            a = torch.eye(m, k, device="cuda", dtype=torch.float16)
            columns = torch.arange(group * n, device="cuda")
            pair = (columns // group) % 8
            b = torch.where(columns % group == 0, 3 * pair + 2, pair)
            b = b.expand(k, -1).contiguous().to(torch.float16)
        else:
            a = torch.randn(m, k, device="cuda", dtype=torch.float16)
            b = torch.randn(k, group * n, device="cuda", dtype=torch.float16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = fn(a, b)
        if operation == "uint8_sub":
            self.assertEqual(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
        self.assertEqual(actual.shape, (m, n))
        FileCheck().check("FlexGemmOutputContraction(group=2").check_not(
            "activation="
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_non_aligned_logical_n(self):
        m, k, n = 64, 64, 68

        def epilogue(acc):
            grouped_values = acc.float().view(m, n, 2)
            return (grouped_values[..., 0] - grouped_values[..., 1]).to(acc.dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, epilogue(a @ b), atol=1e-1, rtol=1e-1)
        self.assertEqual(actual.shape, (m, n))
        FileCheck().check("FlexGemmOutputContraction(group=2").check_not(
            "extern_kernels.mm"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize(
        "case",
        (
            ("scalar_expr", "scalar", "scalar_expr", False),
            ("col_pre", "col", "pre", False),
            ("col_expr_chunked", "col", "col_expr", True),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_output_contraction_with_n_invariant_capture(self, case):
        _, capture_kind, capture_use, chunked = case
        m = k = n = 64

        def epilogue(acc, capture):
            value = acc.float()
            if capture_use == "pre":
                value = value * capture
            if chunked:
                lhs, rhs = value.chunk(2, dim=-1)
            else:
                grouped_values = value.view(m, n, 2)
                lhs, rhs = grouped_values[..., 0], grouped_values[..., 1]
            if capture_use == "scalar_expr":
                return ((lhs - rhs) * (capture + 1.0)).to(acc.dtype)
            if capture_use == "col_expr":
                scale = torch.where(capture > 0, -capture, capture + 1.0)
                return ((lhs - rhs) * scale).to(acc.dtype)
            return (lhs - rhs).to(acc.dtype)

        def fn(a, b, capture):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, capture),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = (
            torch.randn(2 * n, k, device="cuda", dtype=torch.float16).t()
            if chunked
            else torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        )
        capture_shape = (1, 1) if capture_kind == "scalar" else (m, 1)
        capture = torch.randn(*capture_shape, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, capture
        )

        torch.testing.assert_close(
            actual, epilogue(a @ b, capture), atol=0.2, rtol=0.05
        )
        FileCheck().check(f"epilogue_arg_kinds=('{capture_kind}',)").check_not(
            "extern_kernels.mm"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "interleaved group 4 is currently validated only on SM100",
    )
    def test_mm_output_contraction_group4(self):
        m, k, n, group = 64, 64, 64, 4

        def epilogue(acc, scale, col):
            grouped_values = acc.float().view(m, n, group)
            return (
                (grouped_values[..., 0] - grouped_values[..., 1]) * scale
                + grouped_values[..., 2] * col
                - grouped_values[..., 3]
            ).to(acc.dtype)

        def fn(a, b, scale, col):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, scale, col),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, group * n, device="cuda", dtype=torch.float16)
        scale = torch.randn(1, 1, device="cuda", dtype=torch.float32)
        col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, scale, col
        )

        torch.testing.assert_close(
            actual, epilogue(a @ b, scale, col), atol=2e-1, rtol=5e-2
        )
        self.assertEqual(actual.shape, (m, n))
        FileCheck().check("epilogue_arg_kinds=('scalar', 'col')").check(
            "FlexGemmOutputContraction(group=4, chunked=False)"
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize("chunked", (False, True))
    def test_mm_output_contraction_dynamic_m(self, chunked):
        k = n = 64

        def epilogue(acc, col):
            value = acc.float()
            if chunked:
                gate, up = value.chunk(2, dim=-1)
            else:
                grouped_values = value.view(-1, n, 2)
                gate, up = grouped_values[..., 0], grouped_values[..., 1]
            return (torch.nn.functional.silu(gate) * up * col).to(acc.dtype)

        def fn(a, b, col):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, col),
                kernel_options={"backend": "QUACK"},
            )

        b = torch.randn(2 * n, k, device="cuda", dtype=torch.float16)
        b = b.t() if chunked else b.t().contiguous()
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        for index, m in enumerate((64, 128)):
            a = torch.randn(m, k, device="cuda", dtype=torch.float16)
            col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
            if index == 0:
                torch._dynamo.mark_dynamic(a, 0)
                torch._dynamo.mark_dynamic(col, 0)
            torch.testing.assert_close(
                compiled(a, b, col), epilogue(a @ b, col), atol=1e-1, rtol=1e-1
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize("chunked", (False, True))
    def test_mm_output_contraction_partial_n_tile(self, chunked):
        m = k = 64
        n = 80

        def epilogue(acc):
            value = acc.float()
            if chunked:
                lhs, rhs = value.chunk(2, dim=-1)
            else:
                grouped_values = value.view(m, n, 2)
                lhs, rhs = grouped_values[..., 0], grouped_values[..., 1]
            return (lhs - rhs).to(acc.dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={
                    "backend": "QUACK",
                    "config": {"tile_n": 128, "cluster_n": 1, "swap_ab": False},
                },
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = (
            torch.randn(2 * n, k, device="cuda", dtype=torch.float16).t()
            if chunked
            else torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        )
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        torch.testing.assert_close(actual, epilogue(a @ b), atol=0.2, rtol=0.05)
        FileCheck().check("FlexGemmOutputContraction(group=2").check(
            "('tile_n', 128)"
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_concat_layout(self):
        m, k, n = 64, 64, 64

        def epilogue(acc):
            grouped_values = acc.float().view(m, 2, n)
            gate, up = grouped_values.select(-2, -1), grouped_values.select(-2, -2)
            return (torch.nn.functional.silu(gate) * up).to(acc.dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(2 * n, k, device="cuda", dtype=torch.float16).t()
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        torch.testing.assert_close(actual, fn(a, b), atol=2e-1, rtol=2e-2)
        FileCheck().check("FlexGemmOutputContraction(group=2").check(
            "chunked=True"
        ).check_not("activation=").check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_specializes_split_size(self):
        m = k = split_size = 64

        def epilogue(acc):
            grouped_values = acc.float().split(split_size, dim=-1)
            return (grouped_values[0] - grouped_values[1]).to(acc.dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(2 * split_size, k, device="cuda", dtype=torch.float16).t()
        torch._dynamo.mark_static(b, 1)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        torch.testing.assert_close(actual, epilogue(a @ b), atol=0.2, rtol=0.05)
        FileCheck().check("FlexGemmOutputContraction(group=2").check(
            "chunked=True"
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_supports_clustered_m_configs(self):
        from torch._vendor.quack.gemm_config import GemmConfig

        m, k, n, group = 256, 64, 128, 2
        tile_m = 256

        def epilogue(acc, col):
            grouped_values = acc.float().view(m, n, group)
            return (
                torch.nn.functional.silu(grouped_values[..., 0])
                * grouped_values[..., 1]
                * col
            ).to(acc.dtype)

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=tile_m,
                tile_n=256,
                pingpong=False,
                cluster_m=2,
                cluster_n=1,
                device_capacity=10,
            )
        )

        def fn(a, b, col):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, col),
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, group * n, device="cuda", dtype=torch.bfloat16)
        col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, col
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue(a @ b, col),
            epilogue(a.double() @ b.double(), col.double()),
            k,
        )
        FileCheck().check("epilogue_arg_kinds=('col',)").check(
            "FlexGemmOutputContraction(group=2"
        ).check(f"('tile_m', {tile_m})").check("('cluster_m', 2)").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_output_contraction_filters_unsafe_configs(self):
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_from_key,
            gemm_config_key,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmOutputContraction,
            output_contraction_config_supported,
        )
        from torch._inductor.kernel.flex_gemm.lowering import flex_gemm_config_keys

        device = torch.device("cuda")
        candidates = candidate_gemm_configs_for_device(device)
        contraction = FlexGemmOutputContraction(group=2)
        configs = tuple(
            gemm_config_from_key(key)
            for key in flex_gemm_config_keys(
                device, 64, 128, (), tuned=True, output_contraction=contraction
            )
        )
        self.assertTrue(configs)
        self.assertTrue(
            all(output_contraction_config_supported(config, 128) for config in configs)
        )

        unsafe = next(
            config
            for config in candidates
            if not config.swap_ab
            and not output_contraction_config_supported(config, 128)
        )
        with self.assertRaisesRegex(
            NotImplementedError, "incompatible with output contraction"
        ):
            flex_gemm_config_keys(
                device,
                64,
                128,
                (),
                tuned=True,
                output_contraction=contraction,
                explicit_config=dict(gemm_config_key(unsafe)),
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_output_contraction_rejects_unsupported_composition(self):
        m = k = n = 64
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)

        def group_eight(acc):
            grouped_values = acc.view(m, n, 8)
            return sum(grouped_values[..., index] for index in range(8))

        self.assertOutputContractionRejected(
            a,
            torch.randn(k, 8 * n, device="cuda", dtype=torch.float16),
            group_eight,
            "group 2.*group 4",
        )

        scale = torch.randn(1, 2 * n, device="cuda", dtype=torch.float16)

        def n_varying_capture(acc):
            grouped_values = (acc * scale).view(m, n, 2)
            return grouped_values[..., 0] - grouped_values[..., 1]

        b = torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        self.assertOutputContractionRejected(
            a, b, n_varying_capture, "numeric.*captured tensors"
        )

        boolean_scale = torch.ones(1, 1, device="cuda", dtype=torch.bool)

        def boolean_capture(acc):
            grouped_values = (acc * boolean_scale).view(m, n, 2)
            return grouped_values[..., 0] - grouped_values[..., 1]

        self.assertOutputContractionRejected(
            a, b, boolean_capture, "numeric.*captured tensors"
        )

        def chunked(acc):
            gate, up = acc.chunk(2, dim=-1)
            return gate - up

        self.assertOutputContractionRejected(a, b, chunked, "non-contiguous")

        def dynamic_n(acc):
            grouped_values = acc.view(m, -1, 2)
            return grouped_values[..., 0] - grouped_values[..., 1]

        dynamic_b = torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        torch._dynamo.mark_dynamic(dynamic_b, 1)
        self.assertOutputContractionRejected(
            a, dynamic_b, dynamic_n, "statically known physical N"
        )

        def auxiliary(acc):
            grouped_values = acc.view(m, n, 2)
            return grouped_values[..., 0] - grouped_values[..., 1], acc

        self.assertOutputContractionRejected(a, b, auxiliary, "do not compose")

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    def test_mm_chunked_output_contraction_rejects_grouped_reduction(self):
        m, k, physical_n, group = 128, 64, 256, 32

        def epilogue_fn(acc):
            value = acc.float()
            grouped = value.view(m, -1, group)
            scale = grouped.abs().amax(-1, keepdim=True)
            normalized = (grouped / scale).view(m, physical_n)
            gate, up = normalized.chunk(2, dim=-1)
            return (gate - up).to(acc.dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(physical_n, k, device="cuda", dtype=torch.bfloat16).t()
        with self.assertRaisesRegex(
            InductorError,
            "concat-layout output contractions do not compose with grouped reductions",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

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
    def test_mm_tuple_aux_physical_local_reduce_supports_finalize_expression(self):
        m = 128
        n = 128
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), (x.abs().amax(1) + 1.0).sqrt()

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
        FileCheck().check("cutlass.max").check("cute.math.sqrt").check(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("wrapper", ("squeeze", "view"))
    def test_mm_tuple_aux_physical_local_reduce_supports_wrapped_finalize(
        self, wrapper
    ):
        m = 128
        n = 128
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            reduced = x.sum(-1, keepdim=True)
            reduced = (
                reduced.squeeze(-1)
                if wrapper == "squeeze"
                else reduced.view(m, n // group)
            )
            return acc.relu(), reduced + 1.0

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
        FileCheck().check("local_reduce_combine_fn").check(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_rejects_composite_physical_local_reductions(self):
        m = 128
        n = 128
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1) + x.amax(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "single physical local reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_nvfp4_pack_matches_reference(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
        )
        lower = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
        upper = torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf))
        smallest = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
        special = torch.tensor(
            [
                -torch.finfo(torch.float32).max,
                -torch.finfo(torch.float32).tiny,
                -smallest,
                -0.0,
                0.0,
                smallest,
                torch.finfo(torch.float32).tiny,
                torch.finfo(torch.float32).max,
                -torch.inf,
                torch.inf,
            ]
        )
        random = torch.empty(4096).uniform_(
            -8.0, 8.0, generator=torch.Generator().manual_seed(0)
        )
        values = torch.cat(
            (-upper, -boundaries, -lower, special, lower, boundaries, upper, random)
        ).view(-1, 2)
        inputs = (values, values[:8].view(2, 4, 2).permute(1, 0, 2))
        for input in inputs:
            with self.subTest(stride=input.stride()):
                expected = pack_uint4(_f32_to_floatx_unpacked(input, 2, 1)).squeeze(-1)
                actual = nvfp4_pack(input)
                self.assertIs(actual.dtype, torch.float4_e2m1fn_x2)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.view(torch.uint8), expected)

                with FakeTensorMode() as mode:
                    fake = nvfp4_pack(mode.from_tensor(input))
                self.assertIs(fake.dtype, torch.float4_e2m1fn_x2)
                self.assertEqual(fake.shape, actual.shape)
                self.assertEqual(fake.stride(), actual.stride())

    def test_quant_scale_fake_strides_match_eager(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        amax = torch.rand(2, 1).expand(2, 4)
        eager_stride = mx_e8m0_scale(amax).stride()
        with FakeTensorMode() as mode:
            fake_amax = mode.from_tensor(amax)
        self.assertEqual(mx_e8m0_scale(fake_amax).stride(), eager_stride)

    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    def test_quant_scale_rounding_args(self):
        amax = torch.tensor(
            [0.0, 448.0, 449.0, 500.0, 511.0, 512.0, float("inf"), float("nan")],
            device="cuda",
        )
        floor = mx_e8m0_scale(amax, rounding="floor").view(torch.uint8)
        rceil = mx_e8m0_scale(amax, rounding="rceil").view(torch.uint8)
        default = mx_e8m0_scale(amax).view(torch.uint8)

        self.assertEqual(
            floor,
            torch.tensor(
                [0, 127, 127, 127, 127, 128, 247, 255],
                device="cuda",
                dtype=torch.uint8,
            ),
        )
        self.assertEqual(
            rceil,
            torch.tensor(
                [0, 127, 128, 128, 128, 128, 254, 255],
                device="cuda",
                dtype=torch.uint8,
            ),
        )
        self.assertEqual(default, rceil)

        descale_bits = torch.tensor(
            [
                0x00000000,
                0x00000001,
                0x003FFFFF,
                0x00400000,
                0x00400001,
                0x007FFFFF,
                0x00800000,
                0x3F000000,
                0x3F800000,
                0x3F800001,
                0x7F7FFFFF,
                0x7F800000,
                0x7FC00000,
            ],
            device="cuda",
            dtype=torch.int32,
        )
        self.assertEqual(
            mx_e8m0_scale(
                descale_bits.view(torch.float32) * 448.0, rounding="rceil"
            ).view(torch.uint8),
            torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 126, 127, 128, 254, 254, 255],
                device="cuda",
                dtype=torch.uint8,
            ),
        )

        for max_value in (6.0, 57344.0):
            format_amax = torch.tensor(
                [
                    max_value,
                    torch.nextafter(
                        torch.tensor(max_value), torch.tensor(float("inf"))
                    ),
                ],
                device="cuda",
            )
            self.assertEqual(
                mx_e8m0_scale(format_amax, max_value=max_value, rounding="floor").view(
                    torch.uint8
                ),
                torch.tensor([127, 127], device="cuda", dtype=torch.uint8),
            )
            self.assertEqual(
                mx_e8m0_scale(format_amax, max_value=max_value, rounding="rceil").view(
                    torch.uint8
                ),
                torch.tensor([127, 128], device="cuda", dtype=torch.uint8),
            )

        with self.assertRaisesRegex(ValueError, "rounding must be 'floor', 'rceil'"):
            mx_e8m0_scale(amax, rounding="nearest")
        with self.assertRaisesRegex(ValueError, "pack must be 1 or 2"):
            mx_e8m0_scale(amax, pack=4)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("rceil", 448.0, 450.0, 128),
            ("floor", 448.0, 450.0, 127),
            ("rceil", 448.0, 2.0**-140, 0),
            ("rceil", 448.0, float("inf"), 254),
            ("floor", 448.0, float("inf"), 247),
            ("floor", 2.0**-122, 64.0, 255),
        ),
        name_fn=lambda case: f"{case[0]}_max{case[1]:g}_code{case[3]}",
    )
    def test_mm_tuple_aux_mx_scale_rounding(self, case):
        rounding, max_value, value, expected_code = case
        code_check = (
            "cvt.rz.ue8m0x2.f32"
            if rounding == "floor"
            else "cvt.rp.satfinite.ue8m0x2.f32"
        )
        m = 64
        n = 128
        k = 64
        group = 32

        def epilogue_fn(acc):
            x = (acc.float() + value).view(m, -1, group)
            return acc, mx_e8m0_scale(
                x.abs().amax(-1), max_value=max_value, rounding=rounding
            )

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(
            aux.view(torch.uint8),
            torch.full_like(aux.view(torch.uint8), expected_code),
        )
        FileCheck().check(code_check).check_not("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_inline_asm_uses_compact_reduction(self):
        m, n, k, group = 64, 96, 64, 32

        def epilogue_fn(acc):
            reduced = acc.float().view(m, -1, group).amax(-1)
            code = inline_asm_elementwise(
                reduced,
                asm_str="cvt.rzi.u16.f32 $0, $1;",
                constraints="=h,f",
                dtype=torch.uint8,
            )
            return acc, code.float()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = a @ b
        expected_aux = (
            expected.float().view(m, -1, group).amax(-1).to(torch.uint8).float()
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aux, expected_aux)
        FileCheck().check("inline_asm_elementwise_intrinsic(").check_not(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_inline_asm_pack2_broadcasts_and_repeats_scalar(self):
        m = k = n = 64

        def fn(a, b, row, col, scalar):
            def epilogue_fn(acc):
                return inline_asm_elementwise(
                    row,
                    col,
                    scalar,
                    acc.float(),
                    asm_str=(
                        "add.f32 $0, $2, $4; add.f32 $0, $0, $6; "
                        "add.f32 $0, $0, $8; add.f32 $1, $3, $5; "
                        "add.f32 $1, $1, $7; add.f32 $1, $1, $9;"
                    ),
                    constraints="=f,=f,f,f,f,f,f,f,f,f",
                    dtype=torch.float32,
                    pack=2,
                )

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        row = torch.randn(1, n, device="cuda", dtype=torch.float32)
        col = torch.randn(m, 1, device="cuda", dtype=torch.float32)
        scalar = torch.randn(1, 1, device="cuda", dtype=torch.float32)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            row,
            col,
            scalar,
        )

        torch.testing.assert_close(actual, row + col + scalar)
        FileCheck().check("pack=2").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        ((torch.float16, "Float16"), (torch.bfloat16, "BFloat16")),
        name_fn=lambda case: str(case[0]).removeprefix("torch."),
    )
    def test_mm_inline_asm_pack2_restores_16bit_inputs(self, case):
        dtype, cute_type = case
        m = k = n = 64

        def fn(a, b, row):
            def epilogue_fn(acc):
                scalar = torch.full((), 0.5, device="cuda", dtype=dtype)
                return inline_asm_elementwise(
                    row,
                    scalar,
                    acc.float(),
                    asm_str="mov.b16 $0, $2; mov.b16 $1, $3;",
                    constraints="=h,=h,h,h,h,h,f,f",
                    dtype=dtype,
                    pack=2,
                )

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        row = torch.randn(1, n, device="cuda", dtype=dtype)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, row
        )

        self.assertEqual(actual, row.expand(m, n))
        FileCheck().check(f".to(cutlass.{cute_type})").check("pack=2").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("nonpositive_pack", 0, "=f,f", "requires pack >= 1"),
            ("output_count", 2, "=f,f,f", "requires 2 output constraints"),
            ("input_count", 2, "=f,=f,f", "requires 2 input constraints"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_inline_asm_rejects_invalid_pack_contract(self, case):
        _, pack, constraints, error = case

        def fn(a, b):
            def epilogue_fn(acc):
                return inline_asm_elementwise(
                    acc.float(),
                    asm_str="mov.f32 $0, $1;",
                    constraints=constraints,
                    dtype=torch.float32,
                    pack=pack,
                )

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex((ValueError, InductorError), error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("mx", mx_e8m0_scale, "cvt.rp.satfinite.ue8m0x2.f32"),
            ("nvfp4", nvfp4_e4m3_scale, " / 6.0"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_large_n_local_reduce_supports_scale_finalizer(self, case):
        _, scale_fn, code_check = case
        m = 64
        n = 128
        k = 64
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), scale_fn(x.abs().amax(-1))

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(aux.float(), expected_aux.float())
        FileCheck().check(code_check).check("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        ((16, 128), (32, 96)),
        name_fn=lambda case: f"group{case[0]}_n{case[1]}",
    )
    def test_mm_tuple_aux_mx_scale_pack2(self, case):
        group, n = case
        m = k = 16

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), mx_e8m0_scale(x.abs().amax(-1), pack=2)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[:, group : 2 * group] = float("inf")
        b[:, 2 * group : 3 * group] = float("nan")
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        non_nan = ~torch.isnan(expected)
        torch.testing.assert_close(actual[non_nan], expected[non_nan])
        self.assertEqual(aux.view(torch.uint8), expected_aux.view(torch.uint8))
        for expected_code in (0, 254, 255):
            self.assertTrue((aux.view(torch.uint8) == expected_code).any())
        FileCheck().check("pack=2").check("broadcast_to").check_not(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_mx_scale_preserves_nan(self):
        m = k = 16
        n = 128
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), mx_e8m0_scale(x.abs().amax(-1))

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("nan")
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        self.assertEqual(aux.view(torch.uint8), expected_aux.view(torch.uint8))
        self.assertTrue((aux.view(torch.uint8) == 0).any())
        self.assertTrue((aux.view(torch.uint8) == 255).any())
        FileCheck().check("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_mx_scale_zero_preserves_min_value(self):
        m = 16
        n = 128
        k = 16
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc, mx_e8m0_scale(x.abs().amax(-1)).float() * 2

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
        self.assertEqual(aux, torch.full_like(aux, 2.0**-126))
        FileCheck().check("local_reduce_finalize_fn").run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_nvfp4_scale_rounds_before_pointwise(self):
        m = 16
        n = 128
        k = 64
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc, nvfp4_e4m3_scale(x.abs().amax(-1)).float() * 2

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(aux, expected_aux, rtol=0, atol=0)
        FileCheck().check("cutlass.Float8E4M3FN").check("local_reduce_finalize_fn").run(
            code
        )
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("reduction", ("amax", "amin"))
    def test_mm_tuple_aux_physical_extrema_preserve_nan(self, reduction):
        m = 16
        n = 128
        k = 16
        group = 64

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc, getattr(x, reduction)(-1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("nan")
        (_, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        _, expected_aux = epilogue_fn(a @ b)
        torch.testing.assert_close(aux, expected_aux, equal_nan=True)
        self.assertTrue(torch.isnan(aux[:, 0]).all())
        self.assertEqual(aux[:, 1], torch.zeros_like(aux[:, 1]))
        FileCheck().check(f"cutlass.{reduction[1:]}").check(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("case", ("clamp", "clamp_min", "clamp_max"))
    def test_mm_epilogue_clamp_preserves_nan(self, case):
        m = n = k = 16

        def epilogue_fn(acc):
            x = acc.float() + float("nan")
            match case:
                case "clamp":
                    return x.clamp(-1.0, 1.0)
                case "clamp_min":
                    return x.clamp_min(-1.0)
                case "clamp_max":
                    return x.clamp_max(1.0)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertTrue(torch.isnan(actual).all())
        check = FileCheck().check("cutlass_math.")
        match case:
            case "clamp":
                check = check.check("max").check("cutlass_math.min")
            case "clamp_min":
                check = check.check("max")
            case "clamp_max":
                check = check.check("min")
        check.check_not("operator.ne").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_fragment_group_mx_quant_feed_uses_tensorSSA(self):
        m = 64
        n = 128
        k = 64
        group = 32

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
            return (
                (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn),
                scale.squeeze(-1),
            )

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        b[0, 0] = float("nan")
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        x = (a @ b).float().view(m, -1, group)
        expected_aux = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
        expected = (
            (x * expected_aux.float().reciprocal()).view(m, n).to(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(actual.float(), expected.float(), equal_nan=True)
        self.assertEqual(
            aux.view(torch.uint8), expected_aux.squeeze(-1).view(torch.uint8)
        )
        self.assertTrue((aux.view(torch.uint8) == 0).any())
        self.assertTrue((aux.view(torch.uint8) == 255).any())
        FileCheck().check("cvt.rp.satfinite.ue8m0x2.f32").check_not(
            "local_reduce_finalize_fn"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (64, 128))
    @parametrize(
        "case",
        (
            ("sum", lambda x: x.sum(1), "local_reduce_combine_fn"),
            ("mean", lambda x: x.mean(1), " / {group}.0"),
            ("amax", lambda x: x.amax(1), "cutlass.max"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_m_reduce_supports_cta_group(self, case, group):
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
    def test_mm_tuple_aux_local_m_reduce_tuned_internal_transpose(self):
        m = 128
        n = 192
        k = 64
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b.mT),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_flex_gemm_local_reduce_config,
        )
        from torch._inductor.template_heuristics import (
            flex_gemm as flex_gemm_heuristics,
        )

        configs = tuple(
            config
            for config in flex_gemm_heuristics.candidate_gemm_configs_for_device(
                a.device
            )
            if validate_flex_gemm_local_reduce_config(config, group, 0)
        )[:2]
        self.assertEqual(len(configs), 2)
        with mock.patch(
            "torch._inductor.heuristics.template.flex_gemm.candidate_gemm_configs_for_device",
            return_value=configs,
        ):
            actual, aux = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertLocalReduceAuxMatches(actual, aux, a, b.mT, epilogue_fn)

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
            ("sum_group64", 128, 64, lambda x: x.sum(-1), "lhs + rhs"),
            ("mean_group64", 128, 64, lambda x: x.mean(-1), " / 64.0"),
            ("amax_group64", 128, 64, lambda x: x.amax(-1), "cutlass.max"),
            ("sum_group128", 256, 128, lambda x: x.sum(-1), "lhs + rhs"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_n_reduce_supports_cta_subtile_group(self, case):
        _, n, group, reduce_fn, code_check = case
        m = 128

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
        self.assertIn(code_check, code)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)
        self.assertNotIn("local_reduce_strategy", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_local_n_reduce_cta_subtile_group_tuned(self):
        m = 128
        n = 256
        group = 128

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), x.sum(-1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b)[0],
            high_precision_acc.relu(),
            a.shape[1],
        )
        torch.testing.assert_close(
            aux, epilogue_fn(high_precision_acc)[1].float(), atol=1e-3, rtol=1e-3
        )
        self.assertLocalReduceAuxCode(code, group, callbacks=True)
        self.assertIn("cluster_n', 1", code)

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
    def test_local_n_reduce_group32_filters_half_fragment_configs(self):
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
        )
        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_flex_gemm_local_reduce_config,
        )

        candidates = candidate_gemm_configs_for_device(torch.device("cuda"))
        half_fragment_configs = [
            config
            for config in candidates
            if config.tile_m == 128
            and config.tile_n in (128, 160, 224)
            and config.cluster_m > 1
            and validate_flex_gemm_local_reduce_config(config, 16, 1)
        ]
        self.assertTrue(half_fragment_configs)
        self.assertTrue(
            all(
                not validate_flex_gemm_local_reduce_config(config, 32, 1)
                for config in half_fragment_configs
            )
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_reduce_result_feeds_main_output(self):
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = x.sum(-1, keepdim=True) + 1.0
            return (x * scale).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(high_precision_acc),
            a.shape[1],
        )
        self.assertNotIn("local_reduce_feeds_main=True", code)
        self.assertNotIn("local_reduce_combine_fn", code)
        FileCheck().check("cute.ReductionOp.ADD").check_not("local_reduce_out=").run(
            code
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (2, 4, 8, 16, 32))
    def test_mm_local_m_reduce_result_feeds_main_output(self, group):
        m = 128
        n = 64

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            epilogue_fn(high_precision_acc),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feeds_main_and_returns_scale(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n), scale.squeeze(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, scale), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_scale = epilogue_fn(a @ b)
        high_precision_expected, high_precision_scale = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        torch.testing.assert_close(
            scale,
            high_precision_scale.float(),
            atol=1e-3,
            rtol=1e-3,
        )
        FileCheck().check("local_reduce=FlexGemmRuntimeLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 0)
        ).check("out=").check("feeds_main=True").check(
            "callbacks=FlexGemmLocalReduceCallbacks"
        ).check_not("local_reduce_group=").check_not("local_reduce_op").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_result_divides_main_output(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            return (x / scale).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected = epilogue_fn(a @ b)
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            epilogue_fn(high_precision_acc),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_reversed_division(self, consumer):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            transformed = (scale / x).view(m, n)
            if consumer == "main":
                return transformed
            return acc.relu(), transformed

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_post_scale_pointwise(self, consumer):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            normalized = (x / scale + 0.5).view(m, n)
            if consumer == "main":
                return normalized
            return acc.relu(), normalized

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_centered_mean(self, consumer):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            mean = x.mean(1, keepdim=True)
            centered = (x - mean + 0.5).view(m, n)
            if consumer == "main":
                return centered
            return acc.relu(), centered

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "op",
        (
            "add",
            "sub",
            "method_add",
            "method_sub",
            "method_add_alpha",
            "method_sub_alpha",
            "torch_add_alpha",
            "torch_sub_alpha",
        ),
    )
    @parametrize("consumer", ("main", "aux"))
    def test_mm_local_m_reduce_feed_main_supports_reversed_add_sub(self, op, consumer):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            mean = x.mean(1, keepdim=True)
            if op == "add":
                transformed = (mean + x + 0.5).view(m, n)
            elif op == "sub":
                transformed = (mean - x + 0.5).view(m, n)
            elif op == "method_add":
                transformed = mean.add(x).add(0.5).view(m, n)
            elif op == "method_sub":
                transformed = mean.sub(x).add(0.5).view(m, n)
            elif op == "method_add_alpha":
                transformed = mean.add(x, alpha=0.25).add(0.5).view(m, n)
            elif op == "method_sub_alpha":
                transformed = mean.sub(x, alpha=0.25).add(0.5).view(m, n)
            elif op == "torch_add_alpha":
                transformed = torch.add(mean, x, alpha=0.25).add(0.5).view(m, n)
            else:
                transformed = torch.sub(mean, x, alpha=0.25).add(0.5).view(m, n)
            if consumer == "main":
                return transformed
            return acc.relu(), transformed

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_pointwise_wrapper_rejects_two_values(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            sum_scale = x.sum(1, keepdim=True) + 1.0
            max_scale = x.amax(1, keepdim=True) + 1.0
            return (x / sum_scale + x / max_scale).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("cross_warp", 128, 64, "same-warp axis-0 groups <= 32"),
            ("boundary", 136, 17, "fragment width 32"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_local_m_reduce_feed_main_rejects_unsupported_group(self, case):
        # Preserve each case's static geometry instead of inheriting PGO state
        # from the other parameterization.
        torch._dynamo.reset()
        _, m, group, error = case
        n = 64

        def epilogue_fn(acc, group=group, m=m, n=n):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, error):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_specializes_explicit_group(self):
        torch._dynamo.reset()
        m = 128
        n = 64

        def epilogue_fn(acc, group):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + 1.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b, group):
            def epilogue(acc):
                return epilogue_fn(acc, group)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(fn, backend=counter, fullgraph=True)
        for group in (8, 16, 32):
            actual = compiled(a, b, group)
            self.assertMatchesEpilogue(
                actual,
                epilogue_fn(a @ b, group),
                epilogue_fn(a.double() @ b.double(), group),
                a.shape[1],
            )
        self.assertEqual(counter.frame_count, 3)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_same_shape_uses_broadcast_local_reduce(self):
        m = 128
        n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            aux = (x * (x.mean(-1, keepdim=True) + 1.0)).view(m, n)
            return acc.relu(), aux

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

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cute.ReductionOp.ADD").check(" / 16.0").check_not(
            "local_reduce_out="
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_uses_physical_m_broadcast_local_reduce(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            aux = (x * (x.sum(1, keepdim=True) + 1.0).reciprocal()).view(m, n)
            return acc.relu(), aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_divides_by_physical_m_broadcast_local_reduce(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            aux = (x / (x.sum(1, keepdim=True) + 1.0)).view(m, n)
            return acc.relu(), aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_outputs_share_physical_m_feed_main_reduce(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            actual = (x * scale.reciprocal()).view(m, n)
            aux = (x * (scale + 1.0)).view(m, n)
            return actual, aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            (actual, aux),
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_repeated_equivalent_reductions(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x0 = acc.float().view(-1, group, n)
            out = (x0 * x0.sum(1, keepdim=True).reciprocal()).view(m, n)
            x1 = acc.float().view(-1, group, n)
            aux = (x1 * (x1.sum(1, keepdim=True) + 1.0)).view(m, n)
            return out, aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_regrouped_reduction_reuse(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x0 = acc.float().view(-1, group, n)
            scale = x0.sum(1, keepdim=True)
            out = (x0 * scale.reciprocal()).view(m, n)
            x1 = x0.view(m, n).view(-1, group, n)
            aux = (x1 * scale.reciprocal()).view(m, n)
            return out, aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected, expected_aux = epilogue_fn(a @ b)
        high_precision_expected, high_precision_aux = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        self.assertMatchesLowPrecisionEager(
            aux,
            expected_aux,
            high_precision_aux,
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_mixed_grouped_layouts(self):
        m = 128
        n = 64

        def epilogue_fn(acc):
            x8 = acc.float().view(-1, 8, n)
            out = (x8 * x8.sum(1, keepdim=True).reciprocal()).view(m, n)
            x4 = acc.float().view(-1, 4, n)
            aux = (x4 * x4.sum(1, keepdim=True).reciprocal()).view(m, n)
            return out, aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "share one grouped layout"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_composite_physical_reductions(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + x.amax(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_unselected_physical_reduction(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True) + x.square().sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "two-phase local-reduce source"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_rejects_hidden_physical_reduction(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            y = acc.float().relu().view(-1, group, n)
            scaled = x / (x.sum(1, keepdim=True) + 1.0)
            return (scaled + y.sum(1, keepdim=True)).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "one generated physical reduction"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        ("rms_main", "rms_aux", "centered"),
    )
    def test_mm_local_m_reduce_feed_main_rejects_source_expression(self, case):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            if case == "centered":
                mean = x.mean(1, keepdim=True)
                rstd = (x.square().mean(1, keepdim=True) + 1e-5).rsqrt()
                return ((x - mean) * rstd).view(m, n)
            rstd = (x.square().mean(1, keepdim=True) + 1e-5).rsqrt()
            normalized = (x * rstd).view(m, n)
            if case == "rms_main":
                return normalized
            return acc.relu(), normalized

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "two-phase local-reduce source"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_trailing_fp8_quant(self):
        m = 128
        n = 64
        group = 32

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            x = acc.view(-1, group, n)
            return (x * x.sum(1, keepdim=True).reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)
        FileCheck().check("cutlass.Float8E4M3FN").check("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_mx_scale(self):
        m = 128
        n = 64
        group = 32

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = mx_e8m0_scale(x.amax(1, keepdim=True)).float()
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            x = acc.view(-1, group, n)
            scale = mx_e8m0_scale(x.amax(1, keepdim=True)).float()
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)
        FileCheck().check("cvt.rp.satfinite.ue8m0x2.f32").check("feeds_main=True").run(
            code
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_trailing_pointwise_chain(self):
        m = 128
        n = 64
        group = 8

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n) * 0.5 + 1.0

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesEpilogue(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertPhysicalFeedMainCode(code, group)
        FileCheck().check("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_trailing_quant_with_scale_store(self):
        m = 128
        n = 64
        group = 32

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            scale = x.sum(1, keepdim=True)
            quant = (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)
            return quant, scale.squeeze(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        high_precision_acc = a.double() @ b.double()
        high_precision_x = high_precision_acc.view(-1, group, n)
        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b)[0],
            (
                high_precision_x * high_precision_x.sum(1, keepdim=True).reciprocal()
            ).view(m, n),
            a.shape[1],
        )
        torch.testing.assert_close(
            aux,
            high_precision_x.sum(1).float(),
            atol=1e-3,
            rtol=1e-3,
        )
        FileCheck().check("local_reduce=FlexGemmRuntimeLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 0)
        ).check("out=").check("feeds_main=True").check(
            "callbacks=FlexGemmLocalReduceCallbacks"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (4, 8, 16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_sum(self, group):
        m = 128
        n = 128

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = x.sum(-1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cute.ReductionOp.ADD").check_not(
            "local_reduce=FlexGemmRuntimeLocalReducePlan"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (4, 8, 16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_fp8_quant(self, group):
        m = 128
        n = 128

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            x = acc.view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cutlass.Float8E4M3FN").check_not(
            "local_reduce=FlexGemmRuntimeLocalReducePlan"
        ).run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (4, 8, 16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_scale_store(self, group):
        m = 128
        n = 128

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n), scale.squeeze(-1)

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
        self.assertLocalReduceAuxCode(code, group)
        FileCheck().check_not("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (16, 32))
    def test_mm_local_n_reduce_feed_main_fragment_group_tuned(self, group):
        m = 128
        n = 128

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n).to(torch.float8_e4m3fn)

        def high_precision_fn(acc):
            x = acc.view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check("cutlass.Float8E4M3FN").check_not(
            "local_reduce=FlexGemmRuntimeLocalReducePlan"
        ).run(code)
        # Plan-less grouped fragments must gate tuned candidates like a plan:
        # swap_ab configs would regroup the wrong accumulator elements.
        self.assertIn("swap_ab', False", code)
        self.assertNotIn("swap_ab', True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_n_reduce_feed_main_fragment_group_skinny_default_config(self):
        m = 128
        n = 4096
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = x.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 448.0
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        FileCheck().check_not("local_reduce=FlexGemmRuntimeLocalReducePlan").run(code)
        # The skinny-shape default config must satisfy the same grouped-fragment
        # gate as tuned candidates even without a runtime local-reduce plan.
        self.assertIn("swap_ab', False", code)
        self.assertNotIn("swap_ab', True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (64, 128))
    def test_mm_local_n_reduce_feed_main_rejects_multi_fragment_group(self, group):
        torch._dynamo.reset()
        m = 128
        n = 256

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = x.sum(-1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            Exception, "axis-1 groups larger than one TensorSSA fragment"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("fragment_group_tuned", 128, 32, True),
            ("multi_chunk_large_group", 256, 128, False),
            ("tile_n_group", 256, 256, False),
            ("wide_m_tile_n_group", 512, 512, False),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_coda_rmsnorm_rewrite_e2e(self, case):
        _, n, group, tuned = case
        m, k, p = 64, 32, 48
        eps = 1e-5

        def fn(a, b1, gamma, b2):
            def first_epilogue(acc):
                x = acc.float().view(m, -1, group)
                h2 = (acc.float() * gamma).to(torch.bfloat16)
                partial_mean_square = x.square().mean(-1)
                return h2, partial_mean_square

            h2, partial_mean_square = flex_gemm(
                torch.mm,
                (a, b1),
                first_epilogue,
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )
            rstd = (partial_mean_square.mean(-1, keepdim=True) + eps).rsqrt()

            def second_epilogue(acc):
                return acc.float() * rstd

            return flex_gemm(
                torch.mm,
                (h2, b2),
                second_epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, k)
        b1 = self.makeTensor(k, n)
        gamma = self.makeTensor(1, n)
        b2 = self.makeTensor(n, p)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b1, gamma, b2
        )

        acc1 = a @ b1
        h2 = (acc1.float() * gamma).to(torch.bfloat16)
        partial_mean_square = acc1.float().view(m, -1, group).square().mean(-1)
        rstd = (partial_mean_square.mean(-1, keepdim=True) + eps).rsqrt()
        expected = (h2 @ b2).float() * rstd

        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertEqual(code.count("flex_gemm_epilogue("), 2)
        self.assertIn("local_reduce=FlexGemmRuntimeLocalReducePlan", code)
        self.assertIn(
            f"FlexGemmLocalReduceGeometry(group={group}, axis=1)",
            code,
        )
        if group > 32:
            self.assertIn("combine_fn=", code)
        self.assertIn("epilogue_arg_kinds=('row',)", code)
        self.assertIn("epilogue_arg_kinds=('col',)", code)
        self.assertNotIn("local_reduce_feeds_main=True", code)
        self.assertNotIn("local_reduce_op", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_coda_rmsnorm_rewrite_rejects_group_above_config_limit(self):
        m, k, n, p = 64, 32, 1024, 48
        group = 1024
        eps = 1e-5

        def fn(a, b1, gamma, b2):
            def first_epilogue(acc):
                x = acc.float().view(m, -1, group)
                h2 = (acc.float() * gamma).to(torch.bfloat16)
                return h2, x.square().mean(-1)

            h2, partial_mean_square = flex_gemm(
                torch.mm,
                (a, b1),
                first_epilogue,
                kernel_options={"backend": "QUACK"},
            )
            rstd = (partial_mean_square.mean(-1, keepdim=True) + eps).rsqrt()

            def second_epilogue(acc):
                return acc.float() * rstd

            return flex_gemm(
                torch.mm,
                (h2, b2),
                second_epilogue,
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, k)
        b1 = self.makeTensor(k, n)
        gamma = self.makeTensor(1, n)
        b2 = self.makeTensor(n, p)

        with self.assertRaisesRegex(
            Exception,
            "requested group=1024, max supported group=512 for axis=1",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b1, gamma, b2)

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
        self.assertNotIn("output_plan=", code)

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
            (
                "invalid_fast_math_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "fast_math": 1},
                "fast_math kernel option must be bool",
            ),
            (
                "invalid_config_option",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "config": None},
                "config kernel option must be a dict",
            ),
            (
                "unknown_config_field",
                lambda acc: acc.relu(),
                {"backend": "QUACK", "config": {"stages": 4}},
                "unexpected GemmConfig fields",
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


@skipIfNoCuteDSL
@unittest.skipIf(not SM100OrLater, "SM100+ required")
class TestFlexGemmNVFP4Device(FlexGemmTestCase):
    @unittest.skipIf(SM120OrLater, "output contractions are not supported on SM120")
    @parametrize("tuned", (False, True))
    def test_mm_nvfp4_pack_matches_reference(self, device, tuned):
        m = n = 128
        k = 64
        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device=device,
            dtype=torch.bfloat16,
        )
        lower = torch.nextafter(boundaries, torch.full_like(boundaries, float("-inf")))
        upper = torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))
        values = torch.cat(
            (
                -upper,
                -boundaries,
                -lower,
                torch.tensor(
                    [float("-inf"), -0.0, 0.0, float("inf")],
                    device=device,
                    dtype=torch.bfloat16,
                ),
                lower,
                boundaries,
                upper,
            )
        ).repeat(4)[:n]
        a = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
        b = torch.zeros(k, n, device=device, dtype=torch.bfloat16)
        a[:, 0] = 1
        b[0] = values

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: nvfp4_pack(acc.float().view(m, -1, 2)),
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = nvfp4_pack((a @ b).float().view(m, -1, 2))
        self.assertIs(actual.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(actual.view(torch.uint8), expected.view(torch.uint8))
        FileCheck().check("inline_asm_elementwise_intrinsic").check(
            "torch.uint8"
        ).check("aten.view.dtype").check("torch.float4_e2m1fn_x2").run(code)


instantiate_device_type_tests(TestFlexGemmNVFP4Device, globals(), only_for="cuda")


@skipIfNoCuteDSL
@unittest.skipIf(not SM100OrLater, "SM100+ required")
class TestFlexGemmFastMathDevice(FlexGemmTestCase):
    def test_mm_fast_math_kernel_option_controls_cutedsl_math(self, device):
        def epilogue_fn(acc):
            magnitude = acc.abs() + 1.0
            return (
                torch.tanh(acc)
                + torch.sigmoid(acc)
                + torch.exp(-acc.abs())
                + torch.log1p(acc.abs())
                + torch.sqrt(magnitude)
                + torch.rsqrt(magnitude)
            )

        def compile_with_fast_math(a, b, fast_math):
            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(fast_result.double(), expected, atol=0.2, rtol=0.02)
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        for op_name in ("tanh", "exp2", "log", "sqrt", "rsqrt"):
            self.assertIn(f"cute.math.{op_name}", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("fastmath=True", precise_code)

    def test_mm_fast_math_sigmoid_uses_tanh_decomposition(self, device):
        def epilogue_fn(acc):
            return torch.sigmoid(acc)

        def compile_with_fast_math(a, b, fast_math):
            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(fast_result.double(), expected, atol=0.2, rtol=0.02)
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        self.assertIn("cute.math.tanh", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("cute.math.exp2", fast_code)
        self.assertIn("cute.math.exp2", precise_code)
        self.assertNotIn("cute.math.tanh", precise_code)

    def test_mm_fast_math_silu_uses_tanh_decomposition(self, device):
        def epilogue_fn(acc):
            return torch.nn.functional.silu(acc)

        def compile_with_fast_math(a, b, fast_math):
            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(fast_result.double(), expected, atol=0.2, rtol=0.02)
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        self.assertIn("cute.math.tanh", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("cute.math.exp2", fast_code)
        self.assertIn("cute.math.exp2", precise_code)
        self.assertNotIn("cute.math.tanh", precise_code)

    def test_mm_fast_math_gelu_none_uses_tanh_decomposition(self, device):
        def epilogue_fn(acc):
            return torch.nn.functional.gelu(acc, approximate="none")

        def compile_with_fast_math(a, b, fast_math):
            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK", "fast_math": fast_math},
                )

            torch._dynamo.reset()
            return run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        fast_result, (fast_code,) = compile_with_fast_math(a, b, True)
        precise_result, (precise_code,) = compile_with_fast_math(a, b, False)
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(
            fast_result.double(), expected, atol=0.02, rtol=0.002
        )
        torch.testing.assert_close(
            precise_result.double(), expected, atol=0.02, rtol=0.002
        )
        self.assertIn("cute.math.tanh", fast_code)
        self.assertIn("fastmath=True", fast_code)
        self.assertNotIn("cute.math.erf", fast_code)
        self.assertIn("cute.math.erf", precise_code)
        self.assertNotIn("cute.math.tanh", precise_code)

    def test_mm_fast_math_gelu_tanh_preserves_requested_approximation(self, device):
        def epilogue_fn(acc):
            return torch.nn.functional.gelu(acc, approximate="tanh")

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "fast_math": True},
            )

        a = self.makeTensor(128, 64, device=device)
        b = self.makeTensor(64, 128, device=device)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = epilogue_fn(a.double() @ b.double())

        torch.testing.assert_close(actual.double(), expected, atol=0.02, rtol=0.002)
        self.assertIn("cute.math.tanh", code)
        self.assertIn("fastmath=True", code)
        self.assertNotIn("cute.math.erf", code)


instantiate_device_type_tests(TestFlexGemmFastMathDevice, globals(), only_for="cuda")


@skipIfNoCuteDSL
@unittest.skipIf(not SM100OrLater, "SM100+ required")
class TestFlexGemmExplicitConfigDevice(FlexGemmTestCase):
    def test_mm_explicit_config_matches_reference(self, device):
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
            gemm_config_key,
        )

        def epilogue_fn(acc):
            return (acc + 1).relu()

        config = next(
            config
            for config in candidate_gemm_configs_for_device(device)
            if config.swap_ab
        )
        config_key = gemm_config_key(config)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": dict(config_key)},
            )

        a = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
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
        self.assertIn(f"config_key={config_key!r}", code)

    @parametrize("tuned", (False, True))
    def test_mm_partial_config_matches_reference(self, device, tuned):
        from torch._inductor.heuristics.template.flex_gemm import (
            candidate_gemm_configs_for_device,
        )

        def epilogue_fn(acc):
            return (acc + 1).relu()

        config = next(
            config
            for config in candidate_gemm_configs_for_device(device)
            if config.swap_ab
        )
        pinned = {
            name: getattr(config, name)
            for name in ("tile_m", "tile_n", "cluster_m", "cluster_n", "swap_ab")
        }

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={
                    "backend": "QUACK",
                    "config": pinned,
                    "tuned": tuned,
                },
            )

        a = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
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
        for item in pinned.items():
            self.assertIn(repr(item), code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize("group", (64, 128))
    def test_mm_tuple_aux_local_n_reduce_supports_clustered_tile_m256(
        self, device, group
    ):
        from torch._inductor.heuristics.template.flex_gemm import gemm_config_key
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = 256

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), x.sum(-1)

        expected_config = GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            cluster_m=2,
            cluster_n=1,
            device_capacity=10,
        )
        config_key = gemm_config_key(expected_config)
        config = dict(config_key)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertLocalReduceAuxCode(code, group, callbacks=True)
        self.assertIn(f"config_key={config_key!r}", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_tuned_local_reduce_supports_max_autotune(self, device):
        """Keep mutable local-reduce outputs out of deferred template selection."""
        from torch._inductor import config as inductor_config

        m, n, group = 256, 512, 512

        def epilogue_fn(acc):
            partials = acc.float().view(m, -1, group).sum(-1)
            return acc.relu(), partials

        def fn(a, b):
            actual, partials = flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )
            return actual, partials.sum(-1)

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        with inductor_config.patch(max_autotune=True):
            (actual, reduced), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        expected, expected_partials = epilogue_fn(a @ b)
        high_precision_expected, high_precision_partials = epilogue_fn(
            a.double() @ b.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual,
            expected,
            high_precision_expected,
            a.shape[1],
        )
        torch.testing.assert_close(
            reduced,
            high_precision_partials.float().sum(-1),
            atol=1e-3,
            rtol=1e-3,
        )
        self.assertFlexGemmGeneratedCode(code)
        self.assertIn("('tile_m', 256)", code)
        self.assertIn("('tile_n', 512)", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_full_tile_local_reduce_checks_actual_n_warp_layout(self, device):
        """Reject a host-approved full-N group when the kernel layout splits N."""
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = group = 256

        def epilogue_fn(acc):
            partials = acc.float().view(m, -1, group).sum(-1)
            return acc.relu(), partials

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=128,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                device_capacity=10,
            )
        )

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        with (
            mock.patch(
                "torch._inductor.kernel.flex_gemm.lowering."
                "validate_flex_gemm_local_reduce_config",
                return_value=True,
            ),
            self.assertRaisesRegex(
                Exception,
                "full-N GroupedLocalReduce requires one epilogue N-warp partition",
            ),
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize(
        "case",
        (
            ("local_n_g32_tile64", 1, 32, 128, 64, 2, 2),
            ("local_n_g16_tile160", 1, 16, 128, 160, 2, 2),
            ("local_n_g32_tile192", 1, 32, 128, 192, 2, 1),
            ("local_n_g32_tile224", 1, 32, 256, 224, 2, 2),
            ("local_n_g32_tile256", 1, 32, 128, 256, 2, 2),
            ("local_n_g64_tile_m128", 1, 64, 128, 256, 2, 1),
            ("local_n_g128_tile_m128", 1, 128, 128, 256, 2, 1),
            ("local_n_full_tile256_tile_m256", 1, 256, 256, 256, 2, 1),
            ("local_n_full_tile512_tile_m256", 1, 512, 256, 512, 2, 1),
            ("local_m_g128_tile160", 0, 128, 128, 160, 1, 1),
            ("local_m_g64_tile_m256", 0, 64, 256, 256, 2, 1),
            ("local_m_g128_tile_m256", 0, 128, 256, 256, 2, 1),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_reduce_supports_expanded_configs(self, device, case):
        from torch._inductor.heuristics.template.flex_gemm import gemm_config_key
        from torch._vendor.quack.gemm_config import GemmConfig

        _, axis, group, tile_m, tile_n, cluster_m, cluster_n = case
        m = max(tile_m, group, 256)
        n = max(tile_n, group if axis == 1 else 256)

        def epilogue_fn(acc):
            if axis == 1:
                partial = acc.float().view(m, -1, group).sum(-1)
            else:
                partial = acc.float().view(-1, group, n).sum(1)
            return acc.relu(), partial

        expected_config = GemmConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            pingpong=False,
            cluster_m=cluster_m,
            cluster_n=cluster_n,
            device_capacity=10,
        )
        config_key = gemm_config_key(expected_config)
        config = dict(config_key)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, n, device=device, dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        if axis == 1:
            self.assertLocalReduceAuxCode(code, group, callbacks=group > 32)
        else:
            self.assertIn(f"FlexGemmLocalReduceGeometry(group={group}, axis=0)", code)
            self.assertIn("FlexGemmLocalReduceCallbacks(", code)
        self.assertIn(f"config_key={config_key!r}", code)


instantiate_device_type_tests(
    TestFlexGemmExplicitConfigDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
