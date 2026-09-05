# Owner(s): ["module: inductor"]

import contextlib
import dataclasses
import importlib
import math
import struct
import sys
import unittest
from types import SimpleNamespace
from typing import get_args
from unittest import mock

import torch
from torch._higher_order_ops import flex_gemm
from torch._higher_order_ops.flex_gemm import _SUPPORTED_FLEX_GEMM_OP_NAMES
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor.ops_handler import ReductionType
from torch._inductor.utils import run_and_get_code
from torch._subclasses.fake_tensor import is_fake
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import SM100OrLater, SM120OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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
) -> torch.Tensor:
    """Encode E8M0 scales with public tensor operations and inline assembly."""
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
            "mx_e8m0_scale rounding must be 'floor', 'rceil', or None, "
            f"got {rounding!r}"
        )
    rounding = "rceil" if rounding is None else rounding
    max_abs = amax.float()
    if torch.compiler.is_compiling() or is_fake(max_abs):
        if rounding == "rceil":
            conversion_input = max_abs / max_value
            instruction = "cvt.rp.satfinite.ue8m0x2.f32"
            infinity_bits = None
        else:
            max_power = math.floor(math.log2(max_value))
            conversion_input = max_abs * 2.0**-max_power
            instruction = "cvt.rz.ue8m0x2.f32"
            infinity_bits = (
                struct.unpack("<I", struct.pack("<f", 2.0 ** (128 - max_power)))[0]
                if max_power > 0
                else None
            )
        asm = f"{instruction} $0, 0.0, $1;"
        if infinity_bits is not None:
            asm = (
                "{ .reg .pred is_inf; .reg .f32 finite; "
                "testp.infinite.f32 is_inf, $1; "
                f"selp.f32 finite, 0f{infinity_bits:08x}, $1, is_inf; "
                f"{instruction} $0, 0.0, finite; }}"
            )
        return inline_asm_elementwise(
            conversion_input,
            asm_str=asm,
            constraints="=h,r",
            dtype=torch.float8_e8m0fnu,
        )

    if rounding == "floor":
        max_power = math.floor(math.log2(max_value))
        exponent = ((max_abs.view(torch.int32) >> 23) & 0xFF) - 127
        encoded = torch.clamp(exponent - max_power, min=-127, max=128) + 127
    else:
        bits = (max_abs / max_value).view(torch.int32)
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        encoded = exponent + (mantissa != 0).to(torch.int32)
        encoded = torch.where(
            (exponent == 0) & (mantissa <= 0x400000),
            torch.zeros_like(encoded),
            encoded,
        )
        encoded = torch.clamp(encoded, max=254)
    encoded = encoded.to(torch.uint8)
    encoded = torch.where(torch.isnan(max_abs), torch.full_like(encoded, 255), encoded)
    return encoded.view(torch.float8_e8m0fnu)


def nvfp4_e4m3_scale(amax: torch.Tensor, max_value: float = 6.0) -> torch.Tensor:
    """Compute the ordinary E4M3 scale used by NVFP4 test epilogues."""
    if (
        not isinstance(max_value, float)
        or not math.isfinite(max_value)
        or max_value <= 0
    ):
        raise ValueError(
            "nvfp4_e4m3_scale max_value must be a finite positive float, "
            f"got {max_value!r}"
        )
    return torch.clamp(
        amax.float() / max_value,
        min=torch.finfo(torch.float8_e4m3fn).tiny,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)


class TestFlexGemmRuntimeImport(TestCase):
    def test_import_does_not_load_vendored_quack(self):
        for name in list(sys.modules):
            if name == "torch._vendor.quack" or name.startswith("torch._vendor.quack."):
                del sys.modules[name]
        sys.modules.pop("torch._inductor.kernel.flex_gemm.runtime", None)
        importlib.import_module("torch._inductor.kernel.flex_gemm.runtime")
        self.assertNotIn("torch._vendor.quack", sys.modules)

    def test_quack_support_probe_requires_cutlass(self):
        from torch._inductor.kernel.flex_gemm import lowering

        with mock.patch.object(lowering.importlib.util, "find_spec", return_value=None):
            self.assertFalse(lowering.has_flex_gemm_quack())
        with mock.patch.object(
            lowering.importlib.util, "find_spec", return_value=SimpleNamespace()
        ):
            self.assertTrue(lowering.has_flex_gemm_quack())


class TestFlexGemmOutputLayout(TestCase):
    def test_builtin_layout_contracts(self):
        from torch._inductor.kernel.flex_gemm.output_layout import (
            BLOCKED_128X4,
            TRANSPOSED,
        )

        self.assertEqual(BLOCKED_128X4.carrier_shape_fn(2, 129, 5), (2, 2, 2, 512))
        blocked_config = SimpleNamespace(
            device_capacity=10,
            tile_m=256,
            tile_n=256,
            cluster_m=2,
            swap_ab=False,
        )
        self.assertTrue(BLOCKED_128X4.supports_config_fn(blocked_config, 1, 32))
        self.assertFalse(BLOCKED_128X4.supports_config_fn(blocked_config, 0, 32))
        with self.assertRaisesRegex(NotImplementedError, "only axis 1"):
            BLOCKED_128X4.validate_geometry(SimpleNamespace(axis=0))

        carrier = torch.empty(24)
        transposed = TRANSPOSED.runtime_view(carrier, 2, 3, 4)
        self.assertEqual(transposed.shape, (2, 4, 3))
        self.assertEqual(transposed.stride(), (12, 3, 1))
        self.assertTrue(
            TRANSPOSED.supports_config_fn(SimpleNamespace(swap_ab=False), 0, 8)
        )
        self.assertFalse(
            TRANSPOSED.supports_config_fn(SimpleNamespace(swap_ab=True), 0, 8)
        )
        singleton = torch.empty(4).as_strided((4, 1), (1, 4))
        TRANSPOSED.validate_carrier_fn(singleton)
        with self.assertRaisesRegex(ValueError, "must be contiguous"):
            TRANSPOSED.validate_carrier_fn(torch.empty_strided((4, 3), (4, 1)))

    def test_layout_passes_first_class_callbacks_to_quack(self):
        from torch._inductor.kernel.flex_gemm import output_layout

        tensor_fn = mock.Mock()
        fake_shape_fn = mock.Mock()
        constructor = mock.Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        grouped_reduce = SimpleNamespace(GroupedLocalReduceOutputLayout=constructor)
        cutedsl = SimpleNamespace(
            blocked_128x4_output_tensor=tensor_fn,
            blocked_128x4_fake_shape=fake_shape_fn,
        )
        with mock.patch.object(
            output_layout, "output_layout_cutedsl", return_value=cutedsl
        ):
            quack_layout = output_layout.BLOCKED_128X4.quack_layout(grouped_reduce)

        self.assertIs(quack_layout.tensor_fn, tensor_fn)
        self.assertIs(quack_layout.fake_shape_fn, fake_shape_fn)
        self.assertIs(
            quack_layout.carrier_shape_fn,
            output_layout.blocked_128x4_carrier_shape,
        )
        self.assertIs(
            quack_layout.validate_carrier_fn,
            output_layout.blocked_128x4_validate_carrier,
        )
        self.assertEqual(
            output_layout.BLOCKED_128X4.codegen_reference(),
            "flex_gemm_output_layout.BLOCKED_128X4",
        )
        with self.assertRaisesRegex(ValueError, "must be bound"):
            dataclasses.replace(
                output_layout.BLOCKED_128X4, symbol="MISSING"
            ).codegen_reference()

    def test_blocked_layout_owns_carrier_validation(self):
        from torch._inductor.kernel.flex_gemm.output_layout import BLOCKED_128X4

        BLOCKED_128X4.validate_carrier_fn(torch.empty(8))
        with self.assertRaisesRegex(ValueError, "carrier must be contiguous"):
            BLOCKED_128X4.validate_carrier_fn(torch.empty(4, 4).t())


@instantiate_parametrized_tests
class TestFlexGemmRuntimeHelpers(TestCase):
    def test_tensorssa_clamp_codegen_uses_public_cutlass_api(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            FlexGemmTensorSSAOpOverrides,
        )

        self.assertEqual(
            FlexGemmTensorSSAOpOverrides.clamp("x", "lower", "upper"),
            "cutlass.min(cutlass.max(x, lower), upper)",
        )

    def test_epimod_division_respects_fast_math(self):
        from torch._inductor.kernel.flex_gemm.epilogue import FlexGemmEpiModOpOverrides

        self.assertEqual(
            FlexGemmEpiModOpOverrides(False).truediv("a", "b"),
            "epi_math.divide(a, b, fast=False)",
        )
        self.assertEqual(
            FlexGemmEpiModOpOverrides(True).truediv("a", "b"),
            "epi_math.divide(a, b, fast=True)",
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    def test_epimod_cache_identity_includes_all_capture_dtypes(self):
        from torch._inductor.kernel.flex_gemm import runtime

        def epilogue_fn(acc, operand0):
            return {"D": acc + operand0}

        def build(dtype):
            return runtime.flex_gemm_epimod(
                epilogue_fn,
                (torch.empty(1, 8, dtype=dtype),),
                ("row",),
                0,
                None,
                None,
                True,
                None,
            )

        with mock.patch.object(runtime, "_EPIMOD_CACHE", {}):
            fp32 = build(torch.float32)
            self.assertIs(fp32, build(torch.float32))
            self.assertIsNot(fp32, build(torch.bfloat16))

    def test_indexed_output_runtime_plan_validates_physical_buffers(self):
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmEpiModIndexedOutputPlan,
        )

        plan = FlexGemmEpiModIndexedOutputPlan(
            torch.empty(8, dtype=torch.bool),
            torch.empty(8, dtype=torch.int64),
        )
        self.assertEqual(plan.cache_key, (torch.uint8, torch.int64))
        with self.assertRaisesRegex(RuntimeError, "contiguous vector"):
            FlexGemmEpiModIndexedOutputPlan(
                torch.empty(2, 8), torch.empty(8, dtype=torch.int64)
            )
        with self.assertRaisesRegex(RuntimeError, "same shape"):
            FlexGemmEpiModIndexedOutputPlan(
                torch.empty(8), torch.empty(7, dtype=torch.int64)
            )
        with self.assertRaisesRegex(NotImplementedError, "int32 or int64"):
            FlexGemmEpiModIndexedOutputPlan(torch.empty(8), torch.empty(8))

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
        "reduction_type,op_name,init_val",
        (
            ("sum", "ADD", 0.0),
            ("prod", "MUL", 1.0),
            ("max", "MAX", float("-inf")),
            ("min", "MIN", float("inf")),
        ),
    )
    def test_materialize_tensorssa_reduction_reuses_descriptor(
        self, reduction_type, op_name, init_val
    ):
        from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
            tensorssa_reduction,
        )
        from torch._inductor.kernel import gemm_epilogue_codegen

        descriptor = tensorssa_reduction(reduction_type)
        reduction_ops = SimpleNamespace(
            ADD=object(), MUL=object(), MAX=object(), MIN=object()
        )
        cute = SimpleNamespace(ReductionOp=reduction_ops)
        combine = object()
        with mock.patch.object(
            gemm_epilogue_codegen,
            "materialize_epilogue_function",
            side_effect=(combine, lambda: init_val),
        ) as materialize:
            reduction = gemm_epilogue_codegen.materialize_tensorssa_reduction(
                reduction_type, cute
            )

        self.assertIs(reduction.reduce_op, getattr(reduction_ops, op_name))
        self.assertEqual(reduction.init_val, init_val)
        self.assertIs(reduction.combine, combine)
        self.assertEqual(
            [call.args[0] for call in materialize.call_args_list],
            [
                f"def combine(lhs, rhs):\n    return {descriptor.combine_expr}",
                f"def init():\n    return {descriptor.init_val}",
            ],
        )
        for call in materialize.call_args_list:
            self.assertIs(call.args[1], cute)

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

    def test_quant_scale_fake_strides_match_eager(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        amax = torch.rand(2, 1).expand(2, 4)
        ops = (mx_e8m0_scale, nvfp4_e4m3_scale)
        eager_strides = tuple(op(amax).stride() for op in ops)
        with FakeTensorMode() as mode:
            fake_amax = mode.from_tensor(amax)
            for op, eager_stride in zip(ops, eager_strides):
                with self.subTest(op=str(op)):
                    self.assertEqual(op(fake_amax).stride(), eager_stride)

    def test_quant_scale_rounding_args(self):
        amax = torch.tensor(
            [0.0, 448.0, 449.0, 500.0, 511.0, 512.0, float("inf"), float("nan")]
        )
        floor = mx_e8m0_scale(amax, rounding="floor").view(torch.uint8)
        rceil = mx_e8m0_scale(amax, rounding="rceil").view(torch.uint8)

        self.assertEqual(
            floor,
            torch.tensor([0, 127, 127, 127, 127, 128, 247, 255], dtype=torch.uint8),
        )
        self.assertEqual(
            rceil,
            torch.tensor([0, 127, 128, 128, 128, 128, 254, 255], dtype=torch.uint8),
        )
        self.assertEqual(mx_e8m0_scale(amax).view(torch.uint8), rceil)
        with self.assertRaisesRegex(ValueError, "rounding must be"):
            mx_e8m0_scale(amax, rounding="nearest")
        with self.assertRaisesRegex(ValueError, "finite positive float"):
            nvfp4_e4m3_scale(amax, max_value=0.0)

    def test_nvfp4_pack_known_codes(self):
        from torch._higher_order_ops.flex_gemm import nvfp4_pack

        values = torch.tensor(
            [[0.25, 0.75], [-0.25, -0.75], [-6.0, 6.0]], dtype=torch.float32
        )
        packed = nvfp4_pack(values)
        self.assertEqual(packed.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(
            packed.view(torch.uint8),
            torch.tensor([0x20, 0xA8, 0x7F], dtype=torch.uint8),
        )

    def test_to_blocked_matches_reference_and_fake_shape(self):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.testing._internal.common_quantized import (
            to_blocked as reference_to_blocked,
        )

        scale = torch.arange(129 * 5, dtype=torch.float32).view(5, 129).mT
        self.assertFalse(scale.is_contiguous())
        self.assertEqual(to_blocked(scale), to_blocked(scale.contiguous()))
        self.assertEqual(to_blocked(scale), reference_to_blocked(scale))
        self.assertEqual(to_blocked(scale).shape, (2048,))

        with FakeTensorMode() as mode:
            blocked = to_blocked(mode.from_tensor(scale))
            self.assertEqual(blocked.shape, (2048,))
            self.assertEqual(blocked.dtype, scale.dtype)

        with self.assertRaisesRegex(ValueError, "expects a 2-D tensor"):
            to_blocked(torch.ones(4))

    @unittest.skipUnless(importlib.util.find_spec("cutlass"), "requires CuTeDSL")
    def test_quack_feed_main_host_guards_match_runtime_contract(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            validate_local_reduce_feed_main_capability,
        )
        from torch._vendor.quack.grouped_reduce import feed_main_capable

        for axis in (0, 1):
            for group in (2, 8, 16, 32, 64, 128):
                if feed_main_capable(axis, group):
                    validate_local_reduce_feed_main_capability(axis, group)
                else:
                    with self.assertRaises(NotImplementedError):
                        validate_local_reduce_feed_main_capability(axis, group)

    def test_local_reduce_propagates_before_grouped_view_matching(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            FlexGemmLocalReduceAnalysis,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b):
            grouped = torch.mm(a, b).view(4, 6, 2)
            reduced = grouped.sum(dim=-1, keepdim=True).squeeze(-1)
            return reduced.view(4, 2, 3)

        graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 12))
        analysis = FlexGemmLocalReduceAnalysis.from_graph_module(
            graph_module,
            gemm_node(graph_module, torch.ops.aten.mm.default),
        )
        output = next(
            node for node in graph_module.graph.nodes if node.op == "output"
        ).args[0]
        self.assertIsInstance(output, torch.fx.Node)
        self.assertIn(output, analysis.matches)
        self.assertNotIn(output, analysis.grouped_tensors)
        self.assertEqual(
            analysis.matches[output].value_node.target,
            torch.ops.aten.sum.dim_IntList,
        )

    def test_grouped_layout_rejects_inexact_inferred_preserved_dimension(self):
        from torch._inductor.kernel.flex_gemm.quack_reductions import (
            grouped_tensor_layout,
        )

        with self.assertRaisesRegex(
            NotImplementedError, "grouped reshape must split exactly"
        ):
            grouped_tensor_layout((-1, 2, 2), (4, 5))

    @parametrize("unbacked_dim", (1, 2))
    def test_grouped_layout_rejects_unbacked_structural_dimensions(self, unbacked_dim):
        from torch._inductor.kernel.flex_gemm.quack_reductions import (
            grouped_tensor_layout,
        )
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape = [4, -1, 2]
        shape[unbacked_dim] = ShapeEnv().create_unbacked_symint()
        self.assertIsNone(grouped_tensor_layout(tuple(shape), (4, 8)))

    def test_grouped_layout_rejected_backed_group_does_not_guard(self):
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.quack_reductions import (
            grouped_tensor_layout,
        )
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(3, ConstantSource("group"))
        group = shape_env.create_symintnode(symbol, hint=3)

        self.assertIsNone(grouped_tensor_layout((4, 5, group), (4, 8)))
        self.assertEqual(shape_env.guards, [])

    def test_grouped_main_output_recognizer_only_mutates_analysis_on_match(self):
        """Rejected recognitions must not leak grouped layouts or guards."""
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmGroupedMainOutputTransform,
        )
        from torch._inductor.kernel.flex_gemm.epilogue import (
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

        chunked = FlexGemmGroupedMainOutputTransform(group=2, chunked=True)
        for body, expected_transform in (
            (swap_halves_plus_acc, None),
            (silu_mul_halves, chunked),
        ):
            with self.subTest(body=body.__name__):
                graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 16))
                shape_env = None
                if expected_transform is None:
                    shape_env = ShapeEnv()
                    symbol = shape_env.create_symbol(8, ConstantSource("split_size"))
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
                self.assertEqual(analysis.outputs.main_transform, expected_transform)
                registered = [
                    node
                    for node in split_nodes
                    if node in analysis.grouped_main_layouts
                ]
                self.assertEqual(registered, split_nodes if expected_transform else [])
                if shape_env is not None:
                    self.assertEqual(shape_env.guards, [])

    def test_rejected_grouped_select_does_not_install_index_guard(self):
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def duplicate_grouped_lane(a, b):
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

        graph_module = make_fx(duplicate_grouped_lane)(
            torch.randn(4, 8), torch.randn(8, 16)
        )
        shape_env = ShapeEnv()
        index_symbol = shape_env.create_symbol(2, ConstantSource("select_index"))
        index = shape_env.create_symintnode(index_symbol, hint=2)
        group_symbol = shape_env.create_symbol(4, ConstantSource("view_group"))
        group = shape_env.create_symintnode(group_symbol, hint=4)
        view = next(
            node
            for node in graph_module.graph.nodes
            if node.target is torch.ops.aten.view.default
        )
        view.args = (view.args[0], (4, 4, group))
        select = next(
            node
            for node in graph_module.graph.nodes
            if node.target is torch.ops.aten.select.int and node.args[2] == 2
        )
        select.args = (select.args[0], select.args[1], index)

        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        self.assertIsNone(analysis.outputs.main_transform)
        self.assertEqual(analysis.grouped_main_layouts, {})
        self.assertEqual(shape_env.guards, [])

    @parametrize("pointwise", (False, True))
    def test_accepted_grouped_reduction_installs_group_guard(self, pointwise):
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.epilogue import (
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

    def test_active_grouped_layout_installs_own_structural_guard(self):
        from torch._dynamo.source import ConstantSource
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def grouped_reduction_with_sibling(a, b):
            acc = torch.mm(a, b)
            sibling = acc.view(4, 8, 2)
            reduced = acc.view(4, 8, 2).sum(-1, keepdim=True)
            return (sibling * reduced).view(4, 16)

        graph_module = make_fx(grouped_reduction_with_sibling)(
            torch.randn(4, 8), torch.randn(8, 16)
        )
        reduction = next(
            node
            for node in graph_module.graph.nodes
            if node.target is torch.ops.aten.sum.dim_IntList
        )
        sibling_view = next(
            node
            for node in graph_module.graph.nodes
            if node.target is torch.ops.aten.view.default
            and node is not reduction.args[0]
            and tuple(node.meta["val"].shape) == (4, 8, 2)
        )
        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(2, ConstantSource("sibling_group"))
        group = shape_env.create_symintnode(symbol, hint=2)
        sibling_view.args = (sibling_view.args[0], (4, 8, group))

        analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        self.assertEqual(len(shape_env.guards), 1)

    def test_grouped_main_output_does_not_contract_other_axes(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
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
                self.assertIsNone(analysis.outputs.main_transform)

    def test_indexed_output_accepts_gather_from_converted_main(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b, indices):
            main = torch.mm(a, b).to(torch.bfloat16)
            return main, main.gather(1, indices[:, None]).squeeze(1)

        graph_module = make_fx(body)(
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.tensor([0, 7, 8, 15]),
        )
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )

        self.assertIsNotNone(analysis.outputs.indexed_output)

    def test_indexed_output_rejects_strided_indices(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b, indices):
            main = torch.mm(a, b)
            return main, main.gather(1, indices[:, None]).squeeze(1)

        graph_module = make_fx(body)(
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.arange(8)[::2],
        )
        with self.assertRaisesRegex(NotImplementedError, "must be contiguous"):
            analyze_flex_gemm_epilogue(
                graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
            )

    def test_indexed_output_rejects_terminal_dtype_view(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b, indices):
            main = torch.mm(a, b).to(torch.float16).view(torch.bfloat16)
            return main, main.gather(1, indices[:, None]).squeeze(1)

        graph_module = make_fx(body)(
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.tensor([0, 7, 8, 15]),
        )
        with self.assertRaisesRegex(NotImplementedError, "terminal dtype views"):
            analyze_flex_gemm_epilogue(
                graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
            )

    def test_indexed_output_rejects_shared_terminal_conversion(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            flex_gemm_indexed_output_store,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(x, indices):
            logits = x.float()
            selected = logits.gather(1, indices[:, None]).squeeze(1)
            return logits.to(x.dtype), selected.to(x.dtype), selected + 1.0

        graph_module = make_fx(body)(
            torch.randn(4, 8, dtype=torch.bfloat16),
            torch.tensor([0, 1, 2, 3]),
        )
        output = next(node for node in graph_module.graph.nodes if node.op == "output")
        main, indexed, _ = output.args[0]

        self.assertIsNone(flex_gemm_indexed_output_store(main, indexed))

    def test_indexed_output_plan_preserves_aux_order(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b, indices):
            main = torch.mm(a, b)
            ordinary = main + 1.0
            indexed = main.gather(1, indices[:, None]).squeeze(1)
            local = main.float().view(4, 4, 4).sum(-1)
            return main, ordinary, indexed, local

        graph_module = make_fx(body)(
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.tensor([0, 7, 8, 15]),
        )
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        returned = next(
            node for node in graph_module.graph.nodes if node.op == "output"
        ).args[0]

        self.assertEqual(analysis.outputs.returned_aux_outputs, tuple(returned[1:]))
        self.assertEqual(analysis.outputs.aux_outputs, (returned[1],))
        self.assertIs(analysis.outputs.indexed_output.node, returned[2])
        self.assertIs(analysis.outputs.local_reduce.store.node, returned[3])

    def test_indexed_output_rejects_grouped_main_composition(self):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b, indices):
            grouped = torch.mm(a, b).view(4, 8, 2)
            main = grouped.select(-1, 0) - grouped.select(-1, 1)
            return main, main.gather(1, indices[:, None]).squeeze(1)

        graph_module = make_fx(body)(
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.tensor([0, 3, 4, 7]),
        )
        with self.assertRaisesRegex(NotImplementedError, "do not yet compose"):
            analyze_flex_gemm_epilogue(
                graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
            )

    def test_indexed_output_debug_report(self):
        from torch._inductor.kernel.flex_gemm.debug import format_flex_gemm_analysis
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b, indices):
            main = torch.mm(a, b).relu()
            return main, main.gather(1, indices[:, None]).squeeze(1)

        graph_module = make_fx(body)(
            torch.randn(4, 8),
            torch.randn(8, 16),
            torch.tensor([0, 7, 8, 15]),
        )
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        report = format_flex_gemm_analysis(analysis)

        self.assertIn("indexed:\n  output:", report)
        self.assertIn("indices: indices_1: shape=(4,)", report)

    def test_flex_gemm_debug_report(self):
        from torch._inductor.kernel.flex_gemm.debug import (
            flex_gemm_log,
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
            log_flex_gemm_artifact,
        )
        from torch._inductor.kernel.flex_gemm.epilogue import (
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
        report = format_flex_gemm_analysis(analysis)
        details = format_flex_gemm_analysis_details(analysis)
        self.assertIn("outputs:\n  main: relu: shape=(4, 64)", report)
        self.assertIn("auxiliary:\n  (none)", report)
        self.assertIn("main_transform: none", report)
        self.assertIn("geometry: axis=N, group=32", report)
        self.assertIn("consumers: returned", report)
        self.assertIn("output_layout: dense", report)
        self.assertIn("config_constraints:\n  axis=N, group=32", report)
        self.assertIn("grouped_tensors:\n  view:", details)
        self.assertIn("local_reduce_matches:\n", details)
        self.assertIn("grouped_select_indices:\n  (none)", details)
        with self.assertLogs(flex_gemm_log, level="INFO") as records:
            log_flex_gemm_artifact("analysis", lambda: report)
            log_flex_gemm_artifact("analysis_details", lambda: details, verbose=True)
        self.assertEqual(len(records.output), 1)
        self.assertIn(" ===== ANALYSIS =====", records.output[0])

    def test_grouped_main_debug_report(self):
        from torch._inductor.kernel.flex_gemm.debug import (
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
        )
        from torch._inductor.kernel.flex_gemm.epilogue import (
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
            "main_transform: grouped-N, group=2, layout=interleaved",
            format_flex_gemm_analysis(analysis),
        )
        details = format_flex_gemm_analysis_details(analysis)
        self.assertIn("aten.select.int", details)
        self.assertNotIn("grouped_select_indices:\n  (none)", details)

    def test_nvfp4_pack_debug_report(self):
        from torch._higher_order_ops.flex_gemm import nvfp4_pack
        from torch._inductor.kernel.flex_gemm.debug import (
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
        )
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b):
            return nvfp4_pack(torch.mm(a, b).float().view(4, 8, 2))

        with FakeTensorMode() as mode:
            graph_module = make_fx(body, tracing_mode="fake")(
                mode.from_tensor(torch.randn(4, 8)),
                mode.from_tensor(torch.randn(8, 16)),
            )
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        self.assertIn(
            "main_transform: grouped-N, group=2, layout=interleaved",
            format_flex_gemm_analysis(analysis),
        )
        self.assertIn(
            "inline_asm_elementwise",
            format_flex_gemm_analysis_details(analysis),
        )

    def test_to_blocked_debug_report(self):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._inductor.kernel.flex_gemm.debug import (
            format_flex_gemm_analysis,
            format_flex_gemm_analysis_details,
        )
        from torch._inductor.kernel.flex_gemm.epilogue import (
            analyze_flex_gemm_epilogue,
            gemm_node,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        def body(a, b):
            acc = torch.mm(a, b)
            grouped = acc.float().view(4, 2, 32)
            return acc, to_blocked(grouped.abs().amax(-1))

        graph_module = make_fx(body)(torch.randn(4, 8), torch.randn(8, 64))
        analysis = analyze_flex_gemm_epilogue(
            graph_module, gemm_node(graph_module, torch.ops.aten.mm.default)
        )
        self.assertIn(
            "output_layout: blocked_128x4", format_flex_gemm_analysis(analysis)
        )
        self.assertIn(
            "flex_gemm.to_blocked.default",
            format_flex_gemm_analysis_details(analysis),
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


class FlexGemmTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if SM100OrLater:
            from torch._inductor.kernel.flex_gemm import lowering

            if not lowering.has_flex_gemm_quack():
                raise unittest.SkipTest("requires CuTeDSL")

    @contextlib.contextmanager
    def limitEpiModAutotune(self, device):
        """Limit tests after production legality pruning has selected candidates."""
        import torch._vendor.quack.gemm_runtime.autotune as epi_autotune

        prune = epi_autotune._prune_for_mod

        def limited_prune(*args, **kwargs):
            return prune(*args, **kwargs)[:2]

        with (
            mock.patch.object(
                epi_autotune, "_prune_for_mod", side_effect=limited_prune
            ),
            mock.patch.object(epi_autotune, "_MOD_TUNERS", {}),
        ):
            yield

    def quackGemmConfigs(self, device):
        """Return vendored QuACK configs eligible for dense EpiMod calls."""
        from torch._vendor.quack.cute_dsl_utils import get_device_capacity
        from torch._vendor.quack.gemm_config import get_all_configs

        capacity = get_device_capacity(device)[0]
        configs = tuple(
            config
            for config in get_all_configs()
            if config.device_capacity == capacity
            and not config.use_tma_gather
            and config.split_k in (None, 1)
        )
        self.assertTrue(configs)
        return configs

    @staticmethod
    def quackConfigKey(config):
        """Return canonical vendored QuACK config constraints."""
        return tuple(sorted(dataclasses.asdict(config).items()))

    def makeTensor(self, *shape, device="cuda", dtype=torch.bfloat16):
        return torch.testing.make_tensor(
            *shape, device=device, dtype=dtype, low=-0.1, high=0.1
        )

    def assertFlexGemmGeneratedCode(self, code, *checks):
        (
            FileCheck()
            .check("from torch._inductor.kernel.flex_gemm.runtime import (")
            .check("gemm_epimod as flex_gemm_runtime")
            .check("flex_gemm_runtime(")
            .check("tuned=")
            .check("stream=stream")
            .check_not("config_key=")
            .check_not("epilogue_source=")
            .check_not("from torch._vendor.quack")
            .check_not("import torch._vendor.quack")
            .run(code)
        )
        self.assertNotIn("quack_cache_dir", code)
        for check in checks:
            self.assertIn(check, code)

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
                lambda msg: (
                    f"{msg}\nactual error {actual_error.item()} exceeded low precision eager "
                    f"error {eager_error.item()} with fp32_accumulation_eps="
                    f"{fp32_accumulation_eps}, result_rounding_eps="
                    f"{result_rounding_eps}, output_scale={output_scale}, "
                    f"and atol={rounding_atol}"
                )
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

    def assertLocalReduceAuxCode(self, code, group, axis=1):
        """Check generated code passes a structural EpiMod compressed-aux plan."""
        (
            FileCheck()
            .check("local_reduce=FlexGemmEpiModLocalReducePlan")
            .check(self.localReduceGeometryPattern(group, axis))
            .check("out=")
            .check("combine=")
            .run(code)
        )

    def assertMxScaleCode(self, code, rounding="rceil"):
        """Check direct E8M0 conversion code without requiring a named helper."""
        instruction = {
            "floor": "cvt.rz.ue8m0x2.f32",
            "rceil": "cvt.rp.satfinite.ue8m0x2.f32",
        }[rounding]
        self.assertIn(instruction, code)
        self.assertIn("result_type=cutlass.Float8E8M0FNU", code)

    def assertNvfp4ScaleCode(self, code, max_value=6.0):
        """Check direct E4M3 scale rounding in generated code."""
        precise_division = (
            f"epi_math.divide(value, {max_value!r}, fast=False)",
            f"/ cute.full_like(local_reduce0, {max_value!r})",
        )
        self.assertTrue(any(expression in code for expression in precise_division))
        self.assertTrue("torch.float8_e4m3fn" in code or "cutlass.Float8E4M3FN" in code)

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
        """Check generated code uses the QuACK EpiMod feed-main plan."""
        file_check = FileCheck().check("local_reduce=FlexGemmEpiModLocalReducePlan")
        if group is None:
            file_check = file_check.check("FlexGemmLocalReduceGeometry(group=").check(
                "axis=0)"
            )
        else:
            file_check = file_check.check(self.localReduceGeometryPattern(group, 0))
        file_check.check("feeds_main=True").check("combine=").run(code)


@instantiate_parametrized_tests
class TestFlexGemmAnalysis(TestCase):
    def test_nvgemm_backend_overrides_global_backend(self):
        import inspect

        from torch._inductor import config
        from torch._inductor.kernel.flex_gemm import lowering

        graph = torch.fx.Graph()
        bias = graph.placeholder("bias")
        mat1 = graph.placeholder("mat1")
        mat2 = graph.placeholder("mat2")
        output = graph.call_function(torch.ops.aten.addmm.default, (bias, mat1, mat2))
        graph.output(output)
        graph_module = torch.fx.GraphModule({}, graph)
        expected = object()

        def process(lowered_graph, args):
            self.assertFalse(
                lowered_graph.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.addmm.default
                )
            )
            self.assertTrue(
                lowered_graph.graph.find_nodes(
                    op="call_function", target=torch.ops.aten.mm.default
                )
            )
            self.assertTrue(config.max_autotune)
            self.assertEqual(config.max_autotune_gemm_backends, "NVGEMM")
            return expected

        with (
            config.patch(max_autotune=False, max_autotune_gemm_backends="TRITON"),
            mock.patch.object(lowering, "process_subgraph_nodes", side_effect=process),
        ):
            actual = inspect.unwrap(lowering.flex_gemm_lowering)(
                torch.ops.aten.addmm.default,
                SimpleNamespace(graph_module=graph_module),
                (),
                {},
                {"backend": "NVGEMM"},
            )
        self.assertIs(actual, expected)

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

    def test_epimod_local_reduce_plan_validates_consumers(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.runtime import (
            FlexGemmEpiModLocalReducePlan,
        )

        axis0 = FlexGemmLocalReduceGeometry(8, 0)
        with self.assertRaisesRegex(RuntimeError, "local_reduce_out"):
            FlexGemmEpiModLocalReducePlan(axis0, combine="add")
        with self.assertRaisesRegex(RuntimeError, "require a combine"):
            FlexGemmEpiModLocalReducePlan(axis0, out=torch.empty(1))
        with self.assertRaisesRegex(RuntimeError, "prepass finalizers"):
            FlexGemmEpiModLocalReducePlan(
                axis0,
                out=torch.empty(1),
                combine="add",
                prepass_finalize="mean",
            )
        with self.assertRaisesRegex(NotImplementedError, "same-warp axis-0"):
            FlexGemmEpiModLocalReducePlan(
                FlexGemmLocalReduceGeometry(64, 0),
                feeds_main=True,
                combine="add",
            )
        FlexGemmEpiModLocalReducePlan(
            FlexGemmLocalReduceGeometry(16, 1),
            feeds_main=True,
            combine="add",
            prepass=lambda acc: {"local_reduce0": acc},
            prepass_combine="add",
        )
        FlexGemmEpiModLocalReducePlan(axis0, out=torch.empty(1), combine="max")

    def test_local_reduce_plan_uses_explicit_consumers(self):
        from torch._inductor.kernel.flex_gemm.constraints import (
            FlexGemmLocalReduceGeometry,
        )
        from torch._inductor.kernel.flex_gemm.template import (
            FlexGemmEpilogueLocalReduceConfig,
        )

        geometry = FlexGemmLocalReduceGeometry(8, 0)
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
            FlexGemmIndexedOutputStore,
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
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputPlan(
                node,
                indexed_output=FlexGemmIndexedOutputStore(aux, aux, ()),
            )
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputPlan(node, output_storage_nodes=(aux,))
        with self.assertRaisesRegex(RuntimeError, "tensor nodes"):
            FlexGemmLocalReduceMatch(object(), geometry)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputLocalReducePlan(object())
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmOutputLocalReducePlan(match)
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmLocalReduceStore(object())
        with self.assertRaisesRegex(RuntimeError, "output plans"):
            FlexGemmLocalReduceStore(aux, object())
        with self.assertRaisesRegex(NotImplementedError, "tensor outputs"):
            tuple_output_plan(object(), (), analysis)
        with self.assertRaisesRegex(NotImplementedError, "tensor outputs"):
            tuple_output_plan(node, (object(),), analysis)
        FlexGemmOutputPlan(
            node,
            (aux,),
            local_reduce=FlexGemmOutputLocalReducePlan(
                match, store=FlexGemmLocalReduceStore(aux)
            ),
        )
        FlexGemmOutputPlan(
            node,
            (aux,),
            local_reduce=FlexGemmOutputLocalReducePlan(match, feeds_main=True),
        )

    @parametrize(
        "case",
        (
            ("t", lambda x: x.t(), (4, 8), 2),
            ("transpose", lambda x: x.transpose(0, 1), (1, 8), 1),
            ("permute", lambda x: x.permute(1, 0), (8, 1), 1),
            ("identity", lambda x: x.permute(0, 1), (4, 4), None),
        ),
        name_fn=lambda case: case[0],
    )
    def test_local_reduce_output_storage_classifies_transpose(self, case):
        from torch._inductor.kernel.flex_gemm.epilogue import (
            FlexGemmLocalReduceOutputStorage,
            match_flex_gemm_local_reduce_output_storage,
        )
        from torch._inductor.kernel.flex_gemm.output_layout import TRANSPOSED
        from torch.fx.experimental.proxy_tensor import make_fx

        _, transpose, shape, expected_nodes = case
        graph_module = make_fx(lambda x: transpose(x).contiguous())(torch.randn(shape))
        output = next(
            node for node in graph_module.graph.nodes if node.op == "output"
        ).args[0]
        storage = match_flex_gemm_local_reduce_output_storage(output)
        if expected_nodes is None:
            self.assertIsNone(storage)
            return
        self.assertIsInstance(storage, FlexGemmLocalReduceOutputStorage)
        self.assertIs(storage.layout, TRANSPOSED)
        self.assertEqual(len(storage.nodes), expected_nodes)
        self.assertIs(storage.nodes[-1], output)
        self.assertEqual(storage.source.op, "placeholder")


@instantiate_parametrized_tests
class TestFlexGemmEpilogueHOP(FlexGemmTestCase):
    def test_supported_op_names_match_dense_scope(self):
        self.assertEqual(
            _SUPPORTED_FLEX_GEMM_OP_NAMES, "mm/addmm/bmm/baddbmm/scaled_mm"
        )

    def test_scaled_mm_requires_functional_api(self):
        for name, gemm_op in (
            ("packet", torch.ops.aten._scaled_mm_v2),
            ("default", torch.ops.aten._scaled_mm_v2.default),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    RuntimeError, "use torch.nn.functional.scaled_mm"
                ):
                    flex_gemm(gemm_op, (), lambda acc: acc)

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

    def test_default_backend_compiled_matches_reference(self):
        def fn(a, b):
            return flex_gemm(torch.mm, (a, b), lambda acc: acc.relu())

        a = torch.randn(8, 16)
        b = torch.randn(16, 12)
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
        torch.testing.assert_close(actual, (a @ b).relu())

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
        scale = torch.randn(5)

        with self.assertRaisesRegex(
            Exception,
            "captured tensor epilogue args currently must match",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, scale)

    @unittest.skipUnless(importlib.util.find_spec("cutlass"), "requires CuTeDSL")
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
    def test_mm_compiled_uses_current_stream(self):
        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.relu(),
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a.fill_(0.25)
            b.fill_(0.5)
            actual = compiled(a, b)
        torch.cuda.current_stream().wait_stream(stream)
        self.assertEqual(actual, (a @ b).relu())

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
    @unittest.skipIf(SM120OrLater, "QuACK block-scaled GEMM requires SM100/SM110")
    @parametrize(
        "case",
        (
            ("mxfp8_e4m3", False),
            ("nvfp4", False),
            ("mxfp8_e4m3", True),
        ),
        name_fn=lambda case: f"{case[0]}_tuned_{case[1]}",
    )
    def test_scaled_mm_compiled_matches_reference(self, case):
        import torch.nn.functional as F
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand

        format_name, tuned = case
        m = n = k = 256
        a = BlockScaledOperand.quantize(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            format_name,
        )
        b = BlockScaledOperand.quantize(
            torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k),
            format_name,
            dim=-2,
        )
        scale_a = a.scale.flatten()
        scale_b = b.scale.flatten()
        scale_recipe = (
            F.ScalingType.BlockWise1x32
            if format_name == "mxfp8_e4m3"
            else F.ScalingType.BlockWise1x16
        )
        gemm_kwargs = {
            "scale_recipe_a": scale_recipe,
            "scale_recipe_b": scale_recipe,
            "swizzle_a": F.SwizzleType.SWIZZLE_32_4_4,
            "swizzle_b": F.SwizzleType.SWIZZLE_32_4_4,
            "output_dtype": torch.bfloat16,
        }

        def epilogue_fn(acc, row):
            shifted = (acc + row).relu()
            return shifted, shifted * 0.5

        def fn(a_data, b_data, a_scale, b_scale, row):
            return flex_gemm(
                F.scaled_mm,
                (a_data, b_data, a_scale, b_scale),
                lambda acc: epilogue_fn(acc, row),
                gemm_kwargs=gemm_kwargs,
                kernel_options={"backend": "QUACK", "tuned": tuned},
            )

        row = torch.randn(1, n, device="cuda", dtype=torch.bfloat16)
        expected = epilogue_fn(
            F.scaled_mm(
                a.qdata,
                b.qdata,
                scale_a,
                gemm_kwargs["scale_recipe_a"],
                scale_b,
                gemm_kwargs["scale_recipe_b"],
                gemm_kwargs["swizzle_a"],
                gemm_kwargs["swizzle_b"],
                output_dtype=torch.bfloat16,
            ),
            row,
        )
        tune_context = (
            self.limitEpiModAutotune(torch.device("cuda"))
            if tuned
            else contextlib.nullcontext()
        )
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        with tune_context:
            actual, (code,) = run_and_get_code(
                compiled,
                a.qdata,
                b.qdata,
                scale_a,
                scale_b,
                row,
            )
            warm = compiled(a.qdata, b.qdata, scale_a, scale_b, row)

        torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.2)
        torch.testing.assert_close(actual[1], expected[1], rtol=0.02, atol=0.2)
        torch.testing.assert_close(warm[0], actual[0], rtol=0, atol=0)
        torch.testing.assert_close(warm[1], actual[1], rtol=0, atol=0)
        self.assertIn(f"bs_format_a={format_name!r}", code)
        self.assertIn(f"tuned={tuned!r}", code)
        self.assertNotIn("aten._scaled_mm_v2", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "QuACK block-scaled GEMM requires SM100/SM110")
    @parametrize("shared_global_scale", (False, True))
    def test_nvfp4_scaled_mm_global_scales(self, shared_global_scale):
        import torch.nn.functional as F
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand

        m = n = k = 256
        global_a = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        global_b = (
            global_a
            if shared_global_scale
            else torch.tensor([1.5], device="cuda", dtype=torch.float32)
        )
        a = BlockScaledOperand.quantize(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            "nvfp4",
            per_tensor_scale=global_a,
        )
        b = BlockScaledOperand.quantize(
            torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k),
            "nvfp4",
            dim=-2,
            per_tensor_scale=global_b,
        )
        scale_a = a.scale.flatten()
        scale_b = b.scale.flatten()
        recipes = [
            F.ScalingType.BlockWise1x16,
            F.ScalingType.TensorWise,
        ]
        swizzles = [
            F.SwizzleType.SWIZZLE_32_4_4,
            F.SwizzleType.NO_SWIZZLE,
        ]
        gemm_kwargs = {
            "scale_recipe_a": recipes,
            "scale_recipe_b": recipes,
            "swizzle_a": swizzles,
            "swizzle_b": swizzles,
            "output_dtype": torch.bfloat16,
        }

        def epilogue_fn(acc):
            return (acc + 0.125).relu()

        def fn(a_data, b_data, a_scale, a_global, b_scale, b_global):
            return flex_gemm(
                F.scaled_mm,
                (a_data, b_data, [a_scale, a_global], [b_scale, b_global]),
                epilogue_fn,
                gemm_kwargs=gemm_kwargs,
                kernel_options={"backend": "QUACK"},
            )

        expected = epilogue_fn(
            F.scaled_mm(
                a.qdata,
                b.qdata,
                [scale_a, a.per_tensor_scale],
                recipes,
                [scale_b, b.per_tensor_scale],
                recipes,
                swizzles,
                swizzles,
                output_dtype=torch.bfloat16,
            )
        )
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a.qdata,
            b.qdata,
            scale_a,
            a.per_tensor_scale,
            scale_b,
            b.per_tensor_scale,
        )

        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.3)
        self.assertIn("((acc * operand0) * operand1)", code)
        self.assertNotIn("aten._scaled_mm_v2", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "QuACK block-scaled GEMM requires SM100/SM110")
    def test_scaled_mm_grouped_main_and_local_reduce(self):
        import torch.nn.functional as F
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = k = 256
        group = 16
        a = BlockScaledOperand.quantize(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            "mxfp8_e4m3",
        )
        b = BlockScaledOperand.quantize(
            torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k),
            "mxfp8_e4m3",
            dim=-2,
        )
        scale_a = a.scale.flatten()
        scale_b = b.scale.flatten()
        gemm_kwargs = {
            "scale_recipe_a": F.ScalingType.BlockWise1x32,
            "scale_recipe_b": F.ScalingType.BlockWise1x32,
            "swizzle_a": F.SwizzleType.SWIZZLE_32_4_4,
            "swizzle_b": F.SwizzleType.SWIZZLE_32_4_4,
            "output_dtype": torch.bfloat16,
        }
        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
                device_capacity=10,
            )
        )

        def scaled(epilogue_fn):
            def fn(a_data, b_data, a_scale, b_scale):
                return flex_gemm(
                    F.scaled_mm,
                    (a_data, b_data, a_scale, b_scale),
                    epilogue_fn,
                    gemm_kwargs=gemm_kwargs,
                    kernel_options={"backend": "QUACK", "config": config},
                )

            return fn

        base = F.scaled_mm(
            a.qdata,
            b.qdata,
            scale_a,
            gemm_kwargs["scale_recipe_a"],
            scale_b,
            gemm_kwargs["scale_recipe_b"],
            gemm_kwargs["swizzle_a"],
            gemm_kwargs["swizzle_b"],
            output_dtype=torch.bfloat16,
        )

        def grouped_main(acc):
            pairs = acc.float().view(m, -1, 2)
            return (pairs[..., 0] - pairs[..., 1]).to(acc.dtype)

        grouped, (grouped_code,) = run_and_get_code(
            torch.compile(scaled(grouped_main), backend="inductor", fullgraph=True),
            a.qdata,
            b.qdata,
            scale_a,
            scale_b,
        )
        torch.testing.assert_close(grouped, grouped_main(base), rtol=0.03, atol=0.3)
        self.assertIn("GroupedMainOutputTransform(group=2", grouped_code)

        def local_reduce(acc):
            grouped = acc.float().view(m, -1, group)
            return acc.relu(), nvfp4_e4m3_scale(grouped.abs().amax(-1))

        (actual, scale), (reduce_code,) = run_and_get_code(
            torch.compile(scaled(local_reduce), backend="inductor", fullgraph=True),
            a.qdata,
            b.qdata,
            scale_a,
            scale_b,
        )
        expected, expected_scale = local_reduce(base)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.3)
        torch.testing.assert_close(
            scale.float(), expected_scale.float(), rtol=0.125, atol=0.0625
        )
        self.assertIn("local_reduce=FlexGemmEpiModLocalReducePlan", reduce_code)
        self.assertNotIn("aten._scaled_mm_v2", grouped_code + reduce_code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "QuACK block-scaled GEMM requires SM100/SM110")
    def test_scaled_mm_quantized_output(self):
        import torch.nn.functional as F
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand
        from torch._vendor.quack.blockscaled.utils import unpack_scale_blocked_to_2d
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = k = 256
        group = 32
        a = BlockScaledOperand.quantize(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            "mxfp8_e4m3",
        )
        b = BlockScaledOperand.quantize(
            torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k),
            "mxfp8_e4m3",
            dim=-2,
        )
        scale_a = a.scale.flatten()
        scale_b = b.scale.flatten()
        recipe = F.ScalingType.BlockWise1x32
        swizzle = F.SwizzleType.SWIZZLE_32_4_4
        gemm_kwargs = {
            "scale_recipe_a": recipe,
            "scale_recipe_b": recipe,
            "swizzle_a": swizzle,
            "swizzle_b": swizzle,
            "output_dtype": torch.bfloat16,
        }
        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
                device_capacity=10,
            )
        )

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            scale = mx_e8m0_scale(grouped.abs().amax(-1, keepdim=True))
            quantized = (
                (grouped * scale.float().reciprocal())
                .view_as(acc)
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
            )
            return quantized, to_blocked(scale.squeeze(-1))

        def fn(a_data, b_data, a_scale, b_scale):
            return flex_gemm(
                F.scaled_mm,
                (a_data, b_data, a_scale, b_scale),
                epilogue_fn,
                gemm_kwargs=gemm_kwargs,
                kernel_options={"backend": "QUACK", "config": config},
            )

        base = F.scaled_mm(
            a.qdata,
            b.qdata,
            scale_a,
            recipe,
            scale_b,
            recipe,
            swizzle,
            swizzle,
            output_dtype=torch.bfloat16,
        )
        expected, expected_scale = epilogue_fn(base)
        (actual, actual_scale), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a.qdata,
            b.qdata,
            scale_a,
            scale_b,
        )

        scale_shape = (1, (m + 127) // 128, (n // group + 3) // 4, 32, 4, 4)
        actual_dense_scale = unpack_scale_blocked_to_2d(
            actual_scale.view(scale_shape), m, n // group
        ).squeeze(0)
        expected_dense_scale = unpack_scale_blocked_to_2d(
            expected_scale.view(scale_shape), m, n // group
        ).squeeze(0)
        actual_dequant = actual.float() * actual_dense_scale.float().repeat_interleave(
            group, -1
        )
        expected_dequant = (
            expected.float() * expected_dense_scale.float().repeat_interleave(group, -1)
        )

        torch.testing.assert_close(
            actual_dequant, expected_dequant, rtol=0.05, atol=0.5
        )
        self.assertMxScaleCode(code)
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        self.assertNotIn("aten._scaled_mm_v2", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "QuACK block-scaled GEMM requires SM100/SM110")
    def test_scaled_mm_dynamic_m(self):
        import torch.nn.functional as F
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand

        n = k = 256
        b = BlockScaledOperand.quantize(
            torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k),
            "mxfp8_e4m3",
            dim=-2,
        )
        scale_b = b.scale.flatten()
        recipe = F.ScalingType.BlockWise1x32
        swizzle = F.SwizzleType.SWIZZLE_32_4_4
        gemm_kwargs = {
            "scale_recipe_a": recipe,
            "scale_recipe_b": recipe,
            "swizzle_a": swizzle,
            "swizzle_b": swizzle,
            "output_dtype": torch.bfloat16,
        }

        def epilogue_fn(acc):
            return (acc + 0.25).relu()

        def fn(a_data, a_scale):
            return flex_gemm(
                F.scaled_mm,
                (a_data, b.qdata, a_scale, scale_b),
                epilogue_fn,
                gemm_kwargs=gemm_kwargs,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        for index, m in enumerate((128, 256)):
            a = BlockScaledOperand.quantize(
                torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
                "mxfp8_e4m3",
            )
            scale_a = a.scale.flatten()
            if index == 0:
                torch._dynamo.mark_dynamic(a.qdata, 0)
                torch._dynamo.mark_dynamic(scale_a, 0)
            actual = compiled(a.qdata, scale_a)
            expected = epilogue_fn(
                F.scaled_mm(
                    a.qdata,
                    b.qdata,
                    scale_a,
                    recipe,
                    scale_b,
                    recipe,
                    swizzle,
                    swizzle,
                    output_dtype=torch.bfloat16,
                )
            )
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.2)

    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_scaled_mm_unsupported_recipe_falls_back(self):
        import torch.nn.functional as F

        m = n = k = 64
        a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn).t()
        scale_a = torch.tensor([0.5], device="cuda", dtype=torch.float32)
        scale_b = torch.tensor([1.5], device="cuda", dtype=torch.float32)
        gemm_kwargs = {
            "scale_recipe_a": F.ScalingType.TensorWise,
            "scale_recipe_b": F.ScalingType.TensorWise,
            "swizzle_a": F.SwizzleType.NO_SWIZZLE,
            "swizzle_b": F.SwizzleType.NO_SWIZZLE,
            "output_dtype": torch.bfloat16,
        }

        def epilogue_fn(acc):
            return (acc + 0.25).relu()

        def fn(a_data, b_data, a_scale, b_scale):
            return flex_gemm(
                F.scaled_mm,
                (a_data, b_data, a_scale, b_scale),
                epilogue_fn,
                gemm_kwargs=gemm_kwargs,
                kernel_options={"backend": "QUACK"},
            )

        expected = epilogue_fn(
            F.scaled_mm(
                a,
                b,
                scale_a,
                gemm_kwargs["scale_recipe_a"],
                scale_b,
                gemm_kwargs["scale_recipe_b"],
                gemm_kwargs["swizzle_a"],
                gemm_kwargs["swizzle_b"],
                output_dtype=torch.bfloat16,
            )
        )
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            scale_a,
            scale_b,
        )

        torch.testing.assert_close(actual, expected)
        self.assertIn("_scaled_mm_v2", code)
        self.assertNotIn("flex_gemm_runtime", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "QuACK block-scaled GEMM requires SM100/SM110")
    def test_scaled_mm_fast_accum_falls_back(self):
        import torch.nn.functional as F
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand

        m = n = k = 64
        a = BlockScaledOperand.quantize(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            "mxfp8_e4m3",
        )
        b = BlockScaledOperand.quantize(
            torch.randn(k, n, device="cuda", dtype=torch.bfloat16),
            "mxfp8_e4m3",
            dim=-2,
        )
        scale_a = a.scale.flatten()
        scale_b = b.scale.flatten()
        recipe = F.ScalingType.BlockWise1x32
        swizzle = F.SwizzleType.SWIZZLE_32_4_4
        gemm_kwargs = {
            "scale_recipe_a": recipe,
            "scale_recipe_b": recipe,
            "swizzle_a": swizzle,
            "swizzle_b": swizzle,
            "output_dtype": torch.bfloat16,
            "use_fast_accum": True,
        }

        def epilogue_fn(acc):
            return (acc + 0.25).relu()

        def fn(a_data, b_data, a_scale, b_scale):
            return flex_gemm(
                F.scaled_mm,
                (a_data, b_data, a_scale, b_scale),
                epilogue_fn,
                gemm_kwargs=gemm_kwargs,
                kernel_options={"backend": "QUACK"},
            )

        expected = epilogue_fn(
            F.scaled_mm(
                a.qdata,
                b.qdata,
                scale_a,
                recipe,
                scale_b,
                recipe,
                swizzle,
                swizzle,
                output_dtype=torch.bfloat16,
                use_fast_accum=True,
            )
        )
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a.qdata,
            b.qdata,
            scale_a,
            scale_b,
        )

        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.3)
        self.assertIn("_scaled_mm_v2", code)
        self.assertNotIn("flex_gemm_runtime", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @unittest.skipIf(SM120OrLater, "QuACK block-scaled GEMM requires SM100/SM110")
    @parametrize(
        "case",
        (
            ("plain", False, False),
            ("terminal_view", True, False),
            ("strided_indices", False, True),
        ),
        name_fn=lambda case: case[0],
    )
    def test_scaled_mm_indexed_output_falls_back(self, case):
        _, terminal_view, strided_indices = case
        import torch.nn.functional as F
        from torch._vendor.quack.blockscaled.operand import BlockScaledOperand

        m = n = k = 256
        a = BlockScaledOperand.quantize(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            "mxfp8_e4m3",
        )
        b = BlockScaledOperand.quantize(
            torch.randn(k, n, device="cuda", dtype=torch.bfloat16) / math.sqrt(k),
            "mxfp8_e4m3",
            dim=-2,
        )
        scale_a = a.scale.flatten()
        scale_b = b.scale.flatten()
        target_count = 2 * m if strided_indices else m
        targets = torch.arange(target_count, device="cuda", dtype=torch.int64) % n
        if strided_indices:
            targets = targets[::2]
        recipe = F.ScalingType.BlockWise1x32
        swizzle = F.SwizzleType.SWIZZLE_32_4_4
        gemm_kwargs = {
            "scale_recipe_a": recipe,
            "scale_recipe_b": recipe,
            "swizzle_a": swizzle,
            "swizzle_b": swizzle,
            "output_dtype": torch.bfloat16,
        }

        def epilogue(acc, targets):
            main = (acc + 0.25).relu()
            if terminal_view:
                main = main.to(torch.float16).view(torch.bfloat16)
            return main, main.gather(1, targets[:, None]).squeeze(1)

        def fn(a_data, b_data, a_scale, b_scale, targets):
            return flex_gemm(
                F.scaled_mm,
                (a_data, b_data, a_scale, b_scale),
                lambda acc: epilogue(acc, targets),
                gemm_kwargs=gemm_kwargs,
                kernel_options={"backend": "QUACK"},
            )

        base = F.scaled_mm(
            a.qdata,
            b.qdata,
            scale_a,
            recipe,
            scale_b,
            recipe,
            swizzle,
            swizzle,
            output_dtype=torch.bfloat16,
        )
        expected = epilogue(base, targets)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a.qdata,
            b.qdata,
            scale_a,
            scale_b,
            targets,
        )

        if terminal_view:
            self.assertEqual(
                actual[0].view(torch.float16),
                expected[0].view(torch.float16),
                rtol=0.02,
                atol=0.2,
            )
        else:
            self.assertEqual(actual[0], expected[0], rtol=0.02, atol=0.2)
        self.assertEqual(actual[1], actual[0].gather(1, targets[:, None]).squeeze(1))
        self.assertIn("_scaled_mm_v2", code)
        self.assertNotIn("FlexGemmEpiModIndexedOutputPlan", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("interleaved_group2", 2, False, False, 256, None),
            ("chunked_group2", 2, True, False, 256, None),
            ("interleaved_group4", 4, False, False, 256, None),
            ("interleaved_group2_tuned", 2, False, True, 256, None),
            ("interleaved_group2_partial_n", 2, False, False, 192, {"tile_n": 128}),
            ("chunked_group2_partial_n", 2, True, False, 192, {"tile_n": 128}),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_grouped_main_output_matches_reference(self, case):
        _, group, chunked, tuned, n, config = case
        if group == 4 and torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("group-4 grouped main outputs are currently SM100-only")
        m, k = 128, 64
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = (
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
            if chunked
            else torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        )

        def epilogue_fn(acc):
            if chunked:
                lanes = acc.chunk(group, dim=-1)
            else:
                grouped = acc.view(acc.shape[0], acc.shape[1] // group, group)
                lanes = tuple(grouped.select(-1, index) for index in range(group))
            return sum(lanes[1:], lanes[0])

        def fn(lhs, rhs):
            return flex_gemm(
                torch.mm,
                (lhs, rhs),
                epilogue_fn,
                kernel_options={
                    "backend": "QUACK",
                    "tuned": tuned,
                    **({} if config is None else {"config": config}),
                },
            )

        tune_context = (
            self.limitEpiModAutotune(torch.device("cuda"))
            if tuned
            else contextlib.nullcontext()
        )
        with tune_context:
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            k,
        )
        self.assertIn("'main':", code)
        self.assertIn("FlexGemmGroupedMainOutputTransform(", code)
        self.assertIn(f"group={group}", code)
        self.assertIn("fragmentwise=True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_grouped_main_output_uint8(self):
        m = n = k = 64
        group = 2

        def epilogue_fn(acc):
            lanes = acc.float().view(m, n, group)
            return (lanes[..., 0] - lanes[..., 1]).to(torch.uint8)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.eye(m, k, device="cuda", dtype=torch.float16)
        columns = torch.arange(group * n, device="cuda")
        pair = (columns // group) % 8
        b = torch.where(columns % group == 0, 3 * pair + 2, pair)
        b = b.expand(k, -1).contiguous().to(torch.float16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual, fn(a, b))
        self.assertEqual(actual.shape, (m, n))
        self.assertIn("FlexGemmGroupedMainOutputTransform(group=2", code)
        self.assertNotIn("extern_kernels.mm", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_grouped_main_output_rejects_unsafe_explicit_config(self):
        m, n, k, group = 128, 128, 64, 2

        def epilogue_fn(acc):
            grouped = acc.view(m, n // group, group)
            return grouped.select(-1, 0) + grouped.select(-1, 1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={
                    "backend": "QUACK",
                    "config": {"tile_n": 256},
                },
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            Exception, "no supported GemmConfig matches config_constraints"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_grouped_main_output_rejects_unsupported_composition(self):
        m = k = n = 64
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)

        def group_eight(a, b):
            def epilogue(acc):
                lanes = acc.view(m, n, 8)
                return sum(lanes[..., index] for index in range(8))

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        b = torch.randn(k, 8 * n, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(Exception, "group 2 or 4"):
            torch.compile(group_eight, backend="inductor", fullgraph=True)(a, b)

        torch._dynamo.reset()

        def captured(a, b, scale):
            def epilogue(acc):
                lanes = acc.view(m, n, 2)
                return (lanes[..., 0] - lanes[..., 1]) * scale

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        b = torch.randn(k, 2 * n, device="cuda", dtype=torch.float16)
        scale = torch.randn(1, 1, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(Exception, "do not yet support captured tensors"):
            torch.compile(captured, backend="inductor", fullgraph=True)(a, b, scale)

        torch._dynamo.reset()

        def auxiliary(a, b):
            def epilogue(acc):
                lanes = acc.view(m, n, 2)
                return lanes[..., 0] - lanes[..., 1], acc

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={"backend": "QUACK"},
            )

        with self.assertRaisesRegex(Exception, "do not yet compose"):
            torch.compile(auxiliary, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_grouped_main_output_chunked_view(self):
        m, n, k = 128, 128, 64

        def epilogue_fn(acc):
            grouped = acc.view(m, 2, n)
            return grouped.select(1, 0) + grouped.select(1, 1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16).t()
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            k,
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("indices", ((0, 1), (1, 0), (-1, -2)))
    def test_mm_grouped_main_output_specializes_select_indices(self, indices):
        m, n, k = 128, 128, 64
        first, second = indices

        def epilogue_fn(acc):
            grouped = acc.view(m, n, 2)
            return grouped.select(-1, first) - grouped.select(-1, second)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, 2 * n, device="cuda", dtype=torch.bfloat16)
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            k,
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("split_size", (64, 128))
    def test_mm_grouped_main_output_specializes_split_size(self, split_size):
        torch._dynamo.reset()
        m = k = 64

        def epilogue_fn(acc):
            lanes = acc.float().split(split_size, dim=-1)
            return (lanes[0] - lanes[1]).to(acc.dtype)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(2 * split_size, k, device="cuda", dtype=torch.float16).t()
        torch._dynamo.mark_static(b, 1)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        torch.testing.assert_close(actual, epilogue_fn(a @ b), atol=0.2, rtol=0.05)
        FileCheck().check("FlexGemmGroupedMainOutputTransform(group=2").check(
            "chunked=True"
        ).check_not("extern_kernels.mm").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("chunked", (False, True))
    def test_mm_grouped_main_output_dynamic_m(self, chunked):
        group, n, k = 2, 256, 64

        def epilogue_fn(acc):
            if chunked:
                lhs, rhs = acc.chunk(group, dim=-1)
                return lhs - rhs
            grouped = acc.view(acc.shape[0], acc.shape[1] // group, group)
            return grouped.select(-1, 0) - grouped.select(-1, 1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        b = (
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16).t()
            if chunked
            else torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        )
        for index, m in enumerate((128, 192)):
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            if index == 0:
                torch._dynamo.mark_dynamic(a, 0)
            actual = compiled(a, b)
            self.assertMatchesLowPrecisionEager(
                actual,
                epilogue_fn(a @ b),
                epilogue_fn(a.double() @ b.double()),
                k,
            )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_grouped_main_output_dynamic_n(self):
        m = k = 64

        def epilogue_fn(acc):
            lanes = acc.view(m, -1, 2)
            return lanes[..., 0] - lanes[..., 1]

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        for index, output_n in enumerate((64, 68, 80)):
            b = torch.randn(k, 2 * output_n, device="cuda", dtype=torch.float16)
            if index == 0:
                torch._dynamo.mark_dynamic(b, 1)
            actual = compiled(a, b)
            torch.testing.assert_close(actual, fn(a, b), atol=0.2, rtol=0.05)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_chunked_grouped_main_rejects_grouped_reduction(self):
        m, n, k, group = 128, 128, 64, 16

        def epilogue_fn(acc):
            lanes = acc.chunk(2, dim=-1)
            main = lanes[0] + lanes[1]
            aux = acc.float().view(m, -1, group).sum(-1)
            return main, aux

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2 * n, k, device="cuda", dtype=torch.bfloat16).t()
        with self.assertRaisesRegex(
            Exception, "chunked grouped main outputs do not compose"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_chunked_grouped_main_rejects_contiguous_b(self):
        def epilogue_fn(acc):
            lhs, rhs = acc.chunk(2, dim=-1)
            return lhs + rhs

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(Exception, "column-major B storage"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("rounding", ("floor", "rceil"))
    def test_mm_tuple_aux_mx_scale_rounding(self, rounding):
        m, n, k, group = 128, 128, 16, 16
        maxima = torch.tensor(
            [0.0, 1.0, 448.0, 500.0, 512.0, 1024.0, 2.0**-120, 2.0**120],
            device="cuda",
            dtype=torch.bfloat16,
        )
        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        a[:, 0] = 1
        b[0] = maxima.repeat_interleave(group)

        def epilogue_fn(acc):
            grouped = acc.float().abs().view(m, n // group, group)
            return acc, mx_e8m0_scale(grouped.amax(-1), rounding=rounding)

        def fn(lhs, rhs):
            return flex_gemm(
                torch.mm,
                (lhs, rhs),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        (actual, scale), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected, expected_scale = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(scale.view(torch.uint8), expected_scale.view(torch.uint8))
        self.assertMxScaleCode(code, rounding)
        self.assertIn("FlexGemmEpiModLocalReducePlan", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("max_value", (4.0, 6.0))
    def test_mm_tuple_aux_nvfp4_scale_rounding(self, max_value):
        m, n, k, group = 128, 128, 16, 16
        maxima = torch.tensor(
            [0.0, 0.09375, 1.0, 4.0, 6.0, 7.0, 448.0, 2688.0],
            device="cuda",
            dtype=torch.bfloat16,
        )
        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        a[:, 0] = 1
        b[0] = maxima.repeat_interleave(group)

        def epilogue_fn(acc):
            grouped = acc.float().abs().view(m, n // group, group)
            return acc, nvfp4_e4m3_scale(grouped.amax(-1), max_value=max_value)

        def fn(lhs, rhs):
            return flex_gemm(
                torch.mm,
                (lhs, rhs),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        (actual, scale), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected, expected_scale = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(scale.view(torch.uint8), expected_scale.view(torch.uint8))
        self.assertNvfp4ScaleCode(code, max_value)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_prepare_softmax_online_local_reduce(self):
        from torch._inductor import inductor_prims

        m = n = k = 64
        group = 16

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            maximum, _ = inductor_prims.prepare_softmax_online(grouped, -1)
            return acc.relu(), maximum.squeeze(-1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        (actual, maximum), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected, expected_maximum = epilogue_fn(a @ b)

        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.2)
        torch.testing.assert_close(maximum, expected_maximum, rtol=0.02, atol=0.2)
        self.assertIn("combine='max'", code)
        self.assertNotIn("prepare_softmax_online", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (16, 32, 64))
    def test_mm_tuple_aux_mx_scale_preserves_nan(self, group):
        m, n, k = 16, 128, 16

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
        self.assertMxScaleCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_aux_scale_via_inline_asm_matches_named_op(self):
        from torch._higher_order_ops.inline_asm_elementwise import (
            inline_asm_elementwise,
        )

        m, n, k, group = 64, 128, 64, 32
        max_value = 448.0

        def named_epilogue(acc):
            grouped = acc.float().view(m, -1, group)
            return acc, mx_e8m0_scale(grouped.abs().amax(-1))

        def asm_epilogue(acc):
            grouped = acc.float().view(m, -1, group)
            encoded = inline_asm_elementwise(
                grouped.abs().amax(-1) / max_value,
                asm_str="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
                constraints="=h,r",
                dtype=torch.uint16,
            )
            return acc, encoded.to(torch.uint8).float()

        def half_asm_epilogue(acc):
            encoded = inline_asm_elementwise(
                acc.to(torch.bfloat16),
                asm_str="mov.b16 $0, $1;",
                constraints="=h,h",
                dtype=torch.uint16,
            )
            return acc, encoded.float()

        def compile_epilogue(epilogue_fn):
            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK"},
                )

            return torch.compile(fn, backend="inductor", fullgraph=True)

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        _, named_scale = compile_epilogue(named_epilogue)(a, b)
        (_, asm_scale), (code,) = run_and_get_code(compile_epilogue(asm_epilogue), a, b)

        self.assertEqual(asm_scale, named_scale.view(torch.uint8).float())
        self.assertIn("inline_asm_elementwise_intrinsic(", code)

        (_, half_bits), (half_code,) = run_and_get_code(
            compile_epilogue(half_asm_epilogue), a, b
        )
        expected_bits = (a @ b).to(torch.bfloat16).view(torch.uint16).float()
        self.assertEqual(half_bits, expected_bits)
        self.assertIn(".to(cutlass.BFloat16)", half_code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_inline_asm_three_outputs(self):
        m = k = n = 64

        def epilogue_fn(acc):
            first, second, third = inline_asm_elementwise(
                acc.float(),
                asm_str=("mov.f32 $0, $3; add.f32 $1, $3, $3; mul.f32 $2, $3, $3;"),
                constraints="=f,=f,=f,f",
                dtype=(torch.float32, torch.float32, torch.float32),
            )
            return first + second + third

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        expected_input = (a @ b).float()
        self.assertEqual(actual, expected_input * 3 + expected_input.square())
        self.assertEqual(code.count("inline_asm_elementwise_intrinsic("), 1)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_inline_asm_pack2_broadcasts_and_repeats_scalar(self):
        from torch._higher_order_ops.inline_asm_elementwise import (
            inline_asm_elementwise,
        )

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
        from torch._higher_order_ops.inline_asm_elementwise import (
            inline_asm_elementwise,
        )

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
        from torch._higher_order_ops.inline_asm_elementwise import (
            inline_asm_elementwise,
        )
        from torch._inductor.exc import InductorError

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
    def test_mm_tuple_aux_physical_mx_scale_zero_preserves_min_value(self):
        m, n, k, group = 16, 128, 16, 64

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
        self.assertMxScaleCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_physical_nvfp4_scale_rounds_before_pointwise(self):
        m, n, k, group = 16, 128, 64, 64

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
        self.assertNvfp4ScaleCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_tensorSSA_supports_mx_scale_max_value_saturation(self):
        m, n, k, group = 64, 128, 64, 16

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), mx_e8m0_scale(
                x.abs().amax(-1), max_value=2.0**-122, rounding="floor"
            )

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
        self.assertEqual(aux.view(torch.uint8), expected_aux.view(torch.uint8))
        self.assertTrue((aux.view(torch.uint8) == 255).all())
        self.assertMxScaleCode(code, "floor")
        self.assertIn("fragmentwise=True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_nvfp4_fragment_scale_feeds_main_with_e4m3_rounding(self):
        m = n = 64
        group = 16

        def epilogue_fn(acc):
            x = (acc.float() + 7.0).view(m, -1, group)
            scale = nvfp4_e4m3_scale(x.abs().amax(-1, keepdim=True))
            return (x * scale.float().reciprocal()).view(m, n)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.zeros(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(64, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        torch.testing.assert_close(actual, epilogue_fn(a @ b))
        self.assertNvfp4ScaleCode(code)
        self.assertIn("fragmentwise=True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("rounding", ("floor", "rceil"))
    def test_mm_fragment_group_mx_quant_preserves_nan(self, rounding):
        m = n = k = 32
        group = 32

        def epilogue_fn(acc):
            x = (acc.float() + float("nan")).view(m, -1, group)
            scale = mx_e8m0_scale(x.abs().amax(-1, keepdim=True), rounding=rounding)
            quantized = (x * scale.float().reciprocal()).view(m, n)
            return (
                quantized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn),
                scale.squeeze(-1),
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
        quantized, scale = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
        self.assertTrue(torch.isnan(quantized.float()).all())
        self.assertTrue((scale.view(torch.uint8) == 255).all())

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_fragment_group_mx_quant_feed_uses_tensorSSA(self):
        m, n, k, group = 64, 128, 64, 32

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            scale = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
            return (
                (x * scale.float().reciprocal()).view(m, n).to(torch.float8_e4m3fn),
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
        b = torch.ones(k, n, device="cuda", dtype=torch.bfloat16)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        x = (a @ b).float().view(m, -1, group)
        expected_aux = mx_e8m0_scale(x.abs().amax(-1, keepdim=True))
        expected = (
            (x * expected_aux.float().reciprocal()).view(m, n).to(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(actual.float(), expected.float())
        self.assertEqual(
            aux.view(torch.uint8), expected_aux.squeeze(-1).view(torch.uint8)
        )
        self.assertMxScaleCode(code)
        self.assertIn("fragmentwise=True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_m_reduce_feed_main_supports_mx_scale(self):
        m, n, k, group = 128, 64, 64, 32

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

        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        self.assertEqual(actual.dtype, torch.float8_e4m3fn)
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            high_precision_fn(a.double() @ b.double()),
            k,
        )
        self.assertMxScaleCode(code)
        self.assertIn("feeds_main=True", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
        "packed NVFP4 main outputs are currently validated only on SM100",
    )
    def test_mm_nvfp4_pack_matches_reference(self):
        from torch._higher_order_ops.flex_gemm import nvfp4_pack

        m, n, k = 128, 128, 64
        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            device="cuda",
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
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                lower,
                boundaries,
                upper,
            )
        ).repeat(4)[:n]
        a = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.zeros(k, n, device="cuda", dtype=torch.bfloat16)
        a[:, 0] = 1
        b[0] = values

        def epilogue_fn(acc):
            return nvfp4_pack(acc.float().view(m, -1, 2))

        def fn(lhs, rhs):
            return flex_gemm(
                torch.mm,
                (lhs, rhs),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        self.assertEqual(actual, epilogue_fn(a @ b))
        self.assertIn("cvt.rn.satfinite.e2m1x2.f32", code)
        self.assertIn("FlexGemmGroupedMainOutputTransform(group=2", code)

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
        check = FileCheck()
        match case:
            case "clamp":
                check = check.check("cutlass.max").check("cutlass.min")
            case "clamp_min":
                check = check.check("cutlass.max")
            case "clamp_max":
                check = check.check("cutlass.min")
        check.check_not("operator.ne").run(code)

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
                kernel_options={
                    "backend": "QUACK",
                    "tuned": True,
                    "config": {"swap_ab": True},
                },
            )

        with self.limitEpiModAutotune(torch.device("cuda")):
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
    def test_mm_dynamic_shapes_reads_captured_fragment_epilogue_arg(self, case, tuned):
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

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        with self.limitEpiModAutotune("cuda"):
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
    def test_mm_reads_bool_mask_captured_fragment_epilogue_arg(self, case):
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
    def test_mm_reads_bool_mask_captured_scalar_feed_main_arg(self):
        m, k, n, group = 128, 64, 64, 8

        def epilogue_fn(acc, mask):
            x = acc.float().view(-1, group, n)
            mean = x.mean(1, keepdim=True)
            centered = (x - mean).view(m, n)
            return torch.where(mask, centered, -centered)

        def fn(a, b, mask):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, mask),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        mask = torch.randint(0, 2, (m, n), device="cuda", dtype=torch.bool)

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
    def test_mm_preserves_integer_scalar_captured_fragment_epilogue_arg(self):
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
    def test_mm_promotes_low_precision_captured_fragment_epilogue_arg(self, case):
        kind, shape_fn = case

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
    @parametrize("extra_output", ("none", "aux"))
    @parametrize("swap_ab", (False, True))
    def test_mm_bool_main_output(self, extra_output, swap_ab):
        m, k, n = 256, 64, 256

        def epilogue_fn(acc):
            value = acc.float()
            main = value > 0
            return main if extra_output == "none" else (main, value * 0.5)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={
                    "backend": "QUACK",
                    "config": {"swap_ab": swap_ab},
                },
            )

        a = torch.eye(m, k, device="cuda", dtype=torch.bfloat16)
        rows = torch.arange(k, device="cuda")[:, None]
        cols = torch.arange(n, device="cuda")[None, :]
        b = (((rows + cols) % 17) - 8).to(torch.bfloat16)
        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertEqual(actual, epilogue_fn(a @ b))
        self.assertFlexGemmGeneratedCode(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "dtype",
        (torch.uint8, torch.int32),
        name_fn=lambda dtype: str(dtype).removeprefix("torch."),
    )
    def test_mm_generic_integer_main_output_rejected(self, dtype):
        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.to(dtype),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(
            Exception,
            "generic main outputs support only floating-point and bool dtypes",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

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

        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([[-0.25]], device="cuda", dtype=torch.float32)
        with self.limitEpiModAutotune(a.device):
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
                "unsupported FlexGEMM epilogue op",
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
                "unsupported FlexGEMM epilogue op",
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
            ("sum_method", lambda x: x.sum(-1), "combine='add'"),
            ("sum_function", lambda x: torch.sum(x, dim=-1), "combine='add'"),
            ("mean_method", lambda x: x.mean(-1), "finalize='mean'"),
            ("mean_function", lambda x: torch.mean(x, dim=-1), "finalize='mean'"),
            ("prod_method", lambda x: (x * 0.05).prod(-1), "combine='mul'"),
            (
                "prod_function",
                lambda x: torch.prod(x * 0.05, dim=-1),
                "combine='mul'",
            ),
            ("amax_method", lambda x: x.amax(-1), "combine='max'"),
            ("amax_function", lambda x: torch.amax(x, dim=-1), "combine='max'"),
            ("amin_method", lambda x: x.amin(-1), "combine='min'"),
            ("amin_function", lambda x: torch.amin(x, dim=-1), "combine='min'"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_n_reduce_compiled_matches_reference(self, case):
        _, reduce_fn, code_check = case
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
            FileCheck().check(code_check).run(code)
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
        self.assertLocalReduceAuxCode(code, group, axis=axis)

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
            ("sum", lambda x: x.sum(1), "combine='add'"),
            ("mean", lambda x: x.mean(1), "finalize='mean'"),
            ("prod", lambda x: (x * 0.05).prod(1), "combine='mul'"),
            ("amax", lambda x: x.amax(1), "combine='max'"),
            ("amin", lambda x: x.amin(1), "combine='min'"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_m_reduce_compiled_matches_reference(self, case, group):
        _, reduce_fn, code_check = case
        m = 128
        n = 128

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
        self.assertLocalReduceAuxCode(code, group, axis=0)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "index_dtype",
        (torch.int32, torch.int64),
        name_fn=lambda dtype: str(dtype).removeprefix("torch."),
    )
    def test_mm_indexed_output(self, index_dtype):
        from torch._inductor import config as inductor_config

        m, k, n = 65, 64, 512
        targets = torch.arange(m, device="cuda", dtype=index_dtype) % n
        targets[:5] = torch.tensor(
            [0, 127, 128, 256, n - 1], device="cuda", dtype=index_dtype
        )

        def epilogue(acc, targets):
            main = acc.relu()
            return main, main.gather(1, targets[:, None]).squeeze(1)

        def fn(a, b, targets):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, targets),
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        with inductor_config.patch(force_shape_pad=True):
            (actual, selected), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b, targets
            )
        high_precision = (a.double() @ b.double()).relu()

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue(a @ b, targets)[0],
            high_precision,
            k,
        )
        self.assertEqual(selected, actual.gather(1, targets[:, None]).squeeze(1))
        self.assertIn("FlexGemmEpiModIndexedOutputPlan", code)
        self.assertNotIn("extern_kernels.mm", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("float16", lambda acc: acc.float().relu().to(torch.float16)),
            ("bool", lambda acc: acc > 0),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_indexed_output_matches_main_dtype(self, case):
        _, main_fn = case
        m, k, n = 65, 64, 128
        targets = torch.arange(m, device="cuda", dtype=torch.int64) % n

        def epilogue(acc, targets):
            main = main_fn(acc)
            return main, main.gather(1, targets[:, None]).squeeze(1)

        def fn(a, b, targets):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, targets),
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        (actual, indexed), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, targets
        )

        self.assertEqual(indexed, actual.gather(1, targets[:, None]).squeeze(1))
        self.assertEqual(indexed.dtype, actual.dtype)
        self.assertIn("FlexGemmEpiModIndexedOutputPlan", code)
        if actual.dtype is not torch.bool:
            expected = epilogue(a @ b, targets)[0]
            high_precision = epilogue(a.double() @ b.double(), targets)[0]
            self.assertMatchesLowPrecisionEager(actual, expected, high_precision, k)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("transposed", (False, True))
    def test_mm_indexed_output_composes_with_local_reduce(self, transposed):
        m = n = 128
        k = 64
        group = 16
        targets = torch.arange(m, device="cuda", dtype=torch.int64) % n

        def epilogue(acc, targets):
            main = acc.relu()
            local = acc.float().view(m, n // group, group).sum(-1)
            if transposed:
                local = local.t().contiguous()
            ordinary = acc.float() + 0.25
            indexed = main.gather(1, targets[:, None]).squeeze(1)
            return main, local, indexed, ordinary

        def fn(a, b, targets):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue(acc, targets),
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, k)
        b = self.makeTensor(k, n)
        (actual, local, indexed, ordinary), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b, targets
        )
        low_precision = epilogue(a @ b, targets)
        high_precision = epilogue(a.double() @ b.double(), targets)

        self.assertMatchesLowPrecisionEager(
            actual, low_precision[0], high_precision[0], k
        )
        torch.testing.assert_close(
            local, high_precision[1].float(), atol=1e-3, rtol=1e-3
        )
        self.assertEqual(indexed, actual.gather(1, targets[:, None]).squeeze(1))
        self.assertMatchesLowPrecisionEager(
            ordinary, low_precision[3], high_precision[3], k
        )
        self.assertIn("FlexGemmEpiModIndexedOutputPlan", code)
        self.assertIn("FlexGemmEpiModLocalReducePlan", code)
        if transposed:
            self.assertIn("flex_gemm_output_layout.TRANSPOSED", code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_indexed_output_rejects_swap_ab(self):
        m = n = 128
        k = 64
        targets = torch.arange(m, device="cuda", dtype=torch.int64) % n

        def fn(a, b, targets):
            def epilogue(acc):
                main = acc.relu()
                return main, main.gather(1, targets[:, None]).squeeze(1)

            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue,
                kernel_options={
                    "backend": "QUACK",
                    "config": {"swap_ab": True},
                },
            )

        with self.assertRaisesRegex(Exception, "no .*GemmConfig.*config_constraints"):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                self.makeTensor(m, k), self.makeTensor(k, n), targets
            )

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
        self.assertLocalReduceAuxCode(code, group, axis=0)

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
        FileCheck().check("epi_math.sqrt").check("finalize=").run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0)

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
        FileCheck().check("local_reduce_finalize").check("finalize=").run(code)
        self.assertLocalReduceAuxCode(code, group)

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
        with self.assertRaisesRegex(Exception, "supports two grouped reductions only"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

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
        FileCheck().check(f"combine='{reduction[1:]}'").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize("group", (64, 128))
    @parametrize(
        "case",
        (
            ("sum", lambda x: x.sum(1), "combine='add'"),
            ("mean", lambda x: x.mean(1), "finalize='mean'"),
            ("amax", lambda x: x.amax(1), "combine='max'"),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_m_reduce_supports_cta_group(self, case, group):
        _, reduce_fn, code_check = case
        m = 128
        n = 128

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
        self.assertLocalReduceAuxCode(code, group, axis=0)

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

        with self.limitEpiModAutotune(a.device):
            actual, aux = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

        self.assertLocalReduceAuxMatches(actual, aux, a, b.mT, epilogue_fn)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_tuple_aux_contiguous_transpose_rejects_swap(self):
        m = n = group = 128
        config = {
            "tile_m": 256,
            "tile_n": 256,
            "cluster_m": 2,
            "cluster_n": 1,
            "pingpong": False,
            "is_dynamic_persistent": True,
            "swap_ab": True,
        }

        def epilogue_fn(acc):
            reduced = acc.float().view(m, -1, group).sum(-1)
            return acc, reduced.mT.contiguous()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        with self.assertRaisesRegex(
            Exception, "no supported GemmConfig matches config_constraints"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(
                self.makeTensor(m, 64), self.makeTensor(64, n)
            )

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
        FileCheck().check("combine='add'").run(code)
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
            ("sum_group64", 128, 64, lambda x: x.sum(-1), "combine='add'"),
            ("mean_group64", 128, 64, lambda x: x.mean(-1), "finalize='mean'"),
            ("amax_group64", 128, 64, lambda x: x.amax(-1), "combine='max'"),
            ("sum_group128", 256, 128, lambda x: x.sum(-1), "combine='add'"),
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
        self.assertLocalReduceAuxCode(code, group)
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
        FileCheck().check("tuned=True").run(code)
        self.assertLocalReduceAuxCode(code, group)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            (
                "variance_like",
                lambda x: (
                    ((x - x.mean(-1, keepdim=True)).square()).mean(-1) * 0.5 + 1.0
                ),
                " / 4.0",
                False,
            ),
            (
                "sum_keepdim_squeeze",
                lambda x: x.sum(-1, keepdim=True).squeeze(-1),
                "combine='add'",
                False,
            ),
            (
                "stable_logsumexp",
                lambda x: (
                    (x - x.amax(-1, keepdim=True)).exp().sum(-1, keepdim=True).log()
                    + x.amax(-1, keepdim=True)
                ).view(x.shape[0], -1),
                "epi_math.log",
                True,
            ),
            (
                "logsumexp_method",
                lambda x: x.logsumexp(-1),
                "epi_math.log",
                True,
            ),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_reduce_supports_chained_grouped_expressions(self, case):
        case_name, aux_fn, generated_check, checks_max = case
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
        self.assertLocalReduceAuxCode(code, group)
        file_check = FileCheck().check(generated_check)
        if case_name.startswith("variance_like"):
            file_check.check("store_finalize=").check("prepass_combine='add'").check(
                "prepass_finalize='mean'"
            )
        elif checks_max:
            file_check.check("store_finalize=").check("prepass_combine='max'")
        file_check.run(code)

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
        self.assertIn("local_reduce_prepass", code)
        FileCheck().check("local_reduce=FlexGemmEpiModLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 1)
        ).check("feeds_main=True").check("combine='add'").check(
            "prepass_combine='add'"
        ).run(code)

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
        FileCheck().check("local_reduce=FlexGemmEpiModLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 0)
        ).check("out=").check("feeds_main=True").check("combine=").run(code)

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
    def test_mm_local_m_reduce_feed_main_specializes_group_across_functions(self):
        torch._dynamo.reset()
        m, n = 128, 64

        def make_fn(group):
            def epilogue_fn(acc, group=group):
                x = acc.float().view(-1, group, n)
                scale = x.sum(1, keepdim=True) + 1.0
                return (x * scale.reciprocal()).view(m, n)

            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK"},
                )

            return fn, epilogue_fn

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        for group in (8, 16):
            fn, epilogue_fn = make_fn(group)
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
    @parametrize("axis", (0, 1))
    def test_mm_local_reduce_specializes_explicit_group_across_functions(self, axis):
        torch._dynamo.reset()
        m = n = 128

        def make_fn(group):
            def epilogue_fn(acc, group=group):
                if axis == 1:
                    x = acc.float().view(m, n // group, group)
                    scale = x.sum(-1, keepdim=True) + 1.0
                else:
                    x = acc.float().view(m // group, group, n)
                    scale = x.sum(1, keepdim=True) + 1.0
                return (x * scale.reciprocal()).view(m, n)

            def fn(a, b):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={"backend": "QUACK"},
                )

            return fn, epilogue_fn

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16)
        for group in (8, 16):
            fn, epilogue_fn = make_fn(group)
            actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
            self.assertMatchesEpilogue(
                actual,
                epilogue_fn(a @ b),
                epilogue_fn(a.double() @ b.double()),
                a.shape[1],
            )

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
        self.assertIn("local_reduce_prepass", code)
        FileCheck().check("local_reduce=FlexGemmEpiModLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 1)
        ).check("feeds_main=True").check("prepass_combine='add'").check(
            "prepass_finalize='mean'"
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
        FileCheck().check("torch.float8_e4m3fn").run(code)

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
        FileCheck().check("local_reduce=FlexGemmEpiModLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 0)
        ).check("out=").check("feeds_main=True").check("combine='add'").run(code)

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
        self.assertIn("local_reduce_prepass", code)
        FileCheck().check("FlexGemmEpiModLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 1)
        ).check("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case", (("tile_n64", 64), ("tile_n256", 256)), name_fn=lambda case: case[0]
    )
    def test_mm_fragment_group32_forced_config_extremes_match_reference(self, case):
        from torch._vendor.quack.gemm_config import GemmConfig

        torch._dynamo.reset()
        _, tile_n = case
        m, n, k, group = 128, 256, 64, 32

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return (x * x.sum(-1, keepdim=True).reciprocal()).view(m, n)

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=128,
                tile_n=tile_n,
                pingpong=False,
                cluster_m=2,
                cluster_n=2,
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

        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
        accumulator = a.double() @ b.double()
        grouped = accumulator.view(m, -1, group)
        expected = (grouped * grouped.sum(-1, keepdim=True).reciprocal()).view(m, n)
        wrong_group = accumulator.view(m, -1, group // 2)
        wrong_expected = (
            wrong_group * wrong_group.sum(-1, keepdim=True).reciprocal()
        ).view(m, n)
        torch.testing.assert_close(actual.double(), expected, atol=1e-4, rtol=1e-4)
        self.assertGreater((actual.double() - wrong_expected).abs().max(), 1e-2)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    @parametrize(
        "case",
        (
            ("tile128", 128, 128, 1, 1),
            ("tile256_clustered", 256, 256, 2, 1),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_physical_feed_main_forced_config_extremes_match_reference(self, case):
        from torch._vendor.quack.gemm_config import GemmConfig

        torch._dynamo.reset()
        _, tile_m, tile_n, cluster_m, cluster_n = case
        m = n = 256
        group = 32

        def epilogue_fn(acc):
            x = acc.float().view(m // group, group, n)
            return (x * x.sum(1, keepdim=True).reciprocal()).view(m, n)

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=tile_m,
                tile_n=tile_n,
                pingpong=False,
                cluster_m=cluster_m,
                cluster_n=cluster_n,
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

        a = torch.rand(m, 64, device="cuda", dtype=torch.bfloat16) + 0.5
        b = torch.rand(64, n, device="cuda", dtype=torch.bfloat16) + 0.5
        actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
        accumulator = a.double() @ b.double()
        grouped = accumulator.view(m // group, group, n)
        expected = (grouped * grouped.sum(1, keepdim=True).reciprocal()).view(m, n)
        torch.testing.assert_close(actual.double(), expected, atol=1e-4, rtol=1e-4)

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
        self.assertIn("local_reduce_prepass", code)
        FileCheck().check("FlexGemmEpiModLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 1)
        ).check("feeds_main=True").check("torch.float8_e4m3fn").run(code)

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
        FileCheck().check("feeds_main=True").run(code)

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
        FileCheck().check("tuned=True").check(
            "local_reduce=FlexGemmEpiModLocalReducePlan"
        ).check(self.localReduceGeometryPattern(group, 1)).check(
            "feeds_main=True"
        ).check("torch.float8_e4m3fn").run(code)

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
        FileCheck().check("local_reduce=FlexGemmEpiModLocalReducePlan").check(
            self.localReduceGeometryPattern(group, 1)
        ).check("feeds_main=True").run(code)

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_local_n_reduce_prepass_rejects_bool_capture(self):
        m, k, n, group = 128, 64, 128, 16

        def epilogue_fn(acc, mask):
            x = acc.float().view(m, -1, group)
            selected = torch.where(mask.view_as(x), x, x * 0.5)
            scale = selected.sum(-1, keepdim=True)
            return (x * scale.reciprocal()).view(m, n)

        def fn(a, b, mask):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, mask),
                kernel_options={"backend": "QUACK"},
            )

        a = torch.rand(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.rand(k, n, device="cuda", dtype=torch.bfloat16)
        mask = torch.randint(0, 2, (m, n), device="cuda", dtype=torch.bool)
        with self.assertRaisesRegex(
            Exception, "accumulator prepasses do not support captured bool tensors"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b, mask)

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

            first_options = {"backend": "QUACK", "tuned": tuned}
            if tuned:
                first_options["config"] = {
                    "tile_m": 128,
                    "tile_n": 128,
                    "cluster_m": 1,
                    "cluster_n": 1,
                    "swap_ab": False,
                }
            h2, partial_mean_square = flex_gemm(
                torch.mm,
                (a, b1),
                first_epilogue,
                kernel_options=first_options,
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
        self.assertEqual(code.count("flex_gemm_runtime("), 2)
        self.assertIn("local_reduce=FlexGemmEpiModLocalReducePlan", code)
        self.assertIn(
            f"FlexGemmLocalReduceGeometry(group={group}, axis=1)",
            code,
        )
        self.assertIn("combine=", code)
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

        def fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                kernel_options={"backend": "QUACK"},
            )

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

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
    def test_addmm_column_major_c_matches_reference(self):
        m, n, k = 128, 192, 64
        bias = torch.randn(n, m, device="cuda", dtype=torch.bfloat16).t()
        a = torch.randn(k, m, device="cuda", dtype=torch.bfloat16).t()
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return acc.relu()

        def fn(bias, a, b):
            return flex_gemm(
                torch.addmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 1.25, "alpha": 0.5},
                kernel_options={"backend": "QUACK"},
            )

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.addmm(bias, a, b, beta=1.25, alpha=0.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=1.25, alpha=0.5)
            ),
            k,
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
        FileCheck().check("@cute.jit").check("fragmentwise=True").check_not(
            "epi_math"
        ).run(code)

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
    def test_mm_generated_code_reads_captured_fragment_epilogue_arg(self, case):
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
            code,
            "epilogue_args=",
            f"epilogue_arg_kinds=('{kind}',)",
        )

    @skipIfNoCuteDSL
    @unittest.skipIf(not TEST_CUDA, "CUDA required")
    @unittest.skipIf(not SM100OrLater, "SM100+ required")
    def test_mm_generated_code_reads_multiple_captured_fragment_epilogue_args(self):
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

        with self.limitEpiModAutotune(a.device):
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

        with self.limitEpiModAutotune(a.device):
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
    def test_baddbmm_beta_zero_ignores_nan_c(self):
        batch, m, n, k = 2, 128, 128, 64
        bias = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        a = torch.randn(batch, m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(batch, k, n, device="cuda", dtype=torch.bfloat16)

        def epilogue_fn(acc):
            return acc.relu()

        def fn(bias, a, b):
            return flex_gemm(
                torch.baddbmm,
                (bias, a, b),
                epilogue_fn,
                gemm_kwargs={"beta": 0, "alpha": 1.5},
                kernel_options={"backend": "QUACK"},
            )

        actual = torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

        self.assertFalse(torch.isnan(actual).any())
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.baddbmm(bias, a, b, beta=0, alpha=1.5)),
            epilogue_fn(
                torch.baddbmm(bias.double(), a.double(), b.double(), beta=0, alpha=1.5)
            ),
            k,
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
        self.assertFlexGemmGeneratedCode(code)

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
        self.assertFlexGemmGeneratedCode(code)

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

        with self.limitEpiModAutotune(a.device):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(torch.bmm(a, b)),
            epilogue_fn(torch.bmm(a.double(), b.double())),
            a.shape[-1],
        )
        self.assertFlexGemmGeneratedCode(code)

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
        self.assertFlexGemmGeneratedCode(code, "C=")

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

        with self.limitEpiModAutotune(a.device):
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
                "unknown GemmConfig constraint",
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
class TestFlexGemmTransposedOutputDevice(FlexGemmTestCase):
    def test_mm_tuple_aux_local_m_reduce_contiguous_transpose(self, device):
        m, n, k, group = 256, 192, 64, 128
        normalized_input = self.makeTensor(m, n, device=device)
        incoming = self.makeTensor(m, n, device=device)
        row_scale = torch.rand(m, 1, device=device, dtype=torch.float32) + 0.5
        gamma = torch.rand(1, n, device=device, dtype=torch.float32) + 0.5
        zdz = torch.rand(m, 1, device=device, dtype=torch.float32) * 0.1

        def epilogue_fn(acc):
            grad = acc.float()
            normalized = normalized_input.float() * row_scale
            output = incoming.float() + (grad * gamma - normalized * zdz) * row_scale
            partial = (grad * normalized).view(-1, group, n).sum(1)
            return (
                output.to(acc.dtype),
                (normalized * gamma).to(acc.dtype),
                partial.mT.contiguous(),
            )

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = self.makeTensor(m, k, device=device)
        b = self.makeTensor(k, n, device=device)
        with self.limitEpiModAutotune(a.device):
            (actual, normalized, dw), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )
        expected = epilogue_fn(a @ b)
        high_precision_grad = a.double() @ b.double()
        high_precision_normalized = normalized_input.double() * row_scale.double()
        high_precision_output = (
            incoming.double()
            + (
                high_precision_grad * gamma.double()
                - high_precision_normalized * zdz.double()
            )
            * row_scale.double()
        )
        self.assertMatchesLowPrecisionEager(
            actual, expected[0], high_precision_output, k
        )
        self.assertMatchesLowPrecisionEager(
            normalized,
            expected[1],
            high_precision_normalized * gamma.double(),
            1,
        )
        self.assertEqual(
            dw,
            (high_precision_grad * high_precision_normalized)
            .view(-1, group, n)
            .sum(1)
            .mT.float(),
            atol=5e-3,
            rtol=5e-3,
        )
        self.assertTrue(dw.is_contiguous())
        self.assertEqual(dw.stride(), expected[2].stride())
        self.assertEqual(dw.shape, (n, m // group))
        FileCheck().check("tuned=True").check(
            "output_layout=flex_gemm_output_layout.TRANSPOSED"
        ).check_not("extern_kernels.mm").run(code)
        self.assertLocalReduceAuxCode(code, group, axis=0)

    @parametrize(
        "case",
        (
            ("axis_m_singleton_t", "t", 0, 128, 192, 128),
            ("axis_n_two_groups", "transpose", 1, 128, 256, 128),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_contiguous_transpose(self, device, case):
        _, form, axis, m, n, group = case
        k = 64
        tile = torch.randn(m, n, device=device, dtype=torch.float32) * 0.02
        config = {
            "tile_m": 256,
            "tile_n": 256,
            "cluster_m": 2,
            "cluster_n": 1,
            "pingpong": False,
            "is_dynamic_persistent": True,
            "swap_ab": False,
        }

        def epilogue_fn(acc):
            value = acc.float() * tile
            reduced = (
                value.view(-1, group, n).sum(1)
                if axis == 0
                else value.view(m, -1, group).sum(-1)
            )
            match form:
                case "t":
                    transposed = reduced.t()
                case "transpose":
                    transposed = reduced.transpose(0, 1)
            return acc.relu(), transposed.contiguous()

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = self.makeTensor(m, k, device=device)
        b = self.makeTensor(k, n, device=device)
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected = epilogue_fn(a.double() @ b.double())
        self.assertMatchesLowPrecisionEager(
            actual, epilogue_fn(a @ b)[0], expected[0], k
        )
        self.assertEqual(aux, expected[1].float(), atol=5e-3, rtol=5e-3)
        self.assertTrue(aux.is_contiguous())
        self.assertEqual(aux.stride(), expected[1].stride())
        expected_shape = (n, m // group) if axis == 0 else (n // group, m)
        self.assertEqual(aux.shape, expected_shape)
        FileCheck().check("output_layout=flex_gemm_output_layout.TRANSPOSED").check_not(
            "extern_kernels.mm"
        ).run(code)
        self.assertLocalReduceAuxCode(code, group, axis=axis)


instantiate_device_type_tests(
    TestFlexGemmTransposedOutputDevice, globals(), only_for="cuda"
)


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
        def epilogue_fn(acc):
            return (acc + 1).relu()

        config = next(
            config for config in self.quackGemmConfigs(device) if config.swap_ab
        )
        config_key = self.quackConfigKey(config)

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
        self.assertIn("gemm_epimod as flex_gemm_runtime", code)
        self.assertIn(f"config_constraints={tuple(sorted(config_key))!r}", code)

    @parametrize("tuned", (False, True))
    def test_mm_partial_config_matches_reference(self, device, tuned):
        def epilogue_fn(acc):
            return (acc + 1).relu()

        config = next(
            config for config in self.quackGemmConfigs(device) if config.swap_ab
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
        with self.limitEpiModAutotune(device):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b),
            epilogue_fn(a.double() @ b.double()),
            a.shape[1],
        )
        self.assertIn("gemm_epimod as flex_gemm_runtime", code)
        self.assertIn("config_constraints=", code)
        for item in pinned.items():
            self.assertIn(repr(item), code)

    def test_mm_emits_flex_gemm_debug_report(self, device):
        import logging

        from torch._inductor.kernel.flex_gemm.debug import flex_gemm_log

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                torch.relu,
                kernel_options={"backend": "QUACK"},
            )

        a = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
        b = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
        with self.assertLogs(flex_gemm_log, level="DEBUG") as records:
            actual = torch.compile(fn, backend="inductor", fullgraph=True)(a, b)
        self.assertEqual(actual, torch.relu(a @ b))

        concise = "\n".join(
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
        positions = tuple(concise.index(phase) for phase in phases)
        self.assertEqual(positions, tuple(sorted(positions)))
        self.assertIn("gemm_op: aten.mm.default", concise)
        self.assertIn("outputs:\n  main: relu", concise)
        self.assertIn("native config selection: QuACK-owned", concise)
        self.assertNotIn("GENERATED EPILOGUE", concise)

        verbose = "\n".join(
            record.getMessage()
            for record in records.records
            if record.levelno == logging.DEBUG
        )
        self.assertIn(" ===== ANALYSIS DETAILS =====", verbose)
        self.assertIn(" ===== GENERATED EPILOGUE =====", verbose)
        self.assertIn("@cute.jit", verbose)

    def test_addmm_swap_ab_matches_non_swap_and_reference(self, device):
        m, n, k = 128, 192, 64
        bias = self.makeTensor(m, n, device=device)
        a = self.makeTensor(m, k, device=device)
        b = self.makeTensor(k, n, device=device)

        def epilogue_fn(acc):
            return acc.relu()

        def run(swap_ab):
            def fn(bias, a, b):
                return flex_gemm(
                    torch.addmm,
                    (bias, a, b),
                    epilogue_fn,
                    gemm_kwargs={"beta": 0.5, "alpha": 1.5},
                    kernel_options={
                        "backend": "QUACK",
                        "config": {"swap_ab": swap_ab},
                    },
                )

            torch._dynamo.reset()
            return torch.compile(fn, backend="inductor", fullgraph=True)(bias, a, b)

        swapped = run(True)
        non_swapped = run(False)
        self.assertEqual(swapped, non_swapped)
        self.assertMatchesLowPrecisionEager(
            swapped,
            epilogue_fn(torch.addmm(bias, a, b, beta=0.5, alpha=1.5)),
            epilogue_fn(
                torch.addmm(bias.double(), a.double(), b.double(), beta=0.5, alpha=1.5)
            ),
            k,
        )

    def test_mm_swap_ab_captures_and_aux_match_non_swap(self, device):
        m, n, k = 128, 192, 64
        a = self.makeTensor(m, k, device=device)
        b = self.makeTensor(k, n, device=device)
        col_bias = self.makeTensor(m, 1, device=device, dtype=torch.float32)
        row_scale = self.makeTensor(1, n, device=device, dtype=torch.float32)
        tile_bias = self.makeTensor(m, n, device=device, dtype=torch.float32)
        scalar = self.makeTensor(1, 1, device=device, dtype=torch.float32)

        def epilogue_fn(acc):
            biased = ((acc.float() + col_bias) * row_scale + tile_bias) * scalar
            return biased.relu(), acc.float() * row_scale + tile_bias

        def run(swap_ab):
            def fn(a, b, col_bias, row_scale, tile_bias, scalar):
                return flex_gemm(
                    torch.mm,
                    (a, b),
                    epilogue_fn,
                    kernel_options={
                        "backend": "QUACK",
                        "config": {"swap_ab": swap_ab},
                    },
                )

            torch._dynamo.reset()
            return torch.compile(fn, backend="inductor", fullgraph=True)(
                a, b, col_bias, row_scale, tile_bias, scalar
            )

        swapped, swapped_aux = run(True)
        non_swapped, non_swapped_aux = run(False)
        self.assertEqual(swapped, non_swapped)
        self.assertEqual(swapped_aux, non_swapped_aux)
        low_precision_acc = a @ b
        high_precision_acc = a.double() @ b.double()
        self.assertMatchesLowPrecisionEager(
            swapped,
            epilogue_fn(low_precision_acc)[0],
            (
                (
                    (high_precision_acc + col_bias.double()) * row_scale.double()
                    + tile_bias.double()
                )
                * scalar.double()
            ).relu(),
            k,
        )
        self.assertMatchesLowPrecisionEager(
            swapped_aux,
            epilogue_fn(low_precision_acc)[1],
            high_precision_acc * row_scale.double() + tile_bias.double(),
            k,
        )

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_tuple_aux_blocked_128x4_local_n_reduce(self, device):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = k = 256
        group = 32

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            return acc, to_blocked(grouped.amax(-1))

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
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

        a = torch.eye(m, k, device=device, dtype=torch.bfloat16)
        rows = torch.arange(k, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        b = (1 + (rows // 128) * 16 + ((cols // group) // 4) * 4).to(torch.bfloat16)
        (actual, blocked), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected, expected_blocked = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(blocked, expected_blocked)
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        self.assertNotIn("flex_gemm.to_blocked", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize("case", (("mx", 32), ("nvfp4", 16)), name_fn=lambda case: case[0])
    def test_mm_tuple_aux_quant_blocked_128x4_local_n_reduce(self, device, case):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        case_name, group = case
        scale_fn = mx_e8m0_scale if case_name == "mx" else nvfp4_e4m3_scale
        m = n = k = 256

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            return acc, to_blocked(scale_fn(grouped.abs().amax(-1)))

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
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

        a = torch.eye(m, k, device=device, dtype=torch.bfloat16)
        rows = torch.arange(k, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        b = (2.0 ** ((rows % 4) + ((cols // group) % 4))).to(torch.bfloat16)
        (actual, blocked), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected, expected_blocked = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(blocked.view(torch.uint8), expected_blocked.view(torch.uint8))
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        if case_name == "mx":
            self.assertMxScaleCode(code)
        else:
            self.assertNvfp4ScaleCode(code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_mx_quant_blocked_output_feeds_scaled_mm(self, device):
        import torch.nn.functional as F
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        m = hidden = output = k = 256
        group = 32

        def quantize(x):
            grouped = x.float().view(x.shape[0], -1, group)
            scale = mx_e8m0_scale(grouped.abs().amax(-1, keepdim=True))
            quantized = (grouped * scale.float().reciprocal()).view_as(x)
            scale = scale.squeeze(-1)
            return (
                quantized.clamp(-448.0, 448.0).to(torch.float8_e4m3fn),
                scale,
                to_blocked(scale),
            )

        def epilogue_fn(acc):
            quantized, _, blocked_scale = quantize(acc)
            return quantized, blocked_scale

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
                device_capacity=10,
            )
        )

        def fn(a, b, weight, weight_scale):
            activation, activation_scale = flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )
            return F.scaled_mm(
                activation,
                weight.t(),
                scale_a=activation_scale,
                scale_recipe_a=F.ScalingType.BlockWise1x32,
                scale_b=weight_scale,
                scale_recipe_b=F.ScalingType.BlockWise1x32,
                swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                output_dtype=torch.bfloat16,
            )

        a = torch.eye(m, k, device=device, dtype=torch.bfloat16)
        rows = torch.arange(k, device=device)[:, None]
        cols = torch.arange(hidden, device=device)[None, :]
        exponent = (rows // 128) * 2 + ((cols // group) // 4) - 2
        b = (2.0**exponent).to(torch.bfloat16)
        weight_hp = torch.randn(output, hidden, device=device, dtype=torch.bfloat16)
        weight, weight_scale, weight_scale_blocked = quantize(weight_hp)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            weight,
            weight_scale_blocked,
        )
        activation, activation_scale, _ = quantize(a @ b)
        activation_dequant = (
            activation.float()
            * activation_scale.float().repeat_interleave(group, dim=-1)
        )
        weight_dequant = weight.float() * weight_scale.float().repeat_interleave(
            group, dim=-1
        )
        expected = (activation_dequant @ weight_dequant.t()).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=1.0)
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        self.assertIn("_scaled_mm", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_nvfp4_quant_blocked_output_feeds_scaled_mm(self, device):
        import torch.nn.functional as F
        from torch._higher_order_ops.flex_gemm import nvfp4_pack, to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        m = hidden = output = k = 256
        group = 16

        def quantize(x):
            grouped = x.float().view(x.shape[0], -1, group)
            scale = nvfp4_e4m3_scale(grouped.abs().amax(-1, keepdim=True))
            normalized = grouped * scale.float().reciprocal()
            packed = nvfp4_pack(normalized.view(x.shape[0], -1, 2))
            scale = scale.squeeze(-1)
            return packed, scale, to_blocked(scale)

        def epilogue_fn(acc):
            packed, _, blocked_scale = quantize(acc)
            return packed, blocked_scale

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
                device_capacity=10,
            )
        )

        def fn(a, b, weight, weight_scale):
            activation_storage, activation_scale = flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )
            return F.scaled_mm(
                activation_storage.view(torch.float4_e2m1fn_x2),
                weight.t(),
                scale_a=activation_scale,
                scale_recipe_a=F.ScalingType.BlockWise1x16,
                scale_b=weight_scale,
                scale_recipe_b=F.ScalingType.BlockWise1x16,
                swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
                output_dtype=torch.bfloat16,
            )

        a = torch.eye(m, k, device=device, dtype=torch.bfloat16)
        rows = torch.arange(k, device=device)[:, None]
        cols = torch.arange(hidden, device=device)[None, :]
        b = (2.0 ** (((rows + cols) % 7) - 3)).to(torch.bfloat16)
        weight_hp = torch.randn(output, hidden, device=device, dtype=torch.bfloat16)
        weight_storage, _, weight_scale = quantize(weight_hp)
        weight = weight_storage.view(torch.float4_e2m1fn_x2)

        actual, (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True),
            a,
            b,
            weight,
            weight_scale,
        )
        activation_storage, _, activation_scale = quantize(a @ b)
        expected = F.scaled_mm(
            activation_storage.view(torch.float4_e2m1fn_x2),
            weight.t(),
            scale_a=activation_scale,
            scale_recipe_a=F.ScalingType.BlockWise1x16,
            scale_b=weight_scale,
            scale_recipe_b=F.ScalingType.BlockWise1x16,
            swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        self.assertIn("_scaled_mm", code)
        self.assertIn("GroupedMainOutputTransform(group=2", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_tuple_aux_blocked_128x4_zero_fills_padding(self, device):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        m, n, k, group = 129, 80, 256, 16

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            return acc, to_blocked(grouped.amax(-1))

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
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

        a = torch.eye(m, k, device=device, dtype=torch.bfloat16)
        rows = torch.arange(k, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        b = (1 + (rows % 4) + ((cols // group) % 4)).to(torch.bfloat16)
        (actual, blocked), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected, expected_blocked = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(blocked, expected_blocked)
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        self.assertNotIn("triton_poi_fused", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize("blocked", (False, True))
    def test_mm_grouped_main_with_local_reduce_output(self, device, blocked):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        m = n = k = 256
        reduce_group = 16

        def epilogue_fn(acc):
            pairs = acc.float().view(m, -1, 2)
            grouped = acc.float().view(m, -1, reduce_group)
            scale = nvfp4_e4m3_scale(grouped.abs().amax(-1))
            return (
                (pairs[..., 0] - pairs[..., 1]).to(acc.dtype),
                to_blocked(scale) if blocked else scale,
            )

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
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

        a = torch.eye(m, k, device=device, dtype=torch.bfloat16)
        rows = torch.arange(k, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        b = (2.0 ** (((rows + cols) % 7) - 3)).to(torch.bfloat16)
        (actual, scale), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )
        expected, expected_scale = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.5)
        self.assertEqual(scale.view(torch.uint8), expected_scale.view(torch.uint8))
        self.assertIn("GroupedMainOutputTransform(group=2", code)
        self.assertIn("local_reduce=FlexGemmEpiModLocalReducePlan", code)
        if blocked:
            self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_tuple_aux_blocked_128x4_tuned_multitile(self, device):
        from torch._higher_order_ops.flex_gemm import to_blocked

        m = n = 512
        k, group = 64, 32

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            return acc, to_blocked(grouped.amax(-1))

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = torch.zeros(m, k, device=device, dtype=torch.bfloat16)
        a[:k] = torch.eye(k, device=device, dtype=torch.bfloat16)
        rows = torch.arange(k, device=device)[:, None]
        cols = torch.arange(n, device=device)[None, :]
        b = (1 + (rows % 4) + ((cols // group) % 4)).to(torch.bfloat16)
        with self.limitEpiModAutotune(device):
            (actual, blocked), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )
        expected, expected_blocked = epilogue_fn(a @ b)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(blocked, expected_blocked)
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        self.assertIn("tuned=True", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    def test_mm_tuple_aux_blocked_128x4_dynamic_shapes(self, device):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        k, group = 256, 16

        def epilogue_fn(acc):
            grouped = acc.float().view(acc.shape[0], -1, group)
            return acc, to_blocked(grouped.amax(-1))

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=True,
                cluster_m=2,
                cluster_n=1,
                swap_ab=False,
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

        initial_a = torch.eye(128, k, device=device, dtype=torch.bfloat16)
        initial_b = torch.randn(k, 64, device=device, dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(initial_a, 0)
        torch._dynamo.mark_dynamic(initial_b, 1)
        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        for a, b in (
            (initial_a, initial_b),
            (
                torch.eye(129, k, device=device, dtype=torch.bfloat16),
                torch.randn(k, 80, device=device, dtype=torch.bfloat16),
            ),
            (
                torch.eye(256, k, device=device, dtype=torch.bfloat16),
                torch.randn(k, 128, device=device, dtype=torch.bfloat16),
            ),
        ):
            actual, blocked = compiled(a, b)
            expected, expected_blocked = epilogue_fn(a @ b)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(blocked, expected_blocked)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize("quantized", (False, True))
    def test_mm_tuple_aux_blocked_128x4_supports_swap_ab(self, device, quantized):
        from torch._higher_order_ops.flex_gemm import to_blocked
        from torch._vendor.quack.gemm_config import GemmConfig

        m, n, k = 257, 288, 64
        group = 16 if quantized else 32

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            scale = grouped.amax(-1)
            if quantized:
                scale = nvfp4_e4m3_scale(scale)
            return acc, to_blocked(scale)

        def run(a, b, swap_ab):
            config = dataclasses.asdict(
                GemmConfig(
                    tile_m=256,
                    tile_n=256,
                    pingpong=False,
                    is_dynamic_persistent=True,
                    cluster_m=2,
                    cluster_n=1,
                    swap_ab=swap_ab,
                    device_capacity=10,
                )
            )
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK", "config": config},
            )

        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        expected = torch.compile(
            lambda a, b: run(a, b, False), backend="inductor", fullgraph=True
        )(a, b)
        actual, (code,) = run_and_get_code(
            torch.compile(
                lambda a, b: run(a, b, True), backend="inductor", fullgraph=True
            ),
            a,
            b,
        )
        self.assertEqual(actual, expected)
        self.assertIn("flex_gemm_output_layout.BLOCKED_128X4", code)
        self.assertIn("fragmentwise=True", code)
        self.assertIn("('swap_ab', True)", code)

    def test_mm_tuple_aux_blocked_128x4_rejects_axis_m(self, device):
        from torch._higher_order_ops.flex_gemm import to_blocked

        m = n = 128
        group = 16

        def epilogue_fn(acc):
            grouped = acc.float().view(-1, group, n)
            return acc, to_blocked(grouped.amax(1))

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, 64, device=device)
        b = self.makeTensor(64, n, device=device)
        with self.assertRaisesRegex(
            Exception, "blocked local-reduce outputs currently support only axis 1"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_mm_tuple_aux_blocked_128x4_rejects_intermediate_transform(self, device):
        from torch._higher_order_ops.flex_gemm import to_blocked

        m = n = 128
        group = 16

        def epilogue_fn(acc):
            grouped = acc.float().view(m, -1, group)
            return acc, to_blocked(grouped.amax(-1)) + 1

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={"backend": "QUACK"},
            )

        a = self.makeTensor(m, 64, device=device)
        b = self.makeTensor(64, n, device=device)
        with self.assertRaisesRegex(
            Exception, "output layout transforms must be returned directly"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @parametrize("tuned", (False, True))
    def test_mm_swap_ab_supports_local_n_reduce(self, device, tuned):
        m = n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(m, -1, group)
            return acc.relu(), x.sum(-1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={
                    "backend": "QUACK",
                    "config": {"swap_ab": True},
                    "tuned": tuned,
                },
            )

        a = self.makeTensor(m, 64, device=device)
        b = self.makeTensor(64, n, device=device)
        with self.limitEpiModAutotune(device):
            (actual, aux), (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True), a, b
            )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertIn("@cute.jit", code)
        self.assertIn("fragmentwise=True", code)
        self.assertIn("('swap_ab', True)", code)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize(
        "case",
        (
            ("local_n_g16", 16, True),
            ("local_n_g64", 64, True),
            ("local_n_g128_nonpersistent", 128, False),
        ),
        name_fn=lambda case: case[0],
    )
    def test_mm_tuple_aux_local_n_reduce_supports_swap_ab(self, device, case):
        from torch._vendor.quack.gemm_config import GemmConfig

        _, group, dynamic = case
        m, n = 512, 256

        def epilogue_fn(acc):
            partial = acc.float().view(m, -1, group).mean(-1)
            return acc.relu(), partial

        config = dataclasses.asdict(
            GemmConfig(
                tile_m=256,
                tile_n=256,
                pingpong=False,
                is_dynamic_persistent=dynamic,
                cluster_m=2,
                cluster_n=1,
                swap_ab=True,
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
        (actual, aux), (code,) = run_and_get_code(
            torch.compile(fn, backend="inductor", fullgraph=True), a, b
        )

        self.assertLocalReduceAuxMatches(actual, aux, a, b, epilogue_fn)
        self.assertIn("@cute.jit", code)
        self.assertIn("fragmentwise=True", code)
        self.assertIn("('swap_ab', True)", code)
        self.assertIn(f"group={group}", code)

    def test_mm_swap_ab_rejects_unaligned_n(self, device):
        m, n = 128, 293

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: acc.float().relu(),
                kernel_options={
                    "backend": "QUACK",
                    "config": {"swap_ab": True},
                },
            )

        a = self.makeTensor(m, 64, device=device)
        b = self.makeTensor(64, n, device=device)
        with self.assertRaisesRegex(ValueError, "no .*config_constraints.*swap_ab"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_mm_tuned_swap_candidate_captured_args_matches_reference(self, device):
        import torch._vendor.quack.gemm_runtime.autotune as epi_autotune

        m = n = 256
        swap_config = next(
            config
            for config in self.quackGemmConfigs(device)
            if config.swap_ab
            and config.tile_m == 256
            and config.tile_n == 256
            and config.cluster_m == 2
            and config.cluster_n == 1
        )

        def epilogue_fn(acc, row, col):
            return (acc.float() + row) * col

        def fn(a, b, row, col):
            return flex_gemm(
                torch.mm,
                (a, b),
                lambda acc: epilogue_fn(acc, row, col),
                kernel_options={"backend": "QUACK", "tuned": True},
            )

        a = self.makeTensor(m, 64, device=device)
        b = self.makeTensor(64, n, device=device)
        row = self.makeTensor(1, n, device=device, dtype=torch.float32)
        col = self.makeTensor(m, 1, device=device, dtype=torch.float32)
        with (
            mock.patch.object(
                epi_autotune, "_config_space", return_value=(swap_config,)
            ),
            mock.patch.object(epi_autotune, "_MOD_TUNERS", {}),
        ):
            actual, (code,) = run_and_get_code(
                torch.compile(fn, backend="inductor", fullgraph=True),
                a,
                b,
                row,
                col,
            )
        self.assertMatchesLowPrecisionEager(
            actual,
            epilogue_fn(a @ b, row, col),
            epilogue_fn(a.double() @ b.double(), row.double(), col.double()),
            a.shape[1],
        )
        self.assertIn("tuned=True", code)

    def test_mm_swap_ab_rejects_local_n_reduce_feed_main(self, device):
        m = n = 128
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
                kernel_options={
                    "backend": "QUACK",
                    "config": {"swap_ab": True},
                },
            )

        a = self.makeTensor(m, 64, device=device)
        b = self.makeTensor(64, n, device=device)
        with self.assertRaisesRegex(ValueError, "no .*config_constraints.*swap_ab"):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    def test_mm_swap_ab_rejects_local_m_reduce(self, device):
        m = n = 128
        group = 16

        def epilogue_fn(acc):
            x = acc.float().view(-1, group, n)
            return acc.relu(), x.sum(1)

        def fn(a, b):
            return flex_gemm(
                torch.mm,
                (a, b),
                epilogue_fn,
                kernel_options={
                    "backend": "QUACK",
                    "config": {"swap_ab": True},
                },
            )

        a = self.makeTensor(m, 64, device=device)
        b = self.makeTensor(64, n, device=device)
        with self.assertRaisesRegex(
            ValueError, "no supported GemmConfig matches config_constraints"
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(a, b)

    @unittest.skipIf(SM120OrLater, "SM100 config required")
    @parametrize("group", (64, 128))
    def test_mm_tuple_aux_local_n_reduce_supports_clustered_tile_m256(
        self, device, group
    ):
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
        config_key = self.quackConfigKey(expected_config)
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
        self.assertLocalReduceAuxCode(code, group)
        self.assertIn(f"config_constraints={tuple(sorted(config_key))!r}", code)

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
        FileCheck().check("tuned=True").check(
            "local_reduce=FlexGemmEpiModLocalReducePlan"
        ).check(self.localReduceGeometryPattern(group, 1)).run(code)

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
        with self.assertRaisesRegex(
            Exception,
            "no supported GemmConfig matches config_constraints",
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
        config_key = self.quackConfigKey(expected_config)
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
        self.assertLocalReduceAuxCode(code, group, axis=axis)
        self.assertIn(f"config_constraints={tuple(sorted(config_key))!r}", code)


instantiate_device_type_tests(
    TestFlexGemmExplicitConfigDevice, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
