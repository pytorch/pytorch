# Owner(s): ["module: onnx"]
"""Promotion at native operator call sites, without the FX promotion pass."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

import torch
from torch import _prims_common
from torch.onnx._internal.exporter import _core, _fx_passes, _ir_builder, _registration
from torch.onnx._internal.exporter._torchlib.ops import (
    _promotion,
    core,
    factories,
    neural,
    residual,
    sequences,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


def _value(tensor, name="input"):
    packed = torch.view_as_real(tensor) if tensor.is_complex() else tensor
    value = ir.Value(
        name=name,
        shape=ir.Shape(packed.shape),
        type=ir.TensorType(_core.torch_dtype_to_onnx_dtype(packed.dtype)),
    )
    value.meta["torch.onnx.original_type"] = tensor.dtype
    return value


def _evaluate(function, *args):
    graph = ir.Graph([], [], nodes=[], name="promotion", opset_imports={"": 18})
    op = _ir_builder.TranslationContext(_ir_builder.GraphBuilder(graph))
    inputs = []
    feeds = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            value = _value(arg, f"input_{i}")
            graph.inputs.append(value)
            packed = torch.view_as_real(arg) if arg.is_complex() else arg
            feeds[value.name] = packed.numpy()
            inputs.append(value)
        else:
            inputs.append(arg)
    graph.outputs.append(function(op, *inputs))
    proto = onnx.shape_inference.infer_shapes(
        ir.to_proto(ir.Model(graph, ir_version=10)), strict_mode=True
    )
    onnx.checker.check_model(proto, full_check=True)
    result = ReferenceEvaluator(proto).run(None, feeds)[0]
    return torch.from_numpy(np.asarray(result)), graph


@common_utils.instantiate_parametrized_tests
class TestPromotionDtypes(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        graph = ir.Graph([], [], nodes=[], name="dtype")
        self.op = _ir_builder.TranslationContext(_ir_builder.GraphBuilder(graph))

    @common_utils.parametrize(
        "kind", list(_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND)
    )
    def test_dtype_and_rank_lattice_without_tensor_execution(self, kind):
        dtypes = (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        tensors = [
            torch.empty(shape, device="meta", dtype=dtype)
            for dtype, shape in itertools.product(dtypes, ((), (3,)))
        ]
        values = [_value(tensor) for tensor in tensors]
        for i, j in itertools.product(range(len(tensors)), repeat=2):
            with self.subTest(a=tensors[i].dtype, b=tensors[j].dtype, i=i, j=j):
                expected = _prims_common.elementwise_dtypes(
                    tensors[i], tensors[j], type_promotion_kind=kind
                )
                with (
                    mock.patch.object(
                        torch, "empty", side_effect=AssertionError("tensor allocation")
                    ),
                    mock.patch.object(
                        torch.fx.Interpreter,
                        "run",
                        side_effect=AssertionError("graph execution"),
                    ),
                ):
                    self.assertEqual(
                        _promotion.dtypes(
                            self.op, values[i], values[j], kind=kind.name
                        ),
                        expected,
                    )

    @common_utils.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_wrapped_scalars_and_default_dtype(self, default_dtype):
        previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous)
        torch.set_default_dtype(default_dtype)
        for dtype, shape, scalar in itertools.product(
            (torch.bool, torch.int32, torch.float16, torch.float64, torch.complex64),
            ((), (2,)),
            (True, 2, 1.5, 1 + 2j),
        ):
            tensor = torch.empty(shape, dtype=dtype, device="meta")
            value = _value(tensor)
            expected = _prims_common.elementwise_dtypes(
                tensor,
                scalar,
                type_promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            )
            self.assertEqual(_promotion.dtypes(self.op, value, scalar), expected)
        self.assertEqual(
            _promotion.dtypes(self.op, 1, 2, kind="INT_TO_FLOAT")[1], default_dtype
        )

    def test_unknown_rank_is_explicit(self):
        x = ir.Value(type=ir.TensorType(ir.DataType.FLOAT))
        y = ir.Value(type=ir.TensorType(ir.DataType.DOUBLE))
        with self.assertRaisesRegex(ValueError, "requires known ranks"):
            _promotion.dtypes(self.op, x, y)


class TestNativePromotion(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    @common_utils.parametrize(
        "name,other_kind",
        [
            (name, kind)
            for name, kind in itertools.product(
                [
                    "add",
                    "sub",
                    "mul",
                    "div",
                    "eq",
                    "lt",
                    "maximum",
                    "minimum",
                    "logical_and",
                    "pow",
                ],
                ["scalar", "zero_dim", "tensor"],
            )
            if kind != "scalar" or name not in ("maximum", "minimum", "logical_and")
        ],
    )
    def test_mixed_operands(self, device, name, other_kind):
        x = torch.tensor([2, 4], dtype=torch.int32, device=device)
        other = 1.5
        if other_kind != "scalar":
            other = torch.tensor(
                1.5 if other_kind == "zero_dim" else [1.5, 2.5],
                dtype=torch.float64,
                device=device,
            )
        function = getattr(core, f"aten_{name}")
        expected = getattr(torch, name)(x, other)
        actual, _ = _evaluate(function, x, other)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)

    @common_utils.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_integral_division_and_unary_float_result(self, device, default_dtype):
        previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous)
        torch.set_default_dtype(default_dtype)
        x = torch.tensor([2, 5], dtype=torch.int32, device=device)
        for name in ("sin", "sqrt", "sigmoid"):
            function = core.aten_sin if name == "sin" else core.aten_sqrt
            if name == "sigmoid":
                function = residual.aten_sigmoid
            actual, _ = _evaluate(function, x)
            self.assertEqual(actual, getattr(torch, name)(x))
            self.assertEqual(actual.dtype, default_dtype)
        actual, _ = _evaluate(core.aten_div, x, 2)
        self.assertEqual(actual, x / 2)
        self.assertEqual(actual.dtype, default_dtype)

    @common_utils.parametrize("mode", [None, "trunc", "floor"])
    def test_rounding_mode(self, device, mode):
        x = torch.tensor([-7, 7], dtype=torch.int32, device=device)
        y = torch.tensor([2, -2], dtype=torch.int64, device=device)
        actual, _ = _evaluate(lambda op, a, b: core.aten_div_mode(op, a, b, mode), x, y)
        expected = torch.div(x, y, rounding_mode=mode)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)

    @common_utils.parametrize("name", ["add", "mul", "div", "sub"])
    @common_utils.parametrize("other_kind", ["scalar", "real", "complex"])
    def test_complex_original_dtype(self, device, name, other_kind):
        x = torch.tensor([2 + 3j, 4 - 5j], dtype=torch.complex64, device=device)
        other = 1.5
        if other_kind != "scalar":
            dtype = torch.float64 if other_kind == "real" else torch.complex128
            other = torch.tensor([2, 3], dtype=dtype, device=device)
        expected = getattr(torch, name)(x, other)
        actual, _ = _evaluate(getattr(core, f"aten_{name}_complex"), x, other)
        self.assertEqual(actual, torch.view_as_real(expected))
        self.assertEqual(actual.dtype, expected.real.dtype)

    @common_utils.parametrize("name", ["sum", "prod", "mean", "cumsum"])
    @common_utils.parametrize("dtype", [torch.float16, torch.float64])
    def test_reduction_accumulation(self, device, name, dtype):
        x = torch.tensor([1, 2, 3], dtype=dtype, device=device)
        function = getattr(core, f"aten_{name}")
        if name == "cumsum":

            def function(op, value):
                return core.aten_cumsum(op, value, 0)

        actual, graph = _evaluate(function, x)
        expected = torch.cumsum(x, 0) if name == "cumsum" else getattr(torch, name)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)
        if dtype == torch.float16:
            self.assertTrue(
                any(
                    node.op_type == "Cast"
                    and node.outputs[0].dtype == ir.DataType.FLOAT
                    for node in graph
                )
            )

    @common_utils.parametrize("name", ["sum", "prod"])
    @common_utils.parametrize("dtype", [torch.float64, torch.bool])
    def test_explicit_reduction_dtype(self, device, name, dtype):
        x = torch.tensor([1, 0, 3], dtype=torch.int32, device=device)
        function = getattr(core, f"aten_{name}")
        onnx_dtype = _core.torch_dtype_to_onnx_dtype(dtype)
        actual, _ = _evaluate(
            lambda op, value: function(op, value, dtype=onnx_dtype), x
        )
        expected = getattr(torch, name)(x, dtype=dtype)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.dtype, dtype)

    def test_mse_opmath_restores_output_dtype(self, device):
        x = torch.tensor([0.3, 0.6, 0.9], dtype=torch.float16, device=device)
        y = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float16, device=device)
        actual, graph = _evaluate(neural.aten_mse_loss, x, y)
        self.assertEqual(actual, torch.nn.functional.mse_loss(x, y))
        self.assertEqual(actual.dtype, torch.float16)
        self.assertTrue(
            any(
                node.op_type == "Cast" and node.outputs[0].dtype == ir.DataType.FLOAT
                for node in graph
            )
        )

    @common_utils.parametrize("name", ["all", "any"])
    def test_uint8_boolean_reduction(self, device, name):
        x = torch.tensor([1, 0], dtype=torch.uint8, device=device)
        actual, _ = _evaluate(getattr(core, f"aten_{name}"), x)
        self.assertEqual(actual, getattr(torch, name)(x))
        self.assertEqual(actual.dtype, torch.uint8)

    def test_bitwise_mixed_tensor_dtypes(self, device):
        x = torch.tensor([1, 3], dtype=torch.int32, device=device)
        y = torch.tensor([3, 2], dtype=torch.int64, device=device)
        actual, _ = _evaluate(core.aten_bitwise_and, x, y)
        self.assertEqual(actual, x & y)
        self.assertEqual(actual.dtype, torch.int64)

    @common_utils.parametrize(
        "mean_dtype,std_dtype",
        [(torch.float32, torch.float64), (torch.float64, torch.float32)],
    )
    def test_normal_uses_mean_dtype(self, device, mean_dtype, std_dtype):
        mean = torch.tensor([2, 3], dtype=mean_dtype, device=device)
        std = torch.zeros(2, dtype=std_dtype, device=device)
        actual, _ = _evaluate(factories.aten_normal_tensor_tensor, mean, std)
        expected = torch.normal(mean, std)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_sequence_and_where_promotion(self, device):
        x = torch.tensor([1, 2], dtype=torch.int32, device=device)
        y = torch.tensor([3, 4], dtype=torch.float64, device=device)
        actual, _ = _evaluate(lambda op, a, b: sequences.aten_cat(op, [a, b]), x, y)
        self.assertEqual(actual, torch.cat([x, y]))
        actual, _ = _evaluate(
            lambda op, condition: core.aten_where(op, condition, 0, -1000.0),
            torch.tensor([True, False], device=device),
        )
        self.assertEqual(actual, torch.tensor([0.0, -1000.0], device=device))

    @common_utils.parametrize("optimize", [False, True])
    @common_utils.parametrize("mask_kind", ["unsqueeze", "add"])
    def test_public_sdpa_export_without_promotion_graph_rerun(
        self, device, optimize, mask_kind
    ):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(96, 96)
                self.k = torch.nn.Linear(96, 96)
                self.v = torch.nn.Linear(96, 96)
                self.bias = torch.nn.Parameter(torch.randn(3, 49, 49))

            def forward(self, x):
                batch, seq, _ = x.shape
                q = self.q(x).view(batch, seq, 3, 32).transpose(1, 2)
                k = self.k(x).view(batch, seq, 3, 32).transpose(1, 2)
                v = self.v(x).view(batch, seq, 3, 32).transpose(1, 2)
                mask = (
                    self.bias.unsqueeze(0)
                    if mask_kind == "unsqueeze"
                    else self.bias + 0
                )
                result = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask
                )
                return result.transpose(1, 2).reshape(batch, seq, -1)

        model = Model().to(device).eval()
        x = torch.randn(64, 49, 96, device=device)
        registry = _registration.ONNXRegistry.from_torchlib(18)
        with torch.enable_grad():
            expected = model(x).detach()
            ep = torch.export.export(model, (x,), strict=False)
            ep = _fx_passes.decompose_with_registry(ep, registry)
            strides = {
                node.name: node.meta["val"].stride()
                for node in ep.graph.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            }
            with (
                mock.patch.object(
                    _fx_passes, "decompose_with_registry", return_value=ep
                ),
                mock.patch.object(
                    torch.fx.Interpreter,
                    "run",
                    side_effect=AssertionError("promotion graph rerun"),
                ),
            ):
                program = torch.onnx.export(
                    ep, opset_version=18, optimize=optimize, verbose=False
                )
            public_program = torch.onnx.export(
                model, (x,), opset_version=18, optimize=optimize, verbose=False
            )
        self.assertTrue(model.bias.requires_grad)
        self.assertEqual(
            strides,
            {
                node.name: node.meta["val"].stride()
                for node in ep.graph.nodes
                if isinstance(node.meta.get("val"), torch.Tensor)
            },
        )
        for result in (program, public_program):
            onnx.checker.check_model(result.model_proto, full_check=True)
            actual = ReferenceEvaluator(result.model_proto).run(None, {"x": x.numpy()})[
                0
            ]
            self.assertEqual(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)


instantiate_device_type_tests(TestNativePromotion, globals(), only_for="cpu")


if __name__ == "__main__":
    common_utils.run_tests()
