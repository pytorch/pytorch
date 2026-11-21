# Owner(s): ["module: onnx"]
"""PyTorch-owned translations construct IR without ONNX Script evaluation."""

from __future__ import annotations

import contextlib
import operator
from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
import onnxscript
import onnxscript.evaluator
from onnx.reference import ReferenceEvaluator

from torch.onnx._internal.exporter import _ir_builder, _schemas
from torch.onnx._internal.exporter._torchlib import _torchlib_registry
from torch.onnx._internal.exporter._torchlib.ops import core
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import HardwareClassification


@contextlib.contextmanager
def no_script():
    with (
        mock.patch.object(
            _ir_builder,
            "current_builder",
            side_effect=AssertionError("ambient builder lookup"),
        ),
        mock.patch.object(
            onnxscript, "script", side_effect=AssertionError("script called")
        ),
        mock.patch.object(
            onnxscript,
            "TracedOnnxFunction",
            side_effect=AssertionError("tracing called"),
        ),
        mock.patch.object(
            onnxscript.evaluator,
            "default_as",
            side_effect=AssertionError("evaluator called"),
        ),
    ):
        yield


@common_utils.instantiate_parametrized_tests
class TestNativeTorchlib(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def evaluate(self, function, tensors, *, opset=18, **kwargs):
        graph = ir.Graph([], [], nodes=[], name="native", opset_imports={"": opset})
        builder = _ir_builder.GraphBuilder(graph, opset=opset)
        op = _ir_builder.TranslationContext(builder)
        values = [
            ir.Value(
                name=f"input_{i}",
                shape=ir.Shape(tensor.shape),
                type=ir.TensorType(ir.DataType.from_numpy(tensor.dtype)),
            )
            for i, tensor in enumerate(tensors)
        ]
        graph.inputs.extend(values)
        with no_script():
            outputs = function(op, *values, **kwargs)
        if isinstance(outputs, ir.Value):
            outputs = (outputs,)
        graph.outputs.extend(outputs)
        model = ir.Model(graph, ir_version=10)
        model.functions.update(builder.functions)
        proto = ir.to_proto(model)
        # Schema inference fills intermediate shapes, not translation-time tensor casts.
        proto = onnx.shape_inference.infer_shapes(proto, strict_mode=True)
        onnx.checker.check_model(proto)
        return ReferenceEvaluator(proto).run(
            None, dict(zip((value.name for value in values), tensors))
        )

    def test_registration_does_not_infer_or_compile(self):
        def translation(op, x: ir.Value) -> ir.Value:
            return op.builder.call("Neg", x)[0]

        with (
            no_script(),
            mock.patch.object(
                _schemas,
                "op_signature_from_function",
                side_effect=AssertionError("inference called"),
            ),
            mock.patch.object(_torchlib_registry, "_registry", []),
        ):
            self.assertIs(
                _torchlib_registry.onnx_impl(operator.neg)(translation), translation
            )
            self.assertIsNone(_torchlib_registry.get_torchlib_ops()[0].signature)

    @common_utils.parametrize("opset", [18, 20, 23])
    def test_abs_and_complex_abs(self, opset):
        x = np.array([-3.0, 0.0, 4.0], dtype=np.float32)
        self.assertEqual(self.evaluate(core.aten_abs, [x], opset=opset)[0], abs(x))
        complex_x = np.array([[-3.0, 4.0], [0.0, -2.0]], dtype=np.float32)
        self.assertEqual(
            self.evaluate(core.aten_abs_complex, [complex_x], opset=opset)[0],
            np.array([5.0, 2.0], dtype=np.float32),
        )


if __name__ == "__main__":
    common_utils.run_tests()
