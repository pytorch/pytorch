# Owner(s): ["module: onnx"]
"""Native built-in registration and explicit context at the legacy boundary."""

from __future__ import annotations

import operator

import onnx

import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

import torch
from torch.onnx._internal.exporter import (
    _custom_translation,
    _ir_builder,
    _registration,
)
from torch.onnx._internal.exporter._torchlib import _torchlib_registry
from torch.onnx._internal.exporter._torchlib.ops import core
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestNativeRegistration(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_embedding_softmax_native_export(self):
        class EmbeddingSoftmax(torch.nn.Module):
            def forward(self, indices, weight):
                return torch.nn.functional.embedding(indices, weight).softmax(-1)

        model = EmbeddingSoftmax()
        inputs = (torch.tensor([0, 2]), torch.arange(12.0).reshape(3, 4))
        program = torch.onnx.export(
            model, inputs, opset_version=18, optimize=False, verbose=False
        )
        proto = program.model_proto
        onnx.checker.check_model(proto, full_check=True)
        feed = {
            value.name: tensor.numpy()
            for value, tensor in zip(program.model.graph.inputs, inputs)
        }
        result = ReferenceEvaluator(proto).run(None, feed)[0]
        self.assertEqual(torch.from_numpy(result), model(*inputs))

    def test_decorator_returns_plain_function(self):
        size = len(_torchlib_registry._registry)
        self.addCleanup(
            lambda: _torchlib_registry._registry.__delitem__(slice(size, None))
        )

        def identity(op, value):
            return op.Identity(value)

        result = _torchlib_registry.onnx_impl((operator.pos, operator.abs))(identity)
        self.assertIs(result, identity)
        entries = _torchlib_registry._registry[size:]
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(entry.signature is None for entry in entries))
        self.assertTrue(
            all(
                isinstance(entry.onnx_function, _custom_translation.NativeTranslation)
                for entry in entries
            )
        )
        self.assertTrue(
            all(entry.onnx_function.function is identity for entry in entries)
        )

    def test_real_and_complex_direct_construction(self):
        for complex_ in (False, True):
            with self.subTest(complex=complex_):
                shape = [2, 2] if complex_ else [2]
                value = ir.Value(
                    name="x",
                    type=ir.TensorType(ir.DataType.FLOAT),
                    shape=ir.Shape(shape),
                )
                graph = ir.Graph(
                    [value], [], nodes=[], name="abs", opset_imports={"": 18}
                )
                context = _ir_builder.TranslationContext(
                    _ir_builder.GraphBuilder(graph)
                )
                function = core.aten_abs_complex if complex_ else core.aten_abs
                graph.outputs.append(function(context, value))
                graph.outputs[0].shape = ir.Shape([2])
                model = ir.Model(graph, ir_version=9)
                proto = ir.serde.serialize_model(model)
                onnx.checker.check_model(proto, full_check=True)
                inputs = (
                    torch.tensor([[3.0, 4.0], [-5.0, 12.0]])
                    if complex_
                    else torch.tensor([-3.0, 4.0])
                )
                expected = torch.tensor([5.0, 13.0]) if complex_ else inputs.abs()
                result = ReferenceEvaluator(proto).run(None, {"x": inputs.numpy()})[0]
                self.assertEqual(torch.from_numpy(result), expected)

    def test_native_builtins_precede_external_fallback(self):
        registry = _registration.ONNXRegistry.from_torchlib(18)
        for complex_ in (False, True):
            entries = [
                entry
                for entry in registry.get_decomps(torch.ops.aten.abs.default)
                if entry.is_complex == complex_
            ]
            self.assertEqual(len(entries), 1)
            self.assertIsInstance(
                entries[0].onnx_function, _custom_translation.NativeTranslation
            )
            expected = core.aten_abs_complex if complex_ else core.aten_abs
            self.assertIs(entries[0].onnx_function.function, expected)


if __name__ == "__main__":
    run_tests()
