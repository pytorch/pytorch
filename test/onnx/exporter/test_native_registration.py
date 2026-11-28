# Owner(s): ["module: onnx"]
"""Native built-in registration and explicit context at the legacy boundary."""

from __future__ import annotations

import gc
import operator
import weakref

import onnx

import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

import torch
from torch.onnx._internal.exporter import (
    _core,
    _custom_translation,
    _ir_builder,
    _registration,
)
from torch.onnx._internal.exporter._torchlib import _torchlib_registry
from torch.onnx._internal.exporter._torchlib.ops import core, factories
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class TestNativeRegistration(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_declared_symbolic_real_dtype_preserves_complex_meta_shape(self):
        meta = torch.empty(3, dtype=torch.complex64)
        declared = ir.Value(type=ir.TensorType(ir.DataType.FLOAT))
        _core._set_shape_type(declared, meta, complex_to_float=True)
        self.assertEqual(list(declared.shape), [3])
        packed = ir.Value(type=ir.TensorType(ir.DataType.FLOAT))
        _core._set_shape_type(packed, meta, complex_to_float=True, packed_complex=True)
        self.assertEqual(list(packed.shape), [3, 2])

    def test_graph_scoped_constant_cache(self):
        class Add(torch.nn.Module):
            def forward(self, x):
                return (x + 2.0) * 2.0

        programs = [
            torch.onnx.export(
                Add(),
                (torch.ones(3),),
                opset_version=18,
                optimize=False,
                verbose=False,
            )
            for _ in range(2)
        ]
        values = []
        for program in programs:
            nodes = list(program.model.graph)
            add = next(node for node in nodes if node.op_type == "Add")
            mul = next(node for node in nodes if node.op_type == "Mul")
            self.assertIs(add.inputs[1], mul.inputs[1])
            values.append(add.inputs[1])
            onnx.checker.check_model(program.model_proto, full_check=True)
        self.assertIsNot(values[0], values[1])

    def test_dynamic_repeat(self):
        class Repeat(torch.nn.Module):
            def forward(self, x, y):
                return x.repeat(y.shape[0], 1)

        model = Repeat()
        inputs = (torch.ones(1, 3), torch.ones(2))
        program = torch.onnx.export(
            model,
            inputs,
            dynamic_shapes={"x": {}, "y": {0: torch.export.Dim("n")}},
            opset_version=18,
            optimize=False,
            verbose=False,
        )
        proto = program.model_proto
        onnx.checker.check_model(proto, full_check=True)
        feed = {
            value.name: tensor.numpy()
            for value, tensor in zip(
                program.model.graph.inputs, (inputs[0], torch.ones(4))
            )
        }
        result = ReferenceEvaluator(proto).run(None, feed)[0]
        self.assertEqual(torch.from_numpy(result), model(inputs[0], torch.ones(4)))

    def test_arange_probe_does_not_retain_context(self):
        references = []
        expected = torch.arange(3.1, dtype=torch.int64).numel() == 4
        for _ in range(3):
            builder = _ir_builder.GraphBuilder(ir.Graph([], [], nodes=[]), opset=18)
            op = _ir_builder.TranslationContext(builder)
            references.append(weakref.ref(builder))
            self.assertEqual(
                factories._int64_arange_non_integral_uses_float_length(op), expected
            )
            del op, builder
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))

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
