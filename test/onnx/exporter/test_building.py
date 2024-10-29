# Owner(s): ["module: onnx"]
"""Unit tests for the _building module."""

from __future__ import annotations

import numpy as np
import onnxscript
from onnxscript import ir

import torch
from torch.onnx._internal.exporter import _building, _tensors
from torch.testing._internal import common_utils


class TestOpRecorder(common_utils.TestCase):
    def setUp(self):
        self.opset_version = 17
        self.opset = onnxscript.values.Opset("", self.opset_version)
        self.recorder = _building.OpRecorder(opset=self.opset, constant_farm={})

        self.model = ir.Model(
            graph=ir.Graph(
                [],
                [],
                nodes=[],
                opset_imports={
                    "": self.opset_version,
                },
                name="main_graph",
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )

    def test_skippable_castlike_is_ommited(self):
        input_x = _tensors.SymbolicTensor(opset=self.opset, name="input_x")
        input_x.dtype = ir.DataType.FLOAT

        input_y = _tensors.SymbolicTensor(opset=self.opset, name="input_y")
        input_y.dtype = ir.DataType.FLOAT

        with onnxscript.evaluator.default_as(
            tracer := self.recorder,
        ):
            cast = self.opset.CastLike(input_y, input_x)
            _ = self.opset.Add(input_x, cast)

        self.assertEqual(len(tracer.nodes), 1)
        self.assertEqual(tracer.nodes[0].op_type, "Add")

    def test_castlike_is_replaced_with_cast_when_it_is_traced(self):
        input_x = _tensors.SymbolicTensor(opset=self.opset, name="input_x")
        input_x.dtype = ir.DataType.FLOAT

        input_y = _tensors.SymbolicTensor(opset=self.opset, name="input_y")
        input_y.dtype = ir.DataType.INT64

        with onnxscript.evaluator.default_as(
            tracer := self.recorder,
        ):
            cast = self.opset.CastLike(input_y, input_x)
            _ = self.opset.Add(input_x, cast)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Cast")
        self.assertEqual(tracer.nodes[1].op_type, "Add")

    def test_python_constant_added_as_constant_nodes(self):
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3, 4])
        )
        new_shape = [3, 2, 4]

        with onnxscript.evaluator.default_as(
            tracer := self.recorder,
        ):
            _ = self.opset.Reshape(input_x, new_shape)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Constant")
        self.assertEqual(
            tracer.nodes[0].attributes["value"].value.numpy(), np.array(new_shape)
        )
        self.assertEqual(tracer.nodes[1].op_type, "Reshape")

    def test_process_python_sequence_with_allowed_sequence_type(self):
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([2, 4])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([1, 3])
        )

        with onnxscript.evaluator.default_as(
            tracer := self.recorder,
        ):
            _ = self.opset.SequenceAt([input_x, input_y, input_z], 1)

        self.assertEqual(len(tracer.nodes), 3)
        self.assertEqual(tracer.nodes[1].op_type, "SequenceConstruct")

    def test_process_python_sequence_with_variadic_input(self):
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([2, 4])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([1, 3])
        )

        with onnxscript.evaluator.default_as(
            tracer := self.recorder,
        ):
            _ = self.opset.Max(input_x, input_y, 0, input_z)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Constant")

    def test_process_python_sequence_with_an_extra_concat(self):
        input_x = _tensors.SymbolicTensor(
            opset=self.opset, name="input_x", shape=ir.Shape([2, 3])
        )
        input_y = _tensors.SymbolicTensor(
            opset=self.opset, name="input_y", shape=ir.Shape([2, 3])
        )
        input_z = _tensors.SymbolicTensor(
            opset=self.opset, name="input_z", shape=ir.Shape([4, 3])
        )

        with onnxscript.evaluator.default_as(
            tracer := self.recorder,
        ):
            _ = self.opset.Add([input_x, input_y], input_z)

        self.assertEqual(len(tracer.nodes), 2)
        self.assertEqual(tracer.nodes[0].op_type, "Concat")
        self.assertEqual(tracer.nodes[0].attributes["axis"].value, 0)


if __name__ == "__main__":
    common_utils.run_tests()
