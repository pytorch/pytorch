# Owner(s): ["module: onnx"]
r"""Tests for native IR construction without ONNX Script evaluation."""

from __future__ import annotations

import contextvars

import numpy as np

import onnx
import onnx_ir as ir

from torch.onnx._internal.exporter import _ir_builder
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


class TestGraphBuilder(common_utils.TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self.graph = ir.Graph([], [], nodes=[], name="main")
        self.builder = _ir_builder.GraphBuilder(self.graph)
        self.x = ir.Value(
            name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2])
        )

    @parametrize("reverse", [False, True])
    @parametrize("constant", [True, 1, 1.5, [], [1, 2]])
    def test_tensor_binding_precedes_constants(self, reverse, constant):
        args = (constant, self.x) if reverse else (self.x, constant)
        (result,) = self.builder.call("Add", *args)
        self.assertEqual(result.dtype, ir.DataType.FLOAT)
        constant_node = self.graph[0]
        self.assertEqual(constant_node.op_type, "Constant")
        self.assertEqual(constant_node.outputs[0].dtype, ir.DataType.FLOAT)
        (chained,) = self.builder.call("Mul", 2, result)
        self.assertEqual(chained.dtype, ir.DataType.FLOAT)
        self.assertFalse(any(node.op_type == "Cast" for node in self.graph))

    @parametrize(
        "op,value,dtype",
        [
            ("Not", True, ir.DataType.BOOL),
            ("Neg", 1, ir.DataType.INT64),
            ("Neg", 1.5, ir.DataType.FLOAT),
        ],
    )
    def test_unbound_constants(self, op, value, dtype):
        (result,) = self.builder.call(op, value)
        self.assertEqual(result.dtype, dtype)

    def test_fixed_input_dtype(self):
        (result,) = self.builder.call("Reshape", self.x, [1, 2])
        self.assertEqual(self.graph[0].outputs[0].dtype, ir.DataType.INT64)
        self.assertEqual(result.dtype, ir.DataType.FLOAT)

    @parametrize("reverse", [False, True])
    @parametrize("dtype", [ir.DataType.INT32, ir.DataType.FLOAT16, ir.DataType.DOUBLE])
    def test_binding_preserves_tensor_dtype(self, reverse, dtype):
        tensor = ir.Value(type=ir.TensorType(dtype))
        inputs = (2, tensor) if reverse else (tensor, 2)
        (result,) = self.builder.call("Add", *inputs)
        self.assertEqual(result.dtype, dtype)
        self.assertEqual(self.graph[0].outputs[0].dtype, dtype)

    def test_boolean_binding_and_fixed_types(self):
        boolean = ir.Value(type=ir.TensorType(ir.DataType.BOOL))
        (result,) = self.builder.call("And", True, boolean)
        self.assertEqual(result.dtype, ir.DataType.BOOL)
        self.builder.call("Where", True, self.x, 0)
        self.assertEqual(self.graph[-2].outputs[0].dtype, ir.DataType.FLOAT)
        self.assertEqual(self.graph[-3].outputs[0].dtype, ir.DataType.BOOL)

    @parametrize("inputs", [(1, 1.5), (1.5, 1), (True, 1)])
    def test_mixed_unbound_python_constants_rejected(self, inputs):
        with self.assertRaisesRegex(TypeError, "constant|Cast"):
            self.builder.call("Add", *inputs)
        self.assertEqual(len(self.graph), 0)

    def test_tensor_conflict_and_unknown_type(self):
        other = ir.Value(type=ir.TensorType(ir.DataType.DOUBLE))
        with self.assertRaisesRegex(TypeError, "explicit Cast"):
            self.builder.call("Add", self.x, other)
        with self.assertRaisesRegex(ValueError, "unknown type"):
            self.builder.call("Add", self.x, ir.Value())
        with self.assertRaisesRegex(TypeError, "explicit Cast"):
            self.builder.call("Reshape", self.x, other)
        self.assertEqual(len(self.graph), 0)

    def test_constant_identity_and_scope(self):
        first = self.builder.constant([], dtype=ir.DataType.INT64)
        self.assertIs(first, self.builder.constant([], dtype=ir.DataType.INT64))
        self.assertEqual(first.shape, ir.Shape([0]))
        scalar = self.builder.constant(1, dtype=ir.DataType.INT64)
        vector = self.builder.constant([1], dtype=ir.DataType.INT64)
        self.assertIsNot(scalar, vector)
        other = _ir_builder.GraphBuilder(ir.Graph([], [], nodes=[]))
        self.assertIsNot(first, other.constant([], dtype=ir.DataType.INT64))
        with self.assertRaisesRegex(TypeError, "unambiguous"):
            self.builder.call("Neg", [])
        self.builder.call("Neg", first)

    def test_bitwise_constant_cache(self):
        positive = self.builder.constant(0.0, dtype=ir.DataType.FLOAT)
        negative = self.builder.constant(-0.0, dtype=ir.DataType.FLOAT)
        self.assertIsNot(positive, negative)
        self.assertTrue(np.signbit(negative.const_value.numpy()))

    def test_optional_inputs(self):
        (result,) = self.builder.call("Clip", self.x, None, 1)
        self.assertIsNone(self.graph[-1].inputs[1])
        self.assertEqual(result.dtype, ir.DataType.FLOAT)
        self.builder.call("Clip", self.x, None, None)
        self.assertEqual(len(self.graph[-1].inputs), 1)
        with self.assertRaisesRegex(ValueError, "required"):
            self.builder.call("Add", self.x, None)

    def test_variadic_inputs_and_outputs(self):
        (joined,) = self.builder.call("Concat", self.x, self.x, axis=0)
        self.assertEqual(joined.dtype, ir.DataType.FLOAT)
        with self.assertRaisesRegex(ValueError, "num_outputs"):
            self.builder.call("Split", joined)
        outputs = self.builder.call(
            "Split", joined, _num_outputs=2, num_outputs=2, axis=0
        )
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(value.dtype == ir.DataType.FLOAT for value in outputs))
        (values, indices) = self.builder.call("TopK", self.x, [1])
        self.assertEqual(values.dtype, ir.DataType.FLOAT)
        self.assertEqual(indices.dtype, ir.DataType.INT64)

    def test_explicit_cast_shape_and_sequence(self):
        (cast,) = self.builder.call(
            "Cast",
            self.x,
            to=int(ir.DataType.DOUBLE),
            _output_types=[ir.TensorType(ir.DataType.DOUBLE)],
        )
        self.assertEqual(cast.dtype, ir.DataType.DOUBLE)
        (shape,) = self.builder.call("Shape", self.x)
        (dimension,) = self.builder.call("Gather", shape, 0, axis=0)
        (vector,) = self.builder.call("Reshape", dimension, [1])
        (new_shape,) = self.builder.call("Concat", vector, [1], axis=0)
        self.assertEqual(new_shape.dtype, ir.DataType.INT64)
        sequence_type = ir.SequenceType(ir.TensorType(ir.DataType.FLOAT))
        (sequence,) = self.builder.call(
            "SequenceConstruct", self.x, self.x, _output_types=[sequence_type]
        )
        (element,) = self.builder.call(
            "SequenceAt", sequence, 0, _output_types=[self.x.type]
        )
        self.assertEqual(element.dtype, ir.DataType.FLOAT)
        with self.assertRaisesRegex(TypeError, "symbolic"):
            self.builder.call("Neg", [self.x, 1])
        with self.assertRaisesRegex(TypeError, "symbolic shape"):
            self.builder.call("Reshape", self.x, [self.x, 1])

    def test_raw_custom_nodes(self):
        (result,) = self.builder.node(
            "Custom",
            [self.x],
            domain="example",
            version=1,
            attributes={"axis": 1},
            output_types=[self.x.type],
        )
        self.assertEqual(result.dtype, self.x.dtype)
        self.assertEqual(self.graph.opset_imports["example"], 1)
        with self.assertRaisesRegex(ValueError, "version"):
            self.builder.node("Custom", domain="other", output_types=[None])
        with self.assertRaisesRegex(ValueError, "one opset"):
            self.builder.node(
                "Custom", domain="example", version=2, output_types=[None]
            )
        with self.assertRaisesRegex(TypeError, "raw inputs"):
            self.builder.node("Custom", [1], output_types=[None])

    def test_nested_scope_restored_after_exception(self):
        other = _ir_builder.GraphBuilder(ir.Graph([], [], nodes=[]))
        with self.assertRaisesRegex(RuntimeError, "No ONNX IR builder"):
            _ir_builder.current_builder()
        with self.builder.activate():
            self.assertIs(_ir_builder.current_builder(), self.builder)
            with self.assertRaisesRegex(ValueError, "sentinel"):
                with other.activate():
                    self.assertIs(_ir_builder.current_builder(), other)
                    raise ValueError("sentinel")
            self.assertIs(_ir_builder.current_builder(), self.builder)
            with self.builder.activate():
                self.assertIs(_ir_builder.current_builder(), self.builder)
            self.assertIs(_ir_builder.current_builder(), self.builder)
            fresh_context = contextvars.Context()
            with self.assertRaisesRegex(RuntimeError, "No ONNX IR builder"):
                fresh_context.run(_ir_builder.current_builder)
        with self.assertRaisesRegex(RuntimeError, "No ONNX IR builder"):
            _ir_builder.current_builder()

    def test_explicit_subgraphs(self):
        branches = []
        for value in (1, 2):
            graph = ir.Graph([], [], nodes=[], name=f"branch{value}")
            branch_builder = _ir_builder.GraphBuilder(graph)
            result = branch_builder.constant(value, dtype=ir.DataType.INT64)
            result.name = f"branch{value}_result"
            graph.outputs.append(result)
            branches.append(graph)
        (result,) = self.builder.node(
            "If",
            [self.builder.constant(True, dtype=ir.DataType.BOOL)],
            attributes={"then_branch": branches[0], "else_branch": branches[1]},
            output_types=[ir.TensorType(ir.DataType.INT64)],
        )
        self.graph.outputs.append(result)
        result.shape = ir.Shape([])
        onnx.checker.check_model(
            ir.serde.serialize_model(ir.Model(self.graph, ir_version=9))
        )

    def test_explicit_function(self):
        arg = ir.Value(name="arg", type=self.x.type)
        body = ir.Graph([arg], [], nodes=[], name="body")
        (out,) = _ir_builder.GraphBuilder(body).call("Neg", arg)
        body.outputs.append(out)
        function = ir.Function("example", "negate", graph=body, attributes={})
        (result,) = self.builder.call_function(function, self.x, version=1)
        self.assertEqual(result.type, self.x.type)
        self.assertIs(self.builder.functions[function.identifier()], function)
        self.assertEqual(self.graph[-1].domain, "example")
        with self.assertRaisesRegex(ValueError, "expects"):
            self.builder.call_function(function, version=1)
        result.shape = self.x.shape
        self.graph.inputs.append(self.x)
        self.graph.outputs.append(result)
        model = ir.Model(
            self.graph, ir_version=9, functions=list(self.builder.functions.values())
        )
        onnx.checker.check_model(ir.serde.serialize_model(model))

    def test_function_attributes_and_duplicate_definition(self):
        arg = ir.Value(name="arg", type=self.x.type)
        body = ir.Graph([arg], [], nodes=[], name="body", opset_imports={"": 18})
        (out,) = _ir_builder.GraphBuilder(body).node(
            "LeakyRelu",
            [arg],
            attributes={"alpha": ir.RefAttr("alpha", "slope", ir.AttributeType.FLOAT)},
            output_types=[arg.type],
        )
        body.outputs.append(out)
        function = ir.Function(
            "example",
            "activation",
            graph=body,
            attributes={"slope": ir.Attr("slope", ir.AttributeType.FLOAT, None)},
        )
        with self.assertRaisesRegex(ValueError, "requires attribute"):
            self.builder.call_function(function, self.x, version=1)
        self.builder.call_function(function, self.x, version=1, attributes={"slope": 1})
        self.assertEqual(
            self.graph[-1].attributes["slope"].type, ir.AttributeType.FLOAT
        )
        duplicate = ir.Function(
            "example", "activation", graph=body, attributes=function.attributes
        )
        with self.assertRaisesRegex(ValueError, "already defined"):
            self.builder.call_function(
                duplicate, self.x, version=1, attributes={"slope": 1}
            )

    def test_invalid_arity_attributes_and_outputs(self):
        with self.assertRaisesRegex(ValueError, "No ONNX schema"):
            self.builder.call("NotAnOnnxOperator", self.x)
        with self.assertRaisesRegex(ValueError, "inputs"):
            self.builder.call("Add", self.x)
        with self.assertRaisesRegex(ValueError, "no attribute"):
            self.builder.call("Neg", self.x, unexpected=1)
        with self.assertRaisesRegex(ValueError, "requires attribute"):
            self.builder.call("Concat", self.x)
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            self.builder.call(
                "Neg", self.x, _num_outputs=2, _output_types=[self.x.type]
            )
        with self.assertRaisesRegex(TypeError, "conflicts"):
            self.builder.call(
                "Neg", self.x, _output_types=[ir.TensorType(ir.DataType.INT64)]
            )
        self.assertEqual(len(self.graph), 0)

    def test_attributes_use_schema_types(self):
        self.builder.call("LeakyRelu", self.x, alpha=1)
        self.assertEqual(
            self.graph[-1].attributes["alpha"].type, ir.AttributeType.FLOAT
        )

    def test_model_execution(self):
        from onnx.reference import ReferenceEvaluator

        self.graph.inputs.append(self.x)
        (result,) = self.builder.call("Add", 2, self.x)
        result.shape = self.x.shape
        self.graph.outputs.append(result)
        proto = ir.serde.serialize_model(ir.Model(self.graph, ir_version=9))
        onnx.checker.check_model(proto)
        (actual,) = ReferenceEvaluator(proto).run(
            None, {"x": np.array([1, 3], np.float32)}
        )
        self.assertEqual(actual, np.array([3, 5], np.float32))


instantiate_parametrized_tests(TestGraphBuilder)


if __name__ == "__main__":
    common_utils.run_tests()
