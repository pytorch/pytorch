# Owner(s): ["module: onnx"]
"""Unit tests for the ONNXProgram class."""

from __future__ import annotations

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.testing._internal import common_utils


class ONNXProgramRenameAxesTest(common_utils.TestCase):
    """Tests for ONNXProgram.rename_axes method."""

    def _create_onnx_program_with_dynamic_shapes(
        self, input_shapes: list[list[str | int]]
    ) -> torch.onnx.ONNXProgram:
        """Helper to create an ONNXProgram with specified dynamic shapes."""
        from torch.onnx._internal.exporter._onnx_program import ONNXProgram

        inputs = []
        for i, shape in enumerate(input_shapes):
            inputs.append(
                ir.Value(
                    name=f"input_{i}",
                    type=ir.DataType.FLOAT,
                    shape=ir.Shape(shape),
                )
            )

        model = ir.Model(
            ir.Graph(
                inputs=inputs,
                outputs=[
                    ir.Value(
                        name="output",
                        type=ir.DataType.FLOAT,
                        shape=ir.Shape(["s0", "s1"]),
                    )
                ],
                nodes=[],
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )
        return ONNXProgram(model, exported_program=None)

    def test_rename_axes_with_string_keys(self):
        """Test renaming axes using string keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1"], ["s0", "s2"]]
        )

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "sequence_length",
                "s2": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )
        self.assertEqual(
            onnx_program.model.graph.inputs[1].shape,
            ir.Shape(["batch_size", "hidden_size"]),
        )
        # Outputs should also be renamed
        self.assertEqual(
            onnx_program.model.graph.outputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )

    def test_rename_axes_with_symbolic_dim_keys(self):
        """Test renaming axes using SymbolicDim objects as keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1"], ["s0", "s2"]]
        )

        # Get SymbolicDim objects from the model
        batch_dim = onnx_program.model.graph.inputs[0].shape[0]
        seq_dim = onnx_program.model.graph.inputs[0].shape[1]

        onnx_program.rename_axes(
            {
                batch_dim: "batch_size",
                seq_dim: "sequence_length",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length"]),
        )
        # s0 should be renamed in both inputs
        self.assertEqual(
            onnx_program.model.graph.inputs[1].shape[0],
            ir.SymbolicDim("batch_size"),
        )

    def test_rename_axes_with_mixed_string_and_symbolic_dim_keys(self):
        """Test renaming axes using a mix of string and SymbolicDim keys."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1", "s2"]]
        )

        # Get SymbolicDim object from the model
        batch_dim = onnx_program.model.graph.inputs[0].shape[0]

        onnx_program.rename_axes(
            {
                batch_dim: "batch_size",
                "s1": "sequence_length",
                "s2": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "sequence_length", "hidden_size"]),
        )

    def test_rename_axes_with_complex_shape_expressions(self):
        """Test renaming axes in complex shape expressions like 's1 + s2'."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes(
            [["s0", "s1 + s2", "s1*2"]]
        )

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "seq_len",
                "s2": "past_seq_len",
                "s1 + s2": "total_seq_len",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "total_seq_len", "seq_len*2"]),
        )

    def test_rename_axes_with_nonexistent_axis_is_ignored(self):
        """Test that renaming a non-existent axis is silently ignored."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", "s1"]])

        # Include a mapping for a non-existent axis
        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "nonexistent": "should_be_ignored",
            }
        )

        # Only s0 should be renamed
        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", "s1"]),
        )

    def test_rename_axes_with_empty_mapping(self):
        """Test renaming with an empty mapping does nothing."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", "s1"]])

        original_shape = onnx_program.model.graph.inputs[0].shape

        onnx_program.rename_axes({})

        self.assertEqual(onnx_program.model.graph.inputs[0].shape, original_shape)

    def test_rename_axes_with_static_dimensions_unchanged(self):
        """Test that static (integer) dimensions are not affected by renaming."""
        onnx_program = self._create_onnx_program_with_dynamic_shapes([["s0", 10, "s1"]])

        onnx_program.rename_axes(
            {
                "s0": "batch_size",
                "s1": "hidden_size",
            }
        )

        self.assertEqual(
            onnx_program.model.graph.inputs[0].shape,
            ir.Shape(["batch_size", 10, "hidden_size"]),
        )


if __name__ == "__main__":
    common_utils.run_tests()
