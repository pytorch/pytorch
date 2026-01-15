# Owner(s): ["module: onnx"]
"""Unit tests for the _ir_passes module."""

from __future__ import annotations

import onnx_ir as ir

import torch
from torch.onnx._internal.exporter import _ir_passes
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class ONNXIRPassesTest(common_utils.TestCase):
    @common_utils.parametrize(
        "shape_expr, expected_shape_expr",
        [
            ("2*s1", "batch_size*sequence_length"),
            ("s11/s1", "past_sequence_length/sequence_length"),
            ("(s1 + s11)*2", "(masked_sequence_length)*batch_size"),
        ],
    )
    def test__replace_names_in_rename_axis(self, shape_expr, expected_shape_expr):
        rename_mapping = {
            "s1 + s11": "masked_sequence_length",
            "s11": "past_sequence_length",
            "s1": "sequence_length",
            "2": "batch_size",
        }
        new_shape_expr = _ir_passes._replace_names(shape_expr, rename_mapping)
        self.assertEqual(new_shape_expr, expected_shape_expr)

    def test_rename_axis_succeeds_when_mapping_is_not_sorted_and_contains_the_str_not_in_the_model(
        self,
    ):
        model = ir.Model(
            ir.Graph(
                inputs=[
                    ir.Value(
                        name="input_0",
                        type=ir.DataType.FLOAT,
                        shape=ir.Shape(["s0", "s1"]),
                    ),
                    ir.Value(
                        name="input_1",
                        type=ir.DataType.FLOAT,
                        shape=ir.Shape(["s0 + s2", "s1 + s2"]),
                    ),
                    ir.Value(
                        name="input_2",
                        type=ir.DataType.FLOAT,
                        shape=ir.Shape(["s1/(s1 + s2)*2", "(s1 + s2)*2"]),
                    ),
                ],
                outputs=[
                    ir.Value(
                        name="output", type=ir.DataType.FLOAT, shape=ir.Shape("s99")
                    )
                ],
                nodes=[],
            ),
            ir_version=9,
            producer_name="pytorch",
            producer_version=torch.__version__,
        )

        mapping = {
            "s1": "sequence_length",
            "s2": "past_sequence_length",
            "s0": "batch_size",
            "s1 + s2": "masked_sequence_length",
            "s3": "extra_sequence_length",
        }
        _ir_passes.rename_axis(model, mapping)

        self.assertEqual(
            model.graph.inputs[0].shape, ir.Shape(["batch_size", "sequence_length"])
        )
        self.assertEqual(
            model.graph.inputs[1].shape,
            ir.Shape(["batch_size + past_sequence_length", "masked_sequence_length"]),
        )
        self.assertEqual(
            model.graph.inputs[2].shape,
            ir.Shape(
                [
                    "sequence_length/(masked_sequence_length)*2",
                    "(masked_sequence_length)*2",
                ]
            ),
        )


if __name__ == "__main__":
    common_utils.run_tests()
