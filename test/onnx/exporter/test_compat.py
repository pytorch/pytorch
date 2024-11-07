# Owner(s): ["module: onnx"]
"""Unit tests for the _building module."""

from __future__ import annotations

import os
import tempfile

import onnx

import torch
from torch.onnx._internal.exporter import _compat
from torch.testing._internal import common_utils


class SampleModelForDynamicShapes(torch.nn.Module):
    def forward(self, x, b):
        return x.relu(), b.sigmoid()


class NestedModelForDynamicShapes(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.u = torch.nn.Buffer(torch.ones(1))
        self.v = torch.nn.Buffer(torch.ones(1))

    def forward(self, x, ys, zs, c):
        y = ys[0] + ys[1] + zs["a"] + zs["b"]
        self.v.add_(3)
        w = self.u - self.v
        if x.shape[0] < 3 and c.shape[0] != 4:
            return x + w, x + y
        else:
            return x - w, x - y


@common_utils.instantiate_parametrized_tests
class TestCompat(common_utils.TestCase):
    @common_utils.parametrize(
        "dynamic_shapes, input_names, expected_dynamic_axes",
        [
            (
                {
                    "input_x": {
                        0: torch.export.Dim("customx_dim_0"),
                        1: torch.export.Dim("customx_dim_1"),
                    },
                    "input_b": {0: torch.export.Dim("customb_dim_0")},
                },
                None,
                {
                    "input_x": {0: "customx_dim_0", 1: "customx_dim_1"},
                    "input_b": {0: "customb_dim_0"},
                },
            ),
            (
                (
                    {
                        0: torch.export.Dim("customx_dim_0"),
                        1: torch.export.Dim("customx_dim_1"),
                    },
                    {
                        0: torch.export.Dim("customb_dim_0"),
                        1: None,
                        2: torch.export.Dim("customb_dim_2"),
                    },
                ),
                ["input_x", "input_b"],
                {
                    "input_x": {0: "customx_dim_0", 1: "customx_dim_1"},
                    "input_b": {0: "customb_dim_0", 2: "customb_dim_2"},
                },
            ),
            (
                (
                    {
                        0: torch.export.Dim("customx_dim_0"),
                        1: torch.export.Dim("customx_dim_1"),
                    },
                ),
                ["x"],
                {
                    "x": {0: "customx_dim_0", 1: "customx_dim_1"},
                },
            ),
        ],
    )
    def test_from_dynamic_shapes_to_dynamic_axes_success(
        self, dynamic_shapes, input_names, expected_dynamic_axes
    ):
        dynamic_axes = _compat._from_dynamic_shapes_to_dynamic_axes(
            dynamic_shapes=dynamic_shapes, input_names=input_names
        )
        self.assertEqual(dynamic_axes, expected_dynamic_axes)

    def test_dynamic_shapes_supports_nested_input_model_with_input_names_assigned(self):
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            [{0: dim}, {0: dim}],
            {"a": {0: dim}, "b": {0: dim}},
            None,
        )
        # kwargs can still be renamed as long as it's in order
        input_names = ["input_x", "input_y", "input_z", "d", "e", "f"]
        dynamic_axes = _compat._from_dynamic_shapes_to_dynamic_axes(
            dynamic_shapes=dynamic_shapes, input_names=input_names
        )
        expected_dynamic_axes = {
            "input_x": {0: "dim"},
            "input_y": {0: "dim"},
            "input_z": {0: "dim"},
            "d": {0: "dim"},
            "e": {0: "dim"},
        }
        self.assertEqual(dynamic_axes, expected_dynamic_axes)

        model = NestedModelForDynamicShapes()
        input = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )

        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            torch.onnx.export(
                model,
                input,
                filename,
                dynamic_axes=dynamic_axes,
                input_names=input_names,
            )
            onnx_model = onnx.load(filename)

        self.assertTrue(
            all(
                input.type.tensor_type.shape.dim[0].dim_param
                for input in onnx_model.graph.input
            )
        )


if __name__ == "__main__":
    common_utils.run_tests()
