# Owner(s): ["module: onnx"]
"""Simple API tests for the ONNX exporter."""

from __future__ import annotations

import os

import torch
from torch.onnx._internal.exporter import testing as onnx_testing
from torch.testing._internal import common_utils


class SampleModel(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        z = y.relu()
        return (y, z)


class SampleModelTwoInputs(torch.nn.Module):
    def forward(self, x, b):
        y = x + b
        z = y.relu()
        return (y, z)


class SampleModelForDynamicShapes(torch.nn.Module):
    def forward(self, x, b):
        return x.relu(), b.sigmoid()


class TestExportAPIDynamo(common_utils.TestCase):
    """Tests for the ONNX exporter API when dynamo=True."""

    def assert_export(self, *args, **kwargs):
        onnx_program = torch.onnx.export(*args, **kwargs, dynamo=True)
        assert onnx_program is not None
        onnx_testing.assert_onnx_program(onnx_program)

    def test_args_normalization_with_no_kwargs(self):
        self.assert_export(
            SampleModelTwoInputs(),
            (torch.randn(1, 1, 2), torch.randn(1, 1, 2)),
        )

    def test_dynamic_axes_enable_dynamic_shapes_with_fully_specified_axes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": {0: "customx_dim_0", 1: "customx_dim_1", 2: "customx_dim_2"},
                "b": {0: "customb_dim_0", 1: "customb_dim_1", 2: "customb_dim_2"},
            },
        )

    def test_dynamic_axes_enable_dynamic_shapes_with_default_axe_names(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "x": [0, 1, 2],
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_partial_dynamic_shapes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_output_names(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )
        onnx_program = torch.onnx.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamic_axes={"b": [0, 1, 2], "b_out": [0, 1, 2]},
            dynamo=True,
        )
        assert onnx_program is not None
        onnx_testing.assert_onnx_program(onnx_program)

    def test_saved_f_exists_after_export(self):
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            _ = torch.onnx.export(
                SampleModel(), (torch.randn(1, 1, 2),), path, dynamo=True
            )
            self.assertTrue(os.path.exists(path))

    def test_export_supports_script_module(self):
        class ScriptModule(torch.nn.Module):
            def forward(self, x):
                return x

        self.assert_export(torch.jit.script(ScriptModule()), (torch.randn(1, 1, 2),))

    def test_dynamic_shapes_with_fully_specified_axes(self):
        exported_program = torch.export.export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": {
                    0: torch.export.Dim("customx_dim_0"),
                    1: torch.export.Dim("customx_dim_1"),
                    2: torch.export.Dim("customx_dim_2"),
                },
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
        )

        self.assert_export(exported_program)

    def test_partial_dynamic_shapes(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            dynamic_shapes={
                "x": None,
                "b": {
                    0: torch.export.Dim("customb_dim_0"),
                    1: torch.export.Dim("customb_dim_1"),
                    2: torch.export.Dim("customb_dim_2"),
                },
            },
        )

    def test_auto_convert_all_axes_to_dynamic_shapes_with_dynamo_export(self):
        os.environ["TORCH_ONNX_USE_EXPERIMENTAL_LOGIC"] = "1"
        assert os.environ.get("TORCH_ONNX_USE_EXPERIMENTAL_LOGIC") == "1"

        class Nested(torch.nn.Module):
            def forward(self, x):
                (a0, a1), (b0, b1), (c0, c1, c2) = x
                return a0 + a1 + b0 + b1 + c0 + c1 + c2

        inputs = (
            (1, 2),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
            (
                torch.randn(4, 4),
                torch.randn(4, 4),
                torch.randn(4, 4),
            ),
        )

        onnx_program = torch.onnx.dynamo_export(
            Nested(),
            inputs,
            export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
        )
        assert onnx_program is not None
        onnx_testing.assert_onnx_program(onnx_program)

    def test_refine_dynamic_shapes_with_onnx_export(self):
        # NOTE: From test/export/test_export.py

        # refine lower, upper bound
        class TestRefineDynamicShapeModel(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] >= 6 and y.shape[0] <= 16:
                    return x * 2.0, y + 1

        inps = (torch.randn(16), torch.randn(12))
        dynamic_shapes = {
            "x": (torch.export.Dim("dx"),),
            "y": (torch.export.Dim("dy"),),
        }
        self.assert_export(
            TestRefineDynamicShapeModel(), inps, dynamic_shapes=dynamic_shapes
        )

    def test_zero_output_aten_node(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_async.msg(torch.tensor(True), "assertion failed")
                return x + x

        input = torch.randn(2)
        self.assert_export(Model(), (input))


if __name__ == "__main__":
    common_utils.run_tests()
