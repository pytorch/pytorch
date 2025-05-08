# Owner(s): ["module: onnx"]
"""Simple API tests for the ONNX exporter."""

from __future__ import annotations

import io
import os

import numpy as np
from onnxscript import BOOL, FLOAT, ir, opset18 as op

import torch
import torch.onnx._flags
from torch.onnx._internal.exporter import _testing as onnx_testing
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


class NestedModelForDynamicShapes(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        ys: list[torch.Tensor],
        zs: dict[str, torch.Tensor],
        c: torch.Tensor,
    ):
        y = ys[0] + ys[1] + zs["a"] + zs["b"]
        w = 5
        if x.shape[0] < 3 and c.shape[0] != 4:
            return x + w, x + y, c
        else:
            return x - w, x - y, c


class TestExportAPIDynamo(common_utils.TestCase):
    """Tests for the ONNX exporter API when dynamo=True."""

    def assert_export(
        self, *args, strategy: str | None = "TorchExportNonStrictStrategy", **kwargs
    ):
        onnx_program = torch.onnx.export(
            *args, **kwargs, dynamo=True, fallback=False, verbose=False
        )
        assert onnx_program is not None
        onnx_testing.assert_onnx_program(onnx_program, strategy=strategy)

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
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )

    def test_dynamic_axes_supports_output_names(self):
        self.assert_export(
            SampleModelForDynamicShapes(),
            (torch.randn(2, 2, 3), {"b": torch.randn(2, 2, 3)}),
            input_names=["x", "b"],
            dynamic_axes={
                "b": [0, 1, 2],
            },
        )
        self.assert_export(
            SampleModelForDynamicShapes(),
            (
                torch.randn(2, 2, 3),
                torch.randn(2, 2, 3),
            ),
            input_names=["x", "b"],
            output_names=["x_out", "b_out"],
            dynamic_axes={"b": [0, 1, 2], "b_out": [0, 1, 2]},
        )

    def test_saved_f_exists_after_export(self):
        with common_utils.TemporaryFileName(suffix=".onnx") as path:
            _ = torch.onnx.export(
                SampleModel(), (torch.randn(1, 1, 2),), path, dynamo=True
            )
            self.assertTrue(os.path.exists(path))

    def test_dynamic_shapes_with_fully_specified_axes(self):
        ep = torch.export.export(
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
            strict=True,
        )

        self.assert_export(ep, strategy=None)

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
        torch.onnx._flags.USE_EXPERIMENTAL_LOGIC = True

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

    def test_dynamic_shapes_supports_nested_input_model_with_input_names_assigned(self):
        # kwargs can still be renamed as long as it's in order
        input_names = ["input_x", "input_y", "input_z", "d", "e", "f"]

        dynamic_axes = {
            "input_x": {0: "dim"},
            "input_y": {0: "dim"},
            "input_z": {0: "dim"},
            "d": {0: "dim"},
            "e": {0: "dim"},
        }

        model = NestedModelForDynamicShapes()
        input = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )

        self.assert_export(
            model, input, dynamic_axes=dynamic_axes, input_names=input_names
        )

        # Check whether inputs are dynamically shaped
        onnx_program = torch.onnx.export(
            model,
            input,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            dynamo=True,
        )
        self.assertTrue(
            all(
                [
                    input.type.tensor_type.shape.dim[0].dim_param
                    for input in onnx_program.model_proto.graph.input
                ][:-1]
            )
        )

    def test_upgraded_torchlib_impl(self):
        class GeluModel(torch.nn.Module):
            def forward(self, input):
                # Use GELU activation function
                return torch.nn.functional.gelu(input, approximate="tanh")

        input = torch.randn(1, 3, 4, 4)
        onnx_program_op18 = torch.onnx.export(
            GeluModel(),
            input,
            dynamo=True,
        )
        all_nodes_op18 = [n.op_type for n in onnx_program_op18.model.graph]
        self.assertIn("Tanh", all_nodes_op18)
        self.assertNotIn("Gelu", all_nodes_op18)

        onnx_program_op20 = torch.onnx.export(
            GeluModel(),
            input,
            opset_version=20,
            dynamo=True,
        )
        all_nodes_op20 = [n.op_type for n in onnx_program_op20.model.graph]
        self.assertIn("Gelu", all_nodes_op20)

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


class TestCustomTranslationTable(common_utils.TestCase):
    def test_custom_translation_table_overrides_ops(self):
        from onnxscript import opset18 as op

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        def custom_add(self, other):
            # Replace add with sub
            return op.Sub(self, other)

        custom_translation_table = {torch.ops.aten.add.Tensor: custom_add}

        onnx_program = torch.onnx.export(
            Model(),
            (torch.randn(2, 2), torch.randn(2, 2)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)

    def test_custom_translation_table_supports_overloading_ops(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.logical_and.default(x, y)

        def custom_add_bool(self: BOOL, other: BOOL) -> BOOL:
            # Replace add with sub
            return op.Sub(self, other)

        def custom_add(self: FLOAT, other: FLOAT) -> FLOAT:
            # Replace add with mul
            return op.Mul(self, other)

        custom_translation_table = {
            torch.ops.aten.logical_and.default: [custom_add, custom_add_bool],
        }

        onnx_program = torch.onnx.export(
            Model(),
            (torch.tensor(1, dtype=torch.bool), torch.tensor(1, dtype=torch.bool)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        # The dispatcher should pick the correct overload based on the input types
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)
        self.assertNotIn("Mul", all_nodes)

    def test_custom_translation_table_supports_custom_op_as_target(self):
        # Define the custom op and use it in the model
        @torch.library.custom_op("custom::add", mutates_args=())
        def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        @custom_add.register_fake
        def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(a) + torch.empty_like(b)

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return custom_add(x, y)

        def onnx_add(self: FLOAT, other: FLOAT) -> FLOAT:
            # Replace add with Sub
            return op.Sub(self, other)

        custom_translation_table = {
            torch.ops.custom.add.default: onnx_add,
        }

        onnx_program = torch.onnx.export(
            Model(),
            (torch.tensor(1, dtype=torch.bool), torch.tensor(1, dtype=torch.bool)),
            custom_translation_table=custom_translation_table,
            dynamo=True,
        )
        all_nodes = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sub", all_nodes)
        self.assertNotIn("Add", all_nodes)


class TestFakeTensorExport(common_utils.TestCase):
    """Test exporting in fake mode."""

    def test_onnx_program_raises_when_model_defined_in_fake_mode(self):
        with torch.onnx.enable_fake_mode():

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor(42.0))

                def forward(self, x):
                    return self.weight + x

            onnx_program = torch.onnx.export(
                Model(), (torch.tensor(1.0),), dynamo=True, optimize=False
            )
            assert onnx_program is not None
            # Convert to model proto and back to trigger to_bytes method which serializes the tensor
            with self.assertRaises(Exception):
                # The tensors need to be replaced with real tensors
                _ = onnx_program.model_proto

        # Convert to model proto and back to trigger to_bytes method which serializes the tensor
        with self.assertRaises(Exception):
            # It doesn't matter if it is called inside or outside of the enable_fake_mode() context
            _ = onnx_program.model_proto

        # If we replace with concrete tensors, the serialization will succeed.
        # This needs to happen outside of the fake context
        onnx_program.apply_weights({"weight": torch.tensor(42.0)})
        onnx_model = ir.serde.deserialize_model(onnx_program.model_proto)
        np.testing.assert_allclose(
            onnx_model.graph.initializers["weight"].const_value.numpy(), 42.0
        )

    def test_onnx_program_save_raises_when_model_initialized_in_fake_mode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(42.0))

            def forward(self, x):
                return self.weight + x

        with torch.onnx.enable_fake_mode():
            onnx_program = torch.onnx.export(
                Model(), (torch.tensor(1.0),), dynamo=True, optimize=False
            )
            assert onnx_program is not None
            # Convert to model proto and back to trigger to_bytes method which serializes the tensor
            with self.assertRaises(Exception):
                # The tensors need to be replaced with real tensors
                _ = onnx_program.model_proto

        with self.assertRaises(Exception):
            # It doesn't matter if it is called inside or outside of the enable_fake_mode() context
            _ = onnx_program.model_proto

        # If we replace with concrete tensors, the serialization will succeed
        # This needs to happen outside of the fake context
        onnx_program.apply_weights({"weight": torch.tensor(42.0)})
        onnx_model = ir.serde.deserialize_model(onnx_program.model_proto)
        np.testing.assert_allclose(
            onnx_model.graph.initializers["weight"].const_value.numpy(), 42.0
        )

    def test_onnx_program_save_succeeds_when_export_and_save_in_fake_mode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(42.0))

            def forward(self, x):
                return self.weight + x

        real_model = Model()

        with torch.onnx.enable_fake_mode():
            onnx_program = torch.onnx.export(
                real_model, (torch.tensor(1.0),), dynamo=True, optimize=False
            )

            assert onnx_program is not None
            # Convert to model proto and back to trigger to_bytes method which serializes the tensor
            # Note that even though we are calling .model_proto (equivalently .save()) in fake mode,
            # the concrete tensors are maintained.
            # This is due to the usage of torch._subclasses.fake_tensor.unset_fake_temporarily() in
            # TorchTensor.tobytes()
            onnx_model = ir.serde.deserialize_model(onnx_program.model_proto)
            np.testing.assert_allclose(
                onnx_model.graph.initializers["weight"].const_value.numpy(), 42.0
            )

        # This works inside or outside the fake mode
        onnx_model = ir.serde.deserialize_model(onnx_program.model_proto)
        np.testing.assert_allclose(
            onnx_model.graph.initializers["weight"].const_value.numpy(), 42.0
        )

    def test_is_in_onnx_export(self):
        class Mod(torch.nn.Module):
            def forward(self, x):
                def f(x):
                    return x.sin() if torch.onnx.is_in_onnx_export() else x.cos()

                return f(x)

        self.assertFalse(torch.onnx.is_in_onnx_export())
        onnx_program = torch.onnx.export(
            Mod(),
            (torch.randn(3, 4),),
            dynamo=True,
            fallback=False,
        )
        self.assertFalse(torch.onnx.is_in_onnx_export())

        node_names = [n.op_type for n in onnx_program.model.graph]
        self.assertIn("Sin", node_names)

    def test_torchscript_exporter_raises_deprecation_warning(self):
        # Test that the deprecation warning is raised when using torchscript exporter
        with self.assertWarnsRegex(
            DeprecationWarning, "You are using the legacy TorchScript-based ONNX export"
        ):
            torch.onnx.export(
                SampleModel(), (torch.randn(1, 1, 2),), io.BytesIO(), dynamo=False
            )


if __name__ == "__main__":
    common_utils.run_tests()
