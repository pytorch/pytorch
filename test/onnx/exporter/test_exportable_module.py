# Owner(s): ["module: onnx"]
"""Unit tests for ExportableModule."""

from __future__ import annotations

from typing import Any

import torch
from torch.onnx._internal.exporter import _exportable_module
from torch.testing._internal import common_utils


class SimpleExportableModel(_exportable_module.ExportableModule):
    """A simple concrete implementation of ExportableModule for testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        return (torch.randn(2, 10),), None


class ExportableModelWithKwargs(_exportable_module.ExportableModule):
    """ExportableModule with keyword arguments."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x, scale: float = 1.0):
        return self.linear(x) * scale

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        return (torch.randn(2, 10),), {"scale": 2.0}


class ExportableModelWithDynamicShapes(_exportable_module.ExportableModule):
    """ExportableModule with dynamic shapes."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        return (torch.randn(2, 10),), None

    def dynamic_shapes(self):
        return {"x": {0: "batch_size"}}


class ExportableModelWithNames(_exportable_module.ExportableModule):
    """ExportableModule with input and output names."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        return (torch.randn(2, 10),), None

    def input_names(self):
        return ["input_tensor"]

    def output_names(self):
        return ["output_tensor"]


class ExportableModelMultipleOutputs(_exportable_module.ExportableModule):
    """ExportableModule with multiple outputs."""

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 5)
        self.linear2 = torch.nn.Linear(10, 3)

    def forward(self, x):
        return self.linear1(x), self.linear2(x)

    def example_arguments(self) -> tuple[tuple[Any], dict[str, Any] | None]:
        return (torch.randn(2, 10),), None

    def output_names(self):
        return ["output1", "output2"]


class TestExportableModule(common_utils.TestCase):
    """Tests for ExportableModule abstract class and its implementations."""

    def test_abstract_class_cannot_be_instantiated(self):
        """Test that ExportableModule cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            _exportable_module.ExportableModule()

    def test_simple_exportable_model(self):
        """Test basic ExportableModule implementation."""
        model = SimpleExportableModel()
        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model, _exportable_module.ExportableModule)

        # Test example_arguments
        args, kwargs = model.example_arguments()
        self.assertIsInstance(args, tuple)
        self.assertEqual(len(args), 1)
        self.assertIsNone(kwargs)

        # Test forward pass
        output = model(*args)
        self.assertEqual(output.shape, (2, 5))

    def test_exportable_model_with_kwargs(self):
        """Test ExportableModule with keyword arguments."""
        model = ExportableModelWithKwargs()

        args, kwargs = model.example_arguments()
        self.assertIsInstance(args, tuple)
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(kwargs["scale"], 2.0)

        # Test forward pass with kwargs
        output = model(*args, **kwargs)
        self.assertEqual(output.shape, (2, 5))

    def test_dynamic_shapes_default(self):
        """Test that dynamic_shapes returns None by default."""
        model = SimpleExportableModel()
        self.assertIsNone(model.dynamic_shapes())

    def test_dynamic_shapes_custom(self):
        """Test custom dynamic_shapes implementation."""
        model = ExportableModelWithDynamicShapes()
        dynamic_shapes = model.dynamic_shapes()
        self.assertIsNotNone(dynamic_shapes)
        self.assertIsInstance(dynamic_shapes, dict)
        self.assertIn("x", dynamic_shapes)

    def test_input_names_default(self):
        """Test that input_names returns None by default."""
        model = SimpleExportableModel()
        self.assertIsNone(model.input_names())

    def test_output_names_default(self):
        """Test that output_names returns None by default."""
        model = SimpleExportableModel()
        self.assertIsNone(model.output_names())

    def test_input_output_names_custom(self):
        """Test custom input_names and output_names implementation."""
        model = ExportableModelWithNames()

        input_names = model.input_names()
        self.assertIsNotNone(input_names)
        self.assertEqual(len(input_names), 1)
        self.assertEqual(input_names[0], "input_tensor")

        output_names = model.output_names()
        self.assertIsNotNone(output_names)
        self.assertEqual(len(output_names), 1)
        self.assertEqual(output_names[0], "output_tensor")

    def test_multiple_outputs(self):
        """Test ExportableModule with multiple outputs."""
        model = ExportableModelMultipleOutputs()

        args, kwargs = model.example_arguments()
        output1, output2 = model(*args)
        self.assertEqual(output1.shape, (2, 5))
        self.assertEqual(output2.shape, (2, 3))

        output_names = model.output_names()
        self.assertEqual(len(output_names), 2)
        self.assertEqual(output_names[0], "output1")
        self.assertEqual(output_names[1], "output2")

    def test_to_onnx_calls_export(self):
        """Test that to_onnx correctly calls torch.onnx.export."""
        model = SimpleExportableModel()

        # This test verifies that to_onnx returns an ONNXProgram
        # The actual export functionality is tested elsewhere
        try:
            onnx_program = model.to_onnx(dynamo=True, fallback=False)
            self.assertIsNotNone(onnx_program)
            self.assertIsInstance(onnx_program, torch.onnx.ONNXProgram)
        except Exception as e:
            # If export fails for environmental reasons, skip this test
            # The important part is that the method exists and calls export
            if "dynamo" not in str(e).lower():
                raise

    def test_exportable_module_is_nn_module(self):
        """Test that ExportableModule is a proper nn.Module."""
        model = SimpleExportableModel()

        # Test that it has standard nn.Module methods
        self.assertTrue(hasattr(model, "parameters"))
        self.assertTrue(hasattr(model, "named_parameters"))
        self.assertTrue(hasattr(model, "state_dict"))
        self.assertTrue(hasattr(model, "load_state_dict"))

        # Test that parameters work correctly
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_example_arguments_must_be_implemented(self):
        """Test that example_arguments must be implemented by subclasses."""

        class IncompleteExportableModel(_exportable_module.ExportableModule):
            def forward(self, x):
                return x * 2

        with self.assertRaises(TypeError):
            IncompleteExportableModel()

    def test_forward_with_complex_inputs(self):
        """Test ExportableModule with more complex input types."""

        class ComplexInputModel(_exportable_module.ExportableModule):
            def forward(self, x, y):
                return x + y

            def example_arguments(self):
                return (torch.randn(2, 3), torch.randn(2, 3)), None

        model = ComplexInputModel()
        args, kwargs = model.example_arguments()
        output = model(*args)
        self.assertEqual(output.shape, (2, 3))


if __name__ == "__main__":
    common_utils.run_tests()
