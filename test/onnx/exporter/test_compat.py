# Owner(s): ["module: onnx"]
"""Unit tests for the _compat module."""

from __future__ import annotations

import torch
from torch.onnx._internal.exporter import _compat
from torch.testing._internal import common_utils

from torch.utils import _pytree

@common_utils.instantiate_parametrized_tests
class TestPyTreeDynamicAxesShapes(common_utils.TestCase):
        # The test can't be parametrized because the torch.export.Dim generates objects,
        # and we need the exact same object to compare them.
    def test__unflatten_dynamic_shapes_with_inputs_tree_succeeds_on_tuple(self):
        inputs = (torch.randn(1, 2, 3), torch.randn(1, 2, 3))
        x_dim = torch.export.Dim("x_dim_0")
        y_dim = torch.export.Dim("y_dim_1")
        dynamic_shapes = {
            "x": {0: x_dim},
            "y": {1: y_dim},
        }
        unflatten_dynamic_shapes = _compat._unflatten_dynamic_shapes_with_inputs_tree(
            inputs, dynamic_shapes
        )
        
        expected_dynamic_shapes = (
            {0: x_dim},
            {1: y_dim},
        )
        self.assertEqual(unflatten_dynamic_shapes, expected_dynamic_shapes)

    def test__unflatten_dynamic_shapes_with_inputs_tree_succeeds_on_dict(self):
        inputs = {"x": torch.randn(1, 2, 3), "y": torch.randn(1, 2, 3)}
        x_dim = torch.export.Dim("x_dim_0")
        y_dim = torch.export.Dim("y_dim_1")
        dynamic_shapes = {
            "x": {0: x_dim},
            "y": {1: y_dim},
        }
        unflatten_dynamic_shapes = _compat._unflatten_dynamic_shapes_with_inputs_tree(
            inputs, dynamic_shapes
        )
        
        expected_dynamic_shapes = {
            "x": {0: x_dim},
            "y": {1: y_dim},
        }
        self.assertEqual(unflatten_dynamic_shapes, expected_dynamic_shapes)
        
    def test__unflatten_dynamic_shapes_with_inputs_tree_succeeds_on_tuple_of_mixed_structure(self):
        inputs = (torch.randn(1, 2, 3), ({"x0": torch.randn(1, 2, 3)}, {"x1": torch.randn(1, 2, 3)}), (torch.randn(1, 2, 3), torch.randn(1, 2, 3)), [torch.randn(1, 2, 3), torch.randn(1, 2, 3)])
        w_dim_0 = torch.export.Dim("w_dim_0")
        x0_dim_1 = torch.export.Dim("x0_dim_1")
        x0_dim_2 = torch.export.Dim("x0_dim_2")
        x1_dim_1 = torch.export.Dim("x1_dim_1")
        y0_dim_0 = torch.export.Dim("y0_dim_0")
        y0_dim_1 = torch.export.Dim("y0_dim_1")
        y1_dim_2 = torch.export.Dim("y1_dim_2")
        z0_dim_2 = torch.export.Dim("z0_dim_2")
        z1_dim_1 = torch.export.Dim("z1_dim_1")
        dynamic_shapes = {
            "w": {0: w_dim_0},
            "x0": {1: x0_dim_1, 2: x0_dim_2},
            "x1": {1: x1_dim_1},
            "y0": {0: y0_dim_0, 1: y0_dim_1},
            "y1": {2: y1_dim_2},
            "z0": {2: z0_dim_2},
            "z1": {1: z1_dim_1},
            
        }
        unflatten_dynamic_shapes = _compat._unflatten_dynamic_shapes_with_inputs_tree(
            inputs, dynamic_shapes
        )
        expected_dynamic_shapes = (
            {0: w_dim_0},
            ({"x0": {1: x0_dim_1, 2: x0_dim_2}}, {"x1": {1: x1_dim_1}}),
            ({0: y0_dim_0, 1: y0_dim_1}, {2: y1_dim_2}),
            [{2:z0_dim_2}, {1: z1_dim_1}],
        )
        self.assertEqual(unflatten_dynamic_shapes, expected_dynamic_shapes)

if __name__ == "__main__":
    common_utils.run_tests()
