# Owner(s): ["module: onnx"]

"""Tests for onnx export that don't run the exported model."""

import io
import unittest

import onnx
import torch
from torch import Tensor
from torch.onnx import symbolic_helper

from typing import Optional, Type

class TestONNXNoRuntime(unittest.TestCase):
    def run_optional_output_test(self, module_class: Type[torch.nn.Module]):
        # Need scripting to preserve control flow for this test to be meaningful.
        model = torch.jit.script(module_class())
        f = io.BytesIO()
        x = torch.ones(1)
        torch.onnx.export(
            model, (x,), f, opset_version=15,
            # Ensure condition is not constant
            dynamic_axes={"x": {0: "condition"}}, input_names=["x"])
        exported = onnx.load_from_string(f.getvalue())
        output_0_type = exported.graph.output[0].type
        self.assertTrue(output_0_type.HasField("optional_type"))
        output_0_optional_type = output_0_type.optional_type
        self.assertTrue(output_0_optional_type.elem_type.HasField("tensor_type"))
        output_0_tensor_type = output_0_optional_type.elem_type.tensor_type
        self.assertEqual(
            output_0_tensor_type.elem_type,
            symbolic_helper.scalar_type_to_onnx[symbolic_helper.scalar_type_to_pytorch_type.index(x.dtype)])
        self.assertEqual(len(output_0_tensor_type.shape.dim), len(x.shape))

    def test_loop_none_output(self):
        class Model(torch.nn.Module):
            def forward(self, x) -> Optional[Tensor]:
                y: Optional[Tensor] = x
                for _ in range(x.size(0)):
                    y = None
                return y

        self.run_optional_output_test(Model)

    def test_loop_none_input(self):
        class Model(torch.nn.Module):
            def forward(self, x) -> Optional[Tensor]:
                y: Optional[Tensor] = None
                for _ in range(x.size(0)):
                    y = x
                return y

        self.run_optional_output_test(Model)

    def test_if_none_output(self):
        class Model(torch.nn.Module):
            def forward(self, x) -> Optional[Tensor]:
                y: Optional[Tensor] = x
                if x.size(0) > 1:
                    y = None
                return y

        self.run_optional_output_test(Model)

    def test_if_none_input(self):
        class Model(torch.nn.Module):
            def forward(self, x) -> Optional[Tensor]:
                y: Optional[Tensor] = None
                if x.size(0) > 1:
                    y = x
                return y

        self.run_optional_output_test(Model)


if __name__ == "__main__":
    unittest.main()
