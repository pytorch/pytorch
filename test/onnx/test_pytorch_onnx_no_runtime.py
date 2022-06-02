# Owner(s): ["module: onnx"]

"""Tests for onnx export that don't run the exported model."""

import io
import unittest
from typing import Optional, Type

import onnx

import torch
from torch import Tensor
from torch.onnx import symbolic_helper
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class TestOptionalOutput(unittest.TestCase):
    # TODO: Move these tests to test_pytorch_onnx_onnxruntime once
    # ONNX Runtime 1.11 is released and supports opset 16.

    class IfNoneInput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = None
            if x.size(0) > 1:
                y = x
            return y

    class IfNoneOutput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = x
            if x.size(0) > 1:
                y = None
            return y

    class LoopNoneInput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = None
            for _ in range(x.size(0)):
                y = x
            return y

    class LoopNoneOutput(torch.nn.Module):
        def forward(self, x) -> Optional[Tensor]:
            y: Optional[Tensor] = x
            for _ in range(x.size(0)):
                y = None
            return y

    @parametrize(
        "module_class",
        (IfNoneInput, IfNoneOutput, LoopNoneInput, LoopNoneOutput),
        name_fn=lambda module_class: module_class.__name__,
    )
    @parametrize("x_size", (0, 1), name_fn=lambda x_size: str(x_size))
    def test_optional_output(self, module_class: Type[torch.nn.Module], x_size: int):
        # Need scripting to preserve control flow for this test to be meaningful.
        model = torch.jit.script(module_class())
        f = io.BytesIO()
        x = torch.ones(x_size)
        dynamic_axis_name = "condition"
        torch.onnx.export(
            model,
            (x,),
            f,
            opset_version=15,
            # Ensure condition is not constant
            dynamic_axes={"x": {0: dynamic_axis_name}},
            input_names=["x"],
        )
        exported = onnx.load_from_string(f.getvalue())
        expected_elem_type = symbolic_helper.scalar_type_to_onnx[
            symbolic_helper.scalar_type_to_pytorch_type.index(x.dtype)
        ].value
        expected_output_type = onnx.helper.make_optional_type_proto(
            onnx.helper.make_tensor_type_proto(expected_elem_type, (dynamic_axis_name,))
        )
        self.assertEqual(expected_output_type, exported.graph.output[0].type)
        for node in exported.graph.node:
            # Both branches output types should match.
            if node.op_type == "If":
                for attr in node.attribute:
                    if attr.name in ("then_branch", "else_branch"):
                        self.assertEqual(expected_output_type, attr.g.output[0].type)

    def test_uninitialized_optional(self):
        class Module(torch.nn.Module):
            def forward(self, y: Optional[Tensor]) -> Optional[Tensor]:
                if y is not None:
                    if y.shape[1] < 5:
                        if y.size(0) == 1:
                            y = y + 4
                        else:
                            return y
                return y

        y = torch.ones((3, 4), dtype=torch.int)
        torch.onnx.export(
            torch.jit.script(Module()),
            y,
            io.BytesIO(),
            opset_version=15,
            dynamic_axes={"y": {0: "y0", 1: "y1"}},
            input_names=["y"],
        )


instantiate_parametrized_tests(TestOptionalOutput)


if __name__ == "__main__":
    unittest.main()
