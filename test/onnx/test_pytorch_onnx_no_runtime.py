# Owner(s): ["module: onnx"]

"""Tests for onnx export that don't run the exported model."""

import io
import unittest
from typing import Optional, Type

import onnx
import onnx.numpy_helper

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

    def test_maintain_dynamic_shapes_of_unreliable_nodes(self):
        def symbolic_pythonop(ctx: torch.onnx.SymbolicContext, g, *args, **kwargs):
            return g.op("com.microsoft::PythonOp")

        torch.onnx.register_custom_op_symbolic("prim::PythonOp", symbolic_pythonop, 1)
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, "prim::PythonOp", 1)

        # necessay parameters for transformer embeddings
        hidden_size = 48
        max_position_embeddings = 32
        batch_size = 2

        # issue found that autograd.function making downstream
        # node unreliable but with static shape. The issue was first
        # discovered with using Apex FusedLayerNorm in Transformers
        class CustomLayerNorm(torch.autograd.Function):
            @staticmethod
            def forward(ctx, embedding):
                layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-12)
                return layer_norm(embedding)

        class EmbeddingModule(torch.nn.Module):
            def forward(
                self,
                embeddings=None,
            ):
                embedding_output = CustomLayerNorm.apply(embeddings)
                query = embedding_output.transpose(0, 1)
                target_len, batch_size, embedding_dim = query.size()
                # Reshape is used for consuming batch_size, and if it is static,
                # this will be a Constant node in the graph
                query = query.reshape(target_len, batch_size, embedding_dim)
                return query

        embeddings = torch.randn(batch_size, max_position_embeddings, hidden_size)

        f = io.BytesIO()
        torch.onnx.export(
            EmbeddingModule().eval(),
            (embeddings,),
            f,
            input_names=["embeddings"],
            dynamic_axes={
                "embeddings": {
                    0: "batch_size",
                    1: "max_position_embeddings",
                    2: "hidden_size",
                }
            },
            custom_opsets={"com.microsoft": 1},
        )
        model = onnx.load(io.BytesIO(f.getvalue()))

        # If there is a constant node with dim=3 and max_position_embeddings,
        # batch_size, hidden_size as shape, it means the shape becomes static.
        # Normally, with dynamic batch size, this constant node should not exist.
        const_node = [n for n in model.graph.node if n.op_type == "Constant"]
        self.assertNotEqual(len(const_node), 0)
        for node in const_node:
            for a in node.attribute:
                if a.name == "value":
                    shape = onnx.numpy_helper.to_array(a.t)
                    self.assertNotEqual(
                        shape.tolist(),
                        [max_position_embeddings, batch_size, hidden_size],
                    )


instantiate_parametrized_tests(TestOptionalOutput)

if __name__ == "__main__":
    unittest.main()
