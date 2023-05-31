# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export
from torch._export.serde.serialize import ExportedProgramSerializer
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestSerialize(TestCase):
    def test_serialize_multiple_returns_from_node(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w, b):
                return torch.nn.functional.layer_norm(
                    x,
                    x.size()[1:],
                    weight=w,
                    bias=b,
                    eps=1e-5,
                )

        exported_module = export(
            MyModule(),
            (
                torch.ones([512, 512], requires_grad=True),
                torch.ones([512]),
                torch.ones([512]),
            ),
        )

        serialized, _ = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.graph_module.graph.nodes[0]
        self.assertEqual(node.target, "aten.var_mean.correction")
        # aten::native_layer_norm returns 3 tensnors
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_serialize_list_returns(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.split(x, 2)

        input = torch.arange(10.0).reshape(5, 2)
        input.requires_grad = True
        exported_module = export(MyModule(), (input,))

        serialized, _ = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.graph_module.graph.nodes[0]
        self.assertEqual(node.target, "aten.split.Tensor")
        self.assertEqual(len(node.outputs), 1)
        # Input looks like:
        # tensor([[0, 1],
        #         [2, 3],
        #         [4, 5],
        #         [6, 7],
        #         [8, 9]])
        # Output looks like:
        # (tensor([[0, 1],
        #          [2, 3]]),
        #  tensor([[4, 5],
        #          [6, 7]]),
        #  tensor([[8, 9]]))
        self.assertEqual(len(node.outputs[0].as_tensors), 3)

        # check the names are unique
        seen = set()
        for output in node.outputs[0].as_tensors:
            name = output.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_multi_return_some_unused(self) -> None:
        """
        Make sure the serialized output matches the op schema, even if some of
        the arguments are never used in the graph.
        """

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.var_mean.correction(x, [1])[0]

        exported_module = export(
            MyModule(),
            (torch.ones([512, 512], requires_grad=True),),
        )

        serialized, _ = ExportedProgramSerializer().serialize(exported_module)
        node = serialized.graph_module.graph.nodes[0]
        self.assertEqual(node.target, "aten.var_mean.correction")
        self.assertEqual(len(node.outputs), 2)

        # check the names are unique
        seen = set()
        for output in node.outputs:
            name = output.as_tensor.name
            self.assertNotIn(name, seen)
            seen.add(name)

    def test_kwargs_default(self) -> None:
        """
        Tests that the kwargs default values are serialized even if they are not
        specified
        """

        def f(x: torch.Tensor) -> torch.Tensor:
            values = torch.randn(3, 2)
            return torch.searchsorted(x, values, side="right", right=True)

        x, _ = torch.sort(torch.randn(3, 4))
        exported_module = export(f, (x,))
        serialized, _ = ExportedProgramSerializer().serialize(exported_module)

        node = serialized.graph_module.graph.nodes[1]
        self.assertEqual(node.target, "aten.searchsorted.Tensor")
        self.assertEqual(len(node.inputs), 6)
        self.assertEqual(node.inputs[2].arg.as_bool, False)
        self.assertEqual(node.inputs[3].arg.as_bool, True)
        self.assertEqual(node.inputs[4].arg.as_string, "right")
        self.assertEqual(node.inputs[5].arg.as_none, ())


if __name__ == '__main__':
    run_tests()
