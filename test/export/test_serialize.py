# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export
from torch._export.serde.serialize import (
    ExportedProgramSerializer,
    deserialize,
    serialize, GraphModuleOpUpgrader,
)
import torch.utils._pytree as pytree
from torch.testing._internal.common_utils import run_tests, TestCase

TEST_UPGRADERS = {
    "aten::div__Scalar_0_3": (
        "div.Scalar(Tensor self, Scalar other) -> Tensor",
        """
def div__Scalar_0_3(self: torch.Tensor, other) -> torch.Tensor:
  if (self.is_floating_point() or isinstance(other, float)):
    return self.true_divide(other)
  return self.divide(other, rounding_mode='trunc')
        """)
}
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
        node = serialized.graph_module.graph.nodes[-7]
        self.assertEqual(node.target, "torch._ops.aten.var_mean.correction")
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
        node = serialized.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch._ops.aten.split.Tensor")
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
        node = serialized.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch._ops.aten.var_mean.correction")
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

        node = serialized.graph_module.graph.nodes[-1]
        self.assertEqual(node.target, "torch._ops.aten.searchsorted.Tensor")
        self.assertEqual(len(node.inputs), 6)
        self.assertEqual(node.inputs[2].arg.as_bool, False)
        self.assertEqual(node.inputs[3].arg.as_bool, True)
        self.assertEqual(node.inputs[4].arg.as_string, "right")
        self.assertEqual(node.inputs[5].arg.as_none, ())


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestDeserialize(TestCase):
    def check_graph(self, fn, inputs) -> None:
        """Export a graph, serialize it, deserialize it, and compare the results."""
        # TODO(angelayi): test better with some sort of wrapper around all
        # export tests

        ep = export(fn, inputs, [])
        serialized_struct, state_dict = serialize(ep)
        deserialized_ep = deserialize(serialized_struct, state_dict)

        orig_outputs = ep(*inputs)
        loaded_outputs = deserialized_ep(*inputs)

        flat_orig_outputs, _ = pytree.tree_flatten(orig_outputs)
        flat_loaded_outputs, _ = pytree.tree_flatten(loaded_outputs)

        for orig, loaded in zip(flat_orig_outputs, flat_loaded_outputs):
            self.assertTrue(torch.allclose(orig, loaded))

    def test_multi_return(self) -> None:
        """
        Test multiple return from a single node (ex. layer_norm has 2 outputs)
        """
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

        inputs = (
            torch.ones([512, 512], requires_grad=True),
            torch.ones([512]),
            torch.ones([512]),
        )
        self.check_graph(MyModule(), inputs)

    def test_basic(self) -> None:
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = x + x
                x = x * x
                x = x / x
                return x, x.clone()

        inputs = (torch.ones([512], requires_grad=True),)
        self.check_graph(MyModule(), inputs)

class TestOpVersioning(TestCase):
    """Test if serializer/deserializer behaves correctly if version mismatch."""
    def test_creates_upgrader_pass(self):

        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)
        self.assertEqual(len(upgrader.upgrader_passes), 1)

    def test_div_upgrader_replaces_op_with_old_version(self):
        def fn(a: torch.Tensor, b):
            return torch.ops.aten.div.Scalar(a, b)

        inputs = (torch.ones([2, 3]) * 4, 2)
        ep = export(fn, inputs, [])
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)
        ep.transform(*upgrader.upgrader_passes)

        def count_op(graph, target_str):
            return len([n for n in graph.nodes if isinstance(n.target, torch._ops.OpOverload) and n.target.name() == target_str])
        count = count_op(ep.graph, "aten::div.Scalar")
        self.assertEqual(count, 0)
        custom_op_count = count_op(ep.graph, "aten::div__Scalar_0_3")
        self.assertEqual(custom_op_count, 1)

    def test_div_upgrader_pass_return_new_op_after_retrace(self):
        def fn(a: torch.Tensor, b):
            return torch.ops.aten.div.Scalar(a, b)

        inputs = (torch.ones([2, 3]) * 4, 2)
        ep = export(fn, inputs, [])
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)

        def count_op(graph, target_str):
            return len([n for n in graph.nodes if isinstance(n.target, torch._ops.OpOverload) and n.target.name() == target_str])
        count = count_op(ep.graph, "aten::div.Scalar")
        self.assertEqual(count, 1)

        upgraded_ep = upgrader.upgrade(ep)

        custom_op_count = count_op(upgraded_ep.graph, "aten::div__Scalar_0_3")
        self.assertEqual(custom_op_count, 0)

if __name__ == '__main__':
    run_tests()
