# Owner(s): ["module: onnx"]
import torch
import torch._dynamo
import torch.fx

from torch._custom_op import impl as custom_op
from torch.onnx._internal.fx.passes import _utils as pass_utils
from torch.testing._internal import common_utils


class TestFxPasses(common_utils.TestCase):
    def test_set_node_name_correctly_renames_when_new_name_collides_recursively(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        gm, _ = torch._dynamo.export(func, x, y, z)
        torch._dynamo.reset()

        # Purposely name the nodes in a way that will cause a recursive collision later.
        # See :func:`set_node_name` for name collision renaming logic.
        base_name = "tensor"
        nodes = list(gm.graph.nodes)
        for i, node in enumerate(nodes[1:]):
            if i == 0:
                node.name = base_name
            else:
                node.name = f"{base_name}.{i}"

        # Run `set_node_name` and verify that the names are correct.
        name_to_node = {node.name: node for node in gm.graph.nodes}
        pass_utils.set_node_name(nodes[0], base_name, name_to_node)
        assert nodes[0].name == base_name, f"Expected {base_name}, got {nodes[0].name}"
        assert len({node.name for node in nodes}) == len(
            nodes
        ), f"Expected all names to be unique, got {nodes}"

    def test_set_node_name_succeeds_when_no_name_collisions(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        gm, _ = torch._dynamo.export(func, x, y, z)
        torch._dynamo.reset()

        # Run `set_node_name` and verify that the names are correct.
        new_name = "some_tensor"
        nodes = list(gm.graph.nodes)
        name_to_node = {node.name: node for node in nodes}
        pass_utils.set_node_name(nodes[1], new_name, name_to_node)
        assert nodes[1].name == new_name, f"Expected {new_name}, got {nodes[0].name}"
        assert len({node.name for node in nodes}) == len(
            nodes
        ), f"Expected all names to be unique, got {nodes}"

    def test_onnx_dynamo_export_raises_when_model_contains_unsupported_fx_nodes(self):
        @custom_op.custom_op("mylibrary::foo_op")
        def foo_op(x: torch.Tensor) -> torch.Tensor:
            ...

        @custom_op.custom_op("mylibrary::bar_op")
        def bar_op(x: torch.Tensor) -> torch.Tensor:
            ...

        @foo_op.impl_abstract()
        def foo_op_impl_abstract(x):
            return torch.empty_like(x)

        @foo_op.impl("cpu")
        def foo_op_impl(x):
            return x + 1

        @bar_op.impl_abstract()
        def bar_op_impl_abstract(x):
            return torch.empty_like(x)

        @bar_op.impl("cpu")
        def bar_op_impl(x):
            return x + 2

        torch._dynamo.allow_in_graph(foo_op)
        torch._dynamo.allow_in_graph(bar_op)

        def func(x, y, z):
            return foo_op(x) + bar_op(y) + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        with self.assertRaises(torch.onnx.OnnxExporterError) as ctx:
            torch.onnx.dynamo_export(func, x, y, z)
        inner_exception = ctx.exception.__cause__
        self.assertRegex(
            str(inner_exception),
            r"Unsupported FX nodes.*mylibrary\.foo_op.*mylibrary\.bar_op",
        )

        torch._dynamo.reset()


if __name__ == "__main__":
    common_utils.run_tests()
