# Owner(s): ["module: onnx"]
from __future__ import annotations

import torch._dynamo
import torch.fx
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


if __name__ == "__main__":
    common_utils.run_tests()
