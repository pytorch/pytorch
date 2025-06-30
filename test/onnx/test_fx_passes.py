# Owner(s): ["module: onnx"]
import torch
import torch._dynamo
import torch.fx
from torch.onnx._internal.exporter import _fx_passes
from torch.onnx._internal.fx.passes import _utils as pass_utils
from torch.testing._internal import common_utils


class TestFxPasses(common_utils.TestCase):
    def test_set_node_name_correctly_renames_when_new_name_collides_recursively(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        gm, _ = torch._dynamo.export(func)(x, y, z)
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
        assert len({node.name for node in nodes}) == len(nodes), (
            f"Expected all names to be unique, got {nodes}"
        )

    def test_set_node_name_succeeds_when_no_name_collisions(self):
        def func(x, y, z):
            return x + y + z

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        gm, _ = torch._dynamo.export(func)(x, y, z)
        torch._dynamo.reset()

        # Run `set_node_name` and verify that the names are correct.
        new_name = "some_tensor"
        nodes = list(gm.graph.nodes)
        name_to_node = {node.name: node for node in nodes}
        pass_utils.set_node_name(nodes[1], new_name, name_to_node)
        assert nodes[1].name == new_name, f"Expected {new_name}, got {nodes[0].name}"
        assert len({node.name for node in nodes}) == len(nodes), (
            f"Expected all names to be unique, got {nodes}"
        )

    def test_remove_unnecessary_slices(self):
        class Model(torch.nn.Module):
            def forward(self, causal_mask, fill_value):
                causal_mask = causal_mask.clone()
                mask_length = fill_value.shape[-1]
                causal_mask[:, :, :, :mask_length] = fill_value
                return causal_mask

        B = 2
        N = 2
        S = 3
        T = 4
        T2 = 3
        causal_mask = torch.randn(B, N, S, T)
        fill_value = torch.randn(B, N, S, T2)
        inputs = (causal_mask, fill_value)
        model = Model()
        expected = model(*inputs)
        DYN = torch.export.Dim.DYNAMIC
        ep = torch.export.export(model, inputs, dynamic_shapes=({3: DYN}, {3: DYN}))
        node_targets = [
            node.target.name()
            for node in ep.graph.nodes
            if hasattr(node.target, "name")
        ]
        self.assertEqual(
            [
                "aten::sym_size.int",
                "aten::clone",
                "aten::slice.Tensor",
                "aten::slice.Tensor",
                "aten::slice.Tensor",
                "aten::slice.Tensor",
                "aten::copy_",
            ],
            node_targets,
        )
        gm = _fx_passes.remove_unnecessary_slices(ep.module())
        node_targets = [
            node.target.name()
            for node in gm.graph.nodes
            if hasattr(node.target, "name")
        ]
        self.assertEqual(
            [
                "aten::sym_size.int",
                "aten::clone",
                "aten::slice.Tensor",
                "aten::copy_",
            ],
            node_targets,
        )
        torch.testing.assert_close(gm(*inputs), expected)


if __name__ == "__main__":
    common_utils.run_tests()
