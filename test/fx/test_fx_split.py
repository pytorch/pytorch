# Owner(s): ["module: fx"]

import torch
from torch.fx.passes.split_utils import split_by_tags

from torch.testing._internal.common_utils import TestCase


class TestFXSplit(TestCase):
    def test_split_preserve_node_meta(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                x = x + x
                y = y * y
                return x - y

        gm = torch.fx.symbolic_trace(TestModule())
        for node in gm.graph.nodes:
            node.meta["name"] = node.name
            if node.name == "add":
                node.tag = "a"
            elif node.name == "mul":
                node.tag = "b"
            elif node.name == "sub":
                node.tag = "c"

        split_gm = split_by_tags(gm, ["a", "b", "c"])
        for m in split_gm.children():
            for n in m.graph.nodes:
                if n.op != "output":
                    self.assertIn("name", n.meta)
                    self.assertEqual(n.meta["name"], n.name)
