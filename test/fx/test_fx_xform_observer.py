# Owner(s): ["module: fx"]

import copy
import os
import tempfile

import torch
from torch.fx import subgraph_rewriter, symbolic_trace
from torch.fx.passes.graph_transform_observer import GraphTransformObserver
from torch.fx.traceback import NodeSourceAction
from torch.testing._internal.common_utils import TestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_fx.py TESTNAME\n\n"
        "instead."
    )


class TestGraphTransformObserver(TestCase):
    def test_graph_transform_observer(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x)

        def replacement(x):
            return torch.relu(x)

        traced = symbolic_trace(M())

        log_url = tempfile.mkdtemp()

        with GraphTransformObserver(
            traced, "replace_neg_with_relu", log_url=log_url
        ) as ob:
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            self.assertTrue("relu" in ob.created_nodes)
            self.assertTrue("neg" in ob.erased_nodes)

        current_pass_count = GraphTransformObserver.get_current_pass_count()

        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_input_graph.dot",
                )
            )
        )
        self.assertTrue(
            os.path.isfile(
                os.path.join(
                    log_url,
                    f"pass_{current_pass_count}_replace_neg_with_relu_output_graph.dot",
                )
            )
        )

    def test_graph_transform_observer_node_tracking(self):
        class M(torch.nn.Module):
            def forward(self, x):
                val = torch.neg(x)
                return torch.add(val, val)

        def pattern(x):
            return torch.neg(x)

        def replacement(x):
            return torch.relu(x)

        def replacement2(x):
            return torch.cos(x)

        traced = symbolic_trace(M())

        def check_node_source(node_source, node_name, target, id, pass_name, action):
            self.assertEqual(node_source.name, node_name)
            self.assertEqual(node_source.target, target)
            self.assertEqual(node_source.pass_name, pass_name)
            self.assertEqual(node_source.graph_id, id)
            self.assertEqual(node_source.action, action)

        with GraphTransformObserver(traced, "replace_neg_with_relu") as ob:
            subgraph_rewriter.replace_pattern(traced, pattern, replacement)

            self.assertTrue("relu" in ob.created_nodes)
            self.assertTrue("neg" in ob.erased_nodes)

        self.assertEqual(len(traced._replace_hooks), 0)
        self.assertEqual(len(traced._create_node_hooks), 0)
        self.assertEqual(len(traced._erase_node_hooks), 0)
        self.assertEqual(len(traced._deepcopy_hooks), 0)

        for node in traced.graph.nodes:
            if node.name == "relu":
                from_node = node.meta["from_node"]
                self.assertTrue(len(from_node) == 1)
                check_node_source(
                    from_node[0],
                    "neg",
                    str(torch.neg),
                    id(traced.graph),
                    "replace_neg_with_relu",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )

        with GraphTransformObserver(traced, "replace_relu_with_cos") as ob:
            subgraph_rewriter.replace_pattern(traced, replacement, replacement2)

            self.assertTrue("cos" in ob.created_nodes)
            self.assertTrue("relu" in ob.erased_nodes)

        for node in traced.graph.nodes:
            if node.name == "cos":
                from_node = node.meta["from_node"]
                self.assertTrue(len(from_node) == 1)
                check_node_source(
                    from_node[0],
                    "relu",
                    str(torch.relu),
                    id(traced.graph),
                    "replace_relu_with_cos",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )
                check_node_source(
                    from_node[0].from_node[0],
                    "neg",
                    str(torch.neg),
                    id(traced.graph),
                    "replace_neg_with_relu",
                    [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
                )

        class SimpleLinearModel(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        model = SimpleLinearModel()
        gm = torch.export.export(model, (torch.rand(10),)).module()

        with GraphTransformObserver(gm, "test"):
            add_node = gm.graph.call_function(torch.ops.aten.add.default, (1, 1))
            neg_node = next(
                iter([node for node in gm.graph.nodes if node.name == "neg"])
            )
            neg_node.replace_all_uses_with(replace_with=add_node)

        from_node = add_node.meta["from_node"]
        self.assertTrue(len(from_node) == 1)
        check_node_source(
            from_node[0],
            "neg",
            str(torch.ops.aten.neg.default),
            id(gm.graph),
            "test",
            [NodeSourceAction.REPLACE, NodeSourceAction.CREATE],
        )

    def test_graph_transform_observer_deepcopy(self):
        class SimpleLinearModel(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        model = SimpleLinearModel()
        gm = torch.export.export(model, (torch.rand(10),)).module()

        with GraphTransformObserver(gm, "test"):
            gm2 = copy.deepcopy(gm)

        nodes = [node.name for node in gm.graph.nodes]
        nodes2 = [node.name for node in gm2.graph.nodes]
        self.assertEqual(nodes, nodes2)

        # deepcopied graph modules should not have hooks after exiting
        # the context
        self.assertEqual(len(gm2._replace_hooks), 0)
        self.assertEqual(len(gm2._create_node_hooks), 0)
        self.assertEqual(len(gm2._erase_node_hooks), 0)
        self.assertEqual(len(gm2._deepcopy_hooks), 0)
