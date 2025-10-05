# Owner(s): ["module: fx"]

import torch
from torch._inductor.compile_fx import aot_export_module
from torch.export import default_decompositions
from torch.fx.traceback import get_graph_provenance_json, NodeSource, NodeSourceAction
from torch.testing._internal.common_utils import TestCase


CREATE_STR = NodeSourceAction.CREATE.name.lower()


class TestFXNodeSource(TestCase):
    def test_node_source(self):
        node_source = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        self.assertExpectedInline(
            node_source.print_readable().strip(),
            """(name=, pass_name=test_pass, action=create, graph_id=-1)""",
        )
        dummy_source_dict = {
            "name": "",
            "target": "",
            "pass_name": "test_pass",
            "action": CREATE_STR,
            "graph_id": -1,
            "from_node": [],
        }
        self.assertEqual(
            node_source.to_dict(),
            dummy_source_dict,
        )

        self.assertEqual(node_source, NodeSource._from_dict(node_source.to_dict()))

        # Dummy node
        node = torch.fx.Node(
            graph=torch.fx.Graph(),
            name="add",
            op="call_function",
            target=torch.ops.aten.add.Tensor,  # type: ignore[attr-defined]
            args=(torch.tensor(3), torch.tensor(4)),
            kwargs={},
        )
        node.meta["from_node"] = [node_source]

        graph_id = id(node.graph)
        node_source = NodeSource(
            node=node, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        self.assertExpectedInline(
            node_source.print_readable().strip(),
            f"""\
(name=add, pass_name=test_pass, action=create, graph_id={graph_id})
    (name=, pass_name=test_pass, action=create, graph_id=-1)""",
        )
        self.assertEqual(
            node_source.to_dict(),
            {
                "name": "add",
                "target": "aten.add.Tensor",
                "pass_name": "test_pass",
                "action": CREATE_STR,
                "graph_id": graph_id,
                "from_node": [dummy_source_dict],
            },
        )

        # Test two node sources are same
        node_source1 = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        node_source2 = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        self.assertEqual(node_source1, node_source2)

        # Test hash function - equivalent objects should have same hash
        self.assertEqual(hash(node_source1), hash(node_source2))

        # Test two node sources are not same
        node_source3 = NodeSource(
            node=None, pass_name="test_pass_1", action=NodeSourceAction.CREATE
        )
        node_source4 = NodeSource(
            node=None, pass_name="test_pass_2", action=NodeSourceAction.CREATE
        )
        self.assertNotEqual(node_source3, node_source4)

        # Test hash function - different objects should have different hash
        self.assertNotEqual(hash(node_source3), hash(node_source4))

        # Test that equivalent NodeSource objects can be used in sets and dicts
        node_set = {node_source1, node_source2}
        self.assertEqual(len(node_set), 1)  # Should only contain one unique element

        node_dict = {node_source1: "value1", node_source2: "value2"}
        self.assertEqual(len(node_dict), 1)  # Should only contain one key
        self.assertEqual(node_dict[node_source1], "value2")  # Last value should win

        # Test with more complex NodeSource objects
        node_source_with_node = NodeSource(
            node=node, pass_name="test_pass", action=NodeSourceAction.CREATE
        )
        node_source_with_node_copy = NodeSource(
            node=node, pass_name="test_pass", action=NodeSourceAction.CREATE
        )

        # These should be equal and have same hash
        self.assertEqual(node_source_with_node, node_source_with_node_copy)
        self.assertEqual(hash(node_source_with_node), hash(node_source_with_node_copy))

        # Test with different actions
        node_source_replace = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.REPLACE
        )
        node_source_create = NodeSource(
            node=None, pass_name="test_pass", action=NodeSourceAction.CREATE
        )

        # These should be different and have different hashes
        self.assertNotEqual(node_source_replace, node_source_create)
        self.assertNotEqual(hash(node_source_replace), hash(node_source_create))

    def test_graph_provenance(self):
        def check_node_source(node_source_dict, name, pass_name, action):
            self.assertEqual(node_source_dict["name"], name)
            self.assertEqual(node_source_dict["pass_name"], pass_name)
            self.assertEqual(node_source_dict["action"], action)

        def get_first_node_source_and_check(node_source_dict):
            """
            Get the first node source from the from_node list.
            """
            self.assertEqual(len(node_source_dict["from_node"]), 1)
            return node_source_dict["from_node"][0]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(16, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return (x,)

        model = Model()
        example_inputs = (torch.randn(8, 10),)
        ep = torch.export.export(model, example_inputs, strict=True)

        decomposed_ep = ep.run_decompositions(default_decompositions())
        # node decomposed from same ancestor node should have same from_node info
        for node in decomposed_ep.graph.nodes:
            if node.op not in {"placeholder", "output"}:
                assert "from_node" in node.meta

        node_name_to_from_node = {
            node.name: node.meta["from_node"]
            for node in decomposed_ep.graph.nodes
            if node.op not in {"placeholder", "output"}
        }
        same_ancestor_nodes = {
            "permute": "addmm",
            "addmm": "permute",
            "permute_1": "addmm_1",
            "addmm_1": "permute_1",
        }

        for node_name_1 in node_name_to_from_node:
            for node_name_2 in node_name_to_from_node:
                if node_name_2 in {
                    node_name_1,
                    same_ancestor_nodes[node_name_1]
                    if node_name_1 in same_ancestor_nodes
                    else None,
                }:
                    self.assertEqual(
                        node_name_to_from_node[node_name_1],
                        node_name_to_from_node[node_name_2],
                    )
                    self.assertEqual(
                        [
                            NodeSource._from_dict(ns.to_dict())
                            for ns in node_name_to_from_node[node_name_1]
                        ],
                        node_name_to_from_node[node_name_2],
                    )
                else:
                    self.assertNotEqual(
                        node_name_to_from_node[node_name_1],
                        node_name_to_from_node[node_name_2],
                    )
                    self.assertNotEqual(
                        [
                            NodeSource._from_dict(ns.to_dict())
                            for ns in node_name_to_from_node[node_name_1]
                        ],
                        node_name_to_from_node[node_name_2],
                    )

        gm = ep.module()
        provenance = get_graph_provenance_json(gm.graph)
        self.assertEqual(
            set(provenance.keys()), {"relu", "linear", "sigmoid", "linear_1"}
        )

        # Check node "linear" is created from node "x" in PropagateUnbackedSymInts
        key_provenance = provenance["linear"][0]["from_node"]
        self.assertEqual(len(key_provenance), 1)
        key_provenance = key_provenance[0]
        check_node_source(
            key_provenance,
            "x",
            "Interpreter_PropagateUnbackedSymInts",
            CREATE_STR,
        )

        # Check node "x" is then created from another node "x" in FlattenInputOutputSignature
        key_provenance = get_first_node_source_and_check(key_provenance)
        check_node_source(
            key_provenance,
            "x",
            "Interpreter_FlattenInputOutputSignature",
            CREATE_STR,
        )

        gm, graph_signature = aot_export_module(
            gm,
            example_inputs,
            trace_joint=False,
        )

        provenance = get_graph_provenance_json(gm.graph)

        self.assertEqual(
            set(provenance.keys()), {"t", "addmm", "relu", "t_1", "addmm_1", "sigmoid"}
        )
        for key in ["t", "addmm"]:
            # The node provenance hierarchy should be:
            # t -> linear -> x -> x
            #
            # x -> y means x is created from y

            key_provenance = provenance[key]
            self.assertEqual(len(key_provenance), 1)
            key_provenance = key_provenance[0]

            # Check node "t" and "addmm" is created from node "linear" in PropagateUnbackedSymInts
            check_node_source(
                key_provenance,
                "linear",
                "Interpreter_PropagateUnbackedSymInts",
                CREATE_STR,
            )

            # Check node "linear" is then created from node "x" in PropagateUnbackedSymInts
            key_provenance = get_first_node_source_and_check(key_provenance)[
                "from_node"
            ][0]
            check_node_source(
                key_provenance,
                "x",
                "Interpreter_PropagateUnbackedSymInts",
                CREATE_STR,
            )

            # Check node "x" is then created from another node "x" in FlattenInputOutputSignature
            key_provenance = get_first_node_source_and_check(key_provenance)
            check_node_source(
                key_provenance,
                "x",
                "Interpreter_FlattenInputOutputSignature",
                CREATE_STR,
            )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
