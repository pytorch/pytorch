# Owner(s): ["module: fx"]
import json

import torch
from torch._inductor.compile_fx import aot_export_module
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
        ep = torch.export.export(
            model,
            example_inputs,
        )
        gm = ep.module()
        provenance = get_graph_provenance_json(gm.graph)
        provenance = json.loads(provenance)
        self.assertEqual(
            set(provenance.keys()), {"relu", "linear", "sigmoid", "linear_1"}
        )

        # Check node "linear" is created from node "x" in PropagateUnbackedSymInts
        key_provenance = provenance["linear"]
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
        provenance = json.loads(provenance)

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
            key_provenance = get_first_node_source_and_check(key_provenance)
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
