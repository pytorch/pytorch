import json
from typing import Dict, Any

import glow.fb.fx_glow_binding.fx_glow as fx_glow
from torch.fx.experimental.graph_manipulation import serialize_module
from torch.fx.graph_module import GraphModule


class AcceleratedGraphModule:
    def __init__(self, fx_module: GraphModule):
        """Creates the needed data structures to pass to the glow runtime"""
        self.weights: Dict[str, Any] = {}
        self.serialized_graph_json = json.dumps(serialize_module(fx_module, self.weights), indent=4)
        self.binding = fx_glow.fx_glow()
        self.binding.setupHost()

    def load_graph(self, dag):
        """Loads the graph onto the card, requires the dag from the partitioner to be passed in."""
        nodes = {}
        for d in dag.nodes:
            node_info = {}
            inputs = []
            outputs = []
            for n in d.input_nodes:
                if n.op == "call_module":
                    inputs += [str(n.target)]
            for n in d.output_nodes:
                if n.op == "call_module" and str(n.target) != str(n):
                    outputs += [str(n.target)]
            node_info["parents"] = inputs
            node_info["children"] = outputs
            node_info["logical_devices"] = d.logical_device_ids
            nodes[str(d.submodule_node)] = node_info
        nodes_json = json.dumps(nodes, indent=4)
        self.binding.loadNetwork(
            self.serialized_graph_json, self.weights, nodes_json, ["b", "a"], ["add_1"]
        )

    def run_network(self, inputs):
        return self.binding.runNetwork(inputs)
