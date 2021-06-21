from typing import List, Tuple, Union, Dict, Any, Set
from dataclasses import dataclass

import torch
import torch.fx
from torch.fx.node import _get_qualified_name


Tensors = Union[Tuple[torch.Tensor], List[torch.Tensor]]
TensorOrTensors = Union[torch.Tensor, Tensors]
NodeList = List[torch.fx.Node]
NodeSet = Set[torch.fx.Node]
Names = List[str]
CALLABLE_NODE_OPS = {"call_module", "call_function", "call_method"}


def get_node_target(submodules: Dict[str, torch.nn.Module], node: torch.fx.Node) -> str:
    """
    Given a `node` returns its target typename.

    For "call_method" node, return node.target which is the name of that method being called.
    This could potential lead to conflict but should be okay because normally it's on a tensor.

    For "call_function" node, return typename of node.target.

    For "call_module" node, return typename of the module that node.target point to.

    If seeing "_VariableFunctionsClass" in the target name string, it will be replaced by
    "torch". e.g. _VariableFunctionsClass.relu would become torch.relu.
    """

    assert node.op in CALLABLE_NODE_OPS, (
        "Expect op types of " + ", ".join(CALLABLE_NODE_OPS) + f", but found {node.op}"
    )

    if node.op == "call_module":
        assert isinstance(node.target, str)
        return torch.typename(submodules[node.target])
    elif node.op == "call_function":
        target: Any = node.target
        return (
            f"acc_ops.{target.__name__}"
            if target.__module__ == "glow.fb.fx.acc_ops"
            else _get_qualified_name(target)
        )
    else:
        assert isinstance(node.target, str)
        return node.target


class FxNetAccFusionsFinder:
    """
    Finds groups of connected ACC nodes that pass non-tensor data between each other.
    Such groups are called fusion groups.
    """

    def __init__(self, module: torch.fx.GraphModule, acc_nodes: NodeSet):
        self.module = module
        self.nodes = list(module.graph.nodes)
        self.acc_nodes = acc_nodes

    @dataclass
    class FusionGroup:
        # The smallest idx of nodes in the fusion group after topological sorting all the nodes in the model.
        top_node_idx: int

        # Nodes in this fusion group.
        nodes: NodeSet

        # Inputs to this fusion group.
        inputs: NodeSet

        # Nodes that in the fusion group that haven't been processed yet.
        nodes_need_process: NodeSet

        def add_node(self, node):
            """
            Add a node to fusion group.
            """
            if node in self.nodes:
                return

            self.nodes_need_process.add(node)
            self.nodes.add(node)
            self.inputs.discard(node)
            self.inputs.update(
                {
                    n
                    for n in node.all_input_nodes
                    if n.op in CALLABLE_NODE_OPS and n not in self.nodes
                }
            )

    def recursive_add_node(
        self,
        fusion_group: "FxNetAccFusionsFinder.FusionGroup",
        inputs: Union[NodeSet, NodeList],
    ):
        """
        Start from inputs and going reverse topological order. If any upstream node
        is in the fusion group, add all the nodes in this path to fusion group.
        """
        for arg in inputs:
            # Skip placeholder and get_attr because they won't be in the fusion group.
            if arg.op not in CALLABLE_NODE_OPS:
                continue

            # If the node has smaller idx, it's already an upstream node of the fusion
            # group. We don't need to check it anymore.
            if self.nodes.index(arg) < fusion_group.top_node_idx:
                continue

            # If the node is in the fusion group, return True.
            if arg in fusion_group.nodes:
                return True

            # Check the upstream nodes of the node, if any of them is in the fusion group
            # we'll add this node to fusion group and return True.
            if self.recursive_add_node(fusion_group, arg.all_input_nodes):
                fusion_group.add_node(arg)
                return True

        return False

    def __call__(self) -> Dict[torch.fx.Node, NodeSet]:
        result: Dict[torch.fx.Node, NodeSet] = {}
        acc_nodes = list(self.acc_nodes)

        for node in acc_nodes:
            if node in result:
                continue
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if "tensor_meta" in node.meta:
                continue
            if node not in self.acc_nodes:
                continue

            fusion_group: "FxNetAccFusionsFinder.FusionGroup" = self.FusionGroup(
                top_node_idx=self.nodes.index(node),
                nodes={node},
                inputs=set(node.all_input_nodes),
                nodes_need_process={node},
            )
            while fusion_group.nodes_need_process:
                node = fusion_group.nodes_need_process.pop()
                self.recursive_add_node(fusion_group, fusion_group.inputs)

                # Optionally add downstream nodes
                if "tensor_meta" not in node.meta:
                    for user in node.users:
                        if user.op not in CALLABLE_NODE_OPS:
                            continue
                        if user in fusion_group.nodes:
                            continue

                        fusion_group.add_node(user)
                        self.recursive_add_node(fusion_group, fusion_group.inputs)

                # Add some upstream nodes
                for arg in node.all_input_nodes:
                    if arg.op not in CALLABLE_NODE_OPS:
                        continue
                    if "tensor_meta" in arg.meta:
                        continue
                    if arg in fusion_group.nodes:
                        continue

                    fusion_group.add_node(arg)
                    fusion_group.top_node_idx = min(
                        fusion_group.top_node_idx, self.nodes.index(arg)
                    )
                    self.recursive_add_node(fusion_group, fusion_group.inputs)

            if not (set(fusion_group.nodes) <= self.acc_nodes):
                self.acc_nodes -= fusion_group.nodes
            else:
                for n in fusion_group.nodes:
                    result[n] = fusion_group.nodes

        return result
