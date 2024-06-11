# mypy: allow-untyped-defs
from typing import List, Tuple, Union, Dict, Any, Set, Mapping, Optional
import collections
from dataclasses import dataclass
import operator

import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility

__all__ = ['get_acc_ops_name', 'get_node_target', 'is_node_output_tensor', 'FxNetAccFusionsFinder', 'legalize_graph']

Tensors = Union[Tuple[torch.Tensor], List[torch.Tensor]]
TensorOrTensors = Union[torch.Tensor, Tensors]
NodeList = List[torch.fx.Node]
NodeSet = Set[torch.fx.Node]
Names = List[str]
CALLABLE_NODE_OPS = {"call_module", "call_function", "call_method"}


@compatibility(is_backward_compatible=False)
def get_acc_ops_name(k):
    if isinstance(k, str):
        return k
    elif k.__module__ and "acc_ops" in k.__module__:
        return f"acc_ops.{k.__name__}"
    else:
        module = k.__module__.replace('torch._ops', 'torch.ops')  # WAR for bug in how torch.ops assigns module
        return f"{module if module else ''}.{k.__name__}"


@compatibility(is_backward_compatible=False)
def get_node_target(submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> str:
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
        submod = submodules[node.target]
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        return get_acc_ops_name(submod_type)
    elif node.op == "call_function":
        target: Any = node.target
        return (
            f"acc_ops.{target.__name__}"
            if target.__module__ is not None and "acc_ops" in target.__module__
            else _get_qualified_name(target)
        )
    else:
        assert isinstance(node.target, str)
        return node.target

@compatibility(is_backward_compatible=False)
def is_node_output_tensor(node: torch.fx.Node) -> bool:
    """Checks if the node output produces a Tensor or not.

    NOTE: This requires to run `ShapeProp` on the containing fx graph before
    calling this function. This is because it works by checking the `type`
    metadata on the node. This metadata is produced by the `ShapeProp`.
    """
    type_ = node.meta.get("type", None)
    return type_ is not None and issubclass(type_, torch.Tensor)

@compatibility(is_backward_compatible=False)
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
        visited: Optional[NodeSet] = None,
    ):
        """
        Start from inputs and going reverse topological order. If any upstream node
        is in the fusion group, add all the nodes in this path to fusion group.
        """
        for arg in inputs:
            # skip the node if already seen
            if visited is not None:
                if arg in visited:
                    continue
                visited.add(arg)

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
            if self.recursive_add_node(fusion_group, arg.all_input_nodes, visited):
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

            fusion_group: FxNetAccFusionsFinder.FusionGroup = self.FusionGroup(
                top_node_idx=self.nodes.index(node),
                nodes={node},
                inputs=set(node.all_input_nodes),
                nodes_need_process={node},
            )
            while fusion_group.nodes_need_process:
                node = fusion_group.nodes_need_process.pop()
                self.recursive_add_node(
                    fusion_group,
                    fusion_group.inputs,
                    visited=set(),
                )

                # Optionally add downstream nodes
                if "tensor_meta" not in node.meta:
                    for user in node.users:
                        if user.op not in CALLABLE_NODE_OPS:
                            continue
                        if user in fusion_group.nodes:
                            continue

                        fusion_group.add_node(user)
                        self.recursive_add_node(
                            fusion_group,
                            fusion_group.inputs,
                            visited=set(),
                        )

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
                    self.recursive_add_node(
                        fusion_group,
                        fusion_group.inputs,
                        visited=set(),
                    )

            if not (set(fusion_group.nodes) <= self.acc_nodes):
                self.acc_nodes -= fusion_group.nodes
            else:
                for n in fusion_group.nodes:
                    result[n] = fusion_group.nodes

        return result


@compatibility(is_backward_compatible=False)
def legalize_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Replace the graph of the given GraphModule with one that contains the same nodes as the
    original, but in topologically sorted order.

    This is used by the merge_matmul transformation below, which disturbs the topologically sorted
    order of its input GraphModule, so that this order is restored before further transformation.

    Arguments:
        gm: The graph module to topologically sort. It is modified in-place.

    Returns:
        The graph module in-place sorted
    """

    # These operators are used for making runtime assertions before any
    # data-dependent operators occur. We want to prioritize sorting these to
    # ensure that these assertions appear before any data-dependent operations
    # in the graph.
    PRIORITIZED_OPS = [
        operator.add,
        operator.mul,
        operator.sub,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.le,
        operator.lt,
        operator.ge,
        operator.gt,
        operator.eq,
        operator.ne,
        torch.ops.aten.sym_constrain_range.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten._assert_scalar.default,
    ]

    indeg = dict.fromkeys(gm.graph.nodes, 0)
    new_graph = torch.fx.Graph()
    # Track how many unfulfilled dependencies each node has
    for node in gm.graph.nodes:
        for user in node.users:
            indeg[user] += 1
    queue: collections.deque = collections.deque()
    # Add all nodes with no dependencies to the queue
    for node in gm.graph.nodes:
        if indeg[node] == 0:
            queue.append(node)
    env: Dict[torch.fx.Node, torch.fx.Node] = {}
    # Pop nodes from the queue, and add nodes that have had all their
    # dependencies fulfilled
    while len(queue) > 0:
        cur = queue.popleft()
        env[cur] = new_graph.node_copy(cur, lambda x: env[x])
        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                if user.op == "call_function" and user.target in PRIORITIZED_OPS:
                    queue.appendleft(user)
                else:
                    queue.append(user)
    # If the new graph's size is not as large as the old one, then there must be
    # a cycle (i.e. some node's dependencies were not satisfied.)
    if len(new_graph.nodes) < len(gm.graph.nodes):
        raise RuntimeError(f"Input graph has cycles, unable to add {[node for node in indeg if indeg[node] != 0]}")
    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph
    return gm
