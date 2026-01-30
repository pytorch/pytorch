# mypy: allow-untyped-defs
import copy
from dataclasses import dataclass, field
from typing import Optional, Union

import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import map_arg
from torch.fx.passes.utils import HolderModule, lift_subgraph_as_module

from .tools_common import CALLABLE_NODE_OPS, is_node_output_tensor, NodeList


__all__ = [
    "getattr_recursive",
    "setattr_recursive",
    "Component",
    "split_by_tags",
    "move_non_tensor_nodes_on_boundary",
]


@compatibility(is_backward_compatible=False)
def getattr_recursive(obj, name):
    for layer in name.split("."):
        if isinstance(obj, torch.nn.ModuleList):
            if hasattr(obj, "_modules") and layer in obj._modules:
                obj = obj._modules[layer]
            else:
                return None
        elif hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj


@compatibility(is_backward_compatible=False)
def setattr_recursive(obj, attr, value):
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split(".")
        setattr_recursive(getattr(obj, layer[0]), ".".join(layer[1:]), value)


@compatibility(is_backward_compatible=False)
@dataclass
class Component:
    """
    A component serves as a container for a subgraph we want to create afterwards.
    """

    graph: torch.fx.Graph
    order: int
    name: str

    # Stores the placeholder nodes in `graph`.
    input_placeholders: list = field(default_factory=list)

    # Store the nodes in original graph that are placeholder in `graph`.
    orig_inputs: list = field(default_factory=list)

    # Store the nodes in original graph that are outputs in `graph`.
    orig_outputs: list = field(default_factory=list)

    # Mapping from get_attr node in original graph to get_attr node in `graph`.
    getattr_maps: dict[torch.fx.Node, torch.fx.Node] = field(default_factory=dict)
    constructor_args: list[str] = field(default_factory=list)
    gm: Optional[torch.fx.GraphModule] = None


@compatibility(is_backward_compatible=False)
def split_by_tags(
    gm: torch.fx.GraphModule,
    tags: list[str],
    return_fqn_mapping: bool = False,
    return_tuple: bool = False,
    GraphModuleCls: type[torch.fx.GraphModule] = torch.fx.GraphModule,
) -> Union[torch.fx.GraphModule, tuple[torch.fx.GraphModule, dict[str, str]]]:
    """
    Splits a GraphModule using tags on its graph nodes. We honor the order of
    tags. For example, we have tags = ["a", "b", "c"], the function will create
    the initial submodules in the order of "a", "b", "c".

    To set a tag:
    gm.graph.nodes[idx].tag = "mytag"

    This will result in all nodes with the same tag being extracted and placed in their
    own submodule. For placeholder, output and get_attr node, the tag is ignored. placeholder
    and output nodes are created when needed while get_attr nodes get copied to submodules
    where they are used.

    Given the following module def:

    class SimpleModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = torch.nn.Linear(...)
            self.linear2 = torch.nn.Linear(...)
            self.linear3 = torch.nn.Linear(...)

        def forward(self, in1, in2):
            r1 = self.linear1(in1)
            r2 = self.linear2(in2)
            r3 = torch.cat([r1, r2])
            return self.linear3(r3)

    Marking the node corresponding to in1 with the tag sc.REQUEST_ONLY.lower() results in the following split:

    ro:
    def forward(self, in1):
        self = self.root
        linear1 = self.linear1(in1)
        return linear1

    main:
    def forward(self, in2, linear1):
        self = self.root
        linear2 = self.linear2(in2)
        cat_1 = torch.cat([linear1, linear2])
        linear3 = self.linear3(cat_1)
        return linear3

    main:
    def forward(self, in1, in2):
        self = self.root
        ro_0 = self.ro_0(in1)
        main_1 = self.main_1(in2, ro_0)
        return main_1

    Returns:
        split_gm: torch fx graph after split
        orig_to_split_fqn_mapping: a map between the original fqn and the fqn
            after split for call_module and get_attr.
    """

    def flatten(x: torch.fx.node.Argument) -> NodeList:
        """
        Stores nodes in x to a list and returns the list.
        """
        r: NodeList = []
        map_arg(x, r.append)
        return r

    # Mapping from node in original module to node in created submodule.
    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}

    # Mapping from node in original module or created submodules to
    # corresponding component.
    node_to_component: dict[torch.fx.Node, Component] = {}

    # Mapping from tag to the corresponding component.
    tag_to_component: dict[str, Component] = {}

    # Stores all components.
    all_components: list[Component] = []

    # Stores nodes that will be used in main graph.
    used_in_main: dict[torch.fx.Node, None] = {}

    # Main graph after split.
    main_g = torch.fx.Graph()

    # Mapping from node in original module to node in main graph after split.
    main_remapping: dict[torch.fx.Node, torch.fx.Node] = {}

    # Output node of original module.
    output_node: Optional[torch.fx.Node] = None

    # Create a component for each tag, we don't expect to create other components afterwards.
    for tag in tags:
        comp = Component(torch.fx.Graph(), len(all_components), f"{tag}")
        all_components.append(comp)
        tag_to_component[tag] = comp

    # Traverse the nodes in original graph and take care of them.
    for node in gm.graph.nodes:
        if node.op == "output":
            if output_node is not None:
                raise RuntimeError("Multiple output nodes in graph!")
            output_node = node
            continue

        # Placeholders in the original graph get copied to main graph.
        if node.op == "placeholder":
            main_remapping[node] = main_g.placeholder(node.name, type_expr=node.type)
            main_remapping[node].meta = copy.copy(node.meta)
            continue

        # Get_attr nodes are ignored because we are not tagging them.
        # Instead, we copy them directly to the submodules use them afterwards.
        if node.op == "get_attr":
            continue

        # Now we process callable nodes which are nodes with op of call_module,
        # call_function or call_method. Every callable nodes should be tagged.
        assert hasattr(node, "tag"), f"Node does not have tag: {node.format_node()}"

        upstream_components = [
            node_to_component[x]
            for x in flatten(node.args) + flatten(node.kwargs)
            if x.op not in {"placeholder", "get_attr"}
        ]

        comp = tag_to_component[node.tag]
        node_to_component[node] = comp

        # Max order of upperstream components.
        mx = max((c.order for c in upstream_components), default=0)

        # Expect the component for `node` has higher order then its upstream components.
        assert comp.order >= mx, (
            f"Component {comp.name} order must be >= max of its upstream components, order={comp.order} and max={mx}"
        )

        # Map a input of `node` to nodes in the component's graph.
        def remap_func(x):
            # If input is a get_attr node, copy it to current component's graph.
            # Returns the get_attr node in current component's graph.
            if x.op == "get_attr":
                if x not in comp.getattr_maps:
                    comp.getattr_maps[x] = comp.graph.get_attr(
                        x.target, type_expr=x.type
                    )
                    comp.getattr_maps[x].meta = copy.copy(x.meta)
                return comp.getattr_maps[x]

            # If input is not a placeholder, it should have been put into a component
            # already. If it's the current component then we return the corresponding
            # node in the component.
            if x.op != "placeholder" and node_to_component[x] == comp:
                return node_remapping[x]

            # If input is a placeholder or it's in other components, we want to make it
            # as a placeholder in current component's graph.
            if x not in comp.orig_inputs:
                comp.orig_inputs.append(x)
                placeholder = comp.graph.placeholder(x.name, type_expr=x.type)
                placeholder.meta = copy.copy(x.meta)
                comp.input_placeholders.append(placeholder)
                used_in_main[x] = None

            return comp.input_placeholders[comp.orig_inputs.index(x)]

        n = comp.graph.node_copy(node, remap_func)
        n.tag = node.tag  # type: ignore[attr-defined]
        node_remapping[node] = n
        node_to_component[n] = comp

    if output_node is None:
        raise RuntimeError("Graph had no output node!")

    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            # We don't need components mapping for nodes of type "get_attr"
            # that are consumed by the output. Only need to make sure we create
            # corresponding counterparts in the resulting graph.
            main_remapping[x] = main_g.get_attr(x.name, type_expr=x.type)
        else:
            # All component results consumed by the output node should be
            # marked as "used in main".
            used_in_main[x] = None

    # If a node is used in main graph then we mark it as an output in the component
    # it belongs to.
    for n in used_in_main:
        if n.op != "placeholder":
            node_to_component[n].orig_outputs.append(n)

    # Now we create a graphmodule for each component.
    orig_to_split_fqn_mapping: dict[str, str] = {}
    for comp in all_components:
        outs = tuple(map(node_remapping.__getitem__, comp.orig_outputs))

        if return_tuple:
            comp.graph.output(outs)
        else:
            # Take care of the args of FX output node. If there's a single
            # output then the output node args is like (output_single), else
            # if there're multiple outputs then the output node args is like
            # ((output_0, output_1, ...)).
            comp.graph.output(outs[0] if len(outs) == 1 else outs)

        comp.gm, comp_orig_to_split_fqn_mapping = lift_subgraph_as_module(
            gm, subgraph=comp.graph, comp_name=comp.name
        )
        orig_to_split_fqn_mapping.update(comp_orig_to_split_fqn_mapping)

        # Create a call_module node in main graph.
        main_node = main_g.call_module(
            comp.name,
            args=tuple(map(main_remapping.__getitem__, comp.orig_inputs)),
            kwargs=None,
        )

        if len(outs) == 1 and not return_tuple:
            main_remapping[comp.orig_outputs[0]] = main_node
        else:
            for i, o in enumerate(comp.orig_outputs):
                # Use Proxy to record getitem access.
                main_remapping[o] = torch.fx.Proxy(main_node)[i].node  # type: ignore[index]

    main_g.output(map_arg(output_node.args[0], main_remapping.__getitem__))
    main_root = HolderModule({comp.name: comp.gm for comp in all_components})
    main_g._codegen = gm.graph._codegen

    # If the output nodes consumes get_attr directly in the original graph,
    # then we need to make sure get_attr is copied to the new graph.
    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            setattr(main_root, x.name, getattr_recursive(gm, x.target))  # type: ignore[arg-type]

    result_gm = GraphModuleCls(main_root, main_g)
    if return_fqn_mapping:
        return result_gm, orig_to_split_fqn_mapping

    return result_gm


@compatibility(is_backward_compatible=False)
def move_non_tensor_nodes_on_boundary(subgraphs) -> None:
    """
    Move non-tensor nodes on the boundary between subgraphs.

    For each subgraph:

    1. Find nodes whose type is not tensor and any of its children is in another
       subgraph, put them in a queue for next step

    2. Do a BFS on those nodes in the queue,  and run a DFS for each node, let's say node X and it is in subgraph A:

       a. if it is in to_subgraph, return (continue DFS)
       b. if it is in from_subgraph, collect the nodes to nodes_to_move, and continue DFS
       c. otherwise, this means it cannot be moved
       d. also check if node X's parent should be put into the queue. (The queue may
          have duplicated nodes, just process the node once)

    Args:
        subgraphs: List of subgraphs containing nodes to be processed
    """
    # Create a mapping from node to subgraph for quick lookup
    node_to_subgraph: dict[torch.fx.Node, int] = {}
    for i, subgraph in enumerate(subgraphs):
        for node in subgraph.nodes:
            node_to_subgraph[node] = i

    def get_children_in_graph(node: torch.fx.Node) -> list[torch.fx.Node]:
        """Get children nodes that are in callable ops and in some subgraph"""
        return [
            user
            for user in node.users
            if user.op in CALLABLE_NODE_OPS and user in node_to_subgraph
        ]

    def get_parents_in_graph(node: torch.fx.Node) -> list[torch.fx.Node]:
        """Get parent nodes that are in callable ops and in some subgraph"""
        return [
            arg
            for arg in node.all_input_nodes
            if arg.op in CALLABLE_NODE_OPS and arg in node_to_subgraph
        ]

    def has_children_in_other_subgraph(
        node: torch.fx.Node, current_subgraph_idx: int
    ) -> bool:
        """
        Check if the node has any children in a subgraph different from current_subgraph_idx.
        This is the requirement used in both step 1 and step d.
        """
        children = get_children_in_graph(node)
        return any(
            node_to_subgraph[child] != current_subgraph_idx for child in children
        )

    def can_move_node_and_dependencies(
        node: torch.fx.Node, from_subgraph: int, to_subgraph: int
    ) -> tuple[bool, set[torch.fx.Node]]:
        """
        Check if node and its dependencies can be moved from from_subgraph to to_subgraph.
        Returns (can_move, nodes_to_move)

        For node X, do a DFS on its descendants, for each node:
        - if it is in to_subgraph, return (continue DFS)
        - if it is in from_subgraph, collect the nodes to nodes_to_move, and continue DFS
        - otherwise, this means it cannot be moved
        """
        nodes_to_move = set()
        visited = set()
        can_move = True

        def dfs(current_node):
            nonlocal can_move, nodes_to_move

            if current_node in visited:
                return
            visited.add(current_node)

            # Check current node's subgraph
            if current_node not in node_to_subgraph:
                return  # Skip nodes not in any subgraph

            current_subgraph = node_to_subgraph[current_node]

            if current_subgraph == to_subgraph:
                # If it is in to_subgraph, just end DFS
                return
            elif current_subgraph == from_subgraph:
                # If it is in from_subgraph, collect it and continue DFS
                nodes_to_move.add(current_node)
            else:
                # Otherwise, this means it cannot be moved
                can_move = False
                return

            # Continue DFS on children
            children = get_children_in_graph(current_node)
            for child in children:
                if can_move:  # Only continue if we haven't already failed
                    dfs(child)

        # Start DFS from the original node
        dfs(node)

        return can_move, nodes_to_move

    # For each subgraph, find non-tensor nodes with children in other subgraphs
    for subgraph_idx, subgraph in enumerate(subgraphs):
        # non acc nodes cannot be moved to downstream acc graph, so skip
        if not subgraph.is_acc:
            continue
        # Step 1: Find non-tensor nodes with children in other subgraphs
        queue: list[torch.fx.Node] = []
        processed: set[torch.fx.Node] = set()

        for node in subgraph.nodes:
            # Check if node is non-tensor
            if is_node_output_tensor(node):
                continue

            # Check if node meets step 1 requirement: any children in another subgraph
            if has_children_in_other_subgraph(node, subgraph_idx):
                queue.append(node)

        # Step 2: BFS to move nodes that meet the criteria
        while queue:
            current_node = queue.pop(0)

            # Skip if already processed (queue may have duplicates)
            if current_node in processed:
                continue
            processed.add(current_node)

            # Skip if node is no longer in this subgraph (may have been moved)
            if (
                current_node not in node_to_subgraph
                or node_to_subgraph[current_node] != subgraph_idx
            ):
                continue

            children = get_children_in_graph(current_node)
            assert len(children) > 0, (
                "Only node that has children in other subgraph can be moved"
            )

            # Find target subgraph. The children should all be in the same subgraph except current subgraph
            target_subgraph_candidates = set()
            for child in children:
                child_subgraph = node_to_subgraph[child]
                if child_subgraph != subgraph_idx:
                    target_subgraph_candidates.add(child_subgraph)
            # If multiple children live in different subgraphs, the node cannot be moved. User needs to find other ways to move it.
            if len(target_subgraph_candidates) != 1:
                print(
                    f"Cannot move non-tensor node {current_node.name} on boundary because it has children in multiple subgraphs"
                )
                continue

            target_subgraph = target_subgraph_candidates.pop()

            # Check if we can move this node and its dependencies
            can_move, nodes_to_move = can_move_node_and_dependencies(
                current_node, subgraph_idx, target_subgraph
            )

            if can_move:
                # Move all nodes in nodes_to_move to target subgraph
                for node_to_move in nodes_to_move:
                    # Remove from current subgraph
                    subgraph.nodes.remove(node_to_move)
                    # Add to target subgraph
                    subgraphs[target_subgraph].nodes.append(node_to_move)
                    # Update mapping
                    node_to_subgraph[node_to_move] = target_subgraph
                    print(
                        f"In order move the non-tensor node {current_node.name} on boundary, "
                        f"moved node {node_to_move.name} from {'acc' if subgraph.is_acc else 'gpu'}_{subgraph_idx} "
                        f"to {'acc' if subgraphs[target_subgraph].is_acc else 'gpu'}_{target_subgraph}"
                    )

                # Add parents to the queue if they're non-tensor and not already processed
                # and meet the requirement from step 1 (any children in another subgraph)
                parents = get_parents_in_graph(current_node)
                for parent in parents:
                    if (
                        not is_node_output_tensor(parent)
                        and parent not in processed
                        and parent in node_to_subgraph
                        and node_to_subgraph[parent] == subgraph_idx
                    ):
                        # Check if parent meets step 1 requirement: any children in another subgraph
                        assert has_children_in_other_subgraph(parent, subgraph_idx), (
                            f"Parent {parent.name} should have children in another subgraph"
                        )
                        queue.append(parent)
