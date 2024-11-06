from collections import defaultdict
from typing import Dict, List, Set

from sympy.polys.rings import Any

import torch
from torch import fx


def raise_fsdp2_backward_all_gather_ops_if_applicable(
    graph: fx.Graph,
) -> fx.Graph:
    """
    AOTAutograd partitioner does not guarantee that backward hooks are run before the compute ops that depend on them.
    (e.g. in FSDP2 case, backward hooks populate unsharded params, and compute ops read from unsharded params,
    however there is no explicit graph edge dependency from latter to former.) The pathological "use-before-write" scenario
    doesn't come up in unit tests, but does come up in internal models, and it's best to have this guarantee for safety.

    Here, we solve this problem by intentionally moving the FSDP2 backward all-gather ops to be at the top of
    each "FSDP region", so that those unsharded params are guaranteed to be populated before usage.

    We define each "FSDP region" by using resize-to-0 ops as region boundary. Concretely, we have this pattern:
    ```
    resize__0 = torch.ops.inductor.resize_storage_bytes_.default(arg30_1, 0)
    resize__1 = torch.ops.inductor.resize_storage_bytes_.default(arg31_1, 0)
    ...
    (recursive ancestors of as_strided_3 up to graph input)
    resize__2 = torch.ops.inductor.resize_storage_bytes_.default(arg30_1, 64)
    copy__2 = torch.ops.fsdp.copy_.default(arg30_1, as_strided_3)
    (recursive ancestors of as_strided_4 up to graph input)
    resize__3 = torch.ops.inductor.resize_storage_bytes_.default(arg31_1, 128)
    copy__3 = torch.ops.fsdp.copy_.default(arg31_1, as_strided_4)
    ...
    resize__4 = torch.ops.inductor.resize_storage_bytes_.default(arg30_1, 0)
    resize__5 = torch.ops.inductor.resize_storage_bytes_.default(arg31_1, 0)
    ```
    Here, the graph region between two resize-to-0 ops for arg30_1 is an "FSDP region" for arg30_1,
    and we want to move the FSDP2 all-gather related ops that populate arg30_1 to be at the top of the region.
    Note that multiple unsharded params usually bundle together to form an unsharded param group
    (e.g. arg30_1 and arg31_1 in the example above) and they share the same all-gather op.
    So when we move the ops we need to move all the ops related to the unsharded param group together.
    """
    orig_nodes = list(graph.nodes)
    node_to_index = {node: idx for idx, node in enumerate(orig_nodes)}
    node_targets = {node.target for node in orig_nodes if node.op == "call_function"}
    if torch.ops.fsdp.all_gather_copy_in.default not in node_targets:
        # If no FSDP2 all-gather ops, then no-op
        return graph

    # Step 1: Find all resize-to-0 ops on unsharded params
    def _build_unsharded_param_group_to_resize0_groups_mapping() -> Dict[tuple, list]:
        resize0_ops = []
        for node in orig_nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.inductor.resize_storage_bytes_.default
                and node.args[1] == 0
            ):
                resize0_ops.append(node)

        # Create mapping from unsharded param group to list of resize0 groups
        # e.g. (arg30_1, arg31_1) -> [[resize__0, resize__1], [resize__4, resize__5]]
        unsharded_param_group_to_resize0_groups = defaultdict(list)

        # Identify unsharded param group by finding consecutive resize-to-0 ops
        resize0_group: List[fx.Node] = []
        cur_unsharded_params: Set[fx.Node] = set()
        for i, op in enumerate(resize0_ops):
            if i == 0 or node_to_index[op] != node_to_index[resize0_ops[i - 1]] + 1:
                # If the resize-to-0 ops are not consecutive, then store existing group and start a new group
                if cur_unsharded_params:
                    key = tuple(sorted(cur_unsharded_params))
                    unsharded_param_group_to_resize0_groups[key].append(resize0_group)
                resize0_group = []
                cur_unsharded_params = set()
            # Otherwise (i.e. if the resize-to-0 ops are consecutive), add to existing group
            resize0_group.append(op)
            cur_unsharded_params.add(op.args[0])

        # Handle the last unsharded param group
        if cur_unsharded_params:
            key = tuple(sorted(cur_unsharded_params))
            unsharded_param_group_to_resize0_groups[key].append(resize0_group)

        return unsharded_param_group_to_resize0_groups

    unsharded_param_group_to_resize0_groups = (
        _build_unsharded_param_group_to_resize0_groups_mapping()
    )

    # Step 2: In each FSDP region, find the ops that need to be moved up
    nodes_to_insert_after_node: Dict[fx.Node, Set[fx.Node]] = defaultdict(set)
    for (
        cur_unsharded_params,
        resize0_groups,
    ) in unsharded_param_group_to_resize0_groups.items():
        # Iterate through each FSDP region
        for i in range(-1, len(resize0_groups) - 1):
            region_start = None
            if i == -1:
                # First region is defined as last placeholder node of graph to first node of the first resize0 group
                for i, node in enumerate(orig_nodes):
                    if (
                        i + 1 < len(orig_nodes)
                        and orig_nodes[i + 1].op != "placeholder"
                    ):
                        region_start = node
                        break
                region_end = resize0_groups[0][0]
            else:
                # For subsequent regions between two resize0_groups
                current_group = resize0_groups[i]
                next_group = resize0_groups[i + 1]
                region_start = current_group[
                    -1
                ]  # Start of region is the last node of the current resize0 group
                region_end = next_group[
                    0
                ]  # End of region is the first node of next resize0 group
            assert region_start is not None

            # Collect nodes within this FSDP region
            nodes_between = []
            in_region = False
            for node in orig_nodes:
                if node == region_start:
                    in_region = True
                elif node == region_end:
                    in_region = False
                    break
                elif in_region:
                    nodes_between.append(node)

            # Find nodes to move (fsdp.copy_ ops and their ancestors, and resize-to-full op for each fsdp.copy_ op)
            nodes_to_move = set()
            for node in nodes_between:
                if (
                    node.op == "call_function"
                    and node.target == torch.ops.fsdp.copy_.default
                    and node.args[0] in cur_unsharded_params
                ):
                    # Find and add the resize-to-full op
                    for n in nodes_between:
                        if (
                            n.op == "call_function"
                            and n.target
                            == torch.ops.inductor.resize_storage_bytes_.default
                            and n.args[0] == node.args[0]
                            and n.args[1] > 0
                        ):
                            nodes_to_move.add(n)
                            break

                    def _collect_ancestors(n: fx.Node) -> None:
                        if n.op == "placeholder":
                            return
                        nodes_to_move.add(n)

                        def process_arg(arg: Any) -> None:
                            if isinstance(arg, fx.Node):
                                _collect_ancestors(arg)
                            elif isinstance(arg, list):
                                for item in arg:
                                    process_arg(item)

                        for arg in n.args:
                            process_arg(arg)
                        for kwarg in n.kwargs.values():
                            process_arg(kwarg)

                    _collect_ancestors(node.args[1])
                    nodes_to_move.add(node)

            nodes_to_insert_after_node[region_start] |= nodes_to_move

    # Step 3: Rebuild the graph, inserting the raised nodes at the right place
    new_graph = fx.Graph()
    env: Dict[fx.Node, fx.Node] = {}
    nodes_processed = set()

    def insert_node(node: fx.Node) -> None:
        if node not in nodes_processed:
            env[node] = new_graph.node_copy(node, lambda x: env[x])
            nodes_processed.add(node)

    for node in orig_nodes:
        insert_node(node)
        # Sort nodes by index in original graph to ensure ordering correctness
        sorted_succ_nodes = sorted(
            nodes_to_insert_after_node[node], key=lambda n: node_to_index[n]
        )
        for succ_node in sorted_succ_nodes:
            insert_node(succ_node)

    return new_graph
