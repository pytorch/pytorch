import torch
from typing import Dict

def raise_fsdp2_backward_all_gather_ops(graph):
    """
    AOTAutograd partitioner does not guarantee that pre-backward hooks are run before compute ops that depend on it.
    (e.g. in FSDP2 case, pre-backward hooks populate unsharded params, and compute ops read from unsharded params -> there is no explicit graph edge dependency from latter to former)
    Here, we solve this problem by intentionally raising the pre-backward all-gather ops up in the graph to be before the first use op of any of the unsharded params that it writes to.
    This is done per resize0 region (i.e. use resize-to-0 op to define region boundary), to allow for "layer-reuse" scenario as well.
    Concretely, we have this pattern:
    ```
    torch.ops.inductor.resize_storage_bytes_.default(arg30_1, 0)
    torch.ops.inductor.resize_storage_bytes_.default(arg31_1, 0)
    ...
    (recursive ancestors of as_strided_3 up to graph input (root inputs should be purely graph inputs))
    resize_storage_bytes__2 = torch.ops.inductor.resize_storage_bytes_.default(arg30_1, 64)
    copy__2 = torch.ops.fsdp.copy_.default(arg30_1, as_strided_3)
    (recursive ancestors of as_strided_4 up to graph input (root inputs should be purely graph inputs))
    resize_storage_bytes__3 = torch.ops.inductor.resize_storage_bytes_.default(arg31_1, 128)
    copy__3 = torch.ops.fsdp.copy_.default(arg31_1, as_strided_4)
    ...
    torch.ops.inductor.resize_storage_bytes_.default(arg30_1, 0)
    torch.ops.inductor.resize_storage_bytes_.default(arg31_1, 0)
    ```
    and we want to raise all the FSDP2 all-gather related ops (the ops in the middle of the example) to be immediately after the last resize-to-0 op in the previous resize0 op group.
    """
    orig_nodes = list(graph.nodes)
    node_to_index = {node: idx for idx, node in enumerate(orig_nodes)}
    
    # Step 1: Find all resize-to-0 ops on graph inputs
    resize0_ops = []
    for node in orig_nodes:
        if (node.op == "call_function" and 
            node.target == torch.ops.inductor.resize_storage_bytes_.default and
            node.args[1] == 0):
            resize0_ops.append(node)

    # Create mapping from graph input group string to list of resize0 groups
    graph_input_group_to_resize0_groups = defaultdict(list)

    # Group consecutive resize0 ops by checking if their indices are adjacent
    resize0_group = []
    cur_graph_inputs = set()
    for i, op in enumerate(resize0_ops):
        if i == 0 or node_to_index[op] != node_to_index[resize0_ops[i-1]] + 1:
            if cur_graph_inputs:
                # Create key for previous group using sorted tuple
                key = tuple(sorted(cur_graph_inputs))
                graph_input_group_to_resize0_groups[key].append(resize0_group)
            resize0_group = []
            cur_graph_inputs = set()
        resize0_group.append(op)
        cur_graph_inputs.add(op.args[0])

    # Handle the last group
    if cur_graph_inputs:
        key = tuple(sorted(cur_graph_inputs))
        graph_input_group_to_resize0_groups[key].append(resize0_group)

    nodes_to_insert_after_node = defaultdict(set)

    # Process each graph input group
    for cur_graph_inputs, resize0_groups in graph_input_group_to_resize0_groups.items():
        # Process each region, including the first region from graph start to first resize0_group
        for i in range(-1, len(resize0_groups) - 1):
            region_start = None
            if i == -1:
                # For first region (last placeholder node to first node of the first resize0 group)
                for i, node in enumerate(orig_nodes):
                    if i + 1 < len(orig_nodes) and orig_nodes[i + 1].op != "placeholder":
                        region_start = node
                        break
                region_end = resize0_groups[0][0]
            else:
                # For subsequent regions between resize0_groups
                current_group = resize0_groups[i]
                next_group = resize0_groups[i + 1]
                region_start = current_group[-1]  # Start of region is the last node of the current resize0 group
                region_end = next_group[0]  # End of region is the first node of next resize0 group
            assert region_start is not None

            # Collect nodes between region boundaries
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

            # Find nodes to move (fsdp.copy_ ops and their ancestors)
            nodes_to_move = set()
            for node in nodes_between:
                if (node.op == "call_function" and 
                    node.target == torch.ops.fsdp.copy_.default and
                    node.args[0] in cur_graph_inputs):

                    # Find and add the resize-to-full op
                    for n in nodes_between:
                        if (n.op == "call_function" and 
                            n.target == torch.ops.inductor.resize_storage_bytes_.default and
                            n.args[0] == node.args[0] and 
                            n.args[1] > 0):
                            nodes_to_move.add(n)
                            break

                    def collect_ancestors(n):
                        if n.op == "placeholder":
                            return
                        nodes_to_move.add(n)
                        def process_arg(arg):
                            if isinstance(arg, torch.fx.Node):
                                collect_ancestors(arg)
                            elif isinstance(arg, list):
                                for item in arg:
                                    process_arg(item)

                        for arg in n.args:
                            process_arg(arg)
                        for kwarg in n.kwargs.values():
                            process_arg(kwarg)

                    collect_ancestors(node.args[1])
                    nodes_to_move.add(node)

            nodes_to_insert_after_node[region_start] |= nodes_to_move

    new_graph = torch.fx.Graph()
    env: Dict[torch.fx.Node, torch.fx.Node] = {}
    nodes_processed = set()

    def insert_node(node):
        if node not in nodes_processed:
            env[node] = new_graph.node_copy(node, lambda x: env[x])
            nodes_processed.add(node)

    for node in orig_nodes:
        insert_node(node)
        sorted_succ_nodes = sorted(nodes_to_insert_after_node[node], key=lambda n: node_to_index[n])
        for succ_node in sorted_succ_nodes:
            insert_node(succ_node)

    return new_graph
