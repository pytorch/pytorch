"""
This module implements graph deduplication functionality for TorchDynamo's optimization pipeline.
Graph deduplication identifies identical subgraphs in the computational graph and merges them
to reduce redundancy and improve performance. The process involves analyzing regions of the graph,
identifying structurally equivalent regions, and replacing them with a single shared implementation.
This optimization is particularly effective for models with repeated patterns or similar computational
structures across different parts of the network.
"""

import logging
import operator
from collections import defaultdict, deque
from collections.abc import Generator, Iterable
from typing import Optional

import torch
import torch.fx
from torch._dynamo import config
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from .graph_region_tracker import Node, Region
from .graph_utils import _detect_cycles, _get_flat_args, _get_flat_args_unique


# Represents an index into the region
# to select a node and then
# an index into that node's
# flattened arguments
UsageIndex = tuple[int, int]

log = logging.getLogger(__name__)

last_node_to_additional_deps: Optional[dict[Node, OrderedSet[Node]]] = None


def apply_graph_deduplication(output_graph) -> dict[str, torch.fx.GraphModule]:  # type: ignore[no-untyped-def]
    """
    This is the main entry point for applying the graph deduplication pass. \
Deduplication occurs in two phases:
    1. Subgraph creation:
        Subgraph creation works by taking one representative region from each region \
group and creating a subgraph from it, which will then be used to replace all regions \
in the group. This is implemented by first copying all nodes of the region to the new \
subgraph and then finding all inputs which are not within the region and creating placeholders \
for them. For the outputs, all regions in a region group need to be scanned to ensure the \
largest set of outputs is found, and then an output node is created which returns \
a tuple of all outputs.

    2. Graph replacement:
        To replace each region with the extracted subgraph, the node index in the region \
and argument index within the node's flattened args and kwargs are recorded once during \
subgraph creation. This allows us to determine which (external to the region) nodes and \
in which order these nodes are passed as inputs. For the outputs, getitem nodes are created \
for each output, and all nodes in the region with external outputs are replaced by the proper \
getitem node. Finally, all original nodes are erased (there should be no uses of these \
left in the graph).

The deduplication mutates the output_graph argument in place.

Returns a mapping of nodes to their subgraph output replacement node to remap outputs
when they are created in output_graph.
    """

    duplicated_region_groups = output_graph.region_tracker.get_identical_regions(
        output_graph.graph
    )
    node_to_mutated_arg_positions = (
        output_graph.region_tracker.node_to_mutated_arg_positions
    )
    node_to_additional_deps = _populate_additional_deps(
        output_graph.graph, output_graph.region_tracker.node_to_mutated_arg_positions
    )

    sub_gms: dict[str, torch.fx.GraphModule] = {}

    for region_group in duplicated_region_groups:
        inds_with_external_users = _get_all_output_indices(region_group)
        region = region_group[0]
        (
            subgraph,
            external_node_usages,
            node_usage_to_tuple_elems,
            ind_to_tuple_spec,
        ) = _create_subgraph(region, inds_with_external_users)

        # Ignore regions with no args for now, could they possibly be evaluated at compile time?
        if not list(external_node_usages):
            continue

        sub_gm = torch.fx.GraphModule(output_graph.nn_modules, subgraph)
        subgraph_name = output_graph.install_subgraph("subgraph", sub_gm)
        sub_gms[subgraph_name] = sub_gm
        with output_graph.graph.inserting_before():
            get_subgraph_node = output_graph.graph.create_node(
                "get_attr", subgraph_name, (), {}
            )

        for region in region_group:
            _replace_region_with_subgraph(
                output_graph.graph,
                region,
                get_subgraph_node,
                external_node_usages,
                node_usage_to_tuple_elems,
                ind_to_tuple_spec,
                inds_with_external_users,
                subgraph_name,
                node_to_additional_deps,
                node_to_mutated_arg_positions,
            )

    # This is to expose the updated node_to_additional_deps to tests
    global last_node_to_additional_deps
    last_node_to_additional_deps = node_to_additional_deps

    _stable_topological_sort(
        output_graph.graph,
        node_to_additional_deps,
    )
    return sub_gms


def _replace_region_with_subgraph(
    graph: torch.fx.Graph,
    region: Region,
    get_subgraph_node: Node,
    external_node_usages: Iterable[OrderedSet[UsageIndex]],
    node_usage_to_tuple_elems: dict[UsageIndex, OrderedSet[int]],
    ind_to_tuple_spec: dict[int, dict[tuple[int, ...], int]],
    inds_with_external_users: list[int],
    subgraph_name: str,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
    node_to_mutated_arg_positions: dict[Node, OrderedSet[int]],
) -> None:
    sub_args = []
    flattened_getitem_nodes: OrderedSet[Node] = OrderedSet()
    for usages in external_node_usages:
        usage = next(iter(usages))
        node_ind, usage_ind = usage
        node = region[node_ind]
        flattened_args_kwargs = _get_flat_args(node, {})
        for user_ind, node_usage_ind in usages:
            user = region[user_ind]
            if user in node_to_mutated_arg_positions:
                if node_usage_ind in node_to_mutated_arg_positions[user]:
                    log.debug(
                        "NYI: Failed to substitute region %s due to mutation", region
                    )
                    return
        if usage in node_usage_to_tuple_elems:
            tuple_elems = [region[i] for i in node_usage_to_tuple_elems[usage]]
            flattened_getitem_nodes.update(tuple_elems)
            sub_args.extend(tuple_elems)
        else:
            sub_args.append(flattened_args_kwargs[usage_ind])

    # Input/Output aliasing not supported in HOPs today
    # Note: we should use the nodes in the original graph (the region here)
    # because we use the original traced example values for this check
    if _has_aliasing(
        region, sub_args, inds_with_external_users, flattened_getitem_nodes
    ):
        return

    invoke_args = (get_subgraph_node, subgraph_name, *sub_args)

    invoke_subgraph_node = graph.create_node(
        "call_function",
        torch.ops.higher_order.invoke_subgraph,
        invoke_args,  # type: ignore[arg-type]
        {},
    )

    ind = 0
    flattened_output_nodes: OrderedSet[Node] = OrderedSet()
    for external_user_ind in inds_with_external_users:
        node = region[external_user_ind]
        if _is_tuple_node(node):
            tuple_spec = ind_to_tuple_spec[external_user_ind]
            flattened_output_nodes.update(
                _replace_tuple_outputs(
                    node, ind, tuple_spec, invoke_subgraph_node, graph
                )
            )
            ind += len(tuple_spec)
        else:
            subgraph_output = graph.create_node(
                "call_function", operator.getitem, (invoke_subgraph_node, ind), {}
            )
            node.replace_all_uses_with(subgraph_output, propagate_meta=True)
            ind += 1

    # Erase in reverse topological order
    for node in reversed(region):
        if node in flattened_getitem_nodes:
            # Don't erase these, since they will still be used
            continue

        if node not in flattened_output_nodes:
            graph.erase_node(node)

        # Remove any nodes with additional deps
        # This is safe; we've guaranteed that there is
        # no input mutation, so all additional deps
        # will be internal to the subgraph
        node_to_additional_deps.pop(node, None)
        for deps in node_to_additional_deps.values():
            try:
                deps.remove(node)
                deps.add(invoke_subgraph_node)
            except KeyError:
                pass

    if config.graph_deduplication_lint:
        print(_detect_cycles(graph, node_to_additional_deps))
        _stable_topological_sort(graph, node_to_additional_deps)
        graph.lint()


def _get_external_inputs(
    region: Region,
) -> dict[Node, OrderedSet[UsageIndex]]:
    external_node_to_usages = defaultdict[Node, OrderedSet[UsageIndex]](OrderedSet)
    region_unique = set(region)
    for node_ind, node in enumerate(region):
        flattened_args_kwargs = _get_flat_args(node, {})
        for arg_ind, in_node in enumerate(flattened_args_kwargs):
            if isinstance(in_node, Node) and in_node not in region_unique:
                # in_node may occur in multiple nodes' flat_args
                # track this so we can check if the arg is mutated
                # Previously, we only needed to track one occurrence
                # to be able to map that node to a placeholder
                external_node_to_usages[in_node].add((node_ind, arg_ind))

    return external_node_to_usages


def _get_all_output_indices(regions: list[Region]) -> list[int]:
    # Scan all regions to get the set of all possible output nodes indices in the region
    # perhaps we can record this information during region creation for more efficiency?
    inds_with_external_users: set[int] = set()
    for region in regions:
        _get_inds_with_external_users(region, inds_with_external_users)

    return sorted(inds_with_external_users)


def _get_inds_with_external_users(region: Region, inds_unique: set[int]) -> None:
    for ind, node in enumerate(region):
        for user in node.users:
            if user not in region:
                if ind not in inds_unique:
                    inds_unique.add(ind)


def _create_subgraph(
    region: Region,
    inds_with_external_users: list[int],
) -> tuple[
    torch.fx.Graph,
    list[OrderedSet[UsageIndex]],
    dict[UsageIndex, OrderedSet[int]],
    dict[int, dict[tuple[int, ...], int]],
]:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    external_input_to_usages = _get_external_inputs(region)
    external_node_usages = list[OrderedSet[UsageIndex]]()
    region_to_subgraph_node = {}
    flattened_getitem_nodes: OrderedSet[Node] = OrderedSet()
    node_usage_to_tuple_elems: dict[UsageIndex, OrderedSet[int]] = {}

    for node, usage_indices in external_input_to_usages.items():
        # We don't handle tuples as inputs today
        if _is_tuple_node(node):
            # If a node is a tuple we will possibly create multiple placeholders for them
            # and track which nodes we won't copy into the subgraph because they are flattened away
            # Later, when replacing each region with this subgraph, we will create a getitem node
            # externally which will perform the flattening on the outer nodes.
            flattened_node_indices = _get_flattened_node_indices(node, region)
            for ind in flattened_node_indices:
                placeholder = subgraph.placeholder(
                    f"supgraph_input_{node.name}_flattened_{ind}"
                )
                region_to_subgraph_node[region[ind]] = placeholder
                flattened_getitem_nodes.add(region[ind])
            node_usage_to_tuple_elems[next(iter(usage_indices))] = (
                flattened_node_indices
            )
        else:
            placeholder = subgraph.placeholder(f"subgraph_input_{node.name}")
            region_to_subgraph_node[node] = placeholder

        external_node_usages.append(usage_indices)

    def map_arg(node: Node) -> Node:
        if node in region_to_subgraph_node:
            return region_to_subgraph_node[node]
        else:
            return node

    def copy_to_subgraph(node: Node) -> Node:
        subgraph_node = subgraph.node_copy(node, lambda old: map_arg(old))
        region_to_subgraph_node[node] = subgraph_node
        return subgraph_node

    output_list = []
    ind_to_tuple_spec = {}
    for ind, node in enumerate(region):
        if node not in flattened_getitem_nodes:
            subgraph_node = copy_to_subgraph(node)
            if ind in inds_with_external_users:
                # flatten tuple outputs by generating a getitem node tree
                if _is_tuple_node(node):
                    getitem_nodes, ind_to_tuple_spec[ind] = _create_getitem_nodes(
                        node, subgraph_node, subgraph
                    )
                    output_list.extend(getitem_nodes)
                else:
                    output_list.append(subgraph_node)

    subgraph.output(tuple(output_list))

    return subgraph, external_node_usages, node_usage_to_tuple_elems, ind_to_tuple_spec


def _stable_topological_sort_impl(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
    do_sort: bool = True,
) -> bool:
    # Nodes are in exactly one of these four collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = OrderedSet[Node]()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # - `outputs` are always at the end of the graph
    outputs = OrderedSet[Node]()

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()

        if node.target == "output":
            outputs.add(node)
            assert not node.users, "output nodes should have no users"
            continue

        waiting_for = [
            x
            for x in _get_flat_args_unique(node, node_to_additional_deps)
            if x not in ready
        ]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node and do_sort:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    ready.update(outputs)
    return not waiting and len(ready) == len(graph.nodes)


def _stable_topological_sort(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> None:
    assert _stable_topological_sort_impl(graph, node_to_additional_deps)


def _has_cycle(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> bool:
    return not _stable_topological_sort_impl(
        graph, node_to_additional_deps, do_sort=False
    )


def _populate_additional_deps(
    graph: torch.fx.Graph, node_to_mutated_arg_positions: dict[Node, OrderedSet[int]]
) -> dict[Node, OrderedSet[Node]]:
    node_to_additional_deps: dict[Node, OrderedSet[Node]] = defaultdict(OrderedSet)
    _add_mutation_dependencies(node_to_mutated_arg_positions, node_to_additional_deps)
    _add_global_state_dependencies(graph, node_to_additional_deps)
    return node_to_additional_deps


def _add_global_state_dependencies(
    graph: torch.fx.Graph, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> None:
    import torch.amp

    all_nodes = list(graph.nodes)

    # These are targets of the nodes which need to stay in the same relative place in the graph
    global_state_targets = {torch.amp._enter_autocast, torch.amp._exit_autocast}
    all_nodes_dep_on: list[Node] = []

    def prev_cur_nodes(
        all_nodes: list[Node],
    ) -> Generator[tuple[list[Node], Node], None, None]:
        prev_nodes: list[Node] = []
        next_nodes = list(reversed(all_nodes))

        while next_nodes:
            cur_node = next_nodes.pop()
            yield prev_nodes, cur_node
            prev_nodes.append(cur_node)

    for prev_nodes, cur_node in prev_cur_nodes(all_nodes):
        args_unique = _get_flat_args_unique(cur_node, {})
        new_deps = [n for n in all_nodes_dep_on if n not in args_unique]

        if new_deps:
            additional_deps = node_to_additional_deps[cur_node]
            additional_deps.update(new_deps)

        if cur_node.target in global_state_targets:
            additional_deps = node_to_additional_deps[cur_node]
            additional_deps.update(n for n in prev_nodes if n not in args_unique)
            all_nodes_dep_on.append(cur_node)


def _add_mutation_dependencies(
    node_to_mutated_arg_positions: dict[Node, OrderedSet[int]],
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> None:
    for node, indices in node_to_mutated_arg_positions.items():
        flat_args_kwargs = _get_flat_args(node, {})

        # for all mutated args,
        # add dependency on usages which occur after node to ensure
        # node will always be ordered before them
        # also add node as a dependency on usages which
        # occur before node to ensure node is ordered after them
        for index in indices:
            mutated_arg = flat_args_kwargs[index]
            for user in mutated_arg.users:
                if user is node:
                    continue
                elif user < node:
                    node_to_additional_deps[node].add(user)
                elif user > node:
                    node_to_additional_deps[user].add(node)


def _has_aliasing(
    region: Region,
    inputs: list[Node],
    inds_with_external_users: list[int],
    flattened_getitem_nodes: OrderedSet[Node],
) -> bool:
    input_storages: dict[StorageWeakRef, Node] = dict()
    for node in inputs:
        if node in flattened_getitem_nodes:
            continue
        example_value = node.meta["example_value"]
        if isinstance(example_value, torch.Tensor):
            storage = StorageWeakRef(example_value._typed_storage())
            if storage in input_storages:
                # input-input aliasing
                log.debug(
                    "NYI: Failed to substitute region %s due to input-output aliasing detected at nodes %s, %s",
                    region,
                    input_storages[storage],
                    node,
                )
                return True
            input_storages[storage] = node
    output_storages: dict[StorageWeakRef, Node] = dict()
    for i in inds_with_external_users:
        out_node = region[i]
        if out_node in flattened_getitem_nodes:
            continue
        if out_node:
            example_value = out_node.meta["example_value"]
            assert not isinstance(example_value, list)
            if isinstance(example_value, torch.Tensor):
                storage = StorageWeakRef(example_value._typed_storage())
                if storage in output_storages:
                    # output-output aliasing
                    log.debug(
                        "NYI: Failed to substitute region %s due to output-output aliasing detected at nodes %s, %s",
                        region,
                        output_storages[storage],
                        out_node,
                    )
                    return True
                output_storages[storage] = out_node
    intersected_storages = input_storages.keys() & output_storages.keys()
    if len(intersected_storages) > 0:
        # input-output aliasing
        aliased = [
            (input_storages[s], output_storages[s]) for s in intersected_storages
        ]
        aliased = ", ".join([f"{i} and {o}" for i, o in aliased])
        log.debug(
            "NYI: Failed to substitute region %s due to input-output aliasing detected at nodes %s",
            region,
            aliased,
        )
        return True
    return False


def _is_tuple_node(node: Node) -> bool:
    return isinstance(node.meta["example_value"], tuple)


def _get_children_getitems(node: Node) -> Generator[Node, None, None]:
    for user in node.users:
        if user.target == operator.getitem and isinstance(user.args[1], int):
            yield user


def _get_flattened_node_indices(node: Node, region: Region) -> OrderedSet[int]:
    """Returns an ordered set of indices, each representing a node in the region which will be flattened"""
    flattened_node_to_ind = {n: i for i, n in enumerate(region)}
    node_indices: OrderedSet[int] = OrderedSet()
    queue = deque(_get_children_getitems(node))
    while queue:
        cur_node = queue.popleft()
        if any(user in region for user in cur_node.users):
            node_indices.add(flattened_node_to_ind[cur_node])
        for child in _get_children_getitems(cur_node):
            queue.append(child)
    return node_indices


def _create_getitem_nodes(
    node: Node, subgraph_tuple_node: Node, subgraph: torch.fx.Graph
) -> tuple[list[Node], dict[tuple[int, ...], int]]:
    tup = node.meta["example_value"]
    assert isinstance(tup, tuple), "_get_getitem_children expects tuple"

    getitem_nodes: list[Node] = []
    queue = deque([(e, (i,), subgraph_tuple_node) for i, e in enumerate(tup)])
    path_to_output_index = {}

    while queue:
        cur_elem, path, parent = queue.popleft()

        with subgraph.inserting_after(parent):
            new_getitem_node = subgraph.create_node(
                "call_function", operator.getitem, (parent, path[-1]), {}
            )
        new_getitem_node.meta["example_value"] = cur_elem

        path_to_output_index[path] = len(getitem_nodes)
        getitem_nodes.append(new_getitem_node)

        if isinstance(cur_elem, tuple):
            queue.extend(
                [(e, path + (i,), new_getitem_node) for i, e in enumerate(cur_elem)]  # type: ignore[arg-type,misc]
            )

    return getitem_nodes, path_to_output_index  # type: ignore[return-value]


def _replace_tuple_outputs(
    node: Node,
    output_index: int,
    tuple_spec: dict[tuple[int, ...], int],
    invoke_subgraph_node: Node,
    graph: torch.fx.Graph,
) -> OrderedSet[Node]:
    assert _is_tuple_node(node), "_replace_tuple_outputs expects a tuple node"

    queue = deque((c, (c.args[1],)) for c in _get_children_getitems(node))
    erased_nodes: OrderedSet[Node] = OrderedSet()
    while queue:
        cur_node, path = queue.pop()

        for c in _get_children_getitems(cur_node):
            queue.append((c, path + (c.args[1],)))  # type: ignore[return-value, arg-type]

        with graph.inserting_after(invoke_subgraph_node):
            subgraph_output = graph.create_node(
                "call_function",
                operator.getitem,
                (invoke_subgraph_node, output_index + tuple_spec[path]),  # type: ignore[index]
                {},
            )
        cur_node.replace_all_uses_with(subgraph_output, propagate_meta=True)
        graph.erase_node(cur_node)
        erased_nodes.add(cur_node)

    graph.erase_node(node)
    erased_nodes.add(node)
    return erased_nodes
