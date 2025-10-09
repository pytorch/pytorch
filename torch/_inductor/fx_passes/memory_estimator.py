import logging
import operator
from typing import Any, Callable

import torch.fx as fx
from torch._functorch.partitioners import _size_of, get_default_op_list
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def build_memory_profile(
    graph: fx.Graph,
    size_of: Callable[[fx.Node], int],
    is_releasable: Callable[[fx.Node], bool],
) -> list[int]:
    """
    Function to estimate the memory profile of an input FX graph.

    Args:
    - graph (fx.Graph): The input FX graph for which the memory profile
      is to be estimated.
    - size_of (Callable[[fx.Node], int]): A function that returns
      the size of a given node.
    - is_releasable (Callable[[fx.Node], bool]): A function that
      determines if a node's memory can be released (e.g. primal nodes
      cannot be released).

    Returns:
    - List[int]: A list representing the memory profile over the execution
      of the graph, where each entry corresponds to the memory usage at
      a particular point in the execution.
    """

    nodes = list(graph.nodes)
    op_types = get_default_op_list()

    class AliasInfo:
        """
        Class for storing and accessing alias information of a FX graph.

        Attributes:
        - view_to_source: Maps view nodes to their source nodes
        - getitem_to_source: Maps getitem nodes to (source_node, key) tuples
        - source_to_getitems: Maps source nodes to dictionaries of
          {key: getitem_node, "unclaimed": None}
        - source_to_unclaimed_size: Maps source nodes to their storage size
          unclaimed by any getitem_nodes
        """

        def __init__(self, nodes: list[fx.Node]):
            """
            Initialize the AliasInfo class with a list of FX graph nodes.

            Args:
            - nodes (list[fx.Node]): A list of nodes from an FX graph,
              ordered in execution order.

            The constructor analyzes the relationships between nodes in the FX graph
            to populate alias information. It identifies two types of alias nodes:
            getitem and view. For each view, it maps it to its source. For each
            getitem, it maps it to its source and key. It also populates mappings
            for source nodes to their getitems and calculates unclaimed storage sizes.

            """
            # For each view, we map it to its source.
            # Note that we treat getitems of a view (e.g. aten.split) as views.
            self.view_to_source: dict[fx.Node, fx.Node] = {}

            # For each remaining getitem, we map it to its source and key.
            self.getitem_to_source: dict[fx.Node, tuple[fx.Node, Any]] = {}

            # For each none-view source_node of getitems, we map it to a dictionary
            # in the form of {key: getitem_node, ..., "unclaimed": None}, where
            # "unclaimed" is a dummy key that represents all elements in the
            # source_node that is not claimed by any getitems.
            self.source_to_getitems: dict[fx.Node, dict[Any, fx.Node | None]] = {}

            # For each none-view source_node of getitems with at least one unclaimed
            # elements, we map it to its unclaimed storage size.
            self.source_to_unclaimed_size: dict[fx.Node, int] = {}

            for node in nodes:
                is_view = op_types.is_view(node)
                is_getitem = node.target is operator.getitem
                if not (is_view or is_getitem):
                    continue
                assert not (is_view and is_getitem)
                assert node.args and isinstance(node.args[0], fx.Node)
                source = node.args[0]
                if is_view:
                    assert not isinstance(source.meta["val"], list | tuple | dict)
                    if source in self.view_to_source:
                        source = self.view_to_source[source]
                    self.view_to_source[node] = source
                if is_getitem:
                    assert isinstance(source.meta["val"], list | tuple | dict)
                    # Source of getitem can be a view (e.g. aten.split).
                    if source in self.view_to_source:
                        if source in self.view_to_source:
                            source = self.view_to_source[source]
                        # In this case, the getitem node should be treated
                        # the same way as a regular view.
                        self.view_to_source[node] = source
                        continue
                    # Source of getitem cannot be a getitem.
                    assert source not in self.getitem_to_source

                    # There must be a second argument that specifies the key.
                    assert len(node.args) >= 2
                    key = node.args[1]
                    self.getitem_to_source[node] = (source, key)

                    # Populate source_to_getitems.
                    if source not in self.source_to_getitems:
                        self.source_to_getitems[source] = {"unclaimed": None}
                    assert key not in self.source_to_getitems[source]
                    self.source_to_getitems[source][key] = node  # type: ignore[index]

            for source, getitem_map in self.source_to_getitems.items():
                unclaimed_source_size = size_of(source)
                for key, getitem_node in getitem_map.items():
                    if key != "unclaimed" and getitem_node is not None:
                        unclaimed_source_size -= size_of(getitem_node)
                assert unclaimed_source_size >= 0
                if unclaimed_source_size > 0:
                    self.source_to_unclaimed_size[source] = unclaimed_source_size

        def is_view(self, node: fx.Node) -> bool:
            return node in self.view_to_source

        def is_getitem(self, node: fx.Node) -> bool:
            return node in self.getitem_to_source

        def get_source(self, node: fx.Node) -> fx.Node | tuple[fx.Node, Any]:
            if self.is_view(node):
                return self.view_to_source[node]
            if self.is_getitem(node):
                return self.getitem_to_source[node]
            return node

        def is_source_of_getitems(self, node: fx.Node) -> bool:
            return node in self.source_to_getitems

        def get_storage_keys(self, source_node: fx.Node) -> list[Any]:
            assert source_node in self.source_to_getitems
            return list(self.source_to_getitems[source_node].keys())

        def get_unclaimed_storage_size(self, source_node: fx.Node) -> int:
            return self.source_to_unclaimed_size.get(source_node, 0)

        def get_getitem_by_key(self, source: fx.Node, key: Any) -> fx.Node | None:
            assert source in self.source_to_getitems
            assert key in self.source_to_getitems[source]
            return self.source_to_getitems[source][key]

    def _get_last_usage(
        nodes: list[fx.Node], alias_info: AliasInfo
    ) -> dict[fx.Node, list[tuple[fx.Node, Any]]]:
        """
        Determine the last usage point of each storage. This information is used to
        identify when storages can be safely released.

        Args:
        - nodes (list[fx.Node]): A list of nodes from the FX graph, ordered
          in execution order.
        - alias_info (AliasInfo): An instance of AliasInfo containing aliasing
          relationships between nodes in the graph.

        Returns:
        - Dict[fx.Node, list[tuple[fx.Node, Optional[Any]]]]: A mapping
          from each node to a list of storages (represented as tuples of source node
          and key) that are last used by that node. This helps in identifying which
          storages can be released after the node's execution.

        """
        storage_to_last_user: dict[tuple[fx.Node, Any], fx.Node] = {}
        node_to_last_used_storages: dict[fx.Node, list[tuple[fx.Node, Any]]] = {}

        def register_last_uses(use: fx.Node, user: fx.Node) -> None:
            keys: list[Any] = []
            if alias_info.is_view(use):
                # When use is a view (or getitem of a view),
                # user is essentially using the storage allocated at the
                # creation of the source of use.
                use = alias_info.get_source(use)  # type: ignore[assignment]

            if alias_info.is_source_of_getitems(use):  # type: ignore[arg-type]
                # When use is a source of getitems, user is using all separate
                # storages of use.
                keys.extend(alias_info.get_storage_keys(use))  # type: ignore[arg-type]
            elif alias_info.is_getitem(use):  # type: ignore[arg-type]
                # When use is a getitem, user is essentially using a separate
                # storage of the source of use specified by key.
                use, key = alias_info.get_source(use)  # type: ignore[assignment,misc]
                keys.append(key)
            else:
                keys.append(None)

            assert keys

            for key in keys:
                if (use, key) not in storage_to_last_user:  # type: ignore[comparison-overlap]
                    storage_to_last_user[(use, key)] = user  # type: ignore[index]
                    node_to_last_used_storages.setdefault(user, []).append((use, key))  # type: ignore[arg-type]

        for node in reversed(nodes):
            fx.node.map_arg(node.args, lambda n: register_last_uses(n, node))
            fx.node.map_arg(node.kwargs, lambda n: register_last_uses(n, node))

        return node_to_last_used_storages

    alias_info = AliasInfo(nodes)
    node_to_last_used_storages = _get_last_usage(nodes, alias_info)

    # Initialize memory profile
    memory_profile = [0]

    # Process the graph
    for node in nodes:
        if node.op == "placeholder":
            out_mem = size_of(node)
            memory_profile[0] += out_mem
        elif node.op == "output":
            pass
        elif (
            node.op == "call_function"
            or node.op == "call_module"
            or node.op == "call_method"
        ):
            # Aliases don't allocate new memory
            if alias_info.is_view(node) or alias_info.is_getitem(node):
                memory_profile.append(memory_profile[-1])
            else:
                out_mem = size_of(node)
                memory_profile.append(memory_profile[-1] + out_mem)

            # Process storages that are no longer needed after this operation
            storages_to_release = [
                (use, key)
                for use, key in node_to_last_used_storages.get(node, [])
                if is_releasable(use)
            ]
            freed_memory = 0
            for node_to_release, key in storages_to_release:
                released_memory_size = 0
                if key is None:
                    released_memory_size = size_of(node_to_release)
                elif key == "unclaimed":
                    released_memory_size = alias_info.get_unclaimed_storage_size(
                        node_to_release
                    )
                else:
                    getitem_node = alias_info.get_getitem_by_key(node_to_release, key)
                    if getitem_node is not None:
                        released_memory_size = size_of(getitem_node)
                freed_memory += released_memory_size

            assert freed_memory >= 0
            memory_profile.append(memory_profile[-1] - freed_memory)
    return memory_profile


def get_fwd_bwd_interactions(
    fwd_graph: fx.Graph,
    bwd_graph: fx.Graph,
    size_of: Callable[[fx.Node], int],
) -> tuple[int, OrderedSet[str]]:
    """
    Analyze the interactions between the forward (fwd) and backward (bwd) graphs
    to determine memory usage characteristics.

    Args:
    - fwd_graph (fx.Graph): The forward graph representing the forward pass.
    - bwd_graph (fx.Graph): The backward graph representing the backward pass.
    - size_of (Callable[[fx.Node], int]): A function that returns the size
      of a given node.

    Returns:
    - tuple[int, Set[fx.Node]]: A tuple containing:
        1. The baseline memory usage during the backward pass, accounting for
           nodes that persist from the forward pass (i.e., in fwd output but
           not in bwd input).
        2. A set of nodes whose storage cannot be released during the bwd pass.
           These include nodes that are views of primals or in bwd input
           but not in fwd output.
    """

    def get_nodes_in_output(graph: fx.Graph) -> OrderedSet[fx.Node]:
        """
        Get the nodes in the output of a graph.

        Args:
        - graph (fx.Graph): The input graph.

        Returns:
        - list[fx.Node]: A list of nodes in the output of the graph.
        """
        output_node = list(graph.nodes)[-1]
        assert output_node.op == "output"
        nodes_in_output: OrderedSet[fx.Node] = OrderedSet()

        def add_node(node: fx.Node) -> None:
            nodes_in_output.add(node)

        # Using map_arg since output_node.args[0] can be of different types
        # e.g. tuple, list, dict, fx.Node, etc.
        fx.node.map_arg(output_node.args[0], lambda n: add_node(n))
        return nodes_in_output

    op_types = get_default_op_list()

    bwd_baseline_memory = 0
    # placeholder nodes besides primals of the bwd_graph that should also
    # not be deleted during memory profile estimation of the bwd_graph
    do_not_delete: OrderedSet[str] = OrderedSet()

    fwd_outputs = {}
    for node in get_nodes_in_output(fwd_graph):
        is_view_of_primal = False
        if op_types.is_view(node):
            source = node.args[0]
            if isinstance(source, fx.Node) and source.name.startswith("primals"):
                is_view_of_primal = True
        fwd_outputs[node.name] = (size_of(node), is_view_of_primal)
    bwd_inputs: OrderedSet[str] = OrderedSet()
    for node in bwd_graph.nodes:
        if node.op == "placeholder":
            bwd_inputs.add(node.name)
            if node.name.startswith("view"):
                # if node is a view, then it has to be in fwd_outputs
                assert node.name in fwd_outputs
                _, is_view_of_primal = fwd_outputs[node.name]
                if is_view_of_primal:
                    # Add node to do_not_delete because it is a view of a primal
                    do_not_delete.add(node.name)

            # if node is not in fwd_outputs, then add it to do_not_delete
            if node.name not in fwd_outputs:
                do_not_delete.add(node.name)

    # nodes that are in fwd_outputs but not in bwd_inputs take memory storage
    # throughout the bwd pass
    for name, (size, _) in fwd_outputs.items():
        if name not in bwd_inputs:
            bwd_baseline_memory += size

    return bwd_baseline_memory, do_not_delete


def get_peak_memory(
    fwd_graph: fx.Graph,
    bwd_graph: fx.Graph,
) -> int:
    def _safe_size_of(n: fx.Node) -> int:
        try:
            return _size_of(n)
        except Exception:
            log.warning("Failed size_of(%s). Returning 0 instead.", n)
            return 0

    def _is_releasable(n: fx.Node) -> bool:
        # Storages of primals cannot be released during fwd or bwd pass.
        return not n.name.startswith("primals")

    fwd_peak_memory = max(
        build_memory_profile(fwd_graph, _safe_size_of, _is_releasable)
    )

    # tmp change
    bwd_baseline_memory, bwd_do_not_delete = get_fwd_bwd_interactions(
        fwd_graph, bwd_graph, _safe_size_of
    )

    def _is_bwd_releasable(n: fx.Node) -> bool:
        # Storages of nodes in bwd_do_not_delete cannot be released
        # during the bwd pass.
        return _is_releasable(n) and n.name not in bwd_do_not_delete

    bwd_peak_memory = bwd_baseline_memory + max(
        build_memory_profile(bwd_graph, _safe_size_of, _is_bwd_releasable)
    )
    return max(
        fwd_peak_memory,
        bwd_peak_memory,
    )
