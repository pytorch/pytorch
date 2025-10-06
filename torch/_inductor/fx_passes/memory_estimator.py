import itertools
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.fx as fx
from torch.fx.experimental.symbolic_shapes import hint_int
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map_only


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StorageKey:
    storage: torch.UntypedStorage
    device: torch.device

    def __hash__(self):
        return self.storage._cdata

    def __eq__(self, other):
        return self.storage._cdata == other.storage._cdata


class AliasInfo:
    """
    Tracks storage allocation and usage relationships in an FX graph.

    Differentiates between:
    - Fresh allocations: nodes that allocate new storage (not views/aliases)
    - Uses: nodes that use a storage as input
    """

    def __init__(self, nodes: list[fx.Node]):
        # Map from node to the fresh storages it allocates (not views/aliases)
        self.node_to_fresh_allocations: dict[fx.Node, OrderedSet[StorageKey]] = {}

        # Map from storage to the node that originally allocated it
        self.storage_to_allocator: dict[StorageKey, fx.Node] = {}

        # Map from node to all storages it uses as inputs
        self.node_to_storage_uses: dict[fx.Node, OrderedSet[StorageKey]] = {}

        # Map from storage to all nodes that use it
        self.storage_to_uses: dict[StorageKey, OrderedSet[fx.Node]] = defaultdict(
            OrderedSet
        )

        # Map from storage to the last node that uses it
        self.storage_to_last_user: dict[StorageKey, fx.Node] = {}

        # Map from node to storages that have their last use at that node
        self.node_to_storages_last_used: dict[fx.Node, OrderedSet[StorageKey]] = (
            defaultdict(OrderedSet)
        )

        # Track all output storages for each node (for building usage graph)
        self.node_to_output_storages: dict[fx.Node, OrderedSet[StorageKey]] = {}

        # First pass: build storage allocations
        for node in nodes:
            output_storages = self._get_output_storages(node)
            self.node_to_output_storages[node] = output_storages

            fresh_allocations = OrderedSet()
            for storage_key in output_storages:
                if storage_key not in self.storage_to_allocator:
                    self.storage_to_allocator[storage_key] = node
                    fresh_allocations.add(storage_key)

            self.node_to_fresh_allocations[node] = fresh_allocations

        # Second pass: track storage uses (inputs)
        for node in nodes:
            input_storages = self._get_input_storages(node)
            self.node_to_storage_uses[node] = input_storages

            for storage_key in input_storages:
                self.storage_to_uses[storage_key].add(node)

        # Third pass: find last users (iterate in reverse)
        for node in reversed(nodes):
            input_storages = self._get_input_storages(node)
            for storage_key in input_storages:
                if storage_key not in self.storage_to_last_user:
                    self.storage_to_last_user[storage_key] = node
                    self.node_to_storages_last_used[node].add(storage_key)

    def _get_output_storages(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """
        Get all storages from a node's outputs.

        Uses pytree to handle arbitrary nested structures.
        """
        val = node.meta.get("val")
        if val is None:
            return OrderedSet()

        storages = OrderedSet()

        def collect_storage(tensor):
            if hasattr(tensor, "untyped_storage"):
                storages.add(StorageKey(tensor.untyped_storage(), tensor.device))

        # Use tree_map_only to handle FakeTensors in nested structures
        tree_map_only(torch._subclasses.FakeTensor, collect_storage, val)

        return storages

    def _get_input_storages(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """
        Get all storages from a node's inputs.
        """
        input_storages = OrderedSet()

        for input_node in node.all_input_nodes:
            input_storages.update(self.node_to_output_storages[input_node])

        return input_storages

    def get_fresh_allocations(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """Get all fresh storage allocations by this node (not views/aliases)."""
        return self.node_to_fresh_allocations[node]

    def get_storage_uses(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """Get all storages that this node uses as inputs."""
        return self.node_to_storage_uses[node]

    def get_storages_last_used_at(
        self,
        node: fx.Node,
    ) -> OrderedSet[StorageKey]:
        """
        Get storages whose last use is at this node.

        Returns storages that are currently active and have their
        last use at this node.
        """
        return self.node_to_storages_last_used[node]


def _size_of_default(num_bytes: Union[int, torch.SymInt]) -> int:
    return hint_int(num_bytes, fallback=torch._inductor.config.unbacked_symint_fallback)


def build_memory_profile(
    graph: fx.Graph,
    is_releasable: Callable[[fx.Node], bool],
    size_of: Optional[Callable[[Union[int, torch.SymInt]], int]] = None,
) -> list[int]:
    """
    Function to estimate the memory profile of an input FX graph.

    Args:
    - graph (fx.Graph): The input FX graph for which the memory profile
      is to be estimated.
    - is_releasable (Callable[[fx.Node], bool]): A function that
      determines if a node's memory can be released (e.g. primal nodes
      cannot be released).
    - size_of (Callable[[Union[int, torch.SymInt]], int]): A function that converts
      byte counts (possibly symbolic) to concrete integers.

    Returns:
    - List[int]: A list representing the memory profile over the execution
      of the graph, where each entry corresponds to the memory usage at
      a particular point in the execution.
    """

    size_of = size_of or _size_of_default
    nodes = list(graph.nodes)
    alias_info = AliasInfo(nodes)

    # Build memory profile
    current_memory = 0

    for node in itertools.chain(
        graph.find_nodes(op="placeholder"), graph.find_nodes(op="get_attr")
    ):
        for storage_key in alias_info.get_fresh_allocations(node):
            if storage_key.device.type != "cpu":
                current_memory += size_of(storage_key.storage.nbytes())

    memory_profile = [current_memory]

    for node in nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue

        # Process allocations
        for storage_key in alias_info.get_fresh_allocations(node):
            if storage_key.device.type != "cpu":
                current_memory += size_of(storage_key.storage.nbytes())

        memory_profile.append(current_memory)

        # Process deallocations
        for storage_key in alias_info.get_storages_last_used_at(node):
            allocator = alias_info.storage_to_allocator[storage_key]
            if is_releasable(allocator):
                if storage_key.device.type != "cpu":
                    current_memory -= size_of(storage_key.storage.nbytes())

        memory_profile.append(current_memory)

    return memory_profile


def get_peak_memory(
    fwd_graph: fx.Graph,
    bwd_graph: fx.Graph,
) -> int:
    def _is_releasable(n: fx.Node) -> bool:
        # Storages of primals cannot be released during fwd or bwd pass.
        return not n.name.startswith("primals")

    fwd_peak_memory = max(build_memory_profile(fwd_graph, _is_releasable))

    bwd_baseline_memory, bwd_do_not_delete = get_fwd_bwd_interactions(
        fwd_graph, bwd_graph, _size_of_default
    )

    def _is_bwd_releasable(n: fx.Node) -> bool:
        # Storages of nodes in bwd_do_not_delete cannot be released
        # during the bwd pass.
        return _is_releasable(n) and n.name not in bwd_do_not_delete

    bwd_peak_memory = bwd_baseline_memory + max(
        build_memory_profile(bwd_graph, _is_bwd_releasable)
    )
    return max(
        fwd_peak_memory,
        bwd_peak_memory,
    )
