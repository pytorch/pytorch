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


def _is_wait_tensor(node: fx.Node) -> bool:
    """Check if a node is a wait_tensor operation."""
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


@dataclass(frozen=True)
class StorageKey:
    storage: torch.UntypedStorage
    device: torch.device

    def __hash__(self) -> int:
        return self.storage._cdata

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StorageKey):
            return False
        return (
            self.storage._cdata == other.storage._cdata and self.device == other.device
        )


class GraphAliasTracker:
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

        # First pass: build storage allocations and track uses
        for node in nodes:
            # Get output storages
            output_storages = self._get_output_storages(node)
            self.node_to_output_storages[node] = output_storages

            # Track fresh allocations
            fresh_allocations: OrderedSet[StorageKey] = OrderedSet()
            for storage_key in output_storages:
                if storage_key not in self.storage_to_allocator:
                    self.storage_to_allocator[storage_key] = node
                    fresh_allocations.add(storage_key)
            self.node_to_fresh_allocations[node] = fresh_allocations

            # Track input storage uses (safe because inputs were already processed)
            input_storages = self._get_input_storages(node)
            self.node_to_storage_uses[node] = input_storages
            for storage_key in input_storages:
                self.storage_to_uses[storage_key].add(node)

        # Second pass: find last users (iterate in reverse)
        for node in reversed(nodes):
            input_storages = self.node_to_storage_uses[node]
            for storage_key in input_storages:
                if storage_key not in self.storage_to_last_user:
                    self.storage_to_last_user[storage_key] = node
                    self.node_to_storages_last_used[node].add(storage_key)

    @staticmethod
    def _get_output_storages(node: fx.Node) -> OrderedSet[StorageKey]:
        """
        Get all storages from a node's outputs.

        Uses pytree to handle arbitrary nested structures.
        """
        val = node.meta.get("val")
        if val is None:
            return OrderedSet()

        storages: OrderedSet[StorageKey] = OrderedSet()

        def collect_storage(tensor: torch._subclasses.FakeTensor) -> None:
            storages.add(StorageKey(tensor.untyped_storage(), tensor.device))

        # Use tree_map_only to handle FakeTensors in nested structures
        tree_map_only(torch._subclasses.FakeTensor, collect_storage, val)

        return storages

    def _get_input_storages(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """
        Get all storages from a node's inputs.

        For wait_tensor operations, this includes both the direct inputs (the collective handle)
        and all inputs from the corresponding collective start operation, since the wait
        is what actually allows those inputs to be freed.
        """
        input_storages: OrderedSet[StorageKey] = OrderedSet()

        for input_node in node.all_input_nodes:
            input_storages.update(self.node_to_output_storages[input_node])

        # Handle collective start/wait pairs: wait_tensor should also "use" all inputs
        # from the collective start operation, since it's the wait that releases them
        if _is_wait_tensor(node):
            collective_start = node.args[0]
            assert isinstance(collective_start, fx.Node)
            input_storages.update(self.node_to_storage_uses[collective_start])

        return input_storages

    def get_fresh_allocations(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """Get all fresh storage allocations by this node (not views/aliases)."""
        return self.node_to_fresh_allocations[node]

    def get_storage_uses(self, node: fx.Node) -> OrderedSet[StorageKey]:
        """Get all storages that this node uses as inputs."""
        return self.node_to_storage_uses[node]

    def get_storages_last_used(
        self,
        node: fx.Node,
    ) -> OrderedSet[StorageKey]:
        """
        Get storages whose last use is at this node.
        """
        return self.node_to_storages_last_used[node]


def _size_of_default(num_bytes: Union[int, torch.SymInt]) -> int:
    return hint_int(num_bytes, fallback=torch._inductor.config.unbacked_symint_fallback)


def device_filter(device: torch.device) -> bool:
    return device.type != "cpu"


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
    alias_info = GraphAliasTracker(nodes)

    # Build memory profile
    current_memory = 0

    for node in itertools.chain(
        graph.find_nodes(op="placeholder"), graph.find_nodes(op="get_attr")
    ):
        for storage_key in alias_info.get_fresh_allocations(node):
            if device_filter(storage_key.device):
                current_memory += size_of(storage_key.storage.nbytes())

    memory_profile = [current_memory]

    for node in nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue

        # Process allocations
        for storage_key in alias_info.get_fresh_allocations(node):
            if device_filter(storage_key.device):
                current_memory += size_of(storage_key.storage.nbytes())

        memory_profile.append(current_memory)

        # Process deallocations
        for storage_key in alias_info.get_storages_last_used(node):
            allocator = alias_info.storage_to_allocator[storage_key]
            if is_releasable(allocator):
                if device_filter(storage_key.device):
                    current_memory -= size_of(storage_key.storage.nbytes())

        memory_profile.append(current_memory)

    return memory_profile


def get_fwd_bwd_interactions(
    fwd_graph: fx.Graph,
    bwd_graph: fx.Graph,
    size_of: Optional[Callable[[Union[int, torch.SymInt]], int]] = None,
) -> tuple[int, OrderedSet[str]]:
    """
    Analyze the interactions between the forward (fwd) and backward (bwd) graphs
    to determine memory usage characteristics.

    Args:
    - fwd_graph (fx.Graph): The forward graph representing the forward pass.
    - bwd_graph (fx.Graph): The backward graph representing the backward pass.
    - size_of (Callable[[Union[int, torch.SymInt]], int]): A function that converts
      byte counts (possibly symbolic) to concrete integers.

    Returns:
    - tuple[int, OrderedSet[str]]: A tuple containing:
        1. The baseline memory usage during the backward pass, accounting for
           storages that persist from the forward pass (i.e., in fwd output but
           not in bwd input).
        2. A set of node names whose storage cannot be released during the bwd pass.
           These include nodes that use storage from primals or are in bwd input
           but not in fwd output.
    """

    size_of = size_of or _size_of_default

    # Build alias info for forward graph
    fwd_nodes = list(fwd_graph.nodes)
    fwd_alias_info = GraphAliasTracker(fwd_nodes)

    # Identify storages allocated by primal placeholder nodes
    primal_storages: OrderedSet[StorageKey] = OrderedSet()
    for node in fwd_graph.find_nodes(op="placeholder"):
        if node.name.startswith("primals"):
            primal_storages.update(fwd_alias_info.get_fresh_allocations(node))

    # Get storages in forward output
    fwd_output_node = next(iter(reversed(fwd_graph.nodes)))[-1]
    assert fwd_output_node.op == "output"
    fwd_output_storages = fwd_alias_info.get_storage_uses(fwd_output_node)

    # Node names that should not be deleted during memory profile estimation of bwd_graph
    do_not_delete: OrderedSet[str] = OrderedSet()

    # Collect all storages in backward inputs and identify nodes to not delete
    bwd_input_storages: OrderedSet[StorageKey] = OrderedSet()
    for node in bwd_graph.find_nodes(op="placeholder"):
        node_storages = GraphAliasTracker._get_output_storages(node)
        bwd_input_storages.update(node_storages)

        # Check if this node uses primal storage
        if node_storages & primal_storages:
            do_not_delete.add(node.name)

        # Check if this node's storages are not in forward outputs
        # (meaning it's an external input to backward pass)
        if not (node_storages & fwd_output_storages):
            do_not_delete.add(node.name)

    # Calculate baseline memory: storages in fwd output but not in bwd input
    # These storages persist throughout the backward pass
    baseline_storages = fwd_output_storages - bwd_input_storages
    bwd_baseline_memory = 0
    for storage_key in baseline_storages:
        if storage_key.device.type != "cpu":
            bwd_baseline_memory += size_of(storage_key.storage.nbytes())

    return bwd_baseline_memory, do_not_delete


def get_peak_memory(
    fwd_graph: fx.Graph,
    bwd_graph: fx.Graph,
) -> int:
    def _is_releasable(n: fx.Node) -> bool:
        # Storages of primals cannot be released during fwd or bwd pass.
        return not n.name.startswith("primals")

    fwd_peak_memory = max(build_memory_profile(fwd_graph, _is_releasable))

    bwd_baseline_memory, bwd_do_not_delete = get_fwd_bwd_interactions(
        fwd_graph,
        bwd_graph,
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
