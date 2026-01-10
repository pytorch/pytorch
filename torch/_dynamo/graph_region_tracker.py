"""
This module provides functionality for tracking and managing regions in computational graphs.
It supports graph optimization by identifying and grouping similar regions based on their
structure and behavior. The module implements algorithms for:

1. Tracking nodes and their relationships in the computational graph
2. Identifying identical or similar regions across the graph
3. Managing graph regions for optimization purposes
4. Supporting deduplication and other graph transformation passes

The core functionality revolves around the GraphRegionTracker class which maintains
mappings between nodes and their duplicates, enabling efficient graph analysis and
optimization operations.
"""

from __future__ import annotations

import copyreg
import io
import logging
import math
import operator
import pickle
from collections import defaultdict, deque
from dataclasses import fields
from typing import Any, Optional, TYPE_CHECKING, TypeVar

import torch._logging
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_flatten
from .graph_utils import _get_flat_args_unique


T = TypeVar("T")


if TYPE_CHECKING:
    from collections.abc import Callable

    from .symbolic_convert import InstructionTranslatorBase


Node = torch.fx.Node
Region = list[Node]
IdenticalNodes = list[Node]
GlobalStateKey = tuple[
    bool,
    bool,
    int,
    tuple[bool, bool],
    tuple[bool, bool],
    torch.dtype,
    bool,
    bool,
    bool,
    bool,
]

log = logging.getLogger(__name__)
graph_expansion_log = torch._logging.getArtifactLogger(
    __name__, "graph_region_expansion"
)


def debug_log(msg: str, *args) -> None:  # type: ignore[no-untyped-def]
    graph_expansion_log.debug(msg, *args)


def _extract_tensor_metadata_for_node_hash(
    x: torch.Tensor,
) -> tuple[Callable[[T], T], tuple[Any, ...]]:
    from torch._inductor.codecache import _ident, extract_tensor_metadata_for_cache_key

    out = []
    metadata = extract_tensor_metadata_for_cache_key(x)
    for field in fields(metadata):
        out.append(getattr(metadata, field.name))

    return (_ident, tuple(out))


class NodeHashException(Exception):
    pass


class InputPickler(pickle.Pickler):
    def __init__(self) -> None:
        from torch._inductor.codecache import _ident

        stream = io.BytesIO()
        self._stream = stream
        super().__init__(stream)
        self.dispatch_table = copyreg.dispatch_table.copy()
        self.dispatch_table.update(
            {
                FakeTensor: _extract_tensor_metadata_for_node_hash,
                torch.SymInt: lambda x: (_ident, (str(x),)),
                torch.SymBool: lambda x: (_ident, (str(x),)),
                torch.SymFloat: lambda x: (_ident, (str(x),)),
            }
        )
        self.fast = True

    def dumps(self, obj: Any) -> bytes:
        """
        Pickle an object and return a byte string.
        """
        try:
            self.dump(obj)
            return self._stream.getvalue()
        except (TypeError, AttributeError) as e:
            raise NodeHashException from e
        finally:
            self._stream.seek(0)
            self._stream.truncate(0)


def _extract_args(arg: Any) -> Any:
    if isinstance(arg, Node):
        return arg.meta.get("example_value")
    elif isinstance(arg, (torch.Tensor, int)):
        return arg
    else:
        return None


def _normalize_args(
    node: Node,
) -> tuple[tuple[str, ...], tuple[Optional[Any], ...]]:
    flat_args, _ = tree_flatten(node.args)
    sorted_kwargs = sorted(node.kwargs.items(), key=operator.itemgetter(0))
    sorted_keys = tuple(sorted(node.kwargs.keys()))
    flat_kwargs, _ = tree_flatten(sorted_kwargs)
    all_args = flat_args + flat_kwargs
    return (sorted_keys, tuple(_extract_args(arg) for arg in all_args))


def _sort_with_ref_region(
    index_to_rank: dict[int, int], regions: list[list[Any]]
) -> None:
    # sort topologically
    # we need to handle edge cases where some nodes have no dependencies
    # so first we map each node to its ranking
    ref_region = regions[0]
    sorted_indices = sorted(range(len(ref_region)), key=lambda i: index_to_rank[i])
    for region in regions:
        region[:] = [region[i] for i in sorted_indices]


def get_global_state_key() -> GlobalStateKey:
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.get_num_threads(),
        torch._C._get_cublas_allow_fp16_reduced_precision_reduction(),
        torch._C._get_cublas_allow_bf16_reduced_precision_reduction(),
        torch.get_default_dtype(),
        torch.are_deterministic_algorithms_enabled(),
        torch._C._get_cublas_allow_tf32(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch._C._autograd._saved_tensors_hooks_is_enabled(),  # type: ignore[attr-defined]
    )


# This is typical BFS with the caveat
# that a node's children need to be explicitly
# added with the add_children() method
# The flow is yield a node and check if it's valid for all regions
# if not valid, discard and continue onto the next node
# Note: this iterates backward through the graph by looking at args/kwargs
# of a node
class BackwardBfsArgIter:
    def __init__(self, origin: Node) -> None:
        self._cur: Optional[Node] = origin
        self._queue: deque[Optional[Node]] = deque()

    @staticmethod
    def create(origin: Node) -> BackwardBfsArgIter:
        it = BackwardBfsArgIter(origin)
        it.add_children(origin)
        # pop the origin node, since it is the origin of
        # the region and does not need to be considered for addition
        assert it.next()
        return it

    def next(self) -> Optional[Node]:
        ret = self._cur
        if not self._queue:
            self._cur = None
        else:
            self._cur = self._queue.popleft()
        return ret

    def peek(self) -> Optional[Node]:
        return self._cur

    def add_children(self, node: Node) -> None:
        flat_args = _get_flat_args_unique(node, {})
        for arg in flat_args:
            if isinstance(arg, Node):
                self._append(arg)

    def _append(self, arg: Node) -> None:
        if self._cur is None:
            self._cur = arg
        else:
            self._queue.append(arg)

    def __str__(self) -> str:
        return f"BackwardBfsArgIter(cur={self._cur}, queue={self._queue})"


class GraphRegionTracker:
    """
    GraphRegionTracker tracks each node added to the output graph and generates a key based on the source location,
    instruction pointer, input shapes, and global state at the time the node is inserted into the graph. Nodes with
    the same key are grouped together in a list of identical nodes (the value of node_to_duplicates).

    hash_to_duplicates: Dict[str, IdenticalNodes] - A dictionary mapping the key to a list of identical nodes
    node_to_duplicates: Dict[Node, IdenticalNodes] - A dictionary mapping a node to the list of identical nodes it belongs to
    input_pickler: InputPickler - An instance of InputPickler used to generate a node hash
    """

    def __init__(self) -> None:
        self.hash_to_duplicates: dict[str, IdenticalNodes] = defaultdict(list)
        self.node_to_duplicates: dict[Node, IdenticalNodes] = {}
        # Note: position is in flattened args/kwargs list
        self.node_to_mutated_arg_positions: dict[Node, OrderedSet[int]] = {}
        self.input_pickler = InputPickler()

    def _hash_node(
        self, filename: str, lineno: int, instruction_pointer: Optional[int], node: Node
    ) -> str:
        from torch._inductor.codecache import sha256_hash

        key = (
            get_global_state_key(),
            filename,
            lineno,
            instruction_pointer,
            _normalize_args(node),
        )
        return sha256_hash(self.input_pickler.dumps(key))

    def _is_identical(self, n0: Node, n1: Node) -> bool:
        return (
            n0 in self.node_to_duplicates
            and n1 in self.node_to_duplicates
            and self.node_to_duplicates[n0] is self.node_to_duplicates[n1]
            and n0 is not n1
        )

    def track_node(self, tx: InstructionTranslatorBase, node: Node) -> None:
        """
        The main entry point for tracking a node. This function will hash the node argument and group
        nodes with the same hash together. It updates the hash_to_duplicates and node_to_duplicates dictionaries
        to track the new node.
        """
        try:
            if (
                node not in self.node_to_duplicates
            ):  # don't allow nodes to be added twice
                duplicates = self.hash_to_duplicates[
                    self._hash_node(
                        tx.f_code.co_filename, tx.lineno, tx.instruction_pointer, node
                    )
                ]
                duplicates.append(node)
                self.node_to_duplicates[node] = duplicates
        except NodeHashException as e:
            log.debug("Unable to hash node %s with exception %s", node, e)  # noqa: G200

    def track_node_mutations(
        self,
        node: Node,
        flat_args_kwargs: list[Any],
        id_to_initial_version: dict[int, int],
    ) -> None:
        """
        This function tracks which argument positions are mutated by the given node. Subgraph HOP does not support
        input mutations today so we will skip regions which have inputs that are mutated.
        """
        mutated_arg_positions = OrderedSet[int]()
        for i, arg in enumerate(flat_args_kwargs):
            val_id = id(arg)
            if (
                val_id in id_to_initial_version
                and id_to_initial_version[val_id] != arg._version
            ):
                mutated_arg_positions.add(i)

        if mutated_arg_positions:
            self.node_to_mutated_arg_positions[node] = mutated_arg_positions

    def add_node_mutation(
        self,
        node: Node,
        arg_pos: int,
    ) -> None:
        if node in self.node_to_mutated_arg_positions:
            self.node_to_mutated_arg_positions[node].add(arg_pos)
        else:
            self.node_to_mutated_arg_positions[node] = OrderedSet([arg_pos])

    def get_identical_regions(self, graph: torch.fx.Graph) -> list[list[Region]]:
        """
        This function is responsible for extracting the largest regions of identical nodes from the given graph.
        **Note**: This function assumes the nodes that have been tracked with track_node are in the provided graph argument.

        The algorithm proceeds as follows:
        The nodes tracked via track_node above are organized into region groups. The initial region groups look like this:
        [[IdenticalNode1], [IdenticalNode2], [IdenticalNode3]] and each sublist is called a region. For each region group
        (starting at the topologically latest region group), the inner regions are gradually expanded one node at time from
        the flattened args and kwargs of the node in each region provided that for all regions in the group, the nodes being
        added are also identical (ie have the same key computed by track_node). This is checked by verifying that the two
        nodes have the same identical node list in node_to_duplicates.
        """
        topological_ranking = {node: i for i, node in enumerate(graph.nodes)}
        region_groups_with_rank = []
        # needed to detect if replacing a region will create cycles
        node_to_recursive_ancestors = _populate_recursive_ancestor_map(graph)

        # Create region groups; a region group is a group
        # of regions that are all identical. In this initial state
        # each region in the group is a single node, and we discard
        # groups that are only a single region.
        # We track the topological ranking to start with groups later in the graph
        # the reason for this is that we will necessarily create the largest groups first.
        for group in self.hash_to_duplicates.values():
            if len(group) > 1:
                region_group = []
                min_rank = math.inf

                for node in group:
                    # some nodes aren't in the topo ranking?
                    if node in topological_ranking:
                        min_rank = min(min_rank, topological_ranking[node])
                        region_group.append([node])

                if len(region_group) > 1:
                    region_groups_with_rank.append((region_group, min_rank))

        region_groups_with_rank.sort(key=lambda rg: -rg[1])
        region_groups = [rg for rg, _ in region_groups_with_rank]

        # We start from regions later in the graph and expand them earlier
        # as a result, we will create the largest regions first and they won't
        # overlap.
        seen_nodes: set[Node] = set()
        for region_group in region_groups:
            fully_expand_region_group(
                region_group,
                seen_nodes,
                node_to_recursive_ancestors,
                self._is_identical,
            )
            # sort topologically
            # we need to handle edge cases where some nodes have no dependencies
            # so first we map each node to its ranking,
            ref_region = region_group[0]
            index_to_rank = {
                index: topological_ranking[n] for index, n in enumerate(ref_region)
            }
            _sort_with_ref_region(index_to_rank, region_group)

        return [
            region_group for region_group in region_groups if len(region_group[0]) > 1
        ]

    def __str__(self) -> str:
        return f"GraphRegionTracker(hash_to_duplicates={self.hash_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"


class RegionWrapper:
    """Holds state for regions e.g. ancestors and new candidate nodes for consideration"""

    def __init__(
        self, region: Region, node_to_recursive_ancestors: dict[Node, set[Node]]
    ) -> None:
        assert len(region) == 1, "all regions should start with one node"
        node = region[0]
        self.node_to_recursive_ancestors = node_to_recursive_ancestors
        self.iter = BackwardBfsArgIter.create(node)
        self.nodes_unique = OrderedSet([node])
        self.ancestors = set(node_to_recursive_ancestors[node])
        self.region = region

    def next_candidate(self) -> Optional[Node]:
        return self.iter.next()

    def will_inclusion_create_cycle(self, node: Node) -> bool:
        external_users = [user for user in node.users if user not in self.nodes_unique]
        for user in external_users:
            if user in self.ancestors:
                return True

        return False

    def add(self, node: Node) -> None:
        self.nodes_unique.add(node)
        self.region.append(node)
        self.iter.add_children(node)
        self.ancestors.update(self.node_to_recursive_ancestors[node])


def fully_expand_region_group(
    regions: list[Region],
    seen_nodes: set[Node],
    node_to_recursive_ancestors: dict[Node, set[Node]],
    is_identical_fn: Callable[[Node, Node], bool],
) -> None:
    debug_log("--------------------------------------------------")
    debug_log("expanding new region group: %s", regions)

    # All regions should start with 1 node
    assert all(len(region) == 1 for region in regions)
    region_wrappers = [
        RegionWrapper(region, node_to_recursive_ancestors) for region in regions
    ]

    nodes_to_add = OrderedSet[Node]()
    current_node = region_wrappers[0].next_candidate()

    # No children
    if current_node is None:
        return

    # Loop incrementally adding new nodes to each region
    # regions are only expanded if the node to add is valid
    # for ALL regions
    while current_node:
        add_to_all_regions = not region_wrappers[0].will_inclusion_create_cycle(
            current_node
        )
        nodes_to_add.clear()
        nodes_to_add.add(current_node)
        for region_wrapper in region_wrappers[1:]:
            candidate = region_wrapper.next_candidate()

            debug_log("--------------------")
            debug_log(
                "considering candidate: %s, cur_node: %s", candidate, current_node
            )

            if not candidate or not add_to_all_regions:
                add_to_all_regions = False
                continue

            debug_log(
                "candidate in previously claimed nodes?: %s", candidate in seen_nodes
            )
            debug_log("is_identical: %s", is_identical_fn(candidate, current_node))

            add_to_all_regions &= (
                candidate not in seen_nodes
                and candidate not in nodes_to_add
                and candidate.op != "placeholder"
                and candidate.op != "get_attr"
                and is_identical_fn(candidate, current_node)
                and not region_wrapper.will_inclusion_create_cycle(candidate)
            )
            nodes_to_add.add(candidate)

            debug_log(f"add_to_all_regions: {add_to_all_regions}")
            debug_log("--------------------")

        if add_to_all_regions:
            assert len(region_wrappers) == len(nodes_to_add), (
                "Number of nodes to add must equal the number of regions"
            )
            for region_wrapper, node in zip(region_wrappers, nodes_to_add):
                region_wrapper.add(node)
                debug_log("adding %s's children", node)
                debug_log("%s %s", node.args, list(node.kwargs.items()))
                seen_nodes.add(node)

        current_node = region_wrappers[0].next_candidate()

    # Ensure regions are sorted in topological order
    for region in regions:
        region.reverse()

    debug_log("end expand new region group: %s", regions)
    debug_log("--------------------------------------------------")


def _populate_recursive_ancestor_map(graph: torch.fx.Graph) -> dict[Node, set[Node]]:
    node_to_recursive_ancestors: dict[Node, set[Node]] = {}
    for node in graph.nodes:
        node_to_recursive_ancestors[node] = set()
    for node in graph.nodes:
        all_args = _get_flat_args_unique(node, {})
        for arg in all_args:
            if isinstance(arg, Node):
                node_to_recursive_ancestors[node].update(
                    node_to_recursive_ancestors[arg]
                )
                node_to_recursive_ancestors[node].add(arg)
    return node_to_recursive_ancestors
