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

import copyreg
import io
import logging
import math
import pickle
from collections import defaultdict, deque
from dataclasses import fields
from typing import Any, Callable, Optional, TYPE_CHECKING, TypeVar

import torch._logging
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._pytree import tree_flatten


T = TypeVar("T")


if TYPE_CHECKING:
    from .symbolic_convert import InstructionTranslatorBase


Node = torch.fx.Node
Region = list[Node]
IdenticalNodes = list[Node]
GlobalStateKey = tuple[bool, bool, int, bool, bool, torch.dtype, bool, bool, bool, bool]

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


def _extract_tensor_arg(arg: Any) -> Any:
    if isinstance(arg, Node):
        return arg.meta.get("example_value")
    else:
        return None


def _normalize_args(
    node: Node,
) -> tuple[tuple[str, ...], tuple[Optional[Any], ...]]:
    flat_args, _ = tree_flatten(node.args)
    sorted_kwargs = sorted(node.kwargs.items(), key=lambda x: x[0])
    sorted_keys = tuple(sorted(node.kwargs.keys()))
    flat_kwargs, _ = tree_flatten(sorted_kwargs)
    all_args = flat_args + flat_kwargs
    return (sorted_keys, tuple(_extract_tensor_arg(arg) for arg in all_args))


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
    def create(origin: Node) -> "BackwardBfsArgIter":
        it = BackwardBfsArgIter(origin)
        it.add_children(origin)
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
        arg: Any
        flat_args, _ = tree_flatten(node.args)
        for arg in flat_args:
            if isinstance(arg, Node):
                self._append(arg)

        flat_kwargs, _ = tree_flatten(node.kwargs)
        for kwarg in flat_kwargs:
            if isinstance(kwarg, Node):
                self._append(kwarg)

    def _append(self, arg: Node) -> None:
        if self._cur is None:
            self._cur = arg
        else:
            self._queue.append(arg)


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

    def track_node(self, tx: "InstructionTranslatorBase", node: Node) -> None:
        """
        The main entry point for tracking a node. This function will hash the node argument and group
        nodes with the same hash together. It updates the hash_to_duplicates and node_to_duplicates dictionaries
        to track the new node.
        """
        try:
            duplicates = self.hash_to_duplicates[
                self._hash_node(
                    tx.f_code.co_filename, tx.lineno, tx.instruction_pointer, node
                )
            ]
            duplicates.append(node)
            self.node_to_duplicates[node] = duplicates
        except NodeHashException as e:
            log.debug("Unable to hash node %s with exception %s", node, e)

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
            fully_expand_region_group(region_group, seen_nodes, self._is_identical)
            # sort topologically
            for region in region_group:
                region.sort(key=lambda n: topological_ranking[n])

        return [
            region_group for region_group in region_groups if len(region_group[0]) > 1
        ]

    def __str__(self) -> str:
        return f"GraphRegionTracker(hash_to_duplicates={self.hash_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"


def fully_expand_region_group(
    regions: list[Region],
    seen_nodes: set[Node],
    is_identical_fn: Callable[[Node, Node], bool],
) -> None:
    debug_log("--------------------------------------------------")
    debug_log("expanding new region group: %s", regions)

    # All regions should start with 1 node
    assert all(len(region) == 1 for region in regions)
    region_iters = []
    for region in regions:
        (origin,) = region  # Only works for 1 element sets
        region_iters.append(BackwardBfsArgIter.create(origin))

    nodes_to_add: list[Node] = []

    # we already have the origin node in each region
    for region_it in region_iters:
        node = region_it.next()
        assert node
        region_it.add_children(node)

    current_node = region_iters[0].next()
    assert current_node is not None
    # Loop incrementally adding new nodes to each region
    # regions are only expanded if the node to add is valid
    # for ALL regions
    while current_node:
        add_node = True
        nodes_to_add.clear()
        nodes_to_add.append(current_node)
        nodes_to_add_set = set(nodes_to_add)
        for region_it in region_iters[1:]:
            node = region_it.next()

            debug_log("--------------------")
            debug_log("considering adding: %s, cur_node: %s", node, current_node)
            debug_log("previously claimed nodes: %s", node in seen_nodes)
            debug_log("%s", seen_nodes)
            if node:
                debug_log("is_identical: %s", is_identical_fn(node, current_node))
                add_node &= (
                    node not in seen_nodes
                    and node not in nodes_to_add_set
                    and node.op != "placeholder"
                    and is_identical_fn(node, current_node)
                )
                nodes_to_add.append(node)
                nodes_to_add_set.add(node)
            else:
                add_node = False

            debug_log("--------------------")

        if add_node:
            for region, region_it, node in zip(regions, region_iters, nodes_to_add):
                region.append(node)
                debug_log("adding %s's children", node)
                debug_log("%s %s", node.args, list(node.kwargs.items()))
                region_it.add_children(node)
                seen_nodes.add(node)

        current_node = region_iters[0].next()

    # Ensure regions are sorted in topological order
    for region in regions:
        region.reverse()

    debug_log("end expand new region group: %s", regions)
    debug_log("--------------------------------------------------")
