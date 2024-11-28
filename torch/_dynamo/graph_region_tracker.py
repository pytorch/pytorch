import functools
import math
import pickle
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union

import torch.fx
from torch._inductor.codecache import extract_tensor_metadata_for_cache_key, sha256_hash
from torch._subclasses.fake_tensor import TensorMetadata
from torch.utils._pytree import tree_flatten


Node = torch.fx.Node
Region = List[Node]
IdenticalNodes = List[Node]
GlobalStateKey = Tuple[bool, bool, int, bool, bool, torch.dtype, bool, bool, bool]


def get_metadata(node: Node) -> Optional[Union[str, TensorMetadata]]:
    if isinstance(node, torch.fx.Node):
        value = node.meta.get("example_value", None)
        if isinstance(value, torch.Tensor):
            return extract_tensor_metadata_for_cache_key(value)
        elif isinstance(value, torch.SymInt):
            return str(value)
    return None


@functools.lru_cache(128)
def _extract_node_metadata(
    node: Node,
) -> Tuple[Tuple[str, ...], Tuple[Optional[Union[str, TensorMetadata]], ...]]:
    flat_args, _ = tree_flatten(node.args)
    sorted_kwargs = sorted(node.kwargs.items(), key=lambda x: x[0])
    sorted_keys = tuple(sorted(node.kwargs.keys()))
    flat_kwargs, _ = tree_flatten(sorted_kwargs)
    all_args = flat_args + flat_kwargs
    return (sorted_keys, tuple(get_metadata(arg) for arg in all_args))


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
    )


# This is typical BFS with the caveat
# that a node's children need to be explicitly
# added with the add_children() method
# The flow is yield a node and check if it's valid for all regions
# if not valid, discard and continue onto the next node
class BfsRegionIter:
    def __init__(self, origin: Node) -> None:
        self._cur: Tuple[Optional[str], Optional[Node]] = (None, origin)
        self._queue: Deque[Tuple[Optional[str], Optional[Node]]] = deque()

    @staticmethod
    def create(origin: Node) -> "BfsRegionIter":
        it = BfsRegionIter(origin)
        it.add_children(origin)
        return it

    def next(self) -> Tuple[Optional[str], Optional[Node]]:
        ret = self._cur
        if not self._queue:
            self._cur = (None, None)
        else:
            self._cur = self._queue.popleft()
        return ret

    def peek(self) -> Tuple[Optional[str], Optional[Node]]:
        return self._cur

    def add_children(self, node: Node) -> None:
        arg: Any
        for arg in node.args:
            if isinstance(arg, Node):
                self._append((None, arg))

        key: str
        kwarg: Any
        for key, kwarg in node.kwargs.items():
            if isinstance(kwarg, Node):
                self._append((key, kwarg))

    def _append(self, name_arg: Tuple[Optional[str], Node]) -> None:
        if self._cur == (None, None):
            self._cur = name_arg
        else:
            self._queue.append(name_arg)


class GraphRegionTracker:
    def __init__(self) -> None:
        self.loc_to_duplicates: Dict[str, IdenticalNodes] = defaultdict(list)
        self.node_to_duplicates: Dict[Node, IdenticalNodes] = {}
        self.node_to_global_state_hash: Dict[Node, int] = {}

    @staticmethod
    def _get_loc_str(filename: str, lineno: int, instruction_pointer: int) -> str:
        return f"{filename}:{lineno}:{instruction_pointer}"

    @staticmethod
    def _get_key(
        filename: str, lineno: int, instruction_pointer: int, node: Node
    ) -> str:
        return sha256_hash(
            pickle.dumps(
                (
                    get_global_state_key(),
                    GraphRegionTracker._get_loc_str(
                        filename, lineno, instruction_pointer
                    ),
                    _extract_node_metadata(node),
                )
            )
        )

    def track_node(
        self, filename: str, lineno: int, instruction_pointer: int, node: Node
    ) -> None:
        duplicates = self.loc_to_duplicates[
            GraphRegionTracker._get_key(filename, lineno, instruction_pointer, node)
        ]
        duplicates.append(node)
        self.node_to_duplicates[node] = duplicates

    def is_identical(self, n0: Node, n1: Node) -> bool:
        return (
            n0 in self.node_to_duplicates
            and n1 in self.node_to_duplicates
            and self.node_to_duplicates[n0] == self.node_to_duplicates[n1]
            and n0 is not n1
        )

    def get_identical_regions(self, graph: torch.fx.Graph) -> List[List[Region]]:
        topological_ranking = {node: i for i, node in enumerate(graph.nodes)}
        group_ranking = {}
        region_groups = []

        # Create region groups; a region group is a group
        # of regions that are all identical. In this initial state
        # each region in the group is a single node, and we discard
        # groups that are only a single region.
        # We track the topological ranking to start with groups later in the graph
        # the reason for this is that we will necessarily create the largest groups first.
        for group in self.loc_to_duplicates.values():
            if len(group) > 1:
                region_group = []
                min_rank = math.inf
                for node in group:
                    min_rank = min(min_rank, topological_ranking[node])
                    region_group.append([node])

                region_groups.append(region_group)
                group_ranking[id(region_group)] = min_rank

        region_groups.sort(key=lambda g: -group_ranking[id(g)])

        # We start from regions later in the graph and expand them earlier
        # as a result, we will create the largest regions first and they won't
        # overlap.
        seen_nodes: Set[Node] = set()
        for region_group in region_groups:
            fully_expand_region_group(region_group, seen_nodes, self.is_identical)

        return [
            region_group for region_group in region_groups if len(region_group[0]) > 1
        ]

    def __str__(self) -> str:
        return f"GraphRegionTracker(loc_to_duplicates={self.loc_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"


def fully_expand_region_group(
    regions: List[Region],
    seen_nodes: Set[Node],
    is_identical_fn: Callable[[Node, Node], bool],
) -> None:
    # All regions should start with 1 node
    assert all(len(region) == 1 for region in regions)
    region_iters = []
    for region in regions:
        (origin,) = region  # Only works for 1 element sets
        region_iters.append(BfsRegionIter.create(origin))

    nodes_to_add: List[Node] = []

    # we already have the origin node in each region
    for region_it in region_iters:
        _, node = region_it.next()
        assert node
        region_it.add_children(node)

    # arg_name is set for kwargs, None for args
    current_arg_name, current_node = region_iters[0].next()
    assert current_node is not None
    seen_nodes.add(current_node)
    # Loop incrementally adding new nodes to each region
    # regions are only expanded if the node to add is valid
    # for ALL regions
    while current_node:
        add_node = True
        nodes_to_add.clear()
        nodes_to_add.append(current_node)
        for region_it in region_iters[1:]:
            arg_name, node = region_it.next()

            if node:
                add_node &= (
                    current_arg_name == arg_name
                    and node not in seen_nodes
                    and is_identical_fn(node, current_node)
                )
                nodes_to_add.append(node)
                seen_nodes.add(node)
            else:
                add_node = False

        if add_node:
            for region, region_it, node in zip(regions, region_iters, nodes_to_add):
                region.append(node)
                region_it.add_children(node)

        current_arg_name, current_node = region_iters[0].next()

    # Ensure regions are sorted in topological order
    for region in regions:
        region.reverse()
