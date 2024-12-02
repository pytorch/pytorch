import copyreg
import io
import math
import pickle
from collections import defaultdict, deque
from dataclasses import fields
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._pytree import tree_flatten


T = TypeVar("T")


if TYPE_CHECKING:
    from .symbolic_convert import InstructionTranslatorBase


Node = torch.fx.Node
Region = List[Node]
IdenticalNodes = List[Node]
GlobalStateKey = Tuple[bool, bool, int, bool, bool, torch.dtype, bool, bool, bool]


def _extract_tensor_metadata_for_node_hash(
    x: torch.Tensor,
) -> Tuple[Callable[[T], T], Tuple[Any, ...]]:
    from torch._inductor.codecache import _ident, extract_tensor_metadata_for_cache_key

    out = []
    metadata = extract_tensor_metadata_for_cache_key(x)
    for field in fields(metadata):
        out.append(getattr(metadata, field.name))

    return (_ident, tuple(out))


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
) -> Tuple[Tuple[str, ...], Tuple[Optional[Any], ...]]:
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

    def track_node(self, tx: "InstructionTranslatorBase", node: Node) -> None:
        duplicates = self.loc_to_duplicates[
            self._hash_node(
                tx.f_code.co_filename, tx.lineno, tx.instruction_pointer, node
            )
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
    # Loop incrementally adding new nodes to each region
    # regions are only expanded if the node to add is valid
    # for ALL regions
    while current_node:
        add_node = True
        nodes_to_add.clear()
        nodes_to_add.append(current_node)
        nodes_to_add_set = set(nodes_to_add)
        for region_it in region_iters[1:]:
            arg_name, node = region_it.next()

            if node:
                add_node &= (
                    current_arg_name == arg_name
                    and node not in seen_nodes
                    and node not in nodes_to_add_set
                    and is_identical_fn(node, current_node)
                )
                nodes_to_add.append(node)
                nodes_to_add_set.add(node)
            else:
                add_node = False

        if add_node:
            for region, region_it, node in zip(regions, region_iters, nodes_to_add):
                region.append(node)
                region_it.add_children(node)
                seen_nodes.add(node)

        current_arg_name, current_node = region_iters[0].next()

    # Ensure regions are sorted in topological order
    for region in regions:
        region.reverse()
