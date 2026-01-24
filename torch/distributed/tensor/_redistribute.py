# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import dataclasses
import itertools
import logging
import weakref
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from typing import cast, NamedTuple

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed.tensor._collective_utils import one_step_redistribute_cost
from torch.distributed.tensor._dtensor_spec import (
    DTensorSpec,
    ShardOrder,
    ShardOrderEntry,
    TensorMeta,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._debug_mode import get_active_debug_mode


logger = logging.getLogger(__name__)

# Global configuration flag to control the redistribution planning strategy.
# When True, forces the graph-based algorithm using Dijkstra's shortest path.
# When False, prefers the greedy algorithm for faster planning. Uses the graph-based algorithm
# only when necessary to support strided-shard redistribution
_FORCE_MIN_COST_REDISTRIBUTION_PLAN: bool | None = None


@contextlib.contextmanager
def use_min_cost_redistribution_plan(enabled: bool = True):
    """
    Context manager to control the redistribution planning strategy for DTensor operations.

    This context manager allows you to choose between two algorithms for computing the
    sequence of collective operations needed to redistribute a DTensor from one placement
    to another:

    - **Graph-based**: Uses Dijkstra's algorithm to find the minimum-cost path
      through all possible placement transformations. This approach considers the global
      cost of all collective operations and finds the optimal sequence. Best for complex
      redistribution patterns where reducing communication cost and memory overhead is critical.

    - **Greedy**: Uses a heuristic approach that makes locally optimal choices
      at each step. This is faster to compute but may not produce the globally optimal
      transformation sequence. Best for simple redistribution patterns or when planning
      speed is more important than optimal communication.

    **Default Behavior (without this context manager):**

    When this context manager is NOT used, the algorithm selection follows this priority:

    1. **Non-default shard orders**
       → Always use graph-based algorithm (required for correctness)

    2. **Explicit `use_graph_based_transform` parameter** to `_gen_transform_infos_non_cached`
       → Use the specified algorithm (True = graph-based, False = greedy)

    3. **No explicit parameter** (default case)
       → Use greedy algorithm for faster planning

    **Behavior with this context manager:**

    This context manager overrides the default selection by setting the global flag
    `_FORCE_MIN_COST_REDISTRIBUTION_PLAN`, which takes precedence over the explicit
    `use_graph_based_transform` parameter (but not over non-default shard order requirements).

    **Cache Considerations:**

    The redistribution planner caches transform info for performance via the `@cache`
    decorator on `_gen_transform_infos`. If you need to change the algorithm selection
    for the same input specs, clear the cache using `_gen_transform_infos.cache_clear()`
    to ensure the new setting takes effect and doesn't reuse cached results from a
    previous run.

    Args:
        enabled (bool): If True, forces the use of the graph-based algorithm.
                       If False, forces the use of the greedy algorithm.
                       Default: True
    """
    global _FORCE_MIN_COST_REDISTRIBUTION_PLAN

    old_value = _FORCE_MIN_COST_REDISTRIBUTION_PLAN
    _FORCE_MIN_COST_REDISTRIBUTION_PLAN = enabled
    try:
        yield
    finally:
        _FORCE_MIN_COST_REDISTRIBUTION_PLAN = old_value


class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    # logical_shape on this mesh dimension
    logical_shape: list[int]

    def __post_init__(self):
        assert self.mesh_dim >= 0
        assert self.src_dst_placements[0] != self.src_dst_placements[1], (
            "TransformInfo should only be created if it is an op with some effect, not a no-op"
        )


# Global cache for DTensorRedistributePlanner instances
_planner_cache: dict[
    tuple[weakref.ReferenceType[DeviceMesh], TensorMeta],
    "DTensorRedistributePlanner",
] = {}


def get_redistribute_planner(
    device_mesh: DeviceMesh,
    dtensor_meta: TensorMeta,
) -> "DTensorRedistributePlanner":
    """
    Factory function to get or create a DTensorRedistributePlanner instance.
    This function provides transparent caching of planner instances based on
    device mesh and dtensor meta. Multiple calls with the same parameters
    will return the same cached instance for better performance.
    Args:
        device_mesh: The device mesh for the planner
        dtensor_meta: TensorMeta of the DTensor to redistribute
    Returns:
        A DTensorRedistributePlanner instance (potentially cached)
    """
    if _are_we_tracing():
        return DTensorRedistributePlanner(device_mesh, dtensor_meta)

    cache_key = (weakref.ref(device_mesh), dtensor_meta)
    if cache_key not in _planner_cache:
        planner = DTensorRedistributePlanner(device_mesh, dtensor_meta)
        _planner_cache[cache_key] = planner

    return _planner_cache[cache_key]


def clear_redistribute_planner_cache() -> None:
    """Clear the cache of DTensorRedistributePlanner instances."""
    _planner_cache.clear()


class DTensorRedistributePlanner:
    """
    This class is used to plan the collective calls to transform the local shard
    of the DTensor from its current spec to the target spec.
    Suppose there are N tensor dimensions and M mesh dimensions, the total
    possible state size will be (N+2)*M*M!.
    Note: Use get_redistribute_planner() factory function instead of direct
    instantiation for automatic caching.
    """

    @dataclasses.dataclass(frozen=True, slots=True)
    class DistState:
        placements: tuple[Placement, ...]
        tensor_dim_to_mesh_dim: ShardOrder
        _hash: int | None = dataclasses.field(
            default=None, init=False, repr=False, compare=False
        )

        def __str__(self):
            return DTensorSpec.format_shard_order_str(
                self.placements,
                self.tensor_dim_to_mesh_dim,
            )

        def __repr__(self):
            return self.__str__()

        def __post_init__(self):
            # precompute hash after all attributes are set
            object.__setattr__(
                self,
                "_hash",
                self._compute_hash(),
            )

        def __hash__(self) -> int:
            return self._hash if self._hash is not None else self._compute_hash()

        def _compute_hash(self) -> int:
            return hash(
                (
                    self.placements,
                    self.tensor_dim_to_mesh_dim,
                )
            )

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, DTensorRedistributePlanner.DistState):
                return False
            if self._hash != other._hash:
                return False
            return (
                self.placements,
                self.tensor_dim_to_mesh_dim,
            ) == (
                other.placements,
                other.tensor_dim_to_mesh_dim,
            )

    def _to_tuple(self, x):
        """Convert a nested list structure to a nested tuple structure."""
        if isinstance(x, list | tuple):
            return tuple(self._to_tuple(item) for item in x)
        return x

    @staticmethod
    def _dict_to_ShardOrder(x: dict[int, list[int]]) -> ShardOrder:
        """Convert dict to ShardOrder"""
        return tuple(
            ShardOrderEntry(tensor_dim=key, mesh_dims=tuple(value))
            for key, value in sorted(x.items())
            if value
        )

    @staticmethod
    def _ShardOrder_to_dict(x: ShardOrder) -> dict[int, list[int]]:
        """Convert ShardOrder to dict with tensor dim as key"""
        tensor_mesh_dim_dict = defaultdict(list)
        for entry in x:
            tensor_mesh_dim_dict[entry.tensor_dim] = list(entry.mesh_dims)
        return tensor_mesh_dim_dict

    @staticmethod
    def stringify_transform_infos(
        mesh: DeviceMesh,
        transform_infos: Sequence[_TransformInfo],
        src_placement: tuple[Placement, ...],
        src_shard_order: ShardOrder | None = None,
    ) -> str:
        """
        Generate a string representation of the sequence of state transitions
        (placements and shard orders) as described by the given transform_info.

        Args:
            mesh: The DeviceMesh used for the redistribution.
            transform_infos: A sequence of _TransformInfo objects describing each
                transformation step.
            src_placement: The initial tuple of Placement objects.
            src_shard_order: (Optional) The initial ShardOrder representing
                the mapping of tensor dimensions to mesh dimensions. If None,
                the default shard order is computed from src_placement and mesh.

        Returns:
            A string showing the sequence of DistState transitions, separated by '->'.
        """
        assert len(src_placement) == mesh.ndim
        if src_shard_order is None:
            src_shard_order = DTensorSpec.compute_default_shard_order(src_placement)
        cur_placement = list(src_placement)
        shard_order_dict = DTensorRedistributePlanner._ShardOrder_to_dict(
            src_shard_order
        )
        cur_state = DTensorRedistributePlanner.DistState(
            tuple(cur_placement), src_shard_order
        )
        state_list = [
            cur_state,
        ]
        for transform_info in transform_infos:
            src_dim_placement, dst_dim_placement = transform_info.src_dst_placements
            if src_dim_placement.is_shard():
                src_dim = src_dim_placement.dim  # type: ignore[attr-defined]
                assert (
                    src_dim in shard_order_dict and len(shard_order_dict[src_dim]) > 0
                )
                shard_order_dict[src_dim].pop()
            if dst_dim_placement.is_shard():
                dst_dim = dst_dim_placement.dim  # type: ignore[attr-defined]
                if dst_dim not in shard_order_dict:
                    shard_order_dict[dst_dim] = []
                shard_order_dict[dst_dim].append(transform_info.mesh_dim)
            cur_placement[transform_info.mesh_dim] = dst_dim_placement
            new_state = DTensorRedistributePlanner.DistState(
                tuple(cur_placement),
                DTensorRedistributePlanner._dict_to_ShardOrder(shard_order_dict),
            )
            state_list.append(new_state)
        return "->".join([str(s) for s in state_list])

    def __init__(
        self,
        device_mesh: DeviceMesh,
        dtensor_meta: TensorMeta,
    ) -> None:
        """
        Initialize DTensorRedistributePlanner.

        Args:
            device_mesh: The device mesh for this planner
            dtensor_meta: TensorMeta of the DTensor to redistribute
        """
        self.device_mesh = device_mesh
        assert device_mesh._is_current_rank_part_of_mesh()
        assert dtensor_meta is not None
        self.dtensor_meta = dtensor_meta
        self.tensor_dimension = len(dtensor_meta.shape)
        self.strided_shard_placements_in_target: set[_StridedShard] = set()
        self.setup_cost_callbacks()

    def setup_cost_callbacks(
        self,
    ) -> None:
        """
        Set up the cost function for different collective operations.
        Uses communication time estimation based on actual tensor sizes and
        mesh topology for accurate cost modeling.
        """

        def state_to_spec(
            state: DTensorRedistributePlanner.DistState,
        ) -> DTensorSpec:
            return DTensorSpec(
                mesh=self.device_mesh,
                placements=state.placements,
                tensor_meta=self.dtensor_meta,
                shard_order=state.tensor_dim_to_mesh_dim,
            )

        def cost_function(src_state, dst_state):
            return one_step_redistribute_cost(
                state_to_spec(src_state), state_to_spec(dst_state)
            )

        self.cost_function = cost_function

    def get_next_state(
        self,
        placements: tuple[Placement, ...],
        tensor_mesh_dim_tuple: ShardOrder,
    ) -> dict["DTensorRedistributePlanner.DistState", float]:
        # We map tensor dimensions to device mesh axes, similar to JAX-style
        # sharding representation. Notation:
        # S(<tensor_dim>)[<list_of_device_dims>] means tensor dimension
        # <tensor_dim> is sharded on the listed device mesh axes, where
        # <list_of_device_dims> is sorted by device order.
        #
        # To generalize to arbitrary dimensionality, we use the following notation:
        #   S(a)[x, ...]   : tensor dimension 'a' is sharded on device mesh axes x, ... (variadic, possibly empty)
        #   SS(a)[x, ...]  : _StridedShard on tensor dimension 'a' on device mesh axes x, ... (variadic, possibly empty)
        #   R[...]         : replicated on the listed device mesh axes (possibly empty)
        #   P[...]         : partial on the listed device mesh axes (possibly empty)
        # The ellipsis '...' denotes a variadic wildcard, i.e., zero or more device mesh axes.
        #
        # Below are possible transitions from one sharding state to another.
        # We use `S` for Shard, `SS` for _StridedShard, `R` for Replicate, and `P` for Partial.
        #
        # Case 1. Shard(a) -> Shard(b), use all-to-all (a2a), applies to:
        #   S(a)[..., x] -> S(b)[..., x]
        #   or
        #   S(a)[..., x, y]S(b)[..., z, k] -> S(a)[..., x]S(b)[..., z, k, y]
        #   where device order of 'y' > device order of 'z' and 'k'
        #
        # Case 2. Shard() -> Replicate(), use all-gather, applies to:
        #   S(a)[..., x, y, z] -> S(a)[..., x, y]
        #
        # Case 3. Partial() -> Replicate(), use all-reduce, applies to:
        #   P[..., x, y] -> P[..., y] or P[..., x]
        #   Note: this case can be disabled because all-reduce technically is not
        #   a primitive since it combines a reduce-scatter + all-gather.
        #
        # Case 4. Replicate() -> Shard(), use chunk, applies to:
        #   S(a)[..., z] -> S(a)[..., z, y] (`a` can be any tensor dim). Note that
        #   'y' must be after 'z'.
        #
        # Case 5. Partial() -> Shard(), use reduce-scatter, applies to:
        #  P[..., x, y] -> P[..., x]S(a)[..., y] or P[..., x, y] -> P[..., y]S(a)[..., x]
        #
        # Case 6. Replicate() -> Partial(), local math op, applies to:
        #   R* -> P[..., x]
        #
        # (TODO) Case 7. _StridedShard(a) -> Shard(b), use all-to-all (a2a), applies to:
        #   SS(a)[..., x] -> S(b)[..., x]
        #
        # Case 8. _StridedShard() -> Replicate(), use all-gather, applies to:
        #   SS(a)[..., x, y, z] -> SS(a)[..., x, y]
        #
        # (TODO) Case 9. Shard(a) -> _StridedShard(b), use all-to-all (a2a), applies to:
        #   S(a)[..., x] -> SS(b)[..., x]
        #
        # (TODO) Case 10. Partial() -> _StridedShard(), use reduce-scatter, applies to:
        #   P[..., x, y] -> P[..., x]SS(a)[..., y] or P[..., x, y] -> P[..., y]SS(a)[..., x]
        #
        # Case 11. Replicate() -> _StridedShard(), use chunk, applies to:
        #   R* -> SS(a)[..., x]
        #
        # NB: Regarding `_StridedShard``, we only allow changing `Replicate` into
        # `_StridedShard` with the same tensor dim and split_factor that occurs in the
        # target placement.
        #
        # (TODO) Verify device order impact in Partial placement. We may need to handle
        # device ordering for Partial also.

        # list of [DistState, cost]
        all_next_state: dict[DTensorRedistributePlanner.DistState, float] = {}

        tensor_mesh_dim_dict = DTensorRedistributePlanner._ShardOrder_to_dict(
            tensor_mesh_dim_tuple
        )
        cur_dist_state = self.DistState(
            self._to_tuple(placements),
            tensor_mesh_dim_tuple,
        )
        ######################################################################
        # handle case 1: Shard(a) -> Shard(b)
        # For S(a), S(b), only the last device order of S(a) and S(b) can be a2a
        # interchangeably.

        # convert sparse tuple
        for entry in tensor_mesh_dim_tuple:
            src_tensor_dim = entry.tensor_dim
            src_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim][-1]
            if not isinstance(placements[src_mesh_dim], Shard):
                # skip special case like `_StridedShard`
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                if src_tensor_dim == dst_tensor_dim:
                    continue
                # try move the last sharded device dim from
                # Shard(src_tensor_dim) to Shard(dst_tensor_dim)
                move_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim].pop()
                tensor_mesh_dim_dict[dst_tensor_dim].append(move_mesh_dim)
                new_placements = list(placements)
                new_placements[move_mesh_dim] = Shard(dst_tensor_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                # reset content for next iteration
                tensor_mesh_dim_dict[src_tensor_dim].append(move_mesh_dim)
                tensor_mesh_dim_dict[dst_tensor_dim].pop()
        # TODO(zpcore): support discovering submesh to prevent padding when
        # tensor dim is not divisible by the mesh dim.

        ######################################################################
        # handle case 2: Shard() -> Replicate()
        for entry in tensor_mesh_dim_tuple:
            src_tensor_dim = entry.tensor_dim
            src_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim][-1]
            if not isinstance(placements[src_mesh_dim], Shard):
                # skip special case like `_StridedShard`
                continue
            move_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim].pop()
            new_placements = list(placements)
            new_placements[move_mesh_dim] = Replicate()
            dist_state = self.DistState(
                self._to_tuple(new_placements),
                DTensorRedistributePlanner._dict_to_ShardOrder(tensor_mesh_dim_dict),
            )
            tensor_mesh_dim_dict[src_tensor_dim].append(move_mesh_dim)
            all_next_state[dist_state] = self.cost_function(
                cur_dist_state,
                dist_state,
            )

        ######################################################################
        # handle case 3: Partial() -> Replicate()
        for src_mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Partial):
                continue
            new_placements = list(placements)
            new_placements[src_mesh_dim] = Replicate()
            dist_state = self.DistState(
                self._to_tuple(new_placements), tensor_mesh_dim_tuple
            )
            all_next_state[dist_state] = self.cost_function(
                cur_dist_state,
                dist_state,
            )

        ######################################################################
        # handle case 4: Replicate() -> Shard()
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                # try convert placement[mesh_dim] to Shard(dst_tensor_dim)
                new_placements = list(placements)
                new_placements[mesh_dim] = Shard(dst_tensor_dim)
                tensor_mesh_dim_dict[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                tensor_mesh_dim_dict[dst_tensor_dim].pop()

        ######################################################################
        # handle case 5: Partial() -> Shard()
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Partial):
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                # try convert placement[mesh_dim] to Shard(dst_tensor_dim)
                new_placements = list(placements)
                new_placements[mesh_dim] = Shard(dst_tensor_dim)
                tensor_mesh_dim_dict[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                tensor_mesh_dim_dict[dst_tensor_dim].pop()

        ######################################################################
        # handle case 6: Replicate() -> Partial(), default to partial(sum)
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            new_placements = list(placements)
            new_placements[mesh_dim] = Partial()
            dist_state = self.DistState(
                self._to_tuple(new_placements), tensor_mesh_dim_tuple
            )
            all_next_state[dist_state] = self.cost_function(
                cur_dist_state,
                dist_state,
            )

        # Additional cases handling for _StridedShard

        ######################################################################
        # TODO(zpcore): handle case 7: _StridedShard() -> Shard() on the same dim

        ######################################################################
        # handle case 8: _StridedShard() -> Replicate()
        for entry in tensor_mesh_dim_tuple:
            src_tensor_dim = entry.tensor_dim
            src_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim][-1]
            if not isinstance(placements[src_mesh_dim], _StridedShard):
                continue
            move_mesh_dim = tensor_mesh_dim_dict[src_tensor_dim].pop()
            new_placements = list(placements)
            new_placements[move_mesh_dim] = Replicate()
            dist_state = self.DistState(
                self._to_tuple(new_placements),
                DTensorRedistributePlanner._dict_to_ShardOrder(tensor_mesh_dim_dict),
            )
            tensor_mesh_dim_dict[src_tensor_dim].append(move_mesh_dim)
            all_next_state[dist_state] = self.cost_function(
                cur_dist_state,
                dist_state,
            )

        # Early exit if no StridedShard in target
        if not self.strided_shard_placements_in_target:
            return all_next_state

        ######################################################################
        # TODO(zpcore): handle case 9: Shard() -> _StridedShard()

        ######################################################################
        # TODO(zpcore): handle case 10: Partial() -> _StridedShard()

        ######################################################################
        # handle case 11: Replicate() -> _StridedShard()
        for mesh_dim, placement in enumerate(placements):
            if not isinstance(placement, Replicate):
                continue
            for strided_shard_obj in self.strided_shard_placements_in_target:
                dst_tensor_dim = strided_shard_obj.dim
                # try convert placement[mesh_dim] to strided_shard_obj
                new_placements = list(placements)
                new_placements[mesh_dim] = strided_shard_obj
                tensor_mesh_dim_dict[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    DTensorRedistributePlanner._dict_to_ShardOrder(
                        tensor_mesh_dim_dict
                    ),
                )
                all_next_state[dist_state] = self.cost_function(
                    cur_dist_state,
                    dist_state,
                )
                tensor_mesh_dim_dict[dst_tensor_dim].pop()

        return all_next_state

    # TODO(zpcore): if the dst_state contains special placement like
    # `_MaskPartial`, we will never reach that state. Need to support this case.
    def find_min_cost_path(
        self, src_state: DistState, dst_state: DistState
    ) -> list["DTensorRedistributePlanner.DistState"]:
        """
        Find the min cost path from src_state to dst_state using Dijkstra's
        algorithm.

        Args:
            src_state: The source state
            dst_state: The destination state

        Returns:
            A list of states representing the min cost path from src_state to
            dst_state
        """
        import heapq

        # priority queue (cost, counter, state, path) for Dijkstra's algorithm
        # use counter to break ties and avoid comparing DistState objects
        counter = 0
        pq: list[
            tuple[
                float,
                int,
                DTensorRedistributePlanner.DistState,
                list[DTensorRedistributePlanner.DistState],
            ]
        ] = [(0, counter, src_state, [src_state])]
        visited = set()
        while pq:
            cost, _, current_state, path = heapq.heappop(pq)
            if current_state == dst_state:
                return path
            if current_state in visited:
                continue
            visited.add(current_state)
            # get all possible next states and their costs
            next_states = self.get_next_state(
                current_state.placements, current_state.tensor_dim_to_mesh_dim
            )
            for next_state, transition_cost in next_states.items():
                if next_state not in visited:
                    new_cost = cost + transition_cost
                    new_path = path + [next_state]
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, next_state, new_path))
        raise AssertionError(
            f"No path found from src_state {src_state} to dst_state {dst_state}"
        )

    def get_logical_shape(
        self,
        src_state: "DTensorRedistributePlanner.DistState",
        mesh_dim: int,
        full_tensor_shape: tuple[int, ...],
    ) -> list[int]:
        new_logical_shape = list(full_tensor_shape)
        for entry in src_state.tensor_dim_to_mesh_dim:
            tensor_dim = entry.tensor_dim
            mesh_dims = entry.mesh_dims
            assert len(mesh_dims) > 0
            for mdim in mesh_dims:
                if mdim == mesh_dim:
                    continue
                placement = src_state.placements[mdim]
                if isinstance(placement, Shard):
                    new_size, _ = placement.local_shard_size_and_offset(
                        new_logical_shape[tensor_dim],
                        self.device_mesh.size(mesh_dim=mdim),
                        self.device_mesh._sym_get_coordinate(mdim),
                    )
                elif isinstance(placement, _StridedShard):
                    new_size, _ = placement.local_shard_size_and_offset(
                        new_logical_shape[tensor_dim],
                        self.device_mesh.size(mesh_dim=mdim),
                        self.device_mesh._sym_get_coordinate(mdim),
                    )
                else:
                    raise ValueError(f"Unsupported placement type: {placement}")
                new_logical_shape[tensor_dim] = new_size
        return new_logical_shape

    def generate_graph_based_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
        full_tensor_shape: tuple[int, ...],
    ) -> list[_TransformInfo]:
        # In case _StridedShard exists in placements, we let _StridedShard have
        # higher priority to express shard_order.
        # TODO(zpcore): Temporary workaround for backward compatibility where
        # _StridedShard was used to encode device shard order. We should migrate
        # to explicit `shard_order` instead.
        def _try_normalize_spec(
            spec: DTensorSpec,
        ) -> tuple[tuple[Placement, ...], ShardOrder | None]:
            # If any _StridedShard is present, try normalize placements into
            # explicit shard_order.
            if any(isinstance(p, _StridedShard) for p in spec.placements):
                new_placements, shard_order = (
                    DTensorSpec._normalize_placements_into_shard_order(
                        spec.placements, spec.mesh
                    )
                )
            else:
                new_placements, shard_order = spec.placements, spec.shard_order

            if shard_order is not None:
                return new_placements, shard_order

            # Fallback: compute default shard_order (treat _StridedShard as
            # normal shard for order).
            shard_order = DTensorSpec.compute_default_shard_order(
                spec.placements, treat_strided_shard_as_shard=True
            )
            return spec.placements, shard_order

        src_placements, src_shard_order = _try_normalize_spec(src_spec)
        dst_placements, dst_shard_order = _try_normalize_spec(dst_spec)

        if src_shard_order is None or dst_shard_order is None:
            raise ValueError(
                f"Cannot compute redistribution plan from {src_spec} to {dst_spec}: "
                "failed to derive a valid shard_order"
            )

        # In case _StridedShard still exists in placements, collect possible
        # split_factor values in the target placements. Need those values to
        # redistribute from Shard into _StridedShard.
        for placement in dst_placements:
            if isinstance(placement, _StridedShard):
                self.strided_shard_placements_in_target.add(placement)

        src_state = self.DistState(src_placements, src_shard_order)
        dst_state = self.DistState(dst_placements, dst_shard_order)
        transform_infos: list[_TransformInfo] = []
        state_path = self.find_min_cost_path(src_state, dst_state)
        for cur_state, nxt_state in itertools.pairwise(state_path):
            # find the mesh_dim that is different between cur_state and nxt_state
            if cur_state.placements != nxt_state.placements:
                update_mesh_dim = -1
                for mesh_dim, (cur_placement, nxt_placement) in enumerate(
                    zip(cur_state.placements, nxt_state.placements)
                ):
                    if cur_placement != nxt_placement:
                        if update_mesh_dim != -1:
                            raise AssertionError(
                                "Multiple mesh_dims are different between cur_state and nxt_state"
                            )
                        update_mesh_dim = mesh_dim
                        logical_shape = self.get_logical_shape(
                            cur_state, mesh_dim, full_tensor_shape
                        )
                        transform_infos.append(
                            _TransformInfo(
                                mesh_dim=update_mesh_dim,
                                src_dst_placements=(cur_placement, nxt_placement),
                                logical_shape=logical_shape,
                            )
                        )

        return transform_infos

    def generate_greedy_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
    ) -> list[_TransformInfo]:
        """
        Generate the transform infos from the source placements to the target placements.

        To transform from source to target placement it might have multiple steps, i.e. it
        might decompose Si -> Sj into Si -> R -> Sj.
        This would detect if there're mis-aligned/nested shardings between src/dst placements.
        E.g. Suppose the redistribution to perform is (Shard(0), Shard(0)) -> (Replicate(), Shard(0)),
        in this case Shard(0) -> Shard(0) for mesh dimension 1 actually needs resharding, because in
        the former is a nested-sharding of a tensor already already sharded dimension 0, whereas
        the latter is the first sharding on tensor dimension 0.
        """
        # logical shape records the logic tensor shape on the mesh dimension
        # this is useful to ensure uneven sharding gets correct output shape
        initial_logical_shape = list(src_spec.shape)
        mesh_dims_to_logical_shape = [initial_logical_shape]
        transform_infos: list[_TransformInfo] = []
        if self.device_mesh.ndim == 1:
            # if device_mesh is 1D, redistribute is a simple direct
            # transformation (skip if src == dst)
            if src_spec.placements[0] != dst_spec.placements[0]:
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=0,
                        src_dst_placements=(
                            src_spec.placements[0],
                            dst_spec.placements[0],
                        ),
                        logical_shape=initial_logical_shape,
                    )
                )
            return transform_infos

        # Handle multi-dim device mesh placement redistribution First, we need
        # to build the logical shape for each mesh dim for correct allgather
        # uneven shards on each mesh dim (with dynamic padding)
        for i, src in enumerate(src_spec.placements):
            current_logical_shape = mesh_dims_to_logical_shape[i]
            if isinstance(src, Shard):
                if i < self.device_mesh.ndim - 1:
                    # calculate and save the logical shape for this sharding
                    mesh_dim_size = self.device_mesh.size(mesh_dim=i)
                    local_shard_size, _ = src._local_shard_size_and_offset(
                        current_logical_shape[src.dim],
                        mesh_dim_size,
                        self.device_mesh._sym_get_coordinate(i),
                    )
                    new_logical_shape = list(current_logical_shape)
                    new_logical_shape[src.dim] = local_shard_size
                    mesh_dims_to_logical_shape.append(new_logical_shape)
            else:
                mesh_dims_to_logical_shape.append(current_logical_shape)

        # Next, we need to derive the transform infos from src to dst
        # placements, here we use a greedy search with step by step state
        # transformations
        current_placements = list(src_spec.placements)
        target_placements = list(dst_spec.placements)

        if src_spec.num_shards > 1:
            # If src_spec have sharding, it could potentially have sharding that
            # is misaligned with dst_spec a common case of this is nested
            # sharding (i.e. (S(0), S(0)) -> (R, S(0))). In those cases, we
            # first traverse from inner placement to outer placement to detect
            # misaligned shardings and properly replicate nested sharding first.
            for mesh_dim in reversed(range(len(current_placements))):
                current = current_placements[mesh_dim]
                target = target_placements[mesh_dim]
                # If target is not Shard, we can directly redistribute since we
                # are traversing from inner to outer placements here
                if isinstance(target, Shard):
                    # If target is Shard, check for nested sharding on the
                    # tensor dim BEFORE the current mesh_dim
                    shard_dim = target.dim
                    current_mesh_sharding, target_mesh_sharding = [], []
                    for i, (s, p) in enumerate(
                        zip(current_placements, target_placements)
                    ):
                        if i >= mesh_dim:
                            break
                        if s.is_shard(shard_dim):
                            current_mesh_sharding.append(i)
                        if p.is_shard(shard_dim):
                            target_mesh_sharding.append(i)

                    if current_mesh_sharding != target_mesh_sharding:
                        # if current/target_placements have misaligned sharding
                        # on the tensor dim BEFORE the current mesh_dim, we need
                        # to replicate the tensor on the mesh dim first to clear
                        # the nested sharding
                        target = Replicate()

                if current != target:
                    transform_infos.append(
                        _TransformInfo(
                            mesh_dim=mesh_dim,
                            src_dst_placements=(current, target),
                            logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                        )
                    )
                    current_placements[mesh_dim] = target

        # We always traverse from outer placement to inner placement to collect
        # the remaining needed transform infos (i.e. the replication from nested
        # sharding might need to further perform resharding to Shard again)
        for mesh_dim, (current, target) in enumerate(
            zip(current_placements, target_placements)
        ):
            if current != target:
                transform_infos.append(
                    _TransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(current, target),
                        logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                    )
                )
                current_placements[mesh_dim] = target
        return transform_infos


def _gen_transform_infos_non_cached(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    use_graph_based_transform: bool | None = None,
) -> list[_TransformInfo]:
    device_mesh = src_spec.device_mesh
    src_shard_order = src_spec.shard_order
    dst_shard_order = dst_spec.shard_order
    # DTensorSpec should automatically generate shard_order, and it can be () if
    # no shard.
    has_non_default_order = not all(
        DTensorSpec.is_default_device_order(order)
        for order in (src_shard_order, dst_shard_order)
    )
    has_strided_shard = any(
        isinstance(p, _StridedShard)
        for p in (*src_spec.placements, *dst_spec.placements)
    )

    # Determine which transform strategy to use:
    # 1. Non-standard device order or contains _StridedShard → always use graph-based
    # 2. Global flag or explicit parameter True → use graph-based
    # 3. Otherwise → use greedy
    if has_non_default_order or has_strided_shard:
        use_graph_based_transform = True
    elif _FORCE_MIN_COST_REDISTRIBUTION_PLAN is not None:
        use_graph_based_transform = _FORCE_MIN_COST_REDISTRIBUTION_PLAN
    elif use_graph_based_transform is None:
        use_graph_based_transform = False
    assert src_spec.tensor_meta is not None
    drp = get_redistribute_planner(
        device_mesh,
        src_spec.tensor_meta,
    )
    if use_graph_based_transform:
        transform_infos = drp.generate_graph_based_transform_infos(
            src_spec, dst_spec, src_spec.shape
        )
    else:
        transform_infos = drp.generate_greedy_transform_infos(src_spec, dst_spec)
    return transform_infos


@cache
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    use_graph_based_transform: bool | None = None,
) -> list[_TransformInfo]:
    return _gen_transform_infos_non_cached(
        src_spec, dst_spec, use_graph_based_transform
    )


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    use_graph_based_transform: bool | None = None,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = local_tensor
    device_mesh = current_spec.mesh

    if not device_mesh._is_current_rank_part_of_mesh():
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    if _are_we_tracing():
        transform_infos = _gen_transform_infos_non_cached(
            current_spec, target_spec, use_graph_based_transform
        )
    else:
        transform_infos = _gen_transform_infos(
            current_spec, target_spec, use_graph_based_transform
        )

    debug_mode = get_active_debug_mode()
    redistribute_context = (
        debug_mode.record_redistribute_calls(  # type: ignore[union-attr]
            local_tensor,
            current_spec.placements,
            target_spec.placements,
            DTensorRedistributePlanner.stringify_transform_infos(
                device_mesh,
                transform_infos,
                current_spec.placements,
                current_spec.shard_order,
            ),
        )
        if debug_mode is not None
        else contextlib.nullcontext()
    )

    with redistribute_context:
        for transform_info in transform_infos:
            i = transform_info.mesh_dim
            current, target = transform_info.src_dst_placements
            num_chunks = device_mesh.size(mesh_dim=i)

            if current == target:
                # short cut, just use the original local tensor
                new_local_tensor = local_tensor
                continue

            if num_chunks == 1:
                # short cut, if there's only one shard, we don't need to do any collective
                # comm, just use the original local tensor
                new_local_tensor = local_tensor
                continue

            if target.is_replicate():
                # Case 1: target is Replicate
                if current.is_partial():
                    partial_spec = cast(Partial, current)
                    new_local_tensor = partial_spec._reduce_value(
                        local_tensor, device_mesh, i
                    )
                elif current.is_shard():
                    current_placement = cast(Shard, current)
                    new_local_tensor = current_placement._to_replicate_tensor(
                        local_tensor, device_mesh, i, transform_info.logical_shape
                    )
                elif isinstance(current, _StridedShard):
                    new_local_tensor = current._to_replicate_tensor(
                        local_tensor, device_mesh, i, transform_info.logical_shape
                    )
                else:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )

            elif target.is_shard():
                # Case 2: target is Shard
                target_placement = cast(Shard, target)
                if current.is_partial():
                    partial_spec = cast(Partial, current)
                    new_local_tensor = partial_spec._reduce_shard_value(
                        local_tensor, device_mesh, i, target_placement
                    )
                elif current.is_replicate():
                    # split the tensor and return the corresponding cloned local shard
                    new_local_tensor = target_placement._replicate_to_shard(
                        local_tensor, device_mesh, i, device_mesh._sym_get_coordinate(i)
                    )
                elif current.is_shard():
                    shard_spec = cast(Shard, current)
                    if shard_spec.dim != target_placement.dim:
                        new_local_tensor = shard_spec._to_new_shard_dim(
                            local_tensor,
                            device_mesh,
                            i,
                            transform_info.logical_shape,
                            target_placement.dim,
                        )
                elif isinstance(current, _StridedShard):
                    raise NotImplementedError(
                        "Redistribute from _StridedShard to Shard is not implemented yet"
                    )
                else:
                    raise ValueError(
                        f"Unexpected placement {current} for redistribute to target placement {target}"
                    )
            elif target.is_partial():
                if current.is_replicate():
                    partial_spec = cast(Partial, target)
                    new_local_tensor = partial_spec._partition_value(
                        local_tensor, device_mesh, i
                    )
                elif current.is_shard() or isinstance(current, _StridedShard):
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                else:
                    if current != target:
                        raise AssertionError(
                            f"Redistribution from one partial type ({current}) to another ({target}) is unsupported."
                        )
                    # partial -> partial no op, should never hit
                    new_local_tensor = local_tensor
            elif isinstance(target, _StridedShard):
                # Case 4: target is _StridedShard
                if current.is_partial():
                    raise NotImplementedError(
                        "Redistribute from Partial to _StridedShard is not implemented yet"
                    )
                elif current.is_replicate():
                    # split the tensor and return the corresponding local strided shard
                    new_local_tensor = target._replicate_to_strided_shard(
                        local_tensor, device_mesh, i, device_mesh._sym_get_coordinate(i)
                    )
                elif current.is_shard():
                    # Shard -> _StridedShard on potentially different dimensions
                    raise NotImplementedError(
                        "Redistribute from Shard to _StridedShard is not implemented yet"
                    )
                elif isinstance(current, _StridedShard):
                    # _StridedShard -> _StridedShard: go through Replicate
                    # First convert to Replicate, then to _StridedShard
                    replicated = current._to_replicate_tensor(
                        local_tensor, device_mesh, i, transform_info.logical_shape
                    )
                    new_local_tensor = target._replicate_to_strided_shard(
                        replicated, device_mesh, i, device_mesh._sym_get_coordinate(i)
                    )
                else:
                    raise ValueError(
                        f"Unexpected placement {current} for redistribute to target placement {target}"
                    )

            if not async_op and isinstance(
                new_local_tensor, funcol.AsyncCollectiveTensor
            ):
                new_local_tensor = new_local_tensor.wait()
            local_tensor = new_local_tensor
    return new_local_tensor


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        async_op: bool = False,
        forward_dtype: torch.dtype | None = None,
        backward_dtype: torch.dtype | None = None,
    ):
        ctx.async_op = async_op
        ctx.backward_dtype = backward_dtype
        ctx.original_dtype = input._local_tensor.dtype

        if forward_dtype is not None and forward_dtype != input._local_tensor.dtype:
            local_tensor = input._local_tensor.to(dtype=forward_dtype)
            current_spec = DTensorSpec(
                mesh=device_mesh,
                placements=input._spec.placements,
                tensor_meta=TensorMeta(
                    shape=input.shape,
                    stride=input.stride(),
                    dtype=forward_dtype,
                ),
            )
        else:
            local_tensor = input._local_tensor
            current_spec = input._spec

        ctx.current_spec = current_spec

        if current_spec.placements != placements:
            target_spec = DTensorSpec(
                device_mesh, placements, tensor_meta=current_spec.tensor_meta
            )

            output = redistribute_local_tensor(
                local_tensor, current_spec, target_spec, async_op=async_op
            )
        else:
            # use the same local tensor if placements are the same.
            output = local_tensor
            target_spec = current_spec

        # pyrefly: ignore [bad-argument-type]
        return dtensor.DTensor(
            # pyrefly: ignore [bad-argument-count]
            output,
            target_spec,
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore[override]
        previous_spec = ctx.current_spec
        async_op = ctx.async_op
        backward_dtype = ctx.backward_dtype or ctx.original_dtype

        if backward_dtype != grad_output._local_tensor.dtype:
            local_tensor = grad_output._local_tensor.to(dtype=backward_dtype)
            current_spec = DTensorSpec(
                mesh=grad_output._spec.device_mesh,
                placements=grad_output._spec.placements,
                tensor_meta=TensorMeta(
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                    dtype=backward_dtype,
                ),
            )
            previous_spec = DTensorSpec(
                mesh=previous_spec.device_mesh,
                placements=previous_spec.placements,
                tensor_meta=current_spec.tensor_meta,
            )
        else:
            local_tensor = grad_output._local_tensor
            current_spec = grad_output._spec
        # skip the replicate to partial transformation when we are in backward pass
        # In this case we keep the grad as replicate, this is because we don't
        # want to convert the replicated gradients back to partial, although
        # that's logically conform with the same layout, converting the gradients
        # back to partial is actually useless as you would have to do reduce later
        # which would be more expensive than keeping it replicate!

        # for backward shard -> partial, we just do shard -> replicate
        # for backward replicate -> partial, we skip the transformation
        normalized_placements: list[Placement] = []
        for current, target in zip(current_spec.placements, previous_spec.placements):
            if (current.is_shard() or current.is_replicate()) and target.is_partial():
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(target)

        previous_spec = DTensorSpec(
            previous_spec.device_mesh,
            placements=tuple(normalized_placements),
            tensor_meta=previous_spec.tensor_meta,
        )

        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
        )

        if output.dtype != ctx.original_dtype:
            output = output.to(ctx.original_dtype)

        spec = DTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=output.dtype,
            ),
        )
        # pyrefly: ignore [bad-argument-type]
        output_dtensor = dtensor.DTensor(
            # pyrefly: ignore [bad-argument-count]
            output,
            spec,
            # pyrefly: ignore [unexpected-keyword]
            requires_grad=grad_output.requires_grad,
        )

        return (
            output_dtensor,
            None,
            None,
            None,
            None,
            None,
        )
