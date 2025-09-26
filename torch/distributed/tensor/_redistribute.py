# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import dataclasses
import itertools
import logging
<<<<<<< HEAD
import weakref
=======
from enum import auto, Enum
>>>>>>> 858b9a4768c (Support of AllPermute for redistribution)
from functools import cache
from typing import cast, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed._functional_collectives import _are_we_tracing
from torch.distributed.tensor._collective_utils import all_permute_mesh_dim
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._debug_mode import get_active_debug_mode


logger = logging.getLogger(__name__)

from typing import TypeVar


TOrder = TypeVar("TOrder", bound=tuple[tuple[int, ...], ...])

# Print uses JAX-style sharding representation, which maps tensor dimensions to
# mesh dimensions.
use_jax_style_to_print_distribution = True


@dataclasses.dataclass
class TransformInfo:
    pass


@dataclasses.dataclass
class PerDimTransformInfo(TransformInfo):
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    # logical_shape on this mesh dimension
    logical_shape: list[int]


@dataclasses.dataclass
class AllPermuteTransformInfo(TransformInfo):
    # The distribution maps each tensor dimension to a list of ordered mesh
    # dimensions that it is sharded across.
    src_distribution: list[list[int]]
    dst_distribution: list[list[int]]


# TODO(zpcore): add support for combining multiple consecutive collective
# operations of the same type into a single transform
class MergedTransformInfo(TransformInfo):
    pass


class DTensorRedistributePlanner:
    """
    This class is used to plan the collective calls to transform the local shard
    of the DTensor from its current spec to the target spec.

    Suppose there are N tensor dimensions and M mesh dimensions, the total
    possible state size will be (N+2)*M*M!.
    """

    _instances: dict = {}

    @dataclasses.dataclass(frozen=True, slots=True)
    class DistState:
        placements: tuple[Placement, ...]
        tensor_dim_to_mesh_dim: tuple[tuple[int, ...], ...]
        _hash: Optional[int] = dataclasses.field(
            default=None, init=False, repr=False, compare=False
        )

        def __str__(self):
            return DTensorSpec.format_shard_order_str(
                self.placements,
                self.tensor_dim_to_mesh_dim,
                use_jax_style_to_print_distribution,
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

    class TransitionType(Enum):
        NO_OP = auto()  # no op transition
        SHARD_TO_SHARD = auto()
        SHARD_TO_REPLICATE = auto()
        PARTIAL_TO_REPLICATE = auto()
        REPLICATE_TO_SHARD = auto()
        PARTIAL_TO_SHARD = auto()
        REPLICATE_TO_PARTIAL = auto()
        ALL_PERMUTE = auto()
        MERGE_OPS = auto()  # not implemented yet

    @classmethod
    def _create_cache_key(cls, device_mesh, tensor_dimension):
        return (weakref.ref(device_mesh), tensor_dimension)

    def __new__(cls, device_mesh, tensor_dimension):
        cache_key = cls._create_cache_key(device_mesh, tensor_dimension)

        if cache_key not in cls._instances:
            instance = super().__new__(cls)
            object.__setattr__(instance, "_cache_key", cache_key)

            instance._initialized = False
            cls._instances[cache_key] = instance

        return cls._instances[cache_key]

    @classmethod
    def clear_cache(cls):
        cls._instances.clear()

    def __init__(
        self,
        device_mesh,
        tensor_dimension: int,
        # ``global_tensor_size`` is only needed if we want to enable all_permute,
        # because we need to know the local and global tensor shape to make
        # decision.
        global_tensor_size: Optional[torch.Size] = None,
    ) -> None:
        # only initialize once
        if getattr(self, "_initialized", False):
            return
        self.device_mesh = device_mesh
        self.coordinate = device_mesh.get_coordinate()
        assert self.coordinate is not None
        self.tensor_dimension = tensor_dimension
        self.global_tensor_size = global_tensor_size
        self.setup_collective_cost()
        self._initialized = True

    def setup_collective_cost(
        self,
        all_reduce_cost: int = 4,
        all_to_all_cost: int = 1,
        all_gather_cost: int = 2,
        reduce_scatter_cost: int = 2,
        chunk_cost: int = 0,
    ) -> None:
        """
        Set up the cost weights for different collective operations.

        Args:
            all_reduce_cost: Cost weight for all-reduce operations
            all_to_all_cost: Cost weight for all-to-all operations
            all_gather_cost: Cost weight for all-gather operations
        """
        # those can be turned in a handler considering the tensor dim size
        self.all_reduce_cost = all_reduce_cost
        self.all_to_all_cost = all_to_all_cost
        self.all_gather_cost = all_gather_cost
        self.reduce_scatter = reduce_scatter_cost
        self.chunk_cost = chunk_cost

    def _to_tuple(self, x):
        """Convert a nested list structure to a nested tuple structure."""
        if isinstance(x, (list, tuple)):
            return tuple(self._to_tuple(item) for item in x)
        return x

    def all_permute_sharding(
        self,
        placements: tuple[Placement, ...],
        tensor_dim_mesh_dim: tuple[tuple[int, ...], ...],
    ) -> list["DTensorRedistributePlanner.DistState"]:
        """
        We can perform all_permute collective op to transform any distribution
        if their local and global shapes are the same. According to
        https://arxiv.org/pdf/2112.01075, at most one all_permute operation is
        needed in the redistribution sequence from source to target. The
        all_permute can be performed as the final step, or moved before the last
        all_gather to minimize the amount of data involved in the all_permute.

        Example: Given mesh and size {X:4, Y:4, Z:16}, we have
            example 1: [32{X,Y}512, 128] -> [32{Y,X}512, 128]
            example 2: [128{Y}512, 32{X}128] -> [128{X}512, 32{Y}128]
            example 3: [32{X,Y}512, 128] -> [32{Z}512, 128]
        Note: annotation borrowed from https://arxiv.org/pdf/2112.01075 section
        2.1--"Distributed array types"
        """
        # Early exit: ensure each tensor dimension is evenly divisible by the
        # corresponding device mesh dimensions.
        local_tensor_size = [0 for _ in range(self.tensor_dimension)]
        if not self.global_tensor_size:
            return []
        for tensor_dim, mesh_dims in enumerate(tensor_dim_mesh_dim):
            divisor = 1
            for mesh_dim in mesh_dims:
                divisor *= self.device_mesh.size(mesh_dim=mesh_dim)
            if self.global_tensor_size[tensor_dim] % divisor != 0:
                return []
            local_tensor_size[tensor_dim] = (
                self.global_tensor_size[tensor_dim] // divisor
            )

        all_feasible_distribution: list[DTensorRedistributePlanner.DistState] = []
        # find possible device order permutation.
        mesh_dim_permutation = list(
            itertools.permutations(range(self.device_mesh.ndim))
        )
        tensor_dim_to_device_dims = {
            tensor_dim: mesh_dims
            for tensor_dim, mesh_dims in enumerate(tensor_dim_mesh_dim)
            if len(mesh_dims) > 0
        }
        for ordered_devices in mesh_dim_permutation:
            # try different split way based on the number of tensor dim sharding
            # Generate all possible combinations of mesh dimensions to assign to tensor dimensions
            # that will be sharded, based on the number of tensor dimensions that need sharding
            mesh_dim_combinations = list(
                itertools.combinations(
                    range(self.device_mesh.ndim), len(tensor_dim_to_device_dims)
                )
            )

            for try_shard_on_mesh_dims_indexes in mesh_dim_combinations:
                # now we split the list of device mesh dims into
                # ``tensor_dim_to_device_dims`` groups, namely:
                # group 0: [0, try_shard_on_mesh_dims_indexes[0]](both inclusive)
                # group 1: [try_shard_on_mesh_dims_indexes[0]+1, try_shard_on_mesh_dims_indees[1]](both inclusive)
                # group 2: ...
                is_feasible = True
                tensor_dims_get_sharded = list(tensor_dim_to_device_dims.keys())
                assert tensor_dims_get_sharded == len(try_shard_on_mesh_dims_indexes)

                prev_index = 0
                # For each group, check if the tensor dim can be evenly sharded.
                new_tensor_dim_mesh_dim: list[list[int]] = [
                    [] for _ in range(self.tensor_dimension)
                ]
                for tensor_dim, try_shard_on_mesh_dims_index in zip(
                    tensor_dims_get_sharded, try_shard_on_mesh_dims_indexes
                ):
                    maybe_shard_on_mesh_dims = ordered_devices[
                        prev_index : try_shard_on_mesh_dims_index + 1
                    ]
                    prev_index = try_shard_on_mesh_dims_index + 1
                    # Verify that the split produces the same local tensor size
                    # as the original sharding. Currently, we only allow cases
                    # where the tensor dimension size is divisible by the
                    # selected mesh dimensions.
                    divisor = 1
                    for mesh_dim in maybe_shard_on_mesh_dims:
                        divisor *= self.device_mesh.size(mesh_dim=mesh_dim)
                    if self.global_tensor_size[tensor_dim] % divisor != 0:
                        is_feasible = False
                        break
                    new_local_size = self.global_tensor_size[tensor_dim] // divisor
                    if new_local_size != local_tensor_size[tensor_dim]:
                        is_feasible = False
                        break
                    new_tensor_dim_mesh_dim[tensor_dim] = list(maybe_shard_on_mesh_dims)
                if is_feasible:
                    # generate the target distribution.
                    dist_state = self.DistState(
                        placements,
                        self._to_tuple(new_tensor_dim_mesh_dim),
                    )
                    all_feasible_distribution.append(dist_state)
        return all_feasible_distribution

    def get_next_state(
        self,
        placements: tuple[Placement, ...],
        tensor_dim_mesh_dim: tuple[tuple[int, ...], ...],
    ):
        # We map tensor dimensions to device mesh axes, similar to JAX-style
        # sharding representation. Notation:
        # S(<tensor_dim>)[<list_of_device_dims>] means tensor dimension
        # <tensor_dim> is sharded on the listed device mesh axes, where
        # <list_of_device_dims> is sorted by device order.
        #
        # To generalize to arbitrary dimensionality, we use the following notation:
        #   S(a)[x, ...]   : tensor dimension 'a' is sharded on device mesh axes x, ... (variadic, possibly empty)
        #   R[...]         : replicated on the listed device mesh axes (possibly empty)
        #   P[...]         : partial on the listed device mesh axes (possibly empty)
        # The ellipsis '...' denotes a variadic wildcard, i.e., zero or more device mesh axes.
        #
        # Below are possible transitions from one sharding state to another.
        # We use `S` for Shard, `R` for Replicate, and `P` for Partial.
        #
        # Case 1. Shard(a) -> Shard(b), use all-to-all (a2a), applies to:
        #   S(a)[..., x] -> S(b)[..., x]
        #   or
        #   S(a)[..., x, y] S(b)[..., z, k] -> S(a)[..., x] S(b)[..., z, k, y]
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
        #   P[..., x] -> S(a)[..., x] or P[..., x, y] -> P[..., x] S(a)[..., y]
        #
        # Case 6. Replicate() -> Partial(), local math op, applies to:
        #   * -> P[..., x]

        # case 7. all_permute: Handles transitions where the sharding pattern
        # changes only by permuting device mesh axes, allowing data to be
        # rearranged across devices without changing the local/global tensor
        # shape.

        # list of [DistState, [cost, transition type]
        all_next_state: dict[
            DTensorRedistributePlanner.DistState,
            tuple[int, DTensorRedistributePlanner.TransitionType],
        ] = {}

        ######################################################################
        # handle case 1: Shard(a) -> Shard(b)
        # For S(a), S(b), only the last device order of S(a) and S(b) can be a2a
        # interchangeably.
        for src_tensor_dim in range(self.tensor_dimension):
            for dst_tensor_dim in range(self.tensor_dimension):
                if src_tensor_dim == dst_tensor_dim:
                    continue
                # try move the last sharded device dim from
                # Shard(src_tensor_dim) to Shard(dst_tensor_dim)
                if len(tensor_dim_mesh_dim[src_tensor_dim]) > 0:
                    new_tensor_dim_mesh_dim = [
                        list(dim_tuple) for dim_tuple in tensor_dim_mesh_dim
                    ]
                    move_mesh_dim = new_tensor_dim_mesh_dim[src_tensor_dim].pop()
                    new_tensor_dim_mesh_dim[dst_tensor_dim].append(move_mesh_dim)
                    new_placements = list(placements)
                    new_placements[move_mesh_dim] = Shard(dst_tensor_dim)
                    dist_state = self.DistState(
                        self._to_tuple(new_placements),
                        self._to_tuple(new_tensor_dim_mesh_dim),
                    )
                    all_next_state[dist_state] = (
                        self.all_to_all_cost,
                        DTensorRedistributePlanner.TransitionType.SHARD_TO_SHARD,
                    )
        # TODO(zpcore): support discovering submesh to prevent padding when
        # tensor dim is not divisible by the mesh dim.

        ######################################################################
        # handle case 2: Shard() -> Replicate()
        for src_tensor_dim in range(self.tensor_dimension):
            if len(tensor_dim_mesh_dim[src_tensor_dim]) > 0:
                new_tensor_dim_mesh_dim = [
                    list(dim_tuple) for dim_tuple in tensor_dim_mesh_dim
                ]
                move_mesh_dim = new_tensor_dim_mesh_dim[src_tensor_dim].pop()
                new_placements = list(placements)
                new_placements[move_mesh_dim] = Replicate()
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    self._to_tuple(new_tensor_dim_mesh_dim),
                )
                all_next_state[dist_state] = (
                    self.all_gather_cost,
                    DTensorRedistributePlanner.TransitionType.SHARD_TO_REPLICATE,
                )

        ######################################################################
        # handle case 3: Partial() -> Replicate()
        for src_tensor_dim in range(self.tensor_dimension):
            if isinstance(src_tensor_dim, Partial):
                new_placements = list(placements)
                new_placements[src_tensor_dim] = Replicate()
                dist_state = self.DistState(
                    self._to_tuple(new_placements), tensor_dim_mesh_dim
                )
                all_next_state[dist_state] = (
                    self.all_gather_cost,
                    DTensorRedistributePlanner.TransitionType.PARTIAL_TO_REPLICATE,
                )

        ######################################################################
        # handle case 4: Replicate() -> Shard()
        for mesh_dim in range(self.device_mesh.ndim):
            if not isinstance(placements[mesh_dim], Replicate):
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                # try convert placement[mesh_dim] to Shard(dst_tensor_dim)
                new_placements = list(placements)
                new_placements[mesh_dim] = Shard(dst_tensor_dim)
                new_tensor_dim_mesh_dim = [
                    list(dim_tuple) for dim_tuple in tensor_dim_mesh_dim
                ]
                new_tensor_dim_mesh_dim[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    self._to_tuple(new_tensor_dim_mesh_dim),
                )
                all_next_state[dist_state] = (
                    self.chunk_cost,
                    DTensorRedistributePlanner.TransitionType.REPLICATE_TO_SHARD,
                )

        ######################################################################
        # handle case 5: Partial() -> Shard()
        for mesh_dim in range(self.device_mesh.ndim):
            if not isinstance(placements[mesh_dim], Partial):
                continue
            for dst_tensor_dim in range(self.tensor_dimension):
                # try convert placement[mesh_dim] to Shard(dst_tensor_dim)
                new_placements = list(placements)
                new_placements[mesh_dim] = Shard(dst_tensor_dim)
                new_tensor_dim_mesh_dim = [
                    list(dim_tuple) for dim_tuple in tensor_dim_mesh_dim
                ]
                new_tensor_dim_mesh_dim[dst_tensor_dim].append(mesh_dim)
                dist_state = self.DistState(
                    self._to_tuple(new_placements),
                    self._to_tuple(new_tensor_dim_mesh_dim),
                )
                all_next_state[dist_state] = (
                    self.reduce_scatter,
                    DTensorRedistributePlanner.TransitionType.PARTIAL_TO_SHARD,
                )

        ######################################################################
        # handle case 6: Replicate() -> Partial(), default to partial(sum)
        for mesh_dim in range(self.device_mesh.ndim):
            if not isinstance(placements[mesh_dim], Replicate):
                continue
            new_placements = list(placements)
            new_placements[mesh_dim] = Partial()
            dist_state = self.DistState(
                self._to_tuple(new_placements), tensor_dim_mesh_dim
            )
            all_next_state[dist_state] = (
                self.chunk_cost,
                DTensorRedistributePlanner.TransitionType.REPLICATE_TO_PARTIAL,
            )

        ######################################################################
        # handle case 7: allpermute * -> *, swap data when localSize and
        # globalSize type match
        for dist_state in self.all_permute_sharding(placements, tensor_dim_mesh_dim):
            all_next_state[dist_state] = (
                self.all_to_all_cost,
                DTensorRedistributePlanner.TransitionType.ALL_PERMUTE,
            )

        return all_next_state

    def find_min_cost_path(
        self, src_state: DistState, dst_state: DistState
    ) -> list[
        tuple[
            "DTensorRedistributePlanner.DistState",
            "DTensorRedistributePlanner.TransitionType",
        ]
    ]:
        """
        Find the min cost path from src_state to dst_state using Dijkstra's
        algorithm.

        Args:
            src_state: The source state
            dst_state: The destination state

        Returns:
            A list of tuples representing the min cost path from src_state to
            dst_state, where each tuple contains (state, transition_type)
        """
        import heapq

        # Priority queue (cost, counter, state, path_with_transitions) for
        # Dijkstra's algorithm. Use counter to break ties and avoid comparing
        # DistState objects path_with_transitions: list of (state,
        # transition_type) tuples
        counter = 0
        pq: list[
            tuple[
                int,
                int,
                DTensorRedistributePlanner.DistState,
                list[
                    tuple[
                        DTensorRedistributePlanner.DistState,
                        DTensorRedistributePlanner.TransitionType,
                    ]
                ],
            ]
        ] = [
            (
                0,
                counter,
                src_state,
                [(src_state, DTensorRedistributePlanner.TransitionType.NO_OP)],
            )
        ]
        visited = set()
        while pq:
            cost, _, current_state, path_with_transitions = heapq.heappop(pq)
            if current_state == dst_state:
                return path_with_transitions
            if current_state in visited:
                continue
            visited.add(current_state)
            # get all possible next states and their costs
            next_states = self.get_next_state(
                current_state.placements, current_state.tensor_dim_to_mesh_dim
            )
            for next_state, (transition_cost, transition_type) in next_states.items():
                if next_state not in visited:
                    new_cost = cost + transition_cost
                    new_path_with_transitions = path_with_transitions + [
                        (next_state, transition_type)
                    ]
                    counter += 1
                    heapq.heappush(
                        pq, (new_cost, counter, next_state, new_path_with_transitions)
                    )
        raise AssertionError(
            f"No path found from src_state {src_state} to dst_state {dst_state}"
        )

    def get_logical_shape(
        self,
        src_state: "DTensorRedistributePlanner.DistState",
        mesh_dim: int,
        global_tensor_size: tuple[int, ...],
    ):
        new_logical_shape = list(global_tensor_size)
        for tensor_dim, mesh_dims in enumerate(src_state.tensor_dim_to_mesh_dim):
            for mdim in mesh_dims:
                if mdim == mesh_dim:
                    continue
                new_size = Shard._local_shard_size_and_offset(
                    new_logical_shape[tensor_dim],
                    self.device_mesh.size(mesh_dim=mdim),
                    self.coordinate[mdim],
                )[0]
                new_logical_shape[tensor_dim] = new_size
        return new_logical_shape

    def generate_optimal_transform_infos(
        self,
        src_spec: DTensorSpec,
        dst_spec: DTensorSpec,
        global_tensor_size: tuple[int, ...],
    ) -> list[TransformInfo]:
        assert src_spec.shard_order is not None and dst_spec.shard_order is not None
        src_state = self.DistState(src_spec.placements, src_spec.shard_order)
        dst_state = self.DistState(dst_spec.placements, dst_spec.shard_order)

        transform_infos: list[TransformInfo] = []
        state_path_with_transitions = self.find_min_cost_path(src_state, dst_state)

        # Extract just the states for logging
        states_only = [state for state, _ in state_path_with_transitions]
        logger.debug(
            "Path from %s to %s: \n%s",
            src_state,
            dst_state,
            " -> ".join(str(s) for s in states_only),
        )

        # Iterate through consecutive states and their transition types
        for i in range(len(state_path_with_transitions) - 1):
            cur_state, cur_transition_type = state_path_with_transitions[i]
            nxt_state, nxt_transition_type = state_path_with_transitions[i + 1]

            if (
                nxt_transition_type
                == DTensorRedistributePlanner.TransitionType.ALL_PERMUTE
            ):
                transform_infos.append(
                    AllPermuteTransformInfo(
                        src_distribution=[
                            list(x) for x in cur_state.tensor_dim_to_mesh_dim
                        ],
                        dst_distribution=[
                            list(x) for x in nxt_state.tensor_dim_to_mesh_dim
                        ],
                    )
                )
            else:
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
                                cur_state, mesh_dim, global_tensor_size
                            )
                            transform_infos.append(
                                PerDimTransformInfo(
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
    ) -> list[TransformInfo]:
        """
        Generate the transform infos from the source placements to the target placements.

        To transform from source to target placement it might have multiple steps, i.e. it
        might decompose Si -> Sj into Si -> R -> Sj.
        This would detect if there're mis-aligned/nested shardings between src/dst placements.
        E.g. Suppose the redistribution to perform is (Shard(0), Shard(0)) -> (Replicate(), Shard(0)),
        in this case Shard(0) -> Shard(0) for mesh dimension 1 actually needs resharding, because in
        the former is a nested-sharding of a tensor already already sharded dimension 0, whereras
        the latter is the first sharding on tensor dimension 0.
        """
        # logical shape records the logic tensor shape on the mesh dimension
        # this is useful to ensure uneven sharding gets correct output shape
        initial_logical_shape = list(src_spec.shape)
        mesh_dims_to_logical_shape = [initial_logical_shape]
        transform_infos: list[TransformInfo] = []
        if self.device_mesh.ndim == 1:
            # if device_mesh is 1D, redistribute is a simple direct transformation
            transform_infos.append(
                PerDimTransformInfo(
                    mesh_dim=0,
                    src_dst_placements=(src_spec.placements[0], dst_spec.placements[0]),
                    logical_shape=initial_logical_shape,
                )
            )
            return transform_infos

        # Handle multi-dim device mesh placement redistribution
        # First, we need to build the logical shape for each mesh dim
        # for correct allgathering uneven shards on each mesh dim (with dynamic padding)
        for i, src in enumerate(src_spec.placements):
            current_logical_shape = mesh_dims_to_logical_shape[i]
            if isinstance(src, Shard):
                if i < self.device_mesh.ndim - 1:
                    # calculate and save the logical shape for this sharding
                    mesh_dim_size = self.device_mesh.size(mesh_dim=i)
                    local_shard_size, _ = src._local_shard_size_and_offset(
                        current_logical_shape[src.dim],
                        mesh_dim_size,
                        self.coordinate[i],
                    )
                    new_logical_shape = list(current_logical_shape)
                    new_logical_shape[src.dim] = local_shard_size
                    mesh_dims_to_logical_shape.append(new_logical_shape)
            else:
                mesh_dims_to_logical_shape.append(current_logical_shape)

        # Next, we need to derive the transform infos from src to dst placements,
        # here we use a greedy search with step by step state transformations
        current_placements = list(src_spec.placements)
        target_placements = list(dst_spec.placements)

        if src_spec.num_shards > 1:
            # If src_spec have sharding, it could potentially have sharding that is misaligned with dst_spec
            # a common case of this is nested sharding (i.e. (S(0), S(0)) -> (R, S(0))).
            # In those cases, we first traverse from inner placement to outer placement
            # to detect misaligned shardings and properly replicate nested sharding first.
            for mesh_dim in reversed(range(len(current_placements))):
                current = current_placements[mesh_dim]
                target = target_placements[mesh_dim]
                # If target is not Shard, we can directly redistribute since we are traversing from innner
                # to outer placements here
                if isinstance(target, Shard):
                    # If target is Shard, check for nested sharding on the tensor dim BEFORE the current mesh_dim
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
                        # if current/target_placements have misaligned sharding on the tensor dim BEFORE the current
                        # mesh_dim, we need to replicate the tensor on the mesh dim first to clear the nested sharding
                        target = Replicate()

                if current != target:
                    transform_infos.append(
                        PerDimTransformInfo(
                            mesh_dim=mesh_dim,
                            src_dst_placements=(current, target),
                            logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                        )
                    )
                    current_placements[mesh_dim] = target

        # We always traverse from outer placement to inner placement to collect the remaining
        # needed transform infos (i.e. the replication from nested sharding might need to further
        # perform resharding to Shard again)
        for mesh_dim, (current, target) in enumerate(
            zip(current_placements, target_placements)
        ):
            if current != target:
                transform_infos.append(
                    PerDimTransformInfo(
                        mesh_dim=mesh_dim,
                        src_dst_placements=(current, target),
                        logical_shape=mesh_dims_to_logical_shape[mesh_dim],
                    )
                )
                current_placements[mesh_dim] = target
        return transform_infos


def _is_default_device_order(
    placements: tuple[Placement, ...], shard_order: TOrder
) -> bool:
    """
    Check if the device order is the default left-to-right order.
    """
    shard_order_index = [0] * len(shard_order)
    for mesh_dim, p in enumerate(placements):
        if isinstance(p, Shard):
            if mesh_dim != shard_order[p.dim][shard_order_index[p.dim]]:
                return False
            shard_order_index[p.dim] += 1
    return True


def _gen_transform_infos_non_cached(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
) -> list[TransformInfo]:
    transform_infos: list[TransformInfo] = []
    device_mesh = src_spec.device_mesh
    src_shard_order = src_spec.shard_order
    dst_shard_order = dst_spec.shard_order
    assert src_shard_order is not None and dst_shard_order is not None
    assert len(src_shard_order) == len(dst_shard_order)
    if _is_default_device_order(
        src_spec.placements, src_shard_order
    ) and _is_default_device_order(dst_spec.placements, dst_shard_order):
        use_greedy_transform = True
    else:
        # switch to graph search algorithm if the device order is not the default
        use_greedy_transform = False
    drp = DTensorRedistributePlanner(device_mesh, len(src_spec.shape))
    if use_greedy_transform:
        transform_infos = drp.generate_greedy_transform_infos(src_spec, dst_spec)
    else:
        transform_infos = drp.generate_optimal_transform_infos(
            src_spec, dst_spec, src_spec.shape
        )
    return transform_infos


@cache
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
) -> list[TransformInfo]:
    return _gen_transform_infos_non_cached(src_spec, dst_spec)


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
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

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    if _are_we_tracing():
        transform_infos = _gen_transform_infos_non_cached(current_spec, target_spec)
    else:
        transform_infos = _gen_transform_infos(current_spec, target_spec)

    debug_mode = get_active_debug_mode()
    redistribute_context = (
        debug_mode.record_redistribute_calls(  # type: ignore[union-attr]
            local_tensor, current_spec, target_spec
        )
        if debug_mode is not None
        else contextlib.nullcontext()
    )

    with redistribute_context:
        for transform_info in transform_infos:
            if isinstance(transform_info, PerDimTransformInfo):
                i = transform_info.mesh_dim
                current, target = transform_info.src_dst_placements
                device_mesh.size(mesh_dim=i)

                if current == target:
                    # short cut, just use the original local tensor
                    new_local_tensor = local_tensor
                    continue

                logger.debug(
                    "redistribute from %s to %s on mesh dim %s", current, target, i
                )

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
                            local_tensor, device_mesh, i, my_coordinate[i]
                        )
                    else:
                        assert current.is_shard(), (
                            f"Current placement should be shard but found {current}"
                        )
                        shard_spec = cast(Shard, current)
                        if shard_spec.dim != target_placement.dim:
                            new_local_tensor = shard_spec._to_new_shard_dim(
                                local_tensor,
                                device_mesh,
                                i,
                                transform_info.logical_shape,
                                target_placement.dim,
                            )
                elif target.is_partial():
                    if current.is_replicate():
                        partial_spec = cast(Partial, target)
                        # skip the replicate to partial transformation when we are in backward pass
                        # In this case we keep the grad as replicate, this is because we don't
                        # want to convert the replicated gradients back to partial, although
                        # that's logically conform with the same layout, converting the gradients
                        # back to partial is actually useless as you would have to do reduce later
                        # which would be more expensive than keeping it replicate! For this reason,
                        # we keep the replicate grad here.
                        new_local_tensor = (
                            partial_spec._partition_value(local_tensor, device_mesh, i)
                            if not is_backward
                            else local_tensor
                        )
                    elif current.is_shard():
                        if not is_backward:
                            raise RuntimeError(
                                f"redistribute from {current} to {target} not supported yet"
                            )
                        # for backward shard -> partial, we just need to convert the shard to replicate
                        current_placement = cast(Shard, current)
                        new_local_tensor = current_placement._to_replicate_tensor(
                            local_tensor, device_mesh, i, transform_info.logical_shape
                        )
                    else:
                        # partial -> partial no op, should never hit
                        new_local_tensor = local_tensor

            elif isinstance(transform_info, AllPermuteTransformInfo):
                new_local_tensor = all_permute_mesh_dim(
                    local_tensor,
                    current_spec.shape,
                    transform_info.src_distribution,
                    transform_info.dst_distribution,
                    device_mesh,
                )
            else:
                raise NotImplementedError
        if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
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
        shard_order: Optional[TOrder] = None,
        async_op: bool = False,
        forward_dtype: Optional[torch.dtype] = None,
        backward_dtype: Optional[torch.dtype] = None,
    ):
        ctx.async_op = async_op
        ctx.backward_dtype = backward_dtype
        ctx.original_dtype = input._local_tensor.dtype

        if forward_dtype is not None and forward_dtype != input._local_tensor.dtype:
            local_tensor = input._local_tensor.to(dtype=forward_dtype)
            current_spec = DTensorSpec(
                mesh=device_mesh,
                placements=input._spec.placements,
                shard_order=input._spec.shard_order,
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
                device_mesh,
                placements,
                shard_order=shard_order,
                tensor_meta=current_spec.tensor_meta,
            )
            output = redistribute_local_tensor(
                local_tensor,
                current_spec,
                target_spec,
                async_op=async_op,
            )
        else:
            # use the same local tensor if placements are the same.
            output = local_tensor
            target_spec = current_spec

        return dtensor.DTensor(
            output,
            target_spec,
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

        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
            is_backward=True,
        )

        if output.dtype != ctx.original_dtype:
            output = output.to(ctx.original_dtype)

        # normalize the target placement to replicate if it is partial
        normalized_placements: list[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                # keep target placement to replicate instead of partial in this case
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        spec = DTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=output.dtype,
            ),
        )
        output_dtensor = dtensor.DTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return (
            output_dtensor,
            None,
            None,
            None,
            None,
            None,
            None,
        )
