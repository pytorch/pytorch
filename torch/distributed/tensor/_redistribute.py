# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import contextlib
import dataclasses
import logging
import weakref
from functools import cache
from typing import cast, NamedTuple, Optional

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.tensor._api as dtensor
from torch.distributed._functional_collectives import _are_we_tracing
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


class _TransformInfo(NamedTuple):
    mesh_dim: int
    src_dst_placements: tuple[Placement, Placement]
    # logical_shape on this mesh dimension
    logical_shape: list[int]


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
    ) -> None:
        # only initialize once
        if getattr(self, "_initialized", False):
            return
        self.device_mesh = device_mesh
        self.coordinate = device_mesh.get_coordinate()
        assert self.coordinate is not None
        self.tensor_dimension = tensor_dimension
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

        # list of [DistState, cost]
        all_next_state: dict[DTensorRedistributePlanner.DistState, int] = {}

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
                    all_next_state[dist_state] = self.all_to_all_cost
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
                all_next_state[dist_state] = self.all_gather_cost

        ######################################################################
        # handle case 3: Partial() -> Replicate()
        for src_tensor_dim in range(self.tensor_dimension):
            if isinstance(src_tensor_dim, Partial):
                new_placements = list(placements)
                new_placements[src_tensor_dim] = Replicate()
                dist_state = self.DistState(
                    self._to_tuple(new_placements), tensor_dim_mesh_dim
                )
                all_next_state[dist_state] = self.all_gather_cost

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
                all_next_state[dist_state] = self.chunk_cost

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
                all_next_state[dist_state] = self.reduce_scatter

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
            all_next_state[dist_state] = self.chunk_cost

        return all_next_state

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
                int,
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
    ):
        new_logical_shape = list(full_tensor_shape)
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
        full_tensor_shape: tuple[int, ...],
        src_shard_order: TOrder,
        dst_shard_order: TOrder,
    ) -> list[_TransformInfo]:
        src_state = self.DistState(src_spec.placements, src_shard_order)
        dst_state = self.DistState(dst_spec.placements, dst_shard_order)
        transform_infos: list[_TransformInfo] = []
        state_path = self.find_min_cost_path(src_state, dst_state)
        logger.debug(
            "Path from %s to %s: \n%s",
            src_state,
            dst_state,
            " -> ".join(str(s) for s in state_path),
        )
        for cur_state, nxt_state in zip(state_path[:-1], state_path[1:]):
            # find the mesh_dim that is different between cur_state and nxt_state
            if cur_state.placements != nxt_state.placements:
                # skip the transition of device order permutation (no-op)
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
        the former is a nested-sharding of a tensor already already sharded dimension 0, whereras
        the latter is the first sharding on tensor dimension 0.
        """
        # logical shape records the logic tensor shape on the mesh dimension
        # this is useful to ensure uneven sharding gets correct output shape
        initial_logical_shape = list(src_spec.shape)
        mesh_dims_to_logical_shape = [initial_logical_shape]
        transform_infos: list[_TransformInfo] = []
        if self.device_mesh.ndim == 1:
            # if device_mesh is 1D, redistribute is a simple direct transformation
            transform_infos.append(
                _TransformInfo(
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
                        _TransformInfo(
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
                    _TransformInfo(
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
    src_shard_order: TOrder,
    dst_shard_order: TOrder,
) -> list[_TransformInfo]:
    transform_infos: list[_TransformInfo] = []
    device_mesh = src_spec.device_mesh
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
            src_spec, dst_spec, src_spec.shape, src_shard_order, dst_shard_order
        )
    return transform_infos


@cache
def _gen_transform_infos(
    src_spec: DTensorSpec,
    dst_spec: DTensorSpec,
    src_device_order: TOrder,
    dst_device_order: TOrder,
) -> list[_TransformInfo]:
    return _gen_transform_infos_non_cached(
        src_spec, dst_spec, src_device_order, dst_device_order
    )


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    src_shard_order: Optional[TOrder] = None,
    dst_shard_order: Optional[TOrder] = None,
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

    if not src_shard_order:
        src_shard_order = cast(
            TOrder,
            DTensorSpec.compute_default_shard_order(
                current_spec.placements, current_spec.mesh, local_tensor.ndim
            ),
        )
    if not dst_shard_order:
        dst_shard_order = cast(
            TOrder,
            DTensorSpec.compute_default_shard_order(
                target_spec.placements, target_spec.mesh, local_tensor.ndim
            ),
        )

    new_local_tensor = local_tensor
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    if _are_we_tracing():
        transform_infos = _gen_transform_infos_non_cached(
            current_spec, target_spec, src_shard_order, dst_shard_order
        )
    else:
        transform_infos = _gen_transform_infos(
            current_spec, target_spec, src_shard_order, dst_shard_order
        )

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
        shard_order: Optional[TOrder] = None,
        async_op: bool = False,
        forward_dtype: Optional[torch.dtype] = None,
        backward_dtype: Optional[torch.dtype] = None,
    ):
        ctx.async_op = async_op
        ctx.backward_dtype = backward_dtype
        ctx.original_dtype = input._local_tensor.dtype
        ctx.original_device_order = input._spec.shard_order

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
                src_shard_order=input._spec.shard_order,
                dst_shard_order=shard_order,
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
            src_shard_order=current_spec.shard_order,
            dst_shard_order=previous_spec.shard_order,
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
