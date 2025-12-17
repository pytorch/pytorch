#  Copyright (c) Meta Platforms, Inc. and affiliates
import heapq
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, cast, TypeAlias, TypeVar

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    PlacementList,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    generate_redistribute_costs,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement,
    Replicate,
    Shard,
)


logger = logging.getLogger(__name__)


class _ShardingPlaceholder:
    """
    A placeholder for a sharding placement that has a specified tensor dim, but the other
    metadata (e.g. split factor if it's a StridedShard) will be filled in later.
    """

    dim: int

    def __init__(self, dim: int):
        self.dim = dim

    def __repr__(self) -> str:
        return f"_ShardingPlaceholder(dim={self.dim})"


_StrategyTypeT = TypeVar("_StrategyTypeT", bound=StrategyType)
_PlacementT = TypeVar("_PlacementT", bound=Placement)
_ShardingPlaceholderT = TypeVar("_ShardingPlaceholderT", bound=_ShardingPlaceholder)
_SingleDimStrategyFunc: TypeAlias = Callable[
    [OpOverload, ArgsType, KwargsType], list[list[_PlacementT | _ShardingPlaceholderT]]
]
_ExpandedSingleDimStrategyFunc: TypeAlias = Callable[
    [OpOverload, ArgsType, KwargsType], _StrategyTypeT
]


def _insert_single_dim_replication_strategy(
    single_dim_strategies_with_placeholders: list[
        list[Placement | _ShardingPlaceholder]
    ],
    num_input_tensors: int,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """
    Inserts the [Replicate(), Replicate(), ...] strategy after asserting that such strategy does not yet exist.
    """
    for i, strategy in enumerate(single_dim_strategies_with_placeholders):
        assert not all(isinstance(p, Replicate) for p in strategy)
    single_dim_strategies_with_placeholders.append(
        [Replicate()] * (1 + num_input_tensors)
    )
    return single_dim_strategies_with_placeholders


def _fill_single_dim_strategy_placeholders(
    unique_input_placements: set[Placement],
    single_dim_strategies_with_placeholders: list[
        list[Placement | _ShardingPlaceholder]
    ],
) -> list[list[Placement]]:
    """
    Replace any _ShardingPlaceholder with the specific Sharding types used by the inputs in op_schema.
    Supports implicit replication.

    Example:
    single_dim_strategies_with_placeholders = [[Partial(), _ShardingPlaceholder(1), _ShardingPlaceholder(0)]]
    input0: Shard(0)
    input1: StridedShard(1, split_factor=2)
    returns: [
       [Partial(), Shard(1), Shard(0)],
       [Partial(), StridedShard(1, split_factor=2), StridedShard(0, split_factor=2)],
       [Replicate(), Replicate(), Replicate()]
    ]
    """
    shard_builders: dict[str, Callable[[int], Placement]] = {}
    for placement in unique_input_placements:
        if isinstance(placement, _StridedShard):
            key = f"StridedShard(sf={placement.split_factor})"
            if key not in shard_builders:
                sf = placement.split_factor
                shard_builders[key] = lambda tensor_dim: _StridedShard(
                    tensor_dim, split_factor=sf
                )
        elif isinstance(placement, Shard):
            key = "Shard()"
            if key not in shard_builders:
                shard_builders[key] = lambda tensor_dim: Shard(tensor_dim)

    # if any of the placements is a placeholder, we need to expand the strategy
    # to all possible combinations of placements
    expanded_strategies_over_one_mesh_dim: list[list[Placement]] = []
    for s in single_dim_strategies_with_placeholders:
        if any(isinstance(p, _ShardingPlaceholder) for p in s):
            for shard_builder in shard_builders.values():
                expanded_strategy: list[Placement] = []
                for maybe_placeholder in s:
                    if isinstance(maybe_placeholder, _ShardingPlaceholder):
                        # we combine the tensor dim to shard from the placeholder
                        # with other metadata (e.g. split_factor) from the sharding class
                        expanded_strategy.append(shard_builder(maybe_placeholder.dim))
                    else:
                        assert isinstance(maybe_placeholder, Placement)
                        expanded_strategy.append(maybe_placeholder)
                expanded_strategies_over_one_mesh_dim.append(expanded_strategy)
        else:
            assert all(isinstance(p, Placement) for p in s)
            expanded_strategies_over_one_mesh_dim.append(cast(list[Placement], (s)))

    return expanded_strategies_over_one_mesh_dim


def _get_unique_placements(op_schema: OpSchema) -> set[Placement]:
    unique_placements = set()

    def _update_placements(obj: Any):
        if isinstance(obj, DTensorSpec):
            unique_placements.update(obj.placements)
        elif isinstance(obj, OpStrategy):
            assert len(obj.strategies) == 1
            unique_placements.update(obj.strategies[0].output_spec.placements)
        elif isinstance(obj, TupleStrategy):
            for child in obj.children:
                _update_placements(child)

    for obj in op_schema.args_schema:
        _update_placements(obj)

    return unique_placements


def _get_num_tensor_inputs(op_schema: OpSchema) -> int:
    num_inputs = 0
    for obj in op_schema.args_schema:
        if isinstance(obj, OpStrategy):
            num_inputs += 1
        elif isinstance(obj, TupleStrategy):
            num_inputs += len(obj.children)
    return num_inputs


def _expand_single_dim_strategy_to_mesh(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_dim_strategy: _SingleDimStrategyFunc,
) -> _ExpandedSingleDimStrategyFunc:
    """
    Expands the single_mesh_dim impl across all mesh dims, and expands ShardingPlacholder into all
    sharding types used by inputs.

    This supports functional correctness but will generate all possible combinations, which is prohibitively expensive
    for larger numbers of mesh dimensions.

    The expanded_strategy function accesses both the args_schema/kwargs_schema, which contains TensorMeta in place of
    tensor arguments, but also the op_schema which contains OpStrategy in place of Tensor args.
    """

    def _expanded_strategy(
        op_schema: OpSchema,
    ) -> Callable[[OpOverload, ArgsType, KwargsType], StrategyType]:
        def expanded_strategy(
            op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
        ) -> StrategyType:
            # Note: op_schema vs [args_schema, kwargs_schema]
            # -----------------------------------------------
            # Inside `expanded_strategy function we purposefully have access to 2 similar structures.
            # 1) (op, args_schema, kwargs_schema): This is all the single_dim_strategy is allowed to see.
            # importantly, it does not contain information about input placements or meshes - just TensorMeta.
            # 2) op_schema - captured from the parent scope, this contains the input placement and mesh info, needed
            # to actually perform expansion.
            unique_input_placements = _get_unique_placements(op_schema)
            num_inputs = _get_num_tensor_inputs(op_schema)

            # Note: Trees vs Flat Lists
            # -------------------------
            # op_schema.args_schema may contain a TupleStrategy with child strategies for List[Tensor] inputs.
            # args_schema has corresponding TupleStrategy, but with TensorSpec in place of child strategies.
            # CURRENTLY: single_dim_strategy will return a flat list of Placements for each strategy, where any
            # input tuple strategies have been inlined.  I'm not sure if we want to keep doing this, or preserve a pytree
            # structure here.  I'm following the convention in the current DTensor sharding strategies for now.
            # Inside expanded_strategy, we need to carefully align the OpStrategies / Specs from op_schema which are _not_
            # flattened, with the flat Placement list returned from single_dim strategy.
            strategies_over_one_mesh_dim = single_dim_strategy(
                op, args_schema, kwargs_schema
            )
            strategies_over_one_mesh_dim = _insert_single_dim_replication_strategy(
                strategies_over_one_mesh_dim, num_inputs
            )
            expanded_strategies_over_one_mesh_dim = (
                _fill_single_dim_strategy_placeholders(
                    unique_input_placements, strategies_over_one_mesh_dim
                )
            )

            # Note: does not support `allow_unbacked_sharding` which is needed by matmul rules for some compile test
            # currently, we should probably change that test though, since it seems wrong to me to allow sharding unbacked
            # dims
            return expand_to_full_mesh_op_strategy(
                mesh,
                op_schema,
                cast(list[PlacementList], expanded_strategies_over_one_mesh_dim),
            )

        return expanded_strategy

    def _translate_foreach_op_schema(op_schema: OpSchema, index: int) -> OpSchema:
        """Translate foreach op to per-element version of schema."""
        op_parts = str(op_schema.op).split(".")
        base_op_name = op_parts[-2].replace("_foreach_", "")
        foreach_variant = op_parts[-1]

        # select per-element args
        target_args = []
        for arg in op_schema.args_schema:
            if isinstance(arg, TupleStrategy):
                target_args.append(arg.children[index])
            else:  # single arg instead of list
                target_args.append(arg)

        # figure out target op variant
        variant_map = {
            "List": "Tensor",
            "ScalarList": "Scalar",
            "Scalar": "Scalar",
            "Tensor": "Tensor",
            "default": "default",
        }
        target_variant = (
            "default"
            if len(target_args) == 1
            else variant_map.get(foreach_variant, "default")
        )

        # this seems a bit messy
        base_op = getattr(torch.ops.aten, base_op_name)
        target_op = (
            getattr(base_op, target_variant)
            if target_variant in base_op.overloads()
            else base_op.default
        )

        return OpSchema(
            target_op,  # type: ignore[arg-type]
            args_schema=tuple(target_args),
            kwargs_schema=op_schema.kwargs_schema,
        )

    def expanded_foreach_strategy(
        op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> StrategyType:
        tensorlist_len: int | None = None
        for i, obj in enumerate(op_schema.args_schema):
            if isinstance(obj, TupleStrategy):
                if tensorlist_len is None:
                    tensorlist_len = len(obj.children)
                elif len(obj.children) != tensorlist_len:
                    raise AssertionError(
                        f"Expected {tensorlist_len} children in index {i}, but found {len(obj.children)}."
                    )

        if tensorlist_len is None:
            raise AssertionError("Must have at least one tuple input to a foreach op")

        child_strategies: list[StrategyType] = []
        for tensorlist_i in range(tensorlist_len):
            per_index_schema = _translate_foreach_op_schema(op_schema, tensorlist_i)
            per_index_strategy = _expanded_strategy(per_index_schema)
            child_strategies.append(
                per_index_strategy(
                    op, per_index_schema.args_meta, per_index_schema.kwargs_meta
                )
            )

        return TupleStrategy(children=child_strategies)

    # TODO maybe this could be helped by adding a new 'tag' to the OpOverload?
    # Also, i'm guessing that i'll need more info from the registration callsite
    # about which inputs are expected to be lists vs tensors. But maybe I can just infer it all from the runtime
    # inputs?
    if op_schema.op.name().startswith("aten::_foreach_"):
        return expanded_foreach_strategy

    return _expanded_strategy(op_schema)


def _cost_helper(src_placements, dst_placements, mesh, tensor_meta):
    current_spec = DTensorSpec(
        mesh,
        src_placements,
        tensor_meta=tensor_meta,
    )
    new_spec = DTensorSpec(
        mesh,
        dst_placements,
        tensor_meta=tensor_meta,
    )
    return redistribute_cost(current_spec, new_spec)


def _find_lowest_cost_sharding(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_dim_strategy: Callable[
        [OpOverload, ArgsType, KwargsType], list[list[Placement | _ShardingPlaceholder]]
    ],
) -> StrategyType:
    """
    Find the lowest cost sharding for the given op_schema.

    This solves the runtime complexity problem of using _expand_single_dim_strategy_to_mesh by avoiding enumerating
    the product of the single dim strategy over all the mesh dims and then searching the enumerated list for the min
    cost.  Instead, it starts from the input placements and expands a search from there, starting by checking if
    the input placements already describe a legal placement on all mesh dims, and then if not, iterating by taking
    the lowest-cost redistributions in priority-queue order.

    Initial state:
        input placements are in the pq with redistribute cost 0

    Loop over pq:
        pop the lowest cost state from the pq
        if it's a match for any of our single-dim strategies, return it
            - a match is computed by checking each mesh-dim exactly matches one of the single-dim strategies
        else, add all possible next states to the pq
            - next states are computed by taking the current state and redistributing exactly one input tensor
            - each possible next state is added to the pq with the cost of the redistribution
            - next states seen before are not added to the pq

    """
    # Generate single-dim strategies with placeholders
    single_dim_strategies_with_placeholders = single_dim_strategy(
        op_schema.op, op_schema.args_meta, op_schema.kwargs_meta
    )

    # Expand placeholders to get concrete strategies for one mesh dimension
    unique_input_placements = _get_unique_placements(op_schema)
    expanded_strategies_over_one_mesh_dim = _fill_single_dim_strategy_placeholders(
        unique_input_placements, single_dim_strategies_with_placeholders
    )

    # we only consider redistributing to the types of shardings allowed by the expanded rules for each input
    def is_sharding(p: Placement):
        return isinstance(p, (Shard, _StridedShard))

    allowed_sharding_per_input = defaultdict(set)
    for strategy in expanded_strategies_over_one_mesh_dim:
        for input_idx in range(len(strategy) - 1):
            p = strategy[1 + input_idx]
            if is_sharding(p):
                allowed_sharding_per_input[input_idx].add(p)

    logger.debug("Allowed sharding per input idx: %s", allowed_sharding_per_input)

    # Extract input DTensorSpecs from op_schema.args_schema
    input_specs = [
        spec for spec in op_schema.args_schema if isinstance(spec, DTensorSpec)
    ]
    assert len(input_specs) > 0, "broken input"
    # Build src_strategies wrapping input specs for redistribute cost computation
    src_strategies = [OpStrategy([OpSpec(spec)]) for spec in input_specs]

    # Priority queue: (cost, counter, input_placements_tuple)
    # input_placements_tuple is a tuple of placements tuples, one per input tensor
    # Each input's placements tuple has one placement per mesh dimension
    counter: int = 0
    pq: list[tuple[float, int, tuple[tuple[Placement, ...], ...]]] = []
    visited: set[tuple[tuple[Placement, ...], ...]] = set()

    # Start from the current input placements (cost 0 since no redistribution needed)
    initial_input_placements = tuple(spec.placements for spec in input_specs)
    heapq.heappush(pq, (0.0, counter, initial_input_placements))
    counter += 1

    # Explore priority queue to find the first valid strategy
    while pq:
        cost, counter_, input_placements_tuple = heapq.heappop(pq)
        logger.debug(
            "Checking counter=%d cost=%f input_placements_tuple=%s",
            counter_,
            cost,
            input_placements_tuple,
        )

        # Check if current input placements match one of the single-dim strategies for each mesh dim
        # For each mesh dim, we check if the input placements at that dim match any strategy
        selected_output_placements = []
        is_match = True

        for mesh_dim in range(mesh.ndim):
            # Get input placements for this mesh dimension
            input_placements_for_dim = tuple(
                placements[mesh_dim] for placements in input_placements_tuple
            )

            # Check if this matches any single-dim strategy
            match_found = False
            for strategy in expanded_strategies_over_one_mesh_dim:
                # strategy is [output_placement, input1_placement, input2_placement, ...]
                strategy_input_placements = tuple(strategy[1:])
                if strategy_input_placements == input_placements_for_dim:
                    # Found a match for this mesh dim
                    selected_output_placements.append(strategy[0])
                    match_found = True
                    break

            if not match_found:
                is_match = False
                break

        if is_match:
            # All mesh dims matched! Create the result DTensorSpec
            output_placements_tuple = tuple(selected_output_placements)

            # Build input specs from current placements
            arg_specs = [
                DTensorSpec(mesh, placements, tensor_meta=input_spec.tensor_meta)
                for placements, input_spec in zip(input_placements_tuple, input_specs)
            ]

            # Compute redistribute costs from original input specs to required arg_specs
            redistribute_costs = [
                generate_redistribute_costs(src_strategy, arg_spec)
                for src_strategy, arg_spec in zip(src_strategies, arg_specs)
            ]

            # Create and return the first valid OpSpec found
            op_spec = OpSpec(
                output_specs=DTensorSpec(
                    mesh, output_placements_tuple, tensor_meta=None
                ),
                input_specs=arg_specs,
                redistribute_cost=redistribute_costs,
            )

            exhaustive = len(expanded_strategies_over_one_mesh_dim) ** mesh.ndim
            logger.debug(
                "returning cost=%f %s, counter=%d, exhaustive=%d",
                cost,
                op_spec,
                counter,
                exhaustive,
            )
            return OpStrategy([op_spec])

        def can_chunk(placements: tuple[Placement, ...]) -> bool:
            return any(isinstance(p, Replicate) for p in placements)

        def can_allgather(placements: tuple[Placement, ...]) -> bool:
            return any(is_sharding(p) for p in placements)

        # if we can chunk any more, we chunk
        if any(can_chunk(placements) for placements in input_placements_tuple):
            for input_idx in range(len(input_placements_tuple)):
                for mesh_dim in range(mesh.ndim):
                    # if we can chunk, we chunk
                    if isinstance(
                        input_placements_tuple[input_idx][mesh_dim], Replicate
                    ):
                        # copy and convert to list to mutate just this mesh-dim's placements
                        new_input_placements = [
                            list(input_placement)
                            for input_placement in input_placements_tuple
                        ]
                        for sharding in allowed_sharding_per_input[input_idx]:
                            new_input_placements[input_idx][mesh_dim] = sharding
                            new_input_placements_tuple = tuple(
                                tuple(p) for p in new_input_placements
                            )
                            if new_input_placements_tuple in visited:
                                continue
                            new_total_cost = cost + _cost_helper(
                                input_placements_tuple[input_idx],
                                new_input_placements_tuple[input_idx],
                                mesh,
                                input_specs[input_idx].tensor_meta,
                            )
                            visited.add(new_input_placements_tuple)
                            logger.debug(
                                "Pushing chunk (%s) for input %d, mesh_dim %d: "
                                "new_total_cost=%6.2f new_input_placements_tuple=%s, counter=%d",
                                str(sharding),
                                input_idx,
                                mesh_dim,
                                new_total_cost,
                                new_input_placements_tuple,
                                counter,
                            )
                            heapq.heappush(
                                pq,
                                (
                                    new_total_cost,
                                    counter,
                                    new_input_placements_tuple,
                                ),
                            )
                            counter += 1

        # if not, we try all2all
        # TODO

        # if not all2all we try allgather or reducescatter
        # TODO reducescatter
        elif any(can_allgather(placements) for placements in input_placements_tuple):
            for input_idx in range(len(input_placements_tuple)):
                for mesh_dim in range(mesh.ndim):
                    if is_sharding(input_placements_tuple[input_idx][mesh_dim]):
                        new_input_placements = [
                            list(input_placement)
                            for input_placement in input_placements_tuple
                        ]
                        new_input_placements[input_idx][mesh_dim] = Replicate()
                        new_input_placements_tuple = tuple(
                            tuple(p) for p in new_input_placements
                        )
                        if new_input_placements_tuple in visited:
                            continue
                        new_total_cost = cost + _cost_helper(
                            input_placements_tuple[input_idx],
                            new_input_placements_tuple[input_idx],
                            mesh,
                            input_specs[input_idx].tensor_meta,
                        )
                        visited.add(new_input_placements_tuple)
                        logger.debug(
                            "Pushing allgather for input %d, mesh_dim %d: "
                            "new_total_cost=%6.2f new_input_placements_tuple=%s, counter=%d",
                            input_idx,
                            mesh_dim,
                            new_total_cost,
                            new_input_placements_tuple,
                            counter,
                        )
                        heapq.heappush(
                            pq,
                            (
                                new_total_cost,
                                counter,
                                new_input_placements_tuple,
                            ),
                        )
                        counter += 1

        # finally, we try allreduce
        # TODO

    # If we get here, no valid strategy was found - this should not happen
    raise AssertionError(
        f"No valid strategy found for op_schema {op_schema}. "
        f"Explored {len(visited)} strategy combinations."
    )
