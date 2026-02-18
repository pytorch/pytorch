#  Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import heapq
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Optional, TypeAlias, TypeVar, Union

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Placement,
    Replicate,
    Shard,
)
from torch.utils._pytree import tree_map_only


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


@dataclass
class _SingleDimStrategyInfo:
    func: _SingleDimStrategyFunc
    allow_unbacked_sharding: bool | None = field(default=None)

    # Delegate to func so this can be used interchangeably with a raw
    # _SingleDimStrategyFunc (e.g. in tests that call strategy functions directly).
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _insert_single_dim_replication_strategy(
    single_dim_strategies_with_placeholders: list[
        list[Placement | _ShardingPlaceholder]
    ],
    num_outputs: int,
    num_input_tensors: int,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """
    Inserts the [Replicate(), Replicate(), ...] strategy after asserting that such strategy does not yet exist.
    """
    for strategy in single_dim_strategies_with_placeholders:
        assert not all(isinstance(p, Replicate) for p in strategy)
    single_dim_strategies_with_placeholders.insert(
        0, [Replicate()] * (num_outputs + num_input_tensors)
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
                shard_builders[key] = functools.partial(
                    _StridedShard, split_factor=placement.split_factor
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
    # Also include placements from kwargs (e.g., "out" tensor)
    for obj in op_schema.kwargs_schema.values():
        _update_placements(obj)

    return unique_placements


def _get_num_tensor_inputs(op_schema: OpSchema) -> int:
    num_inputs = 0
    for obj in op_schema.args_schema:
        if isinstance(obj, OpStrategy):
            num_inputs += 1
        elif isinstance(obj, TupleStrategy):
            num_inputs += len(obj.children)
    # Also count tensor kwargs (e.g., "out" for out-variant ops)
    for obj in op_schema.kwargs_schema.values():
        if isinstance(obj, OpStrategy):
            num_inputs += 1
        elif isinstance(obj, TupleStrategy):
            num_inputs += len(obj.children)
    return num_inputs


def _expand_single_dim_strategy_to_mesh(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    strategy_info: _SingleDimStrategyInfo,
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
) -> _ExpandedSingleDimStrategyFunc:
    """
    Expands the single_mesh_dim impl across all mesh dims, and expands ShardingPlacholder into all
    sharding types used by inputs.

    This supports functional correctness but will generate all possible combinations, which is prohibitively expensive
    for larger numbers of mesh dimensions.

    The expanded_strategy function accesses both the args_schema/kwargs_schema, which contains TensorMeta in place of
    tensor arguments, but also the op_schema which contains OpStrategy in place of Tensor args.

    Args:
        output_tensor_meta: tensor metadata for the output(s), precomputed during sharding prop
    """
    # Note: circular import, failed to untangle with #168221, reverted
    from torch.distributed.tensor._ops.utils import expand_to_full_mesh_op_strategy

    def _create_expanded_strategy_impl(
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
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

            # Compute num_outputs from output_tensor_meta
            if output_tensor_meta is None:
                num_outputs = 0
            elif isinstance(output_tensor_meta, TensorMeta):
                num_outputs = 1
            else:
                num_outputs = len(output_tensor_meta)

            # Note: Trees vs Flat Lists
            # -------------------------
            # op_schema.args_schema may contain a TupleStrategy with child strategies for List[Tensor] inputs.
            # args_schema has corresponding TupleStrategy, but with TensorSpec in place of child strategies.
            # CURRENTLY: single_dim_strategy will return a flat list of Placements for each strategy, where any
            # input tuple strategies have been inlined.  I'm not sure if we want to keep doing this, or preserve a pytree
            # structure here.  I'm following the convention in the current DTensor sharding strategies for now.
            # Inside expanded_strategy, we need to carefully align the OpStrategies / Specs from op_schema which are _not_
            # flattened, with the flat Placement list returned from single_dim strategy.
            strategies_over_one_mesh_dim = strategy_info.func(
                op, args_schema, kwargs_schema
            )
            strategies_over_one_mesh_dim = _insert_single_dim_replication_strategy(
                strategies_over_one_mesh_dim, num_outputs, num_inputs
            )
            expanded_strategies_over_one_mesh_dim = (
                _fill_single_dim_strategy_placeholders(
                    unique_input_placements, strategies_over_one_mesh_dim
                )
            )

            # Detect inplace ops by checking if the base op name ends with '_'
            op_name = op.name()
            base_name = op_name.split("::")[1].split(".")[0]
            is_inplace = base_name.endswith("_")

            return expand_to_full_mesh_op_strategy(
                mesh,
                op_schema,
                cast(list[PlacementList], expanded_strategies_over_one_mesh_dim),
                output_tensor_meta=output_tensor_meta,
                inplace_op=is_inplace,
                input_index=num_outputs,
                allow_unbacked_sharding=strategy_info.allow_unbacked_sharding,
            )

        return expanded_strategy

    # Create a cached version of the impl
    _cached_create_expanded_strategy = functools.lru_cache(
        _create_expanded_strategy_impl
    )

    def _create_expanded_strategy(
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
    ) -> Callable[[OpOverload, ArgsType, KwargsType], StrategyType]:
        # Try to use cache, but fall back to uncached version if hashing fails
        # (e.g., when TensorMeta contains SymInts from dynamic shapes)
        try:
            return _cached_create_expanded_strategy(op_schema, output_tensor_meta)
        except TypeError:
            # Unhashable types (SymInts), skip caching
            return _create_expanded_strategy_impl(op_schema, output_tensor_meta)

    def _translate_foreach_op_schema(
        op_schema: OpSchema, output_tensor_meta: Sequence[TensorMeta], index: int
    ) -> tuple[OpSchema, TensorMeta]:
        """Translate foreach op to per-element version of schema."""
        op_parts = str(op_schema.op).split(".")
        base_op_name = op_parts[-2].replace("_foreach_", "")
        foreach_variant = op_parts[-1]

        # select per-element inputs, outputs
        target_args, target_kwargs = tree_map_only(
            TupleStrategy,
            lambda x: x.children[index],
            (op_schema.args_schema, op_schema.kwargs_schema),
            is_leaf=lambda x: isinstance(x, TupleStrategy),
        )
        target_output_meta = output_tensor_meta[index]

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

        op_schema = OpSchema(
            target_op,  # type: ignore[arg-type]
            args_schema=tuple(target_args),
            kwargs_schema=op_schema.kwargs_schema,
        )
        return op_schema, target_output_meta

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
            per_index_schema, per_index_output_meta = _translate_foreach_op_schema(
                op_schema,
                output_tensor_meta,  # type: ignore[arg-type]
                tensorlist_i,
            )
            per_index_strategy = _create_expanded_strategy(
                per_index_schema, per_index_output_meta
            )
            child_strategies.append(
                per_index_strategy(
                    op, per_index_schema.args_meta, per_index_schema.kwargs_meta
                )
            )

        return TupleStrategy(children=child_strategies)

    # TODO maybe this could be helped by adding a new 'tag' to the OpOverload?
    if op_schema.op.name().startswith("aten::_foreach_"):
        return expanded_foreach_strategy

    return _create_expanded_strategy(op_schema, output_tensor_meta)


def register_single_dim_strategy(
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo] = None,
    allow_unbacked_sharding: bool | None = None,
) -> Callable[[_SingleDimStrategyFunc], _SingleDimStrategyFunc]:
    """
    Registers a single_dim_strategy function for the given op.

    A single_dim_strategy enumerates all the non-trivial sharding specifications for the operator over a single mesh
    dim. Since it generates the full set of valid shardings regardless of the given input placements, tensor inputs
    are represented via TensorMetas instead of OpStrategies like in op_strategy functions. TensorMeta inputs, along
    with all other op inputs (int, float, bool, etc) inform which sharding placements are valid. Single-dim strategies
    are fed into infra that expands them over multiple mesh dims and fills sharding placeholders with concrete sharding
    types.

    Single-dim-strategies should not list the trivial "all Replicate" rule, which is assumed for all ops.

    Sharding placeholders should be used to represent generic sharding rules in single_dim_strategies.  For example,
    most operators that support sharded inputs and outputs do not care how the shards are laid out globally, as long as
    they are consistent.  It is just as valid to run a matmul on (Shard(0), Replicate -> Shard(0)) as on
    (StridedShard(0), Replicate -> StridedShard(0)).  For this reason, we use a 'ShardingPlaceholder' to represent all
    generic sharding types.  Placeholders will be filled by concrete sharding types seen in runtime inputs, not all
    types known to DTensor.

    """
    # Note: circular import, failed to untangle with #168221, reverted
    from torch.distributed.tensor._api import DTensor
    from torch.distributed.tensor._ops.utils import _get_registration_wrapper

    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]
    registration_wrapper = _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_single_dim_op_strategy,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy,
    )

    # Wrap impl in _SingleDimStrategyInfo here rather than adding a generic
    # transform hook to _get_registration_wrapper, so that single-dim-strategy
    # concerns stay in this module and the shared registration util stays simple.
    def wrapper(impl):
        info = _SingleDimStrategyInfo(
            func=impl,
            allow_unbacked_sharding=allow_unbacked_sharding,
        )
        registration_wrapper(info)
        return impl

    return wrapper


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
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
) -> OpStrategy | None:
    """
    Find the lowest cost sharding for the given op_schema.

    This solves the runtime complexity problem of using _expand_single_dim_strategy_to_mesh by avoiding enumerating
    the product of the single dim strategy over all the mesh dims and then searching the enumerated list for the min
    cost.  Instead, it starts from the input placements and expands a search from there, starting by checking if
    the input placements already describe a legal placement on all mesh dims, and then if not, iterating by taking
    the lowest-cost redistributions in priority-queue order.

    Returns None if any input has _StridedShard placement, signaling the caller to fall back to full expansion.
    """
    # Note: circular import
    from torch.distributed.tensor._ops.utils import (
        generate_redistribute_costs,
        is_tensor_shardable,
    )
    from torch.distributed.tensor.placement_types import Partial

    # Extract input DTensorSpecs from op_schema.args_schema, handling both
    # raw DTensorSpec and OpStrategy wrappers (the caller wraps specs in OpStrategy)
    input_specs: list[DTensorSpec] = []
    for arg in op_schema.args_schema:
        if isinstance(arg, DTensorSpec):
            input_specs.append(arg)
        elif isinstance(arg, OpStrategy):
            assert len(arg.strategies) == 1
            input_specs.append(arg.strategies[0].output_spec)

    assert len(input_specs) > 0, "broken input"
    num_inputs = len(input_specs)

    # Fall back to full expansion if any input has _StridedShard
    for spec in input_specs:
        if any(isinstance(p, _StridedShard) for p in spec.placements):
            return None

    # Generate single-dim strategies with placeholders
    single_dim_strategies_with_placeholders = single_dim_strategy(
        op_schema.op, op_schema.args_meta, op_schema.kwargs_meta
    )

    # Compute num_outputs from strategy structure or output_tensor_meta
    if len(single_dim_strategies_with_placeholders) > 0:
        num_outputs = len(single_dim_strategies_with_placeholders[0]) - num_inputs
    elif output_tensor_meta is None:
        num_outputs = 0
    elif isinstance(output_tensor_meta, TensorMeta):
        num_outputs = 1
    else:
        num_outputs = len(output_tensor_meta)

    # Insert the all-Replicate strategy
    single_dim_strategies_with_placeholders = _insert_single_dim_replication_strategy(
        single_dim_strategies_with_placeholders, num_outputs, num_inputs
    )

    # Expand placeholders to get concrete strategies for one mesh dimension
    unique_input_placements = _get_unique_placements(op_schema)
    expanded_strategies_over_one_mesh_dim = _fill_single_dim_strategy_placeholders(
        unique_input_placements, single_dim_strategies_with_placeholders
    )

    def is_sharding(p: Placement) -> bool:
        return isinstance(p, (Shard, _StridedShard))

    # Precompute allowed placements per input from the expanded rules
    allowed_sharding_per_input: dict[int, set[Placement]] = defaultdict(set)
    allowed_partial_per_input: dict[int, set[Placement]] = defaultdict(set)
    for strategy in expanded_strategies_over_one_mesh_dim:
        for input_idx in range(num_inputs):
            p = strategy[num_outputs + input_idx]
            if is_sharding(p):
                allowed_sharding_per_input[input_idx].add(p)
            elif isinstance(p, Partial):
                allowed_partial_per_input[input_idx].add(p)

    logger.debug("Allowed sharding per input idx: %s", allowed_sharding_per_input)
    logger.debug("Allowed partial per input idx: %s", allowed_partial_per_input)

    # Build strategy lookup: map input placements -> output placements for fast matching
    strategy_lookup: dict[tuple[Placement, ...], tuple[Placement, ...]] = {}
    for strategy in expanded_strategies_over_one_mesh_dim:
        input_key = tuple(strategy[num_outputs:])
        if input_key not in strategy_lookup:
            strategy_lookup[input_key] = tuple(strategy[:num_outputs])

    # Build src_strategies wrapping input specs for redistribute cost computation
    src_strategies = [OpStrategy([OpSpec(spec)]) for spec in input_specs]

    # Resolve output tensor_meta per output index
    if output_tensor_meta is None:
        output_metas: tuple[TensorMeta | None, ...] = (None,) * max(num_outputs, 0)
    elif isinstance(output_tensor_meta, TensorMeta):
        output_metas = (output_tensor_meta,)
    else:
        output_metas = tuple(output_tensor_meta)

    # PQ entry: (cost, counter, input_placements_tuple, transitions)
    # cost is the total redistribute cost from original placements to current state
    # transitions: list of (input_idx, mesh_dim, src_placement, dst_placement)
    _Transitions = list[tuple[int, int, Placement, Placement]]
    counter: int = 0
    pq: list[tuple[float, int, tuple[tuple[Placement, ...], ...], _Transitions]] = []
    visited: set[tuple[tuple[Placement, ...], ...]] = set()

    initial_input_placements = tuple(spec.placements for spec in input_specs)
    heapq.heappush(pq, (0.0, counter, initial_input_placements, []))
    counter += 1

    def _total_cost(
        input_placements_tuple: tuple[tuple[Placement, ...], ...],
    ) -> float:
        """Compute total redistribute cost from original to candidate placements."""
        total = 0.0
        for input_idx in range(num_inputs):
            total += _cost_helper(
                initial_input_placements[input_idx],
                input_placements_tuple[input_idx],
                mesh,
                input_specs[input_idx].tensor_meta,
            )
        return total

    def _push_neighbor(
        input_idx: int,
        mesh_dim: int,
        new_placement: Placement,
        current_placements: tuple[tuple[Placement, ...], ...],
        current_transitions: _Transitions,
    ) -> None:
        nonlocal counter
        new_input_placements = [list(ps) for ps in current_placements]
        old_placement = new_input_placements[input_idx][mesh_dim]
        new_input_placements[input_idx][mesh_dim] = new_placement
        new_tuple = tuple(tuple(ps) for ps in new_input_placements)
        if new_tuple in visited:
            return
        new_cost = _total_cost(new_tuple)
        new_transitions = current_transitions + [
            (input_idx, mesh_dim, old_placement, new_placement)
        ]
        heapq.heappush(pq, (new_cost, counter, new_tuple, new_transitions))
        counter += 1

    while pq:
        cost, counter_, input_placements_tuple, transitions = heapq.heappop(pq)

        if input_placements_tuple in visited:
            continue
        visited.add(input_placements_tuple)

        # Check if current input placements match a single-dim strategy on every mesh dim
        selected_output_placements: list[tuple[Placement, ...]] = []
        is_match = True

        for mesh_dim in range(mesh.ndim):
            input_placements_for_dim = tuple(
                placements[mesh_dim] for placements in input_placements_tuple
            )
            output_for_dim = strategy_lookup.get(input_placements_for_dim)
            if output_for_dim is not None:
                selected_output_placements.append(output_for_dim)
            else:
                is_match = False
                break

        if is_match:
            # Build input specs for the candidate match
            arg_specs = [
                DTensorSpec(mesh, placements, tensor_meta=input_spec.tensor_meta)
                for placements, input_spec in zip(input_placements_tuple, input_specs)
            ]

            # Check all inputs are shardable with these placements
            shardable = all(
                is_tensor_shardable(spec.tensor_meta.shape, spec)
                for spec in arg_specs
                if spec.tensor_meta is not None
            )

            if shardable:
                # Build multi-dim output placements: transpose from per-mesh-dim to per-output
                if num_outputs == 1:
                    output_placements_tuple = tuple(
                        out[0] for out in selected_output_placements
                    )
                    out_meta = output_metas[0] if output_metas else None
                    output_spec: DTensorSpec | tuple[DTensorSpec | None, ...] = (
                        DTensorSpec(mesh, output_placements_tuple, tensor_meta=out_meta)
                    )
                elif num_outputs > 1:
                    multi_output_specs: list[DTensorSpec | None] = []
                    for out_idx in range(num_outputs):
                        out_placements = tuple(
                            out[out_idx] for out in selected_output_placements
                        )
                        meta = (
                            output_metas[out_idx]
                            if out_idx < len(output_metas)
                            else None
                        )
                        multi_output_specs.append(
                            DTensorSpec(mesh, out_placements, tensor_meta=meta)
                        )
                    output_spec = tuple(multi_output_specs)
                else:
                    output_spec = DTensorSpec(mesh, (Replicate(),) * mesh.ndim)

                redistribute_costs = [
                    generate_redistribute_costs(src_strategy, arg_spec)
                    for src_strategy, arg_spec in zip(src_strategies, arg_specs)
                ]

                op_spec = OpSpec(
                    output_specs=output_spec,
                    input_specs=arg_specs,
                    redistribute_cost=redistribute_costs,
                )

                exhaustive = len(expanded_strategies_over_one_mesh_dim) ** mesh.ndim
                logger.debug(
                    "returning cost=%f %s, counter=%d, exhaustive=%d, transitions=%s",
                    cost,
                    op_spec,
                    counter,
                    exhaustive,
                    transitions,
                )
                result = OpStrategy([op_spec])
                result._pq_transitions = transitions  # type: ignore[attr-defined]
                return result

        # Generate all neighbor states (independent blocks, not if/elif)
        for input_idx in range(len(input_placements_tuple)):
            for mesh_dim in range(mesh.ndim):
                current_p = input_placements_tuple[input_idx][mesh_dim]

                # Replicate -> Shard (local chunk, free)
                if isinstance(current_p, Replicate):
                    for sharding in allowed_sharding_per_input[input_idx]:
                        _push_neighbor(
                            input_idx,
                            mesh_dim,
                            sharding,
                            input_placements_tuple,
                            transitions,
                        )
                    # Replicate -> Partial (local, free)
                    for partial in allowed_partial_per_input[input_idx]:
                        _push_neighbor(
                            input_idx,
                            mesh_dim,
                            partial,
                            input_placements_tuple,
                            transitions,
                        )

                # Shard -> Replicate (allgather)
                if is_sharding(current_p):
                    _push_neighbor(
                        input_idx,
                        mesh_dim,
                        Replicate(),
                        input_placements_tuple,
                        transitions,
                    )
                    # Shard -> different Shard (all-to-all)
                    for sharding in allowed_sharding_per_input[input_idx]:
                        if sharding != current_p:
                            _push_neighbor(
                                input_idx,
                                mesh_dim,
                                sharding,
                                input_placements_tuple,
                                transitions,
                            )

                # Partial -> Replicate (allreduce)
                if isinstance(current_p, Partial):
                    _push_neighbor(
                        input_idx,
                        mesh_dim,
                        Replicate(),
                        input_placements_tuple,
                        transitions,
                    )
                    # Partial -> Shard (reduce-scatter)
                    for sharding in allowed_sharding_per_input[input_idx]:
                        _push_neighbor(
                            input_idx,
                            mesh_dim,
                            sharding,
                            input_placements_tuple,
                            transitions,
                        )

    raise AssertionError(
        f"No valid strategy found for op_schema {op_schema}. "
        f"Explored {len(visited)} strategy combinations."
    )
