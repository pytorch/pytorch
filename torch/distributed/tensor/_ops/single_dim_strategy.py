#  Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import heapq
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Optional, TypeAlias, TypeVar, Union
from typing_extensions import TypeIs

import torch
from torch._ops import OpOverload
from torch.distributed.tensor._collective_utils import (
    _compute_placement_transition_cost,
    MeshTopoInfo,
)
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


def _is_sharding(p: Placement) -> TypeIs[Shard | _StridedShard]:
    return isinstance(p, (Shard, _StridedShard))


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


def _build_output_specs(
    mesh: DeviceMesh,
    per_mesh_dim_placements: list[tuple[Placement, ...]],
    num_outputs: int,
    output_metas: tuple[TensorMeta | None, ...],
) -> DTensorSpec | tuple[DTensorSpec | None, ...]:
    """Build output spec(s) by transposing per-mesh-dim placements to per-output.

    per_mesh_dim_placements is indexed [mesh_dim][output_idx]. output_metas must
    have exactly num_outputs elements.
    """
    assert num_outputs > 0
    assert len(output_metas) == num_outputs

    def _placements_for_output(out_idx: int) -> tuple[Placement, ...]:
        return tuple(out[out_idx] for out in per_mesh_dim_placements)

    if num_outputs > 1:
        return tuple(
            DTensorSpec(mesh, _placements_for_output(i), tensor_meta=output_metas[i])
            for i in range(num_outputs)
        )
    else:
        return DTensorSpec(mesh, _placements_for_output(0), tensor_meta=output_metas[0])


class _PreparedSingleDimStrategy:
    """A single-dim strategy materialized for a specific op.

    Expands a strategy function's placeholder-based rules into concrete
    placement rules by filling in the actual shard/partial placements from
    the op_schema. The result is a lookup table (strategy_lookup) that maps
    input placements to output placements for one mesh dimension.

    Provides try_propagate() for matching a multi-dim placement tuple against
    the per-dim rules, and exposes allowed_sharding_per_input /
    allowed_partial_per_input for graph search neighbor generation.
    """

    strategy_lookup: dict[tuple[Placement, ...], tuple[Placement, ...]]
    expanded_strategies: list[list[Placement]]
    num_outputs: int
    num_inputs: int
    output_metas: tuple[TensorMeta | None, ...]
    allowed_sharding_per_input: dict[int, set[Placement]]
    allowed_partial_per_input: dict[int, set[Placement]]
    allow_unbacked_sharding: bool | None

    def __init__(
        self,
        strategy_fn: _SingleDimStrategyInfo
        | Callable[
            [OpOverload, ArgsType, KwargsType],
            list[list[Placement | _ShardingPlaceholder]],
        ],
        op_schema: OpSchema,
        output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None,
        num_inputs: int | None = None,
    ) -> None:
        # Note: circular import
        from torch.distributed.tensor.placement_types import Partial

        if isinstance(strategy_fn, _SingleDimStrategyInfo):
            self.allow_unbacked_sharding = strategy_fn.allow_unbacked_sharding
            func = strategy_fn.func
        else:
            self.allow_unbacked_sharding = None
            func = strategy_fn

        if num_inputs is None:
            num_inputs = _get_num_tensor_inputs(op_schema)
        self.num_inputs = num_inputs

        strategies_with_placeholders = func(
            op_schema.op, op_schema.args_meta, op_schema.kwargs_meta
        )

        # Compute num_outputs from strategy structure or output_tensor_meta
        if len(strategies_with_placeholders) > 0:
            num_outputs = len(strategies_with_placeholders[0]) - num_inputs
        elif output_tensor_meta is None:
            num_outputs = 0
        elif isinstance(output_tensor_meta, TensorMeta):
            num_outputs = 1
        else:
            num_outputs = len(output_tensor_meta)
        self.num_outputs = num_outputs

        strategies_with_placeholders = _insert_single_dim_replication_strategy(
            strategies_with_placeholders, num_outputs, num_inputs
        )

        unique_input_placements = _get_unique_placements(op_schema)
        self.expanded_strategies = _fill_single_dim_strategy_placeholders(
            unique_input_placements, strategies_with_placeholders
        )

        # Build strategy lookup: map input placements -> output placements
        self.strategy_lookup = {}
        for strategy in self.expanded_strategies:
            input_key = tuple(strategy[num_outputs:])
            if input_key not in self.strategy_lookup:
                self.strategy_lookup[input_key] = tuple(strategy[:num_outputs])

        # Precompute allowed placements per input from the expanded rules
        self.allowed_sharding_per_input: dict[int, set[Placement]] = defaultdict(set)
        self.allowed_partial_per_input: dict[int, set[Placement]] = defaultdict(set)
        for strategy in self.expanded_strategies:
            for input_idx in range(num_inputs):
                p = strategy[num_outputs + input_idx]
                if _is_sharding(p):
                    self.allowed_sharding_per_input[input_idx].add(p)
                elif isinstance(p, Partial):
                    self.allowed_partial_per_input[input_idx].add(p)

        # Resolve output tensor_meta per output index
        if output_tensor_meta is None:
            self.output_metas = (None,) * max(num_outputs, 0)
        elif isinstance(output_tensor_meta, TensorMeta):
            self.output_metas = (output_tensor_meta,)
        else:
            self.output_metas = tuple(output_tensor_meta)

    def try_propagate(
        self,
        mesh: DeviceMesh,
        input_placements: tuple[tuple[Placement, ...], ...],
        input_specs: list[DTensorSpec],
    ) -> OpStrategy | None:
        """Try to match input placements against single-dim strategy rules on every mesh dim.

        Checks whether the given input placements independently match a rule in
        strategy_lookup on each mesh dimension, and that all inputs are shardable
        with those placements. If so, returns an OpStrategy with the matched output
        placements and zero redistribute costs.
        """
        from torch.distributed.tensor._ops.utils import is_tensor_shardable

        selected_output_placements: list[tuple[Placement, ...]] = []
        for mesh_dim in range(mesh.ndim):
            input_placements_for_dim = tuple(
                placements[mesh_dim] for placements in input_placements
            )
            output_for_dim = self.strategy_lookup.get(input_placements_for_dim)
            if output_for_dim is not None:
                selected_output_placements.append(output_for_dim)
            else:
                return None

        arg_specs = [
            DTensorSpec(mesh, placements, tensor_meta=input_spec.tensor_meta)
            for placements, input_spec in zip(input_placements, input_specs)
        ]
        if not all(
            is_tensor_shardable(spec.tensor_meta.shape, spec)
            for spec in arg_specs
            if spec.tensor_meta is not None
        ):
            return None

        output_spec = (
            _build_output_specs(
                mesh,
                selected_output_placements,
                self.num_outputs,
                self.output_metas,
            )
            if self.num_outputs > 0
            else None
        )
        return OpStrategy(
            [
                OpSpec(
                    output_specs=output_spec,
                    input_specs=arg_specs,
                    redistribute_cost=[[0.0] for _ in input_specs],
                )
            ]
        )


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
            prepared_strategy = _PreparedSingleDimStrategy(
                strategy_info, op_schema, output_tensor_meta
            )

            # Detect inplace ops by checking if the base op name ends with '_'
            op_name = op.name()
            base_name = op_name.split("::")[1].split(".")[0]
            is_inplace = base_name.endswith("_")

            return expand_to_full_mesh_op_strategy(
                mesh,
                op_schema,
                cast(list[PlacementList], prepared_strategy.expanded_strategies),
                output_tensor_meta=output_tensor_meta,
                inplace_op=is_inplace,
                input_index=prepared_strategy.num_outputs,
                allow_unbacked_sharding=prepared_strategy.allow_unbacked_sharding,
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


_pq_counter: int = 0


@dataclass(order=True)
class _PQEntry:
    """Priority queue entry for the Dijkstra search in _dijkstra_expand_single_dim_strategy_to_mesh.

    Ordered by (cost, counter) for heap comparison. The counter breaks ties
    in FIFO order so that entries with equal cost are explored in insertion
    order rather than by arbitrary tuple comparison on placements.
    """

    cost: float
    counter: int = field(init=False)
    # Per-input placement tuples representing the current search state.
    placements: tuple[tuple[Placement, ...], ...] = field(compare=False)
    # History of (input_idx, mesh_dim, old_placement, new_placement) transitions
    # from the initial state to this state, used for debugging.
    transitions: list[tuple[int, int, Placement, Placement]] = field(compare=False)
    # Accumulated redistribute cost per input (sum of incremental step costs).
    per_input_costs: tuple[float, ...] = field(compare=False)
    # Current communication bytes (in GB) per input, updated as placements change.
    per_input_comm_bytes_gb: tuple[float, ...] = field(compare=False)

    def __post_init__(self) -> None:
        global _pq_counter
        self.counter = _pq_counter
        _pq_counter += 1


def _get_neighbor_placements(
    allowed_sharding: set[Placement],
    allowed_partial: set[Placement],
    current: Placement,
) -> list[Placement]:
    """Return valid placement transitions for one input on one mesh dim.

    Transition rules follow DTensor redistribute semantics:
    - Replicate -> any allowed Shard or Partial (local view, free)
    - Shard -> Replicate (allgather), or different Shard (all-to-all)
    - Partial -> Replicate (allreduce), or any allowed Shard (reduce-scatter)
    """
    # Note: circular import
    from torch.distributed.tensor.placement_types import Partial

    neighbors: list[Placement] = []

    if isinstance(current, Replicate):
        neighbors.extend(allowed_sharding)
        neighbors.extend(allowed_partial)

    elif _is_sharding(current):
        neighbors.append(Replicate())
        neighbors.extend(s for s in allowed_sharding if s != current)

    elif isinstance(current, Partial):
        neighbors.append(Replicate())
        neighbors.extend(allowed_sharding)

    return neighbors


def _dijkstra_expand_single_dim_strategy_to_mesh(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_dim_strategy: _SingleDimStrategyInfo
    | Callable[
        [OpOverload, ArgsType, KwargsType], list[list[Placement | _ShardingPlaceholder]]
    ],
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    _collect_all_matches: set[tuple[tuple[Placement, ...], ...]] | None = None,
) -> OpStrategy | None:
    """
    Find the lowest cost sharding for the given op_schema.

    Uses a Dijkstra-like priority-queue search over input placement states. Each
    state is a tuple of per-input placement tuples, and neighbors are generated
    by changing one placement on one mesh dim for one input. The search
    terminates when a state matches a single-dim strategy on every mesh dim.

    This avoids the O(S^N) exhaustive expansion of _expand_single_dim_strategy_to_mesh
    (S = single-dim strategies, N = mesh dims).  Benchmarks with mm on fake
    process groups show:

        1D(4):     S^N=8,   avg 0.2ms
        2D(2,2):   S^N=64,  avg 2.3ms
        3D(2,2,2): S^N=512, avg 41ms, worst 392ms

    The step count is small (avg 0.6-2.0 pops) but per-step cost is dominated
    by cost computation.  Each transition computes an incremental cost via
    _compute_placement_transition_cost for the single changed placement, matching
    the per-step costs used by graph-based transform info planning.

    Returns None if any input has _StridedShard placement, signaling the caller
    to fall back to full expansion.

    Args:
        _collect_all_matches: Testing-only. When non-None, exhaustively explores the
            full transition graph, adding every shardable match to the set. Still
            returns the optimal (first) match.
    """
    # Extract input DTensorSpecs from OpStrategy-wrapped args.
    # Fall back for TupleStrategy (e.g. index tensors in index_put) since the PQ
    # search doesn't model variable-length tuple inputs.
    input_specs: list[DTensorSpec] = []
    for arg in op_schema.args_schema:
        if isinstance(arg, OpStrategy):
            assert len(arg.strategies) == 1
            input_specs.append(arg.strategies[0].output_spec)
        elif isinstance(arg, TupleStrategy):
            return None

    # Fall back if any kwargs are tensor inputs — the PQ search only tracks
    # positional tensor args and would miss redistribute costs for kwargs.
    for kwarg in op_schema.kwargs_schema.values():
        if isinstance(kwarg, (OpStrategy, TupleStrategy)):
            return None

    assert len(input_specs) > 0, "broken input"
    num_inputs = len(input_specs)

    # Fall back to full expansion if any input has _StridedShard or symbolic shapes
    # (symbolic shapes produce SymFloat costs that can't be compared in the PQ)
    for spec in input_specs:
        if any(isinstance(p, _StridedShard) for p in spec.placements):
            return None
        if spec.tensor_meta is not None and any(
            isinstance(s, torch.SymInt) for s in spec.tensor_meta.shape
        ):
            return None

    prepared_strategy = _PreparedSingleDimStrategy(
        single_dim_strategy, op_schema, output_tensor_meta, num_inputs=num_inputs
    )

    initial_placements = tuple(spec.placements for spec in input_specs)
    first_result: OpStrategy | None = None

    # Fast path: if initial placements already match a strategy, skip search
    fast_result = prepared_strategy.try_propagate(mesh, initial_placements, input_specs)
    if fast_result is not None:
        fast_result._pq_transitions = []  # type: ignore[attr-defined]
        if _collect_all_matches is not None:
            _collect_all_matches.add(initial_placements)
            first_result = fast_result
        else:
            return fast_result

    # Pre-compute mesh topology and per-input comm bytes for cost computation.
    # comm_bytes_gb reflects the local shard size given current placements;
    # it's tracked per PQ entry and updated as placements change.
    mesh_topo = MeshTopoInfo.build_from_mesh(mesh)
    initial_comm_bytes_gb: list[float] = []
    for spec in input_specs:
        assert spec.tensor_meta is not None
        total_bytes = spec.tensor_meta.dtype.itemsize * math.prod(
            spec.tensor_meta.shape
        )
        num_shards = 1
        for i, p in enumerate(spec.placements):
            if p.is_shard():
                num_shards *= mesh_topo.mesh_dim_devices[i]
        initial_comm_bytes_gb.append(total_bytes / num_shards / (1024**3))

    pq: list[_PQEntry] = []
    visited: set[tuple[tuple[Placement, ...], ...]] = set()

    initial_per_input_costs = (0.0,) * num_inputs
    initial_per_input_comm_bytes = tuple(initial_comm_bytes_gb)
    heapq.heappush(
        pq,
        _PQEntry(
            0.0,
            initial_placements,
            [],
            initial_per_input_costs,
            initial_per_input_comm_bytes,
        ),
    )

    def _push_neighbor(
        input_idx: int,
        mesh_dim: int,
        new_placement: Placement,
        source: _PQEntry,
    ) -> None:
        new_input_placements = [list(ps) for ps in source.placements]
        old_placement = new_input_placements[input_idx][mesh_dim]
        new_input_placements[input_idx][mesh_dim] = new_placement
        candidate_placements = tuple(tuple(ps) for ps in new_input_placements)
        if candidate_placements in visited:
            return
        step_cost, new_comm_bytes = _compute_placement_transition_cost(
            old_placement,
            new_placement,
            mesh_topo,
            mesh_dim,
            source.per_input_comm_bytes_gb[input_idx],
        )
        if step_cost == float("inf"):
            return
        changed_cost = source.per_input_costs[input_idx] + step_cost
        new_per_input_costs = (
            source.per_input_costs[:input_idx]
            + (changed_cost,)
            + source.per_input_costs[input_idx + 1 :]
        )
        new_per_input_comm_bytes = (
            source.per_input_comm_bytes_gb[:input_idx]
            + (new_comm_bytes,)
            + source.per_input_comm_bytes_gb[input_idx + 1 :]
        )
        new_cost = sum(new_per_input_costs)
        new_transitions = source.transitions + [
            (input_idx, mesh_dim, old_placement, new_placement)
        ]
        heapq.heappush(
            pq,
            _PQEntry(
                new_cost,
                candidate_placements,
                new_transitions,
                new_per_input_costs,
                new_per_input_comm_bytes,
            ),
        )

    while pq:
        candidate = heapq.heappop(pq)

        if candidate.placements in visited:
            continue
        visited.add(candidate.placements)

        match_result = prepared_strategy.try_propagate(
            mesh, candidate.placements, input_specs
        )
        if match_result is not None:
            # Use pre-computed per-input costs from the PQ search instead of
            # recomputing via generate_redistribute_costs -> _gen_transform_infos.
            match_spec = match_result.strategies[0]
            assert match_spec.input_specs is not None
            op_spec = OpSpec(
                output_specs=match_spec.output_specs,
                input_specs=list(match_spec.input_specs),
                redistribute_cost=[[cost] for cost in candidate.per_input_costs],
            )

            exhaustive = len(prepared_strategy.expanded_strategies) ** mesh.ndim
            logger.debug(
                "returning cost=%f %s, visited=%d, exhaustive=%d, transitions=%s",
                candidate.cost,
                op_spec,
                len(visited),
                exhaustive,
                candidate.transitions,
            )
            result = OpStrategy([op_spec])
            result._pq_transitions = candidate.transitions  # type: ignore[attr-defined]
            if _collect_all_matches is not None:
                _collect_all_matches.add(candidate.placements)
                if first_result is None:
                    first_result = result
            else:
                return result

        # Generate neighbor states
        for mesh_dim in range(mesh.ndim):
            for input_idx in range(len(candidate.placements)):
                current_p = candidate.placements[input_idx][mesh_dim]
                for neighbor_p in _get_neighbor_placements(
                    prepared_strategy.allowed_sharding_per_input[input_idx],
                    prepared_strategy.allowed_partial_per_input[input_idx],
                    current_p,
                ):
                    _push_neighbor(input_idx, mesh_dim, neighbor_p, candidate)

    if _collect_all_matches is not None and first_result is not None:
        return first_result

    raise AssertionError(
        f"No valid strategy found for op_schema {op_schema} "
        f"on {mesh}. "
        f"Explored {len(visited)} strategy combinations."
    )
