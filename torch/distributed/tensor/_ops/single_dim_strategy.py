#  Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Optional, TypeAlias, TypeVar, Union

import torch
from torch._ops import OpOverload
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


def _is_sharding(p: Placement) -> bool:
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
