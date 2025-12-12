#  Copyright (c) Meta Platforms, Inc. and affiliates
import itertools
import logging
from collections.abc import Callable
from typing import Any, cast, TypeAlias, TypeVar

from torch._ops import OpOverload
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_shardable,
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
    # TODO avoid taking num_inputs this way; make it an explicit arg? or a separate util for adding the replicate
    # placement
    num_inputs = len(single_dim_strategies_with_placeholders[0]) - 1
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

    # implicitly allow replicating output, all inputs
    # TODO: op_schema.args_spec is empty in this case, but op_schema.args_schema isn't.  What's the difference?
    expanded_strategies_over_one_mesh_dim.append([Replicate()] * (1 + num_inputs))

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
    unique_input_placements = _get_unique_placements(op_schema)

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
        expanded_strategies_over_one_mesh_dim = _fill_single_dim_strategy_placeholders(
            unique_input_placements, strategies_over_one_mesh_dim
        )

        # TODO: identify differences between this and 'expand_' util
        all_mesh_dim_strategies = [expanded_strategies_over_one_mesh_dim] * mesh.ndim

        # TODO: needs_pytree=True in cat op is not enough to make op_schema.args_strategy unflatten, why?
        # src_strategies = list(op_schema.args_strategy)
        src_strategies = []
        for obj in op_schema.args_schema:
            if isinstance(obj, OpStrategy):
                src_strategies.append(obj)
            elif isinstance(obj, TupleStrategy):
                src_strategies.extend(obj.children)

        all_combinations = itertools.product(*all_mesh_dim_strategies)
        all_strategies = []
        for nd_placements in all_combinations:
            # nd_placements is ([placements mesh_dim 0], [placements mesh_dim 1], ...)
            # where placements is [out, in0, in1, ...]
            # spec_list is inverted, [DTensorSpec0, 1, ...] where each spec has placements for all mesh dims
            spec_list = [
                DTensorSpec(mesh, tuple(placements))
                for placements in zip(*nd_placements)
            ]
            arg_specs = spec_list[1:]
            assert len(src_strategies) == len(arg_specs), (
                f"{len(src_strategies)=}, {len(arg_specs)=}, "
                "if src_strategies is empty, check needs_pytree in op registration"
            )

            # TODO replace isinstance with inheritance or util
            # Note: since we don't look at mesh dims inside single_dim_strategies, we can't tell if tensors are
            # 'shardable' instead, we filter out unshardable strategies after mesh expansion.
            # TODO: make this more robust after adding _ShardingPlaceholder, allowing us to say inside a single_dim_strategy
            # whether we care about even-sharding or other specific properties
            valid = True
            for i, (src_strategy, arg_spec) in enumerate(
                zip(src_strategies, arg_specs)
            ):
                if any(
                    isinstance(p, (Shard, _StridedShard)) for p in arg_spec.placements
                ):
                    # TODO(whc) it doesn't seem safe to me to allow_unbacked_sharding=True,
                    # but it's what we did in matrix_ops so i added it for now to pass
                    # TestDTensorCompile.test_dtensor_matmul_zero_size_shards
                    if not is_tensor_shardable(
                        src_strategy.shape, arg_spec, allow_unbacked_sharding=True
                    ):
                        logger.debug(
                            "expand skipping unshardable %s: arg %d spec %s",
                            nd_placements,
                            i,
                            arg_spec,
                        )
                        valid = False
                        break
            if valid:
                all_strategies.append(
                    OpSpec(
                        output_specs=spec_list[0],
                        input_specs=spec_list[1:],
                        redistribute_cost=[
                            generate_redistribute_costs(src_strategy, arg_spec)
                            for (src_strategy, arg_spec) in zip(
                                src_strategies, arg_specs
                            )
                        ],
                    )
                )
        return OpStrategy(all_strategies)

    return expanded_strategy
