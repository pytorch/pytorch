# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import itertools
import operator
from collections.abc import Callable, Iterable, Sequence
from typing import cast, Optional, TypeAlias, TypeVar, Union

import torch
from torch._prims_common import DimsSequenceType, DimsType
from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor._collective_utils import redistribute_cost
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    OutputSharding,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)


# convenient wrapper to register sharding propagation rules
def register_prop_rule(
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo] = None,
) -> Callable[
    [Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]
]:
    def wrapper(
        impl: Callable[[OpSchema], OutputSharding],
    ) -> Callable[[OpSchema], OutputSharding]:
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(
                overload, impl, schema_info
            )
        return impl

    return wrapper


# Note:
# using TypeVar here allows the registration decorator to preserve the specific type info of the wrapped strategy,
# while hardcoding the typing on the wrapper (e.g. Callable[[OpSchema], StrategyType]) would mean mypy would treat
# the return value of the wrapped strategy as always being a `StrategyType` even if it were a derived class like
# MyStrategyType(StrategyType).
_OpSchemaT = TypeVar("_OpSchemaT", bound=OpSchema)
_StrategyTypeT = TypeVar("_StrategyTypeT", bound=StrategyType)
_ShardingStrategyFunc: TypeAlias = Callable[[_OpSchemaT], _StrategyTypeT]


def register_op_strategy(
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo] = None,
) -> Callable[[_ShardingStrategyFunc], _ShardingStrategyFunc]:
    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]

    def wrapper(impl: _ShardingStrategyFunc) -> _ShardingStrategyFunc:
        if isinstance(op, list):
            overloads = op
        else:
            overloads = [op]

        for overload in overloads:
            curr_schema_info = None
            if schema_info is None:
                specialized_args = [
                    a.name
                    for a in overload._schema.arguments
                    if a.name in arg_names_that_require_specializing_cache_strategy
                ]
                if any(specialized_args):
                    curr_schema_info = RuntimeSchemaInfo(
                        static_kwargkey=specialized_args
                    )
            else:
                curr_schema_info = schema_info
            DTensor._op_dispatcher.sharding_propagator.register_op_strategy(
                overload, impl, curr_schema_info
            )
        return impl

    return wrapper


def replicate_op_strategy(op_schema: OpSchema) -> StrategyType:
    """
    Fallback strategy all use Replication()
    """
    inputs_strategy = op_schema.args_strategy
    # TODO(zpcore): handle kwarg_inputs_strategy
    # kwarg_inputs_strategy = op_schema.kwargs_schema
    output_type = [str(ret.type) for ret in op_schema.op._schema.returns]
    output_len = output_type.count("Tensor")
    # TODO(zpcore): Confirm if view op can be handle properly or not. Prevent
    # handling view ops until confirmed.
    if op_schema.op.is_view:
        raise RuntimeError(
            "fallback strategy is unable to handle view ops until confirmed"
        )
    if "List[Tensor]" in output_type:
        raise RuntimeError(
            "fallback strategy is unable to handle ops with List[Tensor] output "
            "because size of the list may depend on the op's input value"
        )

    mesh = inputs_strategy[0].mesh

    dim_sharding: PlacementList = [Replicate()] * (output_len + len(inputs_strategy))
    single_dim_placement = [dim_sharding]
    return expand_to_full_mesh_op_strategy(
        mesh, op_schema, single_dim_placement, input_index=output_len
    )


def as_list(
    x: Union[list[object], object],
    # pyre-fixme[11]: Annotation `immutable_list` is not defined as a type.
) -> Union[list[object], torch.fx.immutable_collections.immutable_list]:  # type: ignore[valid-type]
    # During tracing, `aten.sum.dim_IntList` uses `immutable_list` for its args,
    # which is an object but treated as a list by the tracer. Therefore, keep
    # `immutable_list` intact here as well.
    if type(x) is list or isinstance(x, torch.fx.immutable_collections.immutable_list):
        return x
    else:
        return [x]


def normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def normalize_dims(dims: DimsType, ndim: int) -> DimsSequenceType:
    """Normalize a dim or a sequence of dims, so that they are all positive."""
    if isinstance(dims, int):
        dims = (normalize_dim(dims, ndim),)
    elif isinstance(dims, list):
        dims = [normalize_dim(dim, ndim) for dim in dims]
    elif isinstance(dims, tuple):
        dims = tuple(normalize_dim(dim, ndim) for dim in dims)
    return dims


def prod(xs: Iterable[int]) -> int:
    return functools.reduce(operator.mul, xs, 1)


def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the spec matches these criteria:
    * any Shard placements in spec refer to valid tensor dims
    * no empty local tensors (uneven sharding OK, as long as last rank has >0 size)
    """
    # number of shards in each tensor dimension
    shards_map = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            if shard_dim >= len(shape):
                return False
            shards_map[shard_dim] *= spec.mesh.size(i)

    for i, dim_size in enumerate(shape):
        # TODO: maybe we should determine is_shardable based on
        #       whether it's evenly sharded or not
        if shards_map[i] > 1 and dim_size < shards_map[i]:
            return False

    return True


def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is evenly shardable according to the spec."""
    # number of shards in each tensor dimension
    shards_map = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            shards_map[shard_dim] *= spec.mesh.size(i)

    for i, dim_size in enumerate(shape):
        if shards_map[i] > 1 and (dim_size % shards_map[i] != 0):
            return False

    return True


def is_tensor_evenly_shardable_on_dim(
    shape: Sequence[int], spec: DTensorSpec, dim: int
) -> bool:
    """Check if the shape is evenly shardable according to the spec on dim."""
    dim = normalize_dim(dim, len(shape))

    num_shards = 1
    for i, placement in enumerate(spec.placements):
        if placement.is_shard():
            shard_dim = cast(Shard, placement).dim
            if shard_dim == dim:
                num_shards *= spec.mesh.size(i)

    return shape[dim] % num_shards == 0


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded."""
    return any(p.is_shard(dim) for p in spec.placements)


def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh."""
    return any(p.is_partial() for p in spec.placements)


def infer_broadcast_dims_map(
    common_shape: torch.Size, input_shape: torch.Size
) -> list[int]:
    # infer the broadcast dims map, where it maps from the common shape dim to the input shape dim
    # this is aligned with the broadcast semantics
    # e.g. if common_shape = [1, 2, 3, 4] and input_shape = [2, 3, 4],
    # broadcast_dims_map will be [-1, 0, 1, 2]
    # meaning that dim 0 in the output has no mapping to the input, and dim 1 in the output maps to dim 0 in the input
    common_ndim = len(common_shape)
    input_ndim = len(input_shape)
    broadcast_dims_map = [-1] * common_ndim
    for idx in range(-1, -1 - input_ndim, -1):
        if input_shape[idx] == common_shape[idx]:
            broadcast_dims_map[common_ndim + idx] = input_ndim + idx
    return broadcast_dims_map


def map_placements_after_broadcast(
    placements: tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: list[int],
    partial_to_replicate: bool = False,
) -> tuple[Placement, ...]:
    """Map each placement based on the output shape after broadcast."""
    new_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, Partial):
            if partial_to_replicate:
                # map the partial placement to replicate
                new_placements.append(Replicate())
            else:
                new_placements.append(placement)
        elif isinstance(placement, Replicate):
            new_placements.append(placement)
        else:
            assert isinstance(placement, Shard | _StridedShard)
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                # there's a map from the common shape shard dim to
                # the input shape shard dim before broadcasting,
                # use that instead
                if isinstance(placement, _StridedShard):
                    new_placements.append(
                        _StridedShard(
                            new_shard_dim, split_factor=placement.split_factor
                        )
                    )
                else:
                    new_placements.append(Shard(new_shard_dim))
            else:
                # there's no map between common shape shard dim and
                # the input shape shard dim before broadcasting,
                # in this case it means implicit broadcasting happen
                # in this dim, so we can just mark it as replicate
                # and implicit broadcast will broadcast automatically
                # to the sharded shape
                new_placements.append(Replicate())

    return tuple(new_placements)


def generate_redistribute_costs(
    src_strategy: OpStrategy, dst_spec: DTensorSpec
) -> list[float]:
    """Generates one row in the 'redistribute_costs' matrix in an OpSpec
    The length of the returned list will match the number of strategies in 'src_strategy'.

    Each value in the row is the cost of redistributing from a particular src_strategy to dst_spec.
    """
    redistribute_costs: list[float] = [
        redistribute_cost(strat.output_spec, dst_spec)
        for strat in src_strategy.strategies
    ]

    return redistribute_costs


def expand_to_full_mesh_op_strategy(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_mesh_dim_strategies: list[PlacementList],
    *,
    input_index: int = 1,
    inplace_op: bool = False,
    is_valid_strategy_cb: Optional[
        Callable[[list[DTensorSpec], tuple[Optional[DTensorSpec], ...]], bool]
    ] = None,
) -> OpStrategy:
    """
    Convenience function to allow writing a sharding strategy considering only a single mesh dimension,
    and have it expanded combinatorically to all mesh dimensions.

    Args:
        mesh (DeviceMesh): the device mesh to expand the strategy to
        op_schema (OpSchema): the op schema
        single_mesh_dim_strategies (list[PlacementList]): the sharding strategies to expand. The outer list is over
            different strategies.  The inner PlacementList is over the outputs and inputs of the op. If input_index is 1,
            a PlacementList looks like [output_placement, input_placement1, input_placement2, ...].
        input_index: the number of outputs of the op, defaults to 1
        inplace_op: whether the op is inplace or not, defaults to False
        is_valid_strategy_cb: a callback function to filter out invalid sharding rules, defaults to None.

    Example: Let's say `my_op(tensor_x, tensor_y) - > output_tensor`  can support sharding or replicating tensor_x,
    but always requires tensor_y to be replicated.  We can specify these valid combinations ignoring mesh dims.
    Then, we can rely on `expand_to_full_mesh_op_strategy` to create every possible combination of these shardings
    over multiple mesh dimensions, filtering out any combinations that are invalid based on the actual mesh dim size.

        single_mesh_dim_strategies = [
            # first strategy: return output sharded on first dim, shard tensor_x on its first dim, replicate tensor_y
            [Shard(0), Shard(0), Replicate()]
            # second strategy: replicate output, and both inputs
            [Replicate(), Replicate(), Replicate()]
        ]
    """
    # Expand the single_mesh_dim_strategies to full mesh dim strategies.
    all_mesh_dim_strategies = [single_mesh_dim_strategies] * mesh.ndim

    strategy_combs = itertools.product(*all_mesh_dim_strategies)

    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list: list[Optional[DTensorSpec]] = []
        for specs in zip(*strategy_comb):
            if specs[0] is not None:
                # TODO: we should fill in tensor_meta here.  If nothing else, it helps the filter strategy callback
                # pyrefly: ignore [bad-argument-type]
                spec_list.append(DTensorSpec(mesh, specs))
            else:
                spec_list.append(None)

        input_specs: list[DTensorSpec] = [
            s for s in spec_list[input_index:] if isinstance(s, DTensorSpec)
        ]

        input_args_strategy = op_schema.args_strategy
        assert len(input_specs) == len(input_args_strategy)
        self_spec = input_args_strategy[0].strategies[0].output_spec

        if inplace_op and self_spec.placements != input_specs[0].placements:
            # if it's inplace op, we would only allow the OpSpec to be added when the
            # input_spec matches the first argument's runtime sharding, otherwise we skip
            continue

        output_specs: tuple[Optional[DTensorSpec], ...]
        if input_index > 1:
            output_specs = tuple(spec_list[:input_index])
        else:
            if spec_list[0] is not None:
                output_specs = spec_list[0]  # type: ignore[assignment]
            else:
                raise RuntimeError("output spec is None")

        # check all inputs are shardable
        if not all(
            is_tensor_shardable(inp.shape, s)
            for inp, s in zip(input_args_strategy, input_specs)
        ):
            continue

        # perform additional op-specific filtering
        if is_valid_strategy_cb is not None:
            if not is_valid_strategy_cb(input_specs, output_specs):
                continue

        redistribute_cost = [
            generate_redistribute_costs(input_strategy, input_spec)
            for input_strategy, input_spec in zip(input_args_strategy, input_specs)
        ]

        strategy = OpSpec(
            output_specs=output_specs,
            input_specs=input_specs,
            redistribute_cost=redistribute_cost,
        )
        all_strategies.append(strategy)
    return OpStrategy(all_strategies)


def shift_shard_dims_after_insert(
    placements: Sequence[Placement], insert_dim: int = 0
) -> Sequence[Placement]:
    normalized_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, Shard) and placement.dim >= insert_dim:
            normalized_placements.append(Shard(placement.dim + 1))
        else:
            normalized_placements.append(placement)
    return normalized_placements


def shift_shard_dims_after_remove(
    placements: Sequence[Placement], remove_dim: int = 0
) -> Sequence[Placement]:
    normalized_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, Shard) and placement.dim > remove_dim:
            normalized_placements.append(Shard(placement.dim - 1))
        else:
            normalized_placements.append(placement)
    return normalized_placements
