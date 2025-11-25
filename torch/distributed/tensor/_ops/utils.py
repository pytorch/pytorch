# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import functools
import heapq
import itertools
import operator
from collections.abc import Callable, Iterable, Sequence
from typing import cast, Optional, Union

import torch
from torch._prims_common import DimsSequenceType, DimsType
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
)
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
    ShardingPlaceholder,
)


def _args_schema_with_tensor_meta(
    args_schema: ArgsType, kwargs_schema: KwargsType
) -> tuple[ArgsType, KwargsType]:
    """
    Replace DTensorSpec with TensorMeta in args_schema, for use with single-dim strategies
    """

    def spec_to_strategy(spec: object) -> object:
        if isinstance(spec, DTensorSpec):
            return spec.tensor_meta
        elif (
            isinstance(spec, (list, tuple))
            and len(spec) > 0
            and isinstance(spec[0], DTensorSpec)
        ):
            raise NotImplementedError("Tuples!")
            #     # tensor list create tuple strategy
            #     tuple_strategy = [spec_to_strategy(s) for s in spec]
            #     tuple_strategy = cast(Sequence[StrategyType], tuple_strategy)
            #     return TupleStrategy(
            #         tuple(tuple_strategy) if isinstance(spec, tuple) else tuple_strategy
            #     )
        else:
            return spec

    args_op_strategy = tuple([spec_to_strategy(a) for a in args_schema])

    kwargs_op_strategy = {k: spec_to_strategy(v) for k, v in kwargs_schema.items()}

    return args_op_strategy, kwargs_op_strategy


def _find_lowest_cost_sharding(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_dim_strategy: Callable[
        [ArgsType, KwargsType], list[list[Placement | ShardingPlaceholder]]
    ],
) -> StrategyType:
    """
    Find the lowest cost sharding for the given op_schema.

    This solves the runtime complexity problem of using _expand_single_dim_strategy_to_mesh by avoiding enumerating
    the product of the single dim strategy over all the mesh dims and then searching the enumerated list for the min
    cost.  Instead, it starts from the input placements and expands a search from there, starting by checking if
    the input placements already describe a legal placement on all mesh dims, and then if not, iterating by taking
    the lowest-cost redistributions in priority-queue order.


    # TODO: this is mostly claude, i haven't done much validation other than reviewing a bunch of times and telling
    # claude to fix things i noticed were wrong, and then running it via bench_singledim.py (see later PR in stack).

    Upshot: it runs, and it speeds things up significantly compared to the full strategy. - especially in the pointwise
    +shard case
    see https://gist.github.com/wconstab/5df63815696c504db2ffadfbf1675d21 for benchmark results

    """

    # Get the arguments schema and prepare for strategy computation
    args_schema, kwargs_schema = _args_schema_with_tensor_meta(
        op_schema.args_schema, op_schema.kwargs_schema
    )

    # Generate single-dim strategies with placeholders
    single_dim_strategies_with_placeholders = single_dim_strategy(
        args_schema, kwargs_schema
    )

    # Expand placeholders to get concrete strategies for one mesh dimension
    expanded_strategies_over_one_mesh_dim = _fill_single_dim_strategy_placeholders(
        mesh, op_schema, single_dim_strategies_with_placeholders
    )

    # Extract input DTensorSpecs from op_schema.args_schema
    input_specs = [
        spec for spec in op_schema.args_schema if isinstance(spec, DTensorSpec)
    ]

    if not input_specs:
        # No input specs, fall back to simple expansion
        return _expand_single_dim_strategy_to_mesh(
            mesh, op_schema, single_dim_strategy
        )(op_schema.args_schema, op_schema.kwargs_schema)

    # Build src_strategies wrapping input specs for redistribute cost computation
    src_strategies = [OpStrategy([OpSpec(spec)]) for spec in input_specs]

    # Priority queue: (cost, counter, input_placements_tuple)
    # input_placements_tuple is a tuple of placements tuples, one per input tensor
    # Each input's placements tuple has one placement per mesh dimension
    counter = 0
    pq: list[tuple[float, int, tuple[tuple[Placement, ...], ...]]] = []
    visited: set[tuple[tuple[Placement, ...], ...]] = set()

    # Start from the current input placements (cost 0 since no redistribution needed)
    initial_input_placements = tuple(spec.placements for spec in input_specs)
    heapq.heappush(pq, (0.0, counter, initial_input_placements))
    counter += 1

    # Explore priority queue to find the first valid strategy
    while pq:
        cost, _, input_placements_tuple = heapq.heappop(pq)

        if input_placements_tuple in visited:
            continue
        visited.add(input_placements_tuple)

        # For each expanded single-dim strategy, check if it's valid with current input placements
        for strategy in expanded_strategies_over_one_mesh_dim:
            # strategy is [output_placement, input1_placement, input2_placement, ...] for a single mesh dim
            # Build a complete strategy combination by replicating this across all mesh dims
            strategy_comb = tuple([tuple(strategy)] * mesh.ndim)

            # Build DTensorSpec for each tensor (output + inputs)
            # We need to include tensor_meta for each spec
            spec_list = []
            for i, specs in enumerate(zip(*strategy_comb)):
                if i == 0:
                    # Output spec - we'll infer tensor_meta from inputs
                    # For now, use the first input's tensor_meta as a placeholder
                    # TODO: properly infer output tensor_meta from the operation
                    spec_list.append(
                        DTensorSpec(mesh, specs, tensor_meta=input_specs[0].tensor_meta)
                    )
                else:
                    # Input spec - use the corresponding input's tensor_meta
                    input_idx = i - 1
                    spec_list.append(
                        DTensorSpec(
                            mesh, specs, tensor_meta=input_specs[input_idx].tensor_meta
                        )
                    )

            output_spec = spec_list[0]
            arg_specs = spec_list[1:]  # Skip output spec, get input specs

            # Check if current input placements match the strategy requirements
            if not all(
                input_placements_tuple[i] == arg_spec.placements
                for i, arg_spec in enumerate(arg_specs)
            ):
                # Current input placements don't match this strategy, skip it
                continue

            # Check if all inputs are shardable
            if not all(
                is_tensor_shardable(src_strat.shape, arg_spec)
                for src_strat, arg_spec in zip(src_strategies, arg_specs)
            ):
                continue

            # Compute redistribute costs from original input specs to required arg_specs
            # Note: The 'cost' variable from the priority queue represents the search cost,
            # but OpSpec needs the full cost matrix from original inputs for sharding propagation
            redistribute_costs = [
                generate_redistribute_costs(src_strategy, arg_spec)
                for src_strategy, arg_spec in zip(src_strategies, arg_specs)
            ]

            # Create and return the first valid OpSpec found
            op_spec = OpSpec(
                output_specs=output_spec,
                input_specs=arg_specs,
                redistribute_cost=redistribute_costs,
            )
            return OpStrategy([op_spec])

        # Explore neighboring input placement states by redistributing inputs
        # This is where we add new states to the priority queue
        for input_idx in range(len(input_specs)):
            for strategy in expanded_strategies_over_one_mesh_dim:
                # Try redistributing this input to match the placement required by this strategy
                new_input_placements = list(input_placements_tuple)
                # Get the required placement for this input from the strategy
                # strategy is [output_placement, input1_placement, input2_placement, ...]
                required_placement_per_dim = strategy[1 + input_idx]
                # Build full placement across all mesh dims
                new_placement = tuple([required_placement_per_dim] * mesh.ndim)
                new_input_placements[input_idx] = new_placement
                new_input_placements_tuple = tuple(new_input_placements)

                if new_input_placements_tuple in visited:
                    continue

                # Compute cost of this redistribution
                current_spec = DTensorSpec(
                    mesh,
                    input_placements_tuple[input_idx],
                    tensor_meta=input_specs[input_idx].tensor_meta,
                )
                new_spec = DTensorSpec(
                    mesh, new_placement, tensor_meta=input_specs[input_idx].tensor_meta
                )
                redistribution_cost = redistribute_cost(current_spec, new_spec)

                if redistribution_cost == float("inf"):
                    continue

                new_total_cost = cost + redistribution_cost

                heapq.heappush(
                    pq, (new_total_cost, counter, new_input_placements_tuple)
                )
                counter += 1

    # If we get here, no valid strategy was found - this should not happen
    raise AssertionError(
        f"No valid strategy found for op_schema {op_schema}. "
        f"Explored {len(visited)} strategy combinations."
    )


def _fill_single_dim_strategy_placeholders(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_dim_strategies_with_placeholders: list[
        list[Placement | ShardingPlaceholder]
    ],
) -> list[list[Placement]]:
    """
    Replace any ShardingPlaceholder with the specific Sharding types used by the inputs in op_schema.
    Supports implicit replication.

    Example:
    single_dim_strategies_with_placeholders = [[Partial(), ShardingPlaceholder(1), ShardingPlaceholder(0)]]
    input0: Shard(0)
    input1: StridedShard(1, split_factor=2)
    returns: [
       [Partial(), Shard(1), Shard(0)],
       [Partial(), StridedShard(1, split_factor=2), StridedShard(0, split_factor=2)],
       [Replicate(), Replicate(), Replicate()]
    ]
    """
    shard_builders: dict[str, Callable[[int], Placement]] = {}
    for spec in op_schema.args_spec:
        for p in spec.placements:
            if isinstance(p, _StridedShard):
                key = f"StridedShard(sf={p.split_factor})"
                if key not in shard_builders:
                    shard_builders[key] = lambda tensor_dim: _StridedShard(
                        tensor_dim, split_factor=p.split_factor
                    )
            elif isinstance(p, Shard):
                key = "Shard()"
                if key not in shard_builders:
                    shard_builders[key] = lambda tensor_dim: Shard(tensor_dim)

    # if any of the placements is a placeholder, we need to expand the strategy
    # to all possible combinations of placements
    expanded_strategies_over_one_mesh_dim: list[list[Placement]] = []
    for s in single_dim_strategies_with_placeholders:
        if not any(isinstance(p, ShardingPlaceholder) for p in s):
            continue
        for shard_builder in shard_builders.values():
            expanded_strategy: list[Placement] = []
            for maybe_placeholder in s:
                if isinstance(maybe_placeholder, ShardingPlaceholder):
                    # we combine the tensor dim to shard from the placeholder
                    # with other metadata (e.g. split_factor) from the sharding class
                    expanded_strategy.append(shard_builder(maybe_placeholder.dim))
                else:
                    assert isinstance(maybe_placeholder, Placement)
                    expanded_strategy.append(maybe_placeholder)
            expanded_strategies_over_one_mesh_dim.append(expanded_strategy)

    # implicitly allow replicating output, all inputs
    expanded_strategies_over_one_mesh_dim.append(
        [Replicate()] * (1 + len(op_schema.args_spec))
    )

    return expanded_strategies_over_one_mesh_dim


def _expand_single_dim_strategy_to_mesh(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    single_dim_strategy: Callable[
        [ArgsType, KwargsType], list[list[Placement | ShardingPlaceholder]]
    ],
) -> Callable[[ArgsType, KwargsType], StrategyType]:
    """
    Expands the single_mesh_dim impl across all mesh dims, and expands ShardingPlacholder into all
    sharding types used by inputs.

    This supports functional correctness but will generate all possible combinations, which is prohibitively expensive
    for larger numbers of mesh dimensions.
    """

    def expanded_strategy(
        args_schema: ArgsType, kwargs_schema: KwargsType
    ) -> StrategyType:
        strategies_over_one_mesh_dim = single_dim_strategy(args_schema, kwargs_schema)
        expanded_strategies_over_one_mesh_dim = _fill_single_dim_strategy_placeholders(
            mesh, op_schema, strategies_over_one_mesh_dim
        )

        # TODO: identify differences between this and 'expand_' util
        all_mesh_dim_strategies = [expanded_strategies_over_one_mesh_dim] * mesh.ndim
        strategy_combs = itertools.product(*all_mesh_dim_strategies)
        all_strategies = []
        for strategy_comb in strategy_combs:
            spec_list = [
                DTensorSpec(mesh, tuple(specs)) for specs in zip(*strategy_comb)
            ]
            arg_specs = spec_list[1:]
            # Sad.. i am wrapping the DTensorSpec back into an OpStrategy to make it compatible with gen_redistribute_costs
            # but I want to avoid having OpStrategy at all in here
            src_strategies = [
                OpStrategy([OpSpec(s)])
                for s in op_schema.args_schema
                if isinstance(s, DTensorSpec)
            ]
            if any(
                not is_tensor_shardable(src_strat.shape, arg_spec)
                for src_strat, arg_spec in zip(src_strategies, arg_specs)
            ):
                # Note: since we don't look at mesh dims inside single_dim_strategies, we can't tell if tensors are 'shardable'
                # instead, we filter out unshardable strategies after mesh expansion
                # TODO: make this more robust after adding ShardingPlaceholder, allowing us to say inside a single_dim_strategy
                # whether we care about even-sharding or other specific properties
                continue

            assert len(arg_specs) == len(src_strategies), (
                "expected one src strategy per arg spec"
            )
            all_strategies.append(
                OpSpec(
                    output_specs=spec_list[0],
                    input_specs=spec_list[1:],
                    redistribute_cost=[
                        generate_redistribute_costs(src_strategy, arg_spec)
                        for (src_strategy, arg_spec) in zip(src_strategies, arg_specs)
                    ],
                )
            )
        return OpStrategy(all_strategies)

    return expanded_strategy


def replicate_op_strategy(op_schema: OpSchema) -> StrategyType:
    """
    Fallback strategy all use Replication()
    """
    args_strategy = op_schema.args_strategy
    kwargs_strategy = op_schema.kwargs_strategy
    inputs_strategy = args_strategy + kwargs_strategy

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
            assert isinstance(placement, Shard)
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                # there's a map from the common shape shard dim to
                # the input shape shard dim before broadcasting,
                # use that instead
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

        args_strategy = op_schema.args_strategy
        kwargs_strategy = op_schema.kwargs_strategy
        input_args_strategy = args_strategy + kwargs_strategy

        if len(input_specs) != len(input_args_strategy):
            raise AssertionError(
                f"input_specs({len(input_specs)}) != strategies({len(input_args_strategy)}: "
                f"{len(args_strategy)} args + {len(kwargs_strategy)} kwargs)"
            )
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
