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
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
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


def _get_registration_wrapper(
    registration_fn,
    op: Union[torch._ops.OpOverload, list[torch._ops.OpOverload]],
    schema_info: Optional[RuntimeSchemaInfo],
    arg_names_that_require_specializing_cache_strategy: Optional[list[str]],
):
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            curr_schema_info = None
            if (
                schema_info is None
                and arg_names_that_require_specializing_cache_strategy is not None
            ):
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
            registration_fn(overload, impl, curr_schema_info)
        return impl

    return wrapper


# convenient wrapper to register sharding propagation rules
def register_prop_rule(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
) -> Callable[
    [Callable[[OpSchema], OutputSharding]], Callable[[OpSchema], OutputSharding]
]:
    return _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy=None,
    )


# Note:
# using TypeVar here allows the registration decorator to preserve the specific type info of the wrapped strategy,
# while hardcoding the typing on the wrapper (e.g. Callable[[OpSchema], StrategyType]) would mean mypy would treat
# the return value of the wrapped strategy as always being a `StrategyType` even if it were a derived class like
# MyStrategyType(StrategyType).
_OpSchemaT = TypeVar("_OpSchemaT", bound=OpSchema)
_StrategyTypeT = TypeVar("_StrategyTypeT", bound=StrategyType)
_ShardingStrategyFunc: TypeAlias = Callable[[_OpSchemaT], _StrategyTypeT]


def register_op_strategy(
    op: torch._ops.OpOverload | list[torch._ops.OpOverload],
    schema_info: RuntimeSchemaInfo | None = None,
) -> Callable[[_ShardingStrategyFunc], _ShardingStrategyFunc]:
    # For every ATen op that accepts any args in this list,
    # the arg itself can impact the strides (and potentially the sharding strategy)
    # of the output tensor.
    # thus, we will detect ATen schemas with any of these args and ensure
    # that they get specialized here.
    arg_names_that_require_specializing_cache_strategy = [
        "memory_format",
    ]
    return _get_registration_wrapper(
        DTensor._op_dispatcher.sharding_propagator.register_op_strategy,
        op,
        schema_info,
        arg_names_that_require_specializing_cache_strategy,
    )


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
    x: list[object] | object,
    # pyre-fixme[11]: Annotation `immutable_list` is not defined as a type.
) -> list[object] | torch.fx.immutable_collections.immutable_list:  # type: ignore[valid-type]
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


def _generate_shard_order_variants(
    placements: tuple[Placement, ...],
    unique_shard_orders: set[tuple[tuple[int, ...], ...]],
    mesh: DeviceMesh,
) -> list[tuple[tuple[int, ...], ...] | None]:
    """
    Generate shard order variants for output specs based on unique shard orders from inputs.

    For each unique shard order pattern from inputs that is compatible with the given placements,
    construct a corresponding shard order for the output.

    Args:
        placements: The placements for the output spec
        unique_shard_orders: Set of shard order patterns (mesh_dims tuples) from inputs
        mesh: The device mesh

    Returns:
        A list of shard_order_raw tuples to use when creating DTensorSpec.
        Each tuple contains mesh_dims tuples for each sharded tensor dimension.
        Returns [None] if no multi-dim sharding exists.
    """

    # Find which tensor dims are sharded across which mesh dims
    tensor_dim_to_mesh_dims: dict[int, list[int]] = {}
    for mesh_dim, placement in enumerate(placements):
        if isinstance(placement, (Shard, _StridedShard)):
            tensor_dim = placement.dim
            if tensor_dim not in tensor_dim_to_mesh_dims:
                tensor_dim_to_mesh_dims[tensor_dim] = []
            tensor_dim_to_mesh_dims[tensor_dim].append(mesh_dim)

    # Check if any tensor dim is sharded across multiple mesh dims
    multi_sharded_dims = {
        tensor_dim: mesh_dims
        for tensor_dim, mesh_dims in tensor_dim_to_mesh_dims.items()
        if len(mesh_dims) > 1
    }

    if not multi_sharded_dims or not unique_shard_orders:
        # No multi-dim sharding or no shard orders from inputs, use default
        return [None]

    # Generate variants by trying each unique shard order pattern from inputs
    shard_order_variants = []
    for shard_order_pattern in unique_shard_orders:
        # Each pattern is a tuple of mesh_dims tuples
        # Try to map this pattern to our multi_sharded_dims
        for mesh_dims_ordering in shard_order_pattern:
            if len(mesh_dims_ordering) < 2:
                continue

            # Try to apply this ordering to each multi-sharded tensor dim
            for tensor_dim, mesh_dims in multi_sharded_dims.items():
                if len(mesh_dims) == len(mesh_dims_ordering):
                    # Check if this ordering is a permutation of our mesh_dims
                    if set(mesh_dims) == set(mesh_dims_ordering):
                        # Create shard_order_raw with this ordering
                        shard_order_raw = (mesh_dims_ordering,)
                        shard_order_variants.append(shard_order_raw)

    # Remove duplicates and add default if no variants found
    shard_order_variants = list(
        set(shard_order_variants) if shard_order_variants else [None]
    )
    return shard_order_variants if shard_order_variants else [None]


def is_tensor_shardable(
    shape: Sequence[int],
    spec: DTensorSpec,
    allow_unbacked_sharding: bool | None = None,
) -> bool:
    """
    Check if the shape is shardable according to the spec.

    This function handles both `Shard` and `_StridedShard` placements:
    - For `Shard`: checks if the tensor dimension size >= number of shards
    - For `_StridedShard`: additionally checks if the dimension is shardable after
      splitting with the placement's `split_factor`

    allow_unbacked_sharding: determines the fallback value if unbacked shapes are involved,
    and the queried shape properties are not statically known.

    e.g. when asking if u0 is shardable on num_shards, and u0 has generic bounds [0, inf],
    the behavior of allow_unbacked_sharding is:

        None: will data-dependent error
        True: assumes shardability; we return True, allowing zero-size shards at runtime when u0 < num_shards.
        False: returns False, and lower-bounding u0, e.g. torch._check(u0 >= num_shards), is needed to enable sharding.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

    assert allow_unbacked_sharding in [None, True, False]
    guard_fn = {
        None: bool,
        True: guard_or_false,
        False: guard_or_true,
    }[allow_unbacked_sharding]

    # number of shards in each tensor dimension
    num_shards = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if isinstance(placement, Shard | _StridedShard):
            shard_dim = placement.dim
            if shard_dim >= len(shape):
                return False
            num_shards[shard_dim] *= spec.mesh.size(i)
            if isinstance(placement, _StridedShard):
                # make sure tensor dim `shard_dim` is shardable after splitting
                # with split_factor
                if guard_fn(
                    shape[shard_dim] < num_shards[shard_dim] * placement.split_factor
                ):
                    return False
            else:
                if guard_fn(shape[shard_dim] < num_shards[shard_dim]):
                    return False

    return True


def is_tensor_evenly_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """Check if the shape is evenly shardable according to the spec."""
    # number of shards in each tensor dimension
    num_shards = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        if isinstance(placement, Shard | _StridedShard):
            shard_dim = placement.dim
            if shard_dim >= len(shape):
                return False
            num_shards[shard_dim] *= spec.mesh.size(i)
            if isinstance(placement, _StridedShard):
                if (
                    shape[shard_dim] % (placement.split_factor * num_shards[shard_dim])
                    != 0
                ):
                    return False
            else:
                if shape[shard_dim] % num_shards[shard_dim] != 0:
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
    output_tensor_meta: TensorMeta | Sequence[TensorMeta | None] | None = None,
    input_index: int = 1,
    inplace_op: bool = False,
    is_valid_strategy_cb: Callable[
        [list[DTensorSpec], tuple[DTensorSpec | None, ...]], bool
    ]
    | None = None,
    unique_shard_orders: set[tuple[tuple[int, ...], ...]] | None = None,
) -> OpStrategy:
    """
    Convenience function to allow writing a sharding strategy considering only a single mesh dimension,
    and have it expanded combinatorially to all mesh dimensions.

    Args:
        mesh (DeviceMesh): the device mesh to expand the strategy to
        op_schema (OpSchema): the op schema
        single_mesh_dim_strategies (list[PlacementList]): the sharding strategies to expand. The outer list is over
            different strategies.  The inner PlacementList is over the outputs and inputs of the op. If input_index is 1,
            a PlacementList looks like [output_placement, input_placement1, input_placement2, ...].
        output_tensor_meta: tensor metadata for the output(s), used to populate DTensorSpec.tensor_meta field
        input_index: the number of outputs of the op, defaults to 1
        inplace_op: whether the op is inplace or not, defaults to False
        is_valid_strategy_cb: a callback function to filter out invalid sharding rules, defaults to None.
        unique_shard_orders: unique shard order patterns from input specs, used to generate shard order variants for outputs

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

    args_strategy = op_schema.args_strategy
    kwargs_strategy = op_schema.kwargs_strategy
    input_args_strategy = args_strategy + kwargs_strategy
    all_strategies = []
    for strategy_comb in strategy_combs:
        spec_list: list[DTensorSpec | None] = []
        spec_index = 0
        for specs in zip(*strategy_comb):
            if specs[0] is not None:
                # Populate tensor_meta field for both output and input specs,
                # including for tuple output cases
                tensor_meta = None
                if spec_index < input_index:
                    if output_tensor_meta is not None:
                        if isinstance(output_tensor_meta, TensorMeta):
                            tensor_meta = output_tensor_meta
                        elif isinstance(output_tensor_meta, (tuple, list)):
                            if spec_index < len(output_tensor_meta):
                                tensor_meta = output_tensor_meta[spec_index]
                else:
                    input_strategy_index = spec_index - input_index
                    if input_strategy_index < len(input_args_strategy):
                        tensor_meta = input_args_strategy[
                            input_strategy_index
                        ].tensor_meta

                # pyrefly: ignore [bad-argument-type]
                spec_list.append(DTensorSpec(mesh, specs, tensor_meta=tensor_meta))
                spec_index += 1
            else:
                spec_list.append(None)

        input_specs: list[DTensorSpec] = [
            s for s in spec_list[input_index:] if isinstance(s, DTensorSpec)
        ]

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

        output_specs: tuple[DTensorSpec | None, ...]
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

        # Generate shard order variants for output specs if needed
        shard_order_variants_list: list[tuple[tuple[int, ...], ...] | None] = [None]
        if unique_shard_orders:
            # For single output case
            if not isinstance(output_specs, tuple):
                output_placements = output_specs.placements
                shard_order_variants_list = _generate_shard_order_variants(
                    output_placements, unique_shard_orders, mesh
                )
            # For multiple outputs, we could extend this later if needed

        # Create an OpSpec for each shard order variant
        for shard_order_raw in shard_order_variants_list:
            # Recreate output_specs with the specific shard_order if needed
            output_specs_with_shard_order: tuple[DTensorSpec | None, ...] | DTensorSpec
            if shard_order_raw is not None and not isinstance(output_specs, tuple):
                from torch.distributed.tensor._dtensor_spec import ShardOrderEntry

                # Convert shard_order_raw to proper ShardOrderEntry format
                # shard_order_raw is ((mesh_dims,),) e.g., ((1, 0),)
                # We need to find which tensor_dim this applies to
                tensor_dim_to_mesh_dims: dict[int, list[int]] = {}
                for mesh_dim, placement in enumerate(output_specs.placements):
                    if isinstance(placement, (Shard, _StridedShard)):
                        tensor_dim = placement.dim
                        if tensor_dim not in tensor_dim_to_mesh_dims:
                            tensor_dim_to_mesh_dims[tensor_dim] = []
                        tensor_dim_to_mesh_dims[tensor_dim].append(mesh_dim)

                # Build ShardOrderEntry tuple
                # shard_order_raw is a tuple of mesh_dims tuples, one for each multi-sharded tensor dim
                # We need to match each mesh_dims tuple to its corresponding tensor_dim
                shard_order_entries = []
                all_matched = True
                for mesh_dims_tuple in shard_order_raw:
                    # Find the tensor_dim that is sharded across these exact mesh_dims
                    matched = False
                    for tensor_dim, mesh_dims_list in sorted(
                        tensor_dim_to_mesh_dims.items()
                    ):
                        if set(mesh_dims_list) == set(mesh_dims_tuple) and len(
                            mesh_dims_list
                        ) == len(mesh_dims_tuple):
                            # Only create entry if this tensor_dim is actually multi-sharded
                            if len(mesh_dims_list) > 1:
                                shard_order_entries.append(
                                    ShardOrderEntry(
                                        tensor_dim=tensor_dim, mesh_dims=mesh_dims_tuple
                                    )
                                )
                                matched = True
                                break

                    if not matched:
                        # This shard_order_raw doesn't apply to this output spec
                        all_matched = False
                        break

                # Only create a variant if all mesh_dims tuples were successfully matched
                if not all_matched or not shard_order_entries:
                    # Skip this shard_order variant, use default instead
                    output_specs_with_shard_order = output_specs
                else:
                    shard_order = tuple(shard_order_entries)
                    try:
                        output_specs_with_shard_order = DTensorSpec(
                            mesh=output_specs.mesh,
                            placements=output_specs.placements,
                            tensor_meta=output_specs.tensor_meta,
                            shard_order=shard_order,
                        )
                    except AssertionError:
                        # Invalid shard_order for these placements, use default
                        output_specs_with_shard_order = output_specs
            else:
                output_specs_with_shard_order = output_specs

            redistribute_cost = [
                generate_redistribute_costs(input_strategy, input_spec)
                for input_strategy, input_spec in zip(input_args_strategy, input_specs)
            ]

            strategy = OpSpec(
                output_specs=output_specs_with_shard_order,
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
