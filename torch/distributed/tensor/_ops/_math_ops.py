# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import (
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import (
    as_list,
    generate_redistribute_costs,
    is_tensor_evenly_shardable,
    is_tensor_evenly_shardable_on_dim,
    normalize_dim,
    normalize_dims,
    register_op_strategy,
)
from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset,
    normalize_to_torch_size,
)
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _StridedShard,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false


aten = torch.ops.aten
prims = torch.ops.prims


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


@dataclass(frozen=True)
class NormReduction:
    norm_type: int | float


ReductionOpType = NormReduction | str


@dataclass(frozen=True)
class _NormPartial(Partial):
    """
    This placement is used for partial p-norm (p not in {inf, -inf, 0}).

    For p-norms, the p-norm over n elements computes (sum_i x_i^p)^(1/p).
    For example, consider 2 ranks, a (4,) tensor sharded on dim-0, and 2-norm:
        Rank 0: [t1, t2] | Rank 1: [t3, t4]
    After computing 2-norm per gradient (partial placement):
        Rank 0: [sqrt(t1^2 + t2^2)] | Rank 1: [sqrt(t3^2 + t4^2)]
    Converting from partial to replicate wants to ultimately get:
        Rank 0/1: [sqrt(t1^2 + t2^2 + t3^2 + t4^2)]
    This is achieved by: x^p -> allreduce sum -> x^(1/p).
    """

    norm_type: int | float = 2

    def __init__(self, norm_type: int | float = 2):
        super().__init__("sum")
        object.__setattr__(self, "norm_type", norm_type)

    def _partition_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        return tensor / math.pow(mesh.size(mesh_dim), 1 / self.norm_type)

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        if not isinstance(shard_spec, Shard):
            raise AssertionError(f"Expected Shard, got {type(shard_spec)}")
        tensor = self._pre_reduce_transform(tensor)
        reduced_tensor = super()._reduce_shard_value(tensor, mesh, mesh_dim, shard_spec)
        return self._post_reduce_transform(reduced_tensor)

    def _reduce_value(
        self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int
    ) -> torch.Tensor:
        tensor = self._pre_reduce_transform(tensor)
        reduced_tensor = super()._reduce_value(tensor, mesh, mesh_dim)
        return self._post_reduce_transform(reduced_tensor)

    def _pre_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor**self.norm_type

    def _post_reduce_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor ** (1.0 / self.norm_type)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _NormPartial):
            return False
        return self.norm_type == other.norm_type

    def __hash__(self) -> int:
        return 1 + hash(self.norm_type)

    def __repr__(self) -> str:
        return f"_NormPartial({self.norm_type})"

    def __str__(self) -> str:
        return f"_NormP({self.norm_type})"


def _infer_reduction_dims(dims_arg: object, ndim: int) -> list[int] | None:
    if dims_arg is None:
        return None
    dims = cast(list[int], as_list(dims_arg))
    dims = cast(list[int], normalize_dims(dims, ndim))
    empty_dims = [[0], [-1], []]
    if ndim == 0 and dims_arg in empty_dims:
        return None
    return dims


def _infer_reduce_dims_map(
    reduction_dims: list[int], input_ndim: int, keep_dim=False
) -> list[int]:
    reduction_dims_map = []
    new_dim_count = 0
    for input_dim in range(input_ndim):
        if input_dim in reduction_dims and not keep_dim:
            # if input dim in reduction dims, mark it as -1
            reduction_dims_map.append(-1)
        else:
            # otherwise mark it as the new dim
            reduction_dims_map.append(new_dim_count)
            new_dim_count += 1

    return reduction_dims_map


def _replicate_dims_start_at(
    placements: Sequence[Placement], start_dim: int = 0
) -> tuple[Placement, ...]:
    new_placements: list[Placement] = []
    for p in placements:
        if p.is_partial() or (_is_shard_like(p) and p.dim >= start_dim):
            new_placements.append(Replicate())  # make it replicate
        else:
            new_placements.append(p)  # keep the placement
    return tuple(new_placements)


# return new_placements which align with placements but skip the skipped_dim
# Precondition: no shard-like placement on skipped_dim (callers must
# replicate it first via replicate_reduction_dims).
def _skip_dim(
    placements: tuple[Placement, ...], skipped_dim: int
) -> tuple[Placement, ...]:
    new_placements: list[Placement] = []
    for p in placements:
        if isinstance(p, _StridedShard) and p.dim >= skipped_dim:
            new_placements.append(_StridedShard(p.dim - 1, split_factor=p.split_factor))
        elif isinstance(p, Shard) and p.dim >= skipped_dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)
    return tuple(new_placements)


def replicate_reduction_dims(
    placements: tuple[Placement, ...], reduction_dims: list[int]
) -> tuple[Placement, ...]:
    # replicate the reduction dims if not reduction_linear
    new_placements: list[Placement] = []

    for p in placements:
        if p.is_partial():
            new_placements.append(Replicate())
        elif _is_shard_like(p) and p.dim in reduction_dims:
            new_placements.append(Replicate())
        else:
            new_placements.append(p)

    return tuple(new_placements)


def map_placements_after_reduction(
    placements: tuple[Placement, ...],
    reduction_dims: list[int],
    reduction_dims_map: list[int],
    reduction_op: ReductionOpType,
) -> tuple[Placement, ...]:
    """
    Map each placement based on the output shape after reduction.
    """
    new_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        else:
            if not _is_shard_like(placement):
                raise AssertionError(
                    f"Expected Shard/_StridedShard, got {type(placement)}"
                )
            shard_dim = placement.dim
            new_shard_dim = reduction_dims_map[shard_dim]
            if new_shard_dim == -1 or shard_dim in reduction_dims:
                # if new_shard_dim collapsed or its in the reduction dims
                # (i.e. for the case where keepdims=True), we generate partial
                new_placements.append(get_placement_from_reduction_op(reduction_op))
            else:
                if isinstance(placement, _StridedShard):
                    new_placements.append(
                        _StridedShard(
                            new_shard_dim, split_factor=placement.split_factor
                        )
                    )
                elif isinstance(placement, Shard):
                    new_placements.append(Shard(new_shard_dim))
    return tuple(new_placements)


def get_placement_from_reduction_op(reduction_op: ReductionOpType) -> Placement:
    if isinstance(reduction_op, NormReduction):
        if reduction_op.norm_type == 0:
            # return P(sum) for easier reduction_linear handling.
            return Partial("sum")
        return _NormPartial(norm_type=reduction_op.norm_type)
    return Partial(reduction_op)


def common_reduction_strategy(
    input_strategy: OpStrategy,
    reduce_dims: list[int],
    keep_dim: bool = False,
    reduction_linear: bool = True,
    reduction_op: ReductionOpType = "sum",
) -> OpStrategy:
    """
    reduction_linear means that the reduction `f` follows this rule:
        f([f(a), f(b)]) = f([a, b])

    reduction linear should be super set of linearity.
    """
    reduction_strategy = OpStrategy([])

    for op_spec in input_strategy.strategies:
        is_reduction_linear = reduction_linear
        if reduction_op == "avg":
            output_spec = op_spec.output_spec
            local_shape = list(output_spec.tensor_meta.shape)  # type:ignore[union-attr]
            for dim in reduce_dims:
                if not is_tensor_evenly_shardable_on_dim(local_shape, output_spec, dim):
                    is_reduction_linear = False
                    break

        for p in op_spec.output_spec.placements:
            if isinstance(p, Partial) and p.reduce_op != reduction_op:
                is_reduction_linear = False
                break

        if not is_reduction_linear:
            input_placements = replicate_reduction_dims(
                op_spec.output_spec.placements, reduce_dims
            )
        else:
            input_placements = op_spec.output_spec.placements

        input_spec = DTensorSpec(
            mesh=input_strategy.mesh,
            placements=input_placements,
            tensor_meta=op_spec.output_spec.tensor_meta,
        )

        reduce_dims_map = _infer_reduce_dims_map(reduce_dims, input_spec.ndim, keep_dim)
        out_placements = map_placements_after_reduction(
            input_spec.placements, reduce_dims, reduce_dims_map, reduction_op
        )
        reduction_strategy.strategies.append(
            OpSpec(
                output_specs=DTensorSpec(
                    mesh=input_strategy.mesh,
                    placements=out_placements,
                ),
                input_specs=(input_spec,),
                redistribute_cost=[
                    generate_redistribute_costs(input_strategy, input_spec)
                ],
            )
        )

    return reduction_strategy


def _reduction_single_dim_strategy(
    args_schema: tuple[Any, ...],
    reduction_dims: list[int] | None,
    keep_dim: bool,
    reduction_linear: bool,
    reduction_op: ReductionOpType,
    extra_partial_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Build single-dim strategies for reduction ops.

    For non-reduction dims: shard input and output on that dim (with dim remapping).
    For reduction dims (if reduction_linear): shard input on reduction dim, output is
    Partial. Also adds partial propagation rules for linear reductions.

    "avg" reductions do not shard reduced dims because averaging uneven local
    averages does not produce the global average.
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    reduce_dims = list(range(ndim)) if reduction_dims is None else reduction_dims

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    for d in range(ndim):
        if d in reduce_dims:
            if reduction_linear and reduction_op != "avg":
                # Shard on reduction dim -> Partial output.
                strategies.append(
                    [
                        get_placement_from_reduction_op(reduction_op),
                        _ShardingPlaceholder(d),
                    ]
                )
        else:
            # Shard on non-reduction dim -> shard output (with dim remapping)
            if keep_dim:
                out_d = d
            else:
                out_d = d - sum(1 for rd in reduce_dims if rd < d)
            strategies.append([_ShardingPlaceholder(out_d), _ShardingPlaceholder(d)])

    # Partial propagation: Partial(reduction_op) input -> Partial(reduction_op) output.
    # Skip for NormReduction: the old OpStrategy path did not propagate
    # _NormPartial through, so keep that behavior.
    if reduction_linear and not isinstance(reduction_op, NormReduction):
        partial_placement = get_placement_from_reduction_op(reduction_op)
        strategies.append([partial_placement, partial_placement])

    if extra_partial_rules:
        strategies.extend(extra_partial_rules)

    return strategies


# Category C: Linear reductions
# all/any are listed here because the remaining reduction helpers share this map,
# but they are registered separately with reduction_linear=False.
LINEAR_REDUCTION_OP_MAP = {
    aten.all.default: "product",
    aten.all.dim: "product",
    aten.any.default: "sum",
    aten.any.dim: "sum",
    aten.any.dims: "sum",
    aten.any.out: "sum",
    aten.sum.default: "sum",
    aten.sum.dim_IntList: "sum",
    prims.sum.default: "sum",
    # These are only valid when there is no padding
    aten.prod.default: "product",
    aten.prod.dim_int: "product",
    aten.prod.int_out: "product",
    prims.prod.default: "product",
    aten.max.default: "max",
    aten.max.out: "max",
    aten.min.default: "min",
    aten.min.out: "min",
    aten.amax.default: "max",
    aten.amax.out: "max",
    prims.amax.default: "max",
    aten.amin.default: "min",
    aten.amin.out: "min",
    prims.amin.default: "min",
    aten.nansum.default: "sum",
}

# all/any are NOT linear reductions: validation showed S(reduction_dim)->P(...)
# produces incorrect results. They only support sharding on non-reduction dims.
NON_LINEAR_BOOL_REDUCTION_OPS = [
    aten.all.default,
    aten.all.dim,
    aten.any.default,
    aten.any.dim,
    aten.any.dims,
    aten.any.out,
]

MEAN_REDUCTION_OPS = [
    aten.mean.default,
    aten.mean.dim,
    aten.mean.out,
]

ARGMAX_ARGMIN_OPS = {
    aten.argmax.default: "max",
    aten.argmin.default: "min",
}


LINEAR_REDUCTION_OPS = [
    op for op in LINEAR_REDUCTION_OP_MAP if op not in NON_LINEAR_BOOL_REDUCTION_OPS
]


@register_single_dim_strategy(
    LINEAR_REDUCTION_OPS,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def linear_reduction_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    reduction_op = (
        "max" if op == aten._foreach_max.default else LINEAR_REDUCTION_OP_MAP[op]
    )
    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=True,
        reduction_op=reduction_op,
    )


register_single_dim_strategy(
    [aten._foreach_max.default],
    schema_info=RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)(linear_reduction_single_dim_strategy)


def _is_spec_evenly_sharded_on_dim(spec: DTensorSpec, dim: int) -> bool:
    shape = spec.shape
    dim = normalize_dim(dim, len(shape))

    num_shards = 1
    for mesh_dim, placement in enumerate(spec.placements):
        if _is_shard_like(placement) and placement.dim == dim:
            num_shards *= spec.mesh.size(mesh_dim)
            if isinstance(placement, _StridedShard):
                num_shards *= placement.split_factor

    return guard_or_false(cast(Any, shape[dim]) % num_shards == 0)


def _mean_full_mesh_strategy_filter(
    mesh: DeviceMesh,
    op_schema: OpSchema,
    input_specs: list[DTensorSpec],
    output_specs: DTensorSpec | tuple[DTensorSpec | None, ...],
) -> bool:
    input_spec = input_specs[0]
    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(op_schema.args_schema[1], input_spec.ndim)
    reduce_dims = list(range(input_spec.ndim)) if dims is None else dims

    return all(_is_spec_evenly_sharded_on_dim(input_spec, dim) for dim in reduce_dims)


@register_single_dim_strategy(
    MEAN_REDUCTION_OPS,
    schema_info=RuntimeSchemaInfo(1),
    full_mesh_strategy_filter=_mean_full_mesh_strategy_filter,
)
def mean_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    reduce_dims = list(range(ndim)) if dims is None else dims
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d in reduce_dims:
            strategies.append([Partial("avg"), _ShardingPlaceholder(d)])
        else:
            if keep_dim:
                out_d = d
            else:
                out_d = d - sum(1 for rd in reduce_dims if rd < d)
            strategies.append([_ShardingPlaceholder(out_d), _ShardingPlaceholder(d)])

    strategies.append([Partial("avg"), Partial("avg")])
    return strategies


@register_single_dim_strategy(
    NON_LINEAR_BOOL_REDUCTION_OPS,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def bool_reduction_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """all/any: only shard on non-reduction dims (no partial propagation)."""
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=False,
        reduction_op="sum",
    )


def _shard_non_reduction_dim(
    args_schema: tuple[Any, ...],
    dim: int,
    keep_dim: bool,
    n_outputs: int,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Shared logic for ops that shard on non-reduction dims with dim remapping.

    Used by dim-reduction ops returning (values, indices) and argmax/argmin.

    Args:
        args_schema: Op args with TensorMeta at position 0.
        dim: The reduction dimension.
        keep_dim: Whether the reduction dim is kept.
        n_outputs: Number of output tensors (1 for argmax, 2 for max.dim).
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")

    ndim = len(input_meta.shape)
    dim = normalize_dim(dim, ndim)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d == dim:
            continue
        out_d = d if keep_dim or d < dim else d - 1
        strategies.append(
            [_ShardingPlaceholder(out_d)]  # pyrefly: ignore[bad-argument-type]
            * n_outputs
            + [_ShardingPlaceholder(d)]
        )
    return strategies


# Category B: Dim-reduction ops returning (values, indices)
# max.dim/min.dim return (values, indices). Indices are local to each shard
# and cannot be combined across ranks, so we force Replicate on reduction dims
# (same approach as argmax/argmin).
@register_single_dim_strategy(
    [aten.max.dim, aten.min.dim],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def max_min_dim_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[1])
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return _shard_non_reduction_dim(args_schema, dim, keep_dim, n_outputs=2)


def argmax_argmin_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """argmax/argmin return indices only. Shard on non-reduction dims."""
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)
    reduce_dims = list(range(ndim)) if dims is None else dims
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])

    if len(reduce_dims) == ndim:
        return []

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d in reduce_dims:
            continue
        out_d = d
        if not keep_dim:
            out_d = d - sum(1 for rd in reduce_dims if rd < d)
        strategies.append([_ShardingPlaceholder(out_d), _ShardingPlaceholder(d)])
    return strategies


@register_op_strategy(list(ARGMAX_ARGMIN_OPS.keys()), schema_info=RuntimeSchemaInfo(1))
def argmax_argmin_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    if not isinstance(input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(input_strategy)}")

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)

    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    reduction_op = ARGMAX_ARGMIN_OPS[op_schema.op]
    return common_reduction_strategy(
        input_strategy,
        reduce_dims,
        keep_dim=keep_dim,
        reduction_linear=False,
        reduction_op=reduction_op,
    )


def _shard_except_dim_strategy(
    n_placements: int,
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
    active_dim: int,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Build single-dim strategies that shard on every dim except an active dim.

    Used by sort-like ops (sort, topk, cummax, cummin), scan ops (cumsum, cumprod,
    logcumsumexp), and softmax-like ops. All outputs and inputs get the same sharding
    placeholder.

    Args:
        n_placements: Total number of placements per rule (outputs + inputs).
        active_dim: The dim to exclude from sharding (e.g. sort dim, softmax dim).
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    active_dim = normalize_dim(active_dim, ndim)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d != active_dim:
            strategies.append([_ShardingPlaceholder(d)] * n_placements)
    return strategies


# Category A: Sort-like and scan ops
# Scan: 1 output + 1 input = 2 placements
_SCAN_N_PLACEMENTS = 2
# Sort-like: 2 outputs (values, indices) + 1 input = 3 placements
_SORT_LIKE_N_PLACEMENTS = 3
# Softmax forward: 1 output + 1 input = 2 placements
_SOFTMAX_FWD_N_PLACEMENTS = 2
# Softmax backward: 1 output + 2 inputs = 3 placements
_SOFTMAX_BWD_N_PLACEMENTS = 3


@register_single_dim_strategy(
    [aten.cumsum.default, aten.cumprod.default, aten.logcumsumexp.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def scan_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = args_schema[1]
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    strategies = _shard_except_dim_strategy(
        _SCAN_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=dim
    )
    # cumsum is linear for sum/avg: partial sums/avgs propagate through
    if op == aten.cumsum.default:
        strategies.append([Partial("sum"), Partial("sum")])
        strategies.append([Partial("avg"), Partial("avg")])
    return strategies


@register_single_dim_strategy(
    [aten.median.default, aten.nanmedian.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def global_median_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Global median reduces all dims — only Replicate is valid."""
    return []


@register_single_dim_strategy(
    [aten.median.dim, aten.nanmedian.dim, aten.mode.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def dim_reduction_with_indices_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else -1
    keep_dim = len(args_schema) > 2 and bool(args_schema[2])
    return _shard_non_reduction_dim(args_schema, dim, keep_dim, n_outputs=2)


@register_single_dim_strategy(
    [aten.kthvalue.default],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
)
def kthvalue_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[2]) if len(args_schema) > 2 else -1
    keep_dim = len(args_schema) > 3 and bool(args_schema[3])
    return _shard_non_reduction_dim(args_schema, dim, keep_dim, n_outputs=2)


@register_single_dim_strategy(
    [aten.cummax.default, aten.cummin.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def cummax_cummin_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    dim = cast(int, args_schema[1])
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=dim
    )


_STD_VAR_OPS = [
    aten.std.correction,
    aten.std.correction_out,
    aten.var.correction,
    aten.var.correction_out,
    aten.var_mean.correction,
    aten.var_mean.correction_out,
    prims.var.default,
]


@register_single_dim_strategy(
    _STD_VAR_OPS,
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def std_var_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = None
    if len(args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], ndim)

    keep_dim = cast(bool, kwargs_schema.get("keepdim", False))
    strategies = _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=False,
        reduction_op="sum",
    )
    num_outputs = len(op._schema.returns)
    if num_outputs == 1 and not any(
        isinstance(arg, TensorMeta) for arg in kwargs_schema.values()
    ):
        return strategies

    expanded_strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for rule in strategies:
        output_placements = [rule[0]] * num_outputs
        input_placements = rule[1:]
        kwarg_placements: list[Placement | _ShardingPlaceholder] = []
        for arg in kwargs_schema.values():
            if isinstance(arg, TensorMeta):
                kwarg_placements.append(output_placements[len(kwarg_placements)])
        expanded_strategies.append(
            output_placements + input_placements + kwarg_placements
        )
    return expanded_strategies


def _get_norm_reduction_op(norm_type: int | float | str) -> ReductionOpType:
    """Get the reduction op for vector/foreach norm based on norm_type.

    For inf/-inf norms, returns simple reduction ops ("max", "min").
    For other norms (including 0), returns NormReduction which produces the
    appropriate Partial placement via get_placement_from_reduction_op.
    """
    if norm_type in (float("inf"), "inf"):
        return "max"
    elif norm_type in (float("-inf"), "-inf"):
        return "min"
    else:
        if not isinstance(norm_type, (int, float)):
            raise AssertionError
        return NormReduction(norm_type)


@register_single_dim_strategy(
    [aten.linalg_vector_norm.default, aten.norm.Scalar],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def vector_norm_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Vector norm: linear reduction with NormPartial.

    S(reduction_dim)->_NormPartial(ord) rules appear as false positives in
    strategy_validation because the validator doesn't enumerate _NormPartial
    as an output placement. The rules are correct at runtime.
    TODO: remove this comment when _NormPartial is deprecated.
    """
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    norm_type = args_schema[1] if len(args_schema) > 1 else 2
    if not isinstance(norm_type, (int, float, str)):
        raise AssertionError(f"Expected int, float, or str, got {type(norm_type)}")
    dim = args_schema[2] if len(args_schema) > 2 else None
    keepdim = args_schema[3] if len(args_schema) > 3 else False
    dims = _infer_reduction_dims(dim, ndim)

    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=cast(bool, keepdim),
        reduction_linear=True,
        reduction_op=_get_norm_reduction_op(norm_type),
    )


register_single_dim_strategy(
    [aten._foreach_norm.Scalar],
    schema_info=RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)(vector_norm_single_dim_strategy)


@register_single_dim_strategy(
    [aten.linalg__powsum.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def powsum_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """linalg__powsum: computes sum(|x|^ord), always reducible with Partial("sum")."""
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dim = args_schema[2] if len(args_schema) > 2 else None
    keepdim = args_schema[3] if len(args_schema) > 3 else False
    dims = _infer_reduction_dims(dim, ndim)

    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=cast(bool, keepdim),
        reduction_linear=True,
        reduction_op="sum",
    )


register_single_dim_strategy(
    [aten._foreach_powsum.Scalar],
    schema_info=RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)(powsum_single_dim_strategy)


_REPLICATE_ONLY_OPS = [
    aten._linalg_svd.default,
    aten.linalg_qr.default,
    # TODO: The diagonal ops can have an improved sharding strategy for
    # shard placements that does not require redistributing to replicate.
    aten.diagonal_copy.default,
    aten.diag_embed.default,
    aten.diag.default,
    aten.diagonal.default,
    aten.tril.default,
    aten.triu.default,
    aten._linalg_eigh.default,
]


@register_single_dim_strategy(
    _REPLICATE_ONLY_OPS,
    schema_info=RuntimeSchemaInfo(1),
)
def linalg_replicate_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Replicate-only ops: only the all-Replicate strategy is valid."""
    return []


# Maps each pooling op to its spatial rank (number of spatial dimensions).
# Batched inputs have layout (N, C, *spatial) with ndim = spatial_rank + 2;
# unbatched inputs drop the batch dim giving ndim = spatial_rank + 1.
POOL_SPATIAL_RANK: dict[torch._ops.OpOverload, int] = {
    aten.avg_pool1d.default: 1,
    aten.avg_pool2d.default: 2,
    aten.avg_pool3d.default: 3,
    aten.adaptive_avg_pool1d.default: 1,
    aten._adaptive_avg_pool2d.default: 2,
    aten._adaptive_avg_pool3d.default: 3,
    aten.adaptive_max_pool1d.default: 1,
    aten.adaptive_max_pool2d.default: 2,
    aten.adaptive_max_pool3d.default: 3,
    aten.fractional_max_pool2d.default: 2,
    aten.fractional_max_pool3d.default: 3,
    aten.max_pool1d_with_indices.default: 1,
    aten.max_pool2d_with_indices.default: 2,
    aten.max_pool3d_with_indices.default: 3,
}

AVG_POOL_OPS = [
    aten.avg_pool1d.default,
    aten.avg_pool2d.default,
    aten.avg_pool3d.default,
    aten.adaptive_avg_pool1d.default,
    aten._adaptive_avg_pool2d.default,
    aten._adaptive_avg_pool3d.default,
]

MAX_POOL_OPS = [
    aten.adaptive_max_pool1d.default,
    aten.adaptive_max_pool2d.default,
    aten.adaptive_max_pool3d.default,
    aten.fractional_max_pool2d.default,
    aten.fractional_max_pool3d.default,
    aten.max_pool1d_with_indices.default,
    aten.max_pool2d_with_indices.default,
    aten.max_pool3d_with_indices.default,
]


@register_single_dim_strategy(
    AVG_POOL_OPS + MAX_POOL_OPS,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def pooling_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")

    ndim = len(input_meta.shape)
    num_outputs = sum(1 for r in op._schema.returns if "Tensor" in str(r.type))
    num_inputs = sum(1 for a in args_schema if isinstance(a, TensorMeta))
    n = num_outputs + num_inputs

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # S(0) for batch dim — always valid
    strategies.append([_ShardingPlaceholder(0)] * n)

    # avg_pool is linear: Partial(sum) and Partial(avg) pass through unchanged.
    if op in AVG_POOL_OPS:
        strategies.append([Partial("sum")] * n)
        strategies.append([Partial("avg")] * n)

    # S(1) is safe when dim 1 is the channel dim (pooling never touches it).
    # Batched inputs have layout (N, C, *spatial) with ndim = spatial_rank + 2.
    spatial_rank = POOL_SPATIAL_RANK[op]
    is_batched = ndim >= spatial_rank + 2
    if is_batched:
        strategies.append([_ShardingPlaceholder(1)] * n)

    return strategies


# Category F: Softmax-like ops
@register_single_dim_strategy(
    [aten._log_softmax.default, aten._softmax.default, aten._safe_softmax.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def softmax_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    softmax_dim = cast(int, args_schema[1])
    return _shard_except_dim_strategy(
        _SOFTMAX_FWD_N_PLACEMENTS,
        op,
        args_schema,
        kwargs_schema,
        active_dim=softmax_dim,
    )


@register_single_dim_strategy(
    [
        aten._log_softmax_backward_data.default,
        aten._softmax_backward_data.default,
    ],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
    allow_unbacked_sharding=False,
)
def softmax_backward_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    softmax_dim = cast(int, args_schema[2])
    return _shard_except_dim_strategy(
        _SOFTMAX_BWD_N_PLACEMENTS,
        op,
        args_schema,
        kwargs_schema,
        active_dim=softmax_dim,
    )


@register_op_strategy(
    [aten.nll_loss_forward.default, aten.nll_loss2d_forward.default],
    schema_info=RuntimeSchemaInfo(3),
)
def nll_loss_forward_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()

    if not len(op_schema.args_schema) == 5:
        raise AssertionError(f"Expected 5 args, got {len(op_schema.args_schema)}")

    (
        input_strategy,
        target_strategy,
        weight_strategy,
        reduction,
        _,
    ) = op_schema.args_schema
    input_strategy = cast(OpStrategy, input_strategy)
    target_strategy = cast(OpStrategy, target_strategy)
    reduction = cast(int, reduction)

    input_shape = input_strategy.shape
    channel_dim = 1 if len(input_shape) >= 2 else 0

    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []

        # make sure input is replicated along the channel dim
        input_src_spec = input_placement_strategy.output_spec
        input_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(
                input_src_spec.placements, [channel_dim]
            ),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_expected_spec)
        )

        # target doesn't have channel dim, and it follows input on other dims
        target_src_spec = target_strategy.strategies[idx].output_spec
        target_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=_skip_dim(input_expected_spec.placements, channel_dim),
            tensor_meta=target_src_spec.tensor_meta,
        )
        op_args_target_specs.append(target_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(target_strategy, target_expected_spec)
        )

        # weight tensor, if given, has to be a Tensor of size input_shape[channel_dim]
        # make sure it is replicated
        if weight_strategy is not None:
            if not isinstance(weight_strategy, OpStrategy):
                raise AssertionError(
                    f"Expected OpStrategy, got {type(weight_strategy)}"
                )
            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_expected_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_expected_spec)
            )

        if reduction == Reduction.NONE.value:
            output_expected_spec = target_expected_spec
            total_weight_expected_spec = DTensorSpec(
                mesh=mesh, placements=tuple([Replicate()] * mesh.ndim)
            )
        else:
            if reduction == Reduction.MEAN.value:
                reduction_op = "avg"
                if not is_tensor_evenly_shardable(
                    target_expected_spec.shape, target_expected_spec
                ):
                    raise ValueError(
                        "The intermediate results of nll_loss cannot be evenly sharded, \
                        resulting in biased mean result."
                    )
            else:  # reduction == Reduction.SUM.value:
                reduction_op = "sum"
            reduce_dims = list(range(target_expected_spec.ndim))
            reduce_dims_map = _infer_reduce_dims_map(
                reduce_dims, target_expected_spec.ndim, keep_dim=False
            )
            out_placements = map_placements_after_reduction(
                target_expected_spec.placements,
                reduce_dims,
                reduce_dims_map,
                reduction_op,
            )
            output_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=out_placements,
            )

            # whether reduction is sum or mean, the total weight has to be summed up if not replicated
            total_weight_placements = map_placements_after_reduction(
                target_expected_spec.placements,
                reduce_dims,
                reduce_dims_map,
                "sum",
            )
            total_weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=total_weight_placements,
            )

        output_strategy.strategies.append(
            OpSpec(
                output_specs=(output_expected_spec, total_weight_expected_spec),
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy(
    [aten.nll_loss_backward.default, aten.nll_loss2d_backward.default],
    schema_info=RuntimeSchemaInfo(4),
)
def nll_loss_backward_strategy(op_schema: OpSchema) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    if not len(op_schema.args_schema) == 7:
        raise AssertionError(f"Expected 7 args, got {len(op_schema.args_schema)}")
    (
        grad_out_strategy,
        input_strategy,
        target_strategy,
        weight_strategy,
        reduction,
        _,
        total_weight_strategy,
    ) = op_schema.args_schema
    grad_out_strategy = cast(OpStrategy, grad_out_strategy)
    input_strategy = cast(OpStrategy, input_strategy)
    target_strategy = cast(OpStrategy, target_strategy)
    reduction = cast(int, reduction)
    total_weight_strategy = cast(OpStrategy, total_weight_strategy)

    input_shape = input_strategy.shape
    channel_dim = 1 if len(input_shape) >= 2 else 0

    grad_in_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []

        # make sure input is replicated along the channel dim
        input_src_spec = input_placement_strategy.output_spec
        input_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(
                input_src_spec.placements, [channel_dim]
            ),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(input_strategy, input_expected_spec)
        )

        # target doesn't have channel dim, and it follows input on other dims
        target_src_spec = target_strategy.strategies[idx].output_spec
        target_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=_skip_dim(input_expected_spec.placements, channel_dim),
            tensor_meta=target_src_spec.tensor_meta,
        )
        op_args_target_specs.append(target_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(target_strategy, target_expected_spec)
        )

        # grad_out follows target if there is no reduction;
        # otherwise, it should be a replicated scalar.
        grad_out_src_spec = grad_out_strategy.strategies[idx].output_spec
        if reduction == Reduction.NONE.value:
            grad_out_expected_spec = target_expected_spec
        else:
            grad_out_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(grad_out_src_spec.placements),
                tensor_meta=grad_out_src_spec.tensor_meta,
            )
        op_args_target_specs.insert(0, grad_out_expected_spec)
        redistribute_costs.insert(
            0, generate_redistribute_costs(grad_out_strategy, grad_out_expected_spec)
        )

        # weight tensor, if given, has to be a Tensor of size input_shape[channel_dim]
        # make sure it is replicated
        if weight_strategy is not None:
            if not isinstance(weight_strategy, OpStrategy):
                raise AssertionError(
                    f"Expected OpStrategy, got {type(weight_strategy)}"
                )
            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_expected_spec)
            redistribute_costs.append(
                generate_redistribute_costs(weight_strategy, weight_expected_spec)
            )

        # total_weight is only used by the backward kernel for reduction='mean'.
        # For reduction='sum' or 'none', it is unused, so no redistribution needed.
        total_weight_src_spec = total_weight_strategy.strategies[idx].output_spec
        if reduction == Reduction.MEAN.value:
            total_weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(total_weight_src_spec.placements),
                tensor_meta=total_weight_src_spec.tensor_meta,
            )
        else:
            total_weight_expected_spec = total_weight_src_spec
        op_args_target_specs.append(total_weight_expected_spec)
        redistribute_costs.append(
            generate_redistribute_costs(
                total_weight_strategy, total_weight_expected_spec
            )
        )

        grad_in_expected_spec = input_expected_spec
        grad_in_strategy.strategies.append(
            OpSpec(
                output_specs=grad_in_expected_spec,
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return grad_in_strategy


@register_single_dim_strategy(
    [aten.native_layer_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def layer_norm_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    normalized_shape = args_schema[1]
    weight_meta = args_schema[2]
    bias_meta = args_schema[3]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(axis):
        # [out, mean, rstd, input, weight?, bias?]
        rule: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(dim),  # out
            _ShardingPlaceholder(dim),  # mean
            _ShardingPlaceholder(dim),  # rstd
            _ShardingPlaceholder(dim),  # input
        ]
        if weight_meta is not None:
            rule.append(Replicate())
        if bias_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)
    return strategies


@register_single_dim_strategy(
    [aten._fused_rms_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def rms_norm_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    normalized_shape = args_schema[1]
    weight_meta = args_schema[2]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(axis):
        # [out, rrms, input, weight?]
        rule: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(dim),  # out
            _ShardingPlaceholder(dim),  # rrms
            _ShardingPlaceholder(dim),  # input
        ]
        if weight_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)
    return strategies


@register_single_dim_strategy(
    [aten.native_layer_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def layer_norm_bwd_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[1]
    normalized_shape = args_schema[2]
    # mean = args_schema[3], rstd = args_schema[4]
    weight_meta = args_schema[5]
    bias_meta = args_schema[6]
    output_mask = args_schema[7]
    compute_d_input = output_mask[0]
    compute_d_weight = weight_meta is not None and output_mask[1]
    compute_d_bias = bias_meta is not None and output_mask[2]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder | None]] = []
    for dim in range(axis):
        # outputs: [d_input, d_weight, d_bias] — always 3 per schema
        # Masked-off outputs use None even if the corresponding input exists.
        rule: list[Placement | _ShardingPlaceholder | None] = [
            _ShardingPlaceholder(dim) if compute_d_input else None,
            Partial("sum") if compute_d_weight else None,
            Partial("sum") if compute_d_bias else None,
        ]
        # inputs: [grad_out, input, mean, rstd, weight?, bias?]
        rule.extend(
            [
                _ShardingPlaceholder(dim),  # grad_out
                _ShardingPlaceholder(dim),  # input
                _ShardingPlaceholder(dim),  # mean
                _ShardingPlaceholder(dim),  # rstd
            ]
        )
        if weight_meta is not None:
            rule.append(Replicate())
        if bias_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)

    return strategies


@register_single_dim_strategy(
    [aten._fused_rms_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def rms_norm_bwd_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    input_meta = args_schema[1]
    normalized_shape = args_schema[2]
    # rstd = args_schema[3]
    weight_meta = args_schema[4]
    output_mask = args_schema[5]
    compute_d_input = output_mask[0]
    compute_d_weight = weight_meta is not None and output_mask[1]

    axis = len(input_meta.shape) - len(normalize_to_torch_size(normalized_shape))

    strategies: list[list[Placement | _ShardingPlaceholder | None]] = []
    for dim in range(axis):
        # outputs: [d_input, d_weight] — always 2 per schema
        # Masked-off outputs use None even if the corresponding input exists.
        # inputs: [grad_out, input, rstd, weight?]
        rule: list[Placement | _ShardingPlaceholder | None] = [
            _ShardingPlaceholder(dim) if compute_d_input else None,
            Partial("sum") if compute_d_weight else None,
            _ShardingPlaceholder(dim),  # grad_out
            _ShardingPlaceholder(dim),  # input
            _ShardingPlaceholder(dim),  # rstd
        ]
        if weight_meta is not None:
            rule.append(Replicate())
        strategies.append(rule)

    return strategies


@register_single_dim_strategy(
    [aten.topk.default],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
)
def topk_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    topk_dim = cast(int, args_schema[2]) if len(args_schema) > 2 else -1
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=topk_dim
    )


@register_single_dim_strategy(
    aten.sort.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def sort_default_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    sort_dim = cast(int, args_schema[1]) if len(args_schema) > 1 else -1
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=sort_dim
    )


@register_single_dim_strategy(
    aten.sort.stable,
    schema_info=RuntimeSchemaInfo(
        1,
        static_kwargkey=["dim", "descending", "stable"],
    ),
    allow_uneven_sharding=True,
)
def sort_stable_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    sort_dim = cast(int, kwargs_schema.get("dim", -1))
    return _shard_except_dim_strategy(
        _SORT_LIKE_N_PLACEMENTS, op, args_schema, kwargs_schema, active_dim=sort_dim
    )


@register_single_dim_strategy(
    [aten.histc.default],
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
)
def histc_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    # histc supports sharded input -> partial output only when min/max are specified
    if len(args_schema) == 4:
        for dim in range(ndim):
            strategies.append([Partial(), _ShardingPlaceholder(dim)])
    return strategies


@register_single_dim_strategy(
    [aten.logsumexp.default],
    schema_info=RuntimeSchemaInfo(
        static_argnum=1,
        static_kwargkey=["keepdim"],
    ),
    allow_uneven_sharding=True,
)
def logsumexp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)

    dims = _infer_reduction_dims(args_schema[1], ndim)
    keep_dim = cast(bool, kwargs_schema.get("keepdim", False))
    return _reduction_single_dim_strategy(
        args_schema,
        reduction_dims=dims,
        keep_dim=keep_dim,
        reduction_linear=False,
        reduction_op="sum",
    )


_LINALG_NUM_PLACEMENTS = {
    # 1 in 1 out
    aten.cholesky_inverse.default: 2,
    aten.linalg_matrix_exp.default: 2,
    # 2 in 1 out
    aten.cholesky_solve.default: 3,
    aten.linalg_householder_product.default: 3,
    aten.linalg_solve_triangular.default: 3,
    # 3 in 1 out
    aten.linalg_ldl_solve.default: 4,
    aten.linalg_lu_solve.default: 4,
    aten.ormqr.default: 4,
    # 1 in 2 out
    aten.geqrf.default: 3,
    aten.linalg_cholesky_ex.default: 3,
    aten.linalg_eig.default: 3,
    aten.linalg_inv_ex.default: 3,
    # 2 in 2 out
    aten.triangular_solve.default: 4,
    # 1 in 3 out
    aten._linalg_det.default: 4,
    aten.linalg_ldl_factor_ex.default: 4,
    aten.linalg_lu.default: 4,
    aten.linalg_lu_factor_ex.default: 4,
    # 2 in 3 out
    aten.lu_unpack.default: 5,
    # 1 in 4 out
    aten._linalg_slogdet.default: 5,
    # 2 in 4 out
    aten._linalg_solve_ex.default: 6,
    # 1 in
    aten._linalg_check_errors.default: 1,
}


def _linalg_batch_dim_strategies(
    ndim: int, n_placements: int
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Build single-dim strategies for linalg ops that operate on the last 1-2 dims.

    Returns sharding on each batch dim (all dims except the last 2), with all
    outputs and inputs sharded on the same dim.
    """
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim - 2):
        strategies.append([_ShardingPlaceholder(dim)] * n_placements)
    return strategies


def _get_ndim(tensor_meta: Any) -> int:
    if not isinstance(tensor_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(tensor_meta)}")
    return len(tensor_meta.shape)


@register_single_dim_strategy(
    [
        aten.cholesky_inverse.default,
        aten.linalg_matrix_exp.default,
        aten.cholesky_solve.default,
        aten.linalg_householder_product.default,
        aten.linalg_solve_triangular.default,
        aten.linalg_ldl_solve.default,
        aten.linalg_lu_solve.default,
        aten.ormqr.default,
        aten.geqrf.default,
        aten.linalg_cholesky_ex.default,
        aten.linalg_eig.default,
        aten.linalg_inv_ex.default,
        aten.triangular_solve.default,
        aten._linalg_det.default,
        aten.linalg_ldl_factor_ex.default,
        aten.linalg_lu.default,
        aten.linalg_lu_factor_ex.default,
        aten.lu_unpack.default,
        aten._linalg_slogdet.default,
        aten._linalg_solve_ex.default,
        aten._linalg_check_errors.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def linalg_batch_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    if op not in _LINALG_NUM_PLACEMENTS:
        raise AssertionError(f"Expected op in _LINALG_NUM_PLACEMENTS, got {op}")

    n_placements = _LINALG_NUM_PLACEMENTS[op]
    strategies = _linalg_batch_dim_strategies(ndim, n_placements=n_placements)

    if op == aten.linalg_solve_triangular.default:
        # solve_triangular(A, B) -> result: linear in B
        strategies.append([Partial(), Replicate(), Partial()])
        strategies.append([Partial("avg"), Replicate(), Partial("avg")])
        # A replicated, B sharded on batch dims (B may have more batch dims than A)
        ndim_b = _get_ndim(args_schema[1])
        for dim in range(ndim_b - 2):
            strategies.append(
                [_ShardingPlaceholder(dim), Replicate(), _ShardingPlaceholder(dim)]
            )
    elif op == aten.cholesky_solve.default:
        # cholesky_solve(B, A) -> result  (B is arg0)
        strategies.append([Partial(), Partial(), Replicate()])
    elif op == aten.linalg_lu_solve.default:
        # linalg_lu_solve(LU, pivots, B) -> result
        strategies.append([Partial(), Replicate(), Replicate(), Partial()])
    elif op == aten.linalg_ldl_solve.default:
        # linalg_ldl_solve(LD, pivots, B) -> result
        strategies.append([Partial(), Replicate(), Replicate(), Partial()])
    elif op == aten.ormqr.default:
        # ormqr(a, tau, C) -> result  (linear in C)
        strategies.append([Partial(), Replicate(), Replicate(), Partial()])
    elif op == aten._linalg_solve_ex.default:
        # _linalg_solve_ex(A, B) -> (result, LU, pivots, info)
        strategies.append(
            [Partial(), Replicate(), Replicate(), Replicate(), Replicate(), Partial()]
        )

    return strategies


# linalg_pinv has optional tensor kwargs atol, rtol (scalar tensors when present).
# Schema: (Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian) -> Tensor
# When atol/rtol are None, num_inputs=1; when present, they add to num_inputs.
@register_single_dim_strategy(
    [aten.linalg_pinv.atol_rtol_tensor],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def linalg_pinv_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    # Count optional tensor kwargs that are actually present
    extra_tensors = sum(
        isinstance(kwargs_schema.get(k), TensorMeta) for k in ("atol", "rtol")
    )
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim - 2):
        s: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim),
        ]
        # atol, rtol are scalar tensors — always Replicate
        s.extend([Replicate()] * extra_tensors)
        strategies.append(s)
    return strategies


# linalg_cross is pointwise on every dim except the cross-product dim (which
# must be size 3).  Shard on any other dim.
@register_single_dim_strategy(
    [aten.linalg_cross.default],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def linalg_cross_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    ndim = _get_ndim(args_schema[0])
    cross_dim = kwargs_schema.get("dim", -1) % ndim
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(ndim):
        if dim == cross_dim:
            continue
        strategies.append([_ShardingPlaceholder(dim)] * 3)
    return strategies


# ---------------------------------------------------------------------------
# Interpolation / upsample / pooling ops
#
# These ops operate on spatial dims and are safely shardable on batch (dim 0)
# and channel (dim 1). grid_sampler is batch-only because the grid tensor has
# no channel dimension.
# ---------------------------------------------------------------------------


@register_single_dim_strategy(
    [
        # Forward ops
        aten.upsample_nearest1d.default,
        aten.upsample_nearest2d.default,
        aten.upsample_nearest3d.default,
        aten._upsample_nearest_exact1d.default,
        aten._upsample_nearest_exact2d.default,
        aten._upsample_nearest_exact3d.default,
        aten._upsample_bilinear2d_aa.default,
        aten._upsample_bicubic2d_aa.default,
        aten._upsample_lanczos2d_aa.default,
        aten.upsample_bicubic2d.default,
        aten.upsample_bilinear2d.default,
        aten.upsample_linear1d.default,
        aten.upsample_trilinear3d.default,
        # Backward ops
        aten.upsample_nearest1d_backward.default,
        aten.upsample_nearest2d_backward.default,
        aten.upsample_nearest3d_backward.default,
        aten._upsample_nearest_exact1d_backward.default,
        aten._upsample_nearest_exact2d_backward.default,
        aten._upsample_nearest_exact3d_backward.default,
        aten._upsample_bilinear2d_aa_backward.default,
        aten._upsample_bicubic2d_aa_backward.default,
        aten._upsample_lanczos2d_aa_backward.default,
        aten.upsample_bicubic2d_backward.default,
        aten.upsample_bilinear2d_backward.default,
        aten.upsample_linear1d_backward.default,
        aten.upsample_trilinear3d_backward.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def interp_upsample_1out_1in_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # 1 output + 1 input = 2 placements; shard on batch (0) and channel (1)
    # Upsample is a linear transformation so Partial(sum/avg) is valid.
    return [
        [_ShardingPlaceholder(0)] * 2,
        [_ShardingPlaceholder(1)] * 2,
        [Partial("sum"), Partial("sum")],
        [Partial("avg"), Partial("avg")],
    ]


@register_single_dim_strategy(
    [
        aten.max_unpool2d.default,
        aten.max_unpool3d.default,
        aten._adaptive_avg_pool2d_backward.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def interp_pool_1out_2in_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # 1 output + 2 inputs = 3 placements; shard on batch (0) and channel (1)
    return [
        [_ShardingPlaceholder(0)] * 3,
        [_ShardingPlaceholder(1)] * 3,
    ]


@register_single_dim_strategy(
    [aten.max_pool2d_with_indices_backward.default],
    schema_info=RuntimeSchemaInfo(1),
)
def pool_backward_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # max_pool2d_with_indices_backward(grad_output, self, ..., indices) -> grad_input
    # 1 output + 3 tensor inputs = 4 placements
    # Order: [output, grad_output, self, indices]
    input_meta = cast(TensorMeta, args_schema[0])
    strategies: list[list[Placement | _ShardingPlaceholder]] = [
        [_ShardingPlaceholder(0)] * 4,
    ]
    if len(input_meta.shape) >= 4:  # batched: (N, C, H, W)
        strategies.append([_ShardingPlaceholder(1)] * 4)
    # The backward is linear in grad_output, so P(sum/avg) pass through.
    # indices must be replicated (integer positions, not reducible).
    # self is only used for shape, so replicate it too.
    r = Replicate()
    for reduce_op in ("sum", "avg"):
        p = Partial(reduce_op)
        strategies.append([p, p, r, r])
    return strategies


@register_single_dim_strategy(
    [aten.grid_sampler_2d.default, aten.grid_sampler_3d.default],
    schema_info=RuntimeSchemaInfo(1),
)
def grid_sampler_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # grid_sampler_{2,3}d(input[N,C,...], grid[N,...,{2,3}]) -> output[N,C,...]
    # grid has no channel dim, so only batch sharding applies to both inputs.
    # Linear in input: P(sum/avg) on input with replicated grid is valid.
    return [
        [_ShardingPlaceholder(0)] * 3,
        [Partial("sum"), Partial("sum"), Replicate()],
        [Partial("avg"), Partial("avg"), Replicate()],
    ]


@register_single_dim_strategy(
    [aten.grid_sampler_2d_backward.default, aten.grid_sampler_3d_backward.default],
    schema_info=RuntimeSchemaInfo(1),
)
def grid_sampler_backward_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    # grid_sampler_{2,3}d_backward: 2 outputs (grad_input, grad_grid) + 3 inputs = 5 placements, batch-only
    # The native implementation only honors output_mask[0]; grad_grid is always computed.
    output_mask = args_schema[6]
    return [
        [
            _ShardingPlaceholder(0) if output_mask[0] else None,
            _ShardingPlaceholder(0),
            _ShardingPlaceholder(0),  # grad_output
            _ShardingPlaceholder(0),  # input
            _ShardingPlaceholder(0),  # grid
        ]
    ]


def _adjust_group_norm_scalars(
    input_specs: list[DTensorSpec], schema: OpSchema
) -> OpSchema:
    """Adjust N, C, HxW scalar args in native_group_norm to local values.

    native_group_norm(input, weight?, bias?, N, C, HxW, group, eps)
    The scalar args are derived from the global input shape by the Python frontend.
    When the input is sharded, we recompute them from the local input shape.
    """
    input_spec = input_specs[0]
    if input_spec.tensor_meta is None:
        raise AssertionError("input_spec must have tensor_meta")
    local_shape, _ = compute_local_shape_and_global_offset(
        input_spec.tensor_meta.shape,
        input_spec.mesh,
        input_spec.placements,
        skip_offset=True,
    )
    # N = local_shape[0], C = local_shape[1], HxW = product of remaining dims
    n_local = local_shape[0]
    c_local = local_shape[1]
    hxw_local = 1
    for d in local_shape[2:]:
        hxw_local *= d
    args = list(schema.args_schema)
    # Find scalar arg positions: tensor slots (DTensorSpec or None for optionals)
    # precede N, C, HxW. Count both to find the offset.
    num_tensor_args = sum(isinstance(a, (DTensorSpec, type(None))) for a in args)
    args[num_tensor_args] = n_local
    args[num_tensor_args + 1] = c_local
    args[num_tensor_args + 2] = hxw_local
    return OpSchema(schema.op, tuple(args), schema.kwargs_schema)


# ---------------------------------------------------------------------------
# Normalization ops
#
# Batch norm reduces over batch (dim 0) + spatial dims (2+), keeping only
# channel (dim 1).  Channel-dim sharding is valid for both forward and
# backward: each shard processes independent channels.
#
# Group norm reduces over (C/groups, spatial) within each group per sample.
# Batch dim (0) is safe to shard — each sample is independent.
# ---------------------------------------------------------------------------

BATCH_NORM_3OUT_OPS = [
    aten.native_batch_norm.default,
    aten._native_batch_norm_legit.default,
    aten._native_batch_norm_legit.no_stats,
    aten._native_batch_norm_legit_no_training.default,
]

BATCH_NORM_4OUT_OPS = [
    aten._batch_norm_with_update.default,
]


@register_single_dim_strategy(
    BATCH_NORM_3OUT_OPS + BATCH_NORM_4OUT_OPS,
    schema_info=RuntimeSchemaInfo(1),
)
def batch_norm_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # Batch norm normalizes per-channel (reduces over batch + spatial dims),
    # so channel-dim sharding is valid: each shard processes independent channels.
    # Unlike group_norm, batch_norm infers shapes from tensors (no scalar N/C/HxW).
    num_outputs = 4 if op in BATCH_NORM_4OUT_OPS else 3
    num_tensor_inputs = sum(isinstance(a, TensorMeta) for a in args_schema)
    # output [N,C,*] shards on dim 1; save_mean, save_invstd [C] shard on dim 0
    rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(1)]
    rule.extend([_ShardingPlaceholder(0)] * 2)  # save_mean, save_invstd
    if num_outputs == 4:
        rule.append(Replicate())  # reserve: opaque cuDNN workspace
    # input [N,C,*] shards on dim 1; weight, bias, running_mean, running_var [C] on dim 0
    rule.append(_ShardingPlaceholder(1))  # input
    rule.extend([_ShardingPlaceholder(0)] * (num_tensor_inputs - 1))
    return [rule]


@register_single_dim_strategy(
    [aten.native_batch_norm_backward.default],
    schema_info=RuntimeSchemaInfo(1),
)
def batch_norm_backward_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder | None]]:
    # Backward reduces over batch + spatial dims, orthogonal to the channel dim,
    # so channel sharding commutes with the op (same principle as forward).
    # Tensors are [N,C,*] -> Shard(1) or [C] -> Shard(0).
    output_mask = args_schema[9]
    num_tensor_inputs = sum(isinstance(a, TensorMeta) for a in args_schema)
    rule: list[Placement | _ShardingPlaceholder | None] = [
        _ShardingPlaceholder(1) if output_mask[0] else None,  # grad_input [N,C,*]
        _ShardingPlaceholder(0) if output_mask[1] else None,  # grad_weight [C]
        _ShardingPlaceholder(0) if output_mask[2] else None,  # grad_bias [C]
        _ShardingPlaceholder(1),  # grad_out [N,C,*]
        _ShardingPlaceholder(1),  # input [N,C,*]
    ]
    rule.extend([_ShardingPlaceholder(0)] * (num_tensor_inputs - 2))
    return [rule]


@register_single_dim_strategy(
    [aten.native_group_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def group_norm_strategy(
    op: torch._ops.OpOverload,
    args_schema: tuple[Any, ...],
    kwargs_schema: dict[str, Any],
) -> list[list[Placement | _ShardingPlaceholder]]:
    # native_group_norm(input, weight?, bias?, N, C, HxW, group, eps) -> (out, mean, rstd)
    # Batch dim (0) is independent. The scalar N/C/HxW args are adjusted to local
    # values by _adjust_group_norm_scalars in the sharding propagation layer.
    num_tensor_inputs = sum(isinstance(a, TensorMeta) for a in args_schema)
    # 3 outputs + input all shard on batch dim
    placements: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(0)] * 4
    # weight and bias (if present) must be Replicate
    placements.extend([Replicate()] * (num_tensor_inputs - 1))
    return [placements]


# Register scalar shape adjuster for group_norm so the sharding propagator
# rewrites the N/C/HxW args to local values when the input is sharded.
from torch.distributed.tensor._api import DTensor


DTensor._op_dispatcher.sharding_propagator.op_to_scalar_shape_adjuster[
    aten.native_group_norm.default
] = _adjust_group_norm_scalars
