# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import itertools
from collections.abc import Callable, Sequence, Sized
from typing import cast

import torch
from torch._ops import OpOverload
from torch._prims_common import IntLike
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    RuntimeSchemaInfo,
    TensorMeta,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import normalize_dim
from torch.distributed.tensor.placement_types import (
    _MaskPartial,
    Partial,
    Placement,
    Replicate,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true


aten = torch.ops.aten
prims = torch.ops.prims

_PASS_THROUGH_PARTIAL_OPS = ("sum", "avg", "max", "min")
_IDEMPOTENT_PARTIAL_OPS = ("avg", "max", "min")


def _same_dim_sharding_strategies(
    ndim: int, num_inputs: int = 1
) -> list[list[Placement | _ShardingPlaceholder]]:
    return [[_ShardingPlaceholder(d)] * (1 + num_inputs) for d in range(ndim)]


def _partial_or_replicate_strategies(
    num_inputs: int, reduce_ops: Sequence[str] = _PASS_THROUGH_PARTIAL_OPS
) -> list[list[Placement | _ShardingPlaceholder]]:
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for reduce_op in reduce_ops:
        partial = Partial(reduce_op)
        for input_placements in itertools.product(
            (Replicate(), partial), repeat=num_inputs
        ):
            if all(p.is_replicate() for p in input_placements):
                continue
            strategies.append([partial, *input_placements])
    return strategies


@register_single_dim_strategy(
    [
        aten.clone.default,
        aten.contiguous.default,
        aten.detach.default,
        aten.detach_.default,
        aten.alias.default,
        aten.view.dtype,
        aten.zero_.default,
        prims.view_of.default,
    ],
    allow_uneven_sharding=True,
)
def propagate_single_input_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    strategies = _same_dim_sharding_strategies(len(input_meta.shape))
    strategies.extend(
        [
            [Partial(reduce_op), Partial(reduce_op)]
            for reduce_op in _PASS_THROUGH_PARTIAL_OPS
        ]
    )
    return strategies


@register_single_dim_strategy(aten.fill_.Scalar, allow_uneven_sharding=True)
def fill_scalar_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    strategies = _same_dim_sharding_strategies(len(input_meta.shape))
    strategies.extend(
        [
            [Partial(reduce_op), Partial(reduce_op)]
            for reduce_op in ("sum", "avg", "max", "min")
        ]
    )
    return strategies


def _partial_needs_reduce_for_dtype_cast(
    reduce_op: str,
    src_dtype: torch.dtype,
    target_dtype: torch.dtype | None,
) -> bool:
    """Return True when reduce_op does not commute with the dtype cast."""
    if target_dtype is None or src_dtype == target_dtype:
        return False
    if target_dtype == torch.bool:
        return True
    if reduce_op in ("max", "min"):
        return False
    return src_dtype.is_floating_point and not target_dtype.is_floating_point


@register_single_dim_strategy(
    aten._to_copy.default,
    schema_info=RuntimeSchemaInfo(static_kwargkey=["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def _to_copy_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    src_dtype = input_meta.dtype
    target_dtype = cast(torch.dtype | None, kwargs_schema.get("dtype", None))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    for reduce_op in Partial.ALL_REDUCE_OPS:
        if not _partial_needs_reduce_for_dtype_cast(reduce_op, src_dtype, target_dtype):
            strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    [
        aten.equal.default,
        aten.is_same_size.default,
    ]
)
def equal_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    self_meta = cast(TensorMeta, args_schema[0])
    other_meta = cast(TensorMeta, args_schema[1])
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(min(len(self_meta.shape), len(other_meta.shape))):
        strategies.append([_ShardingPlaceholder(d), _ShardingPlaceholder(d)])
    return strategies


@register_single_dim_strategy(
    aten.empty_like.default,
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_uneven_sharding=True,
)
def empty_like_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    strategies = _same_dim_sharding_strategies(len(input_meta.shape))
    strategies.extend(
        [
            [Partial(reduce_op), Partial(reduce_op)]
            for reduce_op in _PASS_THROUGH_PARTIAL_OPS
        ]
    )
    return strategies


@register_single_dim_strategy(
    [
        aten.ones_like.default,
        aten.rand_like.default,
        aten.randn_like.default,
        aten.zeros_like.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_uneven_sharding=True,
)
@register_single_dim_strategy(
    [aten.full_like.default],
    schema_info=RuntimeSchemaInfo(2, ["dtype"]),
    allow_uneven_sharding=True,
)
@register_single_dim_strategy(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randint_like.low_dtype_out,
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),
    allow_uneven_sharding=True,
)
def create_like_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    strategies = _same_dim_sharding_strategies(len(input_meta.shape))
    for reduce_op in Partial.ALL_REDUCE_OPS:
        strategies.append([Replicate(), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    [
        aten.new_full.default,
        aten.new_ones.default,
        aten.new_zeros.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_uneven_sharding=True,
)
def new_factory_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    output_shape = args_schema[1]
    if not isinstance(output_shape, list):
        raise AssertionError(f"Expected list, got {type(output_shape)}")

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(len(input_meta.shape)):
        strategies.append([Replicate(), _ShardingPlaceholder(d)])

    if tuple(input_meta.shape) == tuple(output_shape):
        strategies.extend(_same_dim_sharding_strategies(len(input_meta.shape)))

    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Replicate(), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    [
        aten.new_empty.default,
        aten.new_empty_strided.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_uneven_sharding=True,
    include_default_replication=False,
)
def new_empty_factory_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    output_shape = args_schema[1]
    if not isinstance(output_shape, list):
        raise AssertionError(f"Expected list, got {type(output_shape)}")

    strategies: list[list[Placement | _ShardingPlaceholder]] = [
        [Replicate(), Replicate()]
    ]
    for d in range(len(input_meta.shape)):
        strategies.append([Replicate(), _ShardingPlaceholder(d)])
    for reduce_op in Partial.ALL_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])

    if tuple(input_meta.shape) == tuple(output_shape):
        strategies.extend(_same_dim_sharding_strategies(len(input_meta.shape)))

    return strategies


@register_single_dim_strategy(aten.bucketize.Tensor)
def bucketize_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Bucketize returns indices into a sorted boundary tensor.

    Three families of strategies:
    1. Shard the input (and output) on any dim, keep boundaries replicated.
    2. Shard boundaries on dim 0, replicate input, output is Partial("sum").
       Each rank counts how many of its local boundary values each input
       element exceeds; summing across ranks gives the correct global index.
    3. Partial("max") or Partial("min") input with replicated boundaries.
       Bucketize is monotonically non-decreasing in its input, so reducing
       local bucket indices with max (or min) across ranks gives the same
       result as bucketizing the reduced input values.
    """
    input_meta, _boundaries_meta = args_schema
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        strategies.append(
            [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim), Replicate()]
        )
    strategies.append([Partial("sum"), Replicate(), _ShardingPlaceholder(0)])
    for reduce_op in ("max", "min"):
        strategies.append([Partial(reduce_op), Partial(reduce_op), Replicate()])
    return strategies


@register_single_dim_strategy(aten.select.int, schema_info=RuntimeSchemaInfo(1))
def select_int_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    selected_dim = normalize_dim(cast(int, args_schema[1]), len(input_meta.shape))
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(len(input_meta.shape)):
        if d == selected_dim:
            continue
        out_dim = d if d < selected_dim else d - 1
        strategies.append([_ShardingPlaceholder(out_dim), _ShardingPlaceholder(d)])
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.select_backward.default,
    schema_info=RuntimeSchemaInfo(1),
)
def select_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    grad_output_meta = cast(TensorMeta, args_schema[0])
    input_sizes = cast(list[int], args_schema[1])
    dim = normalize_dim(cast(int, args_schema[2]), len(input_sizes))
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(len(grad_output_meta.shape)):
        out_dim = d if d < dim else d + 1
        strategies.append([_ShardingPlaceholder(out_dim), _ShardingPlaceholder(d)])
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def slice_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    defaults = (None, 0, None, None, 1)
    input_meta, dim, start, end, step = args_schema + defaults[len(args_schema) :]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    input_shape = input_meta.shape
    input_ndim = len(input_shape)
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    if start is None:
        start = 0
    if end is None:
        end = input_shape[dim]
    if not isinstance(start, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(start)}")
    if not isinstance(end, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(end)}")
    if statically_known_true(end > input_shape[dim]):
        end = input_shape[dim]
    if not isinstance(step, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(step)}")

    # normalize args
    slice_dim = normalize_dim(dim, input_ndim)  # type: ignore[arg-type]
    start = normalize_dim(start, input_shape[dim])  # type: ignore[arg-type]
    end = normalize_dim(end, input_shape[dim])  # type: ignore[arg-type]

    statically_redundant_slice = (
        statically_known_true(start == 0)
        and statically_known_true(end == input_shape[dim])
        and statically_known_true(step == 1)
    )

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(input_ndim):
        if d != slice_dim or statically_redundant_slice:
            strategies.append([_ShardingPlaceholder(d), _ShardingPlaceholder(d)])
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice_backward.default,
    schema_info=RuntimeSchemaInfo(1),
)
def slice_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    grad_output_meta = cast(TensorMeta, args_schema[0])
    input_sizes = cast(list[int], args_schema[1])
    dim = normalize_dim(cast(int, args_schema[2]), len(input_sizes))
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(len(grad_output_meta.shape)):
        if d != dim:
            strategies.append([_ShardingPlaceholder(d), _ShardingPlaceholder(d)])
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice_scatter.default,
    schema_info=RuntimeSchemaInfo(2),
    allow_uneven_sharding=True,
    allow_replicate_to_partial_redistribution=False,
)
def slice_scatter_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    src_meta = cast(TensorMeta, args_schema[1])
    input_shape = input_meta.shape
    src_shape = src_meta.shape
    slice_dim = cast(int, args_schema[2]) if len(args_schema) > 2 else 0
    slice_dim = normalize_dim(slice_dim, len(input_shape))
    start = args_schema[3] if len(args_schema) > 3 else None
    end = args_schema[4] if len(args_schema) > 4 else None
    step = args_schema[5] if len(args_schema) > 5 else 1
    if start is None:
        start = 0
    if end is None:
        end = input_shape[slice_dim]
    if not isinstance(start, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(start)}")
    if not isinstance(end, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(end)}")
    if not isinstance(step, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(step)}")
    if statically_known_true(end > input_shape[slice_dim]):
        end = input_shape[slice_dim]
    start = normalize_dim(start, input_shape[slice_dim])  # type: ignore[arg-type]
    end = normalize_dim(end, input_shape[slice_dim])  # type: ignore[arg-type]
    same_shape = len(input_shape) == len(src_shape) and all(
        statically_known_true(i == s) for i, s in zip(input_shape, src_shape)
    )
    statically_full_slice = (
        same_shape
        and statically_known_true(start == 0)
        and statically_known_true(end == input_shape[slice_dim])
        and statically_known_true(step == 1)
    )

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if len(input_shape) == len(src_shape):
        for d in range(len(input_shape)):
            if (d != slice_dim or statically_full_slice) and statically_known_true(
                input_shape[d] == src_shape[d]
            ):
                strategies.append(
                    [
                        _ShardingPlaceholder(d),
                        _ShardingPlaceholder(d),
                        _ShardingPlaceholder(d),
                    ]
                )
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op), Partial(reduce_op)])
    for reduce_op in _IDEMPOTENT_PARTIAL_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op), Replicate()])
        strategies.append([Partial(reduce_op), Replicate(), Partial(reduce_op)])
    if statically_full_slice:
        for src_reduce_op in _PASS_THROUGH_PARTIAL_OPS:
            strategies.append([Replicate(), Partial(src_reduce_op), Replicate()])
            strategies.append(
                [Partial(src_reduce_op), Replicate(), Partial(src_reduce_op)]
            )
            for self_reduce_op in _PASS_THROUGH_PARTIAL_OPS:
                strategies.append(
                    [
                        Partial(src_reduce_op),
                        Partial(self_reduce_op),
                        Partial(src_reduce_op),
                    ]
                )
        for out_reduce_op in _IDEMPOTENT_PARTIAL_OPS:
            strategies.append([Partial(out_reduce_op), Replicate(), Replicate()])
            for self_reduce_op in _PASS_THROUGH_PARTIAL_OPS:
                strategies.append(
                    [
                        Partial(out_reduce_op),
                        Partial(self_reduce_op),
                        Replicate(),
                    ]
                )
    return strategies


@register_single_dim_strategy(
    [aten.select_scatter.default],
    schema_info=RuntimeSchemaInfo(1),
)
def select_scatter_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    dim = normalize_dim(cast(int, args_schema[2]), ndim)
    # [output, self, src] — src has the select dim removed
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d == dim:
            continue
        strategies.append(
            [
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d if d < dim else d - 1),
            ]
        )
    return strategies


@register_single_dim_strategy(
    [aten.diagonal_scatter.default],
    schema_info=RuntimeSchemaInfo(1),
)
def diagonal_scatter_single_dim_strategy(
    op: OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    # schema: (self, src, offset=0, dim1=0, dim2=1)
    dim1 = cast(int, args_schema[3]) if len(args_schema) > 3 else 0
    dim2 = cast(int, args_schema[4]) if len(args_schema) > 4 else 1
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    min_d, max_d = min(dim1, dim2), max(dim1, dim2)
    # [output, self, src] — src has dim1/dim2 removed and diagonal appended
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d in (dim1, dim2):
            continue
        removed = (1 if d > min_d else 0) + (1 if d > max_d else 0)
        strategies.append(
            [
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d),
                _ShardingPlaceholder(d - removed),
            ]
        )
    return strategies


@register_single_dim_strategy(aten._local_scalar_dense.default)
def local_scalar_dense_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    return []


@register_single_dim_strategy(
    [
        aten.scatter_.value,
        aten.scatter.value,
        aten.scatter_.src,
        aten.scatter.src,
    ],
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def scatter_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    dim = normalize_dim(cast(int, args_schema[1]), len(input_meta.shape))
    index_meta = cast(TensorMeta, args_schema[2])
    src_meta = args_schema[3] if len(args_schema) > 3 else None
    num_specs = 4 if isinstance(src_meta, TensorMeta) else 3
    src_shape = src_meta.shape if isinstance(src_meta, TensorMeta) else None

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if len(input_meta.shape) == len(index_meta.shape):
        for d in range(len(input_meta.shape)):
            if d == dim or input_meta.shape[d] != index_meta.shape[d]:
                continue
            if src_shape is not None and src_shape[d] != index_meta.shape[d]:
                continue
            strategies.append([_ShardingPlaceholder(d)] * num_specs)
    if num_specs == 4:
        for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
            strategies.append(
                [
                    Partial(reduce_op),
                    Partial(reduce_op),
                    Replicate(),
                    Partial(reduce_op),
                ]
            )
        for reduce_op in _IDEMPOTENT_PARTIAL_OPS:
            strategies.append(
                [Partial(reduce_op), Partial(reduce_op), Replicate(), Replicate()]
            )
            strategies.append(
                [Partial(reduce_op), Replicate(), Replicate(), Partial(reduce_op)]
            )
    return strategies


@register_single_dim_strategy(
    aten.scatter_add.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def scatter_add_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    dim = normalize_dim(cast(int, args_schema[1]), len(input_meta.shape))
    index_meta = cast(TensorMeta, args_schema[2])
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if len(input_meta.shape) == len(index_meta.shape):
        for d in range(len(input_meta.shape)):
            if d != dim and input_meta.shape[d] == index_meta.shape[d]:
                strategies.append([_ShardingPlaceholder(d)] * 4)
    for reduce_op in ("sum", "avg"):
        strategies.append(
            [Partial(reduce_op), Partial(reduce_op), Replicate(), Partial(reduce_op)]
        )
    for reduce_op in _IDEMPOTENT_PARTIAL_OPS:
        strategies.append(
            [Partial(reduce_op), Partial(reduce_op), Replicate(), Replicate()]
        )
        strategies.append(
            [Partial(reduce_op), Replicate(), Replicate(), Partial(reduce_op)]
        )
    strategies.append(
        [
            Partial("sum"),
            Partial("sum"),
            _ShardingPlaceholder(dim),
            _ShardingPlaceholder(dim),
        ]
    )
    return strategies


@register_single_dim_strategy(
    aten.gather.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_uneven_sharding=True,
)
def gather_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    dim = normalize_dim(cast(int, args_schema[1]), len(input_meta.shape))
    index_meta = cast(TensorMeta, args_schema[2])
    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    if dim < len(index_meta.shape) and index_meta.shape[dim] == 1:
        mask_partial = _MaskPartial(offset_shape=input_meta.shape, offset_dim=dim)
        strategies.append(
            [
                mask_partial,
                _ShardingPlaceholder(dim),
                mask_partial,
            ]
        )

    strategies.append(
        [_ShardingPlaceholder(dim), Replicate(), _ShardingPlaceholder(dim)]
    )

    if len(input_meta.shape) == len(index_meta.shape):
        for d in range(len(input_meta.shape)):
            if d != dim:
                strategies.append([_ShardingPlaceholder(d)] * 3)
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op), Replicate()])
    return strategies


@register_single_dim_strategy(
    aten.stack.default,
    RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
)
def stack_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_list = args_schema[0]
    if not isinstance(input_list, (tuple, list)):
        raise AssertionError(type(input_list))
    input_list = tuple(cast(Sequence[TensorMeta], input_list))
    common_input_ndim = len(input_list[0].shape)
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    dim = normalize_dim(dim, common_input_ndim + 1)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(common_input_ndim):
        out_dim = d if d < dim else d + 1
        strategy: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(out_dim)
        ]
        strategy.extend([_ShardingPlaceholder(d)] * len(input_list))
        strategies.append(strategy)
    strategies.extend(_partial_or_replicate_strategies(len(input_list)))
    return strategies


@register_single_dim_strategy(
    aten.cat.default,
    RuntimeSchemaInfo(1, needs_pytree=True),
    allow_uneven_sharding=True,
)
def cat_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_list = args_schema[0]
    # unfortunate naming, but yes it's a TensorList input, and we represent it as a tuple of TensorMeta
    if not isinstance(input_list, (tuple, list)):
        raise AssertionError(type(input_list))
    if not all(isinstance(tm, TensorMeta) for tm in input_list):
        raise AssertionError

    if isinstance(input_list, list):
        input_list = tuple(input_list)

    num_inputs = len(input_list)
    ndim_set = {len(meta.shape) for meta in input_list}
    if len(ndim_set) not in (1, 2):
        raise AssertionError(
            "Expected all cat inputs to be the same ndim, except empty tensors"
        )
    if len(ndim_set) == 2:
        if 0 not in ndim_set:
            raise AssertionError
    common_ndim = max(ndim_set)
    cat_dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    cat_dim = normalize_dim(cat_dim, common_ndim)
    single_dim_strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for i in range(common_ndim):
        if i != cat_dim:
            single_dim_strategies.append([_ShardingPlaceholder(i)] * (1 + num_inputs))
    single_dim_strategies.extend(_partial_or_replicate_strategies(num_inputs))
    # pyrefly: ignore [bad-return]
    return single_dim_strategies


@register_single_dim_strategy(
    aten.index_select.default, schema_info=RuntimeSchemaInfo(1)
)
def index_select_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    values_meta, dim, index_meta = args_schema
    if not isinstance(values_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(values_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    dim = normalize_dim(dim, len(values_meta.shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # Shard values on any non-indexed dim (output has same ndim)
    for d in range(len(values_meta.shape)):
        if d == dim:
            continue
        strategies.append(
            [_ShardingPlaceholder(d), _ShardingPlaceholder(d), Replicate()]
        )

    # Shard index → output sharded on the indexed dim
    strategies.append([_ShardingPlaceholder(dim), Replicate(), _ShardingPlaceholder(0)])

    # Partial passthrough from values
    for reduce_op in Partial.ALL_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op), Replicate()])

    return strategies


@register_single_dim_strategy(
    aten.index.Tensor, schema_info=RuntimeSchemaInfo(needs_pytree=True)
)
def index_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    values_meta, multi_indices_meta = args_schema
    if not isinstance(values_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(values_meta)}")
    if not isinstance(multi_indices_meta, (list, tuple)):
        raise AssertionError(f"Expected list or tuple, got {type(multi_indices_meta)}")

    indexed_dims = [i for i, idx in enumerate(multi_indices_meta) if idx is not None]
    non_indexed_dims = [
        i for i in range(len(values_meta.shape)) if i not in set(indexed_dims)
    ]

    index_metas = [idx for idx in multi_indices_meta if idx is not None]
    if not all(isinstance(m, TensorMeta) for m in index_metas):
        raise AssertionError("Expected all index metas to be TensorMeta")
    broadcast_ndim = max(len(m.shape) for m in index_metas)
    num_indices = len(indexed_dims)

    # Determine where index output dims are inserted in the result
    all_consecutive = all(
        indexed_dims[i + 1] - indexed_dims[i] == 1 for i in range(len(indexed_dims) - 1)
    )
    insert_dim = indexed_dims[0] if all_consecutive else 0

    def values_dim_to_output_dim(d: int) -> int:
        if d < insert_dim:
            return d
        return d + broadcast_ndim - sum(1 for idx_dim in indexed_dims if d > idx_dim)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []

    # Shard values on a non-indexed dim, all indices replicated
    for d in non_indexed_dims:
        out_dim = values_dim_to_output_dim(d)
        rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(out_dim)]
        rule.append(_ShardingPlaceholder(d))
        rule.extend([Replicate()] * num_indices)
        strategies.append(rule)

    # Shard indices on the same broadcast dim.  Each index tensor may
    # have a different ndim, so we map broadcast dim → tensor dim via
    # left-padding.  Tensors with size 1 on that dim are replicated
    # (broadcast semantics).
    for bd in range(broadcast_ndim):
        per_tensor: list[tuple[int, int]] = []  # (tensor_dim, size)
        for m in index_metas:
            offset = broadcast_ndim - len(m.shape)
            if bd < offset:
                per_tensor.append((-1, 1))  # implicit broadcast
            else:
                td = bd - offset
                per_tensor.append((td, m.shape[td]))
        if all(s == 1 for _, s in per_tensor):
            continue  # all broadcast-only, skip
        out_dim = bd + insert_dim
        rule: list[Placement | _ShardingPlaceholder] = [_ShardingPlaceholder(out_dim)]
        rule.append(Replicate())
        for td, s in per_tensor:
            if s > 1:
                rule.append(_ShardingPlaceholder(td))
            else:
                rule.append(Replicate())
        strategies.append(rule)

    # Partial passthrough from values
    for reduce_op in Partial.LINEAR_REDUCE_OPS:
        rule: list[Placement | _ShardingPlaceholder] = [
            Partial(reduce_op),
            Partial(reduce_op),
        ]
        rule.extend([Replicate()] * num_indices)
        strategies.append(rule)

    return strategies


@register_single_dim_strategy(
    [aten.index_put.default, aten.index_put_.default, aten._index_put_impl_.default],
    schema_info=RuntimeSchemaInfo(needs_pytree=True),
)
def index_put_single_dim_strategy(
    op: OpOverload, args: ArgsType, kwargs: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim sharding strategy for index_put(self, indices, values).

    Strategy format: [output, input, *indices, value]

    How index_put works:

      indices is a tuple of index tensors and Nones:
      - an index tensor at entry i means self is indexed on dim i.
      - a None at entry i means all elements along dim i are selected (like :).
      - any trailing dims (if self.ndim > len(indices)) are also not indexed
        (i.e. implicit trailing Nones).

      All non-None index tensors are broadcast together to produce a
      broadcasted indexing shape. Each position in this broadcasted shape
      serves as an indexing coordinate into self. Each coordinate selects a
      tensor element, or a slice (if non-indexed dims exist).

      values is a tensor broadcastable to the indexing output shape.
      When indexed dims are consecutive starting at dim k, this shape is
      (*self[:k], *broadcast_shape, *self[k+n_indexed:]). When indexed
      dims are non-consecutive, it is (*broadcast_shape, *non_indexed_dims).

    Sharding rules (possibly conservative and incomplete):
      - Index tensors: always Replicate (every rank needs all coordinates).
      - Self cannot be sharded on indexed dims (local position != global position).
      - Self and values CAN be sharded on non-indexed dims.
        The exception is broadcasted value dimensions (size 1) - we require Replicate, but can shard self.
      - Additionally, we allow the full Partial rule on non-indexing tensors.

    """
    self_meta = cast(TensorMeta, args[0])
    indices_meta = cast(tuple[TensorMeta | None, ...], args[1])
    values_meta = cast(TensorMeta, args[2])

    # Determine indexed vs non-indexed dims of self.
    indexed_dims = {i for i, idx in enumerate(indices_meta) if idx is not None}
    non_indexed_dims = [d for d in range(len(self_meta.shape)) if d not in indexed_dims]
    n_indexed = len(indexed_dims)
    values_ndim = len(values_meta.shape)

    # Explicitly compute the broadcast shape of the index tensors.
    index_shapes = [idx.shape for idx in indices_meta if idx is not None]
    broadcast_ndim = len(torch.broadcast_shapes(*index_shapes)) if index_shapes else 0

    # Strategy format: [output, input, *indices, value]
    # The infra flattens the indices list and drops None entries, so only
    # non-None index tensors get a placement slot (all Replicate).
    #
    # Values dim mapping depends on whether indexed dims are contiguous:
    #   Contiguous (e.g., (None, idx0, idx1)): broadcast replaces indexed block in-place.
    #     values shape = (*non_indexed_before, *broadcast_shape, *non_indexed_after)
    #   Non-contiguous (e.g., (idx0, None, idx1)): broadcast goes to front.
    #     values shape = (*broadcast_shape, *non_indexed_dim_sizes)
    indexed_dims_sorted = sorted(indexed_dims)
    contiguous_indexed = len(indexed_dims_sorted) <= 1 or (
        indexed_dims_sorted[-1] - indexed_dims_sorted[0] + 1 == len(indexed_dims_sorted)
    )

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for i, self_dim in enumerate(non_indexed_dims):
        if contiguous_indexed and indexed_dims_sorted:
            # Broadcast replaces the indexed block in-place.
            first_indexed = indexed_dims_sorted[0]
            if self_dim < first_indexed:
                values_dim = self_dim
            else:
                values_dim = self_dim - n_indexed + broadcast_ndim
        else:
            # Broadcast goes to front (non-contiguous or no indexed dims).
            values_dim = broadcast_ndim + i

        # values_dim is the position in the result tensor, but values may
        # have fewer dims (right-aligned broadcasting). Convert to the
        # actual values tensor dimension.
        result_ndim = broadcast_ndim + len(non_indexed_dims)
        values_tensor_dim = values_dim - (result_ndim - values_ndim)

        if values_tensor_dim < 0:
            values_placement: Placement | _ShardingPlaceholder = Replicate()
        elif values_meta.shape[values_tensor_dim] == 1:
            values_placement = Replicate()
        else:
            values_placement = _ShardingPlaceholder(values_tensor_dim)

        strategies.append(
            [
                _ShardingPlaceholder(self_dim),
                _ShardingPlaceholder(self_dim),
                *([Replicate()] * n_indexed),
                values_placement,
            ]
        )

    # full-partial rule on non-indexing tensors
    strategies.append(
        [
            Partial(),
            Partial(),
            *([Replicate()] * n_indexed),
            Partial(),
        ]
    )
    return strategies


def _index_dim_strategy(
    args_schema: ArgsType,
    shard_row: Callable[[int], list[Placement | _ShardingPlaceholder]],
    partial_rules: list[list[Placement | _ShardingPlaceholder]] | None = None,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Common strategy for index ops that shard on all dims except the indexed dim.

    Args:
        shard_row: given a dim d, returns the strategy row for sharding on that dim.
        partial_rules: additional Partial passthrough strategies.
    """
    self_meta = cast(TensorMeta, args_schema[0])
    ndim = len(self_meta.shape)
    dim = normalize_dim(cast(int, args_schema[1]), ndim)
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d != dim:
            strategies.append(shard_row(d))
    if partial_rules:
        strategies.extend(partial_rules)
    return strategies


@register_single_dim_strategy(
    [aten.index_fill.int_Scalar, aten.index_fill_.int_Scalar],
    schema_info=RuntimeSchemaInfo(1),
)
def index_fill_scalar_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # index_fill(self, dim, index, value) — fills self[..., index, ...] with scalar value.
    # Partial rules: each rank fills with the same scalar v, then reduces.
    # Only idempotent reduces work: avg(v,v,...,v)=v, max(v,v,...,v)=v, min(v,v,...,v)=v.
    # sum and product fail: sum(v,v,...,v)=nv, product(v,v,...,v)=v^n.
    return _index_dim_strategy(
        args_schema,
        lambda d: [
            _ShardingPlaceholder(d),  # result
            _ShardingPlaceholder(d),  # self
            Replicate(),  # value (scalar, same on all ranks)
        ],
        [[Partial(op), Partial(op), Replicate()] for op in ("avg", "max", "min")],
    )


@register_single_dim_strategy(
    [aten.index_fill.int_Tensor, aten.index_fill_.int_Tensor],
    schema_info=RuntimeSchemaInfo(1),
)
def index_fill_tensor_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # index_fill(self, dim, index, value) — fills self[..., index, ...] with 0-d tensor value.
    # Partial rules: each rank fills with its partial value v_i, then reduces.
    # All reduce ops work because reduce(v_0, ..., v_{n-1}) = V (the global value)
    # regardless of op, since fill is a pure replacement (no mixing with self).
    return _index_dim_strategy(
        args_schema,
        lambda d: [
            _ShardingPlaceholder(d),  # result
            _ShardingPlaceholder(d),  # self
            Replicate(),  # index
            Replicate(),  # value
        ],
        [
            [Partial(op), Partial(op), Replicate(), Partial(op)]
            for op in Partial.ALL_REDUCE_OPS
        ],
    )


@register_single_dim_strategy(
    [aten.index_reduce.default, aten.index_reduce_.default],
    schema_info=RuntimeSchemaInfo(1),
)
def index_reduce_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # index_reduce(self, dim, index, source, reduce) — reduces source into self at index positions.
    # No partial rules: reduce ops are "mean"/"amax"/"amin"/"prod", which don't match
    # any Partial reduce op names ("avg"/"max"/"min"/"product"/"sum").
    return _index_dim_strategy(
        args_schema,
        lambda d: [
            _ShardingPlaceholder(d),  # result
            _ShardingPlaceholder(d),  # self
            Replicate(),  # index
            _ShardingPlaceholder(d),  # source
        ],
    )


@register_single_dim_strategy(
    [
        aten.split.Tensor,
        aten.split_with_sizes.default,
        aten.split_with_sizes_copy.default,
    ],
    RuntimeSchemaInfo(1),
)
def split_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    split_size_or_sections = args_schema[1]
    input_ndim = len(input_meta.shape)
    split_dim = cast(int, args_schema[2]) if len(args_schema) > 2 else 0
    dim = normalize_dim(split_dim, input_ndim)

    def size_split(N, i) -> list:
        # Last chunk will be smaller if the tensor size N
        # along the given dimension dim is not divisible by i.
        if not i > 0:
            raise AssertionError(f"Split size must be positive, got {i}")
        return [i] * (N // i) + ([N % i] if N % i != 0 else [])

    output_size_list = (
        size_split(input_meta.shape[dim], split_size_or_sections)
        if isinstance(split_size_or_sections, IntLike)
        else split_size_or_sections
    )
    if not isinstance(output_size_list, Sized):
        raise AssertionError(f"Expected Sized, got {type(output_size_list)}")

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(input_ndim):
        if d != dim:
            strategy: list[Placement | _ShardingPlaceholder] = [
                _ShardingPlaceholder(d)
            ] * len(output_size_list)
            strategy.append(_ShardingPlaceholder(d))
            strategies.append(strategy)
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op)] * (len(output_size_list) + 1))
    return strategies


# TODO: fix remaining failures in xfail("unbind") in test_dtensor_ops.py
#       and remove this xfail item
@register_single_dim_strategy(
    aten.unbind.int,
    schema_info=RuntimeSchemaInfo(1),
    include_default_replication=False,
    allow_input_redistribution=False,
)
def unbind_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = cast(TensorMeta, args_schema[0])
    input_ndim = len(input_meta.shape)
    input_shape = input_meta.shape
    unbind_dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    unbind_dim = normalize_dim(unbind_dim, input_ndim)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(input_ndim):
        if d == unbind_dim:
            continue
        out_dim = d if d < unbind_dim else d - 1
        strategy: list[Placement | _ShardingPlaceholder] = [
            _ShardingPlaceholder(out_dim)
        ] * input_shape[unbind_dim]
        strategy.append(_ShardingPlaceholder(d))
        strategies.append(strategy)
    for reduce_op in _PASS_THROUGH_PARTIAL_OPS:
        strategies.append([Partial(reduce_op)] * (input_shape[unbind_dim] + 1))
    return strategies


@register_single_dim_strategy(aten.eye.m_out)
def eye_out_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    out_meta = cast(TensorMeta, kwargs_schema["out"])
    return _same_dim_sharding_strategies(len(out_meta.shape))


def _pass_through_partials(
    num_inputs: int = 1,
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Pass-through strategies for all supported reduce ops."""
    return [[Partial(op)] * (1 + num_inputs) for op in ("sum", "avg", "max", "min")]


def _shard_inactive_dims(
    ndim: int, active_dims: set[int], num_inputs: int = 1
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Single-dim strategies: shard on dims the op doesn't touch."""
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(ndim):
        if d not in active_dims:
            strategies.append([_ShardingPlaceholder(d)] * (1 + num_inputs))
    return strategies


@register_single_dim_strategy(aten.roll.default, schema_info=RuntimeSchemaInfo(1))
def roll_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    raw_dims = cast(list[int], args_schema[2]) if len(args_schema) > 2 else []
    # When dims is empty, roll flattens the tensor — all dims are active
    if not raw_dims:
        raw_dims = list(range(ndim))
    active_dims = {normalize_dim(d, ndim) for d in raw_dims}
    return _shard_inactive_dims(ndim, active_dims) + _pass_through_partials()


@register_single_dim_strategy(aten.flip.default, schema_info=RuntimeSchemaInfo(1))
def flip_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    raw_dims = cast(list[int], args_schema[1])
    active_dims = {normalize_dim(d, ndim) for d in raw_dims}
    return _shard_inactive_dims(ndim, active_dims) + _pass_through_partials()


@register_single_dim_strategy(
    [aten._fft_c2c.default, aten._fft_r2c.default, aten._fft_c2r.default],
    schema_info=RuntimeSchemaInfo(1),
)
def fft_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    input_meta = args_schema[0]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    ndim = len(input_meta.shape)
    raw_dims = cast(list[int], args_schema[1])
    active_dims = {normalize_dim(d, ndim) for d in raw_dims}
    return _shard_inactive_dims(ndim, active_dims) + _pass_through_partials()
