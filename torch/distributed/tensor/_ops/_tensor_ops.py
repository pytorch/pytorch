# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Callable, Sequence, Sized
from typing import cast

import torch
from torch._ops import OpOverload
from torch._prims_common import IntLike
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    ArgsType,
    KwargsType,
    OpSchema,
    OpSpec,
    OpStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TensorMeta,
    TupleStrategy,
)
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_dim_sharded,
    normalize_dim,
    register_op_strategy,
    shift_shard_dims_after_insert,
    shift_shard_dims_after_remove,
)
from torch.distributed.tensor.placement_types import (
    _is_shard_like,
    _MaskPartial,
    Partial,
    Placement,
    Replicate,
    Shard,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true


aten = torch.ops.aten
prims = torch.ops.prims

_PARTIAL_PASS_THROUGH_REDUCE_OPS = ("sum", "avg", "min", "max")


@register_single_dim_strategy(
    [
        aten.clone.default,
        aten.contiguous.default,
        aten.detach.default,
        aten.detach_.default,
        aten.alias.default,
        aten.fill_.Scalar,
        aten.view.dtype,
        aten.zero_.default,
        prims.view_of.default,
    ],
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def propagate_single_input_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Propagate placements for ops with one tensor input.

    This is the single-dim form of the old one-to-one OpStrategy: a sharded
    input dim produces the same sharded output dim. Supported Partial
    placements are also forwarded with the same pending reduction.
    """
    input_meta = cast(TensorMeta, args_schema[0])
    strategies: list[list[Placement | _ShardingPlaceholder]] = [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(len(input_meta.shape))
    ]
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
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
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        if not _partial_needs_reduce_for_dtype_cast(reduce_op, src_dtype, target_dtype):
            strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    [
        aten.equal.default,
        aten.is_same_size.default,
    ],
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def equal_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Require both tensor operands to use the same non-partial layout.

    equal/is_same_size compare corresponding local shards, so mismatched
    shardings would compare different global elements. Partial inputs must use
    the implicit Replicate fallback because comparing unreduced partial values
    would be invalid. If either operand is scalar, no sharded rows are
    produced and the implicit Replicate rule is used.
    """
    self_meta = cast(TensorMeta, args_schema[0])
    other_meta = cast(TensorMeta, args_schema[1])
    return [
        [_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)]
        for dim in range(min(len(self_meta.shape), len(other_meta.shape)))
    ]


register_single_dim_strategy(
    aten.empty_like.default,
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)(propagate_single_input_single_dim_strategy)


@register_single_dim_strategy(
    [
        aten.ones_like.default,
        aten.rand_like.default,
        aten.randn_like.default,
        aten.zeros_like.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
@register_single_dim_strategy(
    [aten.full_like.default],
    schema_info=RuntimeSchemaInfo(2, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
@register_single_dim_strategy(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randint_like.low_dtype_out,
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def create_like_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # These factory-like ops create tensors with the input shape but with
    # content independent of the input. Sharding can follow the input, but a
    # Partial input maps to a Replicate output so generated values are not
    # reduced as if they were accumulated partials.
    input_meta = cast(TensorMeta, args_schema[0])
    kw_names = [k for k, v in kwargs_schema.items() if isinstance(v, TensorMeta)]
    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        placement = _ShardingPlaceholder(dim)
        strategies.append(
            [placement, placement]
            + [placement if name == "out" else Replicate() for name in kw_names]
        )
    for reduce_op in Partial.ALL_REDUCE_OPS:
        strategies.append(
            [Replicate(), Partial(reduce_op)] + [Replicate() for _ in kw_names]
        )
    return strategies


@register_single_dim_strategy(
    [
        aten.new_empty.default,
        aten.new_full.default,
        aten.new_ones.default,
        aten.new_zeros.default,
        aten.new_empty_strided.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def new_factory_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    # Replicated output is always legal and is inserted by the single-dim
    # infra. If input and output have the same shape, sharded inputs can also
    # propagate to the new tensor.
    input_meta = cast(TensorMeta, args_schema[0])
    output_shape = cast(Sequence[object], args_schema[1])
    same_shape = tuple(input_meta.shape) == tuple(output_shape)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        placement = _ShardingPlaceholder(dim)
        if same_shape:
            strategies.append([placement, placement])
        else:
            strategies.append([Replicate(), placement])

    # Uninitialized factories (new_empty*) can propagate Partial when the shape
    # matches because the storage is expected to be overwritten later, e.g.
    # autograd's new_empty_strided + copy_ path. Initialized factories keep a
    # Replicate output for Partial inputs to avoid reducing generated constants.
    if op in (aten.new_empty.default, aten.new_empty_strided.default):
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
            if same_shape:
                strategies.append([Partial(reduce_op), Partial(reduce_op)])
            else:
                strategies.append([Replicate(), Partial(reduce_op)])
    else:
        for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
            strategies.append([Replicate(), Partial(reduce_op)])

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


@register_single_dim_strategy(
    aten.select.int,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def select_int_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward all shardings except the selected dimension.

    select removes selected_dim. A shard before that dim keeps its output dim;
    a shard after it shifts left by one. If selected_dim is sharded, the
    strategy search must pick a replicated/resharded fallback.
    """
    input_meta = cast(TensorMeta, args_schema[0])
    selected_dim = normalize_dim(cast(int, args_schema[1]), len(input_meta.shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        if dim == selected_dim:
            continue
        output_dim = dim if dim < selected_dim else dim - 1
        strategies.append([_ShardingPlaceholder(output_dim), _ShardingPlaceholder(dim)])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.select_backward.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def select_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Inverse of select: insert the selected dimension into the output layout."""
    grad_meta = cast(TensorMeta, args_schema[0])
    input_sizes = cast(Sequence[object], args_schema[1])
    dim = normalize_dim(cast(int, args_schema[2]), len(input_sizes))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for grad_dim in range(len(grad_meta.shape)):
        output_dim = grad_dim if grad_dim < dim else grad_dim + 1
        strategies.append(
            [_ShardingPlaceholder(output_dim), _ShardingPlaceholder(grad_dim)]
        )
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice.Tensor,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def slice_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward shardings except on the sliced dimension.

    A nontrivial slice changes offsets along slice_dim, so a local shard on
    that dimension cannot be forwarded without extra global index handling. A
    statically full slice is just a view and can keep that sharding.
    """
    defaults = (None, 0, None, None, 1)
    input_meta, dim, start, end, step = args_schema + defaults[len(args_schema) :]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    input_shape = input_meta.shape

    if start is None:
        start = 0
    if not isinstance(start, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(start)}")
    if end is not None and not isinstance(end, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(end)}")
    if not isinstance(step, IntLike):
        raise AssertionError(f"Expected IntLike, got {type(step)}")
    if end is None or statically_known_true(end > input_shape[dim]):
        end = input_shape[dim]

    slice_dim = normalize_dim(dim, len(input_shape))
    start = start + input_shape[dim] if statically_known_true(start < 0) else start
    end = end + input_shape[dim] if statically_known_true(end < 0) else end

    statically_redundant_slice = (
        statically_known_true(start == 0)
        and statically_known_true(end == input_shape[dim])
        and statically_known_true(step == 1)
    )

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_shape)):
        if dim != slice_dim or statically_redundant_slice:
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice_backward.default,
    schema_info=RuntimeSchemaInfo(1),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def slice_backward_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward grad shardings except along the original sliced dimension."""
    grad_output_meta = args_schema[0]
    input_sizes = args_schema[1]
    dim = args_schema[2]
    if not isinstance(grad_output_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(grad_output_meta)}")
    if not isinstance(input_sizes, Sequence):
        raise AssertionError(f"Expected Sequence, got {type(input_sizes)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    slice_dim = normalize_dim(dim, len(input_sizes))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(grad_output_meta.shape)):
        if dim != slice_dim:
            strategies.append([_ShardingPlaceholder(dim), _ShardingPlaceholder(dim)])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op), Partial(reduce_op)])
    return strategies


@register_single_dim_strategy(
    aten.slice_scatter.default,
    schema_info=RuntimeSchemaInfo(2),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def slice_scatter_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Let input, src, and output follow each other off the scatter dimension.

    slice_scatter replaces a slice of self with src. Non-slice dimensions have
    matching extents and can be sharded locally; the slice dimension stays
    replicated via fallback because local slice coordinates would otherwise
    depend on global offsets.
    """
    defaults = (None, None, 0)
    input_meta, src_meta, dim = args_schema[:3] + defaults[len(args_schema[:3]) :]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(src_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(src_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    slice_dim = normalize_dim(dim, len(input_meta.shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for dim in range(len(input_meta.shape)):
        if dim != slice_dim:
            strategies.append(
                [
                    _ShardingPlaceholder(dim),
                    _ShardingPlaceholder(dim),
                    _ShardingPlaceholder(dim),
                ]
            )
    # Replacement is pointwise by position, so matching Partial placements pass
    # through for the supported reductions.
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append(
            [
                Partial(reduce_op),
                Partial(reduce_op),
                Partial(reduce_op),
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
    """Only allow replication on the input; the output is a local scalar."""
    return []


@register_single_dim_strategy(
    [
        aten.scatter_.value,
        aten.scatter.value,
        aten.scatter_.src,
        aten.scatter.src,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def scatter_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Sharding strategy for scatter/scatter_ value and src overloads.

    scatter writes along the dim argument, so that tensor axis must stay
    replicated. Any other axis can be sharded only if every tensor operand has
    the same size on it; then each rank scatters into its local slice with no
    communication.

    Rows are [output, input, index] for scalar value overloads and
    [output, input, index, src] when src is a tensor. The all-Replicate
    fallback is inserted by the single-dim strategy infra.
    """
    input_meta, dim, index_meta = args_schema[:3]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(index_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(index_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")

    input_shape = input_meta.shape
    index_shape = index_meta.shape
    scatter_dim = normalize_dim(dim, len(input_shape))

    # The scalar value overloads carry no src tensor. scatter.src can also
    # receive a Python number, so only TensorMeta creates a src placement slot.
    src_meta = args_schema[3] if len(args_schema) > 3 else None
    has_tensor_src = isinstance(src_meta, TensorMeta)
    src_shape = src_meta.shape if has_tensor_src else None
    num_specs = 4 if has_tensor_src else 3

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if len(input_shape) == len(index_shape):
        for d in range(len(input_shape)):
            # Shard a non-scatter axis only when input/index/src shards align.
            if d == scatter_dim or input_shape[d] != index_shape[d]:
                continue
            if src_shape is not None and (
                len(src_shape) != len(index_shape) or src_shape[d] != index_shape[d]
            ):
                continue
            placement = _ShardingPlaceholder(d)
            strategies.append([placement] * num_specs)
    return strategies


@register_single_dim_strategy(
    aten.scatter_add.default, schema_info=RuntimeSchemaInfo(1)
)
def scatter_add_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Shard scatter_add only on non-scatter axes with aligned sizes.

    Placement rows are [output, input, index, src]. The scatter axis stays
    replicated because indices address the full extent of that dimension.
    """
    input_meta, dim, index_meta = args_schema[:3]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(index_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(index_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    input_shape = input_meta.shape
    index_shape = index_meta.shape
    scatter_dim = normalize_dim(dim, len(input_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    if len(input_shape) == len(index_shape):
        for d in range(len(input_shape)):
            if d != scatter_dim and input_shape[d] == index_shape[d]:
                placement = _ShardingPlaceholder(d)
                strategies.append([placement, placement, placement, placement])
    return strategies


@register_single_dim_strategy(aten.gather.default, schema_info=RuntimeSchemaInfo(1))
def gather_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Sharding strategy for gather with rows [output, input, index]."""
    input_meta, dim, index_meta = args_schema[:3]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
    if not isinstance(index_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(index_meta)}")
    if not isinstance(dim, int):
        raise AssertionError(f"Expected int, got {type(dim)}")
    input_shape = input_meta.shape
    index_shape = index_meta.shape
    gather_dim = normalize_dim(dim, len(input_shape))

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    # Input sharded on gather_dim: only valid when index has size 1 on that
    # dim. Index/output use a mask partial so reducing combines the local
    # gather results into the requested global value.
    if (
        len(input_shape) > 0
        and gather_dim < len(index_shape)
        and index_shape[gather_dim] == 1
    ):
        mask_partial = _MaskPartial(offset_shape=input_shape, offset_dim=gather_dim)
        strategies.append([mask_partial, Shard(gather_dim), mask_partial])

    # Index sharded on gather_dim: input must be replicated because each rank
    # may request values from the full input; output follows index.
    if len(input_shape) > 0 and gather_dim < len(index_shape):
        strategies.append([Shard(gather_dim), Replicate(), Shard(gather_dim)])

    # Any non-gather dimension can be sharded across all tensors.
    if len(input_shape) == len(index_shape):
        for d in range(len(input_shape)):
            if d != gather_dim:
                placement = _ShardingPlaceholder(d)
                strategies.append([placement, placement, placement])

    return strategies


def _derive_follow_placements_from_tuple_strategy(
    op: torch._ops.OpOverload,
    tuple_strategy: TupleStrategy,
) -> Sequence[Placement]:
    """
    derive the placements to follow from the tuple strategy, mainly used by
    aten.stack, where each operand have the same shape, and correspondingly
    expecting the same sharding
    """

    def merge_placement(
        cur_placement: Placement, new_placement: Placement
    ) -> Placement:
        # semantic if we already have a follow placement, we
        # check each placement for the current arg placement
        # to see if we want to merge/adjust the placement to follow
        # the priority: Partial -> Shard -> Replicate
        # _StridedShard.__eq__ compares both dim and split_factor,
        # so two _StridedShard with different split_factor won't match here.
        if cur_placement == new_placement:
            return cur_placement

        if cur_placement.is_partial():
            if _is_shard_like(new_placement):
                # follow new placement
                return new_placement
            elif new_placement.is_partial():
                # different partial types, we can't merge and have to replicate all here
                return Replicate()
            else:
                # follow partial
                return cur_placement
        elif _is_shard_like(cur_placement):
            if _is_shard_like(new_placement):
                # cur/new placement are different sharding (i.e. different shard dim)
                # currently fallback to replicate all args
                return Replicate()
            else:
                # for partial/replicate, follow the current shard placement
                return cur_placement
        else:
            # current replicate, just follow new placement
            return new_placement

    follow_placements: list[Placement] | None = None
    mesh = tuple_strategy.child_mesh(0)
    for arg_strategy in tuple_strategy.children:
        if not isinstance(arg_strategy, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {type(arg_strategy)}")
        if arg_strategy.mesh != mesh:
            raise ValueError(
                f"All operands in {op} must have the same mesh, "
                f"but got {arg_strategy.mesh} and {mesh}."
            )

        for placement_strategy in arg_strategy.strategies:
            arg_placements = placement_strategy.output_spec.placements
            if follow_placements is None:
                follow_placements = list(arg_placements)
                continue
            if follow_placements is None:
                raise AssertionError(
                    "follow_placements should not be None at this point"
                )
            for mesh_idx in range(mesh.ndim):
                # merge placements with the priority
                follow_placements[mesh_idx] = merge_placement(
                    follow_placements[mesh_idx], arg_placements[mesh_idx]
                )
    if follow_placements is None:
        raise AssertionError("follow placements should not be None!")
    return follow_placements


@register_op_strategy(aten.stack.default, RuntimeSchemaInfo(1, needs_pytree=True))
def stack_strategy(op_schema: OpSchema) -> StrategyType:
    # We keep stack as an OpStrategy because .stack is used w/ NormPartial, and its
    # impossible to include every NormPartial rules as there are infinite possibilities
    # with ord. Future changes to infra could be made, but we prefer to deprecate
    # NormPartial at some point.

    args_schema = op_schema.args_schema
    input_tuple_strategy = args_schema[0]
    if not isinstance(input_tuple_strategy, TupleStrategy):
        raise AssertionError(f"Expected TupleStrategy, got {input_tuple_strategy}")
    input_strategies: list[OpStrategy] = []
    for child in input_tuple_strategy.children:
        if not isinstance(child, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {child}")
        input_strategies.append(child)
    first_input_strategy = input_strategies[0]
    common_input_ndim = first_input_strategy.ndim
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    # normalize the dim to be within the output ndim (input ndim + 1),
    # since stack inserts a new dimension
    dim = normalize_dim(dim, common_input_ndim + 1)

    mesh = first_input_strategy.mesh

    follow_placements = _derive_follow_placements_from_tuple_strategy(
        op_schema.op, input_tuple_strategy
    )

    # create op strategy base on the follow placements
    op_strategy = OpStrategy([])

    input_specs = tuple(
        DTensorSpec(mesh, tuple(follow_placements))
        for _ in range(len(input_tuple_strategy.children))
    )

    # stack op would "insert" new dim, so all sharded dim >= the inserted dim need to
    # be normalized with the new Shard placement
    follow_placements = shift_shard_dims_after_insert(follow_placements, dim)
    output_spec = DTensorSpec(mesh, tuple(follow_placements))
    redistribute_cost = [
        generate_redistribute_costs(input_strategies[i], input_specs[i])
        for i in range(len(input_specs))
    ]
    op_strategy.strategies.append(
        OpSpec(
            output_specs=output_spec,
            input_specs=input_specs,
            redistribute_cost=redistribute_cost,
        )
    )
    return op_strategy


@register_single_dim_strategy(
    aten.cat.default,
    RuntimeSchemaInfo(1, needs_pytree=True),
    allow_unbacked_sharding=False,
)
def cat_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward cat input placements except along the concatenation dimension.

    Concatenation changes only cat_dim; sharding there needs redistribution, so
    this rule offers non-cat dims and leaves the replicate fallback to the
    single-dim infra. Legacy 1-D empty tensors have no real cat layout, so they
    stay Replicate while non-empty tensors follow the output placement.
    """
    input_list = args_schema[0]
    # Unfortunate naming, but yes it's a TensorList input, and we represent it
    # as a tuple of TensorMeta.
    if not isinstance(input_list, (tuple, list)):
        raise AssertionError(type(input_list))
    if not all(isinstance(tm, TensorMeta) for tm in input_list):
        raise AssertionError

    if isinstance(input_list, list):
        input_list = tuple(input_list)

    def is_legacy_cat_empty(meta: TensorMeta) -> bool:
        return len(meta.shape) == 1 and statically_known_true(meta.shape[0] == 0)

    num_inputs = len(input_list)
    if num_inputs == 0:
        raise AssertionError("Expected cat inputs to be non-empty list of tensors")
    legacy_empty_inputs = tuple(is_legacy_cat_empty(meta) for meta in input_list)
    real_ndim_set = {
        len(meta.shape)
        for meta, legacy_empty in zip(input_list, legacy_empty_inputs, strict=True)
        if not legacy_empty
    }
    if len(real_ndim_set) > 1:
        raise AssertionError("Expected all non-empty cat inputs to be the same ndim")
    common_ndim = next(iter(real_ndim_set), 1)
    cat_dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    cat_dim = normalize_dim(cat_dim, common_ndim)
    single_dim_strategies = []
    for i in range(common_ndim):
        if i != cat_dim:
            single_dim_strategies.append(
                [_ShardingPlaceholder(i)]
                + [
                    Replicate() if legacy_empty else _ShardingPlaceholder(i)
                    for legacy_empty in legacy_empty_inputs
                ]
            )
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        single_dim_strategies.append(
            [Partial(reduce_op)]  # pyrefly: ignore [bad-argument-type]
            + [
                Replicate() if legacy_empty else Partial(reduce_op)
                for legacy_empty in legacy_empty_inputs
            ]
        )
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
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
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
            for op in _PARTIAL_PASS_THROUGH_REDUCE_OPS
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
    allow_unbacked_sharding=False,
)
def split_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Forward all shardings except the split dimension.

    Splitting a sharded split_dim would require each output chunk to know
    global chunk boundaries, so the single-dim rule excludes that dim and
    relies on the redistribute-to-replicate fallback.
    """
    input_meta = args_schema[0]
    split_size_or_sections = args_schema[1]
    if not isinstance(input_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta, got {type(input_meta)}")
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
    num_outputs = len(output_size_list)

    strategies: list[list[Placement | _ShardingPlaceholder]] = []
    for d in range(input_ndim):
        if d != dim:
            placement = _ShardingPlaceholder(d)
            # pyrefly: ignore [bad-argument-type]
            strategies.append([placement] * num_outputs + [placement])
    for reduce_op in _PARTIAL_PASS_THROUGH_REDUCE_OPS:
        strategies.append([Partial(reduce_op)] * (num_outputs + 1))
    return strategies


# TODO: fix remaining failures in xfail("unbind") in test_dtensor_ops.py
#       and remove this xfail item
@register_op_strategy(aten.unbind.int, schema_info=RuntimeSchemaInfo(1))
def gen_unbind_strategy(op_schema: OpSchema) -> StrategyType:
    """Forward all shardings except the unbind dimension."""
    input_strategy = op_schema.args_schema[0]
    if not isinstance(input_strategy, OpStrategy):
        raise AssertionError(f"Expected OpStrategy, got {type(input_strategy)}")
    input_ndim = input_strategy.ndim
    input_shape = input_strategy.shape
    unbind_dim = (
        cast(int, op_schema.args_schema[1]) if len(op_schema.args_schema) > 1 else 0
    )
    unbind_dim = normalize_dim(unbind_dim, input_ndim)

    mesh = input_strategy.mesh
    unbind_strategy = OpStrategy([])
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_dim_sharded(arg_spec, dim=unbind_dim):
            raise RuntimeError(
                f"Attempted to unbind along the sharded dimension {unbind_dim}. ",
                "It cannot be performed without redistribution, which is disallowed "
                "by the current operator.",
            )
        # Only add the strategy if the unbind dim is not sharded.
        output_placements = shift_shard_dims_after_remove(
            arg_spec.placements, unbind_dim
        )
        output_specs = tuple(
            DTensorSpec(mesh, tuple(output_placements))
            for _ in range(input_shape[unbind_dim])
        )
        unbind_strategy.strategies.append(
            OpSpec(
                output_specs=output_specs,
                input_specs=(arg_spec,),
                redistribute_cost=[[0.0] * len(input_strategy.strategies)],
            )
        )
    return unbind_strategy


@register_single_dim_strategy(
    aten.eye.m_out,
    schema_info=RuntimeSchemaInfo(static_kwargkey=["out"]),
    allow_unbacked_sharding=True,
    allow_uneven_sharding=True,
)
def eye_out_single_dim_strategy(
    op: OpOverload, args_schema: ArgsType, kwargs_schema: KwargsType
) -> list[list[Placement | _ShardingPlaceholder]]:
    """Strategy for torch.eye(..., out=...).

    Sharded eye needs global index offsets to place the diagonal correctly.
    The local out= kernel does not have that context, so only the implicit
    all-Replicate strategy is valid.
    """
    out_meta = kwargs_schema["out"]
    if not isinstance(out_meta, TensorMeta):
        raise AssertionError(f"Expected TensorMeta for out, got {type(out_meta)}")

    return []


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
