# mypy: allow-untyped-defs
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import sympy

import torch
from torch._prims_common import get_computation_dtype
from torch.utils._ordered_set import OrderedSet

from .elementwise_lowerings import div, square, sub
from .ir import Reduction
from .lowering import (
    _validate_dim,
    aten,
    clone,
    config,
    empty_like,
    ExpandView,
    fallback_handler,
    ir,
    is_boolean_dtype,
    is_integer_dtype,
    is_triton,
    make_pointwise,
    ops,
    PermuteView,
    prims,
    register_lowering,
    squeeze,
    sympy_product,
    to_dtype,
)


if TYPE_CHECKING:
    from .ops_handler import ReductionType


def _validate_reduction_axis(x, axis):
    size = x.get_size()
    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = range(len(size))
    if len(size) == 0:
        assert tuple(axis) in [(), (0,), (-1,)], f"invalid axis: {axis}"
        return []
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(size) if len(size) else 1
        assert 0 <= axis[i] < len(size) or (len(size) == 0 and axis[i] == 0)
    assert len(OrderedSet(axis)) == len(axis), "reduction axis not unique"
    return axis


def _make_reduction_inner(
    x, *, axis, keepdims, dtype, override_return_dtype, reduction_type=None
):
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = OrderedSet[int](_validate_reduction_axis(x, axis))

    kept_sizes = []
    kept_idx = []
    reduced_sizes = []
    reduced_idx = []
    for i in range(len(size)):
        if i in axis:
            reduced_idx.append(i)
            reduced_sizes.append(size[i])
        else:
            kept_idx.append(i)
            kept_sizes.append(size[i])

    # For argmax/argmin compute logical indices when the tensor has non-contiguous layout.
    should_compute_logical_index = False
    if (
        reduction_type in ("argmax", "argmin")
        and len(reduced_sizes) > 1
        and is_triton(x)
    ):
        if isinstance(x.data, PermuteView):
            should_compute_logical_index = True
        elif isinstance(x.data, ir.ReinterpretView) or (
            isinstance(x.data, ir.StorageBox) and isinstance(x.data.data, ir.Buffer)
        ):
            layout = x.get_layout()
            should_compute_logical_index = (
                layout.is_transposed() or not layout.is_contiguous()
            )

    def loader(index, reduction_index):
        assert len(reduction_index) == len(reduced_idx)
        if keepdims:
            assert len(index) == len(size)
            index = [index[i] for i in kept_idx]
        assert len(index) == len(kept_idx)
        new_index = [None] * (len(index) + len(reduction_index))
        for idx, var in itertools.chain(
            zip(kept_idx, index), zip(reduced_idx, reduction_index)
        ):
            new_index[idx] = var
        value = inner_loader(new_index)

        # For argmax/argmin, return tuple with logical linear index if needed
        if should_compute_logical_index:
            rindex = [sympy.expand(i) for i in reduction_index]

            # Compute linear index in row-major order
            # For reduction_ranges = [4, 6]: linear_index = r0 * 6 + r1
            linear_idx = rindex[0]
            for i in range(1, len(rindex)):
                linear_idx = linear_idx * reduced_sizes[i] + rindex[i]

            return (value, ops.index_expr(linear_idx, torch.int64))

        return value

    if keepdims:
        new_size = list(size)
        for i in reduced_idx:
            new_size[i] = sympy.S.One
    else:
        new_size = kept_sizes

    inner_loader = x.make_loader()
    return dict(
        device=x.get_device(),
        dst_dtype=override_return_dtype or x.get_dtype(),
        src_dtype=x.get_dtype(),
        inner_fn=loader,
        ranges=new_size,
        reduction_ranges=reduced_sizes,
    )


def make_reduction(reduction_type: ReductionType, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        # For argmax/argmin on boolean tensors, cast to int32 first to ensure
        # correct comparison in Triton. See https://github.com/pytorch/pytorch/issues/174069
        # Only apply on Triton backend; MPS handles bool comparisons natively.
        if (
            reduction_type in ("argmax", "argmin")
            and x.get_dtype() == torch.bool
            and is_triton(x)
        ):
            x = to_dtype(x, torch.int32)
        kwargs = _make_reduction_inner(
            x,
            axis=axis,
            keepdims=keepdims,
            dtype=dtype,
            override_return_dtype=override_return_dtype,
            reduction_type=reduction_type,
        )
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)
        if isinstance(
            result.data.data,  # type: ignore[attr-defined, attr-type, union-attr]
            Reduction,
        ):  # Only realize if reduction isn't unrolled
            result.realize()
        return result

    return inner


def _make_scan_inner(x, *, axis, dtype):
    if dtype is not None:
        x = to_dtype(x, dtype)
    axis = _validate_dim(x, axis)

    return dict(
        device=x.get_device(),
        dtypes=(x.get_dtype(),),
        inner_fns=(x.make_loader(),),
        size=x.get_size(),
        axis=axis,
    )


@register_lowering(aten.mean)
def mean(x, axis=None, keepdim=False, *, dtype=None):
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    # compute in higher-precision until end of mean lowering
    output_dtype = x.get_dtype()
    if output_dtype in (torch.float16, torch.bfloat16):
        x = to_dtype(x, torch.float)
    sum_result = sum_(x, axis, keepdim)
    denom = sympy_product(size[i] for i in axis)
    denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return to_dtype(div(sum_result, denom), output_dtype)


def var_mean_sum_(x, axis, correction, keepdim, return_mean):
    if correction is None:
        correction = 1

    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    x_mean = mean(x, axis, keepdim=True)
    if return_mean:
        x_mean.realize()

    diffs = square(sub(x, x_mean))
    sum_result = sum_(diffs, axis, keepdim)

    denom = sympy_product(size[i] for i in axis)
    if correction:
        denom = sympy.Max(denom - correction, 0)
    denom = ir.IndexingConstant(index=denom, dtype=x.get_dtype(), device=x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    x_var = div(sum_result, denom)
    if not return_mean:
        return (x_var,)

    x_mean = x_mean if keepdim else squeeze(x_mean, axis)
    return x_var, x_mean


def use_two_step_variance(x, axis, keepdim):
    # two-step algorithm can get better performance in small reductions size
    # while it can accumulate more numerical error than Welford algorithm.
    axis = _validate_reduction_axis(x, axis)
    kwargs = _make_reduction_inner(
        x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
    )

    ranges = kwargs["ranges"]
    reduction_numel = sympy_product(kwargs["reduction_ranges"])
    device = x.get_device()
    if not (device and device.type == "cpu"):
        threshold = config.unroll_reductions_threshold
    else:
        # 1024 is a default value to pass all the UTs about accuracy.
        # A larger threshold can still get performance benefits.
        threshold = config.cpp.use_two_step_variance_threshold
    return (
        isinstance(reduction_numel, sympy.Integer)
        and int(reduction_numel) <= threshold
        and sympy_product(ranges) != 1
    )


def var_mean_welford_(x, axis, *, correction, keepdim, return_mean):
    if correction is None:
        correction = 1

    kwargs = _make_reduction_inner(
        x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None
    )
    loader = kwargs.pop("inner_fn")
    kwargs.pop("dst_dtype")
    kwargs.pop("src_dtype")

    mean, m2, _ = ir.WelfordReduction.create(
        inner_fns=(loader,),
        reduction_type="welford_reduce",
        dtype=x.get_dtype(),
        **kwargs,
    )
    m2.realize()

    dtype = x.get_dtype()
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    rnumel = sympy_product(size[i] for i in axis)

    def get_constant_or_index_expr(x, dtype):
        if isinstance(x, sympy.Expr) and not x.is_number:
            return ops.to_dtype(ops.index_expr(x, torch.int64), dtype)
        return ops.constant(x, dtype)

    def scale_fn(data):
        c = get_constant_or_index_expr(correction, dtype)
        N = get_constant_or_index_expr(rnumel, dtype)
        zero = ops.constant(0, dtype)
        return data / ops.maximum(zero, N - c)

    var = make_pointwise(scale_fn)(m2)

    if return_mean:
        mean.realize()
        return var, mean
    return (var,)


def var_mean_helper_(x, *, axis, correction, keepdim, return_mean):
    out_dtype = x.get_dtype()
    compute_dtype = get_computation_dtype(out_dtype)
    x = to_dtype(x, compute_dtype, copy=False)
    kwargs = dict(
        x=x,
        axis=axis,
        correction=correction,
        keepdim=keepdim,
        return_mean=return_mean,
    )
    output = (
        var_mean_sum_(**kwargs)
        if (
            config.mtia.disable_welford_reduction
            or use_two_step_variance(x, axis=axis, keepdim=keepdim)
        )
        else var_mean_welford_(**kwargs)
    )
    output = tuple(to_dtype(x, out_dtype, copy=False) for x in output)
    return output[0] if not return_mean else output


@register_lowering([aten.var, prims.var])
def var_(x, axis=None, *, correction=None, keepdim=False):
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=False
    )


@register_lowering(aten.var_mean)
def var_mean(x, axis=None, *, correction=None, keepdim=False):
    return var_mean_helper_(
        x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True
    )


@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    fn = make_reduction("sum", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)


fallback_cumsum = fallback_handler(aten.cumsum.default)
fallback_cumprod = fallback_handler(aten.cumprod.default)
fallback_logcumsumexp = fallback_handler(aten.logcumsumexp.default)
fallback_cummax = fallback_handler(aten.cummax.default)
fallback_cummin = fallback_handler(aten.cummin.default)


@register_lowering(aten.cumsum)
def cumsum(x, axis=None, dtype=None):
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    if len(x.get_size()) == 0:
        assert axis in [0, -1]
        dtype = dtype or x.get_dtype()
        return to_dtype(x, dtype, copy=True)

    def combine_fn(a_tuple, b_tuple):
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.add(a, b),)

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if result is None:
        return fallback_cumsum(x, dim=axis, dtype=dtype)
    return result


@register_lowering(aten.cumprod)
def cumprod(x, axis=None, dtype=None):
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    if len(x.get_size()) == 0:
        assert axis in [0, -1]
        dtype = dtype or x.get_dtype()
        return to_dtype(x, dtype, copy=True)

    def combine_fn(a_tuple, b_tuple):
        (a,) = a_tuple
        (b,) = b_tuple
        return (ops.mul(a, b),)

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=combine_fn)
    if result is None:
        return fallback_cumprod(x, dim=axis, dtype=dtype)
    return result


@register_lowering(aten.logcumsumexp)
def logcumsumexp(x, dim):
    def log_add_exp_helper(a_tuple, b_tuple):
        (a,) = a_tuple
        (b,) = b_tuple
        min_v = ops.minimum(a, b)
        max_v = ops.maximum(a, b)
        mask = (min_v != max_v) | (~ops.isinf(min_v))
        return (ops.where(mask, ops.log1p(ops.exp(min_v - max_v)) + max_v, a),)

    dtype = x.get_dtype()
    if len(x.get_size()) == 0:
        assert dim in [0, -1]
        return clone(x)

    kwargs = _make_scan_inner(x, axis=dim, dtype=dtype)
    (result,) = ir.Scan.create(**kwargs, combine_fn=log_add_exp_helper)
    if result is None:
        return fallback_logcumsumexp(x, dim=dim)
    return result


@register_lowering(aten.cummax, type_promotion_kind=None)
def cummax(x, axis=None):
    if len(x.get_size()) == 0:
        assert axis in [0, -1]
        return clone(x), empty_like(x, dtype=torch.int64)

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmax", dtype=dtype, arg_break_ties_left=False
    )

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    kwargs["dtypes"] = (dtype, torch.int64)
    kwargs["inner_fns"] = (
        x.make_loader(),
        lambda idx: ops.index_expr(idx[axis], torch.int64),
    )
    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)  # type: ignore[arg-type]
    if values is None:
        return fallback_cummax(x, dim=axis)
    return values, indices


@register_lowering(aten.cummin, type_promotion_kind=None)
def cummin(x, axis=None):
    if len(x.get_size()) == 0:
        assert axis in [0, -1]
        return clone(x), empty_like(x, dtype=torch.int64)

    dtype = x.get_dtype()
    combine_fn = ir.get_reduction_combine_fn(
        "argmin", dtype=dtype, arg_break_ties_left=False
    )

    kwargs = _make_scan_inner(x, axis=axis, dtype=dtype)
    kwargs["dtypes"] = (dtype, torch.int64)
    kwargs["inner_fns"] = (
        x.make_loader(),
        lambda idx: ops.index_expr(idx[axis], torch.int64),
    )
    values, indices = ir.Scan.create(**kwargs, combine_fn=combine_fn)  # type: ignore[arg-type]
    if values is None:
        return fallback_cummin(x, dim=axis)
    return values, indices


@register_lowering(aten.prod)
def prod(x, axis=None, keepdims=False, *, dtype=None):
    if (
        is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    ) and dtype is None:
        dtype = torch.int64

    fn = make_reduction("prod", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)


@register_lowering(aten.any)
def reduce_any(x, dim=None, keepdim=False):
    x = to_dtype(x, torch.bool)
    return make_reduction("any")(x, axis=dim, keepdims=keepdim)


@register_lowering(aten.max, type_promotion_kind=None)
def reduce_max(x, dim=None, keepdim=False):
    if dim is not None:
        return (
            reduce_amax(x, axis=dim, keepdims=keepdim),
            reduce_argmax(x, axis=dim, keepdims=keepdim),
        )

    return reduce_amax(x, axis=None, keepdims=keepdim)


@register_lowering(aten.min, type_promotion_kind=None)
def reduce_min(x, dim=None, keepdim=False):
    if dim is not None:
        return (
            reduce_amin(x, axis=dim, keepdims=keepdim),
            reduce_argmin(x, axis=dim, keepdims=keepdim),
        )

    return reduce_amin(x, axis=None, keepdims=keepdim)


register_lowering(prims.xor_sum)(make_reduction("xor_sum"))
reduce_amax = register_lowering(aten.amax)(make_reduction("max"))
reduce_amin = register_lowering(aten.amin)(make_reduction("min"))
reduce_argmax = register_lowering(aten.argmax)(
    make_reduction("argmax", override_return_dtype=torch.int64)
)
reduce_argmin = register_lowering(aten.argmin)(
    make_reduction("argmin", override_return_dtype=torch.int64)
)

__all__ = [
    "_make_reduction_inner",
    "_make_scan_inner",
    "_validate_reduction_axis",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "fallback_cummax",
    "fallback_cummin",
    "fallback_cumprod",
    "fallback_cumsum",
    "fallback_logcumsumexp",
    "logcumsumexp",
    "make_reduction",
    "mean",
    "prod",
    "reduce_amax",
    "reduce_amin",
    "reduce_any",
    "reduce_argmax",
    "reduce_argmin",
    "reduce_max",
    "reduce_min",
    "sum_",
    "use_two_step_variance",
    "var_",
    "var_mean",
    "var_mean_helper_",
    "var_mean_sum_",
    "var_mean_welford_",
]
