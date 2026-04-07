# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import functools
import itertools
import operator
from typing import Any, TYPE_CHECKING

import sympy

import torch
from torch.utils._sympy.functions import CeilDiv, FloorDiv

from .elementwise_lowerings import div_prim
from .ir import Reduction
from .lowering import (
    aten,
    ceildiv,
    clone,
    config,
    constant_boundary_condition,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    empty,
    fallback_handler,
    get_promoted_dtype,
    inductor_prims,
    ir,
    ones_like,
    ops,
    Pointwise,
    prims,
    register_lowering,
    TensorBox,
    to_dtype,
    V,
)
from .utils import pad_listlike


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def pooling_size(x, i, kernel_size, stride, padding, ceil_mode, *, dilation=None):
    if dilation is None:
        dilation = [1] * len(padding)

    x_out = FloorDiv(
        x + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) + (stride[i] - 1),
        stride[i],
    )

    if ceil_mode:
        x_alt = FloorDiv(
            x
            + 2 * padding[i]
            - dilation[i] * (kernel_size[i] - 1)
            + 2 * (stride[i] - 1),
            stride[i],
        )
        if V.graph.sizevars.guard_or_false(
            sympy.Ge((x_alt - 1) * stride[i] - x - padding[i], 0)
        ):
            # Sliding windows must start within the input or left padding
            x_alt -= 1  # type: ignore[assignment]
        if V.graph.sizevars.guard_or_false(sympy.Eq(x_out, x_alt)):
            # ceil mode is actually a no-op, lets guard on that
            ceil_mode = False
        else:
            x_out = x_alt
    return x_out, ceil_mode


def should_fallback_max_pool_with_indices(kernel_size, *, n_dim):
    kernel_size = pad_listlike(kernel_size, n_dim)
    window_size = functools.reduce(operator.mul, kernel_size)
    return window_size > 25


def max_pool_checks(
    x, kernel_size, stride, padding, dilation, n_dim, *, assert_fallback=None
):
    if padding == 0:
        padding = [0] * n_dim
    if dilation == 1:
        dilation = [1] * n_dim
    if not stride:
        stride = kernel_size

    kernel_size = pad_listlike(kernel_size, n_dim)
    stride = pad_listlike(stride, n_dim)
    padding = pad_listlike(padding, n_dim)
    dilation = pad_listlike(dilation, n_dim)

    assert isinstance(x, TensorBox)
    assert len(kernel_size) == n_dim
    assert len(stride) == n_dim
    assert len(padding) == n_dim
    assert len(dilation) == n_dim
    assert len(x.get_size()) in (n_dim + 1, n_dim + 2)

    use_fallback = should_fallback_max_pool_with_indices(kernel_size, n_dim=n_dim)
    if assert_fallback is not None:
        assert use_fallback == assert_fallback

    return kernel_size, stride, padding, dilation, use_fallback


def _max_pool_with_offsets(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    *,
    n_dim,
):
    x.realize_hint()
    batch = x.shape[:-n_dim]
    dhw = x.shape[-n_dim:]

    dhw_out, ceil_mode = zip(
        *[
            pooling_size(
                dhw[d], d, kernel_size, stride, padding, ceil_mode, dilation=dilation
            )
            for d in range(n_dim)
        ]
    )

    dtype = x.dtype
    min_value = (
        False
        if dtype is torch.bool
        else (float("-inf") if dtype.is_floating_point else torch.iinfo(dtype).min)
    )

    new_size = list(batch) + list(dhw_out)
    if any(padding) or any(ceil_mode) or any(d > 1 for d in dilation):
        x_loader = constant_boundary_condition(x, min_value, dim=n_dim)
    else:
        x_loader = x.make_loader()

    def fn_inner(idx, reduction_idx):
        prefix = idx[:-n_dim]
        bh = idx[-n_dim:]
        ih = [
            (bh[i] * stride[i]) + (reduction_idx[i] * dilation[i]) - padding[i]
            for i in range(n_dim)
        ]
        return x_loader([*prefix, *ih])

    result = Reduction.create(
        reduction_type="max",
        input_node=x,
        device=x.get_device(),
        dst_dtype=dtype,
        src_dtype=dtype,
        inner_fn=fn_inner,
        ranges=new_size,
        reduction_ranges=kernel_size,
    )
    offsets = Reduction.create(
        reduction_type="argmax",
        input_node=x,
        device=x.get_device(),
        dst_dtype=torch.int64,
        src_dtype=dtype,
        inner_fn=fn_inner,
        ranges=new_size,
        reduction_ranges=kernel_size,
    )
    if isinstance(result.data.data, Reduction):  # type: ignore[attr-defined, union-attr]
        # Only realize if reduction isn't unrolled
        result.realize()
    if isinstance(offsets.data.data, Reduction):  # type: ignore[attr-defined, union-attr]
        # Only realize if reduction isn't unrolled
        offsets.realize()

    return result, offsets


@register_lowering(prims._low_memory_max_pool_with_offsets, type_promotion_kind=None)
def _low_memory_max_pool_with_offsets(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode=False,
):
    n_dim = len(kernel_size)

    # assert we are not on a fallback path, the inductor decomp should have guaranteed this
    kernel_size, stride, padding, dilation, _ = max_pool_checks(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        n_dim,
        assert_fallback=False,
    )

    with config.patch(unroll_reductions_threshold=25):
        result, offsets = _max_pool_with_offsets(
            x,
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            n_dim=n_dim,
        )
        return result, to_dtype(offsets, torch.int8)


def _pool_offsets_to_indices(
    offsets: TensorBox,
    kernel_size: Sequence[int | torch.SymInt],
    input_size: Sequence[int | torch.SymInt],
    increments_to_index: Callable[
        [Sequence[int | torch.SymInt], Sequence[int | torch.SymInt]],
        torch._inductor.virtualized.OpsValue,
    ],
) -> TensorBox:
    n_dim = len(kernel_size)
    offsets_loader = offsets.make_loader()
    window_size = sympy.sympify(functools.reduce(operator.mul, kernel_size))

    def offsets_to_indices(idx):
        offset = offsets_loader(idx)
        offset_sympy = ops.indirect_indexing(offset, window_size)
        reduction_idx = inductor_prims._flattened_index_to_nd(offset_sympy, kernel_size)
        idhw = increments_to_index(idx, reduction_idx)
        return ops.index_expr(
            inductor_prims._flatten_index(idhw, input_size[-n_dim:]), torch.int64
        )

    indices = Pointwise.create(
        device=offsets.get_device(),
        dtype=torch.int64,
        inner_fn=offsets_to_indices,
        ranges=offsets.get_size(),
    )
    return indices


@register_lowering(
    prims._low_memory_max_pool_offsets_to_indices, type_promotion_kind=None
)
def _low_memory_max_pool_offsets_to_indices(
    offsets, kernel_size, input_size, stride, padding, dilation
):
    # TODO: Generalize to other max pooling flavors
    n_dim = len(kernel_size)

    def increments_to_index(idx, reduction_idx):
        bh = idx[-n_dim:]
        return [
            (bh[i] * stride[i]) + (reduction_idx[i] * dilation[i]) - padding[i]
            for i in range(n_dim)
        ]

    return _pool_offsets_to_indices(
        offsets, kernel_size, input_size, increments_to_index
    )


def _max_pool_with_indices(
    x,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    n_dim,
):
    kernel_size, stride, padding, dilation, _ = max_pool_checks(
        x, kernel_size, stride, padding, dilation, n_dim=n_dim
    )

    out, offsets = _max_pool_with_offsets(
        x, kernel_size, stride, padding, dilation, ceil_mode, n_dim=n_dim
    )

    indices = _low_memory_max_pool_offsets_to_indices(
        offsets,
        kernel_size,
        x.shape[-n_dim:],
        stride,
        padding,
        dilation,
    )

    return out, indices


# Fallback when we do not decompose to the low-memory path.
@register_lowering(aten.max_pool2d_with_indices, type_promotion_kind=None)
def max_pool2d_with_indices(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, n_dim=2
    )


# Fallback when we do not decompose to the low-memory path.
@register_lowering(aten.max_pool3d_with_indices, type_promotion_kind=None)
def max_pool3d_with_indices(
    x,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
):
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, n_dim=3
    )


fallback_max_pool2d_with_indices_backward = fallback_handler(
    aten.max_pool2d_with_indices_backward.default,
    add_to_fallback_set=False,
)


@register_lowering(aten.max_pool2d_with_indices_backward, type_promotion_kind=None)
def max_pool2d_with_indices_backward(
    grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
):
    """Lower max_pool2d_with_indices backward for small, non-dilated windows."""
    if padding == 0:
        padding = [0, 0]
    if dilation == 1:
        dilation = [1, 1]
    if not stride:
        stride = kernel_size

    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(dilation) == 2
    assert len(x.get_size()) in (3, 4)

    # we will read this many times, so make sure it is computed
    grad_output.realize_hint()
    gO_stride = grad_output.maybe_get_stride()
    x_stride: Sequence[Any] | None
    if isinstance(x, TensorBox) and isinstance(x.data.data, Pointwise):  # type: ignore[attr-defined]
        data = x.data.data  # type: ignore[attr-defined]
        device = data.get_device()
        assert device is not None
        x_buffer = ir.ComputedBuffer(
            name=None,
            layout=ir.FlexibleLayout(
                device=device,
                dtype=data.get_dtype(),
                size=data.get_size(),
            ),
            data=data,
        )
        x_buffer.decide_layout()
        x_stride = x_buffer.get_stride()
    else:
        x_stride = x.maybe_get_stride()

    is_channels_last = (x_stride is not None and x_stride[1] == 1) or (
        gO_stride is not None and gO_stride[1] == 1
    )
    if any(d != 1 for d in dilation):
        # dilation NYI
        return fallback_max_pool2d_with_indices_backward(
            grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
        )

    *_batch, _height, width = x.get_size()
    *_, pooled_height, pooled_width = grad_output.get_size()

    indices_loader = indices.make_loader()
    grad_loader = grad_output.make_loader()
    new_size = list(x.get_size())

    h_window_size = max(
        max(FloorDiv(h, stride[0]) - max(0, FloorDiv(h - kernel_size[0], stride[0])), 1)
        for h in range(kernel_size[0] * 2)
    )
    w_window_size = max(
        max(FloorDiv(w, stride[1]) - max(0, FloorDiv(w - kernel_size[1], stride[1])), 1)
        for w in range(kernel_size[1] * 2)
    )

    window_size = h_window_size * w_window_size

    if window_size > 25:
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_max_pool2d_with_indices_backward(
            grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices
        )

    indices_size = indices.get_size()

    def fn(idx):
        *prefix, h, w = idx
        index_test = ops.index_expr(h * width + w, torch.int32)
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(
            FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )
        pwstart = ops.index_expr(
            FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)

        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))

        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))
                grad_index = [
                    *prefix,
                    ops.indirect_indexing(
                        ops.minimum(ph, ops.sub(phend, ops.constant(1, torch.int32))),
                        indices_size[-2],
                        check=False,
                    ),
                    ops.indirect_indexing(
                        ops.minimum(pw, ops.sub(pwend, ops.constant(1, torch.int32))),
                        indices_size[-1],
                        check=False,
                    ),
                ]

                index_actual = indices_loader(grad_index)
                grad_part = grad_loader(grad_index)
                check = ops.eq(index_actual, index_test)

                if gradient is None:
                    # don't need mask for 0, 0
                    gradient = ops.where(
                        check, grad_part, ops.constant(0.0, torch.float32)
                    )
                else:
                    mask = ops.and_(
                        ops.and_(
                            ops.lt(ph, phend),
                            ops.lt(pw, pwend),
                        ),
                        check,
                    )
                    gradient = ops.where(mask, ops.add(gradient, grad_part), gradient)
        assert gradient is not None
        return gradient

    out = Pointwise.create(
        device=grad_output.get_device(),
        dtype=grad_output.get_dtype(),
        inner_fn=fn,
        ranges=new_size,
    )
    if is_channels_last:
        return ir.ExternKernel.require_channels_last(out)
    else:
        return out


def pad_adaptive_loader(x, pad_val=0.0):
    x_loader = x.make_loader()

    def load(prefix, increments, start_indices, end_indices):
        ih, iw = increments
        h_start_index, w_start_index = start_indices
        h_end_index, w_end_index = end_indices

        mask = ops.and_(
            ops.lt(
                ops.index_expr(h_start_index + ih, torch.int64),
                ops.index_expr(h_end_index, torch.int64),
            ),
            ops.lt(
                ops.index_expr(w_start_index + iw, torch.int64),
                ops.index_expr(w_end_index, torch.int64),
            ),
        )

        return ops.masked(
            mask,
            lambda: x_loader([*prefix, h_start_index + ih, w_start_index + iw]),
            pad_val,
        )

    return load


def compute_indices_adaptive_pooling(start_index, end_index, h_in, w_in, h_out, w_out):
    h_start_index = functools.partial(start_index, out_dim=h_out, inp_dim=h_in)
    h_end_index = functools.partial(end_index, out_dim=h_out, inp_dim=h_in)

    w_start_index = functools.partial(start_index, out_dim=w_out, inp_dim=w_in)
    w_end_index = functools.partial(end_index, out_dim=w_out, inp_dim=w_in)

    return h_start_index, h_end_index, w_start_index, w_end_index


def _adaptive_pooling_fn(
    start_index, end_index, kernel_maxes, in_sizes, out_sizes, pooling_fn
):
    h_in, w_in = in_sizes
    h_out, w_out = out_sizes

    (
        h_start_index_fn,
        h_end_index_fn,
        w_start_index_fn,
        w_end_index_fn,
    ) = compute_indices_adaptive_pooling(
        start_index, end_index, h_in, w_in, h_out, w_out
    )

    def fn(idx, loader):
        *prefix, bh, bw = idx

        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)

        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)

        result = None
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(
                prefix,
                [ih, iw],
                [h_start_index, w_start_index],
                [h_end_index, w_end_index],
            )
            if result is None:
                result = val
            else:
                result = pooling_fn(val, result)
        return result

    return fn


def _adaptive_pooling_fn_with_idx(
    start_index, end_index, kernel_maxes, in_sizes, out_sizes, pooling_fn
):
    h_in, w_in = in_sizes
    h_out, w_out = out_sizes

    (
        h_start_index_fn,
        h_end_index_fn,
        w_start_index_fn,
        w_end_index_fn,
    ) = compute_indices_adaptive_pooling(
        start_index, end_index, h_in, w_in, h_out, w_out
    )

    def fn(idx, loader):
        *prefix, bh, bw = idx

        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)

        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)

        maxval = None
        maxindex = None
        for ih, iw in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(
                prefix,
                [ih, iw],
                [h_start_index, w_start_index],
                [h_end_index, w_end_index],
            )

            index = ops.index_expr(
                (h_start_index + ih) * w_in + w_start_index + iw, torch.int64
            )

            if maxindex is None:
                maxindex = index
            else:
                maxindex = ops.where(ops.gt(val, maxval), index, maxindex)

            if maxval is None:
                maxval = val
            else:
                maxval = pooling_fn(val, maxval)

        return maxindex

    return fn


fallback_adaptive_avg_pool2d = fallback_handler(
    aten._adaptive_avg_pool2d.default, add_to_fallback_set=False
)


@register_lowering(aten._adaptive_avg_pool2d)
def _adaptive_avg_pool2d(x, output_size):
    if x.get_dtype() == torch.int64:
        # not supported in eager
        raise RuntimeError("'adaptive_avg_pool2d' not implemented for 'Long'")
    assert isinstance(x, TensorBox)
    assert len(output_size) == 2
    x.realize_hint()

    *batch, h_in, w_in = x.get_size()

    h_in = V.graph.sizevars.guard_int(h_in)
    w_in = V.graph.sizevars.guard_int(w_in)

    h_out, w_out = output_size

    # no-op if the same input and output
    if h_in == h_out and w_in == w_out:
        return clone(x)

    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device())
    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [FloorDiv(h_in, h_out), FloorDiv(w_in, w_out)]
        return avg_pool2d(x, kernel_size)

    h_kernel_max = ceildiv((h_in + h_out - 1), h_out)
    w_kernel_max = ceildiv((w_in + w_out - 1), w_out)

    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    window_size = h_kernel_max * w_kernel_max
    if window_size > 25:
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_adaptive_avg_pool2d(x, output_size)

    def start_index(index, out_dim, inp_dim):
        return FloorDiv((index * inp_dim), out_dim)

    def end_index(index, out_dim, inp_dim):
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)

    fn_sum = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.add,
    )

    ones_loader = pad_adaptive_loader(ones_like(x))

    def fn(idx):
        return ops.truediv(
            fn_sum(idx, pad_adaptive_loader(x)), fn_sum(idx, ones_loader)
        )

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    # TODO: should we force these to be realized?
    return rv


fallback_adaptive_max_pool2d = fallback_handler(
    aten.adaptive_max_pool2d.default, add_to_fallback_set=False
)


@register_lowering(aten.adaptive_max_pool2d)
def adaptive_max_pool2d(x, output_size):
    if x.get_dtype() == torch.int64:
        # not supported in eager
        raise RuntimeError("adaptive_max_pool2d not implemented for Long")
    assert isinstance(x, TensorBox)
    assert len(output_size) == 2
    x.realize_hint()

    *batch, h_in, w_in = x.get_size()

    h_in = V.graph.sizevars.guard_int(h_in)
    w_in = V.graph.sizevars.guard_int(w_in)

    h_out, w_out = output_size

    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device()), empty(
            o_size, dtype=torch.int64, device=x.get_device()
        )

    if h_in % h_out == 0 and w_in % w_out == 0:
        # This is handled by a decomposition
        raise ValueError

    h_kernel_max = ceildiv((h_in + h_out - 1), h_out)
    w_kernel_max = ceildiv((w_in + w_out - 1), w_out)

    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    window_size = h_kernel_max * w_kernel_max
    if window_size > 25:
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_adaptive_max_pool2d(x, output_size)

    def start_index(index, out_dim, inp_dim):
        return FloorDiv((index * inp_dim), out_dim)

    def end_index(index, out_dim, inp_dim):
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)

    inner_func_max_val = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.maximum,
    )

    inner_func_max_idx = _adaptive_pooling_fn_with_idx(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[h_in, w_in],
        out_sizes=[h_out, w_out],
        pooling_fn=ops.maximum,
    )

    def inner_fn_max_val(idx):
        return inner_func_max_val(idx, pad_adaptive_loader(x, float("-inf")))

    def inner_fn_max_idx(idx):
        return inner_func_max_idx(idx, pad_adaptive_loader(x, float("-inf")))

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=dtype,
        inner_fn=inner_fn_max_val,
        ranges=new_size,
    )
    ri = Pointwise.create(
        device=x.get_device(),
        dtype=torch.int64,
        inner_fn=inner_fn_max_idx,
        ranges=new_size,
    )
    return rv, ri


def _fractional_pooling_offsets(samples, in_sz, out_sz, kernel_sz, dim, ndims):
    out_sz = out_sz[dim]
    in_sz = in_sz[dim]
    kernel_sz = kernel_sz[dim]
    samples_loader = samples.make_loader()

    def load(prefix, i):
        # Handle indexing for samples tensor correctly for different input dimensions
        # samples tensor always has shape (N, C, 2) for fractional_max_pool2d where:
        # - N=1 for 3D inputs (C,H,W), N=batch_size for 4D inputs (N,C,H,W)
        # - C=num_channels
        # - 2 for the two spatial dimensions (height, width)
        samples_shape = samples.get_size()

        if len(samples_shape) == 3:  # Expected: (N, C, 2)
            if len(prefix) == 1:
                # 3D input case: prefix=(channel,), samples=(1, C, 2)
                # Access: samples[0, channel, dim]
                sample = samples_loader([0, prefix[0], ndims - 1 - dim])
            elif len(prefix) >= 2:
                # 4D+ input case: prefix=(batch, channel, ...), samples=(batch, C, 2)
                # Access: samples[batch, channel, dim]
                sample = samples_loader([prefix[0], prefix[1], ndims - 1 - dim])
            else:
                # Edge case - shouldn't happen for valid fractional pooling
                sample = samples_loader([0, 0, ndims - 1 - dim])
        else:
            # Fallback for unexpected tensor shapes
            sample = samples_loader([*prefix, ndims - 1 - dim])
        i_expr = ops.index_expr(i, samples.get_dtype())
        diff = ops.index_expr(in_sz - kernel_sz, torch.int64)
        out_sz_expr = ops.index_expr(out_sz - 1, torch.int64)
        alpha = ops.truediv(
            ops.to_dtype(diff, torch.float64), ops.to_dtype(out_sz_expr, torch.float64)
        )
        alpha = ops.where(ops.eq(out_sz_expr, 0), 0, alpha)
        seq_i = ops.trunc((i_expr + sample) * alpha) - ops.trunc(sample * alpha)
        seq_i = ops.to_dtype(seq_i, torch.int64)
        mask = ops.lt(i_expr, out_sz_expr)
        return ops.indirect_indexing(ops.where(mask, seq_i, diff), sympy.sympify(in_sz))

    return load


@register_lowering(aten.fractional_max_pool2d)
def fractional_max_pool2d(x, kernel_size, output_size, random_samples):
    return _fractional_max_pool(x, kernel_size, output_size, random_samples, n_dim=2)


@register_lowering(aten.fractional_max_pool3d)
def fractional_max_pool3d(x, kernel_size, output_size, random_samples):
    return _fractional_max_pool(x, kernel_size, output_size, random_samples, n_dim=3)


def _fractional_max_pool(x, kernel_size, output_size, random_samples, n_dim):
    x.realize_hint()
    batch, inp_dhw = x.shape[:-n_dim], x.shape[-n_dim:]

    with config.patch(unroll_reductions_threshold=25):
        dhw_index_fn = [
            _fractional_pooling_offsets(
                samples=random_samples,
                in_sz=inp_dhw,
                out_sz=output_size,
                kernel_sz=kernel_size,
                ndims=n_dim,
                dim=d,
            )
            for d in range(n_dim)
        ]

        x_loader = x.make_loader()

        def fn_inner(idx, reduction_idx):
            prefix = idx[:-n_dim]
            return x_loader([*prefix, *increments_to_index(idx, reduction_idx)])

        def increments_to_index(idx, reduction_idx):
            prefix = idx[:-n_dim]
            bdhw = idx[-n_dim:]
            return [
                dhw_index_fn[d](prefix, bdhw[d]) + reduction_idx[d]
                for d in range(n_dim)
            ]

        new_size = list(batch) + list(output_size)
        dtype = x.get_dtype()
        result = Reduction.create(
            reduction_type="max",
            input_node=x,
            device=x.get_device(),
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
        offsets = Reduction.create(
            reduction_type="argmax",
            input_node=x,
            device=x.get_device(),
            dst_dtype=torch.int64,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
        assert isinstance(result, TensorBox), result
        if isinstance(result.data.data, Reduction):  # type: ignore[attr-defined]
            # Only realize if reduction isn't unrolled
            result.realize()
        assert isinstance(offsets, TensorBox), offsets
        if isinstance(offsets.data.data, Reduction):  # type: ignore[attr-defined]
            # Only realize if reduction isn't unrolled
            offsets.realize()

        indices = _pool_offsets_to_indices(
            offsets, kernel_size, x.shape, increments_to_index
        )
        return result, indices


@register_lowering(aten.upsample_nearest2d_backward.default)
def upsample_nearest2d_backward(
    x, output_size=None, input_size=None, scales_h=None, scales_w=None
):
    x.realize_hint()

    *_batch, inp_h, inp_w = x.get_size()
    inp_h = V.graph.sizevars.guard_int(inp_h)
    inp_w = V.graph.sizevars.guard_int(inp_w)

    # pyrefly: ignore [not-iterable]
    *_batch, out_h, out_w = input_size

    if inp_h % out_h == 0 and inp_w % out_w == 0:
        return avg_pool2d(
            x, [FloorDiv(inp_h, out_h), FloorDiv(inp_w, out_w)], divisor_override=1
        )

    h_kernel_max = ceildiv(inp_h, out_h)
    w_kernel_max = ceildiv(inp_w, out_w)

    def start_index(index, out_dim, inp_dim):
        return CeilDiv(index * inp_dim, sympy.sympify(out_dim))

    def end_index(index, out_dim, inp_dim):
        return start_index((index + 1), out_dim, inp_dim)

    fn_sum = _adaptive_pooling_fn(
        start_index=start_index,
        end_index=end_index,
        kernel_maxes=[h_kernel_max, w_kernel_max],
        in_sizes=[inp_h, inp_w],
        out_sizes=[out_h, out_w],
        pooling_fn=ops.add,
    )

    def fn(idx):
        return fn_sum(idx, pad_adaptive_loader(x))

    rv = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        ranges=list(input_size),
    )

    return rv


@register_lowering(aten.avg_pool2d, type_promotion_kind=None)
def avg_pool2d(
    x,
    kernel_size,
    stride=(),
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return _avg_poolnd(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dim=2,
    )


@register_lowering(aten.avg_pool3d, type_promotion_kind=None)
def avg_pool3d(
    x,
    kernel_size,
    stride=(),
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return _avg_poolnd(
        x,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
        dim=3,
    )


fallbacks_avg_poolnd = [
    fallback_handler(aten.avg_pool1d.default, add_to_fallback_set=False),
    fallback_handler(aten.avg_pool2d.default, add_to_fallback_set=False),
    fallback_handler(aten.avg_pool3d.default, add_to_fallback_set=False),
]


def _avg_poolnd(
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dim,
):
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0] * dim
    kernel_size = pad_listlike(kernel_size, dim)
    stride = pad_listlike(stride, dim)
    padding = pad_listlike(padding, dim)

    assert isinstance(x, TensorBox)
    assert len(kernel_size) == dim
    assert len(stride) == dim
    assert len(padding) == dim
    assert len(x.get_size()) in (dim + 1, dim + 2)

    x.realize_hint()
    batch = x.get_size()[:-dim]
    h = x.get_size()[-dim:]

    h_out, ceil_modes = zip(
        *[
            pooling_size(h[i], i, kernel_size, stride, padding, ceil_mode)
            for i in range(dim)
        ]
    )

    if any(padding) or any(ceil_modes):
        x_loader = constant_boundary_condition(x, 0.0, dim=dim)
        had_padding = True
    else:
        x_loader = x.make_loader()
        had_padding = False

    new_size = list(batch) + list(h_out)
    dtype = x.get_dtype()
    # compute in higher-precision until scaling
    output_dtype = get_promoted_dtype(
        x,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        return_compute_dtype=True,
    )

    def fn_inner(idx, reduction_idx):
        prefix = idx[:-dim]
        bh = idx[-dim:]
        ih = reduction_idx
        ih = [bh[i] * stride[i] + ih[i] - padding[i] for i in range(dim)]
        return x_loader([*prefix, *ih])

    window_size = functools.reduce(operator.mul, kernel_size)

    if window_size > 25 and any(
        V.graph.sizevars.statically_known_true(sympy.Ne(k, s))
        for k, s in zip(kernel_size, stride)
    ):
        fallback = fallbacks_avg_poolnd[dim - 1]
        return fallback(
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    # TODO: remove this when #100331 is merged. We only do this
    # for window_size <=25 to avoid performance regressions compared
    # to the previous algorithm which unrolled manually for <=25
    context = (
        config.patch(unroll_reductions_threshold=25)
        if window_size <= 25
        else contextlib.nullcontext()
    )

    device = x.get_device()
    assert device is not None

    with context:
        rv = Reduction.create(
            reduction_type="sum",
            input_node=x,
            device=device,
            dst_dtype=output_dtype,
            src_dtype=dtype,
            inner_fn=fn_inner,
            ranges=new_size,
            reduction_ranges=kernel_size,
        )
    if hasattr(rv.data, "data") and isinstance(rv.data.data, Reduction):
        # Only realize if reduction isn't unrolled
        rv.realize()

    if not had_padding or divisor_override:
        divisor = divisor_override if divisor_override else window_size
        result = div_prim(rv, divisor)
    else:

        def fn_count(idx):
            bh = idx[-dim:]

            divide_factors = []
            for i in range(dim):
                hstart = bh[i] * stride[i] - padding[i]
                hend = sympy.Min(hstart + kernel_size[i], h[i] + padding[i])
                if not count_include_pad:
                    hstart = sympy.Max(hstart, 0)
                    hend = sympy.Min(hend, h[i])
                factor = ops.index_expr(hend - hstart, torch.int32)
                divide_factors.append(factor)
            return functools.reduce(ops.mul, divide_factors)

        divide_factor = Pointwise.create(
            device=x.get_device(),
            dtype=dtype,
            inner_fn=fn_count,
            ranges=new_size,
        )
        result = div_prim(rv, divide_factor)

    return to_dtype(result, dtype)


fallback_avg_pool2d_backward = fallback_handler(
    aten.avg_pool2d_backward.default, add_to_fallback_set=False
)


@register_lowering(aten.avg_pool2d_backward, type_promotion_kind=None)
def avg_pool2d_backward(
    grad_output,
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    """Lower avg_pool2d backward when the pooling window is small enough to inline."""
    assert divisor_override is None or divisor_override != 0, "divisor must be not zero"
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]

    assert isinstance(grad_output, TensorBox)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(x.get_size()) in (3, 4)

    grad_output.realize_hint()  # we will read this many times, so make sure it is computed

    *_, height, width = x.get_size()

    _h_out, ceil_mode1 = pooling_size(
        height, 0, kernel_size, stride, padding, ceil_mode
    )
    _w_out, ceil_mode2 = pooling_size(width, 1, kernel_size, stride, padding, ceil_mode)

    grad_loader = grad_output.make_loader()

    had_padding = padding[0] or padding[1] or ceil_mode1 or ceil_mode2

    *_, pooled_height, pooled_width = grad_output.get_size()
    new_size = list(x.get_size())
    dtype = x.get_dtype()

    h_window_size = max(
        max(FloorDiv(h, stride[0]) - max(0, FloorDiv(h - kernel_size[0], stride[0])), 1)
        for h in range(kernel_size[0] * 2)
    )
    w_window_size = max(
        max(FloorDiv(w, stride[1]) - max(0, FloorDiv(w - kernel_size[1], stride[1])), 1)
        for w in range(kernel_size[1] * 2)
    )

    window_size = h_window_size * w_window_size
    if window_size > 25:
        # Kernel size too big. Results in hard-to-optimize Triton code. Use fallback.
        return fallback_avg_pool2d_backward(
            grad_output,
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def compute_pool_size_without_padding(ph, pw):
        """
        This computes the scaling factor that we will divide an element
        by when `count_include_pad=False`
        """
        stride_h = ops.constant(stride[0], torch.int32)
        stride_w = ops.constant(stride[1], torch.int32)
        pad_h = ops.constant(padding[0], torch.int32)
        pad_w = ops.constant(padding[1], torch.int32)
        kernel_h = ops.constant(kernel_size[0], torch.int32)
        kernel_w = ops.constant(kernel_size[1], torch.int32)
        hstart = ops.sub(ops.mul(ph, stride_h), pad_h)
        wstart = ops.sub(ops.mul(pw, stride_w), pad_w)
        hend = ops.minimum(
            ops.add(hstart, kernel_h),
            ops.add(ops.index_expr(height, torch.int32), pad_h),
        )
        wend = ops.minimum(
            ops.add(wstart, kernel_w),
            ops.add(ops.index_expr(width, torch.int32), pad_w),
        )
        hstart = ops.maximum(hstart, ops.constant(0, torch.int32))
        wstart = ops.maximum(wstart, ops.constant(0, torch.int32))
        hend = ops.minimum(hend, ops.index_expr(height, torch.int32))
        wend = ops.minimum(wend, ops.index_expr(width, torch.int32))
        divide_factor = ops.mul(ops.sub(hend, hstart), ops.sub(wend, wstart))
        return divide_factor

    def fn(idx):
        *prefix, h, w = idx
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(
            FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32
        )
        pwstart = ops.index_expr(
            FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32
        )
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)

        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))

        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))

                if divisor_override is not None:
                    scale = divisor_override
                elif count_include_pad or not had_padding:
                    scale = kernel_size[0] * kernel_size[1]
                else:
                    scale = compute_pool_size_without_padding(ph, pw)

                part = ops.truediv(
                    grad_loader(
                        [
                            *prefix,
                            ops.indirect_indexing(
                                ops.minimum(
                                    ph, ops.sub(phend, ops.constant(1, torch.int32))
                                ),
                                pooled_height,
                                check=False,
                            ),
                            ops.indirect_indexing(
                                ops.minimum(
                                    pw, ops.sub(pwend, ops.constant(1, torch.int32))
                                ),
                                pooled_width,
                                check=False,
                            ),
                        ]
                    ),
                    scale,
                )

                mask = ops.and_(
                    ops.lt(ph, phend),
                    ops.lt(pw, pwend),
                )
                if gradient is None:
                    gradient = ops.where(mask, part, ops.constant(0.0, torch.float32))
                else:
                    gradient = ops.where(mask, ops.add(gradient, part), gradient)
        assert gradient is not None
        return gradient

    rv = Pointwise.create(
        device=grad_output.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    return rv


fallback_avg_pool3d_backward = fallback_handler(
    aten.avg_pool3d_backward.default, add_to_fallback_set=False
)


@register_lowering(aten.avg_pool3d_backward, type_promotion_kind=None)
def avg_pool3d_backward(
    grad_output,
    x,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override=None,
):
    """Lower avg_pool3d backward when the pooling window is small enough to inline."""
    assert divisor_override is None or divisor_override != 0, "divisor must be not zero"
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0, 0]

    assert isinstance(grad_output, TensorBox)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 3
    assert len(stride) == 3
    assert len(padding) == 3
    assert len(x.get_size()) in (4, 5)

    grad_output.realize_hint()

    *_batch, depth, height, width = x.get_size()

    _d_out, ceil_mode_d = pooling_size(
        depth, 0, kernel_size, stride, padding, ceil_mode
    )
    _h_out, ceil_mode_h = pooling_size(
        height, 1, kernel_size, stride, padding, ceil_mode
    )
    _w_out, ceil_mode_w = pooling_size(
        width, 2, kernel_size, stride, padding, ceil_mode
    )

    grad_loader = grad_output.make_loader()
    had_padding = any(padding) or ceil_mode_d or ceil_mode_h or ceil_mode_w

    *_, pooled_depth, pooled_height, pooled_width = grad_output.get_size()
    new_size = list(x.get_size())
    dtype = x.get_dtype()

    d_window_size, h_window_size, w_window_size = (
        max(
            max(d // stride[i] - max(0, (d - kernel_size[i]) // stride[i]), 1)
            for d in range(kernel_size[i] * 2)
        )
        for i in range(3)
    )

    window_size = d_window_size * h_window_size * w_window_size
    if window_size > 125:
        # Kernel size too big. Results in hard-to-optimize Triton code.
        return fallback_avg_pool3d_backward(
            grad_output,
            x,
            kernel_size,
            stride,
            padding,
            ceil_mode,
            count_include_pad,
            divisor_override,
        )

    def compute_pool_size_without_padding(pd, ph, pw):
        stride_d, stride_h, stride_w = (ops.constant(s, torch.int32) for s in stride)
        pad_d, pad_h, pad_w = (ops.constant(p, torch.int32) for p in padding)
        kernel_d, kernel_h, kernel_w = (
            ops.constant(k, torch.int32) for k in kernel_size
        )

        dstart, hstart, wstart = (
            ops.sub(ops.mul(p, s), pad)
            for p, s, pad in zip(
                [pd, ph, pw], [stride_d, stride_h, stride_w], [pad_d, pad_h, pad_w]
            )
        )
        dend, hend, wend = (
            ops.minimum(
                ops.add(start, k), ops.add(ops.index_expr(dim, torch.int32), pad)
            )
            for start, k, dim, pad in zip(
                [dstart, hstart, wstart],
                [kernel_d, kernel_h, kernel_w],
                [depth, height, width],
                [pad_d, pad_h, pad_w],
            )
        )
        dstart, hstart, wstart = (
            ops.maximum(start, ops.constant(0, torch.int32))
            for start in [dstart, hstart, wstart]
        )
        dend, hend, wend = (
            ops.minimum(end, ops.index_expr(dim, torch.int32))
            for end, dim in zip([dend, hend, wend], [depth, height, width])
        )
        divide_factor = ops.mul(
            ops.mul(ops.sub(dend, dstart), ops.sub(hend, hstart)), ops.sub(wend, wstart)
        )
        return divide_factor

    def fn(idx):
        *prefix, d, h, w = idx
        d, h, w = (v + pad for v, pad in zip([d, h, w], padding))

        pdstart, phstart, pwstart = (
            ops.index_expr(FloorDiv(v - k + s, s), torch.int32)
            for v, k, s in zip([d, h, w], kernel_size, stride)
        )

        pdend, phend, pwend = (
            ops.index_expr(FloorDiv(v, s) + 1, torch.int32)
            for v, s in zip([d, h, w], stride)
        )

        pdstart, phstart, pwstart = (
            ops.maximum(pstart, ops.constant(0, torch.int32))
            for pstart in [pdstart, phstart, pwstart]
        )
        pdend, phend, pwend = (
            ops.minimum(pend, ops.index_expr(pooled_dim, torch.int32))
            for pend, pooled_dim in zip(
                [pdend, phend, pwend], [pooled_depth, pooled_height, pooled_width]
            )
        )

        gradient = None
        # Iterate over the 3D region to accumulate gradients
        for pd_ in range(d_window_size):
            for ph_ in range(h_window_size):
                for pw_ in range(w_window_size):
                    pd, ph, pw = (
                        ops.add(pstart, ops.constant(p_, torch.int32))
                        for pstart, p_ in zip(
                            [pdstart, phstart, pwstart], [pd_, ph_, pw_]
                        )
                    )

                    if divisor_override is not None:
                        scale = divisor_override
                    elif count_include_pad or not had_padding:
                        scale = kernel_size[0] * kernel_size[1] * kernel_size[2]
                    else:
                        scale = compute_pool_size_without_padding(pd, ph, pw)

                    part = ops.truediv(
                        grad_loader(
                            [
                                *prefix,
                                ops.indirect_indexing(
                                    ops.minimum(
                                        pd, ops.sub(pdend, ops.constant(1, torch.int32))
                                    ),
                                    pooled_depth,
                                    check=False,
                                ),
                                ops.indirect_indexing(
                                    ops.minimum(
                                        ph, ops.sub(phend, ops.constant(1, torch.int32))
                                    ),
                                    pooled_height,
                                    check=False,
                                ),
                                ops.indirect_indexing(
                                    ops.minimum(
                                        pw, ops.sub(pwend, ops.constant(1, torch.int32))
                                    ),
                                    pooled_width,
                                    check=False,
                                ),
                            ]
                        ),
                        scale,
                    )

                    mask = ops.and_(
                        ops.and_(ops.lt(pd, pdend), ops.lt(ph, phend)),
                        ops.lt(pw, pwend),
                    )
                    if gradient is None:
                        gradient = ops.where(
                            mask, part, ops.constant(0.0, torch.float32)
                        )
                    else:
                        gradient = ops.where(mask, ops.add(gradient, part), gradient)
        assert gradient is not None
        return gradient

    rv = Pointwise.create(
        device=grad_output.get_device(),
        dtype=dtype,
        inner_fn=fn,
        ranges=new_size,
    )
    return rv


__all__ = [
    "_adaptive_avg_pool2d",
    "_adaptive_pooling_fn",
    "_adaptive_pooling_fn_with_idx",
    "_avg_poolnd",
    "_fractional_max_pool",
    "_fractional_pooling_offsets",
    "_low_memory_max_pool_offsets_to_indices",
    "_low_memory_max_pool_with_offsets",
    "_max_pool_with_indices",
    "_max_pool_with_offsets",
    "_pool_offsets_to_indices",
    "adaptive_max_pool2d",
    "avg_pool2d",
    "avg_pool2d_backward",
    "avg_pool3d",
    "avg_pool3d_backward",
    "compute_indices_adaptive_pooling",
    "fallback_adaptive_avg_pool2d",
    "fallback_adaptive_max_pool2d",
    "fallback_avg_pool2d_backward",
    "fallback_avg_pool3d_backward",
    "fallback_max_pool2d_with_indices_backward",
    "fallbacks_avg_poolnd",
    "fractional_max_pool2d",
    "fractional_max_pool3d",
    "max_pool2d_with_indices",
    "max_pool2d_with_indices_backward",
    "max_pool3d_with_indices",
    "max_pool_checks",
    "pad_adaptive_loader",
    "pooling_size",
    "should_fallback_max_pool_with_indices",
    "upsample_nearest2d_backward",
]
