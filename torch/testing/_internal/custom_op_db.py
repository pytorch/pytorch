# mypy: allow-untyped-defs
import torch
import functools
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
from torch.testing._internal.autograd_function_db import (
    sample_inputs_numpy_cube,
    sample_inputs_numpy_mul,
    sample_inputs_numpy_mul_scalar,
    sample_inputs_numpy_sort,
    sample_inputs_numpy_take,
)
from torch import Tensor
from torch.types import Number
from typing import *  # noqa: F403

# Note: [custom op db]
#
# This is a collection of custom operator test cases written as OpInfos
# so they can easily be consumed by OpInfo-based tests to check if subsystems
# support them correctly.

def to_numpy(tensor):
    return tensor.cpu().numpy()

@torch.library.custom_op("_torch_testing::numpy_cube", mutates_args=())
def numpy_cube(x: Tensor) -> tuple[Tensor, Tensor]:
    x_np = to_numpy(x)
    dx = torch.tensor(3 * x_np ** 2, device=x.device)
    return torch.tensor(x_np ** 3, device=x.device), dx

@numpy_cube.register_fake
def _(x):
    return x.clone(), x.clone()

def numpy_cube_setup_context(ctx, inputs, output):
    x, = inputs
    _cube, dx = output
    ctx.save_for_backward(x, dx)

def numpy_cube_backward(ctx, grad_out, grad_dx):
    x, dx = ctx.saved_tensors
    grad_x = numpy_mul(grad_out, dx) + 6 * numpy_mul(grad_dx, x)
    return grad_x

numpy_cube.register_autograd(numpy_cube_backward, setup_context=numpy_cube_setup_context)

def numpy_cube_vmap(info, in_dims, x):
    result = numpy_cube(x)
    return result, (in_dims[0], in_dims[0])

numpy_cube.register_vmap(numpy_cube_vmap)

@torch.library.custom_op("_torch_testing::numpy_mul", mutates_args=())
def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
    return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)

@numpy_mul.register_fake
def _(x, y):
    assert x.device == y.device
    return (x * y).contiguous()

def numpy_mul_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)

def numpy_mul_backward(ctx, grad_out):
    x, y = ctx.saved_tensors
    grad_x = grad_out * y if ctx.needs_input_grad[0] else None
    grad_y = grad_out * x if ctx.needs_input_grad[1] else None
    return grad_x, grad_y

numpy_mul.register_autograd(numpy_mul_backward, setup_context=numpy_mul_setup_context)

def numpy_mul_vmap(info, in_dims, x, y):
    x_bdim, y_bdim = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
    result = x * y
    result = result.movedim(-1, 0)
    return result, 0

numpy_mul.register_vmap(numpy_mul_vmap)

@torch.library.custom_op("_torch_testing::numpy_mul_scalar", mutates_args=())
def numpy_mul_scalar(x: Tensor, *, scalar: float) -> Tensor:
    return torch.tensor(to_numpy(x) * scalar, device=x.device)

@numpy_mul_scalar.register_fake
def _(x, *, scalar):
    return (x * scalar).contiguous()

def numpy_mul_scalar_setup_context(ctx, inputs, keyword_only_inputs, output):
    ctx.scalar = keyword_only_inputs["scalar"]

def numpy_mul_scalar_backward(ctx, grad_out):
    grad_x = grad_out * ctx.scalar
    return grad_x

numpy_mul_scalar.register_autograd(numpy_mul_scalar_backward, setup_context=numpy_mul_scalar_setup_context)

def numpy_mul_scalar_vmap(info, in_dims, x, *, scalar):
    x_bdim, = in_dims
    x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
    result = x * scalar
    result = result.movedim(-1, 0)
    return result, 0

numpy_mul_scalar.register_vmap(numpy_mul_scalar_vmap)

@torch.library.custom_op("_torch_testing::numpy_sort", mutates_args=())
def numpy_sort(x: Tensor, dim: int) -> tuple[Tensor, Tensor, Tensor]:
    device = x.device
    x = to_numpy(x)
    ind = np.argsort(x, axis=dim)
    ind_inv = np.argsort(ind, axis=dim)
    result = np.take_along_axis(x, ind, axis=dim)
    return (
        torch.tensor(result, device=device),
        torch.tensor(ind, device=device),
        torch.tensor(ind_inv, device=device),
    )

@numpy_sort.register_fake
def _(x, dim):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.long), torch.empty_like(x, dtype=torch.long)

def numpy_sort_setup_context(ctx, inputs, output):
    _out, ind, ind_inv = output
    ctx.dim = inputs[1]
    ctx.save_for_backward(ind, ind_inv)
    ctx.mark_non_differentiable(ind, ind_inv)

def numpy_sort_backward(ctx, grad_out, grad_ind, grad_ind_inv):
    ind, ind_inv = ctx.saved_tensors
    return numpy_take(grad_out, ind_inv, ind, ctx.dim), None

numpy_sort.register_autograd(numpy_sort_backward, setup_context=numpy_sort_setup_context)

def numpy_sort_vmap(info, in_dims, x, dim):
    x_bdim, _ = in_dims
    x = x.movedim(x_bdim, 0)
    dim = dim if dim >= 0 else dim + x.dim() - 1
    result = numpy_sort(x, dim + 1)
    return result, (0, 0, 0)

numpy_sort.register_vmap(numpy_sort_vmap)

@torch.library.custom_op("_torch_testing::numpy_take", mutates_args=())
def numpy_take(x: Tensor, ind: Tensor, ind_inv: Tensor, dim: int) -> Tensor:
    device = x.device
    x = to_numpy(x)
    ind = to_numpy(ind)
    return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

@numpy_take.register_fake
def _(x, ind, ind_inv, dim):
    assert x.device == ind.device
    assert x.device == ind_inv.device
    assert ind.dtype == torch.long
    assert ind_inv.dtype == torch.long
    return torch.empty_like(x)

def numpy_take_setup_context(ctx, inputs, output):
    _x, ind, ind_inv, dim = inputs
    ctx.dim = dim
    ctx.save_for_backward(ind, ind_inv)

def numpy_take_backward(ctx, grad_out):
    ind, ind_inv = ctx.saved_tensors
    grad_x = numpy_take(grad_out, ind_inv, ind, ctx.dim)
    return grad_x, None, None, None

numpy_take.register_autograd(numpy_take_backward, setup_context=numpy_take_setup_context)

def numpy_take_vmap(info, in_dims, x, ind, ind_inv, dim):
    x_bdim, ind_bdim, ind_inv_bdim, _ = in_dims

    # wrap dim
    logical_dim = x.dim() if x_bdim is None else x_bdim - 1
    dim = dim if dim >= 0 else dim + logical_dim

    def expand_bdim(x, x_bdim):
        if x_bdim is None:
            return x.expand(info.batch_size, *x.shape)
        return x.movedim(x_bdim, 0)

    x = expand_bdim(x, x_bdim)
    ind = expand_bdim(ind, ind_bdim)
    ind_inv = expand_bdim(ind_inv, ind_inv_bdim)

    return numpy_take(x, ind, ind_inv, dim + 1), 0

numpy_take.register_vmap(numpy_take_vmap)

@torch.library.custom_op("_torch_testing::numpy_nonzero", mutates_args=())
def numpy_nonzero(x: Tensor) -> Tensor:
    x_np = to_numpy(x)
    res = np.stack(np.nonzero(x_np), axis=1)
    if res.shape[0] <= 1:
        raise RuntimeError("not supported")
    return torch.tensor(res, device=x.device)

@numpy_nonzero.register_fake
def _(x):
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    shape = [i0, x.dim()]
    result = x.new_empty(shape, dtype=torch.long)
    return result

def sample_inputs_numpy_nonzero(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shape = 10
    result = make_arg(shape, low=0.9, high=2)
    mask = make_tensor(shape, low=0, high=2, device=device, dtype=torch.long)
    with torch.no_grad():
        result *= mask

    yield SampleInput(result, args=())

def numpy_nonzero_vmap(info, in_dims, x):
    raise NotImplementedError("Operator is data-dependent and cannot be vmapped.")

numpy_nonzero.register_vmap(numpy_nonzero_vmap)

@torch.library.custom_op("_torch_testing::numpy_view_copy", mutates_args=())
def numpy_view_copy(x: Tensor, shape: Sequence[int]) -> Tensor:
    return torch.tensor(np.copy(to_numpy(x).reshape(shape)), device=x.device)

@numpy_view_copy.register_fake
def _(x, shape) -> Tensor:
    return x.clone().view(shape).clone()

def numpy_view_copy_setup_context(ctx, inputs, output) -> None:
    ctx.x_shape = inputs[0].shape

def numpy_view_copy_backward(ctx, grad_out):
    return torch.ops._torch_testing.numpy_view_copy(grad_out, ctx.x_shape), None

numpy_view_copy.register_autograd(numpy_view_copy_backward, setup_context=numpy_view_copy_setup_context)

def numpy_view_copy_vmap(info, in_dims, x, shape):
    x_bdim, _ = in_dims
    x = x.movedim(x_bdim, 0)
    x_shape = x.shape[0]
    batch_shape = (x_shape, *shape)
    result = numpy_view_copy(x, batch_shape)
    return result, 0

numpy_view_copy.register_vmap(numpy_view_copy_vmap)

def sample_inputs_numpy_view_copy(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    result = make_arg(2, 3, 4, low=0.9, high=2)
    yield SampleInput(result, args=([2, 12],))

@torch.library.custom_op('_torch_testing::numpy_cat', mutates_args=())
def numpy_cat(xs: Sequence[Tensor], dim: int) -> Tensor:
    assert len(xs) > 0
    assert all(x.device == xs[0].device for x in xs)
    assert all(x.dtype == xs[0].dtype for x in xs)
    np_xs = [to_numpy(x) for x in xs]
    np_out = np.concatenate(np_xs, axis=dim)
    return torch.tensor(np_out, device=xs[0].device)

@numpy_cat.register_fake
def _(xs, dim):
    assert len(xs) > 0
    assert all(x.device == xs[0].device for x in xs)
    assert all(x.dtype == xs[0].dtype for x in xs)
    return torch.cat(xs, dim=dim)

def numpy_cat_setup_context(ctx, inputs, output):
    xs, dim = inputs
    ctx.dim_sizes = [x.shape[dim] for x in xs]
    ctx.dim = dim

def numpy_cat_backward(ctx, grad_out):
    dim_sizes = ctx.dim_sizes
    dim = ctx.dim

    splits = list(np.cumsum(dim_sizes)[:-1])
    grad_xs = torch.ops._torch_testing.numpy_split_copy(grad_out, splits, dim)
    return grad_xs, None

numpy_cat.register_autograd(numpy_cat_backward, setup_context=numpy_cat_setup_context)

def numpy_cat_vmap(info, in_dims, x, dim):
    x_bdim, = in_dims
    result = numpy_cat(x, dim)
    return result, x_bdim

numpy_cat.register_vmap(numpy_cat_vmap)

def sample_inputs_numpy_cat(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    r0 = make_arg(2, 3, 4, low=0.9, high=2)
    r1 = make_arg(4, 3, 4, low=0.9, high=2)
    r2 = make_arg(5, 3, 4, low=0.9, high=2)
    yield SampleInput([r0, r1, r2], args=(0,))

@torch.library.custom_op('_torch_testing::numpy_split_copy', mutates_args=())
def numpy_split_copy(x: Tensor, splits: Sequence[int], dim: int) -> List[Tensor]:
    x_np = to_numpy(x)
    arrs = np.split(x_np, splits, axis=dim)
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs]

@numpy_split_copy.register_fake
def _(x, splits, dim):
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)]

def numpy_split_copy_setup_context(ctx, inputs, output):
    _, _, dim = inputs
    ctx.dim = dim

def numpy_split_copy_backward(ctx, grad_out):
    result = torch.ops._torch_testing.numpy_cat(grad_out, dim=ctx.dim)
    return result, None, None

numpy_split_copy.register_autograd(numpy_split_copy_backward, setup_context=numpy_split_copy_setup_context)

def numpy_split_copy_vmap(info, in_dims, x, splits, dim):
    x_bdim, _ , _ = in_dims
    x = x.movedim(x_bdim, 0)
    result = numpy_split_copy(x, splits, dim + 1)
    return result, 0

numpy_split_copy.register_vmap(numpy_split_copy_vmap)

def sample_inputs_numpy_split_copy(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    x = make_arg(2, 9, low=0.9, high=2)
    yield SampleInput(x, args=([1, 3, 6], 1))

@torch.library.custom_op('_torch_testing::numpy_split_copy_with_int', mutates_args=())
def numpy_split_copy_with_int(x: Tensor, splits: Sequence[int], dim: int) -> tuple[List[Tensor], int]:
    x_np = to_numpy(x)
    arrs = np.split(x_np, splits, axis=dim)
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs], len(splits)

@numpy_split_copy_with_int.register_fake
def _(x, splits, dim):
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)], len(splits)

def numpy_split_copy_with_int_setup_context(ctx, inputs, output):
    _, _, dim = inputs
    ctx.dim = dim

def numpy_split_copy_with_int_backward(ctx, grad_out, _):
    return torch.ops._torch_testing.numpy_cat(grad_out, dim=ctx.dim), None, None

numpy_split_copy_with_int.register_autograd(
    numpy_split_copy_with_int_backward,
    setup_context=numpy_split_copy_with_int_setup_context)

def numpy_split_copy_with_int_vmap(info, in_dims, x, splits, dim):
    x_bdim, _ , _ = in_dims
    x = x.movedim(x_bdim, 0)
    result, len_split = numpy_split_copy_with_int(x, splits, dim + 1)
    return (result, len_split), ([0 for _ in range(len(result))], None)

numpy_split_copy_with_int.register_vmap(numpy_split_copy_with_int_vmap)

@torch.library.custom_op("_torch_testing::numpy_nms", mutates_args=())
def numpy_nms(boxes: Tensor, scores: Tensor, iou_threshold: Number) -> Tensor:
    # Adapted from Ross Girshick's fast-rcnn implementation at
    # https://github.com/rbgirshick/fast-rcnn/blob/master/lib/utils/nms.py
    assert boxes.device == scores.device
    device = boxes.device

    boxes = to_numpy(boxes)
    scores = to_numpy(scores)

    N = boxes.shape[0]
    assert boxes.shape == (N, 4)
    assert scores.shape == (N,)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    result = torch.tensor(np.stack(keep), device=device)
    # Needed for data-dependent condition :(
    assert result.size(0) >= 2
    return result

@numpy_nms.register_fake
def _(boxes, scores, iou_threshold):
    assert boxes.device == scores.device
    N = boxes.shape[0]
    assert boxes.shape == (N, 4)
    assert scores.shape == (N,)

    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    result = boxes.new_empty([i0], dtype=torch.int64)
    return result

def numpy_nms_vmap(info, in_dims, boxes, scores, iou_threshold):
    raise NotImplementedError("Operator is data-dependent and cannot be vmapped.")

numpy_nms.register_vmap(numpy_nms_vmap)

def sample_inputs_numpy_nms(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype)
    N = 64
    xs = make_arg([N], low=0, high=28)
    dx = make_arg([N], low=0, high=4)
    ys = make_arg([N], low=0, high=28)
    dy = make_arg([N], low=0, high=4)
    boxes = torch.stack([xs, ys, xs + dx, ys + dy], dim=1).requires_grad_(requires_grad)
    scores = make_arg([N], low=0, high=1, requires_grad=requires_grad)
    iou_threshold = make_arg([], low=0, high=1).item()

    yield SampleInput(boxes, args=(scores, iou_threshold))

custom_op_db = [
    OpInfo(
        'NumpyCubeCustomOp',
        op=numpy_cube._opoverload,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulCustomOp',
        op=numpy_mul._opoverload,
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulScalarCustomOp',
        op=numpy_mul_scalar._opoverload,
        sample_inputs_func=sample_inputs_numpy_mul_scalar,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpySortCustomOp',
        op=numpy_sort._opoverload,
        sample_inputs_func=sample_inputs_numpy_sort,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyTakeCustomOp',
        op=numpy_take._opoverload,
        sample_inputs_func=sample_inputs_numpy_take,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyNonzeroCustomOp',
        op=numpy_nonzero._opoverload,
        sample_inputs_func=sample_inputs_numpy_nonzero,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpyNMSCustomOp',
        op=torch.ops._torch_testing.numpy_nms,
        sample_inputs_func=sample_inputs_numpy_nms,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpyViewCopyCustomOp',
        op=torch.ops._torch_testing.numpy_view_copy,
        sample_inputs_func=sample_inputs_numpy_view_copy,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=True,
        supports_out=False,
    ),
    OpInfo(
        'NumpyCatCustomOp',
        op=torch.ops._torch_testing.numpy_cat,
        sample_inputs_func=sample_inputs_numpy_cat,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpySplitCopyCustomOp',
        op=torch.ops._torch_testing.numpy_split_copy,
        sample_inputs_func=sample_inputs_numpy_split_copy,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_autograd=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_out=False,
    ),
    OpInfo(
        'NumpySplitCopyWithIntCustomOp',
        op=torch.ops._torch_testing.numpy_split_copy_with_int,
        sample_inputs_func=sample_inputs_numpy_split_copy,
        dtypes=all_types_and(torch.bool, torch.half),
        gradcheck_wrapper=lambda op, *args, **kwargs: op(*args, **kwargs)[0],
        supports_autograd=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_out=False,
    ),
]


# ==============================================================
# some mechanical test cases
# ==============================================================

lib = torch.library.Library("_torch_testing", "FRAGMENT")  # noqa: TOR901

lib.define("source0(Tensor x) -> Tensor")

@torch.library.register_fake("_torch_testing::source0", lib=lib)
def _(x):
    return x.clone()

lib.define("source1(Tensor x) -> Tensor")

def source1_fake(x):
    return x.clone()

torch.library.register_fake("_torch_testing::source1", source1_fake, lib=lib)

lib.define("source2(Tensor x) -> Tensor")

@torch.library.register_fake("_torch_testing::source2", lib=lib)
def _(x):
    return x.clone()

lib.define("source3(Tensor x) -> Tensor")

def source3_fake(x):
    return x.clone()

torch.library.register_fake("_torch_testing::source3", source3_fake, lib=lib)


@torch.library.custom_op("_torch_testing::source4", mutates_args=())
def source4(x: Tensor) -> Tensor:
    return x.clone()

@source4.register_fake
def _(x):
    return x.clone()

@torch.library.custom_op("_torch_testing::source5", mutates_args=())
def source5(x: Tensor) -> Tensor:
    return x.clone()

def source5_fake(x):
    return x.clone()

source5.register_fake(source5_fake)
