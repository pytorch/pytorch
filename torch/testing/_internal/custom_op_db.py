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
    sample_inputs_numpy_sort,
    sample_inputs_numpy_take,
)
from torch import Tensor
from torch.types import Number
from typing import *  # noqa: F403
import torch._custom_ops as custom_ops

# Note: [custom op db]
#
# This is a collection of custom operator test cases written as OpInfos
# so they can easily be consumed by OpInfo-based tests to check if subsystems
# support them correctly.

def to_numpy(tensor):
    return tensor.cpu().numpy()

@custom_ops.custom_op('_torch_testing::numpy_cube')
def numpy_cube(x: Tensor) -> Tuple[Tensor, Tensor]:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_cube')
def numpy_cube_impl(x):
    x_np = to_numpy(x)
    dx = torch.tensor(3 * x_np ** 2, device=x.device)
    return torch.tensor(x_np ** 3, device=x.device), dx

@custom_ops.impl_abstract('_torch_testing::numpy_cube')
def numpy_cube_abstract(x):
    return x.clone(), x.clone()

@custom_ops.impl_save_for_backward('_torch_testing::numpy_cube')
def numpy_cube_save_for_backward(inputs, output):
    return (inputs.x, output[1])

@custom_ops.impl_backward('_torch_testing::numpy_cube')
def numpy_cube_backward(ctx, saved, grad_out, grad_dx):
    x, dx = saved
    grad_x = torch.ops._torch_testing.numpy_mul(grad_out, dx) + 6 * torch.ops._torch_testing.numpy_mul(grad_dx, x)
    return {'x': grad_x}

@custom_ops.custom_op('_torch_testing::numpy_mul')
def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_mul')
def numpy_mul_impl(x, y):
    return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)

@custom_ops.impl_abstract('_torch_testing::numpy_mul')
def numpy_mul_abstract(x, y):
    assert x.device == y.device
    return (x * y).contiguous()

@custom_ops.impl_save_for_backward('_torch_testing::numpy_mul')
def numpy_mul_save_for_backward(inputs, output):
    saved = {}
    saved['x_requires_grad'] = inputs.x.requires_grad
    saved['y_requires_grad'] = inputs.y.requires_grad
    # Optimization: only save what is necessary
    saved['y'] = inputs.y if inputs.x.requires_grad else None
    saved['x'] = inputs.x if inputs.y.requires_grad else None
    return saved

@custom_ops.impl_backward('_torch_testing::numpy_mul')
def numpy_mul_backward(ctx, saved, grad_out):
    grad_x = grad_out * saved['y'] if saved['x_requires_grad'] else None
    grad_y = grad_out * saved['x'] if saved['x_requires_grad'] else None
    return {'y': grad_y, 'x': grad_x}

@custom_ops.custom_op('_torch_testing::numpy_sort')
def numpy_sort(x: Tensor, dim: int) -> Tuple[Tensor, Tensor, Tensor]:
    raise NotImplementedError()

@custom_ops.impl("_torch_testing::numpy_sort")
def numpy_sort_impl(x, dim):
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

@custom_ops.impl_abstract('_torch_testing::numpy_sort')
def numpy_sort_abstract(x, dim):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.long), torch.empty_like(x, dtype=torch.long)

@custom_ops.impl_save_for_backward('_torch_testing::numpy_sort')
def numpy_sort_save_for_backward(inputs, output):
    out, ind, ind_inv = output
    return [inputs.dim, ind, ind_inv]

@custom_ops.impl_backward('_torch_testing::numpy_sort', output_differentiability=[True, False, False])
def numpy_sort_backward(ctx, saved, grad_out, grad_ind, grad_ind_inv):
    dim, ind, ind_inv = saved
    return {'x': torch.ops._torch_testing.numpy_take(grad_out, ind_inv, ind, dim)}

@custom_ops.custom_op('_torch_testing::numpy_take')
def numpy_take(x: Tensor, ind: Tensor, ind_inv: Tensor, dim: int) -> Tensor:
    raise NotImplementedError()

@custom_ops.impl("_torch_testing::numpy_take")
def numpy_take_impl(x, ind, ind_inv, dim):
    device = x.device
    x = to_numpy(x)
    ind = to_numpy(ind)
    return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

@custom_ops.impl_abstract('_torch_testing::numpy_take')
def numpy_take_abstract(x, ind, ind_inv, dim):
    assert x.device == ind.device
    assert x.device == ind_inv.device
    assert ind.dtype == torch.long
    assert ind_inv.dtype == torch.long
    return torch.empty_like(x)

@custom_ops.impl_save_for_backward('_torch_testing::numpy_take')
def numpy_take_save_for_backward(inputs, output):
    return {
        'dim': inputs.dim,
        'ind': inputs.ind,
        'ind_inv': inputs.ind_inv,
    }

@custom_ops.impl_backward('_torch_testing::numpy_take')
def numpy_take_backward(ctx, saved, grad_out):
    return {
        'x': torch.ops._torch_testing.numpy_take(grad_out, saved['ind_inv'], saved['ind'], saved['dim']),
        'ind': None,
        'ind_inv': None,
    }

@custom_ops.custom_op('_torch_testing::numpy_nonzero')
def numpy_nonzero(x: Tensor) -> Tensor:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_nonzero')
def numpy_nonzero_impl(x):
    x_np = to_numpy(x)
    res = np.stack(np.nonzero(x_np), axis=1)
    if res.shape[0] <= 1:
        raise RuntimeError("not supported")
    return torch.tensor(res, device=x.device)

@custom_ops.impl_abstract('_torch_testing::numpy_nonzero')
def numpy_nonzero_abstract(x):
    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    shape = [x.dim(), i0]
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

@custom_ops.custom_op('_torch_testing::numpy_view_copy')
def numpy_view_copy(x: Tensor, shape: Sequence[int]) -> Tensor:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_view_copy')
def numpy_view_copy_impl(x, shape) -> Tensor:
    return torch.tensor(np.copy(to_numpy(x).reshape(shape)), device=x.device)

@custom_ops.impl_abstract('_torch_testing::numpy_view_copy')
def numpy_view_copy_abstract(x, shape) -> Tensor:
    return x.clone().view(shape).clone()

@custom_ops.impl_save_for_backward('_torch_testing::numpy_view_copy')
def numpy_view_copy_save_for_backward(inputs, output) -> Tensor:
    return inputs.x.shape

@custom_ops.impl_backward('_torch_testing::numpy_view_copy')
def numpy_view_copy_backward(ctx, x_shape, grad_out) -> Tensor:
    return {'x': torch.ops._torch_testing.numpy_view_copy(grad_out, x_shape)}

def sample_inputs_numpy_view_copy(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    result = make_arg(2, 3, 4, low=0.9, high=2)
    yield SampleInput(result, args=([2, 12],))

@custom_ops.custom_op('_torch_testing::numpy_cat')
def numpy_cat(xs: Sequence[Tensor], dim: int) -> Tensor:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_cat')
def numpy_cat_impl(xs, dim):
    assert len(xs) > 0
    assert all(x.device == xs[0].device for x in xs)
    assert all(x.dtype == xs[0].dtype for x in xs)
    np_xs = [to_numpy(x) for x in xs]
    np_out = np.concatenate(np_xs, axis=dim)
    return torch.tensor(np_out, device=xs[0].device)

@custom_ops.impl_abstract('_torch_testing::numpy_cat')
def numpy_cat_abstract(xs, dim):
    assert len(xs) > 0
    assert all(x.device == xs[0].device for x in xs)
    assert all(x.dtype == xs[0].dtype for x in xs)
    return torch.cat(xs, dim=dim)

@custom_ops.impl_save_for_backward('_torch_testing::numpy_cat')
def numpy_cat_save_for_backward(inputs, output):
    dim_sizes = [x.shape[inputs.dim] for x in inputs.xs]
    return dim_sizes, inputs.dim

@custom_ops.impl_backward('_torch_testing::numpy_cat')
def numpy_cat_backward(ctx, saved, grad_out):
    dim_sizes, dim = saved
    splits = list(np.cumsum(dim_sizes)[:-1])
    grad_xs = torch.ops._torch_testing.numpy_split_copy(grad_out, splits, dim)
    return {'xs': grad_xs}

def sample_inputs_numpy_cat(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    r0 = make_arg(2, 3, 4, low=0.9, high=2)
    r1 = make_arg(4, 3, 4, low=0.9, high=2)
    r2 = make_arg(5, 3, 4, low=0.9, high=2)
    yield SampleInput([r0, r1, r2], args=(0,))

@custom_ops.custom_op('_torch_testing::numpy_split_copy')
def numpy_split_copy(x: Tensor, sections: Sequence[int], dim: int) -> List[Tensor]:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_split_copy')
def numpy_split_copy_impl(x, splits, dim):
    x_np = to_numpy(x)
    arrs = np.split(x_np, splits, axis=dim)
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs]

@custom_ops.impl_abstract('_torch_testing::numpy_split_copy')
def numpy_split_copy_abstract(x, splits, dim):
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)]

@custom_ops.impl_save_for_backward('_torch_testing::numpy_split_copy')
def numpy_split_copy_save_for_backward(inputs, output):
    return inputs.dim

@custom_ops.impl_backward('_torch_testing::numpy_split_copy')
def numpy_split_copy_backward(ctx, saved, grad_out):
    dim = saved
    return {'x': torch.ops._torch_testing.numpy_cat(grad_out, dim=dim)}

def sample_inputs_numpy_split_copy(opinfo, device, dtype, requires_grad, **kwargs):
    make_arg = functools.partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    x = make_arg(2, 9, low=0.9, high=2)
    yield SampleInput(x, args=([1, 3, 6], 1))

@custom_ops.custom_op('_torch_testing::numpy_split_copy_with_int')
def numpy_split_copy_with_int(x: Tensor, sections: Sequence[int], dim: int) -> Tuple[List[Tensor], int]:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_split_copy_with_int')
def numpy_split_copy_with_int_impl(x, splits, dim):
    x_np = to_numpy(x)
    arrs = np.split(x_np, splits, axis=dim)
    return [torch.tensor(arr, device=x.device, dtype=x.dtype) for arr in arrs], len(splits)

@custom_ops.impl_abstract('_torch_testing::numpy_split_copy_with_int')
def numpy_split_copy_with_int_abstract(x, splits, dim):
    return [xi.clone() for xi in torch.tensor_split(x, splits, dim)], len(splits)

@custom_ops.impl_save_for_backward(
    '_torch_testing::numpy_split_copy_with_int')
def numpy_split_copy_with_int_save_for_backward(inputs, output):
    return inputs.dim

@custom_ops.impl_backward(
    '_torch_testing::numpy_split_copy_with_int',
    output_differentiability=[True, False])
def numpy_split_copy_with_int_backward(ctx, saved, grad_out, _):
    dim = saved
    return {'x': torch.ops._torch_testing.numpy_cat(grad_out, dim=dim)}

@custom_ops.custom_op('_torch_testing::numpy_nms')
def numpy_nms(boxes: Tensor, scores: Tensor, iou_threshold: Number) -> Tensor:
    raise NotImplementedError()

@custom_ops.impl('_torch_testing::numpy_nms')
def numpy_nms_impl(boxes, scores, iou_threshold):
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

    result = np.stack(keep)
    result = torch.tensor(np.stack(keep), device=device)
    # Needed for data-dependent condition :(
    assert result.size(0) >= 2
    return result

@custom_ops.impl_abstract('_torch_testing::numpy_nms')
def numpy_nms_abstract(boxes, scores, iou_threshold):
    assert boxes.device == scores.device
    N = boxes.shape[0]
    assert boxes.shape == (N, 4)
    assert scores.shape == (N,)

    ctx = torch._custom_op.impl.get_ctx()
    i0 = ctx.create_unbacked_symint()
    result = boxes.new_empty([i0, 4])
    return result

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
        op=torch.ops._torch_testing.numpy_cube,
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulCustomOp',
        op=torch.ops._torch_testing.numpy_mul,
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpySortCustomOp',
        op=torch.ops._torch_testing.numpy_sort,
        sample_inputs_func=sample_inputs_numpy_sort,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyTakeCustomOp',
        op=torch.ops._torch_testing.numpy_take,
        sample_inputs_func=sample_inputs_numpy_take,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyNonzeroCustomOp',
        op=torch.ops._torch_testing.numpy_nonzero,
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
