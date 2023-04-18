import torch
import functools
from torch.testing import make_tensor
from torch.testing._internal.opinfo.core import (
    OpInfo,
    SampleInput,
)
from torch.testing._internal.common_dtype import all_types_and
import numpy as np
from torch._custom_op import custom_op
from torch.testing._internal.autograd_function_db import (
    sample_inputs_numpy_cube,
    sample_inputs_numpy_mul,
    sample_inputs_numpy_sort,
    sample_inputs_numpy_take,
)

# Note: [custom op db]
#
# This is a collection of custom operator test cases written as OpInfos
# so they can easily be consumed by OpInfo-based tests to check if subsystems
# support them correctly.

def to_numpy(tensor):
    return tensor.cpu().numpy()

@custom_op('(Tensor x) -> (Tensor, Tensor)', ns='_torch_testing')
def numpy_cube(x):
    ...

@numpy_cube.impl('cpu')
@numpy_cube.impl('cuda')
def numpy_cube_impl(x):
    x_np = to_numpy(x)
    dx = torch.tensor(3 * x_np ** 2, device=x.device)
    return torch.tensor(x_np ** 3, device=x.device), dx

@numpy_cube.impl_fake()
def numpy_cube_fake(ctx, x):
    return x.clone(), x.clone()

@custom_op('(Tensor x, Tensor y) -> Tensor', ns='_torch_testing')
def numpy_mul(x, y):
    ...

@numpy_mul.impl('cpu')
@numpy_mul.impl('cuda')
def numpy_mul_impl(x, y):
    return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)

@numpy_mul.impl_fake()
def numpy_mul_fake(ctx, x, y):
    return (x * y).contiguous()

@custom_op('(Tensor x, int dim) -> (Tensor, Tensor, Tensor)', ns='_torch_testing')
def numpy_sort(x, dim):
    ...

@numpy_sort.impl('cpu')
@numpy_sort.impl('cuda')
def numpy_sort_impl(x, dim):
    device = x.device
    x = to_numpy(x)
    ind = np.argsort(x, axis=dim)
    ind_inv = np.argsort(ind, axis=dim)
    result = np.take_along_axis(x, ind, axis=dim)
    return (
        torch.tensor(x, device=device),
        torch.tensor(ind, device=device),
        torch.tensor(ind_inv, device=device),
    )

@numpy_sort.impl_fake()
def numpy_sort_fake(ctx, x, dim):
    return torch.empty_like(x), torch.empty_like(x, dtype=torch.long), torch.empty_like(x, dtype=torch.long)

@custom_op('(Tensor x, Tensor ind, Tensor ind_inv, int dim) -> Tensor', ns='_torch_testing')
def numpy_take(x, ind, ind_inv, dim):
    ...

@numpy_take.impl('cpu')
@numpy_take.impl('cuda')
def numpy_take_impl(x, ind, ind_inv, dim):
    device = x.device
    x = to_numpy(x)
    ind = to_numpy(ind)
    return torch.tensor(np.take_along_axis(x, ind, dim), device=device)

@numpy_take.impl_fake()
def numpy_take_fake(ctx, x, ind, ind_inv, dim):
    return torch.empty_like(x)

@custom_op('(Tensor x) -> Tensor', ns='_torch_testing')
def numpy_nonzero(x):
    ...

@numpy_nonzero.impl(['cpu', 'cuda'])
def numpy_nonzero_impl(x):
    x_np = to_numpy(x)
    res = np.stack(np.nonzero(x_np), axis=1)
    if res.shape[0] <= 1:
        raise RuntimeError("not supported")
    return torch.tensor(res, device=x.device)

@numpy_nonzero.impl_fake()
def numpy_nonzero_fake(ctx, x):
    i0 = ctx.new_data_dependent_symint()
    ctx.constrain_range(i0, min=2)
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


@custom_op('(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor', ns='_torch_testing')
def numpy_nms(boxes, scores, iou_threshold):
    ...

@numpy_nms.impl(['cpu', 'cuda'])
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

@numpy_nms.impl_fake()
def numpy_nms_fake(ctx, boxes, scores, iou_threshold):
    assert boxes.device == scores.device
    N = boxes.shape[0]
    assert boxes.shape == (N, 4)
    assert scores.shape == (N,)

    i0 = ctx.new_data_dependent_symint()
    ctx.constrain_range(i0, min=2)
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

# CustomOp isn't deepcopy-able, so we wrap in a function that is.
def wrap_for_opinfo(op):
    def inner(*args, **kwargs):
        return op(*args, **kwargs)
    return inner

custom_op_db = [
    OpInfo(
        'NumpyCubeCustomOp',
        op=wrap_for_opinfo(numpy_cube),
        sample_inputs_func=sample_inputs_numpy_cube,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyMulCustomOp',
        op=wrap_for_opinfo(numpy_mul),
        sample_inputs_func=sample_inputs_numpy_mul,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpySortCustomOp',
        op=wrap_for_opinfo(numpy_sort),
        sample_inputs_func=sample_inputs_numpy_sort,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyTakeCustomOp',
        op=wrap_for_opinfo(numpy_take),
        sample_inputs_func=sample_inputs_numpy_take,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyNonzeroCustomOp',
        op=wrap_for_opinfo(numpy_nonzero),
        sample_inputs_func=sample_inputs_numpy_nonzero,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
    OpInfo(
        'NumpyNMSCustomOp',
        op=wrap_for_opinfo(numpy_nms),
        sample_inputs_func=sample_inputs_numpy_nms,
        dtypes=all_types_and(torch.bool, torch.half),
        supports_out=False,
    ),
]
