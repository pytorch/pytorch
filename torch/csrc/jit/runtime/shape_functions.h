R"=====("  ### DO NOT REMOVE THIS STRING!!!
# this file is included in torch/csrc/jit/runtime/symbolic_shape_registry.cpp
# at compile time and turned into a "raw" string
# there's a matching one at the bottom
# mypy: ignore-errors
# flake8: noqa

from typing import List, Any, Optional, Tuple, TypeVar, Union
number = TypeVar('number', bound=Union[int, float])

import torch

import inspect
import warnings
from importlib.machinery import SourceFileLoader

import os
shape_function_fp = f"{os.path.dirname(os.path.realpath(__file__))}/shape_functions_1.h"
try:
    _shapes_1 = SourceFileLoader("shape_functions", shape_function_fp).load_module() # type: ignore
    globals().update(inspect.getmembers(_shapes_1))
except Exception as e:
    warnings.warn(f"Couldn't load shape functions from {shape_function_fp}")



####    SHAPE COMPUTE FUNCTIONS START   ###


def index_select(self: List[int], dim: int, index: List[int]):
    dim = maybe_wrap_dim(dim, len(self))
    numel = multiply_integers(index)
    assert len(index) <= 1
    assert dim == 0 or dim < len(self)
    result_size: List[int] = []
    for i in range(len(self)):
        if dim == i:
            result_size.append(numel)
        else:
            result_size.append(self[i])
    return result_size


def embedding(
    weight: List[int],
    indices: List[int],
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
):
    assert len(weight) == 2
    if len(indices) == 1:
        return index_select(weight, 0, indices)
    size = _copy(indices)
    size.append(weight[1])
    return size


def max_int():
    return 9223372036854775807


def slice(
    self: List[int], dim: int, start: Optional[int], end: Optional[int], step: int
):
    ndim = len(self)
    assert ndim != 0
    dim = maybe_wrap_dim(dim, ndim)
    start_val = start if start is not None else 0
    end_val = end if end is not None else max_int()
    assert step > 0
    if start_val == max_int():
        start_val = 0
    if start_val < 0:
        start_val += self[dim]
    if end_val < 0:
        end_val += self[dim]
    if start_val < 0:
        start_val = 0
    elif start_val >= self[dim]:
        start_val = self[dim]
    if end_val < start_val:
        end_val = start_val
    elif end_val >= self[dim]:
        end_val = self[dim]
    len = end_val - start_val
    out = _copy(self)
    out[dim] = (len + step - 1) // step
    return out


def check_cat_no_zero_dim(tensors: List[List[int]]):
    for tensor in tensors:
        assert len(tensor) > 0

def legacy_cat_wrap_dim(dim: int, tensor_sizes: List[List[int]]):
    out_dim: Optional[int] = None
    for size in tensor_sizes:
        if not (len(size) == 1 and size[0] == 0):
            if out_dim is None:
                out_dim = maybe_wrap_dim(dim, len(size))
    if out_dim is None:
        out_dim = dim
    return out_dim


def should_skip(tensor: List[int]):
    return numel(tensor) == 0 and len(tensor) == 1


def check_cat_shape_except_dim(
    first: List[int], second: List[int], dimension: int, index: int
):
    first_dims = len(first)
    second_dims = len(second)
    assert first_dims == second_dims, "Tensors must have same number of dimensions"
    for dim in range(0, first_dims):
        if dim != dimension:
            assert (
                first[dim] == second[dim]
            ), "Sizes of tensors must match except in dimension"


def cat(tensors: List[List[int]], dim: int):
    check_cat_no_zero_dim(tensors)
    dim = legacy_cat_wrap_dim(dim, tensors)
    assert len(tensors) > 0
    not_skipped_tensor: Optional[List[int]] = None
    for tensor in tensors:
        if not should_skip(tensor):
            not_skipped_tensor = tensor
    if not_skipped_tensor is None:
        return [0]

    cat_dim_size = 0

    for i in range(len(tensors)):
        tensor = tensors[i]
        if not should_skip(tensor):
            check_cat_shape_except_dim(not_skipped_tensor, tensor, dim, i)
            cat_dim_size = cat_dim_size + tensor[dim]

    result_size = _copy(not_skipped_tensor)
    result_size[dim] = cat_dim_size
    return result_size


def select(self: List[int], dim: int, index: int):
    ndim = len(self)
    assert ndim != 0
    dim = maybe_wrap_dim(dim, ndim)
    size = self[dim]
    assert not (index < -size or index >= size)
    if index < 0:
        index += size
    out: List[int] = []
    for i in range(ndim):
        if i != dim:
            out.append(self[i])
    return out


def matmul(tensor1: List[int], tensor2: List[int]):
    dim_tensor1 = len(tensor1)
    dim_tensor2 = len(tensor2)
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        return dot(tensor1, tensor2)
    elif dim_tensor1 == 2 and dim_tensor2 == 1:
        return mv(tensor1, tensor2)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        return squeeze(mm(unsqueeze(tensor1, 0), tensor2), 0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        return mm(tensor1, tensor2)
    elif dim_tensor1 >= 1 and dim_tensor2 >= 1:
        # We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
        # we track m1 vs m2 separately even though they must match for nicer error messages
        n = tensor1[-2] if dim_tensor1 > 1 else 1
        m1 = tensor1[-1]
        batch_tensor1: List[int] = []
        # TODO: handling of slice
        for i in range(dim_tensor1 - 2):
            batch_tensor1.append(tensor1[i])
        m2 = tensor2[-1] if dim_tensor2 > 1 else 1
        p = tensor2[-1]
        batch_tensor2: List[int] = []
        # TODO: handling of slice
        for i in range(dim_tensor2 - 2):
            batch_tensor2.append(tensor2[i])

        # expand the batch portion (i.e. cut off matrix dimensions and expand rest)
        expand_batch_portion = broadcast(batch_tensor1, batch_tensor2)

        # todo: copy ?
        output_shape = expand_batch_portion
        if dim_tensor1 > 1:
            output_shape.append(n)

        if dim_tensor2 > 1:
            output_shape.append(p)

        return output_shape
    else:
        assert False, "both  arguments to matmul need to be at least 1D"


def t(self: List[int]):
    assert len(self) <= 2
    self_len = len(self)
    if self_len == 0:
        out: List[int] = []
        return out
    elif self_len == 1:
        return [self[0]]
    else:
        return [self[1], self[0]]


def transpose(self: List[int], dim0: int, dim1: int):
    ndims = len(self)
    dim0 = maybe_wrap_dim(dim0, ndims)
    dim1 = maybe_wrap_dim(dim1, ndims)
    if dim0 == dim1:
        return _copy(self)
    out: List[int] = []
    for i in range(ndims):
        if i == dim0:
            out.append(self[dim1])
        elif i == dim1:
            out.append(self[dim0])
        else:
            out.append(self[i])
    return out


def linear(input: List[int], weight: List[int], bias: Optional[List[int]]):
    out = matmul(input, t(weight))
    if bias is not None:
        assert broadcast(bias, out) == out
    return out


def addmm(self: List[int], mat1: List[int], mat2: List[int], beta: Any, alpha: Any):
    return broadcast(self, mm(mat1, mat2))


def check_non_negative(array: List[int]) -> bool:
    # TODO: look into rewriting with early return and getting loop unrolling to fire
    non_negative = False
    for val in array:
        if val < 0:
            non_negative = True
    return non_negative


def check_shape_forward(
    input: List[int],
    weight_sizes: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    k = len(input)
    weight_dim = len(weight_sizes)

    # TODO: assertions could be expanded with the error messages
    assert not check_non_negative(padding)
    assert not check_non_negative(stride)

    assert weight_dim == k
    assert weight_sizes[0] >= groups
    assert (weight_sizes[0] % groups) == 0
    # only handling not transposed
    assert input[1] == weight_sizes[1] * groups
    assert bias is None or (len(bias) == 1 and bias[0] == weight_sizes[0])

    for i in range(2, k):
        assert (input[i] + 2 * padding[i - 2]) >= (
            dilation[i - 2] * (weight_sizes[i] - 1) + 1
        )

    # this is not handling transposed convolution yet


def conv_output_size(
    input_size: List[int],
    weight_size: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    check_shape_forward(
        input_size, weight_size, bias, stride, padding, dilation, groups
    )

    has_dilation = len(dilation) > 0
    dim = len(input_size)
    output_size: List[int] = []
    input_batch_size_dim = 0
    weight_output_channels_dim = 0
    output_size.append(input_size[input_batch_size_dim])
    output_size.append(weight_size[weight_output_channels_dim])

    for d in range(2, dim):
        dilation_ = dilation[d - 2] if has_dilation else 1
        kernel = dilation_ * (weight_size[d] - 1) + 1
        output_size.append(
            (input_size[d] + (2 * padding[d - 2]) - kernel) // stride[d - 2] + 1
        )
    return output_size


def conv1d(
    input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    assert len(weight) == 3
    assert len(input) == 3
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


def conv2d(
    input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    assert len(weight) == 4
    assert len(input) == 4
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


def batch_norm(
    input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    running_mean: Optional[List[int]],
    running_var: Optional[List[int]],
    training: bool,
    momentum: float,
    eps: float,
    cudnn_enabled: bool,
):
    out: List[int] = []
    for elem in input:
        out.append(elem)
    return out


def conv3d(
    input: List[int],
    weight: List[int],
    bias: Optional[List[int]],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
):
    assert len(weight) == 5
    assert len(input) == 5
    return conv_output_size(input, weight, bias, stride, padding, dilation, groups)


def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max)
    if dim < 0:
        dim += dim_post_expr
    return dim


def zero_dim_tensor(input: Any):
    out: List[int] = []
    return out


def multiply_integers(li: List[int]):
    out = 1
    for elem in li:
        out = out * elem
    return out


def arange_end(end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any):
    assert end >= 0
    return [int(torch.ceil(end))]


def arange_start(
    start: number, end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    assert end >= 0
    assert end >= start
    return [int(torch.ceil(end - start))]


def arange_start_step(
    start: number, end: number, step: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    assert step != 0
    if step < 0:
        assert start >= end
    else:
        assert end >= start
    return [int(torch.ceil((end - start) / step))]


def permute(input: List[int], dims: List[int]):
    assert len(input) == len(dims)
    ndim = len(dims)
    seen_dims: List[int] = []
    newSizes: List[int] = []
    for i in range(ndim):
        dim = maybe_wrap_dim(dims[i], ndim)
        seen_dims.append(dim)
        newSizes.append(input[dim])
    for i in range(1, ndim):
        for j in range(i):
            assert seen_dims[i] != seen_dims[j]
    return newSizes


def flatten(input: List[int], start_dim: int, end_dim: int):
    start_dim = maybe_wrap_dim(start_dim, len(input))
    end_dim = maybe_wrap_dim(end_dim, len(input))
    assert start_dim <= end_dim
    if len(input) == 0:
        return [1]
    if start_dim == end_dim:
        # TODO: return self
        out: List[int] = []
        for elem in input:
            out.append(elem)
        return out
    slice_numel = 1
    for i in range(start_dim, end_dim + 1):
        slice_numel *= input[i]
    # TODO: use slicing when slice optimization has landed
    # slice_numel = multiply_integers(input[start_dim:end_dim - start_dim + 1])
    shape: List[int] = []
    for i in range(start_dim):
        shape.append(input[i])
    shape.append(slice_numel)
    for i in range(end_dim + 1, len(input)):
        shape.append(input[i])
    return shape


def quantized_prepacked_conv2d(input: List[int], conv2dOpContext: Any):
    assert isinstance(
        conv2dOpContext, __torch__.torch.classes.quantized.Conv2dPackedParamsBase
    )
    (weight, bias, stride, padding, dilation, groups) = unchecked_cast(
        Tuple[List[int], Optional[List[int]], List[int], List[int], List[int], int],
        ops.quantized.conv2d_unpack_sizes(conv2dOpContext),
    )
    return conv2d(input, weight, bias, stride, padding, dilation, groups)


####    SHAPE COMPUTE FUNCTIONS END   ###
### DO NOT REMOVE THIS STRING!!!
")====="
