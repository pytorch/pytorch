R"=====("  ### DO NOT REMOVE THIS STRING!!!
# this file is included in torch/csrc/jit/runtime/symbolic_shape_registry.cpp
# at compile time and turned into a "raw" string
# there's a matching one at the bottom

from typing import List, Any, Optional, Tuple

from numpy import number

import torch

####    SHAPE COMPUTE FUNCTIONS START   ###


def broadcast(a: List[int], b: List[int]):
    dimsA = len(a)
    dimsB = len(b)
    ndim = max(dimsA, dimsB)
    expandedSizes: List[int] = []

    for i in range(ndim):
        offset = ndim - 1 - i
        dimA = dimsA - 1 - offset
        dimB = dimsB - 1 - offset
        sizeA = a[dimA] if (dimA >= 0) else 1
        sizeB = b[dimB] if (dimB >= 0) else 1

        if sizeA != sizeB and sizeA != 1 and sizeB != 1:
            # TODO: only assertion error is bound in C++ compilation right now
            raise AssertionError(
                "The size of tensor a {} must match the size of tensor b ("
                "{}) at non-singleton dimension {}".format(sizeA, sizeB, i)
            )

        expandedSizes.append(sizeB if sizeA == 1 else sizeA)

    return expandedSizes


def adaptive_avg_pool2d(self: List[int], out: List[int]):
    assert len(out) == 2
    assert len(self) == 3 or len(self) == 4
    for i in range(1, len(self)):
        assert self[i] != 0

    shape: List[int] = []
    for i in range(0, len(self) - 2):
        shape.append(self[i])
    for elem in out:
        shape.append(elem)
    return shape


def _copy(self: List[int]):
    out: List[int] = []
    for elem in self:
        out.append(elem)
    return out


def unary(self: List[int]):
    return _copy(self)


def broadcast_inplace(a: List[int], b: List[int]):
    dimsA = len(a)
    dimsB = len(b)
    if dimsB > dimsA:
        raise AssertionError(
            "The dims of tensor b ({}) must be less than or equal to"
            "the dims of tensor a ({}) ".format(dimsB, dimsA)
        )
    for dimA in range(dimsA):
        dimB = dimsB - dimsA + dimA
        sizeA = a[dimA]
        sizeB = b[dimB] if (dimB >= 0) else 1
        if dimB >= 0:
            assert sizeA == sizeB
        else:
            if sizeA != sizeB and sizeB != 1:
                # TODO: only assertion error is bound in C++ compilation right now
                raise AssertionError(
                    "The size of tensor a {} must match the size of tensor b ("
                    "{}) at non-singleton dimension {}".format(sizeA, sizeB, dimA)
                )
    return _copy(a)


def expand(self: List[int], sizes: List[int]):
    assert len(sizes) >= len(self)
    ndim = len(sizes)
    tensor_dim = len(self)
    if ndim == 0:
        return _copy(sizes)
    out: List[int] = []
    for i in range(ndim):
        offset = ndim - 1 - i
        dim = tensor_dim - 1 - offset
        size = self[dim] if dim >= 0 else 1
        targetSize = sizes[i]
        if targetSize == -1:
            assert dim >= 0
            targetSize = size
        if size != targetSize:
            assert size == 1
            size = targetSize
        out.append(size)
    return out


def expand_one_unused(self: List[int], sizes: List[int], inp0: Any):
    return expand(self, sizes)


def infer_size_impl(shape: List[int], numel: int) -> List[int]:
    newsize = 1
    infer_dim: Optional[int] = None
    for dim in range(len(shape)):
        if shape[dim] == -1:
            if infer_dim is not None:
                raise AssertionError("only one dimension can be inferred")
            infer_dim = dim
        elif shape[dim] >= 0:
            newsize *= shape[dim]
        else:
            raise AssertionError("invalid shape dimensions")
    if not (
        numel == newsize
        or (infer_dim is not None and newsize > 0 and numel % newsize == 0)
    ):
        raise AssertionError("invalid shape")
    out = _copy(shape)
    if infer_dim is not None:
        out[infer_dim] = numel // newsize
    return out


def numel(sizes: List[int]):
    numel = 1
    for elem in sizes:
        numel *= elem
    return numel


def view(self: List[int], sizes: List[int]):
    return infer_size_impl(sizes, numel(self))


def view_one_unused(self: List[int], sizes: List[int], *, implicit: bool = False):
    return view(self, sizes)


def mean_dim(self: List[int], dims: List[int], keep_dim: bool, dt: Any):
    out: List[int] = []
    for idx in range(len(self)):
        is_mean_dim: bool = False
        for reduce_dim in dims:
            if idx == maybe_wrap_dim(reduce_dim, len(self)):
                is_mean_dim = True
        if is_mean_dim:
            if keep_dim:
                out.append(1)
        else:
            out.append(self[idx])
    return out


# note: python already rounds down towards negative infinity on integer division, special arithmetic not needed
def div_rtn(x: int, y: int):
    return x // y


def pooling_output_shape_pad_lr(
    inputSize: int,
    kernelSize: int,
    pad_l: int,
    pad_r: int,
    stride: int,
    dilation: int,
    ceil_mode: bool,
):
    outputSize = (
        div_rtn(
            inputSize
            + pad_l
            + pad_r
            - dilation * (kernelSize - 1)
            - 1
            + (stride - 1 if ceil_mode else 0),
            stride,
        )
        + 1
    )
    if ceil_mode:
        if (outputSize - 1) * stride >= inputSize + pad_l:
            outputSize = outputSize - 1
    return outputSize


def pooling_output_shape(
    inputSize: int,
    kernelSize: int,
    pad_l: int,
    stride: int,
    dilation: int,
    ceil_mode: bool,
):
    assert stride != 0, "stride should not be zeero"
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad_l, pad_l, stride, dilation, ceil_mode
    )


def pool2d_shape_check(
    input: List[int],
    kH: int,
    kW: int,
    dH: int,
    dW: int,
    padH: int,
    padW: int,
    dilationH: int,
    dilationW: int,
    nInputPlane: int,
    inputHeight: int,
    inputWidth: int,
    outputHeight: int,
    outputWidth: int,
):
    ndim = len(input)
    nOutputPlane = nInputPlane

    assert kW > 0 and kH > 0
    assert dW > 0 and dH > 0
    assert dilationH > 0 and dilationW > 0

    valid_dims = input[1] != 0 and input[2] != 0
    assert (
        ndim == 3
        and input[0] != 0
        and valid_dims
        or (ndim == 4 and valid_dims and input[3] != 0)
    )

    assert kW // 2 >= padW and kH // 2 >= padH
    assert outputWidth >= 1 and outputHeight >= 1


def max_pool2d(
    input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool,
):
    assert (
        len(kernel_size) == 1 or len(kernel_size) == 2
    ), "max_pool2d: kernel_size must either be a single int, or a tuple of two ints"
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]

    assert (
        len(stride) == 0 or len(stride) == 1 or len(stride) == 2
    ), "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints"
    dH = kH if len(stride) == 0 else stride[0]
    dW = kW if len(stride) == 0 else dH if len(stride) == 1 else stride[1]

    assert (
        len(padding) == 1 or len(padding) == 2
    ), "max_pool2d: padding must be either be a single int, or a tuple of two ints"
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    assert (
        len(dilation) == 1 or len(dilation) == 2
    ), "max_pool2d: dilation must be either a single int, or a tuple of two ints"
    dilationH = dilation[0]
    dilationW = dilationH if len(dilation) == 1 else dilation[1]

    assert len(input) == 3 or len(input) == 4

    nbatch = input[-4] if len(input) == 4 else 1
    nInputPlane = input[-3]
    inputHeight = input[-2]
    inputWidth = input[-1]

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, dilationH, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, dilationW, ceil_mode)

    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        dilationH,
        dilationW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
    )

    if len(input) == 3:
        return [nInputPlane, outputHeight, outputWidth]
    else:
        return [nbatch, nInputPlane, outputHeight, outputWidth]


def max_pool2d_with_indices(
    input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    ceil_mode: bool,
):
    out = max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    return (out, out)


def upsample_nearest2d(
    input: List[int],
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]],
):
    out: List[int] = []
    out.append(input[0])
    out.append(input[1])
    if output_size is not None:
        assert (
            scale_factors is None
        ), "Must specify exactly one of output_size and scale_factors"
        assert len(output_size) == 2
        out.append(output_size[0])
        out.append(output_size[1])
        return out

    if scale_factors is not None:
        assert (
            output_size is None
        ), "Must specify exactly one of output_size and scale_factors"
        assert len(scale_factors) == 2
        out.append(int(input[2] * scale_factors[0]))
        out.append(int(input[3] * scale_factors[1]))
        return out
    assert 0, "Either output_size or scale_factors must be presented"


def my_upsample_bilinear2d(
    input: List[int], output_size: List[int], scale_factors: Optional[List[float]]
):
    out: List[int] = []
    out.append(input[0])
    out.append(input[1])
    if output_size is not None:
        assert (
            scale_factors is None
        ), "Must specify exactly one of output_size and scale_factors"
        assert len(output_size) == 2
        out.append(output_size[0])
        out.append(output_size[1])
        return out

    if scale_factors is not None:
        assert (
            output_size is None
        ), "Must specify exactly one of output_size and scale_factors"
        assert len(scale_factors) == 2
        out.append(int(input[2] * scale_factors[0]))
        out.append(int(input[3] * scale_factors[1]))
        return out
    assert 0, "Either output_size or scale_factors must be presented"


def mm(self: List[int], mat2: List[int]):
    assert len(self) == 2, "self must be a matrix"
    assert len(mat2) == 2, "mat2 must be a matrix"

    assert self[1] == mat2[0]
    return [self[0], mat2[1]]


def dot(self: List[int], tensor: List[int]):
    assert len(self) == 1 and len(tensor) == 1
    assert self[0] == tensor[0]
    out: List[int] = []
    return out


def mv(self: List[int], vec: List[int]):
    assert len(self) == 2 and len(vec) == 1
    assert self[1] == vec[0]
    # TODO: return self
    return [self[0]]


def unsqueeze(li: List[int], dim: int):
    dim = maybe_wrap_dim(dim, len(li) + 1)
    out = _copy(li)
    out.insert(dim, 1)
    return out


def squeeze_nodim(li: List[int]):
    out: List[int] = []
    for i in range(len(li)):
        if li[i] != 1:
            out.append(li[i])
    return out


def squeeze(li: List[int], dim: int):
    out: List[int] = []
    wrapped_dim = maybe_wrap_dim(dim, len(li))
    for i in range(len(li)):
        if i == wrapped_dim:
            if li[i] != 1:
                out.append(li[i])
        else:
            out.append(li[i])
    return out


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


def avg_pool2d(
    input: List[int],
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: Optional[int],
):
    assert len(kernel_size) == 1 or len(kernel_size) == 2
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]

    assert len(stride) >= 0 or len(stride) <= 2
    dH = kH if len(stride) == 0 else stride[0]
    dW = kW if len(stride) == 0 else (dH if len(stride) == 1 else stride[1])
    assert len(padding) == 1 or len(padding) == 2
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    assert divisor_override is None or divisor_override != 0
    nbatch = input[0] if len(input) == 4 else 1
    nInputPlane = input[-3]
    inputHeight = input[-2]
    inputWidth = input[-1]
    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)
    pool2d_shape_check(
        input,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        1,
        1,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
    )
    if len(input) == 3:
        return [nInputPlane, outputHeight, outputWidth]
    else:
        return [nbatch, nInputPlane, outputHeight, outputWidth]


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
        if len(size) != 0 and size != [0] and out_dim is not None:
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
    assert not check_non_negative(padding), "negative padding"
    assert not check_non_negative(stride), "negative stride"

    assert weight_dim == k, "weight_dim != k"
    assert weight_sizes[0] >= groups, "weight sizes[0] not >= groups"
    assert (weight_sizes[0] % groups) == 0, "weight sizes not divisable by groups"
    # only handling not transposed
    assert input[1] == weight_sizes[1] * groups, "weight sizes[1] != groups"
    assert bias is None or (len(bias) == 1 and bias[0] == weight_sizes[0])

    for i in range(k, k):
        assert (input[i] + 2 * padding[i - 2]) >= (
            dilation[i - 2] * (weight_sizes[i] - 1) + 1
        ), "kernel size can't be bigger than input size"

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
        # most common case of just enough padding to fit kernel and stride evenly
        # paves the input
        # if 2 * padding[d - 2] == kernel - 1 and input_size[d] % stride[d - 2] == 0:
        #     append_value = int(input_size[d] / stride[d - 2])
        # elif 2 * padding[d - 2] == kernel - 1:
        #     append_value = input_size[d] // stride[d - 2]
        # elif (input_size[d] + (2 * padding[d - 2]) - (kernel - 1)) % stride[d - 2] == 0:
        #     append_value = int(
        #         (input_size[d] + (2 * padding[d - 2]) - (kernel - 1)) / stride[d - 2]
        #     )
        # else:
        #     append_value = (
        #         input_size[d] + (2 * padding[d - 2]) - (kernel - 1)
        #     ) // stride[d - 2]
        append_value = (input_size[d] + (2 * padding[d - 2]) - (kernel - 1)) // stride[
            d - 2
        ]
        output_size.append(append_value)

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
    for i in range(start_dim, end_dim - start_dim + 1):
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
    weight, bias, stride, padding, dilation, groups = unchecked_cast(
        Tuple[List[int], Optional[List[int]], List[int], List[int], List[int], int],
        ops.quantized.conv2d_unpack_sizes(conv2dOpContext),
    )
    return conv2d(input, weight, bias, stride, padding, dilation, groups)


####    SHAPE COMPUTE FUNCTIONS END   ###
### DO NOT REMOVE THIS STRING!!!
")====="
