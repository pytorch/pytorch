from typing import List, Any, Optional, TypeVar, Union, Dict, Callable
import math
number = Union[int, float]
# flake8: noqa

import torch


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

def broadcast_three(a: List[int], b: List[int], c: List[int]):
    return broadcast(broadcast(a, b), c)

def broadcast_one_three(a: List[int], b: Any, c: List[int]):
    return broadcast(a, c)

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

def max_dim(self: List[int], dim: int, keep_dim: bool):
    out = mean_dim(self, [dim], keep_dim, None)
    return out, out

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
    if len(stride) == 0:
        dW = kW
    elif len(stride) == 1:
        dW = dH
    else:
        dW = stride[1]

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
    elif start_val > self[dim]:
        start_val = self[dim]
    if end_val < start_val:
        end_val = start_val
    elif end_val >= self[dim]:
        end_val = self[dim]
    slice_len = end_val - start_val
    out = _copy(self)
    out[dim] = (slice_len + step - 1) // step
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
    weight: Optional[List[int]],
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
    return [int(math.ceil(end))]


def arange_start(
    start: number, end: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    assert end >= 0
    assert end >= start
    return [int(math.ceil(end - start))]


def arange_start_step(
    start: number, end: number, step: number, inp0: Any, inp1: Any, inp2: Any, inp3: Any
):
    assert step != 0
    if step < 0:
        assert start >= end
    else:
        assert end >= start
    return [int(math.ceil((end - start) / step))]


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

shape_compute_graph_mapping : Dict[str, torch._C.ScriptFunction] = {}
script_func_map: Dict[Callable, torch._C.ScriptFunction] = {}

def add_shape_compute_mapping(operator_schema: str, func: Callable):
    global shape_compute_graph_mapping
    global shape_compute_graph_set

    if func in script_func_map:
        shape_compute_graph_mapping[operator_schema] = script_func_map[func]
        return

    scripted_func = torch.jit.script(func)

    torch._C._jit_pass_inline(scripted_func.graph)

    for _ in range(2):
        torch._C._jit_pass_peephole(scripted_func.graph)
        torch._C._jit_pass_constant_propagation(scripted_func.graph)

    script_func_map[func] = scripted_func
    shape_compute_graph_mapping[operator_schema] = scripted_func

add_shape_compute_mapping("aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)", unary)
add_shape_compute_mapping("aten::rsub.Tensor(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", unary)
add_shape_compute_mapping("aten::dropout(Tensor input, float p, bool train) -> Tensor", unary)
add_shape_compute_mapping("aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor", adaptive_avg_pool2d)
add_shape_compute_mapping("prim::NumToTensor.Scalar(Scalar a) -> Tensor", zero_dim_tensor)
add_shape_compute_mapping("prim::NumToTensor.bool(bool a) -> Tensor", zero_dim_tensor)
add_shape_compute_mapping("aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", unary)
add_shape_compute_mapping("aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))", unary)
add_shape_compute_mapping("aten::arange(Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", arange_end)
add_shape_compute_mapping("aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", arange_start)
add_shape_compute_mapping("aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", arange_start_step)
add_shape_compute_mapping("aten::squeeze(Tensor(a) self) -> Tensor(a)", squeeze_nodim)
add_shape_compute_mapping("aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)", squeeze)
add_shape_compute_mapping("aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", unsqueeze)
add_shape_compute_mapping("aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)", slice)
add_shape_compute_mapping("aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)", select)
add_shape_compute_mapping("aten::index_select(Tensor self, int dim, Tensor index) -> Tensor", index_select)
add_shape_compute_mapping("aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, "
                          "float eps=1e-05, bool cudnn_enable=True) -> Tensor", unary)
add_shape_compute_mapping("aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor", unary)
add_shape_compute_mapping("aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor", unary)
add_shape_compute_mapping("aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)", unary)
add_shape_compute_mapping("aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor", embedding)
add_shape_compute_mapping("aten::mm(Tensor self, Tensor mat2) -> Tensor", mm)
add_shape_compute_mapping("aten::dot(Tensor self, Tensor tensor) -> Tensor", dot)
add_shape_compute_mapping("aten::mv(Tensor self, Tensor vec) -> Tensor", mv)
add_shape_compute_mapping("aten::matmul(Tensor self, Tensor other) -> Tensor", matmul)
add_shape_compute_mapping("aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", linear)
add_shape_compute_mapping("aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor", max_pool2d)
add_shape_compute_mapping("aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)", max_pool2d_with_indices)
add_shape_compute_mapping("aten::t(Tensor(a) self) -> Tensor(a)", t)
add_shape_compute_mapping("aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)", transpose)
add_shape_compute_mapping("aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor", conv1d)
add_shape_compute_mapping("aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor", conv2d)
add_shape_compute_mapping("aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor", batch_norm)
add_shape_compute_mapping("aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor", conv3d)
add_shape_compute_mapping("aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)", flatten)
add_shape_compute_mapping("aten::cat(Tensor[] tensors, int dim=0) -> Tensor", cat)
add_shape_compute_mapping("aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)", permute)
add_shape_compute_mapping("aten::view(Tensor(a) self, int[] size) -> Tensor(a)", view)
add_shape_compute_mapping("aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)", expand)
add_shape_compute_mapping("aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)", expand_one_unused)
add_shape_compute_mapping("aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", mean_dim)
add_shape_compute_mapping("aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", mean_dim)
add_shape_compute_mapping("aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)", max_dim)
add_shape_compute_mapping("aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor", zero_dim_tensor)
add_shape_compute_mapping("aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor", zero_dim_tensor)
add_shape_compute_mapping("aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", addmm)
add_shape_compute_mapping("aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)", upsample_nearest2d)
add_shape_compute_mapping("aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor", unary)
add_shape_compute_mapping("aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor", unary)
add_shape_compute_mapping("aten::dequantize(Tensor self) -> Tensor", unary)
add_shape_compute_mapping("quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc", broadcast)

# TODO: migrate over all of symbolic_shape_registry_util.cpp
# These are duplicated here so that the functions will be serialiazed
add_shape_compute_mapping("aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor", broadcast_three)
add_shape_compute_mapping("aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor", broadcast_one_three)
add_shape_compute_mapping("aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)", broadcast_inplace)

# quantized_conv_prepack TODO
