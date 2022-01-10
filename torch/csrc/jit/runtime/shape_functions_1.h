R"=====("  ### DO NOT REMOVE THIS STRING!!!
# this file is included in torch/csrc/jit/runtime/symbolic_shape_registry.cpp
# at compile time and turned into a "raw" string
# there's a matching one at the bottom
# mypy: ignore-errors
# flake8: noqa

from typing import List, Any, Optional, Tuple, TypeVar, Union
number = TypeVar('number', bound=Union[int, float])

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


####    SHAPE COMPUTE FUNCTIONS END   ###
### DO NOT REMOVE THIS STRING!!! #
")====="
