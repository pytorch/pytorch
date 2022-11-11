import math
from typing import List, Optional, Union

import torch
import torch._prims_common as utils
from torch import Tensor
from torch._decomp import _add_op_to_registry, global_decomposition_table, meta_table
from torch._ops import OpOverload
from torch._prims_common import (
    check,
    corresponding_complex_dtype,
    corresponding_real_dtype,
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    FloatLike,
    IntLike,
)

from torch._prims_common.wrappers import out_wrapper
from torch._refs import _broadcast_shapes

from torch._subclasses.fake_tensor import check_no_bool_index_tensors
from torch.utils._pytree import tree_map


aten = torch.ops.aten

_meta_lib_dont_use_me_use_register_meta = torch.library.Library("aten", "IMPL", "Meta")


def register_meta(op):
    def wrapper(fn):
        def register(op):
            _add_op_to_registry(meta_table, op, fn)

        tree_map(register, op)
        return fn

    return wrapper


def toRealValueType(dtype):
    from_complex = {
        torch.complex32: torch.half,
        torch.cfloat: torch.float,
        torch.cdouble: torch.double,
    }
    return from_complex.get(dtype, dtype)


@register_meta(aten._fft_c2c.default)
def meta_fft_c2c(self, dim, normalization, forward):
    assert self.dtype.is_complex
    return self.new_empty(self.size())


@register_meta(aten._fft_r2c.default)
def meta_fft_r2c(self, dim, normalization, onesided):
    assert self.dtype.is_floating_point
    output_sizes = list(self.size())

    if onesided:
        last_dim = dim[-1]
        last_dim_halfsize = (output_sizes[last_dim] // 2) + 1
        output_sizes[last_dim] = last_dim_halfsize

    return self.new_empty(
        output_sizes, dtype=utils.corresponding_complex_dtype(self.dtype)
    )


@register_meta(aten.randperm.generator_out)
def meta_randperm(n, *, generator=None, out):
    assert out.ndim == 1 and out.size(0) == n
    return out


@register_meta(aten.randint.default)
def meta_randint(
    high, size, *, dtype=torch.long, layout=None, device=None, pin_memory=None
):
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.randint.low)
def meta_randint_low(
    low, high, size, *, dtype=torch.long, layout=None, device=None, pin_memory=None
):
    return torch.empty(
        size, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta([aten._fft_c2r.default, aten._fft_c2r.out])
@out_wrapper()
def meta_fft_c2r(self, dim, normalization, lastdim):
    assert self.dtype.is_complex
    output_sizes = list(self.size())
    output_sizes[dim[-1]] = lastdim
    return self.new_empty(output_sizes, dtype=toRealValueType(self.dtype))


@register_meta(aten.copy_.default)
def meta_copy_(self, src, non_blocking=False):
    return self


def inferUnsqueezeGeometry(tensor, dim):
    result_sizes = list(tensor.size())
    result_strides = list(tensor.stride())
    new_stride = 1 if dim >= tensor.dim() else result_sizes[dim] * result_strides[dim]
    result_sizes.insert(dim, 1)
    result_strides.insert(dim, new_stride)
    return result_sizes, result_strides


@register_meta(aten.unsqueeze_.default)
def meta_unsqueeze_(self, dim):
    dim = maybe_wrap_dim(dim, self.dim() + 1)
    g_sizes, g_strides = inferUnsqueezeGeometry(self, dim)
    self.as_strided_(g_sizes, g_strides)
    return self


# Implementations below are taken from https://github.com/albanD/subclass_zoo/blob/main/python_meta_tensor.py
@register_meta(aten.index_select.default)
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)


@register_meta(aten.index_select.out)
def meta_index_select_out(self, dim, index, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.index_select(self, dim, index))


@register_meta([aten.max.default, aten.max.unary_out])
@out_wrapper()
def meta_max(self):
    return self.new_empty(())


@register_meta(aten.max.dim)
def meta_max_dim(self, dim, keepdim=False):
    dim = utils.reduction_dims(self.shape, (dim,))
    output_shape = _compute_reduction_shape(self, dim, keepdim)
    return (
        self.new_empty(output_shape),
        self.new_empty(output_shape, dtype=torch.long),
    )


@register_meta([aten.min.default])
def meta_min(self):
    return self.new_empty(())


@register_meta(aten.angle.default)
def meta_angle(self):
    if self.is_complex():
        result_dtype = corresponding_real_dtype(self.dtype)
    else:
        _, result_dtype = elementwise_dtypes(
            self, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )
    return torch.empty_like(self, dtype=result_dtype)


@register_meta(aten.angle.out)
def meta_angle_out(self, out):
    torch._resize_output_(out, self.size(), self.device)
    return out.copy_(torch.angle(self))


def squareCheckInputs(self, f_name):
    assert (
        self.dim() >= 2
    ), f"{f_name}: The input tensor must have at least 2 dimensions."
    assert self.size(-1) == self.size(
        -2
    ), f"{f_name}: A must be batches of square matrices, but they are {self.size(-2)} by {self.size(-1)} matrices"


def checkUplo(uplo: str):
    uplo_uppercase = uplo.upper()
    assert (
        len(uplo) == 1 and uplo_uppercase == "U" or uplo_uppercase == "L"
    ), f"Expected UPLO argument to be 'L' or 'U', but got {uplo}"


# @register_meta(aten.linalg_eigh.default)
def meta_linalg_eigh(self, uplo="L"):
    squareCheckInputs(self, "linalg_eigh")
    checkUplo(uplo)
    real_dtype = toRealValueType(self.dtype)
    assert self.dim() >= 2
    values = self.new_empty(self.shape, dtype=real_dtype)
    values.transpose_(-2, -1)
    vectors = self.new_empty(self.shape[:-1])
    return (values, vectors)


# From aten/src/ATen/native/ReflectionPad.cpp
@register_meta(
    [aten.reflection_pad2d_backward.default, aten.replication_pad2d_backward.default]
)
def meta_pad2d_backward(grad_output, self, padding):
    dim_w = 2
    dim_h = 1
    dim_plane = 0
    nbatch = 1

    self_shape = self.shape
    if self.dim() == 4:
        nbatch = self_shape[0]
        dim_w += 1
        dim_h += 1
        dim_plane += 1

    pad_l = padding[0]
    pad_r = padding[1]
    pad_t = padding[2]
    pad_b = padding[3]

    nplane = self_shape[dim_plane]
    input_h = self_shape[dim_h]
    input_w = self_shape[dim_w]
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    check(
        output_w == grad_output.shape[dim_w],
        lambda: f"gradOutput width unexpected. Expected: {output_w}, Got: {grad_output.shape[dim_w]}",
    )
    check(
        output_h == grad_output.shape[dim_h],
        lambda: f"gradOutput height unexpected. Expected: {output_h}, Got: {grad_output.shape[dim_h]}",
    )
    return self.new_empty(self.shape)


@register_meta(aten.reflection_pad2d.default)
def meta_pad2d(self, padding):
    valid_dims = self.size(1) != 0 and self.size(2) != 0
    check(
        (self.ndim == 3 and valid_dims)
        or (self.ndim == 4 and valid_dims and self.size(3) != 0),
        lambda: f"3D or 4D (batch mode) tensor expected for input, but got: {self}",
    )
    if self.ndim == 4:
        nbatch, nplane, input_h, input_w = self.shape
    else:
        nbatch = 1
        nplane, input_h, input_w = self.shape

    pad_l, pad_r, pad_t, pad_b = padding

    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r

    if self.ndim == 3:
        return self.new_empty((nplane, output_h, output_w))
    else:
        return self.new_empty((nbatch, nplane, output_h, output_w))


@register_meta([aten.bernoulli.default, aten.bernoulli.out])
@out_wrapper()
def meta_bernoulli(self, *, generator=None):
    # https://github.com/pytorch/pytorch/issues/88612
    return torch.empty_like(self).contiguous()


@register_meta(aten.bernoulli_.float)
def meta_bernoulli_(self, p=0.5, generator=None):
    return self


@register_meta(aten.bernoulli.p)
def meta_bernoulli_p(self, p=0.5, generator=None):
    # https://github.com/pytorch/pytorch/issues/88612
    return torch.empty_like(self).contiguous()


@register_meta(aten._fused_moving_avg_obs_fq_helper.default)
def meta__fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    check(
        ch_axis < self.dim(),
        lambda: "Error in fused_moving_avg_obs_fake_quant_cpu: ch_axis must be < self.dim()",
    )
    mask = torch.empty_like(self, dtype=torch.bool)
    return (torch.empty_like(self), mask)


def dot_check(self, other):
    check(
        self.dim() == 1 and other.dim() == 1,
        lambda: f"1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors",
    )


@register_meta(aten.dot.default)
def meta_dot(self, tensor):
    dot_check(self, tensor)
    return self.new_empty(())


@register_meta([aten.mm.default])
def meta_mm(a, b):
    check(a.dim() == 2, lambda: "a must be 2D")
    check(b.dim() == 2, lambda: "b must be 2D")
    N, M1 = a.shape
    M2, P = b.shape
    check(M1 == M2, lambda: "a and b must have same reduction dim")
    return a.new_empty(N, P)


def _compute_reduction_shape(self, dims, keepdim):
    if keepdim:
        return tuple(self.shape[i] if i not in dims else 1 for i in range(self.ndim))

    return utils.compute_reduction_output_shape(self.shape, dims)


# FakeTensors (meta tensors with a device) will report device as meta
# when running meta kernels. Here, access the "fake device" of FakeTensor if it
# exists so meta kernels which have diverge per device will be more
# accurate when run with FakeTensors
def device_hint(tensor) -> "str":
    if isinstance(tensor, torch._subclasses.FakeTensor):
        return tensor.fake_device.type
    else:
        return "cuda"  # default to cuda


@register_meta(aten.convolution.default)
def meta_conv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    is_transposed: bool,
    output_padding: List[int],
    groups: int,
):
    def _formula(ln: int, p: int, d: int, k: int, s: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output

        See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
        Returns:
            The output length
        """
        return (ln + 2 * p - d * (k - 1) - 1) // s + 1

    def _formula_transposed(ln: int, p: int, d: int, k: int, s: int, op: int) -> int:
        """
        Formula to apply to calculate the length of some dimension of the output
        if transposed convolution is used.
        See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

        Args:
            ln: length of the dimension
            p: padding in that dim
            d: dilation in that dim
            k: kernel size in that dim
            s: stride in that dim
            op: output padding in that dim

        Returns:
            The output length
        """
        return (ln - 1) * s - 2 * p + d * (k - 1) + op + 1

    def calc_conv_nd_return_shape(
        dims: torch.Size,
        kernel_size: torch.Size,
        stride: Union[List[int], int],
        padding: Union[List[int], int],
        dilation: Union[List[int], int],
        output_padding: Optional[Union[List[int], int]] = None,
    ):
        ret_shape = []
        if isinstance(stride, IntLike):
            stride = [stride] * len(dims)
        elif len(stride) == 1:
            stride = [stride[0]] * len(dims)

        if isinstance(padding, IntLike):
            padding = [padding] * len(dims)
        elif len(padding) == 1:
            padding = [padding[0]] * len(dims)

        if isinstance(dilation, IntLike):
            dilation = [dilation] * len(dims)
        elif len(dilation) == 1:
            dilation = [dilation[0]] * len(dims)

        output_padding_list: Optional[List[int]] = None
        if output_padding:
            if isinstance(output_padding, IntLike):
                output_padding_list = [output_padding] * len(dims)
            elif len(output_padding) == 1:
                output_padding_list = [output_padding[0]] * len(dims)
            else:
                output_padding_list = output_padding

        for i in range(len(dims)):
            # If output_padding is present, we are dealing with a transposed convolution
            if output_padding_list:
                ret_shape.append(
                    _formula_transposed(
                        dims[i],
                        padding[i],
                        dilation[i],
                        kernel_size[i],
                        stride[i],
                        output_padding_list[i],
                    )
                )
            else:
                ret_shape.append(
                    _formula(
                        dims[i], padding[i], dilation[i], kernel_size[i], stride[i]
                    )
                )
        return ret_shape

    def is_channels_last(ten):
        return torch._prims_common.suggest_memory_format(ten) == torch.channels_last

    def pick_memory_format():
        if device_hint(input_tensor) == "cuda":
            if is_channels_last(input_tensor) or is_channels_last(weight):
                return torch.channels_last
        else:
            if is_channels_last(input_tensor):
                return torch.channels_last
        if input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif input_tensor.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format

    kernel_size = weight.shape[2:]
    dims = input_tensor.shape[2:]
    if is_transposed:
        out_channels = groups * weight.shape[1]

        shape_out = calc_conv_nd_return_shape(
            dims,
            kernel_size,
            stride,
            padding,
            dilation,
            output_padding,
        )

    else:
        out_channels = weight.shape[0]
        if weight.shape[1] * groups != input_tensor.shape[1]:
            raise RuntimeError("Invalid channel dimensions")
        shape_out = calc_conv_nd_return_shape(
            dims, kernel_size, stride, padding, dilation
        )
    out = input_tensor.new_empty((input_tensor.shape[0], out_channels, *shape_out))

    out = out.to(memory_format=pick_memory_format())  # type: ignore[call-overload]
    return out


# from check_dim_size() in aten/src/ATen/TensorUtils.cpp.
def check_dim_size(tensor, dim, dim_size, size):
    check(
        tensor.dim() == dim and tensor.shape[dim_size] == size,
        lambda: f"Expected a tensor of dimension {dim} and tensor.size[{dim_size}] == {size}, "
        + f"but got : dimension {tensor.dim()} and tensor.size[{dim_size}] = {tensor.shape[dim_size]}",
    )


@register_meta(aten.avg_pool2d.default)
def meta_avg_pool2d(
    input,
    kernel_size,
    stride=(),
    padding=(0,),
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    def unpack(name, val):
        check(
            len(val) in [1, 2],
            lambda: f"avg_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)
    check(
        len(stride) in [0, 1, 2],
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    elif len(stride) == 1:
        dH, dW = stride[0], stride[0]
    else:
        dH, dW = unpack("stride", stride)

    padH, padW = unpack("padding", padding)

    check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    nbatch = input.size(-4) if input.dim() == 4 else 1
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    memory_format = utils.suggest_memory_format(input)
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
        memory_format,
    )

    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return torch.empty(
        size, dtype=input.dtype, device=input.device, memory_format=memory_format
    )


# from avg_pool2d_backward_shape_check() in aten/src/ATen/native/Pool.h.
def avg_pool2d_backward_shape_check(
    input,
    gradOutput,
    nbatch,
    kH,
    kW,
    dH,
    dW,
    padH,
    padW,
    nInputPlane,
    inputHeight,
    inputWidth,
    outputHeight,
    outputWidth,
    mem_format,
):
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
        mem_format,
    )

    ndim = input.dim()
    nOutputPlane = nInputPlane

    check_dim_size(gradOutput, ndim, ndim - 3, nOutputPlane)
    check_dim_size(gradOutput, ndim, ndim - 2, outputHeight)
    check_dim_size(gradOutput, ndim, ndim - 1, outputWidth)


# Don't override the C++ registration.
@register_meta(aten.avg_pool2d_backward.default)
def meta_avg_pool2d_backward(
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    # From aten/src/ATen/native/AveragePool2d.cpp structured kernel meta func.
    check(
        len(kernel_size) == 1 or len(kernel_size) == 2,
        lambda: "avg_pool2d: kernel_size must either be a single int, or a tuple of two ints",
    )
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]
    check(
        len(stride) == 0 or len(stride) == 1 or len(stride) == 2,
        lambda: "avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    dH = kH if len(stride) == 0 else stride[0]
    dW = kW if len(stride) == 0 else dH if len(stride) == 1 else stride[1]
    check(
        len(padding) == 1 or len(padding) == 2,
        lambda: "avg_pool2d: padding must either be a single int, or a tuple of two ints",
    )
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]

    check(
        divisor_override is None or divisor_override != 0,
        lambda: "divisor must be not zero",
    )

    input_size = input.shape
    nbatch = input_size[-4] if input.dim() == 4 else 1
    nInputPlane = input_size[-3]
    inputHeight = input_size[-2]
    inputWidth = input_size[-1]

    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)

    mem_format = utils.suggest_memory_format(input)

    avg_pool2d_backward_shape_check(
        input,
        gradOutput_,
        nbatch,
        kH,
        kW,
        dH,
        dW,
        padH,
        padW,
        nInputPlane,
        inputHeight,
        inputWidth,
        outputHeight,
        outputWidth,
        mem_format,
    )

    return torch.empty(
        input_size, dtype=input.dtype, device=input.device, memory_format=mem_format
    )


@register_meta(aten._adaptive_avg_pool2d.default)
def meta_adaptive_avg_pool2d(self, output_size):
    check(
        self.ndim == 3 or self.ndim == 4,
        lambda: f"Expected 3D or 4D tensor, but got {self.shape}",
    )
    output_shape = self.shape[:-2] + tuple(output_size)
    memory_format = utils.suggest_memory_format(self)
    # need to set memory_format to preserve the memory format of the input
    # channel last input should have channel last output
    return torch.empty(
        output_shape, dtype=self.dtype, device=self.device, memory_format=memory_format
    )


@register_meta(aten._adaptive_avg_pool3d.default)
def meta_adaptive_avg_pool3d(self, output_size):
    check(
        self.ndim == 4 or self.ndim == 5,
        lambda: f"Expected 4D or 5D tensor, but got {self.shape}",
    )
    return self.new_empty(self.shape[:-3] + tuple(output_size))


@register_meta(aten._adaptive_avg_pool2d_backward.default)
def meta__adaptive_avg_pool2d_backward(grad_out, self):
    ndim = grad_out.ndim
    for i in range(1, ndim):
        check(
            grad_out.size(i) > 0,
            lambda: f"adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero \
                      size for non-batch dimensions, {grad_out.shape} with dimension {i} being empty",
        )
    check(
        ndim == 3 or ndim == 4,
        lambda: f"adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got {self.shape}",
    )
    check(
        self.dtype == grad_out.dtype,
        lambda: f"expected dtype {self.dtype} for `grad_output` but got dtype {grad_out.dtype}",
    )
    return self.new_empty(self.shape)


@register_meta(aten.repeat_interleave.Tensor)
def meta_repeat_interleave_Tensor(repeats, output_size=None):
    if output_size is None:
        raise RuntimeError("cannot repeat_interleave a meta tensor without output_size")
    return repeats.new_empty(output_size)


@register_meta([aten.complex.default, aten.complex.out])
@out_wrapper()
def meta_complex(real, imag):
    assert real.dtype.is_floating_point
    assert imag.dtype.is_floating_point
    out_shape = _broadcast_shapes(real.shape, imag.shape)
    return real.new_empty(out_shape, dtype=corresponding_complex_dtype(real.dtype))


@register_meta(aten.vdot.default)
def vdot(self, other):
    if not self.is_complex:
        return torch.dot(self, other)

    if self.is_conj():
        if other.is_conj():
            return torch.vdot(other.conj(), self.conj())
        else:
            return torch.dot(self.conj(), other)
    elif other.is_conj():
        return torch.dot(self, other.conj()).conj()

    dot_check(self, other)
    return self.new_empty(())


# Leaving this function around because a python implementation
# of indexing shape inference is useful,
# but not registering it to the dispatcher because we already
# get shape inference through structured kernels
@register_meta(aten.index.Tensor)
def meta_index_Tensor(self, indices):
    check_no_bool_index_tensors(aten.index.Tensor, self, indices)
    check(indices, lambda: "at least one index must be provided")
    # aten::index is the internal advanced indexing implementation
    # checkIndexTensorTypes and expandTensors
    result: List[Optional[Tensor]] = []
    for i, index in enumerate(indices):
        if index is not None:
            check(
                index.dtype in [torch.long, torch.int, torch.int8, torch.bool],
                lambda: "tensors used as indices must be long, int, byte or bool tensors",
            )
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero()
                k = len(result)
                check(
                    k + index.ndim <= self.ndim,
                    lambda: f"too many indices for tensor of dimension {self.ndim}",
                    IndexError,
                )
                for j in range(index.ndim):
                    check(
                        index.shape[j] == self.shape[k + j],
                        lambda: f"The shape of the mask {index.shape} at index {i} "
                        f"does not match the shape of the indexed tensor {self.shape} at index {k + j}",
                        IndexError,
                    )
                    result.append(nonzero.select(1, j))
            else:
                result.append(index)
        else:
            result.append(index)
    indices = result
    check(
        len(indices) <= self.ndim,
        lambda: f"too many indices for tensor of dimension {self.ndim} (got {len(indices)})",
    )
    # expand_outplace
    import torch._refs as refs  # avoid import cycle in mypy

    indices = list(refs._maybe_broadcast(*indices))
    # add missing null tensors
    while len(indices) < self.ndim:
        indices.append(None)

    # hasContiguousSubspace
    #   true if all non-null tensors are adjacent
    # See:
    # https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    # https://stackoverflow.com/questions/53841497/why-does-numpy-mixed-basic-advanced-indexing-depend-on-slice-adjacency
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        else:
            if index is not None:
                break
    else:
        has_contiguous_subspace = True

    # transposeToFront
    # This is the logic that causes the newly inserted dimensions to show up
    # at the beginning of the tensor, if they're not contiguous
    if not has_contiguous_subspace:
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        self = self.permute(dims)
        indices = transposed_indices

    # AdvancedIndex::AdvancedIndex
    # Now we can assume the indices have contiguous subspace
    # This is simplified from AdvancedIndex which goes to more effort
    # to put the input and indices in a form so that TensorIterator can
    # take them.  If we write a ref for this, probably that logic should
    # get implemented
    before_shape: List[int] = []
    after_shape: List[int] = []
    replacement_shape: List[int] = []
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(self.shape[dim])
            else:
                before_shape.append(self.shape[dim])
        else:
            replacement_shape = list(index.shape)
    return self.new_empty(before_shape + replacement_shape + after_shape)


@register_meta([aten.convolution_backward.default])
def meta_convolution_backward(
    grad_output_,
    input_,
    weight_,
    bias_sizes_opt,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    # High level logic taken from slow_conv3d_backward_cpu which should
    # be representative of all convolution_backward impls
    backend_grad_input = None
    backend_grad_weight = None
    backend_grad_bias = None

    if output_mask[0]:
        backend_grad_input = grad_output_.new_empty(input_.size())
    if output_mask[1]:
        backend_grad_weight = grad_output_.new_empty(weight_.size())
    if output_mask[2]:
        backend_grad_bias = grad_output_.new_empty(bias_sizes_opt)

    return (backend_grad_input, backend_grad_weight, backend_grad_bias)


@register_meta([aten.addbmm.default, aten.addbmm.out])
@out_wrapper()
def meta_addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    dim1 = batch1.size(1)
    dim2 = batch2.size(2)
    self = self.expand((dim1, dim2))
    check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")
    check(
        batch1.size(0) == batch2.size(0),
        lambda: f"batch1 and batch2 must have same number of batches, got {batch1.size(0)} and {batch2.size(0)}",
    )
    check(
        batch1.size(2) == batch2.size(1),
        lambda: (
            f"Incompatible matrix sizes for bmm ({batch1.size(1)}x{batch1.size(2)} "
            f"and {batch2.size(1)}x{batch2.size(2)})"
        ),
    )
    check(
        self.size(0) == dim1 and self.size(1) == dim2,
        lambda: "self tensor does not match matmul output shape",
    )
    return self.new_empty(self.size())


@register_meta(aten._cdist_forward.default)
def meta_cdist_forward(x1, x2, p, compute_mode):
    check(
        x1.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X1 got: {x1.dim()}D",
    )
    check(
        x2.dim() >= 2,
        lambda: f"cdist only supports at least 2D tensors, X2 got: {x2.dim()}D",
    )
    check(
        x1.size(-1) == x2.size(-1),
        lambda: f"X1 and X2 must have the same number of columns. X1: {x1.size(-1)} X2: {x2.size(-1)}",
    )
    check(
        utils.is_float_dtype(x1.dtype),
        lambda: "cdist only supports floating-point dtypes, X1 got: {x1.dtype}",
    )
    check(
        utils.is_float_dtype(x2.dtype),
        lambda: "cdist only supports floating-point dtypes, X2 got: {x2.dtype}",
    )
    check(p >= 0, lambda: "cdist only supports non-negative p values")
    check(
        compute_mode >= 0 and compute_mode <= 2,
        lambda: f"possible modes: 0, 1, 2, but was: {compute_mode}",
    )
    r1 = x1.size(-2)
    r2 = x2.size(-2)
    batch_tensor1 = x1.shape[:-2]
    batch_tensor2 = x2.shape[:-2]
    output_shape = list(torch.broadcast_shapes(batch_tensor1, batch_tensor2))
    output_shape.extend([r1, r2])
    return x1.new_empty(output_shape)


@register_meta(aten._embedding_bag.default)
def meta_embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    check(
        indices.dtype in (torch.long, torch.int),
        lambda: f"expected indices to be long or int, got {indices.dtype}",
    )
    check(
        offsets.dtype in (torch.long, torch.int),
        lambda: f"expected offsets to be long or int, got {offsets.dtype}",
    )
    check(
        utils.is_float_dtype(weight.dtype),
        lambda: f"expected weight to be floating point type, got {weight.dtype}",
    )

    num_bags = offsets.size(0)
    if include_last_offset:
        check(
            num_bags >= 1, lambda: "include_last_offset: numBags should be at least 1"
        )
        num_bags -= 1

    output = weight.new_empty(num_bags, weight.size(1))
    MODE_SUM, MODE_MEAN, MODE_MAX = range(3)

    if per_sample_weights is not None:
        check(
            mode == MODE_SUM,
            lambda: "embedding_bag: per_sample_weights only supported with mode='sum'",
        )
        check(
            per_sample_weights.dtype == weight.dtype,
            lambda: f"expected weight ({weight.dtype}) and per_sample_weights ({per_sample_weights.dtype}) to have same dtype",
        )
        check(
            per_sample_weights.ndim == 1,
            lambda: f"expected per_sample_weights to be 1D tensor, got {per_sample_weights.ndim}D",
        )
        check(
            per_sample_weights.numel() == indices.numel(),
            lambda: (
                f"expected per_sample_weights.numel() ({per_sample_weights.numel()} "
                f"to be the same as indices.numel() ({indices.numel()})"
            ),
        )

    def is_fast_path_index_select_scale(src, scale, output, padding_idx):
        return (
            is_fast_path_index_select(src, output, padding_idx) and scale.stride(0) == 1
        )

    def is_fast_path_index_select(src, output, padding_idx):
        return (
            (src.dtype == torch.float or src.dtype == torch.half)
            and src.stride(1) == 1
            and output.stride(1) == 1
            and padding_idx < 0
        )

    def is_fast_path(src, scale, output, padding_idx):
        if scale is not None:
            return is_fast_path_index_select_scale(src, scale, output, padding_idx)
        else:
            return is_fast_path_index_select(src, output, padding_idx)

    if device_hint(offsets) != "cpu":
        offset2bag = indices.new_empty(indices.size(0))
        bag_size = indices.new_empty(offsets.size())
        if mode == MODE_MAX:
            max_indices = indices.new_empty(num_bags, weight.size(1))
        else:
            max_indices = indices.new_empty(0)
    else:
        fast_path_sum = is_fast_path(weight, per_sample_weights, output, padding_idx)
        if mode == MODE_MEAN or mode == MODE_MAX or not fast_path_sum:
            offset2bag = offsets.new_empty(indices.size(0))
        else:
            offset2bag = offsets.new_empty(0)
        bag_size = offsets.new_empty(num_bags)
        max_indices = offsets.new_empty(bag_size.size())
    return output, offset2bag, bag_size, max_indices


@register_meta(aten._embedding_bag_forward_only.default)
def meta_embedding_bag_forward_only(weight, indices, offsets, *args):
    output, offset2bag, bag_size, max_indices = meta_embedding_bag(
        weight, indices, offsets, *args
    )
    if device_hint(offsets) == "cpu":
        bag_size = offsets.new_empty(offsets.size())
    return output, offset2bag, bag_size, max_indices


def _get_reduction_dtype(input, dtype, promote_int_to_long=True):
    # if specified, dtype takes precedence
    if dtype:
        return dtype

    if input.dtype.is_floating_point or input.dtype.is_complex:
        return input.dtype
    elif promote_int_to_long:
        return torch.long

    return input.dtype


@register_meta([aten.nansum.default, aten.nansum.out])
@out_wrapper()
def meta_nansum(input, dims=None, keepdim=False, *, dtype=None):
    output_dtype = _get_reduction_dtype(input, dtype, promote_int_to_long=True)
    dims = utils.reduction_dims(input.shape, dims)
    output_shape = _compute_reduction_shape(input, dims, keepdim)
    return input.new_empty(output_shape, dtype=output_dtype)


@register_meta(aten.nanmedian.default)
def meta_nanmedian(input):
    output_shape = utils.compute_reduction_output_shape(
        input.shape, tuple(range(input.dim()))
    )
    return input.new_empty(output_shape)


@register_meta([aten.nanmedian.dim, aten.nanmedian.dim_values])
@out_wrapper("values", "indices")
def meta_nanmedian_dim(input, dim=-1, keepdim=False):
    dim = utils.reduction_dims(input.shape, (dim,))
    output_shape = _compute_reduction_shape(input, dim, keepdim)
    return (
        input.new_empty(output_shape),
        input.new_empty(output_shape, dtype=torch.long),
    )


@register_meta(aten.logical_not_.default)
def meta_logical_not_(self):
    return self


@register_meta(aten.repeat.default)
def meta_repeat(self, repeats):
    check(
        len(repeats) >= self.dim(),
        lambda: "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor",
    )
    # Add new leading dimensions to the tensor if the
    # number of target dimensions is larger than the
    # number of source dimensions.
    num_new_dimensions = len(repeats) - self.dim()
    padded_size = (1,) * num_new_dimensions + tuple(self.shape)
    target_size = [padded_size[i] * repeats[i] for i in range(len(repeats))]
    return self.new_empty(target_size)


@register_meta(aten.zero_.default)
def meta_zero_(self):
    return self


@register_meta(
    [
        aten.mul_.Scalar,
        aten.div_.Scalar,
        aten.mul_.Tensor,
        aten.div_.Tensor,
        aten.logical_and_.default,
        aten.logical_or_.default,
        aten.logical_xor_.default,
    ],
)
def meta_binop_inplace(self, other):
    return self


@register_meta(
    [
        aten.add_.Scalar,
        aten.sub_.Scalar,
        aten.add_.Tensor,
        aten.sub_.Tensor,
    ],
)
def meta_binop_inplace_alpha(self, other, alpha=1):
    return self


@register_meta(aten.zero.default)
def meta_zero(self):
    return self.new_empty(self.shape)


@register_meta([aten.fill_.Tensor, aten.fill_.Scalar])
def meta_fill_(self, val):
    return self


@register_meta([aten.fill.Tensor, aten.fill.Scalar])
def meta_fill(self, val):
    return torch.empty_like(self)


@register_meta(aten.relu_.default)
def meta_relu_(self):
    return self


@register_meta(aten.index_put.default)
def meta_index_put(self, indices, values, accumulate=False):
    return torch.empty_like(self)


@register_meta(aten.masked_fill_.Scalar)
def meta_masked_fill_(self, mask, value):
    return self


@register_meta(aten.index_put_.default)
def meta_index_put_(self, indices, values, accumulate=False):
    return self


@register_meta(aten.alias.default)
def meta_alias(self):
    return self.view(self.shape)


def common_meta_baddbmm_bmm(batch1, batch2, is_bmm, self_baddbmm=None):
    check(batch1.dim() == 3, lambda: "batch1 must be a 3D tensor")
    check(batch2.dim() == 3, lambda: "batch2 must be a 3D tensor")

    batch1_sizes = batch1.size()
    batch2_sizes = batch2.size()

    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    res_rows = batch1_sizes[1]
    res_cols = batch2_sizes[2]
    output_size = (bs, res_rows, res_cols)

    check(
        batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size,
        lambda: f"Expected size for first two dimensions of batch2 tensor to be: [{bs}"
        f", {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}].",
    )

    # TODO: handle out

    output = batch2.new_empty(output_size)

    if not is_bmm and self_baddbmm is not None:
        check(self_baddbmm.dim() == 3, lambda: "self must be a 3D tensor")
        check(
            self_baddbmm.size() == output_size,
            lambda: "Expected an input tensor shape with shape {output_size} but got shape: {self.size()}",
        )

    return output


@register_meta(aten.bmm.default)
def meta_bmm(self, mat2):
    return common_meta_baddbmm_bmm(self, mat2, True)


def div_rtn(x, y):
    q = x // y
    r = x % y
    # WARNING: explicit bool conversion here is necessary;
    # would be fixed by SymBool
    if r != 0 and (bool(r < 0) != bool(y < 0)):
        q -= 1
    return q


def pooling_output_shape_pad_lr(
    inputSize, kernelSize, pad_l, pad_r, stride, dilation, ceil_mode
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
            outputSize -= 1
    return outputSize


def pooling_output_shape(inputSize, kernelSize, pad, stride, dilation, ceil_mode):
    check(stride != 0, lambda: "stride should not be zero")
    check(pad >= 0, lambda: f"pad must be non-negative, but got pad: {pad}")
    check(
        pad <= kernelSize // 2,
        lambda: f"pad should be at most half of kernel size, but got pad={pad} and kernel_size={kernelSize}",
    )
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode
    )


def pool2d_shape_check(
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
    memory_format,
):
    ndim = input.dim()
    nOutputPlane = nInputPlane

    check(
        kW > 0 and kH > 0,
        lambda: "kernel size should be greater than zero, but got kH: {kH}, kW: {kW}",
    )
    check(
        dW > 0 and dH > 0,
        lambda: "stride should be greater than zero, but got dH: {dH}, dW: {dW}",
    )
    check(
        dilationH > 0 and dilationW > 0,
        lambda: "dilation should be greater than zero, but got dilationH: {dilationH}, dilationW: {dilationW}",
    )

    valid_dims = input.size(1) != 0 and input.size(2) != 0

    if memory_format == torch.channels_last:
        check(
            ndim == 4 and valid_dims and input.size(3) != 0,
            lambda: "Expected 4D (batch mode) tensor expected for input with channels_last layout"
            " with optional 0 dim batch size for input, but got: {input.size()}",
        )
    else:
        check(
            (ndim == 3 and input.size(0) != 0 and valid_dims)
            or (ndim == 4 and valid_dims and input.size(3) != 0),
            lambda: f"Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got: {input.size()}",
        )

    check(
        kW // 2 >= padW and kH // 2 >= padH,
        lambda: "pad should be smaller than or equal to half of kernel size, but got "
        f"padW = {padW}, padH = {padH}, kW = {kW}, kH = {kH}",
    )

    check(
        outputWidth >= 1 and outputHeight >= 1,
        lambda: f"Given input size: ({nInputPlane}x{inputHeight}x{inputWidth}). "
        f"Calculated output size: ({nOutputPlane}x{outputHeight}x{outputWidth}). "
        "Output size is too small",
    )


@register_meta(aten.max_pool2d_with_indices.default)
def meta_max_pool2d_with_indices(
    input, kernel_size, stride=(), padding=(0,), dilation=(1,), ceil_mode=False
):
    # Reference: aten/src/ATen/native/DilatedMaxPool2d.cpp
    def unpack(name, val):
        check(
            len(val) in [1, 2],
            lambda: f"max_pool2d: {name} must either be a single int, or a tuple of two ints",
        )
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return H, W

    kH, kW = unpack("kernel_size", kernel_size)

    check(
        len(stride) in [0, 1, 2],
        lambda: "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints",
    )
    if len(stride) == 0:
        dH, dW = kH, kW
    else:
        dH, dW = unpack("stride", stride)

    padH, padW = unpack("padding", padding)
    dilationH, dilationW = unpack("dilation", dilation)

    memory_format = utils.suggest_memory_format(input)
    if memory_format == torch.channels_last:
        check(
            input.dim() == 4,
            lambda: "non-empty 4D (batch mode) tensor expected for input with channels_last layout",
        )
    elif memory_format == torch.contiguous_format:
        check(
            input.dim() in [3, 4],
            lambda: "non-empty 3D or 4D (batch mode) tensor expected for input",
        )
    else:
        check(
            False,
            lambda: "Unsupport memory format. Supports only ChannelsLast, Contiguous",
        )

    nbatch = input.size(-4) if input.dim() == 4 else 1
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)

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
        memory_format,
    )

    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return (
        torch.empty(
            size, dtype=input.dtype, device=input.device, memory_format=memory_format
        ),
        torch.empty(
            size, dtype=torch.int64, device=input.device, memory_format=memory_format
        ),
    )


@register_meta([aten.full.default])
def full(size, fill_value, *args, **kwargs):
    return torch.empty(size, *args, **kwargs)


@register_meta(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randn_like.default,
        aten.rand_like.default,
        aten.full_like.default,
        aten.zeros_like.default,
        aten.ones_like.default,
    ]
)
def meta_like(self, *args, **kwargs):
    return aten.empty_like.default(self, **kwargs)


# hacky: Please remove after math.ceil works with arange
@register_meta(aten.arange.default)
def arange(end, **kwargs):
    if isinstance(end, FloatLike):
        end = math.ceil(end)  # type: ignore[arg-type]

    def is_integral(x):
        return isinstance(x, IntLike) or isinstance(x, bool)

    set_to_integral_dtype = kwargs.get("dtype", None) is None and is_integral(end)
    if set_to_integral_dtype:
        kwargs["dtype"] = torch.int64

    return aten.empty([end], **kwargs)


@register_meta(aten.arange.start)
def arange_start(start, end, **kwargs):
    return aten.arange(end - start, **kwargs)


@register_meta(aten.select.int)
def meta_select(self, dim, index):
    ndim = self.dim()
    check(
        ndim != 0, lambda: "select() cannot be applied to a 0-dim tensor.", IndexError
    )

    dim = dim if dim >= 0 else dim + ndim
    size = self.size(dim)

    check(
        not (-index > size or index >= size),
        lambda: f"select(): index {index} out of range for tensor of size "
        f"{self.size()} at dimension {dim}",
        IndexError,
    )

    index = index if index >= 0 else index + size

    new_size = list(self.size())
    new_stride = list(self.stride())

    new_storage_offset = self.storage_offset() + index * new_stride[dim]
    del new_size[dim]
    del new_stride[dim]

    return self.as_strided(new_size, new_stride, new_storage_offset)


@register_meta(aten.select_scatter.default)
def meta_select_scatter(self, src, dim, index):
    return torch.empty_like(self)


@register_meta(aten.slice_scatter.default)
def meta_slice_scatter(self, src, dim=0, start=None, end=None, step=1):
    return torch.empty_like(self)


# TODO: Deduplicate this with canonicalize_dim
def maybe_wrap_dim(dim: int, dim_post_expr: int, wrap_scalar: bool = True):
    if dim_post_expr <= 0:
        assert wrap_scalar
        dim_post_expr = 1
    min = -dim_post_expr
    max = dim_post_expr - 1
    assert not (dim < min or dim > max), f"dim {dim} out of bounds ({min}, {max})"
    if dim < 0:
        dim += dim_post_expr
    return dim


def ensure_nonempty_size(t, dim):
    return 1 if t.dim() == 0 else t.shape[dim]


# From aten/src/ATen/native/ScatterGatherChecks.h
def gather_shape_check(self, dim, index):
    self_dims = max(self.dim(), 1)
    index_dims = max(index.dim(), 1)
    check(
        self_dims == index_dims,
        lambda: "Index tensor must have the same number of dimensions as input tensor",
    )
    for i in range(self_dims):
        if i != dim:
            check(
                ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i),
                lambda: f"Size does not match at dimension {i} expected index {index.shape}"
                + f" to be smaller than self {self.shape} apart from dimension {dim}",
            )


@register_meta(aten.gather.default)
def meta_gather(self, dim, index, sparse_grad=False):
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    is_index_empty = index.numel() == 0
    if not is_index_empty:
        check(
            index.dtype == torch.long,
            lambda: f"gather(): Expected dtype int64 for index, but got {index.dtype}",
        )
        gather_shape_check(self, wrapped_dim, index)
    return self.new_empty(index.shape)


# From aten/src/ATen/native/TensorAdvancedIndexing.cpp
def get_operator_enum(reduce_, use_new_options=False):
    if use_new_options:
        if reduce_ == "sum":
            return "REDUCE_ADD"
        elif reduce_ == "prod":
            return "REDUCE_MULTIPLY"
        elif reduce_ == "mean":
            return "REDUCE_MEAN"
        elif reduce_ == "amax":
            return "REDUCE_MAXIMUM"
        elif reduce_ == "amin":
            return "REDUCE_MINIMUM"
        check(
            False,
            lambda: "reduce argument must be either sum, prod, mean, amax or amin.",
        )
        return
    else:
        if reduce_ == "add":
            return "REDUCE_ADD"
        elif reduce_ == "multiply":
            return "REDUCE_MULTIPLY"
        check(False, lambda: "reduce argument must be either add or multiply.")
        return


# From aten/src/ATen/native/ScatterGatherChecks.h
def scatter_gather_dtype_check(method_name, self, index, src_opt=None):
    if index.numel() != 0:
        check(
            index.dtype == torch.long,
            lambda: f"{method_name}(): Expected dtype int64 for index",
        )

    if src_opt is not None:
        check(
            self.dtype == src_opt.dtype,
            lambda: f"{method_name}(): Expected self.dtype to be equal to src.dtype",
        )


def ensure_nonempty_dim(dim):
    return max(dim, 1)


# From aten/src/ATen/native/ScatterGatherChecks.h
def scatter_shape_check(self, dim, index, src_opt=None):
    if index.numel() == 0:
        return
    check(
        ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
        lambda: "Index tensor must have the same number of dimensions as self tensor",
    )

    is_wrong_shape = False
    self_dims = ensure_nonempty_dim(self.dim())

    # Check: index.size(d) <= self.size(d) for all d != dim
    for d in range(self_dims):
        index_d_size = ensure_nonempty_size(index, d)
        if d == dim:
            continue
        if index_d_size > ensure_nonempty_size(self, d):
            is_wrong_shape = True
            break

    # Check: index.size(d) <= src.size(d) for all d if src is Tensor
    if not is_wrong_shape and src_opt is not None:
        for d in range(self_dims):
            index_d_size = ensure_nonempty_size(index, d)
            if index_d_size > ensure_nonempty_size(src_opt, d):
                is_wrong_shape = True
                break

    if src_opt is not None:
        check(
            ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()),
            lambda: "Index tensor must have the same number of dimensions as self tensor",
        )
        check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be smaller than self {self.shape}"
            + f" apart from dimension {dim} and to be smaller than src {src_opt.shape}",
        )
    else:
        check(
            not is_wrong_shape,
            lambda: f"Expected index {index.shape} to be smaller than self {self.shape}"
            + f" apart from dimension {dim}",
        )


# From aten/src/ATen/native/TensorAdvancedIndexing.cpp
def scatter_meta_impl(self, dim, index, src=None, reduce_=None, use_new_options=False):
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    scatter_gather_dtype_check("scatter", self, index, src)
    scatter_shape_check(self, wrapped_dim, index, src)
    if reduce_ is not None:
        # Check if we have a valid reduce operator.
        get_operator_enum(reduce_, use_new_options)


@register_meta(aten.scatter_add.default)
def meta_scatter_add(self, dim, index, src):
    scatter_meta_impl(self, dim, index, src, "add")
    return self.new_empty(self.shape)


@register_meta(aten.scatter_add_)
def meta_scatter_add_(self, dim, index, src):
    scatter_meta_impl(self, dim, index, src, "add")
    return self


@register_meta(aten.scatter)
@out_wrapper()
def meta_scatter(self, dim, index, src_or_value, reduce=None):
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    scatter_meta_impl(self, dim, index, src, reduce)
    return self.new_empty(self.shape)


@register_meta(aten.scatter_)
def meta_scatter_(self, dim, index, src_or_value, reduce=None):
    src = src_or_value if isinstance(src_or_value, torch.Tensor) else None
    scatter_meta_impl(self, dim, index, src, reduce)
    return self


@register_meta([aten.scatter_reduce.two, aten.scatter_reduce.two_out])
@out_wrapper()
def meta_scatter_reduce_two(self, dim, index, src, reduce, include_self=True):
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    return self.new_empty(self.shape)


@register_meta(aten.scatter_reduce_.two)
def meta_scatter_reduce__two(self, dim, index, src, reduce, include_self=True):
    scatter_meta_impl(self, dim, index, src, reduce, use_new_options=True)
    return self


@register_meta(aten.upsample_nearest2d.vec)
def upsample_nearest2d_vec(input, output_size, scale_factors):
    mem_format = utils.suggest_memory_format(input)
    spatial_dimensions = input.dim() - 2

    input_shape = input.shape
    if output_size is not None:
        assert scale_factors is None
        out_size = output_size
    elif scale_factors is not None:
        assert output_size is None
        out_size = []
        for i in range(spatial_dimensions):
            sym_float = (input_shape[i + 2] / 1) * scale_factors[i]
            assert sym_float >= 0
            out_size.append(math.floor(sym_float))

    output_height = out_size[0]
    output_width = out_size[1]
    nbatch = input_shape[0]
    channels = input_shape[1]
    return input.new_empty((nbatch, channels, output_height, output_width)).to(
        memory_format=mem_format
    )


@register_meta([aten.sort.default, aten.sort.stable])
def meta_sort(self, stable=None, dim=-1, descending=False):
    return torch.empty_like(self), torch.empty_like(self, dtype=torch.int64)


def zero_numel_check_dims(self, dim, fn_name):
    if self.ndim == 0:
        check(
            dim == 0 or dim == -1,
            lambda: f"{fn_name}: Expected reduction dim -1 or 0 for scalar but got {dim}",
            IndexError,
        )
    else:
        check(
            self.size(dim) != 0,
            lambda: f"{fn_name}: Expected reduction dim {dim} to have non-zero size.",
            IndexError,
        )


# From aten/src/ATen/native/ReduceOps.cpp
def check_argmax_argmin(name, self, dim):
    if dim is not None:
        dim = maybe_wrap_dim(dim, self.dim())
        zero_numel_check_dims(self, dim, name)
    else:
        check(
            self.numel() != 0,
            lambda: f"{name}: Expected reduction dim to be specified for input.numel() == 0.",
        )


@register_meta([aten.argmax.default, aten.argmin.default])
def argmax_argmin_meta(self, dim=None, keepdim=False):
    check_argmax_argmin("argmax", self, dim)
    dims = utils.reduction_dims(self.shape, (dim,) if dim is not None else None)
    shape = _compute_reduction_shape(self, dims, keepdim)
    return self.new_empty(shape, dtype=torch.int64)


@register_meta(aten.scalar_tensor.default)
def scalar_tensor(s, dtype=None, layout=None, device=None, pin_memory=None):
    return torch.empty(
        (), dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_meta(aten.topk.default)
def topk_meta(self, k, dim=-1, largest=True, sorted=True):
    # From aten/src/ATen/native/Sorting.cpp
    dim = maybe_wrap_dim(dim, self.dim(), wrap_scalar=True)
    check(
        k >= 0 and k <= (self.size(dim) if self.dim() > 0 else 1),
        lambda: "selected index k out of range",
    )
    sliceSize = 1 if self.dim() == 0 else self.size(dim)
    check(k >= 0 and k <= sliceSize, lambda: "k not in range for dimension")

    topKSize = list(self.shape)
    if len(topKSize) > 0:
        topKSize[dim] = k
    return self.new_empty(topKSize), self.new_empty(topKSize, dtype=torch.int64)


# We must also trigger meta registrations from PrimTorch ref
# decompositions
import torch._refs
import torch._refs.nn.functional
import torch._refs.special


def activate_meta():

    activate_meta_table = {}

    # For a given op, we pick the most specific decomp function from
    # global_decomp_table in the precedence order of meta > post_autograd > pre_autograd
    for type in ["meta", "post_autograd", "pre_autograd"]:
        registry = global_decomposition_table[type]

        for opo in registry:
            if opo not in activate_meta_table:
                activate_meta_table[opo] = registry[opo]

    for op_overload, fn in activate_meta_table.items():
        assert isinstance(op_overload, OpOverload)

        op_overload.py_impl(torch._C.DispatchKey.Meta)(fn)

        if torch._C._dispatch_has_kernel_for_dispatch_key(
            op_overload.name(), "CompositeImplicitAutograd"
        ):
            # Internally, we shouldn't be registering meta kernels for any operators that
            # have CompositeImplicitAutograd kernels.
            # Instead, we should be letting those decompositions run, and writing meta kernels
            # only for the base operators.
            pass
        elif op_overload.is_view:
            # Attempting to register a python meta kernel for a view operator.
            # We shouldn't do this, because the output will report as not having aliased storages.
            # All view ops have meta kernels in C++ today, so we should use those instead.
            pass
        elif op_overload.name() in {
            "aten::empty_strided",  # causing infinite recursion, test_meta.py
            "aten::clone",  # causing infinite recursion
            "aten::_to_copy",  # causing infinite recursion, test_serialization.py -k test_tensor_subclass_getstate_overwrite  # noqa: B950
            "aten::copy_",  # Exception not raised, test_torch.py -k test_storage_meta_errors_cpu_int64  # noqa: B950
            "aten::constant_pad_nd",  # requires_grad mismatch, test_ops.py -k test_fake_crossref_backward_amp_istft_cuda_float32  # noqa: B950
            "aten::rot90",  # requires_grad mismatch! test_ops.py -k test_fake_crossref_backward_amp_rot90_cuda_float32  # noqa: B950
        }:
            pass
        else:
            _meta_lib_dont_use_me_use_register_meta.impl(op_overload, fn)


activate_meta()
