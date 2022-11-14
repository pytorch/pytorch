import functools
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import product
from typing import Callable, cast, Iterable, List, Optional, Tuple

import torch
import torch._prims_common as utils
import torch.nn.functional as F
from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import _maybe_resize_out, _safe_copy_out, out_wrapper
from torch.utils._pytree import tree_flatten, tree_map

DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]

# None of these functions are publicly accessible; get at them
# from torch._decomps
__all__: List[str] = []

aten = torch.ops.aten


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


# This wraps a decomposition and performs various type promotion logic within it, depending on the strategy provided
# We're currently re-using ELEMENTWISE_TYPE_PROMOTION_KIND, although some of the usages are on non-elementwise ops
# Will need to validate the non-elementwise uses
def type_casts(
    f: Callable,
    type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND,
    compute_dtype_only: bool = False,
):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        flat_args = [
            x for x in tree_flatten((args, kwargs))[0] if isinstance(x, Tensor)
        ]
        computation_dtype, result_dtype = utils.elementwise_dtypes(
            *flat_args, type_promotion_kind=type_promotion
        )

        # TODO: pretty sure this is not quite right
        def increase_prec(x):
            if isinstance(x, Tensor):
                return x.to(computation_dtype)
            else:
                return x

        def decrease_prec(x):
            if isinstance(x, Tensor):
                return x.to(result_dtype)
            else:
                return x

        r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
        if compute_dtype_only:
            return r
        else:
            return tree_map(decrease_prec, r)

    return inner


compute_only_pw_cast_for_opmath = partial(
    type_casts,
    type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    compute_dtype_only=True,
)
pw_cast_for_opmath = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
reduction_complex_to_real = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
)
pw_cast_for_int_to_real = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)

# This expands x until x.dim() == dim. Might be useful as an operator
def _unsqueeze_to_dim(x: Tensor, dim: int):
    for _ in range(dim - x.dim()):
        x = x.unsqueeze(-1)
    return x


@register_decomposition(aten.tanh_backward)
@pw_cast_for_opmath
def tanh_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (1 - y * y).conj_physical()


@register_decomposition(aten.sigmoid_backward)
@pw_cast_for_opmath
def sigmoid_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (y * (1 - y)).conj_physical()


@register_decomposition(aten.softplus_backward)
@pw_cast_for_opmath
def softplus_backward(out_grad: Tensor, x: Tensor, beta: float, threshold: float):
    z = (x * beta).exp()
    return torch.where((x * beta) > threshold, out_grad, out_grad * z / (z + 1.0))


@register_decomposition(aten.elu_backward)
@pw_cast_for_opmath
def elu_backward(
    grad_output: Tensor,
    alpha: float,
    scale: float,
    input_scale: float,
    is_result: bool,
    self_or_result: Tensor,
):
    negcoef = alpha * scale
    poscoef = scale
    negiptcoef = input_scale
    if is_result:
        return torch.where(
            self_or_result <= 0,
            grad_output * negiptcoef * (self_or_result + negcoef),
            self_or_result * poscoef,
        )
    else:
        return torch.where(
            self_or_result <= 0,
            grad_output * negiptcoef * negcoef * torch.exp(self_or_result * negiptcoef),
            grad_output * poscoef,
        )


@register_decomposition(aten.hardsigmoid)
@pw_cast_for_opmath
def hardsigmoid(self: Tensor) -> Tensor:
    return torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardsigmoid_backward)
@pw_cast_for_opmath
def hardsigmoid_backward(grad_output: Tensor, self: Tensor):
    return torch.where(
        (self > -3.0) & (self < 3.0),
        grad_output * (1.0 / 6.0),
        0.0,
    )


@register_decomposition(aten.hardtanh_backward)
@pw_cast_for_opmath
def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
):
    return torch.where((self <= min_val) | (self >= max_val), 0.0, grad_output)


@register_decomposition(aten.hardshrink_backward)
@pw_cast_for_opmath
def hardshrink_backward(grad_out: Tensor, self: Tensor, lambd: float):
    return torch.where((self >= -lambd) & (self <= lambd), 0.0, grad_out)


@register_decomposition(aten.hardswish)
@pw_cast_for_opmath
def hardswish(self: Tensor) -> Tensor:
    return self * torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardswish_backward)
@pw_cast_for_opmath
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    return torch.where(
        self < -3,
        0.0,
        torch.where(self <= 3, grad_output * ((self / 3) + 0.5), grad_output),
    )


@register_decomposition(aten.threshold_backward)
@pw_cast_for_opmath
def threshold_backward(grad_output: Tensor, self: Tensor, threshold: float):
    return torch.where(self <= threshold, 0.0, grad_output)


@register_decomposition(aten.leaky_relu_backward)
@pw_cast_for_opmath
def leaky_relu_backward(
    grad_output: Tensor, self: Tensor, negative_slope: float, self_is_result: bool
):
    return torch.where(self > 0, grad_output, grad_output * negative_slope)


@register_decomposition(aten.gelu_backward)
@pw_cast_for_opmath
def gelu_backward(grad: Tensor, self: Tensor, approximate: str = "none"):
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "tanh":
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        x_sq = self * self
        x_cube = x_sq * self
        inner = kBeta * (self + kKappa * x_cube)
        tanh_inner = torch.tanh(inner)

        left = 0.5 * self
        right = 1 + tanh_inner

        left_derivative = 0.5 * right

        tanh_derivative = 1 - tanh_inner * tanh_inner
        inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)
        right_derivative = left * tanh_derivative * inner_derivative

        return grad * (left_derivative + right_derivative)
    else:
        kAlpha = M_SQRT1_2
        kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
        cdf = 0.5 * (1 + torch.erf(self * kAlpha))
        pdf = kBeta * torch.exp(self * self * -0.5)
        return grad * (cdf + self * pdf)


@register_decomposition(aten.mish_backward)
@pw_cast_for_opmath
def mish_backward(grad_output: Tensor, input: Tensor):
    input_tanh_softplus = torch.tanh(F.softplus(input))
    input_sigmoid = torch.sigmoid(input)
    out = input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)
    return grad_output * (input_tanh_softplus + out)


@register_decomposition(aten.silu)
@pw_cast_for_opmath
def silu(self: Tensor) -> Tensor:
    return self * torch.sigmoid(self)


@register_decomposition(aten.silu_backward)
@pw_cast_for_opmath
def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    sigmoid = 1 / (1 + torch.exp(-self))
    return grad_output * sigmoid * (1 + self * (1 - sigmoid))


@register_decomposition(aten.softshrink_backward)
def softshrink_backward(grad_output: Tensor, self: Tensor, lambd: float) -> Tensor:
    return torch.where((self >= -lambd) & (self <= lambd), 0.0, grad_output)


@register_decomposition(aten.prelu_backward)
@pw_cast_for_opmath
def prelu_backward(
    grad_output: Tensor, self: Tensor, weight: Tensor
) -> Tuple[Tensor, Tensor]:
    # Logic is more complicated than I would like.  Basically, weight can either
    # be a scalar or a vector of size [C], and in the forward pass it's
    # broadcast against [N, C, ...]. So now, we need to do the corresponding
    # reduction, which is harder than we'd like...
    cur_weight = weight
    for _ in range(2, grad_output.dim()):
        cur_weight = cur_weight.unsqueeze(-1)
    input_grad = torch.where(self > 0, grad_output, cur_weight * grad_output)
    weight_grad_collector = torch.where(self > 0, 0.0, self * grad_output)
    out = weight_grad_collector.sum_to_size(cur_weight.shape)
    while out.dim() > weight.dim():
        out = out.squeeze(-1)
    return (input_grad, out)


@register_decomposition(aten.rrelu_with_noise_backward)
@pw_cast_for_opmath
def rrelu_with_noise_backward(
    grad_output: Tensor,
    self: Tensor,
    noise: Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool,
) -> Tensor:
    if training and upper - lower > 1e-6:
        return grad_output.mul(noise)
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu_backward(
            grad_output, self, negative_slope, self_is_result
        )


@register_decomposition(aten.log_sigmoid_backward)
@pw_cast_for_opmath
def log_sigmoid_backward(grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor:
    in_negative = self < 0
    max_deriv = torch.where(in_negative, 1, 0)
    sign = torch.where(in_negative, 1, -1)
    z = torch.exp(-torch.abs(self))
    return grad_output * (max_deriv - sign * (z / (1 + z)))
    # CPU has a special formula that uses buffer, but disabled for convenience sake
    # return (max_deriv - sign * (buffer / (1 + buffer))) * grad_output


def apply_loss_reduction(loss: Tensor, reduction: int):
    if reduction == Reduction.MEAN.value:
        return torch.mean(loss)
    elif reduction == Reduction.SUM.value:
        return torch.sum(loss)
    else:
        return loss


def to_real_dtype(dtype: torch.dtype):
    if dtype == torch.complex32:
        return torch.float16
    elif dtype == torch.complex64:
        return torch.float32
    elif dtype == torch.complex128:
        return torch.float64


# TODO: None of these loss castings are quite correct, see
# https://github.com/pytorch/pytorch/issues/76870. Also, the ATen kernels
# perform the pointwise portion in opmath, but don't maintain it between the
# pointwise portion and the reduction


@register_decomposition(aten.mse_loss)
@pw_cast_for_opmath
def mse_loss(
    self: Tensor, target: Tensor, reduction: int = Reduction.MEAN.value
) -> Tensor:
    loss = (self - target) ** 2
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.mse_loss_backward)
@pw_cast_for_opmath
def mse_loss_backward(
    grad_output: Tensor, input: Tensor, target: Tensor, reduction: int
):
    norm = 2.0 / input.numel() if reduction == Reduction.MEAN.value else 2.0
    return norm * (input - target) * grad_output


@register_decomposition(aten.huber_loss_backward.default)
@pw_cast_for_opmath
def huber_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, delta: float
):
    norm = 1.0 / self.numel() if reduction == Reduction.MEAN.value else 1.0
    x = self - target
    return torch.where(
        x < -delta,
        -norm * grad_output * delta,
        torch.where(x > delta, norm * grad_output * delta, norm * x * grad_output),
    )


# We cannot use @out_wrapper() here, because the output tensor is not named 'out', it's 'grad_input'
@register_decomposition(aten.huber_loss_backward.out)
@pw_cast_for_opmath
def huber_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    delta: float,
    grad_input: Tensor,
):
    result = huber_loss_backward(grad_output, self, target, reduction, delta)
    _maybe_resize_out(grad_input, result.shape)
    return _safe_copy_out(copy_from=result, copy_to=grad_input, exact_dtype=True)


def _nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    channel_dim = 0 if self.dim() < 2 else 1
    if reduction == Reduction.MEAN.value:
        grad_output = grad_output / total_weight

    target = target.unsqueeze(channel_dim)
    grad_input = torch.zeros_like(self)
    grad_input = torch.scatter(grad_input, channel_dim, target, -1.0)

    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    if weight is not None:
        new_shape = [1 for _ in range(self.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        grad_output = grad_output * weight

    has_ignore_index = ignore_index >= 0
    if has_ignore_index:
        grad_output = torch.where(target != ignore_index, grad_output, 0)

    return grad_input * grad_output


@register_decomposition(aten.glu_backward)
@pw_cast_for_opmath
def glu_backward(grad_output: Tensor, self: Tensor, dim: int) -> Tensor:
    assert self.dim() > 0, "glu does not support 0-dimensional tensors"
    wrap_dim = utils.canonicalize_dim(self.dim(), dim)
    nIn = self.size(wrap_dim)
    assert (
        nIn % 2 == 0
    ), f"Halving dimension must be even, but dimension {wrap_dim} is size {nIn}"
    inputSize = nIn // 2
    firstHalf = self.narrow(wrap_dim, 0, inputSize)
    secondHalf = self.narrow(wrap_dim, inputSize, inputSize)
    gradInputFirstHalf = torch.sigmoid(secondHalf)
    gradInputSecondHalf = (
        (1.0 - gradInputFirstHalf) * gradInputFirstHalf * firstHalf * grad_output
    )
    gradInputFirstHalf = gradInputFirstHalf * grad_output
    return torch.cat([gradInputFirstHalf, gradInputSecondHalf], dim=wrap_dim)


@register_decomposition(aten.nll_loss_backward)
def nll_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    assert 0 <= self.dim() <= 2, "input tensor should be 1D or 2D"
    assert (
        target.dim() <= 1
    ), "0D or 1D target tensor expected, multi-target not supported"

    no_batch_dim = self.dim() == 1 and target.dim() == 0
    assert no_batch_dim or (
        self.shape[0] == target.shape[0]
    ), f"size mismatch (got input: {self.shape}, target: {target.shape})"
    assert total_weight.numel() == 1, (
        "expected total_weight to be a single element tensor, got: ",
        f"{total_weight.shape} ({total_weight.numel()} elements)",
    )

    assert (
        weight is None or weight.numel() == self.shape[-1]
    ), "weight tensor should be defined either for all or no classes"

    if reduction == Reduction.NONE.value and self.dim() == 2:
        assert grad_output.dim() == 1 and grad_output.shape[0] == self.shape[0], (
            f"Expected a tensor of dimension 1 and tensor.size[0] == {self.shape[0]} but "
            f"got: dimension {grad_output.dim()} and tensor.size[0] == {grad_output.shape[0]}"
        )
    else:
        assert (
            grad_output.dim() <= 1 and grad_output.numel() == 1
        ), f"Expected a single element grad_output tensor, but got: {grad_output.shape}"

    return _nll_loss_backward(
        grad_output, self, target, weight, reduction, ignore_index, total_weight
    )


@register_decomposition(aten.nll_loss2d_backward)
def nll_loss2d_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
    total_weight: Tensor,
) -> Tensor:
    assert (
        self.dim() == 4
    ), f"only batches of spatial inputs supported (4D tensors), but got input of dimension: {self.dim()}"

    assert (
        target.dim() == 3
    ), f"only batches of spatial targets supported (3D tensors) but got targets of dimension: {target.dim()}"

    assert (
        self.shape[0] == target.shape[0]
        and self.shape[2] == target.shape[1]
        and self.shape[3] == target.shape[2]
    ), f"size mismatch (got input: {self.shape}, target: {target.shape}"

    assert total_weight.numel() == 1, (
        "expected total_weight to be a single element tensor, "
        f"got: {total_weight.shape} ( {total_weight.numel()}, elements)"
    )

    return _nll_loss_backward(
        grad_output, self, target, weight, reduction, ignore_index, total_weight
    )


@register_decomposition(aten.binary_cross_entropy)
@pw_cast_for_opmath
def binary_cross_entropy(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    # We cannot currently model this without introducing data-dependent control flow
    # TORCH_CHECK(
    #     (input_val >= 0) && (input_val <= 1),
    #     "all elements of input should be between 0 and 1"
    # )
    loss = (target - 1) * torch.maximum(
        torch.log(1 - self), self.new_full((), -100)
    ) - target * torch.maximum(torch.log(self), self.new_full((), -100))
    if weight is not None:
        loss = loss * weight
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.binary_cross_entropy_backward)
@pw_cast_for_opmath
def binary_cross_entropy_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    EPSILON = 1e-12
    result = grad_output * (self - target) / torch.clamp(self * (1 - self), min=EPSILON)
    if weight is not None:
        result = result * weight
    if reduction == Reduction.MEAN.value:
        result = result / self.numel()
    return result


@register_decomposition(aten.soft_margin_loss)
@out_wrapper()
@pw_cast_for_opmath
def soft_margin_loss(
    input: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    loss = torch.log1p(torch.exp(-input * target))
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.soft_margin_loss_backward)
@pw_cast_for_opmath
def soft_margin_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    grad_input = target * grad_output * (torch.sigmoid(target * self) - 1)
    if reduction == Reduction.MEAN.value:
        grad_input = grad_input / self.numel()
    return grad_input


@register_decomposition(aten._euclidean_dist)
def _euclidean_dist(x1: Tensor, x2: Tensor) -> Tensor:
    x1_norm = x1.pow(2).sum(-1, True)
    x1_pad = torch.ones_like(x1_norm, memory_format=torch.contiguous_format)
    x2_norm = x2.pow(2).sum(-1, True)
    x2_pad = torch.ones_like(x2_norm, memory_format=torch.contiguous_format)
    x1_ = torch.cat([x1.mul(-2), x1_norm, x1_pad], -1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], -1)
    result = x1_.matmul(x2_.mT)
    return result.clamp_min(0).sqrt()


@register_decomposition(aten.slice_backward)
def slice_backward(
    grad_output: Tensor,
    input_sizes: List[int],
    dim: int,
    start: int,
    end: int,
    step: int,
):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.slice_scatter(grad_input, grad_output, dim, start, end, step)


@register_decomposition(aten.slice.Tensor)
def slice_forward(
    # Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1
    self: Tensor,
    dim: int = 0,
    start: Optional[int] = None,
    end: Optional[int] = None,
    step: int = 1,
):

    ndim = self.dim()
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    dim = utils.canonicalize_dim(self.dim(), dim)
    sizes = list(self.size())
    strides = list(self.stride())

    if step <= 0:
        raise RuntimeError("slice step must be positive")

    start_val = start if start is not None else 0
    end_val = end if end is not None else sys.maxsize  # 2^63 – 1

    if start_val < 0:
        start_val += sizes[dim]

    if end_val < 0:
        end_val += sizes[dim]

    if start_val < 0:
        start_val = 0
    elif start_val >= sizes[dim]:
        start_val = sizes[dim]

    if end_val < start_val:
        end_val = start_val
    elif end_val >= sizes[dim]:
        end_val = sizes[dim]

    storage_offset = self.storage_offset() + start_val * strides[dim]
    len = end_val - start_val
    sizes[dim] = (len + step - 1) // step
    strides[dim] *= step

    if self.is_quantized:
        raise NotImplementedError(
            "Slice decomposition for quantized tensors aren't implemented"
        )
    else:
        return self.as_strided(sizes, strides, storage_offset)


@register_decomposition(aten.select_backward)
def select_backward(grad_output: Tensor, input_sizes: List[int], dim: int, index: int):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.select_scatter(grad_input, grad_output, dim, index)


@register_decomposition(aten.diagonal_backward)
def diagonal_backward(
    grad_output: Tensor, input_sizes: List[int], offset: int, dim1: int, dim2: int
):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.diagonal_scatter(grad_input, grad_output, offset, dim1, dim2)


def _cast_grad_to_input_dtype(
    grad_output: Tensor, grad_input: Tensor, input_dtype: torch.dtype
):
    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input


@register_decomposition(aten._softmax_backward_data)
@compute_only_pw_cast_for_opmath
def _softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    new_grad_output = grad_output * output
    grad_input = new_grad_output - output * torch.sum(
        new_grad_output, dim=dim, keepdim=True
    )
    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)


@register_decomposition(aten._log_softmax_backward_data)
@compute_only_pw_cast_for_opmath
def _log_softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: torch.dtype
):
    grad_input = grad_output - torch.exp(output) * torch.sum(
        grad_output, dim=dim, keepdim=True
    )
    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype)


def _im2col_col2im_indices_along_dim(
    input_d, kernel_d, dilation_d, padding_d, stride_d, device
):
    """Utility function to implement im2col and col2im"""
    blocks_d = input_d + padding_d * 2 - dilation_d * (kernel_d - 1)

    arange_kw = partial(torch.arange, dtype=torch.int64, device=device)

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = arange_kw(0, blocks_d, stride_d).unsqueeze(0)

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = arange_kw(0, kernel_d * dilation_d, dilation_d).unsqueeze(-1)

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    return blocks_d_indices + kernel_grid


@register_decomposition(aten.im2col)
@out_wrapper()
@pw_cast_for_opmath
def im2col(
    input: Tensor,
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> Tensor:
    utils.check(len(kernel_size) == 2, lambda: "im2col(): only 2D kernel supported")
    utils.check(len(dilation) == 2, lambda: "im2col(): only 2D dilation supported")
    utils.check(len(padding) == 2, lambda: "im2col(): only 2D padding supported")
    utils.check(len(stride) == 2, lambda: "im2col(): only 2D stride supported")

    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        utils.check(
            cond, lambda: "{param_name} should be greater {'than' zero, but got {param}"
        )

    check_positive(kernel_size, "kernel_size")
    check_positive(dilation, "dilation")
    check_positive(dilation, "padding", strict=False)
    check_positive(stride, "stride")

    shape = input.shape
    ndim = len(shape)
    utils.check(
        ndim in (3, 4) and all(d != 0 for d in shape[-3:]),
        lambda: "Expected 3D or 4D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )
    output_size = tuple(
        1 + (out + 2 * pad - dil * (ker - 1) - 1) // st
        for out, pad, dil, ker, st in zip(
            shape[-2:], padding, dilation, kernel_size, stride
        )
    )
    utils.check(
        all(c > 0 for c in output_size),
        lambda: f"Given an input with spacial size {tuple(shape[-2:])}, "
        f"kernel_size={kernel_size}, dilation={dilation}, "
        f"padding={padding}, stride={stride}, "
        "the calculated shape of the array of sliding blocks "
        f"is {output_size}, but its components must be at least one.",
    )
    batched_input = ndim == 4
    if not batched_input:
        input = input.unsqueeze(0)

    batch_dim, channel_dim, input_h, input_w = input.shape

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    blocks_row_indices = _im2col_col2im_indices_along_dim(
        input_h, kernel_h, dilation_h, padding_h, stride_h, input.device
    )
    blocks_col_indices = _im2col_col2im_indices_along_dim(
        input_w, kernel_w, dilation_w, padding_w, stride_w, input.device
    )

    # Note that F.pad takes (padding_left, padding_right, padding_top, padding_bottom)
    # ugh
    padded_input = F.pad(input, (padding_w, padding_w, padding_h, padding_h))

    blocks_row_indices = blocks_row_indices.unsqueeze(-1).unsqueeze(-1)
    output = padded_input[:, :, blocks_row_indices, blocks_col_indices]
    output = output.permute(0, 1, 2, 4, 3, 5)
    num_blocks_row = blocks_row_indices.size(1)
    num_blocks_col = blocks_col_indices.size(1)
    output = output.reshape(
        batch_dim, channel_dim * kernel_h * kernel_w, num_blocks_row * num_blocks_col
    )

    if not batched_input:
        output = output.squeeze(0)
    return output


@register_decomposition(aten.col2im)
@out_wrapper()
@pw_cast_for_opmath
def col2im(
    input: Tensor,
    output_size: List[int],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> Tensor:
    utils.check(len(output_size) == 2, lambda: "only 2D output_size supported")
    utils.check(len(kernel_size) == 2, lambda: "only 2D kernel supported")
    utils.check(len(dilation) == 2, lambda: "only 2D dilation supported")
    utils.check(len(padding) == 2, lambda: "only 2D padding supported")
    utils.check(len(stride) == 2, lambda: "only 2D stride supported")

    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        utils.check(
            cond, lambda: "{param_name} should be greater than zero, but got {param}"
        )

    check_positive(kernel_size, "kernel_size")
    check_positive(dilation, "dilation")
    check_positive(padding, "padding", strict=False)
    check_positive(stride, "stride")
    check_positive(output_size, "output_size")

    shape = input.shape
    ndim = len(shape)
    utils.check(
        ndim in (2, 3) and all(d != 0 for d in shape[-2:]),
        lambda: "Expected 2D or 3D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )
    prod_kernel_size = kernel_size[0] * kernel_size[1]
    utils.check(
        shape[-2] % prod_kernel_size == 0,
        lambda: "Expected size of input's first non-batch dimension to be divisible by the "
        f"product of kernel_size, but got input.shape[-2] = {shape[-2]} and "
        f"kernel_size={kernel_size}",
    )
    col = [
        1 + (out + 2 * pad - dil * (ker - 1) - 1) // st
        for out, pad, dil, ker, st in zip(
            output_size, padding, dilation, kernel_size, stride
        )
    ]
    L = col[0] * col[1]
    utils.check(
        shape[-1] == L,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    utils.check(
        L > 0,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    batched_input = ndim == 3
    if not batched_input:
        input = input.unsqueeze(0)

    shape = input.shape

    out_h, out_w = output_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    kernel_h, kernel_w = kernel_size

    # col2im is defined as the backwards of im2col, so we differentiate its decomposition by hand
    input = input.reshape([shape[0], shape[1] // prod_kernel_size] + kernel_size + col)
    input = input.permute(0, 1, 2, 4, 3, 5)

    indices_row = _im2col_col2im_indices_along_dim(
        out_h, kernel_h, dilation_h, padding_h, stride_h, input.device
    )
    indices_row = _unsqueeze_to_dim(indices_row, 4)
    indices_col = _im2col_col2im_indices_along_dim(
        out_w, kernel_w, dilation_w, padding_w, stride_w, input.device
    )

    output_padded_size = [o + 2 * p for o, p in zip(output_size, padding)]
    output = input.new_zeros(
        [shape[0], shape[1] // prod(kernel_size)] + output_padded_size
    )
    idx = (None, None, indices_row, indices_col)
    output = torch.ops.aten.index_put(output, idx, input, accumulate=True)
    output = F.pad(output, (-padding_w, -padding_w, -padding_h, -padding_h))

    if not batched_input:
        output = output.squeeze(0)
    return output


@register_decomposition(aten.native_dropout_backward)
@pw_cast_for_opmath
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    return grad_output * (mask.type_as(grad_output) * scale)


@register_decomposition(aten.unfold_backward)
def unfold_backward(
    grad: Tensor, input_size: List[int], dimension: int, size: int, step: int
) -> Tensor:
    if len(input_size) == 0:
        return torch.squeeze_copy(grad, 0)
    dim = utils.canonicalize_dim(len(input_size), dimension)
    idx = torch.arange(input_size[dim], device=grad.device, dtype=torch.int32)
    idx = idx.unfold(0, size, step).flatten()
    grad = grad.movedim(-1, dim + 1).flatten(dim, dim + 1)
    # nb. At the moment this generates two kernels in triton
    # It could potentially be fused into one call to scatter_reduce,
    # in the case step <= size provided scatter_reduce generates 1 kernel
    grad_input = grad.new_zeros(input_size)
    return torch.index_add(grad_input, dim, idx, grad)


@register_decomposition(aten.logit_backward.default)
@pw_cast_for_opmath
def logit_backward(
    grad_output: Tensor, self: Tensor, eps: Optional[float] = None
) -> Tensor:
    if eps is not None:
        lo = eps
        hi = 1.0 - lo
        return torch.where(
            torch.logical_and(self >= lo, self <= hi),
            grad_output / (self * (1.0 - self)),
            0.0,
        )
    else:
        return torch.where(
            torch.logical_and(self >= 0.0, self <= 1.0),
            grad_output / (self * (1.0 - self)),
            self.new_full((), float("nan")),
        )


@register_decomposition(aten.native_dropout)
def native_dropout(input: Tensor, p: float, train: Optional[bool]):
    if train:
        bool_mask = torch.rand_like(input) > p
        res = bool_mask * input * float(1.0 / (1.0 - p))
        return (res, bool_mask)
    else:
        return (input, torch.ones_like(input, dtype=torch.bool))


@register_decomposition(aten._softmax)
@out_wrapper()
def _softmax(x: Tensor, dim: int, half_to_float: bool):
    # eager softmax returns a contiguous tensor. Ensure that decomp also returns
    # a contiguous tensor.
    x = x.contiguous()
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(computation_dtype)
    x_max = torch.amax(x, dim, keepdim=True)
    unnormalized = torch.exp(x - x_max)
    result = unnormalized / torch.sum(unnormalized, dim, keepdim=True)
    if not half_to_float:
        result = result.to(result_dtype)
    return result


@register_decomposition(aten._log_softmax)
@out_wrapper()
def _log_softmax(x: Tensor, dim: int, half_to_float: bool):
    # eager log_softmax returns a contiguous tensor. Ensure that decomp also
    # returns a contiguous tensor.
    x = x.contiguous()
    if half_to_float:
        assert x.dtype == torch.half
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        x, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    x = x.to(computation_dtype)
    x_max = torch.amax(x, dim, keepdim=True)
    shifted = x - x_max
    shifted_logsumexp = torch.log(torch.sum(torch.exp(shifted), dim, keepdim=True))
    result = shifted - shifted_logsumexp
    if not half_to_float:
        result = result.to(result_dtype)
    return result


@register_decomposition(aten.rsub.Tensor)
def rsub_Tensor(self: Tensor, other: Tensor, alpha: float = 1) -> Tensor:
    return torch.sub(other, self, alpha=alpha)


@register_decomposition(aten.rsub.Scalar)
def rsub_Scalar(self: Tensor, other: float, alpha: float = 1) -> Tensor:
    return torch.sub(other, self, alpha=alpha)


@register_decomposition(aten.embedding)
def embedding(
    weight: Tensor,
    indices: Tensor,
    padding_idx: int = -1,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    assert weight.dim() == 2, "'weight' must be 2-D"
    # TODO: Assert not ported over yet
    #   auto indices_arg = TensorArg(indices, "indices", 1);
    #   checkScalarTypes("embedding", indices_arg, {kLong, kInt});

    if indices.dim() == 1:
        return weight.index_select(0, indices)

    size = list(indices.shape)
    for d in weight.shape[1:]:
        size.append(d)

    return weight.index_select(0, indices.reshape(-1)).view(size)


# TODO: Correct the type promotion semantics
@register_decomposition(aten.embedding_dense_backward)
def embedding_dense_backward(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    numel = indices.numel()
    grad = grad_output.reshape(numel, grad_output.size(-1))
    grad_weight = grad_output.new_zeros((num_weights, grad_output.shape[-1]))
    indices_rank1 = indices.reshape(numel)
    if scale_grad_by_freq:
        counts = indices.new_zeros((num_weights,))
        ones = indices.new_ones((numel,))
        counts = counts.index_put([indices_rank1], ones, accumulate=True)
        grad_weights_scale = counts[indices_rank1]
        grad = grad / grad_weights_scale.unsqueeze(1)
    skip_padding = (indices_rank1 != padding_idx).unsqueeze(1)
    skip_padding = skip_padding.expand_as(grad)
    zero_grad = torch.full_like(grad, 0)
    return grad_weight.index_put(
        [indices_rank1], torch.where(skip_padding, grad, zero_grad), accumulate=True
    )


def prod(x: List[int]):
    r = 1
    for i in x:
        r *= i
    return r


@register_decomposition(aten.split_with_sizes)
def split_with_sizes(
    self: Tensor, split_sizes: List[int], dim: int = 0
) -> List[Tensor]:
    num_splits = len(split_sizes)
    splits = []
    start_idx = 0
    for i in range(num_splits):
        length = split_sizes[i]
        splits.append(self.narrow(dim, start_idx, length))
        start_idx += length
    return splits


@register_decomposition(aten.split.Tensor)
def split(self: Tensor, split_size: int, dim: int = 0) -> List[Tensor]:
    input_sizes = self.shape
    dim_size = input_sizes[dim]
    if split_size == 0:
        assert dim_size == 0
        return [self]
    chunks = (dim_size + split_size - 1) // split_size
    split_sizes = [split_size for i in range(chunks)]
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size)
    return torch.split(self, split_sizes, dim)


# TODO: this doesn't appear to have enough precision in bfloat16
@register_decomposition(aten.addmm)
@out_wrapper()
@pw_cast_for_opmath
def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: int = 1, alpha: int = 1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mm(mat1, mat2)
    if beta == 0:
        return out

    # The output of aten.addmm is contiguous, we need to match this behavior in the decomposition.
    # The original implementation 'beta * self + out' would return a strided tensor if `self` is strided.
    # We thus use `out`, the output of torch.mm, which is always contiguous, as the first argument for addition.
    # This is relying on TensorIterator's behavior that it takes higher precedence on the stride of first input.
    # Alternative, we can write `(beta * self + out).contiguous()`, but it introduces another copy in some cases.
    # This implementation is not ideal, and we should revisit this when we have a better solution.
    return out + beta * self


# This computes the mean and variance along the specifized normalization dims,
# then normalizes along those dims. Finally, it returns the mean and variance of
# the normalized dims. Note that it intentionally leaves outputs upcasted.
# Example:
# input: [2, 3, 4, 5], norm_dims: [1, 3]
# mean: [2, 1, 4, 1]
def normalize(input, norm_dims, eps):
    computation_dtype = utils.get_computation_dtype(input.dtype)
    input_acc = input.to(dtype=computation_dtype)
    biased_var = torch.var(input_acc, dim=norm_dims, unbiased=False, keepdim=True)
    mean = torch.mean(input_acc, dim=norm_dims, keepdim=True)
    rstd = torch.rsqrt(biased_var + eps)

    out = (input - mean) * rstd
    return out, mean, rstd


@register_decomposition(aten.native_group_norm_backward)
@pw_cast_for_opmath
def native_group_norm_backward(
    grad_output: Tensor,
    input: Tensor,
    mean: Tensor,
    rstd: Tensor,
    gamma: Optional[Tensor],
    N: int,
    C: int,
    HxW: int,
    group: int,
    output_mask: List[bool],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    utils.check_same_device(
        grad_output, input, mean, rstd, allow_cpu_scalar_tensors=False
    )
    utils.check_same_shape(input, grad_output, allow_cpu_scalar_tensors=False)
    utils.check_same_shape(mean, rstd, allow_cpu_scalar_tensors=False)
    utils.check(
        input.numel() == N * C * HxW,
        lambda: f"Expect input to have { N * C * HxW} elements",
    )
    utils.check(
        mean.shape == (N, group),
        lambda: f"Expect mean to have shape ({N}, {group}, but got {mean.shape}",
    )
    utils.check(
        gamma is None or gamma.numel() == C,
        lambda: f"Expect gamma to have {C} elements but got {gamma.numel() if gamma is not None else -1}",
    )

    cpg, _rem = divmod(C, group)
    utils.check(
        _rem == 0,
        lambda: f"Expect number of channels {C} to be evenly-divisible by number of groups {group}",
    )

    # Compute Internal gradients
    ds = torch.mul(grad_output, input).view(N, C, HxW).sum(dim=[2])
    db = grad_output.view(N, C, HxW).sum(dim=[2])

    d_input: Optional[Tensor] = None
    d_gamma: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    if output_mask[0]:
        s = 1.0 / (HxW * cpg)
        if gamma is not None:
            ds_val = torch.mul(ds, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            db_val = torch.mul(db, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            c1 = torch.mul(
                rstd.unsqueeze(-1),
                gamma.reshape(1, group, cpg),
            )
        else:
            ds_val = ds.reshape(N, group, cpg).sum(2)
            db_val = db.reshape(N, group, cpg).sum(2)
            c1 = torch.mul(
                rstd.unsqueeze(-1),
                torch.ones((1, group, cpg), device=rstd.device),
            )
        c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
        c3 = -c2 * mean - db_val * rstd * s

        c1 = c1.unsqueeze(-1)
        c2 = _unsqueeze_to_dim(c2, 4)
        c3 = _unsqueeze_to_dim(c3, 4)
        d_input = (
            torch.mul(grad_output.reshape(N, group, cpg, HxW), c1)
            + torch.mul(input.reshape(N, group, cpg, HxW), c2)
            + c3
        )
        d_input = d_input.reshape(input.shape).to(input.dtype)
    if output_mask[1]:
        d_gamma = (
            (
                (ds.view(N, group, cpg) - db.view(N, group, cpg) * mean.unsqueeze(-1))
                * rstd.unsqueeze(-1)
            )
            .sum(dim=[0])
            .reshape(C)
        )
    if output_mask[2]:
        d_bias = db.sum(dim=[0])

    return (d_input, d_gamma, d_bias)


def _maybe_cast(x: Optional[Tensor], dtype) -> Optional[Tensor]:
    if x is not None:
        return x.to(dtype)
    return x


# TODO: Take a closer look at the type promotion semantics
@register_decomposition(aten.native_layer_norm_backward)
def native_layer_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: List[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: List[bool],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    input_shape = input.shape
    input_ndim = input.dim()
    computation_dtype = utils.get_computation_dtype(input.dtype)
    grad_out_cast, input_cast, weight_cast, bias_cast = [
        x.to(computation_dtype).contiguous() if x is not None else x
        for x in (grad_out, input, weight, bias)
    ]
    assert grad_out_cast is not None

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices: List[int] = []
    outer_dim_indices: List[int] = []
    for i in range(input_ndim):
        if i >= axis:
            inner_dim_indices.append(i)
        else:
            outer_dim_indices.append(i)

    N = prod(inner_dims)  # type: ignore[arg-type]
    M = prod(outer_dims)  # type: ignore[arg-type]
    if M <= 0 or N <= 0:
        return (
            input.new_zeros(input_shape) if output_mask[0] else None,
            input.new_zeros(input_shape[axis:]) if output_mask[1] else None,
            input.new_zeros(input_shape[axis:]) if output_mask[2] else None,
        )

    x_hat = (input_cast - mean) * rstd
    if weight_cast is not None:
        grad_x_hat = grad_out_cast * weight_cast
    else:
        grad_x_hat = grad_out_cast
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)

    inner = a - b - c3
    d_input: Optional[Tensor] = None
    d_weight: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    if output_mask[0]:
        d_input = (rstd / N) * inner

    if output_mask[1] and weight_cast is not None:
        if len(outer_dim_indices) > 0:
            d_weight = torch.sum(grad_out_cast * x_hat, outer_dim_indices, False)
        else:
            d_weight = grad_out_cast * x_hat

    if output_mask[2] and bias_cast is not None:
        if len(outer_dim_indices) > 0:
            d_bias = torch.sum(grad_out_cast, outer_dim_indices, False)
        else:
            d_bias = grad_out_cast.clone()

    return (
        _maybe_cast(d_input, input.dtype),
        _maybe_cast(d_weight, input.dtype),
        _maybe_cast(d_bias, input.dtype),
    )


@register_decomposition(aten.native_batch_norm)
def native_batch_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    reduction_dims = [0] + list(range(2, input.dim()))
    computation_dtype = utils.get_computation_dtype(input.dtype)
    if training:
        output, mean, rstd = normalize(input, reduction_dims, eps)

        save_mean = _squeeze_multiple(mean, reduction_dims)
        save_rstd = _squeeze_multiple(rstd, reduction_dims)
        if running_mean is not None:
            running_mean.copy_(momentum * save_mean + (1 - momentum) * running_mean)
        if running_var is not None:
            n = input.numel() / input.shape[1]
            # This doesn't strictly match eager's numerics, which accumulates var sum and then directly applies the correction
            # But... that would require re-implementing var here, for negligible numerics gain on a tensor whose
            # numerics probably don't matter.
            unbiased_var = torch.var(input, reduction_dims, unbiased=False) * (
                n / (n - 1)
            )
            running_var.copy_(momentum * unbiased_var + (1 - momentum) * running_var)
    else:
        assert running_mean is not None and running_var is not None
        running_mean = running_mean.to(dtype=computation_dtype, copy=True)
        running_var = running_var.to(dtype=computation_dtype, copy=True)
        mean = running_mean
        invstd = 1 / (torch.sqrt(running_var + eps))
        # Very annoying inconsistency where CPU and CUDA give different shapes
        if input.device.type != "cpu":
            save_mean = running_mean
            save_rstd = invstd
        else:
            save_mean = input.new_zeros((0,))
            save_rstd = input.new_zeros((0,))
        mean = _unsqueeze_to_dim(mean, input.dim() - 1)
        invstd = _unsqueeze_to_dim(invstd, input.dim() - 1)
        output = (input - mean) * invstd

    if weight is None:
        weight = input.new_ones(())

    if bias is None:
        bias = input.new_zeros(())

    weight = _unsqueeze_to_dim(weight, input.dim() - 1)
    bias = _unsqueeze_to_dim(bias, input.dim() - 1)
    output = output * weight + bias
    if input.device.type == "cpu":
        save_mean = save_mean.to(dtype=input.dtype)
        save_rstd = save_rstd.to(dtype=input.dtype)
    return output.to(dtype=input.dtype), save_mean, save_rstd


@register_decomposition(aten._fused_dropout)
@pw_cast_for_opmath
def _fused_dropout_decomposition(input, p, generator=None):
    mask = (torch.rand_like(input) < p).to(dtype=torch.uint8)
    res = mask.type_as(input) * input * (1.0 / p)
    return (res, mask)


@register_decomposition(aten._to_copy)
def _to_copy(
    x: Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    layout=None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    non_blocking: bool = False,
    memory_format: Optional[torch.memory_format] = None,
):
    assert not layout or layout == torch.strided, "TODO"
    assert not pin_memory, "TODO"
    assert device is not None or dtype is not None or memory_format is not None
    dtype_converted = False
    if device is not None and device != x.get_device():
        # avoid conversions on cpu
        if dtype is not None and device.type == "cpu":
            x = torch._prims.convert_element_type(x, dtype)
            dtype_converted = True
        x = torch._prims.device_put(x, device)
    if dtype is not None and not dtype_converted:
        x = torch._prims.convert_element_type(x, dtype)
    if memory_format is not None:  # no ref/prim for memory format
        out = torch.empty_like(x, memory_format=memory_format)
        out.copy_(x)
        return out  # type: ignore[call-overload]
    return x


@pw_cast_for_int_to_real
def xlogy(self: Tensor, other: Tensor) -> Tensor:
    return aten.where(
        aten.isnan(self),
        self,
        aten.where(
            self == aten.new_zeros(self, ()),
            aten.new_zeros(self, ()),
            self * aten.log(other),
        ),
    )


@register_decomposition(aten.std.correction)
@reduction_complex_to_real
def std_decomposition(
    x: Tensor,
    dim: Optional[List[int]],
    correction: Optional[int] = None,
    keepdim: bool = False,
):
    return torch.sqrt(torch.var(x, dim, correction=correction, keepdim=keepdim))


# Questionable decompositions
# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition([aten.detach, aten.lift, aten.lift_fresh])
def nop_decomposition(x):
    return aten.alias(x)


@register_decomposition(aten.cudnn_batch_norm)
def cudnn_batch_norm(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    exponential_average_factor: float,
    epsilon: float,
):
    a, b, c = aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        exponential_average_factor,
        epsilon,
    )
    # Cudnn return running mean and variance when training is True
    if training:
        return (a, b, c, input.new_zeros((0,), dtype=torch.uint8))
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
        input.new_zeros((0,), dtype=torch.uint8),
    )


def _broadcast_batch_norm_backward(x, broadcast_mask):
    for axis, mask in enumerate(broadcast_mask):
        if mask == 1 and not (axis < x.ndim and x.shape[axis] == broadcast_mask[axis]):
            x = x.unsqueeze(axis)
    return x


@register_decomposition(aten.native_batch_norm_backward)
def native_batch_norm_backward(
    grad_out: Tensor,
    input: Tensor,
    weight: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_invstd: Optional[Tensor],
    train: bool,
    eps: float,
    output_mask: List[bool],
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    input_dtype = input.dtype
    computation_dtype = utils.get_computation_dtype(input.dtype)
    (
        grad_out_cast,
        input_cast,
        weight_cast,
        running_mean_cast,
        running_var_cast,
        save_mean_cast,
        save_invstd_cast,
    ) = [
        x.to(computation_dtype) if x is not None else x
        for x in (
            grad_out,
            input,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
        )
    ]
    input_shape = input.shape
    input_rank = input.dim()
    assert input_rank >= 2, "rank of the input must be at least 2"

    axis = 1
    num_features = prod(list(input_shape)) / input_shape[axis]
    mean = save_mean_cast
    invstd = save_invstd_cast
    if train:
        assert save_mean_cast is not None and save_invstd_cast is not None
    else:
        assert running_mean_cast is not None and running_var_cast is not None
        mean = running_mean_cast
        invstd = torch.rsqrt(running_var_cast + eps)

    broadcast_mask: List[int] = [1] * input_rank
    broadcast_mask[axis] = input_shape[axis]

    reduction_axes: List[int] = []
    for i in range(input_rank):
        if i != axis:
            reduction_axes.append(i)

    mean = _broadcast_batch_norm_backward(mean, broadcast_mask)  # type: ignore[arg-type]
    norm = 1.0 / num_features
    grad_output_sum = torch.sum(grad_out_cast, reduction_axes)  # type: ignore[arg-type]
    dot_p = torch.sum(grad_out_cast * (input_cast - mean), reduction_axes)  # type: ignore[operator]

    grad_mean = _broadcast_batch_norm_backward(grad_output_sum * norm, broadcast_mask)
    proj_scale = _broadcast_batch_norm_backward(torch.mul(dot_p * norm, invstd * invstd), broadcast_mask)  # type: ignore[operator]

    if weight_cast is None:
        grad_scale = _broadcast_batch_norm_backward(invstd, broadcast_mask) * 1.0  # type: ignore[arg-type]
    else:
        grad_scale = _broadcast_batch_norm_backward(
            invstd * weight_cast, broadcast_mask
        )

    if train:
        proj = (input_cast - mean) * proj_scale  # type: ignore[operator]
        grad_input = ((grad_out_cast - proj) - grad_mean) * grad_scale
    else:
        grad_input = grad_out_cast * grad_scale

    if output_mask[1]:
        grad_weight = dot_p * invstd
    else:
        grad_weight = None  # "None" doesn't work with vjp, should use zeros for vjp

    if output_mask[2]:
        grad_bias = grad_output_sum
    else:
        grad_bias = None  # "None" doesn't work with vjp, should use zeros for vjp

    return (
        grad_input.to(input_dtype),
        _maybe_cast(grad_weight, input_dtype),
        _maybe_cast(grad_bias, input_dtype),
    )


@register_decomposition(aten.cudnn_batch_norm_backward)
def cudnn_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_var: Optional[Tensor],
    epsilon: float,
    reserveSpace: Tensor,
):
    return aten.native_batch_norm_backward(
        grad_output,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        epsilon,
        [True, True, True],
    )


@register_decomposition(aten._adaptive_avg_pool2d)
@pw_cast_for_opmath
def adaptive_avg_pool2d(input: Tensor, output_size: Tuple[int, int]):
    # Preconditions
    device = input.device
    shape = input.shape
    ndim = len(shape)
    utils.check(
        ndim in (3, 4),
        lambda: f"adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got {ndim}",
    )
    for d in input.shape[-2:]:
        utils.check(
            d != 0,
            lambda: "adaptive_avg_pool2d(): Expected input to have non-zero size for "
            f"non-batch dimensions, but input has shape {tuple(shape)}.",
        )

    # Optimisation (we should also do this in the kernel implementation)
    if shape[-2] % output_size[-2] == 0 and shape[-1] % output_size[-1] == 0:
        stride = tuple(i // o for i, o in zip(shape[-2:], output_size))
        kernel = tuple(
            i - (o - 1) * s for i, o, s in zip(shape[-2:], output_size, stride)
        )
        return torch.nn.functional.avg_pool2d(input, kernel, stride)

    def start_index(a, b, c):
        return torch.div(a * c, b, rounding_mode="trunc")

    def end_index(a, b, c):
        return torch.div((a + 1) * c + b - 1, b, rounding_mode="trunc")

    def compute_idx(in_size, out_size):
        orange = torch.arange(out_size, device=device, dtype=torch.int64)
        i0 = start_index(orange, out_size, in_size)
        # Let length = end_index - start_index, i.e. the length of the pooling kernels
        # length.max() can be computed analytically as follows:
        maxlength = in_size // out_size + 1
        in_size_mod = in_size % out_size
        # adaptive = True iff there are kernels with different lengths
        adaptive = not (in_size_mod == 0 or out_size % in_size_mod == 0)
        if adaptive:
            maxlength += 1
        elif in_size_mod == 0:
            maxlength -= 1

        range_max = torch.arange(maxlength, device=device, dtype=torch.int64)
        idx = i0.unsqueeze(-1) + range_max
        if adaptive:
            # Need to clamp to avoid accesing out-of-bounds memory
            # TODO make minimum accept scalars
            maxval = torch.scalar_tensor(
                in_size - 1, dtype=idx.dtype, device=idx.device
            )
            idx = torch.minimum(idx, maxval)

            # Compute the lenghts
            i1 = end_index(orange, out_size, in_size)
            length = i1 - i0
        else:
            length = maxlength
        return idx, length, range_max, adaptive

    # length is not None if it's constant, otherwise we'll need to compute it
    idxh, length_h, range_max_h, adaptive_h = compute_idx(shape[-2], output_size[-2])
    idxw, length_w, range_max_w, adaptive_w = compute_idx(shape[-1], output_size[-1])

    vals = input[..., _unsqueeze_to_dim(idxh, 4), idxw]
    # Shortcut for the simpler case
    if not adaptive_h and not adaptive_w:
        return torch.mean(vals, dim=(-3, -1))

    def maybe_mask(vals, length, range_max, adaptive, dim):
        if isinstance(length, IntLike):
            return vals, length
        else:
            # zero-out the things we didn't really want to select
            assert dim < 0
            # hack
            mask = range_max >= length.unsqueeze(-1)
            if dim == -2:
                mask = _unsqueeze_to_dim(mask, 4)
            vals = torch.masked_fill(vals, mask, 0.0)
            # Compute the length of each window
            length = _unsqueeze_to_dim(length, -dim)
            return vals, length

    vals, length_h = maybe_mask(
        vals, length_h, range_max_h, adaptive=adaptive_h, dim=-2
    )
    vals, length_w = maybe_mask(
        vals, length_w, range_max_w, adaptive=adaptive_w, dim=-1
    )

    # We unroll the sum as we assume that the kernels are going to be small
    ret = None
    for i, j in product(range(vals.shape[-3]), range(vals.shape[-1])):
        if ret is None:
            ret = vals[..., i, :, j]
        else:
            ret = ret + vals[..., i, :, j]
    return ret / (length_h * length_w)


@register_decomposition(aten.index_add_)
def index_add_(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    dim = utils.canonicalize_dims(x.ndim, dim)
    utils.check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    if alpha != 1:
        python_type = utils.dtype_to_type(x.dtype)
        utils.check(
            python_type == bool
            or utils.is_weakly_lesser_type(type(alpha), python_type),
            lambda: f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!",
        )
        tensor = tensor * alpha
    idx = (None,) * dim + (index,)
    torch.ops.aten.index_put_(x, idx, tensor, accumulate=True)
    return x


def _squeeze_multiple(self: Tensor, dims: List[int]) -> Tensor:
    ndim = self.dim()
    wrapped_dims = utils.canonicalize_dims(ndim, dims)
    assert isinstance(wrapped_dims, tuple)
    for idx in range(ndim - 1, -1, -1):
        if idx in wrapped_dims:
            self = self.squeeze(idx)
    return self


@register_decomposition(aten.logsumexp.default)
@pw_cast_for_int_to_real
def logsumexp(self: Tensor, dim: List[int], keepdim: bool = False) -> Tensor:
    if self.numel() == 0:
        return torch.sum(torch.exp(self), dim, keepdim).log()
    maxes = torch.amax(self, dim, keepdim=True)
    maxes_squeezed = maxes if keepdim else _squeeze_multiple(maxes, dim)
    maxes_squeezed = torch.masked_fill(
        maxes_squeezed, maxes_squeezed.abs() == float("inf"), 0
    )
    result = torch.sum(torch.exp(self - maxes), dim, keepdim)
    return result.log().add(maxes_squeezed)


# nb: Should use acc_t, not op_math
@register_decomposition(aten.log_sigmoid_forward)
@out_wrapper("output", "buffer")
@pw_cast_for_opmath
def log_sigmoid_forward(self: Tensor) -> Tuple[Tensor, Tensor]:
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer


@register_decomposition(aten.norm)
@out_wrapper()
@reduction_complex_to_real
def norm(
    self: Tensor,
    p: Optional[float] = None,
    dim: List[int] = None,
    keepdim: bool = False,
    dtype: Optional[torch.dtype] = None,
):
    if p is None:
        p = 2.0
    return torch.linalg.vector_norm(self, p, dim, keepdim, dtype=dtype)


@register_decomposition(torch.ops.aten.upsample_bilinear2d.vec)
@register_decomposition(torch.ops.aten.upsample_bilinear2d.vec, type="pre_autograd")
@pw_cast_for_opmath
def upsample_bilinear2d_vec(
    input: Tensor,
    output_size: Optional[List[int]],
    align_corners: bool,
    scale_factors: Optional[List[float]],
) -> Tensor:
    # get dimensions of original image
    n_batch, n_channels, in_h, in_w = input.shape

    if output_size is not None:
        out_h = float(output_size[0])
        out_w = float(output_size[1])
    elif scale_factors is not None:
        out_h = in_h * scale_factors[0]
        out_w = in_w * scale_factors[1]

    # Calculate horizontal and vertical scaling factor
    if out_h > 1:
        if align_corners:
            h_scale_factor = (in_h - 1) / (int(out_h) - 1)
        else:
            h_scale_factor = in_h / out_h
    else:
        h_scale_factor = 0.0

    if out_w > 1:
        if align_corners:
            w_scale_factor = (in_w - 1) / (int(out_w) - 1)
        else:
            w_scale_factor = in_w / out_w
    else:
        w_scale_factor = 0.0

    i = torch.arange(int(out_h), dtype=input.dtype, device=input.device)
    j = torch.arange(int(out_w), dtype=input.dtype, device=input.device)

    if align_corners:
        x = h_scale_factor * i
        y = w_scale_factor * j
    else:
        x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
        y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)

    x_floor = torch.floor(x).to(torch.int64)
    x_ceil = torch.ceil(x).clamp(max=in_h - 1).to(torch.int64)
    y_floor = torch.floor(y).to(torch.int64)
    y_ceil = torch.ceil(y).clamp(max=in_w - 1).to(torch.int64)

    x_view = x.unsqueeze(1)
    x_floor_view = x_floor.unsqueeze(1)
    x_ceil_view = x_ceil.unsqueeze(1)

    v1 = input[:, :, x_floor_view, y_floor]
    v2 = input[:, :, x_ceil_view, y_floor]
    v3 = input[:, :, x_floor_view, y_ceil]
    v4 = input[:, :, x_ceil_view, y_ceil]

    xscale2 = x_view - x_floor_view
    xscale1 = 1.0 - xscale2

    yscale2 = y - y_floor
    yscale1 = 1.0 - yscale2

    q1 = torch.mul(v1, xscale1) + torch.mul(v2, xscale2)
    q2 = torch.mul(v3, xscale1) + torch.mul(v4, xscale2)
    result = torch.mul(q1, yscale1) + torch.mul(q2, yscale2)

    # convert output to correct memory format, if necessary
    input_memory_format = utils.suggest_memory_format(input)
    result = result.contiguous(memory_format=input_memory_format)

    return result


# We should be applying decompositions after all transformations
@register_decomposition(aten.is_same_size.default)
def is_same_size(a: Tensor, b: Tensor) -> bool:
    return a.shape == b.shape


@register_decomposition([aten._reshape_alias, aten._unsafe_view])
def _reshape_alias(x, shape, *args):
    return aten.view(x, shape)


@register_decomposition(aten.nll_loss_forward)
def nll_loss_forward(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> Tuple[Tensor, Tensor]:
    assert self.dim() > 0 and self.dim() <= 2, "input tensor should be 1D or 2D"
    assert (
        target.dim() <= 1
    ), "0D or 1D target tensor expected, multi-target not supported"

    no_batch_dim = self.dim() == 1 and target.dim() == 0
    assert no_batch_dim or (
        self.shape[0] == target.shape[0]
    ), f"size mismatch (got input: {self.shape}, target: {target.shape})"

    n_classes = self.shape[-1]

    assert weight is None or (
        weight.dim() == 1 and weight.numel() == n_classes
    ), f"weight tensor should be defined either for all {n_classes} classes or no classes but got weight tensor of shape: {weight.shape}"  # noqa: B950

    # self can be [N, C] or [C]
    # target can be [N] or []

    n_dims = self.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    if weight is not None:
        w = weight.unsqueeze(0) if n_dims > 1 else weight
        self = self * w

    target_ = target.unsqueeze(channel_dim)
    # target can be [N, 1] or [1]

    result = -torch.gather(self, channel_dim, target_).squeeze(channel_dim)

    if ignore_index >= 0:
        result = torch.where(target != ignore_index, result, 0)

    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = self.new_full((), 0.0)
        return result, total_weight

    if weight is not None:
        w = weight.unsqueeze(0).expand(self.shape) if n_dims > 1 else weight
        wsum = torch.gather(w, channel_dim, target_).squeeze(channel_dim)
        if ignore_index >= 0:
            wsum = torch.where(target != ignore_index, wsum, 0)
        total_weight = wsum.sum()
    elif ignore_index >= 0:
        total_weight = (target != ignore_index).sum().to(self)
    else:
        total_weight = self.new_full((), 1.0 * result.numel())

    if reduction == Reduction.SUM.value:
        result = result.sum()
    elif reduction == Reduction.MEAN.value:
        if weight is None:
            result = result.sum() / total_weight if ignore_index >= 0 else result.mean()
        else:
            result = result.sum() / total_weight

    return result, total_weight


# These are adapted from aten/src/ATen/native/UpSample.h, wich is based on
# https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
def _upsample_cubic_convolution1(x: Tensor, A: float) -> Tensor:
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _upsample_cubic_convolution2(x: Tensor, A: float) -> Tensor:
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


def _upsample_get_cubic_coefficients(t: Tensor) -> TensorSequenceType:
    A = -0.75
    return (
        _upsample_cubic_convolution2(t + 1.0, A),
        _upsample_cubic_convolution1(t, A),
        _upsample_cubic_convolution1(1.0 - t, A),
        _upsample_cubic_convolution2(2.0 - t, A),
    )


def _upsample_cubic_interp1d(coeffs: TensorSequenceType, ts: Tensor) -> Tensor:
    coeffs2 = _upsample_get_cubic_coefficients(ts)
    return _sum_tensors(c1 * c2 for (c1, c2) in zip(coeffs, coeffs2))


# Need this instead of just sum() to keep mypy happy
def _sum_tensors(ts: Iterable[Tensor]) -> Tensor:
    return reduce(torch.add, ts)


@register_decomposition(aten.grid_sampler_2d)
@pw_cast_for_opmath
def grid_sampler_2d(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> Tensor:
    utils.check(
        interpolation_mode in (0, 1, 2),
        lambda: f"Invalid interpolation mode {interpolation_mode}",
    )
    utils.check(
        padding_mode in (0, 1, 2), lambda: f"Invalid padding mode {padding_mode}"
    )

    def unnormalize(coords: Tensor, size: int) -> Tensor:
        # Rescale coordinates from [-1, 1] to:
        #   [0, size - 1] if align_corners is True
        #   [-.5, size -.5] if align_corners is False
        mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
        ofs = size * 0.5 - 0.5
        return coords * mul + ofs

    # Reflects coordinates until they fall between low and high (inclusive).
    # The bounds are passed as twice their value so that half-integer values
    # can be represented as ints.
    def reflect_coordinates(coords: Tensor, twice_low: int, twice_high: int) -> Tensor:
        if twice_low == twice_high:
            return torch.zeros_like(coords)
        coords_min = twice_low / 2
        coords_span = (twice_high - twice_low) / 2
        coords2 = (coords - coords_min).abs()
        extra = torch.fmod(coords2, coords_span)
        flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
        return torch.where(
            flips & 1 == 0, extra + coords_min, coords_span + coords_min - extra
        )

    def compute_coordinates(coords: Tensor, size: int) -> Tensor:
        if padding_mode == 0:  # Zero
            return coords
        elif padding_mode == 1:  # Borders
            return torch.clamp(coords, 0, size - 1)
        else:  # padding_mode == 2, Reflection
            if align_corners:
                coords_reflected = reflect_coordinates(coords, 0, 2 * (size - 1))
            else:
                coords_reflected = reflect_coordinates(coords, -1, 2 * size - 1)
            return torch.clamp(coords_reflected, 0, size - 1)

    def compute_source_index(coords: Tensor, size: int) -> Tensor:
        coords_un = unnormalize(coords, size)
        return compute_coordinates(coords_un, size)

    N, C, iH, iW = a.shape
    _, oH, oW, _ = grid.shape

    def in_bounds_cond(xs: Tensor, ys: Tensor) -> Tensor:
        return torch.logical_and(
            0 <= xs, torch.logical_and(xs < iW, torch.logical_and(0 <= ys, ys < iH))
        )

    N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1)
    C_idx = torch.arange(C, device=a.device).view(1, C, 1, 1)

    def clip(xs: Tensor, ys: Tensor, ws: Tensor) -> TensorSequenceType:
        cond = in_bounds_cond(xs, ys)
        # To clip to inside valid coordinates, we map the coordinates
        # to (x, y) = (0, 0) and also set the weight to 0
        # We also change the shape of the tensor to the appropriate one for
        # broadcasting with N_idx, C_idx for the purposes of advanced indexing
        return tuple(
            torch.where(cond, t, 0).view(N, 1, oH, oW)
            for t in (xs.to(dtype=torch.int64), ys.to(dtype=torch.int64), ws)
        )

    def get_summand(ix: Tensor, iy: Tensor, w) -> Tensor:
        # Perform clipping, index into input tensor and multiply by weight
        idx_x, idx_y, w_ = clip(ix, iy, w)
        return a[N_idx, C_idx, idx_y, idx_x] * w_

    x = grid[..., 0]
    y = grid[..., 1]

    if interpolation_mode == 0:  # Bilinear
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nw, iy_nw = ix.floor(), iy.floor()
        ix_ne, iy_ne = ix_nw + 1, iy_nw
        ix_sw, iy_sw = ix_nw, iy_nw + 1
        ix_se, iy_se = ix_ne, iy_sw

        w_nw = (ix_se - ix) * (iy_se - iy)
        w_ne = (ix - ix_sw) * (iy_sw - iy)
        w_sw = (ix_ne - ix) * (iy - iy_ne)
        w_se = (ix - ix_nw) * (iy - iy_nw)

        return _sum_tensors(
            get_summand(ix, iy, w)
            for (ix, iy, w) in (
                (ix_nw, iy_nw, w_nw),
                (ix_ne, iy_ne, w_ne),
                (ix_sw, iy_sw, w_sw),
                (ix_se, iy_se, w_se),
            )
        )
    elif interpolation_mode == 1:  # Nearest
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nearest = ix.round()
        iy_nearest = iy.round()

        return get_summand(ix_nearest, iy_nearest, 1)
    else:  # interpolation_mode == 2, Bicubic
        ix = unnormalize(x, iW)
        iy = unnormalize(y, iH)

        ix_nw = ix.floor()
        iy_nw = iy.floor()

        tx = ix - ix_nw
        ty = iy - iy_nw

        def get_value_bounded(ix: Tensor, iy: Tensor) -> Tensor:
            x = compute_coordinates(ix, iW)
            y = compute_coordinates(iy, iH)
            return get_summand(x, y, 1)

        def get_coeff(ofs: int) -> Tensor:
            iy_ofs = iy_nw + (ofs - 1)
            cs = (
                get_value_bounded(ix_nw - 1, iy_ofs),
                get_value_bounded(ix_nw, iy_ofs),
                get_value_bounded(ix_nw + 1, iy_ofs),
                get_value_bounded(ix_nw + 2, iy_ofs),
            )
            return _upsample_cubic_interp1d(cs, tx.unsqueeze(1))

        coeffs = tuple((get_coeff(ofs) for ofs in range(4)))
        return _upsample_cubic_interp1d(coeffs, ty.unsqueeze(1))


@register_decomposition(aten.mv)
@out_wrapper()
@pw_cast_for_opmath
def mv(self, vec):
    utils.check(
        self.dim() == 2 and vec.dim() == 1,
        lambda: f"matrix @ vector expected, got {self.dim()}, {vec.dim()}",
    )
    utils.check(
        self.size(1) == vec.size(0),
        lambda: f"size mismatch, got {self.size(0)}x{self.size(1)},{vec.size(0)}",
    )
    return (self * vec).sum(dim=1)


@register_decomposition(aten.dot)
@out_wrapper()
@pw_cast_for_opmath
def dot(self, other):
    if self.is_complex():
        if self.is_conj():
            if other.is_conj():
                return torch.dot(self.conj(), other.conj()).conj()
            else:
                return torch.vdot(self.conj(), other)
        elif other.is_conj():
            return torch.vdot(other.conj(), self)

    utils.check(
        self.dim() == 1 and other.dim() == 1,
        lambda: f"1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors",
    )
    utils.check(
        self.dtype == other.dtype,
        lambda: f"dot : expected both vectors to have same dtype, but found {self.dtype} and {other.dtype}",
    )

    def numel_error():
        return (
            f"inconsistent tensor size, expected tensor [{self.numel()}] and src [{other.numel()}] to have the"
            f"same number of elements, but got {self.numel()} and {other.numel()} elements respectively"
        )

    utils.check(self.numel() == other.numel(), numel_error)

    return (self * other).sum()


@register_decomposition(aten.binary_cross_entropy_with_logits)
def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=Reduction.MEAN.value
):
    max_val = (-self).clamp_min(0)
    if pos_weight is not None:
        log_weight = (pos_weight - 1) * target + 1
        loss = (1 - target) * self + log_weight * (
            ((-max_val).exp() + (-self - max_val).exp()).log() + max_val
        )
    else:
        loss = (
            (1 - target) * self
            + max_val
            + ((-max_val).exp() + (-self - max_val).exp()).log()
        )

    if weight is not None:
        loss = loss * weight

    return apply_loss_reduction(loss, reduction)


def should_fold(tensor1: torch.Tensor, dim_tensor2: int) -> bool:
    dim_tensor1 = tensor1.ndim
    if dim_tensor1 >= 3 and (dim_tensor2 == 1 or dim_tensor2 == 2):
        t1_sizes_ptr = tensor1.shape
        t1_strides = tensor1.stride()
        if (
            dim_tensor1 == 3
            and dim_tensor2 == 2
            and t1_strides[-1] != 1
            and t1_strides[0] == t1_sizes_ptr[1] * t1_sizes_ptr[2]
        ):
            # First dim is slowest moving, and then the following two dims are
            # transposed. This can happen for example by permute(0, 2, 1).
            # First 2 dims could be folded to use mm but would require permutation
            # with actual data movement, which can be instead handled by BMM with each
            # GEMM transposed.
            # This can be generalized to a tensor with dim X + Y + Z where X, Y, and Z
            # dims are contiguous, Y dims and Z dims are transposed, and X, Y, Z > 0.
            # For example, this can happen by permute(0, 1, 5, 2, 3, 4), where X = 2,
            # Y = 3, and Z = 1.
            return False
        else:
            return True
    else:
        return False


@torch.ops.aten.matmul.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def matmul(tensor1, tensor2):
    dim_tensor1 = tensor1.dim()
    dim_tensor2 = tensor2.dim()
    assert dim_tensor1 != 0 and dim_tensor2 != 0
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        return torch.dot(tensor1, tensor2)
    elif dim_tensor1 == 2 and dim_tensor2 == 1:
        return torch.mv(tensor1, tensor2)
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        return torch.squeeze(torch.mm(torch.unsqueeze(tensor1, 0), tensor2), 0)
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        # if tensor1.shape[1] != tensor2.shape[0]:
        #     breakpoint()
        return torch.mm(tensor1, tensor2)
    elif should_fold(tensor1, dim_tensor2) or should_fold(tensor2, dim_tensor1):
        # NB: Much of this was written with Copilot! (although still had to fix a bunch of issues)

        # dim_tensor1 >=3 && (dim_tensor2 == 1 || dim_tensor2 == 2) ||
        # dim_tensor2 >=3 && (dim_tensor1 == 1 || dim_tensor1 == 2)
        # and some condition on the strides is fulfilled

        # optimization: use mm instead of bmm by folding the batch of the larger tensor
        # into its leading matrix dimension
        transpose = dim_tensor2 > dim_tensor1
        t1 = tensor2.mT if transpose else tensor1
        t2 = (
            tensor2 if not transpose else (tensor1.t() if dim_tensor1 == 2 else tensor1)
        )
        # Invariant: t1.dim() >= 3 && (t2.dim() == 1 || t2.dim() == 2)
        #            and t1 and t2 are matmul-compatible

        # Why not t1.view(-1, sizes_1[-1])?
        # If the last dim is 0, then view(-1, 0) won't work because the -1 becomes ambiguous.
        # This can happen in e.g. [3, 5, 0] @ [0, 0].
        sizes_1 = t1.shape
        output_shape = list(sizes_1[:-1])
        folded_dim1 = reduce(operator.mul, output_shape)

        # Readjust output_shape if we are multiplying by a matrix
        t2_is_matrix = t2.dim() == 2
        if t2_is_matrix:
            output_shape.append(t2.shape[1])
        t1_folded = t1.reshape(folded_dim1, sizes_1[-1])
        if t2_is_matrix:
            # FIXME This path always does an unnecessary copy when transpose == True as the returned
            # result from BLAS is already C-transposed
            output = t1_folded.mm(t2).view(output_shape)
            return output.mT.contiguous() if transpose else output
        else:
            return t1_folded.mv(t2).view(output_shape)

    elif dim_tensor1 >= 1 and dim_tensor2 >= 1:
        # We are multiplying b1 x n x m1 by x2 x m2 x p (where b1 can be a list);
        # we track m1 vs m2 separately even though they must match for nicer error messages
        n = tensor1.size(-2) if dim_tensor1 > 1 else 1
        m1 = tensor1.size(-1)
        batch_tensor1 = tensor1.shape[:-2]
        m2 = tensor2.size(-2) if dim_tensor2 > 1 else tensor2.size(-1)
        p = tensor2.size(-1) if dim_tensor2 > 1 else 1
        batch_tensor2: List[int] = []
        # TODO: handling of slice
        for i in range(dim_tensor2 - 2):
            batch_tensor2.append(tensor2.size(i))

        # expand the batch portion (i.e. cut off matrix dimensions and expand rest)
        expand_batch_portion = list(
            torch.broadcast_shapes(batch_tensor1, batch_tensor2)
        )

        tensor1_expand_size = expand_batch_portion + [n, m1]
        tensor2_expand_size = expand_batch_portion + [m2, p]

        expand_batch_product = prod(expand_batch_portion)

        # HACK: We need reshape with symint support
        tensor1_expanded = tensor1.expand(tensor1_expand_size).reshape(
            expand_batch_product, n, m1
        )
        tensor2_expanded = tensor2.expand(tensor2_expand_size).reshape(
            expand_batch_product, m2, p
        )

        output_shape = expand_batch_portion
        if dim_tensor1 > 1:
            output_shape.append(n)

        if dim_tensor2 > 1:
            output_shape.append(p)

        return tensor1_expanded.bmm(tensor2_expanded).view(output_shape)
    else:
        utils.check(False, lambda: "both arguments to matmul need to be at least 1D")


@register_decomposition(aten.upsample_bicubic2d.default)
@pw_cast_for_opmath
def upsample_bicubic2d_default(
    a: Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scale_h: Optional[float] = None,
    scale_w: Optional[float] = None,
) -> Tensor:
    N, C, iH, iW = a.shape
    oH, oW = output_size

    def compute_scale(in_size, out_size, align_corners, scale=None):
        if align_corners:
            return (in_size - 1) / (out_size - 1) if out_size > 1 else 0
        else:
            return 1 / scale if scale is not None and scale > 0 else in_size / out_size

    def compute_source_index(scale, dst_index, align_corners):
        if align_corners:
            return scale * dst_index
        else:
            return scale * (dst_index + 0.5) - 0.5

    height_scale = compute_scale(iH, oH, align_corners, scale_h)
    width_scale = compute_scale(iW, oW, align_corners, scale_w)

    N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1)
    C_idx = torch.arange(C, device=a.device).view(1, C, 1, 1)
    out_y = torch.arange(oH, device=a.device).view((1, 1, oH, 1))
    out_x = torch.arange(oW, device=a.device).view((1, 1, 1, oW))

    real_x = compute_source_index(width_scale, out_x, align_corners)
    in_x = real_x.floor()
    t_x = real_x - in_x
    ix = in_x.to(dtype=torch.int64)

    real_y = compute_source_index(height_scale, out_y, align_corners)
    in_y = real_y.floor()
    t_y = real_y - in_y
    iy = in_y.to(dtype=torch.int64)

    iys_ofs = (iy - 1, iy, iy + 1, iy + 2)
    ixs_ofs = (ix - 1, ix, ix + 1, ix + 2)

    def load_bounded(ys, xs):
        y_idx = torch.clamp(ys, 0, iH - 1)
        x_idx = torch.clamp(xs, 0, iW - 1)
        return a[N_idx, C_idx, y_idx, x_idx]

    def get_x_interp(y):
        coeffs_x = tuple((load_bounded(y, x_ofs) for x_ofs in ixs_ofs))
        return _upsample_cubic_interp1d(coeffs_x, t_x)

    coeffs_y = tuple((get_x_interp(y_ofs) for y_ofs in iys_ofs))
    return _upsample_cubic_interp1d(coeffs_y, t_y)


@register_decomposition(aten.upsample_bicubic2d.vec)
@out_wrapper()
@pw_cast_for_opmath
def upsample_bicubic2d_vec(
    a: Tensor,
    output_size: Optional[Tuple[int, int]],
    align_corners: bool,
    scale_factors: Optional[Tuple[float, float]] = None,
) -> Tensor:
    utils.check(
        bool(output_size) + bool(scale_factors) == 1,
        lambda: "Must specify exactly one of output_size and scale_factors.",
    )
    if output_size is None:
        assert scale_factors is not None
        output_size = cast(
            Tuple[int, int],
            tuple(int(w * scale) for w, scale in zip(a.shape[2:], scale_factors)),
        )
    scale_h, scale_w = scale_factors if scale_factors else (None, None)
    return upsample_bicubic2d_default(a, output_size, align_corners, scale_h, scale_w)


def register_inplace(aten_op, outplace_op):
    @register_decomposition(aten_op)
    def inplace_op(*args, **kwargs):
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


register_inplace(aten.add_, aten.add)
register_inplace(aten.sub_, aten.sub)
register_inplace(aten.mul_, aten.mul)
register_inplace(aten.relu_, aten.relu)
register_inplace(aten.hardtanh_, aten.hardtanh)
register_inplace(aten.hardswish_, aten.hardswish)
register_inplace(aten.leaky_relu_, aten.leaky_relu)
register_inplace(aten.silu_, aten.silu)
