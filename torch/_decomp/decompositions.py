import functools
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union

import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _safe_copy_out,
    out_wrapper,
)
from torch.fx.experimental.symbolic_shapes import expect_true, guard_int
from torch.utils._pytree import tree_flatten, tree_map

DispatchKey = torch._C.DispatchKey  # type: ignore[attr-defined]

# None of these functions are publicly accessible; get at them
# from torch._decomps
__all__: List[str] = []

aten = torch._ops.ops.aten


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
pw_cast_for_int_to_real = partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
)


# This expands x until x.dim() == dim. Might be useful as an operator
def _unsqueeze_to_dim(x: Tensor, dim: int) -> Tensor:
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
            grad_output * poscoef,
        )
    else:
        return torch.where(
            self_or_result <= 0,
            grad_output * negiptcoef * negcoef * torch.exp(self_or_result * negiptcoef),
            grad_output * poscoef,
        )


@register_decomposition([aten.fill.Scalar])
def fill_scalar(self, value):
    return torch.full_like(self, value)


@register_decomposition([aten.fill.Tensor])
def fill_tensor(self, value: Tensor):
    torch._check(
        value.dim() == 0,
        lambda: f"fill only supports 0-dimension value tensor but got tensor with {value.dim()} dimensions",
    )
    return aten.copy(self, value)


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
def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
):
    return torch.where((self <= min_val) | (self >= max_val), 0.0, grad_output)


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


@register_decomposition(aten._prelu_kernel)
def _prelu_kernel(self: Tensor, weight: Tensor) -> Tensor:
    return torch.where(self > 0, self, weight * self)


@register_decomposition(aten._prelu_kernel_backward)
def _prelu_kernel_backward(
    grad_output: Tensor,
    self: Tensor,
    weight: Tensor,
) -> Tuple[Tensor, Tensor]:
    input_grad = torch.where(self > 0, grad_output, weight * grad_output)
    weight_grad = torch.where(self > 0, 0.0, self * grad_output)
    return (input_grad, weight_grad)


@register_decomposition(aten.rrelu_with_noise)
@aten.rrelu_with_noise.default.py_impl(DispatchKey.AutogradCUDA)
@pw_cast_for_opmath
def rrelu_with_noise(
    self: Tensor,
    noise: Tensor,
    lower: float,
    upper: float,
    training: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    assert generator is None
    if training:
        not_positive = self <= 0
        r = aten.uniform(self, lower, upper)
        output = torch.where(not_positive, self * r, self)
        noise.copy_(torch.where(not_positive, r, 1))
        return output
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu(self, negative_slope)


@register_decomposition(aten.rrelu_with_noise_)
@aten.rrelu_with_noise_.default.py_impl(DispatchKey.AutogradCUDA)
@pw_cast_for_opmath
def rrelu_with_noise_(
    self: Tensor,
    noise: Tensor,
    lower: float,
    upper: float,
    training: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    return self.copy_(rrelu_with_noise(self, noise, lower, upper, training, generator))


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


@register_decomposition(aten.smooth_l1_loss)
@pw_cast_for_opmath
def smooth_l1_loss(
    self: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
    beta: float = 1.0,
):
    loss = (self - target).abs()
    loss = torch.where(loss < beta, 0.5 * loss**2 / beta, loss - 0.5 * beta)
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.smooth_l1_loss_backward.default)
@pw_cast_for_opmath
def smooth_l1_loss_backward(
    grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, beta: float
):
    norm = 1.0 / self.numel() if reduction == Reduction.MEAN.value else 1.0
    x = self - target
    abs_x = torch.abs(x)
    norm_grad = norm * grad_output
    return torch.where(
        abs_x < beta,
        norm_grad * x / beta,
        norm_grad * torch.sign(x),
    )


@register_decomposition(aten.smooth_l1_loss_backward.grad_input)
@pw_cast_for_opmath
def smooth_l1_loss_backward_out(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int,
    beta: float,
    grad_input: Tensor,
):
    result = smooth_l1_loss_backward(grad_output, self, target, reduction, beta)
    _maybe_resize_out(grad_input, result.shape)
    return _safe_copy_out(copy_from=result, copy_to=grad_input, exact_dtype=True)


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
    safe_target = torch.where(target != ignore_index, target, 0)
    grad_input = torch.zeros_like(self)
    grad_input = torch.scatter(grad_input, channel_dim, safe_target, -1.0)

    if grad_input.dim() > grad_output.dim() > 0:
        grad_output = grad_output.unsqueeze(channel_dim)

    if weight is not None:
        new_shape = [1 for _ in range(self.dim())]
        new_shape[channel_dim] = weight.shape[0]
        weight = weight.reshape(new_shape)
        grad_output = grad_output * weight

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
        torch.log1p(-self), self.new_full((), -100)
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


@register_decomposition(aten.dist)
def dist(input: Tensor, other: Tensor, p: float = 2):
    return aten.norm(input - other, p=p)


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
    elif start_val > sizes[dim]:
        start_val = sizes[dim]

    if end_val < start_val:
        end_val = start_val
    elif end_val > sizes[dim]:
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

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()

    return _cast_grad_to_input_dtype(grad_output, grad_input, input_dtype).contiguous()


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

    # Broadcast and add kernel starting positions (indices) with
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
    torch._check(len(kernel_size) == 2, lambda: "im2col(): only 2D kernel supported")
    torch._check(len(dilation) == 2, lambda: "im2col(): only 2D dilation supported")
    torch._check(len(padding) == 2, lambda: "im2col(): only 2D padding supported")
    torch._check(len(stride) == 2, lambda: "im2col(): only 2D stride supported")

    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        torch._check(
            cond, lambda: "{param_name} should be greater {'than' zero, but got {param}"
        )

    check_positive(kernel_size, "kernel_size")
    check_positive(dilation, "dilation")
    check_positive(dilation, "padding", strict=False)
    check_positive(stride, "stride")

    shape = input.shape
    ndim = len(shape)
    torch._check(
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
    torch._check(
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
    torch._check(len(output_size) == 2, lambda: "only 2D output_size supported")
    torch._check(len(kernel_size) == 2, lambda: "only 2D kernel supported")
    torch._check(len(dilation) == 2, lambda: "only 2D dilation supported")
    torch._check(len(padding) == 2, lambda: "only 2D padding supported")
    torch._check(len(stride) == 2, lambda: "only 2D stride supported")

    def check_positive(param, param_name, strict=True):
        cond = all(p > 0 for p in param) if strict else all(p >= 0 for p in param)
        torch._check(
            cond, lambda: "{param_name} should be greater than zero, but got {param}"
        )

    check_positive(kernel_size, "kernel_size")
    check_positive(dilation, "dilation")
    check_positive(padding, "padding", strict=False)
    check_positive(stride, "stride")
    check_positive(output_size, "output_size")

    shape = input.shape
    ndim = len(shape)
    torch._check(
        ndim in (2, 3) and all(d != 0 for d in shape[-2:]),
        lambda: "Expected 2D or 3D (batch mode) tensor for input with possible 0 batch size "
        f"and non-zero dimensions, but got: {tuple(shape)}",
    )
    prod_kernel_size = kernel_size[0] * kernel_size[1]
    torch._check(
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
    torch._check(
        shape[-1] == L,
        lambda: f"Given output_size={output_size}, kernel_size={kernel_size}, "
        f"dilation={dilation}, padding={padding}, stride={stride}, "
        f"expected input.size(-1) to be {L} but got {shape[-1]}.",
    )
    torch._check(
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
    output = aten._unsafe_index_put(output, idx, input, accumulate=True)
    output = F.pad(output, (-padding_w, -padding_w, -padding_h, -padding_h))

    if not batched_input:
        output = output.squeeze(0)
    return output


@register_decomposition(aten.native_dropout_backward)
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    # According to the CUDA kernel implementation we should have this test;
    # but it seems to fail tests!
    # torch._check(mask.dtype == torch.bool, lambda: f"Mask should be Bool Scalar Type {mask.dtype}")

    # Mimicking CUDA kernel's behavior for output stride: output follow input's memory format
    # This different from TensorIterator's behavior
    r = (grad_output * (mask.type_as(grad_output) * scale)).clone(
        memory_format=utils.suggest_memory_format(grad_output)
    )
    return r


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
    index = (None,) * dim + (idx,)
    return aten._unsafe_index_put(grad_input, index, grad, accumulate=True).contiguous()


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
    if train and p != 0:
        if p == 1:
            return (torch.zeros_like(input), torch.zeros_like(input, dtype=torch.bool))
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
    if x.numel() == 0:
        unnormalized = torch.exp(x)
    else:
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
    if x.numel() == 0:
        shifted = x
    else:
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
    # Nb. scale_grad_by_freq is not used in the forward
    if indices.ndim <= 1:
        # We need this one as weight[indices] calls item() in these cases
        out = weight.index_select(0, indices)
        if indices.ndim == 0:
            out = out.squeeze(0)
        return out
    else:
        return weight[indices]


@register_decomposition(aten.embedding_dense_backward)
def embedding_dense_backward(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    computation_dtype, result_dtype = utils.elementwise_dtypes(
        grad_output, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    grad_output = grad_output.to(computation_dtype)
    indices = _maybe_convert_to_dtype(indices, torch.long)  # type: ignore[assignment]
    if scale_grad_by_freq:
        counts = indices.new_zeros((num_weights,))
        ones = torch.ones_like(indices)
        counts = aten._unsafe_index_put(counts, [indices], ones, accumulate=True)
        grad_weights_scale = counts[indices]
        grad_output = grad_output / grad_weights_scale.unsqueeze(-1)

    mask = _unsqueeze_to_dim(indices == padding_idx, grad_output.ndim)
    grad = grad_output.masked_fill(mask, 0)
    grad_weight = grad_output.new_zeros(
        (num_weights,) + grad_output.shape[indices.ndim :]
    )
    return aten._unsafe_index_put(grad_weight, [indices], grad, accumulate=True).to(
        result_dtype
    )


def prod(x: List[int]):
    r = 1
    for i in x:
        r *= i
    return r


@register_decomposition([aten.split_with_sizes, aten.unsafe_split_with_sizes])
def split_with_sizes(
    self: Tensor, split_sizes: List[int], dim: int = 0
) -> List[Tensor]:
    torch._check_with(
        ValueError,
        sum(split_sizes) == self.shape[dim],
        lambda: "Split sizes don't add up to the tensor's size in the given dimension",
    )
    num_splits = len(split_sizes)
    splits = []
    start_idx = 0
    for i in range(num_splits):
        length = split_sizes[i]
        torch._check(
            length >= 0,
            lambda: "split_with_sizes expects split_sizes have only non-negative entries",
        )
        # We know this is true thanks to the sum, but this assertion helps
        # out our internal reasoning
        expect_true(start_idx + length <= self.shape[dim])
        splits.append(self.narrow(dim, start_idx, length))
        start_idx += length
    return splits


@register_decomposition([aten.split.Tensor, aten.unsafe_split.Tensor])
def split(self: Tensor, split_size: int, dim: int = 0) -> Tuple[Tensor, ...]:
    input_sizes = self.shape
    dim_size = input_sizes[dim]
    if split_size == 0:
        assert dim_size == 0
        return (self,)
    chunks = (dim_size + split_size - 1) // split_size
    chunks = guard_int(chunks)
    split_sizes = [split_size for i in range(chunks)]
    split_sizes[-1] = split_size - (split_size * chunks - dim_size)
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


@register_decomposition(aten._addmm_activation)
@out_wrapper()
@pw_cast_for_opmath
def _addmm_activation(
    self: Tensor,
    mat1: Tensor,
    mat2: Tensor,
    beta: int = 1,
    alpha: int = 1,
    use_gelu: bool = False,
):
    out = addmm(self, mat1, mat2, beta, alpha)
    if use_gelu:
        if self.is_cuda:
            return aten.gelu(out, approximate="tanh")
        else:
            return aten.gelu(out)
    return aten.relu(out)


@register_decomposition(aten.addmv)
@out_wrapper()
@pw_cast_for_opmath
def addmv(self: Tensor, mat1: Tensor, vec: Tensor, beta: int = 1, alpha: int = 1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mv(mat1, vec)
    if beta == 0:
        return out
    return out + beta * self


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
    torch._check(
        input.numel() == N * C * HxW,
        lambda: f"Expect input to have { N * C * HxW} elements",
    )
    torch._check(
        mean.shape == (N, group),
        lambda: f"Expect mean to have shape ({N}, {group}, but got {mean.shape}",
    )
    torch._check(
        gamma is None or gamma.numel() == C,
        lambda: f"Expect gamma to have {C} elements but got {gamma.numel() if gamma is not None else -1}",
    )

    cpg, _rem = divmod(C, group)
    torch._check(
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
    grad_out_cast, input_cast, weight_cast, bias_cast = (
        x.to(computation_dtype).contiguous() if x is not None else x
        for x in (grad_out, input, weight, bias)
    )
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


def native_batch_norm_helper(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
    functional: bool,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    reduction_dims = [0] + list(range(2, input.dim()))
    computation_dtype = utils.get_computation_dtype(input.dtype)
    new_running_mean = running_mean
    new_running_var = running_var
    if training:
        computation_dtype = utils.get_computation_dtype(input.dtype)
        input_acc = input.to(dtype=computation_dtype)
        biased_var, mean = torch.var_mean(
            input_acc, dim=reduction_dims, correction=0, keepdim=True
        )
        rstd = torch.rsqrt(biased_var + eps)

        output = (input - mean) * rstd

        save_mean = torch.squeeze(mean, reduction_dims)
        save_rstd = torch.squeeze(rstd, reduction_dims)
        if running_mean is not None:
            new_running_mean = momentum * save_mean + (1 - momentum) * running_mean
            if not functional:
                running_mean.copy_(new_running_mean)
        if running_var is not None:
            n = input.numel() / input.shape[1]
            # This doesn't strictly match eager's numerics, which accumulates var sum and then directly applies the correction
            # But... that would require re-implementing var here, for negligible numerics gain on a tensor whose
            # numerics probably don't matter.
            squeezed_var = torch.squeeze(biased_var, reduction_dims)
            unbiased_var = squeezed_var * (n / (n - 1))
            new_running_var = momentum * unbiased_var + (1 - momentum) * running_var
            if not functional:
                running_var.copy_(new_running_var)
    else:
        assert running_mean is not None and running_var is not None
        running_mean = running_mean.to(dtype=computation_dtype, copy=True)
        new_running_mean = running_mean
        running_var = running_var.to(dtype=computation_dtype, copy=True)
        new_running_var = running_var
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

    if weight is not None:
        weight = weight.flatten()
        weight = _unsqueeze_to_dim(weight, input.dim() - 1)
        output = output * weight

    if bias is not None:
        bias = bias.flatten()
        bias = _unsqueeze_to_dim(bias, input.dim() - 1)
        output = output + bias

    if input.device.type == "cpu":
        save_mean = save_mean.to(dtype=input.dtype)
        save_rstd = save_rstd.to(dtype=input.dtype)
    return (
        output.to(dtype=input.dtype),
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
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
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


# TODO: this decomposition is NOT here to stay. We would much prefer replacing native_batch_norm
# with our new correctly schema'd _native_batch_norm_legit and its variants, but
# we cannot do that immediately in the C++ because it would be forwards incompatible
# with some mobile use cases.
#
# Since this change is most impactful for aot autograd/functionalization, we simply
# register this decomposition on the Autograd key for the python dispatcher (which is
# currently only used by aot autograd/functionalization and no one else, really).
# In two weeks or so, we should remove this decomposition and phase out the current native_batch_norm
# to be _native_batch_norm_legit and have the right schema (stating that there are input mutations).
@aten.native_batch_norm.default.py_impl(DispatchKey.Autograd)
def native_batch_norm_decomposition(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    if running_mean is None and running_var is None:
        return aten._native_batch_norm_legit(
            input, weight, bias, training, momentum, eps
        )
    if running_mean is None:
        raise RuntimeError(
            "running_mean is None, but running_var is provided. "
            "They should both be None or both be provided."
        )
    if running_var is None:
        raise RuntimeError(
            "running_var is None, but running_mean is provided. "
            "They should both be None or both be provided."
        )
    if training:
        # HACK: batch norm consolidation should clean this up so this op doesn't take in a training arg.
        return aten._native_batch_norm_legit(
            input, weight, bias, running_mean, running_var, training, momentum, eps
        )
    else:
        return aten._native_batch_norm_legit_no_training(
            input, weight, bias, running_mean, running_var, momentum, eps
        )


@aten.unsafe_chunk.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def unsafe_chunk_py_impl(tensor, chunks, dim=0) -> List[Tensor]:
    dim_size = tensor.size(dim)
    split_size = (dim_size + chunks - 1) // chunks

    if split_size == 0 and dim_size == 0:
        split_sizes = [split_size for _ in chunks]
        split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size)
        return torch.ops.aten.unsafe_split_with_sizes.default(tensor, split_sizes, dim)
    return torch.ops.aten.unsafe_split.Tensor(tensor, split_size, dim)


@register_decomposition(aten._native_batch_norm_legit_no_training.default)
def _native_batch_norm_legit_no_training(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    return aten._native_batch_norm_legit.default(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        False,  # training
        momentum,
        eps,
    )


@register_decomposition(aten._native_batch_norm_legit.default)
def _native_batch_norm_legit(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


@register_decomposition(aten._native_batch_norm_legit.no_stats)
def _native_batch_norm_legit_no_stats(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input, weight, bias, None, None, training, momentum, eps, False
    )
    return output, save_mean, save_rstd


@register_decomposition(aten._native_batch_norm_legit_functional.default)
def _native_batch_norm_legit_functional(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        output,
        save_mean,
        save_rstd,
        new_running_mean,
        new_running_var,
    ) = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, training, momentum, eps, True
    )
    assert new_running_mean is not None, "new_running_mean should not be None"
    assert new_running_var is not None, "new_running_var should not be None"
    return output, save_mean, save_rstd, new_running_mean, new_running_var


@register_decomposition(aten._fused_dropout)
@pw_cast_for_opmath
def _fused_dropout_decomposition(input, p, generator=None):
    assert generator is None
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
    if device is None and dtype is None and memory_format is None:
        return x.clone()
    dtype_converted = False
    if device is not None and device != x.device:
        # avoid conversions on cpu
        if dtype is not None and device.type == "cpu":
            x = torch._prims.convert_element_type(x, dtype)
            dtype_converted = True
        x = torch._prims.device_put(x, device)
    if dtype is not None and not dtype_converted:
        x = torch._prims.convert_element_type(x, dtype)
    if memory_format is not None:  # no ref/prim for memory format
        return torch.clone(x, memory_format=memory_format)
    return x


# Questionable decompositions
# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition([aten.detach, aten.lift, aten.lift_fresh])
def nop_decomposition(x):
    return aten.alias(x)


# Also register to the Autograd dispatch key, so this decomp can run above autograd.
# native_batch_norm needs to decompose into other ops before autograd.
@aten.cudnn_batch_norm.default.py_impl(DispatchKey.Autograd)
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
    if weight is not None:
        weight_dtype = weight.dtype
    else:
        weight_dtype = input_dtype
    computation_dtype = utils.get_computation_dtype(input.dtype)
    (
        grad_out_cast,
        input_cast,
        weight_cast,
        running_mean_cast,
        running_var_cast,
        save_mean_cast,
        save_invstd_cast,
    ) = (
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
    )
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
        _maybe_cast(grad_weight, weight_dtype),
        _maybe_cast(grad_bias, weight_dtype),
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
    torch._check(
        ndim in (3, 4),
        lambda: f"adaptive_avg_pool2d(): Expected 3D or 4D tensor, but got {ndim}",
    )
    for d in input.shape[-2:]:
        torch._check(
            d != 0,
            lambda: "adaptive_avg_pool2d(): Expected input to have non-zero size for "
            f"non-batch dimensions, but input has shape {tuple(shape)}.",
        )

    # TODO: decompose integer path
    if input.dtype in [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]:
        return torch.nn.functional.adaptive_avg_pool2d(input, output_size)

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
    return _index_add(x, dim, index, tensor, inplace=True, alpha=alpha)


@register_decomposition(aten.index_add)
@out_wrapper()
def index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    alpha: NumberType = 1,
):
    return _index_add(x, dim, index, tensor, inplace=False, alpha=alpha)


def _index_add(
    x: TensorLike,
    dim: int,
    index: TensorLike,
    tensor: TensorLike,
    *,
    inplace: bool,
    alpha: NumberType = 1,
):
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    if alpha != 1:
        python_type = utils.dtype_to_type(x.dtype)
        torch._check(
            python_type == bool
            or utils.is_weakly_lesser_type(type(alpha), python_type),
            lambda: f"alpha argument of type {type(alpha)} cannot be safely cast to type {python_type}!",
        )
        tensor = tensor * alpha
    # Treat scalars as elements of \R^1
    zero_dim = x.ndim == 0
    x1 = x.unsqueeze(0) if zero_dim else x
    idx = (None,) * dim + (index,)
    index_put = aten.index_put_ if inplace else aten.index_put
    out = index_put(x1, idx, tensor, accumulate=True)
    if inplace:
        return x
    else:
        return out.squeeze(0) if zero_dim else out.contiguous()


@register_decomposition(aten.index_copy_)
def index_copy_(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return _index_copy(x, dim, index, tensor, inplace=True)


@register_decomposition(aten.index_copy)
@out_wrapper()
def index_copy(x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike):
    return _index_copy(x, dim, index, tensor, inplace=False)


def _index_copy(
    x: TensorLike, dim: int, index: TensorLike, tensor: TensorLike, *, inplace: bool
):
    dim = utils.canonicalize_dims(x.ndim, dim)
    torch._check(
        index.ndim <= 1,
        lambda: f"Index should have dimension 1 or 0 (got {index.ndim})",
    )
    # Treat scalars as elements of \R^1
    zero_dim = x.ndim == 0
    x1 = x.unsqueeze(0) if zero_dim else x
    idx = (None,) * dim + (index,)
    index_put = aten.index_put_ if inplace else aten.index_put
    out = index_put(x1, idx, tensor)
    if inplace:
        return x
    else:
        return out.squeeze(0) if zero_dim else out.contiguous()


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


@register_decomposition(aten.uniform)
def uniform(
    x: Tensor,
    low: Union[bool, int, float] = 0.0,
    high: Union[bool, int, float] = 1.0,
):
    return prims._uniform_helper(
        x.shape,
        low=sym_float(low),
        high=sym_float(high),
        dtype=x.dtype,
        device=x.device,
    )


@register_decomposition(aten.uniform_)
def uniform_(self, low=0, high=1, generator=None):
    assert generator is None
    return self.copy_(uniform(self, low, high))


# aten/src/ATen/native/UpSample.cpp compute_output_size
def upsample_compute_output_size(input_size, output_size, scale_factors):
    spatial_dimensions = len(input_size) - 2
    if output_size is not None:
        torch._check(
            scale_factors is None,
            lambda: "Must specify exactly one of output_size and scale_factors",
        )
        torch._check(len(output_size) == spatial_dimensions, lambda: "")
        return output_size
    if scale_factors is not None:
        # NB: this isn't necessary lol
        torch._check(
            output_size is None,
            lambda: "Must specify exactly one of output_size and scale_factors",
        )
        torch._check(len(scale_factors) == spatial_dimensions, lambda: "")
        output_size = []
        for i, s in enumerate(scale_factors):
            if int(s) == s:
                output_size.append(input_size[i + 2] * int(s))
            else:
                output_size.append(sym_int(input_size[i + 2] * s))
        return output_size
    torch._check(
        False, lambda: "Must specify exactly one of output_size and scale_factors"
    )


def get_scale_value(scales, idx):
    if scales is None:
        return None
    return scales[idx]


@register_decomposition(aten.upsample_nearest1d.vec)
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.Autograd)
def upsample_nearest1d_vec(input, output_size, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scale = get_scale_value(scale_factors, 0)

    return aten.upsample_nearest1d.default(input, osize, scale)


@register_decomposition(aten.upsample_nearest2d.vec)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.Autograd)
def upsample_nearest2d_vec(input, output_size, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scale_h = get_scale_value(scale_factors, 0)
    scale_w = get_scale_value(scale_factors, 1)

    return aten.upsample_nearest2d.default(input, osize, scale_h, scale_w)


@register_decomposition(aten.upsample_nearest3d.vec)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.Autograd)
def upsample_nearest3d_vec(input, output_size, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scale_d = get_scale_value(scale_factors, 0)
    scale_h = get_scale_value(scale_factors, 1)
    scale_w = get_scale_value(scale_factors, 2)

    return aten.upsample_nearest3d.default(input, osize, scale_d, scale_h, scale_w)


def _compute_upsample_nearest_indices(input, output_size, scales):
    # For each dim in output_size, compute the set of input indices used
    # to produce the upsampled output.
    indices = []
    num_spatial_dims = len(output_size)
    input_dtype = torch.float if input.dtype == torch.uint8 else input.dtype
    for d in range(num_spatial_dims):
        # Math matches aten/src/ATen/native/cpu/UpSampleKernel.cpp
        # Indices are computed as following:
        # scale = isize / osize
        # input_index = floor(output_index * scale)
        # Same as OpenCV INTER_NEAREST
        osize = output_size[d]
        output_indices = torch.arange(osize, dtype=input_dtype, device=input.device)
        isize = input.shape[-num_spatial_dims + d]
        scale = isize / (isize * scales[d]) if scales[d] is not None else isize / osize
        input_indices = (output_indices * scale).to(torch.int64)
        for _ in range(num_spatial_dims - 1 - d):
            input_indices = input_indices.unsqueeze(-1)
        indices.append(input_indices)
    return tuple(indices)


@register_decomposition(aten.upsample_nearest1d.default)
@aten.upsample_nearest1d.default.py_impl(DispatchKey.Autograd)
@pw_cast_for_opmath
def upsample_nearest1d(
    input: Tensor,
    output_size: List[int],
    scales: Optional[float] = None,
) -> Tensor:
    (l_indices,) = _compute_upsample_nearest_indices(input, output_size, (scales,))
    return aten._unsafe_index(input, (None, None, l_indices))


@register_decomposition(aten.upsample_nearest2d.default)
@aten.upsample_nearest2d.default.py_impl(DispatchKey.Autograd)
@pw_cast_for_opmath
def upsample_nearest2d(
    input: Tensor,
    output_size: List[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    h_indices, w_indices = _compute_upsample_nearest_indices(
        input, output_size, (scales_h, scales_w)
    )
    result = aten._unsafe_index(input, (None, None, h_indices, w_indices))

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    _, n_channels, _, _ = input.shape
    if input.device.type == "cuda" and n_channels < 4:
        memory_format = torch.contiguous_format

    result = result.contiguous(memory_format=memory_format)

    return result


@register_decomposition(aten.upsample_nearest3d.default)
@aten.upsample_nearest3d.default.py_impl(DispatchKey.Autograd)
@pw_cast_for_opmath
def upsample_nearest3d(
    input: Tensor,
    output_size: List[int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    d_indices, h_indices, w_indices = _compute_upsample_nearest_indices(
        input, output_size, (scales_d, scales_h, scales_w)
    )
    result = aten._unsafe_index(input, (None, None, d_indices, h_indices, w_indices))

    return result


def gather_params(params, has_biases, has_projections):
    if has_biases and has_projections:
        group_size = 5
    elif has_biases:
        group_size = 4
    elif has_projections:
        group_size = 3
    else:
        group_size = 2

    assert len(params) % group_size == 0, len(params)
    return [
        tuple(params[i : i + group_size]) for i in range(0, len(params), group_size)
    ]


def params_hiddens(params, hiddens, i, bidirectional):
    if bidirectional:
        cur_params, cur_hidden = params[2 * i], hiddens[2 * i]
        bidir_params, bidir_hidden = params[2 * i + 1], hiddens[2 * i + 1]
    else:
        cur_params, cur_hidden = params[i], hiddens[i]
        bidir_params, bidir_hidden = None, None

    return cur_params, cur_hidden, bidir_params, bidir_hidden


def update_hidden_for_packed(cur_hidden, last_batch_size, batch_size, hiddens):
    assert last_batch_size > batch_size
    hiddens.append(cur_hidden.narrow(0, batch_size, last_batch_size - batch_size))
    return cur_hidden.narrow(0, 0, batch_size)


def update_hidden_for_packed_reverse(
    cur_hidden, last_batch_size, batch_size, inp_hidden
):
    if last_batch_size == batch_size:
        return cur_hidden
    assert last_batch_size < batch_size
    return torch.concat(
        (
            cur_hidden,
            inp_hidden.narrow(0, last_batch_size, batch_size - last_batch_size),
        )
    )


def one_layer_rnn_data(
    inp, hidden, params, has_biases, hidden_fn, batch_sizes, reverse=False
):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None

    step_output = []
    hiddens: List[torch.Tensor] = []

    last_batch_size = batch_sizes[-1] if reverse else batch_sizes[0]
    cur_hidden = hidden.narrow(0, 0, last_batch_size)
    split_inp = torch.split(inp, list(batch_sizes))
    if reverse:
        split_inp = split_inp[::-1]
    for inp in split_inp:
        i = inp.shape[0]

        if last_batch_size == i:
            pass  # don't update cur_hidden
        # this will only happen when reverse=False, since batch sizes are sorted largest -> smallest
        elif reverse:
            cur_hidden = update_hidden_for_packed_reverse(
                cur_hidden, last_batch_size, i, hidden
            )
        else:
            cur_hidden = update_hidden_for_packed(
                cur_hidden, last_batch_size, i, hiddens
            )

        cur_hidden = hidden_fn(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias)
        last_batch_size = i
        step_output.append(cur_hidden)

    if reverse:
        step_output.reverse()
    else:
        hiddens.append(cur_hidden)
        hiddens.reverse()

    out = torch.cat(step_output, 0)
    hidden_out = torch.cat(hiddens, 0) if not reverse else cur_hidden
    return out, hidden_out


def rnn_cell(nonlinearity):
    def inner(i, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
        return nonlinearity(F.linear(cur_hidden, hh_weight, hh_bias) + i)

    return inner


def rnn_cell_data(nonlinearity):
    def inner(i, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
        i = F.linear(i, ih_weight, ih_bias)
        return nonlinearity(F.linear(cur_hidden, hh_weight, hh_bias) + i)

    return inner


def one_layer_rnn(inp, hidden, params, has_biases, hidden_fn, reverse=False):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None

    precomputed_input = F.linear(inp, ih_weight, ih_bias)
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input
    cur_hidden = hidden.unsqueeze(0)
    step_output = []
    for i in precomputed_input:
        cur_hidden = hidden_fn(i, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias)
        step_output.append(cur_hidden)

    if reverse:
        step_output.reverse()

    out = torch.cat(step_output, 0)

    return out, cur_hidden.squeeze(0)


def mkldnn_one_layer_lstm(inp, hidden, params, has_biases, reverse=False):
    w0 = params[0]
    w1 = params[1]
    if has_biases:
        w2 = params[2]
        w3 = params[3]
    else:
        w2 = torch.zeros(w0.size())
        w3 = torch.zeros(w1.size())

    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)

    batch_sizes: List[int] = []
    mode = 2  # third_party/ideep/include/ideep/abstract_types.hpp: ideep::rnn_kind::LSTM = 2
    hidden_size = hx.size(2)
    num_layers = 1

    # _rnn_helper already handles bidirectional and batch_first so we hard-code them to False here
    bidirectional = False
    batch_first = False

    train = False
    # If batch_first, inp has been permuted in _rnn_helper. Convert to contiguous here.
    # Same as aten/src/ATen/native/mkldnn/RNN.cpp: mkldnn_rnn: input = input.contiguous();
    inp = inp.contiguous()
    hx = hx.contiguous()
    cx = cx.contiguous()
    outputs = torch.ops.aten.mkldnn_rnn_layer.default(
        inp,
        w0,
        w1,
        w2,
        w3,
        hx,
        cx,
        reverse,
        batch_sizes,
        mode,
        hidden_size,
        num_layers,
        has_biases,
        bidirectional,
        batch_first,
        train,
    )
    y, hy, cy = outputs[0], outputs[1], outputs[2]
    return y, (hy.squeeze(0), cy.squeeze(0))


def _rnn_helper(
    input,
    hidden,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
    layer_fn,
):
    input = input.transpose(0, 1) if batch_first else input
    final_hiddens = []

    for i in range(num_layers):
        cur_params, cur_hidden, bidir_params, bidir_hidden = params_hiddens(
            params, hidden, i, bidirectional
        )
        dropout = dropout if (train and num_layers < i - 1) else 0.0
        fwd_inp, fwd_hidden = layer_fn(input, cur_hidden, cur_params, has_biases)
        final_hiddens.append(fwd_hidden)

        if bidirectional:
            bwd_inp, bwd_hidden = layer_fn(
                input, bidir_hidden, bidir_params, has_biases, reverse=True
            )
            final_hiddens.append(bwd_hidden)

        if bidirectional:
            input = torch.cat([fwd_inp, bwd_inp], fwd_inp.dim() - 1)
        else:
            input = fwd_inp

        if dropout != 0 and train and i < num_layers - 1:
            input = torch.dropout(input, dropout, train=True)

    input = input.transpose(0, 1) if batch_first else input
    return input, final_hiddens


@register_decomposition(aten.rnn_tanh.input)
@aten.rnn_tanh.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_tanh.input.py_impl(DispatchKey.Autograd)
def rnn_tanh_input(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    hidden = hx.unbind(0)
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        partial(one_layer_rnn, hidden_fn=rnn_cell(torch.tanh)),
    )
    return out, torch.stack(final_hiddens, 0)


@register_decomposition(aten.rnn_relu.input)
@aten.rnn_relu.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_relu.input.py_impl(DispatchKey.Autograd)
def rnn_relu_input(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    hidden = hx.unbind(0)
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        partial(one_layer_rnn, hidden_fn=rnn_cell(torch.relu)),
    )
    return out, torch.stack(final_hiddens, 0)


@register_decomposition(aten.rnn_relu.data)
@aten.rnn_relu.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_relu.data.py_impl(DispatchKey.Autograd)
def rnn_relu_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    hidden = hx.unbind(0)
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(
        data,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,
        partial(
            one_layer_rnn_data,
            batch_sizes=batch_sizes,
            hidden_fn=rnn_cell_data(torch.relu),
        ),
    )
    return out, torch.stack(final_hiddens, 0)


@register_decomposition(aten.rnn_tanh.data)
@aten.rnn_tanh.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.rnn_tanh.data.py_impl(DispatchKey.Autograd)
def rnn_tanh_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    hidden = hx.unbind(0)
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(
        data,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,
        partial(
            one_layer_rnn_data,
            batch_sizes=batch_sizes,
            hidden_fn=rnn_cell_data(torch.tanh),
        ),
    )
    return out, torch.stack(final_hiddens, 0)


def lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim):
    gates = F.linear(hx, hh_weight, hh_bias) + inp
    chunked_gates = gates.chunk(4, chunk_dim)
    in_gate = chunked_gates[0].sigmoid()
    forget_gate = chunked_gates[1].sigmoid()
    cell_gate = chunked_gates[2].tanh()
    out_gate = chunked_gates[3].sigmoid()
    cy = forget_gate * cx + (in_gate * cell_gate)
    hy = out_gate * cy.tanh()
    hy = hy if hr_weight is None else F.linear(hy, hr_weight, None)

    return hy, cy


def one_layer_lstm(inp, hidden, params, has_biases, reverse=False):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = (
        params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    )

    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)

    precomputed_input = F.linear(inp, ih_weight, ih_bias)
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input
    step_output = []
    for inp in precomputed_input:
        hx, cx = lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=2)
        step_output.append(hx)

    if reverse:
        step_output.reverse()

    out = torch.cat(step_output, 0)

    return out, (hx.squeeze(1), cx.squeeze(1))


def one_layer_lstm_data(inp, hidden, params, has_biases, batch_sizes, reverse=False):
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = (
        params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    )

    step_output = []
    hiddens = []

    last_batch_size = batch_sizes[-1] if reverse else batch_sizes[0]
    split_inp = torch.split(inp, list(batch_sizes))
    if reverse:
        split_inp = split_inp[::-1]

    orig_hx = hidden[0]
    orig_cx = hidden[1]
    hx, cx = orig_hx.narrow(0, 0, last_batch_size), orig_cx.narrow(
        0, 0, last_batch_size
    )

    for inp in split_inp:
        i = inp.shape[0]
        inp = F.linear(inp, ih_weight, ih_bias)

        # this will only happen when reverse=False, since batch sizes are sorted largest -> smallest
        if i < last_batch_size:
            hiddens.append(
                (
                    hx.narrow(0, i, last_batch_size - i),
                    cx.narrow(0, i, last_batch_size - i),
                )
            )
            hx, cx = hx.narrow(0, 0, i), cx.narrow(0, 0, i)

        # this will only happen when reverse=True
        if i > last_batch_size:
            hx = torch.concat(
                (hx, orig_hx.narrow(0, last_batch_size, i - last_batch_size)), 0
            )
            cx = torch.concat(
                (cx, orig_cx.narrow(0, last_batch_size, i - last_batch_size)), 0
            )

        hx, cx = lstm_cell(inp, hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=1)
        last_batch_size = i
        step_output.append(hx)

    if reverse:
        step_output.reverse()
        hidden_out = (hx, cx)
    else:
        hiddens.append((hx, cx))
        hiddens.reverse()
        hidden0, hidden1 = zip(*hiddens)
        hidden_out = torch.cat(hidden0, 0), torch.cat(hidden1, 0)

    out = torch.cat(step_output, 0)
    return out, hidden_out


def select_one_layer_lstm_function(input, hx, params):
    r"""Check whether we could use decompose lstm with mkldnn_rnn_layer.
    All the below conditions need to be met:
        * ``torch._C._has_mkldnn`` returns ``True``.
        * All the input args are on CPU.
        * The dtypes of args are either torch.float or torch.bfloat16.
        * Inference.
        * ``has_projections`` returns ``False``.

    Args:
        * input: the input sequence to LSTM
        * hx: a tuple of the input hidden state and cell state ``(h_0, c_0)`` to LSTM
        * params: the weight and bias tensors of LSTM
    """

    def use_mkldnn(input, hx, params):
        if not torch._C._has_mkldnn:
            return False

        tensors = [input] + list(hx) + list(chain.from_iterable(params))
        devices = {t.device for t in tensors}
        if len(devices) != 1:
            return False

        device = devices.pop()
        if device != torch.device("cpu"):
            return False
        # With autocast, possible to have mixed dtype here
        dtypes = {t.dtype for t in tensors}
        for dtype in dtypes:
            if dtype not in [torch.float, torch.bfloat16]:
                return False

        if input.requires_grad:
            return False

        has_projections = hx[0].size(2) != hx[1].size(2)
        if has_projections:
            return False

        return True

    # mkldnn_one_layer_lstm does not depend on seq_len while one_layer_lstm
    # will expand over the seq_len dim
    if use_mkldnn(input, hx, params):
        return mkldnn_one_layer_lstm
    else:
        return one_layer_lstm


@register_decomposition(aten.lstm.input)
@aten.lstm.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.lstm.input.py_impl(DispatchKey.Autograd)
def lstm_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    assert len(hx) == 2, "lstm expects two hidden states"
    params = gather_params(params, has_biases, hx[0].size(2) != hx[1].size(2))
    hidden = list(zip(hx[0], hx[1]))
    layer_fn = select_one_layer_lstm_function(input, hx, params)
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        layer_fn,
    )
    final_hiddens = list(zip(*final_hiddens))
    return out, torch.stack(final_hiddens[0], 0), torch.stack(final_hiddens[1], 0)


@register_decomposition(aten.lstm.data)
@aten.lstm.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.lstm.data.py_impl(DispatchKey.Autograd)
def lstm_data_impl(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    assert len(hx) == 2, "lstm expects two hidden states"
    params = gather_params(params, has_biases, hx[0].size(2) != hx[1].size(2))
    hidden = list(zip(hx[0], hx[1]))
    out, final_hiddens = _rnn_helper(
        data,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,
        partial(one_layer_lstm_data, batch_sizes=batch_sizes),
    )
    final_hiddens = list(zip(*final_hiddens))
    return out, torch.stack(final_hiddens[0], 0), torch.stack(final_hiddens[1], 0)


def gru_cell(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
    chunked_igates = inp.chunk(3, 1)
    chunked_hgates = F.linear(cur_hidden, hh_weight, hh_bias).chunk(3, 2)
    reset_gate = (chunked_hgates[0] + chunked_igates[0]).sigmoid()
    input_gate = (chunked_hgates[1] + chunked_igates[1]).sigmoid()
    new_gate = (chunked_igates[2] + (chunked_hgates[2] * reset_gate)).tanh()
    return (cur_hidden - new_gate) * input_gate + new_gate


def gru_cell_data(inp, cur_hidden, ih_weight, ih_bias, hh_weight, hh_bias):
    chunked_igates = F.linear(inp, ih_weight, ih_bias).chunk(3, 1)
    chunked_hgates = F.linear(cur_hidden, hh_weight, hh_bias).chunk(3, 1)
    reset_gate = (chunked_hgates[0] + chunked_igates[0]).sigmoid()
    input_gate = (chunked_hgates[1] + chunked_igates[1]).sigmoid()
    new_gate = (chunked_igates[2] + (chunked_hgates[2] * reset_gate)).tanh()
    return (cur_hidden - new_gate) * input_gate + new_gate


@register_decomposition(aten.gru.data)
@aten.gru.data.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.gru.data.py_impl(DispatchKey.Autograd)
def gru_impl_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
):
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(
        data,
        hx.unbind(0),
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        False,
        partial(one_layer_rnn_data, batch_sizes=batch_sizes, hidden_fn=gru_cell_data),
    )
    return out, torch.stack(final_hiddens, 0)


@register_decomposition(aten.gru.input)
@aten.gru.input.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.gru.input.py_impl(DispatchKey.Autograd)
def gru_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    params = gather_params(params, has_biases, False)
    out, final_hiddens = _rnn_helper(
        input,
        hx.unbind(0),
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        partial(one_layer_rnn, hidden_fn=gru_cell),
    )
    return out, torch.stack(final_hiddens, 0)


@register_decomposition(aten._upsample_bilinear2d_aa.vec)
@aten._upsample_bilinear2d_aa.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_bilinear2d_aa.vec.py_impl(DispatchKey.Autograd)
def upsample_bilinear2d_aa_vec(input, output_size, align_corners, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scale_h = get_scale_value(scale_factors, 0)
    scale_w = get_scale_value(scale_factors, 1)
    return torch.ops.aten._upsample_bilinear2d_aa(
        input, osize, align_corners, scale_h, scale_w
    )


@register_decomposition(aten.upsample_bilinear2d.vec)
@aten.upsample_bilinear2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_bilinear2d.vec.py_impl(DispatchKey.Autograd)
def upsample_bilinear2d_vec(input, output_size, align_corners, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scale_h = get_scale_value(scale_factors, 0)
    scale_w = get_scale_value(scale_factors, 1)
    return aten.upsample_bilinear2d.default(
        input, osize, align_corners, scale_h, scale_w
    )


@register_decomposition(aten.upsample_bilinear2d.default)
@aten.upsample_bilinear2d.default.py_impl(DispatchKey.Autograd)
@pw_cast_for_opmath
def upsample_bilinear2d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # get dimensions of original image
    n_batch, n_channels, in_h, in_w = input.shape

    out_h = output_size[0]
    out_w = output_size[1]

    # Calculate horizontal and vertical scaling factor
    # TODO: Figure out if scales_h/scales_w matters here
    if out_h > 1:
        if align_corners:
            h_scale_factor = (in_h - 1) / (out_h - 1)
        else:
            h_scale_factor = 1.0 / scales_h if scales_h is not None else in_h / out_h
    else:
        h_scale_factor = 0.0

    if out_w > 1:
        if align_corners:
            w_scale_factor = (in_w - 1) / (out_w - 1)
        else:
            w_scale_factor = 1.0 / scales_w if scales_w is not None else in_w / out_w
    else:
        w_scale_factor = 0.0

    i = torch.arange(out_h, dtype=input.dtype, device=input.device)
    j = torch.arange(out_w, dtype=input.dtype, device=input.device)

    if align_corners:
        x = h_scale_factor * i
        y = w_scale_factor * j
    else:
        x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
        y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)

    x_floor = x.to(torch.int64)
    x_ceil = torch.ceil(x).clamp(max=in_h - 1).to(torch.int64)
    y_floor = y.to(torch.int64)
    y_ceil = torch.ceil(y).clamp(max=in_w - 1).to(torch.int64)

    x_view = x.unsqueeze(1)
    x_floor_view = x_floor.unsqueeze(1)
    x_ceil_view = x_ceil.unsqueeze(1)

    v1 = aten._unsafe_index(input, [None, None, x_floor_view, y_floor])
    v2 = aten._unsafe_index(input, [None, None, x_ceil_view, y_floor])
    v3 = aten._unsafe_index(input, [None, None, x_floor_view, y_ceil])
    v4 = aten._unsafe_index(input, [None, None, x_ceil_view, y_ceil])

    xscale2 = x_view - x_floor_view
    xscale1 = 1.0 - xscale2

    yscale2 = y - y_floor
    yscale1 = 1.0 - yscale2

    q1 = torch.mul(v1, xscale1) + torch.mul(v2, xscale2)
    q2 = torch.mul(v3, xscale1) + torch.mul(v4, xscale2)
    result = torch.mul(q1, yscale1) + torch.mul(q2, yscale2)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    if input.device.type == "cuda" and n_channels < 16:
        memory_format = torch.contiguous_format

    result = result.contiguous(memory_format=memory_format)

    return result


# We should be applying decompositions after all transformations
@register_decomposition(aten.is_same_size.default)
def is_same_size(a: Tensor, b: Tensor) -> bool:
    return a.shape == b.shape


@register_decomposition([aten._reshape_alias, aten._unsafe_view])
def _reshape_alias(x, shape, *args):
    return aten.view(x, shape)


@register_decomposition([aten._unsafe_index])
def _index(x, indices):
    return aten.index(x, indices)


def _nll_loss_forward(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> Tuple[Tensor, Tensor]:
    # self can be [N, C] or [C]
    # target can be [N] or []

    n_dims = self.dim()
    channel_dim = 1
    if n_dims < 2:
        channel_dim = 0

    if weight is not None:
        if n_dims > 1:
            shape = [
                1,
            ] * n_dims
            shape[channel_dim] = weight.shape[0]
            w = weight.view(shape)
        else:
            w = weight
        self = self * w
    safe_target = torch.where(target != ignore_index, target, 0)
    safe_target_ = safe_target.unsqueeze(channel_dim)
    # target can be [N, 1] or [1]

    result = -torch.gather(self, channel_dim, safe_target_).squeeze(channel_dim)

    result = torch.where(target != ignore_index, result, 0)

    if reduction == Reduction.NONE.value and n_dims > 1:
        total_weight = self.new_full((), 0.0)
        return result, total_weight

    if weight is not None:
        w = w.expand(self.shape)
        wsum = torch.gather(w, channel_dim, safe_target_).squeeze(channel_dim)
        wsum = torch.where(target != ignore_index, wsum, 0)
        total_weight = wsum.sum()
    else:
        total_weight = (target != ignore_index).sum().to(self)

    if reduction == Reduction.SUM.value:
        result = result.sum()
    elif reduction == Reduction.MEAN.value:
        result = result.sum() / total_weight

    return result, total_weight


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

    return _nll_loss_forward(self, target, weight, reduction, ignore_index)


@register_decomposition(aten.nll_loss2d_forward)
def nll_loss2d_forward(
    self: Tensor,
    target: Tensor,
    weight: Optional[Tensor],
    reduction: int,
    ignore_index: int,
) -> Tuple[Tensor, Tensor]:
    return _nll_loss_forward(self, target, weight, reduction, ignore_index)


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


def _linspace_from_neg_one(
    num_steps: int, align_corners: bool, dtype: torch.dtype, device: torch.device
):
    if num_steps <= 1:
        return torch.tensor(0, device=device, dtype=dtype)

    a = ((num_steps - 1) / num_steps) if not align_corners else 1
    return torch.linspace(-a, a, steps=num_steps, device=device, dtype=dtype)


def _make_base_grid_4d(theta: Tensor, h: int, w: int, align_corners: bool):
    dtype = theta.dtype
    device = theta.device

    # Using padding and summation generates a single kernel vs using torch.stack where 3 kernels generated
    # corresponding to each individual tensor: grid_x, grid_y, grid_one
    grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, w, 1)
    grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(h, 1, 1)
    grid_one = torch.ones((1, 1, 1), dtype=dtype, device=device)

    # this is just a temporary hack and we should use torch.stack here once #104480 is merged
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(2, 0), mode="constant", value=0)
    return grid_x + grid_y + grid_one


def _make_base_grid_5d(theta: Tensor, d: int, h: int, w: int, align_corners: bool):
    dtype = theta.dtype
    device = theta.device

    grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, 1, w, 1)
    grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(1, h, 1, 1)
    grid_z = _linspace_from_neg_one(d, align_corners, dtype, device).view(d, 1, 1, 1)
    grid_one = torch.ones((1, 1, 1, 1), dtype=dtype, device=device)

    # this is just a temporary hack and we should use torch.stack here once #104480 is merged
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 3), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 2), mode="constant", value=0)
    grid_z = torch.nn.functional.pad(grid_z, pad=(2, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(3, 0), mode="constant", value=0)
    return grid_x + grid_y + grid_z + grid_one


def _affine_grid_generator_4d(theta: Tensor, size: List[int], align_corners: bool):
    n, _, h, w = size
    base_grid = _make_base_grid_4d(theta, h, w, align_corners=align_corners)
    # base_grid shape is (h, w, 3) and theta shape is (n, 2, 3)
    # We do manually a matrix multiplication which is faster than mm()
    # (h * w, 3, 1) * (n, 1, 3, 2) -> (n, h * w, 2)
    grid = (base_grid.view(-1, 3, 1) * theta.mT.unsqueeze(1)).sum(-2)
    return grid.view(n, h, w, 2)


def _affine_grid_generator_5d(theta: Tensor, size: List[int], align_corners: bool):
    n, _, d, h, w = size
    base_grid = _make_base_grid_5d(theta, d, h, w, align_corners=align_corners)
    # base_grid shape is (d, h, w, 4) and theta shape is (n, 3, 4)
    # We do manually a matrix multiplication which is faster than mm()
    # (d * h * w, 4, 1) * (n, 1, 4, 3) -> (n, h * w, 3)
    grid = (base_grid.view(-1, 4, 1) * theta.mT.unsqueeze(1)).sum(-2)
    return grid.view(n, d, h, w, 3)


@register_decomposition(aten.affine_grid_generator)
@pw_cast_for_opmath
def affine_grid_generator(theta: Tensor, size: List[int], align_corners: bool):
    torch._check(
        len(size) in (4, 5),
        lambda: "affine_grid_generator needs 4d (spatial) or 5d (volumetric) inputs.",
    )
    if len(size) == 4:
        return _affine_grid_generator_4d(theta, size, align_corners=align_corners)
    else:
        return _affine_grid_generator_5d(theta, size, align_corners=align_corners)


@register_decomposition(aten.grid_sampler_2d)
@pw_cast_for_opmath
def grid_sampler_2d(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> Tensor:
    torch._check(
        interpolation_mode in (0, 1, 2),
        lambda: f"Invalid interpolation mode {interpolation_mode}",
    )
    torch._check(
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

        coeffs = tuple(get_coeff(ofs) for ofs in range(4))
        return _upsample_cubic_interp1d(coeffs, ty.unsqueeze(1))


@register_decomposition(aten.mv)
@out_wrapper()
@pw_cast_for_opmath
def mv(self, vec):
    torch._check(
        self.dim() == 2 and vec.dim() == 1,
        lambda: f"matrix @ vector expected, got {self.dim()}, {vec.dim()}",
    )
    torch._check(
        self.size(1) == vec.size(0),
        lambda: f"size mismatch, got input ({self.size(0)}x{self.size(1)}), vec ({vec.size(0)})",
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

    torch._check(
        self.dim() == 1 and other.dim() == 1,
        lambda: f"1D tensors expected, but got {self.dim()}D and {other.dim()}D tensors",
    )
    torch._check(
        self.dtype == other.dtype,
        lambda: f"dot : expected both vectors to have same dtype, but found {self.dtype} and {other.dtype}",
    )

    def numel_error():
        return (
            f"inconsistent tensor size, expected tensor [{self.numel()}] and src [{other.numel()}] to have the"
            f"same number of elements, but got {self.numel()} and {other.numel()} elements respectively"
        )

    torch._check(self.numel() == other.numel(), numel_error)

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


def should_fold(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
    # For comments of the logic of this function see eager in /native/LinearAlgebra.cpp

    t1, t2 = (tensor1, tensor2) if tensor1.ndim >= tensor2.ndim else (tensor2, tensor1)

    if not (t1.ndim >= 3 and t2.ndim <= 2):
        return False
    if t2.requires_grad:
        return True
    if tensor1.ndim == 2:
        return False
    if t1.numel() == 0:
        return True

    t1_shape = t1.shape
    t1_stride = t1.stride()
    return all(
        st1 == st2 * s2
        for (st1, st2, s2) in zip(t1_stride[:-2], t1_stride[1:-1], t1_shape[1:-1])
    )


@aten.matmul.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@out_wrapper()
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
        return torch.mm(tensor1, tensor2)
    elif should_fold(tensor1, tensor2):
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

        # This will almost always be a view.
        # It may not be a view if t2->requires_grad(). See should_fold in aten/ for an explanation
        t1_folded = t1.reshape(folded_dim1, sizes_1[-1])
        if t2_is_matrix:
            # This copies if we perform a 2D @ 3D and the first tensor requires_grad
            # See should_fold native/LinearAlgebra.cpp for why.
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

        # Same optimization for the gradients as that in should_fold
        # If we're going to broadcast, we force it to go through the should_fold branch
        if (
            dim_tensor1 == 3
            and dim_tensor2 == 3
            and batch_tensor1[0] != batch_tensor2[0]
        ):
            if batch_tensor1[0] == 1 and tensor1.requires_grad():
                return matmul(tensor1.squeeze(0), tensor2)
            if batch_tensor2[0] == 1 and tensor2.requires_grad():
                return matmul(tensor1, tensor2.squeeze(0))

        # expand the batch portion (i.e. cut off matrix dimensions and expand rest)
        expand_batch_portion = list(
            torch.broadcast_shapes(batch_tensor1, batch_tensor2)
        )

        tensor1_expand_size = expand_batch_portion + [n, m1]

        expand_batch_product = prod(expand_batch_portion)

        # HACK: We need reshape with symint support
        tensor1_expanded = tensor1.expand(tensor1_expand_size).reshape(
            expand_batch_product, n, m1
        )

        vector_rhs = dim_tensor2 == 1
        if vector_rhs:
            tensor2_expand_size = expand_batch_portion + [m2]
            tensor2_expanded = (
                tensor2.expand(tensor2_expand_size)
                .reshape(expand_batch_product, m2)
                .unsqueeze(2)
            )
        else:
            tensor2_expand_size = expand_batch_portion + [m2, p]
            tensor2_expanded = tensor2.expand(tensor2_expand_size).reshape(
                expand_batch_product, m2, p
            )

        output_shape = expand_batch_portion
        if dim_tensor1 > 1:
            output_shape.append(n)

        if dim_tensor2 > 1:
            output_shape.append(p)

        if vector_rhs:
            return tensor1_expanded.bmm(tensor2_expanded).squeeze(-1).view(output_shape)
        else:
            return tensor1_expanded.bmm(tensor2_expanded).view(output_shape)
    else:
        torch._check(False, lambda: "both arguments to matmul need to be at least 1D")


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
        return aten._unsafe_index(a, [N_idx, C_idx, y_idx, x_idx])

    def get_x_interp(y):
        coeffs_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)
        return _upsample_cubic_interp1d(coeffs_x, t_x)

    coeffs_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)
    result = _upsample_cubic_interp1d(coeffs_y, t_y)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(a)
    result = result.contiguous(memory_format=memory_format)
    return result


@register_decomposition(aten.upsample_bicubic2d.vec)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.Autograd)
@out_wrapper()
def upsample_bicubic2d_vec(
    a: Tensor,
    output_size: Optional[Tuple[int, int]],
    align_corners: bool,
    scale_factors: Optional[Tuple[float, float]] = None,
) -> Tensor:
    torch._check(
        bool(output_size) + bool(scale_factors) == 1,
        lambda: "Must specify exactly one of output_size and scale_factors.",
    )
    if output_size is None:
        assert scale_factors is not None
        output_size = cast(
            Tuple[int, int],
            tuple(
                sym_int(sym_float(w) * scale)
                for w, scale in zip(a.shape[2:], scale_factors)
            ),
        )
    scale_h, scale_w = scale_factors if scale_factors else (None, None)
    return aten.upsample_bicubic2d.default(
        a, output_size, align_corners, scale_h, scale_w
    )


@register_decomposition(aten.aminmax)
@out_wrapper("min", "max")
def aminmax(self, *, dim=None, keepdim=False):
    amin = torch.amin(self, dim=dim, keepdim=keepdim)
    amax = torch.amax(self, dim=dim, keepdim=keepdim)
    return amin, amax


@register_decomposition(aten.nansum)
@out_wrapper()
def nansum(self, dim=None, keepdim=False, *, dtype=None):
    return aten.sum(torch.where(torch.isnan(self), 0, self), dim, keepdim, dtype=dtype)


@register_decomposition([aten.arange.default, aten.arange.out])
@out_wrapper()
def arange_default(
    end: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
):
    return aten.arange.start_step(
        0, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_decomposition([aten.arange.start])
def arange_start(
    start: NumberType,
    end: NumberType,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
):
    return aten.arange.start_step(
        start, end, 1, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory
    )


@register_decomposition(aten.multi_margin_loss)
@aten.multi_margin_loss.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: NumberType = 1,
    margin: NumberType = 1,
    weight: Optional[Tensor] = None,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    input = torch.atleast_2d(input)
    target = torch.atleast_1d(target)
    nframe = input.shape[0]
    dim = input.shape[1]
    torch._check(p == 1 or p == 2, lambda: "only p == 1 and p == 2 supported")
    torch._check(
        input.ndim == 2 and dim != 0,
        lambda: f"Expected non-empty vector or matrix with optional 0-dim batch size, but got: {input.shape}",
    )
    torch._check(
        target.ndim == 1 and target.numel() == nframe,
        lambda: f"inconsistent target size, expected {nframe} but got {target.shape}",
    )
    if weight is not None:
        weight = torch.atleast_1d(weight)
        torch._check(
            weight.ndim == 1 and weight.numel() == dim,  # type: ignore[union-attr]
            lambda: f"inconsistent weight size, expected {dim} but got {weight.shape}",  # type: ignore[union-attr]
        )
    target = target.unsqueeze(1)
    u = torch.gather(input, dim=1, index=target)
    z = margin - u + input
    z = z.clamp_min(0)
    z = z if p == 1 else z * z
    if weight is not None:
        z = z * weight[target]
    idx = torch.arange(dim, device=input.device)
    z = torch.where(idx != target, z, 0)
    if reduction == Reduction.MEAN.value:
        return z.mean()
    elif reduction == Reduction.SUM.value:
        return z.sum() / z.shape[1]
    else:
        return z.mean(dim=1)


@register_decomposition(aten.multilabel_margin_loss_forward)
@aten.multilabel_margin_loss_forward.default.py_impl(DispatchKey.Autograd)
@out_wrapper("output", "is_target")
def multilabel_margin_loss_forward(
    input: Tensor,
    target: Tensor,
    reduction: int,
) -> Tuple[Tensor, Tensor]:
    orig_input_shape = input.shape
    orig_target_shape = target.shape
    input = torch.atleast_2d(input)
    target = torch.atleast_2d(target)
    dim = input.shape[1]
    torch._check(
        len(orig_input_shape) <= 2 and dim != 0,
        lambda: f"Expected non-empty vector or matrix with optional 0-dim batch size, but got: {orig_input_shape}",
    )
    torch._check(
        len(orig_target_shape) <= 2 and orig_target_shape == orig_input_shape,
        lambda: f"inconsistent target size: {orig_target_shape} for input of size: {orig_input_shape}",
    )
    # ignores labels after the first -1, detects when -1 is not present
    idx = torch.arange(dim, device=target.device)
    is_end = target == -1
    end_idx = torch.amin(torch.where(is_end, idx, dim), dim=-1, keepdim=True)
    # target indices
    target_mask = idx < end_idx
    # masks target to be able to use gather, which doesn't allow -1
    tidx0 = torch.where(target_mask, target, 0)
    u = torch.gather(input, dim=-1, index=tidx0)
    # is_target
    tidx1 = torch.where(target_mask, target, -1)
    is_target = torch.any(idx == tidx1.unsqueeze(dim=-1), dim=1)
    # loss
    z = 1.0 - u.T.unsqueeze(dim=-1) + input
    z = z.clamp_min(0)
    z = z / dim
    # masks loss
    z = torch.where(is_target, 0, z)
    # reduction
    if reduction == Reduction.MEAN.value:
        z = z.sum(dim=(0, -1)).mean()
    elif reduction == Reduction.SUM.value:
        z = z.sum()
    else:
        z = z.sum(dim=(0, -1))
    # result
    is_target = is_target.to(input.dtype).reshape(orig_target_shape)
    return z, is_target


def register_inplace(aten_op, outplace_op):
    @register_decomposition(aten_op)
    def inplace_op(*args, **kwargs):
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


register_inplace(aten.addbmm_, aten.addbmm)
register_inplace(aten.addmm_, aten.addmm)
register_inplace(aten.addmv_, aten.addmv)
register_inplace(aten.baddbmm_, aten.baddbmm)
register_inplace(aten.fill_, aten.fill)
register_inplace(aten.gelu_, aten.gelu)
register_inplace(aten.hardswish_, aten.hardswish)
register_inplace(aten.hardtanh_, aten.hardtanh)
register_inplace(aten.hardsigmoid_, aten.hardsigmoid)
register_inplace(aten.__iand__, aten.__and__)
register_inplace(aten.__ilshift__, aten.__lshift__)
register_inplace(aten.index_put_, aten.index_put)
register_inplace(aten.index_reduce_, aten.index_reduce)
register_inplace(aten.__ior__, aten.__or__)
register_inplace(aten.__irshift__, aten.__rshift__)
register_inplace(aten.__ixor__, aten.__xor__)
register_inplace(aten.leaky_relu_, aten.leaky_relu)
register_inplace(aten.logit_, aten.logit)
register_inplace(aten.relu_, aten.relu)
register_inplace(aten.renorm_, aten.renorm)
register_inplace(aten.round_, aten.round)
register_inplace(aten.scatter_, aten.scatter)
register_inplace(aten.scatter_add_, aten.scatter_add)
register_inplace(aten.scatter_reduce_, aten.scatter_reduce)
register_inplace(aten.silu_, aten.silu)
