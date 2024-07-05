# mypy: allow-untyped-defs
import builtins
import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Any, Callable, cast, Iterable, List, Optional, Tuple, Union

import torch
import torch._meta_registrations
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import (
    IntLike,
    NumberType,
    suggest_memory_format,
    TensorLike,
    TensorSequenceType,
)
from torch._prims_common.wrappers import (
    _maybe_convert_to_dtype,
    _maybe_resize_out,
    _safe_copy_out,
    out_wrapper,
)
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map

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
            x for x in pytree.arg_tree_leaves(*args, **kwargs) if isinstance(x, Tensor)
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
@out_wrapper("grad_input")
@pw_cast_for_opmath
def tanh_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (1 - y * y).conj_physical()


@register_decomposition(aten.sigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def sigmoid_backward(out_grad: Tensor, y: Tensor):
    return out_grad * (y * (1 - y)).conj_physical()


@register_decomposition(aten.softplus_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def softplus_backward(out_grad: Tensor, x: Tensor, beta: float, threshold: float):
    z = (x * beta).exp()
    return torch.where((x * beta) > threshold, out_grad, out_grad * z / (z + 1.0))


@register_decomposition(aten.elu_backward)
@out_wrapper("grad_input")
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
@out_wrapper()
@pw_cast_for_opmath
def hardsigmoid(self: Tensor) -> Tensor:
    return torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardsigmoid_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def hardsigmoid_backward(grad_output: Tensor, self: Tensor):
    return torch.where(
        (self > -3.0) & (self < 3.0),
        grad_output * (1.0 / 6.0),
        0.0,
    )


@register_decomposition(aten.hardtanh_backward)
@out_wrapper("grad_input")
def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
):
    return torch.where((self <= min_val) | (self >= max_val), 0.0, grad_output)


@register_decomposition(aten.hardswish)
@out_wrapper()
@pw_cast_for_opmath
def hardswish(self: Tensor) -> Tensor:
    return self * torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardswish_backward)
@out_wrapper()
@pw_cast_for_opmath
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    return torch.where(
        self < -3,
        0.0,
        torch.where(self <= 3, grad_output * ((self / 3) + 0.5), grad_output),
    )


@register_decomposition(aten.threshold_backward)
@out_wrapper("grad_input")
def threshold_backward(grad_output: Tensor, self: Tensor, threshold: float):
    return torch.where(self <= threshold, 0, grad_output)


@register_decomposition(aten.leaky_relu_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def leaky_relu_backward(
    grad_output: Tensor, self: Tensor, negative_slope: float, self_is_result: bool
):
    return torch.where(self > 0, grad_output, grad_output * negative_slope)


@register_decomposition(aten.gelu_backward)
@out_wrapper("grad_input")
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
@out_wrapper()
@pw_cast_for_opmath
def silu(self: Tensor) -> Tensor:
    return self * torch.sigmoid(self)


@register_decomposition(aten.silu_backward)
@out_wrapper("grad_input")
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
@out_wrapper()
@pw_cast_for_opmath
def rrelu_with_noise(
    self: Tensor,
    noise: Tensor,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
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
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    return self.copy_(rrelu_with_noise(self, noise, lower, upper, training, generator))


@register_decomposition(aten.rrelu_with_noise_backward)
@out_wrapper()
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
@out_wrapper("grad_input")
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
@out_wrapper()
@pw_cast_for_opmath
def mse_loss(
    self: Tensor, target: Tensor, reduction: int = Reduction.MEAN.value
) -> Tensor:
    loss = (self - target) ** 2
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.mse_loss_backward)
@out_wrapper("grad_input")
@pw_cast_for_opmath
def mse_loss_backward(
    grad_output: Tensor, input: Tensor, target: Tensor, reduction: int
):
    norm = 2.0 / input.numel() if reduction == Reduction.MEAN.value else 2.0
    return norm * (input - target) * grad_output


@register_decomposition(aten.smooth_l1_loss)
@out_wrapper()
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
@out_wrapper("grad_input")
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
@out_wrapper("grad_input")
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
@out_wrapper("grad_input")
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
@out_wrapper()
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
@out_wrapper("grad_input")
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
@out_wrapper("grad_input")
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
@out_wrapper()
def dist(input: Tensor, other: Tensor, p: float = 2):
    return aten.norm(input - other, p=p)


@register_decomposition(aten._euclidean_dist)
@out_wrapper()
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
@out_wrapper()
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
    from torch.fx.experimental.symbolic_shapes import (
        guard_size_oblivious,
        statically_known_true,
    )

    ndim = self.dim()
    if ndim == 0:
        raise RuntimeError("slice() cannot be applied to a 0-dim tensor.")
    dim = utils.canonicalize_dim(self.dim(), dim)
    sizes = list(self.size())
    strides = list(self.stride())

    if step <= 0:
        raise RuntimeError("slice step must be positive")

    start_val = start if start is not None else 0
    end_val = end if end is not None else sys.maxsize  # 2^63 - 1

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
    elif statically_known_true(end_val == sys.maxsize) or guard_size_oblivious(
        end_val > sizes[dim]
    ):
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
@out_wrapper()
def select_backward(grad_output: Tensor, input_sizes: List[int], dim: int, index: int):
    grad_input = grad_output.new_zeros(input_sizes)
    return torch.select_scatter(grad_input, grad_output, dim, index)


@register_decomposition(aten.diagonal_backward)
@out_wrapper()
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
@out_wrapper("grad_input")
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
@out_wrapper()
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
@out_wrapper()
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
@out_wrapper()
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


@register_decomposition(aten.dropout)
@aten.dropout.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.dropout.default.py_impl(DispatchKey.Autograd)
def dropout(input: Tensor, p: float, train: Optional[bool]):
    if train and p != 0:
        return aten.native_dropout(input, p, train)[0]
    else:
        return input.clone()


@register_decomposition(aten.native_dropout)
@out_wrapper("out0", "out1")
def native_dropout(input: Tensor, p: float, train: Optional[bool]):
    if train and p != 0:
        if p == 1:
            return (torch.zeros_like(input), torch.zeros_like(input, dtype=torch.bool))
        if not input.dtype.is_floating_point:
            raise RuntimeError(
                "result type Float can't be cast to the desired output type Long"
            )
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


@register_decomposition(aten.embedding)
@out_wrapper()
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
@out_wrapper()
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


def _pad_chunk(
    tensors: List[Tensor],
    dim: int,
    num_chunks: int,
) -> List[Tensor]:
    padded_tensors = []
    for tensor in tensors:
        tensor_size = tensor.size()
        pad_along_dim = (tensor_size[dim] + num_chunks - 1) // num_chunks * num_chunks
        if pad_along_dim != tensor_size[dim]:
            # Use aten.constant_pad_nd instead of copy_ for functionalization
            pad = [0] * 2 * (tensor.ndim - dim - 1) + [
                0,
                pad_along_dim - tensor_size[dim],
            ]
            tensor = aten.constant_pad_nd(tensor, pad, 0)
        view_size = tensor_size[:dim] + torch.Size([num_chunks, -1])
        padded_tensors.append(tensor.view(view_size))
    return padded_tensors


def have_same_ndims(tensors: List[Tensor]):
    ndim = tensors[0].ndim
    for tensor in tensors:
        if tensor.ndim != ndim:
            return False
    return True


def leading_dimension_matches(tensors: List[Tensor], dim: int):
    leading_dim_sizes = tensors[0].size()[:dim]
    for tensor in tensors:
        torch._check(
            tensor.size()[:dim] == leading_dim_sizes,
            lambda: "_chunk_cat expects same sizes of 0,...,dim-1 dimensions for all tensors",
        )


def _preprocess_chunk_cat_inputs(
    tensors: List[Tensor],
    dim: int,
    num_chunks: int,
):
    torch._check(num_chunks >= 1, lambda: "_chunk_cat expects positive num_chunks")
    torch._check(
        len(tensors) > 0, lambda: "_chunk_cat expects a non-empty input tensor list"
    )
    expected_dtype = tensors[0].dtype
    expected_device = tensors[0].device
    for tensor in tensors:
        torch._check(tensor.numel() > 0, lambda: "_chunk_cat expects non-empty tensor")
        torch._check(
            tensor.dtype == expected_dtype,
            lambda: "_chunk_cat expects all input tensors with the same dtype",
        )
        torch._check(
            tensor.device == expected_device,
            lambda: "_chunk_cat expects all inputs tensors on the same device",
        )
    if have_same_ndims(tensors):
        dim = utils.canonicalize_dim(tensors[0].dim(), dim)
    else:
        torch._check(
            dim >= 0,
            lambda: "_chunk_cat expects non-negative dim when input tensors have different ndims",
        )
        for tensor in tensors:
            torch._check(
                dim < tensor.ndim,
                lambda: "_chunk_cat expects dim < ndim for all input tensors",
            )
    leading_dimension_matches(tensors, dim)
    return dim


@register_decomposition([aten._chunk_cat.default, aten._chunk_cat.out])
def _chunk_cat(
    tensors: List[Tensor],
    dim: int,
    num_chunks: int,
    out: Optional[Tensor] = None,
) -> Tensor:
    dim = _preprocess_chunk_cat_inputs(tensors, dim, num_chunks)
    padded_tensors = _pad_chunk(tensors, dim, num_chunks)
    if out is None:
        return torch.cat(padded_tensors, dim + 1)
    else:
        torch.cat(padded_tensors, dim + 1, out=out)
        return out


@register_decomposition(aten.split_with_sizes)
def split_with_sizes(
    self: Tensor, split_sizes: List[int], dim: int = 0
) -> List[Tensor]:
    # NB: Perform the check_is_size tests first so that the
    # sum test does not try to do a replacement
    for i in range(len(split_sizes)):
        torch._check_is_size(
            split_sizes[i],
            lambda: "split_with_sizes expects split_sizes have only non-negative entries",
        )
    torch._check_with(
        ValueError,
        sum(split_sizes) == self.shape[dim],
        lambda: f"Split sizes add up to {sum(split_sizes)} but got the tensor's size of {self.shape[dim]}",
    )
    num_splits = len(split_sizes)
    splits = []
    start_idx = 0

    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import expect_true

    for i in range(num_splits):
        length = split_sizes[i]
        # We know this is true thanks to the sum, but this assertion helps
        # out our internal reasoning
        expect_true(start_idx + length <= self.shape[dim])
        splits.append(self.narrow(dim, start_idx, length))
        start_idx += length
    return splits


# out_wrapper currently does not allow optional outputs
@register_decomposition(
    [aten.split_with_sizes_copy.default, aten.split_with_sizes_copy.out]
)
def split_with_sizes_copy(
    self: Tensor,
    split_sizes: List[int],
    dim: int = 0,
    out: Optional[List[Tensor]] = None,
) -> Optional[List[Tensor]]:
    splits = split_with_sizes(self, split_sizes, dim=dim)
    if out is None:
        return [s.clone(memory_format=torch.contiguous_format) for s in splits]
    else:
        for output, split in zip(out, splits):
            _maybe_resize_out(output, split.shape)
            _safe_copy_out(copy_from=split, copy_to=output, exact_dtype=True)
        return None


@register_decomposition(aten.unsafe_split.Tensor)
def unsafe_split(input: Tensor, split_size: int, dim: int = 0) -> Tuple[Tensor, ...]:
    return aten.split.Tensor(input, split_size, dim)


@register_decomposition(aten.unsafe_split_with_sizes.default)
def unsafe_split_with_sizes(
    input: Tensor, split_sizes: List[int], dim: int = 0
) -> Tuple[Tensor, ...]:
    return aten.split_with_sizes.default(input, split_sizes, dim)


@register_decomposition(aten.split.Tensor)
def split(self: Tensor, split_size: int, dim: int = 0) -> Tuple[Tensor, ...]:
    input_sizes = self.shape
    dim_size = input_sizes[dim]
    if split_size == 0:
        assert dim_size == 0
        return (self,)
    chunks = (dim_size + split_size - 1) // split_size

    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import guard_int

    chunks = guard_int(chunks)
    split_sizes = [split_size for i in range(chunks)]
    split_sizes[-1] = split_size - (split_size * chunks - dim_size)
    return torch.split(self, split_sizes, dim)


@aten.tensor_split.tensor_indices_or_sections.py_impl(
    DispatchKey.CompositeImplicitAutograd
)
def tensor_split_tensor_indices_or_sections_py_impl(
    self: Tensor,
    tensor_indices_or_sections: Tensor,
    dim: int = 0,
) -> Tuple[Tensor, ...]:
    assert tensor_indices_or_sections.device.type == "cpu"
    assert tensor_indices_or_sections.dtype == torch.int64
    split_dim = tensor_indices_or_sections.dim()
    torch._check(
        split_dim == 1 or split_dim == 0,
        lambda: "tensor_split expected tensor_indices_or_sections to be a zero-dimensional "
        f"or one-dimensional tensor, but got a tensor with {split_dim} dims",
    )
    if split_dim == 0:
        sections = tensor_indices_or_sections.item()
        assert isinstance(sections, IntLike)
        return self.tensor_split(sections, dim)
    else:
        indices = [i.item() for i in tensor_indices_or_sections]
        # WARNING: Tempted to torch._check_is_size on the indices here?  You
        # can't: tensor_split works with negative values in indices:
        #
        # >>> torch.tensor_split(torch.randn(10), torch.tensor([-5, 5]))
        # (tensor([ 0.3540,  2.1074, -0.8507,  1.1639,  0.3055]), tensor([]),
        # tensor([-0.4285,  1.0692, -0.1776,  0.9362,  1.6143]))
        #
        # Sorry, I don't make the rules.  Explicitly do the item call in user
        # code if you KNOW that they are non-negative.
        return self.tensor_split(indices, dim)


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


@register_decomposition(aten.native_group_norm_backward.default)
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


# out_wrapper currently does not allow optional outputs
@register_decomposition(aten.native_group_norm_backward.out)
def native_group_norm_backward_out(
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
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    result = native_group_norm_backward(
        grad_output, input, mean, rstd, gamma, N, C, HxW, group, output_mask
    )
    grad_input = (out0, out1, out2)
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)

    return grad_input


def _maybe_cast(x: Optional[Tensor], dtype) -> Optional[Tensor]:
    if x is not None:
        return x.to(dtype)
    return x


# TODO: Take a closer look at the type promotion semantics
@register_decomposition(aten.native_layer_norm_backward.default)
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
    mean = _unsqueeze_to_dim(mean, input_cast.dim())  # type: ignore[union-attr]
    rstd = _unsqueeze_to_dim(rstd, input_cast.dim())  # type: ignore[union-attr]
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


# out_wrapper currently does not allow optional outputs
@register_decomposition(aten.native_layer_norm_backward.out)
def native_layer_norm_backward_out(
    grad_out: Tensor,
    input: Tensor,
    normalized_shape: List[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: List[bool],
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    result = native_layer_norm_backward(
        grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask
    )
    grad_input = (out0, out1, out2)
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)

    return grad_input


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
@out_wrapper("out", "save_mean", "save_invstd")
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
@aten.native_batch_norm.default.py_impl(DispatchKey.CompositeImplicitAutograd)
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


def _get_batch_norm_reserve_tensor(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    eps: float,
    training: bool,
) -> Tensor:
    """
    Return a reserve tensor for batch norm, used only by cudnn to pass forward state to the
    backward pass. This is needed for `_batch_norm_with_update` and `_batch_norm_no_update`,
    which support a variety of backends including cudnn. We create this tensor here to get
    the correct shape in the traced graph if we detect that will call the cudnn kernel,
    and rely on DCE to avoid materializing this tensor.
    """
    backend = torch._C._select_batch_norm_backend(  # type: ignore[attr-defined]
        input, weight, bias, running_mean, running_var, True, eps
    )
    reserve_size = 0
    if backend == torch._C._BatchNormBackend.Cudnn:  # type: ignore[attr-defined]
        reserve_size = torch._C._get_cudnn_batch_norm_reserve_space_size(input, training)  # type: ignore[attr-defined]
    return torch.empty(
        reserve_size, dtype=torch.uint8, layout=input.layout, device=input.device
    )


@register_decomposition(aten._batch_norm_with_update.default)
def _batch_norm_with_update(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        True,  # training
        momentum,
        eps,
        False,  # functional
    )
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=True
    )
    return output, save_mean, save_rstd, reserve


@register_decomposition(aten._batch_norm_with_update_functional.default)
def _batch_norm_with_update_functional(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        output,
        save_mean,
        save_rstd,
        new_rm,
        new_rv,
    ) = native_batch_norm_helper(
        input, weight, bias, running_mean, running_var, True, momentum, eps, True
    )
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=True
    )
    assert new_rm is not None, "new_running_mean should not be None"
    assert new_rv is not None, "new_running_var should not be None"
    return (output, save_mean, save_rstd, reserve, new_rm, new_rv)


@register_decomposition(aten._batch_norm_no_update.default)
def _batch_norm_no_update(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    running_mean: Tensor,
    running_var: Tensor,
    momentum: float,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    output, save_mean, save_rstd, _, _ = native_batch_norm_helper(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        False,  # training
        momentum,
        eps,
        False,  # functional
    )
    reserve = _get_batch_norm_reserve_tensor(
        input, weight, bias, running_mean, running_var, eps, training=False
    )
    return output, save_mean, save_rstd, reserve


@register_decomposition(aten._fused_dropout)
@out_wrapper("out0", "out1")
@pw_cast_for_opmath
def _fused_dropout_decomposition(input, p, generator=None):
    assert generator is None
    mask = (torch.rand_like(input) < p).to(dtype=torch.uint8)
    res = mask.type_as(input) * input * (1.0 / p)
    return (res, mask)


@register_decomposition(aten._to_copy)
@out_wrapper()
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
        dtype_converted = True

    if memory_format is not None:  # no ref/prim for memory format
        return torch.clone(x, memory_format=memory_format)
    return x


# Questionable decompositions
# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition([aten.detach, aten.lift, aten.lift_fresh])
@out_wrapper()
def nop_decomposition(x):
    return aten.alias(x)


# Also register to the Autograd dispatch key, so this decomp can run above autograd.
# native_batch_norm needs to decompose into other ops before autograd.
@aten.cudnn_batch_norm.default.py_impl(DispatchKey.Autograd)
@register_decomposition(aten.cudnn_batch_norm)
@out_wrapper("out0", "out1", "out2", "out3")
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
        if mask == 1 and not (axis < x.ndim and x.shape[axis] == mask):
            x = x.unsqueeze(axis)
    return x


@register_decomposition(aten.batch_norm_backward.default)
def batch_norm_backward(
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
    reserve: Tensor,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    return native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    )


@register_decomposition(aten.native_batch_norm_backward.default)
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


# out_wrapper currently does not allow optional outputs
@register_decomposition(aten.native_batch_norm_backward.out)
def native_batch_norm_backward_out(
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
    *,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    result = native_batch_norm_backward(
        grad_out,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        train,
        eps,
        output_mask,
    )
    grad_input = (out0, out1, out2)
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)

    return grad_input


@register_decomposition(aten.miopen_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
def miopen_batch_norm_backward(
    input: Tensor,
    grad_output: Tensor,
    weight: Tensor,
    running_mean: Optional[Tensor],
    running_var: Optional[Tensor],
    save_mean: Optional[Tensor],
    save_var: Optional[Tensor],
    epsilon: float,
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


@register_decomposition(aten.cudnn_batch_norm_backward)
@out_wrapper("out0", "out1", "out2")
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
@out_wrapper()
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
            # Need to clamp to avoid accessing out-of-bounds memory
            # TODO make minimum accept scalars
            maxval = torch.scalar_tensor(
                in_size - 1, dtype=idx.dtype, device=idx.device
            )
            idx = torch.minimum(idx, maxval)

            # Compute the length
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
    index_size = index.size(0) if index.ndim == 1 else 1
    tensor_size = tensor.size(dim) if tensor.ndim > 0 else 1
    torch._check(
        tensor_size == index_size,
        lambda: f"Number of indices ({index_size}) should be equal to tensor.size(dim) ({tensor_size}), for {dim=}",
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


@register_decomposition(aten.pad_sequence.default)
@aten.pad_sequence.default.py_impl(DispatchKey.CompositeImplicitAutograd)
def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    torch._check(len(sequences) > 0, lambda: "received an empty list of sequences")
    sequences_size = len(sequences)
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(x.size(0) for x in sequences)
    if batch_first:
        out_dims = (sequences_size, max_len)
    else:
        out_dims = (max_len, sequences_size)
    out_dims = out_dims + trailing_dims
    out = sequences[0].new_full(out_dims, padding_value)
    dim_paddings = (0, 0) * len(trailing_dims)
    for i in range(sequences_size):
        currseq = sequences[i]
        row = aten.constant_pad_nd(
            currseq, dim_paddings + (0, max_len - currseq.size(0)), padding_value
        )
        if batch_first:
            out = aten.select_scatter(out, row, dim=0, index=i)
        else:
            out = aten.select_scatter(out, row, dim=1, index=i)
    return out


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
    index = index.unsqueeze(0) if index.ndim == 0 else index
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
@out_wrapper()
def uniform(
    x: Tensor,
    low: Union[bool, int, float] = 0.0,
    high: Union[bool, int, float] = 1.0,
    generator: Optional[torch.Generator] = None,
):
    return prims._uniform_helper(
        x.shape,
        low=sym_float(low),
        high=sym_float(high),
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )


@register_decomposition(aten.uniform_)
def uniform_(self, low=0, high=1, generator=None):
    return self.copy_(uniform(self, low, high, generator))


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
@register_decomposition(aten.upsample_nearest2d.vec)
@register_decomposition(aten.upsample_nearest3d.vec)
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest1d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest2d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest3d.vec.py_impl(DispatchKey.Autograd)
def _upsample_nearest_vec(
    input: Tensor,
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]],
) -> Tensor:
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scales = (
        scale_factors if scale_factors else [None] * len(osize)  # type: ignore[list-item]
    )
    return _upsample_nearest(input, osize, scales)


@register_decomposition(aten._upsample_nearest_exact1d.vec)
@register_decomposition(aten._upsample_nearest_exact2d.vec)
@register_decomposition(aten._upsample_nearest_exact3d.vec)
@aten._upsample_nearest_exact1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact1d.vec.py_impl(DispatchKey.Autograd)
@aten._upsample_nearest_exact2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact2d.vec.py_impl(DispatchKey.Autograd)
@aten._upsample_nearest_exact3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact3d.vec.py_impl(DispatchKey.Autograd)
def _upsample_nearest_exact_vec(
    input: Tensor,
    output_size: Optional[List[int]],
    scale_factors: Optional[List[float]],
) -> Tensor:
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scales = (
        scale_factors if scale_factors else [None] * len(osize)  # type: ignore[list-item]
    )
    return _upsample_nearest(input, osize, scales, exact=True)


def _compute_upsample_nearest_indices(input, output_size, scales, exact=False):
    # For each dim in output_size, compute the set of input indices used
    # to produce the upsampled output.
    indices = []
    num_spatial_dims = len(output_size)
    offset = 0.5 if exact else 0.0

    for d in range(num_spatial_dims):
        # Math matches aten/src/ATen/native/cpu/UpSampleKernel.cpp
        #
        # Indices are computed as following:
        # scale = isize / osize
        # Case: exact=False
        # input_index = floor(output_index * scale)
        # Same as OpenCV INTER_NEAREST
        #
        # Case: exact=False
        # index_f32 = (output_index + 0.5) * scale - 0.5
        # input_index = round(index_f32)
        # Same as Pillow and Scikit-Image/Scipy ndi.zoom
        osize = output_size[d]
        isize = input.shape[-num_spatial_dims + d]
        scale = isize / (isize * scales[d]) if scales[d] is not None else isize / osize

        output_indices = torch.arange(osize, dtype=torch.float32, device=input.device)
        input_indices = ((output_indices + offset) * scale).to(torch.int64)
        for _ in range(num_spatial_dims - 1 - d):
            input_indices = input_indices.unsqueeze(-1)
        indices.append(input_indices)
    return indices


@register_decomposition([aten.upsample_nearest1d.default, aten.upsample_nearest1d.out])
@aten.upsample_nearest1d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest1d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest1d(
    input: Tensor,
    output_size: List[int],
    scales: Optional[float] = None,
) -> Tensor:
    return _upsample_nearest(input, output_size, [scales])


@register_decomposition(
    [aten._upsample_nearest_exact1d.default, aten._upsample_nearest_exact1d.out]
)
@aten._upsample_nearest_exact1d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact1d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest_exact1d(
    input: Tensor,
    output_size: List[int],
    scales: Optional[float] = None,
) -> Tensor:
    return _upsample_nearest(input, output_size, [scales], exact=True)


@register_decomposition([aten.upsample_nearest2d.default, aten.upsample_nearest2d.out])
@aten.upsample_nearest2d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest2d(
    input: Tensor,
    output_size: List[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    return _upsample_nearest(input, output_size, [scales_h, scales_w])


@register_decomposition(
    [aten._upsample_nearest_exact2d.default, aten._upsample_nearest_exact2d.out]
)
@aten._upsample_nearest_exact2d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def _upsample_nearest_exact2d(
    input: Tensor,
    output_size: List[int],
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    return _upsample_nearest(input, output_size, [scales_h, scales_w], exact=True)


@register_decomposition([aten.upsample_nearest3d.default, aten.upsample_nearest3d.out])
@aten.upsample_nearest3d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_nearest3d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def upsample_nearest3d(
    input: Tensor,
    output_size: List[int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    return _upsample_nearest(input, output_size, [scales_d, scales_h, scales_w])


@register_decomposition(
    [aten._upsample_nearest_exact3d.default, aten._upsample_nearest_exact3d.out]
)
@aten._upsample_nearest_exact3d.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_nearest_exact3d.default.py_impl(DispatchKey.Autograd)
@out_wrapper(preserve_memory_format=True, exact_dtype=True)
def _upsample_nearest_exact3d(
    input: Tensor,
    output_size: List[int],
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    return _upsample_nearest(
        input, output_size, [scales_d, scales_h, scales_w], exact=True
    )


@pw_cast_for_opmath
def _upsample_nearest(
    input: Tensor,
    output_size: List[int],
    scales: List[Optional[float]],
    exact: bool = False,
) -> Tensor:
    spatial_indices = _compute_upsample_nearest_indices(
        input, output_size, scales, exact=exact
    )

    indices = [None, None] + spatial_indices
    result = aten._unsafe_index(input, indices)

    if result.ndim == 4:
        # convert output to correct memory format, if necessary
        memory_format = utils.suggest_memory_format(input)

        # following "heuristic: only use channels_last path when it's faster than the contiguous path"
        n_channels = input.shape[1]
        if input.device.type == "cuda" and n_channels < 4:
            memory_format = torch.contiguous_format

        result = result.contiguous(memory_format=memory_format)
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
            input = torch.cat([fwd_inp, bwd_inp], fwd_inp.dim() - 1)  # type: ignore[possibly-undefined]
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
        * ``torch._C._get_mkldnn_enabled()`` returns ``True``.
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
        if not torch._C._get_mkldnn_enabled():
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


@register_decomposition(aten._upsample_bicubic2d_aa.vec)
@aten._upsample_bicubic2d_aa.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten._upsample_bicubic2d_aa.vec.py_impl(DispatchKey.Autograd)
def upsample_bicubic2d_aa_vec(input, output_size, align_corners, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scale_h = get_scale_value(scale_factors, 0)
    scale_w = get_scale_value(scale_factors, 1)
    return torch.ops.aten._upsample_bicubic2d_aa(
        input, osize, align_corners, scale_h, scale_w
    )


@register_decomposition(aten.upsample_bilinear2d.vec)
@register_decomposition(aten.upsample_trilinear3d.vec)
@aten.upsample_linear1d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_linear1d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_bilinear2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_bilinear2d.vec.py_impl(DispatchKey.Autograd)
@aten.upsample_trilinear3d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_trilinear3d.vec.py_impl(DispatchKey.Autograd)
def _upsample_linear_vec(input, output_size, align_corners, scale_factors):
    osize = upsample_compute_output_size(input.size(), output_size, scale_factors)
    scales = scale_factors if scale_factors else [None] * len(osize)
    return _upsample_linear(input, osize, align_corners, scales)


@register_decomposition([aten.upsample_linear1d.default, aten.upsample_linear1d.out])
@out_wrapper()
def upsample_linear1d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_w: Optional[float] = None,
) -> Tensor:
    return _upsample_linear(input, output_size, align_corners, [scales_w])


@register_decomposition(
    [aten.upsample_bilinear2d.default, aten.upsample_bilinear2d.out]
)
@aten.upsample_bilinear2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
def upsample_bilinear2d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    return _upsample_linear(input, output_size, align_corners, [scales_h, scales_w])


@register_decomposition(
    [aten.upsample_trilinear3d.default, aten.upsample_trilinear3d.out]
)
@out_wrapper()
def upsample_trilinear3d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_d: Optional[float] = None,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    return _upsample_linear(
        input, output_size, align_corners, [scales_d, scales_h, scales_w]
    )


def _compute_scale(in_size, out_size, align_corners, scale=None):
    if align_corners:
        return (in_size - 1.0) / (out_size - 1.0) if out_size > 1 else 0
    else:
        return 1.0 / scale if scale is not None and scale > 0 else in_size / out_size


def _compute_source_index(scale, dst_index, align_corners):
    if align_corners:
        return scale * dst_index
    else:
        return scale * (dst_index + 0.5) - 0.5


def _sum_tensors_uint8(
    src: Iterable[Tensor], weights: Iterable[Tensor], weights_precision: Tensor
) -> Tensor:
    output = _sum_tensors(
        s.to(torch.int32) * c.to(torch.int32) for s, c in zip(src, weights)
    ) + (1 << (weights_precision - 1))
    output = output >> weights_precision
    return torch.clamp(output, 0, 255).to(torch.uint8)


def _compute_weight_precision(weights: TensorSequenceType) -> Tensor:
    max_weight = torch.stack(weights).max()
    max_weight_precision = 22
    precisions = torch.arange(max_weight_precision, device=max_weight.device)
    values = 0.5 + max_weight * (1 << (precisions + 1))
    mask = values >= (1 << 15)
    return max_weight_precision - mask.sum()


@pw_cast_for_opmath
def _upsample_linear(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales: List[Optional[float]],
) -> Tensor:
    # get dimensions of original image
    n_batch, n_channels = input.shape[:2]
    inp_sizes = input.shape[2:]
    n_dims = len(inp_sizes)

    _, dtype = utils.elementwise_dtypes(
        input,
        type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )

    def get_values(inp_size, out_size, scales, nsqueeze):
        # First Calculate scaling factor
        scale_factor = _compute_scale(inp_size, out_size, align_corners, scales)
        # We have to create arange with int64 dtype and use .to in order to avoid
        # additional kernels creation in inductor and get a perf slowdown
        i = torch.arange(out_size, device=input.device).to(dtype=dtype)

        x_f32 = _compute_source_index(scale_factor, i, align_corners).clamp(min=0.0)
        x_f32 = x_f32.reshape(x_f32.shape[0], *[1] * (nsqueeze))
        x = x_f32.to(torch.int64)
        xp1 = (x + 1).clamp(max=inp_size - 1)
        return x_f32, x, xp1

    values = [
        get_values(inp_size, out_size, scales, n_dims - 1 - i)
        for i, (inp_size, out_size, scales) in enumerate(
            zip(inp_sizes, output_size, scales)
        )
    ]
    xs_f32, xs, xp1s = list(zip(*values))

    vs = []
    for a in product(*[[0, 1]] * n_dims):
        idx = [None, None] + [xs[k] if a[k] == 0 else xp1s[k] for k in range(n_dims)]
        v = aten._unsafe_index(input, idx)
        v = _maybe_convert_to_dtype(v, dtype)
        vs.append(v)

    for i in reversed(range(n_dims)):
        xscale = (xs_f32[i] - xs[i]).clamp(0.0, 1.0).to(dtype)
        vs = [
            # x1 * (1 - alpha) + x2 * alpha == x1 + (x2 - x1) * alpha
            v1 + torch.mul(v2 - v1, xscale)
            for v1, v2 in zip(vs[::2], vs[1::2])
        ]

    assert len(vs) == 1
    result = vs[0]

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    if input.device.type == "cuda" and n_channels < 16:
        memory_format = torch.contiguous_format

    assert isinstance(result, torch.Tensor)

    result = result.contiguous(memory_format=memory_format)

    if not input.is_floating_point():
        result = result.round()

    return result


# We should be applying decompositions after all transformations
@register_decomposition(aten.is_same_size.default)
def is_same_size(a: Tensor, b: Tensor) -> bool:
    return a.shape == b.shape


@register_decomposition([aten._reshape_alias, aten._unsafe_view])
@out_wrapper()
def _reshape_alias(x, shape, *args):
    return aten.view(x, shape)


@register_decomposition([aten._unsafe_index])
def _unsafe_index(x, indices):
    return aten.index(x, indices)


@register_decomposition([aten._unsafe_masked_index])
def _unsafe_masked_index(x, mask, indices, fill):
    for index in indices:
        if index is not None:
            torch._check(
                index.dtype in [torch.long, torch.int],
                lambda: "tensors used as indices must be long or int tensors",
            )

    torch._check(
        mask.dtype == torch.bool,
        lambda: "tensors used as masks must be bool tensors",
    )

    if x.numel() == 0:
        meta_result = torch._meta_registrations.meta_index_Tensor(x, indices)
        return x.new_full(meta_result.shape, fill)

    for i in range(len(indices)):
        index = indices[i]
        if index is not None:
            indices[i] = index.clamp(min=0, max=x.size(i) - 1)

    return aten._unsafe_index(x, indices).masked_fill(~mask, fill)


@register_decomposition([aten._unsafe_masked_index_put_accumulate])
def _unsafe_masked_index_put_accumulate(x, mask, indices, values):
    for index in indices:
        if index is not None:
            torch._check(
                index.dtype in [torch.long, torch.int],
                lambda: "tensors used as indices must be long or int tensors",
            )

    torch._check(
        mask.dtype == torch.bool,
        lambda: "tensors used as masks must be bool tensors",
    )

    if x.numel() == 0:
        return x.clone()

    for i in range(len(indices)):
        index = indices[i]
        if index is not None:
            indices[i] = index.clamp(min=-x.size(i), max=x.size(i) - 1)

    masked_value = values.masked_fill(~mask, 0)
    return aten._unsafe_index_put(x, indices, masked_value, accumulate=True)


@register_decomposition(aten.constant_pad_nd)
@out_wrapper()
def constant_pad_nd(
    input: Tensor,
    pad: Tuple[int, ...],
    value: NumberType = 0,
) -> Tensor:
    # Avoid importing sympy at a module level
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    if builtins.all(statically_known_true(p <= 0) for p in pad):
        import torch._refs as refs

        return refs.constant_pad_nd(input, pad, value)

    torch._check(
        len(pad) % 2 == 0,
        lambda: "constant_pad_nd requires an even number of padding",
    )

    dim = len(pad) // 2
    inp_shape = input.shape[-dim:]
    nc_dim = input.dim() - dim

    pad_left = [pad[2 * (dim - 1 - i)] for i in range(dim)]
    pad_right = [pad[2 * (dim - 1 - i) + 1] for i in range(dim)]

    if input.numel() == 0:
        shape = list(input.shape)
        for i in range(dim):
            shape[input.ndim - 1 - i] += pad[2 * i] + pad[2 * i + 1]
        result = input.new_full(shape, value)
        memory_format = utils.suggest_memory_format(input)
        result = result.contiguous(memory_format=memory_format)
        return result

    out_indices = [
        torch.arange(
            -pad_left[i], inp_shape[i] + pad_right[i], device=input.device
        ).reshape(-1, *[1] * (dim - 1 - i))
        for i in range(dim)
    ]

    indices: List[Any] = [None] * input.dim()
    for i in range(dim):
        indices[i + nc_dim] = out_indices[i]

    conds = []
    for i in range(dim):
        view_shape = [1] * input.dim()
        view_shape[nc_dim + i] = out_indices[i].shape[0]
        idx = out_indices[i].view(view_shape)
        conds.append(torch.logical_and(idx >= 0, idx < input.shape[nc_dim + i]))
    mask = reduce(torch.logical_and, conds)
    result = aten._unsafe_masked_index(input, mask, indices, value)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)
    result = result.contiguous(memory_format=memory_format)
    return result


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
@out_wrapper("output", "total_weight")
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
@out_wrapper("output", "total_weight")
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

    if t.device == torch.device("cpu"):
        tt1 = torch.stack([t, 1.0 - t], dim=0)
        tt2 = torch.stack([t + 1.0, 2.0 - t], dim=0)
        w03 = _upsample_cubic_convolution2(tt2, A)
        w12 = _upsample_cubic_convolution1(tt1, A)
        w0, w3 = torch.unbind(w03, dim=0)
        w1, w2 = torch.unbind(w12, dim=0)
        return w0, w1, w2, w3
    else:
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
@out_wrapper()
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


def _grid_sampler_2d(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
    _expand_grid: bool = True,
) -> Tensor:
    # This method is a copy of grid_sampler_2d implementation and introduced with additional arg _expand_grid to
    # optionally expand the input grid for performance reasons.
    # Experimenting locally it was found that compiled CUDA code is accelerated by ~5x
    # and CPU code by ~2x on bicubic mode, if we expand the grid from (N, H, W, 2) into (N, C, H, W, 2)
    # However, this leads to a slowdown around ~0.8x on CPU bilinear mode, channels first.
    # Thus we apply this hack to not expand the grid for this case.

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
    _, oH, oW, two = grid.shape
    assert two == 2

    if _expand_grid:
        # Let's expand grid to [N, C, oH, oW, 2]
        # This allows to generate a single triton cuda kernel instead of two kernels.
        # Two kernels are due source indices, weights have shape (N, 1, oH, oW), xnumel=N*oH*oW
        # and output has shape (N, C, oH, oW), xnumel=N*C*oH*oW
        # Expanding grid to (N, C, oH, oW, two) unifies xnumel to N*C*oH*oW
        grid = grid.view(N, 1, oH, oW, two).expand(N, C, oH, oW, 2)

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
        c = C if _expand_grid else 1
        return tuple(
            torch.where(cond, t, 0).view(N, c, oH, oW)
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

        if not _expand_grid:
            tx = tx.unsqueeze(1)
            ty = ty.unsqueeze(1)

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
            return _upsample_cubic_interp1d(cs, tx)

        coeffs = tuple(get_coeff(ofs) for ofs in range(4))
        return _upsample_cubic_interp1d(coeffs, ty)


@register_decomposition(aten.grid_sampler_2d)
@out_wrapper()
@pw_cast_for_opmath
def grid_sampler_2d(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> Tensor:
    return _grid_sampler_2d(
        a,
        grid=grid,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


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


@register_decomposition(aten.binary_cross_entropy_with_logits)
@out_wrapper()
def binary_cross_entropy_with_logits(
    self, target, weight=None, pos_weight=None, reduction=Reduction.MEAN.value
):
    if pos_weight is not None:
        log_weight = (pos_weight - 1) * target + 1
        loss = (1 - target) * self - (log_weight * F.logsigmoid(self))
    else:
        loss = (1 - target) * self - F.logsigmoid(self)

    if weight is not None:
        loss = loss * weight

    return apply_loss_reduction(loss, reduction)


def should_fold(tensor1: torch.Tensor, tensor2: torch.Tensor, is_out: bool) -> bool:
    # For comments of the logic of this function see eager in /native/LinearAlgebra.cpp

    t1, t2 = (tensor1, tensor2) if tensor1.ndim >= tensor2.ndim else (tensor2, tensor1)

    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    if not (t1.ndim >= 3 and t2.ndim <= 2):
        return False
    if t2.requires_grad and not is_out:
        return True
    if tensor1.ndim == 2:
        return False
    if guard_size_oblivious(t1.numel() == 0):
        return True

    t1_shape = t1.shape
    t1_stride = t1.stride()
    return all(
        st1 == st2 * s2
        for (st1, st2, s2) in zip(t1_stride[:-2], t1_stride[1:-1], t1_shape[1:-1])
    )


@aten.matmul.default.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.matmul.out.py_impl(DispatchKey.CompositeImplicitAutograd)
@out_wrapper(pass_is_out=True)
def matmul(tensor1, tensor2, *, is_out=False):
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
    elif should_fold(tensor1, tensor2, is_out):
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
            if batch_tensor1[0] == 1 and tensor1.requires_grad:
                return matmul(tensor1.squeeze(0), tensor2)
            if batch_tensor2[0] == 1 and tensor2.requires_grad:
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


@register_decomposition([aten.upsample_bicubic2d.default, aten.upsample_bicubic2d.out])
@aten.upsample_bicubic2d.default.py_impl(DispatchKey.Autograd)
@out_wrapper()
@pw_cast_for_opmath
def upsample_bicubic2d_default(
    input: Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scale_h: Optional[float] = None,
    scale_w: Optional[float] = None,
) -> Tensor:
    # get dimensions of original image
    _, _, in_h, in_w = input.shape

    # Calculate horizontal and vertical scaling factor
    h_scale_factor = _compute_scale(in_h, output_size[0], align_corners, scale_h)
    w_scale_factor = _compute_scale(in_w, output_size[1], align_corners, scale_w)

    _, dtype = utils.elementwise_dtypes(
        input, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    )

    # We have to create arange with int64 dtype and use .to in order to avoid
    # additional kernels creation in inductor and get a perf slowdown
    i = torch.arange(output_size[0], device=input.device).to(dtype=dtype)
    j = torch.arange(output_size[1], device=input.device).to(dtype=dtype)

    x_float = _compute_source_index(w_scale_factor, j, align_corners)
    y_float = _compute_source_index(h_scale_factor, i, align_corners)
    y_float = y_float.unsqueeze(-1)

    x = x_float.floor()
    y = y_float.floor()

    # We should also clamp xscale/yscale
    # See guard_index_and_lambda in UpSample.h
    yscale = (y_float - y).clamp(0.0, 1.0)
    xscale = (x_float - x).clamp(0.0, 1.0)
    x = x.to(torch.int64)
    y = y.to(torch.int64)

    iys_ofs = (y - 1, y, y + 1, y + 2)
    ixs_ofs = (x - 1, x, x + 1, x + 2)

    weights_x = _upsample_get_cubic_coefficients(xscale)
    weights_y = _upsample_get_cubic_coefficients(yscale)

    weights_precision_x, weights_precision_y = None, None
    if input.dtype == torch.uint8:
        weights_precision_x = _compute_weight_precision(weights_x)
        weights_precision_y = _compute_weight_precision(weights_y)

        weights_x = [
            (w * (1 << weights_precision_x) + torch.sign(w) * 0.5).to(torch.int16)
            for w in weights_x
        ]
        weights_y = [
            (w * (1 << weights_precision_y) + torch.sign(w) * 0.5).to(torch.int16)
            for w in weights_y
        ]

    def load_bounded(ys, xs):
        y_idx = torch.clamp(ys, 0, in_h - 1)
        x_idx = torch.clamp(xs, 0, in_w - 1)
        v = aten._unsafe_index(input, [None, None, y_idx, x_idx])
        return v

    def get_x_interp(y):
        src_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)
        if input.dtype == torch.uint8:
            assert weights_precision_x is not None
            return _sum_tensors_uint8(src_x, weights_x, weights_precision_x)
        return _sum_tensors(c1 * c2 for (c1, c2) in zip(src_x, weights_x))

    src_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)
    if input.dtype == torch.uint8:
        assert weights_precision_y is not None
        result = _sum_tensors_uint8(src_y, weights_y, weights_precision_y)
    else:
        result = _sum_tensors(c1 * c2 for (c1, c2) in zip(src_y, weights_y))

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)
    result = result.contiguous(memory_format=memory_format)
    return result


@register_decomposition(aten.upsample_bicubic2d.vec)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.CompositeImplicitAutograd)
@aten.upsample_bicubic2d.vec.py_impl(DispatchKey.Autograd)
@out_wrapper()
@pw_cast_for_opmath
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
    return upsample_bicubic2d_default(a, output_size, align_corners, scale_h, scale_w)


@register_decomposition(aten.reflection_pad1d)
@register_decomposition(aten.reflection_pad2d)
@register_decomposition(aten.reflection_pad3d)
@pw_cast_for_opmath
@out_wrapper()
def _reflection_pad(a: Tensor, padding: Tuple[int, ...]) -> Tensor:
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return middle - 1 - (middle - 1 - dim_idx.abs()).abs()

    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )


@register_decomposition(aten.replication_pad1d)
@register_decomposition(aten.replication_pad2d)
@register_decomposition(aten.replication_pad3d)
@pw_cast_for_opmath
@out_wrapper()
def _replication_pad(a: Tensor, padding: Tuple[int, ...]) -> Tensor:
    def idx(left, middle, right):
        dim_idx = torch.arange(-left, middle + right, device=a.device)
        return torch.clamp(dim_idx, 0, middle - 1)

    return _reflection_or_replication_pad(
        a,
        padding,
        idx,
    )


def _reflection_or_replication_pad(
    a: Tensor,
    padding: Tuple[int, ...],
    idx_fn: Callable[[int, int, int], Tensor],
) -> Tensor:
    dim = len(padding) // 2
    torch._check(
        a.dim() in (dim + 1, dim + 2),
        lambda: f"reflection_pad{dim}d requires {dim + 1}D or {dim + 2}D input",
    )
    inp_shape = a.shape[-dim:]
    nc_dim = a.dim() - dim

    padding_left = [padding[2 * (dim - 1 - i)] for i in range(dim)]
    padding_right = [padding[2 * (dim - 1 - i) + 1] for i in range(dim)]

    result = a
    for i in range(dim):
        idx: List[Any] = [None] * result.dim()
        idx[i + nc_dim] = idx_fn(padding_left[i], inp_shape[i], padding_right[i])
        result = aten._unsafe_index(result, idx)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(result)
    result = result.contiguous(memory_format=memory_format)
    return result


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


@register_decomposition(out_dtype)
def out_dtype_decomp(*args, **kwargs):
    from torch._higher_order_ops.out_dtype import out_dtype_dense

    return out_dtype_dense(*args, **kwargs)


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


# scaled_dot_product_attention used to be decomposed in pre-autograd, given that
# it calls _scaled_dot_product_attention_math and
# _scaled_dot_product_attention_math only has a CompositeImplicitAutograd
# kernel. As a result it's decomposed into ops with finer granularity.
# However recent PRs (#103826 #105131 #115913) added new logic in
# scaled_dot_product_attention and now it calls
# _scaled_dot_product_flash_attention_for_cpu in export path. This results
# in _scaled_dot_product_flash_attention_for_cpu showing up in export result.
# This decomposition ensures scaled_dot_product_attention is still decomposed
# the same way as before, i.e., going through
# _scaled_dot_product_attention_math. Notice that this decomp rule should be
# excluded by inductor.
@register_decomposition(aten._scaled_dot_product_flash_attention_for_cpu.default)
def scaled_dot_product_flash_attention_for_cpu(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    attn_mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    dtype = query.dtype
    torch._check(
        torch.is_floating_point(query),
        lambda: f"query must be FP32, FP64, BF16, FP16 but got {query.dtype}",
    )
    torch._check(
        query.dim() == 4 and key.dim() == 4 and value.dim() == 4,
        lambda: f"q, k, v must be a 4 dimensional tensor, got {query.dim()}, {key.dim()}, {value.dim()}",
    )
    torch._check(
        dropout_p == 0.0, lambda: f"dropout probability must be zero, got {dropout_p}"
    )
    torch._check(
        query.shape[3] == value.shape[3] and key.shape[3] == value.shape[3],
        lambda: "q, k, v should have the same head size",
    )

    output, attn = aten._scaled_dot_product_attention_math.default(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        dropout_mask=None,
        scale=scale,
    )
    # Why this change?
    # In pre-dispatch export scaled_dot_product_attention is executed via
    # * flash_attention.
    # flash_attention allocates output tensor as (N, L, H, E)
    #   it then transposes that to get (N, H, L, E) which is supposed to be the return
    # tensor dim for scaled_dot_product_attention
    # assume x: [N, H, L, E] is the output sdpa
    # In MHA code, this output is then permuted via (2, 0, 1, 3) to get
    # (L, N, H, E) dim tensor
    # x = x.permute(2, 0, 1, 3).contiguous() and the viewed via
    # x = x.view(L * N, H * E)
    # During pre autograd dispatch call to contiguous is not traced because
    # flash_attention output after the x.permute is already contiguous
    # on which the view is valid
    # However, during 2nd stage export, post-dispatch, we run _match variant
    # instead of flash* to get the decomposition. _match variant returns
    # x: [N, H, L, E] applying x.permute(2, 0, 1, 3) returns
    # x: [L, N, H, E] and without converting this to contiguous tensor
    # subsequent view is not valid and the export fails
    # solution is to maintain the return tensor view from the decomp to be
    # exactly same as *flash* variant.
    # flash variants output is contiguous as [N, L, H, E]
    # _match variant out is contiguous as [N, H, L, E]
    # out = out.transpose(1, 2).contiguous gets output as contiguous
    # in [N, L, H, E].
    # Subsrequent transpose(1, 2) then returns a view on which
    # aforementioned code snippet, as showm below, is valid
    # x = x.permute(2, 0, 1, 3).contiguous() and the viewed via
    # x = x.view(L * N, H * E)

    # Really the invariant you want to maintain is:
    # pre-dispatch op-output and its decomposed representation must
    # return tensor with same view and dims
    output = output.transpose(1, 2).contiguous(memory_format=torch.contiguous_format)
    return (output.transpose(1, 2), attn)


def register_inplace(aten_op, outplace_op):
    @register_decomposition(aten_op)
    def inplace_op(*args, **kwargs):
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


@register_decomposition([aten.baddbmm])
@out_wrapper()
@pw_cast_for_opmath
def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    result = torch.bmm(batch1, batch2)
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    if beta == 0:
        return result
    if not isinstance(beta, numbers.Number) or beta != 1:
        self = self * beta
    return self + result


@register_decomposition(aten.floor_divide)
@out_wrapper()
def floor_divide(self, other):
    return torch.div(self, other, rounding_mode="floor")


@register_decomposition(aten.sym_numel)
def sym_numel(t):
    return functools.reduce(operator.mul, t.shape, 1)


@register_decomposition([aten.sum.default, aten.sum.out])
def sum_default(
    self: Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    if out is None:
        return aten.sum.dim_IntList(self, [], dtype=dtype)
    else:
        return aten.sum.IntList_out(self, [], dtype=dtype, out=out)


@register_decomposition([aten.squeeze.default, aten.squeeze.dim])
def squeeze_default(self: Tensor, dim: Optional[int] = None):
    if dim is None:
        return aten.squeeze.dims(self, list(range(self.dim())))
    else:
        return aten.squeeze.dims(self, [dim])


@register_decomposition(torch.ops.aten._weight_norm_interface)
def _weight_norm_interface(v, g, dim=0):
    # https://github.com/pytorch/pytorch/blob/852f8526c52190125446adc9a6ecbcc28fb66182/aten/src/ATen/native/WeightNorm.cpp#L58
    keep_dim = tuple(i for i in range(len(v.shape)) if i != dim)
    # align with cuda behavior, keep norm in 'float' when g is 'bfloat16'
    norm_dtype = torch.float if g.dtype == torch.bfloat16 else None
    norm = v.norm(2, keep_dim, keepdim=True, dtype=norm_dtype)
    return v * (g / norm.to(g.dtype)), norm


@register_decomposition(aten.isin)
@out_wrapper()
def isin(elements, test_elements, *, assume_unique=False, invert=False):
    # handle when either elements or test_elements are Scalars (they can't both be)
    if not isinstance(elements, torch.Tensor):
        elements = torch.tensor(elements, device=test_elements.device)
    if not isinstance(test_elements, torch.Tensor):
        test_elements = torch.tensor(test_elements, device=elements.device)

    if test_elements.numel() < 10.0 * pow(elements.numel(), 0.145):
        return isin_default(elements, test_elements, invert=invert)
    else:
        return isin_sorting(
            elements, test_elements, assume_unique=assume_unique, invert=invert
        )


def isin_default(elements, test_elements, *, invert=False):
    if elements.numel() == 0:
        return torch.empty_like(elements, dtype=torch.bool)

    x = elements.view(*elements.shape, *((1,) * test_elements.ndim))
    if not invert:
        cmp = x == test_elements
    else:
        cmp = x != test_elements
    dim = tuple(range(-1, -test_elements.ndim - 1, -1))
    return cmp.any(dim=dim)


def isin_sorting(elements, test_elements, *, assume_unique=False, invert=False):
    elements_flat = elements.flatten()
    test_elements_flat = test_elements.flatten()
    if assume_unique:
        # This is the same as the aten implementation. For
        # assume_unique=False, we cannot use unique() here, so we use a
        # version with searchsorted instead.
        all_elements = torch.cat([elements_flat, test_elements_flat])
        sorted_elements, sorted_order = torch.sort(all_elements, stable=True)

        duplicate_mask = sorted_elements[1:] == sorted_elements[:-1]
        duplicate_mask = torch.constant_pad_nd(duplicate_mask, [0, 1], False)

        if invert:
            duplicate_mask = duplicate_mask.logical_not()

        mask = torch.empty_like(duplicate_mask)
        mask = mask.index_copy(0, sorted_order, duplicate_mask)

        return mask[0 : elements.numel()]
    else:
        sorted_test_elements, _ = torch.sort(test_elements_flat)
        idx = torch.searchsorted(sorted_test_elements, elements_flat)
        test_idx = torch.where(idx < sorted_test_elements.numel(), idx, 0)
        cmp = sorted_test_elements[test_idx] == elements_flat
        cmp = cmp.logical_not() if invert else cmp
        return cmp.reshape(elements.shape)


@register_decomposition(aten.take)
@out_wrapper()
def take(self, index):
    flattened = self.reshape(-1)
    return flattened[index]


@register_decomposition(aten.resize_as)
def resize_as(self, other, memory_format=None):
    if memory_format is None:
        memory_format = torch.contiguous_format
    if memory_format == torch.preserve_format:
        memory_format = suggest_memory_format(other)
    return aten.resize(self, other.shape, memory_format=memory_format)


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
