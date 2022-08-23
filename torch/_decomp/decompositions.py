import functools
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
import torch._prims_common as utils
import torch.nn.functional as F
from torch import Tensor
from torch._decomp import register_decomposition
from torch._prims_common.wrappers import out_wrapper
from torch.utils._pytree import tree_flatten, tree_map

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
def type_casts(f: Callable, type_promotion: utils.ELEMENTWISE_TYPE_PROMOTION_KIND):
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
        return tree_map(decrease_prec, r)

    return inner


pw_cast_for_opmath = functools.partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
)
reduction_complex_to_real = functools.partial(
    type_casts, type_promotion=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
)
pw_cast_for_int_to_real = functools.partial(
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


@register_decomposition(aten.elu)
@pw_cast_for_opmath
def elu(
    self: Tensor, alpha: float = 1, scale: float = 1, input_scale: float = 1
) -> Tensor:
    negcoef = alpha * scale
    poscoef = scale
    negiptcoef = input_scale
    return torch.where(
        self > 0, self * poscoef, (torch.exp(self * negiptcoef) - 1) * negcoef
    )


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
        grad_output.new_zeros(()),
    )


@register_decomposition(aten.hardtanh_backward)
@pw_cast_for_opmath
def hardtanh_backward(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
):
    return torch.where(
        (self <= min_val) | (self >= max_val), grad_output.new_zeros(()), grad_output
    )


@register_decomposition(aten.hardshrink_backward)
@pw_cast_for_opmath
def hardshrink_backward(grad_out: Tensor, self: Tensor, lambd: float):
    return torch.where(
        (self >= -lambd) & (self <= lambd), grad_out.new_zeros(()), grad_out
    )


@register_decomposition(aten.hardswish)
@pw_cast_for_opmath
def hardswish(self: Tensor) -> Tensor:
    return self * torch.clamp(torch.clamp(self + 3, min=0), max=6) / 6


@register_decomposition(aten.hardswish_backward)
@pw_cast_for_opmath
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    return torch.where(
        self < -3,
        grad_output.new_zeros(()),
        torch.where(self <= 3, grad_output * ((self / 3) + 0.5), grad_output),
    )


@register_decomposition(aten.threshold_backward)
@pw_cast_for_opmath
def threshold_backward(grad_output: Tensor, self: Tensor, threshold: float):
    return torch.where(self <= threshold, grad_output.new_zeros(()), grad_output)


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
    return torch.where(
        (self >= -lambd) & (self <= lambd), grad_output.new_zeros(()), grad_output
    )


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
    weight_grad_collector = torch.where(
        self > 0, grad_output.new_zeros(()), self * grad_output
    )
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


@register_decomposition(aten.huber_loss)
@pw_cast_for_opmath
def huber_loss(
    self: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
    delta: float = 1.0,
) -> Tensor:
    assert delta > 0, "huber_loss does not support non-positive values for delta."
    z = (self - target).abs()
    loss = torch.where(z < delta, 0.5 * z * z, delta * (z - 0.5 * delta))
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.huber_loss_backward)
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
    loss = torch.log(1.0 + torch.exp(-input * target))
    return apply_loss_reduction(loss, reduction)


@register_decomposition(aten.soft_margin_loss_backward)
@pw_cast_for_opmath
def soft_margin_loss_backward(
    grad_output: Tensor,
    self: Tensor,
    target: Tensor,
    reduction: int = Reduction.MEAN.value,
) -> Tensor:
    norm = 1.0 / self.numel() if reduction == Reduction.MEAN.value else 1.0
    z = torch.exp(-target * self)
    grad_input = -norm * target * z / (1.0 + z) * grad_output
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


@register_decomposition(aten._softmax_backward_data)
@pw_cast_for_opmath
def _softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: int
):
    new_grad = grad_output * output
    return new_grad - output * torch.sum(new_grad, dim=dim, keepdim=True)


@register_decomposition(aten._log_softmax_backward_data)
@pw_cast_for_opmath
def _log_softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: int
):
    grad_input = grad_output - torch.exp(output) * torch.sum(
        grad_output, dim=dim, keepdim=True
    )
    return grad_input


# TODO: the type annotations on arguments are not quite right


@register_decomposition(aten.im2col_backward)
def im2col_backward(
    grad_output: Tensor,
    input_size: List[int],
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> Tensor:
    return F.fold(grad_output, input_size, kernel_size, dilation, padding, stride)  # type: ignore[arg-type]


@register_decomposition(aten.col2im_backward)
def col2im_backward(
    grad_output: Tensor,
    kernel_size: List[int],
    dilation: List[int],
    padding: List[int],
    stride: List[int],
) -> Tensor:
    return F.unfold(grad_output, kernel_size, dilation, padding, stride)  # type: ignore[arg-type]


@register_decomposition(aten.native_dropout_backward)
@pw_cast_for_opmath
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    return grad_output * (mask.type_as(grad_output) * scale)


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
            self.new_zeros(()),
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


# TODO: Correct the type promotion semantics
@register_decomposition(aten._softmax)
@pw_cast_for_opmath
def _softmax(x: Tensor, dim: int, half_to_float: bool):
    x_max = torch.amax(x, dim, keepdim=True)
    unnormalized = torch.exp(x - x_max)
    return unnormalized / torch.sum(unnormalized, dim, keepdim=True)


# TODO: Correct the type promotion semantics
@register_decomposition(aten._log_softmax)
@pw_cast_for_opmath
def _log_softmax(x: Tensor, dim: int, half_to_float: bool):
    x_max = torch.amax(x, dim, keepdim=True)
    shifted = x - x_max
    shifted_logsumexp = torch.log(torch.sum(torch.exp(shifted), dim, keepdim=True))
    return shifted - shifted_logsumexp


# Remove special case when https://github.com/pytorch/pytorch/pull/72949 is landed.
@register_decomposition(aten.addcmul)
@pw_cast_for_opmath
def addcmul(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1):
    if self.is_floating_point() or self.is_complex():
        return self + value * tensor1 * tensor2
    else:
        return self + int(value) * tensor1 * tensor2


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


@register_decomposition(aten.split_with_sizes, disable_meta=True)
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


@register_decomposition(aten.split.Tensor, disable_meta=True)
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
@pw_cast_for_opmath
def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: int = 1, alpha: int = 1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mm(mat1, mat2)
    if beta == 0:
        return out
    return beta * self + out


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


@register_decomposition(aten.native_group_norm.default, disable_meta=True)
def native_group_norm(
    input: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    orig_shape = input.shape
    input = input.view(N, group, C // group, HxW)
    reduction_dims = [2, 3]
    out, mean, rstd = normalize(input, reduction_dims, eps)
    mean = _squeeze_multiple(mean, reduction_dims)
    rstd = _squeeze_multiple(rstd, reduction_dims)
    out = out.view(orig_shape)
    if weight is not None:
        weight = _unsqueeze_to_dim(weight, out.dim() - 1)
        out = out * weight
    if bias is not None:
        bias = _unsqueeze_to_dim(bias, out.dim() - 1)
        out = out + bias

    out = out.to(dtype=input.dtype)
    mean = mean.to(dtype=input.dtype)
    rstd = rstd.to(dtype=input.dtype)
    return (out, mean, rstd)


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
        x.to(computation_dtype) if x is not None else x
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
            input.new_zeros(input_shape),
            input.new_zeros(input_shape[axis:]),
            input.new_zeros(input_shape[axis:]),
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
            d_bias = grad_out_cast

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
        running_mean = running_mean.to(dtype=computation_dtype)
        running_var = running_var.to(dtype=computation_dtype)
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


@register_decomposition(aten.xlogy.Tensor)
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


@register_decomposition(aten.var.correction)
@reduction_complex_to_real
def var_correction(
    x: Tensor,
    dims: Optional[List[int]],
    correction: Optional[int] = None,
    keepdim: bool = False,
):
    if dims is None:
        dims = []

    if x.is_complex():
        # For complex, calculate variance of real and imaginary components
        # separately then add to get overall variance.
        real_in = x.real
        var_real = torch.var(real_in, dims, correction=correction, keepdim=keepdim)
        imag_in = x.imag
        var_imag = torch.var(imag_in, dims, correction=correction, keepdim=keepdim)
        return var_real + var_imag

    if correction is None:
        correction = 0

    if len(dims) == 0:
        n = prod(x.shape)  # type: ignore[arg-type]
    else:
        n = 1
        for dim in dims:
            n *= x.shape[dim]

    mean = torch.mean(x, dims, True)
    sub = x - mean
    sq = sub * sub
    sum = torch.sum(sq, dims, keepdim)

    if correction:
        n = n - correction

    return sum / n


@register_decomposition(aten.std.correction)
@reduction_complex_to_real
def std_decomposition(
    x: Tensor, dims: List[int], correction: int = 0, keepdim: bool = False
):
    return torch.sqrt(torch.var(x, dims, correction=correction, keepdim=keepdim))


# Questionable decompositions
# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition(
    [aten.detach, aten.lift, aten.lift_fresh, aten.alias], disable_meta=True
)
def nop_decomposition(x):
    return x


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
        input.new_zeros((0,)),
        input.new_zeros((0,)),
        input.new_zeros((0,), dtype=torch.uint8),
    )


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

    mean = torch.reshape(mean, broadcast_mask)  # type: ignore[arg-type]
    norm = 1.0 / num_features
    grad_output_sum = torch.sum(grad_out_cast, reduction_axes)  # type: ignore[arg-type]
    dot_p = torch.sum(grad_out_cast * (input_cast - mean), reduction_axes)

    grad_mean = torch.reshape(grad_output_sum * norm, broadcast_mask)
    proj_scale = torch.reshape(torch.mul(dot_p * norm, invstd * invstd), broadcast_mask)  # type: ignore[operator]

    if weight_cast is None:
        grad_scale = torch.reshape(invstd, broadcast_mask) * 1.0  # type: ignore[arg-type]
    else:
        grad_scale = torch.reshape(invstd * weight_cast, broadcast_mask)

    if train:
        proj = (input_cast - mean) * proj_scale
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


@register_decomposition(aten.transpose.int, disable_meta=True)
def transpose_int(self: Tensor, dim0: int, dim1: int) -> Tensor:
    dim0, dim1 = utils.canonicalize_dims(self.dim(), (dim0, dim1))  # type: ignore[misc]

    # NB: these no-op views force this operator to return a
    # fresh TensorImpl, which is important for autograd to
    # work correctly (assert will fail if you don't do it)
    if self.dim() <= 1:
        return self.view(self.shape)

    if dim0 == dim1:
        return self.view(self.shape)
    perm = list(range(self.dim()))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return torch.permute(self, perm)


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
    return result


# We should be applying decompositions after all transformations
@register_decomposition(aten.is_same_size.default)
def is_same_size(a: Tensor, b: Tensor) -> bool:
    return a.shape == b.shape


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
