import torch
from torch import Tensor
from typing import Optional, List
from enum import Enum

aten = torch.ops.aten

decomposition_table = {}


def register_decomposition(aten_op, registry=None):
    def decomposition_decorator(f):
        nonlocal registry
        if registry is None:
            registry = decomposition_table
        registry[aten_op] = f
        return f
    return decomposition_decorator


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


@register_decomposition(aten.tanh_backward)
def tanh_backward_decomposition(out_grad: Tensor, y: Tensor):
    return out_grad * (1 - y * y)


@register_decomposition(aten.sigmoid_backward)
def sigmoid_backward_decomposition(out_grad: Tensor, y: Tensor):
    return out_grad * (y * (1 - y))


@register_decomposition(aten.softplus_backward)
# The out argument seems to always be ignored?
def softplus_backward_decomposition(out_grad: Tensor, x: Tensor, beta: float, threshold: float):
    z = (x * beta).exp()
    return aten.where((x * beta) > threshold, out_grad, out_grad * z / (z + 1.0))


@register_decomposition(aten.elu_backward)
def elu_backward_decomposition(
    grad_output: Tensor, alpha: float, scale: float, input_scale: float, is_result: bool, self_or_result: Tensor
):
    negcoef = alpha * scale
    poscoef = scale
    negiptcoef = input_scale
    if is_result:
        return aten.where(
            self_or_result <= 0,
            grad_output * negiptcoef * (self_or_result + negcoef),
            self_or_result * poscoef
        )
    else:
        return aten.where(
            self_or_result <= 0,
            grad_output * negiptcoef * negcoef * aten.exp(self_or_result * negiptcoef),
            grad_output * poscoef
        )


@register_decomposition(aten.hardsigmoid_backward)
def hardsigmoid_backward_decomposition(grad_output: Tensor, self: Tensor):
    return aten.where((self > -3.0) & (self < 3.0), grad_output * (1.0 / 6.0), aten.new_zeros(grad_output, ()))


@register_decomposition(aten.hardtanh_backward)
def hardtanh_backward_decomposition(grad_output: Tensor, self: Tensor, min_val: float, max_val: float):
    return aten.where((self <= min_val) | (self >= max_val), aten.new_zeros(grad_output, ()), grad_output)


@register_decomposition(aten.hardshrink_backward)
def hardshrink_backward(grad_out: Tensor, self: Tensor, lambd: float):
    return aten.where((self >= -lambd) & (self <= lambd), aten.new_zeros(grad_out, ()), grad_out)


@register_decomposition(aten.hardswish_backward)
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    return aten.where(self < -3, aten.new_zeros(grad_output, ()), aten.where(self <= 3, grad_output * ((self / 3) + 0.5), grad_output))


@register_decomposition(aten.threshold_backward)
def threshold_backward_decomposition(grad_output: Tensor, self: Tensor, threshold: float):
    return aten.where(self <= threshold, aten.new_zeros(grad_output, (1,)), grad_output)


@register_decomposition(aten.leaky_relu_backward)
def leaky_relu_backward_decomposition(grad_output: Tensor, self: Tensor, negative_slope: float, self_is_result: bool):
    return aten.where(self > 0, grad_output, grad_output * negative_slope)


@register_decomposition(aten.gelu_backward)
def gelu_backward_decomposition(grad: Tensor, self: Tensor):
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    kAlpha = M_SQRT1_2
    kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
    cdf = 0.5 * (1 + aten.erf(self * kAlpha))
    pdf = kBeta * aten.exp(self * self * -0.5)
    return grad * (cdf + self * pdf)


@register_decomposition(aten.mish_backward)
def mish_backward_decomposition(grad_output: Tensor, input: Tensor):
    input_tanh_softplus = aten.tanh(aten.softplus(input))
    input_sigmoid = aten.sigmoid(input)
    out = (input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus))
    return grad_output * (input_tanh_softplus + out)


@register_decomposition(aten.silu_backward)
def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    sigmoid = 1 / (1 + aten.exp(aten.neg(self)))
    return grad_output * sigmoid * (1 + self * (1 - sigmoid))

# whyyyy does log_sigmoid do 2 different things for CPU and CUDA >:(


@register_decomposition(aten.log_sigmoid_backward)
def log_sigmoid_backward(grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor:
    in_negative = self < 0
    max_deriv = aten.where(in_negative, 1, 0)
    sign = aten.where(in_negative, 1, -1)
    if grad_output.is_cuda:  # buffer is not used on CUDA
        z = aten.exp(-aten.abs(self))
        return grad_output * (max_deriv - sign * (z / (1 + z)))
    else:
        return (max_deriv - sign * (buffer / (1 + buffer))) * grad_output


@register_decomposition(aten.mse_loss_backward)
def mse_loss_backward(grad_output: Tensor, input: Tensor, target: Tensor, reduction: int):
    norm = 2. / input.numel() if reduction == Reduction.MEAN.value else 2.
    return norm * (input - target) * grad_output


@register_decomposition(aten.huber_loss_backward)
def huber_loss_backward(grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, delta: float):
    norm = 1. / self.numel() if reduction == Reduction.MEAN.value else 1.
    x = self - target
    return aten.where(
        x < -delta,
        -norm * grad_output * delta,
        aten.where(x > delta, norm * grad_output * delta, norm * x * grad_output)
    )


@register_decomposition(aten.slice_backward)
def slice_backward(grad_output: Tensor, input_sizes: List[int], dim: int, start: int, end: int, step: int):
    grad_input = aten.new_zeros(grad_output, input_sizes)
    return aten.slice_scatter(grad_input, grad_output, dim, start, end, step)


@register_decomposition(aten.select_backward)
def select_backward(grad_output: Tensor, input_sizes: List[int], dim: int, index: int):
    grad_input = aten.new_zeros(grad_output, input_sizes)
    return aten.select_scatter(grad_input, grad_output, dim, index)


@register_decomposition(aten.diagonal_backward)
def diagonal_backward(grad_output: Tensor, input_sizes: List[int], offset: int, dim1: int, dim2: int):
    grad_input = aten.new_zeros(grad_output, input_sizes)
    return aten.diagonal_scatter(grad_input, grad_output, offset, dim1, dim2)


# @register_decomposition(aten.cudnn_batch_norm)
# def cudnn_batch_norm(input: Tensor, weight: Tensor, bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, exponential_average_factor: float, epsilon: float):
#     a, b, c = aten.native_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
#     return (a,b, c, aten.new_empty(input, (1,)))

# @register_decomposition(aten.cudnn_batch_norm_backward)
# def cudnn_batch_norm_backward(input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor], save_mean: Optional[Tensor], save_var: Optional[Tensor], epsilon: float, reserveSpace: Tensor):
#     return aten.native_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_var, True, epsilon, [True, True, True])


@register_decomposition(aten._softmax_backward_data)
def _softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int):
    new_grad = grad_output * output
    return (new_grad - output * aten.sum(new_grad, dim=dim, keepdim=True))


@register_decomposition(aten._log_softmax_backward_data)
def _log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int):
    grad_input = grad_output - aten.exp(output) * aten.sum(grad_output, dim=dim, keepdim=True)
    return grad_input


@register_decomposition(aten.im2col_backward)
def im2col_backward(
    grad_output: Tensor, input_size: List[int], kernel_size: List[int],
    dilation: List[int], padding: List[int], stride: List[int]
) -> Tensor:
    return aten.col2im(grad_output, input_size, kernel_size, dilation, padding, stride)


@register_decomposition(aten.native_dropout_backward)
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    return grad_output * (mask.type_as(grad_output) * scale)


@register_decomposition(aten.logit_backward)
def logit_backward(grad_output: Tensor, self: Tensor, eps: Optional[float] = None) -> Tensor:
    if eps is not None:
        lo = eps
        hi = 1.0 - lo
        return aten.where(
            aten.logical_and(self >= lo, self <= hi),
            grad_output / (self * (1.0 - self)),
            aten.new_zeros(self, ()))
    else:
        return aten.where(
            aten.logical_and(self >= 0.0, self <= 1.0),
            grad_output / (self * (1.0 - self)),
            aten.new_full(self, (), float('nan')))


@register_decomposition(aten.native_dropout)
def native_dropout_decomposition(input, p, generator=None):
    bool_mask = aten.rand_like(input) < p
    res = bool_mask * input * float(1.0 / p)
    return [res, bool_mask]


@register_decomposition(aten._softmax)
def _softmax(x: Tensor, dim: int, half_to_float: bool):
    x_max = aten.max(x, dim, keepdim=True)[0]
    unnormalized = aten.exp(x - x_max)
    return unnormalized / aten.sum(unnormalized, dim, keepdim=True)


@register_decomposition(aten._log_softmax)
def _log_softmax(x: Tensor, dim: int, half_to_float: bool):
    x_max = aten.max(x, dim, keepdim=True)[0]
    shifted = x - x_max
    shifted_logsumexp = aten.log(aten.sum(aten.exp(shifted), dim, keepdim=True))
    return shifted - shifted_logsumexp


@register_decomposition(aten.addcdiv)
def addcdiv(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: float=1):
    return self + value * (tensor1 / tensor2)

@register_decomposition(aten.addcmul)
def addcmul(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: float=1):
    if self.is_floating_point():
        return self + value * tensor1 * tensor2
    else:
        return self + int(value) * tensor1 * tensor2

@register_decomposition(aten.embedding_dense_backward)
def embedding_dense_backward(grad_output: Tensor, indices: Tensor, num_weights: int, padding_idx: int, scale_grad_by_freq: bool):
    numel = indices.numel()
    grad = grad_output.view(numel, grad_output.size(-1))
    grad_weight = aten.new_zeros(grad_output, (num_weights, grad_output.shape[-1]))
    indices_rank1 = indices.view(numel)
    if scale_grad_by_freq:
        counts = aten.new_zeros(indices, (num_weights,))
        ones = aten.new_ones(indices, (numel,))
        counts = aten.index_put(counts, [indices_rank1], ones, accumulate=True)
        grad_weights_scale = aten.index(counts, [indices_rank1])
        grad = grad / grad_weights_scale.unsqueeze(1)
    skip_padding = (indices_rank1 != padding_idx).unsqueeze(1)
    skip_padding = skip_padding.expand_as(grad)
    zero_grad = aten.full_like(grad, 0)
    return aten.index_put(grad_weight, [indices_rank1], aten.where(skip_padding, grad, zero_grad), accumulate=True)

# @register_decomposition(aten.addmm)
# def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta=1, alpha=1):
#     if not self.is_floating_point():
#         beta = int(beta)
#         alpha = int(alpha)
#     out = alpha * aten.mm(mat1, mat2)
#     if beta == 0:
#         return out
#     return beta * self + out


@register_decomposition(aten.clamp_min)
def clamp_min(self: Tensor, min: float):
    return aten.clamp(self, min=min)


@register_decomposition(aten.clamp_max)
def clamp_max(self: Tensor, min: float):
    return aten.clamp(self, max=max)

@register_decomposition(aten._fused_dropout)
def _fused_dropout_decomposition(input, p, generator=None):
    mask = aten.to(aten.rand_like(input) < p, dtype=torch.uint8)
    res = mask.type_as(input) * input * (1./p)
    return [res, mask]


# Questionable decompositions
@register_decomposition(aten._s_where)
def _s_where_canonicalization(a, b, c):
    return aten.where(a, b, c)


# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition(aten.detach)
def detach_decomposition(x):
    return x


@register_decomposition(aten.var)
def var_decomposition(x, dims, correction=0, keepdim=False):
    if dims is None:
        dims = []

    if isinstance(dims, (tuple, list)) and len(dims) == 0:
        n = x.numel()
    else:
        n = 1
        for dim in dims:
            n *= x.shape[dim]

    mean = aten.mean(x, dims, True)
    sub = x - mean
    sq = sub * sub
    sum = aten.sum(sq, dims, keepdim)

    if correction:
        n = n - correction

    return sum / n


@register_decomposition(aten.std)
def std_decomposition(x, dims, correction=0, keepdim=False):
    return aten.sqrt(aten.var(x, dims, correction=correction, keepdim=keepdim))
