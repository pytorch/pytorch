import torch
from torch import Tensor
from typing import Any, Dict, NamedTuple, Optional, Set, Tuple, List, Callable, Union
from enum import Enum

aten = torch.ops.aten

decomposition_table = {}

def register_decomposition(aten_op):
    def decomposition_decorator(f):
        decomposition_table[aten_op] = f
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
def softplus_backward_decomposition(out_grad: Tensor, x: Tensor, beta: float, threshold: float, out):
    z = (x * beta).exp()
    return aten.where((x * beta) > threshold, out_grad, out_grad * z / (z + 1.0))

@register_decomposition(aten.elu_backward)
def elu_backward_decomposition(grad_output: Tensor, alpha: float, scale: float, input_scale: float, is_result: bool, self_or_result: Tensor):
    negcoef = alpha * scale
    poscoef = scale
    negiptcoef = input_scale
    if is_result:
        return aten.where(self_or_result <= 0, grad_output * negiptcoef * (self_or_result + negcoef), self_or_result * poscoef)
    else:
        return aten.where(self_or_result <= 0, grad_output * negiptcoef * negcoef * aten.exp(self_or_result * negiptcoef), grad_output * poscoef)

@register_decomposition(aten.hardsigmoid_backward)
def hardsigmoid_backward_decomposition(grad_output: Tensor, self: Tensor):
    return aten.where((self > -3.0) & (self < 3.0), grad_output * (1.0/6.0), aten.new_zeros(grad_output, ()))

@register_decomposition(aten.hardtanh_backward)
def hardtanh_backward_decomposition(grad_output: Tensor, self: Tensor, min_val: float, max_val: float):
    return aten.where((self <= min_val) | (self >= max_val), aten.new_zeros(grad_output, ()), grad_output)

@register_decomposition(aten.hardshrink_backward)
def hardshrink_backward(grad_out: Tensor, self: Tensor, lambd: float):
    return aten.where((self >= -lambd) & (self <= lambd), aten.new_zeros(grad_out, ()), grad_out)

@register_decomposition(aten.threshold_backward)
def threshold_backward_decomposition(grad_output: Tensor, self: Tensor, threshold: float):
    return aten.where(self <= threshold, aten.new_zeros(grad_output, ()), grad_output)

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
  return grad_output * (input_tanh_softplus + (input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)))

# whyyyy does log_sigmoid do 2 different things for CPU and CUDA >:(
@register_decomposition(aten.log_sigmoid_backward)
def log_sigmoid_backward(grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor:
    in_negative = self < 0
    max_deriv = aten.where(in_negative, 1, 0)
    sign = aten.where(in_negative, 1, -1)
    if grad_output.is_cuda: # buffer is not used on CUDA
        z = aten.exp(-aten.abs(self))
        return grad_output * (max_deriv - sign * (z / (1 + z)))
    else:
        return (max_deriv - sign * (buffer / (1 + buffer))) * grad_output

@register_decomposition(aten.mse_loss_backward)
def mse_loss_backward(grad_output: Tensor, input: Tensor, target: Tensor, reduction: int):
    norm = 2./input.numel() if reduction == Reduction.MEAN.value else 2.
    return norm * (input - target) * grad_output

@register_decomposition(aten.huber_loss_backward)
def huber_loss_backward(grad_output: Tensor, self: Tensor, target: Tensor, reduction: int, delta: float):
    norm = 1./self.numel() if reduction == Reduction.MEAN.value else 1.
    x = self - target
    return aten.where(x < -delta, -norm * grad_output * delta, aten.where(x > delta, norm * grad_output * delta, norm * x * grad_output))

@register_decomposition(aten.slice_backward)
def slice_backward(grad_output: Tensor, input_sizes: List[int], dim: int, start: int, end: int, step:int):
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

@register_decomposition(aten._softmax_backward_data)
def _softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int):
    new_grad = grad_output * output
    return (new_grad - output * aten.sum(new_grad, dim=dim, keepdim=True))

@register_decomposition(aten._log_softmax_backward_data)
def _log_softmax_backward_data(grad_output: Tensor, output: Tensor, dim: int, input_dtype: int):
    grad_input = grad_output - aten.exp(output) * aten.sum(grad_output, dim=dim, keepdim=True)
    return grad_input

@register_decomposition(aten.im2col_backward)
def im2col_backward(grad_output: Tensor, input_size: List[int], kernel_size: List[int], dilation: List[int], padding: List[int], stride: List[int]) -> Tensor:
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

# @register_decomposition(aten._fused_dropout)
# def _fused_dropout_decomposition(input, p, generator=None):
#     mask = aten.to(aten.rand_like(input) < p, dtype=torch.uint8)
#     res = mask.type_as(input) * input * (1./p)
#     return [res, mask]


@register_decomposition(aten._s_where)
def _s_where_canonicalization(a, b, c):
    return aten.where(a, b, c)