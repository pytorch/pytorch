import torch
from torch import Tensor
from typing import Optional, List, Tuple
from enum import Enum
from collections import defaultdict
from torch.utils._pytree import tree_map

aten = torch.ops.aten
aten.__origin__ = None

decomposition_table = {}


def register_decomposition(aten_op, registry=None):
    def decomposition_decorator(f):
        nonlocal registry
        if registry is None:
            registry = decomposition_table

        def add_op_to_table(aten_op):
            # Converts aten.foo to aten.foo.default
            # Done so I can be lazy and not write default on all of these ops
            if not isinstance(aten_op, torch._ops.OpOverload):
                op_overload = aten_op.default
            else:
                op_overload = aten_op
            registry[op_overload] = f
        # To handle allowing multiple aten_ops at once
        tree_map(add_op_to_table, aten_op)
        return f
    return decomposition_decorator


def get_decompositions(aten_ops: List[torch._ops.OpOverload]):
    packets_to_overloads = defaultdict(list)
    for op in decomposition_table:
        packets_to_overloads[op.overloadpacket].append(op)
    decompositions = {}
    for op in aten_ops:
        if op in packets_to_overloads:
            if len(packets_to_overloads[op]) == 1:
                op_overload = packets_to_overloads[op][0]
                decompositions[op_overload] = decomposition_table[op_overload]
            else:
                raise RuntimeError(f"Multiple decompositions for overloads found for {op}: {packets_to_overloads[op]}, please specify")
    return decompositions


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
def gelu_backward_decomposition(grad: Tensor, self: Tensor, approximate: str = 'none'):
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "none":
        kAlpha = M_SQRT1_2
        kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
        cdf = 0.5 * (1 + aten.erf(self * kAlpha))
        pdf = kBeta * aten.exp(self * self * -0.5)
        return grad * (cdf + self * pdf)
    else:
        kBeta = M_SQRT2 * M_2_SQRTPI * 0.5
        kKappa = 0.044715
        x_sq = self * self
        x_cube = x_sq * self
        inner = kBeta * (self + kKappa * x_cube)
        tanh_inner = aten.tanh(inner)

        left = 0.5 * self
        right = 1 + tanh_inner

        left_derivative = 0.5 * right

        tanh_derivative = 1 - tanh_inner * tanh_inner
        inner_derivative = kBeta * (1 + 3 * kKappa * x_sq)
        right_derivative = left * tanh_derivative * inner_derivative

        return grad * (left_derivative + right_derivative)


@register_decomposition(aten.mish_backward)
def mish_backward_decomposition(grad_output: Tensor, input: Tensor):
    input_tanh_softplus = aten.tanh(aten.softplus(input))
    input_sigmoid = aten.sigmoid(input)
    out = (input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus))
    return grad_output * (input_tanh_softplus + out)


@register_decomposition(aten.silu)
def silu(self: Tensor) -> Tensor:
    return self * aten.sigmoid(self)


@register_decomposition(aten.silu_backward)
def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    sigmoid = 1 / (1 + aten.exp(aten.neg(self)))
    return grad_output * sigmoid * (1 + self * (1 - sigmoid))


@register_decomposition(aten.softshrink_backward)
def softshrink_backward(grad_output: Tensor, self: Tensor, lambd: float) -> Tensor:
    return aten.where((self >= -lambd) & (self <= lambd), aten.new_zeros(grad_output, ()), grad_output)


@register_decomposition(aten.prelu_backward)
def prelu_backward(grad_output: Tensor, self: Tensor, weight: Tensor) -> Tuple[Tensor, Tensor]:
    # Logic is more complicated than I would like.  Basically, weight can either
    # be a scalar or a vector of size [C], and in the forward pass it's
    # broadcast against [N, C, ...]. So now, we need to do the corresponding
    # reduction, which is harder than we'd like...
    cur_weight = weight
    for _ in range(2, grad_output.dim()):
        cur_weight = cur_weight.unsqueeze(-1)
    input_grad = aten.where(self > 0, grad_output, cur_weight * grad_output)
    weight_grad_collector = aten.where(self > 0, aten.new_zeros(grad_output, ()), self * grad_output)
    out = aten.sum_to_size(weight_grad_collector, cur_weight.shape)
    while out.dim() > weight.dim():
        out = out.squeeze(-1)
    return (input_grad, out)


@register_decomposition(aten.rrelu_with_noise_backward)
def rrelu_with_noise_backward(grad_output: Tensor, self: Tensor, noise: Tensor, lower: float, upper: float, training: bool, self_is_result: bool) -> Tensor:
    if training and upper - lower > 1e-6:
        return grad_output.mul(noise)
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu_backward(grad_output, self, negative_slope, self_is_result)


@register_decomposition(aten.log_sigmoid_backward)
def log_sigmoid_backward(grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor:
    in_negative = self < 0
    max_deriv = aten.where(in_negative, 1, 0)
    sign = aten.where(in_negative, 1, -1)
    z = aten.exp(-aten.abs(self))
    return grad_output * (max_deriv - sign * (z / (1 + z)))
    # CPU has a special formula that uses buffer, but disabled for convenience sake
    # return (max_deriv - sign * (buffer / (1 + buffer))) * grad_output


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


@register_decomposition(aten.binary_cross_entropy_backward)
def binary_cross_entropy_backward(grad_output: Tensor, self: Tensor, target: Tensor, weight: Optional[Tensor] = None, reduction: int = Reduction.MEAN.value) -> Tensor:
    if weight is None:
        weight = aten.new_ones(self, ())
    result = weight * (self - target) / self / (1 - self)
    if reduction == Reduction.MEAN.value:
        result = result * (1.0 / self.numel())
    return result * grad_output


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


@register_decomposition(aten.col2im_backward)
def col2im_backward(
    grad_output: Tensor, kernel_size: List[int],
    dilation: List[int], padding: List[int], stride: List[int]
) -> Tensor:
    return aten.im2col(grad_output, kernel_size, dilation, padding, stride)


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
def addcdiv(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1):
    return self + value * (tensor1 / tensor2)


# Remove special case when https://github.com/pytorch/pytorch/pull/72949 is landed.
@register_decomposition(aten.addcmul)
def addcmul(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1):
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


def prod(x: List[int]):
    r = 1
    for i in x:
        r *= i
    return r


@register_decomposition(aten.native_layer_norm)
def native_layer_norm(input: Tensor, normalized_shape: List[int], weight: Optional[Tensor], bias: Optional[Tensor], eps: float) -> Tuple[Tensor, Tensor, Tensor]:
    input_shape = input.shape
    input_ndim = input.dim()

    axis = input_ndim - len(normalized_shape)
    M = prod(input_shape[:axis])

    # Hmm... not sure how I get around this...
    # Basically, native_batch_norm doesn't support 0-entry tensors, while
    # native_layer_norm does (and is tested by OpInfos!)
    if M > 0:
        input_reshaped = input.view(1, M, -1)
    else:
        return (input, aten.new_empty(input, (0,)), aten.new_empty(input, (0,)))

    # Unlike Batch Normalization, which applies scalar scale and bias for each
    # entire channel/plane with the affine option, Layer Normalization applies
    # per-element scale and bias. E.g. For input {N, C, H, W}, weight for
    # batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
    out, mean, rstd = aten.native_batch_norm(
        input_reshaped, weight=None, bias=None, running_mean=None,
        running_var=None, training=True, momentum=0.0, eps=eps)
    out = out.view(input_shape)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias

    stat_shape = list(input_shape[:axis])
    for _ in range(axis, input.dim()):
        stat_shape.append(1)
    mean = mean.view(stat_shape)
    rstd = rstd.view(stat_shape)
    return (out, mean, rstd)


@register_decomposition(aten.split_with_sizes)
def split_with_sizes(self: Tensor, split_sizes: List[int], dim: int = 0) -> List[Tensor]:
    num_splits = len(split_sizes)
    splits = []
    start_idx = 0
    for i in range(num_splits):
        length = split_sizes[i]
        splits.append(self.narrow(dim, start_idx, length))
        start_idx += length
    return splits


@register_decomposition(aten.split)
def split(self: Tensor, split_size: int, dim: int = 0) -> List[Tensor]:
    input_sizes = self.shape
    dim_size = input_sizes[dim]
    if split_size == 0:
        assert(dim_size == 0)
        return [self]
    chunks = (dim_size + split_size - 1) // split_size
    split_sizes = [split_size for i in range(chunks)]
    split_sizes[chunks - 1] = split_size - (split_size * chunks - dim_size)
    return aten.split_with_sizes(self, split_sizes, dim)


@register_decomposition(aten.addmm)
def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: int = 1, alpha: int = 1):
    if not self.is_floating_point():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * aten.mm(mat1, mat2)
    if beta == 0:
        return out
    return beta * self + out


@register_decomposition(aten.clamp_min)
def clamp_min(self: Tensor, min: float):
    return aten.clamp(self, min=min)


@register_decomposition(aten.clamp_max)
def clamp_max(self: Tensor, max: float):
    return aten.clamp(self, max=max)


@register_decomposition(aten._fused_dropout)
def _fused_dropout_decomposition(input, p, generator=None):
    mask = aten.to(aten.rand_like(input) < p, dtype=torch.uint8)
    res = mask.type_as(input) * input * (1./p)
    return [res, mask]


@register_decomposition(aten.logical_and)
def logical_and(self: Tensor, other: Tensor) -> Tensor:
    return self.to(dtype=torch.bool) & other.to(dtype=torch.bool)


@register_decomposition(aten.logical_or)
def logical_or(self: Tensor, other: Tensor) -> Tensor:
    return self.to(dtype=torch.bool) | other.to(dtype=torch.bool)


@register_decomposition(aten.logical_xor)
def logical_xor(self: Tensor, other: Tensor) -> Tensor:
    return self.to(dtype=torch.bool) ^ other.to(dtype=torch.bool)


@register_decomposition(aten.logical_not)
def logical_not(self: Tensor) -> Tensor:
    return ~self.to(dtype=torch.bool)


# Commented out due to requiring type conversions for correct behavior on OpInfo tests
# @register_decomposition(aten.xlogy)
# def xlogy(self: Tensor, other: Tensor) -> Tensor:
#     return aten.where(aten.isnan(self),
#                       self,
#                       aten.where(self == aten.new_zeros(self, ()),
#                                  aten.new_zeros(self, ()),
#                                  self * aten.log(other)))


@register_decomposition(aten.var)
def var_decomposition(x: Tensor, dims: List[int], correction: int = 0, keepdim: bool = False):
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
def std_decomposition(x: Tensor, dims: List[int], correction: int = 0, keepdim: bool = False):
    return aten.sqrt(aten.var(x, dims, correction=correction, keepdim=keepdim))


# Questionable decompositions
# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition(aten.detach)
def detach_decomposition(x):
    return x


# @register_decomposition(aten.cudnn_batch_norm)
# def cudnn_batch_norm(input: Tensor, weight: Tensor, bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, exponential_average_factor: float, epsilon: float):
#     a, b, c = aten.native_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
#     return (a,b, c, aten.new_empty(input, (1,)))

# @register_decomposition(aten.cudnn_batch_norm_backward)
# def cudnn_batch_norm_backward(input: Tensor, grad_output: Tensor, weight: Tensor, running_mean: Optional[Tensor], running_var: Optional[Tensor], save_mean: Optional[Tensor], save_var: Optional[Tensor], epsilon: float, reserveSpace: Tensor):
#     return aten.native_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_var, True, epsilon, [True, True, True])
