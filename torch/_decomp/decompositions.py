import torch
from torch import Tensor
from torch._decomp import register_decomposition
from enum import Enum
from typing import Tuple, Optional, List
import torch.nn.functional as F
import functools
from torch.utils._pytree import tree_map

# None of these functions are publicly accessible; get at them
# from torch._decomps
__all__: List[str] = []

aten = torch.ops.aten


class Reduction(Enum):
    NONE = 0
    MEAN = 1
    SUM = 2


def cast_for_opmath(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        orig_prec = None

        # TODO: pretty sure this is not quite right
        def increase_prec(x):
            if isinstance(x, Tensor) and x.dtype in (torch.float16, torch.bfloat16):
                nonlocal orig_prec
                if orig_prec is None:
                    orig_prec = x.dtype
                else:
                    assert orig_prec == x.dtype
                return x.to(torch.float32)
            else:
                return x

        def decrease_prec(x):
            if isinstance(x, Tensor) and x.dtype is torch.float32:
                assert orig_prec is not None
                return x.to(orig_prec)
            else:
                return x

        r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
        if orig_prec:
            return tree_map(decrease_prec, r)
        else:
            return r

    return inner


@register_decomposition(aten.tanh_backward)
@cast_for_opmath
def tanh_backward_decomposition(out_grad: Tensor, y: Tensor):
    return out_grad * (1 - y * y)


@register_decomposition(aten.sigmoid_backward)
@cast_for_opmath
def sigmoid_backward_decomposition(out_grad: Tensor, y: Tensor):
    return out_grad * (y * (1 - y))


@register_decomposition(aten.softplus_backward)
def softplus_backward_decomposition(
    out_grad: Tensor, x: Tensor, beta: float, threshold: float
):
    z = (x * beta).exp()
    return torch.where((x * beta) > threshold, out_grad, out_grad * z / (z + 1.0))


@register_decomposition(aten.elu_backward)
def elu_backward_decomposition(
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


@register_decomposition(aten.hardsigmoid_backward)
@cast_for_opmath
def hardsigmoid_backward_decomposition(grad_output: Tensor, self: Tensor):
    return torch.where(
        (self > -3.0) & (self < 3.0),
        grad_output * (1.0 / 6.0),
        grad_output.new_zeros(()),
    )


@register_decomposition(aten.hardtanh_backward)
def hardtanh_backward_decomposition(
    grad_output: Tensor, self: Tensor, min_val: float, max_val: float
):
    return torch.where(
        (self <= min_val) | (self >= max_val), grad_output.new_zeros(()), grad_output
    )


@register_decomposition(aten.hardshrink_backward)
def hardshrink_backward(grad_out: Tensor, self: Tensor, lambd: float):
    return torch.where(
        (self >= -lambd) & (self <= lambd), grad_out.new_zeros(()), grad_out
    )


@register_decomposition(aten.hardswish_backward)
@cast_for_opmath
def hardswish_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    return torch.where(
        self < -3,
        grad_output.new_zeros(()),
        torch.where(self <= 3, grad_output * ((self / 3) + 0.5), grad_output),
    )


@register_decomposition(aten.threshold_backward)
def threshold_backward_decomposition(
    grad_output: Tensor, self: Tensor, threshold: float
):
    return torch.where(self <= threshold, grad_output.new_zeros(()), grad_output)


@register_decomposition(aten.leaky_relu_backward)
@cast_for_opmath
def leaky_relu_backward(
    grad_output: Tensor, self: Tensor, negative_slope: float, self_is_result: bool
):
    return torch.where(self > 0, grad_output, grad_output * negative_slope)


@register_decomposition(aten.gelu_backward)
@cast_for_opmath
def gelu_backward_decomposition(grad: Tensor, self: Tensor, approximate: str = "none"):
    M_SQRT2 = 1.41421356237309504880
    M_SQRT1_2 = 0.70710678118654752440
    M_2_SQRTPI = 1.12837916709551257390
    if approximate == "none":
        kAlpha = M_SQRT1_2
        kBeta = M_2_SQRTPI * M_SQRT1_2 * 0.5
        cdf = 0.5 * (1 + torch.erf(self * kAlpha))
        pdf = kBeta * torch.exp(self * self * -0.5)
        return grad * (cdf + self * pdf)
    else:
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


@register_decomposition(aten.mish_backward)
def mish_backward_decomposition(grad_output: Tensor, input: Tensor):
    input_tanh_softplus = torch.tanh(F.softplus(input))
    input_sigmoid = torch.sigmoid(input)
    out = input * input_sigmoid * (1 - input_tanh_softplus * input_tanh_softplus)
    return grad_output * (input_tanh_softplus + out)


@register_decomposition(aten.silu)
def silu(self: Tensor) -> Tensor:
    return self * torch.sigmoid(self)


@register_decomposition(aten.silu_backward)
def silu_backward(grad_output: Tensor, self: Tensor) -> Tensor:
    sigmoid = 1 / (1 + torch.exp(-self))
    return grad_output * sigmoid * (1 + self * (1 - sigmoid))


@register_decomposition(aten.softshrink_backward)
def softshrink_backward(grad_output: Tensor, self: Tensor, lambd: float) -> Tensor:
    return torch.where(
        (self >= -lambd) & (self <= lambd), grad_output.new_zeros(()), grad_output
    )


@register_decomposition(aten.prelu_backward)
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
        return leaky_relu_backward(grad_output, self, negative_slope, self_is_result)


@register_decomposition(aten.log_sigmoid_backward)
@cast_for_opmath
def log_sigmoid_backward(grad_output: Tensor, self: Tensor, buffer: Tensor) -> Tensor:
    in_negative = self < 0
    max_deriv = torch.where(in_negative, 1, 0)
    sign = torch.where(in_negative, 1, -1)
    z = torch.exp(-torch.abs(self))
    return grad_output * (max_deriv - sign * (z / (1 + z)))
    # CPU has a special formula that uses buffer, but disabled for convenience sake
    # return (max_deriv - sign * (buffer / (1 + buffer))) * grad_output


@register_decomposition(aten.mse_loss_backward)
def mse_loss_backward(
    grad_output: Tensor, input: Tensor, target: Tensor, reduction: int
):
    norm = 2.0 / input.numel() if reduction == Reduction.MEAN.value else 2.0
    return norm * (input - target) * grad_output


@register_decomposition(aten.huber_loss_backward)
@cast_for_opmath
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


@register_decomposition(aten.binary_cross_entropy_backward)
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
@cast_for_opmath
def _softmax_backward_data(
    grad_output: Tensor, output: Tensor, dim: int, input_dtype: int
):
    new_grad = grad_output * output
    return new_grad - output * torch.sum(new_grad, dim=dim, keepdim=True)


@register_decomposition(aten._log_softmax_backward_data)
@cast_for_opmath
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
def native_dropout_backward(grad_output: Tensor, mask: Tensor, scale: float):
    return grad_output * (mask.type_as(grad_output) * scale)


@register_decomposition(aten.logit_backward)
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
def native_dropout_decomposition(input, p, generator=None):
    bool_mask = torch.rand_like(input) < p
    res = bool_mask * input * float(1.0 / p)
    return [res, bool_mask]


@register_decomposition(aten._softmax)
@cast_for_opmath
def _softmax(x: Tensor, dim: int, half_to_float: bool):
    x_max = torch.max(x, dim, keepdim=True)[0]
    unnormalized = torch.exp(x - x_max)
    return unnormalized / torch.sum(unnormalized, dim, keepdim=True)


@register_decomposition(aten._log_softmax)
@cast_for_opmath
def _log_softmax(x: Tensor, dim: int, half_to_float: bool):
    x_max = torch.max(x, dim, keepdim=True)[0]
    shifted = x - x_max
    shifted_logsumexp = torch.log(torch.sum(torch.exp(shifted), dim, keepdim=True))
    return shifted - shifted_logsumexp


@register_decomposition(aten.addcdiv)
@cast_for_opmath
def addcdiv(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1):
    return self + value * (tensor1 / tensor2)


# Remove special case when https://github.com/pytorch/pytorch/pull/72949 is landed.
@register_decomposition(aten.addcmul)
@cast_for_opmath
def addcmul(self: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1):
    if self.is_floating_point() or self.is_complex():
        return self + value * tensor1 * tensor2
    else:
        return self + int(value) * tensor1 * tensor2


@register_decomposition(aten.embedding_dense_backward)
def embedding_dense_backward(
    grad_output: Tensor,
    indices: Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
):
    numel = indices.numel()
    grad = grad_output.view(numel, grad_output.size(-1))
    grad_weight = grad_output.new_zeros((num_weights, grad_output.shape[-1]))
    indices_rank1 = indices.view(numel)
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


@register_decomposition(aten.native_layer_norm)
def native_layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    input_shape = input.shape
    input_ndim = input.dim()

    axis = input_ndim - len(normalized_shape)
    M = prod(input_shape[:axis])  # type: ignore[arg-type]

    # Hmm... not sure how I get around this...
    # Basically, native_batch_norm doesn't support 0-entry tensors, while
    # native_layer_norm does (and is tested by OpInfos!)
    if M > 0:
        input_reshaped = input.view(1, M, -1)
    else:
        return (input, input.new_zeros((0,)), input.new_zeros((0,)))

    # Unlike Batch Normalization, which applies scalar scale and bias for each
    # entire channel/plane with the affine option, Layer Normalization applies
    # per-element scale and bias. E.g. For input {N, C, H, W}, weight for
    # batchnorm has shape {C} while weight for layernorm has shape {H, W} or {W}.
    out, mean, rstd = aten.native_batch_norm(
        input_reshaped,
        weight=None,
        bias=None,
        running_mean=None,
        running_var=None,
        training=True,
        momentum=0.0,
        eps=eps,
    )
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
def addmm(self: Tensor, mat1: Tensor, mat2: Tensor, beta: int = 1, alpha: int = 1):
    if not self.is_floating_point() and not self.is_complex():
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mm(mat1, mat2)
    if beta == 0:
        return out
    return beta * self + out


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

    x_hat = (input - mean) * rstd
    if weight is not None:
        grad_x_hat = grad_out * weight
    else:
        grad_x_hat = grad_out
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)

    inner = a - b - c3

    if output_mask[0]:
        d_input: Optional[Tensor] = (rstd / N) * inner
    else:
        d_input = None

    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            d_weight: Optional[Tensor] = torch.sum(grad_out * x_hat, outer_dim_indices, False)
        else:
            d_weight = grad_out * x_hat
    else:
        d_weight = None

    if output_mask[2] and bias is not None:
        if len(outer_dim_indices) > 0:
            d_bias: Optional[Tensor] = torch.sum(grad_out, outer_dim_indices, False)
        else:
            d_bias = grad_out
    else:
        d_bias = None
    return (d_input, d_weight, d_bias)


@register_decomposition(aten.clamp_min)
def clamp_min(self: Tensor, min: float):
    return torch.clamp(self, min=min)


@register_decomposition(aten.clamp_max)
def clamp_max(self: Tensor, max: float):
    return torch.clamp(self, max=max)


@register_decomposition(aten._fused_dropout)
def _fused_dropout_decomposition(input, p, generator=None):
    mask = (torch.rand_like(input) < p).to(dtype=torch.uint8)
    res = mask.type_as(input) * input * (1.0 / p)
    return [res, mask]


# TODO: these logical decomps are buggy for complex inputs


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


# TODO: var and std OpInfo doesn't the next two decomps, how to get here?


@register_decomposition(aten.var.correction)
@cast_for_opmath
def var_decomposition(
    x: Tensor, dims: List[int], correction: int = 0, keepdim: bool = False
):
    if dims is None:
        dims = []

    if isinstance(dims, (tuple, list)) and len(dims) == 0:
        n = x.numel()
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
@cast_for_opmath
def std_decomposition(
    x: Tensor, dims: List[int], correction: int = 0, keepdim: bool = False
):
    return torch.sqrt(torch.var(x, dims, correction=correction, keepdim=keepdim))


# Questionable decompositions
# This is only valid if we're running the graph without autograd, such as if the backward pass has been traced.
# Note that this decomposition causes issues with in-place ops
@register_decomposition(aten.detach)
def detach_decomposition(x):
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
        return (a, b, c, input.new_zeros((1,)))
    return (a, input.new_zeros((1,)), input.new_zeros((1,)), input.new_zeros((1,)))


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
