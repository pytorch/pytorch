import torch
from torch import Tensor
import torch._decomp
from typing import Tuple, List, Optional

aten = torch.ops.aten

decomposition_table = torch._decomp.decomposition_table
register_decomposition = torch._decomp.register_decomposition
get_decompositions = torch._decomp.get_decompositions

# Decompositions have been ported to torch._decomp inside of PyTorch core. The only decompositions here are temporary or hacks. Please submit your contributions to PyTorch core!


def maybe_register_decomposition(op):
    def decorator(f):
        try:
            return register_decomposition(op)(f)
        except Exception:
            return f
    return decorator


# Functions where we need a special decomposition for jvp but there's another version that
# should be used more generally (ex. for jvp we need to recompute the mean and variance for
# the backwards of a normalization function. Without jvp, it should used the saved value)
decomposition_table_for_jvp = {}


def register_decomposition_for_jvp(fn):
    return register_decomposition(fn, registry=decomposition_table_for_jvp)


@maybe_register_decomposition(aten.trace.default)
def trace(self: Tensor) -> Tensor:
    return torch.sum(torch.diag(self))


@maybe_register_decomposition(aten.log_sigmoid_forward.default)
def log_sigmoid_forward(self: Tensor) -> Tuple[Tensor, Tensor]:
    min = torch.minimum(self.new_zeros(()), self)
    z = torch.exp(-torch.abs(self))
    if self.is_cuda:
        buffer = self.new_zeros((0,))
    else:
        buffer = z
    return min - torch.log1p(z), buffer


@register_decomposition_for_jvp(aten.native_layer_norm_backward)
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
    inner_dim_indices = list(range(axis, input_ndim))
    outer_dim_indices = list(range(0, axis))

    N = 1
    for i in inner_dims:
        N *= i
    M = 1
    for i in outer_dims:
        M *= i
    if M <= 0 or N <= 0:
        return (
            input.new_zeros(input_shape),
            input.new_zeros(input_shape[axis:]),
            input.new_zeros(input_shape[axis:]),
        )

    # this is exactly the same as the other decomposition except for here. We recompute the mean and variance
    # so that they track gradients through input
    mean_ = torch.mean(input, dim=inner_dim_indices, keepdim=True)
    var = torch.var(input, dim=inner_dim_indices, unbiased=False, keepdim=True)
    eps = torch.pow(1 / rstd, 2) - var  # this makes me so sad inside
    eps = eps.detach()
    rstd_ = 1 / torch.sqrt(var + eps)

    x_hat = (input - mean_) * rstd_
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
        d_input: Optional[Tensor] = (rstd_ / N) * inner
    else:
        d_input = torch.zeros_like(input)  # should be None but doesn't work with vjp

    if output_mask[1] and weight is not None:
        if len(outer_dim_indices) > 0:
            d_weight: Optional[Tensor] = torch.sum(
                grad_out * x_hat, outer_dim_indices, False
            )
        else:
            d_weight = grad_out * x_hat
    elif weight is not None:
        d_weight = torch.zeros_like(weight)  # should be None but doesn't work with vjp
    else:
        d_weight = torch.zeros(())  # should be None but doesn't work with vjp

    if output_mask[2] and bias is not None:
        if len(outer_dim_indices) > 0:
            d_bias: Optional[Tensor] = torch.sum(grad_out, outer_dim_indices, False)
        else:
            d_bias = grad_out
    elif bias is not None:
        d_bias = torch.zeros_like(bias)  # should be None but doesn't work with vjp
    else:
        d_bias = torch.zeros(())  # should be None but doesn't work with vjp

    return (d_input, d_weight, d_bias)
