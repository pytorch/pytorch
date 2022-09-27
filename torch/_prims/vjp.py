# Module for defining vector-Jacobian product (VJP) functions for PrimTorch
# primitives.

import math
from typing import Callable, Dict

import torch
import torch._prims_common as utils

prims = torch.ops.prims


def _dim_size(a, dims):
    dims = utils.canonicalize_dims(a.ndim, dims)
    reduction_size = 1
    for idx, size in enumerate(a.size()):
        if idx in dims:
            reduction_size *= size
    return reduction_size


def _restore_reduced_dims(a, dims, shape):
    if a.size() == shape:
        return a
    unsqueezed_a = torch._prims.expand_dims(a, dims, len(shape))
    return torch._refs.expand(unsqueezed_a, shape)


# Reference: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml#L1109-L1115
def _amax_amin_vjp(grad, result, self, dims):
    expanded_grad = _restore_reduced_dims(grad, dims, self.shape)
    expanded_result = _restore_reduced_dims(result, dims, self.shape)
    mask = torch.eq(expanded_result, self)
    return (expanded_grad / torch.sum(mask, dims, keepdim=True)) * mask, *(None,) * len(
        dims
    )


def _sum_vjp(grad, result, self, dims):
    # Return None for each dim.
    return _restore_reduced_dims(grad, dims, self.shape), *(None,) * len(dims)


def _mean_vjp(grad, self, dims):
    mean_local_grad = 1.0 / _dim_size(self, dims)
    return _restore_reduced_dims(grad, dims, self.shape) * mean_local_grad


def _var_vjp(grad, result, self, dims, correction):
    var_reduction_size = _dim_size(self, dims)
    var_reduction_size -= correction
    constant = 2.0 / var_reduction_size

    # expand grad and mean tensors to self tensor size
    expanded_grad = _restore_reduced_dims(grad, dims, self.shape)
    mean = torch._refs.mean(self, dims, keepdim=True)
    expanded_mean = torch._refs.broadcast_to(mean, self.shape)
    var_local_grad = constant * prims.sub(self, expanded_mean)
    # Return None for each dim and for correction argument.
    return expanded_grad * var_local_grad, *(None,) * (len(dims) + 1)


def _broadcast_in_dim_vjp(grad, result, self, shape, broadcast_dimensions):
    pre_expand_shape = [
        1 if i not in broadcast_dimensions else x for i, x in enumerate(shape)
    ]
    grad = grad.sum_to_size(pre_expand_shape)
    squeeze_dims = [i for i in range(0, grad.ndim) if i not in broadcast_dimensions]
    grad = prims.squeeze(grad, squeeze_dims)
    return grad, *(None,) * (len(shape) + len(broadcast_dimensions))


def _squeeze_vjp(grad, result, self, dims):
    return torch._prims.expand_dims(grad, dims, self.ndim), *(None,) * len(dims)


def _transpose_vjp(grad, result, self, permutation):
    return prims.transpose(grad, permutation), *(None,) * len(permutation)


def _var_mean_vjp(grad_var, grad_mean, var, mean, self, dims, correction):
    grad = _var_vjp(grad_var, var, self, dims, correction)[0] + _mean_vjp(
        grad_mean, self, dims
    )
    return grad, *(None,) * (len(dims) + 1)


# vjp_implementations["prim_name"] gives a callable that takes in the backward
# gradient (one for each output), the forward outputs and the forward inputs.
# This callable returns a tuple of VJP results for each input. If the input was
# a single tensor, then the VJP result is a single tensor, not a tuple.
vjp_implementations: Dict[str, Callable] = {
    "abs": lambda grad, result, self: prims.mul(grad, prims.sign(self)),
    "acos": lambda grad, result, self: prims.mul(
        grad, prims.neg(prims.rsqrt(prims.sub(1, prims.pow(self, 2))))
    ),
    "add": lambda grad, result, self, other: (prims.view_of(grad), prims.view_of(grad)),
    "amax": _amax_amin_vjp,
    "amin": _amax_amin_vjp,
    "asin": lambda grad, result, self: prims.mul(
        grad, prims.rsqrt(prims.sub(1, prims.pow(self, 2)))
    ),
    "atan": lambda grad, result, self: prims.mul(
        grad, prims.reciprocal(prims.add(1, prims.pow(self, 2)))
    ),
    "atan2": lambda grad, result, self, other: (
        prims.mul(
            grad, prims.div(other, prims.add(prims.pow(self, 2), prims.pow(other, 2)))
        ),
        prims.mul(
            grad,
            prims.div(
                prims.neg(self), prims.add(prims.pow(self, 2), prims.pow(other, 2))
            ),
        ),
    ),
    "atanh": lambda grad, result, self: prims.mul(
        grad, prims.reciprocal(prims.sub(1, prims.pow(self, 2)))
    ),
    "broadcast_in_dim": _broadcast_in_dim_vjp,
    "ceil": lambda grad, result, self: prims.mul(grad, 0),
    "convert_element_type": lambda grad, result, self, dtype: (
        prims.convert_element_type(grad, self.dtype),
        None,
    ),
    "cos": lambda grad, result, self: prims.mul(grad, prims.neg(prims.sin(self))),
    "cosh": lambda grad, result, self: prims.mul(grad, prims.sinh(self)),
    "div": lambda grad, result, self, other: (
        prims.div(grad, other),
        prims.mul(prims.mul(prims.neg(grad), self), prims.pow(other, -2)),
    ),
    "erf": lambda grad, result, self: prims.mul(
        grad,
        prims.mul(2 / math.sqrt(math.pi), prims.exp(prims.neg(prims.pow(self, 2)))),
    ),
    "erfc": lambda grad, result, self: prims.mul(
        grad,
        prims.mul(-2 / math.sqrt(math.pi), prims.exp(prims.neg(prims.pow(self, 2)))),
    ),
    "exp": lambda grad, result, self: prims.mul(grad, result),
    "expm1": lambda grad, result, self: prims.mul(grad, prims.add(result, 1)),
    "floor": lambda grad, result, self: prims.mul(grad, 0),
    "fmod": lambda grad, result, self, other: (
        grad,
        prims.mul(prims.neg(grad), prims.trunc(prims.div(self, other))),
    ),
    "lgamma": lambda grad, result, self: prims.mul(grad, prims.digamma(self)),
    "log": lambda grad, result, self: prims.div(grad, self),
    "log10": lambda grad, result, self: prims.div(grad, prims.mul(self, math.log(10))),
    "log1p": lambda grad, result, self: prims.div(grad, prims.add(self, 1)),
    "log2": lambda grad, result, self: prims.div(grad, prims.mul(self, math.log(2))),
    "mul": lambda grad, result, self, other: (
        prims.mul(grad, other),
        prims.mul(grad, self),
    ),
    "neg": lambda grad, result, self: prims.neg(grad),
    "pow": lambda grad, result, self, other: (
        prims.mul(grad, prims.mul(other, prims.pow(self, prims.sub(other, 1)))),
        prims.mul(grad, prims.mul(prims.log(self), result)),
    ),
    "reciprocal": lambda grad, result, self: prims.mul(
        grad, prims.neg(prims.pow(result, 2))
    ),
    "remainder": lambda grad, result, self, other: (
        grad,
        prims.neg(prims.mul(grad, prims.floor(prims.div(self, other)))),
    ),
    "round": lambda grad, result, self: prims.mul(grad, 0),
    "rsqrt": lambda grad, result, self: prims.mul(
        grad, prims.mul(-0.5, prims.div(result, self))
    ),
    "sign": lambda grad, result, self: prims.mul(grad, 0),
    "sin": lambda grad, result, self: prims.mul(grad, prims.cos(self)),
    "sinh": lambda grad, result, self: prims.mul(grad, prims.cosh(self)),
    "sqrt": lambda grad, result, self: prims.mul(grad, prims.div(0.5, result)),
    "squeeze": _squeeze_vjp,
    "sub": lambda grad, result, self, other: (grad, prims.neg(grad)),
    "sum": _sum_vjp,
    "tan": lambda grad, result, self: prims.mul(
        grad, prims.add(1, prims.pow(result, 2))
    ),
    "tanh": lambda grad, result, self: prims.mul(
        grad, prims.sub(1, prims.pow(result, 2))
    ),
    "transpose": _transpose_vjp,
    "trunc": lambda grad, result, self: prims.mul(grad, 0),
    "var": _var_vjp,
    "var_mean": _var_mean_vjp,
    "view_of": lambda grad, result, self: prims.view_of(grad),
    "where": lambda grad, result, condition, self, other: (
        None,
        prims.where(condition, grad, 0),
        prims.where(condition, 0, grad),
    ),
}
