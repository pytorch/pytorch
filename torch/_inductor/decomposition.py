import functools
import logging
import math
import numbers

import torch
import torch._decomp as decomp
from torch import Tensor
from torch._decomp import get_decompositions
from torch._prims_common import is_boolean_dtype, is_integer_dtype

from . import config

log = logging.getLogger(__name__)
aten = torch.ops.aten
log = logging.getLogger(__name__)

decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.addcmul,
        aten.avg_pool2d_backward,
        aten.binary_cross_entropy_with_logits,
        aten.clamp_max,
        aten.clamp_min,
        aten.col2im,
        aten.cudnn_batch_norm,
        aten.cudnn_batch_norm_backward,
        aten.detach,
        aten.dot,
        aten.elu,
        aten.elu_backward,
        aten._embedding_bag,
        aten.embedding_dense_backward,
        aten.expand_as,
        aten.eye,
        aten.flip,
        aten._fused_moving_avg_obs_fq_helper,
        aten.gelu,
        aten.gelu_backward,
        aten.glu_backward,
        aten.grid_sampler_2d,
        aten.hardsigmoid,
        aten.hardsigmoid_backward,
        aten.hardswish,
        aten.hardswish_backward,
        aten.hardtanh,
        aten.hardtanh_backward,
        aten.im2col,
        aten.index_add,
        aten.index_add_,
        aten.index_select,
        aten.l1_loss,
        aten.leaky_relu,
        aten.leaky_relu_backward,
        aten.linalg_vector_norm,
        aten.logit,
        aten.logit_backward,
        aten._log_softmax,
        aten._log_softmax_backward_data,
        aten.logsumexp.default,
        aten.max_pool2d_with_indices_backward,
        aten.mse_loss,
        aten.mse_loss_backward,
        aten.mv,
        aten.narrow,
        aten.native_batch_norm,
        aten.native_batch_norm_backward,
        aten.native_dropout_backward,
        aten.native_group_norm,
        aten.native_group_norm_backward,
        aten.native_layer_norm,
        aten.native_layer_norm_backward,
        aten.new_empty,
        aten.new_full,
        aten.new_ones,
        aten.nll_loss_backward,
        aten.nll_loss_forward,
        aten.norm,
        aten.reflection_pad2d_backward,
        aten._reshape_alias,
        aten.select_backward,
        aten.select_scatter,
        aten.sgn,
        aten.sigmoid_backward,
        aten.silu,
        aten.silu_backward,
        aten.slice_backward,
        aten._softmax,
        aten._softmax_backward_data,
        aten.softplus,
        aten.softplus_backward,
        aten.stack,
        aten.std_mean.correction,
        aten.t,
        aten.tanh_backward,
        aten.threshold_backward,
        aten.transpose.int,
        aten.tril.default,
        aten.unfold,
        aten.unfold_backward,
        aten.upsample_bilinear2d.vec,
        aten.upsample_nearest2d_backward,
        aten.softplus,
        aten.softplus_backward,
    ]
)


def register_decomposition(ops):
    for op in [ops] if callable(ops) else ops:
        if op in decompositions:
            log.warning(f"duplicate decomp: {ops}")
    return decomp.register_decomposition(ops, decompositions)


@register_decomposition([aten.clamp])
def clamp(x, min=None, max=None):
    if min is not None:
        x = torch.maximum(x, torch.tensor(min, dtype=x.dtype, device=x.device))
    if max is not None:
        x = torch.minimum(x, torch.tensor(max, dtype=x.dtype, device=x.device))
    return x


@register_decomposition([aten.tanh])
def tanh(x):
    return 2.0 / (1.0 + torch.exp(-2.0 * x)) - 1.0


# TorchInductor-only decomposition. It should not be taken to core.
# See https://github.com/pytorch/torchdynamo/pull/1120
@register_decomposition([aten.floor_divide.default])
def floordiv(a, b):
    return aten.div.Tensor_mode(a, b, rounding_mode="floor")


@register_decomposition([aten.addmm])
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    if config.triton.mm != "aten":
        out = torch.mm(mat1, mat2)
        if not isinstance(alpha, numbers.Number) or alpha != 1:
            out = out * alpha
        if not isinstance(beta, numbers.Number) or beta != 1:
            input = input * beta
        return input + out
    else:
        return NotImplemented  # go directly to lowering


@register_decomposition([aten.rsqrt])
def rsqrt(x):
    return torch.reciprocal(torch.sqrt(x))


@register_decomposition([aten.log2])
def log2(x):
    return torch.log(x) * (1.0 / math.log(2.0))


@register_decomposition([aten.round.decimals])
def round_dec(x, decimals=0):
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)


@register_decomposition([aten.special_erf, aten.erf])
def special_erf(x):
    # TODO(jansel): this might be crazy slow.  Triton doesn't have the
    #               cuda ::erf() builtin.  I've made a feature request for this,
    #               so it may be coming soon.

    # from https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    sign = torch.sign(x)
    x = torch.abs(x)

    # A & S 7.1.26
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * torch.exp(-x * x)

    return sign * y


@register_decomposition([aten.rsub.Tensor, aten.rsub.Scalar])
def rsub(a, b):
    if isinstance(b, numbers.Number):
        b = torch.tensor(b, dtype=a.dtype, device=a.device)
    return b - a


@register_decomposition([aten.masked_fill])
def masked_fill(value, mask, other):
    if isinstance(other, numbers.Number):
        other = torch.tensor(other, dtype=value.dtype, device=value.device)
    assert other.numel() == 1 and other.ndim == 0
    if other.device != value.device and other.numel() == 1:
        other = other.to(value.device)
    if other.dtype != value.dtype:
        # TODO: error out on improper complex conversions
        other = other.to(value.dtype)
    return torch.where(mask, other, value)


@register_decomposition([aten.nan_to_num])
def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    if is_boolean_dtype(x.dtype) or is_integer_dtype(x.dtype):
        return x

    if nan is None:
        nan = 0.0
    if posinf is None:
        posinf = torch.finfo(x.dtype).max
    if neginf is None:
        neginf = torch.finfo(x.dtype).min
    nan, posinf, neginf = (
        torch.tensor(v, dtype=x.dtype, device=x.device) for v in (nan, posinf, neginf)
    )
    x = torch.where(x != x, nan, x)
    x = torch.where(x == float("inf"), posinf, x)
    x = torch.where(x == float("-inf"), neginf, x)
    return x


@register_decomposition([aten.all.default])
def all(input):
    return torch.logical_not(torch.any(torch.logical_not(input)))


@register_decomposition([aten.all.dim])
def all_dim(input, dim, keeepdim=False):
    return torch.logical_not(torch.any(torch.logical_not(input), dim, keeepdim))


@register_decomposition(aten.hardswish_)
def hardswish_(x):
    return x.copy_(aten.hardswish(x))


@register_decomposition(aten.hardtanh_)
def hardtanh_(x, min_val=-1, max_val=1):
    return x.copy_(aten.hardtanh(x, min_val, max_val))


@register_decomposition(aten.leaky_relu_)
def leaky_relu_(x, negative_slope=0.01):
    return x.copy_(aten.leaky_relu(x, negative_slope))


@register_decomposition(aten.silu_)
def silu_(x):
    return x.copy_(aten.silu(x))


@register_decomposition(aten.masked_fill_)
def masked_fill_(x, mask, value):
    return x.copy_(aten.masked_fill(x, mask, value))


@register_decomposition([aten.log1p])
def log1p(x):
    return torch.log(x + 1)


@register_decomposition([aten.baddbmm])
def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    result = torch.bmm(batch1, batch2)
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    if not isinstance(beta, numbers.Number) or beta != 1:
        self = self * beta
    return self + result


@register_decomposition([aten.conj_physical])
def conj_physical(self):
    assert not self.is_complex(), "TODO: implement this"
    return self


@register_decomposition([aten.lift, aten.detach_])
def lift(self):
    return self


@register_decomposition([aten.fill.Scalar])
def fill_scalar(self, value):
    return torch.full_like(self, value)


@register_decomposition([aten.fill.Tensor])
def fill_tensor(self, value: Tensor):
    assert value.dim() == 0, "aten.fill.Tensor only supports 0-dimension value tensor"
    return torch.full_like(self, value.item())


@register_decomposition([aten.bernoulli.default])
def bernoulli(self, *, generator=None):
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < self


@register_decomposition([aten.bernoulli.p])
def bernoulli_p(self, p=0.5, *, generator=None):
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < p


"""
Some decomps result in differences from eager related to randomness.
We put these decomps in a separate table `extra_random_decomps` to allow
turning them on and off via `config.fallback_random`.
"""
extra_random_decomps = get_decompositions([aten.native_dropout])
register_extra_random_decomp = functools.partial(
    decomp.register_decomposition, registry=extra_random_decomps
)


@register_extra_random_decomp([aten.bernoulli_])
def bernoulli_(self, p=0.5):
    return self.copy_(torch.rand_like(self) < p)


@functools.lru_cache(None)
def fast_random_decomps():
    return {**decompositions, **extra_random_decomps}


def select_decomp_table():
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    return fast_random_decomps()
