import functools
import logging
import math
import numbers

import torch
import torch._decomp as decomp
from torch import Tensor
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._decomp.decompositions import pw_cast_for_opmath
from torch.utils._mode_utils import no_dispatch

from . import config, utils

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

inductor_decompositions = get_decompositions(
    [
        aten.arange,
        aten.bitwise_and_,
        aten.bitwise_or_,
        aten.clamp_min_,
        aten.flip,
        aten.lcm,
        aten.linalg_vector_norm,
        aten.sin_,
        aten.sqrt_,
        aten.std,
        aten.std_mean,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.unsafe_split,
    ]
)
decompositions = {**core_aten_decompositions(), **inductor_decompositions}


def register_decomposition(ops):
    for op in [ops] if callable(ops) else ops:
        if op in decompositions:
            log.warning(f"duplicate decomp: {ops}")
    return decomp.register_decomposition(ops, decompositions)


@register_decomposition([aten.clamp])
@pw_cast_for_opmath
def clamp(x, min=None, max=None):
    if min is not None:
        x = x.clamp_min(min)
    if max is not None:
        x = x.clamp_max(max)
    return x


# TorchInductor-only decomposition. It should not be taken to core.
# See https://github.com/pytorch/torchdynamo/pull/1120
@register_decomposition([aten.floor_divide.default])
def floordiv(a, b):
    return aten.div.Tensor_mode(a, b, rounding_mode="floor")


# Not really sure how to put this into the main library.  PrimTorch wants
# empty_permuted to go to the prim, and typically users don't really want
# to decompose to empty_strided (but inductor is OK with it, because we are
# cool with strides and everything goes to empty_strided)
@register_decomposition([aten.empty_permuted.default])
def empty_permuted(size, physical_layout, **kwargs):
    perm = [0] * len(size)
    for p, l in enumerate(physical_layout):
        perm[l] = p
    return torch.empty([size[l] for l in physical_layout], **kwargs).permute(perm)





def check_device(a: Tensor, b: Tensor):
    return a.is_cuda and b.is_cuda


@register_decomposition([aten.mm])
def mm_decomp(mat1, mat2):
    return NotImplemented  # go directly to lowering

@register_decomposition([aten.bmm])
def bmm_decomp(mat1, mat2):
    return NotImplemented  # go directly to lowering



@register_decomposition([aten.convolution_backward])
def convolution_backward(
    grad_output,
    input,
    weight,
    bias_sizes,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    output_mask,
):
    if not output_mask[2] or grad_output.device.type != "cuda":
        return NotImplemented
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, grad_weight, _ = aten.convolution_backward(
        grad_output,
        input,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        [output_mask[0], output_mask[1], False],
    )
    return (grad_inp, grad_weight, grad_bias)


@register_decomposition([aten.log2])
def log2(x):
    return torch.log(x) * (1.0 / math.log(2.0))


@register_decomposition([aten.round.decimals])
def round_dec(x, decimals=0):
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)


@register_decomposition([aten.all.default])
def all(input):
    return torch.logical_not(torch.any(torch.logical_not(input)))


@register_decomposition([aten.all.dim])
def all_dim(input, dim, keepdim=False):
    return torch.logical_not(torch.any(torch.logical_not(input), dim, keepdim))


# NB: this decomposition is not stride accurate, do not put it in the main
# library
@register_decomposition(aten.copy)
def copy(self, src, non_blocking=False):
    intermediate = src.to(self, non_blocking)
    if self.size() != intermediate.size():
        return aten.expand_copy.default(intermediate, self.size())
    else:
        return intermediate


@register_decomposition([aten.baddbmm])
def baddbmm(self, batch1, batch2, beta=1, alpha=1):
    result = torch.bmm(batch1, batch2)
    if not isinstance(alpha, numbers.Number) or alpha != 1:
        result = result * alpha
    if beta == 0:
        return result
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


@register_decomposition([aten.bernoulli.default])
def bernoulli(self, *, generator=None):
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < self


@register_decomposition([aten.fmin, prims.fmin])
def fmin(self, other):
    return torch.where(torch.isnan(other) | (other > self), self, other)


@register_decomposition([aten.fmax, prims.fmax])
def fmax(self, other):
    return torch.where(torch.isnan(other) | (other < self), self, other)


@register_decomposition([aten.narrow_copy])
def narrow_copy(self, dim, start, length):
    return torch.narrow(self, dim, start, length).clone()


@register_decomposition([aten.expand_copy])
def expand_copy(self, size, *, implicit=False):
    return aten.expand(self, size, implicit=implicit).clone()


@register_decomposition([aten.view_copy.default])
def view_copy_default(self, size):
    return aten.view(self, size).clone()


@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(self, dtype):
    return self.to(dtype).clone()


"""
Some decomps result in differences from eager related to randomness.
We put these decomps in a separate table `extra_random_decomps` to allow
turning them on and off via `config.fallback_random`.
"""
extra_random_decomps = get_decompositions(
    [
        aten.native_dropout,
        aten.cauchy,
        aten.cauchy_,
        aten.exponential,
        aten.exponential_,
        aten.geometric,
        aten.geometric_,
        aten.normal,
        aten.normal_,
        aten.normal_functional,
        aten.log_normal,
        aten.log_normal_,
        aten.uniform_,
    ]
)
register_extra_random_decomp = functools.partial(
    decomp.register_decomposition, registry=extra_random_decomps
)


@register_extra_random_decomp([aten.bernoulli_])
def bernoulli_(self, p=0.5):
    return self.copy_(torch.rand_like(self, dtype=torch.float32) < p)


@register_extra_random_decomp([aten.bernoulli.p])
def bernoulli_p(self, p=0.5, *, generator=None):
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < p


@functools.lru_cache(None)
def fast_random_decomps():
    return {**decompositions, **extra_random_decomps}


def select_decomp_table():
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    return fast_random_decomps()
