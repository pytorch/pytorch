import functools
import logging
import math
import numbers
from itertools import chain

import torch
import torch._decomp as decomp
import torch.ao.quantization.fx._decomposed
from torch import Tensor
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._decomp.decompositions import pw_cast_for_opmath
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch.utils._mode_utils import no_dispatch

from . import config, utils

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed

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
            log.warning("duplicate decomp: %s", ops)
    return decomp.register_decomposition(ops, decompositions)


@register_decomposition(aten._unsafe_view.default)
def _unsafe_view(self, size):
    # this makes pattern matching easier
    return self.view(size)


# TODO: for now, inductor doesn't handle asserts
# because the condition is symbool -> tensor in the graph.
@register_decomposition([aten._assert_async.msg])
def assert_async_msg_decomp(tensor, msg):
    return


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


def get_alignment_size(x):
    if x.dtype == torch.float16 or x.dtype == torch.half or x.dtype == torch.bfloat16:
        return 8
    elif x.dtype == torch.float32 or x.dtype == torch.float:
        return 4
    else:
        return 0


def check_device(a: Tensor, b: Tensor):
    return a.is_cuda and b.is_cuda


def is_symbolic(a: Tensor, b: Tensor):
    return any(
        isinstance(x, torch.SymInt)
        for x in chain(a.size(), a.stride(), b.size(), b.stride())
    )


def get_padded_length(x, alignment_size):
    if alignment_size == 0 or x % alignment_size == 0:
        return 0
    return int((x // alignment_size + 1) * alignment_size) - x


def pad_dim(x, padded_length, dim):
    if padded_length == 0:
        return x
    pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
    return torch.cat([x, pad], dim=dim)


@register_decomposition([aten.addmm])
def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    if (
        config.shape_padding
        and check_device(mat1, mat2)
        and not is_symbolic(mat1, mat2)
        and should_pad_bench(mat1, mat2, torch.ops.aten.addmm, input=input)
    ):
        m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
        k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
        n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
        if m_padded_length != 0 or k_padded_length != 0 or n_padded_length != 0:
            return pad_addmm(
                input, mat1, mat2, m_padded_length, k_padded_length, n_padded_length
            )

    return NotImplemented  # go directly to lowering


def pad_addmm(input, mat1, mat2, m_padded_length, k_padded_length, n_padded_length):
    # addmm decomp with padding will go through pad_addmm multiple times if multiple dimensions are needed to be padded
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
    elif m_padded_length != 0:
        mat1 = pad_dim(mat1, m_padded_length, 0)

    if input is not None and k_padded_length == 0:
        if n_padded_length != 0:
            if input.dim() == 2:
                input = pad_dim(input, n_padded_length, 1)
            elif input.dim() == 1:
                input = pad_dim(input, n_padded_length, 0)
        elif m_padded_length != 0 and input.dim() == 2:
            input = pad_dim(input, m_padded_length, 0)

    if k_padded_length != 0:
        return torch.ops.aten.addmm(input, mat1, mat2)
    elif n_padded_length != 0:
        return torch.ops.aten.addmm(input, mat1, mat2)[:, :-n_padded_length]
    else:
        return torch.ops.aten.addmm(input, mat1, mat2)[:-m_padded_length, :]


def should_pad_bench(mat1, mat2, op, input=None):
    assert utils.has_triton()

    import triton.testing

    do_bench = functools.partial(
        triton.testing.do_bench,
        warmup=5,
        rep=100,
        fast_flush=True,
    )

    with no_dispatch():
        if op is torch.ops.aten.mm or op is torch.ops.aten.addmm:
            m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
            k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
            n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))
        elif op is torch.ops.aten.bmm:
            m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
            k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
            n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))
        else:
            return False

        if m_padded_length == k_padded_length == n_padded_length == 0:
            return False

        mat1 = torch.randn_like(mat1)
        mat2 = torch.randn_like(mat2)
        warmup = 5
        rep = 100
        if op is torch.ops.aten.bmm or op is torch.ops.aten.mm:
            ori_time = do_bench(
                lambda: op(mat1, mat2),
            )
        else:
            if input is not None:
                input = torch.randn_like(input)
            ori_time = do_bench(
                lambda: op(input, mat1, mat2),
            )

        mat1_pad = torch.randn_like(mat1)
        mat2_pad = torch.randn_like(mat2)

        if op is torch.ops.aten.addmm:
            input_pad = None
            if input is not None and input.is_cuda:
                input_pad = torch.randn_like(input)
            pad_time = do_bench(
                lambda: pad_addmm(
                    input_pad,
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
            )
        elif op is torch.ops.aten.mm:
            pad_time = do_bench(
                lambda: pad_mm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
            )
        else:
            pad_time = do_bench(
                lambda: pad_bmm(
                    mat1_pad,
                    mat2_pad,
                    m_padded_length,
                    k_padded_length,
                    n_padded_length,
                ),
            )

        # Shape padding introduces additional memory ops. Based on microbenchmarks, 1.1x represents a reasonable
        # tradeoff between performance improvement from shape padding and overhead from additional memory ops
        # TODO: Build a learned model which would be better than this heuristic
        return ori_time > pad_time * 1.1


@register_decomposition([aten.mm])
def mm_decomp(mat1, mat2):
    if (
        config.shape_padding
        and check_device(mat1, mat2)
        and not is_symbolic(mat1, mat2)
        and should_pad_bench(mat1, mat2, torch.ops.aten.mm)
    ):
        m_padded_length = get_padded_length(mat1.shape[0], get_alignment_size(mat1))
        k_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
        n_padded_length = get_padded_length(mat2.shape[1], get_alignment_size(mat2))

        if m_padded_length != 0 or k_padded_length != 0 or n_padded_length != 0:
            return pad_mm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)

    return NotImplemented  # go directly to lowering


def pad_mm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length):
    # mm_decomp will go through pad_mm multiple times if multiple dimensions are needed to be padded
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 1)
        mat2 = pad_dim(mat2, k_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 1)
        return torch.ops.aten.mm(mat1, mat2)[:, :-n_padded_length]
    else:
        mat1 = pad_dim(mat1, m_padded_length, 0)
        return torch.ops.aten.mm(mat1, mat2)[:-m_padded_length, :]


@register_decomposition([aten.bmm])
def bmm_decomp(mat1, mat2):
    if (
        config.shape_padding
        and check_device(mat1, mat2)
        and not is_symbolic(mat1, mat2)
        and should_pad_bench(mat1, mat2, torch.ops.aten.bmm)
    ):
        m_padded_length = get_padded_length(mat1.shape[1], get_alignment_size(mat1))
        k_padded_length = get_padded_length(mat1.shape[2], get_alignment_size(mat1))
        n_padded_length = get_padded_length(mat2.shape[2], get_alignment_size(mat2))

        if k_padded_length != 0 or n_padded_length != 0 or m_padded_length != 0:
            pad_bmm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length)

    return NotImplemented  # go directly to lowering


def pad_bmm(mat1, mat2, m_padded_length, k_padded_length, n_padded_length):
    # bmm_decomp will go through pad_bmm multiple times if multiple dimensions are needed to be padded
    if k_padded_length != 0:
        mat1 = pad_dim(mat1, k_padded_length, 2)
        mat2 = pad_dim(mat2, k_padded_length, 1)
        return torch.ops.aten.bmm(mat1, mat2)
    elif n_padded_length != 0:
        mat2 = pad_dim(mat2, n_padded_length, 2)
        return torch.ops.aten.bmm(mat1, mat2)[:, :, :-n_padded_length].contiguous()
    else:
        mat1 = pad_dim(mat1, m_padded_length, 1)
        return torch.ops.aten.bmm(mat1, mat2)[:, :-m_padded_length, :].contiguous()


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


@register_decomposition([aten.cat.default])
def cat(tensors, dim=0):
    if len(tensors) == 1:
        return tensors[0].clone()
    return NotImplemented


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


@register_decomposition([aten.native_dropout])
def native_dropout(input: Tensor, p: float, train: bool):
    if not train or p == 0:
        return (input, torch.ones_like(input, dtype=torch.bool))
    if p == 1:
        return (torch.zeros_like(input), torch.zeros_like(input, dtype=torch.bool))
    # intentionally don't decompose to improve pattern matching
    return NotImplemented


# The difference between quantize_per_tensor.default and quantize_per_tensor.tensor is
# scale and zero_point is scalar or scalar tensor
@register_decomposition(quantized_decomposed.quantize_per_tensor.default)
def quantize_per_tensor_default_decomp_impl(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_scale = 1.0 / scale
    return torch.clamp(
        torch.round(input * inv_scale) + zero_point, quant_min, quant_max
    ).to(dtype)


# The difference between dequantize_per_tensor.default and dequantize_per_tensor.tensor is
# scale and zero_point is scalar or scalar tensor
@register_decomposition(quantized_decomposed.dequantize_per_tensor.default)
def dequantize_per_tensor_default_decomp_impl(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (input.to(torch.float32) - zero_point) * scale


@register_decomposition(quantized_decomposed.quantize_per_tensor.tensor)
def quantize_per_tensor_tensor_decomp_impl(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    inv_scale = 1.0 / scale
    return torch.clamp(
        torch.round(input * inv_scale) + zero_point, quant_min, quant_max
    ).to(dtype)


@register_decomposition(quantized_decomposed.dequantize_per_tensor.tensor)
def dequantize_per_tensor_tensor_decomp_impl(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return (input.to(torch.float32) - zero_point) * scale


@functools.lru_cache(None)
def fast_random_decomps():
    return {**decompositions, **extra_random_decomps}


def select_decomp_table():
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    return fast_random_decomps()
