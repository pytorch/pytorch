import functools
import logging
import math
import numbers
import typing

import torch
import torch._decomp as decomp
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
    core_aten_decompositions,
    get_decompositions,
    remove_decompositions,
)
from torch._decomp.decompositions import pw_cast_for_opmath
from torch._decomp.decompositions_for_rng import extra_random_decomps

from . import config

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed

inductor_decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.arange,
        aten.bitwise_and_,
        aten.bitwise_or_,
        aten.clamp_min_,
        aten.dist,
        aten.empty_like,
        aten.flip,
        aten.gelu,
        aten.hardtanh,
        aten.index_select,
        aten.lcm,
        aten.leaky_relu,
        aten.linalg_vector_norm,
        aten._log_softmax,
        aten.max_pool2d_with_indices_backward,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten.native_batch_norm,
        aten.native_group_norm,
        aten.native_layer_norm,
        aten._softmax,
        aten.sin_,
        aten.sqrt_,
        aten.std,
        aten.std_mean,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.unsafe_split,
        aten.upsample_bilinear2d.vec,
    ]
)
decompositions = {**core_aten_decompositions(), **inductor_decompositions}

# Remove unwanted decompositions included via the core ATen decompositions from
# the Inductor decomp table.
decomps_to_exclude = [
    aten._unsafe_index,
]

remove_decompositions(decompositions, decomps_to_exclude)


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


# Following `assert_async_msg_decomp` and implement as non-op.
@register_decomposition([aten._functional_assert_async.msg])
def functional_assert_async_msg_decomp(tensor, msg):
    return


@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(symbol, *, min=None, max=None):
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


@register_decomposition([aten.bmm])
def bmm(self, batch2):
    if self.device == "cpu":
        if self.size(1) == 1 and batch2.size(-1) == 1:
            return torch.sum(
                self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True
            ).unsqueeze(1)
    return NotImplemented


@register_decomposition([aten.mm])
def mm(self, input2):
    if self.device == "cpu":
        if (
            self.size(-1) == 1
            and input2.size(0) == 1
            and (self.dtype == input2.dtype)
            and ((torch.numel(self) + torch.numel(input2)) <= 32)
        ):
            return torch.cat([self[i, :] * input2 for i in range(self.size(0))])
        if self.size(0) == 1 and input2.size(-1) == 1:
            return torch.sum(
                self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
    return NotImplemented


@register_decomposition([aten.cat.default])
def cat(tensors, dim=0):
    def non_empty_tensor(x):
        # special case for cat'ing with an empty tensor -
        # just drop the 'empty' inputs so they don't confuse the logic below.
        return len(x.shape) > 1 or x.shape[0] > 0

    filtered_tensors = list(filter(non_empty_tensor, tensors))

    if len(filtered_tensors) == 1:
        return tensors[0].clone()
    elif 1 < len(filtered_tensors) < len(tensors):
        # on the first call, when we remove empty tensors, we redispatch recursively
        return aten.cat.default(filtered_tensors, dim)
    # when no 'filtering' has occured, we raise to prevent infinite recursion (no more decomposition needed)
    return NotImplemented


@register_decomposition([aten.angle])
def angle(x):
    if x.is_complex():
        return torch.where(
            torch.isnan(x.real), float("nan"), torch.atan2(x.imag, x.real)
        )
    else:
        # when x is real number
        #   if x >= 0, return 0
        #   if x < 0, return pi
        #   if x is nan, return nan
        ret = torch.where(x < 0, math.pi, 0.0)
        nan = torch.where(torch.isnan(x), float("nan"), 0.0)
        return ret + nan


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


@register_decomposition(aten.rand_like)
def rand_like(self, *, dtype=None, device=None, **kwargs):
    return torch.rand(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    )


@register_decomposition(aten.randn_like)
def randn_like(self, *, dtype=None, device=None, **kwargs):
    return torch.randn(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    )


@register_decomposition(aten.full_like)
def full_like(
    self,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    requires_grad=False,
    memory_format=torch.preserve_format,
):
    return torch.full(
        [*self.size()],
        fill_value,
        dtype=dtype or self.dtype,
        layout=layout or self.layout,
        device=device or self.device,
        requires_grad=requires_grad or self.requires_grad,
    )


@register_decomposition(aten.randint_like.default)
def randint_like(self, high, *, dtype=None, device=None, **kwargs):
    return aten.randint.low(
        0,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    )


@register_decomposition(aten.randint_like.low_dtype)
def randint_like_low(self, low, high, *, dtype=None, device=None, **kwargs):
    return aten.randint.low(
        low,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    )


@register_decomposition(aten.randint.default)
def randint(high, size, **kwargs):
    return aten.randint.low(0, high, size, **kwargs)


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


@register_decomposition(aten._foreach_addcmul.Scalar)
def _foreach_addcmul_scalar(self, left_tensors, right_tensors, scalar=1):
    return aten._foreach_add.List(
        self, aten._foreach_mul.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_addcdiv.Scalar)
def _foreach_addcdiv_scalar(self, left_tensors, right_tensors, scalar=1):
    return aten._foreach_add.List(
        self, aten._foreach_div.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(start_tensors, end_tensors, weight):
    return aten._foreach_add.List(
        start_tensors,
        aten._foreach_mul.Scalar(
            aten._foreach_sub.List(end_tensors, start_tensors), weight
        ),
    )


@aten.miopen_batch_norm.default.py_impl(torch._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    running_mean: typing.Optional[torch.Tensor],
    running_var: typing.Optional[torch.Tensor],
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

    if training:
        return (a, b, c)
    return (
        a,
        weight.new_zeros((0,)),
        weight.new_zeros((0,)),
    )


@functools.lru_cache(None)
def fast_random_decomps():
    return {**decompositions, **extra_random_decomps}


def select_decomp_table():
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    return fast_random_decomps()
