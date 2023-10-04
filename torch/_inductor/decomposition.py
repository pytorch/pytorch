import functools
import logging
import math
import typing
from typing import Optional

import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
    core_aten_decompositions,
    get_decompositions,
    remove_decompositions,
)
from torch._decomp.decompositions import (
    _grid_sampler_2d as decomp_grid_sampler_2d,
    pw_cast_for_opmath,
)
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype

from . import config, inductor_prims

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
        out_dtype,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.upsample_bilinear2d.vec,
    ]
)
decompositions = {**core_aten_decompositions(), **inductor_decompositions}

# Remove unwanted decompositions included via the core ATen decompositions from
# the Inductor decomp table.
decomps_to_exclude = [
    aten._unsafe_index,
    aten._scaled_dot_product_flash_attention.default,  # See comments in torch/_decomp/decompositions.py
    aten.clamp_max,
    aten.clamp_min,
    aten.glu,  # inductor lowers this directly
    aten.split.Tensor,  # inductor lowers this directly
    aten.unbind,  # inductor lowers this directly
]

remove_decompositions(decompositions, decomps_to_exclude)


def register_decomposition(ops):
    for op in [ops] if callable(ops) else ops:
        if op in decompositions:
            log.warning("duplicate decomp: %s", ops)
    return decomp.register_decomposition(ops, decompositions)


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


@register_decomposition([aten.full])
def full(size, fill_value, **kwargs):
    dtype = kwargs.get("dtype")
    if dtype is None:
        kwargs["dtype"] = type_to_dtype(type(fill_value))
        return aten.full(size, fill_value, **kwargs)
    return NotImplemented


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


@register_decomposition([aten.bmm])
@pw_cast_for_opmath
def bmm(self, batch2):
    if config.coordinate_descent_tuning:
        if self.shape[1] == 1:
            out = (self.unsqueeze(-1) * batch2.unsqueeze(1)).sum(dim=2)
            return out
    if self.device.type == "cpu":
        if self.size(1) == 1 and batch2.size(-1) == 1:
            return torch.sum(
                self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True
            ).unsqueeze(1)
    return NotImplemented


@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(self, mat1, mat2, beta=1, alpha=1):
    if self.device.type == "cpu":
        if mat1.size(0) == 1 and mat2.size(-1) == 1:
            out = torch.sum(
                mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
            return alpha * out + beta * self
        if mat1.size(0) == 1 and mat2.size(0) <= 16 and mat2.size(1) <= 16:
            out = (mat1.T * mat2).sum(dim=0, keepdim=True)
            return alpha * out + beta * self
    return NotImplemented


@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(self, input2):
    # Our matrix vector multiplies only achieve peak bandwidth with coordinate descent tuning.
    # todo: Look into why and fix it (hopefully)
    if config.coordinate_descent_tuning:
        if self.shape[0] == 1 or input2.shape[1] == 1:
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    if self.device.type == "cpu":
        if (
            self.size(-1) == 1
            and self.size(0) > 0
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


def get_like_layout(
    tensor: torch.Tensor, memory_format: Optional[torch.memory_format]
) -> torch.memory_format:
    # TODO: _to_copy tensor to stride permutation
    if memory_format in (torch.preserve_format, None):
        return utils.suggest_memory_format(tensor)
    else:
        return memory_format


@register_decomposition(aten.rand_like)
def rand_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    return torch.rand(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_decomposition(aten.randn_like)
def randn_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    return torch.randn(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


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
        requires_grad=requires_grad,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_decomposition(aten.randint_like.default)
def randint_like(self, high, *, dtype=None, device=None, memory_format=None, **kwargs):
    return aten.randint.low(
        0,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_decomposition(aten.randint_like.low_dtype)
def randint_like_low(
    self, low, high, *, dtype=None, device=None, memory_format=None, **kwargs
):
    return aten.randint.low(
        low,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


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


@register_decomposition(torch.ops.quantized.embedding_bag_byte_unpack)
def q_embedding_bag_byte_unpack_decomp(packed):
    def bitcast_u8_to_f32(u8):
        x, y, z, w = (u8[..., n].to(torch.int32) for n in (0, 1, 2, 3))
        return (x + (y << 8) + (z << 16) + (w << 24)).view(torch.float32)[..., None]

    scales = bitcast_u8_to_f32(packed[..., -8:-4])
    offsets = bitcast_u8_to_f32(packed[..., -4:])
    return packed[..., :-8].to(torch.float32) * scales + offsets


@register_decomposition([aten.grid_sampler_2d])
@pw_cast_for_opmath
def grid_sampler_2d(
    a: torch.Tensor,
    grid: torch.Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> torch.Tensor:
    # We do not expand the grid (_expand_grid=False) on cpu for performance reasons
    # Experimenting locally it was found that compiled CUDA code is accelerated by ~5x
    # and CPU code by ~2x on bicubic mode, if we expand the grid from (N, H, W, 2) into (N, C, H, W, 2)
    # However, this leads to a slowdown around ~0.8x on CPU bilinear mode, channels first.
    # Thus we apply this hack to not expand the grid for this case.
    _expand_grid = not (
        a.device == torch.device("cpu")
        and interpolation_mode == 0
        and a.is_contiguous(memory_format=torch.contiguous_format)
    )

    output = decomp_grid_sampler_2d(
        a,
        grid=grid,
        interpolation_mode=interpolation_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
        _expand_grid=_expand_grid,
    )
    return output


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


@register_decomposition(aten.masked_scatter)
def masked_scatter(self, mask, source):
    if self.device.type == "cuda":
        # This two-step algorithm is the same as eager CUDA, for eager CPU we
        # use a 1-shot serial iteration.
        self, mask = aten.broadcast_tensors([self, mask])
        source_idx = mask.reshape(-1).cumsum(0) - 1
        return inductor_prims.masked_scatter_with_index(self, mask, source_idx, source)
    return NotImplemented
