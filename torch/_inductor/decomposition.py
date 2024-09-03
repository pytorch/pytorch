# mypy: allow-untyped-decorators
import functools
import logging
import math
import sys
import typing
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
from torch._dynamo.utils import counters
from torch._higher_order_ops.out_dtype import out_dtype
from torch._inductor.utils import pad_listlike
from torch._prims_common import (
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    type_to_dtype,
)
from torch.fx.experimental.symbolic_shapes import definitely_true, guard_size_oblivious

from . import config, inductor_prims
from .utils import (
    is_gpu,
    needs_fallback_due_to_atomic_add_limitations,
    use_scatter_fallback,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized
_quantized = torch.ops._quantized
quantized_decomposed = torch.ops.quantized_decomposed

inductor_decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.addmv,
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
        aten._batch_norm_with_update,
        aten._batch_norm_with_update_functional,
        aten._batch_norm_no_update,
        aten.batch_norm_backward,
        aten.native_batch_norm,
        aten.native_group_norm,
        aten.native_layer_norm,
        aten.nll_loss2d_backward,
        aten._softmax,
        aten.sin_,
        aten.sqrt_,
        out_dtype,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.upsample_bilinear2d.vec,
        quantized.linear_dynamic_fp16_unpacked_weight,
        _quantized.wrapped_quantized_linear,
    ]
)
decompositions = {**core_aten_decompositions(), **inductor_decompositions}

# Remove unwanted decompositions included via the core ATen decompositions from
# the Inductor decomp table.
decomps_to_exclude = [
    aten._unsafe_index,
    aten._unsafe_masked_index,
    aten._unsafe_masked_index_put_accumulate,
    aten._scaled_dot_product_flash_attention_for_cpu.default,  # See comments in torch/_decomp/decompositions.py
    aten._softmax_backward_data,
    aten.clamp_max,
    aten.clamp_min,
    aten.glu,  # inductor lowers this directly
    aten.select_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.slice_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.split.Tensor,  # inductor lowers this directly
    aten.squeeze,  # inductor lowers this directly
    aten.sum,  # inductor lowers this directly
    aten.unbind,  # inductor lowers this directly
]

remove_decompositions(decompositions, decomps_to_exclude)


def register_decomposition(
    ops: List[Union[torch._ops.OperatorBase, torch._ops.OpOverloadPacket]]
) -> Callable[..., Any]:
    for op in [ops] if callable(ops) else ops:  # type: ignore[attr-defined]
        if op in decompositions:
            log.warning("duplicate decomp: %s", ops)
    return decomp.register_decomposition(ops, decompositions)


# TODO: for now, inductor doesn't handle asserts
# because the condition is symbol -> tensor in the graph.
@register_decomposition([aten._assert_async.msg])
def assert_async_msg_decomp(tensor: torch.Tensor, msg: str) -> None:
    return


# Following `assert_async_msg_decomp` and implement as non-op.
@register_decomposition([aten._functional_assert_async.msg])
def functional_assert_async_msg_decomp(tensor: torch.Tensor, msg: str) -> None:
    return


@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(
    symbol: torch.SymInt,
    *,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
) -> None:
    return


@register_decomposition([aten.clamp])
@pw_cast_for_opmath
def clamp(
    x: torch.Tensor,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
) -> torch.Tensor:
    if min is not None:
        x = x.clamp_min(min)
    if max is not None:
        x = x.clamp_max(max)
    return x


@register_decomposition([aten.full])
def full(
    size: List[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = kwargs.get("dtype")
    if dtype is None:
        kwargs["dtype"] = type_to_dtype(type(fill_value))
        return torch.full(size, fill_value, **kwargs)
    return NotImplemented


# Not really sure how to put this into the main library.  PrimTorch wants
# empty_permuted to go to the prim, and typically users don't really want
# to decompose to empty_strided (but inductor is OK with it, because we are
# cool with strides and everything goes to empty_strided)
@register_decomposition([aten.empty_permuted.default])
def empty_permuted(
    size: List[Union[int, torch.SymInt]],
    physical_layout: List[int],
    **kwargs: Any,
) -> torch.Tensor:
    perm = [0] * len(size)
    for p, l in enumerate(physical_layout):
        perm[l] = p
    return torch.empty([size[l] for l in physical_layout], **kwargs).permute(perm)


@register_decomposition([aten.convolution_backward])
def convolution_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias_sizes: List[int],
    stride: Union[int, List[int]],
    padding: Union[int, List[int]],
    dilation: Union[int, List[int]],
    transposed: bool,
    output_padding: List[int],
    groups: int,
    output_mask: List[bool],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not output_mask[2] or not is_gpu(grad_output.device.type):
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


@register_decomposition([aten.round.decimals])
def round_dec(x: torch.Tensor, decimals: int = 0) -> torch.Tensor:
    ten_pow_decimals = 10.0**decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)


@register_decomposition([aten.bmm])
@pw_cast_for_opmath
def bmm(
    self: torch.Tensor,
    batch2: torch.Tensor,
) -> torch.Tensor:
    if config.coordinate_descent_tuning:
        if guard_size_oblivious(self.shape[1] == 1) or guard_size_oblivious(
            batch2.shape[2] == 1
        ):
            out = (self.unsqueeze(-1) * batch2.unsqueeze(1)).sum(dim=2)
            return out
    if self.device.type == "cpu":
        if guard_size_oblivious(self.size(1) == 1) and guard_size_oblivious(
            batch2.size(-1) == 1
        ):
            counters["inductor"]["decompose_bmm"] += 1
            return torch.sum(
                self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True
            ).unsqueeze(1)
    return NotImplemented


@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(
    self: torch.Tensor,
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    beta: torch.types.Number = 1,
    alpha: torch.types.Number = 1,
) -> torch.Tensor:
    if self.device.type == "cpu":
        if guard_size_oblivious(mat1.size(0) == 1) and guard_size_oblivious(
            mat2.size(-1) == 1
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = torch.sum(
                mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
            return alpha * out + beta * self
        if (
            guard_size_oblivious(mat1.size(0) == 1)
            and definitely_true(mat2.size(0) <= 16)
            and definitely_true(mat2.size(1) <= 16)
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = (mat1.T * mat2).sum(dim=0, keepdim=True)
            return alpha * out + beta * self
    return NotImplemented


@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(
    self: torch.Tensor,
    input2: torch.Tensor,
) -> torch.Tensor:
    # Our matrix vector multiplies only achieve peak bandwidth with coordinate descent tuning.
    # todo: Look into why and fix it (hopefully)
    if config.coordinate_descent_tuning:
        if guard_size_oblivious(self.shape[0] == 1) or guard_size_oblivious(
            input2.shape[1] == 1
        ):
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    if self.device.type == "cpu":
        if (
            guard_size_oblivious(self.size(-1) == 1)
            and guard_size_oblivious(self.size(0) > 0)
            and guard_size_oblivious(input2.size(0) == 1)
            and (self.dtype == input2.dtype)
            and definitely_true((torch.numel(self) + torch.numel(input2)) <= 32)
        ):
            counters["inductor"]["decompose_mm"] += 1
            return torch.cat([self[i, :] * input2 for i in range(self.size(0))])
        if guard_size_oblivious(self.size(0) == 1) and guard_size_oblivious(
            input2.size(-1) == 1
        ):
            counters["inductor"]["decompose_mm"] += 1
            return torch.sum(
                self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
    return NotImplemented


# This pass does two things:
# - Eliminate cat when there is only one tensor input
# - Normalize cat calls, so that legacy empty 1-D tensors are removed (NB: we
#   don't remove ALL empty tensors, only the naughty ones)
@register_decomposition([aten.cat.default])
def cat(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

    def non_empty_tensor(x: torch.Tensor) -> bool:
        # For better or worse, this is a valid cat:
        #
        #   torch.cat([torch.randn(2, 2, 4), torch.randn(0), torch.randn(3, 2, 4)])
        #
        # We'd like to eliminate naughtiness like this for downstream passes
        # like split_cat.  The easiest way is to just drop such inputs
        # (guarding that they are non-zero).
        #
        # Is it permissible for this filtering to be size-oblivious?  A case
        # where this could matter is cat([(2, 2), (u0,)], dim=0); if u0
        # happened to be zero, we would have liked to have filtered it out.
        # But actually, the ONLY way this could have passed is if u0 == 0,
        # so by the time we get here we have already installed a deferred
        # runtime assert forcing u0 to be zero.  So if this hasn't happened,
        # we know that the unbacked SymInt has appropriate size and there are
        # no problems.
        if len(x.shape) == 1 and guard_size_oblivious(x.shape[0] == 0):
            return False

        if dim < len(x.shape) and guard_size_oblivious(x.shape[dim] == 0):
            return False

        return True

    filtered_tensors = list(filter(non_empty_tensor, tensors))

    if len(filtered_tensors) == 1:
        return filtered_tensors[0].clone()
    elif 1 < len(filtered_tensors) < len(tensors):
        # on the first call, when we remove empty tensors, we redispatch recursively
        return aten.cat.default(filtered_tensors, dim)

    # optimization, avoid concat for single, repeated input
    if len(filtered_tensors) > 1 and all(
        t is filtered_tensors[0] for t in filtered_tensors
    ):
        inp = filtered_tensors[0]
        shape = list(inp.shape)
        dim = dim + len(inp.shape) if dim < 0 else dim
        shape.insert(dim, len(filtered_tensors))
        return inp.unsqueeze(dim).expand(*shape).flatten(dim, dim + 1).clone()

    # when no 'filtering' has occurred, we raise to prevent infinite recursion (no more decomposition needed)
    return NotImplemented


@register_decomposition([aten.angle])
def angle(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return torch.where(
            torch.isnan(x.real), float("nan"), torch.atan2(x.imag, x.real)
        )

    # when x is real number
    #   if x >= 0, return 0
    #   if x < 0, return pi
    #   if x is nan, return nan
    _, dtype = elementwise_dtypes(
        x,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    pi = torch.scalar_tensor(math.pi, dtype=dtype, device=x.device)
    ret = torch.where(x < 0, pi, 0.0)
    return torch.where(torch.isnan(x), float("nan"), ret)


@register_decomposition([aten.add])
def add(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: Optional[torch.types.Number] = None,
) -> torch.Tensor:
    # Require both x and y to be complex tensors.
    x_is_complex_tensor = torch.is_tensor(x) and x.is_complex()
    y_is_complex_tensor = torch.is_tensor(y) and y.is_complex()
    if not x_is_complex_tensor or not y_is_complex_tensor:
        return NotImplemented
    z = y
    if alpha is not None:
        z = alpha * y
    complex_type = torch.promote_types(x.dtype, y.dtype)

    # For complex typed `x`, `x.view(x.real.dtype)` doubles the last dimension and can cause problem
    # when broadcasting the add.
    def reshape_tensor_complex(tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from [*initial_dims, last_dim] to *initial_dims, last_dim/2, 2]"""
        # Get the current shape of the tensor
        *initial_dims, last_dim = tensor.shape

        # Check if the last dimension is even. We should never reach here since `x.view(x.real.dtype)`
        # doubles the last dimension for complex numbers.
        if last_dim % 2 != 0:
            raise AssertionError(
                "The size of the last dimension must be even to reshape it to [..., last_dim/2, 2]"
            )

        # Reshape the tensor
        new_shape = (*initial_dims, last_dim // 2, 2)
        reshaped_tensor = tensor.view(new_shape)
        return reshaped_tensor

    x_reshaped = reshape_tensor_complex(x.view(x.real.dtype))
    z_reshaped = reshape_tensor_complex(z.view(y.real.dtype))
    result = torch.flatten(x_reshaped + z_reshaped, start_dim=-2).view(complex_type)
    return result


@register_decomposition([aten.conj_physical])
def conj_physical(self: torch.Tensor) -> torch.Tensor:
    assert not self.is_complex(), "TODO: implement this"
    return self


@register_decomposition([aten.lift, aten.detach_])
def lift(self: torch.Tensor) -> torch.Tensor:
    return self


@register_decomposition([aten.bernoulli.default])
def bernoulli(
    self: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    assert generator is None
    return (torch.rand_like(self, dtype=torch.float32) < self).to(self.dtype)


@register_decomposition([aten.fmin, prims.fmin])
def fmin(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(other) | (other > self), self, other)


@register_decomposition([aten.fmax, prims.fmax])
def fmax(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(other) | (other < self), self, other)


@register_decomposition(aten.amax)
def amax(
    self: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if self.dtype == torch.bool:
        return torch.any(self, dim=dim, keepdim=keepdim)
    return NotImplemented


@register_decomposition(aten.amin)
def amin(
    self: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if self.dtype == torch.bool:
        return torch.all(self, dim=dim, keepdim=keepdim)
    return NotImplemented


@register_decomposition([aten.narrow_copy])
def narrow_copy(
    self: torch.Tensor,
    dim: int,
    start: int,
    length: int,
) -> torch.Tensor:
    return torch.narrow(self, dim, start, length).clone()


@register_decomposition([aten.view_copy.default])
def view_copy_default(
    self: torch.Tensor,
    size: List[Union[int, torch.SymInt]],
) -> torch.Tensor:
    return aten.view(self, size).clone()


@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(
    self: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return self.to(dtype).clone()


def get_like_layout(
    tensor: torch.Tensor,
    memory_format: Optional[torch.memory_format] = None,
) -> torch.memory_format:
    # TODO: _to_copy tensor to stride permutation
    if memory_format is torch.preserve_format or memory_format is None:
        return utils.suggest_memory_format(tensor)
    else:
        return memory_format


@register_decomposition(aten.rand_like)
def rand_like(
    self: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    memory_format: Optional[torch.memory_format] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return torch.rand(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_decomposition(aten.randn_like)
def randn_like(
    self: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    memory_format: Optional[torch.memory_format] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return torch.randn(
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_decomposition(aten.full_like)
def full_like(
    self: torch.Tensor,
    fill_value: Union[int, float],
    *,
    dtype: Optional[torch.dtype] = None,
    layout: Optional[torch.layout] = None,
    device: Optional[torch.device] = None,
    pin_memory: bool = False,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.preserve_format,
) -> torch.Tensor:
    return torch.full(
        [*self.size()],
        fill_value,
        dtype=dtype or self.dtype,
        layout=layout or self.layout,
        device=device or self.device,
        requires_grad=requires_grad,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_decomposition(aten.randint_like.default)
def randint_like(
    self: torch.Tensor,
    high: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    memory_format: Optional[torch.memory_format] = None,
    **kwargs: Any,
) -> torch.Tensor:
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
    self: torch.Tensor,
    low: int,
    high: int,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    memory_format: Optional[torch.memory_format] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return aten.randint.low(
        low,
        high,
        [*self.size()],
        dtype=dtype or self.dtype,
        device=device or self.device,
        **kwargs,
    ).to(memory_format=get_like_layout(self, memory_format))


@register_decomposition(aten.randint.default)
def randint(
    high: int,
    size: List[Union[int, torch.SymInt]],
    **kwargs: Any,
) -> torch.Tensor:
    return aten.randint.low(0, high, size, **kwargs)


@register_decomposition(quantized.linear_dynamic_fp16_unpacked_weight.default)
def linear_dynamic_fp16_unpacked_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    packed_weight = torch.ops._quantized.wrapped_fbgemm_pack_gemm_matrix_fp16(weight)
    return torch.ops._quantized.wrapped_fbgemm_linear_fp16_weight(
        input, packed_weight, bias, weight.size()[0]
    )


@register_decomposition(_quantized.wrapped_quantized_linear.default)
def wrapped_quantized_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    input_zero_point: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    bias: torch.Tensor,
    out_scale: torch.Tensor,
    out_zero_point: torch.Tensor,
    out_channel: int,
) -> torch.Tensor:
    packed_weight = torch.ops._quantized.wrapped_linear_prepack(
        weight, weight_scale, weight_zero_point, bias
    )
    return torch.ops._quantized.wrapped_quantized_linear_prepacked(
        input,
        input_scale,
        input_zero_point,
        packed_weight,
        out_scale,
        out_zero_point,
        out_channel,
    )


@register_decomposition(torch.ops.quantized.embedding_bag_byte_unpack)
def q_embedding_bag_byte_unpack_decomp(packed: torch.Tensor) -> torch.Tensor:
    def bitcast_u8_to_f32(u8: torch.Tensor) -> torch.Tensor:
        x, y, z, w = (u8[..., n].to(torch.int32) for n in (0, 1, 2, 3))
        if sys.byteorder == "little":
            return (x + (y << 8) + (z << 16) + (w << 24)).view(torch.float32)[..., None]
        else:
            return ((x << 24) + (y << 16) + (z << 8) + w).view(torch.float32)[..., None]

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
def _foreach_addcmul_scalar(
    self: List[torch.Tensor],
    left_tensors: List[torch.Tensor],
    right_tensors: List[torch.Tensor],
    scalar: float = 1,
) -> List[torch.Tensor]:
    return aten._foreach_add.List(
        self, aten._foreach_mul.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_addcdiv.Scalar)
def _foreach_addcdiv_scalar(
    self: List[torch.Tensor],
    left_tensors: List[torch.Tensor],
    right_tensors: List[torch.Tensor],
    scalar: float = 1,
) -> List[torch.Tensor]:
    return aten._foreach_add.List(
        self, aten._foreach_div.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(
    start_tensors: List[torch.Tensor],
    end_tensors: List[torch.Tensor],
    weight: torch.types.Number,
) -> List[torch.Tensor]:
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
def fast_random_decomps() -> Dict[Any, Callable[..., Any]]:
    return {**decompositions, **extra_random_decomps}


# TODO(aakhundov): replace this (and the above) Any by more
# specific type and fix all the cascading mypy errors
def select_decomp_table() -> Dict[Any, Callable[..., Any]]:
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    return fast_random_decomps()


@register_decomposition(aten.masked_scatter)
def masked_scatter(
    self: torch.Tensor,
    mask: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    from .codegen.common import BackendFeature, has_backend_feature

    if has_backend_feature(self.device, BackendFeature.MASKED_SCATTER_WITH_INDEX):
        # This two-step algorithm is the same as eager CUDA, for eager CPU we
        # use a 1-shot serial iteration.
        self, mask = aten.broadcast_tensors([self, mask])
        source_idx = mask.reshape(-1).cumsum(0) - 1
        self_flat, mask_flat, source_flat = (x.flatten() for x in (self, mask, source))
        result = aten._unsafe_masked_index(source_flat, mask_flat, [source_idx], 0)
        return torch.where(mask_flat, result, self_flat).view(self.shape)
    return NotImplemented


@register_decomposition(quantized_decomposed.choose_qparams.tensor)
def choose_qparams_tensor(
    input: torch.Tensor,
    quant_min: int,
    quant_max: int,
    eps: float,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    min_val, max_val = torch.aminmax(input)
    scale = (max_val - min_val) / float(quant_max - quant_min)
    scale = torch.max(scale, torch.Tensor([eps]))
    zero_point = quant_min - torch.round(min_val / scale).to(torch.int)
    zero_point = torch.clamp(zero_point, quant_min, quant_max)
    return scale.to(torch.float64), zero_point.to(torch.int64)


@register_decomposition(aten.put)
def put(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    flattened = self.flatten()
    flattened = torch.index_put(
        flattened, [index], source.reshape(index.shape), accumulate
    )
    return flattened.reshape(self.shape)


@register_decomposition(aten.put_)
def put_(
    self: torch.Tensor,
    index: torch.Tensor,
    source: torch.Tensor,
    accumulate: bool = False,
) -> torch.Tensor:
    out = aten.put(self, index, source, accumulate=accumulate)
    return self.copy_(out)


@register_decomposition(aten._softmax_backward_data.default)
@pw_cast_for_opmath
def _softmax_backward_data(
    grad_output: torch.Tensor,
    output: torch.Tensor,
    dim: int,
    input_dtype: torch.dtype,
) -> torch.Tensor:
    new_grad_output = grad_output * output
    sum_new_grad = torch.sum(new_grad_output, dim=dim, keepdim=True)
    # grad_input = new_grad_output - output * sum_new_grad
    grad_input = inductor_prims.fma(-output, sum_new_grad, new_grad_output)

    # CPU kernel doesn't respect input_dtype, but following check doesn't work for meta tensor
    # if grad_output.device == torch.device("cpu"):
    #     return grad_input.contiguous()

    if grad_output.dtype != input_dtype:
        grad_input = grad_input.to(input_dtype)
    return grad_input.contiguous()


@register_decomposition(aten.index_reduce)
def index_reduce(
    self: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    reduction_type: str,
    *,
    include_self: bool = True,
) -> torch.Tensor:
    if reduction_type == "mean" and not needs_fallback_due_to_atomic_add_limitations(
        self.dtype
    ):
        true_division = self.dtype.is_floating_point or self.dtype.is_complex
        ones = torch.ones_like(src)
        if include_self:
            out = self
            counts = torch.ones_like(self).index_add(dim, index, ones)
        else:
            out = self.index_fill(dim, index, 0)
            counts = torch.zeros_like(self).index_add(dim, index, ones)
            counts = counts.masked_fill(counts < 1, 1)
        out = out.index_add(dim, index, src)
        return out / counts if true_division else out // counts

    if use_scatter_fallback(
        aten.scatter_reduce_.two,
        reduction_type,
        self.dtype,
        src.dtype,
        src.device.type,
        True,
    ):
        return NotImplemented

    repeats = self.shape[dim + 1 :].numel() * self.shape[:dim].numel()
    index_shape = (index.numel(), *self.shape[dim + 1 :], *self.shape[:dim])
    perm = (*range(self.ndim - dim, self.ndim), 0, *range(1, self.ndim - dim))
    scatter_index = (
        index.to(torch.int64)
        .repeat_interleave(repeats)
        .reshape(index_shape)
        .permute(perm)
    )
    return self.scatter_reduce(
        dim,
        scatter_index,
        src,
        reduction_type,
        include_self=include_self,
    )


@register_decomposition(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x: torch.Tensor,
    kernel_size: List[int],
    stride: Optional[Union[int, List[int]]] = None,
    padding: Union[int, List[int]] = 0,
    dilation: Union[int, List[int]] = 1,
    ceil_mode: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dilation == 1:
        dilation = [1, 1]

    if padding == 0:
        padding = [0, 0]

    if not stride:
        stride = kernel_size

    kernel_size = pad_listlike(kernel_size, 2)
    dilation = pad_listlike(dilation, 2)
    padding = pad_listlike(padding, 2)
    stride = pad_listlike(stride, 2)

    window_size = kernel_size[0] * kernel_size[1]
    # We fallback when using non-default dilation or when the window size is too large
    if (
        torch._inductor.lowering.should_fallback_max_pool2d_with_indices(
            kernel_size, dilation
        )
        or window_size > torch.iinfo(torch.int8).max
    ):
        return NotImplemented

    vals, offsets = prims._low_memory_max_pool2d_with_offsets(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    indices = prims._low_memory_max_pool2d_offsets_to_indices(
        offsets,
        kernel_size[1],
        x.size(-1),
        stride,
        padding,
    )
    return vals, indices
