# mypy: allow-untyped-decorators
import functools
import logging
import math
import operator
import sys
import typing
from collections.abc import Callable
from typing import Any, Optional, TypeAlias, TypeVar, Union
from typing_extensions import ParamSpec

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
    _index_add,
    embedding_dense_backward as decomp_embedding_dense_backward,
    pw_cast_for_opmath,
    pw_cast_for_opmath_non_tensor_args,
)
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._dynamo.utils import counters
from torch._environment import is_fbcode
from torch._higher_order_ops.out_dtype import out_dtype
from torch._inductor.utils import pad_listlike
from torch._prims_common import (
    elementwise_dtypes,
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    type_to_dtype,
)
from torch.fx.experimental.symbolic_shapes import guard_or_false, statically_known_true

from . import config, inductor_prims
from .utils import (
    is_gpu,
    needs_fallback_due_to_atomic_add_limitations,
    use_scatter_fallback,
)


_T = TypeVar("_T")
_P = ParamSpec("_P")

_GenericOperator: TypeAlias = Union[
    torch._ops.OperatorBase, torch._ops.OpOverloadPacket
]

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
quantized = torch.ops.quantized
_quantized = torch.ops._quantized
quantized_decomposed = torch.ops.quantized_decomposed

inductor_decompositions = get_decompositions(
    [
        aten._adaptive_avg_pool2d_backward,
        aten.index_select,
        aten.addmv,
        aten.arange,
        aten.bitwise_and_,
        aten.bitwise_or_,
        aten.clamp_min_,
        aten.dist,
        aten.elu,
        aten.empty_like,
        aten.flip,
        aten.gelu,
        aten.hardtanh,
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
        aten.permute_copy,
        aten.rrelu_with_noise_backward,
        aten._softmax,
        aten.sin_,
        aten.sqrt_,
        out_dtype,
        aten._to_copy,
        aten.tril_indices,
        aten.triu_indices,
        aten.unbind_copy.int,
        aten.upsample_bilinear2d.vec,
        quantized.linear_dynamic_fp16_unpacked_weight,
        _quantized.wrapped_quantized_linear,
    ]
)
decompositions = {**core_aten_decompositions(), **inductor_decompositions}

# Remove unwanted decompositions included via the core ATen decompositions from
# the Inductor decomp table.
decomps_to_exclude: list[Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket]] = [
    aten._unsafe_index,
    aten._unsafe_masked_index,
    aten._unsafe_masked_index_put_accumulate,
    aten._scaled_dot_product_flash_attention_for_cpu.default,  # See comments in torch/_decomp/decompositions.py
    aten._softmax_backward_data,
    aten.clamp_max,
    aten.clamp_min,
    aten.embedding_dense_backward,  # we fall back on xpu
    aten.index_add,  # we conditionally call this decomp
    aten.glu,  # inductor lowers this directly
    aten.select_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.slice_scatter,  # need to be in the ATen graph in order for it to work with the re-inplacing pass
    aten.split.Tensor,  # inductor lowers this directly
    aten.squeeze,  # inductor lowers this directly
    aten.sum,  # inductor lowers this directly
    aten.unbind,  # inductor lowers this directly
    aten.baddbmm,  # upcasts to fp32, perf issue
]

remove_decompositions(decompositions, decomps_to_exclude)


def register_decomposition(
    ops: Union[_GenericOperator, list[_GenericOperator]],
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    for op in ops if isinstance(ops, list) else [ops]:
        if op in decompositions:
            log.warning("duplicate decomp: %s", ops)
    return decomp.register_decomposition(ops, decompositions)


@register_decomposition([aten.embedding_dense_backward])
def _embedding_dense_backward(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    num_weights: int,
    padding_idx: int,
    scale_grad_by_freq: bool,
) -> torch.Tensor:
    # TODO: check if XE4 still need this fallback
    # check torch.xpu.get_device_properties(grad_output.device).architecture
    if grad_output.is_xpu:
        return NotImplemented
    # We can write a util function to update decomp table if we have more ops to fallback.
    return decomp_embedding_dense_backward(
        grad_output, indices, num_weights, padding_idx, scale_grad_by_freq
    )


@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(
    symbol: torch.SymInt,
    *,
    min: Optional[torch.types.Number] = None,
    max: Optional[torch.types.Number] = None,
) -> None:
    return


@register_decomposition([aten.clamp])
@pw_cast_for_opmath_non_tensor_args
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
    size: list[Union[int, torch.SymInt]],
    fill_value: torch.types.Number,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = kwargs.get("dtype")
    if dtype is None:
        kwargs["dtype"] = type_to_dtype(type(fill_value))
        return torch.full(size, fill_value, **kwargs)
    return NotImplemented


@register_decomposition([aten.index_add])
def index_add(
    x: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    tensor: torch.Tensor,
    *,
    alpha: torch.types.Number = 1,
) -> torch.Tensor:
    # If we are not in fbcode and dtype is bfloat16
    # fallback to index_add kernel
    # see https://github.com/pytorch/pytorch/issues/137425 for details
    if not is_fbcode() and x.dtype == torch.bfloat16:
        return NotImplemented
    else:
        return _index_add(x, dim, index, tensor, inplace=False, alpha=alpha)


# Not really sure how to put this into the main library.  PrimTorch wants
# empty_permuted to go to the prim, and typically users don't really want
# to decompose to empty_strided (but inductor is OK with it, because we are
# cool with strides and everything goes to empty_strided)
@register_decomposition([aten.empty_permuted.default])
def empty_permuted(
    size: list[Union[int, torch.SymInt]],
    physical_layout: list[int],
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
    bias_sizes: list[int],
    stride: Union[int, list[int]],
    padding: Union[int, list[int]],
    dilation: Union[int, list[int]],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # TODO: Re-enable for mps once our reductions are performant enough
    # (https://github.com/pytorch/pytorch/issues/150121)
    if config.coordinate_descent_tuning and self.device.type not in ["cpu", "mps"]:
        if statically_known_true(self.shape[1] == 1) or statically_known_true(
            batch2.shape[2] == 1
        ):
            out = (self.unsqueeze(-1) * batch2.unsqueeze(1)).sum(dim=2)
            return out
    if self.device.type == "cpu":
        if statically_known_true(self.size(1) == 1) and statically_known_true(
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
    out_dtype: Optional[torch.dtype] = None,
    beta: torch.types.Number = 1,
    alpha: torch.types.Number = 1,
) -> torch.Tensor:
    if self.device.type == "cpu":
        if statically_known_true(mat1.size(0) == 1) and statically_known_true(
            mat2.size(-1) == 1
        ):
            counters["inductor"]["decompose_addmm"] += 1
            out = torch.sum(
                mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True
            ).unsqueeze(0)
            return alpha * out + beta * self
        if (
            statically_known_true(mat1.size(0) == 1)
            and guard_or_false(mat2.size(0) <= 16)
            and guard_or_false(mat2.size(1) <= 16)
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
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Our matrix vector multiplies only achieve peak bandwidth with coordinate descent tuning.
    # todo: Look into why and fix it (hopefully)

    # TODO: Re-enable for mps once our reductions are performant enough
    # (https://github.com/pytorch/pytorch/issues/150121)
    if config.coordinate_descent_tuning and self.device.type not in ["cpu", "mps"]:
        if statically_known_true(self.shape[0] == 1) or statically_known_true(
            input2.shape[1] == 1
        ):
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    if self.device.type == "cpu":
        if (
            statically_known_true(self.size(-1) == 1)
            and statically_known_true(self.size(0) > 0)
            and statically_known_true(input2.size(0) == 1)
            and (self.dtype == input2.dtype)
            and guard_or_false((torch.numel(self) + torch.numel(input2)) <= 32)
        ):
            counters["inductor"]["decompose_mm"] += 1
            return self * input2
        if statically_known_true(self.size(0) == 1) and statically_known_true(
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
    tensors: list[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
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
        if len(x.shape) == 1 and guard_or_false(x.shape[0] == 0):
            return False

        if dim < len(x.shape) and guard_or_false(x.shape[dim] == 0):
            return False

        return True

    filtered_tensors = list(filter(non_empty_tensor, tensors))

    if len(filtered_tensors) == 1:
        # check dtype promotion
        promoted_dtype = elementwise_dtypes(
            *tensors,
            type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )[1]
        filtered_t = filtered_tensors[0]
        return (
            filtered_t.clone()
            if promoted_dtype == filtered_t.dtype
            else filtered_t.to(dtype=promoted_dtype)
        )
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

    def _requires_fallback(tensor: torch.Tensor) -> bool:
        if tensor.ndim == 0:
            return False
        # Viewing complex tensors as their real dtype requires the last stride to be 1.
        return tensor.stride()[-1] != 1

    output_size_zero = False
    if x.ndim == 0 and y.ndim == 0:
        output_size_zero = True

    if x.ndim == 0:
        x = x.reshape(1)
    if y.ndim == 0:
        y = y.reshape(1)

    z = y
    if alpha is not None:
        z = alpha * y
    complex_type = torch.promote_types(x.dtype, y.dtype)

    if _requires_fallback(x) or _requires_fallback(z):
        return NotImplemented

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

    # Manually resolve complex tensors, as .is_conj() is unreliable after cloning during compilation.
    x = x + 0
    z = z + 0

    x_reshaped = reshape_tensor_complex(x.view(x.real.dtype))
    z_reshaped = reshape_tensor_complex(z.view(y.real.dtype))
    result = torch.flatten(x_reshaped + z_reshaped, start_dim=-2).view(complex_type)

    if output_size_zero:
        return result[0]
    return result


@register_decomposition([aten.conj_physical])
def conj_physical(self: torch.Tensor) -> torch.Tensor:
    if self.is_complex():
        return NotImplemented
    return self


@register_decomposition([aten.lift, aten.detach_])
def lift(self: torch.Tensor) -> torch.Tensor:
    return self


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
    size: list[Union[int, torch.SymInt]],
) -> torch.Tensor:
    return aten.view(self, size).clone()


@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(
    self: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return self.to(dtype).clone()


def _get_shape_permutation_like(
    self: torch.Tensor,
) -> tuple[utils.ShapeType, utils.StrideType]:
    physical_layout, _ = utils.compute_elementwise_output_logical_to_physical_perm(self)
    shape = [self.shape[l] for l in physical_layout]

    permutation = [0] * len(shape)
    for p, l in enumerate(physical_layout):
        permutation[l] = p

    return (shape, permutation)


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
    dtype = self.dtype if dtype is None else dtype
    layout = self.layout if layout is None else layout
    device = self.device if device is None else device

    if memory_format != torch.preserve_format:
        result = torch.full(
            self.shape,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        return result.to(memory_format=memory_format)

    else:
        assert layout == torch.strided
        shape, permutation = _get_shape_permutation_like(self)
        result = torch.full(
            shape,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        if permutation == list(range(len(permutation))):
            return result
        return result.permute(permutation).clone()


def _rand_like(
    rand_fn: Callable[..., torch.Tensor],
    self: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    memory_format: torch.memory_format = torch.preserve_format,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = self.dtype if dtype is None else dtype
    device = self.device if device is None else device

    if memory_format != torch.preserve_format:
        return rand_fn(
            self.shape,
            dtype=dtype,
            device=device,
            **kwargs,
        ).to(memory_format=memory_format)

    shape, permutation = _get_shape_permutation_like(self)
    result = rand_fn(
        shape,
        dtype=dtype,
        device=device,
        **kwargs,
    )
    if permutation == list(range(len(permutation))):
        return result
    return result.permute(permutation).clone()


@register_decomposition(aten.rand_like)
def rand_like(self: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return _rand_like(torch.rand, self, **kwargs)


@register_decomposition(aten.randn_like)
def randn_like(self: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return _rand_like(torch.randn, self, **kwargs)


@register_decomposition(aten.randint_like.default)
def randint_like(self: torch.Tensor, high: int, **kwargs: Any) -> torch.Tensor:
    return _rand_like(functools.partial(aten.randint.low, 0, high), self, **kwargs)


@register_decomposition(aten.randint_like.low_dtype)
def randint_like_low(
    self: torch.Tensor, low: int, high: int, **kwargs: Any
) -> torch.Tensor:
    return _rand_like(functools.partial(aten.randint.low, low, high), self, **kwargs)


@register_decomposition(aten.randint.default)
def randint(
    high: int,
    size: list[Union[int, torch.SymInt]],
    **kwargs: Any,
) -> torch.Tensor:
    return aten.randint.low(0, high, size, **kwargs)


@register_decomposition(quantized.linear_dynamic_fp16_unpacked_weight.default)
def linear_dynamic_fp16_unpacked_weight(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
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
    packed_weight = torch.ops._quantized._wrapped_linear_prepack(
        weight, weight_scale, weight_zero_point, bias
    )
    return torch.ops._quantized._wrapped_quantized_linear_prepacked(
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
    self: list[torch.Tensor],
    left_tensors: list[torch.Tensor],
    right_tensors: list[torch.Tensor],
    scalar: float = 1,
) -> list[torch.Tensor]:
    return aten._foreach_add.List(
        self, aten._foreach_mul.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_addcdiv.Scalar)
def _foreach_addcdiv_scalar(
    self: list[torch.Tensor],
    left_tensors: list[torch.Tensor],
    right_tensors: list[torch.Tensor],
    scalar: float = 1,
) -> list[torch.Tensor]:
    return aten._foreach_add.List(
        self, aten._foreach_div.List(left_tensors, right_tensors), alpha=scalar
    )


@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(
    start_tensors: list[torch.Tensor],
    end_tensors: list[torch.Tensor],
    weight: torch.types.Number,
) -> list[torch.Tensor]:
    return aten._foreach_add.List(
        start_tensors,
        aten._foreach_mul.Scalar(
            aten._foreach_sub.List(end_tensors, start_tensors), weight
        ),
    )


@register_decomposition(aten._foreach_lerp.ScalarList)
def _foreach_lerp_scalarlist(
    start_tensors: list[torch.Tensor],
    end_tensors: list[torch.Tensor],
    scalars: list[torch.types.Number],
) -> list[torch.Tensor]:
    return aten._foreach_add.List(
        start_tensors,
        aten._foreach_mul.ScalarList(
            aten._foreach_sub.List(end_tensors, start_tensors), scalars
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


@functools.cache
def fast_random_decomps() -> dict[Any, Callable[..., Any]]:
    return {**decompositions, **extra_random_decomps}


# TODO(aakhundov): replace this (and the above) Any by more
# specific type and fix all the cascading mypy errors
def select_decomp_table() -> dict[Any, Callable[..., Any]]:
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions
    if config.fallback_embedding_bag_byte_unpack:
        # remove q_embedding_bag_byte_unpack_decomp from decompositions
        decompositions.pop(torch.ops.quantized.embedding_bag_byte_unpack.default, None)
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
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _max_pool_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: Optional[Union[int, list[int]]],
    padding: Union[int, list[int]],
    dilation: Union[int, list[int]],
    ceil_mode: bool,
    dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dilation == 1:
        dilation = [1] * dim

    if padding == 0:
        padding = [0] * dim

    if not stride:
        stride = kernel_size

    # pyrefly: ignore [bad-assignment]
    kernel_size = pad_listlike(kernel_size, dim)
    # pyrefly: ignore [bad-assignment]
    dilation = pad_listlike(dilation, dim)
    # pyrefly: ignore [bad-assignment]
    padding = pad_listlike(padding, dim)
    # pyrefly: ignore [bad-assignment]
    stride = pad_listlike(stride, dim)

    window_size = functools.reduce(operator.mul, kernel_size)
    # We fallback when using non-default dilation or when the window size is too large
    if (
        torch._inductor.lowering.should_fallback_max_pool_with_indices(
            kernel_size, n_dim=dim
        )
        or window_size > torch.iinfo(torch.int8).max
    ):
        return NotImplemented

    vals, offsets = prims._low_memory_max_pool_with_offsets(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )
    indices = prims._low_memory_max_pool_offsets_to_indices(
        offsets,
        kernel_size,
        x.shape[-dim:],
        stride,
        padding,
        dilation,
    )
    return vals, indices


@register_decomposition(aten.max_pool2d_with_indices)
def max_pool2d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: Optional[Union[int, list[int]]] = None,
    padding: Union[int, list[int]] = 0,
    dilation: Union[int, list[int]] = 1,
    ceil_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=2
    )


@register_decomposition(aten.max_pool3d_with_indices)
def max_pool3d_with_indices(
    x: torch.Tensor,
    kernel_size: list[int],
    stride: Optional[Union[int, list[int]]] = None,
    padding: Union[int, list[int]] = 0,
    dilation: Union[int, list[int]] = 1,
    ceil_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _max_pool_with_indices(
        x, kernel_size, stride, padding, dilation, ceil_mode, dim=3
    )


@register_decomposition(aten.adaptive_max_pool2d)
def adaptive_max_pool2d(
    x: torch.Tensor, output_size: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    *batch, h_in, w_in = x.shape
    h_out, w_out = output_size

    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return x.new_empty(o_size), x.new_empty(o_size, dtype=torch.int64)

    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [h_in // h_out, w_in // w_out]
        return aten.max_pool2d_with_indices(x, kernel_size)

    return NotImplemented


@register_decomposition(aten.searchsorted.Scalar)
def searchsorted_scalar(
    sorted_sequence: torch.Tensor,
    self: torch.types.Number,
    *,
    out_int32: bool = False,
    right: bool = False,
    side: Optional[str] = None,
    sorter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return aten.searchsorted(
        sorted_sequence,
        torch.tensor([self], device=sorted_sequence.device),
        out_int32=out_int32,
        right=right,
        side=side,
        sorter=sorter,
    )[0]


@register_decomposition(aten.rrelu_with_noise_functional)
def rrelu_with_noise_functional(
    self: torch.Tensor,
    noise: torch.Tensor,
    lower: float = 0.125,
    upper: float = 0.3333333333333333,
    training: bool = False,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if training:
        not_positive = self <= 0
        r = aten.uniform(self, lower, upper, generator=generator)
        output = torch.where(not_positive, self * r, self)
        noise_out = torch.where(not_positive, r, 1)
        return output, noise_out
    else:
        negative_slope = (lower + upper) / 2
        return aten.leaky_relu(self, negative_slope), torch.Tensor()


@register_decomposition(aten.repeat_interleave.Tensor)
def repeat_interleave_Tensor(
    repeat: torch.Tensor,
    output_size: Optional[int] = None,
) -> torch.Tensor:
    if config.triton.autotune_at_compile_time:
        # We can't compile-time auto-tune this because
        # it expects specific data in `repeat`
        return NotImplemented
    if output_size is None or type(output_size) is not int:
        return NotImplemented
    if repeat.device.type == "mps":
        return NotImplemented
    assert repeat.dtype in [torch.int32, torch.int64]
    assert repeat.ndim == 1
    cumsum = repeat.cumsum(0)
    pos = torch.arange(output_size, device=repeat.device)
    indices = torch.searchsorted(
        cumsum, pos, out_int32=(repeat.dtype == torch.int32), right=True
    )
    return torch.clamp(indices, max=repeat.size(0) - 1)


# intentionally not regiestered
def conv1d_to_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: tuple[int] = (1,),
    padding: tuple[int] = (0,),
    dilation: tuple[int] = (1,),
    groups: int = 1,
) -> torch.Tensor:
    # Shapes:
    # input:  (N, C_in, L_in)
    # weight: (C_out, C_in // groups, K)
    # bias:   (C_out,)
    assert input.dim() == 3 and weight.dim() == 3, (
        "Expect (N,C_in,L) and (C_out,C_in//groups,K)"
    )

    # pyrefly: ignore [bad-assignment]
    stride = stride[0]
    # pyrefly: ignore [bad-assignment]
    padding = padding[0]
    # pyrefly: ignore [bad-assignment]
    dilation = dilation[0]

    # Unsqueeze to make input 2D: (N,C,L) -> (N,C,L,1)
    input_2d = input.unsqueeze(-1)
    # Unsqueeze kernel: (C_out,C_in/groups,K) -> (C_out,C_in/groups,K,1)
    weight_2d = weight.unsqueeze(-1)

    # Call conv2d with adjusted args
    out_2d = aten.conv2d.default(
        input_2d,
        weight_2d,
        bias,
        stride=(stride, 1),
        padding=(padding, 0),
        dilation=(dilation, 1),
        groups=groups,
    )

    # Squeeze dummy dimension back out: (N,C_out,L_out,1) -> (N,C_out,L_out)
    return out_2d.squeeze(-1)
