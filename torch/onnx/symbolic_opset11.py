# mypy: allow-untyped-defs
# mypy: disable-error-code=arg-type
"""This file exports ONNX ops for opset 11."""

from __future__ import annotations

import functools
import sys
import warnings
from typing import TYPE_CHECKING

import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import (
    _type_utils,
    errors,
    symbolic_helper,
    symbolic_opset10 as opset10,
    symbolic_opset9 as opset9,
    utils,
)
from torch.onnx._internal import jit_utils, registration


if TYPE_CHECKING:
    from collections.abc import Sequence


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = [
    "add",
    "append",
    "arange",
    "argsort",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "cat",
    "chunk",
    "clamp_max",
    "clamp_min",
    "clamp",
    "constant_pad_nd",
    "cumsum",
    "Delete",
    "embedding_bag",
    "embedding_renorm",
    "flatten",
    "gather",
    "hardtanh",
    "hstack",
    "im2col",
    "index_fill",
    "index",
    "index_copy",
    "index_put",
    "insert",
    "linalg_det",
    "linalg_vector_norm",
    "logdet",
    "masked_scatter",
    "masked_select",
    "mm",
    "narrow",
    "normal",
    "pad",
    "pixel_shuffle",
    "pop",
    "prim_constant_chunk",
    "reflection_pad",
    "relu6",
    "remainder",
    "replication_pad",
    "round",
    "scatter",
    "select",
    "size",
    "sort",
    "split_with_sizes",
    "split",
    "squeeze",
    "stack",
    "topk",
    "unbind",
    "unique_dim",
    "unsqueeze",
    "vstack",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=11)


@_onnx_symbolic("aten::hardtanh")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "f")
def hardtanh(g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float):
    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.FLOAT
    )
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(min_val, dtype=scalar_type.dtype()),
    )
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(max_val, dtype=scalar_type.dtype()),
    )
    return symbolic_helper._op_with_optional_float_cast(
        g, "Clip", self, min_val, max_val, opset_before=12
    )


@_onnx_symbolic("aten::clamp")
def clamp(g: jit_utils.GraphContext, self, min, max):
    def _cast_if_not_none(tensor, dtype):
        if tensor is not None and not symbolic_helper._is_none(tensor):
            return g.op(
                "Cast",
                tensor,
                to_i=dtype.onnx_type(),
            )
        else:
            return tensor

    scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        min = _cast_if_not_none(min, scalar_type)
        max = _cast_if_not_none(max, scalar_type)

    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    elif symbolic_helper._is_none(max):
        return clamp_min(g, self, min)
    else:
        if (
            symbolic_helper._get_tensor_rank(min) == 0
            and symbolic_helper._get_tensor_rank(max) == 0
        ):
            return symbolic_helper._op_with_optional_float_cast(
                g, "Clip", self, min, max, opset_before=12
            )
        else:
            return clamp_max(g, clamp_min(g, self, min), max)


@_onnx_symbolic("aten::clamp_min")
@symbolic_helper.parse_args("v", "v")
def clamp_min(g: jit_utils.GraphContext, self, min):
    min = g.op("Cast", min, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    if symbolic_helper._get_tensor_rank(min) == 0:
        max = opset9.unused(g)
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        return symbolic_helper._op_with_optional_float_cast(
            g, "Max", self, min, opset_before=12
        )


@_onnx_symbolic("aten::clamp_max")
@symbolic_helper.parse_args("v", "v")
def clamp_max(g: jit_utils.GraphContext, self, max):
    max = g.op("Cast", max, to_i=_type_utils.JitScalarType.from_value(self).onnx_type())
    if symbolic_helper._get_tensor_rank(max) == 0:
        min = opset9.unused(g)
        return symbolic_helper._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        return symbolic_helper._op_with_optional_float_cast(
            g, "Min", self, max, opset_before=12
        )


@_onnx_symbolic("aten::relu6")
def relu6(g: jit_utils.GraphContext, input):
    scalar_type = _type_utils.JitScalarType.from_value(
        input, _type_utils.JitScalarType.FLOAT
    )
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(0, dtype=scalar_type.dtype()),
    )
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(6, dtype=scalar_type.dtype()),
    )
    return clamp(g, input, min_val, max_val)


@_onnx_symbolic("aten::select")
# Opset 11 gather accepts negative indices
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
def select(g: jit_utils.GraphContext, self, dim, index):
    return g.op("Gather", self, index, axis_i=dim)


@_onnx_symbolic("aten::index_put")
def index_put(
    g: jit_utils.GraphContext, self, indices_list_value, values, accumulate=False
):
    if symbolic_helper._is_packed_list(indices_list_value):
        indices_list = symbolic_helper._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]
    accumulate = symbolic_helper._parse_arg(accumulate, "b")

    if len(indices_list) == 0:
        return values

    if len(indices_list) > 1:
        for idx_ in range(len(indices_list)):
            if symbolic_helper._is_bool(indices_list[idx_]):
                indices_list[idx_] = g.op("NonZero", indices_list[idx_])
        index = indices_list[0]

        for ind in indices_list[1:]:
            index = opset9.add(g, index, ind)
        broadcast_index_shape = g.op("Shape", index)
        indices_list = [
            symbolic_helper._unsqueeze_helper(
                g, opset9.expand(g, ind, broadcast_index_shape, None), [-1]
            )
            for ind in indices_list
        ]
        index = g.op("Concat", *indices_list, axis_i=-1)
    else:
        # Replace index_put node with masked_scatter or masked_fill
        # when inputs to the index_put node contains a single boolean input.
        #
        # index_put -> masked_fill
        #   * input index contains single tensor of Bool type (e.g.: %24 <- %23).
        #   * input value contains single element (e.g.: %18).
        #
        # Torch IR
        #   %mask : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) = aten::clone(%0, %6)
        #   %16 : Bool(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) =
        #               aten::to(%8, %26, %27, %11, %12, %28, %29, %15)
        #   %18 : Float(requires_grad=0, device=cpu) = prim::Constant[value={1}]()
        #   %23 : Bool(8, strides=[1], device=cpu) = aten::view(%16, %22)
        #   %24 : Tensor?[] = prim::ListConstruct(%23)
        #   %25 : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) =
        #                aten::index_put(%mask, %24, %18, %30)
        #   return (%25)
        #
        #
        # index_put -> masked_scatter
        #   * input index contains single tensor of Bool type (e.g.: %32 <- %31).
        #   * input value contains multiple elements (e.g.: %28).
        #
        # Torch IR
        #   %mask : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu) = aten::clone(%0, %6)
        #   %28 : Float(8, strides=[1], requires_grad=0, device=cpu)
        #                = prim::Constant[value= 1  1  1  1  1  1  1  1 [ CPUFloatType{8} ]]()
        #   %15 : Bool(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu)
        #                = aten::ne(%mask, %some_const)
        #   %23 : Bool(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu)
        #                = aten::to(%15, %34, %35, %18, %19, %36, %37, %22)
        #   %38 : Long(requires_grad=0, device=cpu) = prim::Constant[value={0}]()
        #   %30 : int[] = prim::Constant[value=[-1]]()
        #   %31 : Bool(8, strides=[1], device=cpu) = aten::view(%23, %30)
        #   %32 : Tensor?[] = prim::ListConstruct(%31)
        #   %33 : Float(2, 2, 2, strides=[4, 2, 1], requires_grad=0, device=cpu)
        #               = aten::index_put(%mask, %32, %28, %38)
        #   return (%33)
        index = indices_list[0]
        bool_inp = index
        if symbolic_helper._is_bool(bool_inp):
            rank = symbolic_helper._get_tensor_rank(values)
            if rank is not None and rank == 0:
                return opset9.masked_fill(g, self, bool_inp, values)
            mask_rank = symbolic_helper._get_tensor_rank(bool_inp)
            self_rank = symbolic_helper._get_tensor_rank(self)
            if (
                mask_rank is not None
                and self_rank is not None
                and self_rank > mask_rank
            ):
                # Unsqueeze 'bool_inp' to be broadcastable to shape of 'self'.
                bool_inp = symbolic_helper._unsqueeze_helper(
                    g, bool_inp, list(range(mask_rank, self_rank))
                )
            return masked_scatter(g, self, bool_inp, values)
        broadcast_index_shape = g.op("Shape", index)
        index = symbolic_helper._unsqueeze_helper(g, index, [-1])
    sub_data_shape = symbolic_helper._slice_helper(
        g, g.op("Shape", self), axes=[0], starts=[len(indices_list)], ends=[sys.maxsize]
    )
    values_shape = g.op("Concat", broadcast_index_shape, sub_data_shape, axis_i=0)
    # Check if values is a singular value and expand accordingly
    rank = symbolic_helper._get_tensor_rank(values)
    if rank is not None and rank == 0:
        values = opset9.expand(g, values, values_shape, None)
    values = symbolic_helper._reshape_helper(g, values, values_shape)

    self_scalar_type = _type_utils.JitScalarType.from_value(
        self, _type_utils.JitScalarType.UNDEFINED
    )
    if self_scalar_type != _type_utils.JitScalarType.UNDEFINED:
        values_scalar_type = _type_utils.JitScalarType.from_value(
            values, _type_utils.JitScalarType.UNDEFINED
        )
        if self_scalar_type != values_scalar_type:
            values = g.op("Cast", values, to_i=self_scalar_type.onnx_type())
    elif accumulate:
        raise errors.SymbolicValueError("self does not have a valid scalar type.", self)

    if accumulate:
        zeros = g.op(
            "ConstantOfShape",
            g.op("Shape", self),
            value_t=torch.tensor([0], dtype=self_scalar_type.dtype()),
        )
        result = g.op("ScatterND", zeros, index, values)
        result = add(g, self, result)
    else:
        result = g.op("ScatterND", self, index, values)

    return result


@_onnx_symbolic("aten::pixel_shuffle")
@symbolic_helper.parse_args("v", "i")
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor):
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None and rank != 4:
        return symbolic_helper._unimplemented("pixel_shuffle", "only support 4d input")
    return g.op("DepthToSpace", self, blocksize_i=upscale_factor, mode_s="CRD")


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[symbolic_helper._apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[symbolic_helper._apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[symbolic_helper._apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[symbolic_helper._apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[symbolic_helper._apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[symbolic_helper._apply_params("upsample_trilinear3d", 5, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bicubic2d",
    decorate=[symbolic_helper._apply_params("upsample_bicubic2d", 4, "cubic")],
)
def _interpolate(name: str, dim: int, interpolate_mode: str):
    return symbolic_helper._interpolate_helper(name, dim, interpolate_mode)


@_onnx_symbolic("aten::__interpolate")
@symbolic_helper.quantized_args(True, False, False, False, False, False, False)
def __interpolate(
    g: jit_utils.GraphContext,
    input,
    size,
    scale_factor,
    mode,
    align_corners,
    recompute_scale_factor,
    antialias,
):
    return symbolic_helper.__interpolate_helper(
        g, input, size, scale_factor, mode, align_corners, recompute_scale_factor
    )


@_onnx_symbolic("aten::gather")
@symbolic_helper.parse_args("v", "i", "v", "v")
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=False):
    if symbolic_helper._maybe_get_const(sparse_grad, "i"):
        return symbolic_helper._unimplemented("gather", "sparse_grad == True")
    return g.op("GatherElements", self, index, axis_i=dim)


@_onnx_symbolic("aten::scatter")
@symbolic_helper.parse_args("v", "i", "v", "v")
def scatter(g: jit_utils.GraphContext, self, dim, index, src):
    src_type = _type_utils.JitScalarType.from_value(src)
    src = symbolic_helper._maybe_get_scalar(src)
    if symbolic_helper._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim)
    else:
        # Check if scalar "src" has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        if _type_utils.JitScalarType.from_value(self) != src_type:
            src = g.op(
                "Cast",
                src,
                to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
            )
        return g.op(
            "ScatterElements", self, index, opset9.expand_as(g, src, index), axis_i=dim
        )


@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
def cumsum(g: jit_utils.GraphContext, self, dim, dtype=None):
    dim_tensor = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int))
    if dtype and dtype.node().kind() != "prim::Constant":
        parsed_dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        cast = g.op(
            "Cast", self, to_i=_type_utils.JitScalarType(parsed_dtype).onnx_type()
        )
    else:
        cast = self
    csum = g.op("CumSum", cast, dim_tensor)
    return csum


@_onnx_symbolic("aten::masked_select")
def masked_select(g: jit_utils.GraphContext, self, mask):
    index = opset9.nonzero(g, opset9.expand_as(g, mask, self))
    return g.op("GatherND", self, index)


@_onnx_symbolic("aten::masked_scatter")
def masked_scatter(g: jit_utils.GraphContext, self, mask, source):
    index = opset9.nonzero(g, opset9.expand_as(g, mask, self))
    # NOTE: source can have more elements than needed.
    # It could also have arbitrary shape.
    # This is not supported by ONNX::ScatterND, so we need to flatten and slice source tensor.
    source = symbolic_helper._reshape_helper(g, source, torch.LongTensor([-1]))
    source = symbolic_helper._slice_helper(
        g,
        source,
        axes=torch.LongTensor([0]),
        starts=torch.LongTensor([0]),
        ends=opset9.size(g, index, torch.LongTensor([0])),
    )
    return g.op("ScatterND", self, index, source)


@_onnx_symbolic("aten::len")
def _len(g: jit_utils.GraphContext, self):
    if (
        symbolic_helper._is_tensor_list(self)
        or self.node().kind() == "onnx::SplitToSequence"
    ):
        return g.op("SequenceLength", self)
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return symbolic_helper._squeeze_helper(g, sz_0, [0])


@_onnx_symbolic("aten::__getitem_")
def __getitem_(g: jit_utils.GraphContext, self, i):
    if symbolic_helper._is_tensor_list(self):
        # SequenceAt requires that the input be a List of Tensors
        return g.op("SequenceAt", self, i)
    else:
        from torch.onnx.symbolic_opset9 import __getitem_ as getitem

        return getitem(g, self, i)


@_onnx_symbolic("aten::_set_item")
def _set_item(g: jit_utils.GraphContext, tensor_list, i, v):
    tensor_list = g.op("SequenceErase", tensor_list, i)
    return g.op("SequenceInsert", tensor_list, v, i)


@_onnx_symbolic("aten::append")
def append(g: jit_utils.GraphContext, self, tensor):
    return g.op("SequenceInsert", self, tensor)


@_onnx_symbolic("aten::add")
def add(g: jit_utils.GraphContext, self, other, alpha=None):
    if symbolic_helper._is_value(self) and symbolic_helper._is_tensor_list(self):
        tensor_list_node = other.node()
        if tensor_list_node.kind() != "prim::ListConstruct":
            return symbolic_helper._unimplemented(
                "add", "does not support adding dynamic tensor list to another"
            )
        tensors = symbolic_helper._unpack_list(other)
        l = self
        for t in tensors:
            l = g.op("SequenceInsert", l, t)
        return l

    return opset9.add(g, self, other, alpha)


@_onnx_symbolic("aten::insert")
def insert(g: jit_utils.GraphContext, self, pos, tensor):
    return g.op("SequenceInsert", self, tensor, pos)


@_onnx_symbolic("aten::pop")
def pop(g: jit_utils.GraphContext, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)


@_onnx_symbolic("aten::Delete")
def Delete(g: jit_utils.GraphContext, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)


@_onnx_symbolic("aten::cat")
@symbolic_helper.quantized_args(True)
def cat(g: jit_utils.GraphContext, tensor_list, dim):
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.cat(g, tensor_list, dim)
    else:
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim)


@_onnx_symbolic("aten::stack")
def stack(g: jit_utils.GraphContext, tensor_list, dim):
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.stack(g, tensor_list, dim)
    else:
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim, new_axis_i=1)


@_onnx_symbolic("aten::_unique2")
@symbolic_helper.parse_args("v", "i", "i", "i")
def _unique2(g: jit_utils.GraphContext, self, sorted, return_inverse, return_counts):
    u, _indices, inverse_indices, counts = g.op(
        "Unique", self, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


@_onnx_symbolic("aten::unique_dim")
@symbolic_helper.parse_args("v", "i", "i", "i", "i")
def unique_dim(
    g: jit_utils.GraphContext, self, dim, sorted, return_inverse, return_counts
):
    u, _indices, inverse_indices, counts = g.op(
        "Unique", self, axis_i=dim, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def sort(g: jit_utils.GraphContext, self, dim, descending, out=None):
    return symbolic_helper._sort_helper(g, self, dim, descending=descending, out=out)


@_onnx_symbolic("aten::argsort")
@symbolic_helper.parse_args("v", "i", "i", "none")
def argsort(g: jit_utils.GraphContext, self, dim, descending, out=None):
    _, indices = symbolic_helper._sort_helper(
        g, self, dim, descending=descending, out=out
    )
    return indices


@_onnx_symbolic("aten::round")
@symbolic_helper.parse_args("v", "i")
def round(g: jit_utils.GraphContext, self, decimals=0):
    if not symbolic_helper._is_fp(self):
        return self
    if decimals == 0:
        return g.op("Round", self)
    mul = g.op("Mul", self, g.op("Constant", value_t=torch.tensor(pow(10, decimals))))
    round = g.op("Round", mul)
    return g.op(
        "Mul", round, g.op("Constant", value_t=torch.tensor(pow(10, -1 * decimals)))
    )


@_onnx_symbolic("aten::remainder")
def remainder(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_fp(input) or symbolic_helper._is_fp(other):
        return opset9.remainder(g, input, other)
    return g.op("Mod", input, other, fmod_i=0)


@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split(g: jit_utils.GraphContext, self, split_size_or_sizes, dim, _outputs=None):
    if not symbolic_helper._is_split_static(split_size_or_sizes, _outputs):
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        if _outputs is None:
            return split_out
        # Convert to multiple slice nodes iff number of splits and number of outputs are statically known.
        if (
            symbolic_helper._is_packed_list(split_size_or_sizes)
            and len(symbolic_helper._unpack_list(split_size_or_sizes)) == _outputs
        ):
            split_sizes = [
                symbolic_helper._unsqueeze_helper(g, v, [0])
                for v in symbolic_helper._unpack_list(split_size_or_sizes)
            ]
            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            res = []
            for i in range(_outputs):
                end = g.op(
                    "Add", start, split_sizes[i]
                )  # split_sizes is a list of same length as _outputs
                res.append(g.op("Slice", self, start, end, axis))
                start = end
            return res
        return [
            g.op(
                "SequenceAt",
                split_out,
                g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)),
            )
            for i in range(_outputs)
        ]
    else:
        return opset9.split(g, self, split_size_or_sizes, dim, _outputs)


@_onnx_symbolic("aten::split_with_sizes")
@symbolic_helper.parse_args("v", "v", "i", "i")
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
def unbind(g: jit_utils.GraphContext, self, dim=0, _outputs=None):
    if _outputs is None:
        return g.op(
            "SplitToSequence",
            self,
            g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)),
            axis_i=dim,
            keepdims_i=0,
        )
    else:
        return opset9.unbind(g, self, dim, _outputs)


def _prepare_onnx_paddings(g: jit_utils.GraphContext, input, pad):
    """Generate paddings in ONNX order based on pad in pytorch.

    Args:
        input: the input tensor.
        pad: the paddings in pytorch.
            The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ..., dim_m_begin, dim_m_end,
            where m is in range [0, n].
    """
    if (
        not symbolic_helper._is_packed_list(pad)
        and symbolic_helper._is_list(pad)
        and symbolic_helper._is_scalar_list(pad)
    ):
        pad = g.op("ConcatFromSequence", pad, axis_i=0, new_axis_i=1)
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # Assume zero-dimensions in the beginning, pad the "pad" sequence with zeros in the beginning
    pad_len = opset9.size(g, pad, g.op("Constant", value_t=torch.tensor([0])))
    # Set extension = [0] * (dim * 2 - len(pad))
    rank = symbolic_helper._get_tensor_rank(input)
    if rank is None:
        rank = g.op("Size", g.op("Shape", input))
    else:
        rank = g.op("Constant", value_t=torch.tensor(rank, dtype=torch.int64))
    extension = g.op(
        "Sub",
        g.op("Mul", rank, g.op("Constant", value_t=torch.tensor(2, dtype=torch.int64))),
        pad_len,
    )
    # Concat pad with extension: paddings = [dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, 0, 0, ... ]
    # Currently ONNX only supports int64 type for Pad
    pad = g.op("Cast", pad, to_i=_C_onnx.TensorProtoDataType.INT64)
    paddings = g.op(
        "Concat",
        pad,
        g.op(
            "ConstantOfShape", extension, value_t=torch.tensor([0], dtype=torch.int64)
        ),
        axis_i=0,
    )
    # Reshape and reverse order and collate first beginnings and then ends
    # paddings = [[..., 0, dim_n-1_begin, dim_n_begin],
    #               [..., 0, dim_n-1_end, dim_n_end]]
    # Reshape back to 1-D paddings = [..., 0, dim_n - 1_begin, dim_n_begin, ..., 0, dim_n - 1_end, dim_n_end]
    paddings = symbolic_helper._reshape_helper(
        g, paddings, g.op("Constant", value_t=torch.tensor([-1, 2]))
    )
    paddings = g.op("Transpose", opset10.flip(g, paddings, [0]), perm_i=[1, 0])
    paddings = symbolic_helper._reshape_helper(
        g, paddings, g.op("Constant", value_t=torch.tensor([-1]))
    )
    padding_c = g.op("Cast", paddings, to_i=_C_onnx.TensorProtoDataType.INT64)
    return padding_c


@_onnx_symbolic("aten::constant_pad_nd")
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value=None):
    mode = "constant"
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, input)
    pad = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, pad, value, mode_s=mode)


@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
def reflection_pad(g: jit_utils.GraphContext, input, padding):
    mode = "reflect"
    paddings = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, paddings, mode_s=mode)


@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
def replication_pad(g: jit_utils.GraphContext, input, padding):
    mode = "edge"
    paddings = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, paddings, mode_s=mode)


@_onnx_symbolic("aten::pad")
def pad(
    g: jit_utils.GraphContext,
    input: _C.Value,
    pad: _C.Value,
    mode: _C.Value,
    value: _C.Value,
):
    mode = symbolic_helper._parse_arg(mode, "s")
    if mode == "replicate":
        return replication_pad(g, input, pad)
    elif mode == "reflect":
        return reflection_pad(g, input, pad)
    elif mode == "constant":
        return constant_pad_nd(g, input, pad, value)
    elif mode == "circular":
        return opset9._pad_circular(g, input, pad)
    else:
        raise errors.SymbolicValueError(f"Unrecognized padding mode {mode}", input)


@_onnx_symbolic("aten::linalg_det")
def linalg_det(g: jit_utils.GraphContext, self):
    return g.op("Det", self)


@_onnx_symbolic("aten::logdet")
def logdet(g: jit_utils.GraphContext, input):
    return opset9.log(g, linalg_det(g, input))


@_onnx_symbolic("aten::arange")
def arange(g: jit_utils.GraphContext, *args):
    def _get_arange_dtype(dtype):
        dtype = symbolic_helper._maybe_get_const(dtype, "i")
        return dtype

    if len(args) == 2 and all(isinstance(val, int) for val in args):
        # aten::arange(Scalar start, Scalar end)
        dtype = torch.int64
        # Start index.
        start = g.op(
            "Constant",
            value_t=torch.tensor(args[0], dtype=dtype),
        )
        # End (exclusive) index.
        end = g.op(
            "Constant",
            value_t=torch.tensor(args[1], dtype=dtype),
        )
        # Step size from start to end indexes.
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=dtype),
        )
        return g.op("Range", start, end, delta_default)
    elif len(args) == 2 or len(args) == 5:
        if len(args) == 2:
            # aten::arange(Scalar end, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[1])
        type_, end, start, step = symbolic_helper._arange_cast_helper(
            g, end=args[0], dtype=dtype
        )
        start_default = g.op(
            "Constant",
            value_t=torch.tensor(0, dtype=type_.dtype()),
        )
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=type_.dtype()),
        )
        return g.op("Range", start_default, end, delta_default)
    elif len(args) == 4 or len(args) == 7:
        if len(args) == 4:
            # aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[3])
        _, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], step=args[2], dtype=dtype
        )
        return g.op("Range", start, end, step)
    elif len(args) == 6:
        # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[2])
        type_, end, start, step = symbolic_helper._arange_cast_helper(
            g, start=args[0], end=args[1], dtype=dtype
        )
        delta_default = g.op(
            "Constant",
            value_t=torch.tensor(1, dtype=type_.dtype()),
        )
        return g.op("Range", start, end, delta_default)
    else:
        return symbolic_helper._unimplemented(
            "aten::arange", f"with {len(args)} arguments"
        )


@_onnx_symbolic("aten::_dim_arange")
@symbolic_helper.parse_args("v", "i")
def _dim_arange(g: jit_utils.GraphContext, like, dim):
    like_shape = g.op("Shape", like)
    stop = g.op(
        "Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0
    )
    return arange(g, stop, 4, None, None, None)


@_onnx_symbolic("aten::size")
@symbolic_helper.quantized_args(True, quantize_output=False)
def size(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    return symbolic_helper._size_helper(g, self, dim)


@_onnx_symbolic("aten::squeeze")
def squeeze(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Squeeze", self)

    # dim as a tensor
    if not symbolic_helper._is_constant(dim):
        return symbolic_helper._squeeze_helper(g, self, [dim])

    dim = symbolic_helper._get_const(dim, "i", "dim")

    input_rank = symbolic_helper._get_tensor_rank(self)
    adjusted_dim = dim
    if input_rank is not None and dim < 0:
        adjusted_dim += input_rank
    dim_size = symbolic_helper._get_tensor_dim_size(self, adjusted_dim)
    if (dim < 0 and input_rank is None) or dim_size is None:
        # If onnx shape inference is not on, export always as dynamic.
        # Because we cannot tell if observed static shape is also static at runtime.
        # create "cond" node (condition is shape[i]==1)
        dim_constant = g.op("Constant", value_t=torch.tensor([dim]))
        size = symbolic_helper._size_helper(g, self, dim_constant)
        const_one = g.op("Constant", value_t=torch.ones(1, dtype=torch.int64))
        cond = g.op("Equal", size, const_one)
        # create the "If" node and add the "then" and "else" blocks to it.
        if_op, (if_context, else_context), _ = jit_utils.add_op_with_blocks(
            g, "If", cond, n_blocks=2
        )
        squeeze_ = symbolic_helper._squeeze_helper(if_context, self, [dim])
        utils._add_output_to_block(if_context.block, squeeze_)
        identity_ = else_context.op("Identity", self)
        utils._add_output_to_block(else_context.block, identity_)
        return if_op

    # For static input shape
    dim = adjusted_dim
    if dim_size > 1:
        warnings.warn(
            "This model contains a squeeze operation on dimension "
            + str(dim)
            + ". The size of "
            + "this dimension in the given input is "
            + str(dim_size)
            + ". The model will "
            + "be exported without the squeeze node. If the model is intended to be used with dynamic "
            + "input shapes, please export with dynamic_axes argument."
        )
        return self
    return symbolic_helper._squeeze_helper(g, self, [dim])


@_onnx_symbolic("aten::unsqueeze")
def unsqueeze(g: jit_utils.GraphContext, self, dim):
    if symbolic_helper._is_constant(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")

    return symbolic_helper._unsqueeze_helper(g, self, [dim])


@_onnx_symbolic("aten::mm")
def mm(g: jit_utils.GraphContext, self, other):
    return g.op("Gemm", self, other, beta_f=0.0, alpha_f=1.0)


@_onnx_symbolic("aten::index")
def index(g: jit_utils.GraphContext, self, index):
    if symbolic_helper._is_packed_list(index):
        indices = symbolic_helper._unpack_list(index)
    else:
        indices = [index]

    # Handle single mask index.
    if len(indices) == 1:
        index = indices[0]
        if not symbolic_helper._is_none(index) and (
            symbolic_helper._is_bool(index)
            or _type_utils.JitScalarType.from_value(index)
            == _type_utils.JitScalarType.UINT8
        ):
            index = opset9.nonzero(g, index)
            return g.op("GatherND", self, index)
    return opset9.index(g, self, index)


@_onnx_symbolic("aten::index_fill")
def index_fill(g: jit_utils.GraphContext, self, dim, index, value):
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, self)
    expanded_value = opset9.expand(g, value, expanded_index_shape, None)
    return scatter(g, self, dim, expanded_index, expanded_value)


@_onnx_symbolic("aten::index_copy")
def index_copy(g: jit_utils.GraphContext, self, dim, index, source):
    _expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    return scatter(g, self, dim, expanded_index, source)


@_onnx_symbolic("aten::bitwise_right_shift")
@_onnx_symbolic("aten::__rshift_")
def __rshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(self):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
        )

    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.UINT8
    ):
        return g.op("BitShift", self, other, direction_s="RIGHT")

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
    )
    rshift = g.op("Div", self, two_pow)
    return rshift


@_onnx_symbolic("aten::bitwise_left_shift")
@_onnx_symbolic("aten::__lshift_")
def __lshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if _type_utils.JitScalarType.from_value(
        other, _type_utils.JitScalarType.UNDEFINED
    ) != _type_utils.JitScalarType.from_value(self):
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
        )

    if (
        _type_utils.JitScalarType.from_value(self, _type_utils.JitScalarType.UNDEFINED)
        == _type_utils.JitScalarType.UINT8
    ):
        return g.op("BitShift", self, other, direction_s="LEFT")

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=_type_utils.JitScalarType.from_value(self).onnx_type(),
    )
    lshift = g.op("Mul", self, two_pow)
    return lshift


def _get_im2col_indices_along_dim(
    g: jit_utils.GraphContext, input_d, kernel_size_d, dilation_d, padding_d, stride_d
):
    # Input is always 4-D (N, C, H, W)
    # Calculate indices of sliding blocks along spatial dimension
    # Slide kernel over input each dim d:
    # each dimension d ranges from 0 to input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # with steps = stride

    blocks_d = g.op(
        "Add", input_d, g.op("Constant", value_t=torch.tensor(padding_d * 2))
    )
    blocks_d = g.op(
        "Sub",
        blocks_d,
        g.op("Constant", value_t=torch.tensor(dilation_d * (kernel_size_d - 1))),
    )

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = g.op(
        "Range",
        g.op("Constant", value_t=torch.tensor(0)),
        blocks_d,
        g.op("Constant", value_t=torch.tensor(stride_d)),
    )

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = torch.arange(0, kernel_size_d * dilation_d, dilation_d)
    kernel_grid = g.op("Constant", value_t=kernel_grid.unsqueeze(0))

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    blocks_d_indices = symbolic_helper._unsqueeze_helper(
        g, blocks_d_indices, [0]
    )  # Reshape to [1, -1]
    kernel_mask = symbolic_helper._reshape_helper(
        g, kernel_grid, g.op("Constant", value_t=torch.tensor([-1, 1]))
    )
    block_mask = g.op("Add", blocks_d_indices, kernel_mask)

    return block_mask


def _get_im2col_padded_input(g: jit_utils.GraphContext, input, padding_h, padding_w):
    # Input is always 4-D tensor (N, C, H, W)
    # Padding tensor has the following format: (padding_h, padding_w)
    # Reshape the padding to follow ONNX format: (dim1_begin, dim2_begin,...,dim1_end, dim2_end,...)
    pad = g.op("Constant", value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    return g.op("Pad", input, pad)


def _get_im2col_output_shape(g: jit_utils.GraphContext, input, kernel_h, kernel_w):
    batch_dim = size(g, input, g.op("Constant", value_t=torch.tensor(0)))
    channel_dim = size(g, input, g.op("Constant", value_t=torch.tensor(1)))
    channel_unfolded = g.op(
        "Mul", channel_dim, g.op("Constant", value_t=torch.tensor(kernel_h * kernel_w))
    )

    return g.op(
        "Concat",
        symbolic_helper._unsqueeze_helper(g, batch_dim, [0]),
        symbolic_helper._unsqueeze_helper(g, channel_unfolded, [0]),
        g.op("Constant", value_t=torch.tensor([-1])),
        axis_i=0,
    )


@_onnx_symbolic("aten::im2col")
@symbolic_helper.parse_args("v", "is", "is", "is", "is")
def im2col(g: jit_utils.GraphContext, input, kernel_size, dilation, padding, stride):
    # Input is always 4-D tensor (N, C, H, W)
    # All other args are int[2]

    input_h = size(g, input, g.op("Constant", value_t=torch.tensor(2)))
    input_w = size(g, input, g.op("Constant", value_t=torch.tensor(3)))

    stride_h, stride_w = stride[0], stride[1]
    padding_h, padding_w = padding[0], padding[1]
    dilation_h, dilation_w = dilation[0], dilation[1]
    kernel_h, kernel_w = kernel_size[0], kernel_size[1]

    blocks_row_indices = _get_im2col_indices_along_dim(
        g, input_h, kernel_h, dilation_h, padding_h, stride_h
    )
    blocks_col_indices = _get_im2col_indices_along_dim(
        g, input_w, kernel_w, dilation_w, padding_w, stride_w
    )

    output_shape = _get_im2col_output_shape(g, input, kernel_h, kernel_w)
    padded_input = _get_im2col_padded_input(g, input, padding_h, padding_w)

    # For a 4D matrix of size (1, 1, 3, 3) as below with kernel_size=2, stride=1, and dilation=1
    # [[[[1., 2., 3.,],
    #    [4., 5., 6.,],
    #    [7., 8., 9.,]]]]
    # First gather indices along rows (dim=2) with blocks_row_indices = [[0,1], [1,2]] to get:
    # [[[[[1., 2., 3.],
    #     [4., 5., 6.]],
    #    [[4., 5., 6.],
    #     [7., 8., 9.]]]]]
    # And then gather along cols (dim=4) with blocks_row_indices = [[0,1], [1,2]] to get:
    # [[[[[[1., 2.],
    #      [4., 5.]],
    #     [[2., 3.],
    #      [5., 6]]],
    #    [[[4., 5.],
    #      [7., 8.]],
    #     [[5., 6.],
    #      [8., 9.]]]]]]
    # Transpose dims 3 (depth) and 4 (rows), and then reshape to output shape (1, 1, 4, 4) to get:
    #  [[[1., 2., 4., 5.],
    #    [2., 3., 5., 6.],
    #    [4., 5., 7., 8.],
    #    [5., 6., 8., 9.]]]
    output = g.op("Gather", padded_input, blocks_row_indices, axis_i=2)
    output = g.op("Gather", output, blocks_col_indices, axis_i=4)
    output = g.op("Transpose", output, perm_i=[0, 1, 2, 4, 3, 5])
    return symbolic_helper._reshape_helper(g, output, output_shape)


@_onnx_symbolic("aten::narrow")
def narrow(g: jit_utils.GraphContext, input, dim, start, length):
    end = g.op("Add", start, length)
    return symbolic_helper._slice_helper(g, input, axes=dim, starts=start, ends=end)


@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
def flatten(g: jit_utils.GraphContext, input, start_dim, end_dim):
    dim = symbolic_helper._get_tensor_rank(input)
    if dim == 1:
        return input
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1:
        if end_dim == -1 or (dim is not None and end_dim == dim - 1):
            return g.op("Flatten", input, axis_i=start_dim)
    elif start_dim == 0:
        if end_dim == -2 or (dim is not None and end_dim == dim - 2):
            return g.op("Flatten", input, axis_i=end_dim + 1)
    if dim is None:
        return symbolic_helper._unimplemented(
            "dim",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )
    # if end_dim is negative add dim
    if end_dim < 0:
        end_dim = dim + end_dim

    return symbolic_helper._flatten_helper(g, input, start_dim, end_dim, dim)


@_onnx_symbolic("aten::linalg_vector_norm")
@symbolic_helper.parse_args("v", "f", "is", "b", "v")
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self,
    ord,
    dim: Sequence[int] | None,
    keepdim: bool,
    dtype,
):
    return symbolic_helper._linalg_vector_norm_helper(g, self, ord, dim, keepdim, dtype)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(
    g: jit_utils.GraphContext,
    embedding_matrix,
    indices,
    offsets,
    scale_grad_by_freq,
    mode,
    sparse,
    per_sample_weights,
    include_last_offset,
    padding_idx,
):
    return symbolic_helper._embedding_bag_helper(
        g,
        embedding_matrix,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
    )


@_onnx_symbolic("aten::embedding_renorm")
@symbolic_helper.parse_args("v", "v", "f", "f")
def embedding_renorm(g: jit_utils.GraphContext, weight, indices, max_norm, norm_type):
    unique_indices = g.op("Unique", indices)
    partial_weight = g.op("Gather", weight, unique_indices)
    norm_i = int(norm_type)
    if norm_i == 1:
        norm_type = "ReduceL1"
    elif norm_i == 2:
        norm_type = "ReduceL2"
    else:
        raise errors.SymbolicValueError(
            f"Unsupported: ONNX export of embedding_renorm with norm: {norm_i}. "
            "Only 1. and 2. are supported.",
            weight,
        )
    partial_weight_norm = g.op(norm_type, partial_weight, axes_i=[1], keepdims_i=1)
    # https://github.com/pytorch/pytorch/blob/0a07488ed2c47765e337e290bd138c0e6e459cbd/aten/src/ATen/native/Embedding.cpp#L177
    # Add 1e-7 to prevent division by zero.
    partial_weight_norm_ = g.op(
        "Add", partial_weight_norm, g.op("Constant", value_t=torch.tensor(1e-7))
    )
    max_norm = torch.tensor(max_norm)
    scales = g.op("Div", max_norm, partial_weight_norm_)
    partial_weight_renorm = g.op("Mul", partial_weight, scales)
    partial_weight_renorm = g.op(
        "Where",
        g.op("Greater", partial_weight_norm, max_norm),
        partial_weight_renorm,
        partial_weight,
    )
    return g.op(
        "ScatterND",
        weight,
        symbolic_helper._unsqueeze_helper(g, unique_indices, [1]),
        partial_weight_renorm,
    )


@_onnx_symbolic("aten::chunk")
def chunk(g: jit_utils.GraphContext, self, chunks, dim):
    # Calculate chunk size for dynamic chunk
    dim_size = g.op("Gather", g.op("Shape", self), dim, axis_i=0)
    chunk_size_s = g.op(
        "Sub", chunks, g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
    )
    chunk_size = g.op("Div", g.op("Add", dim_size, chunk_size_s), chunks)
    # Create splits vector
    chunk_vec = [
        opset9.expand(g, chunk_size, chunk_size_s, None),
        g.op("Sub", dim_size, g.op("Mul", chunk_size, chunk_size_s)),
    ]
    chunk_vec = g.op("Concat", *chunk_vec, axis_i=0)
    return split(g, self, chunk_vec, dim)


@_onnx_symbolic("aten::normal")
def normal(
    g: jit_utils.GraphContext,
    mean,
    std,
    sizes=None,
    generator=None,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
):
    # If you can sample from a given distribution with mean 0 and variance 1, then you can easily sample from a
    # scale-location transformation of that distribution, which has mean mu and variance sigma's square. If x is a sample
    # from a mean 0 and variance 1 distribution then
    #       sigma x+mu
    # is a sample with mean mu and variance sigma's square.
    if sizes is not None and not symbolic_helper._is_none(sizes):
        mean = opset9.expand(g, mean, sizes, None)
    result = opset9.mul(g, std, g.op("RandomNormalLike", mean))
    return add(g, result, mean)


@_onnx_symbolic("aten::atleast_1d")
def atleast_1d(g: jit_utils.GraphContext, self: torch._C.Value):
    # NOTE: If it's 0D, reshape to 1D

    # NOTE: self could be a packed list or a tensor
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            if tensor_rank == 0:
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1]))
                )
            new_tensor_list.append(new_tensor)
        return g.op("SequenceConstruct", *new_tensor_list)

    tensor_rank = symbolic_helper._get_tensor_rank(self)
    if tensor_rank == 0:
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1]))
        )
    return self


@_onnx_symbolic("aten::atleast_2d")
def atleast_2d(g: jit_utils.GraphContext, self: torch._C.Value):
    # NOTE: If it's 0D, reshape to 2D
    #       If it's 1D, unsqueeze to 2D

    # NOTE: self could be a packed list or a tensor
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            if tensor_rank == 0:
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1, 1]))
                )
            elif tensor_rank == 1:
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[0]
                )
            new_tensor_list.append(new_tensor)
        return g.op("SequenceConstruct", *new_tensor_list)

    tensor_rank = symbolic_helper._get_tensor_rank(self)
    if tensor_rank == 0:
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1, 1]))
        )
    elif tensor_rank == 1:
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[0])
    return self


@_onnx_symbolic("aten::atleast_3d")
def atleast_3d(g: jit_utils.GraphContext, self: torch._C.Value):
    # NOTE: If it's 0D, reshape to 3D
    #       If it's 1D, unsqueeze to 3D
    #       If it's 2D, unsqueeze to 3D

    # NOTE: self could be a packed list or a tensor
    if symbolic_helper._is_value(self) and symbolic_helper._is_packed_list(self):
        tensor_list = symbolic_helper._unpack_list(self)
        new_tensor_list = []
        for tensor in tensor_list:
            new_tensor = tensor
            tensor_rank = symbolic_helper._get_tensor_rank(tensor)
            if tensor_rank == 0:
                new_tensor = symbolic_helper._reshape_helper(
                    g, new_tensor, g.op("Constant", value_t=torch.tensor([1, 1, 1]))
                )
            elif tensor_rank == 1:
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[0]
                )
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[-1]
                )
            elif tensor_rank == 2:
                new_tensor = symbolic_helper._unsqueeze_helper(
                    g, new_tensor, axes_i=[-1]
                )
            new_tensor_list.append(new_tensor)
        return g.op("SequenceConstruct", *new_tensor_list)

    tensor_rank = symbolic_helper._get_tensor_rank(self)
    if tensor_rank == 0:
        self = symbolic_helper._reshape_helper(
            g, self, g.op("Constant", value_t=torch.tensor([1, 1, 1]))
        )
    elif tensor_rank == 1:
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[0])
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[-1])
    elif tensor_rank == 2:
        self = symbolic_helper._unsqueeze_helper(g, self, axes_i=[-1])
    return self


@_onnx_symbolic("prim::ConstantChunk")
def prim_constant_chunk(g: jit_utils.GraphContext, self, chunks, dim):
    input_shape = g.op("Shape", self)
    axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
    input_shape_dim = g.op("Gather", input_shape, axis, axis_i=0)
    start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
    chunk_size = g.op("Constant", value_t=torch.tensor([chunks], dtype=torch.long))
    chunk_size_minus_1 = g.op(
        "Constant", value_t=torch.tensor([chunks - 1], dtype=torch.long)
    )
    input_shape_dim_shift = g.op("Add", input_shape_dim, chunk_size_minus_1)
    chunk_dim = g.op("Div", input_shape_dim_shift, chunk_size)
    res = []
    for i in range(chunks):
        index = g.op("Constant", value_t=torch.tensor([i + 1], dtype=torch.long))
        end = g.op("Mul", chunk_dim, index)
        res.append(g.op("Slice", self, start, end, axis))
        start = end
    return res


@_onnx_symbolic("aten::hstack")
def hstack(g: jit_utils.GraphContext, tensor_list: _C.Value):
    tensor_list = atleast_1d(g, tensor_list)
    first_tensor = g.op(
        "SequenceAt",
        tensor_list,
        g.op("Constant", value_t=torch.tensor(0, dtype=torch.long)),
    )
    first_tensor_shape = g.op("Shape", first_tensor)
    first_tensor_dim = g.op("Size", first_tensor_shape)

    const_one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.long))
    equal_to_one = g.op("Equal", first_tensor_dim, const_one)

    (
        if_op_greater,
        (if_context_equal, else_context_equal),
        _,
    ) = jit_utils.add_op_with_blocks(g, "If", equal_to_one, n_blocks=2, outputs=1)
    result_if = if_context_equal.op(
        "ConcatFromSequence", tensor_list, axis_i=0, new_axis_i=0
    )
    utils._add_output_to_block(if_context_equal.block, result_if)
    result_else = else_context_equal.op(
        "ConcatFromSequence", tensor_list, axis_i=1, new_axis_i=0
    )
    utils._add_output_to_block(else_context_equal.block, result_else)
    result = if_op_greater.node().output()

    return result


@_onnx_symbolic("aten::vstack")
def vstack(g: jit_utils.GraphContext, tensor_list: _C.Value):
    tensor_list = atleast_2d(g, tensor_list)
    return g.op("ConcatFromSequence", tensor_list, axis_i=0, new_axis_i=0)
