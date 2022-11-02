"""This file exports ONNX ops for opset 11."""

import functools
import sys
import warnings
from typing import Optional, Sequence, Union

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
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils, registration

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in README.md

__all__ = [
    "add",
    "append",
    "arange",
    "argsort",
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
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=11)


def _apply_params(*args, **kwargs):
    """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""

    def _apply(fn):
        return fn(*args, **kwargs)

    return _apply


@_onnx_symbolic("aten::hardtanh")
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "f", "f")
@_beartype.beartype
def hardtanh(g: jit_utils.GraphContext, self: _C.Value, min_val: float, max_val: float):
    dtype = self.type().scalarType()
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType.from_name(dtype)
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(min_val, dtype=scalar_type.dtype()),
    )
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(max_val, dtype=scalar_type.dtype()),
    )
    return opset9._op_with_optional_float_cast(
        g, "Clip", self, min_val, max_val, opset_before=12
    )


@_onnx_symbolic("aten::clamp")
@_beartype.beartype
def clamp(g: jit_utils.GraphContext, self, min, max):
    dtype = self.type().scalarType()

    @_beartype.beartype
    def _cast_if_not_none(tensor, dtype):
        if tensor is not None and not symbolic_helper._is_none(tensor):
            return g.op(
                "Cast",
                tensor,
                to_i=_type_utils.JitScalarType.from_name(dtype).onnx_type(),
            )
        else:
            return tensor

    if dtype is not None:
        min = _cast_if_not_none(min, dtype)
        max = _cast_if_not_none(max, dtype)

    if symbolic_helper._is_none(min):
        return clamp_max(g, self, max)
    elif symbolic_helper._is_none(max):
        return clamp_min(g, self, min)
    else:
        if (
            symbolic_helper._get_tensor_rank(min) == 0
            and symbolic_helper._get_tensor_rank(max) == 0
        ):
            return opset9._op_with_optional_float_cast(
                g, "Clip", self, min, max, opset_before=12
            )
        else:
            return clamp_max(g, clamp_min(g, self, min), max)


@_onnx_symbolic("aten::clamp_min")
@symbolic_helper.parse_args("v", "v")
@_beartype.beartype
def clamp_min(g: jit_utils.GraphContext, self, min):
    dtype = self.type().scalarType()
    min = g.op("Cast", min, to_i=_type_utils.JitScalarType.from_name(dtype).onnx_type())
    if symbolic_helper._get_tensor_rank(min) == 0:
        max = opset9.unused(g)
        return opset9._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        return opset9._op_with_optional_float_cast(g, "Max", self, min, opset_before=12)


@_onnx_symbolic("aten::clamp_max")
@symbolic_helper.parse_args("v", "v")
@_beartype.beartype
def clamp_max(g: jit_utils.GraphContext, self, max):
    dtype = self.type().scalarType()
    max = g.op("Cast", max, to_i=_type_utils.JitScalarType.from_name(dtype).onnx_type())
    if symbolic_helper._get_tensor_rank(max) == 0:
        min = opset9.unused(g)
        return opset9._op_with_optional_float_cast(
            g, "Clip", self, min, max, opset_before=12
        )
    else:
        return opset9._op_with_optional_float_cast(g, "Min", self, max, opset_before=12)


@_onnx_symbolic("aten::relu6")
@_beartype.beartype
def relu6(g: jit_utils.GraphContext, input):
    relu_ = opset9._op_with_optional_float_cast(g, "Relu", input, opset_before=14)
    dtype = input.type().scalarType()
    if dtype is None:
        scalar_type = _type_utils.JitScalarType.FLOAT
    else:
        scalar_type = _type_utils.JitScalarType.from_name(dtype)
    min_val = g.op(
        "Constant",
        value_t=torch.tensor(0, dtype=scalar_type.dtype()),
    )
    max_val = g.op(
        "Constant",
        value_t=torch.tensor(6, dtype=scalar_type.dtype()),
    )
    return clamp(g, relu_, min_val, max_val)


@_onnx_symbolic("aten::select")
# Opset 11 gather accepts negative indices
@symbolic_helper.quantized_args(True)
@symbolic_helper.parse_args("v", "i", "v")
@_beartype.beartype
def select(g: jit_utils.GraphContext, self, dim, index):
    return g.op("Gather", self, index, axis_i=dim)


@_onnx_symbolic("aten::index_put")
@_beartype.beartype
def index_put(
    g: jit_utils.GraphContext, self, indices_list_value, values, accumulate=False
):
    if symbolic_helper._is_packed_list(indices_list_value):
        indices_list = symbolic_helper._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]
    if symbolic_helper.is_caffe2_aten_fallback():
        args = [self] + indices_list + [values, accumulate]
        return g.at("index_put", *args)

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

    dtype = self.type().scalarType()
    if dtype is not None and dtype != values.type().scalarType():
        values = g.op(
            "Cast", values, to_i=_type_utils.JitScalarType.from_name(dtype).onnx_type()
        )
    scalar_type = _type_utils.JitScalarType.from_name(dtype)

    if accumulate:
        zeros = g.op(
            "ConstantOfShape",
            g.op("Shape", self),
            value_t=torch.tensor([0], dtype=scalar_type.dtype()),
        )
        result = g.op("ScatterND", zeros, index, values)
        result = add(g, self, result)
    else:
        result = g.op("ScatterND", self, index, values)

    return result


@_onnx_symbolic("aten::pixel_shuffle")
@symbolic_helper.parse_args("v", "i")
@_beartype.beartype
def pixel_shuffle(g: jit_utils.GraphContext, self, upscale_factor):
    rank = symbolic_helper._get_tensor_rank(self)
    if rank is not None and rank != 4:
        return symbolic_helper._unimplemented("pixel_shuffle", "only support 4d input")
    return g.op("DepthToSpace", self, blocksize_i=upscale_factor, mode_s="CRD")


@_onnx_symbolic(
    "aten::upsample_nearest1d",
    decorate=[_apply_params("upsample_nearest1d", 3, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest2d",
    decorate=[_apply_params("upsample_nearest2d", 4, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_nearest3d",
    decorate=[_apply_params("upsample_nearest3d", 5, "nearest")],
)
@_onnx_symbolic(
    "aten::upsample_linear1d",
    decorate=[_apply_params("upsample_linear1d", 3, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bilinear2d",
    decorate=[_apply_params("upsample_bilinear2d", 4, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_trilinear3d",
    decorate=[_apply_params("upsample_trilinear3d", 5, "linear")],
)
@_onnx_symbolic(
    "aten::upsample_bicubic2d",
    decorate=[_apply_params("upsample_bicubic2d", 4, "cubic")],
)
@_beartype.beartype
def _interpolate(name: str, dim: int, interpolate_mode: str):
    return symbolic_helper._interpolate_helper(name, dim, interpolate_mode)


@_onnx_symbolic("aten::__interpolate")
@symbolic_helper.quantized_args(True, False, False, False, False, False, False)
@_beartype.beartype
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
@_beartype.beartype
def gather(g: jit_utils.GraphContext, self, dim, index, sparse_grad=False):
    if symbolic_helper._maybe_get_const(sparse_grad, "i"):
        return symbolic_helper._unimplemented("gather", "sparse_grad == True")
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("gather", self, dim, index, sparse_grad)
    return g.op("GatherElements", self, index, axis_i=dim)


@_onnx_symbolic("aten::scatter")
@symbolic_helper.parse_args("v", "i", "v", "v")
@_beartype.beartype
def scatter(g: jit_utils.GraphContext, self, dim, index, src):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("scatter", self, dim, index, src, overload_name="src")
    src_type = src.type().scalarType()
    src = symbolic_helper._maybe_get_scalar(src)
    if symbolic_helper._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim)
    else:
        # Check if scalar "src" has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        if self.type().scalarType() != src_type:
            src = g.op(
                "Cast",
                src,
                to_i=_type_utils.JitScalarType.from_name(
                    self.type().scalarType()
                ).onnx_type(),
            )
        return g.op(
            "ScatterElements", self, index, opset9.expand_as(g, src, index), axis_i=dim
        )


@_onnx_symbolic("aten::cumsum")
@symbolic_helper.parse_args("v", "i", "none")
@_beartype.beartype
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
@_beartype.beartype
def masked_select(g: jit_utils.GraphContext, self, mask):
    index = opset9.nonzero(g, opset9.expand_as(g, mask, self))
    return g.op("GatherND", self, index)


@_onnx_symbolic("aten::masked_scatter")
@_beartype.beartype
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
        dynamic_slice=True,
    )
    return g.op("ScatterND", self, index, source)


@_onnx_symbolic("aten::len")
@_beartype.beartype
def _len(g: jit_utils.GraphContext, self):
    if (
        symbolic_helper._is_tensor_list(self)
        or self.node().kind() == "onnx::SplitToSequence"
    ):
        return g.op("SequenceLength", self)
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return symbolic_helper._squeeze_helper(g, sz_0, [0])


@_onnx_symbolic("aten::__getitem_")
@_beartype.beartype
def __getitem_(g: jit_utils.GraphContext, self, i):
    if symbolic_helper._is_tensor_list(self):
        # SequenceAt requires that the input be a List of Tensors
        return g.op("SequenceAt", self, i)
    else:
        from torch.onnx.symbolic_opset9 import __getitem_ as getitem

        return getitem(g, self, i)


@_onnx_symbolic("aten::_set_item")
@_beartype.beartype
def _set_item(g: jit_utils.GraphContext, tensor_list, i, v):
    tensor_list = g.op("SequenceErase", tensor_list, i)
    return g.op("SequenceInsert", tensor_list, v, i)


@_onnx_symbolic("aten::append")
@_beartype.beartype
def append(g: jit_utils.GraphContext, self, tensor):
    return g.op("SequenceInsert", self, tensor)


@_onnx_symbolic("aten::add")
@_beartype.beartype
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
@_beartype.beartype
def insert(g: jit_utils.GraphContext, self, pos, tensor):
    return g.op("SequenceInsert", self, tensor, pos)


@_onnx_symbolic("aten::pop")
@_beartype.beartype
def pop(g: jit_utils.GraphContext, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)


@_onnx_symbolic("aten::Delete")
@_beartype.beartype
def Delete(g: jit_utils.GraphContext, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)


@_onnx_symbolic("aten::cat")
@_beartype.beartype
def cat(g: jit_utils.GraphContext, tensor_list, dim):
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.cat(g, tensor_list, dim)
    else:
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim)


@_onnx_symbolic("aten::stack")
@_beartype.beartype
def stack(g: jit_utils.GraphContext, tensor_list, dim):
    if symbolic_helper._is_packed_list(tensor_list):
        return opset9.stack(g, tensor_list, dim)
    else:
        dim = symbolic_helper._get_const(dim, "i", "dim")
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim, new_axis_i=1)


@_onnx_symbolic("aten::_unique2")
@symbolic_helper.parse_args("v", "i", "i", "i")
@_beartype.beartype
def _unique2(g: jit_utils.GraphContext, self, sorted, return_inverse, return_counts):
    u, indices, inverse_indices, counts = g.op(
        "Unique", self, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


@_onnx_symbolic(
    "aten::avg_pool1d",
    decorate=[_apply_params("avg_pool1d", torch.nn.modules.utils._single)],
)
@_onnx_symbolic(
    "aten::avg_pool2d",
    decorate=[_apply_params("avg_pool2d", torch.nn.modules.utils._pair)],
)
@_onnx_symbolic(
    "aten::avg_pool3d",
    decorate=[_apply_params("avg_pool3d", torch.nn.modules.utils._triple)],
)
@_beartype.beartype
def _avg_pool(name, tuple_fn):
    @symbolic_helper.quantized_args(True, False, False, False, False, False, False)
    @symbolic_helper.parse_args("v", "is", "is", "is", "i", "i", "none")
    @_beartype.beartype
    def symbolic_fn(
        g,
        input: _C.Value,
        kernel_size: Sequence[int],
        stride: Sequence[int],
        padding: Union[int, Sequence[int]],
        ceil_mode: int,
        count_include_pad: int,
        divisor_override=None,
    ):
        # Although onnx::AvgPool provides count_include_pad and ceil_mode,
        # The corner case of Average Pooling with ceil_mode on
        # PyTorch allows sliding window go off bound, which leads to
        # this accommodation.
        # More detail on https://github.com/pytorch/pytorch/issues/57178
        if not stride:
            stride = kernel_size
        padding = symbolic_helper._avgpool_helper(
            tuple_fn, padding, kernel_size, stride, divisor_override, name
        )
        assert isinstance(padding, tuple)
        adjusted_padding = padding
        if count_include_pad:
            input = g.op(
                "Pad",
                input,
                g.op("Constant", value_t=torch.tensor(((0,) * 2 + padding) * 2)),
                mode_s="constant",
            )
            adjusted_padding = (0,) * len(padding)
        if ceil_mode:
            padding_ceil = opset9.get_pool_ceil_padding(
                input, kernel_size, stride, padding
            )
            adjusted_padding = adjusted_padding + tuple(
                a + b for (a, b) in zip(padding_ceil, adjusted_padding)
            )
        else:
            adjusted_padding = adjusted_padding * 2
        output = g.op(
            "AveragePool",
            input,
            kernel_shape_i=tuple_fn(kernel_size),
            strides_i=tuple_fn(stride),
            pads_i=adjusted_padding,
        )
        return output

    return symbolic_fn


@_onnx_symbolic("aten::unique_dim")
@symbolic_helper.parse_args("v", "i", "i", "i", "i")
@_beartype.beartype
def unique_dim(
    g: jit_utils.GraphContext, self, dim, sorted, return_inverse, return_counts
):
    u, indices, inverse_indices, counts = g.op(
        "Unique", self, axis_i=dim, sorted_i=sorted, outputs=4
    )
    return u, inverse_indices, counts


@_onnx_symbolic("aten::topk")
@symbolic_helper.parse_args("v", "v", "i", "i", "i", "none")
@_beartype.beartype
def topk(g: jit_utils.GraphContext, self, k, dim, largest, sorted, out=None):
    return symbolic_helper._topk_helper(
        g, self, k, dim, largest=largest, sorted=sorted, out=out
    )


@_onnx_symbolic("aten::sort")
@symbolic_helper.parse_args("v", "i", "i", "none")
@_beartype.beartype
def sort(g: jit_utils.GraphContext, self, dim, decending, out=None):
    return symbolic_helper._sort_helper(g, self, dim, decending=decending, out=out)


@_onnx_symbolic("aten::argsort")
@symbolic_helper.parse_args("v", "i", "i", "none")
@_beartype.beartype
def argsort(g: jit_utils.GraphContext, self, dim, decending, out=None):
    _, indices = symbolic_helper._sort_helper(
        g, self, dim, decending=decending, out=out
    )
    return indices


@_onnx_symbolic("aten::round")
@_beartype.beartype
def round(g: jit_utils.GraphContext, self):
    return g.op("Round", self)


@_onnx_symbolic("aten::remainder")
@_beartype.beartype
def remainder(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_fp(input) or symbolic_helper._is_fp(other):
        return opset9.remainder(g, input, other)
    return g.op("Mod", input, other, fmod_i=0)


@_onnx_symbolic("aten::split")
@symbolic_helper.parse_args("v", "v", "i", "i")
@_beartype.beartype
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
@_beartype.beartype
def split_with_sizes(g: jit_utils.GraphContext, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


@_onnx_symbolic("aten::unbind")
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
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


@_beartype.beartype
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
@_beartype.beartype
def constant_pad_nd(g: jit_utils.GraphContext, input, padding, value=None):
    mode = "constant"
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, input)
    pad = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, pad, value, mode_s=mode)


@_onnx_symbolic("aten::reflection_pad1d")
@_onnx_symbolic("aten::reflection_pad2d")
@_onnx_symbolic("aten::reflection_pad3d")
@_beartype.beartype
def reflection_pad(g: jit_utils.GraphContext, input, padding):
    mode = "reflect"
    paddings = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, paddings, mode_s=mode)


@_onnx_symbolic("aten::replication_pad1d")
@_onnx_symbolic("aten::replication_pad2d")
@_onnx_symbolic("aten::replication_pad3d")
@_beartype.beartype
def replication_pad(g: jit_utils.GraphContext, input, padding):
    mode = "edge"
    paddings = _prepare_onnx_paddings(g, input, padding)
    return g.op("Pad", input, paddings, mode_s=mode)


@_onnx_symbolic("aten::pad")
@_beartype.beartype
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
@_beartype.beartype
def linalg_det(g: jit_utils.GraphContext, self):
    return g.op("Det", self)


@_onnx_symbolic("aten::logdet")
@_beartype.beartype
def logdet(g: jit_utils.GraphContext, input):
    return opset9.log(g, linalg_det(g, input))


@_onnx_symbolic("aten::arange")
@_beartype.beartype
def arange(g: jit_utils.GraphContext, *args):
    def _get_arange_dtype(dtype):
        dtype = symbolic_helper._maybe_get_const(dtype, "i")
        return dtype

    if len(args) == 2 or len(args) == 5:
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
@_beartype.beartype
def _dim_arange(g: jit_utils.GraphContext, like, dim):
    like_shape = g.op("Shape", like)
    stop = g.op(
        "Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0
    )
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.op("_caffe2::Range", stop)
    return arange(g, stop, 4, None, None, None)


@_onnx_symbolic("aten::size")
@_beartype.beartype
def size(g: jit_utils.GraphContext, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    return symbolic_helper._size_helper(g, self, dim)


@_onnx_symbolic("aten::squeeze")
@_beartype.beartype
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
@_beartype.beartype
def unsqueeze(g: jit_utils.GraphContext, self, dim):
    if symbolic_helper._is_constant(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")

    return symbolic_helper._unsqueeze_helper(g, self, [dim])


@_onnx_symbolic("aten::mm")
@_beartype.beartype
def mm(g: jit_utils.GraphContext, self, other):
    return g.op("Gemm", self, other, beta_f=0.0, alpha_f=1.0)


@_onnx_symbolic("aten::index")
@_beartype.beartype
def index(g: jit_utils.GraphContext, self, index):
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("index", self, index, overload_name="Tensor")

    if symbolic_helper._is_packed_list(index):
        indices = symbolic_helper._unpack_list(index)
    else:
        indices = [index]

    # Handle single mask index.
    if len(indices) == 1:
        index = indices[0]
        if not symbolic_helper._is_none(index) and (
            symbolic_helper._is_bool(index) or index.type().scalarType() == "Byte"
        ):
            index = opset9.nonzero(g, index)
            return g.op("GatherND", self, index)
    return opset9.index(g, self, index)


@_onnx_symbolic("aten::index_fill")
@_beartype.beartype
def index_fill(g: jit_utils.GraphContext, self, dim, index, value):
    dim_value = symbolic_helper._parse_arg(dim, "i")
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at(
            "index_fill",
            self,
            index,
            value,
            overload_name="int_Scalar",
            dim_i=dim_value,
        )

    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    value = symbolic_helper._maybe_get_scalar(value)
    value = symbolic_helper._if_scalar_type_as(value, self)
    expanded_value = opset9.expand(g, value, expanded_index_shape, None)
    return scatter(g, self, dim, expanded_index, expanded_value)


@_onnx_symbolic("aten::index_copy")
@_beartype.beartype
def index_copy(g: jit_utils.GraphContext, self, dim, index, source):
    dim_value = symbolic_helper._parse_arg(dim, "i")
    if symbolic_helper.is_caffe2_aten_fallback():
        return g.at("index_copy", self, index, source, dim_i=dim_value)
    expanded_index_shape, expanded_index = symbolic_helper._index_fill_reshape_helper(
        g, self, dim, index
    )
    return scatter(g, self, dim, expanded_index, source)


@_onnx_symbolic("aten::__rshift_")
@_beartype.beartype
def __rshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_name(
                self.type().scalarType()
            ).onnx_type(),
        )

    if self.type().scalarType() == "Byte":
        return g.op("BitShift", self, other, direction_s="RIGHT")

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=_type_utils.JitScalarType.from_name(self.type().scalarType()).onnx_type(),
    )
    rshift = g.op("Div", self, two_pow)
    return rshift


@_onnx_symbolic("aten::__lshift_")
@_beartype.beartype
def __lshift_(g: jit_utils.GraphContext, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op(
            "Cast",
            other,
            to_i=_type_utils.JitScalarType.from_name(
                self.type().scalarType()
            ).onnx_type(),
        )

    if self.type().scalarType() == "Byte":
        return g.op("BitShift", self, other, direction_s="LEFT")

    two = g.op("Constant", value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not symbolic_helper._is_fp(self):
        other = g.op("Cast", other, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    two_pow = g.op("Pow", two, other)
    two_pow = g.op(
        "Cast",
        two_pow,
        to_i=_type_utils.JitScalarType.from_name(self.type().scalarType()).onnx_type(),
    )
    lshift = g.op("Mul", self, two_pow)
    return lshift


@_beartype.beartype
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


@_beartype.beartype
def _get_im2col_padded_input(g: jit_utils.GraphContext, input, padding_h, padding_w):
    # Input is always 4-D tensor (N, C, H, W)
    # Padding tensor has the following format: (padding_h, padding_w)
    # Reshape the padding to follow ONNX format: (dim1_begin, dim2_begin,...,dim1_end, dim2_end,...)
    pad = g.op("Constant", value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    return g.op("Pad", input, pad)


@_beartype.beartype
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
@_beartype.beartype
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
@_beartype.beartype
def narrow(g: jit_utils.GraphContext, input, dim, start, length):
    end = g.op("Add", start, length)
    return symbolic_helper._slice_helper(
        g, input, axes=dim, starts=start, ends=end, dynamic_slice=True
    )


@_onnx_symbolic("aten::flatten")
@symbolic_helper.quantized_args(True, False, False)
@symbolic_helper.parse_args("v", "i", "i")
@_beartype.beartype
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
@_beartype.beartype
def linalg_vector_norm(
    g: jit_utils.GraphContext,
    self,
    ord,
    dim: Optional[Sequence[int]],
    keepdim: bool,
    dtype,
):
    if ord == 0:
        if dim is None:
            self = symbolic_helper._reshape_helper(
                g, self, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64))
            )
            keepdim = False

        cond_op = g.op(
            "Not", g.op("Equal", self, g.op("Constant", value_t=torch.LongTensor([0])))
        )
        cond_op = g.op(
            "Cast",
            cond_op,
            to_i=_type_utils.JitScalarType.from_name(
                self.type().scalarType()
            ).onnx_type(),
        )
        return symbolic_helper._reducesum_helper(
            g, cond_op, axes_i=dim, keepdims_i=keepdim
        )
    else:
        return opset9.linalg_vector_norm(g, self, ord, dim, keepdim, dtype)


@_onnx_symbolic("aten::embedding_bag")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
@_beartype.beartype
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
    if scale_grad_by_freq and GLOBALS.export_training:
        return symbolic_helper._onnx_unsupported(
            "embedding_bag with scale_grad_by_freq for training mode"
        )
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")

    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL)
    zero = g.op("Constant", value_t=torch.tensor([0]))

    indices_len = symbolic_helper._unsqueeze_helper(
        g,
        symbolic_helper._size_helper(
            g, indices, g.op("Constant", value_t=torch.tensor(0))
        ),
        [0],
    )
    if not include_last_offset:
        offsets = [offsets, indices_len]
        offsets = g.op("Concat", *offsets, axis_i=0)

    # Offsets holds the starting index position of each bag. So we create a list of the indices slices (determined by
    # offsets) and gather those indices in indices_row. Then we use this subset of indices to gather from embeddings.
    # The embeddings output is a loop scan output, so we can avoid creating a sequence and inserting elements in.
    offsets_starts = symbolic_helper._slice_helper(
        g, offsets, axes=[0], starts=[0], ends=[sys.maxsize], steps=[1]
    )
    offsets_ends = symbolic_helper._slice_helper(
        g, offsets, axes=[0], starts=[1], ends=[sys.maxsize], steps=[1]
    )

    loop_len = symbolic_helper._size_helper(
        g, offsets_ends, g.op("Constant", value_t=torch.tensor(0))
    )

    loop, (loop_context,), _ = jit_utils.add_op_with_blocks(
        g, "Loop", loop_len, loop_condition, n_blocks=1
    )
    loop_block = loop_context.block

    # FIXME(justinchuby): We need to handle what happens when we call b.op on a node return
    block_input_iter = utils._add_input_to_block(loop_block)
    cond = utils._add_input_to_block(loop_block)

    indices_start = loop_context.op(
        "Gather", offsets_starts, block_input_iter, axis_i=0
    )
    indices_end = loop_context.op("Gather", offsets_ends, block_input_iter, axis_i=0)
    indices_start = symbolic_helper._unsqueeze_helper(loop_context, indices_start, [0])
    indices_end = symbolic_helper._unsqueeze_helper(loop_context, indices_end, [0])

    indices_row = loop_context.op("Slice", indices, indices_start, indices_end, zero)
    embeddings = loop_context.op("Gather", embedding_matrix, indices_row, axis_i=0)
    if not symbolic_helper._is_none(per_sample_weights):
        per_sample_weights_row = loop_context.op(
            "Slice", per_sample_weights, indices_start, indices_end, zero
        )
        per_sample_weights_row = symbolic_helper._unsqueeze_helper(
            loop_context, per_sample_weights_row, [1]
        )
        embeddings = loop_context.op("Mul", embeddings, per_sample_weights_row)
    if mode == 0:
        embeddings = symbolic_helper._reducesum_helper(
            loop_context, embeddings, axes_i=[0], keepdims_i=0
        )
    elif mode == 1:
        embeddings = loop_context.op("ReduceMean", embeddings, axes_i=[0], keepdims_i=0)
    else:
        embeddings = loop_context.op("ReduceMax", embeddings, axes_i=[0], keepdims_i=0)

    cond_out = loop_context.op(
        "Cast", loop_condition, to_i=_C_onnx.TensorProtoDataType.BOOL
    )
    utils._add_output_to_block(loop_block, cond_out)
    utils._add_output_to_block(loop_block, embeddings)

    # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
    # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
    return loop.node().output(), None, None, None


@_onnx_symbolic("aten::embedding_renorm")
@symbolic_helper.parse_args("v", "v", "f", "f")
@_beartype.beartype
def embedding_renorm(g: jit_utils.GraphContext, weight, indices, max_norm, norm_type):
    unique_indices = g.op("Unique", indices)
    partial_weight = g.op("Gather", weight, unique_indices)
    norm_type = int(norm_type)
    if norm_type == 1:
        norm_type = "ReduceL1"
    elif norm_type == 2:
        norm_type = "ReduceL2"
    else:
        raise errors.SymbolicValueError(
            f"Unsupported: ONNX export of embedding_renorm with norm: {norm_type}. "
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
@_beartype.beartype
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
@_beartype.beartype
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
    # scale-location transformation of that distribution, which has mean μ and variance σ's square. If x is a sample
    # from a mean 0 and variance 1 distribution then
    #       σx+μ
    # is a sample with mean μ and variance σ's square.
    if sizes is not None and not symbolic_helper._is_none(sizes):
        mean = opset9.expand(g, mean, sizes, None)
    result = opset9.mul(g, std, g.op("RandomNormalLike", mean))
    return add(g, result, mean)


@_onnx_symbolic("prim::ConstantChunk")
@_beartype.beartype
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
