
import torch
from torch.nn.modules.utils import _single, _pair, _triple
import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _unimplemented
import torch.onnx.symbolic_opset9

from sys import maxsize

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 10
# Opset 10 is supported by ONNX release 1.5.0
# release on 04/24/19


def div(g, self, other, *args):
    if len(args) == 0:
        return torch.onnx.symbolic_opset9.true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@parse_args("v", "v", "s")
def _div_rounding_mode(g, self, other, rounding_mode):
    if rounding_mode == "floor":
        return _floor_divide(g, self, other)
    else:
        return torch.onnx.symbolic_opset9._div_rounding_mode(g, self, other, rounding_mode)


def _floor_divide(g, self, other):
    if sym_help._is_fp(self) or sym_help._is_fp(other):
        out = torch.onnx.symbolic_opset9.true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        # Integer division does trunction rounding
        div = g.op("Div", self, other)
        # Division is negative if: self < 0 != other < 0
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        negative = g.op("Xor",
                        g.op("Less", self, zero),
                        g.op("Less", other, zero))

        # For negative numbers with self % other != 0, subtract 1 to round down instead of up
        mod = g.op("Mod", self, other, fmod_i=0)
        fixup_mask = g.op("And", negative,
                          g.op("Not", g.op("Equal", mod, zero)))

        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        fixup = g.op("Sub", div, one)
        return g.op("Where", fixup_mask, fixup, div)


@parse_args("v", "i", "i", "none")
def sort(g, self, dim, decending, out=None):
    return sym_help._sort_helper(g, self, dim, decending=decending, out=out)


@parse_args("v", "v", "i", "i", "i", "none")
def topk(g, self, k, dim, largest, sorted, out=None):
    return sym_help._topk_helper(g, self, k, dim, largest=largest, sorted=sorted, out=out)


def _max_pool(name, tuple_fn, ndims, return_indices):
    @parse_args("v", "is", "is", "is", "is", "i")
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        if not stride:
            stride = kernel_size
        kwargs = {
            "kernel_shape_i": tuple_fn(kernel_size),
            "pads_i": tuple_fn(padding) * 2,
            "strides_i": tuple_fn(stride),
            "ceil_mode_i": ceil_mode,
        }
        if set(tuple_fn(dilation)) != {1}:
            kwargs["dilations_i"] = tuple_fn(dilation)
        # easy but hacky way to get flattened indices values
        # to be used to convert the indices values to non-flattened.
        # In ONNX the indices are computed as a flatten 1-D tensor,
        # so the values in indices are in [0, N x C x D1 x ... x Dn).
        # To convert the indices to the same format used by Pytorch,
        # we first execute a maxpool with a kernel and stride of 1 on the same input.
        # This will result in a tensor of indices in which each index will have it's own value.
        # Using this tensor as a reference, we extract the first index of each axis and subtract
        # it from each index of this axis in the indices to convert.
        # This step will result in a tensor were each dimension has values of indices within
        # the dimension it is in.
        # For more information :
        # https://github.com/pytorch/pytorch/pull/16455#issuecomment-460776407
        if return_indices:
            r, indices = g.op("MaxPool", input, outputs=2, **kwargs)
            _, flattened_indices = g.op("MaxPool", input, outputs=2,
                                        kernel_shape_i=[1 for _ in range(ndims)],
                                        strides_i=[1 for _ in range(ndims)])
            # convert indices to have non-flattened indices values
            from torch.onnx.symbolic_opset9 import sub
            s = sym_help._slice_helper(g, flattened_indices, axes=[2 + i for i in range(ndims)],
                                       starts=tuple_fn(0), ends=tuple_fn(1))
            indices = sub(g, indices, s)
            return r, indices
        else:
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            return r

    return symbolic_fn


max_pool1d = _max_pool("max_pool1d", _single, 1, return_indices=False)
max_pool2d = _max_pool("max_pool2d", _pair, 2, return_indices=False)
max_pool3d = _max_pool("max_pool3d", _triple, 3, return_indices=False)
max_pool1d_with_indices = _max_pool("max_pool1d_with_indices", _single, 1, return_indices=True)
max_pool2d_with_indices = _max_pool("max_pool2d_with_indices", _pair, 2, return_indices=True)
max_pool3d_with_indices = _max_pool("max_pool3d_with_indices", _triple, 3, return_indices=True)


def _avg_pool(name, tuple_fn):
    @parse_args("v", "is", "is", "is", "i", "i", "none")
    def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
        if not stride:
            stride = kernel_size
        padding = sym_help._avgpool_helper(tuple_fn, padding, kernel_size, stride, divisor_override, name)
        if count_include_pad:
            input = g.op("Pad", input,
                         pads_i=((0,) * 2 + padding) * 2,
                         mode_s="constant",
                         value_f=0.)
            padding = (0,) * len(padding)
        output = g.op("AveragePool", input,
                      kernel_shape_i=tuple_fn(kernel_size),
                      strides_i=tuple_fn(stride),
                      pads_i=padding * 2,
                      ceil_mode_i=ceil_mode)
        return output
    return symbolic_fn


avg_pool1d = _avg_pool("avg_pool1d", _single)
avg_pool2d = _avg_pool("avg_pool2d", _pair)
avg_pool3d = _avg_pool("avg_pool3d", _triple)


def _interpolate(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = sym_help._get_interpolate_attributes(g, interpolate_mode, args)
        sym_help._interpolate_warning(interpolate_mode)
        align_corners = sym_help._maybe_get_scalar(align_corners)
        if align_corners:
            return _unimplemented(name, "align_corners == True")
        if scales is None:
            scales = sym_help._interpolate_size_to_scales(g, input, output_size, dim)
        return g.op("Resize", input, scales, mode_s=interpolate_mode)
    return symbolic_fn


upsample_nearest1d = _interpolate("upsample_nearest1d", 3, "nearest")
upsample_nearest2d = _interpolate("upsample_nearest2d", 4, "nearest")
upsample_nearest3d = _interpolate("upsample_nearest3d", 5, "nearest")
upsample_linear1d = _interpolate("upsample_linear1d", 3, "linear")
upsample_bilinear2d = _interpolate("upsample_bilinear2d", 4, "linear")
upsample_trilinear3d = _interpolate("upsample_trilinear3d", 5, "linear")

def __interpolate(g, input, size, scale_factor, mode , align_corners, recompute_scale_factor):
    scales, mode = sym_help._interpolate_get_scales_and_mode(g, input, size, scale_factor,
                                                             mode , align_corners)
    return g.op("Resize", input, scales, mode_s=mode)


def _slice(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    if dynamic_slice:
        starts = sym_help._unsqueeze_helper(g, starts, [0])
        ends = sym_help._unsqueeze_helper(g, ends, [0])
        if isinstance(axes, int):
            axes = g.op("Constant", value_t=torch.tensor(axes))
        axes = sym_help._unsqueeze_helper(g, axes, [0])
    else:
        assert len(starts) == len(ends)
        assert len(starts) == len(axes)
        assert steps is None or len(starts) == len(steps)
        if len(starts) == 1 and starts[0] == 0 and ends[0] == 9223372036854775807\
           and (steps is None or (len(steps) == 1 and steps[0] == 1)):
            return input
        axes = g.op("Constant", value_t=torch.tensor(axes))
        starts = g.op("Constant", value_t=torch.tensor(starts))
        ends = g.op("Constant", value_t=torch.tensor(ends))
    if steps is None:
        return g.op("Slice", input, starts, ends, axes)
    steps = g.op("Constant", value_t=torch.tensor(steps))
    return g.op("Slice", input, starts, ends, axes, steps)


def slice(g, self, *args):
    if len(args) == 4:
        # aten::slice(Tensor self, int dim, int? start=None, int? end=None, int step=1) -> Tensor
        dim, start, end, step = args
    elif len(args) == 3:
        # aten::slice(t[] l, int? start=None, int? end=None, int step=1) -> t[]
        start, end, step = args
        dim = 0
    else:
        raise NotImplementedError("Unknown aten::slice signature")
    is_start_none = start.node().kind() == "prim::Constant" and start.type().kind() == 'NoneType'
    is_end_none = end.node().kind() == "prim::Constant" and end.type().kind() == 'NoneType'
    is_start_onnx_const = start.node().kind() == 'onnx::Constant'
    is_end_onnx_const = end.node().kind() == 'onnx::Constant'
    step = sym_help._parse_arg(step, 'i')
    if (not is_start_none and not is_start_onnx_const) or \
       (not isinstance(end, int) and not is_end_none and not is_end_onnx_const) or \
       (not isinstance(dim, int) and dim.node().kind() != 'onnx::Constant'):
        dynamic_slice = True
        if is_start_none:
            start = g.op("Constant", value_t=torch.tensor(0))
        if is_end_none:
            end = g.op("Constant", value_t=torch.tensor(9223372036854775807))
    else:
        start = [0 if is_start_none else sym_help._parse_arg(start, 'i')]
        end = [9223372036854775807 if is_end_none else sym_help._parse_arg(end, 'i')]
        dim = [sym_help._parse_arg(dim, 'i')]
        dynamic_slice = False
    return sym_help._slice_helper(g, self, axes=dim, starts=start, ends=end, steps=[step], dynamic_slice=dynamic_slice)


@parse_args("v", "is")
def flip(g, input, dims):
    return sym_help._slice_helper(g, input, axes=dims,
                                  starts=[-1] * len(dims),
                                  ends=[-9223372036854775807] * len(dims),
                                  steps=[-1] * len(dims))


def fmod(g, input, other):
    return g.op("Mod", input, other, fmod_i=1)


@parse_args("v", "v", "v", "i", "i", "i", "v", "i", "i")
def embedding_bag(g,
                  embedding_matrix,
                  indices,
                  offsets,
                  scale_grad_by_freq,
                  mode,
                  sparse,
                  per_sample_weights,
                  include_last_offset,
                  padding_idx):
    if scale_grad_by_freq and sym_help._training_mode:
        return sym_help._onnx_unsupported("embedding_bag with scale_grad_by_freq for training mode")
    if padding_idx is not None and padding_idx >= 0:
        raise RuntimeError("embedding_bag with padding_idx")
    from torch.onnx.symbolic_opset9 import select
    import warnings
    warnings.warn("Export of embedding_bag with dynamic input/offsets shape is not supported in opset 10. "
                  "Please use opset 11 or higher to export model for dynamic input shape.'")
    offsets_dim_0 = sym_help._get_tensor_dim_size(offsets, 0)
    if offsets_dim_0 is not None:
        if include_last_offset:
            offset_len = offsets_dim_0 - 1
            offsets_extended = offsets
        else:
            offset_len = offsets_dim_0
            offsets_extended = [offsets, g.op("Constant", value_t=torch.tensor([maxsize]))]
            offsets_extended = g.op("Concat", *offsets_extended, axis_i=0)
        list_ = []
        for i in range(offset_len):
            start_ = sym_help._unsqueeze_helper(g, select(g, offsets_extended, torch.tensor(0), torch.tensor(i)), [0])
            end_ = sym_help._unsqueeze_helper(g, select(g, offsets_extended, torch.tensor(0), torch.tensor(i + 1)), [0])
            axes_ = g.op("Constant", value_t=torch.tensor([0]))
            indices_row = g.op("Slice", indices, start_, end_, axes_)

            embeddings = g.op("Gather", embedding_matrix, indices_row)
            if not sym_help._is_none(per_sample_weights):
                per_sample_weights_row = g.op("Slice", per_sample_weights, start_, end_, axes_)
                per_sample_weights_row = sym_help._unsqueeze_helper(g, per_sample_weights_row, [1])
                embeddings = g.op("Mul", embeddings, per_sample_weights_row)
            if mode == 0:
                embeddings = sym_help._reducesum_helper(g, embeddings, axes_i=[0], keepdims_i=0)
            elif mode == 1:
                embeddings = g.op("ReduceMean", embeddings, axes_i=[0], keepdims_i=0)
            else:
                embeddings = g.op("ReduceMax", embeddings, axes_i=[0], keepdims_i=0)

            embeddings = sym_help._unsqueeze_helper(g, embeddings, [0])
            list_.append(embeddings)

        output = g.op("Concat", *list_, axis_i=0)
        # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
        # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
        return output, None, None, None
    else:
        return sym_help._onnx_unsupported("embedding_bag with unknown shape of offsets for opset 10 is not supported. "
                                          "please use opset 11 or higher.")


@parse_args("v", "t", "i", "i", "i")
def fake_quantize_per_tensor_affine(g, inputs, scale, zero_point, quant_min=-128, quant_max=127):
    if quant_min not in [0, -128] or quant_max not in [127, 255]:
        raise RuntimeError(
            "ONNX defines [0, 255] for quint8 and [-128, 127] for qint8, got [{}, {}]".format(quant_min, quant_max))
    scale = scale.float().data  # Avoid exporter generating double type
    zero_point_dtype = torch.int8 if quant_min == -128 else torch.uint8
    zero_point = torch.tensor(zero_point, dtype=zero_point_dtype)  # ONNX requires zero_point to be tensor
    return g.op("DequantizeLinear", g.op("QuantizeLinear", inputs, scale, zero_point), scale, zero_point)

def isinf(g, input):
    from torch.onnx.symbolic_opset9 import _cast_Double  # type: ignore[attr-defined]
    return g.op("IsInf", _cast_Double(g, input, False))
