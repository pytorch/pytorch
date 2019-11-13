from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.onnx.symbolic_helper as sym_help
import warnings

from torch.onnx.symbolic_helper import parse_args, _unimplemented
from torch.onnx.symbolic_helper import _black_list_in_opset
from torch.onnx.symbolic_opset9 import expand
from torch.nn.modules.utils import _single, _pair, _triple

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 11

black_listed_operators = [
    "hardtanh"
]


for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)


def clamp(g, self, min, max):
    dtype = self.type().scalarType()

    def _cast_if_not_none(tensor, dtype):
        if tensor is not None and not sym_help._is_none(tensor):
            return g.op("Cast", tensor, to_i=sym_help.cast_pytorch_to_onnx[dtype])
        else:
            return tensor

    if dtype is not None:
        min = _cast_if_not_none(min, dtype)
        max = _cast_if_not_none(max, dtype)
    return g.op("Clip", self, min, max)


@parse_args('v', 'i')
def pixel_shuffle(g, self, upscale_factor):
    dims = self.type().sizes()
    if len(dims) != 4:
        return _unimplemented("pixel_shuffle", "only support 4d input")
    return g.op("DepthToSpace", self, blocksize_i=upscale_factor, mode_s="CRD")

def interpolate_quantized(g, input, output_size, dim, interpolate_mode, align_corners=None):
    align_corners = sym_help._maybe_get_scalar(align_corners)
    coordinate_transformation_mode = "asymmetric" if interpolate_mode == "nearest" \
        else "align_corners" if align_corners else "pytorch_half_pixel"
    empty_tensor = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
    input_size = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
    output_size = g.op("Cast", output_size, to_i=sym_help.cast_pytorch_to_onnx["Long"])
    output_size = g.op("Concat", input_size, output_size, axis_i=0)

    output = g.op("Resize",
                  input,
                  empty_tensor,  # roi only takes effect whith coordinate_transformation_mode="tf_crop_and_resize"
                  empty_tensor,  # scales is not needed since we are sending out_size
                  output_size,
                  coordinate_transformation_mode_s=coordinate_transformation_mode,
                  cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                  mode_s=interpolate_mode,  # nearest, linear, or cubic
                  nearest_mode_s="floor")  # only valid when mode="nearest"
    sym_help._quantized_ops.add(output)
    return output

def _interpolate(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, align_corners=None):
        if input in sym_help._quantized_ops:
            return interpolate_quantized(g, input, output_size, dim, interpolate_mode, align_corners)
        align_corners = sym_help._maybe_get_scalar(align_corners)
        coordinate_transformation_mode = "asymmetric" if interpolate_mode == "nearest" \
            else "align_corners" if align_corners else "pytorch_half_pixel"
        empty_tensor = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        input_size = input.type().sizes()
        input_size = g.op("Constant", value_t=torch.tensor(input_size[0:2], dtype=torch.int64))
        output_size = g.op("Cast", output_size, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        output_size = g.op("Concat", input_size, output_size, axis_i=0)

        return g.op("Resize",
                    input,
                    empty_tensor,  # roi only takes effect whith coordinate_transformation_mode="tf_crop_and_resize"
                    empty_tensor,  # scales is not needed since we are sending out_size
                    output_size,
                    coordinate_transformation_mode_s=coordinate_transformation_mode,
                    cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                    mode_s=interpolate_mode,  # nearest, linear, or cubic
                    nearest_mode_s="floor")  # only valid when mode="nearest"
    return symbolic_fn


upsample_nearest1d = _interpolate('upsample_nearest1d', 3, "nearest")
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, "nearest")
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, "nearest")
upsample_linear1d = _interpolate('upsample_linear1d', 3, "linear")
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, "linear")
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, "linear")
upsample_bicubic2d = _interpolate('upsample_bicubic2d', 4, "cubic")


def __interpolate(g, input, size, scale_factor, mode, align_corners):
    mode = sym_help._maybe_get_const(mode, 's')
    if 'linear' in mode:
        mode = 'linear'
    if 'cubic' in mode:
        mode = 'cubic'
    align_corners = sym_help._maybe_get_const(align_corners, 'b')
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    coordinate_transformation_mode = "asymmetric" if mode == "nearest" \
        else "align_corners" if align_corners else "pytorch_half_pixel"
    # roi only takes effect whith coordinate_transformation_mode="tf_crop_and_resize"
    roi = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))

    if not sym_help._is_none(size) :
        input_size = g.op("Shape", input)
        input_size = sym_help._slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        # in some cases size is not a packed list but size is a scalar
        # We need to also verify that (sym_help._maybe_get_const(size, 't').dim() == 0)
        # but this information is not always available. Try to get the dim,
        # and if not assume that it is not a scalar.
        try:
            is_scalar = not sym_help._is_packed_list(size) and ((sym_help._maybe_get_const(size, 't').dim() == 0))
        except AttributeError:
            is_scalar = not sym_help._is_packed_list(size)
            if not is_scalar:
                warnings.warn("Cannot verify if the output_size is a scalar while exporting interpolate. Assuming that it is not a scalar.")

        if is_scalar:
            if not input.type().dim():
                return sym_help._unimplemented("interpolate (with a scalar output_size)", "missing input shape (try giving an array of output_size values)")
            size = unsqueeze(g, size, 0)
            size = [size for i in range(input.type().dim() - 2)]
            size = g.op("Concat", *size, axis_i=0)
        size = g.op("Concat", input_size, size, axis_i=0)
        scales = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
        return g.op("Resize",
                    input,
                    roi,
                    scales,
                    size,
                    coordinate_transformation_mode_s=coordinate_transformation_mode,
                    cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                    mode_s=mode,  # nearest, linear, or cubic
                    nearest_mode_s="floor")
    else:  # if not sym_help._is_none(scales)
        if not input.type().dim():
            return sym_help._unimplemented("interpolate (with scales)", "missing input shape")
        scales = sym_help._interpolate_get_scales(g, scale_factor, input.type().dim())
        return g.op("Resize",
                    input,
                    roi,
                    scales,
                    coordinate_transformation_mode_s=coordinate_transformation_mode,
                    cubic_coeff_a_f=-0.75,  # only valid when mode="cubic"
                    mode_s=mode,  # nearest, linear, or cubic
                    nearest_mode_s="floor")  # only valid when mode="nearest"

@parse_args('v', 'i', 'v', 'v')
def gather(g, self, dim, index, sparse_grad=False):
    if sym_help._maybe_get_const(sparse_grad, 'i'):
        return _unimplemented("gather", "sparse_grad == True")
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, dim, index, sparse_grad, operator_s="gather")
    return g.op("GatherElements", self, index, axis_i=dim)


@parse_args('v', 'i', 'v', 'v')
def scatter(g, self, dim, index, src):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, dim, index, src, operator_s="scatter")
    return g.op("ScatterElements", self, index, src, axis_i=dim)


@parse_args('v', 'i', 'none')
def cumsum(g, self, dim, dtype=None):
    dim_tensor = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int))
    csum = g.op("CumSum", self, dim_tensor)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        csum = g.op("Cast", csum, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return csum


def masked_select(g, self, mask):
    from torch.onnx.symbolic_opset9 import nonzero, expand_as
    index = nonzero(g, expand_as(g, mask, self))
    return g.op('GatherND', self, index)


def masked_scatter(g, self, mask, source):
    from torch.onnx.symbolic_opset9 import nonzero, expand_as, view, size
    index = nonzero(g, expand_as(g, mask, self))
    # NOTE: source can have more elements than needed.
    # It could also have arbitrary shape.
    # This is not supported by ONNX::ScatterND, so we need to flatten and slice source tensor.
    source = view(g, source, torch.LongTensor([-1]))
    source = sym_help._slice_helper(g, source,
                                    axes=torch.LongTensor([0]),
                                    starts=torch.LongTensor([0]),
                                    ends=size(g, index, torch.LongTensor([0])),
                                    dynamic_slice=True)
    return g.op('ScatterND', self, index, source)


@parse_args('v', 'i', 'i', 'i')
def _unique2(g, self, sorted, return_inverse, return_counts):
    u, indices, inverse_indices, counts = g.op("Unique", self, sorted_i=sorted, outputs=4)
    return u, inverse_indices, counts


def _avg_pool(name, tuple_fn):
    @parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
    def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
        padding = sym_help._avgpool_helper(tuple_fn, padding, kernel_size, stride, divisor_override, name)
        if count_include_pad:
            input = g.op("Pad", input,
                         g.op("Constant", value_t=torch.tensor(((0,) * 2 + padding) * 2)), mode_s='constant')
            padding = (0,) * len(padding)
        output = g.op("AveragePool", input,
                      kernel_shape_i=tuple_fn(kernel_size),
                      strides_i=tuple_fn(stride),
                      pads_i=padding * 2,
                      ceil_mode_i=ceil_mode)
        return output
    return symbolic_fn


avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)


@parse_args('v', 'i', 'i', 'i', 'i')
def unique_dim(g, self, dim, sorted, return_inverse, return_counts):
    u, indices, inverse_indices, counts = g.op("Unique", self, axis_i=dim, sorted_i=sorted, outputs=4)
    return u, inverse_indices, counts


@parse_args('v', 'v', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=None):
    return sym_help._topk_helper(g, self, k, dim, largest=largest, sorted=sorted, out=out)


@parse_args('v', 'i', 'i', 'none')
def sort(g, self, dim, decending, out=None):
    return sym_help._sort_helper(g, self, dim, decending=decending, out=out)


def round(g, self):
    return g.op("Round", self)


# Generate paddings in ONNX order based on pad in pytorch.
# Arguments:
#     dim: the dimension of the tensor.
#     pad: the paddings in pytorch.
#          The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ..., dim_m_begin, dim_m_end,
#          where m is in range [0, n].
def _prepare_onnx_paddings(g, dim, pad):
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    # Assume zero-dimensions in the beginning, pad the "pad" sequence with zeros in the beginning
    pad_len = torch.onnx.symbolic_opset9.size(g, pad, g.op("Constant", value_t=torch.tensor([0])))
    # Set extension = [0] * (dim * 2 - len(pad))
    extension = g.op("Sub", g.op("Mul", g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int64)),
                     g.op("Constant", value_t=torch.tensor(2, dtype=torch.int64))), pad_len)
    # Concat pad with extension: paddings = [dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, 0, 0, ... ]
    # Currently ONNX only supports int64 type for Pad
    pad = g.op("Cast", pad, to_i=sym_help.cast_pytorch_to_onnx['Long'])
    paddings = g.op("Concat", pad, g.op("ConstantOfShape", extension, value_t=torch.tensor([0], dtype=torch.int64)), axis_i=0)
    # Reshape and reverse order and collate first beginnings and then ends
    # paddings = [[..., 0, dim_n-1_begin, dim_n_begin],
    #               [..., 0, dim_n-1_end, dim_n_end]]
    # Reshape back to 1-D paddings = [..., 0, dim_n - 1_begin, dim_n_begin, ..., 0, dim_n - 1_end, dim_n_end]
    paddings = g.op("Reshape", paddings, g.op("Constant", value_t=torch.tensor([-1, 2])))
    paddings = g.op("Transpose", torch.onnx.symbolic_opset10.flip(g, paddings, [0]), perm_i=[1, 0])
    paddings = g.op("Reshape", paddings, g.op("Constant", value_t=torch.tensor([-1])))
    padding_c = g.op("Cast", paddings, to_i=sym_help.cast_pytorch_to_onnx['Long'])
    return padding_c


def constant_pad_nd(g, input, padding, value=None):
    mode = "constant"
    value = sym_help._maybe_get_scalar(value)
    value = sym_help._if_scalar_type_as(g, value, input)
    pad = _prepare_onnx_paddings(g, input.type().dim(), padding)
    return g.op("Pad", input, pad, value, mode_s=mode)


def reflection_pad(g, input, padding):
    mode = "reflect"
    paddings = _prepare_onnx_paddings(g, input.type().dim(), padding)
    return g.op("Pad", input, paddings, mode_s=mode)


def replication_pad(g, input, padding):
    mode = "edge"
    paddings = _prepare_onnx_paddings(g, input.type().dim(), padding)
    return g.op("Pad", input, paddings, mode_s=mode)


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def det(g, self):
    return g.op("Det", self)


def arange(g, *args):
    def _get_arange_dtype(dtype):
        dtype = sym_help._maybe_get_const(dtype, 'i')
        return dtype

    if len(args) == 5:
        # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[1])
        type, end, start, step = sym_help._arange_cast_helper(g, end=args[0], dtype=dtype)
        start_default = g.op("Constant", value_t=torch.tensor(0, dtype=sym_help.scalar_type_to_pytorch_type[type]))
        delta_default = g.op("Constant", value_t=torch.tensor(1, dtype=sym_help.scalar_type_to_pytorch_type[type]))
        arange_tensor = g.op("Range", start_default, end, delta_default)
    elif len(args) == 6:
        # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[2])
        type, end, start, step = sym_help._arange_cast_helper(g, start=args[0], end=args[1], dtype=dtype)
        delta_default = g.op("Constant", value_t=torch.tensor(1, dtype=sym_help.scalar_type_to_pytorch_type[type]))
        arange_tensor = g.op("Range", start, end, delta_default)
    elif len(args) == 7:
        # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[3])
        type, end, start, step = sym_help._arange_cast_helper(g, start=args[0], end=args[1], step=args[2], dtype=dtype)
        arange_tensor = g.op("Range", start, end, step)
    else:
        raise NotImplementedError("Unknown aten::arange signature taking " + str(len(args)) + " arguments.")
    return arange_tensor


@parse_args('v', 'i')
def _dim_arange(g, like, dim):
    like_shape = g.op('Shape', like)
    stop = g.op("Gather", like_shape, g.op("Constant", value_t=torch.tensor(dim)), axis_i=0)
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("_caffe2::Range", stop)
    return arange(g, stop, 4, None, None, None)


def size(g, self, dim):
    return sym_help._size_helper(g, self, dim)


def squeeze(g, self, dim=None):
    if dim is None:
        dims = []
        for i, size in enumerate(self.type().sizes()):
            if size == 1:
                dims.append(i)
    else:
        dims = [sym_help._get_const(dim, 'i', 'dim')]
    return g.op("Squeeze", self, axes_i=dims)


@parse_args('v', 'i')
def unsqueeze(g, self, dim):
    return g.op("Unsqueeze", self, axes_i=[dim])


def mm(g, self, other):
    return g.op("Gemm", self, other, beta_f=0.0, alpha_f=1.0)


def index_fill(g, self, dim, index, value):
    dim_value = sym_help._parse_arg(dim, 'i')
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, index, value, dim_i=dim_value, operator_s="index_fill")
    expanded_index_shape, expanded_index = sym_help._index_fill_reshape_helper(g, self, dim, index)
    value = sym_help._maybe_get_scalar(value)
    value = sym_help._if_scalar_type_as(g, value, self)
    expanded_value = expand(g, value, expanded_index_shape, None)
    return scatter(g, self, dim, expanded_index, expanded_value)


def index_copy(g, self, dim, index, source):
    dim_value = sym_help._parse_arg(dim, 'i')
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, index, source, dim_i=dim_value, operator_s="index_copy")
    expanded_index_shape, expanded_index = sym_help._index_fill_reshape_helper(g, self, dim, index)
    return scatter(g, self, dim, expanded_index, source)
