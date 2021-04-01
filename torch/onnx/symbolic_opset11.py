
from sys import maxsize

import torch
import torch.onnx.symbolic_helper as sym_help
import warnings
import numpy

from torch.onnx.symbolic_helper import parse_args, _unimplemented, _is_tensor_list
from torch.onnx.symbolic_opset9 import expand, unused
from torch.nn.modules.utils import _single, _pair, _triple
from torch.onnx.utils import _add_block, _add_input_to_block, _add_output_to_block

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 11


@parse_args('v', 'f', 'f')
def hardtanh(g, self, min_val, max_val):
    dtype = self.type().scalarType()
    if dtype is None:
        dtype = 6  # float
    else:
        dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    min_val = g.op("Constant", value_t=torch.tensor(min_val, dtype=sym_help.scalar_type_to_pytorch_type[dtype]))
    max_val = g.op("Constant", value_t=torch.tensor(max_val, dtype=sym_help.scalar_type_to_pytorch_type[dtype]))
    return g.op("Clip", self, min_val, max_val)


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


def clamp_min(g, self, min):
    max = unused(g)
    return clamp(g, self, min, max)


def clamp_max(g, self, max):
    min = unused(g)
    return clamp(g, self, min, max)


# Opset 11 gather accepts negative indices
@parse_args('v', 'i', 'v')
def select(g, self, dim, index):
    return g.op("Gather", self, index, axis_i=dim)


def index_put(g, self, indices_list_value, values, accumulate=False):
    if sym_help._is_packed_list(indices_list_value):
        indices_list = sym_help._unpack_list(indices_list_value)
    else:
        indices_list = [indices_list_value]
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        args = [self] + indices_list + [values, accumulate]
        return g.op("ATen", *args, operator_s='index_put')

    from torch.onnx.symbolic_opset9 import add, expand
    accumulate = sym_help._parse_arg(accumulate, 'b')

    if len(indices_list) == 0:
        return values

    index = indices_list[0]

    if len(indices_list) > 1:
        for ind in indices_list[1:]:
            index = add(g, index, ind)
        broadcast_index_shape = g.op("Shape", index)
        indices_list = [
            sym_help._unsqueeze_helper(g, expand(g, ind, broadcast_index_shape, None), [-1]) for ind in indices_list
        ]
        index = g.op("Concat", *indices_list, axis_i=-1)
    else:
        # Replace index_put node with masked_scatter or masked_fill
        # when inputs to the index_put node contains boolean inputs
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
        bool_inp = index
        if bool_inp.type() is not None and bool_inp.type().scalarType() == 'Bool':
            rank = sym_help._get_tensor_rank(values)
            if rank is not None and rank == 0:
                from torch.onnx.symbolic_opset9 import masked_fill
                return masked_fill(g, self, bool_inp, values)
            return masked_scatter(g, self, bool_inp, values)
        broadcast_index_shape = g.op("Shape", index)
        index = sym_help._unsqueeze_helper(g, index, [-1])
    sub_data_shape = sym_help._slice_helper(
        g, g.op("Shape", self), axes=[0], starts=[len(indices_list)], ends=[maxsize])
    values_shape = g.op("Concat", broadcast_index_shape, sub_data_shape, axis_i=0)
    # Check if values is a singular value and expand accordingly
    rank = sym_help._get_tensor_rank(values)
    if rank is not None and rank == 0:
        values = expand(g, values, values_shape, None)
    values = g.op("Reshape", values, values_shape)

    dtype = self.type().scalarType()
    if dtype is not None and dtype != values.type().scalarType():
        values = g.op("Cast", values, to_i=sym_help.cast_pytorch_to_onnx[dtype])
    dtype = sym_help.scalar_type_to_onnx.index(sym_help.cast_pytorch_to_onnx[dtype])
    dtype = sym_help.scalar_type_to_pytorch_type[dtype]

    if accumulate:
        zeros = g.op("ConstantOfShape", g.op("Shape", self), value_t=torch.tensor([0], dtype=dtype))
        result = g.op("ScatterND", zeros, index, values)
        result = add(g, self, result)
    else:
        result = g.op("ScatterND", self, index, values)

    return result


@parse_args('v', 'i')
def pixel_shuffle(g, self, upscale_factor):
    rank = sym_help._get_tensor_rank(self)
    if rank is not None and rank != 4:
        return _unimplemented("pixel_shuffle", "only support 4d input")
    return g.op("DepthToSpace", self, blocksize_i=upscale_factor, mode_s="CRD")


def _interpolate(name, dim, interpolate_mode):
    return sym_help._interpolate_helper(name, dim, interpolate_mode)


upsample_nearest1d = _interpolate('upsample_nearest1d', 3, "nearest")
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, "nearest")
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, "nearest")
upsample_linear1d = _interpolate('upsample_linear1d', 3, "linear")
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, "linear")
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, "linear")
upsample_bicubic2d = _interpolate('upsample_bicubic2d', 4, "cubic")


def __interpolate(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    return sym_help.__interpolate_helper(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor)

@parse_args('v', 'i', 'v', 'v')
def gather(g, self, dim, index, sparse_grad=False):
    if sym_help._maybe_get_const(sparse_grad, 'i'):
        return _unimplemented("gather", "sparse_grad == True")
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, dim, index, sparse_grad, operator_s="gather")
    return g.op("GatherElements", self, index, axis_i=dim)


@parse_args('v', 'i', 'v', 'v')
def scatter(g, self, dim, index, src):
    from torch.onnx.symbolic_opset9 import expand_as
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, dim, index, src, operator_s="scatter")
    src_type = src.type().scalarType()
    src = sym_help._maybe_get_scalar(src)
    if sym_help._is_value(src):
        return g.op("ScatterElements", self, index, src, axis_i=dim)
    else:
        # Check if scalar 'src' has same type as self (PyTorch allows different
        # type for scalar src (but not when src is tensor)). If not, insert Cast node.
        if self.type().scalarType() != src_type:
            src = g.op("Cast", src, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])
        return g.op("ScatterElements", self, index, expand_as(g, src, index), axis_i=dim)


@parse_args('v', 'i', 'none')
def cumsum(g, self, dim, dtype=None):
    dim_tensor = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int))
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        cast = g.op("Cast", self, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    else:
        cast = self
    csum = g.op("CumSum", cast, dim_tensor)
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


def _len(g, self):
    if _is_tensor_list(self) or self.node().kind() == "onnx::SplitToSequence":
        return g.op("SequenceLength", self)
    sz_0 = size(g, self, g.op("Constant", value_t=torch.LongTensor([0])))
    return sym_help._squeeze_helper(g, sz_0, [0])


def __getitem_(g, self, i):
    if sym_help._is_tensor_list(self):
        # SequenceAt requires that the input be a List of Tensors
        return g.op("SequenceAt", self, i)
    else:
        from torch.onnx.symbolic_opset9 import __getitem_ as getitem
        return getitem(g, self, i)


def append(g, self, tensor):
    return g.op("SequenceInsert", self, tensor)


def add(g, self, other, alpha=None):
    if sym_help._is_value(self) and sym_help._is_tensor_list(self):
        tensor_list_node = other.node()
        if tensor_list_node.kind() != "prim::ListConstruct":
            return _unimplemented("add", "does not support adding dynamic tensor list to another")
        tensors = sym_help._unpack_list(other)
        l = self
        for t in tensors:
            l = g.op("SequenceInsert", l, t)
        return l

    return torch.onnx.symbolic_opset9.add(g, self, other, alpha)

def insert(g, self, pos, tensor):
    return g.op("SequenceInsert", self, tensor, pos)


def pop(g, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)

def Delete(g, tensor_list, dim):
    return g.op("SequenceErase", tensor_list, dim)

def cat(g, tensor_list, dim):
    if sym_help._is_packed_list(tensor_list):
        from torch.onnx.symbolic_opset9 import cat as cat_opset9
        return cat_opset9(g, tensor_list, dim)
    else:
        dim = sym_help._get_const(dim, 'i', 'dim')
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim)


def stack(g, tensor_list, dim):
    if sym_help._is_packed_list(tensor_list):
        from torch.onnx.symbolic_opset9 import stack as stack_opset9
        return stack_opset9(g, tensor_list, dim)
    else:
        dim = sym_help._get_const(dim, 'i', 'dim')
        return g.op("ConcatFromSequence", tensor_list, axis_i=dim, new_axis_i=1)


@parse_args('v', 'i', 'i', 'i')
def _unique2(g, self, sorted, return_inverse, return_counts):
    u, indices, inverse_indices, counts = g.op("Unique", self, sorted_i=sorted, outputs=4)
    return u, inverse_indices, counts


def _avg_pool(name, tuple_fn):
    @parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
    def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
        padding = sym_help._avgpool_helper(tuple_fn, padding, kernel_size, stride, divisor_override, name)
        if not stride:
            stride = kernel_size
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


@parse_args('v', 'v', 'i', 'i')
def split(g, self, split_size_or_sizes, dim, _outputs=None):
    if not sym_help._is_split_static(split_size_or_sizes, _outputs):
        split_out = g.op("SplitToSequence", self, split_size_or_sizes, axis_i=dim)
        if _outputs is None:
            return split_out
        # Convert to multiple slice nodes iff number of splits and number of outputs are statically known.
        if sym_help._is_packed_list(split_size_or_sizes) and len(sym_help._unpack_list(split_size_or_sizes)) == _outputs:
            split_sizes = [sym_help._unsqueeze_helper(g, v, [0]) for v in sym_help._unpack_list(split_size_or_sizes)]
            start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
            axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
            res = []
            for i in range(_outputs):
                end = g.op("Add", start, split_sizes[i])  # split_sizes is a list of same length as _outputs
                res.append(g.op("Slice", self, start, end, axis))
                start = end
            return res
        return [g.op("SequenceAt", split_out, g.op("Constant", value_t=torch.tensor([i], dtype=torch.long)))
                for i in range(_outputs)]
    else:
        return torch.onnx.symbolic_opset9.split(g, self, split_size_or_sizes, dim, _outputs)


@parse_args('v', 'v', 'i', 'i')
def split_with_sizes(g, self, split_sizes, dim, _outputs=None):
    return split(g, self, split_sizes, dim, _outputs)


@parse_args('v', 'i', 'i')
def unbind(g, self, dim=0, _outputs=None):
    if _outputs is None:
        return g.op("SplitToSequence", self, g.op("Constant", value_t=torch.tensor(1, dtype=torch.long)), axis_i=dim, keepdims_i=0)
    else:
        return torch.onnx.symbolic_opset9.unbind(g, self, dim, _outputs)


# Generate paddings in ONNX order based on pad in pytorch.
# Args:
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
    pad = _prepare_onnx_paddings(g, sym_help._get_tensor_rank(input), padding)
    return g.op("Pad", input, pad, value, mode_s=mode)


def reflection_pad(g, input, padding):
    mode = "reflect"
    paddings = _prepare_onnx_paddings(g, sym_help._get_tensor_rank(input), padding)
    return g.op("Pad", input, paddings, mode_s=mode)


def replication_pad(g, input, padding):
    mode = "edge"
    paddings = _prepare_onnx_paddings(g, sym_help._get_tensor_rank(input), padding)
    return g.op("Pad", input, paddings, mode_s=mode)


reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad


def det(g, self):
    return g.op("Det", self)


def logdet(g, input):
    from torch.onnx.symbolic_opset9 import log
    return log(g, det(g, input))


def arange(g, *args):
    def _get_arange_dtype(dtype):
        dtype = sym_help._maybe_get_const(dtype, 'i')
        return dtype

    if len(args) == 2 or len(args) == 5:
        if len(args) == 2:
            # aten::arange(Scalar end, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[1])
        type, end, start, step = sym_help._arange_cast_helper(g, end=args[0], dtype=dtype)
        start_default = g.op("Constant", value_t=torch.tensor(0, dtype=sym_help.scalar_type_to_pytorch_type[type]))
        delta_default = g.op("Constant", value_t=torch.tensor(1, dtype=sym_help.scalar_type_to_pytorch_type[type]))
        arange_tensor = g.op("Range", start_default, end, delta_default)
    elif len(args) == 4 or len(args) == 7:
        if len(args) == 4:
            # aten::arange(Scalar start, Scalar end, Scalar step, Tensor out)
            dtype = None
        else:
            # aten::arange(Scalar start, Scalar end, Scalar step, ScalarType dtype, Layout, Device, bool pin_memory)
            dtype = _get_arange_dtype(args[3])
        type, end, start, step = sym_help._arange_cast_helper(g, start=args[0], end=args[1], step=args[2], dtype=dtype)
        arange_tensor = g.op("Range", start, end, step)
    elif len(args) == 6:
        # aten::arange(Scalar start, Scalar end, ScalarType dtype, Layout, Device, bool pin_memory)
        dtype = _get_arange_dtype(args[2])
        type, end, start, step = sym_help._arange_cast_helper(g, start=args[0], end=args[1], dtype=dtype)
        delta_default = g.op("Constant", value_t=torch.tensor(1, dtype=sym_help.scalar_type_to_pytorch_type[type]))
        arange_tensor = g.op("Range", start, end, delta_default)
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


def size(g, self, dim=None):
    if dim is None:
        return g.op("Shape", self)
    return sym_help._size_helper(g, self, dim)


def squeeze(g, self, dim=None):
    if dim is None:
        return g.op("Squeeze", self)

    dim = sym_help._get_const(dim, 'i', 'dim')

    input_rank = sym_help._get_tensor_rank(self)
    adjusted_dim = dim
    if input_rank is not None and dim < 0:
        adjusted_dim += input_rank
    dim_size = sym_help._get_tensor_dim_size(self, adjusted_dim)
    if (dim < 0 and input_rank is None) or dim_size is None:
        # If onnx shape inference is not on, export always as dynamic.
        # Because we cannot tell if observed static shape is also static at runtime.
        # create 'cond' node (condition is shape[i]==1)
        dim_constant = g.op("Constant", value_t=torch.tensor([dim]))
        size = sym_help._size_helper(g, self, dim_constant)
        const_one = g.op("Constant", value_t=torch.ones(1, dtype=torch.int64))
        cond = g.op("Equal", size, const_one)
        # create the 'If' node and add the 'then' and 'else' blocks to it.
        if_node_outputs = g.op("If", cond)
        if_node = if_node_outputs.node()
        if_block = torch.onnx.utils._add_block(if_node)
        squeeze_ = sym_help._squeeze_helper(if_block, self, [dim])
        torch.onnx.utils._add_output_to_block(if_block, squeeze_)
        else_block = torch.onnx.utils._add_block(if_node)
        identity_ = else_block.op("Identity", self)
        torch.onnx.utils._add_output_to_block(else_block, identity_)
        return if_node_outputs

    # For static input shape
    dim = adjusted_dim
    if dim_size > 1:
        warnings.warn("This model contains a squeeze operation on dimension " + str(dim) + ". The size of " +
                      "this dimension in the given input is " + str(dim_size) + ". The model will " +
                      "be exported without the squeeze node. If the model is intended to be used with dynamic " +
                      "input shapes, please export with dynamic_axes argument.")
        return self
    return sym_help._squeeze_helper(g, self, [dim])


@parse_args('v', 'i')
def unsqueeze(g, self, dim):
    return sym_help._unsqueeze_helper(g, self, [dim])

def mm(g, self, other):
    return g.op("Gemm", self, other, beta_f=0.0, alpha_f=1.0)


def index(g, self, index):
    if sym_help._operator_export_type == torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
        return g.op("ATen", self, index, operator_s="index")

    if sym_help._is_packed_list(index):
        indices = sym_help._unpack_list(index)
    else:
        indices = [index]

    # Handle single mask index.
    if len(indices) == 1:
        index = indices[0]
        if not sym_help._is_none(index) and (index.type().scalarType() == "Bool" or index.type().scalarType() == "Byte"):
            from torch.onnx.symbolic_opset9 import nonzero
            index = nonzero(g, index)
            return g.op('GatherND', self, index)
    from torch.onnx.symbolic_opset9 import index as index_opset9
    return index_opset9(g, self, index)


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


def __rshift_(g, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])

    if self.type().scalarType() == 'Byte':
        return g.op('BitShift', self, other, direction_s="RIGHT")

    two = g.op('Constant', value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not sym_help._is_fp(self):
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx['Float'])
    two_pow = g.op('Pow', two, other)
    two_pow = g.op('Cast', two_pow, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])
    rshift = g.op('Div', self, two_pow)
    return rshift


def __lshift_(g, self, other):
    # make sure to cast other to self's type
    # (when self is long, make sure that other is not float)
    if other.type().scalarType() != self.type().scalarType():
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])

    if self.type().scalarType() == 'Byte':
        return g.op('BitShift', self, other, direction_s="LEFT")

    two = g.op('Constant', value_t=torch.tensor(2, dtype=torch.float32))
    # exponent (same type as self) has to be float or double in onnx::Pow
    if not sym_help._is_fp(self):
        other = g.op("Cast", other, to_i=sym_help.cast_pytorch_to_onnx['Float'])
    two_pow = g.op('Pow', two, other)
    two_pow = g.op('Cast', two_pow, to_i=sym_help.cast_pytorch_to_onnx[self.type().scalarType()])
    lshift = g.op('Mul', self, two_pow)
    return lshift


def _get_im2col_indices_along_dim(g, input_d, kernel_size_d, dilation_d, padding_d, stride_d):
    # Input is always 4-D (N, C, H, W)
    # Calculate indices of sliding blocks along spatial dimension
    # Slide kernel over input each dim d:
    # each dimension d ranges from 0 to input[d]+2xpadding[d]-dilation[d]x(kernel_size[d]-1)
    # with steps = stride

    blocks_d = g.op("Add", input_d, g.op("Constant", value_t=torch.tensor(padding_d * 2)))
    blocks_d = g.op("Sub", blocks_d, g.op("Constant", value_t=torch.tensor(dilation_d * (kernel_size_d - 1))))

    # Stride kernel over input and find starting indices along dim d
    blocks_d_indices = g.op("Range", g.op("Constant", value_t=torch.tensor(0)),
                            blocks_d, g.op("Constant", value_t=torch.tensor(stride_d)))

    # Apply dilation on kernel and find its indices along dim d
    kernel_grid = numpy.arange(0, kernel_size_d * dilation_d, dilation_d)
    kernel_grid = g.op("Constant", value_t=torch.tensor([kernel_grid]))

    # Broadcast and add kernel staring positions (indices) with
    # kernel_grid along dim d, to get block indices along dim d
    blocks_d_indices = sym_help._unsqueeze_helper(g, blocks_d_indices, [0])  # Reshape to [1, -1]
    kernel_mask = g.op('Reshape', kernel_grid, g.op('Constant', value_t=torch.tensor([-1, 1])))
    block_mask = g.op("Add", blocks_d_indices, kernel_mask)

    return block_mask


def _get_im2col_padded_input(g, input, padding_h, padding_w):
    # Input is always 4-D tensor (N, C, H, W)
    # Padding tensor has the following format: (padding_h, padding_w)
    # Reshape the padding to follow ONNX format: (dim1_begin, dim2_begin,...,dim1_end, dim2_end,...)
    pad = g.op("Constant", value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    return g.op("Pad", input, pad)


def _get_im2col_output_shape(g, input, kernel_h, kernel_w):
    batch_dim = size(g, input, g.op("Constant", value_t=torch.tensor(0)))
    channel_dim = size(g, input, g.op("Constant", value_t=torch.tensor(1)))
    channel_unfolded = g.op("Mul", channel_dim,
                            g.op("Constant", value_t=torch.tensor(kernel_h * kernel_w)))

    return g.op("Concat",
                sym_help._unsqueeze_helper(g, batch_dim, [0]),
                sym_help._unsqueeze_helper(g, channel_unfolded, [0]),
                g.op("Constant", value_t=torch.tensor([-1])), axis_i=0)


@parse_args('v', 'is', 'is', 'is', 'is')
def im2col(g, input, kernel_size, dilation, padding, stride):
    # Input is always 4-D tensor (N, C, H, W)
    # All other args are int[2]

    input_h = size(g, input, g.op("Constant", value_t=torch.tensor(2)))
    input_w = size(g, input, g.op("Constant", value_t=torch.tensor(3)))

    stride_h, stride_w = stride[0], stride[1]
    padding_h, padding_w = padding[0], padding[1]
    dilation_h, dilation_w = dilation[0], dilation[1]
    kernel_h, kernel_w = kernel_size[0], kernel_size[1]

    blocks_row_indices = _get_im2col_indices_along_dim(g, input_h, kernel_h, dilation_h, padding_h, stride_h)
    blocks_col_indices = _get_im2col_indices_along_dim(g, input_w, kernel_w, dilation_w, padding_w, stride_w)

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
    return g.op("Reshape", output, output_shape)


def narrow(g, input, dim, start, length):
    from torch.onnx.symbolic_helper import _slice_helper
    end = g.op("Add", start, length)
    return _slice_helper(g, input, axes=dim, starts=start, ends=end, dynamic_slice=True)


@parse_args('v', 'i', 'i')
def flatten(g, input, start_dim, end_dim):
    dim = sym_help._get_tensor_rank(input)
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim == 1:
        if (end_dim == -1 or (dim is not None and end_dim == dim - 1)):
            return g.op("Flatten", input, axis_i=start_dim)
    elif start_dim == 0:
        if (end_dim == -2 or (dim is not None and end_dim == dim - 2)):
            return g.op("Flatten", input, axis_i=end_dim + 1)
    if dim is None:
        return _unimplemented("dim",
                              "ONNX and PyTorch use different strategies to split the input. "
                              "Input rank must be known at export time.")
    # if end_dim is negative add dim
    if end_dim < 0 :
        end_dim = dim + end_dim

    return sym_help._flatten_helper(g, input, start_dim, end_dim, dim)


@parse_args('v', 'v', 'v', 'i', 'i', 'i', 'v', 'i')
def embedding_bag(g,
                  embedding_matrix,
                  indices,
                  offsets,
                  scale_grad_by_freq,
                  mode,
                  sparse,
                  per_sample_weights,
                  include_last_offset):
    if scale_grad_by_freq and sym_help._training_mode:
        return sym_help._onnx_unsupported('embedding_bag with scale_grad_by_freq for training mode')

    loop_condition = g.op("Constant", value_t=torch.tensor(1))
    loop_condition = g.op("Cast", loop_condition, to_i=9)
    zero = g.op("Constant", value_t=torch.tensor([0]))

    indices_len = sym_help._unsqueeze_helper(g,
                                             sym_help._size_helper(g, indices, g.op("Constant", value_t=torch.tensor(0))),
                                             [0])
    if not include_last_offset:
        offsets = [offsets, indices_len]
        offsets = g.op("Concat", *offsets, axis_i=0)

    # Offsets holds the starting index position of each bag. So we create a list of the indices slices (determined by
    # offsets) and gather those indices in indices_row. Then we use this subset of indices to gather from embeddings.
    # The embeddings output is a loop scan output, so we can avoid creating a sequence and inserting elements in.
    offsets_starts = sym_help._slice_helper(g, offsets, axes=[0], starts=[0], ends=[maxsize], steps=[1])
    offsets_ends = sym_help._slice_helper(g, offsets, axes=[0], starts=[1], ends=[maxsize], steps=[1])

    loop_len = sym_help._size_helper(g, offsets_ends, g.op("Constant", value_t=torch.tensor(0)))
    loop = g.op("Loop", loop_len, loop_condition)

    loop_block = _add_block(loop.node())
    block_input_iter = _add_input_to_block(loop_block)
    cond = _add_input_to_block(loop_block)

    indices_start = loop_block.op("Gather", offsets_starts, block_input_iter, axis_i=0)
    indices_end = loop_block.op("Gather", offsets_ends, block_input_iter, axis_i=0)
    indices_start = sym_help._unsqueeze_helper(loop_block, indices_start, [0])
    indices_end = sym_help._unsqueeze_helper(loop_block, indices_end, [0])

    indices_row = loop_block.op("Slice", indices, indices_start, indices_end, zero)
    embeddings = loop_block.op("Gather", embedding_matrix, indices_row, axis_i=0)
    if not sym_help._is_none(per_sample_weights):
        per_sample_weights_row = loop_block.op("Slice", per_sample_weights,
                                               indices_start,
                                               indices_end,
                                               zero)
        per_sample_weights_row = sym_help._unsqueeze_helper(loop_block, per_sample_weights_row, [1])
        embeddings = loop_block.op("Mul", embeddings, per_sample_weights_row)
    if mode == 0:
        embeddings = sym_help._reducesum_helper(loop_block, embeddings, axes_i=[0], keepdims_i=0)
    elif mode == 1:
        embeddings = loop_block.op("ReduceMean", embeddings, axes_i=[0], keepdims_i=0)
    else:
        embeddings = loop_block.op("ReduceMax", embeddings, axes_i=[0], keepdims_i=0)

    cond_out = loop_block.op("Cast", loop_condition, to_i=9)
    _add_output_to_block(loop_block, cond_out)
    _add_output_to_block(loop_block, embeddings)

    # aten::embedding_bag returns a tuple of 4 elements: output, offset2bag, bag_size, max_indices.
    # But the last three outputs are not used in torch.nn.EmbeddingBag or torch.nn.functional.embedding_bag.
    return loop.node().output(), None, None, None


def prim_ConstantChunk(g, self, chunks, dim):
    input_shape = g.op("Shape", self)
    axis = g.op("Constant", value_t=torch.tensor([dim], dtype=torch.long))
    input_shape_dim = g.op("Gather", input_shape, axis, axis_i=0)
    start = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
    chunk_size = g.op("Constant", value_t=torch.tensor([chunks], dtype=torch.long))
    chunk_size_minus_1 = g.op("Constant", value_t=torch.tensor([chunks - 1], dtype=torch.long))
    input_shape_dim_shift = g.op("Add", input_shape_dim, chunk_size_minus_1)
    chunk_dim = g.op("Div", input_shape_dim_shift, chunk_size)
    res = []
    for i in range(chunks):
        index = g.op("Constant", value_t=torch.tensor([i + 1], dtype=torch.long))
        end = g.op("Mul", chunk_dim, index)
        res.append(g.op("Slice", self, start, end, axis))
        start = end
    return res
