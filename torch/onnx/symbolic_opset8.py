import torch
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_opset9 as sym_opset9

from torch.onnx.symbolic_helper import parse_args, _unimplemented, _black_list_in_opset, _try_get_scalar_type
from torch.onnx.symbolic_opset9 import wrap_logical_op_with_cast_to, _cast_Float

import warnings

# Note [ONNX operators that are added/updated from opset 8 to opset 9]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New operators:
#   Compress
#   ConstantOfShape
#   EyeLike
#   MaxUnpool
#   OneHot
#   Sinh
#   Cosh
#   Asinh
#   Acosh
#   Atanh
#   Shrink
#   IsNaN
#   Sign
#   Erf
#   Scatter
#   Where
#   NonZero
#   TfIdfVectorizer
#   MeanVarianceNormalization
#
# Updated operators:
#   BatchNormalization: removed spatial attribute.
#   Greater, Less, Constant, MatMul, PRelu, Gemm, Flatten: more data types{integers} supported.
#   Cast: more data types{string} supported.
#   Upsample: moved scales from attribute to input.
#   Scan

black_listed_operators = [
    "nonzero", "where", "scatter", "scatter_add", "erf", "sign", "isnan", "gather",
]

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)


def upsample_nearest2d(g, input, output_size, align_corners=None):
    align_corners = sym_help._maybe_get_scalar(align_corners)
    if align_corners:
        return _unimplemented("upsample_neareset2d", "align_corners == True")

    output_size = sym_help._maybe_get_const(output_size, 'is')
    if sym_help._is_value(output_size):
        return _unimplemented("upsample_nearest2d", "torch._C.Value (output_size) indexing")
    else:
        height_scale = float(output_size[-2]) / input.type().sizes()[-2]
        width_scale = float(output_size[-1]) / input.type().sizes()[-1]
        scales = [1., 1., height_scale, width_scale]
        return g.op("Upsample", input, mode_s="nearest",
                    scales_f=scales)


# NOTE: We should create a wrapper for this kind of operation, after resolving the shape/type propagation
#       issue for "cast" operators. Some symbolic functions depend on shape information of input tensor, which
#       is lost after casting.
def _try_cast_integer_to_float(g, *args):
    floating_scalar_types = ['Half', 'Float', 'Double']
    old_type = None
    # Cast the input tensor to Float if its scalarType is known and is not floating number.
    # If casting is performed, return the old scalarType, otherwise return None.
    if args[0].type().kind() == "DimensionedTensorType" or args[0].type().kind() == "CompleteTensorType":
        old_type = args[0].type().scalarType()
        if old_type not in floating_scalar_types:
            args = tuple(_cast_Float(g, arg, False) for arg in args)
        else:
            return (None,) + args
    else:
        warnings.warn("Only floating datatype is supported for these operators: "
                      "{Greater, Less, MatMul, PRelu, Gemm, Flatten}. This might cause "
                      "the onnx model to be incorrect, if inputs have integer datatypes.")
    return (old_type,) + args


def _cast_to_type(g, input, to_type):
    if to_type is None:
        return input
    return getattr(sym_opset9, '_cast_{}'.format(to_type))(g, input, False)


def _comparison_operator(g, input, other, op_name):
    other = sym_help._maybe_get_scalar(other)
    other = sym_help._if_scalar_type_as(g, other, input)
    _, input, other = _try_cast_integer_to_float(g, input, other)
    return g.op(op_name, input, other)


# NOTE: For symbolics {gt, lt, bmm, matmul, prelu, mm, addmm, view, flatten},
#       integer input type not supported in opset8. Cast to float if possible.
@wrap_logical_op_with_cast_to('Byte')
def gt(g, input, other):
    return _comparison_operator(g, input, other, "Greater")


@wrap_logical_op_with_cast_to('Byte')
def lt(g, input, other):
    return _comparison_operator(g, input, other, "Less")


def bmm(g, self, other):
    if _try_get_scalar_type(self):
        old_type, self, other = _try_cast_integer_to_float(g, self, other)
        return _cast_to_type(g, g.op("MatMul", self, other), old_type)
    else:
        return g.op("MatMul", self, other)


def matmul(g, self, other):
    return bmm(g, self, other)


def prelu(g, self, weight):
    if self.isCompleteTensor():
        self_sizes = self.type().sizes()
        if self_sizes and len(self_sizes) > 2:
            weight = g.op("Unsqueeze", weight, axes_i=list(range(1, len(self_sizes) - 1)))
    if _try_get_scalar_type(self):
        old_type, self, weight = _try_cast_integer_to_float(g, self, weight)
        return _cast_to_type(g, g.op("PRelu", self, weight), old_type)
    else:
        return g.op("PRelu", self, weight)


def mm(g, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    ty = sym_help._try_get_scalar_type(self, other).lower()
    C = g.constant(0, [1], ty)
    if _try_get_scalar_type(self):
        old_type, self, other, C = _try_cast_integer_to_float(g, self, other, C)
        return _cast_to_type(g, g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0), old_type)
    else:
        return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0)


@parse_args('v', 'v', 'v', 't', 't')
def addmm(g, self, mat1, mat2, beta, alpha):
    if _try_get_scalar_type(self):
        old_type, self, mat1, mat2 = _try_cast_integer_to_float(g, self, mat1, mat2)
        return _cast_to_type(
            g, g.op("Gemm", mat1, mat2, self,
                    beta_f=sym_help._scalar(beta), alpha_f=sym_help._scalar(alpha)), old_type)
    else:
        return g.op("Gemm", mat1, mat2, self, beta_f=sym_help._scalar(beta), alpha_f=sym_help._scalar(alpha))


def view(g, self, size):
    size = sym_help._maybe_get_const(size, 'is')
    if sym_help._is_value(size):
        shape = size
    else:
        if self.isCompleteTensor():
            self_sizes = self.type().sizes()
            if self_sizes and len(size) == 2 and self_sizes[0] == size[0]:
                old_type, self = _try_cast_integer_to_float(g, self)
                return _cast_to_type(g, g.op("Flatten", self, axis_i=1), old_type)
        shape = g.op("Constant", value_t=torch.LongTensor(size))
    return g.op("Reshape", self, shape)


def flatten(g, input, start_dim, end_dim):
    start_dim_i = sym_help._get_const(start_dim, 'i', 'start_dim')
    end_dim_i = sym_help._get_const(end_dim, 'i', 'end_dim')

    dim = input.type().dim()
    if end_dim_i < 0 :
        end_dim_i = dim + end_dim_i
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim_i == 1 and end_dim_i == dim - 1 :
        if _try_get_scalar_type(input):
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(g, g.op("Flatten", input, axis_i=start_dim_i), old_type)
        else:
            return g.op("Flatten", input, axis_i=start_dim_i)
    if start_dim_i == 0 and end_dim_i == dim - 2 :
        if _try_get_scalar_type(input):
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(g, g.op("Flatten", input, axis_i=end_dim_i + 1), old_type)
        else:
            return g.op("Flatten", input, axis_i=end_dim_i + 1)

    return sym_opset9.flatten(g, input, start_dim, end_dim)


def _constant_fill(g, sizes, dtype, const_value):
    if not sym_help.scalar_type_to_pytorch_type[dtype].is_floating_point:
        result = g.op(
            "ConstantFill", sizes, dtype_i=sym_help.cast_pytorch_to_onnx["Float"], input_as_shape_i=1, value_f=const_value)
        return sym_help._cast_func_template(sym_help.scalar_type_to_onnx[dtype], g, result, None)
    else:
        return g.op("ConstantFill", sizes, dtype_i=sym_help.scalar_type_to_onnx[dtype], input_as_shape_i=1, value_f=const_value)


@parse_args('v', 'i', 'v', 'v', 'v')
def zeros(g, sizes, dtype, layout, device, pin_memory=False):
    # NOTE: no way to set device and layout in ONNX, so we ignore it
    return _constant_fill(g, sizes, dtype, 0)


@parse_args('v', 'i', 'v', 'v', 'v')
def zeros_like(g, input, dtype, layout, device, pin_memory=False):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, 0)


@parse_args('v', 'i', 'v', 'v', 'v')
def ones(g, sizes, dtype, layout, device, pin_memory=False):
    return _constant_fill(g, sizes, dtype, 1)


@parse_args('v', 'i', 'v', 'v', 'v')
def ones_like(g, input, dtype, layout, device, pin_memory=False):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, 1)


def full(g, sizes, value, dtype, layout, device, pin_memory=False):
    const_value = sym_help._maybe_get_const(value, 't')
    if sym_help._is_value(const_value):
        tmp = zeros(g, sizes, dtype, layout, device)
        return sym_opset9.add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = sym_help._get_const(dtype, 'i', 'dtype')
        return _constant_fill(g, sizes, dtype, const_value)


@parse_args('v', 'f', 'i', 'v', 'v', 'v')
def full_like(g, input, fill_value, dtype, layout, device, pin_memory=False):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, fill_value)
