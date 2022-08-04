"""
Note [ONNX operators that are added/updated from opset 8 to opset 9]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    Compress
    ConstantOfShape
    EyeLike
    MaxUnpool
    OneHot
    Sinh
    Cosh
    Asinh
    Acosh
    Atanh
    Shrink
    IsNaN
    Sign
    Erf
    Scatter
    Where
    NonZero
    TfIdfVectorizer
    MeanVarianceNormalization

Updated operators:
    BatchNormalization: removed spatial attribute.
    Greater, Less, Constant, MatMul, PRelu, Gemm, Flatten: more data types{integers} supported.
    Cast: more data types{string} supported.
    Upsample: moved scales from attribute to input.
    Scan
"""

import warnings

import torch
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9

block_listed_operators = [
    "nonzero",
    "where",
    "scatter",
    "scatter_add",
    "erf",
    "sign",
    "isnan",
    "gather",
    "arange",
    "masked_fill",
    "index_fill",
    "index_copy",
    "repeat_interleave",
    "isnan",
    "any",
    "all",
]

for block_listed_op in block_listed_operators:
    vars()[block_listed_op] = symbolic_helper._block_list_in_opset(block_listed_op)
    vars()[block_listed_op].__module__ = "torch.onnx.symbolic_opset8"


def _interpolate(name, dim, interpolate_mode):
    def symbolic_fn(g, input, output_size, *args):
        scales, align_corners = symbolic_helper._get_interpolate_attributes(
            g, interpolate_mode, args
        )
        symbolic_helper._interpolate_warning(interpolate_mode)
        align_corners = symbolic_helper._maybe_get_scalar(align_corners)
        if align_corners:
            return symbolic_helper._unimplemented(name, "align_corners == True")
        output_size = symbolic_helper._maybe_get_const(output_size, "is")
        if symbolic_helper._is_value(output_size):
            return symbolic_helper._unimplemented(
                name, "torch._C.Value (output_size) indexing"
            )
        if scales is None:
            scales = [
                1.0
                if i < 2
                else float(output_size[-(dim - i)])
                / float(input.type().sizes()[-(dim - i)])
                for i in range(0, dim)
            ]
        return g.op("Upsample", input, mode_s=interpolate_mode, scales_f=scales)

    return symbolic_fn


upsample_nearest1d = _interpolate("upsample_nearest1d", 3, "nearest")
upsample_nearest2d = _interpolate("upsample_nearest2d", 4, "nearest")
upsample_nearest3d = _interpolate("upsample_nearest3d", 5, "nearest")
upsample_linear1d = _interpolate("upsample_linear1d", 3, "linear")
upsample_bilinear2d = _interpolate("upsample_bilinear2d", 4, "linear")
upsample_trilinear3d = _interpolate("upsample_trilinear3d", 5, "linear")


def __interpolate(
    g, input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias
):
    align_corners = symbolic_helper._maybe_get_const(align_corners, "b")
    if not symbolic_helper._is_none(align_corners) and align_corners:
        return symbolic_helper._unimplemented("interpolate", "align_corners == True")

    if not symbolic_helper._is_none(scale_factor) and symbolic_helper._is_value(
        scale_factor
    ):
        return symbolic_helper._unimplemented(
            "interpolate", "dynamic scales in opset 8"
        )

    if not symbolic_helper._is_none(size) and symbolic_helper._is_value(size):
        return symbolic_helper._unimplemented("interpolate", "dynamic size in opset 8")

    scales, mode = symbolic_helper._interpolate_get_scales_and_mode(
        g, input, size, scale_factor, mode, align_corners
    )
    return g.op("Upsample", input, mode_s=mode, scales_f=scales)


# NOTE: We should create a wrapper for this kind of operation, after resolving the shape/type propagation
#       issue for "cast" operators. Some symbolic functions depend on shape information of input tensor, which
#       is lost after casting.
def _try_cast_integer_to_float(g, *args):
    floating_scalar_types = ["Half", "Float", "Double"]
    old_type = None
    # Cast the input tensor to Float if its scalarType is known and is not floating number.
    # If casting is performed, return the old scalarType, otherwise return None.
    arg0_type = args[0].type().scalarType()
    if arg0_type is not None:
        old_type = arg0_type
        if old_type not in floating_scalar_types:
            # TODO(justinchuby): Remove the type ignore hint once _cast_Float is
            # properly defined.
            # NOTE: _cast_Float is generated programmatically so we need to make the
            # type checker happy with ignore[attr-defined].
            args = tuple(opset9._cast_Float(g, arg, False) for arg in args)  # type: ignore[attr-defined]
        else:
            return (None,) + args
    else:
        warnings.warn(
            "Only floating datatype is supported for these operators: "
            "{Greater, Less, MatMul, PRelu, Gemm, Flatten}. This might cause "
            "the onnx model to be incorrect, if inputs have integer datatypes."
        )
    return (old_type,) + args


def _cast_to_type(g, input, to_type):
    if to_type is None:
        return input
    return getattr(opset9, f"_cast_{to_type}")(g, input, False)


def _comparison_operator(g, input, other, op_name):
    other = symbolic_helper._maybe_get_scalar(other)
    other = symbolic_helper._if_scalar_type_as(g, other, input)
    _, input, other = _try_cast_integer_to_float(g, input, other)
    return g.op(op_name, input, other)


# NOTE: For symbolics {gt, lt, bmm, matmul, prelu, mm, addmm, view, flatten},
#       integer input type not supported in opset8. Cast to float if possible.
def gt(g, input, other):
    return _comparison_operator(g, input, other, "Greater")


def lt(g, input, other):
    return _comparison_operator(g, input, other, "Less")


def bmm(g, self, other):
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, other = _try_cast_integer_to_float(g, self, other)
        return _cast_to_type(g, g.op("MatMul", self, other), old_type)
    else:
        return g.op("MatMul", self, other)


def matmul(g, self, other):
    return bmm(g, self, other)


def prelu(g, self, weight):
    self_rank = symbolic_helper._get_tensor_rank(self)
    weight_sizes = symbolic_helper._get_tensor_sizes(weight)
    if self_rank is not None and self_rank > 2:
        weight = g.op("Unsqueeze", weight, axes_i=list(range(1, self_rank - 1)))
    elif self_rank == 0 and weight_sizes == [1]:
        # self and weight are both scalar but weight has rank == 1, squeeze weight.
        weight = symbolic_helper._squeeze_helper(g, weight, [0])
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, weight = _try_cast_integer_to_float(g, self, weight)
        return _cast_to_type(g, g.op("PRelu", self, weight), old_type)
    else:
        return g.op("PRelu", self, weight)


def mm(g, self, other):
    # Create a dummy C tensor. Only needed for API purposes, the value is
    # since beta = 0
    ty = symbolic_helper._try_get_scalar_type(self, other).lower()
    C = g.constant(0, [1], ty)
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, other, C = _try_cast_integer_to_float(g, self, other, C)
        return _cast_to_type(
            g, g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0), old_type
        )
    else:
        return g.op("Gemm", self, other, C, beta_f=0.0, alpha_f=1.0)


@symbolic_helper.parse_args("v", "v", "v", "t", "t")
def addmm(g, self, mat1, mat2, beta, alpha):
    if symbolic_helper._try_get_scalar_type(self):
        old_type, self, mat1, mat2 = _try_cast_integer_to_float(g, self, mat1, mat2)
        return _cast_to_type(
            g,
            g.op(
                "Gemm",
                mat1,
                mat2,
                self,
                beta_f=symbolic_helper._scalar(beta),
                alpha_f=symbolic_helper._scalar(alpha),
            ),
            old_type,
        )
    else:
        return g.op(
            "Gemm",
            mat1,
            mat2,
            self,
            beta_f=symbolic_helper._scalar(beta),
            alpha_f=symbolic_helper._scalar(alpha),
        )


def flatten(g, input, start_dim, end_dim):
    start_dim_i = symbolic_helper._get_const(start_dim, "i", "start_dim")
    end_dim_i = symbolic_helper._get_const(end_dim, "i", "end_dim")

    dim = input.type().dim()
    if end_dim_i < 0:
        end_dim_i = dim + end_dim_i
    # use ONNX's Flatten operator for cases where the output shape is 2D
    if start_dim_i == 1 and end_dim_i == dim - 1:
        if symbolic_helper._try_get_scalar_type(input):
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(
                g, g.op("Flatten", input, axis_i=start_dim_i), old_type
            )
        else:
            return g.op("Flatten", input, axis_i=start_dim_i)
    if start_dim_i == 0 and end_dim_i == dim - 2:
        if symbolic_helper._try_get_scalar_type(input):
            old_type, input = _try_cast_integer_to_float(g, input)
            return _cast_to_type(
                g, g.op("Flatten", input, axis_i=end_dim_i + 1), old_type
            )
        else:
            return g.op("Flatten", input, axis_i=end_dim_i + 1)

    return opset9.flatten(g, input, start_dim, end_dim)


def _constant_fill(g, sizes, dtype, const_value):
    if dtype is None:
        dtype = symbolic_helper.ScalarType.FLOAT
    if not symbolic_helper.scalar_type_to_pytorch_type[dtype].is_floating_point:
        result = g.op(
            "ConstantFill",
            sizes,
            dtype_i=symbolic_helper.cast_pytorch_to_onnx["Float"],
            input_as_shape_i=1,
            value_f=const_value,
        )
        return symbolic_helper._cast_func_template(
            symbolic_helper.scalar_type_to_onnx[dtype], g, result, None
        )
    else:
        return g.op(
            "ConstantFill",
            sizes,
            dtype_i=symbolic_helper.scalar_type_to_onnx[dtype],
            input_as_shape_i=1,
            value_f=const_value,
        )


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty(g, sizes, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros(g, sizes, dtype, layout, device, pin_memory)


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def empty_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros_like(g, input, dtype, layout, device, pin_memory)


@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def zeros(g, sizes, dtype, layout, device, pin_memory=False):
    # NOTE: no way to set device and layout in ONNX, so we ignore it
    return _constant_fill(g, sizes, dtype, 0)


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def zeros_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, 0)


@symbolic_helper.parse_args("v", "i", "v", "v", "v")
def ones(g, sizes, dtype, layout, device, pin_memory=False):
    return _constant_fill(g, sizes, dtype, 1)


@symbolic_helper.parse_args("v", "i", "v", "v", "v", "v")
def ones_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, 1)


def full(g, sizes, value, dtype, layout, device, pin_memory=False):
    const_value = symbolic_helper._maybe_get_const(value, "t")
    if symbolic_helper._is_value(const_value):
        tmp = zeros(g, sizes, dtype, layout, device)
        return opset9.add(g, tmp, value, g.op("Constant", value_t=torch.tensor(1)))
    else:
        dtype = symbolic_helper._get_const(dtype, "i", "dtype")
        return _constant_fill(g, sizes, dtype, const_value)


@symbolic_helper.parse_args("v", "f", "i", "v", "v", "v", "v")
def full_like(
    g, input, fill_value, dtype, layout, device, pin_memory=False, memory_format=None
):
    shape = g.op("Shape", input)
    return _constant_fill(g, shape, dtype, fill_value)


def repeat(g, self, repeats):
    if not symbolic_helper._is_value(repeats):
        repeats = g.op("Constant", value_t=torch.LongTensor(repeats))
    if symbolic_helper._is_packed_list(repeats):
        repeat_size_len = len(symbolic_helper._unpack_list(repeats))
    else:
        const_repeats = symbolic_helper._maybe_get_const(repeats, "is")
        repeat_size_len = len(const_repeats)
    if self.isCompleteTensor():
        sizes = self.type().sizes()
        diff_dims = repeat_size_len - len(sizes)
        if diff_dims > 0:
            self = opset9.view(
                g, self, g.op("Constant", value_t=torch.tensor([1] * diff_dims + sizes))
            )
    return g.op("Tile", self, repeats)
