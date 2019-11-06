import torch
import torch.onnx.symbolic_helper as sym_help

from torch.onnx.symbolic_helper import parse_args, _unimplemented
from torch.onnx.symbolic_helper import _black_list_in_opset


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 11

black_listed_operators = [
    "eq", "ne", "sort", "topk", "hardtanh"
]

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)


def clamp(g, self, min, max):
    dtype = self.type().scalarType()

    def _cast_if_not_none(tensor, dtype):
        if tensor is not None and not tensor.node().mustBeNone():
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
    dim_tensor = g.op("Constant", value_t=torch.tensor(dim))
    csum = g.op("CumSum", self, dim_tensor)
    if dtype and dtype.node().kind() != 'prim::Constant':
        parsed_dtype = sym_help._get_const(dtype, 'i', 'dtype')
        csum = g.op("Cast", csum, to_i=sym_help.scalar_type_to_onnx[parsed_dtype])
    return csum


@parse_args('v', 'i', 'i', 'i')
def _unique2(g, self, sorted, return_inverse, return_counts):
    u, indices, inverse_indices, counts = g.op("Unique", self, sorted_i=sorted, outputs=4)
    return u, inverse_indices, counts


@parse_args('v', 'i', 'i', 'i', 'i')
def unique_dim(g, self, dim, sorted, return_inverse, return_counts):
    u, indices, inverse_indices, counts = g.op("Unique", self, axis_i=dim, sorted_i=sorted, outputs=4)
    return u, inverse_indices, counts


def round(g, self):
    return g.op("Round", self)
