import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import _black_list_in_opset

import warnings


# Note [ONNX operators that are added/updated from opset 7 to opset 8]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New operators:
#   Expand
#
# Updated operators:
#   Min, Max, Sum, Mean: supports multidirectional broadcasting.
#   MaxPool: added optional indices output.
#   Scan

black_listed_operators = [
    "scan", "expand", "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"
]


# NOTE: max, min, sum, mean: broadcasting is not supported in opset 7.
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
def max(g, self, dim_or_y=None, keepdim=None):
    # torch.max(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMax", self, keepdims_i=0)
    # torch.max(input, other)
    if keepdim is None:
        warnings.warn("Multidirectional broadcasting is not supported in opset 7. "
                      "This might cause the onnx model to be incorrect, if inputs to max operators "
                      "have different shapes")
        return g.op("Max", self, dim_or_y)
    # torch.max(input, dim, keepdim)
    else:
        dim = sym_help._get_const(dim_or_y, 'i', 'dim')
        keepdim = sym_help._get_const(keepdim, 'i', 'keepdim')
        max = g.op("ReduceMax", self, axes_i=[dim], keepdims_i=keepdim)
        indices = g.op('ArgMax', self, axis_i=dim, keepdims_i=keepdim)
        return max, indices


def min(g, self, dim_or_y=None, keepdim=None):
    # torch.min(input)
    if dim_or_y is None and keepdim is None:
        return g.op("ReduceMin", self, keepdims_i=0)
    # torch.min(input, other)
    if keepdim is None:
        warnings.warn("Multidirectional broadcasting is not supported in opset 7. "
                      "This might cause the onnx model to be incorrect, if inputs to min operators "
                      "have different shapes")
        return g.op("Min", self, dim_or_y)
    # torch.min(input, dim, keepdim)
    else:
        dim = sym_help._get_const(dim_or_y, 'i', 'dim')
        keepdim = sym_help._get_const(keepdim, 'i', 'keepdim')
        min = g.op("ReduceMin", self, axes_i=[dim], keepdims_i=keepdim)
        indices = g.op('ArgMin', self, axis_i=dim, keepdims_i=keepdim)
        return min, indices

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)
