from torch.onnx.symbolic_helper import _black_list_in_opset

import torch.onnx.symbolic_opset9 as sym_opset9

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
    "scan", "expand", "expand_as",
    "adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d",
    "max_pool1d_with_indices", "max_pool2d_with_indices", "max_pool3d_with_indices"
]


# NOTE: max, min, sum, mean: broadcasting is not supported in opset 7.
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
def max(g, self, dim_or_y=None, keepdim=None):
    # torch.max(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn("Multidirectional broadcasting is not supported in opset 7. "
                      "This might cause the onnx model to be incorrect, if inputs to max operators "
                      "have different shapes")
    return sym_opset9.max(g, self, dim_or_y, keepdim)


def min(g, self, dim_or_y=None, keepdim=None):
    # torch.min(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn("Multidirectional broadcasting is not supported in opset 7. "
                      "This might cause the onnx model to be incorrect, if inputs to min operators "
                      "have different shapes")
    return sym_opset9.min(g, self, dim_or_y, keepdim)


for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)
