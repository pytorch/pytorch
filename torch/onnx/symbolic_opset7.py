from torch.onnx.symbolic_helper import _block_list_in_opset, parse_args
import torch.onnx.symbolic_helper as sym_help

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

block_listed_operators = [
    "scan", "expand", "expand_as", "meshgrid",
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


def div(g, self, other, *args):
    if len(args) == 0:
        return sym_opset9.true_divide(g, self, other)
    else:
        return _div_rounding_mode(g, self, other, *args)


@parse_args("v", "v", "s")
def _div_rounding_mode(g, self, other, rounding_mode):
    if rounding_mode == "floor":
        return _floor_divide(g, self, other)
    else:
        return sym_opset9._div_rounding_mode(g, self, other, rounding_mode)


def _floor_divide(g, self, other):
    if sym_help._is_fp(self) or sym_help._is_fp(other):
        out = sym_opset9.true_divide(g, self, other)
        return g.op("Floor", out)
    else:
        raise RuntimeError("Integer floor division requires ONNX opset 9 or greater")


for block_listed_op in block_listed_operators:
    vars()[block_listed_op] = _block_list_in_opset(block_listed_op)
