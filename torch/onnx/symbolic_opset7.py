"""
Note [ONNX operators that are added/updated from opset 7 to opset 8]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
  Expand

Updated operators:
  Min, Max, Sum, Mean: supports multidirectional broadcasting.
  MaxPool: added optional indices output.
  Scan
"""

import functools
import warnings

from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import jit_utils, registration


_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=7)

block_listed_operators = (
    "scan",
    "expand",
    "expand_as",
    "meshgrid",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
)


# NOTE: max, min, sum, mean: broadcasting is not supported in opset 7.
# torch.max (same for torch.min) actually has two interfaces smashed together:
# torch.max(x, dim, keepdim) and torch.max(x, y)
@_onnx_symbolic("aten::max")
def max(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.max(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to max operators "
            "have different shapes",
            stacklevel=2,
        )
    return opset9.max(g, self, dim_or_y, keepdim)


@_onnx_symbolic("aten::min")
def min(g: jit_utils.GraphContext, self, dim_or_y=None, keepdim=None):
    # torch.min(input, other)
    if keepdim is None and dim_or_y is not None:
        warnings.warn(
            "Multidirectional broadcasting is not supported in opset 7. "
            "This might cause the onnx model to be incorrect, if inputs to min operators "
            "have different shapes",
            stacklevel=2,
        )
    return opset9.min(g, self, dim_or_y, keepdim)


for block_listed_op in block_listed_operators:
    _onnx_symbolic(f"aten::{block_listed_op}")(
        symbolic_helper._block_list_in_opset(block_listed_op)
    )
