import torch
import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from torch.onnx.symbolic_helper import parse_args, _unimplemented, _black_list_in_opset
import torch.onnx.symbolic_opset9


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 10
# Opset 10 is supported by ONNX release 1.5.0
# release on 04/24/19


# Blacklist operators for this opset version.
# These operators have been updated in ONNX but not re-implemented here.
# It is very important to blacklist these operators to avoid exporting
# models with mixed versions of operators.
# TODO : add support for the blacklisted operators in black_listed_operators
black_listed_operators = ["flip", "slice", "upsample_nearest2d", "upsample_bilinear2d"]

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)


# Add new operator here
@parse_args('v', 'i', 'i', 'i', 'i')
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")
    k = g.op("Constant", value_t=torch.tensor(k, dtype=torch.int64))
    from torch.onnx.symbolic_opset9 import unsqueeze
    k = unsqueeze(g, k, 0)
    return g.op("TopK", self, k, axis_i=dim, outputs=2)
