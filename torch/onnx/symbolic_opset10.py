import torch
import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from torch.onnx.symbolic_helper import parse_args, _unimplemented
import torch.onnx.symbolic_opset9


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files]


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
