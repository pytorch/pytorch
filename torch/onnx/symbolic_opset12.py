from __future__ import absolute_import, division, print_function, unicode_literals

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg, _unimplemented

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 12

@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op("Einsum", *tensors, equation_s=equation)

def broadcast_tensors(g, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    if (tensors.type().dim() != 2 and tensors[0].type().sizes() != tensors[1].type().sizes()):
        return _unimplemented("broadcast_tensors", "cannot broadcast")
    #out = [g.op("Expand", t, shape) for t in tensors]
    return g.op("prim::ListConstruct", *tensors)

def mse_loss(g, input, target, reduction):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = sym_help._maybe_get_const(reduction, 'i')
    reduction_vals = ['none', 'mean', 'sum']
    reduction = reduction_vals[reduction]
    return g.op("MeanSquaredDistance", input, target, reduction_s=reduction)
