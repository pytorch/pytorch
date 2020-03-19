from __future__ import absolute_import, division, print_function, unicode_literals

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
import torch


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 12

@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op("Einsum", *tensors, equation_s=equation)

def nll_loss(g, self, target, weight, reduction, ignore_index):
    # none reduction : onnx::Constant[value={0}]
    # mean reduction : onnx::Constant[value={1}]
    # sum reduction : onnx::Constant[value={2}]
    reduction = sym_help._maybe_get_const(reduction, 'i')
    reduction_vals = ['none', 'mean', 'sum']
    reduction = reduction_vals[reduction]

    # when ignore_index is not specified, ignore_index == onnx::Constant[value={-100}]
    ignore_index = sym_help._maybe_get_const(ignore_index, 'i')
    if ignore_index == -100:
        if weight.node().mustBeNone():
            return g.op("NegativeLogLikelihoodLoss", self, target, reduction_s=reduction)
        else:
            return g.op("NegativeLogLikelihoodLoss", self, target, weight, reduction_s=reduction)

    # ignore_index specifies a target value that is ignored (not the index of the ignored value),
    # if ignore_index is specified, create a not-equal mask on the target and
    # set targets and weights to 0 for the ignored index

    ignore_index = g.op("Constant", value_t=torch.tensor(ignore_index, dtype=torch.int64))
    mask = g.op("Equal", target, ignore_index)
    mask = g.op("Not", mask)

    target_mask = g.op("Cast", mask, to_i=7)  # INT64
    target = g.op("Mul", target, target_mask)

    weight_mask = g.op("Cast", mask, to_i=1)  # FLOAT
    if weight.node().mustBeNone():
        weight = weight_mask
    else:
        weight = g.op("Mul", weight, weight_mask)

    return g.op("NegativeLogLikelihoodLoss", self, target, weight, reduction_s=reduction)

def nll_loss2d(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)
