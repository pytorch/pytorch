from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 12

black_listed_operators = [
    "ArgMin", "ArgMax"
]

@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op("Einsum", *tensors, equation_s=equation)


@parse_args('v', 'f', 'i')
def dropout(g, input, p, train):
    sym_help.assert_training_mode(train, "dropout")
    # in eval mode, dropout is non-op - if the node's train param is set to False, dropout is non-op
    if not sym_help._training_mode:
        return input
    p = g.op("Constant", value_t=torch.tensor(p))
    r, _ = g.op("Dropout", input, p, outputs=2)
    return r


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

    # if ignore_index is specified, compute nllloss with no reduction and apply the reduction afterwards
    if weight.node().mustBeNone():
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, reduction_s=reduction, ignore_index_i=ignore_index)
    else:
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, weight, reduction_s=reduction, ignore_index_i=ignore_index)

    return nllloss

def nll_loss2d(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)
