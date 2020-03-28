from __future__ import absolute_import, division, print_function, unicode_literals

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args


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
    if sym_help._maybe_get_const(ignore_index, 'i') == -100:
        if weight.node().mustBeNone():
            return g.op("NegativeLogLikelihoodLoss", self, target, reduction_s=reduction)
        else:
            return g.op("NegativeLogLikelihoodLoss", self, target, weight, reduction_s=reduction)

    # if ignore_index is specified, compute nllloss with no reduction and apply the reduction afterwards
    if weight.node().mustBeNone():
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, reduction_s='none')
    else:
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, weight, reduction_s='none')

    from torch.onnx.symbolic_opset9 import zeros_like, ones_like, eq, where, index_select
    zeros = zeros_like(g, nllloss)
    ignored_mask = eq(g, target, ignore_index)
    nllloss = where(g, ignored_mask, zeros, nllloss)

    if reduction == 'none':
        return nllloss

    nllloss = g.op("ReduceSum", nllloss)

    if reduction == 'sum':
        return nllloss

    # reduction == 'mean'
    # if reduction = mean, we want to divide the reduced sum of nllloss
    # by the sum of the non ignored weights (if weights are available),
    # or by the number of non ignored targets (if weights are not available);
    # denominator acts like a mask of which indices to ignore and is then
    # multiplied by weight to set the ignored ones to 0, before summing
    # the values in it
    zeros = zeros_like(g, target)
    ones = ones_like(g, target)
    denominator = where(g, ignored_mask, zeros, ones)
    if not sym_help._is_none(weight):
        # take(weight, target) on 1D tensor weight
        weight = index_select(g, weight, 0, target)
        denominator = g.op("Mul", denominator, weight)

    # denominator is the number of elements if weights are not provided,
    # otherwise it is the sum of the non ignored weights
    denominator = g.op("ReduceSum", denominator)
    nllloss = g.op("Div", nllloss, denominator)
    return nllloss

def nll_loss2d(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)
