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

    if weight.node().mustBeNone():
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, reduction_s=reduction)
    else:
        nllloss = g.op("NegativeLogLikelihoodLoss", self, target, weight, reduction_s=reduction)

    # when ignore_index is not specified, ignore_index == onnx::Constant[value={-100}]
    if sym_help._maybe_get_const(ignore_index, 'i') == -100:
        return nllloss

    # if ignore_index
    zeros = zeros_like(g, nllloss)
    ignored_mask = eq(g, target, ignore_index)
    nllloss = where(g, ignored_mask, zeros, nllloss)

    if reduction == 'sum' or reduction == 'mean':
        zeros = zeros_like(g, target)
        ones = ones_like(g, target)
        nb_elem = where(g, ignored_mask, zeros, ones)
        if not sym_help._is_none(weight):
            # take(weight, target)
            weight_flattened = g.op('Reshape', weight, g.op("Constant", value_t=torch.tensor([-1], dtype=torch.int64)))
            weight = index_select(g, weight_flattened, 0, target)
            weight = reshape_as(g, weight, target)

            nb_elem = g.op("Div", nb_elem, weight)

        nb_elem = g.op("ReduceSum", nb_elem)
        nllloss = g.op("ReduceSum", nllloss)

        if reduction == 'mean':
            nllloss = g.op("Div", nllloss, nb_elem)
    return nllloss

def nll_loss2d(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)
