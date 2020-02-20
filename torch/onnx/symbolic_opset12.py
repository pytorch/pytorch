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

@parse_args('s', 'v', 'v')
def crossentropyloss(g, input, target, weight, reduction, ignore_index):
    
    loss = g.op("SoftmaxCrossEntropyLoss", input, target, reduction_s='none')
    if not sym_help._is_none(weight):
        loss = g.op("SoftmaxCrossEntropyLoss", input, target, weight, reduction_s='none')

    # when ignore_index is not specified, ignore_index == onnx::Constant[value={-100}]
    if sym_help._maybe_get_const(ignore_index, 'i') == -100:
        return loss

    # if ignore_index
    zeros = zeros_like(g, loss)
    ignored_mask = eq(g, target, ignore_index)
    loss = where(g, ignored_mask, zeros, loss)

    if reduction == 'sum' or reduction == 'mean':
        zeros = zeros_like(g, target)
        ones = ones_like(g, target)
        nb_elem = where(g, ignored_mask, zeros, ones)
        if not sym_help._is_none(weight):
            weight = index_select(g, weight_flattened, 0, target)
            weight = reshape_as(g, weight, target)

            nb_elem = g.op("Mul", nb_elem, weight)
        
        nb_elem = g.op("ReduceSum", nb_elem)
        loss = g.op("ReduceSum", nllloss)

        if reduction == 'mean':
            loss = g.op("Div", nllloss, nb_elem)

    return loss
