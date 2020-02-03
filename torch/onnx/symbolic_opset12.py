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
    # in eval mode, dropout is non-op - if the node's train param is set to False, dropout is non-op
    if not sym_help._training_mode or not train:
        return input
    p = g.op("Constant", value_t=torch.tensor(p))
    return g.op("Dropout", input, p, outputs=1)


@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    input_sizes = input.type().sizes()

    if weight is None or sym_help._is_none(weight):
        assert len(input_sizes) > 1
        weight_value = torch.tensor([1.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or sym_help._is_none(bias):
        assert len(input_sizes) > 1
        bias_value = torch.tensor([0.] * input_sizes[1]).type(
            'torch.' + input.type().scalarType() + 'Tensor')
        bias = g.op("Constant", value_t=bias_value)

    if not sym_help._training_mode or not training:
        out = g.op("BatchNormalization", input, weight, bias, running_mean, running_var,
                   epsilon_f=eps,
                   momentum_f=1 - momentum,
                   outputs=1)
        return out
    else:
        training_mode = g.op("Constant", value_t=torch.tensor(True))
        res, new_running_mean, new_running_var, saved_mean, saved_var = g.op("BatchNormalization",
                                                                             input,
                                                                             weight, bias,
                                                                             running_mean, running_var, training_mode,
                                                                             epsilon_f=eps,
                                                                             momentum_f=1 - momentum,
                                                                             outputs=5)
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        saved_mean.setDebugName("batch_norm_dead_output-" + saved_mean.debugName())
        saved_var.setDebugName("batch_norm_dead_output-" + saved_var.debugName())
        return res
