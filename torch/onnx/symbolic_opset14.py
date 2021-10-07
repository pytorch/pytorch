# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 14
import torch

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args

# Note [ONNX operators that are added/updated in opset 14]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# New operators:
#   HardSwish, Trilu
#
# Updated operators:
#   Reshape
#   Add, Sub, Mul, Div
#   GRU, LSTM, RNN
#   BatchNorm, Cumsum, Relu

@parse_args("v")
def hardswish(g, self):
    return g.op("HardSwish", self)

@parse_args("v", "i")
def tril(g, self, diagonal, out=None):
    k = g.op("Constant", value_t=torch.tensor(diagonal, dtype=torch.int64))
    return g.op("Trilu", self, k, upper_i=0)

@parse_args("v", "i")
def triu(g, self, diagonal, out=None):
    k = g.op("Constant", value_t=torch.tensor(diagonal, dtype=torch.int64))
    return g.op("Trilu", self, k, upper_i=1)

@parse_args("v", "v")
def reshape(g, self, shape):
    return sym_help._reshape_helper(g, self, shape)

@parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    sym_help.check_training_mode(training, "batch_norm")
    weight, bias, running_mean, running_var = sym_help._batchnorm_helper(g, input, weight, bias, running_mean, running_var)
    out = g.op("BatchNormalization", input, weight, bias, running_mean, running_var,
               epsilon_f=eps,
               momentum_f=1 - momentum,
               training_mode_i=0 if not training else 1,
               outputs=1 if not training else 3)
    if not training:
        return out
    else:
        res, new_running_mean, new_running_var = out
        new_running_mean.setType(running_mean.type())
        new_running_var.setType(running_var.type())
        return res
