# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 14
import torch
from torch.onnx.symbolic_opset9 import _var_mean

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_helper import _block_list_in_opset

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
    shape = sym_help._maybe_get_const(shape, "is")
    if not sym_help._is_value(shape):
        shape = g.op("Constant", value_t=torch.LongTensor(shape))
    return sym_help._reshape_helper(g, self, shape)

@parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    sym_help.assert_training_mode(training, "batch_norm")
    batch_size = sym_help._get_tensor_dim_size(input, 0)
    channel_size = sym_help._get_tensor_dim_size(input, 1)

    if weight is None or sym_help._is_none(weight):
        if channel_size is None:
            raise RuntimeError("Unsupported: ONNX export of batch_norm for unknown "
                               "channel size.")
        weight_value = torch.tensor([1.] * channel_size).type(
            "torch." + input.type().scalarType() + "Tensor")
        weight = g.op("Constant", value_t=weight_value)
    if bias is None or sym_help._is_none(bias):
        if channel_size is None:
            raise RuntimeError("Unsupported: ONNX export of batch_norm for unknown "
                               "channel size.")
        bias_value = torch.tensor([0.] * channel_size).type(
            "torch." + input.type().scalarType() + "Tensor")
        bias = g.op("Constant", value_t=bias_value)
    # If track_running_stats is set to False batch statistics are instead used during evaluation time
    if running_mean is None or sym_help._is_none(running_mean) or running_var is None or sym_help._is_none(running_var):
        assert batch_size is not None and channel_size is not None
        reshape_in = g.op("Reshape", input,
                          g.op("Constant", value_t=torch.tensor([batch_size, channel_size, -1], dtype=torch.int64)))
        trans_in = g.op("Transpose", reshape_in, perm_i=[0, 2, 1])
        running_var, running_mean = _var_mean(g, trans_in,
                                              g.op("Constant", value_t=torch.tensor([0, 1], dtype=torch.int64)),
                                              False, False)
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
