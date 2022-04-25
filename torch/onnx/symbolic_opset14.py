# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 14
import torch

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, args_have_same_dtype

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
    # NOTE: Due to bug in ORT https://github.com/microsoft/onnxruntime/issues/10664
    #       Reshape export cannot utilize the new allowzero attribute introduced in opset 14.
    return sym_help._reshape_helper(g, self, shape, allowzero=0)

@parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):

    if torch.is_autocast_enabled() and \
            not args_have_same_dtype([input, weight, bias, running_mean, running_var]) and \
            sym_help._export_onnx_opset_version < 15:
        return sym_help._onnx_opset_unsupported_detailed("BatchNormalization", 14, 15,
                                                         "All input tensors must have the same `dtype`."
                                                         " Turn off Autocast or export using opset version 15.")

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


class Quantized:
    """
    https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter#quantized-model-export
    """
    domain = "quantized"

    @staticmethod
    def hardswish(g, x, op_scale, op_zero_point):
        x, _, _, _ = sym_help.dequantize_helper(g, x)

        output = hardswish(g, x)

        return sym_help.quantize_helper(g, output, op_scale, op_zero_point)
