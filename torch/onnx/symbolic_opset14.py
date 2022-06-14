"""This file exports ONNX ops for opset 14.

Note [ONNX operators that are added/updated in opset 14]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New operators:
    HardSwish, Trilu

Updated operators:
    Reshape
    Add, Sub, Mul, Div
    GRU, LSTM, RNN
    BatchNorm, Cumsum, Relu
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

import torch
from torch.onnx import symbolic_helper
from torch.onnx._globals import GLOBALS


@symbolic_helper.parse_args("v")
def hardswish(g, self):
    return g.op("HardSwish", self)


@symbolic_helper.parse_args("v", "i")
def tril(g, self, diagonal, out=None):
    k = g.op("Constant", value_t=torch.tensor(diagonal, dtype=torch.int64))
    return g.op("Trilu", self, k, upper_i=0)


@symbolic_helper.parse_args("v", "i")
def triu(g, self, diagonal, out=None):
    k = g.op("Constant", value_t=torch.tensor(diagonal, dtype=torch.int64))
    return g.op("Trilu", self, k, upper_i=1)


@symbolic_helper.parse_args("v", "v")
def reshape(g, self, shape):
    # NOTE: Due to bug in ORT https://github.com/microsoft/onnxruntime/issues/10664
    #       Reshape export cannot utilize the new allowzero attribute introduced in opset 14.
    return symbolic_helper._reshape_helper(g, self, shape, allowzero=0)


@symbolic_helper.parse_args("v", "v", "v", "v", "v", "i", "f", "f", "i")
def batch_norm(
    g,
    input,
    weight,
    bias,
    running_mean,
    running_var,
    training,
    momentum,
    eps,
    cudnn_enabled,
):

    if (
        torch.is_autocast_enabled()
        and not symbolic_helper.args_have_same_dtype(
            [input, weight, bias, running_mean, running_var]
        )
        and GLOBALS.export_onnx_opset_version < 15
    ):
        return symbolic_helper._onnx_opset_unsupported_detailed(
            "BatchNormalization",
            14,
            15,
            "All input tensors must have the same `dtype`."
            " Turn off Autocast or export using opset version 15.",
        )

    symbolic_helper.check_training_mode(training, "batch_norm")
    weight, bias, running_mean, running_var = symbolic_helper._batchnorm_helper(
        g, input, weight, bias, running_mean, running_var
    )
    out = g.op(
        "BatchNormalization",
        input,
        weight,
        bias,
        running_mean,
        running_var,
        epsilon_f=eps,
        momentum_f=1 - momentum,
        training_mode_i=0 if not training else 1,
        outputs=1 if not training else 3,
    )
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
        x, _, _, _ = symbolic_helper.dequantize_helper(g, x)

        output = hardswish(g, x)

        return symbolic_helper.quantize_helper(g, output, op_scale, op_zero_point)
