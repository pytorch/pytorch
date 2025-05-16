# mypy: allow-untyped-defs
"""This file exports ONNX ops for opset 22.

Note [ONNX Operators that are added/updated in opset 22]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-22-of-the-default-onnx-operator-set
New operators:
    - DFT
    - IDFT
    - HammingWindow
    - HannWindow
    - BlackmanWindow
"""

import functools
import torch
from torch import _C
from torch.onnx import symbolic_helper
from torch.onnx._internal import jit_utils, registration

__all__ = [
    "dft",
    "idft",
    "hamming_window",
    "hann_window",
    "blackman_window",
]

_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=22)


@_onnx_symbolic("aten::fft_fft")
@symbolic_helper.parse_args("v", "i", "s", "i")
def dft(g, input, dim, norm, lastdim):
    return g.op("DFT", input, axis_i=dim, inverse_i=0, norm_s=norm)


@_onnx_symbolic("aten::fft_ifft")
@symbolic_helper.parse_args("v", "i", "s", "i")
def idft(g, input, dim, norm, lastdim):
    return g.op("DFT", input, axis_i=dim, inverse_i=1, norm_s=norm)


@_onnx_symbolic("aten::hamming_window")
@symbolic_helper.parse_args("i", "b", "f", "i")
def hamming_window(g, window_length, periodic, alpha, dtype):
    return g.op(
        "HammingWindow",
        g.op("Constant", value_t=torch.tensor(window_length)),
        alpha_f=alpha,
        periodic_i=int(periodic)
    )


@_onnx_symbolic("aten::hann_window")
@symbolic_helper.parse_args("i", "b", "i")
def hann_window(g, window_length, periodic, dtype):
    return g.op(
        "HannWindow",
        g.op("Constant", value_t=torch.tensor(window_length)),
        periodic_i=int(periodic)
    )


@_onnx_symbolic("aten::blackman_window")
@symbolic_helper.parse_args("i", "b", "f", "i")
def blackman_window(g, window_length, periodic, beta, dtype):
    return g.op(
        "BlackmanWindow",
        g.op("Constant", value_t=torch.tensor(window_length)),
        beta_f=beta,
        periodic_i=int(periodic)
    )
