"""This file exports ONNX ops for opset 17.

Note [ONNX Operators that are added/updated in opset 17]

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/main/docs/Changelog.md#version-17-of-the-default-onnx-operator-set
New operators:
    BlackmanWindow
    DFT
    HammingWindow
    HannWindow
    LayerNormalization
    MelWeightMatrix
    STFT
    SequenceMap
"""

# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

import math
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import _C
from torch._C import _onnx as _C_onnx
from torch.onnx import _type_utils, symbolic_helper


@symbolic_helper.parse_args("v", "i", "i", "i", "v", "b", "b", "b", "none")
def stft(
    g,
    self: _C.Value,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: _C.Value,
    normalized: bool,
    onesided: bool,
    return_complex: bool,
) -> _C.Value:
    # return_complex is always True since version 1.8.0
    # n_fft: Size of the FFT. frame_length in ONNX
    # hop_length: Stride of the FFT. frame_step in ONNX
    # window: Window to use in FFT. window in ONNX
    input_dtype = _type_utils.JitScalarType.from_name(self.type().scalarType()).dtype()
    frame_step = g.op("Constant", value_t=torch.tensor(hop_length, dtype=torch.int64))
    frame_length = g.op("Constant", value_t=torch.tensor(n_fft, dtype=torch.int64))
    stft_result = g.op(
        "STFT",
        self,
        frame_step,
        window,
        frame_length,
        onesided_i=onesided,
    )
    # Reconstruct the reel output to complex output
    # TODO
    # Normalization multiplies the result by 1 / sqrt(frame_length)
    if normalized:
        sqrt_frame_length = torch.tensor(math.sqrt(n_fft), dtype=input_dtype)
        return g.op("Div", stft_result, g.op("Constant", value_t=sqrt_frame_length))

    return stft_result
