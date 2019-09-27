r"""Importing this file includes common utility methods for checking quantized
tensors and modules.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
from contextlib import contextmanager

"""Computes the output shape given convolution parameters."""
def _conv_output_shape(input_size, kernel_size, padding, stride, dilation,
                       output_padding=0):
    return np.floor((input_size + 2 * padding - kernel_size - (kernel_size - 1)
                     * (dilation - 1)) / stride) + 2 * output_padding + 1

# Quantization references
def _quantize(x, scale, zero_point, qmin=None, qmax=None, dtype=np.uint8):
    """Quantizes a numpy array."""
    if qmin is None:
        qmin = np.iinfo(dtype).min
    if qmax is None:
        qmax = np.iinfo(dtype).max
    qx = np.round(x / scale + zero_point).astype(np.int64)
    qx = np.clip(qx, qmin, qmax)
    qx = qx.astype(dtype)
    return qx


def _dequantize(qx, scale, zero_point):
    """Dequantizes a numpy array."""
    x = (qx.astype(np.float) - zero_point) * scale
    return x


def _requantize(x, multiplier, zero_point, qmin=0, qmax=255, qtype=np.uint8):
    """Requantizes a numpy array, i.e., intermediate int32 or int16 values are
    converted back to given type"""
    qx = (x * multiplier).round() + zero_point
    qx = np.clip(qx, qmin, qmax).astype(qtype)
    return qx

def _calculate_dynamic_qparams(X, dtype):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if dtype == torch.qint8:
        qmin, qmax = -128, 127
    else:  # dtype == torch.quint8
        qmin, qmax = 0, 255
    n_levels = 255.0
    min_val = X.min()
    max_val = X.max()
    if min_val == max_val:
        scale = 1.0
        zero_point = 0
    else:
        max_val = max(max_val, 0.0)
        min_val = min(min_val, 0.0)
        scale = (max_val - min_val) / n_levels
        scale = max(scale, np.finfo(np.float32).eps)
        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
    return [float(scale), int(zero_point)]

@contextmanager
def enable_mobile_quantized_engine():
    torch.backends.quantized.engine = 'qnnpack'
    try:
        yield
    finally:
        qengines = torch.backends.quantized.get_supported_qengines()
        if 'fbgemm' in qengines:
            torch.backends.quantized.engine = 'fbgemm'
        else:
            torch.backends.quantized.engine = 'none'
