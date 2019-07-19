r"""Importing this file includes common utility methods for checking quantized
tensors and modules.
"""
from collections import Iterable
import copy

from .common_utils import TEST_NUMPY

if TEST_NUMPY:
    import numpy as np

def _clip(a, a_min, a_max):
    """Clips the values to min, max."""
    if TEST_NUMPY:
        return np.clip(a, a_min, a_max)
    a = copy.deepcopy(a)
    if isinstance(a, Iterable):
        for idx in len(a):
            a[idx] = _clip(a[idx], a_min, a_max)
        return a
    else:
        return min(a_max, max(a_min, a))

def _round(a, ndigits=None):
    """Rounds the values to some (optional) number of digits."""
    if TEST_NUMPY:
        return np.round(a, decimals=ndigits)
    a = copy.deepcopy(a)
    if isinstance(a, Iterable):
        for idx in len(a):
            a[idx] = _round(a[idx], ndigits)
        return a
    else:
        return round(a, ndigits)

# Quantization references
def _quantize(x, scale, zero_point, qmin=None, qmax=None, dtype=torch.uint8):
    """Quantizes a numpy array."""
    if qmin is None:
        qmin = torch.iinfo(dtype).min
    if qmax is None:
        qmax = torch.iinfo(dtype).max
    qx = torch.tensor(x, dtype=torch.float32)
    qx = (qx / scale + zero_point).round().to(torch.int64)
    qx = qx.clamp(qmin, qmax)
    qx = qx.to(dtype)
    qx = qx.numpy()
    if not isinstance(x, Iterable):
        qx = qx[0]
    return qx


def _dequantize(qx, scale, zero_point):
    """Dequantizes a numpy array."""
    x = torch.tensor(qx, dtype=torch.float)
    x = (x - zero_point) * scale
    x = x.numpy()
    if not isinstance(qx, Iterable):
        x = x[0]
    return x


def _requantize(x, multiplier, zero_point, qmin=0, qmax=255, qtype=torch.uint8):
    """Requantizes a numpy array, i.e., intermediate int32 or int16 values are
    converted back to given type"""
    qx = torch.tensor(x, dtype=torch.float)
    qx = (qx * multiplier).round() + zero_point
    qx = qx.clamp(qmin, qmax).to(qtype).numpy()
    if not isinstance(x, Iterable):
        qx = qx[0]
    return qx
