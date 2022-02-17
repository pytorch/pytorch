r"""Importing this file includes common utility methods for checking quantized
tensors and modules.
"""
import numpy as np
import torch
from contextlib import contextmanager
from torch.testing._internal.common_utils import TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_PPC, IS_MACOS, IS_WINDOWS

supported_qengines = torch.backends.quantized.supported_engines
supported_qengines.remove('none')
# Note: We currently do not run QNNPACK tests on WINDOWS and MACOS as it is flaky. Issue #29326
# QNNPACK is not supported on PPC
# QNNPACK throws ASAN heap-buffer-overflow error.
if 'qnnpack' in supported_qengines and any([IS_PPC, TEST_WITH_ASAN, TEST_WITH_TSAN, TEST_WITH_UBSAN, IS_MACOS, IS_WINDOWS]):
    supported_qengines.remove('qnnpack')

def _conv_output_shape(input_size, kernel_size, padding, stride, dilation,
                       output_padding=0):
    """Computes the output shape given convolution parameters."""
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
    x = (qx.astype(float) - zero_point) * scale
    return x


def _requantize(x, multiplier, zero_point, qmin=0, qmax=255, qtype=np.uint8):
    """Requantizes a numpy array, i.e., intermediate int32 or int16 values are
    converted back to given type"""
    qx = (x * multiplier).round() + zero_point
    qx = np.clip(qx, qmin, qmax).astype(qtype)
    return qx

def _calculate_dynamic_qparams(X, dtype, reduce_range=False):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if dtype == torch.qint8:
        if reduce_range:
            qmin, qmax = -64, 63
        else:
            qmin, qmax = -128, 127
    else:  # dtype == torch.quint8
        if reduce_range:
            qmin, qmax = 0, 127
        else:
            qmin, qmax = 0, 255
    min_val = X.min()
    max_val = X.max()
    if min_val == max_val:
        scale = 1.0
        zero_point = 0
    else:
        max_val = max(max_val, 0.0)
        min_val = min(min_val, 0.0)
        scale = (max_val - min_val) / (qmax - qmin)
        scale = max(scale, np.finfo(np.float32).eps)
        zero_point = qmin - round(min_val / scale)
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
    return [float(scale), int(zero_point)]

def _calculate_dynamic_per_channel_qparams(X, dtype):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    qmin, qmax = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    n_levels = qmax - qmin
    scale = np.zeros(X.shape[0], dtype=np.float64)
    zero_point = np.zeros(X.shape[0], dtype=np.int64)
    for i in range(zero_point.shape[0]):
        min_val = X.min()
        max_val = X.max()
        if min_val == max_val:
            scale[i] = 1.0
            zero_point[i] = 0
        else:
            max_val = max(max_val, 0.0)
            min_val = min(min_val, 0.0)
            scale[i] = (max_val - min_val) / n_levels
            scale[i] = max(scale[i], np.finfo(np.float32).eps)
            zero_point[i] = qmin - round(min_val / scale[i])
            zero_point[i] = max(qmin, zero_point[i])
            zero_point[i] = min(qmax, zero_point[i])

    return scale, zero_point

def _snr(x, x_hat):
    """Calculates the signal to noise ratio and returns the signal and noise
    power, as well as the SNR in dB.
    If the input is a list/tuple this function is called recursively on each
    element. The result will have the same nested structure as the inputs.

    Args:
        x, x_hat: Either a tensor or a nested list/tuple of tensors.
    Returns:
        signal, noise, SNR(in dB): Either floats or a nested list of floats
    """
    if isinstance(x, (list, tuple)):
        assert(len(x) == len(x_hat))
        res = []
        for idx in range(len(x)):
            res.append(_snr(x[idx], x_hat[idx]))
        return res
    if x_hat.is_quantized:
        x_hat = x_hat.dequantize()
    if x.is_quantized:
        x = x.dequantize()
    noise = (x - x_hat).norm()
    if noise == 0:
        return 0.0, float('inf'), float('inf')
    signal = x.norm()
    snr = signal / noise
    snr_db = 20 * snr.log10()
    return signal, noise, snr_db

@contextmanager
def override_quantized_engine(qengine):
    previous = torch.backends.quantized.engine
    torch.backends.quantized.engine = qengine
    try:
        yield
    finally:
        torch.backends.quantized.engine = previous

@contextmanager
def override_cpu_allocator_for_qnnpack(qengine_is_qnnpack):
    try:
        if qengine_is_qnnpack:
            torch._C._set_default_mobile_cpu_allocator()
        yield
    finally:
        if qengine_is_qnnpack:
            torch._C._unset_default_mobile_cpu_allocator()

# TODO: Update all quantization tests to use this decorator.
# Currently for some of the tests it seems to have inconsistent params
# for fbgemm vs qnnpack.
def override_qengines(qfunction):
    def test_fn(*args, **kwargs):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                # qfunction should not return anything.
                qfunction(*args, **kwargs)
    return test_fn

def qengine_is_fbgemm():
    return torch.backends.quantized.engine == 'fbgemm'
def qengine_is_qnnpack():
    return torch.backends.quantized.engine == 'qnnpack'

# Helper function used to simulate per-channel fake-quant against any axis
def _permute_to_axis_zero(X, axis):
    new_axis_list = list(range(X.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = X.permute(tuple(new_axis_list))
    return y, new_axis_list

# Reference method for fake quantize
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_channel_affine_reference(X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    res = torch.zeros_like(X)

    for i in range(X.size()[0]):
        res[i] = (torch.clamp(torch.round(X[i] * (1.0 / per_channel_scale[i]) +
                  per_channel_zero_point[i]), quant_min, quant_max) - per_channel_zero_point[i]) * per_channel_scale[i]

    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

# Reference method for the gradient of the fake quantize operator
# Note: because scale/zero_point are left as float in the actual kernel, this mimics how fake_quant works for float16/64
def _fake_quantize_per_channel_affine_grad_reference(dY, X, per_channel_scale, per_channel_zero_point, axis, quant_min, quant_max):
    dtype = X.dtype
    X, permute_axis_list = _permute_to_axis_zero(X.to(torch.float32), axis)
    Xq = torch.zeros_like(X)
    for i in range(X.size()[0]):
        Xq[i] = torch.round(X[i] * (1.0 / per_channel_scale[i]) + per_channel_zero_point[i])
    Xq = Xq.permute(tuple(permute_axis_list))
    mask = (Xq >= quant_min) * (Xq <= quant_max)
    res = torch.zeros_like(dY)
    res[mask] = dY[mask]
    return res.to(dtype)

def to_tensor(X, device):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    else:
        X = X.clone().detach()
    return X.to(device=torch.device(device), dtype=torch.float32)
