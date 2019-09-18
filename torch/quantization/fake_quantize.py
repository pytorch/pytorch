from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import default_observer
from functools import partial

class FakeQuantize(Module):
    ''' Simulate the quantize and dequantize operations in training time.
    Args:
        `qconfig`: object that encodes configuration info for quantization
        `observer_module`: Observer module that records stats of weights and
        activations
        `calcqparam`: A function that calculates quantization parameters
        given the stats
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 quant_min=0, quant_max=255):
        super(FakeQuantize, self).__init__()
        assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.fake_quant_enabled = True
        self.observer_enabled = True
        self.observer = default_observer(dtype=dtype, qscheme=qscheme)()
        self.scale = None
        self.zero_point = None

    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled = enabled
        return self

    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    def enable_observer(self, enabled=True):
        self.observer_enabled = enabled

    def disable_observer(self):
        return self.enable_observer(False)

    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled:
            X = self.observer(X)
            scale, zero_point = self.calculate_qparams()
            self.scale, self.zero_point = float(scale), int(zero_point)
        if self.fake_quant_enabled:
            X = torch.fake_quantize_per_tensor_affine(
                X, self.scale, self.zero_point, self.quant_min,
                self.quant_max)
        return X

    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

def fake_quant(fake_quant_cls, **kwargs):
    return partial(fake_quant_cls, **kwargs)

def default_fake_quant(**kwargs):
    return fake_quant(FakeQuantize, **kwargs)

def default_weight_fake_quant(**kwargs):
    kwargs.setdefault('dtype', torch.qint8)
    kwargs.setdefault('qscheme', torch.per_tensor_symmetric)
    kwargs.setdefault('quant_min', -128)
    kwargs.setdefault('quant_max', 127)
    return fake_quant(FakeQuantize, **kwargs)
