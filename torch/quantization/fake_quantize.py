from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import default_observer, _with_args

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
        self.enabled = True
        self.observer = default_observer(dtype=dtype, qscheme=qscheme)
        self.scale = None
        self.zero_point = None

    def enable(self, enabled=True):
        self.enabled = enabled
        return self

    def disable(self):
        return self.enable(False)

    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    def forward(self, X):
        if self.enabled:
            self.observer(X)
            scale, zero_point = self.calculate_qparams()
            self.scale, self.zero_point = float(scale), int(zero_point)
            X = torch.fake_quantize_per_tensor_affine(
                X, self.scale, self.zero_point, self.quant_min,
                self.quant_max)
        return X

    with_args = classmethod(_with_args)

default_fake_quant = FakeQuantize

default_weight_fake_quant = FakeQuantize.with_args(dtype=torch.qint8,
                                                   qscheme=torch.per_tensor_symmetric,
                                                   quant_min=-128,
                                                   quant_max=127)
