from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import default_observer

DTYPE_RANGE = {
    torch.quint8: [0, 255],
    torch.qint8: [-128, 127],
    torch.qint32: [-2**31, 2**31-1]
}

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
                 quant_min=0, quant_max=255, enable_fq=True):
        super(FakeQuantize, self).__init__()
        range = torch.iinfo(dtype).min, torch.iinfo(dtype).max #DTYPE_RANGE[dtype]
        assert range[0] <= quant_min, 'quant_min out of bound'
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        assert quant_max <= range[1], 'quant_max out of bound'
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.enable_fq = enable_fq
        self.observer = default_observer(dtype=dtype, qscheme=qscheme)()
        self.scale = None
        self.zero_point = None

    def enable(self):
        self.enable_fq = True

    def calculate_qparams(self):
        return self.observer.calculate_qparams()

    def forward(self, X):
        if self.training and self.enable_fq:
            self.observer(X)
            self.scale, self.zero_point = self.calculate_qparams()
            X = torch.fake_quantize_per_tensor_affine(
                X, self.scale.double(), self.zero_point.long(), self.quant_min,
                self.quant_max)
        return X
