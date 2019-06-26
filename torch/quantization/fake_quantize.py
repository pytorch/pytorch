from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .._jit_internal import weak_module
from .observer import Observer, observer

@weak_module
class FakeQuantize(Module):
    ''' Simulate the quantize and dequantize operations in training time.
    Args:
        `fq_config`: object that encodes configuration info for quantization
        `observer_module`: Observer module that records stats of weights and
        activations
        `calcqparam`: A function that calculates quantization parameters
        given the stats
    '''

    def __init__(self, fq_config, quant_delay=0):
        super(FakeQuantize, self).__init__()
        assert fq_config['qscheme'] == torch.per_tensor_affine
        observer_factory = observer(Observer, {
            'dtype': fq_config.get('dtype', torch.qint8),
            'qscheme': fq_config.get('qscheme', torch.per_tensor_affine)})
        self.observer = observer_factory()
        self.fq_config = fq_config
        self.quant_min = fq_config.get('quant_min', -128)
        self.quant_max = fq_config.get('quant_max', 127)
        self.quant_delay = quant_delay
        self.iter = 1
        self.scale = None
        self.zero_point = None

    def forward(self, X):
        self.observer(X)
        self.scale, self.zero_point = self.observer.calculate_qparams()
        X = torch.fake_quantize_per_tensor_affine(X, self.scale.double(), self.zero_point.long(), self.quant_min,
            self.quant_max, self.quant_delay, self.iter)
        self.iter = self.iter + 1
        return X
