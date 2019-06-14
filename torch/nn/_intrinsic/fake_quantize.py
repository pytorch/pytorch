from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch._ops import ops
from ..modules.module import Module
from ..._jit_internal import weak_module

@weak_module
class DefaultObserverModule(Module):
    def __init__(self, qconfig, alpha=0.9):
        super(DefaultObserverModule, self).__init__()
        self.stats = ()
        self.alpha = alpha

    def forward(self, X):
        if not self.stats:
            self.stats = torch.min(X), torch.max(X)
        else:
            self.stats = alpha * self.stats[0] + (1-alpha) * torch.min(X), \
                alpha * self.stats[1] + (1-alpha) * torch.max(X),
        return self.stats

def get_default_calcqparam(qconfig):
    def default_calcqpram(stats):
        min, max = stats
        qmin, qmax = qconfig['qmin'], qconfig['qmax']
        scale = (max.item() - min.item()) / (qmax - qmin)
        zero_point = qmin
        return scale, zero_point
    return default_calcqpram

@weak_module
class FakeQuantize(Module):
    ''' Simulate the quantize and dequantize operations in training time.
    Args:
        `qconfig`: object that encodes configuration info for quantization
        `observer_module`: Observer module that records stats of weights and
        activations
        `calcqparam`: A function that calculates quantization parameters
        given the stats
    '''

    def __init__(self, qconfig, quant_delay=0):
        super(FakeQuantize, self).__init__()
        # assert (qconfig['qscheme'] == torch.per_tensor_affine)
        if 'observer_module' in qconfig:
            assert 'calcqparam' in qconfig
            self.observer = qconfig['observer_module']
            self.calcqparam = qconfig['calcqparam']
        else:
            self.observer = DefaultObserverModule(qconfig)
            self.calcqparam = get_default_calcqparam(qconfig)
        self.qconfig = qconfig
        self.qmin = qconfig['qmin']
        self.qmax = qconfig['qmax']
        self.quant_delay = quant_delay
        self.iter = 0
        self.scale = None
        self.zero_point = None

    def forward(self, X):
        self.scale, self.zero_point = self.calcqparam(self.observer(X))
        print('FakeQuantize forward', X.requires_grad)
        X = ops.quantized.fake_quantize_per_tensor_affine_forward(X, scale=self.scale, zero_point=self.zero_point, quant_min=self.qmin,
            quant_max=self.qmax, quant_delay=self.quant_delay, iter=self.iter)
        print('FakeQuantize forward', X.requires_grad)
        self.iter += 1
        return X

    def backward(self, X_grad):
        X_grad = ops.quantized.fake_quantize_per_tensor_affine_backward(X_grad, scale=self.scale, zero_point=self.zero_point, quant_min=self.qmin,
        quant_max=self.qmax, quant_delay=self.quant_delay, iter=self.iter)
        return X_grad
