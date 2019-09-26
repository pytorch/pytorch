from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import MinMaxObserver, _with_args

class FakeQuantize(Module):
    ''' Simulate the quantize and dequantize operations in training time.
    Args:
        `qconfig`: object that encodes configuration info for quantization
        `observer_module`: Observer module that records stats of weights and
        activations
        `calcqparam`: A function that calculates quantization parameters
        given the stats
    '''
    def __init__(self, observer=MinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(FakeQuantize, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.fake_quant_enabled = True
        self.observer_enabled = True
        self.observer = observer(**observer_kwargs)
        assert torch.iinfo(self.observer.dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.observer.dtype).max, 'quant_max out of bound'
        self.scale = None
        self.zero_point = None
        self.dtype = self.observer.dtype

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
            X = torch.fake_quantize_per_tensor_affine(X, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return X

    with_args = classmethod(_with_args)

    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.scale = state_dict.pop(prefix + 'scale')
        self.zero_point = state_dict.pop(prefix + 'zero_point')
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                        missing_keys, unexpected_keys, error_msgs)

default_fake_quant = FakeQuantize
default_weight_fake_quant = FakeQuantize.with_args(observer=MinMaxObserver, quant_min=-128, quant_max=127, dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
