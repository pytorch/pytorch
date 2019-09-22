from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import default_observer, default_l2_observer, default_per_channel_weight_observer
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

    def __init__(self, observer=default_observer(), quant_min=0, quant_max=255):
        super(FakeQuantize, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.fake_quant_enabled = True
        self.observer_enabled = True
        self.observer = observer()
        assert torch.iinfo(self.observer.dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.observer.dtype).max, 'quant_max out of bound'
        self.scale = None
        self.zero_point = None
        self.dtype = self.observer.dtype
        self.qscheme = self.observer.qscheme

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
        #return X
        if self.observer_enabled:
            with torch.no_grad():
                X = self.observer(X)
                self.scale, self.zero_point = self.calculate_qparams()
        if self.fake_quant_enabled:
        #    TODO: Check if axis is zero
                if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                    X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point, self.quant_min,
                    self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(
                    X, float(self.scale), int(self.zero_point), self.quant_min,
                    self.quant_max)
        return X

    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'quant_min'] = self.quant_min
        destination[prefix + 'quant_max'] = self.quant_max
        destination[prefix + 'dtype'] = self.dtype
        destination[prefix + 'qscheme'] = self.qscheme
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point
        destination[prefix + 'fake_quant_enabled'] = self.fake_quant_enabled
        destination[prefix + 'observer_enabled'] = self.observer_enabled

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        self.quant_min = int(state_dict.pop(prefix + 'quant_min'))
        self.quant_max = int(state_dict.pop(prefix + 'quant_max'))
        self.dtype = state_dict.pop(prefix + 'dtype')
        self.qscheme = state_dict.pop(prefix + 'qscheme')
        self.scale = state_dict.pop(prefix + 'scale')
        self.zero_point = state_dict.pop(prefix + 'zero_point')
        self.fake_quant_enabled = bool(state_dict.pop(prefix + 'fake_quant_enabled'))
        self.observer_enabled = bool(state_dict.pop(prefix + 'observer_enabled'))
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                        missing_keys, unexpected_keys, error_msgs)

def fake_quant(fake_quant_cls, **kwargs):
    return partial(fake_quant_cls, **kwargs)

def default_fake_quant(**kwargs):
    kwargs.setdefault('observer', default_observer(dtype=torch.quint8, qscheme = torch.per_tensor_affine, reduce_range =True))
    return fake_quant(FakeQuantize, **kwargs)

def l2_fake_quant(**kwargs):
    kwargs.setdefault('observer', default_l2_observer(reduce_range = True))
    return fake_quant(FakeQuantize, **kwargs)

def default_per_channel_weight_fake_quant(**kwargs):
    kwargs.setdefault('quant_min', -128)
    kwargs.setdefault('quant_max', 127)
    kwargs.setdefault('observer', default_per_channel_weight_observer(dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range =False))
    return fake_quant(FakeQuantize, **kwargs)

def default_weight_fake_quant(**kwargs):
    kwargs.setdefault('quant_min', -128)
    kwargs.setdefault('quant_max', 127)
    kwargs.setdefault('observer', default_observer(dtype=torch.qint8, qscheme = torch.per_tensor_symmetric, reduce_range =False))
    return fake_quant(FakeQuantize, **kwargs)

def disable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.enable_fake_quant()

def disable_observer(mod):
    if type(mod) == FakeQuantize:
        mod.disable_observer()

def enable_observer(mod):
    if type(mod) == FakeQuantize:
        mod.enable_observer()
