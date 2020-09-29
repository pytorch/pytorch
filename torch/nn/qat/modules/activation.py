import torch
import torch.nn as nn
from torch.nn import Module

class Sigmoid(Module):
    r""" qat version of :class:`~torch.nn.Sigmoid`
    Since quantized sigmoid has fixed quantization parameters, we need
    to simulate the behvaior in quantization aware training.
    """
    _FLOAT_MODULE = nn.Sigmoid

    def __init__(self, activation_post_process_ctr):
        super().__init__()
        from torch.quantization import default_affine_fixed_qparams_fake_quant
        self.fake_quant = activation_post_process_ctr()

    def forward(self, x):
        return self.fake_quant(torch.sigmoid(x))


    @classmethod
    def from_float(cls, mod):
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qat_sigmoid = cls(torch.quantization.default_affine_fixed_qparams_fake_quant)
        return qat_sigmoid
