from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch.nn.functional as F

class Hardswish(nn.Hardswish):
    r"""
    A Hardswish module attached with FakeQuantize modules for both output
    activation and weight, used for quantization aware training.

    Similar to `torch.nn.Hardswish`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Hardswish

    def __init__(self, qconfig=None):
        super(Hardswish, self).__init__()
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(F.hardswish(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_hardswish = cls(qconfig=qconfig)
        return qat_hardswish

class ELU(nn.ELU):
    r"""This is the QAT equivalent of :class:`torch.nn.ELU`.
    """
    _FLOAT_MODULE = nn.ELU

    def __init__(self, alpha, qconfig=None):
        super(ELU, self).__init__(alpha)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(F.elu(input, self.alpha))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_elu = cls(mod.alpha, qconfig=qconfig)
        return qat_elu
