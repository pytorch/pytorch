from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn

class GroupNorm(nn.GroupNorm):
    r"""
    A GroupNorm module attached with FakeQuantize modules for both output
    activation and weight, used for quantization aware training.

    Similar to `torch.nn.GroupNorm`, with FakeQuantize modules initialized to
    default.

    Attributes:
        activation_post_process: fake quant module for output activation
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.GroupNorm

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True,
                 qconfig=None):
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig
        self.activation_post_process = qconfig.activation()

    def forward(self, input):
        return self.activation_post_process(
            super(GroupNorm, self).forward(input))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by
            torch.quantization utilities or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, ' qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'

        qconfig = mod.qconfig
        qat_groupnorm = cls(
            mod.num_groups, mod.num_channels, mod.eps, mod.affine,
            qconfig=qconfig)
        return qat_groupnorm
