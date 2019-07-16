from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.modules.conv import Conv2d as NNConv2d
from torch.nn.modules.utils import _pair
from torch.quantization.QConfig import default_qat_qconfig
import torch.nn.functional as F

class Conv2d(NNConv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for both output
    activation and weight, used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow, TODO: rename?
        weight_fake_quant: fake quant module for weight

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)
        # fake quant module for output activation
        self.observer = default_qat_qconfig.activation()
        self.weight_fake_quant = default_qat_qconfig.weight()

    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return self.observer(F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                                 self.weight_fake_quant(self.weight), self.bias, self.stride,
                                 _pair(0), self.dilation, self.groups))
        return self.observer(F.conv2d(input, self.weight_fake_quant(self.weight), self.bias, self.stride,
                                      self.padding, self.dilation, self.groups))

    # TODO: support initializing from qconfig
    @staticmethod
    def from_float(mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == NNConv2d, 'nnq.Linear.from_float only works for nn.Linear'
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert hasattr(mod, 'observer'), 'Input float module must have observer attached'
        qat_conv = Conv2d(mod.in_channels, mod.out_channels, mod.kernel_size,
                          stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                          groups=mod.groups, bias=mod.bias is not None,
                          padding_mode=mod.padding_mode)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        qat_conv.observer = mod.qconfig.activation()
        qat_conv.weight_fake_quant = mod.qconfig.weight()
        return qat_conv
