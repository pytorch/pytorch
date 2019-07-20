from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn import Conv2d as NNConv2d
from torch.quantization.QConfig import default_qat_qconfig

class Conv2d(NNConv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for both output
    activation and weight, used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow
        weight_fake_quant: fake quant module for weight
    """

    __FLOAT_MODULE__ = NNConv2d

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 activation_fake_quant=default_qat_qconfig.activation,
                 weight_fake_quant=default_qat_qconfig.weight):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)
        self.observer = activation_fake_quant()
        self.weight_fake_quant = weight_fake_quant()

    def forward(self, input):
        return self.observer(
            self.conv2d_forward(input, self.weight_fake_quant(self.weight)))

    @classmethod
    def from_float(cls, mod, qconfig=None):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls.__FLOAT_MODULE__, ' nnq.' + cls.__name__ + '.from_float only works for ' + \
            cls.__FLOAT_MODULE__.__name__
        if not qconfig:
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            assert mod.qconfig, 'Input float module must have a valid qconfig'
            qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode,
                       activation_fake_quant=qconfig.activation,
                       weight_fake_quant=qconfig.weight)
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        return qat_conv
