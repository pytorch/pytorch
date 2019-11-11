from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.quantized.functional

class ReLU(torch.nn.ReLU):
    r"""Applies quantized rectified linear unit function element-wise:

    :math:`\text{ReLU}(x)= \max(x_0, x)`, where :math:`x_0` is the zero point.

    Please see https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU
    for more documentation on ReLU.

    Args:
        inplace: (Currently not supported) can optionally do the operation in-place.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.quantized.ReLU()
        >>> input = torch.randn(2)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return torch.nn.quantized.functional.relu(input, inplace=self.inplace)

    def _get_name(self):
        return 'QuantizedReLU'

    @staticmethod
    def from_float(mod):
        return ReLU(mod.inplace)


class ReLU6(torch.nn.ReLU):
    r"""Applies the element-wise function:

    :math:`\text{ReLU6}(x) = \min(\max(x_0, x), q(6))`, where :math:`x_0` is the
    zero_point, and :math:`q(6)` is the quantized representation of number 6.

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.quantized.ReLU6()
        >>> input = torch.randn(2)
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    def __init__(self, inplace=False):
        super(ReLU6, self).__init__(inplace)
        self.inplace = inplace

    def forward(self, input):
        return torch.ops.quantized.relu6(input, self.inplace)

    def _get_name(self):
        return 'QuantizedReLU6'

    @staticmethod
    def from_float(mod):
        return ReLU6(mod.inplace)
