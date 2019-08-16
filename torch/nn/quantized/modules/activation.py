from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .. import functional as F
from torch.nn import ReLU as NNReLU

class ReLU(NNReLU):
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
        >>> input = torch.quantize_linear(input, 1.0, 0, dtype=torch.qint32)
        >>> output = m(input)
    """
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(inplace)
        assert not inplace, 'torch.nn.quantized.ReLU does not support inplace'

    def forward(self, input):
        return F.relu(input)

    @staticmethod
    def from_float(mod):
        return ReLU(mod.inplace)
