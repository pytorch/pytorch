
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn.quantized.modules._base_modules import _BaseQuantizedModule

class QFunctional(torch.nn.Module, _BaseQuantizedModule):
    r"""Wrapper class for quantized operatitons.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add_relu``).

    .. Examples::

        >>> q_add = QFunctional('add')
        >>> a = torch.quantize_linear(torch.tensor(3.0), 1.0, 0, torch.qint32)
        >>> b = torch.quantize_linear(torch.tensor(4.0), 1.0, 0, torch.qint32)
        >>> q_add.add_relu(a, b)

    Valid operations:
        - add + ReLU
    """

    _FLOAT_MODULE = torch.nn._intrinsic.FloatFunctional

    def __init__(self):
        super(QFunctional, self).__init__()

    def forward(self, x):
        raise RuntimeError("Functional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def add_relu(self, a, b):
        return torch.ops.quantized.add_relu(a, b, scale=self.scale,
                                            zero_point=self.zero_point)
