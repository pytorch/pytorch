from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.nn import Conv2d, ReLU, Linear, BatchNorm2d

class ConvReLU2d(torch.nn.Sequential):
    def __init__(self, conv, relu):
        assert type(conv) == Conv2d and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(type(conv), type(relu))
        super(ConvReLU2d, self).__init__(conv, relu)

class LinearReLU(torch.nn.Sequential):
    def __init__(self, linear, relu):
        assert type(linear) == Linear and type(relu) == ReLU, \
            'Incorrect types for input modules{}{}'.format(type(linear), type(relu))
        super(LinearReLU, self).__init__(linear, relu)

class ConvBn2d(torch.nn.Sequential):
    def __init__(self, conv, bn):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(type(conv), type(bn))
        super(ConvBn2d, self).__init__(conv, bn)

class ConvBnReLU2d(torch.nn.Sequential):
    def __init__(self, conv, bn, relu):
        assert type(conv) == Conv2d and type(bn) == BatchNorm2d and \
            type(relu) == ReLU, 'Incorrect types for input modules{}{}{}' \
            .format(type(conv), type(bn), type(relu))
        super(ConvBnReLU2d, self).__init__(conv, bn, relu)


class FloatFunctional(torch.nn.Module):
    r"""State collector class for float operatitons.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add_relu``).

    .. Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add_relu(a, b)

    Valid operations:
        - add + ReLU
    """
    def __init__(self):
        super(FloatFunctional, self).__init__()
        self.observer = torch.nn.Identity()

    def forward(self, x):
        raise RuntimeError("FloatFunctional is not intended to use the " +
                           "'forward'. Please use the underlying operation")

    def add_relu(self, a, b):
        c = torch.add(a, b)
        c = torch.nn.functional.relu(c)
        c = self.observer(c)
        return c
