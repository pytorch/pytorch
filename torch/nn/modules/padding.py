from .module import Module
from .utils import _pair, _quadruple, _ntuple
from .. import functional as F


# TODO: grad_output size asserts in THNN


class ConstantPad1d(Module):
    r"""Pads the input tensor boundaries with a constant value.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in both boundaries. If a 2-tuple, uses (paddingLeft, paddingRight)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ConstantPad1d(3, 3.5)
        >>> input = torch.randn(16, 2, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ConstantPad1d((3, 5), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value):
        super(ConstantPad1d, self).__init__()
        self.padding = _pair(padding)
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ConstantPad2d(Module):
    r"""Pads the input tensor boundaries with a constant value.

    For Nd-padding, use nn.functional.pad().

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ConstantPad2d(3, 3.5)
        >>> input = torch.randn(16, 3, 320, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ConstantPad2d((3, 3, 6, 6), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value):
        super(ConstantPad2d, self).__init__()
        self.padding = _quadruple(padding)
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ConstantPad3d(Module):
    r"""Pads the input tensor boundaries with a constant value.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 6-tuple, uses
            (paddingLeft, paddingRight, paddingTop, paddingBottom, paddingFront, paddingBack)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = D_{in} + paddingFront + paddingBack`
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ConstantPad3d(3, 3.5)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value):
        super(ConstantPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReflectionPad1d(Module):
    r"""Pads the input tensor using the reflection of the input boundary.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 2-tuple, uses (paddingLeft, paddingRight)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ReflectionPad1d(3)
        >>> input = torch.randn(16, 3, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ReflectionPad1d((3, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReflectionPad1d, self).__init__()
        self.padding = _pair(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReflectionPad2d(Module):
    r"""Pads the input tensor using the reflection of the input boundary.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ReflectionPad2d(3)
        >>> input = torch.randn(16, 3, 320, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ReflectionPad2d((3, 3, 6, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReplicationPad1d(Module):
    r"""Pads the input tensor using replication of the input boundary.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 2-tuple, uses (paddingLeft, paddingRight)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ReplicationPad1d(3)
        >>> input = torch.randn(16, 3, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ReplicationPad1d((3, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReplicationPad1d, self).__init__()
        self.padding = _pair(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReplicationPad2d(Module):
    r"""Pads the input tensor using replication of the input boundary.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ReplicationPad2d(3)
        >>> input = torch.randn(16, 3, 320, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ReplicationPad2d((3, 3, 6, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReplicationPad3d(Module):
    r"""Pads the input tensor using replication of the input boundary.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 6-tuple, uses (paddingLeft, paddingRight,
            paddingTop, paddingBottom, paddingFront, paddingBack)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = D_{in} + paddingFront + paddingBack`
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ReplicationPad3d(3)
        >>> input = torch.randn(16, 3, 8, 320, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ReplicationPad3d((3, 3, 6, 6, 1, 1))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ReplicationPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ZeroPad2d(ConstantPad2d):
    r"""Pads the input tensor boundaries with zero.

    Args:
        padding (int, tuple): the size of the padding. If is int, uses the same
            padding in all boundaries. If a 4-tuple, uses (paddingLeft, paddingRight, paddingTop, paddingBottom)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
          :math:`W_{out} = W_{in} + paddingLeft + paddingRight`

    Examples::

        >>> m = nn.ZeroPad2d(3)
        >>> input = torch.randn(16, 3, 320, 480)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ZeroPad2d((3, 3, 6, 6))
        >>> output = m(input)

    """

    def __init__(self, padding):
        super(ZeroPad2d, self).__init__(padding, 0)
