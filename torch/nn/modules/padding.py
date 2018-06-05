from .module import Module
from .utils import _pair, _quadruple, _ntuple
from .. import functional as F


# TODO: grad_output size asserts in THNN


class _ConstantPadNd(Module):

    def __init__(self, value):
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, input):
        return F.pad(input, self.padding, 'constant', self.value)

    def extra_repr(self):
        return 'padding={}, value={}'.format(self.padding, self.value)


class ConstantPad1d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses (`paddingLeft`, `paddingRight`)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 4)
        >>> input

        (0 ,.,.) =
          0.1875  0.5046 -1.0074  2.0005
         -0.3540 -1.8645  1.1530  0.0632
        [torch.FloatTensor of size (1,2,4)]

        >>> m(input)

        (0 ,.,.) =
          3.5000  3.5000  0.1875  0.5046 -1.0074  2.0005  3.5000  3.5000
          3.5000  3.5000 -0.3540 -1.8645  1.1530  0.0632  3.5000  3.5000
        [torch.FloatTensor of size (1,2,8)]

        >>> # using different paddings
        >>> m = nn.ConstantPad1d((3, 1), 3.5)
        >>> m(input)

        (0 ,.,.) =
          3.5000  3.5000  3.5000  0.1875  0.5046 -1.0074  2.0005  3.5000
          3.5000  3.5000  3.5000 -0.3540 -1.8645  1.1530  0.0632  3.5000
        [torch.FloatTensor of size (1,2,8)]

    """

    def __init__(self, padding, value):
        super(ConstantPad1d, self).__init__(value)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (`paddingLeft`, `paddingRight`,
            `paddingTop`, `paddingBottom`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + \textit{paddingTop} + \textit{paddingBottom}`
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ConstantPad2d(2, 3.5)
        >>> input = torch.randn(1, 2, 2)
        >>> input

        (0 ,.,.) =
         -0.2295 -0.9774
         -0.3335 -1.4178
        [torch.FloatTensor of size (1,2,2)]

        >>> m(input)

        (0 ,.,.) =
          3.5000  3.5000  3.5000  3.5000  3.5000  3.5000
          3.5000  3.5000  3.5000  3.5000  3.5000  3.5000
          3.5000  3.5000 -0.2295 -0.9774  3.5000  3.5000
          3.5000  3.5000 -0.3335 -1.4178  3.5000  3.5000
          3.5000  3.5000  3.5000  3.5000  3.5000  3.5000
          3.5000  3.5000  3.5000  3.5000  3.5000  3.5000
        [torch.FloatTensor of size (1,6,6)]

        >>> # using different paddings
        >>> m = nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        >>> m(input)

        (0 ,.,.) =
          3.5000  3.5000  3.5000  3.5000  3.5000
          3.5000  3.5000  3.5000  3.5000  3.5000
          3.5000  3.5000  3.5000 -0.2295 -0.9774
          3.5000  3.5000  3.5000 -0.3335 -1.4178
          3.5000  3.5000  3.5000  3.5000  3.5000
        [torch.FloatTensor of size (1,5,5)]

    """

    def __init__(self, padding, value):
        super(ConstantPad2d, self).__init__(value)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (`paddingLeft`, `paddingRight`, `paddingTop`, `paddingBottom`, `paddingFront`, `paddingBack`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = D_{in} + \textit{paddingFront} + \textit{paddingBack}`
          :math:`H_{out} = H_{in} + \textit{paddingTop} + \textit{paddingBottom}`
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ConstantPad3d(3, 3.5)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings
        >>> m = nn.ConstantPad3d((3, 3, 6, 6, 0, 1), 3.5)
        >>> output = m(input)

    """

    def __init__(self, padding, value):
        super(ConstantPad3d, self).__init__(value)
        self.padding = _ntuple(6)(padding)


class _ReflectionPadNd(Module):

    def forward(self, input):
        return F.pad(input, self.padding, 'reflect')

    def extra_repr(self):
        return '{}'.format(self.padding)


class ReflectionPad1d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses (`paddingLeft`, `paddingRight`)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ReflectionPad1d(2)
        >>> input = torch.arange(8).reshape(1, 2, 4)
        >>> input

        (0 ,.,.) =
          0  1  2  3
          4  5  6  7
        [torch.FloatTensor of size (1,2,4)]

        >>> m(input)

        (0 ,.,.) =
           2   1   0   1   2   3   2   1
           6   5   4   5   6   7   6   5
        [torch.FloatTensor of size (1,2,8)]

        >>> # using different paddings
        >>> m = nn.ReflectionPad1d((3, 1))
        >>> m(input)

        (0 ,.,.) =
           3   2   1   0   1   2   3   2
           7   6   5   4   5   6   7   6
        [torch.FloatTensor of size (1,2,8)]

    """

    def __init__(self, padding):
        super(ReflectionPad1d, self).__init__()
        self.padding = _pair(padding)


class ReflectionPad2d(_ReflectionPadNd):
    r"""Pads the input tensor using the reflection of the input boundary.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (`paddingLeft`, `paddingRight`,
            `paddingTop`, `paddingBottom`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + \textit{paddingTop} + \textit{paddingBottom}`
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ReflectionPad2d(2)
        >>> input = torch.arange(9).reshape(1, 1, 3, 3)
        >>> input

        (0 ,0 ,.,.) =
          0  1  2
          3  4  5
          6  7  8
        [torch.FloatTensor of size (1,1,3,3)]

        >>> m(input)

        (0 ,0 ,.,.) =
           8   7   6   7   8   7   6
           5   4   3   4   5   4   3
           2   1   0   1   2   1   0
           5   4   3   4   5   4   3
           8   7   6   7   8   7   6
           5   4   3   4   5   4   3
           2   1   0   1   2   1   0
        [torch.FloatTensor of size (1,1,7,7)]

        >>> # using different paddings
        >>> m = nn.ReflectionPad2d((1, 1, 2, 0))
        >>> m(input)

        (0 ,0 ,.,.) =
          7  6  7  8  7
          4  3  4  5  4
          1  0  1  2  1
          4  3  4  5  4
          7  6  7  8  7
        [torch.FloatTensor of size (1,1,5,5)]

    """

    def __init__(self, padding):
        super(ReflectionPad2d, self).__init__()
        self.padding = _quadruple(padding)


class _ReplicationPadNd(Module):

    def forward(self, input):
        return F.pad(input, self.padding, 'replicate')

    def extra_repr(self):
        return '{}'.format(self.padding)


class ReplicationPad1d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 2-`tuple`, uses (`paddingLeft`, `paddingRight`)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ReplicationPad1d(2)
        >>> input = torch.arange(8).reshape(1, 2, 4)
        >>> input

        (0 ,.,.) =
          0  1  2  3
          4  5  6  7
        [torch.FloatTensor of size (1,2,4)]

        >>> m(input)

        (0 ,.,.) =
           0   0   0   1   2   3   3   3
           4   4   4   5   6   7   7   7
        [torch.FloatTensor of size (1,2,8)]

        >>> # using different paddings
        >>> m = nn.ReplicationPad1d((3, 1))
        >>> m(input)

        (0 ,.,.) =
           0   0   0   0   1   2   3   3
           4   4   4   4   5   6   7   7
        [torch.FloatTensor of size (1,2,8)]

    """

    def __init__(self, padding):
        super(ReplicationPad1d, self).__init__()
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (`paddingLeft`, `paddingRight`,
            `paddingTop`, `paddingBottom`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + \textit{paddingTop} + \textit{paddingBottom}`
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ReplicationPad2d(2)
        >>> input = torch.arange(9).reshape(1, 1, 3, 3)
        >>> input

        (0 ,0 ,.,.) =
          0  1  2
          3  4  5
          6  7  8
        [torch.FloatTensor of size (1,1,3,3)]

        >>> m(input)

        (0 ,0 ,.,.) =
           0   0   0   1   2   2   2
           0   0   0   1   2   2   2
           0   0   0   1   2   2   2
           3   3   3   4   5   5   5
           6   6   6   7   8   8   8
           6   6   6   7   8   8   8
           6   6   6   7   8   8   8
        [torch.FloatTensor of size (1,1,7,7)]

        >>> # using different paddings
        >>> m = nn.ReplicationPad2d((1, 1, 2, 0))
        >>> m(input)

        (0 ,0 ,.,.) =
          0  0  1  2  2
          0  0  1  2  2
          0  0  1  2  2
          3  3  4  5  5
          6  6  7  8  8
        [torch.FloatTensor of size (1,1,5,5)]

    """

    def __init__(self, padding):
        super(ReplicationPad2d, self).__init__()
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    r"""Pads the input tensor using replication of the input boundary.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses (`paddingLeft`, `paddingRight`,
            `paddingTop`, `paddingBottom`, `paddingFront`, `paddingBack`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = D_{in} + \textit{paddingFront} + \textit{paddingBack}`
          :math:`H_{out} = H_{in} + \textit{paddingTop} + \textit{paddingBottom}`
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

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


class ZeroPad2d(ConstantPad2d):
    r"""Pads the input tensor boundaries with zero.

    For `N`d-padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 4-`tuple`, uses (`paddingLeft`, `paddingRight`,
            `paddingTop`, `paddingBottom`)

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = H_{in} + \textit{paddingTop} + \textit{paddingBottom}`
          :math:`W_{out} = W_{in} + \textit{paddingLeft} + \textit{paddingRight}`

    Examples::

        >>> m = nn.ZeroPad2d(2)
        >>> input = torch.randn(1, 1, 3, 3)
        >>> input

        (0 ,0 ,.,.) =
          1.4418 -1.9812 -0.3815
         -0.3828 -0.6833 -0.2376
          0.1433  0.0211  0.4311
        [torch.FloatTensor of size (1,1,3,3)]

        >>> m(input)

        (0 ,0 ,.,.) =
          0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
          0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
          0.0000  0.0000  1.4418 -1.9812 -0.3815  0.0000  0.0000
          0.0000  0.0000 -0.3828 -0.6833 -0.2376  0.0000  0.0000
          0.0000  0.0000  0.1433  0.0211  0.4311  0.0000  0.0000
          0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
          0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
        [torch.FloatTensor of size (1,1,7,7)]

        >>> # using different paddings
        >>> m = nn.ZeroPad2d((1, 1, 2, 0))
        >>> m(input)

        (0 ,0 ,.,.) =
          0.0000  0.0000  0.0000  0.0000  0.0000
          0.0000  0.0000  0.0000  0.0000  0.0000
          0.0000  1.4418 -1.9812 -0.3815  0.0000
          0.0000 -0.3828 -0.6833 -0.2376  0.0000
          0.0000  0.1433  0.0211  0.4311  0.0000
        [torch.FloatTensor of size (1,1,5,5)]

    """

    def __init__(self, padding):
        super(ZeroPad2d, self).__init__(padding, 0)
