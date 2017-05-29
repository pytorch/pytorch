import math
import torch
from torch.nn.parameter import Parameter
from .. import functional as F
from .module import Module
from .utils import _pair, _triple


class _SubsamplingNd(Module):

    def __init__(self, in_channels, size, stride):
        super(_SubsamplingNd, self).__init__()
        self.in_channels = in_channels
        self.size = size
        self.stride = stride
        self.weight = Parameter(torch.Tensor(in_channels))
        self.bias = Parameter(torch.Tensor(in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = 1.0
        for k in self.size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, size={size}, stride={stride})')
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Subsampling2d(_SubsamplingNd):
    r"""Subsamples the input by averaging over input neighborhoods of given size and stride,
    multiplying by weight and adding bias.
    The module preserves the number of channels (``in_channels == out_channels``).
    Weight and bias are learned 1D tensors of size (in_channels).

    The output value of the layer with input size :math:`(N, C_{in}, H, W)` can be described
    as

    .. math::

        \begin{array}{ll}
        out(N_i, C_{in_j}) = bias(C_{in_j}
                      + weight(C_{in_j}) * 1_{size_0,size_1} \star input(N_i, C_{in_j})
        \end{array}

    where :math:`\star` is the valid 2D `cross-correlation`_ operator and
    :math:`1_{size_0,size_1}` is a one tensor of rank 2, and size :math:`(size_0,size_1)`

    | :attr:`size` controls the size of the identity tensor
    | :attr:`stride` controls the stride for the cross-correlation

    The parameters :attr:`size` and :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case the first `int` is used for the height dimension
          and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        size (int or tuple): Size of the averaging operation
        stride (int or tuple, optional): Stride of the averaging operation

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{in}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in} - size[1]) / stride[1] + 1)
          :math:`W_{out} = floor((W_{in} - size[0]) / stride[0] + 1)

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (in_channels)
        bias (Tensor):   the learnable bias of the module of shape (in_channels)

    Examples:

        >>> # With square averaging and equal stride
        >>> m = nn.Subsample2d(10, 3, stride=2)
        >>> # non-square averaging and unequal stride
        >>> m = nn.Subsample2d(10, (2,3), stride=(1,2))
        >>> input = autograd.Variable(torch.randn(10,2,8,8)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    """

    def __init__(self, in_channels, size, stride=1):
        size = _pair(size)
        stride = _pair(stride)
        super(Subsampling2d, self).__init__(in_channels, size, stride)

    def forward(self, input):
        return F.subsample(input, self.weight, self.bias, self.size, self.stride)


class Subsampling3d(_SubsamplingNd):
    r"""Subsamples the input by averaging over input neighborhoods of given size and stride,
    multiplying by weight and adding bias.
    The module preserves the number of channels (``in_channels == out_channels``).
    Weight and bias are learned 1D tensors of size (in_channels).

    The output value of the layer with input size :math:`(N, C_{in}, D, H, W)` can be described
    as

    .. math::

        \begin{array}{ll}
        out(N_i, C_{in_j}) = bias(C_{in_j}
                      + weight(C_{in_j}) * 1_{size_0,size_1,size_2} \star input(N_i, C_{in_j})
        \end{array}

    where :math:`\star` is the valid 2D `cross-correlation`_ operator and
    :math:`1_{size_0,size_1,size_2}` is a one tensor of rank 3 and size :math:`(size_0,size_1,size_2)`

    | :attr:`size` controls the size of the identity tensor
    | :attr:`stride` controls the stride for the cross-correlation

    The parameters :attr:`size` and :attr:`stride` can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height
          and width dimension
        - a ``tuple`` of three ints -- in which case the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        size (int or tuple): Size of the averaging operation
        stride (int or tuple, optional): Stride of the averaging operation

    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{in}, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = floor((D_{in} - size[2]) / stride[2] + 1)
          :math:`H_{out} = floor((H_{in} - size[1]) / stride[1] + 1)
          :math:`W_{out} = floor((W_{in} - size[0]) / stride[0] + 1)

    Attributes:
        weight (Tensor): the learnable weights of the module of shape (in_channels)
        bias (Tensor):   the learnable bias of the module of shape (in_channels)

    Examples:

        >>> # With square averaging and equal stride
        >>> m = nn.Subsample2d(10, 3, stride=2)
        >>> # non-square averaging and unequal stride
        >>> m = nn.Subsample2d(10, (2,2,3), stride=(1,1,2))
        >>> input = autograd.Variable(torch.randn(10,2,8,8,8)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    """

    def __init__(self, in_channels, size, stride=1):
        size = _triple(size)
        stride = _triple(stride)
        super(Subsampling3d, self).__init__(in_channels, size, stride)

    def forward(self, input):
        return F.subsample(input, self.weight, self.bias, self.size, self.stride)
