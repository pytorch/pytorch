from numbers import Integral

from .module import Module
from .. import functional as F
from .utils import _pair


class _UpsamplingBase(Module):

    def __init__(self, size=None, scale_factor=None):
        super(_UpsamplingBase, self).__init__()
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if scale_factor is not None and not isinstance(scale_factor, Integral):
            raise ValueError('scale_factor must be of integer type')
        self.size = _pair(size)
        self.scale_factor = scale_factor

    def __repr__(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        return self.__class__.__name__ + '(' + info + ')'


class UpsamplingNearest2d(_UpsamplingBase):
    """
    Applies a 2D nearest neighbor upsampling to an input signal composed of several input
    channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When `size` is given, it is the output size of the image (h, w).

    Args:
        size (tuple, optional): a tuple of ints (H_out, W_out) output sizes
        scale_factor (int, optional): the multiplier for the image height / width

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in} * scale_factor`
          :math:`W_{out} = floor((W_{in}  * scale_factor`

    Examples::

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.UpsamplingNearest2d(scale_factor=2)
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1  1  2  2
          1  1  2  2
          3  3  4  4
          3  3  4  4
        [torch.FloatTensor of size 1x1x4x4]

    """

    def forward(self, input):
        return F.upsample_nearest(input, self.size, self.scale_factor)


class UpsamplingBilinear2d(_UpsamplingBase):
    """
    Applies a 2D bilinear upsampling to an input signal composed of several input
    channels.

    To specify the scale, it takes either the :attr:`size` or the :attr:`scale_factor`
    as it's constructor argument.

    When `size` is given, it is the output size of the image (h, w).

    Args:
        size (tuple, optional): a tuple of ints (H_out, W_out) output sizes
        scale_factor (int, optional): the multiplier for the image height / width

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in} * scale_factor`
          :math:`W_{out} = floor((W_{in}  * scale_factor`

    Examples::

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.UpsamplingBilinear2d(scale_factor=2)
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1.0000  1.3333  1.6667  2.0000
          1.6667  2.0000  2.3333  2.6667
          2.3333  2.6667  3.0000  3.3333
          3.0000  3.3333  3.6667  4.0000
        [torch.FloatTensor of size 1x1x4x4]

    """

    def forward(self, input):
        return F.upsample_bilinear(input, self.size, self.scale_factor)
