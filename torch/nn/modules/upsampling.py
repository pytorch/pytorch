from numbers import Integral
import warnings

from .module import Module
from .. import functional as F
from .utils import _pair, _triple


class Upsample(Module):
    """
    Upsamples a given multi-channel 2D (spatial) or 3D (volumetric) data.

    The input data is assumed to be of the form `minibatch x channels x [depth] x height x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor, bilinear and trilinear upsampling,
    with bilinear only available for 4D Tensor inputs and trilinear for 4D Tensor inputs.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (tuple, optional): a tuple of ints ([D_out], H_out, W_out) output sizes
        scale_factor (int / tuple of ints, optional): the multiplier for the image height / width / depth
        mode (string, optional): the upsampling algorithm: nearest | bilinear | trilinear. Default: nearest

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(N, C, D_{out}, H_{out}, W_{out})` where
          :math:`D_{out} = floor(D_{in} * scale\_factor)` or `size[-3]`
          :math:`H_{out} = floor(H_{in} * scale\_factor)` or `size[-2]`
          :math:`W_{out} = floor(W_{in}  * scale\_factor)` or `size[-1]`

    Examples::

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.Upsample(scale_factor=2, mode='bilinear')
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1.0000  1.3333  1.6667  2.0000
          1.6667  2.0000  2.3333  2.6667
          2.3333  2.6667  3.0000  3.3333
          3.0000  3.3333  3.6667  4.0000
        [torch.FloatTensor of size 1x1x4x4]

        >>> inp
        Variable containing:
        (0 ,0 ,.,.) =
          1  2
          3  4
        [torch.FloatTensor of size 1x1x2x2]

        >>> m = nn.Upsample(scale_factor=2, mode='nearest')
        >>> m(inp)
        Variable containing:
        (0 ,0 ,.,.) =
          1  1  2  2
          1  1  2  2
          3  3  4  4
          3  3  4  4
        [torch.FloatTensor of size 1x1x4x4]


    """

    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, input):
        return F.upsample(input, self.size, self.scale_factor, self.mode)

    def __repr__(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return self.__class__.__name__ + '(' + info + ')'


class UpsamplingNearest2d(Upsample):
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
          :math:`H_{out} = floor(H_{in} * scale\_factor)`
          :math:`W_{out} = floor(W_{in}  * scale\_factor)`

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
    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingNearest2d, self).__init__(size, scale_factor, mode='nearest')

    def forward(self, input):
        warnings.warn("nn.UpsamplingNearest2d is deprecated. Use nn.Upsample instead.")
        return super(UpsamplingNearest2d, self).forward(input)


class UpsamplingBilinear2d(Upsample):
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
          :math:`H_{out} = floor(H_{in} * scale\_factor)`
          :math:`W_{out} = floor(W_{in}  * scale\_factor)`

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
    def __init__(self, size=None, scale_factor=None):
        super(UpsamplingBilinear2d, self).__init__(size, scale_factor, mode='bilinear')

    def forward(self, input):
        warnings.warn("nn.UpsamplingBilinear2d is deprecated. Use nn.Upsample instead.")
        return super(UpsamplingBilinear2d, self).forward(input)
