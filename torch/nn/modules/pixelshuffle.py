from .module import Module
from .. import functional as F


class PixelShuffle(Module):
    r"""Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(C, H \times r, W \times r)`.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.

    This is the inverse operation of :class:`~torch.nn.PixelUnshuffle`.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(N, C \times \text{upscale_factor}^2, H, W)`
        - Output: :math:`(N, C, H \times \text{upscale_factor}, W \times \text{upscale_factor})`

    Examples::

        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = torch.randn(1, 9, 4, 4)
        >>> output = pixel_shuffle(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class PixelUnshuffle(Module):
    r"""Rearranges elements in a tensor of shape
    :math:`(C, H \times r, W \times r)` to a tensor of shape
    :math:`(*, C \times r^2, H, W)`.

    This is the inverse operation of :class:`~torch.nn.PixelShuffle`, which is
    used in implementing efficient sub-pixel convolution with a stride of
    :math:`1/r`.

    Args:
        downscale_factor (int): factor to decrease spatial resolution by

    Shape:
        - Input: :math:`(N, C, H \times \text{downscale_factor}, W \times \text{downscale_factor})`
        - Output: :math:`(N, C \times \text{downscale_factor}^2, H, W)`

    Examples::

        >>> pixel_unshuffle = nn.PixelUnshuffle(3)
        >>> input = torch.randn(1, 1, 12, 12)
        >>> output = pixel_unshuffle(input)
        >>> print(output.size())
        torch.Size([1, 9, 4, 4])

        >>> # inverse of nn.PixelShuffle
        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = torch.randn(1, 9, 4, 4)
        >>> output = pixel_shuffle(input)
        >>> recovered = pixel_unshuffle(output)
        >>> print((input - recovered).abs().max())
        tensor(0.)

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self):
        return 'downscale_factor={}'.format(self.downscale_factor)
