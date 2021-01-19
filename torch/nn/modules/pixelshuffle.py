from .module import Module
from .. import functional as F

from torch import Tensor


class PixelShuffle(Module):
    r"""Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \div \text{upscale\_factor}^2

    .. math::
        H_{out} = H_{in} \times \text{upscale\_factor}

    .. math::
        W_{out} = W_{in} \times \text{upscale\_factor}

    Examples::

        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = torch.randn(1, 9, 4, 4)
        >>> output = pixel_shuffle(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """
    __constants__ = ['upscale_factor']
    upscale_factor: int

    def __init__(self, upscale_factor: int) -> None:
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self) -> str:
        return 'upscale_factor={}'.format(self.upscale_factor)


class PixelUnshuffle(Module):
    r"""Reverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements
    in a tensor of shape :math:`(*, C, H \times r, W \times r)` to a tensor of shape
    :math:`(*, C \times r^2, H, W)`, where r is a downscale factor.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.

    Args:
        downscale_factor (int): factor to decrease spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \times \text{downscale\_factor}^2

    .. math::
        H_{out} = H_{in} \div \text{downscale\_factor}

    .. math::
        W_{out} = W_{in} \div \text{downscale\_factor}

    Examples::

        >>> pixel_unshuffle = nn.PixelUnshuffle(3)
        >>> input = torch.randn(1, 1, 12, 12)
        >>> output = pixel_unshuffle(input)
        >>> print(output.size())
        torch.Size([1, 9, 4, 4])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """
    __constants__ = ['downscale_factor']
    downscale_factor: int

    def __init__(self, downscale_factor: int) -> None:
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input: Tensor) -> Tensor:
        return F.pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self) -> str:
        return 'downscale_factor={}'.format(self.downscale_factor)
