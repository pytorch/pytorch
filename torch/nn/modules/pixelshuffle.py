from .module import Module
from .. import functional as F


class PixelShuffle(Module):
    r"""Rearranges elements in a Tensor of shape :math:`(*, C * r^2, H, W]` to a
    tensor of shape :math:`(C, H * r, W * r)`.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(N, C * {upscale\_factor}^2, H, W)`
        - Output: :math:`(N, C, H * {upscale\_factor}, W * {upscale\_factor})`

    Examples::

        >>> ps = nn.PixelShuffle(3)
        >>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
        >>> output = ps(input)
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

    def __repr__(self):
        return self.__class__.__name__ + ' (upscale_factor=' + str(self.upscale_factor) + ')'
