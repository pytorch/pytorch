from .module import Module


class PixelShuffle(Module):
    """Rearranges elements in a tensor of shape [*, C*r^2, H, W] to a
    tensor of shape [C, H*r, W*r]. This is useful for implementing
    efficient sub-pixel convolution with a stride of 1/r.
    "Real-Time Single Image and Video Super-Resolution Using an Efficient
    Sub-Pixel Convolutional Neural Network" - Shi et. al (2016) for more details
    Args:
        upscale_factor (int): factor to increase spatial resolution by
    Input Shape: [*, channels*upscale_factor^2, height, width]
    Output Shape:[*, channels, height*upscale_factor, width*upscale_factor]
    Examples:
        >>> ps = nn.PixelShuffle(3)
        >>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        batch_size, channels, in_height, in_width = input.size()
        channels //= self.upscale_factor ** 2

        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = input.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor,
            in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        return shuffle_out.view(batch_size, channels, out_height, out_width)

    def __repr__(self):
        return self.__class__.__name__ + ' (upscale_factor=' + str(self.upscale_factor) + ')'
