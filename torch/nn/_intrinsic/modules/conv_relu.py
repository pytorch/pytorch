from __future__ import absolute_import, division, print_function, unicode_literals

from torch.nn.modules import Conv2d

class ConvReLU2d(Conv2d):
    r"""Conv2D followed by a RELU operation on input.


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input),m(-input)))
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        super(ConvReLU2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

    def forward(self, input):
        output = super(ConvReLU2d, self).forward(input)
        return F.relu(output, inplace=True)

    @staticmethod
    def from_modules(conv, relu):
        weight = conv.weight
        bias = conv.bias
        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation
        transposed = conv.transposed
        output_padding = conv.output_padding
        groups = conv.groups
        padding_mode = conv.padding_mode

        convrelu = Conv2dReLU(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups,
                              bias is not None, padding_mode)
        return convrelu
