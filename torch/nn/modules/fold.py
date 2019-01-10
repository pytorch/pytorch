from .module import Module
from .. import functional as F


class Fold(Module):
    """
    De-interleaves vectors of length :math:`\prod(kernel_size)` from the "channel"
    dimension of the input tensor to generate blocks of size :math:`kernel_size`
    of the output.  These blocks populate the "spatial" dimensions [2:]
    of the output via a sliding window with positions determined by the
    padding, stride and dilation values.  The "channel" dimension 1 of the output
    is determined by the vectors interleaevd position in the "channel" dimension
    of the input.

    Each element of the output batch dimension 0 has :math:`C / \prod(kernel_size)`
    channels (dimension 1) and spatial dimensions [2:] of shape :math:`output_size`.

    | If :attr:`padding` is non-zero, then the input is implicitly
    zero-padded on both sides by :attr:`padding` number of points
    | :attr:`dilation` controls the intenal spacing between the kernel points in the output.
    It is harder to describe, but this `link`_ has a nice visualization of what
    dilation does.

    Args:
        output_size (int or tuple): the shape of the spatial dimensions [2:] of the output
        kernel_size (int or tuple): the size of the sliding blocks to convert
                                    to columns.
        stride (int or tuple): the stride of the sliding blocks in the input
                               spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    | If :attr:`output_size`, :attr:`kernel_size`, :attr:`dilation`,
    :attr:`padding` or :attr:`stride` is of length 1 then
    their value will be replicated across all spatial dimensions

    | For the case of two output spatial dimensions this operation is sometimes called col2im

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N * C * \prod(kernel_size), L_{out},)` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    Examples::

        >>> # output_size (3, 3) kernel_size (2, 2), dilation (1, 1), padding (0, 0), stride (1, 1)
        >>> fold = nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1))
        >>> input = torch.randn(1, 36, 1)
        >>> output = unfold(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    """

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super(Fold, self).__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return F.fold(input, self.output_size, self.kernel_size, self.dilation,
                      self.padding, self.stride)

    def extra_repr(self):
        return 'output_size={output_size}, kernel_size={kernel_size}, ' \
            'dilation={dilation}, padding={padding}, stride={stride}'.format(
                **self.__dict__
            )


class Unfold(Module):
    """

    Converts each sliding :math:`kernel_size` block of the "spatial" dimensions [2:]
    of the input tensor into a column of the output. These columns are interleaved
    with the "channel" dimension 1 such that in the output the channel dimension combines
    both the spatial position of the block within the input and the original
    channel position. We denote size of the "batch" dimension 0 as :math:`N`.

    Each element of the output batch dimension 0 has :math:`C * \prod(kernel_size)`
    rows and contains as many columns as there are :math:`kernel_size` neighborhoods
    of the input according to the padding, stride and dilation values.

    | If :attr:`padding` is non-zero, then the input is implicitly
    zero-padded on both sides by :attr:`padding` number of points before reshaping
    | :attr:`dilation` controls the internal spacing between the kernel points.
    It is harder to describe, but this `link`_ has a nice visualization of what
    dilation does.

    Args:
        kernel_size (int or tuple): the size of the sliding blocks to convert
                                    to columns.
        stride (int or tuple, optional): the stride of the sliding blocks in the input
                                         spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    | If :attr:`kernel_size`, :attr:`dilation`, :attr:`padding` or :attr:`stride`
    is of length 1 then their value will be replicated across all spatial dimensions

    | For the case of two input spatial dimensions this operation is sometimes called im2col

    Shape:
        - Input: :math:`(N, C, L_{in})`
        - Output: :math:`(N, C * \prod(kernel_size), L_{out},)` where
          :math:`L_{out} = floor((L_{in} + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    Examples::

        >>> # kernel_size (2, 2), dilation (1, 1), padding (0, 0), stride (1, 1)
        >>> unfold = nn.Unfold((3, 3), (1, 1), (0, 0), (1, 1))
        >>> input = torch.randn(2, 4, 3, 3)
        >>> output = unfold(input)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    """

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, input):
        return F.unfold(input, self.kernel_size, self.dilation,
                        self.padding, self.stride)

    def extra_repr(self):
        return 'kernel_size={kernel_size}, dilation={dilation}, padding={padding},' \
            ' stride={stride}'.format(**self.__dict__)
