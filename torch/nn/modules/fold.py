# coding=utf-8
from .module import Module
from .. import functional as F
from ..._jit_internal import weak_module, weak_script_method


@weak_module
class Fold(Module):
    r"""Combines an array of sliding local blocks into a large containing
    tensor.

    Consider a batched :attr:`input` tensor containing sliding local blocks,
    e.g., patches of images, of shape :math:`(N, C \times  \prod(\text{kernel\_size}), L)`,
    where :math:`N` is batch dimension, :math:`C \times \prod(\text{kernel\_size})`
    is the number of values within a block (a block has :math:`\prod(\text{kernel\_size})`
    spatial locations each containing a :math:`C`-channeled vector), and
    :math:`L` is the total number of blocks. (This is exactly the
    same specification as the output shape of :class:`~torch.nn.Unfold`.) This
    operation combines these local blocks into the large :attr:`output` tensor
    of shape :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)`
    by summing the overlapping values. Similar to :class:`~torch.nn.Unfold`, the
    arguments must satisfy

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`d` is over all spatial dimensions.

    * :attr:`output_size` describes the spatial shape of the large containing
      tensor of the sliding local blocks. It is useful to resolve the ambiguity
      when multiple input shapes map to same number of sliding blocks, e.g.,
      with ``stride > 0``.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        output_size (int or tuple): the shape of the spatial dimensions of the
                                    output (i.e., ``output.sizes()[2:]``)
        kernel_size (int or tuple): the size of the sliding blocks
        stride (int or tuple): the stride of the sliding blocks in the input
                               spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    * If :attr:`output_size`, :attr:`kernel_size`, :attr:`dilation`,
      :attr:`padding` or :attr:`stride` is an int or a tuple of length 1 then
      their values will be replicated across all spatial dimensions.

    * For the case of two output spatial dimensions this operation is sometimes
      called ``col2im``.

    .. note::
        :class:`~torch.nn.Fold` calculates each combined value in the resulting
        large tensor by summing all values from all containing blocks.
        :class:`~torch.nn.Unfold` extracts the values in the local blocks by
        copying from the large tensor. So, if the blocks overlap, they are not
        inverses of each other.

    .. warning::
        Currently, only 4-D output tensors (batched image-like tensors) are
        supported.

    Shape:
        - Input: :math:`(N, C \times \prod(\text{kernel\_size}), L)`
        - Output: :math:`(N, C, \text{output\_size}[0], \text{output\_size}[1], \dots)` as described above

    Examples::

        >>> fold = nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
        >>> input = torch.randn(1, 3 * 2 * 2, 12)
        >>> output = fold(input)
        >>> output.size()
        torch.Size([1, 3, 4, 5])

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    """
    __constants__ = ['output_size', 'kernel_size', 'dilation', 'padding',
                     'stride']

    def __init__(self, output_size, kernel_size, dilation=1, padding=0, stride=1):
        super(Fold, self).__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    @weak_script_method
    def forward(self, input):
        return F.fold(input, self.output_size, self.kernel_size, self.dilation,
                      self.padding, self.stride)

    def extra_repr(self):
        return 'output_size={output_size}, kernel_size={kernel_size}, ' \
            'dilation={dilation}, padding={padding}, stride={stride}'.format(
                **self.__dict__
            )


@weak_module
class Unfold(Module):
    r"""Extracts sliding local blocks from a batched input tensor.

    Consider an batched :attr:`input` tensor of shape :math:`(N, C, *)`,
    where :math:`N` is the batch dimension, :math:`C` is the channel dimension,
    and :math:`*` represent arbitrary spatial dimensions. This operation flattens
    each sliding :attr:`kernel_size`-sized block within the spatial dimensions
    of :attr:`input` into a column (i.e., last dimension) of a 3-D :attr:`output`
    tensor of shape :math:`(N, C \times \prod(\text{kernel\_size}), L)`, where
    :math:`C \times \prod(\text{kernel\_size})` is the total number of values
    within each block (a block has :math:`\prod(\text{kernel\_size})` spatial
    locations each containing a :math:`C`-channeled vector), and :math:`L` is
    the total number of such blocks:

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial\_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel\_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    where :math:`\text{spatial\_size}` is formed by the spatial dimensions
    of :attr:`input` (:math:`*` above), and :math:`d` is over all spatial
    dimensions.

    Therefore, indexing :attr:`output` at the last dimension (column dimension)
    gives all values within a certain block.

    The :attr:`padding`, :attr:`stride` and :attr:`dilation` arguments specify
    how the sliding blocks are retrieved.

    * :attr:`stride` controls the stride for the sliding blocks.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension before
      reshaping.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.

    Args:
        kernel_size (int or tuple): the size of the sliding blocks
        stride (int or tuple, optional): the stride of the sliding blocks in the input
                                         spatial dimensions. Default: 1
        padding (int or tuple, optional): implicit zero padding to be added on
                                          both sides of input. Default: 0
        dilation (int or tuple, optional): a parameter that controls the
                                           stride of elements within the
                                           neighborhood. Default: 1

    * If :attr:`kernel_size`, :attr:`dilation`, :attr:`padding` or
      :attr:`stride` is an int or a tuple of length 1, their values will be
      replicated across all spatial dimensions.

    * For the case of two input spatial dimensions this operation is sometimes
      called ``im2col``.

    .. note::
        :class:`~torch.nn.Fold` calculates each combined value in the resulting
        large tensor by summing all values from all containing blocks.
        :class:`~torch.nn.Unfold` extracts the values in the local blocks by
        copying from the large tensor. So, if the blocks overlap, they are not
        inverses of each other.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C \times \prod(\text{kernel\_size}), L)` as described above

    Examples::

        >>> unfold = nn.Unfold(kernel_size=(2, 3))
        >>> input = torch.randn(2, 5, 3, 4)
        >>> output = unfold(input)
        >>> # each patch contains 30 values (2x3=6 vectors, each of 5 channels)
        >>> # 4 blocks (2x3 kernels) in total in the 3x4 input
        >>> output.size()
        torch.Size([2, 30, 4])

        >>> # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
        >>> inp = torch.randn(1, 3, 10, 12)
        >>> w = torch.randn(2, 3, 4, 5)
        >>> inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        >>> out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        >>> out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        >>> # or equivalently (and avoiding a copy),
        >>> # out = out_unf.view(1, 2, 7, 8)
        >>> (torch.nn.functional.conv2d(inp, w) - out).abs().max()
        tensor(1.9073e-06)

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    """
    __constants__ = ['kernel_size', 'dilation', 'padding', 'stride']

    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super(Unfold, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    @weak_script_method
    def forward(self, input):
        return F.unfold(input, self.kernel_size, self.dilation,
                        self.padding, self.stride)

    def extra_repr(self):
        return 'kernel_size={kernel_size}, dilation={dilation}, padding={padding},' \
            ' stride={stride}'.format(**self.__dict__)
