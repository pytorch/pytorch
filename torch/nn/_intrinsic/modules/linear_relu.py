from __future__ import absolute_import, division, print_function, unicode_literals

from torch.nn.modules import Linear


class LinearReLU(Linear):

    r"""Applies linear to the incoming data followed by a ReLU: :math:`relu(y = xA^T + b)`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

        inplace: can optionally do the relu operation in-place. Default: ``False``


    Examples::

        >>> m = nn._intrinsic.LinearRelu(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    def __init__(self, in_features, out_features, bias=True):

        super(LinearReLU, self).__init__(
            in_features, out_features, bias)

    def forward(self, input):
        output = super(LinearReLU, self).forward(input)
        return F.relu(output, inplace=True)

    @staticmethod
    def from_modules(linear, relu):
        weight = linear.weight
        bias = linear.bias
        in_features = linear.in_features
        out_features = linear.out_features

        linearrelu = LinearReLU(in_features, out_features, bias)
        return linearrelu
