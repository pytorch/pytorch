import math

import torch
from torch.nn.parameter import Parameter

from .module import Module


class Linear(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True

    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.bias is None:
            return self._backend.Linear()(input, self.weight)
        else:
            return self._backend.Linear()(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class BiLinear(Module):
    r"""Applies a bilinear transformation to the incoming data: :math:`\forall k: y_k = x_1 A_k x_2 + b`

    Args:
        in_features_1: size of each input sample of the first input
        in_features_2: size of each input sample of the second input
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True

    Shape:
        - Input: :math:`[(N, in\_features\_1), (N, in\_features\_2)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features_1 x in_features_2)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = nn.BiLinear(10, 20, 30)
        >>> input_1 = autograd.Variable(torch.randn(128, 20))
        >>> input_2 = autograd.Variable(torch.randn(128, 30))
        >>> output = m([input_1, input_2])
        >>> print(output.size())
    """

    def __init__(self, in_features_1, in_features_2, out_features, bias=True):
        super(BiLinear, self).__init__()
        self.in_features_1 = in_features_1
        self.in_features_2 = in_features_2
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features_1, in_features_2))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        assert len(input) == 2
        output = torch.cat([input[0].mm(self.weight[i]).mul(input[1]).sum(1)
                            for i in range(self.out_features)], 1)
        if self.bias is not None:
            output.add_(self.bias.expand_as(output))
        return output

    def __repr__(self):
        buf = self.__class__.__name__ + ' ((' \
            + str(self.in_features_1) + ',' \
            + str(self.in_features_2) + ') -> ' \
            + str(self.out_features) + ')'
        if self.bias is None:
            buf += ' without bias'
        return buf


# TODO: PartialLinear - maybe in sparse?
