import math

import torch
from torch.autograd import Variable

from .module import Module


class Linear(Module):
    """Applies a linear transformation to the incoming data, y = Ax + b
    The input is a 2D mini-batch of samples, each of size in_features
    The output will be a 2D Tensor of size mini-batch x out_features

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
    Input Shape: [*, in_features] : Input can be of shape minibatch x in_features
    Output Shape:[*, out_features]  : Output is of shape minibatch x out_features
    Members:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples:
        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features

        super(Linear, self).__init__(
            weight=torch.Tensor(out_features, in_features),
            bias=torch.Tensor(out_features) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.bias is None:
            return self._backend.Linear()(input, self.weight)
        else:
            return self._backend.Linear()(input, self.weight, self.bias)


# TODO: Bilinear
# TODO: PartialLinear - maybe in sparse?
