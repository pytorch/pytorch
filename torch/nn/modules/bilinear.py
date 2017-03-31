import math

import torch
from torch.nn.parameter import Parameter

from .module import Module


class Bilinear(Module):

    def __init__(self, in_features1, in_features2, out_features, bias=True):
        super(Bilinear, self).__init__()
        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features1, in_features2))

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

    def forward(self, input1, input2):
        return self._backend.Bilinear()(input1, input2, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features1) + ' x ' \
            + str(self.in_features2) + ' -> ' \
            + str(self.out_features) + ')'
