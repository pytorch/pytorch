import math

import torch
from torch.autograd import Variable

from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Variable(torch.DoubleTensor(out_features, in_features))
        self.bias = Variable(torch.DoubleTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self._backend.Linear()(input, self.weight, self.bias)
