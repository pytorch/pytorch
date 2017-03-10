import torch
from torch.autograd.function import InplaceFunction
from itertools import repeat


class Dropout(InplaceFunction):

    def __init__(self, p=0.5, train=False, inplace=False, v2=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.train = train
        self.inplace = inplace
        self.v2 = v2

    def _make_noise(self, input):
        return input.new().resize_as_(input)

    def forward(self, input):
        if self.inplace:
            self.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if self.p > 0:
            if self.train:
                self.noise = self._make_noise(input)
                self.noise.bernoulli_(1 - self.p)
                if not self.v2:
                    self.noise.div_(1 - self.p)
                if self.p == 1:
                    self.noise.fill_(0)
                self.noise = self.noise.expand_as(input)
                output.mul_(self.noise)
            elif self.v2:
                output.mul_(1 - self.p)

        return output

    def backward(self, grad_output):
        if self.p > 0:
            if self.train:
                return grad_output.mul(self.noise)
            elif self.v2:
                return grad_output.mul(1 - self.p)
        return grad_output


class FeatureDropout(Dropout):

    def _make_noise(self, input):
        return input.new().resize_(input.size(0), input.size(1),
                                   *repeat(1, input.dim() - 2))
