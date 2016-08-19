import math
import torch
from torch.autograd import Variable

from .module import Module

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kh, kw, dh=1, dw=1, padh=0, padw=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = kh
        self.kw = kw
        self.dh = dh
        self.dw = dw
        self.padh = padh
        self.padw = padw

        self.weight = Variable(torch.DoubleTensor(self.out_channels, self.in_channels, self.kh, self.kw))
        self.bias = Variable(torch.DoubleTensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kh * self.kw * self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def __call__(self, input):
        return self._backend.Conv2d(self.kw, self.kh, self.dw, self.dh, self.padw, self.padh)(input, self.weight, self.bias)[0]

