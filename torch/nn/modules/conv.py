import math
import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = _pair(ksize)
        self.dh, self.dw = _pair(stride)
        self.padh, self.padw = _pair(pad)

        self.weight = Variable(torch.DoubleTensor(self.out_channels, self.in_channels, self.kh, self.kw))
        if nobias:
            self.bias = None
        else:
            self.bias = Variable(torch.DoubleTensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kh * self.kw * self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        conv2d = self._backend.Conv2d(self.kw, self.kh, self.dw, self.dh, self.padw, self.padh)
        if self.bias is None:
            return conv2d(input, self.weight)
        else:
            return conv2d(input, self.weight, self.bias)
