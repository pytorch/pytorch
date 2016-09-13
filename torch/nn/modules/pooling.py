import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair

class MaxPool2d(Module):

    def __init__(self, ksize, stride=None, pad=0, dil=1, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kh, self.kw = _pair(ksize)
        self.dh, self.dw = _pair(stride or ksize)
        self.padh, self.padw = _pair(pad)
        self.dilh, self.dilw = _pair(dil)
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return self._backend.MaxPool2d(self.kw, self.kh, self.dw, self.dh, self.padw, self.padh, self.ceil_mode)(input)

class AvgPool2d(Module):

    def __init__(self, ksize, stride=None, pad=0, ceil_mode=False, count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.kh, self.kw = _pair(ksize)
        self.dh, self.dw = _pair(stride or ksize)
        self.padh, self.padw = _pair(pad)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return self._backend.AvgPool2d(self.kw, self.kh, self.dw, self.dh, self.padw, self.padh, self.ceil_mode, self.count_include_pad)(input)
