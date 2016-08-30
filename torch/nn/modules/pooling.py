import torch
from torch.autograd import Variable

from .module import Module

class MaxPool2d(Module):

    def __init__(self, kh, kw, dh=None, dw=None, padh=0, padw=0, dilh=1, dilw=1, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kw = kw
        self.kh = kh
        self.dw = dw or kw
        self.dh = dh or kh
        self.padw = padw
        self.padh = padh
        self.dilh = dilh
        self.dilw = dilw
        self.ceil_mode = ceil_mode

    def __call__(self, input):
        return self._backend.MaxPool2d(self.kw, self.kh, self.dw, self.dh, self.padw, self.padh, self.dilh, self.dilw, self.ceil_mode)(input)[0]

class AvgPool2d(Module):

    def __init__(self, kh, kw, dh=None, dw=None, padh=0, padw=0, ceil_mode=False, count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.kw = kw
        self.kh = kh
        self.dw = dw or kw
        self.dh = dh or kh
        self.padw = padw
        self.padh = padh
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def __call__(self, input):
        return self._backend.AvgPool2d(self.kw, self.kh, self.dw, self.dh, self.padw, self.padh, self.ceil_mode, self.count_include_pad)(input)[0]
