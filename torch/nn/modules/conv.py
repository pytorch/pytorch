import math
import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class Conv1d(Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1):
        super(Conv1d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride

        kernel_elements = self.in_features * self.kernel_size
        self.weight = Variable(torch.DoubleTensor(out_features,
                kernel_elements))
        self.bias = Variable(torch.DoubleTensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features * self.kernel_size)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        func = self._backend.Conv1d(self.kernel_size, self.stride,
                self.in_features, self.out_features)
        return func(input, self.weight, self.bias)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, dilation=None, no_bias=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = _pair(kernel_size)
        self.dh, self.dw = _pair(stride)
        self.padh, self.padw = _pair(padding)
        self.is_dilated = dilation is not None
        if self.is_dilated:
            self.dilh, self.dilw = _pair(dilation)

        self.weight = Variable(torch.DoubleTensor(self.out_channels,
                self.in_channels, self.kh, self.kw))
        if no_bias:
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
        if self.is_dilated:
            func = self._backend.DilatedConv2d(self.kw, self.kh, self.dw,
                    self.dh, self.padw, self.padh, self.dilh, self.dilw)
        else:
            func = self._backend.Conv2d(self.kw, self.kh, self.dw, self.dh,
                    self.padw, self.padh)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)


class FullConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, output_padding=0, no_bias=False):
        super(FullConv2d, self).__init__(in_channels, out_channels, kernel_size,
                stride, padding, no_bias)
        self.out_padh, self.out_padw = _pair(output_padding)

    def forward(self, input):
        func = self._backend.FullConv2d(self.kw, self.kh, self.dw, self.dh,
                self.padw, self.padh, self.out_padh, self.out_padw)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)


class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, no_bias=False):
        super(Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride)
        self.padt, self.padh, self.padw = _triple(padding)

        self.weight = Variable(torch.DoubleTensor(self.out_channels,
                self.in_channels, self.kt, self.kh, self.kw))
        if no_bias:
            self.bias = None
        else:
            self.bias = Variable(torch.DoubleTensor(self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kt * self.kh * self.kw * self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        func = self._backend.Conv3d(self.kt, self.kw, self.kh, self.dt,
                self.dw, self.dh, self.padt, self.padw, self.padh)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)


class FullConv3d(Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0, no_bias=False):
        super(Conv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride)
        self.padt, self.padh, self.padw = _triple(padding)

        self.weight = Variable(torch.DoubleTensor(self.in_channels,
                self.out_channels, self.kt, self.kh, self.kw))
        if no_bias:
            self.bias = None
        else:
            self.bias = Variable(torch.DoubleTensor(self.out_channels))

        self.reset_parameters()

    def forward(self, input):
        func = self._backend.FullConv3d(self.kt, self.kw, self.kh,
                self.dt, self.dw, self.dh, self.padt, self.padw, self.padh)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)


# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: FullConv2dMap
