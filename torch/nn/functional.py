import numbers
import torch
from . import functions
from .modules import utils

def conv2d(input, weight, bias=None, stride=1, pad=0, groups=1):
    state = functions.conv.Conv2d(stride, pad, groups)
    return bias and state(input, weight, bias) or state(input, weight)

def conv2d_transpose(input, weight, bias=None, *args, **kwargs):
    state = functions.conv.ConvTranspose2d(*args, **kwargs)
    return bias and state(input, weight, bias) or state(input, weight)


def avg_pool2d(input, *args, **kwargs):
    return torch.nn.AvgPool2d(*args, **kwargs)(input)

def avg_pool3d(input, *args, **kwargs):
    return torch.nn.AvgPool3d(*args, **kwargs)(input)


def max_pool1d(input, *args, **kwargs):
    return functions.thnn.MaxPool1d(*args, **kwargs)(input)

def max_pool2d(input, kernel_size, stride=1, padding=0, dilation=1,
        return_indices=False, ceil_mode=False):
    kernel_size = utils._pair(kernel_size)
    stride = utils._pair(stride)
    padding = utils._pair(padding)
    dilation = utils._pair(dilation)
    return functions.thnn.MaxPool2d(
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            return_indices, ceil_mode)(input)

def max_pool3d(input, *args, **kwargs):
    return functions.thnn.MaxPool3d(*args, **kwargs)(input)


def linear(input, weight, bias=None):
    return functions.linear.Linear()(input, weight, bias)

def batch_norm(input, weight, bias, *args, **kwargs):
    return functions.thnn.BatchNorm(*args, **kwargs)(input, weight, bias)

def softmax(input):
    return functions.thnn.auto.Softmax()(input)

def logsoftmax(input):
    return functions.thnn.LogSoftmax()(input)

def cross_entropy(x, y):
    return torch.nn.CrossEntropyLoss()(x, y)

def dropout(x, *args, **kwargs):
    return functions.dropout.Dropout(*args, **kwargs)(x)

def relu(input):
    return torch.nn.ReLU()(input)
