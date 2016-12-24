import numbers
import torch
from . import functions
from .modules import utils

# Convolution

def conv1d(input, weight, bias=None, stride=1, padding=0, groups=1):
    state = functions.conv.Conv1d(stride, padding, groups)
    return bias and state(input, weight, bias) or state(input, weight)

def conv2d(input, weight, bias=None, stride=1, padding=0, groups=1):
    state = functions.conv.Conv2d(stride, padding, groups)
    return bias and state(input, weight, bias) or state(input, weight)

def conv3d(input, weight, bias=None, stride=1, padding=0, groups=1):
    state = functions.conv.Conv3d(stride, padding, groups)
    return bias and state(input, weight, bias) or state(input, weight)

def conv2d_transpose(input, weight, bias=None, stride=1, padding=0, groups=1, out_pad=0):
    state = functions.conv.ConvTranspose2d(stride, padding, groups, out_pad)
    return bias and state(input, weight, bias) or state(input, weight)


# Pooling

def avg_pool2d(input, *args, **kwargs):
    return torch.nn.AvgPool2d(*args, **kwargs)(input)

def avg_pool3d(input, *args, **kwargs):
    return torch.nn.AvgPool3d(*args, **kwargs)(input)


# share the same interface
def max_pool1d(input, kernel_size, stride=1, padding=0, dilation=1,
        ceil_mode=False, return_indices=False):
    return functions.thnn.MaxPool1d(kernel_size, stride, padding, dilation,
            return_indices, ceil_mode)(input)

def max_pool2d(input, kernel_size, stride=1, padding=0, dilation=1,
        ceil_mode=False, return_indices=False):
    return functions.thnn.MaxPool2d(kernel_size, stride, padding, dilation,
            return_indices, ceil_mode)(input)

def max_pool3d(input, kernel_size, stride=1, padding=0, dilation=1,
        ceil_mode=False, return_indices=False):
    return functions.thnn.MaxPool3d(kernel_size, stride, padding, dilation,
            return_indices, ceil_mode)(input)


# Activation functions

def dropout(x, *args, **kwargs):
    return functions.dropout.Dropout(*args, **kwargs)(x)

def relu(input):
    return torch.nn.ReLU()(input)

def elu(input):
    return functions.thnn.auto.ELU()(input)

def sigmoid(input):
    return functions.thnn.Sigmoid()(input)

def tanh(input):
    return torch.tanh(input)

def softsign(input):
    return functions.activation.Softsign()(input)

def softmin(input):
    return functions.thnn.Softmin()(input)

def softmax(input):
    return functions.thnn.auto.Softmax()(input)

def logsoftmax(input):
    return functions.thnn.LogSoftmax()(input)

# etc.

def linear(input, weight, bias=None):
    return functions.linear.Linear()(input, weight, bias)

def batch_norm(input, weight, bias, *args, **kwargs):
    return functions.thnn.BatchNorm(*args, **kwargs)(input, weight, bias)

def cross_entropy(x, y):
    return torch.nn.CrossEntropyLoss()(x, y)

