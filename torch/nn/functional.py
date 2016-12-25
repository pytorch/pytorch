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

def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    kw, kh = utils._pair(kernel_size)
    out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    return out.mul(kw * kh).pow(1./norm_type)


# Activation functions

def dropout(input, p=0.5, training=False, inplace=False):
    return functions.dropout.Dropout(p, training, inplace)(input)

def threshold(input, threshold, value, inplace=False):
    return functions.thnn.auto.Threshold(threshold, value, inplace)(input)

def relu(input, inplace=False):
    return functions.thnn.auto.Threshold(0, 0, inplace)(input)

def hardtanh(input, min_val=-1., max_val=1., inplace=False):
    return functions.thnn.auto.Hardtanh(min_val, max_val, inplace)(input)

def relu6(input, inplace=False):
    return functions.thnn.auto.Hardtanh(0, 6, inplace)(input)

def elu(input, alpha=1., inplace=False):
    return functions.thnn.auto.ELU(alpha, inplace)(input)

def leaky_relu(input, negative_slope=1e-2, inplace=False):
    return functions.thnn.auto.LeakyReLU(negative_slope, inplace)(input)

def prelu(input, weight):
    return functions.thnn.PReLU()(input, weight)

def rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False):
    return functions.thnn.RReLU(lower, upper, training, inplace)(input)

def sigmoid(input):
    return functions.thnn.Sigmoid()(input)

def logsigmoid(input):
    return functions.thnn.LogSigmoid()(input)

def tanh(input):
    return torch.tanh(input)

def hardshrink(input, lambd=0.5):
    return functions.thnn.auto.Hardshrink(lambd)(input)

def tanhshrink(input):
    return input - torch.tanh(input)

def softsign(input):
    return functions.activation.Softsign()(input)

def softplus(input, beta=1, threshold=20):
    return functions.thnn.auto.Softplus(beta, threshold)(input)

def softmin(input):
    return functions.thnn.Softmin()(input)

def softmax(input):
    return functions.thnn.auto.Softmax()(input)

def softshrink(input, lambd=0.5):
    return functions.thnn.auto.Softshrink(lambd)(input)

def log_softmax(input):
    return functions.thnn.LogSoftmax()(input)


# etc.

def linear(input, weight, bias=None):
    state = functions.linear.Linear()
    return bias and state(input, weight, bias) or state(input, weight)

def batch_norm(input, running_mean, running_var, weight=None, bias=None,
        training=False, momentum=0.1, eps=1e-5):
    state = functions.thnn.BatchNorm(running_mean, running_var,
            training, momentum, eps)
    return weight and state(input, weight, bias) or state(input)

def nll_loss(input, target, weight=None, size_average=True):
    return functions.thnn.NLLLoss(size_average, weight=weight)(input, target)

def sparse_cross_entropy_with_logits(input, target, weight=None, size_average=True):
    return nll_loss(log_softmax(input), target, weight, size_average)

def sigmoid_cross_entropy_with_logits(input, target, weight=None, size_average=True):
    return functions.thnn.BCELoss(size_average, weight=weight)(input, target)
