"""
Creating extensions using numpy and scipy
=========================================

In this tutorial, we shall go through two tasks:

1. Create a neural network layer with no parameters.

    -  This calls into **numpy** as part of it’s implementation

2. Create a neural network layer that has learnable weights

    -  This calls into **SciPy** as part of it’s implementation
"""

import torch
from torch.autograd import Function
from torch.autograd import Variable

###############################################################
# Parameter-less example
# ----------------------
#
# This layer doesn’t particularly do anything useful or mathematically
# correct.
#
# It is aptly named BadFFTFunction
#
# **Layer Implementation**

from numpy.fft import rfft2, irfft2


class BadFFTFunction(Function):

    def forward(self, input):
        numpy_input = input.numpy()
        result = abs(rfft2(numpy_input))
        return torch.FloatTensor(result)

    def backward(self, grad_output):
        numpy_go = grad_output.numpy()
        result = irfft2(numpy_go)
        return torch.FloatTensor(result)

# since this layer does not have any parameters, we can
# simply declare this as a function, rather than as an nn.Module class


def incorrect_fft(input):
    return BadFFTFunction()(input)

###############################################################
# **Example usage of the created layer:**

input = Variable(torch.randn(8, 8), requires_grad=True)
result = incorrect_fft(input)
print(result.data)
result.backward(torch.randn(result.size()))
print(input.grad)

###############################################################
# Parametrized example
# --------------------
#
# This implements a layer with learnable weights.
#
# It implements the Cross-correlation with a learnable kernel.
#
# In deep learning literature, it’s confusingly referred to as
# Convolution.
#
# The backward computes the gradients wrt the input and gradients wrt the
# filter.
#
# **Implementation:**
#
# *Please Note that the implementation serves as an illustration, and we
# did not verify it’s correctness*

from scipy.signal import convolve2d, correlate2d
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class ScipyConv2dFunction(Function):

    def forward(self, input, filter):
        result = correlate2d(input.numpy(), filter.numpy(), mode='valid')
        self.save_for_backward(input, filter)
        return torch.FloatTensor(result)

    def backward(self, grad_output):
        input, filter = self.saved_tensors
        grad_input = convolve2d(grad_output.numpy(), filter.t().numpy(), mode='full')
        grad_filter = convolve2d(input.numpy(), grad_output.numpy(), mode='valid')
        return torch.FloatTensor(grad_input), torch.FloatTensor(grad_filter)


class ScipyConv2d(Module):

    def __init__(self, kh, kw):
        super(ScipyConv2d, self).__init__()
        self.filter = Parameter(torch.randn(kh, kw))

    def forward(self, input):
        return ScipyConv2dFunction()(input, self.filter)

###############################################################
# **Example usage:**

module = ScipyConv2d(3, 3)
print(list(module.parameters()))
input = Variable(torch.randn(10, 10), requires_grad=True)
output = module(input)
print(output)
output.backward(torch.randn(8, 8))
print(input.grad)
