from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn
import typing

from torch.nn.modules.utils import _pair

class MaxPool2d(nn.Module):
    r"""
    Provide as much information upfront as possible to enable decoupling op
    creation from op execution for maximum performance.
    """

    __annotations__  = {
        'context': typing.Optional[torch.Tensor]
    }

    def __init__(self, max_pool,
                 output_min=-math.inf, output_max=+math.inf,
                 input_channels=0, output_channels=0):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel = _pair(max_pool.kernel_size)
        self.padding = _pair(max_pool.padding)
        self.stride = _pair(max_pool.stride)
        self.dilation = _pair(max_pool.dilation)
        self.ceil_mode = max_pool.ceil_mode
        self.output_min = output_min
        self.output_max = output_max
        self.construct()

    def construct(self):
        if ((self.input_channels > 0) and
            (self.output_channels > 0)):
            self.context = torch.ops.mobile.max_pool2d_create(
                self.input_channels,
                self.output_channels,
                self.kernel,
                self.padding,
                self.stride,
                self.dilation,
                self.ceil_mode,
                self.output_min,
                self.output_max)
        else:
            self.context = None

    @torch.jit.export
    def __getstate__(self):
        return (
            self.input_channels,
            self.output_channels,
            self.kernel,
            self.padding,
            self.stride,
            self.dilation,
            self.ceil_mode,
            self.output_min,
            self.output_max,
            self.training,
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.input_channels = state[0]
        self.output_channels = state[1]
        self.kernel = state[2]
        self.padding = state[3]
        self.stride = state[4]
        self.dilation = state[5]
        self.ceil_mode = state[6]
        self.output_min = state[7]
        self.output_max = state[8]
        self.training = state[9]
        self.construct()

    def forward(self, input):
        if (self.context is not None):
            assert input1.size(1) == self.input_channels, "Invalid input!"

            return torch.ops.mobile.max_pool2d_run(
                self.context,
                input)

        return torch.ops.mobile.max_pool2d(
            input,
            self.kernel,
            self.padding,
            self.stride,
            self.dilation,
            self.ceil_mode,
            self.output_min,
            self.output_max)


class MaxPool2dReLU(nn.Module):
    def __init__(self, max_pool2d,
                 input_channels=None, output_channels=None):
        super().__init__()

        if type(relu) is nn.ReLU:
            self.max_pool = MaxPool2d(0, math.inf, max_pool2d, input_channels, output_channels)

        elif type(relu) is nn.ReLU6:
            self.max_pool = MaxPool2d(0, 6, max_pool2d, input_channels, output_channels)

        else:
            assert False, "Unknown ReLU type {}!".format(type(relu))

    def forward(self, input):
        return self.max_pool(input)
