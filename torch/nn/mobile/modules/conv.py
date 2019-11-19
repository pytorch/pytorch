from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn

class Conv2d(nn.Module):
    r"""
    Provide as much information upfront as possible to enable decoupling op
    creation from op execution for maximum performance.
    """

    def __init__(self,
                 conv,
                 output_min=-math.inf, output_max=+math.inf):
        super().__init__()

        self.weight = conv.weight
        self.bias = conv.bias if conv.bias is not None else None
        self.padding = conv.padding
        self.stride = conv.stride
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.transposed = conv.transposed
        self.output_min = output_min
        self.output_max = output_max
        self.construct()

    def construct(self):
        self.context = torch.ops.mobile.conv2d_create(
            self.weight,
            self.bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.transposed,
            self.output_min,
            self.output_max)

    @torch.jit.export
    def __getstate__(self):
        return (
            self.weight,
            self.bias,
            self.padding,
            self.stride,
            self.dilation,
            self.groups,
            self.transposed,
            self.output_min,
            self.output_max,
            self.training,
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.weight = state[0]
        self.bias = state[1]
        self.padding = state[2]
        self.stride = state[3]
        self.dilation = state[4]
        self.groups = state[5]
        self.transposed = state[6]
        self.output_min = state[7]
        self.output_max = state[8]
        self.training = state[9]
        self.construct()

    def forward(self, input):
        return torch.ops.mobile.conv2d_run(
            self.context,
            input)


class Conv2dReLU(nn.Module):
    def __init__(self, convrelu):
        super().__init__()

        conv = convrelu[0]
        relu = convrelu[1]

        if type(relu) is nn.ReLU:
            self.conv = Conv2d(conv, 0)

        elif type(relu) is nn.ReLU6:
            self.conv = Conv2d(conv, 0, 6)

        else:
            assert False, "Unknown ReLU type {}!".format(type(relu))

    def forward(self, input):
        return self.conv(input)

