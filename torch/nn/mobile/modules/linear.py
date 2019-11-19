from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn

class Linear(nn.Module):
    r"""
    Provide as much information upfront as possible to enable decoupling op
    creation from op execution for maximum performance.
    """

    def __init__(self,
                 linear,
                 output_min=-math.inf, output_max=+math.inf):
        super().__init__()

        self.weight = linear.weight
        self.bias = linear.bias if linear.bias is not None else None
        self.output_min = output_min
        self.output_max = output_max
        self.construct()

    def construct(self):
        self.context = torch.ops.mobile.linear_create(
            self.weight,
            self.bias,
            self.output_min,
            self.output_max)

    @torch.jit.export
    def __getstate__(self):
        return (
            self.weight,
            self.bias,
            self.output_min,
            self.output_max,
            self.training,
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.weight = state[0]
        self.bias = state[1]
        self.output_min = state[2]
        self.output_max = state[3]
        self.training = state[4]
        self.construct()

    def forward(self, input):
        return torch.ops.mobile.linear_run(
            self.context,
            input)

class LinearReLU(nn.Module):
    def __init__(self, linear):
        super().__init__()

        if type(relu) is nn.ReLU:
            self.linear = Linear(linear, 0)

        elif type(relu) is nn.ReLU6:
            self.linear = Linear(linear, 0, 6)

        else:
            assert False, "Unknown ReLU type {}!".format(type(relu))

    def forward(self, input):
        return self.linear(input)
