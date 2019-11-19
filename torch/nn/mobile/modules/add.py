from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn
import typing

class Add(nn.Module):
    r"""
    Provide as much information upfront as possible to enable decoupling op
    creation from op execution for maximum performance.
    """

    __annotations__  = {
        'context': typing.Optional[torch.Tensor]
    }

    def __init__(self,
                 output_min=-math.inf, output_max=+math.inf,
                 input1_channels=0, input2_channels=0, output_channels=0):
        super().__init__()

        self.input1_channels = input1_channels
        self.input2_channels = input2_channels
        self.output_channels = output_channels
        self.output_min = output_min
        self.output_max = output_max
        self.construct()

    def construct(self):
        if ((self.input1_channels > 0) and
            (self.input2_channels > 0) and
            (self.output_channels > 0)):
            self.context = torch.ops.mobile.add_create(
                self.input1_channels,
                self.input2_channels,
                self.output_channels,
                self.output_min,
                self.output_max)
        else:
            self.context = None

    @torch.jit.export
    def __getstate__(self):
        return (
            self.input1_channels,
            self.input2_channels,
            self.output_channels,
            self.output_min,
            self.output_max,
            self.training,
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.input1_channels = state[0]
        self.input2_channels = state[1]
        self.output_channels = state[2]
        self.output_min = state[3]
        self.output_max = state[4]
        self.training = state[5]
        self.construct()

    def forward(self, input1, input2, output=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor

        if self.context is not None:
            assert input1.size(1) == self.input1_channels, "Invalid input1!"
            assert input2.size(1) == self.input2_channels, "Invalid input2!"

            return torch.ops.mobile.add_run(
                self.context,
                output,
                input1,
                input2)
        else:
            return torch.ops.mobile.add(
                output,
                input1,
                input2,
                self.output_min,
                self.output_max)


class AddReLU(nn.Module):
    def __init__(self, relu,
                 input1_channels=0, input2_channels=0, output_channels=0):
        super().__init__()

        if type(relu) is nn.ReLU:
            self.add = Add(0, math.inf, input1_channels, input2_channels, output_channels)

        elif type(relu) is nn.ReLU6:
            self.add = Add(0, 6, input1_channels, input2_channels, output_channels)

        else:
            assert False, "Unknown ReLU type {}!".format(type(relu))

    def forward(self, input1, input2, output=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor

        return self.add(input1, input2, output)
