from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import torch
import torch.nn as nn
import typing

class Clamp(nn.Module):
    r"""
    Provide as much information upfront as possible to enable decoupling op
    creation from op execution for maximum performance.
    """

    __annotations__  = {
        'context': typing.Optional[torch.Tensor]
    }

    def __init__(self,
                 output_min, output_max,
                 input_channels=0, output_channels=0):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.output_min = output_min
        self.output_max = output_max
        self.construct()

    def construct(self):
        if ((self.input_channels > 0) and
            (self.output_channels > 0)):
            self.context = torch.ops.mobile.clamp_create(
                self.input_channels,
                self.output_channels,
                self.output_min,
                self.output_max)
        else:
            self.context = None

    @torch.jit.export
    def __getstate__(self):
        return (
            self.input_channels,
            self.output_channels,
            self.output_min,
            self.output_max,
            self.training,
        )

    @torch.jit.export
    def __setstate__(self, state):
        self.input_channels = state[0]
        self.output_channels = state[1]
        self.output_min = state[2]
        self.output_max = state[3]
        self.training = state[4]
        self.construct()

    def forward(self, input, output=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor

        if (self.context is not None):
            assert input.size(1) == self.input_channels, "Invalid input!"

            return torch.ops.mobile.clamp_run(
                self.context,
                input)

        else:
            return torch.ops.mobile.clamp(
                input,
                self.output_min,
                self.output_max)
