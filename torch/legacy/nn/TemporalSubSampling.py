import math
import torch
from .Module import Module


class TemporalSubSampling(Module):

    def __init__(self, inputFrameSize, kW, dW=1):
        super(TemporalSubSampling, self).__init__()

        self.inputFrameSize = inputFrameSize
        self.kW = kW
        self.dW = dW

        self.weight = torch.Tensor(inputFrameSize)
        self.bias = torch.Tensor(inputFrameSize)
        self.gradWeight = torch.Tensor(inputFrameSize)
        self.gradBias = torch.Tensor(inputFrameSize)

        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.kW)

        self.weight.uniform_(-stdv, stdv)
        self.bias.uniform_(-stdv, stdv)

    def updateOutput(self, input):
        self._backend.TemporalSubSampling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.kW,
            self.dW,
            self.inputFrameSize
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return
        self._backend.TemporalSubSampling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.kW,
            self.dW
        )
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self._backend.TemporalSubSampling_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.kW,
            self.dW,
            scale
        )
