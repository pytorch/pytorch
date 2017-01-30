import math
import torch
from .Module import Module


class TemporalConvolution(Module):

    def __init__(self, inputFrameSize, outputFrameSize, kW, dW=1):
        super(TemporalConvolution, self).__init__()

        self.inputFrameSize = inputFrameSize
        self.outputFrameSize = outputFrameSize
        self.kW = kW
        self.dW = dW

        self.weight = torch.Tensor(outputFrameSize, inputFrameSize * kW)
        self.bias = torch.Tensor(outputFrameSize)
        self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize * kW)
        self.gradBias = torch.Tensor(outputFrameSize)

        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.kW * self.inputFrameSize)

        self.weight.uniform_(-stdv, stdv)
        self.bias.uniform_(-stdv, stdv)

    def updateOutput(self, input):
        self._backend.TemporalConvolution_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.kW,
            self.dW,
            self.inputFrameSize,
            self.outputFrameSize
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return
        self._backend.TemporalConvolution_updateGradInput(
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
        self._backend.TemporalConvolution_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.kW,
            self.dW,
            scale
        )
