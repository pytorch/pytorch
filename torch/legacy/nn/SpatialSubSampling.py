import math
import torch
from .Module import Module


class SpatialSubSampling(Module):

    def __init__(self, nInputPlane, kW, kH, dW=1, dH=1):
        super(SpatialSubSampling, self).__init__()

        self.nInputPlane = nInputPlane
        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH

        self.weight = torch.Tensor(nInputPlane)
        self.bias = torch.Tensor(nInputPlane)
        self.gradWeight = torch.Tensor(nInputPlane)
        self.gradBias = torch.Tensor(nInputPlane)

        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.kW * self.kH)

        self.weight.uniform_(-stdv, stdv)
        self.bias.uniform_(-stdv, stdv)

    def updateOutput(self, input):
        self._backend.SpatialSubSampling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.kW, self.kH,
            self.dW, self.dH
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        self._backend.SpatialSubSampling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.kW, self.kH,
            self.dW, self.dH
        )
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self._backend.SpatialSubSampling_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.kW, self.kH,
            self.dW, self.dH,
            scale
        )
