import random
import math
import torch
from .Module import Module


class SpatialFullConvolutionMap(Module):

    def __init__(self, conMatrix, kW, kH, dW=1, dH=1):
        super(SpatialFullConvolutionMap, self).__init__()

        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH
        self.connTable = conMatrix
        self.nInputPlane = int(self.connTable.select(1, 0).max()) + 1
        self.nOutputPlane = int(self.connTable.select(1, 1).max()) + 1

        self.weight = torch.Tensor(self.connTable.size(0), kH, kW)
        self.gradWeight = torch.Tensor(self.connTable.size(0), kH, kW)

        self.bias = torch.Tensor(self.nOutputPlane)
        self.gradBias = torch.Tensor(self.nOutputPlane)

        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
            self.weight.uniform_(-stdv, stdv)
            self.bias.uniform_(-stdv, stdv)
        else:
            ninp = torch.Tensor(self.nOutputPlane).zero_()
            for i in range(self.connTable.size(0)):
                idx = int(self.connTable[i][1])
                ninp[idx] += 1
            for k in range(self.connTable.size(0)):
                idx = int(self.connTable[k][1])
                stdv = 1. / math.sqrt(self.kW * self.kH * ninp[idx])
                self.weight[k].uniform_(-stdv, stdv)
            for k in range(self.bias.size(0)):
                stdv = 1. / math.sqrt(self.kW * self.kH * ninp[k])
                # TODO: torch.uniform
                self.bias[k] = random.uniform(-stdv, stdv)

    def updateOutput(self, input):
        self._backend.SpatialFullConvolutionMap_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.connTable,
            self.nInputPlane,
            self.nOutputPlane,
            self.dW, self.dH
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.SpatialFullConvolutionMap_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.bias,
            self.connTable,
            self.nInputPlane,
            self.nOutputPlane,
            self.dW, self.dH
        )
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self._backend.SpatialFullConvolutionMap_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.connTable,
            self.nInputPlane,
            self.nOutputPlane,
            self.dW, self.dH,
            scale
        )
