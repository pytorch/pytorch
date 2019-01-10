import math
import torch
from .Module import Module


class VolumetricFullConvolution(Module):

    def __init__(self, nInputPlane, nOutputPlane,
                 kT, kW, kH,                 # kernel size
                 dT=1, dW=1, dH=1,           # stride
                 padT=0, padW=0, padH=0,     # padding
                 adjT=0, adjW=0, adjH=0):    # extra output adjustment
        super(VolumetricFullConvolution, self).__init__()

        self.nInputPlane = nInputPlane
        self.nOutputPlane = nOutputPlane
        self.kW = kW
        self.kH = kH
        self.kT = kT
        self.dW = dW
        self.dH = dH
        self.dT = dT
        self.padW = padW
        self.padH = padH
        self.padT = padT
        self.adjW = adjW
        self.adjH = adjH
        self.adjT = adjT

        if self.adjW > self.dW - 1 or self.adjH > self.dH - 1 or self.adjT > self.dT - 1:
            raise RuntimeError('adjW, adjH and adjT must be smaller than self.dW - 1, '
                               ' self.dH - 1 and self.dT - 1 respectively')

        self.weight = torch.Tensor(nInputPlane, nOutputPlane, kT, kH, kW)
        self.gradWeight = torch.Tensor(nInputPlane, nOutputPlane, kT, kH, kW)
        self.bias = torch.Tensor(self.nOutputPlane)
        self.gradBias = torch.Tensor(self.nOutputPlane)

        self.ones = torch.Tensor()
        self.finput = torch.Tensor()
        self.fgradInput = torch.Tensor()
        self._gradOutput = None

        self.reset()

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            nInputPlane = self.nInputPlane
            kT = self.kT
            kH = self.kH
            kW = self.kW
            stdv = 1. / math.sqrt(kW * kH * kT * nInputPlane)

        self.weight.uniform_(-stdv, stdv)
        self.bias.uniform_(-stdv, stdv)

    def _makeContiguous(self, input, gradOutput=None):
        if not input.is_contiguous():
            if self._input is None:
                self._input = input.new()
            self._input.resize_as_(input).copy_(input)
            input = self._input

        if gradOutput is not None:
            if not gradOutput.is_contiguous():
                if self._gradOutput is None:
                    self._gradOutput = gradOutput.new()
                self._gradOutput.resize_as_(gradOutput).copy_(gradOutput)
                gradOutput = self._gradOutput
            return input, gradOutput

        return input

    def _calculateAdj(targetSize, ker, pad, stride):
        return (targetSize + 2 * pad - ker) % stride

    def updateOutput(self, input):
        inputTensor = input
        adjT, adjW, adjH = self.adjT, self.adjW, self.adjH

        # The input can be a table where the second element indicates the target
        # output size, in which case the adj factors are computed automatically
        if isinstance(input, list):
            inputTensor = input[0]
            targetTensor = input[1]
            tDims = targetTensor.dim()
            tT = targetTensor.size(tDims - 3)
            tH = targetTensor.size(tDims - 2)
            tW = targetTensor.size(tDims - 1)
            adjT = self._calculateAdj(tT, self.kT, self.padT, self.dT)
            adjW = self._calculateAdj(tW, self.kW, self.padW, self.dW)
            adjH = self._calculateAdj(tH, self.kH, self.padH, self.dH)

        inputTensor = self._makeContiguous(inputTensor)
        self._backend.VolumetricFullConvolution_updateOutput(
            self._backend.library_state,
            inputTensor,
            self.output,
            self.weight,
            self.bias,
            self.finput,
            self.fgradInput,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            adjT, adjW, adjH
        )

        return self.output

    def updateGradInput(self, input, gradOutput):
        inputTensor = input
        adjT, adjW, adjH = self.adjT, self.adjW, self.adjH

        # The input can be a table where the second element indicates the target
        # output size, in which case the adj factors are computed automatically
        if isinstance(input, list):
            inputTensor = input[0]
            targetTensor = input[1]
            tDims = targetTensor.dim()
            tT = targetTensor.size(tDims - 3)
            tH = targetTensor.size(tDims - 2)
            tW = targetTensor.size(tDims - 1)
            adjT = self._calculateAdj(tT, self.kT, self.padT, self.dT)
            adjW = self._calculateAdj(tW, self.kW, self.padW, self.dW)
            adjH = self._calculateAdj(tH, self.kH, self.padH, self.dH)
            # Momentarily extract the gradInput tensor
            if isinstance(self.gradInput, list):
                self.gradInput = self.gradInput[0]

        inputTensor, gradOutput = self._makeContiguous(inputTensor, gradOutput)
        self._backend.VolumetricFullConvolution_updateGradInput(
            self._backend.library_state,
            inputTensor,
            gradOutput,
            self.gradInput,
            self.weight,
            self.finput,
            self.fgradInput,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            adjT, adjW, adjH
        )

        if isinstance(input, list):
            # Create a zero tensor to be expanded and used as gradInput[1].
            if self.zeroScalar is None:
                self.zeroScalar = input[1].new(1).zero_()
            self.ones.resize_(input[1].dim()).fill_(1)
            zeroTensor = self.zeroScalar.view(self.ones.tolist()).expand_as(input[1])
            self.gradInput = [self.gradInput, zeroTensor]

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        inputTensor = input
        adjT, adjW, adjH = self.adjT, self.adjW, self.adjH

        # The input can be a table where the second element indicates the target
        # output size, in which case the adj factors are computed automatically
        if isinstance(input, list):
            inputTensor = input[0]
            targetTensor = input[1]
            tDims = targetTensor.dim()
            tT = targetTensor.size(tDims - 3)
            tH = targetTensor.size(tDims - 2)
            tW = targetTensor.size(tDims - 1)
            adjT = self._calculateAdj(tT, self.kT, self.padT, self.dT)
            adjW = self._calculateAdj(tW, self.kW, self.padW, self.dW)
            adjH = self._calculateAdj(tH, self.kH, self.padH, self.dH)

        inputTensor, gradOutput = self._makeContiguous(inputTensor, gradOutput)
        self._backend.VolumetricFullConvolution_accGradParameters(
            self._backend.library_state,
            inputTensor,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.finput,
            self.fgradInput,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            adjT, adjW, adjH,
            scale
        )

    def type(self, type, tensorCache=None):
        self.finput = torch.Tensor()
        self.fgradInput = torch.Tensor()
        return super(VolumetricFullConvolution, self).type(type, tensorCache)

    def __repr__(self):
        s = super(VolumetricFullConvolution, self).__repr__()
        s += '({} -> {}, {}x{}x{}'.format(self.nInputPlane, self.nOutputPlane, self.kT, self.kW, self.kH)
        if self.dT != 1 or self.dW != 1 or self.dH != 1 or \
                self.padT != 0 or self.padW != 0 or self.padH != 0 or \
                self.adjT != 0 or self.adjW != 0 or self.adjH != 0:
            s += ', {}, {}, {}'.format(self.dT, self.dW, self.dH)

        if self.padT != 0 or self.padW != 0 or self.padH != 0 or \
                self.adjT != 0 or self.adjW != 0 or self.adjH != 0:
            s += ', {}, {}, {}'.format(self.padT, self.padW, self.padH)

        if self.adjT != 0 or self.adjW != 0 or self.adjH != 0:
            s += ', {}, {}, {}'.format(self.adjT, self.adjW, self.adjH)

        s += ')'
        return s
