import math
import torch
from torch.legacy import nn

class SpatialFullConvolution(nn.Module):

    def __init__(self, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=None, adjW=0, adjH=0):
        super(SpatialFullConvolution, self).__init__()

        self.nInputPlane = nInputPlane
        self.nOutputPlane = nOutputPlane
        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH or padW or 0
        self.adjW = adjW
        self.adjH = adjH

        if self.adjW > self.dW - 1 or self.adjH > self.dH - 1:
            raise ValueError('adjW and adjH must be smaller than self.dW - 1 and self.dH - 1 respectively')

        self.weight = torch.Tensor(nInputPlane, nOutputPlane, kH, kW)
        self.gradWeight = torch.Tensor(nInputPlane, nOutputPlane, kH, kW)
        self.bias = torch.Tensor(self.nOutputPlane)
        self.gradBias = torch.Tensor(self.nOutputPlane)

        self.ones = torch.Tensor()
        self.finput = None
        self.fgradInput = None
        self.zeroScalar = None

        self.reset()

    def noBias(self):
        self.bias = None
        self.gradBias = None
        return self

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            nInputPlane = self.nInputPlane
            kH = self.kH
            kW = self.kW
            stdv = 1/math.sqrt(kW*kH*nInputPlane)

        self.weight.uniform_(-stdv, stdv)
        if self.bias:
            self.bias.uniform_(-stdv, stdv)

    def _makeContiguous(self, input, gradOutput=None):
        if not input.isContiguous():
           self._input = self._input or input.new()
           self._input.resizeAs_(input).copy(input)
           input = self._input

        if gradOutput is not None:
            if not gradOutput.isContiguous():
                self._gradOutput = self._gradOutput or gradOutput.new()
                self._gradOutput.resizeAs_(gradOutput).copy(gradOutput)
                gradOutput = self._gradOutput
            return input, gradOutput

        return input

    def _calculateAdj(self, targetSize, ker, pad, stride):
        return (targetSize + 2 * pad - ker) % stride

    def updateOutput(self, input):
        inputTensor = input
        adjW, adjH = self.adjW, self.adjH

        # The input can be a table where the second element indicates the target
        # output size, in which case the adj factors are computed automatically
        if isinstance(input, list):
            inputTensor = input[0]
            targetTensor = input[1]
            tDims = targetTensor.dim()
            tH = targetTensor.size(tDims-2)
            tW = targetTensor.size(tDims-1)
            adjW = self._calculateAdj(tW, self.kW, self.padW, self.dW)
            adjH = self._calculateAdj(tH, self.kH, self.padH, self.dH)
            self.finput = self.finput or input[0].new()
            self.fgradInput = self.fgradInput or input[0].new()
        else:
            self.finput = self.finput or input.new()
            self.fgradInput = self.fgradInput or input.new()


        inputTensor = self._makeContiguous(inputTensor)
        self._backend.SpatialFullConvolution_updateOutput(
            self._backend.library_state,
            inputTensor,
            self.output,
            self.weight,
            self.bias,
            self.finput,
            self.fgradInput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            adjW, adjH
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
            return
        inputTensor = input
        adjW, adjH = self.adjW, self.adjH

        # The input can be a table where the second element indicates the target
        # output size, in which case the adj factors are computed automatically
        if isinstance(input, list):
            inputTensor = input[0]
            targetTensor = input[1]
            tDims = targetTensor.dim()
            tH = targetTensor.size(tDims-2)
            tW = targetTensor.size(tDims-1)
            adjW = self._calculateAdj(tW, self.kW, self.padW, self.dW)
            adjH = self._calculateAdj(tH, self.kH, self.padH, self.dH)
        # Momentarily extract the gradInput tensor
        if isinstance(self.gradInput, list):
            self.gradInput = self.gradInput[0]

        inputTensor, gradOutput = self._makeContiguous(inputTensor, gradOutput)
        self._backend.SpatialFullConvolution_updateGradInput(
            self._backend.library_state,
            inputTensor,
            gradOutput,
            self.gradInput,
            self.weight,
            self.finput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            adjW, adjH
        )

        if isinstance(input, list):
            # Create a zero tensor to be expanded and used as gradInput[1].
            self.zeroScalar = self.zeroScalar or input[1].new(1).zero_()
            self.ones.resize_(input[1].dim()).fill_(1)
            zeroTensor =  self.zeroScalar.viewAs(self.ones).expandAs(input[1])
            self.gradInput = [self.gradInput, zeroTensor]

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        inputTensor = input
        adjW, adjH = self.adjW, self.adjH

        # The input can be a table where the second element indicates the target
        # output size, in which case the adj factors are computed automatically
        if isinstance(inputTensor, list):
            inputTensor = input[0]
            targetTensor = input[1]
            tDims = targetTensor.dim()
            tH = targetTensor.size(tDims-2)
            tW = targetTensor.size(tDims-1)
            adjW = calculateAdj(tW, self.kW, self.padW, self.dW)
            adjH = calculateAdj(tH, self.kH, self.padH, self.dH)

        inputTensor, gradOutput = self._makeContiguous(inputTensor, gradOutput)
        self._backend.SpatialFullConvolution_accGradParameters(
            self._backend.library_state,
            inputTensor,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.finput,
            self.fgradInput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            adjW, adjH,
            scale
        )

    def type(self, type=None, tensorCache=None):
        self.finput = self.finput and torch.Tensor()
        self.fgradInput = self.fgradInput and torch.Tensor()
        return super(SpatialFullConvolution, self).type(type, tensorCache)

    def __repr__(self):
        s = super(SpatialFullConvolution, self).__repr__()
        s += '({} -> {}, {}x{}'.format(self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
        if self.dW != 1 or self.dH != 1 or self.padW != 0 or self.padH != 0:
            s += ', {}, {}'.format(self.dW, self.dH)

        if (self.padW or self.padH) and (self.padW != 0 or self.padH != 0):
            s += ', {}, {}'.format(self.padW, self.padH)

        if (self.adjW or self.adjH) and (self.adjW != 0 or self.adjH != 0):
            s += ', {}, {}'.format(self.adjW, self.adjH)

        s += ')'
        if self.bias is None:
            s += ' without bias'
        return s

    def clearState(self):
        nn.utils.clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
        return super(SpatialFullConvolution, self).clearState()


