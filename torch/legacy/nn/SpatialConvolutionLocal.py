import math
import torch
from .Module import Module
from .utils import clear


class SpatialConvolutionLocal(Module):

    def __init__(self, nInputPlane, nOutputPlane, iW, iH, kW, kH, dW=1, dH=1, padW=0, padH=None):
        super(SpatialConvolutionLocal, self).__init__()

        self.nInputPlane = nInputPlane
        self.nOutputPlane = nOutputPlane
        self.kW = kW
        self.kH = kH
        self.iW = iW
        self.iH = iH

        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH if padH is not None else padW
        self.oW = int(math.floor((self.padW * 2 + iW - self.kW) / self.dW)) + 1
        self.oH = int(math.floor((self.padH * 2 + iH - self.kH) / self.dH)) + 1
        assert 1 <= self.oW and 1 <= self.oH

        self.weight = torch.Tensor(self.oH, self.oW, nOutputPlane, nInputPlane, kH, kW)
        self.bias = torch.Tensor(nOutputPlane, self.oH, self.oW)
        self.gradWeight = torch.Tensor().resize_as_(self.weight)
        self.gradBias = torch.Tensor().resize_as_(self.bias)

        self.reset()
        self.finput = None
        self.fgradInput = None
        self._gradOutput = None

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.kW * self.kH * self.nInputPlane)

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

    def _viewWeight(self):
        self.weight = self.weight.view(self.oH * self.oW, self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
        if self.gradWeight is not None and self.gradWeight.dim() > 0:
            self.gradWeight = self.gradWeight.view(
                self.oH * self.oW, self.nOutputPlane, self.nInputPlane * self.kH * self.kW)

    def _unviewWeight(self):
        self.weight = self.weight.view(self.oH, self.oW, self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
        if self.gradWeight is not None and self.gradWeight.dim() > 0:
            self.gradWeight = self.gradWeight.view(
                self.oH, self.oW, self.nOutputPlane, self.nInputPlane, self.kH, self.kW)

    def _checkInputSize(self, input):
        if input.ndimension() == 3:
            if input.size(0) != self.nInputPlane or input.size(1) != self.iH or input.size(1) != self.iW:
                raise RuntimeError(
                    'Given input size: ({}x{}x{}) inconsistent with expected input size: ({}x{}x{}).'.format(
                        input.size(0), input.size(1), input.size(2), self.nInputPlane, self.iH, self.iW))
        elif input.ndimension() == 4:
            if input.size(1) != self.nInputPlane or input.size(2) != self.iH or input.size(3) != self.iW:
                raise RuntimeError(
                    'Given input size: ({}x{}x{}x{}) inconsistent with expected input size: (*x{}x{}x{}).'.format(
                        input.size(0), input.size(1), input.size(2), input.size(3), self.nInputPlane, self.iH, self.iW))
        else:
            raise RuntimeError('3D or 4D (batch mode) tensor expected')

    def _checkOutputSize(self, input, output):
        if output.ndimension() != input.ndimension():
            raise RuntimeError('inconsistent dimension between output and input.')

        if output.ndimension() == 3:
            if output.size(0) != self.nOutputPlane or output.size(1) != self.oH or output.size(2) != self.oW:
                raise RuntimeError(
                    'Given output size: ({}x{}x{}) inconsistent with expected output size: ({}x{}x{}).'.format(
                        output.size(0), output.size(1), output.size(2), self.nOutputPlane, self.oH, self.oW))
        elif output.ndimension() == 4:
            if output.size(1) != self.nOutputPlane or output.size(2) != self.oH or output.size(3) != self.oW:
                raise RuntimeError('Given output size: ({}x{}x{}x{}) inconsistent with expected output size: '
                                   '(batchsize x{}x{}x{}).'.format(
                                       output.size(0), output.size(1), output.size(2),
                                       output.size(3), self.nOutputPlane, self.oH, self.oW))
        else:
            raise RuntimeError('3D or 4D(batch mode) tensor expected')

    def updateOutput(self, input):
        if self.finput is None:
            self.finput = input.new()
        if self.fgradInput is None:
            self.fgradInput = input.new()
        self._checkInputSize(input)
        self._viewWeight()
        input = self._makeContiguous(input)
        self._backend.SpatialConvolutionLocal_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.finput,
            self.fgradInput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            self.iW, self.iH,
            self.oW, self.oH
        )
        self._unviewWeight()
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        self._checkInputSize(input)
        self._checkOutputSize(input, gradOutput)

        self._viewWeight()
        input, gradOutput = self._makeContiguous(input, gradOutput)
        self._backend.SpatialConvolutionLocal_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.finput,
            self.fgradInput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            self.iW, self.iH,
            self.oW, self.oH
        )
        self._unviewWeight()
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self._checkInputSize(input)
        self._checkOutputSize(input, gradOutput)
        input, gradOutput = self._makeContiguous(input, gradOutput)
        self._viewWeight()
        self._backend.SpatialConvolutionLocal_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradWeight,
            self.gradBias,
            self.finput,
            self.fgradInput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            self.iW, self.iH,
            self.oW, self.oH,
            scale
        )
        self._unviewWeight()

    def type(self, type=None, tensorCache=None):
        if self.finput is not None:
            self.finput = torch.Tensor()
        if self.fgradInput is not None:
            self.fgradInput = torch.Tensor()
        return super(SpatialConvolutionLocal, self).type(type, tensorCache)

    def __tostring__(self, ):
        s = super(SpatialConvolution, self).__repr__()
        s += '({} -> {}, {}x{}, {}x{}'.format(self.nInputPlane, self.nOutputPlane, self.iW, self.iH, self.kW, self.kH)
        if self.dW != 1 or self.dH != 1 or self.padW != 0 or self.padH != 0:
            s += ', {}, {}'.format(self.dW, self.dH)

        if self.padW != 0 or self.padH != 0:
            s += ', {}, {}'.format(self.padW, self.padH)

        s += ')'
        return s

    def clearState(self):
        clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
        return super(SpatialConvolutionLocal, self).clearState()
