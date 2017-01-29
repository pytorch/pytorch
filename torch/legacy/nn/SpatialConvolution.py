import math
import torch
from .Module import Module
from .utils import clear


class SpatialConvolution(Module):

    def __init__(self, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=None):
        super(SpatialConvolution, self).__init__()

        self.nInputPlane = nInputPlane
        self.nOutputPlane = nOutputPlane
        self.kW = kW
        self.kH = kH

        self.dW = dW
        self.dH = dH
        self.padW = padW
        self.padH = padH if padH is not None else padW

        self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
        self.bias = torch.Tensor(nOutputPlane)
        self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
        self.gradBias = torch.Tensor(nOutputPlane)

        self.reset()
        self._input = None
        self._gradOutput = None
        self.finput = None
        self.fgradInput = None

    def noBias(self):
        self.bias = None
        self.gradBias = None
        return self

    def reset(self, stdv=None):
        if stdv is not None:
            stdv = stdv * math.sqrt(3)
        else:
            stdv = 1. / math.sqrt(self.kW * self.kH * self.nInputPlane)

        self.weight.uniform_(-stdv, stdv)
        if self.bias is not None:
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

    def _init(self):
        if self.finput is None:
            self.finput = self.weight.new()
        if self.fgradInput is None:
            self.fgradInput = self.weight.new()

    # function to re-view the weight layout in a way that would make the MM ops happy
    def _viewWeight(self):
        self.weight = self.weight.view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)
        if self.gradWeight is not None and self.gradWeight.dim() > 0:
            self.gradWeight = self.gradWeight.view(self.nOutputPlane, self.nInputPlane * self.kH * self.kW)

    def _unviewWeight(self):
        self.weight = self.weight.view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
        if self.gradWeight is not None and self.gradWeight.dim() > 0:
            self.gradWeight = self.gradWeight.view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)

    def updateOutput(self, input):
        self._init()
        self._viewWeight()
        input = self._makeContiguous(input)
        self._backend.SpatialConvolutionMM_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.bias,
            self.finput,
            self.fgradInput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH
        )
        self._unviewWeight()
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        self._init()
        self._viewWeight()
        input, gradOutput = self._makeContiguous(input, gradOutput)
        self._backend.SpatialConvolutionMM_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.finput,
            self.fgradInput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH
        )
        self._unviewWeight()
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self._init()
        input, gradOutput = self._makeContiguous(input, gradOutput)
        self._viewWeight()
        self._backend.SpatialConvolutionMM_accGradParameters(
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
            scale
        )
        self._unviewWeight()

    def type(self, type=None, tensorCache={}):
        if self.finput is not None:
            self.finput = torch.Tensor()
        if self.fgradInput is not None:
            self.fgradInput = torch.Tensor()
        return super(SpatialConvolution, self).type(type, tensorCache)

    def __repr__(self):
        s = super(SpatialConvolution, self).__repr__()
        s += '({} -> {}, {}x{}'.format(self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
        if self.dW != 1 or self.dH != 1 or self.padW != 0 or self.padH != 0:
            s += ', {}, {}'.format(self.dW, self.dH)

        if self.padW != 0 or self.padH != 0:
            s += ', {}, {}'.format(self.padW, self.padH)

        s += ')'
        if self.bias is None:
            s += ' without bias'
        return s

    def clearState(self):
        clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
        return super(SpatialConvolution, self).clearState()
