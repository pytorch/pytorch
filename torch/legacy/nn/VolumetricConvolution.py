import math
import torch
from .Module import Module
from .utils import clear

class VolumetricConvolution(Module):

    def __init__(self, nInputPlane, nOutputPlane, kT, kW, kH, dT=1, dW=1, dH=1, padT=0, padW=None, padH=None):
        super(VolumetricConvolution, self).__init__()

        self.nInputPlane = nInputPlane
        self.nOutputPlane = nOutputPlane
        self.kT = kT
        self.kW = kW
        self.kH = kH
        self.dT = dT
        self.dW = dW
        self.dH = dH
        self.padT = padT
        self.padW = padW or self.padT
        self.padH = padH or self.padW

        self.weight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
        self.bias = torch.Tensor(nOutputPlane)
        self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kT, kH, kW)
        self.gradBias = torch.Tensor(nOutputPlane)
        self.reset()

        self.finput = None
        self.fgradInput = None

    def reset(self, stdv=None):
        if stdv is not None:
           stdv = stdv * math.sqrt(3)
        else:
           stdv = 1. / math.sqrt(self.kT*self.kW*self.kH*self.nInputPlane)

        self.weight.uniform_(-stdv, stdv)
        self.bias.uniform_(-stdv, stdv)

    def _makeContiguous(self, input, gradOutput=None):
        if not input.isContiguous():
           self._input = self._input or input.new()
           self._input.resizeAs_(input).copy_(input)
           input = self._input

        if gradOutput is not None:
            if not gradOutput.isContiguous():
                self._gradOutput = self._gradOutput or gradOutput.new()
                self._gradOutput.resizeAs_(gradOutput).copy_(gradOutput)
                gradOutput = self._gradOutput
            return input, gradOutput

        return input

    # function to re-view the weight layout in a way that would make the MM ops happy
    def _viewWeight(self):
        self.weight = self.weight.view(self.nOutputPlane, self.nInputPlane * self.kT * self.kH * self.kW)
        if self.gradWeight and self.gradWeight.dim() > 0:
            self.gradWeight = self.gradWeight.view(self.nOutputPlane, self.nInputPlane * self.kT * self.kH * self.kW)

    def _unviewWeight(self):
        self.weight = self.weight.view(self.nOutputPlane, self.nInputPlane, self.kT, self.kH, self.kW)
        if self.gradWeight and self.gradWeight.dim() > 0:
            self.gradWeight = self.gradWeight.view(self.nOutputPlane, self.nInputPlane, self.kT, self.kH, self.kW)

    def updateOutput(self, input):
        self.finput = self.finput or input.new()
        self.fgradInput = self.fgradInput or input.new()
        if input.type() == 'torch.cuda.FloatTensor':
            self._backend.VolumetricConvolution_updateOutput(
                self._backend.library_state,
                input,
                self.output,
                self.weight,
                self.bias,
                self.finput,
                self.fgradInput,
                self.dT, self.dW, self.dH,
                self.padT, self.padW, self.padH
            )
        else:
            self._viewWeight()
            input = self._makeContiguous(input)
            self._backend.VolumetricConvolutionMM_updateOutput(
                self._backend.library_state,
                input,
                self.output,
                self.weight,
                self.bias,
                self.finput,
                self.kT, self.kW, self.kH,
                self.dT, self.dW, self.dH,
                self.padT, self.padW, self.padH
            )
            self._unviewWeight()

        return self.output

    def updateGradInput(self, input, gradOutput):
        if not self.gradInput:
            return
        if input.type() == 'torch.cuda.FloatTensor':
            self._backend.VolumetricConvolution_updateGradInput(
                self._backend.library_state,
                input,
                gradOutput,
                self.gradInput,
                self.weight,
                self.finput,
                self.dT, self.dW, self.dH,
                self.padT, self.padW, self.padH
            )
        else:
            self._viewWeight()
            input, gradOutput = self._makeContiguous(input, gradOutput)
            self._backend.VolumetricConvolutionMM_updateGradInput(
                self._backend.library_state,
                input,
                gradOutput,
                self.gradInput,
                self.weight,
                self.finput,
                self.fgradInput,
                self.kT, self.kW, self.kH,
                self.dT, self.dW, self.dH,
                self.padT, self.padW, self.padH
            )
            self._unviewWeight()

        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        if input.type() == 'torch.cuda.FloatTensor':
            self._backend.VolumetricConvolution_accGradParameters(
                self._backend.library_state,
                input,
                gradOutput,
                self.gradWeight,
                self.gradBias,
                self.finput,
                self.fgradInput,
                self.dT, self.dW, self.dH,
                self.padT, self.padW, self.padH,
                scale
            )
        else:
            input, gradOutput = self._makeContiguous(input, gradOutput)
            self._viewWeight()
            self._backend.VolumetricConvolutionMM_accGradParameters(
                self._backend.library_state,
                input,
                gradOutput,
                self.gradWeight,
                self.gradBias,
                self.finput,
                scale
            )
            self._unviewWeight()

    def type(self, type, tensorCache=None):
        clear(self, 'finput', 'fgradInput')
        return super(VolumetricConvolution, self).type(type, tensorCache)

    def clearState(self, ):
        clear(self, 'finput', 'fgradInput', '_input', '_gradOutput')
        return super(VolumetricConvolution, self).clearState()

    def __repr__(self):
        s = super(VolumetricConvolution, self).__repr__()
        s += '({} -> {}, {}x{}x{}'.format(self.nInputPlane, self.nOutputPlane, self.kT, self.kW, self.kH)
        if self.dT != 1 or self.dW != 1 or self.dH != 1 or \
           self.padT != 0 or self.padW != 0 or self.padH != 0:
               s += ', {}, {}, {}'.format(self.dT, self.dW, self.dH)

        if self.padT != 0 or self.padW != 0 or self.padH != 0:
               s += ', {}, {}, {}'.format(self.padT, self.padW, self.padH)

        s += ')'
        return s

