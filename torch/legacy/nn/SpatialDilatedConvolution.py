import torch
from .SpatialConvolution import SpatialConvolution


class SpatialDilatedConvolution(SpatialConvolution):

    def __init__(self, nInputPlane, nOutputPlane, kW, kH, dW=1, dH=1, padW=0, padH=None, dilationH=1, dilationW=None):
        super(SpatialDilatedConvolution, self).__init__(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)

        self.dilationH = dilationH
        self.dilationW = dilationW if dilationW is not None else dilationH

    def updateOutput(self, input):
        if self.finput is None:
            self.finput = self.weight.new()
        if self.fgradInput is None:
            self.fgradInput = self.weight.new()
        input = self._makeContiguous(input)
        self._backend.SpatialDilatedConvolution_updateOutput(
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
            self.dilationH, self.dilationW
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput is None:
            return

        input, gradOutput = self._makeContiguous(input, gradOutput)
        if self.fgradInput is None:
            self.fgradInput = self.weight.new()
        self._backend.SpatialDilatedConvolution_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.finput,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            self.dilationH, self.dilationW
        )
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        input, gradOutput = self._makeContiguous(input, gradOutput)
        if self.fgradInput is None:
            self.fgradInput = self.weight.new()
        self._backend.SpatialDilatedConvolution_accGradParameters(
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
            self.dilationH, self.dilationW,
            scale
        )

    def __repr__(self):
        s = super(SpatialConvolution, self).__repr__()
        s += '({} -> {}, {}x{}'.format(self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
        if self.dW != 1 or self.dH != 1 or self.padW != 0 or self.padH != 0:
            s += ', {}, {}'.format(self.dW, self.dH)

        if self.padW != 0 or self.padH != 0:
            s += ', {}, {}'.format(self.padW, self.padH)

        s += ', {}, {}'.format(self.dilationW, self.dilationH)

        s += ')'
        if self.bias is None:
            s += ' without bias'
        return s
