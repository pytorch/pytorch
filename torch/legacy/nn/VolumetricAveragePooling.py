import torch
from .Module import Module


class VolumetricAveragePooling(Module):

    def __init__(self, kT, kW, kH, dT=1, dW=1, dH=1, padT=0,padW=0,padH=0):
        super(VolumetricAveragePooling, self).__init__()

        self.kT = kT
        self.kH = kH
        self.kW = kW
        self.dT = dT
        self.dW = dW
        self.dH = dH
        self.padT = padT
        self.padH = padH
        self.padW = padW
        self.ceil_mode = False
        self.count_include_pad = True




    def updateOutput(self, input):

        self._backend.VolumetricAveragePooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT,self.padW,self.padH,
            self.ceil_mode, self.count_include_pad)
        return self.output

    def updateGradInput(self, input, gradOutput):

        self._backend.VolumetricAveragePooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padH,self.padW,
            self.ceil_mode,self.count_include_pad
        )
        return self.gradInput

    def __repr__(self):
        s = super(VolumetricAveragePooling, self).__repr__()
        s += '({}x{}x{}, {}, {}, {}'.format(self.kT, self.kW, self.kH)
        if (self.padT or self.padW or self.padH) and (self.padT !=0 or self.padW != 0 or self.padH != 0):
            s += ', {}, {},{}'.format(self.padT, self.padW, self.padH)
        s += ')'
        return s
