import torch
from torch.legacy import nn

class VolumetricAveragePooling(nn.Module):

    def __init__(self, kT, kW, kH, dT=None, dW=None, dH=None):
        super(VolumetricAveragePooling, self).__init__()
        self.kT = kT
        self.kH = kH
        self.kW = kW
        self.dT = dT or kT
        self.dW = dW or kW
        self.dH = dH or kH

    def updateOutput(self, input):
        self._backend.VolumetricAveragePooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.VolumetricAveragePooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH
        )
        return self.gradInput

    def __repr__(self):
        s = super(VolumetricAveragePooling, self).__repr__()
        s += '({}x{}x{}, {}, {}, {}'.format(self.kT, self.kW, self.kH, self.dT, self.dW, self.dH)
        if self.padT != 0 or self.padW != 0 or self.padH != 0:
            s += ', {}, {}, {}'.format(self.padT, self.padW, self.padH)
        s += ')'
        return s

