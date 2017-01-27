import torch
from .Module import Module
from .utils import clear


class VolumetricMaxPooling(Module):

    def __init__(self, kT, kW, kH, dT=None, dW=None, dH=None, padT=0, padW=0, padH=0):
        super(VolumetricMaxPooling, self).__init__()

        self.kT = kT
        self.kH = kH
        self.kW = kW
        self.dT = dT or kT
        self.dW = dW or kW
        self.dH = dH or kH

        self.padT = padT
        self.padW = padW
        self.padH = padH

        self.ceil_mode = False
        self.indices = torch.LongTensor()

    def ceil(self):
        self.ceil_mode = True
        return self

    def floor(self):
        self.ceil_mode = False
        return self

    def updateOutput(self, input):
        dims = input.dim()
        self.itime = input.size(dims - 3)
        self.iheight = input.size(dims - 2)
        self.iwidth = input.size(dims - 1)

        if self.indices is None:
            self.indices = input.new()
        self.indices = self.indices.long()
        self._backend.VolumetricMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.indices,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            self.ceil_mode
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.VolumetricMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.indices,
            self.kT, self.kW, self.kH,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH,
            self.ceil_mode
        )
        return self.gradInput

    def clearState(self):
        clear(self, 'indices')
        return super(VolumetricMaxPooling, self).clearState()

    def __repr__(self):
        s = super(VolumetricMaxPooling, self).__repr__()
        s += '({}x{}x{}, {}, {}, {}'.format(self.kT, self.kW, self.kH, self.dT, self.dW, self.dH)
        if self.padT != 0 or self.padW != 0 or self.padH != 0:
            s += ', {}, {}, {}'.format(self.padT, self.padW, self.padH)
        s += ')'
        return s
