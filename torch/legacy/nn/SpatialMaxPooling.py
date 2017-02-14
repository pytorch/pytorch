import torch
from .Module import Module
from .utils import clear


class SpatialMaxPooling(Module):

    def __init__(self, kW, kH, dW=None, dH=None, padW=0, padH=0):
        super(SpatialMaxPooling, self).__init__()

        dW = dW or kW
        dH = dH or kH

        self.kW = kW
        self.kH = kH
        self.dW = dW
        self.dH = dH

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
        if not hasattr(self, 'indices') or self.indices is None:
            self.indices = input.new()
        self.indices = self.indices.long()

        dims = input.dim()
        self.iheight = input.size(dims - 2)
        self.iwidth = input.size(dims - 1)

        self._backend.SpatialMaxPooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.indices,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            self.ceil_mode
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.SpatialMaxPooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.indices,
            self.kW, self.kH,
            self.dW, self.dH,
            self.padW, self.padH,
            self.ceil_mode
        )
        return self.gradInput

    def __repr__(self):
        s = super(SpatialMaxPooling, self).__repr__()
        s += '({}x{}, {}, {}'.format(self.kW, self.kH, self.dW, self.dH)
        if (self.padW or self.padH) and (self.padW != 0 or self.padH != 0):
            s += ', {}, {}'.format(self.padW, self.padH)
        s += ')'

        return s

    def clearState(self):
        clear(self, 'indices')
        return super(SpatialMaxPooling, self).clearState()
