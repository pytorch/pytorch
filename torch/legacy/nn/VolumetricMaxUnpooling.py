import torch
from .Module import Module
from .VolumetricMaxPooling import VolumetricMaxPooling


class VolumetricMaxUnpooling(Module):

    def __init__(self, poolingModule):
        super(VolumetricMaxUnpooling, self).__init__()
        assert isinstance(poolingModule, VolumetricMaxPooling)
        assert poolingModule.kT == poolingModule.dT
        assert poolingModule.kH == poolingModule.dH
        assert poolingModule.kW == poolingModule.dW
        self.pooling = poolingModule

    def _setParams(self):
        self.indices = self.pooling.indices
        self.otime = self.pooling.itime
        self.oheight = self.pooling.iheight
        self.owidth = self.pooling.iwidth
        self.dT = self.pooling.dT
        self.dH = self.pooling.dH
        self.dW = self.pooling.dW
        self.padT = self.pooling.padT
        self.padH = self.pooling.padH
        self.padW = self.pooling.padW

    def updateOutput(self, input):
        self._setParams()
        self._backend.VolumetricMaxUnpooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.indices,
            self.otime, self.owidth, self.oheight,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._setParams()
        self._backend.VolumetricMaxUnpooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.indices,
            self.otime, self.owidth, self.oheight,
            self.dT, self.dW, self.dH,
            self.padT, self.padW, self.padH
        )
        return self.gradInput

    def __repr__(self):
        return 'nn.VolumetricMaxUnpooling associated to ' + self.pooling.__repr__()
