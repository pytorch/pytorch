import torch
from .Module import Module
from .SpatialMaxPooling import SpatialMaxPooling


class SpatialMaxUnpooling(Module):

    def __init__(self, poolingModule):
        super(SpatialMaxUnpooling, self).__init__()
        assert isinstance(poolingModule, SpatialMaxPooling)
        assert poolingModule.kH == poolingModule.dH
        assert poolingModule.kW == poolingModule.dW
        self.pooling = poolingModule

    def _setParams(self):
        self.indices = self.pooling.indices
        self.oheight = self.pooling.iheight
        self.owidth = self.pooling.iwidth

    def updateOutput(self, input):
        self._setParams()
        self._backend.SpatialMaxUnpooling_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.indices,
            self.owidth, self.oheight
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._setParams()
        self._backend.SpatialMaxUnpooling_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.indices,
            self.owidth, self.oheight
        )
        return self.gradInput

    def __repr__(self):
        return 'nn.SpatialMaxUnpooling associated to ' + self.pooling.__repr__()
