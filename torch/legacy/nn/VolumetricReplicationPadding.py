import torch
from .Module import Module


class VolumetricReplicationPadding(Module):

    def __init__(self, pleft, pright=None, ptop=None, pbottom=None, pfront=None, pback=None):
        super(VolumetricReplicationPadding, self).__init__()
        self.pleft = pleft
        self.pright = pright or pleft
        self.ptop = ptop or pleft
        self.pbottom = pbottom or pleft
        self.pfront = pfront or pleft
        self.pback = pback or pleft

    def updateOutput(self, input):
        assert input.dim() == 5
        self._backend.VolumetricReplicationPadding_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.pleft, self.pright,
            self.ptop, self.pbottom,
            self.pfront, self.pback
        )

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 5 and gradOutput.dim() == 5
        assert input.size(0) == gradOutput.size(0)
        assert input.size(1) == gradOutput.size(1)
        assert input.size(2) + self.pfront + self.pback == gradOutput.size(2)
        assert input.size(3) + self.ptop + self.pbottom == gradOutput.size(3)
        assert input.size(4) + self.pleft + self.pright == gradOutput.size(4)

        self._backend.VolumetricReplicationPadding_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.pleft, self.pright,
            self.ptop, self.pbottom,
            self.pfront, self.pback
        )

        return self.gradInput

    def __repr__(self):
        s = super(VolumetricReplicationPadding, self).__repr__()
        s += '({}, {}, {}, {}, {}, {})'.format(self.pleft, self.pright,
                                               self.ptop, self.pbottom,
                                               self.pfront, self.pback
                                               )
        return s
