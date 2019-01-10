import torch
from .Module import Module


class SpatialReplicationPadding(Module):

    def __init__(self, pad_l, pad_r=None, pad_t=None, pad_b=None):
        super(SpatialReplicationPadding, self).__init__()
        self.pad_l = pad_l
        self.pad_r = pad_r if pad_r is not None else pad_l
        self.pad_t = pad_t if pad_t is not None else pad_l
        self.pad_b = pad_b if pad_b is not None else pad_l

    def updateOutput(self, input):
        assert input.dim() == 4
        self._backend.SpatialReplicationPadding_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.pad_l, self.pad_r, self.pad_t, self.pad_b
        )

        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.dim() == 4 and gradOutput.dim() == 4
        assert input.size(0) == gradOutput.size(0) and \
            input.size(1) == gradOutput.size(1) and \
            input.size(2) + self.pad_t + self.pad_b == gradOutput.size(2) and \
            input.size(3) + self.pad_l + self.pad_r == gradOutput.size(3)

        self._backend.SpatialReplicationPadding_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.pad_l, self.pad_r, self.pad_t, self.pad_b
        )

        return self.gradInput

    def __repr__(self):
        s = super(SpatialReplicationPadding, self).__repr__()
        s += '({}, {}, {}, {})'.format(self.pad_l, self.pad_r, self.pad_t, self.pad_b)
        return s
