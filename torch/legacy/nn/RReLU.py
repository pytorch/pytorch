import torch
from .Module import Module
from .utils import clear


class RReLU(Module):

    def __init__(self, lower=1. / 8, upper=1. / 3, inplace=False):
        super(RReLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

        assert self.lower <= self.upper and self.lower >= 0 and self.upper >= 0
        self.noise = torch.Tensor()
        self.train = True

    def updateOutput(self, input):
        self._backend.RReLU_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.noise,
            self.lower,
            self.upper,
            self.train,
            self.inplace,
            torch.default_generator if not input.is_cuda else 0
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.RReLU_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.noise,
            self.lower,
            self.upper,
            self.train,
            self.inplace
        )
        return self.gradInput

    def __repr__(self):
        return super(RReLU, self).__repr__() + '({:.4f}, {:.4f})'.format(self.lower, self.upper)

    def clearState(self):
        clear(self, 'noise')
        return super(RReLU, self).clearState()
