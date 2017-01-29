import torch
from .Module import Module


class LeakyReLU(Module):

    def __init__(self, negval=1 / 100, inplace=False):
        super(LeakyReLU, self).__init__()
        if isinstance(negval, bool):
            inplace = negval
            self.negval = 1 / 100
        else:
            self.negval = negval

        # default for inplace is False
        self.inplace = inplace
        if self.negval < 0:
            # TODO: warning here
            self.inplace = False

    def updateOutput(self, input):
        self._backend.LeakyReLU_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.negval,
            self.inplace
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.LeakyReLU_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.negval,
            self.inplace
        )
        return self.gradInput

    def __repr__(self):
        return str(type(self)) + '({:.4f})'.format(self.negval)
