import torch
from .Module import Module


class Threshold(Module):

    def __init__(self, threshold=0, value=0, inplace=False):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value

        # default for inplace is False
        self.inplace = inplace
        self.validateParameters()

    def updateOutput(self, input):
        self.validateParameters()
        self._backend.Threshold_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.threshold,
            self.value,
            self.inplace
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.validateParameters()
        self._backend.Threshold_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.threshold,
            self.value,
            self.inplace
        )
        return self.gradInput

    def validateParameters(self):
        if self.inplace:
            if self.value > self.threshold:
                raise RuntimeError('in-place processing requires value ({}) to not '
                                   'exceed threshold ({})'.format(self.value, self.threshold))
