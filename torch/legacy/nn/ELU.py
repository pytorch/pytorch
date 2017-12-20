# -*- coding: utf8 -*-
import torch
from .Module import Module


class ELU(Module):
    """
            Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter
            Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
            http.//arxiv.org/pdf/1511.07289.pdf
    """

    def __init__(self, alpha=1., inplace=False):
        assert type(alpha) == float
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def updateOutput(self, input):
        self._backend.ELU_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.alpha,
            1.0,
            self.inplace
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.ELU_updateGradInput(
            self._backend.library_state,
            gradOutput,
            self.gradInput,
            self.output,
            self.alpha,
            1.0
        )
        return self.gradInput

    def __repr__(self):
        return '{}(alpha={:.3f})'.format(str(type(self)), self.alpha)
