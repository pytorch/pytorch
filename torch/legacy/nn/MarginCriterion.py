import torch
from .Criterion import Criterion


class MarginCriterion(Criterion):

    def __init__(self, margin=1, sizeAverage=True):
        super(MarginCriterion, self).__init__()
        self.sizeAverage = True
        self.margin = margin
        self.output_tensor = None

    def updateOutput(self, input, target):
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.MarginCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.sizeAverage,
            self.margin
        )
        self.output = self.output_tensor[0].item()
        return self.output

    def updateGradInput(self, input, target):
        self._backend.MarginCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            self.gradInput,
            self.sizeAverage,
            self.margin
        )
        return self.gradInput
