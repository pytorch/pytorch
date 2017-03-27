import torch
from .Criterion import Criterion


class SoftMarginCriterion(Criterion):

    def __init__(self, ):
        super(SoftMarginCriterion, self).__init__()
        self.sizeAverage = True
        self.output_tensor = None

    def updateOutput(self, input, target):
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.SoftMarginCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.sizeAverage
        )
        self.output = self.output_tensor[0]
        return self.output

    def updateGradInput(self, input, target):
        self._backend.SoftMarginCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            self.gradInput,
            self.sizeAverage
        )
        return self.gradInput
