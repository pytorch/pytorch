import torch
from .Criterion import Criterion


class SmoothL1Criterion(Criterion):

    def __init__(self, sizeAverage=True):
        super(SmoothL1Criterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.output_tensor = None

    def updateOutput(self, input, target):
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.SmoothL1Criterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.sizeAverage,
            True,  # reduce
        )
        self.output = self.output_tensor[0].item()
        return self.output

    def updateGradInput(self, input, target):
        implicit_gradOutput = torch.ones(1).type_as(input)
        self._backend.SmoothL1Criterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            implicit_gradOutput,
            self.gradInput,
            self.sizeAverage,
            True,  # reduce
        )
        return self.gradInput
