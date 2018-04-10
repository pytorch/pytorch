import torch
from .Criterion import Criterion


class AbsCriterion(Criterion):

    def __init__(self, sizeAverage=True):
        super(AbsCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.output_tensor = torch.Tensor(1)

    def updateOutput(self, input, target):
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.AbsCriterion_updateOutput(
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
        self._backend.AbsCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            implicit_gradOutput,
            self.gradInput,
            self.sizeAverage,
            True,  # reduce
        )
        return self.gradInput
