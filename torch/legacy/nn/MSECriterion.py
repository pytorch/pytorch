import torch
from .Criterion import Criterion


class MSECriterion(Criterion):

    def __init__(self, sizeAverage=True):
        super(MSECriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.output_tensor = None

    def updateOutput(self, input, target):
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.MSECriterion_updateOutput(
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
        implicit_gradOutput = torch.Tensor([1]).type(input.type())

        self._backend.MSECriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            implicit_gradOutput,
            self.gradInput,
            self.sizeAverage,
            True,  # reduce
        )
        return self.gradInput
