import torch
from .Criterion import Criterion


class DistKLDivCriterion(Criterion):

    def __init__(self, sizeAverage=True):
        super(DistKLDivCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.output_tensor = torch.Tensor(1)

    def updateOutput(self, input, target):
        assert input.is_same_size(target)
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.DistKLDivCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.sizeAverage
        )
        self.output = self.output_tensor[0]
        return self.output

    def updateGradInput(self, input, target):
        assert input.is_same_size(target)
        self._backend.DistKLDivCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            self.gradInput,
            self.sizeAverage
        )
        return self.gradInput
