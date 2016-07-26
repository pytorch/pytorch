import torch
from torch.legacy import nn

class DistKLDivCriterion(nn.Criterion):

    def __init__(self, sizeAverage=True):
        super(DistKLDivCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.output_tensor = torch.Tensor(1)

    def updateOutput(self, input, target):
        assert input.isSameSizeAs(target)
        self.output_tensor = self.output_tensor or input.new(1)
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
        assert input.isSameSizeAs(target)
        self._backend.DistKLDivCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            self.gradInput,
            self.sizeAverage
        )
        return self.gradInput

