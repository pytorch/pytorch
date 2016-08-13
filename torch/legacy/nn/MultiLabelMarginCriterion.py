import torch
from .Criterion import Criterion

class MultiLabelMarginCriterion(Criterion):

    def __init__(self, sizeAverage=True):
        super(MultiLabelMarginCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.isTarget = torch.Tensor()
        self.output_tensor = None

    def updateOutput(self, input, target):
        self.output_tensor = self.output_tensor or input.new(1)
        self._backend.MultiLabelMarginCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.isTarget,
            self.sizeAverage
        )
        self.output = self.output_tensor[0]
        return self.output

    def updateGradInput(self, input, target):
        self._backend.MultiLabelMarginCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            self.gradInput,
            self.isTarget,
            self.sizeAverage
        )
        return self.gradInput

