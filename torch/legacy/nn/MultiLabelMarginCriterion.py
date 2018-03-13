import torch
from .Criterion import Criterion


class MultiLabelMarginCriterion(Criterion):

    def __init__(self, sizeAverage=True):
        super(MultiLabelMarginCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.isTarget = torch.Tensor()
        self.output_tensor = None

    def updateOutput(self, input, target):
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        target = target.long()
        self._backend.MultiLabelMarginCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.isTarget,
            self.sizeAverage,
            True,  # reduce
        )
        self.output = self.output_tensor[0].item()
        return self.output

    def updateGradInput(self, input, target):
        target = target.long()
        implicit_gradOutput = torch.ones(1).type_as(input)
        self._backend.MultiLabelMarginCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            implicit_gradOutput,
            self.gradInput,
            self.isTarget,
            self.sizeAverage,
            True,  # reduce
        )
        return self.gradInput
