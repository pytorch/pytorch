import torch
from torch.legacy import nn

class MultiMarginCriterion(nn.Criterion):

    def __init__(self, p=1, weights=None, margin=1, sizeAverage=True):
        super(MultiMarginCriterion, self).__init__()
        if p != 1 and p != 2:
            raise ValueError("only p == 1 and p == 2 supported")
        self.p = p
        self.margin = margin
        self.sizeAverage = sizeAverage
        if weights is not None:
            assert weights.dim() == 1
        self.weights = weights
        self.output_tensor = None

    def updateOutput(self, input, target):
        self.output_tensor = self.output_tensor or input.new(1)
        self._backend.MultiMarginCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.sizeAverage,
            self.p,
            self.weights,
            self.margin
        )
        self.output = self.output_tensor[0]
        return self.output


    def updateGradInput(self, input, target):
        self._backend.MultiMarginCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            self.gradInput,
            self.sizeAverage,
            self.p,
            self.weights,
            self.margin
        )
        return self.gradInput

