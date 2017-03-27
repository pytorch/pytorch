import torch
from .Criterion import Criterion


class SpatialClassNLLCriterion(Criterion):

    def __init__(self, weights=None, sizeAverage=True):
        assert weights is None or weights.dim() == 1
        super(SpatialClassNLLCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.weights = weights

        self.output_tensor = torch.zeros(1)
        self.total_weight_tensor = torch.ones(1)

    def updateOutput(self, input, target):
        self._backend.SpatialClassNLLCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.sizeAverage,
            self.weights,
            self.total_weight_tensor
        )
        self.output = self.output_tensor[0]
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput.resize_as_(input).zero_()
        self._backend.SpatialClassNLLCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            self.gradInput,
            self.sizeAverage,
            self.weights,
            self.total_weight_tensor
        )
        return self.gradInput
