import torch
from .Criterion import Criterion


class SpatialClassNLLCriterion(Criterion):

    def __init__(self, weights=None, sizeAverage=True, ignore_index=-100):
        assert weights is None or weights.dim() == 1
        super(SpatialClassNLLCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.weights = weights
        self.ignore_index = ignore_index

        self.output_tensor = torch.zeros(1)
        self.total_weight_tensor = torch.ones(1)

    def updateOutput(self, input, target):
        if not hasattr(self, 'ignore_index'):
            self.ignore_index = -100
        self._backend.SpatialClassNLLCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            self.sizeAverage,
            self.weights,
            self.total_weight_tensor,
            self.ignore_index,
            True,  # reduce
        )
        self.output = self.output_tensor[0].item()
        return self.output

    def updateGradInput(self, input, target):
        self.gradInput.resize_as_(input).zero_()
        implicit_gradOutput = torch.ones(1).type_as(input)
        self._backend.SpatialClassNLLCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            implicit_gradOutput,
            self.gradInput,
            self.sizeAverage,
            self.weights,
            self.total_weight_tensor,
            self.ignore_index,
            True,  # reduce
        )
        return self.gradInput
