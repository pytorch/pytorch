import torch
from .Criterion import Criterion


class ClassNLLCriterion(Criterion):

    def __init__(self, weights=None, sizeAverage=True, ignore_index=-100):
        super(ClassNLLCriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.ignore_index = ignore_index

        if weights is not None:
            assert weights.dim() == 1
        self.weights = weights

        self.output_tensor = torch.zeros(1)
        self.total_weight_tensor = torch.ones(1)

    def updateOutput(self, input, target):
        self.ignore_index = getattr(self, "ignore_index", -100)
        target = target.long()
        self._backend.ClassNLLCriterion_updateOutput(
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
        target = target.long()
        implicit_gradOutput = torch.ones(1).type_as(input)

        self._backend.ClassNLLCriterion_updateGradInput(
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
