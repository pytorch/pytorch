import torch
from torch.nn.functional import _Reduction
from .Criterion import Criterion


class MultiMarginCriterion(Criterion):

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
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        target = target.long()
        self._backend.MultiMarginCriterion_updateOutput(
            self._backend.library_state,
            input,
            target,
            self.output_tensor,
            _Reduction.legacy_get_enum(self.sizeAverage, True, emit_warning=False),
            self.p,
            self.weights,
            self.margin,
        )
        self.output = self.output_tensor[0].item()
        return self.output

    def updateGradInput(self, input, target):
        target = target.long()
        implicit_gradOutput = torch.ones(1).type_as(input)
        self._backend.MultiMarginCriterion_updateGradInput(
            self._backend.library_state,
            input,
            target,
            implicit_gradOutput,
            self.gradInput,
            _Reduction.legacy_get_enum(self.sizeAverage, True, emit_warning=False),
            self.p,
            self.weights,
            self.margin,
        )
        return self.gradInput
