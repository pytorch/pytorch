import torch
from torch.nn.functional import _Reduction
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
            _Reduction.legacy_get_enum(self.sizeAverage, True, emit_warning=False),
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
            _Reduction.legacy_get_enum(self.sizeAverage, True, emit_warning=False),
        )
        return self.gradInput
