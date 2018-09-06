import torch
from .Criterion import Criterion
from .utils import clear


class L1Cost(Criterion):

    def __init__(self):
        super(L1Cost, self).__init__()
        self.output_tensor = torch.Tensor(1)

    def updateOutput(self, input, target=None):
        assert target is None
        if self.output_tensor is None:
            self.output_tensor = input.new(1)
        self._backend.L1Cost_updateOutput(
            self._backend.library_state,
            input,
            self.output_tensor
        )
        self.output = self.output_tensor[0].item()
        return self.output

    def updateGradInput(self, input, target=None):
        assert target is None
        self._backend.L1Cost_updateGradInput(
            self._backend.library_state,
            input,
            None,
            self.gradInput
        )
        return self.gradInput

    def clearState(self):
        clear(self, 'output_tensor')
        return super(L1Cost, self).clearState()
