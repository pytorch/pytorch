import torch
from torch.legacy import nn

class L1Cost(nn.Criterion):

    def __init__(self):
        super(L1Cost, self).__init__()
        self.output_tensor = torch.Tensor(1)

    def updateOutput(self, input, target=None):
        assert target is None
        self.output_tensor = self.output_tensor or input.new(1)
        self._backend.L1Cost_updateOutput(
            self._backend.library_state,
            input,
            self.output_tensor
        )
        self.output = self.output_tensor[0]
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
        # TODO: this shouldn't call set
        if self.output_tensor:
            self.output_tensor.set()
        return super(L1Cost, self).clearState()

