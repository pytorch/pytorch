import torch
from torch.legacy import nn

class AbsCriterion(nn.Module):

    def __init__(self, sizeAverage):
        super(AbsCriterion, self).__init__()
        if sizeAverage != nil:
            self.sizeAverage = sizeAverage
        else:
            self.sizeAverage = True

    def updateOutput(self, input, target):
        self.output_tensor = self.output_tensor or input.new(1)
        self._backend.AbsCriterion_updateOutput(
            self._backend.library_state,
            input._cdata,
            target._cdata,
            self.output_tensor._cdata,
            self.sizeAverage
        )
        self.output = self.output_tensor[1]
        return self.output


    def updateGradInput(self, input, target):
        self._backend.AbsCriterion_updateGradInput(
            self._backend.library_state,
            input._cdata,
            target._cdata,
            self.gradInput._cdata,
            self.sizeAverage
        )
        return self.gradInput

