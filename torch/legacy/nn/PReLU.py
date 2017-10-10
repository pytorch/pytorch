import torch
from .Module import Module
from .utils import clear


class PReLU(Module):

    def __init__(self, nOutputPlane=0):
        super(PReLU, self).__init__()
        # if no argument provided, use shared model (weight is scalar)
        self.nOutputPlane = nOutputPlane
        self.weight = torch.Tensor(nOutputPlane or 1).fill_(0.25)
        self.gradWeight = torch.Tensor(nOutputPlane or 1)

    def updateOutput(self, input):
        self._backend.PReLU_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.PReLU_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight
        )
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        self._backend.PReLU_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.gradWeight,
            scale
        )
        return self.gradWeight

    def clearState(self):
        clear(self, 'gradWeightBuf', 'gradWeightBuf2')
        return super(PReLU, self).clearState()
