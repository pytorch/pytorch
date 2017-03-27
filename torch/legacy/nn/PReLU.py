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
        self.gradWeightBuf = None
        self.gradWeightBuf2 = None

    def updateOutput(self, input):
        self._backend.PReLU_updateOutput(
            self._backend.library_state,
            input,
            self.output,
            self.weight,
            self.nOutputPlane
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self._backend.PReLU_updateGradInput(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.nOutputPlane
        )
        return self.gradInput

    def accGradParameters(self, input, gradOutput, scale=1):
        if self.gradWeightBuf is None:
            self.gradWeightBuf = input.new()
        if self.gradWeightBuf2 is None:
            self.gradWeightBuf2 = input.new()
        self._backend.PReLU_accGradParameters(
            self._backend.library_state,
            input,
            gradOutput,
            self.gradInput,
            self.weight,
            self.gradWeight,
            self.gradWeightBuf,
            self.gradWeightBuf2,
            self.nOutputPlane,
            scale
        )
        return self.gradWeight

    def clearState(self):
        clear(self, 'gradWeightBuf', 'gradWeightBuf2')
        return super(PReLU, self).clearState()
