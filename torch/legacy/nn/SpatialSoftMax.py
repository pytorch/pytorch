import torch
from .Module import Module


class SpatialSoftMax(Module):

    def updateOutput(self, input):
        self.output = torch.softmax(
            input,
            0 if input.dim() == 1 or input.dim() == 3 else 1
        )
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = torch.softmax_backward_data(
            gradOutput,
            self.output,
            0 if input.dim() == 1 or input.dim() == 3 else 1,
            input
        )
        return self.gradInput
