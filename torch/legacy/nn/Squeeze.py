import torch
from torch.legacy import nn

class Squeeze(nn.Module):

    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def updateOutput(self, input):
        dim = self.dim
        # TODO: this operates inplace
        self.output.set(input.squeeze(dim) if dim is not None else input.squeeze())
        return self.output


    def updateGradInput(self, input, gradOutput):
        assert input.nElement() == gradOutput.nElement()
        self.gradInput.set(gradOutput.viewAs(input))
        return self.gradInput

