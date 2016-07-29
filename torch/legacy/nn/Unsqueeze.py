import torch
from torch.legacy import nn

class Unsqueeze(nn.Module):

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def updateOutput(self, input):
        nn.utils.addSingletonDimension(self.output, input, self.dim)
        return self.output

    def updateGradInput(self, input, gradOutput):
        assert input.nElement() == gradOutput.nElement()
        self.gradInput.view(gradOutput, input.size())
        return self.gradInput

    def __repr__(self):
        return super(Unsqueeze, self).__repr__() + '({})'.format(self.dim)

