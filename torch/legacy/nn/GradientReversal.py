import torch
from torch.legacy import nn

class GradientReversal(nn.Module):

    def __init__(self, lambd=1):
        super(GradientReversal, self).__init__()
        self.lambd = lambd

    def setLambda(self, lambd):
        self.lambd = lambd

    def updateOutput(self, input):
        self.output.set(input)
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput.resizeAs(gradOutput)
        self.gradInput.copy(gradOutput)
        self.gradInput.mul(-self.lambd)
        return self.gradInput

