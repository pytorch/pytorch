import torch
from .Module import Module


class CSubTable(Module):

    def __init__(self, ):
        super(CSubTable, self).__init__()
        self.gradInput = [torch.Tensor(), torch.Tensor()]

    def updateOutput(self, input):
        self.output.resize_as_(input[0]).copy_(input[0])
        self.output.add_(-1, input[1])
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.gradInput[0] is None:
            self.gradInput[0] = input[0].new()
        if self.gradInput[1] is None:
            self.gradInput[1] = input[1].new()
        self.gradInput[0].resize_as_(input[0]).copy_(gradOutput)
        self.gradInput[1].resize_as_(input[1]).copy_(gradOutput).mul_(-1)

        self.gradInput = self.gradInput[:2]
        return self.gradInput
