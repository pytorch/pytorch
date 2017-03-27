import torch
from .Module import Module


class CAddTable(Module):

    def __init__(self, inplace=False):
        super(CAddTable, self).__init__()
        self.inplace = inplace
        self.gradInput = []

    def updateOutput(self, input):
        if self.inplace:
            self.output.set_(input[0])
        else:
            self.output.resize_as_(input[0]).copy_(input[0])

        for i in range(1, len(input)):
            self.output.add_(input[i])

        return self.output

    def updateGradInput(self, input, gradOutput):
        for i in range(len(input)):
            if i >= len(self.gradInput):
                assert i == len(self.gradInput)
                self.gradInput.append(input[0].new())

            if self.inplace:
                self.gradInput[i].set_(gradOutput)
            else:
                self.gradInput[i].resize_as_(input[i]).copy_(gradOutput)

        del self.gradInput[len(input):]

        return self.gradInput
