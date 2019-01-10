import torch
from .Module import Module
from .utils import clear


class CMulTable(Module):

    def __init__(self, ):
        super(CMulTable, self).__init__()
        self.gradInput = []

    def updateOutput(self, input):
        self.output.resize_as_(input[0]).copy_(input[0])
        for i in range(1, len(input)):
            self.output.mul_(input[i])

        return self.output

    def updateGradInput_efficient(self, input, gradOutput):
        if self.tout is None:
            self.tout = input[0].new()
        self.tout.resize_as_(self.output)
        for i in range(len(input)):
            if len(self.gradInput) <= i:
                assert i == len(self.gradInput)
                self.gradInput.append(input[0].new())
            self.gradInput[i].resize_as_(input[i]).copy_(gradOutput)
            self.tout.copy_(self.output).div_(input[i])
            self.gradInput[i].mul_(self.tout)

        self.gradInput = self.gradInput[:len(input)]
        return self.gradInput

    def updateGradInput(self, input, gradOutput):
        for i in range(len(input)):
            if len(self.gradInput) <= i:
                assert i == len(self.gradInput)
                self.gradInput.append(input[0].new())
            self.gradInput[i].resize_as_(input[i]).copy_(gradOutput)
            for j in range(len(input)):
                if i != j:
                    self.gradInput[i].mul_(input[j])

        self.gradInput = self.gradInput[:len(input)]
        return self.gradInput

    def clearState(self):
        clear(self, 'tout')
        return super(CMulTable, self).clearState()
