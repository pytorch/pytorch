import torch
from torch.legacy import nn

class CMulTable(nn.Module):

    def __init__(self, ):
        super(CMulTable, self).__init__()
        self.gradInput = []

    def updateOutput(self, input):
        self.output.resizeAs(input[0]).copy(input[0])
        for i in range(1, len(input)):
            self.output.cmul(input[i])

        return self.output

    def updateGradInput_efficient(self, input, gradOutput):
        self.tout = self.tout or input[0].new()
        self.tout.resizeAs(self.output)
        for i in range(len(input)):
            if i not in self.gradInput:
                self.gradInput[i] = input[0].new()
            self.gradInput[i].resizeAs(input[i]).copy(gradOutput)
            self.tout.copy(self.output).cdiv(input[i])
            self.gradInput[i].cmul(self.tout)

        self.gradInput = self.gradInput[:len(input)]
        return self.gradInput

    def updateGradInput(self, input, gradOutput):
        for i in range(len(input)):
            if i not in self.gradInput:
                self.gradInput[i] = input[0].new()
            self.gradInput[i].resizeAs(input[i]).copy(gradOutput)
            for j in range(len(input)):
                if i != j:
                    self.gradInput[i].cmul(input[j])

        self.gradInput = self.gradInput[:len(input)]
        return self.gradInput

    def clearState(self):
        if self.tout: self.tout.set()
        return super(CMulTable, self).clearState(self)
