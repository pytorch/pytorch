import torch
from .Module import Module


class Contiguous(Module):

    def updateOutput(self, input):
        if not input.is_contiguous():
            self.output.resize_as_(input).copy_(input)
        else:
            self.output.set_(input)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if not gradOutput.is_contiguous():
            self.gradInput.resize_as_(gradOutput).copy_(gradOutput)
        else:
            self.gradInput.set_(gradOutput)

        return self.gradInput
