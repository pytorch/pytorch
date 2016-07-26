import torch
from torch.legacy import nn

# This module acts as an L1 latent state regularizer, adding the
# [gradOutput] to the gradient of the L1 loss. The [input] is copied to
# the [output].

class L1Penalty(nn.Module):

    # TODO: why sizeAverage=False by default?
    def __init__(self, l1weight, sizeAverage=False, provideOutput=True):
        super(L1Penalty, self).__init__()
        self.l1weight = l1weight
        self.sizeAverage = sizeAverage
        self.provideOutput = provideOutput

    def updateOutput(self, input):
        m = self.l1weight
        if self.sizeAverage:
            m = m / input.nElement()

        loss = m * input.norm(1)
        self.loss = loss
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        m = self.l1weight
        if self.sizeAverage:
            m = m / input.nElement()

        self.gradInput.resizeAs(input).copy(input).sign().mul(m)

        if self.provideOutput:
            self.gradInput.add(gradOutput)

        return self.gradInput

