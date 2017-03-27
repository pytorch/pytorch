import torch
from .Module import Module

# This module acts as an L1 latent state regularizer, adding the
# [gradOutput] to the gradient of the L1 loss. The [input] is copied to
# the [output].


class L1Penalty(Module):

    def __init__(self, l1weight, sizeAverage=False, provideOutput=True):
        super(L1Penalty, self).__init__()
        self.l1weight = l1weight
        self.sizeAverage = sizeAverage
        self.provideOutput = provideOutput

    def updateOutput(self, input):
        m = self.l1weight
        if self.sizeAverage:
            m = m / input.nelement()

        loss = m * input.norm(1)
        self.loss = loss
        self.output = input
        return self.output

    def updateGradInput(self, input, gradOutput):
        m = self.l1weight
        if self.sizeAverage:
            m = m / input.nelement()

        self.gradInput.resize_as_(input).copy_(input).sign_().mul_(m)

        if self.provideOutput:
            self.gradInput.add_(gradOutput)

        return self.gradInput
