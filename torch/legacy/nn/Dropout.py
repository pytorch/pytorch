import torch
from torch.legacy import nn

class Dropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.train = True
        self.noise = torch.Tensor()

    def updateOutput(self, input):
        if self.inplace:
            self.output.set_(input)
        else:
            self.output.resizeAs_(input).copy(input)

        if self.p > 0 and self.train:
            self.noise.resizeAs_(input)
            self.noise.bernoulli_(1-self.p)
            self.noise.div_(1-self.p)
            self.output.mul_(self.noise)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.inplace:
            self.gradInput.set_(gradOutput)
        else:
            self.gradInput.resizeAs_(gradOutput).copy(gradOutput)

        if self.p > 0 and self.train:
            self.gradInput.mul_(self.noise) # simply mask the gradients with the noise vector

        return self.gradInput

    def setp(self, p):
        self.p = p

    def __repr__(self):
        return super(Dropout, self).__repr__() + '({:.4f})'.format(self.p)

    def clearState(self):
        nn.utils.clear(self, 'noise')
        return super(Dropout, self).clearState()

