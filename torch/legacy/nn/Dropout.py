import torch
from torch.legacy import nn

# TODO: is it 100% to remove v2?
class Dropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        self.p = p
        self.inplace = inplace
        self.train = True
        self.noise = torch.Tensor()

    def updateOutput(self, input):
        if self.inplace:
            self.output.set(input)
        else:
            self.output.resizeAs(input).copy(input)

        if self.p > 0 and self.train:
            self.noise.resizeAs(input)
            self.noise.bernoulli(1-self.p)
            self.noise.div(1-self.p)
            self.output.cmul(self.noise)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.inplace:
            self.gradInput.set(gradOutput)
        else:
            self.gradInput.resizeAs(gradOutput).copy(gradOutput)

        if self.p > 0 and self.train:
            self.gradInput.cmul(self.noise) # simply mask the gradients with the noise vector

        return self.gradInput

    def setp(self, p):
        self.p = p

    def __repr__(self):
        return '%s({:.4f})'.format(str(type(self)), self.p)

    def clearState(self):
        # TODO: this probably shouldn't call set
        if self.noise:
           self.noise.set()
        return Parent.clearState(self)

