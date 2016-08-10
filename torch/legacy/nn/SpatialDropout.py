import torch
from torch.legacy import nn

class SpatialDropout(nn.Module):

    def __init__(self, p=0.5):
        super(SpatialDropout, self).__init__()
        self.p = p
        self.train = True
        self.noise = torch.Tensor()

    def updateOutput(self, input):
        self.output.resizeAs_(input).copy(input)
        if self.train:
            if input.dim() == 4:
                self.noise.resize_(input.size(0), input.size(1), 1, 1)
            else:
                raise RuntimeError('Input must be 4D (nbatch, nfeat, h, w)')

            self.noise.bernoulli_(1-self.p)
            # We expand the random dropouts to the entire feature map because the
            # features are likely correlated accross the map and so the dropout
            # should also be correlated.
            self.output.mul_(self.noise.expandAs(input))
        else:
            self.output.mul_(1-self.p)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.train:
            self.gradInput.resizeAs_(gradOutput).copy(gradOutput)
            self.gradInput.mul_(self.noise.expandAs(input)) # simply mask the gradients with the noise vector
        else:
            raise RuntimeError('backprop only defined while training')

        return self.gradInput

    def setp(self, p):
        self.p = p

    def __repr__(self):
        return super(SpatialDropout, self).__repr__()

    def clearState(self):
        nn.utils.clear(self, 'noise')
        return super(SpatialDropout, self).clearState()

