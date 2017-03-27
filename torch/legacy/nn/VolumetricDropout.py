import torch
from .Module import Module
from .utils import clear


class VolumetricDropout(Module):

    def __init__(self, p=0.5):
        super(VolumetricDropout, self).__init__()
        self.p = p
        self.train = True
        self.noise = torch.Tensor()

    def updateOutput(self, input):
        self.output.resize_as_(input).copy_(input)
        if self.train:
            assert input.dim() == 5
            self.noise.resize_(input.size(0), input.size(1), 1, 1, 1)

            self.noise.bernoulli_(1 - self.p)
            # We expand the random dropouts to the entire feature map because the
            # features are likely correlated accross the map and so the dropout
            # should also be correlated.
            self.output.mul_(self.noise.expand_as(input))
        else:
            self.output.mul_(1 - self.p)

        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.train:
            self.gradInput.resize_as_(gradOutput).copy_(gradOutput)
            self.gradInput.mul_(self.noise.expand_as(input))  # simply mask the gradients with the noise vector
        else:
            raise RuntimeError('backprop only defined while training')

        return self.gradInput

    def setp(self, p):
        self.p = p

    def __repr__(self):
        return super(VolumetricDropout, self).__repr__() + '({:.4f})'.format(self.p)

    def clearState(self):
        clear(self, 'noise')
        return super(VolumetricDropout, self).clearState()
