import torch
from torch.legacy import nn

class BCECriterion(nn.Criterion):
    eps = 1e-12

    def __init__(self, weights=None, sizeAverage=True):
        if weights is not None and weights.dim() != 1:
            raise ValueError("weights input should be 1D Tensor")

        super(BCECriterion, self).__init__()
        self.sizeAverage = sizeAverage
        self.buffer = None
        self.weights = weights

    def updateOutput(self, input, target):
         # - log(input) * target - log(1 - input) * (1 - target)
        if input.nElement() != target.nElement():
            raise RuntimeError("input and target size mismatch")

        self.buffer = self.buffer or input.new()

        buffer = self.buffer
        weights = self.weights

        buffer.resizeAs_(input)

        if weights is not None and target.dim() != 1:
            weights = self.weights.view(1, target.size(1)).expandAs(target)

        # log(input) * target
        torch.add(buffer, input, self.eps).log_()
        if weights is not None:
            buffer.mul_(weights)

        output = torch.dot(target, buffer)

        # log(1 - input) * (1 - target)
        torch.mul(buffer, input, -1).add_(1+self.eps).log_()
        if weights is not None:
            buffer.mul_(weights)

        output = output + torch.sum(buffer)
        output = output - torch.dot(target, buffer)

        if self.sizeAverage:
            output = output / input.nElement()

        self.output = - output

        return self.output


    def updateGradInput(self, input, target):
         # - (target - input) / ( input (1 - input) )
         # The gradient is slightly incorrect:
         # It should have be divided by (input + self.eps) (1 - input + self.eps)
         # but it is divided by input (1 - input + self.eps) + self.eps
         # This modification requires less memory to be computed.
         if input.nElement() != target.nElement():
            raise RuntimeError("input and target size mismatch")

         self.buffer = self.buffer or input.new()

         buffer = self.buffer
         weights = self.weights
         gradInput = self.gradInput

         if weights is not None and target.dim() != 1:
             weights = self.weights.view(1, target.size(1)).expandAs(target)


         buffer.resizeAs_(input)
         # - x ( 1 + self.eps -x ) + self.eps
         torch.add(buffer, input, -1).add_(-self.eps).mul_(input).add_(-self.eps)

         gradInput.resizeAs_(input)
         # y - x
         torch.add(gradInput, target, -1, input)
         # - (y - x) / ( x ( 1 + self.eps -x ) + self.eps )
         gradInput.div_(buffer)

         if weights is not None:
             gradInput.mul_(weights)

         if self.sizeAverage:
             gradInput.div_(target.nElement())

         return gradInput

