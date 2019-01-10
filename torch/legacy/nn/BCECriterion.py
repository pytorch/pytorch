import torch
from .Criterion import Criterion

# TODO: use THNN


class BCECriterion(Criterion):
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
        if input.nelement() != target.nelement():
            raise RuntimeError("input and target size mismatch")

        if self.buffer is None:
            self.buffer = input.new()

        buffer = self.buffer
        weights = self.weights

        buffer.resize_as_(input)

        if weights is not None and target.dim() != 1:
            weights = self.weights.view(1, target.size(1)).expand_as(target)

        # log(input) * target
        torch.add(input, self.eps, out=buffer).log_()
        if weights is not None:
            buffer.mul_(weights)

        target_1d = target.contiguous().view(-1)
        # don't save a 1-d view of buffer: it should already be contiguous, and it's
        # used as non-1d tensor later.
        output = torch.dot(target_1d, buffer.contiguous().view(-1))

        # log(1 - input) * (1 - target)
        torch.mul(input, -1, out=buffer).add_(1 + self.eps).log_()
        if weights is not None:
            buffer.mul_(weights)

        output = output + torch.sum(buffer)
        output = output - torch.dot(target_1d, buffer.contiguous().view(-1))

        if self.sizeAverage:
            output = output / input.nelement()

        self.output = - output.item()

        return self.output

    def updateGradInput(self, input, target):
        # - (target - input) / ( input (1 - input) )
        # The gradient is slightly incorrect:
        # It should have be divided by (input + self.eps) (1 - input + self.eps)
        # but it is divided by input (1 - input + self.eps) + self.eps
        # This modification requires less memory to be computed.
        if input.nelement() != target.nelement():
            raise RuntimeError("input and target size mismatch")

        if self.buffer is None:
            self.buffer = input.new()

        buffer = self.buffer
        weights = self.weights
        gradInput = self.gradInput

        if weights is not None and target.dim() != 1:
            weights = self.weights.view(1, target.size(1)).expand_as(target)

        buffer.resize_as_(input)
        # - x ( 1 + self.eps -x ) + self.eps
        torch.add(input, -1, out=buffer).add_(-self.eps).mul_(input).add_(-self.eps)

        gradInput.resize_as_(input)
        # y - x
        torch.add(target, -1, input, out=gradInput)
        # - (y - x) / ( x ( 1 + self.eps -x ) + self.eps )
        gradInput.div_(buffer)

        if weights is not None:
            gradInput.mul_(weights)

        if self.sizeAverage:
            gradInput.div_(target.nelement())

        return gradInput
